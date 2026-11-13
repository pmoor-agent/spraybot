use std::collections::BTreeMap;

use geo::algorithm::bounding_rect::BoundingRect;
use geo::algorithm::buffer::{Buffer, BufferStyle};
use geo::algorithm::intersects::Intersects;
use geo::{Coord, LineString, MultiPolygon, Point, Rect};
use glam::DVec2;

use crate::core::types::{NormalizedPrimitive, PalettePlan, PrimitiveGeom};

/// Configuration for overlap-aware topological color batching.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchConfig {
    /// Spray feather expansion margin (meters) added to footprints when checking overlap.
    pub feather_margin_m: f64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            feather_margin_m: 0.008,
        }
    }
}

/// Contiguous sequence of same-color primitives painted before swapping cans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorBatch {
    pub slot_index: usize,
    pub primitive_ids: Vec<usize>,
}

/// Computes the buffered 2D spray footprint, AABB, and representative center for a primitive.
fn primitive_footprint(
    prim: &NormalizedPrimitive,
    feather_margin_m: f64,
) -> (MultiPolygon<f64>, Option<Rect<f64>>, DVec2) {
    let margin = feather_margin_m.max(0.0);
    let footprint = match &prim.geometry {
        PrimitiveGeom::FilledPolygon(mp) => {
            if margin > 1e-7 {
                mp.buffer(margin)
            } else {
                mp.clone()
            }
        }
        PrimitiveGeom::StrokedPath { points, width } => {
            let radius = (width * 0.5 + margin).max(1e-4);
            if points.is_empty() {
                MultiPolygon::empty()
            } else if points.len() == 1 {
                Point::new(points[0].x, points[0].y).buffer(radius)
            } else {
                let coords: Vec<Coord<f64>> =
                    points.iter().map(|p| Coord { x: p.x, y: p.y }).collect();
                let ls = LineString::new(coords);
                ls.buffer_with_style(BufferStyle::new(radius))
            }
        }
    };

    let bbox = footprint.bounding_rect();
    let center = if let Some(rect) = bbox {
        let c = rect.center();
        DVec2::new(c.x, c.y)
    } else {
        match &prim.geometry {
            PrimitiveGeom::StrokedPath { points, .. } if !points.is_empty() => points[0],
            _ => DVec2::ZERO,
        }
    };

    (footprint, bbox, center)
}

#[inline]
fn rects_overlap(a: &Rect<f64>, b: &Rect<f64>) -> bool {
    a.min().x <= b.max().x
        && a.max().x >= b.min().x
        && a.min().y <= b.max().y
        && a.max().y >= b.min().y
}

#[inline]
fn slot_for_primitive(prim: &NormalizedPrimitive, idx: usize, palette: &PalettePlan) -> usize {
    if palette.primitive_color_map.is_empty() {
        return 0;
    }
    if prim.id < palette.primitive_color_map.len() {
        palette.primitive_color_map[prim.id]
    } else if idx < palette.primitive_color_map.len() {
        palette.primitive_color_map[idx]
    } else {
        0
    }
}

/// Schedules primitives into overlap-safe color batches minimizing rack swaps and air travel.
pub fn batch_primitives(
    primitives: &[NormalizedPrimitive],
    palette: &PalettePlan,
    config: &BatchConfig,
) -> Vec<ColorBatch> {
    let n = primitives.len();
    if n == 0 {
        return Vec::new();
    }

    // 1. Precompute buffered footprints, AABBs, centers, and assigned palette slots.
    let mut footprints = Vec::with_capacity(n);
    let mut bboxes = Vec::with_capacity(n);
    let mut centers = Vec::with_capacity(n);
    let mut slots = Vec::with_capacity(n);

    for (idx, prim) in primitives.iter().enumerate() {
        let (fp, bb, center) = primitive_footprint(prim, config.feather_margin_m);
        footprints.push(fp);
        bboxes.push(bb);
        centers.push(center);
        slots.push(slot_for_primitive(prim, idx, palette));
    }

    // 2. Build overlap DAG: i -> j if i < j, slot(i) != slot(j), and buffered footprints intersect.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree = vec![0usize; n];

    for i in 0..n {
        let Some(bb_i) = bboxes[i] else {
            continue;
        };
        for j in (i + 1)..n {
            if slots[i] == slots[j] {
                continue;
            }
            let Some(bb_j) = bboxes[j] else {
                continue;
            };
            if !rects_overlap(&bb_i, &bb_j) {
                continue;
            }
            if footprints[i].intersects(&footprints[j]) {
                adj[i].push(j);
                in_degree[j] += 1;
            }
        }
    }

    // 3. Greedy topological color batching over the 0-in-degree frontier.
    let mut frontier: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut scheduled = vec![false; n];
    let mut scheduled_count = 0usize;
    let mut batches: Vec<ColorBatch> = Vec::new();
    // Start near rack side of canvas so first batch starts efficiently.
    let mut last_pos = DVec2::new(-1.17, 1.55);

    while scheduled_count < n {
        debug_assert!(!frontier.is_empty(), "overlap DAG must be acyclic");

        // Group currently ready frontier primitives by palette slot.
        let mut by_slot: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &idx in &frontier {
            by_slot.entry(slots[idx]).or_default().push(idx);
        }

        // Choose color maximizing (ready_batch_size, unlocked_primitives_count), then nearest to last_pos.
        let mut best_slot = 0usize;
        let mut best_score = (0usize, 0usize);
        let mut best_min_dist = f64::INFINITY;

        for (&slot, members) in &by_slot {
            let ready_count = members.len();
            // Count how many distinct successors would become ready (in_degree drops to 0) if this batch runs.
            let mut unlocked_count = 0usize;
            for &u in members {
                for &v in &adj[u] {
                    if in_degree[v] == 1 {
                        unlocked_count += 1;
                    }
                }
            }
            let min_dist = members
                .iter()
                .map(|&u| centers[u].distance(last_pos))
                .fold(f64::INFINITY, f64::min);

            let score = (ready_count, unlocked_count);
            if score > best_score || (score == best_score && min_dist < best_min_dist) {
                best_score = score;
                best_min_dist = min_dist;
                best_slot = slot;
            }
        }

        // Emit all ready primitives of `best_slot` (and keep draining if any newly ready of same slot,
        // though DAG edges only connect different colors).
        let mut batch_indices = by_slot.remove(&best_slot).unwrap_or_default();
        frontier.retain(|&idx| slots[idx] != best_slot);

        // Order batch by nearest-neighbor + 2-opt to minimize rapid air moves.
        order_nearest_neighbor_with_2opt(&mut batch_indices, &centers, last_pos);
        if let Some(&last_idx) = batch_indices.last() {
            last_pos = centers[last_idx];
        }

        let mut primitive_ids = Vec::with_capacity(batch_indices.len());
        for &u in &batch_indices {
            scheduled[u] = true;
            scheduled_count += 1;
            primitive_ids.push(primitives[u].id);

            for &v in &adj[u] {
                if !scheduled[v] {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 {
                        frontier.push(v);
                    }
                }
            }
        }

        batches.push(ColorBatch {
            slot_index: best_slot,
            primitive_ids,
        });
    }

    batches
}

fn order_nearest_neighbor_with_2opt(indices: &mut Vec<usize>, centers: &[DVec2], start_pos: DVec2) {
    if indices.len() <= 1 {
        return;
    }

    // Greedy nearest-neighbor pass starting from `start_pos`.
    let mut remaining = std::mem::take(indices);
    let mut ordered = Vec::with_capacity(remaining.len());
    let mut cur = start_pos;
    while !remaining.is_empty() {
        let mut best_pos = 0usize;
        let mut best_dist = f64::INFINITY;
        for (k, &idx) in remaining.iter().enumerate() {
            let d = centers[idx].distance(cur);
            if d < best_dist {
                best_dist = d;
                best_pos = k;
            }
        }
        let chosen = remaining.swap_remove(best_pos);
        cur = centers[chosen];
        ordered.push(chosen);
    }

    // 2-opt refinement pass (keeps start_pos -> ordered[0] included in cost).
    let m = ordered.len();
    if m >= 4 {
        let mut improved = true;
        let mut passes = 0;
        while improved && passes < 16 {
            improved = false;
            passes += 1;
            for i in 0..(m - 1) {
                let prev = if i == 0 { start_pos } else { centers[ordered[i - 1]] };
                let a = centers[ordered[i]];
                for j in (i + 1)..m {
                    let b = centers[ordered[j]];
                    let cur_cost = prev.distance(a)
                        + if j + 1 < m {
                            b.distance(centers[ordered[j + 1]])
                        } else {
                            0.0
                        };
                    let new_cost = prev.distance(b)
                        + if j + 1 < m {
                            a.distance(centers[ordered[j + 1]])
                        } else {
                            0.0
                        };
                    if new_cost + 1e-6 < cur_cost {
                        ordered[i..=j].reverse();
                        improved = true;
                        break;
                    }
                }
            }
        }
    }

    *indices = ordered;
}

#[cfg(test)]
mod tests {
    use geo::{Coord, LineString, MultiPolygon, Polygon};

    use super::*;
    use crate::core::palette::quantize_palette;

    fn rect_prim(id: usize, x0: f64, y0: f64, x1: f64, y1: f64, color: [u8; 3]) -> NormalizedPrimitive {
        let poly = Polygon::new(
            LineString::new(vec![
                Coord { x: x0, y: y0 },
                Coord { x: x1, y: y0 },
                Coord { x: x1, y: y1 },
                Coord { x: x0, y: y1 },
                Coord { x: x0, y: y0 },
            ]),
            Vec::new(),
        );
        NormalizedPrimitive {
            id,
            color_srgb: color,
            opacity: 1.0,
            geometry: PrimitiveGeom::FilledPolygon(MultiPolygon::new(vec![poly])),
        }
    }

    #[test]
    fn preserves_z_order_on_overlapping_different_colors() {
        let red = [220, 30, 30];
        let blue = [30, 60, 220];
        // Back red -> middle blue (overlaps red) -> front red (overlaps blue).
        // Cannot merge back red and front red because blue sits strictly between them!
        let prims = vec![
            rect_prim(0, 0.0, 1.0, 0.3, 1.3, red),
            rect_prim(1, 0.1, 1.1, 0.4, 1.4, blue),
            rect_prim(2, 0.15, 1.15, 0.25, 1.25, red),
        ];
        let palette = quantize_palette(&prims, 16, 42);
        let batches = batch_primitives(&prims, &palette, &BatchConfig::default());

        let flat: Vec<usize> = batches.iter().flat_map(|b| b.primitive_ids.iter().copied()).collect();
        assert_eq!(flat, vec![0, 1, 2], "overlapping sandwich must preserve 0 -> 1 -> 2 Z-order");
        assert_eq!(batches.len(), 3, "must not merge red across overlapping blue");
    }

    #[test]
    fn batches_disjoint_same_color_shapes_together() {
        let red = [220, 30, 30];
        let blue = [30, 60, 220];
        // Alternating colors in SVG, but all shapes are spatially far apart (> 8mm).
        let prims = vec![
            rect_prim(0, -0.8, 1.0, -0.6, 1.2, red),
            rect_prim(1, -0.4, 1.0, -0.2, 1.2, blue),
            rect_prim(2, 0.0, 1.0, 0.2, 1.2, red),
            rect_prim(3, 0.4, 1.0, 0.6, 1.2, blue),
            rect_prim(4, 0.8, 1.0, 1.0, 1.2, red),
        ];
        let palette = quantize_palette(&prims, 16, 42);
        let batches = batch_primitives(&prims, &palette, &BatchConfig::default());

        assert_eq!(
            batches.len(),
            2,
            "disjoint 2-color shapes should coalesce into just 2 can batches, got {}",
            batches.len()
        );
        let total: usize = batches.iter().map(|b| b.primitive_ids.len()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn feather_margin_detects_near_touching_overlap() {
        let red = [220, 30, 30];
        let blue = [30, 60, 220];
        // Separated by 5mm (< 2 * 8mm feather expansion = 16mm), so 8mm buffered footprints intersect!
        let prims = vec![
            rect_prim(0, 0.0, 1.0, 0.2, 1.2, red),
            rect_prim(1, 0.205, 1.0, 0.4, 1.2, blue),
            rect_prim(2, 0.0, 1.0, 0.4, 1.2, red),
        ];
        let palette = quantize_palette(&prims, 16, 42);
        let batches = batch_primitives(&prims, &palette, &BatchConfig::default());
        let flat: Vec<usize> = batches.iter().flat_map(|b| b.primitive_ids.iter().copied()).collect();
        let pos0 = flat.iter().position(|&id| id == 0).unwrap();
        let pos1 = flat.iter().position(|&id| id == 1).unwrap();
        let pos2 = flat.iter().position(|&id| id == 2).unwrap();
        assert!(pos0 < pos1 && pos1 < pos2, "8mm feather overlap must preserve 0 -> 1 -> 2");
    }
}
