use std::collections::HashMap;

use geo::algorithm::area::Area;
use geo::algorithm::bool_ops::{BooleanOps, FillRule as GeoFillRule};
use geo::algorithm::bounding_rect::BoundingRect;
use geo::algorithm::buffer::{Buffer, BufferStyle};
use geo::algorithm::contains::Contains;
use geo::line_measures::LengthMeasurable;
use geo::{Coord, Line, LineString, MultiLineString, MultiPolygon, Polygon};
use glam::{DVec2, Vec3};

use crate::core::batching::ColorBatch;
use crate::core::types::{
    JobOp, JobPlan, NormalizedPrimitive, PalettePlan, PrimitiveGeom, StrokeWaypoint,
};

/// Minimum polygon area (m^2) below which degenerate slivers are skipped.
const MIN_AREA_M2: f64 = 1e-6;

/// Physical spray nozzle and toolpath planning parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannerConfig {
    pub cone_half_angle_rad: f32,
    pub d_min: f32,
    pub d_max: f32,
    pub d_perimeter: f32,
    pub lead_length: f32,
    pub min_feed_speed: f32,
    pub max_feed_speed: f32,
    pub hatch_overlap_step_ratio: f32,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            cone_half_angle_rad: 12.0_f32.to_radians(),
            d_min: 0.04,
            d_max: 0.26,
            d_perimeter: 0.05,
            lead_length: 0.035,
            min_feed_speed: 0.12,
            max_feed_speed: 0.85,
            hatch_overlap_step_ratio: 0.65,
        }
    }
}

impl PlannerConfig {
    /// Effective core spot width `w(d) = 2 * d * tan(theta_c)` at standoff `d`.
    #[inline]
    pub fn spot_width(&self, standoff: f32) -> f32 {
        2.0 * standoff * self.cone_half_angle_rad.tan()
    }

    /// Computes standoff- and opacity-compensated nozzle speed `v_spray` in `[min_feed_speed, max_feed_speed]`.
    ///
    /// Calibrated around `0.33 m/s` at `d = 0.12m`, `opacity = 1.0`, scaling inversely with `d * opacity`.
    #[inline]
    pub fn compensated_feed_speed(&self, standoff: f32, opacity: f32) -> f32 {
        let d = standoff.max(1e-4);
        let op = opacity.clamp(0.05, 1.0);
        (0.040 / (d * op)).clamp(self.min_feed_speed, self.max_feed_speed)
    }
}

/// Generates a full 3D [`JobPlan`] (can swaps, shakes, contour+hatch strokes, park home).
pub fn plan_job(
    primitives: &[NormalizedPrimitive],
    _palette: &PalettePlan,
    batches: &[ColorBatch],
    config: &PlannerConfig,
) -> JobPlan {
    let by_id: HashMap<usize, &NormalizedPrimitive> =
        primitives.iter().map(|p| (p.id, p)).collect();

    let mut ops: Vec<JobOp> = Vec::new();

    for batch in batches {
        let mut batch_strokes: Vec<JobOp> = Vec::new();
        for &prim_id in &batch.primitive_ids {
            let Some(&prim) = by_id.get(&prim_id) else {
                continue;
            };
            plan_primitive(prim, config, &mut batch_strokes);
        }

        if batch_strokes.is_empty() {
            continue;
        }

        ops.push(JobOp::PickCan {
            slot: batch.slot_index,
        });
        ops.push(JobOp::ShakeCan);
        ops.extend(batch_strokes);
        ops.push(JobOp::ReturnCan {
            slot: batch.slot_index,
        });
    }

    ops.push(JobOp::ParkHome);
    JobPlan { ops }
}

fn plan_primitive(prim: &NormalizedPrimitive, config: &PlannerConfig, out: &mut Vec<JobOp>) {
    let opacity = prim.opacity.clamp(0.01, 1.0);
    match &prim.geometry {
        PrimitiveGeom::StrokedPath { points, width } => {
            if points.len() < 2 {
                return;
            }
            let w_max = config.spot_width(config.d_max) as f64;
            if *width <= w_max {
                let standoff = ((*width as f32) / (2.0 * config.cone_half_angle_rad.tan()))
                    .clamp(config.d_min, config.d_max);
                emit_polyline_stroke(points, standoff, opacity, config, out);
            } else {
                let coords: Vec<Coord<f64>> =
                    points.iter().map(|p| Coord { x: p.x, y: p.y }).collect();
                let ls = LineString::new(coords);
                let poly = ls.buffer_with_style(BufferStyle::new((*width * 0.5).max(1e-4)));
                plan_multipolygon(&poly, opacity, config, out);
            }
        }
        PrimitiveGeom::FilledPolygon(mp) => {
            plan_multipolygon(mp, opacity, config, out);
        }
    }
}

fn plan_multipolygon(
    mp: &MultiPolygon<f64>,
    opacity: f32,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    for poly in &mp.0 {
        if poly.unsigned_area() < MIN_AREA_M2 {
            continue;
        }
        plan_single_polygon(poly, opacity, config, out);
    }
}

fn plan_single_polygon(
    poly: &Polygon<f64>,
    opacity: f32,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    let w_perim = config.spot_width(config.d_perimeter) as f64;
    let inset_perim = 0.45 * w_perim;
    let perim_mp = filter_valid_mp(poly.buffer(-inset_perim));

    if perim_mp.0.is_empty() {
        // Thin sliver narrower than 2 * inset_perim: fall back to medial pass at d_min.
        let poly_mp = MultiPolygon::new(vec![poly.clone()]);
        plan_thin_sliver_medial(&poly_mp, opacity, config, 0, out);
        return;
    }

    // 1. Perimeter pass at d_perimeter (exterior + all hole rings).
    for p in &perim_mp.0 {
        emit_ring_stroke(p.exterior(), config.d_perimeter, opacity, config, out);
        for hole in p.interiors() {
            emit_ring_stroke(hole, config.d_perimeter, opacity, config, out);
        }
    }

    // 2. Interior boustrophedon zig-zag hatch: inset further by 0.6 * w(0.05).
    let inset_hatch = inset_perim + 0.6 * w_perim;
    let hatch_mp = filter_valid_mp(poly.buffer(-inset_hatch));
    if !hatch_mp.0.is_empty() {
        plan_boustrophedon_hatch(&hatch_mp, &perim_mp, opacity, config, out);
    }
}

fn emit_ring_stroke(
    ring: &LineString<f64>,
    standoff: f32,
    opacity: f32,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    let pts: Vec<DVec2> = ring.coords().map(|c| DVec2::new(c.x, c.y)).collect();
    emit_polyline_stroke(&pts, standoff, opacity, config, out);
}

fn plan_boustrophedon_hatch(
    hatch_mp: &MultiPolygon<f64>,
    containment_mp: &MultiPolygon<f64>,
    opacity: f32,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    let Some(bbox) = hatch_mp.bounding_rect() else {
        return;
    };

    let width = bbox.width();
    let height = bbox.height();
    let minor_dim = width.min(height) as f32;

    // Choose d_fill in [0.08, 0.22] proportional to minor dimension.
    let w_fill_min = config.spot_width(0.08);
    let w_fill_max = config.spot_width(0.22);
    let target_w = (minor_dim * 0.35).clamp(w_fill_min, w_fill_max);
    let d_fill = (target_w / (2.0 * config.cone_half_angle_rad.tan())).clamp(0.08, 0.22);

    let step_over = (config.hatch_overlap_step_ratio as f64) * (config.spot_width(d_fill) as f64);
    let step_over = step_over.max(0.005);

    // Orient scanlines along the major axis to minimize turnaround count.
    let horizontal = width >= height;
    let (u_min, u_max, v_min, v_max) = if horizontal {
        (bbox.min().y, bbox.max().y, bbox.min().x - 0.01, bbox.max().x + 0.01)
    } else {
        (bbox.min().x, bbox.max().x, bbox.min().y - 0.01, bbox.max().y + 0.01)
    };

    let span = u_max - u_min;
    if span <= 1e-6 {
        return;
    }

    let mut scan_coords: Vec<f64> = Vec::new();
    let mut u = u_min + 0.25 * step_over;
    if u >= u_max {
        scan_coords.push(0.5 * (u_min + u_max));
    } else {
        while u < u_max {
            scan_coords.push(u);
            u += step_over;
        }
        // Ensure near-far edge is not left uncovered.
        if let Some(&last_u) = scan_coords.last() {
            if u_max - last_u > 0.55 * step_over {
                scan_coords.push((u_max - 0.25 * step_over).max(last_u + 1e-4));
            }
        }
    }

    let lines: Vec<LineString<f64>> = scan_coords
        .iter()
        .map(|&coord| {
            if horizontal {
                LineString::new(vec![Coord { x: v_min, y: coord }, Coord { x: v_max, y: coord }])
            } else {
                LineString::new(vec![Coord { x: coord, y: v_min }, Coord { x: coord, y: v_max }])
            }
        })
        .collect();
    let scan_mls = MultiLineString::new(lines);
    let clipped = hatch_mp.clip(&scan_mls, false);

    // Bucket clipped segments by scanline row index.
    let mut rows: Vec<Vec<(DVec2, DVec2)>> = vec![Vec::new(); scan_coords.len()];
    for ls in clipped.0 {
        let coords: Vec<Coord<f64>> = ls.coords().copied().collect();
        if coords.len() < 2 {
            continue;
        }
        let a = DVec2::new(coords[0].x, coords[0].y);
        let b = DVec2::new(coords.last().unwrap().x, coords.last().unwrap().y);
        if a.distance(b) < 1e-5 {
            continue;
        }
        let mid_u = if horizontal {
            0.5 * (a.y + b.y)
        } else {
            0.5 * (a.x + b.x)
        };
        let mut best_row = 0usize;
        let mut best_du = f64::INFINITY;
        for (ri, &rc) in scan_coords.iter().enumerate() {
            let du = (mid_u - rc).abs();
            if du < best_du {
                best_du = du;
                best_row = ri;
            }
        }
        // Canonicalize left->right (or bottom->top) before boustrophedon reversal.
        let (p0, p1) = if horizontal {
            if a.x <= b.x { (a, b) } else { (b, a) }
        } else if a.y <= b.y {
            (a, b)
        } else {
            (b, a)
        };
        rows[best_row].push((p0, p1));
    }

    for row in &mut rows {
        row.sort_by(|sa, sb| {
            let ka = if horizontal { sa.0.x } else { sa.0.y };
            let kb = if horizontal { sb.0.x } else { sb.0.y };
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Build boustrophedon continuous polylines, retracting across holes/concavities.
    let mut current_poly: Vec<DVec2> = Vec::new();
    let mut prev_row: Option<usize> = None;

    for (row_idx, row_segs) in rows.into_iter().enumerate() {
        if row_segs.is_empty() {
            if current_poly.len() >= 2 {
                emit_polyline_stroke(&current_poly, d_fill, opacity, config, out);
            }
            current_poly.clear();
            prev_row = None;
            continue;
        }

        let reverse_dir = row_idx % 2 == 1;
        let ordered_segs: Vec<(DVec2, DVec2)> = if reverse_dir {
            row_segs.into_iter().rev().map(|(a, b)| (b, a)).collect()
        } else {
            row_segs
        };

        for (seg_idx, (s_start, s_end)) in ordered_segs.into_iter().enumerate() {
            let can_connect = if seg_idx == 0 && prev_row == Some(row_idx.saturating_sub(1)) {
                if let Some(&last_pt) = current_poly.last() {
                    let link = Line::new(
                        Coord {
                            x: last_pt.x,
                            y: last_pt.y,
                        },
                        Coord {
                            x: s_start.x,
                            y: s_start.y,
                        },
                    );
                    last_pt.distance(s_start) <= 2.5 * step_over && containment_mp.contains(&link)
                } else {
                    false
                }
            } else {
                false
            };

            if can_connect {
                current_poly.push(s_start);
                current_poly.push(s_end);
            } else {
                if current_poly.len() >= 2 {
                    emit_polyline_stroke(&current_poly, d_fill, opacity, config, out);
                }
                current_poly.clear();
                current_poly.push(s_start);
                current_poly.push(s_end);
            }
        }
        prev_row = Some(row_idx);
    }

    if current_poly.len() >= 2 {
        emit_polyline_stroke(&current_poly, d_fill, opacity, config, out);
    }
}

/// Binary-searches the largest inset yielding a non-empty medial core so thin glyphs/slivers
/// are traced cleanly along their centerlines at `d = d_min` without overspray or vanishing.
fn plan_thin_sliver_medial(
    mp: &MultiPolygon<f64>,
    opacity: f32,
    config: &PlannerConfig,
    depth: usize,
    out: &mut Vec<JobOp>,
) {
    let w_min = config.spot_width(config.d_min) as f64;
    let max_inset = 0.45 * (config.spot_width(config.d_perimeter) as f64);

    for poly in &mp.0 {
        let area = poly.unsigned_area();
        if area < MIN_AREA_M2 {
            continue;
        }

        // Binary search largest inset r in (0, max_inset] where poly.buffer(-r) remains non-empty.
        let mut lo = 0.0_f64;
        let mut hi = max_inset;
        let mut best_core: Option<MultiPolygon<f64>> = None;
        let mut best_r = 0.0_f64;

        for _ in 0..8 {
            let mid = 0.5 * (lo + hi);
            if mid < 1e-5 {
                break;
            }
            let inset = poly.buffer(-mid);
            if !inset.0.is_empty() && inset.unsigned_area() > 1e-9 {
                best_core = Some(inset);
                best_r = mid;
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let Some(core) = best_core else {
            // Ultra-thin polygon (< 0.02mm wide): fall back to major-axis spine segment.
            if let Some(bb) = poly.bounding_rect() {
                let pts = if bb.width() >= bb.height() {
                    let y = bb.center().y;
                    vec![DVec2::new(bb.min().x, y), DVec2::new(bb.max().x, y)]
                } else {
                    let x = bb.center().x;
                    vec![DVec2::new(x, bb.min().y), DVec2::new(x, bb.max().y)]
                };
                emit_polyline_stroke(&pts, config.d_min, opacity, config, out);
            }
            continue;
        };

        // Emit exterior loop of each medial core component (skip inner hole loop when thin to avoid double pass).
        for core_poly in &core.0 {
            let ext_len = core_poly.exterior().length(&geo::Euclidean);
            if ext_len < 1e-5 {
                continue;
            }
            let pts: Vec<DVec2> = core_poly
                .exterior()
                .coords()
                .map(|c| DVec2::new(c.x, c.y))
                .collect();
            emit_polyline_stroke(&pts, config.d_min, opacity, config, out);
        }

        // If the shape had mixed thicknesses (e.g., thick stem + thin serif), paint uncovered portions.
        if depth < 2 {
            let covered = core.buffer(0.48 * w_min.max(best_r * 2.0));
            let poly_mp = MultiPolygon::new(vec![poly.clone()]);
            let remainder = filter_valid_mp(poly_mp.difference_with_fill_rule(&covered, GeoFillRule::EvenOdd));
            if !remainder.0.is_empty() && remainder.unsigned_area() > 0.15 * area {
                plan_thin_sliver_medial(&remainder, opacity, config, depth + 1, out);
            }
        }
    }
}

fn filter_valid_mp(mp: MultiPolygon<f64>) -> MultiPolygon<f64> {
    MultiPolygon::new(
        mp.0.into_iter()
            .filter(|p| p.unsigned_area() >= MIN_AREA_M2)
            .collect(),
    )
}

/// Wraps a continuous 2D path with tangential lead-in/lead-out (`valve_on = false`)
/// and emits a [`JobOp::PaintStroke`].
fn emit_polyline_stroke(
    raw_points: &[DVec2],
    standoff: f32,
    opacity: f32,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    // Clean near-duplicate consecutive vertices while preserving closure if closed.
    let mut pts: Vec<DVec2> = Vec::with_capacity(raw_points.len());
    for &p in raw_points {
        if pts.last().is_none_or(|last| last.distance(p) > 1e-5) {
            pts.push(p);
        }
    }
    if pts.len() < 2 {
        return;
    }

    let total_len: f64 = pts.windows(2).map(|w| w[0].distance(w[1])).sum();
    if total_len < 1e-5 {
        return;
    }

    let start_dir = (pts[1] - pts[0]).normalize_or_zero();
    let end_dir = (pts[pts.len() - 1] - pts[pts.len() - 2]).normalize_or_zero();
    if start_dir.length_squared() < 0.5 || end_dir.length_squared() < 0.5 {
        return;
    }

    let feed_speed = config.compensated_feed_speed(standoff, opacity);
    let lead = config.lead_length as f64;
    let lead_in_pt = pts[0] - start_dir * lead;
    let lead_out_pt = pts[pts.len() - 1] + end_dir * lead;

    let make_wp = |p: DVec2, valve_on: bool| StrokeWaypoint {
        pos: Vec3::new(p.x as f32, p.y as f32, 0.0),
        normal: Vec3::Z,
        standoff,
        feed_speed,
        valve_on,
        opacity_scale: opacity,
    };

    let mut waypoints = Vec::with_capacity(pts.len() + 2);
    waypoints.push(make_wp(lead_in_pt, false));
    for p in pts {
        waypoints.push(make_wp(p, true));
    }
    waypoints.push(make_wp(lead_out_pt, false));

    out.push(JobOp::PaintStroke {
        points_canvas_3d: waypoints,
    });
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::core::batching::{BatchConfig, batch_primitives};
    use crate::core::palette::quantize_palette;
    use crate::core::svg_ingest::{IngestConfig, ingest_svg, ingest_svg_file};

    fn square_poly(cx: f64, cy: f64, half: f64) -> MultiPolygon<f64> {
        let poly = Polygon::new(
            LineString::new(vec![
                Coord { x: cx - half, y: cy - half },
                Coord { x: cx + half, y: cy - half },
                Coord { x: cx + half, y: cy + half },
                Coord { x: cx - half, y: cy + half },
                Coord { x: cx - half, y: cy - half },
            ]),
            Vec::new(),
        );
        MultiPolygon::new(vec![poly])
    }

    #[test]
    fn contour_and_hatch_generated_for_large_polygon() {
        let prims = vec![NormalizedPrimitive {
            id: 0,
            color_srgb: [255, 100, 50],
            opacity: 1.0,
            geometry: PrimitiveGeom::FilledPolygon(square_poly(0.0, 1.55, 0.20)),
        }];
        let palette = quantize_palette(&prims, 16, 42);
        let batches = batch_primitives(&prims, &palette, &BatchConfig::default());
        let cfg = PlannerConfig::default();
        let job = plan_job(&prims, &palette, &batches, &cfg);

        // Expect PickCan, ShakeCan, >= 2 PaintStrokes (1 perimeter + >= 1 hatch), ReturnCan, ParkHome
        assert!(matches!(job.ops.first(), Some(JobOp::PickCan { slot: 0 })));
        assert!(matches!(job.ops.get(1), Some(JobOp::ShakeCan)));
        assert!(matches!(job.ops.last(), Some(JobOp::ParkHome)));

        let strokes: Vec<&Vec<StrokeWaypoint>> = job
            .ops
            .iter()
            .filter_map(|op| match op {
                JobOp::PaintStroke { points_canvas_3d } => Some(points_canvas_3d),
                _ => None,
            })
            .collect();
        assert!(strokes.len() >= 2, "expected perimeter + hatch strokes, got {}", strokes.len());

        // First stroke should be perimeter at d_perimeter = 0.05m
        assert!((strokes[0][0].standoff - cfg.d_perimeter).abs() < 1e-6);
        // Interior hatch stroke should use d_fill in [0.08, 0.22]
        assert!(
            strokes[1][0].standoff >= 0.08 - 1e-6 && strokes[1][0].standoff <= 0.22 + 1e-6,
            "hatch standoff {} out of [0.08, 0.22]",
            strokes[1][0].standoff
        );
    }

    #[test]
    fn thin_sliver_text_fallback_produces_strokes_at_d_min() {
        // 8mm wide x 12cm tall sliver (< w(0.04)=17mm, empty after 0.45*w(0.05)=9.5mm inset)
        let poly = Polygon::new(
            LineString::new(vec![
                Coord { x: -0.004, y: 1.49 },
                Coord { x: 0.004, y: 1.49 },
                Coord { x: 0.004, y: 1.61 },
                Coord { x: -0.004, y: 1.61 },
                Coord { x: -0.004, y: 1.49 },
            ]),
            Vec::new(),
        );
        let prims = vec![NormalizedPrimitive {
            id: 0,
            color_srgb: [255, 255, 255],
            opacity: 1.0,
            geometry: PrimitiveGeom::FilledPolygon(MultiPolygon::new(vec![poly])),
        }];
        let palette = quantize_palette(&prims, 16, 42);
        let batches = batch_primitives(&prims, &palette, &BatchConfig::default());
        let cfg = PlannerConfig::default();
        let job = plan_job(&prims, &palette, &batches, &cfg);

        let strokes: Vec<&Vec<StrokeWaypoint>> = job
            .ops
            .iter()
            .filter_map(|op| match op {
                JobOp::PaintStroke { points_canvas_3d } => Some(points_canvas_3d),
                _ => None,
            })
            .collect();
        assert!(!strokes.is_empty(), "thin sliver must not vanish");
        assert!((strokes[0][0].standoff - cfg.d_min).abs() < 1e-6);
    }

    #[test]
    fn lead_in_and_lead_out_valve_invariants_across_full_showcase_svgs() {
        let cfg = PlannerConfig::default();
        for svg_path in [
            "assets/examples/01_geometric_shapes.svg",
            "assets/examples/02_street_art_poster.svg",
            "assets/examples/03_palette_stress.svg",
        ] {
            let prims = ingest_svg_file(Path::new(svg_path), &IngestConfig::default())
                .expect("ingest showcase SVG");
            let palette = quantize_palette(&prims, 16, 42);
            let batches = batch_primitives(&prims, &palette, &BatchConfig::default());
            let job = plan_job(&prims, &palette, &batches, &cfg);

            assert!(matches!(job.ops.last(), Some(JobOp::ParkHome)));
            let mut stroke_count = 0usize;
            for op in &job.ops {
                if let JobOp::PaintStroke { points_canvas_3d: wps } = op {
                    stroke_count += 1;
                    assert!(wps.len() >= 4, "stroke in {svg_path} needs >= 4 waypoints (lead-in + geom + lead-out)");
                    assert!(!wps.first().unwrap().valve_on, "lead-in must have valve_on = false");
                    assert!(!wps.last().unwrap().valve_on, "lead-out must have valve_on = false");
                    for wp in &wps[1..wps.len() - 1] {
                        assert!(wp.valve_on, "geometry waypoints must have valve_on = true");
                        assert!(
                            wp.standoff >= cfg.d_min - 1e-6 && wp.standoff <= cfg.d_max + 1e-6,
                            "standoff {} out of bounds",
                            wp.standoff
                        );
                        assert!(
                            wp.feed_speed >= cfg.min_feed_speed - 1e-6
                                && wp.feed_speed <= cfg.max_feed_speed + 1e-6,
                            "feed_speed {} out of bounds",
                            wp.feed_speed
                        );
                    }
                    let lead_in_dist = wps[0].pos.distance(wps[1].pos);
                    let lead_out_dist = wps[wps.len() - 2].pos.distance(wps[wps.len() - 1].pos);
                    assert!((lead_in_dist - cfg.lead_length).abs() < 1e-4);
                    assert!((lead_out_dist - cfg.lead_length).abs() < 1e-4);
                }
            }
            assert!(stroke_count > 10, "expected many strokes for {svg_path}");
        }
    }

    #[test]
    fn small_text_glyphs_never_vanish_in_planner() {
        let svg = br##"
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600" width="1000" height="600">
            <text x="100" y="300" font-family="Liberation Sans, sans-serif" font-size="22" font-weight="700" fill="#FFFFFF">SPRAYBOT TEST 01</text>
        </svg>"##;
        let prims = ingest_svg(svg, &IngestConfig::default()).expect("ingest text SVG");
        assert!(!prims.is_empty(), "text SVG should yield glyph outline primitives");
        let palette = quantize_palette(&prims, 16, 42);
        let batches = batch_primitives(&prims, &palette, &BatchConfig::default());
        let job = plan_job(&prims, &palette, &batches, &PlannerConfig::default());

        let stroke_count = job
            .ops
            .iter()
            .filter(|op| matches!(op, JobOp::PaintStroke { .. }))
            .count();
        assert!(stroke_count > 0, "small text glyphs must produce paint strokes");
    }

    #[test]
    fn wide_stroke_buffers_to_contour_and_hatch_and_speed_compensates() {
        let cfg = PlannerConfig::default();
        // Speed compensation checks
        let v_close_opaque = cfg.compensated_feed_speed(0.04, 1.0);
        let v_mid = cfg.compensated_feed_speed(0.12, 1.0);
        let v_far = cfg.compensated_feed_speed(0.26, 1.0);
        let v_half_op = cfg.compensated_feed_speed(0.12, 0.5);
        assert!(v_close_opaque >= v_mid && v_mid >= v_far);
        assert!((v_half_op - (2.0 * v_mid).min(cfg.max_feed_speed)).abs() < 1e-4);

        // Wide stroke > w(0.26) (~0.1105m) should buffer into filled polygon (perimeter + hatch)
        let prims = vec![NormalizedPrimitive {
            id: 0,
            color_srgb: [230, 57, 70],
            opacity: 1.0,
            geometry: PrimitiveGeom::StrokedPath {
                points: vec![DVec2::new(-0.5, 1.5), DVec2::new(0.5, 1.5)],
                width: 0.16,
            },
        }];
        let palette = quantize_palette(&prims, 16, 42);
        let batches = batch_primitives(&prims, &palette, &BatchConfig::default());
        let job = plan_job(&prims, &palette, &batches, &cfg);
        let strokes: Vec<&Vec<StrokeWaypoint>> = job
            .ops
            .iter()
            .filter_map(|op| match op {
                JobOp::PaintStroke { points_canvas_3d } => Some(points_canvas_3d),
                _ => None,
            })
            .collect();
        assert!(
            strokes.len() >= 2,
            "extra-wide 0.16m stroke should expand into contour + hatch strokes, got {}",
            strokes.len()
        );
        assert!((strokes[0][0].standoff - cfg.d_perimeter).abs() < 1e-6);
    }

    #[test]
    fn hatch_retracts_across_interior_hole() {
        // 0.60m x 0.60m square with a large 0.24m x 0.24m center hole
        let outer = LineString::new(vec![
            Coord { x: -0.30, y: 1.25 },
            Coord { x: 0.30, y: 1.25 },
            Coord { x: 0.30, y: 1.85 },
            Coord { x: -0.30, y: 1.85 },
            Coord { x: -0.30, y: 1.25 },
        ]);
        let hole = LineString::new(vec![
            Coord { x: -0.12, y: 1.43 },
            Coord { x: -0.12, y: 1.67 },
            Coord { x: 0.12, y: 1.67 },
            Coord { x: 0.12, y: 1.43 },
            Coord { x: -0.12, y: 1.43 },
        ]);
        let hole_poly = Polygon::new(hole.clone(), Vec::new());
        let poly = Polygon::new(outer, vec![hole]);
        let prims = vec![NormalizedPrimitive {
            id: 0,
            color_srgb: [94, 43, 255],
            opacity: 1.0,
            geometry: PrimitiveGeom::FilledPolygon(MultiPolygon::new(vec![poly])),
        }];
        let palette = quantize_palette(&prims, 16, 42);
        let batches = batch_primitives(&prims, &palette, &BatchConfig::default());
        let cfg = PlannerConfig::default();
        let job = plan_job(&prims, &palette, &batches, &cfg);

        // No active spray segment (between two valve_on=true waypoints) may cross deep inside the hole!
        let inner_hole = hole_poly.buffer(-0.02);
        for op in &job.ops {
            if let JobOp::PaintStroke { points_canvas_3d: wps } = op {
                for pair in wps.windows(2) {
                    if pair[0].valve_on && pair[1].valve_on {
                        let seg = Line::new(
                            Coord { x: pair[0].pos.x as f64, y: pair[0].pos.y as f64 },
                            Coord { x: pair[1].pos.x as f64, y: pair[1].pos.y as f64 },
                        );
                        assert!(
                            !geo::algorithm::intersects::Intersects::intersects(&inner_hole, &seg),
                            "active spray segment {:?} -> {:?} crossed interior hole!",
                            pair[0].pos,
                            pair[1].pos
                        );
                    }
                }
            }
        }
    }
}
