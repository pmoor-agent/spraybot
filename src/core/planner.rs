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
    GradientFillMeta, JobOp, JobPlan, NormalizedPrimitive, NozzleCap, PalettePlan, PrimitiveGeom,
    StrokeWaypoint,
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

/// Effective core spot width `w(d) = 2 * d * tan(theta_c)` for `cap` at standoff `d`.
#[inline]
pub fn spot_width_cap(cap: NozzleCap, standoff: f32) -> f32 {
    2.0 * standoff * cap.cone_half_angle_rad().tan()
}

/// Returns `(d_min, d_perimeter, d_max)` standoff bounds for `cap`.
#[inline]
fn cap_standoff_bounds(cap: NozzleCap, config: &PlannerConfig) -> (f32, f32, f32) {
    match cap {
        NozzleCap::Fat => (config.d_min, config.d_perimeter, config.d_max),
        NozzleCap::Skinny => (0.035, 0.038, 0.160),
    }
}

/// Computes standoff-compensated feed speed depending ONLY on `(cap, standoff)`.
#[inline]
pub fn compensated_feed_speed_cap(config: &PlannerConfig, cap: NozzleCap, standoff: f32) -> f32 {
    let d = standoff.max(1e-4);
    match cap {
        NozzleCap::Fat => (0.040 / d).clamp(config.min_feed_speed, config.max_feed_speed),
        NozzleCap::Skinny => (0.012 / d).clamp(0.10, 0.42),
    }
}

/// Beer-Lambert inverse perceptual-to-flow mapping from target alpha to `opacity_scale`.
#[inline]
pub fn flow_scale_for_target_alpha(target_alpha: f32, optical_depth: f32) -> f32 {
    if target_alpha >= 0.96 {
        return 1.0;
    }
    let clamped = target_alpha.clamp(0.02, 0.96);
    (-(1.0 - clamped).ln() / optical_depth).clamp(0.015, 1.0)
}

#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let span = edge1 - edge0;
    if span.abs() < 1e-12 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Generates a full 3D [`JobPlan`] (can swaps, shakes, cap swaps, contour+hatch strokes, park home).
pub fn plan_job(
    primitives: &[NormalizedPrimitive],
    palette: &PalettePlan,
    batches: &[ColorBatch],
    config: &PlannerConfig,
) -> JobPlan {
    let by_id: HashMap<usize, &NormalizedPrimitive> =
        primitives.iter().map(|p| (p.id, p)).collect();

    let max_slot = batches
        .iter()
        .map(|b| b.slot_index + 1)
        .max()
        .unwrap_or(0)
        .max(palette.cans.len());
    let mut slot_caps = vec![NozzleCap::Fat; max_slot];
    let mut ops: Vec<JobOp> = Vec::new();

    for batch in batches {
        if batch.slot_index >= slot_caps.len() {
            slot_caps.resize(batch.slot_index + 1, NozzleCap::Fat);
        }

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

        let current_cap = slot_caps[batch.slot_index];
        let mut current_group: Vec<JobOp> = Vec::new();
        let mut other_group: Vec<JobOp> = Vec::new();
        for op in batch_strokes {
            let stroke_cap = match &op {
                JobOp::PaintStroke { points_canvas_3d } => points_canvas_3d
                    .first()
                    .map(|wp| wp.cap)
                    .unwrap_or(NozzleCap::Fat),
                _ => NozzleCap::Fat,
            };
            if stroke_cap == current_cap {
                current_group.push(op);
            } else {
                other_group.push(op);
            }
        }

        ops.push(JobOp::PickCan {
            slot: batch.slot_index,
        });
        ops.push(JobOp::ShakeCan);
        ops.extend(current_group);
        if !other_group.is_empty() {
            let other_cap = match current_cap {
                NozzleCap::Fat => NozzleCap::Skinny,
                NozzleCap::Skinny => NozzleCap::Fat,
            };
            ops.push(JobOp::SwapCap { cap: other_cap });
            ops.extend(other_group);
            slot_caps[batch.slot_index] = other_cap;
        }
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
            let cap = if *width < 0.016 {
                NozzleCap::Skinny
            } else {
                NozzleCap::Fat
            };
            let (d_min, _, d_max) = cap_standoff_bounds(cap, config);
            let w_max = spot_width_cap(cap, d_max) as f64;
            if *width <= w_max {
                let standoff = ((*width as f32) / (2.0 * cap.cone_half_angle_rad().tan()))
                    .clamp(d_min, d_max);
                let flow_scale = flow_scale_for_target_alpha(opacity, 3.15);
                emit_uniform_polyline_stroke(points, standoff, flow_scale, cap, config, out);
            } else {
                let coords: Vec<Coord<f64>> =
                    points.iter().map(|p| Coord { x: p.x, y: p.y }).collect();
                let ls = LineString::new(coords);
                let poly = ls.buffer_with_style(BufferStyle::new((*width * 0.5).max(1e-4)));
                plan_multipolygon(&poly, opacity, config, out);
            }
        }
        PrimitiveGeom::FilledPolygon(mp) => {
            if let Some(meta) = &prim.gradient_meta {
                plan_gradient_multipolygon(mp, opacity, meta, config, out);
            } else {
                plan_multipolygon(mp, opacity, config, out);
            }
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
    let area = poly.unsigned_area();
    if area < MIN_AREA_M2 {
        return;
    }

    let fat_inset = 0.45 * spot_width_cap(NozzleCap::Fat, config.d_perimeter) as f64;
    let minor_dim = poly
        .bounding_rect()
        .map(|bb| bb.width().min(bb.height()))
        .unwrap_or(0.0);
    let fat_perim_mp = filter_valid_mp(poly.buffer(-fat_inset));
    let use_skinny = minor_dim < 0.024
        || fat_perim_mp.0.is_empty()
        || fat_perim_mp.unsigned_area() < 0.15 * area;

    let (cap, perim_mp, d_perim, inset_perim, w_perim) = if use_skinny {
        let (_, d_p, _) = cap_standoff_bounds(NozzleCap::Skinny, config);
        let w_p = spot_width_cap(NozzleCap::Skinny, d_p) as f64;
        let inset_p = 0.45 * w_p;
        let mp = filter_valid_mp(poly.buffer(-inset_p));
        (NozzleCap::Skinny, mp, d_p, inset_p, w_p)
    } else {
        let (_, d_p, _) = cap_standoff_bounds(NozzleCap::Fat, config);
        let w_p = spot_width_cap(NozzleCap::Fat, d_p) as f64;
        (NozzleCap::Fat, fat_perim_mp, d_p, fat_inset, w_p)
    };

    if perim_mp.0.is_empty() {
        // Thin sliver narrower than 2 * inset_perim: fall back to medial pass at cap's d_min.
        let poly_mp = MultiPolygon::new(vec![poly.clone()]);
        plan_thin_sliver_medial(&poly_mp, opacity, cap, config, 0, out);
        return;
    }

    // 1. Perimeter pass at d_perimeter (exterior + all hole rings).
    let perim_flow = flow_scale_for_target_alpha(opacity, 3.15);
    for p in &perim_mp.0 {
        emit_ring_stroke(p.exterior(), d_perim, perim_flow, cap, config, out);
        for hole in p.interiors() {
            emit_ring_stroke(hole, d_perim, perim_flow, cap, config, out);
        }
    }

    // 2. Interior boustrophedon zig-zag hatch: inset further by 0.6 * w_perim.
    let inset_hatch = inset_perim + 0.6 * w_perim;
    let hatch_mp = filter_valid_mp(poly.buffer(-inset_hatch));
    if !hatch_mp.0.is_empty() {
        plan_boustrophedon_hatch(&hatch_mp, &perim_mp, opacity, cap, config, out);
    }
}

/// Plans airbrushed gradient bands using wide `Fat` cap scanlines with zero perimeter strokes.
fn plan_gradient_multipolygon(
    mp: &MultiPolygon<f64>,
    opacity: f32,
    meta: &GradientFillMeta,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    match meta {
        GradientFillMeta::Linear {
            origin,
            dir,
            band_width,
            fade_in_s,
            fade_out_s,
        } => {
            let d_fill = ((*band_width as f32 * 0.45)
                / (2.0 * NozzleCap::Fat.cone_half_angle_rad().tan()))
            .clamp(0.12, 0.25);
            let step_over = (0.62 * spot_width_cap(NozzleCap::Fat, d_fill) as f64).max(0.012);
            let perp = DVec2::new(-dir.y, dir.x);

            for poly in &mp.0 {
                if poly.unsigned_area() < MIN_AREA_M2 {
                    continue;
                }
                let poly_mp = MultiPolygon::new(vec![poly.clone()]);
                let chains = build_oriented_boustrophedon_polylines(
                    &poly_mp,
                    &poly_mp,
                    *origin,
                    *dir,
                    perp,
                    step_over,
                    None,
                );
                for chain in chains {
                    let pts_with_op: Vec<(DVec2, f32)> = chain
                        .into_iter()
                        .map(|pt| {
                            let s = (pt - *origin).dot(*dir);
                            let w_in = fade_in_s
                                .map_or(1.0, |(a, b)| 0.04 + 0.96 * smoothstep(a, b, s));
                            let w_out =
                                fade_out_s.map_or(1.0, |(a, b)| 1.0 - 0.55 * smoothstep(a, b, s));
                            let target_alpha = (opacity * (w_in * w_out) as f32).clamp(0.02, 1.0);
                            (pt, flow_scale_for_target_alpha(target_alpha, 4.2))
                        })
                        .collect();
                    emit_polyline_stroke(&pts_with_op, d_fill, NozzleCap::Fat, config, out);
                }
            }
        }
        GradientFillMeta::Radial {
            center,
            band_width,
            fade_in_r,
            fade_out_r,
        } => {
            let d_fill = ((*band_width as f32 * 0.45)
                / (2.0 * NozzleCap::Fat.cone_half_angle_rad().tan()))
            .clamp(0.12, 0.25);
            let step_over = (0.62 * spot_width_cap(NozzleCap::Fat, d_fill) as f64).max(0.012);

            for poly in &mp.0 {
                if poly.unsigned_area() < MIN_AREA_M2 {
                    continue;
                }
                let Some(bbox) = poly.bounding_rect() else {
                    continue;
                };
                let (step_dir, scan_dir) = if bbox.width() >= bbox.height() {
                    (DVec2::Y, DVec2::X)
                } else {
                    (DVec2::X, DVec2::Y)
                };
                let poly_mp = MultiPolygon::new(vec![poly.clone()]);
                let chains = build_oriented_boustrophedon_polylines(
                    &poly_mp,
                    &poly_mp,
                    DVec2::ZERO,
                    step_dir,
                    scan_dir,
                    step_over,
                    Some(0.015),
                );
                for chain in chains {
                    let pts_with_op: Vec<(DVec2, f32)> = chain
                        .into_iter()
                        .map(|pt| {
                            let r = pt.distance(*center);
                            let w_in = fade_in_r
                                .map_or(1.0, |(a, b)| 0.04 + 0.96 * smoothstep(a, b, r));
                            let w_out =
                                fade_out_r.map_or(1.0, |(a, b)| 1.0 - 0.55 * smoothstep(a, b, r));
                            let target_alpha = (opacity * (w_in * w_out) as f32).clamp(0.02, 1.0);
                            (pt, flow_scale_for_target_alpha(target_alpha, 4.2))
                        })
                        .collect();
                    emit_polyline_stroke(&pts_with_op, d_fill, NozzleCap::Fat, config, out);
                }
            }
        }
    }
}

fn emit_ring_stroke(
    ring: &LineString<f64>,
    standoff: f32,
    opacity_scale: f32,
    cap: NozzleCap,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    let pts: Vec<DVec2> = ring.coords().map(|c| DVec2::new(c.x, c.y)).collect();
    emit_uniform_polyline_stroke(&pts, standoff, opacity_scale, cap, config, out);
}

fn plan_boustrophedon_hatch(
    hatch_mp: &MultiPolygon<f64>,
    containment_mp: &MultiPolygon<f64>,
    opacity: f32,
    cap: NozzleCap,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    let Some(bbox) = hatch_mp.bounding_rect() else {
        return;
    };

    let width = bbox.width();
    let height = bbox.height();
    let minor_dim = width.min(height) as f32;

    let (d_fill_min, d_fill_max) = match cap {
        NozzleCap::Fat => (0.08_f32, 0.22_f32),
        NozzleCap::Skinny => (0.045_f32, 0.14_f32),
    };
    let w_fill_min = spot_width_cap(cap, d_fill_min);
    let w_fill_max = spot_width_cap(cap, d_fill_max);
    let target_w = (minor_dim * 0.35).clamp(w_fill_min, w_fill_max);
    let d_fill =
        (target_w / (2.0 * cap.cone_half_angle_rad().tan())).clamp(d_fill_min, d_fill_max);

    let min_step = match cap {
        NozzleCap::Fat => 0.005,
        NozzleCap::Skinny => 0.0025,
    };
    let step_over =
        ((config.hatch_overlap_step_ratio as f64) * (spot_width_cap(cap, d_fill) as f64))
            .max(min_step);

    // Orient scanlines along the major axis to minimize turnaround count.
    let (step_dir, scan_dir) = if width >= height {
        (DVec2::Y, DVec2::X)
    } else {
        (DVec2::X, DVec2::Y)
    };

    let chains = build_oriented_boustrophedon_polylines(
        hatch_mp,
        containment_mp,
        DVec2::ZERO,
        step_dir,
        scan_dir,
        step_over,
        None,
    );
    let hatch_flow = flow_scale_for_target_alpha(opacity, 4.2);
    for chain in chains {
        emit_uniform_polyline_stroke(&chain, d_fill, hatch_flow, cap, config, out);
    }
}

/// Clips parallel scanlines along `scan_dir` (stepped along `step_dir` from `origin`)
/// and connects adjacent segments into boustrophedon polylines when inside `containment_mp`.
fn build_oriented_boustrophedon_polylines(
    region_mp: &MultiPolygon<f64>,
    containment_mp: &MultiPolygon<f64>,
    origin: DVec2,
    step_dir: DVec2,
    scan_dir: DVec2,
    step_over: f64,
    max_subdivide_m: Option<f64>,
) -> Vec<Vec<DVec2>> {
    let mut s_min = f64::INFINITY;
    let mut s_max = f64::NEG_INFINITY;
    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;

    for poly in &region_mp.0 {
        for c in poly.exterior().coords() {
            let rel = DVec2::new(c.x, c.y) - origin;
            let s = rel.dot(step_dir);
            let u = rel.dot(scan_dir);
            s_min = s_min.min(s);
            s_max = s_max.max(s);
            u_min = u_min.min(u);
            u_max = u_max.max(u);
        }
    }

    if !s_min.is_finite() || !u_min.is_finite() {
        return Vec::new();
    }
    let span = s_max - s_min;
    if span <= 1e-6 {
        return Vec::new();
    }

    u_min -= 0.01;
    u_max += 0.01;

    let mut scan_coords: Vec<f64> = Vec::new();
    let mut s_cur = s_min + 0.25 * step_over;
    if s_cur >= s_max {
        scan_coords.push(0.5 * (s_min + s_max));
    } else {
        while s_cur < s_max {
            scan_coords.push(s_cur);
            s_cur += step_over;
        }
        // Ensure near-far edge is not left uncovered.
        if let Some(&last_s) = scan_coords.last() {
            if s_max - last_s > 0.55 * step_over {
                scan_coords.push((s_max - 0.25 * step_over).max(last_s + 1e-4));
            }
        }
    }

    let lines: Vec<LineString<f64>> = scan_coords
        .iter()
        .map(|&s_val| {
            let p0 = origin + step_dir * s_val + scan_dir * u_min;
            let p1 = origin + step_dir * s_val + scan_dir * u_max;
            LineString::new(vec![Coord { x: p0.x, y: p0.y }, Coord { x: p1.x, y: p1.y }])
        })
        .collect();
    let scan_mls = MultiLineString::new(lines);
    let clipped = region_mp.clip(&scan_mls, false);

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
        let mid_s = (0.5 * (a + b) - origin).dot(step_dir);
        let mut best_row = 0usize;
        let mut best_ds = f64::INFINITY;
        for (ri, &rc) in scan_coords.iter().enumerate() {
            let ds = (mid_s - rc).abs();
            if ds < best_ds {
                best_ds = ds;
                best_row = ri;
            }
        }
        // Canonicalize along scan_dir before boustrophedon reversal.
        let ua = (a - origin).dot(scan_dir);
        let ub = (b - origin).dot(scan_dir);
        let (p0, p1) = if ua <= ub { (a, b) } else { (b, a) };
        rows[best_row].push((p0, p1));
    }

    for row in &mut rows {
        row.sort_by(|sa, sb| {
            let ka = (sa.0 - origin).dot(scan_dir);
            let kb = (sb.0 - origin).dot(scan_dir);
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let push_segment = |poly: &mut Vec<DVec2>, a: DVec2, b: DVec2| {
        if let Some(max_spacing) = max_subdivide_m {
            let seg_len = a.distance(b);
            let n_sub = (seg_len / max_spacing).ceil().max(1.0) as usize;
            for k in 0..=n_sub {
                let pt = a.lerp(b, k as f64 / n_sub as f64);
                if poly.last().is_none_or(|last| last.distance(pt) > 1e-5) {
                    poly.push(pt);
                }
            }
        } else {
            if poly.last().is_none_or(|last| last.distance(a) > 1e-5) {
                poly.push(a);
            }
            if poly.last().is_none_or(|last| last.distance(b) > 1e-5) {
                poly.push(b);
            }
        }
    };

    // Build boustrophedon continuous polylines, retracting across holes/concavities.
    let mut chains: Vec<Vec<DVec2>> = Vec::new();
    let mut current_poly: Vec<DVec2> = Vec::new();
    let mut prev_row: Option<usize> = None;

    for (row_idx, row_segs) in rows.into_iter().enumerate() {
        if row_segs.is_empty() {
            if current_poly.len() >= 2 {
                chains.push(std::mem::take(&mut current_poly));
            } else {
                current_poly.clear();
            }
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
                if let Some(&last_pt) = current_poly.last() {
                    if let Some(max_spacing) = max_subdivide_m {
                        let link_len = last_pt.distance(s_start);
                        if link_len > max_spacing {
                            push_segment(&mut current_poly, last_pt, s_start);
                        }
                    }
                }
                push_segment(&mut current_poly, s_start, s_end);
            } else {
                if current_poly.len() >= 2 {
                    chains.push(std::mem::take(&mut current_poly));
                } else {
                    current_poly.clear();
                }
                push_segment(&mut current_poly, s_start, s_end);
            }
        }
        prev_row = Some(row_idx);
    }

    if current_poly.len() >= 2 {
        chains.push(current_poly);
    }

    chains
}

/// Binary-searches the largest inset yielding a non-empty medial core so thin glyphs/slivers
/// are traced cleanly along their centerlines at `d = d_min` without overspray or vanishing.
fn plan_thin_sliver_medial(
    mp: &MultiPolygon<f64>,
    opacity: f32,
    cap: NozzleCap,
    config: &PlannerConfig,
    depth: usize,
    out: &mut Vec<JobOp>,
) {
    let (d_min, d_perim, _) = cap_standoff_bounds(cap, config);
    let w_min = spot_width_cap(cap, d_min) as f64;
    let max_inset = 0.45 * (spot_width_cap(cap, d_perim) as f64);
    let medial_flow = flow_scale_for_target_alpha(opacity, 3.15);

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
                emit_uniform_polyline_stroke(&pts, d_min, medial_flow, cap, config, out);
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
            emit_uniform_polyline_stroke(&pts, d_min, medial_flow, cap, config, out);
        }

        // If the shape had mixed thicknesses (e.g., thick stem + thin serif), paint uncovered portions.
        if depth < 2 {
            let covered = core.buffer(0.48 * w_min.max(best_r * 2.0));
            let poly_mp = MultiPolygon::new(vec![poly.clone()]);
            let remainder = filter_valid_mp(poly_mp.difference_with_fill_rule(&covered, GeoFillRule::EvenOdd));
            if !remainder.0.is_empty() && remainder.unsigned_area() > 0.15 * area {
                plan_thin_sliver_medial(&remainder, opacity, cap, config, depth + 1, out);
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

#[inline]
fn emit_uniform_polyline_stroke(
    raw_points: &[DVec2],
    standoff: f32,
    opacity_scale: f32,
    cap: NozzleCap,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    let pts: Vec<(DVec2, f32)> = raw_points.iter().map(|&p| (p, opacity_scale)).collect();
    emit_polyline_stroke(&pts, standoff, cap, config, out);
}

/// Wraps a continuous 2D path with per-point opacity scales and tangential lead-in/lead-out
/// (`valve_on = false`), emitting a [`JobOp::PaintStroke`].
fn emit_polyline_stroke(
    raw_points: &[(DVec2, f32)],
    standoff: f32,
    cap: NozzleCap,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    // Clean near-duplicate consecutive vertices while preserving closure if closed.
    let mut pts: Vec<(DVec2, f32)> = Vec::with_capacity(raw_points.len());
    for &(p, op) in raw_points {
        if pts.last().is_none_or(|last| last.0.distance(p) > 1e-5) {
            pts.push((p, op.clamp(0.015, 1.0)));
        }
    }
    if pts.len() < 2 {
        return;
    }

    let total_len: f64 = pts.windows(2).map(|w| w[0].0.distance(w[1].0)).sum();
    if total_len < 1e-5 {
        return;
    }

    let start_dir = (pts[1].0 - pts[0].0).normalize_or_zero();
    let end_dir = (pts[pts.len() - 1].0 - pts[pts.len() - 2].0).normalize_or_zero();
    if start_dir.length_squared() < 0.5 || end_dir.length_squared() < 0.5 {
        return;
    }

    let feed_speed = compensated_feed_speed_cap(config, cap, standoff);
    let lead = config.lead_length as f64;
    let lead_in_pt = pts[0].0 - start_dir * lead;
    let lead_out_pt = pts[pts.len() - 1].0 + end_dir * lead;

    let make_wp = |p: DVec2, opacity_scale: f32, valve_on: bool| StrokeWaypoint {
        pos: Vec3::new(p.x as f32, p.y as f32, 0.0),
        normal: Vec3::Z,
        standoff,
        feed_speed,
        valve_on,
        opacity_scale,
        cap,
    };

    let mut waypoints = Vec::with_capacity(pts.len() + 2);
    waypoints.push(make_wp(lead_in_pt, pts[0].1, false));
    for &(p, op) in &pts {
        waypoints.push(make_wp(p, op, true));
    }
    waypoints.push(make_wp(lead_out_pt, pts[pts.len() - 1].1, false));

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
            gradient_meta: None,
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
        // 8mm wide x 12cm tall sliver selects Skinny cap and is wide enough for Skinny perimeter (0.038m)
        let poly_8mm = Polygon::new(
            LineString::new(vec![
                Coord { x: -0.004, y: 1.49 },
                Coord { x: 0.004, y: 1.49 },
                Coord { x: 0.004, y: 1.61 },
                Coord { x: -0.004, y: 1.61 },
                Coord { x: -0.004, y: 1.49 },
            ]),
            Vec::new(),
        );
        // 2.4mm wide x 12cm tall sliver (< 2 * 0.45 * w_skinny(0.038) ≈ 3.1mm) falls back to medial at d_min = 0.035m
        let poly_2_4mm = Polygon::new(
            LineString::new(vec![
                Coord { x: -0.0012, y: 1.49 },
                Coord { x: 0.0012, y: 1.49 },
                Coord { x: 0.0012, y: 1.61 },
                Coord { x: -0.0012, y: 1.61 },
                Coord { x: -0.0012, y: 1.49 },
            ]),
            Vec::new(),
        );
        let cfg = PlannerConfig::default();
        for (poly, expected_d) in [(poly_8mm, 0.038_f32), (poly_2_4mm, 0.035_f32)] {
            let prims = vec![NormalizedPrimitive {
                id: 0,
                color_srgb: [255, 255, 255],
                opacity: 1.0,
                geometry: PrimitiveGeom::FilledPolygon(MultiPolygon::new(vec![poly])),
                gradient_meta: None,
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
            assert!(!strokes.is_empty(), "thin sliver must not vanish");
            assert_eq!(strokes[0][0].cap, NozzleCap::Skinny);
            assert!(
                (strokes[0][0].standoff - expected_d).abs() < 1e-6,
                "expected standoff {expected_d}, got {}",
                strokes[0][0].standoff
            );
        }
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
                        let (d_min, _, d_max) = cap_standoff_bounds(wp.cap, &cfg);
                        assert!(
                            wp.standoff >= d_min - 1e-6 && wp.standoff <= d_max + 1e-6,
                            "standoff {} out of {:?} bounds [{d_min}, {d_max}]",
                            wp.standoff,
                            wp.cap
                        );
                        let (v_min, v_max) = match wp.cap {
                            NozzleCap::Fat => (cfg.min_feed_speed, cfg.max_feed_speed),
                            NozzleCap::Skinny => (0.10, 0.42),
                        };
                        assert!(
                            wp.feed_speed >= v_min - 1e-6 && wp.feed_speed <= v_max + 1e-6,
                            "feed_speed {} out of {:?} bounds [{v_min}, {v_max}]",
                            wp.feed_speed,
                            wp.cap
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
            gradient_meta: None,
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
            gradient_meta: None,
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

    #[test]
    fn gradient_bands_emit_zero_perimeter_strokes_and_airbrushed_opacity() {
        let cfg = PlannerConfig::default();
        // Linear gradient middle band with both fade_in_s and fade_out_s.
        let band_width = 0.20;
        let prim_linear = NormalizedPrimitive {
            id: 0,
            color_srgb: [214, 52, 88],
            opacity: 1.0,
            geometry: PrimitiveGeom::FilledPolygon(square_poly(0.0, 1.55, 0.25)),
            gradient_meta: Some(GradientFillMeta::Linear {
                origin: DVec2::new(0.0, 1.80),
                dir: DVec2::new(0.0, -1.0),
                band_width,
                fade_in_s: Some((0.55 * band_width, 1.18 * band_width)),
                fade_out_s: Some((2.04 * band_width, 2.20 * band_width)),
            }),
        };
        let palette = quantize_palette(&[prim_linear.clone()], 16, 42);
        let batches = batch_primitives(&[prim_linear.clone()], &palette, &BatchConfig::default());
        let job = plan_job(&[prim_linear], &palette, &batches, &cfg);

        let strokes: Vec<&Vec<StrokeWaypoint>> = job
            .ops
            .iter()
            .filter_map(|op| match op {
                JobOp::PaintStroke { points_canvas_3d } => Some(points_canvas_3d),
                _ => None,
            })
            .collect();
        assert!(!strokes.is_empty());
        let mut min_op = f32::INFINITY;
        let mut max_op = 0.0_f32;
        for wps in &strokes {
            for wp in *wps {
                assert_eq!(wp.cap, NozzleCap::Fat);
                assert!(
                    wp.standoff >= 0.12 - 1e-6 && wp.standoff <= 0.25 + 1e-6,
                    "gradient must emit zero d_perimeter=0.05 perimeter strokes, got standoff={}",
                    wp.standoff
                );
                min_op = min_op.min(wp.opacity_scale);
                max_op = max_op.max(wp.opacity_scale);
            }
        }
        assert!(min_op < 0.25 && max_op > 0.85, "expected smooth fade modulation: min={min_op}, max={max_op}");

        // Radial gradient band must subdivide scanlines to <= 0.015m and modulate radially.
        let prim_radial = NormalizedPrimitive {
            id: 0,
            color_srgb: [0, 128, 255],
            opacity: 1.0,
            geometry: PrimitiveGeom::FilledPolygon(square_poly(0.0, 1.55, 0.20)),
            gradient_meta: Some(GradientFillMeta::Radial {
                center: DVec2::new(0.0, 1.55),
                band_width: 0.15,
                fade_in_r: Some((0.05, 0.12)),
                fade_out_r: Some((0.18, 0.24)),
            }),
        };
        let palette_r = quantize_palette(&[prim_radial.clone()], 16, 42);
        let batches_r = batch_primitives(&[prim_radial.clone()], &palette_r, &BatchConfig::default());
        let job_r = plan_job(&[prim_radial], &palette_r, &batches_r, &cfg);
        let strokes_r: Vec<&Vec<StrokeWaypoint>> = job_r
            .ops
            .iter()
            .filter_map(|op| match op {
                JobOp::PaintStroke { points_canvas_3d } => Some(points_canvas_3d),
                _ => None,
            })
            .collect();
        assert!(!strokes_r.is_empty());
        let mut r_min_op = f32::INFINITY;
        let mut r_max_op = 0.0_f32;
        for wps in &strokes_r {
            let active = &wps[1..wps.len() - 1];
            for pair in active.windows(2) {
                let d = pair[0].pos.distance(pair[1].pos);
                assert!(d <= 0.0155, "radial scanline spacing {d} exceeded 0.015m");
            }
            for wp in active {
                assert_eq!(wp.cap, NozzleCap::Fat);
                assert!(wp.standoff >= 0.12 - 1e-6 && wp.standoff <= 0.25 + 1e-6);
                r_min_op = r_min_op.min(wp.opacity_scale);
                r_max_op = r_max_op.max(wp.opacity_scale);
            }
        }
        assert!(r_min_op < 0.30 && r_max_op > 0.85, "expected radial opacity feathering: min={r_min_op}, max={r_max_op}");

        // Verify feed speed depends ONLY on (cap, standoff), never divided by opacity.
        let v_fat_1 = compensated_feed_speed_cap(&cfg, NozzleCap::Fat, 0.12);
        let v_skinny = compensated_feed_speed_cap(&cfg, NozzleCap::Skinny, 0.038);
        assert!((v_fat_1 - 0.040 / 0.12).abs() < 1e-5);
        assert!((v_skinny - (0.012_f32 / 0.038_f32).clamp(0.10, 0.42)).abs() < 1e-5);
    }

    #[test]
    fn dual_cap_and_swap_cap_scheduling_on_poster() {
        let cfg = PlannerConfig::default();
        let prims = ingest_svg_file(
            Path::new("assets/examples/02_street_art_poster.svg"),
            &IngestConfig::default(),
        )
        .expect("ingest 02_street_art_poster");
        let palette = quantize_palette(&prims, 16, 42);
        let batches = batch_primitives(&prims, &palette, &BatchConfig::default());
        let job = plan_job(&prims, &palette, &batches, &cfg);

        // Must emit SwapCap { cap: NozzleCap::Skinny } for small text & thin strokes.
        let skinny_swaps = job
            .ops
            .iter()
            .filter(|op| matches!(op, JobOp::SwapCap { cap: NozzleCap::Skinny }))
            .count();
        assert!(
            skinny_swaps >= 1,
            "expected SwapCap {{ cap: Skinny }} on 02_street_art_poster.svg"
        );

        // Verify at most 1 SwapCap per PickCan..ReturnCan batch and active cap matches every stroke.
        let max_slot = batches.iter().map(|b| b.slot_index + 1).max().unwrap_or(0);
        let mut slot_caps = vec![NozzleCap::Fat; max_slot];
        let mut held_slot: Option<usize> = None;
        let mut swaps_in_batch = 0usize;

        for op in &job.ops {
            match op {
                JobOp::PickCan { slot } => {
                    assert!(held_slot.is_none());
                    held_slot = Some(*slot);
                    swaps_in_batch = 0;
                }
                JobOp::SwapCap { cap } => {
                    let s = held_slot.expect("SwapCap must occur while holding a can");
                    swaps_in_batch += 1;
                    assert!(swaps_in_batch <= 1, "at most 1 SwapCap per batch");
                    assert_ne!(slot_caps[s], *cap, "redundant SwapCap to same cap");
                    slot_caps[s] = *cap;
                }
                JobOp::PaintStroke { points_canvas_3d } => {
                    let s = held_slot.expect("PaintStroke must occur while holding a can");
                    let expected = slot_caps[s];
                    for wp in points_canvas_3d {
                        assert_eq!(wp.cap, expected, "stroke cap mismatch against active slot cap");
                    }
                }
                JobOp::ReturnCan { slot } => {
                    assert_eq!(held_slot, Some(*slot));
                    held_slot = None;
                }
                _ => {}
            }
        }
        assert!(held_slot.is_none());
    }
}
