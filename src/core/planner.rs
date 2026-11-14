use std::collections::HashMap;

use geo::algorithm::area::Area;
use geo::algorithm::bool_ops::{BooleanOps, FillRule as GeoFillRule};
use geo::algorithm::bounding_rect::BoundingRect;
use geo::algorithm::buffer::{Buffer, BufferStyle, LineJoin};
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

fn isoperimetric_quotient(poly: &Polygon<f64>) -> f64 {
    let perim_len = poly.exterior().length(&geo::Euclidean);
    if perim_len <= 1e-9 {
        return 0.0;
    }
    let area = poly.unsigned_area();
    (4.0 * std::f64::consts::PI * area) / (perim_len * perim_len)
}

fn ring_needs_hole_pass(ext: &LineString<f64>, hole: &LineString<f64>, max_gap: f64) -> bool {
    let max_gap_sq = max_gap * max_gap;
    hole.coords().any(|hc| {
        let hp = DVec2::new(hc.x, hc.y);
        let mut min_d_sq = f64::INFINITY;
        for w in ext.0.windows(2) {
            let a = DVec2::new(w[0].x, w[0].y);
            let b = DVec2::new(w[1].x, w[1].y);
            let ab = b - a;
            let len_sq = ab.length_squared();
            let t = if len_sq > 1e-18 {
                ((hp - a).dot(ab) / len_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            min_d_sq = min_d_sq.min((a + ab * t).distance_squared(hp));
        }
        min_d_sq > max_gap_sq
    })
}

fn emit_medial_core_between(
    poly: &Polygon<f64>,
    lo_offset: f64,
    hi_offset: f64,
    standoff: f32,
    spot_w: f64,
    flow: f32,
    cap: NozzleCap,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    let mut lo = lo_offset;
    let mut hi = hi_offset;
    let mut best_core = poly.buffer(-lo);
    if best_core.0.is_empty() || best_core.unsigned_area() <= 1e-9 {
        return;
    }
    for _ in 0..8 {
        let mid = 0.5 * (lo + hi);
        if mid - lo < 1e-5 {
            break;
        }
        let inset = poly.buffer(-mid);
        if !inset.0.is_empty() && inset.unsigned_area() > 1e-9 {
            best_core = inset;
            lo = mid;
        } else {
            hi = mid;
        }
    }

    for core_poly in &best_core.0 {
        if core_poly.exterior().length(&geo::Euclidean) < 1e-5 {
            continue;
        }
        emit_ring_stroke(core_poly.exterior(), standoff, flow, cap, config, out);
        for hole in core_poly.interiors() {
            if hole.length(&geo::Euclidean) >= 1e-5
                && ring_needs_hole_pass(core_poly.exterior(), hole, 0.35 * spot_w)
            {
                emit_ring_stroke(hole, standoff, flow, cap, config, out);
            }
        }
    }
}

/// Detects narrower bridges or sub-lobes of `poly` that collapsed between `prev_offset` and
/// `next_offset` while wider regions survived in `next_mp`, and emits a centered medial pass
/// for each collapsed component at a safe standoff bounded by `prev_d`.
#[allow(clippy::too_many_arguments)]
fn emit_partially_collapsed_bridges(
    poly: &Polygon<f64>,
    prev_offset: f64,
    next_offset: f64,
    next_mp: &MultiPolygon<f64>,
    prev_d: f32,
    prev_w: f64,
    fill_flow: f32,
    cap: NozzleCap,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    let probe_offset = prev_offset + 0.25 * prev_w;
    if probe_offset >= next_offset - 1e-5 {
        return;
    }
    let probe_mp = filter_valid_mp(poly.buffer(-probe_offset));
    if probe_mp.0.is_empty() {
        return;
    }

    let cover_margin = (next_offset - probe_offset) + 0.002;
    let cover_style = BufferStyle::new(cover_margin).line_join(LineJoin::Miter(0.35));
    let covered_by_next = filter_valid_mp(next_mp.buffer_with_style(cover_style));
    let collapsed = filter_valid_mp(probe_mp.difference(&covered_by_next));

    let tan_theta = cap.cone_half_angle_rad().tan() as f64;
    let (d_min, _, _) = cap_standoff_bounds(cap, config);
    let d_safe_probe = (probe_offset / (0.48 * 2.0 * tan_theta)) as f32;
    let d_bridge = prev_d.min(d_safe_probe.max(d_min));
    let min_mean_half_width = 0.001_f64.min(0.05 * prev_w);

    for cp in &collapsed.0 {
        let area = cp.unsigned_area();
        if area < 2e-5 {
            continue;
        }
        let perim = cp.exterior().length(&geo::Euclidean);
        if perim <= 1e-5 || (2.0 * area / perim) < min_mean_half_width {
            continue;
        }
        emit_medial_core_between(
            cp,
            0.0,
            next_offset - probe_offset,
            d_bridge,
            prev_w,
            fill_flow,
            cap,
            config,
            out,
        );
    }
}

/// Peels concentric contour-offset rings inward from the perimeter pass, bridging any
/// narrow residual centerlines with a medial pass and handing off large hole-free cores
/// to boustrophedon hatching with tight overlap.
#[allow(clippy::too_many_arguments)]
fn plan_concentric_infill(
    poly: &Polygon<f64>,
    perim_mp: &MultiPolygon<f64>,
    inset_perim: f64,
    d_perim: f32,
    w_perim: f64,
    max_concentric_rings: usize,
    opacity: f32,
    cap: NozzleCap,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    let fill_flow = flow_scale_for_target_alpha(opacity, 3.6);
    let (d_fill_min, d_fill_max) = match cap {
        NozzleCap::Fat => (0.08_f32, 0.20_f32),
        NozzleCap::Skinny => (0.045_f32, 0.12_f32),
    };
    let tan_theta = cap.cone_half_angle_rad().tan() as f64;

    let mut prev_offset = inset_perim;
    let mut prev_d = d_perim;
    let mut prev_w = w_perim;
    let mut rings_emitted = 0usize;
    const MAX_SAFETY_RINGS: usize = 48;

    for ring_idx in 1..=MAX_SAFETY_RINGS {
        let step_ratio = if ring_idx == 1 { 0.42 } else { 0.50 };
        let offset = prev_offset + step_ratio * prev_w;
        let ring_mp = filter_valid_mp(poly.buffer(-offset));

        if ring_mp.0.is_empty() {
            let half_gap_offset = prev_offset + 0.275 * prev_w;
            emit_medial_core_between(
                poly,
                half_gap_offset,
                offset,
                prev_d,
                prev_w,
                fill_flow,
                cap,
                config,
                out,
            );
            return;
        }

        emit_partially_collapsed_bridges(
            poly,
            prev_offset,
            offset,
            &ring_mp,
            prev_d,
            prev_w,
            fill_flow,
            cap,
            config,
            out,
        );

        let d_safe_max = (offset / (0.48 * 2.0 * tan_theta)) as f32;
        let d_k = if d_safe_max >= d_fill_min {
            d_safe_max.clamp(d_fill_min, d_fill_max)
        } else {
            d_safe_max.clamp(d_perim, d_fill_max)
        };
        let w_k = spot_width_cap(cap, d_k) as f64;

        for p in &ring_mp.0 {
            emit_ring_stroke(p.exterior(), d_k, fill_flow, cap, config, out);
            for hole in p.interiors() {
                emit_ring_stroke(hole, d_k, fill_flow, cap, config, out);
            }
        }

        prev_offset = offset;
        prev_d = d_k;
        prev_w = w_k;
        rings_emitted += 1;

        if rings_emitted >= max_concentric_rings {
            let handoff_offset = prev_offset + 0.32 * prev_w;
            let remaining = filter_valid_mp(poly.buffer(-handoff_offset));
            if remaining.0.is_empty() {
                let half_gap_offset = prev_offset + 0.275 * prev_w;
                emit_medial_core_between(
                    poly,
                    half_gap_offset,
                    handoff_offset,
                    prev_d,
                    prev_w,
                    fill_flow,
                    cap,
                    config,
                    out,
                );
                return;
            }
            if remaining.0.iter().all(|p| p.interiors().is_empty()) {
                emit_partially_collapsed_bridges(
                    poly,
                    prev_offset,
                    handoff_offset,
                    &remaining,
                    prev_d,
                    prev_w,
                    fill_flow,
                    cap,
                    config,
                    out,
                );
                let max_safe_d = ((handoff_offset / (0.45 * 2.0 * tan_theta)) as f32).max(d_perim);
                plan_boustrophedon_hatch_bounded(
                    &remaining,
                    perim_mp,
                    opacity,
                    cap,
                    config,
                    Some(max_safe_d),
                    out,
                );
                return;
            }
        }
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

    // 2. Concentric contour-offset infill (full pocket for holed/circular shapes, 1 seam-bridging
    //    peel + medial/boustrophedon handoff for general elongated/polygonal shapes).
    let use_full_concentric = !poly.interiors().is_empty() || isoperimetric_quotient(poly) >= 0.60;
    let max_concentric_rings = if use_full_concentric { 4 } else { 1 };
    plan_concentric_infill(
        poly,
        &perim_mp,
        inset_perim,
        d_perim,
        w_perim,
        max_concentric_rings,
        opacity,
        cap,
        config,
        out,
    );
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
            let step_over = (0.54 * spot_width_cap(NozzleCap::Fat, d_fill) as f64).max(0.012);
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
            let step_over = (0.54 * spot_width_cap(NozzleCap::Fat, d_fill) as f64).max(0.012);

            for poly in &mp.0 {
                if poly.unsigned_area() < MIN_AREA_M2 {
                    continue;
                }
                let (r_min, r_max) = radial_bounds(poly, *center);
                if r_max <= r_min + 1e-5 {
                    continue;
                }

                let mut radii: Vec<f64> = Vec::new();
                let r_start = r_min + 0.35 * step_over;
                if r_start >= r_max {
                    radii.push((0.5 * (r_min + r_max)).max(0.005));
                } else {
                    let mut r_cur = r_start;
                    while r_cur < r_max - 1e-5 {
                        radii.push(r_cur);
                        r_cur += step_over;
                    }
                    if let Some(&last_r) = radii.last() {
                        if r_max - last_r > 0.55 * step_over {
                            radii.push((r_max - 0.30 * step_over).max(last_r + 1e-4));
                        }
                    }
                }

                let poly_mp = MultiPolygon::new(vec![poly.clone()]);
                let mut emitted_any = false;
                for r in radii {
                    if r < 1e-4 {
                        continue;
                    }
                    let n_seg =
                        ((2.0 * std::f64::consts::PI * r) / 0.012).ceil().max(16.0) as usize;
                    let mut coords = Vec::with_capacity(n_seg + 1);
                    for k in 0..=n_seg {
                        let theta = (k % n_seg) as f64 * (2.0 * std::f64::consts::PI / n_seg as f64);
                        coords.push(Coord {
                            x: center.x + r * theta.cos(),
                            y: center.y + r * theta.sin(),
                        });
                    }
                    let circle_mls = MultiLineString::new(vec![LineString::new(coords)]);
                    let clipped = poly_mp.clip(&circle_mls, false);
                    let chains = merge_clipped_circle_chains(clipped.0);
                    for chain in chains {
                        if chain.len() < 2 {
                            continue;
                        }
                        let pts_with_op: Vec<(DVec2, f32)> = chain
                            .into_iter()
                            .map(|pt| {
                                let dist = pt.distance(*center);
                                let w_in = fade_in_r
                                    .map_or(1.0, |(a, b)| 0.04 + 0.96 * smoothstep(a, b, dist));
                                let w_out = fade_out_r
                                    .map_or(1.0, |(a, b)| 1.0 - 0.55 * smoothstep(a, b, dist));
                                let target_alpha =
                                    (opacity * (w_in * w_out) as f32).clamp(0.02, 1.0);
                                (pt, flow_scale_for_target_alpha(target_alpha, 4.2))
                            })
                            .collect();
                        let prev_len = out.len();
                        emit_polyline_stroke(&pts_with_op, d_fill, NozzleCap::Fat, config, out);
                        if out.len() > prev_len {
                            emitted_any = true;
                        }
                    }
                }

                // Degenerate radial sliver fallback: preserve coverage with dense subdivided scanlines.
                if !emitted_any {
                    if let Some(bbox) = poly.bounding_rect() {
                        let (step_dir, scan_dir) = if bbox.width() >= bbox.height() {
                            (DVec2::Y, DVec2::X)
                        } else {
                            (DVec2::X, DVec2::Y)
                        };
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
                                    let dist = pt.distance(*center);
                                    let w_in = fade_in_r
                                        .map_or(1.0, |(a, b)| 0.04 + 0.96 * smoothstep(a, b, dist));
                                    let w_out = fade_out_r
                                        .map_or(1.0, |(a, b)| 1.0 - 0.55 * smoothstep(a, b, dist));
                                    let target_alpha =
                                        (opacity * (w_in * w_out) as f32).clamp(0.02, 1.0);
                                    (pt, flow_scale_for_target_alpha(target_alpha, 4.2))
                                })
                                .collect();
                            emit_polyline_stroke(&pts_with_op, d_fill, NozzleCap::Fat, config, out);
                        }
                    }
                }
            }
        }
    }
}

fn min_dist_to_ring(pt: DVec2, ring: &LineString<f64>) -> f64 {
    let mut min_d_sq = f64::INFINITY;
    for w in ring.0.windows(2) {
        let a = DVec2::new(w[0].x, w[0].y);
        let b = DVec2::new(w[1].x, w[1].y);
        let ab = b - a;
        let len_sq = ab.length_squared();
        let t = if len_sq > 1e-18 {
            ((pt - a).dot(ab) / len_sq).clamp(0.0, 1.0)
        } else {
            0.0
        };
        min_d_sq = min_d_sq.min((a + ab * t).distance_squared(pt));
    }
    min_d_sq.sqrt()
}

fn radial_bounds(poly: &Polygon<f64>, center: DVec2) -> (f64, f64) {
    let r_max = poly
        .exterior()
        .coords()
        .map(|c| DVec2::new(c.x, c.y).distance(center))
        .fold(0.0_f64, f64::max);
    let r_min = if poly.contains(&geo::Point::new(center.x, center.y)) {
        0.0
    } else {
        let mut d = min_dist_to_ring(center, poly.exterior());
        for hole in poly.interiors() {
            d = d.min(min_dist_to_ring(center, hole));
        }
        d
    };
    (r_min, r_max)
}

fn merge_clipped_circle_chains(lines: Vec<LineString<f64>>) -> Vec<Vec<DVec2>> {
    let mut chains: Vec<Vec<DVec2>> = lines
        .into_iter()
        .filter_map(|ls| {
            let pts: Vec<DVec2> = ls.coords().map(|c| DVec2::new(c.x, c.y)).collect();
            (pts.len() >= 2).then_some(pts)
        })
        .collect();
    if chains.len() < 2 {
        return chains;
    }

    let mut merged = true;
    while merged {
        merged = false;
        'outer: for i in 0..chains.len() {
            if chains[i].len() >= 3 && chains[i].first().unwrap().distance(*chains[i].last().unwrap()) < 1e-4 {
                continue;
            }
            for j in 0..chains.len() {
                if i == j {
                    continue;
                }
                if chains[j].len() >= 3 && chains[j].first().unwrap().distance(*chains[j].last().unwrap()) < 1e-4 {
                    continue;
                }
                let end_i = *chains[i].last().unwrap();
                let start_j = *chains[j].first().unwrap();
                if end_i.distance(start_j) < 1e-4 {
                    let tail = chains.remove(j);
                    let dst_idx = if j < i { i - 1 } else { i };
                    chains[dst_idx].extend(tail.into_iter().skip(1));
                    merged = true;
                    break 'outer;
                }
            }
        }
    }
    chains
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

#[allow(dead_code)]
fn plan_boustrophedon_hatch(
    hatch_mp: &MultiPolygon<f64>,
    containment_mp: &MultiPolygon<f64>,
    opacity: f32,
    cap: NozzleCap,
    config: &PlannerConfig,
    out: &mut Vec<JobOp>,
) {
    plan_boustrophedon_hatch_bounded(hatch_mp, containment_mp, opacity, cap, config, None, out);
}

fn plan_boustrophedon_hatch_bounded(
    hatch_mp: &MultiPolygon<f64>,
    containment_mp: &MultiPolygon<f64>,
    opacity: f32,
    cap: NozzleCap,
    config: &PlannerConfig,
    max_d_fill: Option<f32>,
    out: &mut Vec<JobOp>,
) {
    let Some(bbox) = hatch_mp.bounding_rect() else {
        return;
    };

    let width = bbox.width();
    let height = bbox.height();
    let minor_dim = width.min(height) as f32;

    let (d_fill_min, d_fill_max_cap) = match cap {
        NozzleCap::Fat => (0.08_f32, 0.22_f32),
        NozzleCap::Skinny => (0.045_f32, 0.14_f32),
    };
    let d_fill_max = max_d_fill
        .map(|d| d.clamp(d_fill_min, d_fill_max_cap))
        .unwrap_or(d_fill_max_cap);
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
    if chains.is_empty() && hatch_mp.unsigned_area() >= MIN_AREA_M2 {
        plan_thin_sliver_medial(hatch_mp, opacity, cap, config, 0, out);
        return;
    }
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

/// Computes per-vertex arc-length-smoothed forward unit 2D tangents over a ~0.025m radius window
/// so sharp corners, closed contour loops, and turnarounds transition smoothly without wrist snaps.
fn compute_smoothed_tangents(
    pts: &[(DVec2, f32)],
    window_radius: f64,
    is_closed: bool,
) -> Vec<DVec2> {
    let n = pts.len();
    let mut s = Vec::with_capacity(n);
    s.push(0.0_f64);
    for i in 1..n {
        s.push(s[i - 1] + pts[i].0.distance(pts[i - 1].0));
    }
    let total_len = s[n - 1];
    let eff_radius = if is_closed {
        window_radius.min(0.35 * total_len).max(1e-6)
    } else {
        window_radius
    };

    let seg_dirs: Vec<DVec2> = pts
        .windows(2)
        .map(|w| (w[1].0 - w[0].0).normalize_or_zero())
        .collect();

    let mut tangents = Vec::with_capacity(n);
    for i in 0..n {
        let si = s[i];
        let lo = si - eff_radius;
        let hi = si + eff_radius;
        let mut acc = DVec2::ZERO;

        let shifts: &[f64] = if is_closed {
            &[-total_len, 0.0, total_len]
        } else {
            &[0.0]
        };

        for &shift in shifts {
            for (j, &dir) in seg_dirs.iter().enumerate() {
                let a = s[j] + shift;
                let b = s[j + 1] + shift;
                if b <= lo || a >= hi {
                    continue;
                }
                let u0 = (a - si).clamp(-eff_radius, eff_radius);
                let u1 = (b - si).clamp(-eff_radius, eff_radius);
                if u1 <= u0 {
                    continue;
                }
                // Exact integral of tent kernel (1 - |u|/R) over [u0, u1].
                let weight = (u1 - u0) - (u1 * u1.abs() - u0 * u0.abs()) / (2.0 * eff_radius);
                acc += dir * weight.max(0.0);
            }
        }

        let unit = if acc.length_squared() > 1e-10 {
            acc.normalize()
        } else if is_closed && (i == 0 || i == n - 1) {
            let sum = seg_dirs[seg_dirs.len() - 1] + seg_dirs[0];
            if sum.length_squared() > 1e-10 {
                sum.normalize()
            } else {
                seg_dirs[0]
            }
        } else if i == 0 {
            seg_dirs[0]
        } else if i == n - 1 {
            seg_dirs[seg_dirs.len() - 1]
        } else {
            let sum = seg_dirs[i - 1] + seg_dirs[i];
            if sum.length_squared() > 1e-10 {
                sum.normalize()
            } else {
                seg_dirs[i - 1]
            }
        };
        tangents.push(unit);
    }
    tangents
}

/// Computes a natural human drag-aligned spray aim vector from smoothed 2D stroke tangent `t_xy`,
/// clamping upward pitch cleanly to keep the can body clear of the canvas wall (`Z = 0`).
fn compute_drag_aim_dir(t_xy: DVec2, total_len: f64, standoff: f32, cap: NozzleCap) -> Vec3 {
    let (target_theta, ramp_len, min_cos_phi) = match cap {
        NozzleCap::Fat => (0.235_f64, 0.045_f64, 0.94_f64),
        NozzleCap::Skinny => (0.078_f64, 0.025_f64, 0.995_f64),
    };
    let len_scale = (total_len / ramp_len).clamp(0.0, 1.0);
    let theta = target_theta * len_scale;
    let sin_t = theta.sin();

    // Gentle downward cant (~1.4 deg) when comfortable, scaled with stroke length.
    let downward_cant = 0.025_f64 * len_scale;
    let mut dx = t_xy.x * sin_t;
    let mut dy = t_xy.y * sin_t - downward_cant;

    // Enforce cap-specific cone/stiffness limit (cos_phi >= 0.995 for Skinny, >= 0.94 for Fat).
    let max_r_xy = (1.0 - min_cos_phi * min_cos_phi).max(0.0).sqrt();
    let r_xy = dx.hypot(dy);
    if r_xy > max_r_xy && r_xy > 1e-12 {
        let scale = max_r_xy / r_xy;
        dx *= scale;
        dy *= scale;
    }

    // Critical clearance clamp: when tilted upward (aim_dir.y > 0), keep can top clear of Z = 0:
    // standoff - 0.105 * aim_dir.y.max(0.0) - 0.036 >= 0.020 (or clamp positive y to 0 when close).
    let max_up_y = ((standoff as f64 - 0.036 - 0.020) / 0.105 - 1e-5).max(0.0);
    if dy > max_up_y {
        dy = max_up_y;
    }

    let dz = -(1.0 - dx * dx - dy * dy).max(0.0).sqrt();
    Vec3::new(dx as f32, dy as f32, dz as f32).normalize_or(Vec3::NEG_Z)
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

    let is_closed = pts.len() >= 4 && pts[0].0.distance(pts[pts.len() - 1].0) < 1e-4;
    let tangents = compute_smoothed_tangents(&pts, 0.025, is_closed);
    let feed_speed = compensated_feed_speed_cap(config, cap, standoff);
    let lead = config.lead_length as f64;
    let lead_in_pt = pts[0].0 - start_dir * lead;
    let lead_out_pt = pts[pts.len() - 1].0 + end_dir * lead;

    let make_wp = |p: DVec2, t_xy: DVec2, opacity_scale: f32, valve_on: bool| StrokeWaypoint {
        pos: Vec3::new(p.x as f32, p.y as f32, 0.0),
        normal: Vec3::Z,
        aim_dir: compute_drag_aim_dir(t_xy, total_len, standoff, cap),
        standoff,
        feed_speed,
        valve_on,
        opacity_scale,
        cap,
    };

    let mut waypoints = Vec::with_capacity(pts.len() + 2);
    waypoints.push(make_wp(lead_in_pt, tangents[0], pts[0].1, false));
    for (&(p, op), &t_xy) in pts.iter().zip(tangents.iter()) {
        waypoints.push(make_wp(p, t_xy, op, true));
    }
    waypoints.push(make_wp(
        lead_out_pt,
        tangents[tangents.len() - 1],
        pts[pts.len() - 1].1,
        false,
    ));

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

    #[test]
    fn fat_hatch_stroke_emits_drag_aligned_aim_dir() {
        let cfg = PlannerConfig::default();
        let mut out = Vec::new();
        emit_uniform_polyline_stroke(
            &[DVec2::new(-0.20, 1.50), DVec2::new(0.20, 1.50)],
            0.14,
            1.0,
            NozzleCap::Fat,
            &cfg,
            &mut out,
        );
        let JobOp::PaintStroke { points_canvas_3d: wps } = &out[0] else {
            panic!("expected PaintStroke");
        };
        assert_eq!(wps.len(), 4);
        // Lead-in and lead-out smoothly inherit start/end tangents.
        assert!((wps[0].aim_dir - wps[1].aim_dir).length() < 1e-6);
        assert!((wps[3].aim_dir - wps[2].aim_dir).length() < 1e-6);

        for wp in wps {
            let xy_mag = wp.aim_dir.truncate().length();
            assert!(xy_mag > 0.15, "Fat drag tilt xy mag {xy_mag} should exceed 0.15");
            assert!(wp.aim_dir.z < -0.95, "Fat aim_dir.z {} should be < -0.95", wp.aim_dir.z);
            assert!(wp.aim_dir.x > 0.15, "aim_dir.x {} should align with +X drag", wp.aim_dir.x);
            assert!((wp.aim_dir.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn skinny_detail_stroke_emits_subtle_tilt_and_respects_clearance_clamp() {
        let cfg = PlannerConfig::default();
        let mut out = Vec::new();
        emit_uniform_polyline_stroke(
            &[DVec2::new(-0.04, 1.50), DVec2::new(0.04, 1.50)],
            0.045,
            1.0,
            NozzleCap::Skinny,
            &cfg,
            &mut out,
        );
        let JobOp::PaintStroke { points_canvas_3d: wps } = &out[0] else {
            panic!("expected PaintStroke");
        };
        for wp in wps {
            let xy_mag = wp.aim_dir.truncate().length();
            assert!(
                xy_mag > 0.04 && xy_mag <= 0.11,
                "Skinny xy tilt {xy_mag} out of (0.04, 0.11]"
            );
            assert!(wp.aim_dir.z < -0.99, "Skinny aim_dir.z {} should be < -0.99", wp.aim_dir.z);
            assert!(wp.aim_dir.z <= -0.995 + 1e-5, "Skinny cos_phi should stay >= 0.995");
        }

        // Upward close Skinny pass (standoff = 0.038m) must clamp positive aim_dir.y <= 0.0.
        let mut close_up = Vec::new();
        emit_uniform_polyline_stroke(
            &[DVec2::new(0.0, 1.45), DVec2::new(0.0, 1.55)],
            0.038,
            1.0,
            NozzleCap::Skinny,
            &cfg,
            &mut close_up,
        );
        let JobOp::PaintStroke { points_canvas_3d: close_wps } = &close_up[0] else {
            panic!("expected PaintStroke");
        };
        for wp in close_wps {
            assert!(
                wp.aim_dir.y <= 1e-6,
                "close upward Skinny pass must clamp positive aim_dir.y, got {}",
                wp.aim_dir.y
            );
            assert!(wp.nozzle_pos().z >= 0.03);
        }

        // Marginal standoff (0.065m) upward pass must satisfy standoff - 0.105 * y - 0.036 >= 0.020.
        let mut mid_up = Vec::new();
        emit_uniform_polyline_stroke(
            &[DVec2::new(0.0, 1.45), DVec2::new(0.0, 1.55)],
            0.065,
            1.0,
            NozzleCap::Fat,
            &cfg,
            &mut mid_up,
        );
        let JobOp::PaintStroke { points_canvas_3d: mid_wps } = &mid_up[0] else {
            panic!("expected PaintStroke");
        };
        for wp in mid_wps {
            let clearance = wp.standoff - 0.105 * wp.aim_dir.y.max(0.0) - 0.036;
            assert!(
                clearance >= 0.020 - 1e-5,
                "can-top clearance {clearance} fell below 0.020m"
            );
        }
    }

    #[test]
    fn sharp_90_deg_corner_stroke_smooths_aim_dir_transition() {
        let cfg = PlannerConfig::default();
        // 3-point sharp 90 deg corner (+X then +Y).
        let mut out_sparse = Vec::new();
        emit_uniform_polyline_stroke(
            &[
                DVec2::new(0.0, 1.40),
                DVec2::new(0.20, 1.40),
                DVec2::new(0.20, 1.60),
            ],
            0.12,
            1.0,
            NozzleCap::Fat,
            &cfg,
            &mut out_sparse,
        );
        let JobOp::PaintStroke { points_canvas_3d: wps } = &out_sparse[0] else {
            panic!("expected PaintStroke");
        };
        // Active corner vertex at index 2 should smoothly bisect (+X, +Y).
        let corner_xy = wps[2].aim_dir.truncate().normalize();
        assert!(
            corner_xy.x > 0.55 && corner_xy.y > 0.55,
            "corner vertex should blend both legs smoothly, got {:?}",
            wps[2].aim_dir
        );
        for pair in wps.windows(2) {
            let xy_a = pair[0].aim_dir.truncate().normalize();
            let xy_b = pair[1].aim_dir.truncate().normalize();
            let angle_deg = xy_a.dot(xy_b).clamp(-1.0, 1.0).acos().to_degrees();
            assert!(
                angle_deg < 55.0,
                "90-deg corner snapped by {angle_deg:.1} deg in a single step"
            );
        }

        // Dense 90 deg corner (8mm spacing) transitions gradually over the ~0.025m window.
        let mut dense_pts = Vec::new();
        for i in 0..=15 {
            dense_pts.push(DVec2::new(i as f64 * 0.008, 1.40));
        }
        for i in 1..=15 {
            dense_pts.push(DVec2::new(0.12, 1.40 + i as f64 * 0.008));
        }
        let mut out_dense = Vec::new();
        emit_uniform_polyline_stroke(&dense_pts, 0.12, 1.0, NozzleCap::Fat, &cfg, &mut out_dense);
        let JobOp::PaintStroke { points_canvas_3d: dense_wps } = &out_dense[0] else {
            panic!("expected PaintStroke");
        };
        for pair in dense_wps.windows(2) {
            let step_deg = pair[0]
                .aim_dir
                .dot(pair[1].aim_dir)
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            assert!(
                step_deg < 8.0,
                "dense corner aim_dir jumped {step_deg:.2} deg between adjacent vertices"
            );
            let xy_a = pair[0].aim_dir.truncate().normalize();
            let xy_b = pair[1].aim_dir.truncate().normalize();
            let xy_step_deg = xy_a.dot(xy_b).clamp(-1.0, 1.0).acos().to_degrees();
            assert!(
                xy_step_deg < 40.0,
                "dense corner XY tilt jumped {xy_step_deg:.2} deg between adjacent vertices"
            );
        }
    }

    #[test]
    fn poster_strokes_preserve_drag_tilt_and_nozzle_z_clearance() {
        let cfg = PlannerConfig::default();
        let prims = ingest_svg_file(
            Path::new("assets/examples/02_street_art_poster.svg"),
            &IngestConfig::default(),
        )
        .expect("ingest 02_street_art_poster");
        let palette = quantize_palette(&prims, 16, 42);
        let batches = batch_primitives(&prims, &palette, &BatchConfig::default());
        let job = plan_job(&prims, &palette, &batches, &cfg);

        let mut saw_tilted_fat = false;
        let mut saw_tilted_skinny = false;
        for op in &job.ops {
            if let JobOp::PaintStroke { points_canvas_3d } = op {
                for wp in points_canvas_3d {
                    assert!(
                        (wp.aim_dir.length() - 1.0).abs() < 1e-5,
                        "aim_dir must be normalized: {:?}",
                        wp.aim_dir
                    );
                    assert!(
                        wp.aim_dir.z <= -0.94 + 1e-5,
                        "aim_dir.z {} violated <= -0.94 bound",
                        wp.aim_dir.z
                    );
                    let noz = wp.nozzle_pos();
                    assert!(
                        noz.z >= 0.03,
                        "nozzle_pos.z {} fell below 0.03m (standoff={}, aim={:?})",
                        noz.z,
                        wp.standoff,
                        wp.aim_dir
                    );
                    if wp.standoff >= 0.056 {
                        let clearance = wp.standoff - 0.105 * wp.aim_dir.y.max(0.0) - 0.036;
                        assert!(
                            clearance >= 0.020 - 1e-5,
                            "can-top clearance {clearance} < 0.020 at standoff={}",
                            wp.standoff
                        );
                    } else {
                        assert!(
                            wp.aim_dir.y <= 1e-5,
                            "close pass (standoff={}) had positive aim_dir.y={}",
                            wp.standoff,
                            wp.aim_dir.y
                        );
                    }
                    let xy_mag = wp.aim_dir.truncate().length();
                    match wp.cap {
                        NozzleCap::Fat if xy_mag > 0.15 => saw_tilted_fat = true,
                        NozzleCap::Skinny if xy_mag > 0.04 && xy_mag <= 0.11 => {
                            saw_tilted_skinny = true;
                        }
                        _ => {}
                    }
                }
            }
        }
        assert!(saw_tilted_fat, "expected tilted Fat strokes in poster job");
        assert!(saw_tilted_skinny, "expected tilted Skinny strokes in poster job");
    }

    fn circle_ring(cx: f64, cy: f64, r: f64, n: usize, ccw: bool) -> LineString<f64> {
        let mut coords = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let k = if ccw { i % n } else { (n - (i % n)) % n };
            let theta = k as f64 * (2.0 * std::f64::consts::PI / n as f64);
            coords.push(Coord {
                x: cx + r * theta.cos(),
                y: cy + r * theta.sin(),
            });
        }
        LineString::new(coords)
    }

    fn rect_ring(x0: f64, y0: f64, x1: f64, y1: f64, ccw: bool) -> LineString<f64> {
        if ccw {
            LineString::new(vec![
                Coord { x: x0, y: y0 },
                Coord { x: x1, y: y0 },
                Coord { x: x1, y: y1 },
                Coord { x: x0, y: y1 },
                Coord { x: x0, y: y0 },
            ])
        } else {
            LineString::new(vec![
                Coord { x: x0, y: y0 },
                Coord { x: x0, y: y1 },
                Coord { x: x1, y: y1 },
                Coord { x: x1, y: y0 },
                Coord { x: x0, y: y0 },
            ])
        }
    }

    fn rasterize_job_to_canvas(
        prims: &[NormalizedPrimitive],
        cfg: &PlannerConfig,
        width: u32,
        height: u32,
    ) -> (JobPlan, crate::sim::canvas::CanvasPaintBuffer) {
        use crate::core::kinematics::KinematicChain;
        use crate::core::trajectory::CompiledTimeline;
        use crate::core::types::StudioConfig;
        use crate::sim::canvas::CanvasPaintBuffer;

        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);
        let palette = quantize_palette(prims, 16, 42);
        let batches = batch_primitives(prims, &palette, &BatchConfig::default());
        let job = plan_job(prims, &palette, &batches, cfg);
        let timeline = CompiledTimeline::compile(&job, &chain, &studio)
            .expect("trajectory compile for raster test");

        let mut buf = CanvasPaintBuffer::new(
            width,
            height,
            studio.canvas_center,
            studio.canvas_width,
            studio.canvas_height,
            42,
        );
        buf.deposit_timeline_interval(&timeline, &palette, 0.0, timeline.duration());
        (job, buf)
    }

    fn sample_coverage(
        buf: &crate::sim::canvas::CanvasPaintBuffer,
        x: f32,
        y: f32,
        paint_srgb: [u8; 3],
    ) -> f32 {
        let (px, py) = buf
            .world_to_pixel(glam::Vec2::new(x, y))
            .expect("sample point within canvas bounds");
        let idx = (py as usize) * (buf.width as usize) + (px as usize);
        let luma = |rgb: [f32; 3]| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        let gesso = luma(buf.gesso_linear);
        let paint = luma(crate::sim::canvas::srgb_u8_to_linear_rgb(paint_srgb));
        let denom = (gesso - paint).max(1e-5);
        ((gesso - luma(buf.linear_rgb[idx])) / denom).clamp(0.0, 1.0)
    }

    #[test]
    fn circular_annulus_uses_concentric_rings_with_solid_polar_coverage() {
        let cx = 0.0_f64;
        let cy = 1.55_f64;
        let r_out = 0.25_f64;
        let r_in = 0.20_f64;
        let paint_srgb = [15, 25, 45];
        let annulus = Polygon::new(
            circle_ring(cx, cy, r_out, 72, true),
            vec![circle_ring(cx, cy, r_in, 72, false)],
        );
        let prim = NormalizedPrimitive {
            id: 0,
            color_srgb: paint_srgb,
            opacity: 1.0,
            geometry: PrimitiveGeom::FilledPolygon(MultiPolygon::new(vec![annulus.clone()])),
            gradient_meta: None,
        };
        let cfg = PlannerConfig::default();

        // Compare op count against horizontal boustrophedon infill on the same annulus.
        let inset_old = 1.05 * spot_width_cap(NozzleCap::Fat, cfg.d_perimeter) as f64;
        let old_hatch_mp = filter_valid_mp(annulus.buffer(-inset_old));
        let mut old_hatch_ops = Vec::new();
        plan_boustrophedon_hatch(
            &old_hatch_mp,
            &MultiPolygon::new(vec![annulus.clone()]),
            1.0,
            NozzleCap::Fat,
            &cfg,
            &mut old_hatch_ops,
        );
        let boustrophedon_total_strokes = 2 + old_hatch_ops.len();

        let (job, buf) = rasterize_job_to_canvas(&[prim], &cfg, 1024, 614);
        let strokes: Vec<&Vec<StrokeWaypoint>> = job
            .ops
            .iter()
            .filter_map(|op| match op {
                JobOp::PaintStroke { points_canvas_3d } => Some(points_canvas_3d),
                _ => None,
            })
            .collect();

        println!(
            "Annulus stroke comparison: concentric={} strokes vs boustrophedon={} strokes",
            strokes.len(),
            boustrophedon_total_strokes
        );
        assert!(
            strokes.len() <= 6,
            "5cm annulus should plan <= 6 smooth concentric strokes, got {}",
            strokes.len()
        );
        assert!(
            strokes.len() * 2 < boustrophedon_total_strokes,
            "concentric ({}) should use < 1/2 the strokes of boustrophedon ({})",
            strokes.len(),
            boustrophedon_total_strokes
        );

        // Verify every concentric ring closes smoothly with wraparound drag-tilt continuity.
        for wps in &strokes {
            let active = &wps[1..wps.len() - 1];
            assert!(active.len() >= 16);
            let closure_gap = active.first().unwrap().pos.distance(active.last().unwrap().pos);
            assert!(
                closure_gap < 1e-4,
                "concentric ring did not close cleanly: gap={closure_gap}"
            );
            let aim_jump = active
                .first()
                .unwrap()
                .aim_dir
                .distance(active.last().unwrap().aim_dir);
            assert!(
                aim_jump < 1e-4,
                "closed ring wraparound aim_dir jumped by {aim_jump}"
            );
        }

        // Verify rasterized coverage around all 360 degrees (specifically North & South poles) and across ring width.
        for deg in (0..360).step_by(5) {
            let rad = (deg as f32).to_radians();
            for r in [0.212_f32, 0.225_f32, 0.238_f32] {
                let x = cx as f32 + r * rad.cos();
                let y = cy as f32 + r * rad.sin();
                let cov = sample_coverage(&buf, x, y, paint_srgb);
                assert!(
                    cov > 0.88,
                    "annulus gap at deg={deg}, r={r:.3} ({x:.3}, {y:.3}): coverage={cov:.3}"
                );
            }
        }
        assert!(
            sample_coverage(&buf, cx as f32, cy as f32, paint_srgb) < 0.02,
            "interior hole center must stay clean"
        );
        assert!(
            sample_coverage(&buf, cx as f32, (cy + 0.30) as f32, paint_srgb) < 0.02,
            "exterior outside annulus must stay clean"
        );
    }

    #[test]
    fn holed_shield_and_fcz_stems_have_zero_contour_hatch_halo() {
        let cx = 0.0_f64;
        let cy = 1.55_f64;
        let paint_srgb = [15, 30, 80];
        // FCZ-like holed circle shield: r=0.22m minus two vertical rectangular cutouts and a crossbar cutout.
        let outer = circle_ring(cx, cy, 0.22, 64, true);
        let hole_left = rect_ring(cx - 0.11, cy - 0.10, cx - 0.04, cy + 0.10, false);
        let hole_right = rect_ring(cx + 0.04, cy - 0.10, cx + 0.11, cy + 0.10, false);
        let shield = Polygon::new(outer, vec![hole_left, hole_right]);

        // Medium Fat-cap F-like stem (3.4cm wide, right where old inset_hatch vanished) + thin Skinny-cap Z-like stem (6mm wide).
        let fat_stem = Polygon::new(rect_ring(0.32, 1.40, 0.354, 1.70, true), Vec::new());
        let skinny_stem = Polygon::new(rect_ring(0.42, 1.45, 0.426, 1.65, true), Vec::new());

        let prims = vec![
            NormalizedPrimitive {
                id: 0,
                color_srgb: paint_srgb,
                opacity: 1.0,
                geometry: PrimitiveGeom::FilledPolygon(MultiPolygon::new(vec![shield])),
                gradient_meta: None,
            },
            NormalizedPrimitive {
                id: 1,
                color_srgb: paint_srgb,
                opacity: 1.0,
                geometry: PrimitiveGeom::FilledPolygon(MultiPolygon::new(vec![
                    fat_stem,
                    skinny_stem,
                ])),
                gradient_meta: None,
            },
        ];

        let cfg = PlannerConfig::default();
        let (_job, buf) = rasterize_job_to_canvas(&prims, &cfg, 1024, 614);

        // Check shield outer frame (north, south, west, east) & central vertical bridge between the two holes.
        for &(sx, sy, label) in &[
            (0.0_f32, 1.71_f32, "shield north frame"),
            (0.0_f32, 1.39_f32, "shield south frame"),
            (-0.165_f32, 1.55_f32, "shield west arc"),
            (0.165_f32, 1.55_f32, "shield east arc"),
            (0.0_f32, 1.55_f32, "shield center bridge"),
            (0.0_f32, 1.62_f32, "shield upper bridge"),
            (0.0_f32, 1.48_f32, "shield lower bridge"),
            (0.337_f32, 1.55_f32, "3.4cm Fat stem center"),
            (0.337_f32, 1.64_f32, "3.4cm Fat stem upper"),
            (0.423_f32, 1.55_f32, "6mm Skinny stem center"),
        ] {
            let cov = sample_coverage(&buf, sx, sy, paint_srgb);
            assert!(
                cov > 0.85,
                "white halo / gap at {label} ({sx:.3}, {sy:.3}): coverage={cov:.3}"
            );
        }

        // Interior cutouts must remain unpainted.
        assert!(
            sample_coverage(&buf, -0.075, 1.55, paint_srgb) < 0.05,
            "left shield cutout must stay clean"
        );
        assert!(
            sample_coverage(&buf, 0.075, 1.55, paint_srgb) < 0.05,
            "right shield cutout must stay clean"
        );
    }

    #[test]
    fn fcz_holed_shield_central_bridge_between_cutouts_has_solid_coverage() {
        let cx = 0.0_f64;
        let cy = 1.55_f64;
        let paint_srgb = [11, 46, 107];
        // Matches FCZ shield proportions: r_out=40.6cm, 15.2cm central bridge between ~11.2cm
        // vertical cutouts, ~18-20cm side lobes, and top semicircle cutout above horizontal bar.
        let r_out = 0.4065_f64;
        let r_in = 0.3730_f64;
        let outer = circle_ring(cx, cy, r_out, 96, true);

        // Top semicircle cutout (CW) above y = cy + 0.0357.
        let y_top_bar = cy + 0.0357;
        let dy_top = y_top_bar - cy;
        let theta_top = (dy_top / r_in).asin();
        let mut top_hole_coords = Vec::new();
        for i in 0..=48 {
            let t = std::f64::consts::PI - theta_top
                - (i as f64 / 48.0) * (std::f64::consts::PI - 2.0 * theta_top);
            top_hole_coords.push(Coord {
                x: cx + r_in * t.cos(),
                y: cy + r_in * t.sin(),
            });
        }
        top_hole_coords.push(top_hole_coords[0]);
        let top_hole = LineString::new(top_hole_coords);

        // Two vertical bottom-semicircle cutouts (CW) leaving a 15.2cm central bridge (x in [-0.076, 0.076]).
        let make_bottom_cutout = |x0: f64, x1: f64| -> LineString<f64> {
            let y_top = cy - 0.0089;
            let mut coords = vec![Coord { x: x0, y: y_top }, Coord { x: x1, y: y_top }];
            let t1 = ((x1 - cx) / r_in).acos();
            let t0 = ((x0 - cx) / r_in).acos();
            // Bottom semicircle angles are negative (-t1 .. -t0).
            for i in 0..=24 {
                let t = -t1 + (i as f64 / 24.0) * (t1 - t0);
                coords.push(Coord {
                    x: cx + r_in * t.cos(),
                    y: cy + r_in * t.sin(),
                });
            }
            coords.push(Coord { x: x0, y: y_top });
            LineString::new(coords)
        };
        let hole_left = make_bottom_cutout(-0.188, -0.076);
        let hole_right = make_bottom_cutout(0.076, 0.188);

        let shield = Polygon::new(outer, vec![top_hole, hole_left, hole_right]);
        let prim = NormalizedPrimitive {
            id: 0,
            color_srgb: paint_srgb,
            opacity: 1.0,
            geometry: PrimitiveGeom::FilledPolygon(MultiPolygon::new(vec![shield])),
            gradient_meta: None,
        };

        let cfg = PlannerConfig::default();
        let (_job, buf) = rasterize_job_to_canvas(&[prim], &cfg, 1280, 768);

        // Vertical centerline (and +/-1.5cm) of the 15.2cm central bridge must have solid alpha >= 0.80.
        for step in 0..=20 {
            let sy = 1.22_f32 + (step as f32 / 20.0) * (1.53_f32 - 1.22_f32);
            for sx in [-0.015_f32, 0.0_f32, 0.015_f32] {
                let cov = sample_coverage(&buf, sx, sy, paint_srgb);
                assert!(
                    cov >= 0.80,
                    "unpainted stripe in FCZ central bridge at ({sx:.3}, {sy:.3}): coverage={cov:.3}"
                );
            }
        }

        // Horizontal divider bar and side lobes must also be solidly covered.
        for &(sx, sy, label) in &[
            (0.0_f32, 1.563_f32, "horizontal divider center"),
            (-0.28_f32, 1.40_f32, "left side lobe"),
            (0.28_f32, 1.40_f32, "right side lobe"),
        ] {
            let cov = sample_coverage(&buf, sx, sy, paint_srgb);
            assert!(
                cov >= 0.85,
                "gap at {label} ({sx:.3}, {sy:.3}): coverage={cov:.3}"
            );
        }

        // Cutouts must remain clean (bridge medial pass must not bleed across into cutouts).
        for &(sx, sy, label) in &[
            (-0.132_f32, 1.44_f32, "left vertical cutout upper"),
            (-0.132_f32, 1.32_f32, "left vertical cutout lower"),
            (0.132_f32, 1.44_f32, "right vertical cutout upper"),
            (0.132_f32, 1.32_f32, "right vertical cutout lower"),
            (0.0_f32, 1.72_f32, "top semicircle cutout"),
        ] {
            let cov = sample_coverage(&buf, sx, sy, paint_srgb);
            assert!(
                cov < 0.05,
                "paint overspray inside {label} ({sx:.3}, {sy:.3}): coverage={cov:.3}"
            );
        }
    }

    #[test]
    fn radial_gradient_bands_plan_concentric_arcs_with_solid_coverage() {
        let center = DVec2::new(0.0, 1.55);
        let paint_srgb = [20, 40, 160];
        // 1. Verify radial annulus band emits smooth concentric circular strokes.
        let band_poly = Polygon::new(
            circle_ring(center.x, center.y, 0.22, 64, true),
            vec![circle_ring(center.x, center.y, 0.10, 64, false)],
        );
        let prim_band = NormalizedPrimitive {
            id: 0,
            color_srgb: paint_srgb,
            opacity: 1.0,
            geometry: PrimitiveGeom::FilledPolygon(MultiPolygon::new(vec![band_poly])),
            gradient_meta: Some(GradientFillMeta::Radial {
                center,
                band_width: 0.12,
                fade_in_r: None,
                fade_out_r: None,
            }),
        };

        let cfg = PlannerConfig::default();
        let (job, buf) = rasterize_job_to_canvas(&[prim_band], &cfg, 512, 307);
        let strokes: Vec<&Vec<StrokeWaypoint>> = job
            .ops
            .iter()
            .filter_map(|op| match op {
                JobOp::PaintStroke { points_canvas_3d } => Some(points_canvas_3d),
                _ => None,
            })
            .collect();
        assert!(
            !strokes.is_empty() && strokes.len() <= 6,
            "expected compact concentric radial rings, got {} strokes",
            strokes.len()
        );
        for wps in &strokes {
            let active = &wps[1..wps.len() - 1];
            let radii: Vec<f64> = active
                .iter()
                .map(|wp| DVec2::new(wp.pos.x as f64, wp.pos.y as f64).distance(center))
                .collect();
            let mean_r = radii.iter().sum::<f64>() / radii.len() as f64;
            for r in radii {
                assert!(
                    (r - mean_r).abs() < 2e-3,
                    "radial gradient stroke deviated from concentric circle: r={r:.4} vs mean={mean_r:.4}"
                );
            }
            for wp in active {
                assert!(wp.standoff >= 0.12 - 1e-6 && wp.standoff <= 0.25 + 1e-6);
            }
        }

        for deg in (0..360).step_by(15) {
            let rad = (deg as f32).to_radians();
            let x = center.x as f32 + 0.16 * rad.cos();
            let y = center.y as f32 + 0.16 * rad.sin();
            let cov = sample_coverage(&buf, x, y, paint_srgb);
            assert!(
                cov > 0.85,
                "radial gradient annulus gap at deg={deg}: coverage={cov:.3}"
            );
        }
    }
}
