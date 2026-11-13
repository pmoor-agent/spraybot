use std::path::Path;

use geo::algorithm::area::Area;
use geo::algorithm::bool_ops::{BooleanOps, FillRule as GeoFillRule};
use geo::algorithm::orient::{Direction, Orient};
use geo::{Coord, LineString, MultiPolygon, Polygon};
use glam::{DVec2, Vec2};
use i_overlay::core::fill_rule::FillRule as OverlayFillRule;
use i_overlay::float::simplify::SimplifyShape;
use kurbo::{Affine, BezPath, PathEl, Point as KurboPoint};

use crate::core::types::{GradientFillMeta, NormalizedPrimitive, PrimitiveGeom, StudioConfig};

/// Minimum polygon area (m^2) below which degenerate slivers are discarded.
const MIN_POLYGON_AREA_M2: f64 = 1e-6;

/// Configuration for SVG parsing, curve flattening, and canvas coordinate normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct IngestConfig {
    pub canvas_center: Vec2,
    pub canvas_size: Vec2,
    pub margin: f32,
    pub flatten_tolerance_m: f64,
    pub gradient_bands: usize,
}

impl IngestConfig {
    pub fn from_studio(studio: &StudioConfig) -> Self {
        Self {
            canvas_center: Vec2::new(studio.canvas_center.x, studio.canvas_center.y),
            canvas_size: Vec2::new(studio.canvas_width, studio.canvas_height),
            margin: studio.canvas_margin,
            flatten_tolerance_m: 0.0005,
            gradient_bands: 5,
        }
    }
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self::from_studio(&StudioConfig::default())
    }
}

/// Parses SVG bytes into world-canvas-normalized primitives in painter's Z-order.
pub fn ingest_svg(
    svg_bytes: &[u8],
    config: &IngestConfig,
) -> Result<Vec<NormalizedPrimitive>, String> {
    let mut opt = usvg::Options::default();
    opt.fontdb_mut().load_system_fonts();

    let tree =
        usvg::Tree::from_data(svg_bytes, &opt).map_err(|err| format!("failed to parse SVG: {err}"))?;

    let svg_size = tree.size();
    let svg_w = svg_size.width() as f64;
    let svg_h = svg_size.height() as f64;
    if svg_w <= 0.0 || svg_h <= 0.0 {
        return Err(format!("invalid SVG dimensions: {svg_w}x{svg_h}"));
    }

    let world_from_svg = compute_world_from_svg_affine(svg_w, svg_h, config);
    let world_scale = world_from_svg.as_coeffs()[0].abs();

    let mut primitives = Vec::new();
    collect_group(
        tree.root(),
        1.0,
        None,
        &world_from_svg,
        world_scale,
        config,
        &mut primitives,
    );

    Ok(primitives)
}

/// Convenience wrapper reading an SVG file from disk and calling [`ingest_svg`].
pub fn ingest_svg_file(
    path: &Path,
    config: &IngestConfig,
) -> Result<Vec<NormalizedPrimitive>, String> {
    let bytes =
        std::fs::read(path).map_err(|err| format!("failed to read SVG '{}': {err}", path.display()))?;
    ingest_svg(&bytes, config)
}

/// Computes the uniform aspect-preserving Y-up world transform from SVG pixel coordinates.
fn compute_world_from_svg_affine(svg_w: f64, svg_h: f64, config: &IngestConfig) -> Affine {
    let avail_w = (config.canvas_size.x - 2.0 * config.margin).max(1e-6) as f64;
    let avail_h = (config.canvas_size.y - 2.0 * config.margin).max(1e-6) as f64;
    let scale = (avail_w / svg_w).min(avail_h / svg_h);

    let cx = config.canvas_center.x as f64;
    let cy = config.canvas_center.y as f64;
    // Maps (x, y) in [0, svg_w] x [0, svg_h] (Y-down) to world canvas (Y-up) centered at (cx, cy).
    Affine::new([
        scale,
        0.0,
        0.0,
        -scale,
        cx - 0.5 * svg_w * scale,
        cy + 0.5 * svg_h * scale,
    ])
}

#[inline]
fn usvg_transform_to_affine(t: usvg::Transform) -> Affine {
    Affine::new([
        t.sx as f64,
        t.ky as f64,
        t.kx as f64,
        t.sy as f64,
        t.tx as f64,
        t.ty as f64,
    ])
}

fn collect_group(
    group: &usvg::Group,
    parent_opacity: f32,
    parent_clip: Option<&MultiPolygon<f64>>,
    world_from_svg: &Affine,
    world_scale: f64,
    config: &IngestConfig,
    out: &mut Vec<NormalizedPrimitive>,
) {
    let group_opacity = (parent_opacity * group.opacity().get()).clamp(0.0, 1.0);
    if group_opacity <= 0.0 {
        return;
    }

    let own_clip = group.clip_path().and_then(|cp| {
        let clip_poly = extract_clip_polygon(cp, world_from_svg, config.flatten_tolerance_m);
        if clip_poly.0.is_empty() {
            None
        } else {
            Some(clip_poly)
        }
    });

    let combined_clip_storage: Option<MultiPolygon<f64>> = match (parent_clip, own_clip) {
        (Some(p), Some(c)) => Some(p.intersection_with_fill_rule(&c, GeoFillRule::EvenOdd)),
        (Some(p), None) => Some(p.clone()),
        (None, Some(c)) => Some(c),
        (None, None) => None,
    };
    let active_clip = combined_clip_storage.as_ref();

    for node in group.children() {
        match node {
            usvg::Node::Group(child_group) => {
                collect_group(
                    child_group,
                    group_opacity,
                    active_clip,
                    world_from_svg,
                    world_scale,
                    config,
                    out,
                );
            }
            usvg::Node::Path(path) => {
                collect_path(
                    path,
                    group_opacity,
                    active_clip,
                    world_from_svg,
                    world_scale,
                    config,
                    out,
                );
            }
            usvg::Node::Text(text) => {
                collect_group(
                    text.flattened(),
                    group_opacity,
                    active_clip,
                    world_from_svg,
                    world_scale,
                    config,
                    out,
                );
            }
            usvg::Node::Image(image) => {
                if let usvg::ImageKind::SVG(sub_tree) = image.kind() {
                    let sub_affine = *world_from_svg * usvg_transform_to_affine(image.abs_transform());
                    let coeffs = sub_affine.as_coeffs();
                    let sub_scale = ((coeffs[0].hypot(coeffs[1])) + (coeffs[2].hypot(coeffs[3]))) * 0.5;
                    collect_group(
                        sub_tree.root(),
                        group_opacity,
                        active_clip,
                        &sub_affine,
                        sub_scale,
                        config,
                        out,
                    );
                }
            }
        }
    }
}

fn extract_clip_polygon(
    clip_path: &usvg::ClipPath,
    world_from_svg: &Affine,
    tolerance_m: f64,
) -> MultiPolygon<f64> {
    let mut polys: Vec<MultiPolygon<f64>> = Vec::new();
    collect_clip_group(clip_path.root(), world_from_svg, tolerance_m, &mut polys);
    if polys.is_empty() {
        return MultiPolygon::empty();
    }
    let mut acc = polys.remove(0);
    for p in polys {
        acc = acc.union_with_fill_rule(&p, GeoFillRule::NonZero);
    }
    acc
}

fn collect_clip_group(
    group: &usvg::Group,
    world_from_svg: &Affine,
    tolerance_m: f64,
    out: &mut Vec<MultiPolygon<f64>>,
) {
    for node in group.children() {
        match node {
            usvg::Node::Group(g) => collect_clip_group(g, world_from_svg, tolerance_m, out),
            usvg::Node::Path(p) => {
                let path_to_world = *world_from_svg * usvg_transform_to_affine(p.abs_transform());
                let subpaths = flatten_usvg_path(p.data(), &path_to_world, tolerance_m);
                let rule = p
                    .fill()
                    .map(|f| f.rule())
                    .unwrap_or(usvg::FillRule::NonZero);
                let mp = subpaths_to_multipolygon(&subpaths, rule);
                if !mp.0.is_empty() {
                    out.push(mp);
                }
            }
            usvg::Node::Text(t) => {
                collect_clip_group(t.flattened(), world_from_svg, tolerance_m, out);
            }
            usvg::Node::Image(_) => {}
        }
    }
}

fn collect_path(
    path: &usvg::Path,
    group_opacity: f32,
    clip: Option<&MultiPolygon<f64>>,
    world_from_svg: &Affine,
    world_scale: f64,
    config: &IngestConfig,
    out: &mut Vec<NormalizedPrimitive>,
) {
    if !path.is_visible() {
        return;
    }

    let path_abs = path.abs_transform();
    let path_to_world = *world_from_svg * usvg_transform_to_affine(path_abs);
    let subpaths = flatten_usvg_path(path.data(), &path_to_world, config.flatten_tolerance_m);
    if subpaths.is_empty() {
        return;
    }

    match path.paint_order() {
        usvg::PaintOrder::FillAndStroke => {
            if let Some(fill) = path.fill() {
                emit_fill(
                    fill,
                    &subpaths,
                    &path_to_world,
                    group_opacity,
                    clip,
                    config,
                    out,
                );
            }
            if let Some(stroke) = path.stroke() {
                emit_stroke(
                    stroke,
                    &subpaths,
                    path_abs,
                    group_opacity,
                    world_scale,
                    out,
                );
            }
        }
        usvg::PaintOrder::StrokeAndFill => {
            if let Some(stroke) = path.stroke() {
                emit_stroke(
                    stroke,
                    &subpaths,
                    path_abs,
                    group_opacity,
                    world_scale,
                    out,
                );
            }
            if let Some(fill) = path.fill() {
                emit_fill(
                    fill,
                    &subpaths,
                    &path_to_world,
                    group_opacity,
                    clip,
                    config,
                    out,
                );
            }
        }
    }
}

fn emit_fill(
    fill: &usvg::Fill,
    subpaths: &[FlattenedSubpath],
    path_to_world: &Affine,
    group_opacity: f32,
    clip: Option<&MultiPolygon<f64>>,
    config: &IngestConfig,
    out: &mut Vec<NormalizedPrimitive>,
) {
    let opacity = (group_opacity * fill.opacity().get()).clamp(0.0, 1.0);
    if opacity <= 0.0 {
        return;
    }

    let mut mp = subpaths_to_multipolygon(subpaths, fill.rule());
    if let Some(clip_mp) = clip {
        mp = mp.intersection_with_fill_rule(clip_mp, GeoFillRule::EvenOdd);
    }
    mp = filter_valid_polygons(mp);
    if mp.0.is_empty() {
        return;
    }

    match fill.paint() {
        usvg::Paint::Color(c) => {
            push_primitive(
                out,
                [c.red, c.green, c.blue],
                opacity,
                PrimitiveGeom::FilledPolygon(mp),
            );
        }
        usvg::Paint::LinearGradient(lg) => {
            emit_linear_gradient_bands(
                lg,
                &mp,
                path_to_world,
                opacity,
                config.gradient_bands.max(1),
                out,
            );
        }
        usvg::Paint::RadialGradient(rg) => {
            emit_radial_gradient_bands(
                rg,
                &mp,
                path_to_world,
                opacity,
                config.gradient_bands.max(1),
                out,
            );
        }
        usvg::Paint::Pattern(pat) => {
            if let Some((color, pat_op)) = fallback_pattern_color(pat.root()) {
                push_primitive(
                    out,
                    color,
                    (opacity * pat_op).clamp(0.0, 1.0),
                    PrimitiveGeom::FilledPolygon(mp),
                );
            }
        }
    }
}

fn emit_stroke(
    stroke: &usvg::Stroke,
    subpaths: &[FlattenedSubpath],
    path_abs: usvg::Transform,
    group_opacity: f32,
    world_scale: f64,
    out: &mut Vec<NormalizedPrimitive>,
) {
    let opacity = (group_opacity * stroke.opacity().get()).clamp(0.0, 1.0);
    if opacity <= 0.0 {
        return;
    }

    // Scale stroke width from path local space through path_abs into world canvas meters.
    let abs_scale = ((path_abs.sx as f64).hypot(path_abs.ky as f64)
        + (path_abs.kx as f64).hypot(path_abs.sy as f64))
        * 0.5;
    let width_m = (stroke.width().get() as f64) * abs_scale * world_scale;
    if width_m <= 1e-6 {
        return;
    }

    let (color_srgb, stop_opacity) = match stroke.paint() {
        usvg::Paint::Color(c) => ([c.red, c.green, c.blue], 1.0_f32),
        usvg::Paint::LinearGradient(lg) => average_stop_color(lg.stops()),
        usvg::Paint::RadialGradient(rg) => average_stop_color(rg.stops()),
        usvg::Paint::Pattern(pat) => match fallback_pattern_color(pat.root()) {
            Some(pair) => pair,
            None => return,
        },
    };
    let final_opacity = (opacity * stop_opacity).clamp(0.0, 1.0);
    if final_opacity <= 0.0 {
        return;
    }

    for sp in subpaths {
        let mut pts = sp.points.clone();
        if sp.closed && pts.len() >= 2 {
            if pts.first().unwrap().distance(*pts.last().unwrap()) > 1e-6 {
                pts.push(pts[0]);
            }
        }
        if pts.len() < 2 {
            continue;
        }
        push_primitive(
            out,
            color_srgb,
            final_opacity,
            PrimitiveGeom::StrokedPath {
                points: pts,
                width: width_m,
            },
        );
    }
}

fn push_primitive(
    out: &mut Vec<NormalizedPrimitive>,
    color_srgb: [u8; 3],
    opacity: f32,
    geometry: PrimitiveGeom,
) {
    push_primitive_with_meta(out, color_srgb, opacity, geometry, None);
}

fn push_primitive_with_meta(
    out: &mut Vec<NormalizedPrimitive>,
    color_srgb: [u8; 3],
    opacity: f32,
    geometry: PrimitiveGeom,
    gradient_meta: Option<GradientFillMeta>,
) {
    let id = out.len();
    out.push(NormalizedPrimitive {
        id,
        color_srgb,
        opacity,
        geometry,
        gradient_meta,
    });
}

#[derive(Debug, Clone)]
struct FlattenedSubpath {
    points: Vec<DVec2>,
    closed: bool,
}

fn flatten_usvg_path(
    path: &usvg::tiny_skia_path::Path,
    transform: &Affine,
    tolerance_m: f64,
) -> Vec<FlattenedSubpath> {
    use usvg::tiny_skia_path::PathSegment;

    let mut bez = BezPath::new();
    for seg in path.segments() {
        match seg {
            PathSegment::MoveTo(p) => {
                bez.move_to(KurboPoint::new(p.x as f64, p.y as f64));
            }
            PathSegment::LineTo(p) => {
                bez.line_to(KurboPoint::new(p.x as f64, p.y as f64));
            }
            PathSegment::QuadTo(p1, p2) => {
                bez.quad_to(
                    KurboPoint::new(p1.x as f64, p1.y as f64),
                    KurboPoint::new(p2.x as f64, p2.y as f64),
                );
            }
            PathSegment::CubicTo(p1, p2, p3) => {
                bez.curve_to(
                    KurboPoint::new(p1.x as f64, p1.y as f64),
                    KurboPoint::new(p2.x as f64, p2.y as f64),
                    KurboPoint::new(p3.x as f64, p3.y as f64),
                );
            }
            PathSegment::Close => {
                bez.close_path();
            }
        }
    }

    bez.apply_affine(*transform);

    let tol = tolerance_m.clamp(1e-5, 0.0005);
    let mut subpaths: Vec<FlattenedSubpath> = Vec::new();
    let mut current: Option<FlattenedSubpath> = None;

    kurbo::flatten(bez.iter(), tol, |el| match el {
        PathEl::MoveTo(p) => {
            if let Some(prev) = current.take() {
                if prev.points.len() >= 2 {
                    subpaths.push(prev);
                }
            }
            current = Some(FlattenedSubpath {
                points: vec![DVec2::new(p.x, p.y)],
                closed: false,
            });
        }
        PathEl::LineTo(p) => {
            if let Some(ref mut cur) = current {
                let pt = DVec2::new(p.x, p.y);
                if cur.points.last().is_none_or(|last| last.distance(pt) > 1e-7) {
                    cur.points.push(pt);
                }
            }
        }
        PathEl::ClosePath => {
            if let Some(mut cur) = current.take() {
                cur.closed = true;
                if cur.points.len() >= 2 {
                    subpaths.push(cur);
                }
            }
        }
        PathEl::QuadTo(..) | PathEl::CurveTo(..) => {}
    });

    if let Some(last) = current.take() {
        if last.points.len() >= 2 {
            subpaths.push(last);
        }
    }

    subpaths
}

fn subpaths_to_multipolygon(
    subpaths: &[FlattenedSubpath],
    rule: usvg::FillRule,
) -> MultiPolygon<f64> {
    let contours: Vec<Vec<[f64; 2]>> = subpaths
        .iter()
        .filter_map(|sp| {
            let mut pts: Vec<[f64; 2]> = sp.points.iter().map(|p| [p.x, p.y]).collect();
            if pts.len() >= 2 {
                let first = pts[0];
                let last = *pts.last().unwrap();
                if (first[0] - last[0]).hypot(first[1] - last[1]) < 1e-7 {
                    pts.pop();
                }
            }
            if pts.len() >= 3 { Some(pts) } else { None }
        })
        .collect();

    if contours.is_empty() {
        return MultiPolygon::empty();
    }

    let overlay_rule = match rule {
        usvg::FillRule::NonZero => OverlayFillRule::NonZero,
        usvg::FillRule::EvenOdd => OverlayFillRule::EvenOdd,
    };

    let shapes = contours.simplify_shape(overlay_rule);
    let polygons: Vec<Polygon<f64>> = shapes
        .into_iter()
        .filter_map(|mut shape| {
            if shape.is_empty() {
                return None;
            }
            let ext = contour_to_ring(&shape.remove(0))?;
            let holes: Vec<LineString<f64>> = shape
                .into_iter()
                .filter_map(|c| contour_to_ring(&c))
                .collect();
            let poly = Polygon::new(ext, holes).orient(Direction::Default);
            if poly.unsigned_area() >= MIN_POLYGON_AREA_M2 {
                Some(poly)
            } else {
                None
            }
        })
        .collect();

    MultiPolygon::new(polygons)
}

fn contour_to_ring(pts: &[[f64; 2]]) -> Option<LineString<f64>> {
    if pts.len() < 3 {
        return None;
    }
    let mut coords: Vec<Coord<f64>> = pts.iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
    if let (Some(first), Some(last)) = (coords.first().copied(), coords.last().copied()) {
        if (first.x - last.x).hypot(first.y - last.y) > 1e-9 {
            coords.push(first);
        }
    }
    if coords.len() >= 4 {
        Some(LineString::new(coords))
    } else {
        None
    }
}

fn filter_valid_polygons(mp: MultiPolygon<f64>) -> MultiPolygon<f64> {
    let kept: Vec<Polygon<f64>> = mp
        .0
        .into_iter()
        .filter(|p| p.unsigned_area() >= MIN_POLYGON_AREA_M2)
        .map(|p| p.orient(Direction::Default))
        .collect();
    MultiPolygon::new(kept)
}

fn emit_linear_gradient_bands(
    lg: &usvg::LinearGradient,
    mp: &MultiPolygon<f64>,
    path_to_world: &Affine,
    base_opacity: f32,
    bands: usize,
    out: &mut Vec<NormalizedPrimitive>,
) {
    let grad_to_world = *path_to_world * usvg_transform_to_affine(lg.transform());
    let p1_k = grad_to_world * KurboPoint::new(lg.x1() as f64, lg.y1() as f64);
    let p2_k = grad_to_world * KurboPoint::new(lg.x2() as f64, lg.y2() as f64);
    let p1 = DVec2::new(p1_k.x, p1_k.y);
    let p2 = DVec2::new(p2_k.x, p2_k.y);
    let axis = p2 - p1;
    let axis_len = axis.length();

    // Sub-threshold size or degenerate axis collapses to area/offset-weighted stop color.
    let area = mp.unsigned_area();
    if axis_len < 1e-4 || area < 1e-4 || bands <= 1 {
        let (color, stop_op) = average_stop_color(lg.stops());
        let final_op = (base_opacity * stop_op).clamp(0.0, 1.0);
        if final_op > 0.0 {
            push_primitive(out, color, final_op, PrimitiveGeom::FilledPolygon(mp.clone()));
        }
        return;
    }

    let dir = axis / axis_len;
    let perp = DVec2::new(-dir.y, dir.x);
    let band_width = axis_len / bands as f64;
    let back_overlap = 0.45 * band_width;
    let front_pad = 0.20 * band_width;

    // Project polygon vertices onto (dir, perp) basis to bound band rectangles cleanly.
    let mut s_min = f64::INFINITY;
    let mut s_max = f64::NEG_INFINITY;
    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    for poly in &mp.0 {
        for c in poly.exterior().coords() {
            let rel = DVec2::new(c.x, c.y) - p1;
            let s = rel.dot(dir);
            let u = rel.dot(perp);
            s_min = s_min.min(s);
            s_max = s_max.max(s);
            u_min = u_min.min(u);
            u_max = u_max.max(u);
        }
    }
    let u0 = u_min - 0.5;
    let u1 = u_max + 0.5;

    for b in 0..bands {
        let t_sample = b as f64 / (bands - 1) as f64;
        let (color, stop_op) = sample_gradient_stops(lg.stops(), t_sample as f32, lg.spread_method());
        let band_opacity = (base_opacity * stop_op).clamp(0.0, 1.0);
        if band_opacity <= 0.0 {
            continue;
        }

        let s_nom0 = b as f64 * band_width;
        let s_nom1 = (b + 1) as f64 * band_width;
        let s0 = if b == 0 {
            s_min.min(0.0) - 0.5
        } else {
            s_nom0 - back_overlap
        };
        let s1 = if b + 1 == bands {
            s_max.max(axis_len) + 0.5
        } else {
            s_nom1 + front_pad
        };
        if s1 <= s0 + 1e-7 {
            continue;
        }

        let fade_in_s = (b > 0).then_some((s_nom0 - 0.45 * band_width, s_nom0 + 0.18 * band_width));
        let fade_out_s =
            (b + 1 < bands).then_some((s_nom1 + 0.04 * band_width, s_nom1 + 0.20 * band_width));

        let slab = oriented_rect_polygon(p1, dir, perp, s0, s1, u0, u1);
        let clipped = filter_valid_polygons(mp.intersection_with_fill_rule(&slab, GeoFillRule::EvenOdd));
        if !clipped.0.is_empty() {
            push_primitive_with_meta(
                out,
                color,
                band_opacity,
                PrimitiveGeom::FilledPolygon(clipped),
                Some(GradientFillMeta::Linear {
                    origin: p1,
                    dir,
                    band_width,
                    fade_in_s,
                    fade_out_s,
                }),
            );
        }
    }
}

fn emit_radial_gradient_bands(
    rg: &usvg::RadialGradient,
    mp: &MultiPolygon<f64>,
    path_to_world: &Affine,
    base_opacity: f32,
    bands: usize,
    out: &mut Vec<NormalizedPrimitive>,
) {
    let grad_to_world = *path_to_world * usvg_transform_to_affine(rg.transform());
    let radius = rg.r().get() as f64;
    let center_world = grad_to_world * KurboPoint::new(rg.cx() as f64, rg.cy() as f64);
    let edge_world = grad_to_world * KurboPoint::new(rg.cx() as f64 + radius, rg.cy() as f64);
    let world_radius = center_world.distance(edge_world);

    let area = mp.unsigned_area();
    if radius <= 1e-6 || world_radius < 1e-4 || area < 1e-4 || bands <= 1 {
        let (color, stop_op) = average_stop_color(rg.stops());
        let final_op = (base_opacity * stop_op).clamp(0.0, 1.0);
        if final_op > 0.0 {
            push_primitive(out, color, final_op, PrimitiveGeom::FilledPolygon(mp.clone()));
        }
        return;
    }

    let cx = rg.cx() as f64;
    let cy = rg.cy() as f64;
    let center = DVec2::new(center_world.x, center_world.y);
    let band_width = world_radius / bands as f64;
    let band_width_local = radius / bands as f64;
    let back_overlap_local = 0.45 * band_width_local;
    let front_pad_local = 0.20 * band_width_local;

    for b in 0..bands {
        let t_sample = b as f64 / (bands - 1) as f64;
        let (color, stop_op) = sample_gradient_stops(rg.stops(), t_sample as f32, rg.spread_method());
        let band_opacity = (base_opacity * stop_op).clamp(0.0, 1.0);
        if band_opacity <= 0.0 {
            continue;
        }

        let r0 = if b == 0 {
            0.0
        } else {
            (b as f64 * band_width_local - back_overlap_local).max(0.0)
        };
        let r1 = (b + 1) as f64 * band_width_local
            + if b + 1 < bands { front_pad_local } else { 0.0 };

        let clipped = if b == 0 {
            let outer = circle_polygon_in_world(cx, cy, r1, &grad_to_world, 48);
            mp.intersection_with_fill_rule(&outer, GeoFillRule::EvenOdd)
        } else if b + 1 == bands {
            // Outer band extends to cover Pad region outside gradient circle.
            let inner = circle_polygon_in_world(cx, cy, r0, &grad_to_world, 48);
            mp.difference_with_fill_rule(&inner, GeoFillRule::EvenOdd)
        } else {
            let outer = circle_polygon_in_world(cx, cy, r1, &grad_to_world, 48);
            let inner = circle_polygon_in_world(cx, cy, r0, &grad_to_world, 48);
            let ring = outer.difference_with_fill_rule(&inner, GeoFillRule::EvenOdd);
            mp.intersection_with_fill_rule(&ring, GeoFillRule::EvenOdd)
        };

        let valid = filter_valid_polygons(clipped);
        if !valid.0.is_empty() {
            let r_nom0 = b as f64 * band_width;
            let r_nom1 = (b + 1) as f64 * band_width;
            let fade_in_r =
                (b > 0).then_some((r_nom0 - 0.45 * band_width, r_nom0 + 0.18 * band_width));
            let fade_out_r = (b + 1 < bands)
                .then_some((r_nom1 + 0.04 * band_width, r_nom1 + 0.20 * band_width));
            push_primitive_with_meta(
                out,
                color,
                band_opacity,
                PrimitiveGeom::FilledPolygon(valid),
                Some(GradientFillMeta::Radial {
                    center,
                    band_width,
                    fade_in_r,
                    fade_out_r,
                }),
            );
        }
    }
}

fn oriented_rect_polygon(
    origin: DVec2,
    dir: DVec2,
    perp: DVec2,
    s0: f64,
    s1: f64,
    u0: f64,
    u1: f64,
) -> Polygon<f64> {
    let p00 = origin + dir * s0 + perp * u0;
    let p10 = origin + dir * s1 + perp * u0;
    let p11 = origin + dir * s1 + perp * u1;
    let p01 = origin + dir * s0 + perp * u1;
    let ring = LineString::new(vec![
        Coord { x: p00.x, y: p00.y },
        Coord { x: p10.x, y: p10.y },
        Coord { x: p11.x, y: p11.y },
        Coord { x: p01.x, y: p01.y },
        Coord { x: p00.x, y: p00.y },
    ]);
    Polygon::new(ring, Vec::new()).orient(Direction::Default)
}

fn circle_polygon_in_world(
    cx: f64,
    cy: f64,
    radius: f64,
    grad_to_world: &Affine,
    segments: usize,
) -> Polygon<f64> {
    let mut coords = Vec::with_capacity(segments + 1);
    for i in 0..segments {
        let angle = (i as f64 / segments as f64) * std::f64::consts::TAU;
        let (sin_a, cos_a) = angle.sin_cos();
        let pt = *grad_to_world * KurboPoint::new(cx + radius * cos_a, cy + radius * sin_a);
        coords.push(Coord { x: pt.x, y: pt.y });
    }
    coords.push(coords[0]);
    Polygon::new(LineString::new(coords), Vec::new()).orient(Direction::Default)
}

fn sample_gradient_stops(
    stops: &[usvg::Stop],
    t_raw: f32,
    spread: usvg::SpreadMethod,
) -> ([u8; 3], f32) {
    if stops.is_empty() {
        return ([0, 0, 0], 1.0);
    }
    if stops.len() == 1 {
        let c = stops[0].color();
        return ([c.red, c.green, c.blue], stops[0].opacity().get());
    }

    let t = match spread {
        usvg::SpreadMethod::Pad => t_raw.clamp(0.0, 1.0),
        usvg::SpreadMethod::Repeat => t_raw.rem_euclid(1.0),
        usvg::SpreadMethod::Reflect => {
            let m = t_raw.rem_euclid(2.0);
            if m <= 1.0 { m } else { 2.0 - m }
        }
    };

    if t <= stops[0].offset().get() {
        let c = stops[0].color();
        return ([c.red, c.green, c.blue], stops[0].opacity().get());
    }
    let last = stops.last().unwrap();
    if t >= last.offset().get() {
        let c = last.color();
        return ([c.red, c.green, c.blue], last.opacity().get());
    }

    for pair in stops.windows(2) {
        let s0 = &pair[0];
        let s1 = &pair[1];
        let o0 = s0.offset().get();
        let o1 = s1.offset().get();
        if t >= o0 && t <= o1 {
            let span = (o1 - o0).max(1e-6);
            let frac = ((t - o0) / span).clamp(0.0, 1.0);
            let c0 = s0.color();
            let c1 = s1.color();
            let lerp_u8 = |a: u8, b: u8| -> u8 {
                (a as f32 + (b as f32 - a as f32) * frac)
                    .round()
                    .clamp(0.0, 255.0) as u8
            };
            let op = s0.opacity().get() + (s1.opacity().get() - s0.opacity().get()) * frac;
            return ([lerp_u8(c0.red, c1.red), lerp_u8(c0.green, c1.green), lerp_u8(c0.blue, c1.blue)], op);
        }
    }

    let c = last.color();
    ([c.red, c.green, c.blue], last.opacity().get())
}

fn average_stop_color(stops: &[usvg::Stop]) -> ([u8; 3], f32) {
    if stops.is_empty() {
        return ([0, 0, 0], 1.0);
    }
    if stops.len() == 1 {
        let c = stops[0].color();
        return ([c.red, c.green, c.blue], stops[0].opacity().get());
    }

    // Trapezoidal integral over offset in [0, 1].
    let mut r_acc = 0.0_f64;
    let mut g_acc = 0.0_f64;
    let mut b_acc = 0.0_f64;
    let mut a_acc = 0.0_f64;
    let mut w_acc = 0.0_f64;

    // Leading pad before first stop
    let first_off = stops[0].offset().get() as f64;
    if first_off > 0.0 {
        let c = stops[0].color();
        r_acc += c.red as f64 * first_off;
        g_acc += c.green as f64 * first_off;
        b_acc += c.blue as f64 * first_off;
        a_acc += stops[0].opacity().get() as f64 * first_off;
        w_acc += first_off;
    }

    for pair in stops.windows(2) {
        let dt = (pair[1].offset().get() - pair[0].offset().get()).max(0.0) as f64;
        if dt > 0.0 {
            let c0 = pair[0].color();
            let c1 = pair[1].color();
            r_acc += 0.5 * (c0.red as f64 + c1.red as f64) * dt;
            g_acc += 0.5 * (c0.green as f64 + c1.green as f64) * dt;
            b_acc += 0.5 * (c0.blue as f64 + c1.blue as f64) * dt;
            a_acc += 0.5 * (pair[0].opacity().get() as f64 + pair[1].opacity().get() as f64) * dt;
            w_acc += dt;
        }
    }

    // Trailing pad after last stop
    let last = stops.last().unwrap();
    let tail = (1.0 - last.offset().get() as f64).max(0.0);
    if tail > 0.0 {
        let c = last.color();
        r_acc += c.red as f64 * tail;
        g_acc += c.green as f64 * tail;
        b_acc += c.blue as f64 * tail;
        a_acc += last.opacity().get() as f64 * tail;
        w_acc += tail;
    }

    if w_acc <= 1e-9 {
        let c = stops[0].color();
        return ([c.red, c.green, c.blue], stops[0].opacity().get());
    }

    (
        [
            (r_acc / w_acc).round().clamp(0.0, 255.0) as u8,
            (g_acc / w_acc).round().clamp(0.0, 255.0) as u8,
            (b_acc / w_acc).round().clamp(0.0, 255.0) as u8,
        ],
        (a_acc / w_acc).clamp(0.0, 1.0) as f32,
    )
}

fn fallback_pattern_color(group: &usvg::Group) -> Option<([u8; 3], f32)> {
    for node in group.children() {
        match node {
            usvg::Node::Path(p) => {
                if let Some(fill) = p.fill() {
                    if let usvg::Paint::Color(c) = fill.paint() {
                        return Some(([c.red, c.green, c.blue], fill.opacity().get()));
                    }
                }
                if let Some(stroke) = p.stroke() {
                    if let usvg::Paint::Color(c) = stroke.paint() {
                        return Some(([c.red, c.green, c.blue], stroke.opacity().get()));
                    }
                }
            }
            usvg::Node::Group(g) => {
                if let Some(res) = fallback_pattern_color(g) {
                    return Some(res);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::algorithm::bounding_rect::BoundingRect;
    use geo::algorithm::contains::Contains;
    use geo::Point;

    #[test]
    fn aspect_scaling_and_y_up_flip() {
        let svg = br##"
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 500" width="1000" height="500">
            <rect x="0" y="0" width="100" height="50" fill="#FF0000"/>
            <rect x="900" y="450" width="100" height="50" fill="#00FF00"/>
        </svg>"##;

        let cfg = IngestConfig::default();
        let prims = ingest_svg(svg, &cfg).expect("parse aspect SVG");
        assert_eq!(prims.len(), 2);

        // Paintable bounds: width = 2.34m, height = 1.34m -> scale = min(2.34/1000, 1.34/500) = 0.00234
        // SVG (0,0) top-left maps to high Y; SVG (1000,500) bottom-right maps to low Y.
        let top_left_bbox = match &prims[0].geometry {
            PrimitiveGeom::FilledPolygon(mp) => mp.bounding_rect().unwrap(),
            _ => panic!("expected filled polygon"),
        };
        let bot_right_bbox = match &prims[1].geometry {
            PrimitiveGeom::FilledPolygon(mp) => mp.bounding_rect().unwrap(),
            _ => panic!("expected filled polygon"),
        };

        // Top-left SVG rect must have smaller X and LARGER Y than bottom-right SVG rect (Y-up!)
        assert!(top_left_bbox.min().x < bot_right_bbox.min().x);
        assert!(top_left_bbox.min().y > bot_right_bbox.max().y);

        // Aspect ratio preserved: SVG 1000x500 -> 2.34m x 1.17m centered at (0.0, 1.55)
        let full_w = bot_right_bbox.max().x - top_left_bbox.min().x;
        let full_h = top_left_bbox.max().y - bot_right_bbox.min().y;
        assert!((full_w - 2.34).abs() < 1e-4, "expected width 2.34, got {full_w}");
        assert!((full_h - 1.17).abs() < 1e-4, "expected height 1.17, got {full_h}");
        let center_x = 0.5 * (top_left_bbox.min().x + bot_right_bbox.max().x);
        let center_y = 0.5 * (bot_right_bbox.min().y + top_left_bbox.max().y);
        assert!((center_x - 0.0).abs() < 1e-4);
        assert!((center_y - 1.55).abs() < 1e-4);
    }

    #[test]
    fn evenodd_holes_preserved_from_geometric_shapes() {
        let path = Path::new("assets/examples/01_geometric_shapes.svg");
        let prims = ingest_svg_file(path, &IngestConfig::default()).expect("ingest 01_geometric_shapes");

        // Find the purple octagon (#5E2BFF = [94, 43, 255]) with 2 interior holes
        let octagon = prims
            .iter()
            .find(|p| p.color_srgb == [0x5E, 0x2B, 0xFF])
            .expect("purple octagon present");

        match &octagon.geometry {
            PrimitiveGeom::FilledPolygon(mp) => {
                assert_eq!(mp.0.len(), 1, "octagon should be a single polygon");
                assert_eq!(
                    mp.0[0].interiors().len(),
                    2,
                    "evenodd octagon should preserve both square & diamond holes"
                );
            }
            _ => panic!("expected filled polygon for octagon"),
        }

        // Find the teal 6-point star (#00C49A) with 1 centered hexagon hole
        let hex_star = prims
            .iter()
            .find(|p| p.color_srgb == [0x00, 0xC4, 0x9A])
            .expect("teal star present");
        match &hex_star.geometry {
            PrimitiveGeom::FilledPolygon(mp) => {
                assert_eq!(mp.0.len(), 1);
                assert_eq!(mp.0[0].interiors().len(), 1, "evenodd star should preserve center hole");
            }
            _ => panic!("expected filled polygon for star"),
        }

        // Verify wide and thin strokes extracted with distinct meter widths
        let strokes: Vec<f64> = prims
            .iter()
            .filter_map(|p| match &p.geometry {
                PrimitiveGeom::StrokedPath { width, .. } => Some(*width),
                _ => None,
            })
            .collect();
        assert!(strokes.len() >= 6, "expected strokes from 01_geometric_shapes");
        let max_w = strokes.iter().copied().fold(0.0_f64, f64::max);
        let min_w = strokes.iter().copied().fold(f64::INFINITY, f64::min);
        // 60px * (1.34 / 600) = 0.134m > w(0.26) = 0.1105m
        assert!(max_w > 0.12, "60px stroke should exceed 0.12m, got {max_w}");
        assert!(min_w < 0.025, "8px stroke should be < 0.025m, got {min_w}");
    }

    #[test]
    fn text_outlines_and_linear_gradient_bands_in_poster() {
        let path = Path::new("assets/examples/02_street_art_poster.svg");
        let prims = ingest_svg_file(path, &IngestConfig::default()).expect("ingest 02_street_art_poster");

        // Linear gradient sky (twilightSky) should produce 5 sliced bands at the start (ids 0..5)
        let sky_bands: Vec<&NormalizedPrimitive> = prims
            .iter()
            .take(5)
            .filter(|p| matches!(p.geometry, PrimitiveGeom::FilledPolygon(_)))
            .collect();
        assert_eq!(sky_bands.len(), 5, "twilightSky linearGradient should slice into 5 bands");

        // Bands should have distinct colors ordered from dark twilight (top, higher Y) to warm gold (bottom, lower Y)
        let top_y = match &sky_bands[0].geometry {
            PrimitiveGeom::FilledPolygon(mp) => mp.bounding_rect().unwrap().center().y,
            _ => unreachable!(),
        };
        let bot_y = match &sky_bands[4].geometry {
            PrimitiveGeom::FilledPolygon(mp) => mp.bounding_rect().unwrap().center().y,
            _ => unreachable!(),
        };
        assert!(top_y > bot_y, "gradient band 0 (y={top_y}) should sit above band 4 (y={bot_y}) in Y-up");
        assert_ne!(sky_bands[0].color_srgb, sky_bands[4].color_srgb);

        // Text elements (SPRAY, BOT, URBAN MURAL STUDIO, UNIT-07...) should be converted into filled polygons
        // Verify semi-transparent shadow layers preserve opacity < 1.0
        let semi_transparent_count = prims.iter().filter(|p| p.opacity < 0.99).count();
        assert!(
            semi_transparent_count >= 10,
            "expected semi-transparent primitives (sun rays, shadows, mist), got {semi_transparent_count}"
        );

        // Specifically verify text glyph polygons exist in the upper-right region (SVG x >= 540, y <= 320 -> world x > 0.08, y > 1.50)
        let right_upper_glyph_polys = prims
            .iter()
            .filter(|p| match &p.geometry {
                PrimitiveGeom::FilledPolygon(mp) => {
                    if let Some(bb) = mp.bounding_rect() {
                        bb.min().x > 0.08 && bb.min().y > 1.48 && bb.width() < 0.9 && bb.height() < 0.4
                    } else {
                        false
                    }
                }
                _ => false,
            })
            .count();
        assert!(
            right_upper_glyph_polys >= 2,
            "expected flattened <text> outline primitives in poster upper-right, got {right_upper_glyph_polys}"
        );
    }

    #[test]
    fn radial_gradient_produces_concentric_bands() {
        let svg = br##"
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
            <defs>
                <radialGradient id="rg" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#FFFFFF"/>
                    <stop offset="100%" stop-color="#0000FF"/>
                </radialGradient>
            </defs>
            <circle cx="100" cy="100" r="80" fill="url(#rg)"/>
        </svg>"##;

        let prims = ingest_svg(svg, &IngestConfig::default()).expect("ingest radial gradient");
        assert_eq!(prims.len(), 5, "radial gradient should slice into 5 bands");

        // Center point (0.0, 1.55) should lie inside band 0 (innermost disk) and outside band 2 (annulus)
        let center = Point::new(0.0, 1.55);
        match (&prims[0].geometry, &prims[2].geometry) {
            (PrimitiveGeom::FilledPolygon(inner), PrimitiveGeom::FilledPolygon(mid_ring)) => {
                assert!(inner.contains(&center), "innermost band should contain center");
                assert!(!mid_ring.contains(&center), "annulus band should exclude center");
            }
            _ => panic!("expected filled polygons"),
        }

        // Radial bands should carry Radial gradient metadata and proportional fade ranges.
        match &prims[1].gradient_meta {
            Some(GradientFillMeta::Radial {
                band_width,
                fade_in_r,
                fade_out_r,
                ..
            }) => {
                assert!(*band_width > 0.0);
                let (fi_a, fi_b) = fade_in_r.expect("band 1 should fade in");
                let (fo_a, fo_b) = fade_out_r.expect("band 1 should fade out");
                assert!(((fi_b - fi_a) - 0.63 * band_width).abs() < 1e-6);
                assert!(((fo_b - fo_a) - 0.16 * band_width).abs() < 1e-6);
            }
            other => panic!("expected Radial gradient meta on band 1, got {other:?}"),
        }
    }

    #[test]
    fn proportional_gradient_overlap_scales_with_bounds() {
        let small_svg = br##"
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" width="1000" height="1000">
            <defs>
                <linearGradient id="g_small" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stop-color="#1B1035"/>
                    <stop offset="100%" stop-color="#FFD166"/>
                </linearGradient>
            </defs>
            <rect x="0" y="0" width="100" height="100" fill="url(#g_small)"/>
            <rect x="500" y="500" width="50" height="50" fill="#00FF00"/>
        </svg>"##;

        let large_svg = br##"
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" width="1000" height="1000">
            <defs>
                <linearGradient id="g_large" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stop-color="#1B1035"/>
                    <stop offset="100%" stop-color="#FFD166"/>
                </linearGradient>
            </defs>
            <rect x="0" y="0" width="800" height="800" fill="url(#g_large)"/>
        </svg>"##;

        let cfg = IngestConfig::default();
        let small_prims = ingest_svg(small_svg, &cfg).expect("ingest small gradient");
        let large_prims = ingest_svg(large_svg, &cfg).expect("ingest large gradient");

        assert_eq!(small_prims.len(), 6);
        assert_eq!(large_prims.len(), 5);
        assert!(small_prims[5].gradient_meta.is_none(), "non-gradient rect must have gradient_meta = None");

        // Exact endpoint colors preserved on bands 0 and 4.
        assert_eq!(small_prims[0].color_srgb, [0x1B, 0x10, 0x35]);
        assert_eq!(small_prims[4].color_srgb, [0xFF, 0xD1, 0x66]);
        assert_eq!(large_prims[0].color_srgb, [0x1B, 0x10, 0x35]);
        assert_eq!(large_prims[4].color_srgb, [0xFF, 0xD1, 0x66]);

        let measure_band_overlap = |prims: &[NormalizedPrimitive]| -> (f64, f64) {
            let b0 = match &prims[0].geometry {
                PrimitiveGeom::FilledPolygon(mp) => mp.bounding_rect().unwrap(),
                _ => panic!("expected filled polygon"),
            };
            let b1 = match &prims[1].geometry {
                PrimitiveGeom::FilledPolygon(mp) => mp.bounding_rect().unwrap(),
                _ => panic!("expected filled polygon"),
            };
            let bw = match &prims[1].gradient_meta {
                Some(GradientFillMeta::Linear {
                    band_width,
                    fade_in_s,
                    fade_out_s,
                    ..
                }) => {
                    let (fi_a, fi_b) = fade_in_s.expect("middle band must fade in");
                    let (fo_a, fo_b) = fade_out_s.expect("middle band must fade out");
                    assert!(((fi_a - 0.55 * band_width)).abs() < 1e-6);
                    assert!(((fi_b - 1.18 * band_width)).abs() < 1e-6);
                    assert!(((fo_a - 2.04 * band_width)).abs() < 1e-6);
                    assert!(((fo_b - 2.20 * band_width)).abs() < 1e-6);
                    *band_width
                }
                other => panic!("expected Linear gradient meta, got {other:?}"),
            };
            assert!(
                matches!(
                    &prims[0].gradient_meta,
                    Some(GradientFillMeta::Linear { fade_in_s: None, fade_out_s: Some(_), .. })
                ),
                "first band should have fade_in_s = None"
            );
            assert!(
                matches!(
                    &prims[4].gradient_meta,
                    Some(GradientFillMeta::Linear { fade_in_s: Some(_), fade_out_s: None, .. })
                ),
                "last band should have fade_out_s = None"
            );
            (bw, b0.max().x - b1.min().x)
        };

        let (small_bw, small_overlap) = measure_band_overlap(&small_prims);
        let (large_bw, large_overlap) = measure_band_overlap(&large_prims);

        // Large rect is 8x wider than small rect -> band_width and overlap must both scale 8x.
        assert!((large_bw / small_bw - 8.0).abs() < 1e-4);
        assert!((small_overlap - 0.65 * small_bw).abs() < 1e-5);
        assert!((large_overlap - 0.65 * large_bw).abs() < 1e-5);
        assert!((large_overlap / small_overlap - 8.0).abs() < 1e-3);
    }
}
