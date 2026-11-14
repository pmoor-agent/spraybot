use std::f32::consts::PI;
use std::thread::{self, JoinHandle};

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::mesh::Indices;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use crossbeam_channel::{Receiver, Sender};

use crate::core::trajectory::CompiledTimeline;
use crate::core::types::{NozzleCap, PalettePlan, RobotFrameState};
use crate::sim::state::{AppConfig, SimClock, SimStepSet, SprayJobRes};

/// Spray cone half-angle (12 degrees).
pub const CONE_HALF_ANGLE_RAD: f32 = 12.0_f32 * PI / 180.0;

/// Warm unprimed gesso off-white (#F4F1EA).
pub const GESSO_SRGB: [u8; 3] = [244, 241, 234];

/// Bare matte canvas roughness and saturated painted acrylic satin roughness.
pub const BARE_CANVAS_ROUGHNESS: f32 = 0.88;
pub const PAINTED_ACRYLIC_ROUGHNESS: f32 = 0.32;

/// Calibrated Beer-Lambert flow rate so single normal passes reach ~96% coverage cleanly.
const SPRAY_FLOW_RATE: f32 = 0.075;

/// Thickness increment factor per Beer-Lambert step so 1-2 passes transition to satin acrylic.
const THICKNESS_ACCUM_SCALE: f32 = 0.38;

/// Base adaptive time step (seconds) for swept-segment timeline deposition.
const SWEPT_STEP_DT: f64 = 0.008;

/// Maximum segment nozzle displacement (meters) before subdividing a swept interval.
const MAX_SWEPT_DISPLACEMENT_M: f32 = 0.04;

/// Minimum nozzle direction dot product before subdividing a sharply turning swept interval.
const MIN_SWEPT_DIR_COS: f32 = 0.995;

/// Maximum midpoint chord sagitta (meters) before subdividing a curved swept interval.
const MAX_SWEPT_SAGITTA_M: f32 = 0.002;

/// Maximum recursion depth for subdividing fast or curved swept intervals.
const MAX_DEPOSIT_SUBDIV_DEPTH: u8 = 4;

/// Recursion depth for spray start/stop and cap/slot transition intervals.
const TRANSITION_SUBDIV_DEPTH: u8 = 3;

/// Registers the dynamic PBR wall canvas and ray-plane spray deposition systems.
pub fn plugin(app: &mut App) {
    app.add_systems(Startup, setup_canvas)
        .add_systems(Update, update_canvas_deposition.in_set(SimStepSet::CanvasAndFx));
}

#[derive(Component)]
struct WallCanvasMesh;

#[derive(Resource)]
struct CanvasResources {
    albedo_tex: Handle<Image>,
    metallic_roughness_tex: Handle<Image>,
    tex_width: u32,
    tex_height: u32,
    worker: CanvasPaintWorker,
    last_reset_epoch: u64,
    last_dispatched_t_sim: f64,
}

/// CPU backing buffers for linear-RGB spray blending and paint thickness accumulation.
#[derive(Debug, Clone)]
pub struct CanvasPaintBuffer {
    pub width: u32,
    pub height: u32,
    pub canvas_center: Vec3,
    pub canvas_width: f32,
    pub canvas_height: f32,
    pub seed: u64,
    pub gesso_linear: [f32; 3],
    pub linear_rgb: Vec<[f32; 3]>,
    pub thickness: Vec<f32>,
}

impl CanvasPaintBuffer {
    /// Creates a clean gesso canvas backing buffer matching the given studio dimensions.
    pub fn new(
        width: u32,
        height: u32,
        canvas_center: Vec3,
        canvas_width: f32,
        canvas_height: f32,
        seed: u64,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let gesso_linear = srgb_u8_to_linear_rgb(GESSO_SRGB);
        let count = (width as usize) * (height as usize);
        Self {
            width,
            height,
            canvas_center,
            canvas_width,
            canvas_height,
            seed,
            gesso_linear,
            linear_rgb: vec![gesso_linear; count],
            thickness: vec![0.0; count],
        }
    }

    /// Resets the CPU backing buffers back to clean unprimed gesso and zero paint thickness.
    pub fn reset(&mut self) {
        self.linear_rgb.fill(self.gesso_linear);
        self.thickness.fill(0.0);
    }

    /// Maps a world `(x, y)` coordinate on the canvas plane to `(px, py)` texel indices.
    ///
    /// Top-left `(-1.25, 2.30)` maps to `(0, 0)` and bottom-right `(+1.25, 0.80)` maps to `(W-1, H-1)`,
    /// matching Bevy's `Rectangle` UV convention.
    #[inline]
    pub fn world_to_pixel(&self, world_xy: Vec2) -> Option<(u32, u32)> {
        let min_x = self.canvas_center.x - 0.5 * self.canvas_width;
        let max_y = self.canvas_center.y + 0.5 * self.canvas_height;
        let u = (world_xy.x - min_x) / self.canvas_width;
        let v = (max_y - world_xy.y) / self.canvas_height;
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }
        let px = ((u * self.width as f32).floor() as i32).clamp(0, self.width as i32 - 1) as u32;
        let py = ((v * self.height as f32).floor() as i32).clamp(0, self.height as i32 - 1) as u32;
        Some((px, py))
    }

    /// Maps texel center `(px, py)` back to world-space 3D coordinates on the canvas plane.
    #[inline]
    pub fn pixel_to_world(&self, px: u32, py: u32) -> Vec3 {
        let u = (px as f32 + 0.5) / self.width as f32;
        let v = (py as f32 + 0.5) / self.height as f32;
        let wx = (self.canvas_center.x - 0.5 * self.canvas_width) + u * self.canvas_width;
        let wy = (self.canvas_center.y + 0.5 * self.canvas_height) - v * self.canvas_height;
        Vec3::new(wx, wy, self.canvas_center.z)
    }

    /// Computes PBR perceptual roughness in `[0.32, 0.88]` from accumulated paint thickness `h`.
    #[inline]
    pub fn roughness_at(&self, px: u32, py: u32) -> f32 {
        let idx = (py as usize) * (self.width as usize) + (px as usize);
        roughness_from_thickness(self.thickness[idx])
    }

    /// Deposits one sub-step of aerosol spray from `(nozzle_pos, nozzle_dir)` onto the canvas plane
    /// using the default `Fat` nozzle cap.
    ///
    /// Returns the inclusive dirty pixel bounding box `(min_x, min_y, max_x, max_y)` if any texel changed.
    pub fn splat_cone(
        &mut self,
        nozzle_pos: Vec3,
        nozzle_dir: Vec3,
        paint_linear: [f32; 3],
        flow_scale: f32,
        dt_sub: f32,
    ) -> Option<(u32, u32, u32, u32)> {
        self.splat_cone_with_cap(
            nozzle_pos,
            nozzle_dir,
            paint_linear,
            flow_scale,
            dt_sub,
            NozzleCap::Fat,
        )
    }

    /// Deposits one sub-step of aerosol spray from `(nozzle_pos, nozzle_dir)` onto the canvas plane
    /// using the specified [`NozzleCap`] cone half-angle and orifice flow factor.
    ///
    /// Returns the inclusive dirty pixel bounding box `(min_x, min_y, max_x, max_y)` if any texel changed.
    pub fn splat_cone_with_cap(
        &mut self,
        nozzle_pos: Vec3,
        nozzle_dir: Vec3,
        paint_linear: [f32; 3],
        flow_scale: f32,
        dt_sub: f32,
        cap: NozzleCap,
    ) -> Option<(u32, u32, u32, u32)> {
        if flow_scale <= 0.0 || dt_sub <= 0.0 {
            return None;
        }
        let dir = nozzle_dir.normalize_or_zero();
        if dir.length_squared() < 0.5 || dir.z >= -1e-4 {
            return None;
        }

        let z_plane = self.canvas_center.z;
        let t_hit = (z_plane - nozzle_pos.z) / dir.z;
        if !(0.002..=1.2).contains(&t_hit) {
            return None;
        }

        let hit = nozzle_pos + dir * t_hit;
        let tan_half = cap.cone_half_angle_rad().tan();
        let spot_radius = (t_hit * tan_half).max(1e-4);
        let cos_phi = (-dir.z).clamp(0.25, 1.0);
        let max_extent = 1.45 * spot_radius / cos_phi;

        let (px_min, py_min, px_max, py_max) = self.world_bounds_to_pixels(
            hit.x - max_extent,
            hit.x + max_extent,
            hit.y - max_extent,
            hit.y + max_extent,
        )?;

        // Scale area by 1 / cos(phi) to conserve paint mass across oblique incidence ellipses.
        let plume_area = (PI * spot_radius * spot_radius / cos_phi).max(1e-6);
        let dose_coeff =
            SPRAY_FLOW_RATE * cap.orifice_flow_factor() * flow_scale * dt_sub / plume_area;

        let canvas_left = self.canvas_center.x - 0.5 * self.canvas_width;
        let canvas_top = self.canvas_center.y + 0.5 * self.canvas_height;
        let w = self.width as usize;

        let mut dirty_x0 = u32::MAX;
        let mut dirty_y0 = u32::MAX;
        let mut dirty_x1 = 0_u32;
        let mut dirty_y1 = 0_u32;

        for py in py_min..=py_max {
            let v_coord = (py as f32 + 0.5) / self.height as f32;
            let wy = canvas_top - v_coord * self.canvas_height;
            let row_offset = (py as usize) * w;

            for px in px_min..=px_max {
                let u_coord = (px as f32 + 0.5) / self.width as f32;
                let wx = canvas_left + u_coord * self.canvas_width;

                let rel = Vec3::new(wx - nozzle_pos.x, wy - nozzle_pos.y, z_plane - nozzle_pos.z);
                let axial = rel.dot(dir);
                if axial <= 1e-4 {
                    continue;
                }

                let perp_sq = (rel - dir * axial).length_squared();
                let r_local = axial * tan_half;
                let u_cone_sq = perp_sq / (r_local * r_local).max(1e-8);
                if u_cone_sq > 1.35 * 1.35 {
                    continue;
                }

                let u_cone = u_cone_sq.sqrt();
                let gaussian = (-2.2 * u_cone_sq).exp();
                let falloff = apply_rim_modulation(px, py, self.seed, u_cone, gaussian);

                let alpha_step = 1.0 - (-dose_coeff * falloff).exp();
                if alpha_step <= 1e-5 {
                    continue;
                }

                let idx = row_offset + (px as usize);
                self.deposit_texel(idx, paint_linear, alpha_step, alpha_step);

                dirty_x0 = dirty_x0.min(px);
                dirty_y0 = dirty_y0.min(py);
                dirty_x1 = dirty_x1.max(px);
                dirty_y1 = dirty_y1.max(py);
            }
        }

        if dirty_x0 <= dirty_x1 && dirty_y0 <= dirty_y1 {
            Some((dirty_x0, dirty_y0, dirty_x1, dirty_y1))
        } else {
            None
        }
    }

    /// Deposits a continuous swept spray interval `(pos0, dir0) -> (pos1, dir1)` over `dt` seconds
    /// using an analytical Gaussian swept-chord integral on a single bounding-box pass.
    ///
    /// Falls back to [`Self::splat_cone_with_cap`] when ray-plane displacement is below `0.25 * avg_r`.
    #[allow(clippy::too_many_arguments)]
    pub fn splat_swept_segment_with_cap(
        &mut self,
        pos0: Vec3,
        dir0: Vec3,
        pos1: Vec3,
        dir1: Vec3,
        paint_linear: [f32; 3],
        flow_scale: f32,
        dt: f32,
        cap: NozzleCap,
    ) -> Option<(u32, u32, u32, u32)> {
        if flow_scale <= 0.0 || dt <= 0.0 {
            return None;
        }

        let d0 = dir0.normalize_or_zero();
        let d1 = dir1.normalize_or_zero();
        let z_plane = self.canvas_center.z;

        let t0 = if d0.length_squared() >= 0.5 && d0.z < -1e-4 {
            (z_plane - pos0.z) / d0.z
        } else {
            -1.0
        };
        let t1 = if d1.length_squared() >= 0.5 && d1.z < -1e-4 {
            (z_plane - pos1.z) / d1.z
        } else {
            -1.0
        };
        let t0_valid = (0.002..=1.2).contains(&t0);
        let t1_valid = (0.002..=1.2).contains(&t1);

        match (t0_valid, t1_valid) {
            (false, false) => return None,
            (true, false) => {
                return self.splat_cone_with_cap(pos0, d0, paint_linear, flow_scale, dt * 0.5, cap);
            }
            (false, true) => {
                return self.splat_cone_with_cap(pos1, d1, paint_linear, flow_scale, dt * 0.5, cap);
            }
            (true, true) => {}
        }

        let h0 = pos0 + d0 * t0;
        let h1 = pos1 + d1 * t1;
        let tan_half = cap.cone_half_angle_rad().tan();
        let r0 = (t0 * tan_half).max(1e-4);
        let r1 = (t1 * tan_half).max(1e-4);
        let avg_r = 0.5 * (r0 + r1);

        let seg_xy = Vec2::new(h1.x - h0.x, h1.y - h0.y);
        let seg_len = seg_xy.length();
        let dir_mid = (d0 + d1).normalize_or(d0);
        if seg_len < 0.25 * avg_r {
            let pos_mid = 0.5 * (pos0 + pos1);
            return self.splat_cone_with_cap(pos_mid, dir_mid, paint_linear, flow_scale, dt, cap);
        }

        let cos_phi = (-dir_mid.z).clamp(0.25, 1.0);
        let min_cos = ((-d0.z).clamp(0.25, 1.0))
            .min((-d1.z).clamp(0.25, 1.0))
            .min(cos_phi);
        let ext = 1.45 * r0.max(r1) / min_cos;

        let (px_min, py_min, px_max, py_max) = self.world_bounds_to_pixels(
            h0.x.min(h1.x) - ext,
            h0.x.max(h1.x) + ext,
            h0.y.min(h1.y) - ext,
            h0.y.max(h1.y) + ext,
        )?;

        // Project onto the plane perpendicular to dir_mid so oblique incidence stretches cleanly.
        let dir_xy = Vec2::new(dir_mid.x, dir_mid.y);
        let seg_m = Vec3::new(seg_xy.x, seg_xy.y, 0.0) - dir_mid * seg_xy.dot(dir_xy);
        let seg_m_len = seg_m.length();
        if seg_m_len < 1e-6 {
            let pos_mid = 0.5 * (pos0 + pos1);
            return self.splat_cone_with_cap(pos_mid, dir_mid, paint_linear, flow_scale, dt, cap);
        }

        let seg_m_dir = seg_m / seg_m_len;
        let inv_seg_m_len = 1.0 / seg_m_len;
        let r = avg_r;
        let inv_r_sq = 1.0 / (r * r);
        let l_over_r = seg_m_len / r;
        let r_over_l = r / seg_m_len;
        let k_erf = 2.2_f32.sqrt() * l_over_r;
        let prefactor = r_over_l * (PI / 8.8_f32).sqrt();

        // Scale area by 1 / cos(phi) to conserve paint mass across oblique incidence ellipses.
        let plume_area = (PI * r * r / cos_phi).max(1e-6);
        let dose_coeff =
            SPRAY_FLOW_RATE * cap.orifice_flow_factor() * flow_scale * dt / plume_area;

        let canvas_left = self.canvas_center.x - 0.5 * self.canvas_width;
        let canvas_top = self.canvas_center.y + 0.5 * self.canvas_height;
        let w = self.width as usize;

        let mut dirty_x0 = u32::MAX;
        let mut dirty_y0 = u32::MAX;
        let mut dirty_x1 = 0_u32;
        let mut dirty_y1 = 0_u32;

        for py in py_min..=py_max {
            let v_coord = (py as f32 + 0.5) / self.height as f32;
            let wy = canvas_top - v_coord * self.canvas_height;
            let row_offset = (py as usize) * w;

            for px in px_min..=px_max {
                let u_coord = (px as f32 + 0.5) / self.width as f32;
                let wx = canvas_left + u_coord * self.canvas_width;

                let d_xy = Vec2::new(wx - h0.x, wy - h0.y);
                let d_m = Vec3::new(d_xy.x, d_xy.y, 0.0) - dir_mid * d_xy.dot(dir_xy);
                let along_m = d_m.dot(seg_m_dir);
                let s = along_m * inv_seg_m_len;
                let perp_m_sq = (d_m - seg_m_dir * along_m).length_squared().max(0.0);
                let u_perp_sq = perp_m_sq * inv_r_sq;
                if u_perp_sq > 1.35 * 1.35 {
                    continue;
                }

                let s_clamped = s.clamp(0.0, 1.0);
                let axial_gap = l_over_r * (s - s_clamped);
                let u_min_sq = u_perp_sq + axial_gap * axial_gap;
                if u_min_sq > 1.35 * 1.35 {
                    continue;
                }

                let chord_integral =
                    prefactor * (erf_f32(k_erf * (1.0 - s)) + erf_f32(k_erf * s));
                let base_falloff = (-2.2 * u_perp_sq).exp() * chord_integral;
                let u_rim = u_min_sq.sqrt();
                let falloff = apply_rim_modulation(px, py, self.seed, u_rim, base_falloff);

                let total_dose = dose_coeff * falloff;
                let alpha_step = 1.0 - (-total_dose).exp();
                if alpha_step <= 1e-5 {
                    continue;
                }

                let idx = row_offset + (px as usize);
                self.deposit_texel(idx, paint_linear, alpha_step, total_dose);

                dirty_x0 = dirty_x0.min(px);
                dirty_y0 = dirty_y0.min(py);
                dirty_x1 = dirty_x1.max(px);
                dirty_y1 = dirty_y1.max(py);
            }
        }

        if dirty_x0 <= dirty_x1 && dirty_y0 <= dirty_y1 {
            Some((dirty_x0, dirty_y0, dirty_x1, dirty_y1))
        } else {
            None
        }
    }

    /// Deposits spray paint across timeline interval `[t_from, t_to]` using adaptive swept segments
    /// (~`0.008s` step), subdividing on sharp turns or high displacement and falling back to point
    /// splats at spray start/stop transitions.
    pub fn deposit_timeline_interval(
        &mut self,
        timeline: &CompiledTimeline,
        palette: &PalettePlan,
        t_from: f64,
        t_to: f64,
    ) -> Option<(u32, u32, u32, u32)> {
        if t_to <= t_from + 1e-9 {
            return None;
        }

        let mut dirty: Option<(u32, u32, u32, u32)> = None;
        let mut t = t_from;
        let mut s0 = timeline.sample(t);

        while t < t_to - 1e-9 {
            let dt_step = (t_to - t).min(SWEPT_STEP_DT);
            let t_next = t + dt_step;
            let s1 = timeline.sample(t_next);

            if s0.spray_active || s1.spray_active {
                let rect = self.deposit_sub_interval(timeline, palette, t, t_next, &s0, &s1, 0);
                dirty = merge_dirty_rect(dirty, rect);
            }

            t = t_next;
            s0 = s1;
        }

        dirty
    }

    #[allow(clippy::too_many_arguments)]
    fn deposit_sub_interval(
        &mut self,
        timeline: &CompiledTimeline,
        palette: &PalettePlan,
        t0: f64,
        t1: f64,
        s0: &RobotFrameState,
        s1: &RobotFrameState,
        depth: u8,
    ) -> Option<(u32, u32, u32, u32)> {
        let dt = t1 - t0;
        if dt <= 1e-9 || (!s0.spray_active && !s1.spray_active) {
            return None;
        }

        let same_continuous_spray = s0.spray_active
            && s1.spray_active
            && s0.held_can_slot.is_some()
            && s0.held_can_slot == s1.held_can_slot
            && s0.active_cap == s1.active_cap;

        if same_continuous_spray {
            let disp = (s1.nozzle_world_pos - s0.nozzle_world_pos).length();
            let dir_cos = s0
                .nozzle_world_dir
                .normalize_or_zero()
                .dot(s1.nozzle_world_dir.normalize_or_zero());
            let flow_diff = (s0.flow_scale - s1.flow_scale).abs();

            let mut needs_subdiv = disp > MAX_SWEPT_DISPLACEMENT_M
                || dir_cos < MIN_SWEPT_DIR_COS
                || flow_diff > 0.15;

            let t_mid = 0.5 * (t0 + t1);
            let mut s_mid_opt: Option<RobotFrameState> = None;
            if !needs_subdiv && depth < MAX_DEPOSIT_SUBDIV_DEPTH {
                let s_mid = timeline.sample(t_mid);
                let chord_mid = 0.5 * (s0.nozzle_world_pos + s1.nozzle_world_pos);
                let sagitta = (s_mid.nozzle_world_pos - chord_mid).length();
                if sagitta > MAX_SWEPT_SAGITTA_M
                    || !s_mid.spray_active
                    || s_mid.held_can_slot != s0.held_can_slot
                    || s_mid.active_cap != s0.active_cap
                {
                    needs_subdiv = true;
                }
                s_mid_opt = Some(s_mid);
            }

            if needs_subdiv && depth < MAX_DEPOSIT_SUBDIV_DEPTH {
                let s_mid = s_mid_opt.unwrap_or_else(|| timeline.sample(t_mid));
                let left =
                    self.deposit_sub_interval(timeline, palette, t0, t_mid, s0, &s_mid, depth + 1);
                let right =
                    self.deposit_sub_interval(timeline, palette, t_mid, t1, &s_mid, s1, depth + 1);
                return merge_dirty_rect(left, right);
            }

            let slot = s0.held_can_slot?;
            let can = palette.cans.get(slot)?;
            let paint_linear = srgb_u8_to_linear_rgb(can.color_srgb);
            let flow_scale = 0.5 * (s0.flow_scale + s1.flow_scale);
            return self.splat_swept_segment_with_cap(
                s0.nozzle_world_pos,
                s0.nozzle_world_dir,
                s1.nozzle_world_pos,
                s1.nozzle_world_dir,
                paint_linear,
                flow_scale,
                dt as f32,
                s0.active_cap,
            );
        }

        // Spray start/stop or cap/slot transition: subdivide and fall back to point splat.
        let t_mid = 0.5 * (t0 + t1);
        if depth < TRANSITION_SUBDIV_DEPTH {
            let s_mid = timeline.sample(t_mid);
            let left =
                self.deposit_sub_interval(timeline, palette, t0, t_mid, s0, &s_mid, depth + 1);
            let right =
                self.deposit_sub_interval(timeline, palette, t_mid, t1, &s_mid, s1, depth + 1);
            return merge_dirty_rect(left, right);
        }

        let frame = timeline.sample(t_mid);
        if !frame.spray_active {
            return None;
        }
        let slot = frame.held_can_slot?;
        let can = palette.cans.get(slot)?;
        let paint_linear = srgb_u8_to_linear_rgb(can.color_srgb);
        self.splat_cone_with_cap(
            frame.nozzle_world_pos,
            frame.nozzle_world_dir,
            paint_linear,
            frame.flow_scale,
            dt as f32,
            frame.active_cap,
        )
    }

    #[inline]
    fn world_bounds_to_pixels(
        &self,
        min_x_world: f32,
        max_x_world: f32,
        min_y_world: f32,
        max_y_world: f32,
    ) -> Option<(u32, u32, u32, u32)> {
        let canvas_left = self.canvas_center.x - 0.5 * self.canvas_width;
        let canvas_right = self.canvas_center.x + 0.5 * self.canvas_width;
        let canvas_bottom = self.canvas_center.y - 0.5 * self.canvas_height;
        let canvas_top = self.canvas_center.y + 0.5 * self.canvas_height;

        if max_x_world < canvas_left
            || min_x_world > canvas_right
            || max_y_world < canvas_bottom
            || min_y_world > canvas_top
        {
            return None;
        }

        let u_min = ((min_x_world - canvas_left) / self.canvas_width).clamp(0.0, 1.0);
        let u_max = ((max_x_world - canvas_left) / self.canvas_width).clamp(0.0, 1.0);
        // Note: world top (+Y) is v = 0, world bottom (-Y) is v = 1.
        let v_min = ((canvas_top - max_y_world) / self.canvas_height).clamp(0.0, 1.0);
        let v_max = ((canvas_top - min_y_world) / self.canvas_height).clamp(0.0, 1.0);

        let px_min =
            ((u_min * self.width as f32).floor() as i32).clamp(0, self.width as i32 - 1) as u32;
        let px_max =
            ((u_max * self.width as f32).ceil() as i32).clamp(0, self.width as i32 - 1) as u32;
        let py_min =
            ((v_min * self.height as f32).floor() as i32).clamp(0, self.height as i32 - 1) as u32;
        let py_max =
            ((v_max * self.height as f32).ceil() as i32).clamp(0, self.height as i32 - 1) as u32;

        if px_min > px_max || py_min > py_max {
            None
        } else {
            Some((px_min, py_min, px_max, py_max))
        }
    }

    #[inline]
    fn deposit_texel(
        &mut self,
        idx: usize,
        paint_linear: [f32; 3],
        alpha_step: f32,
        thickness_dose: f32,
    ) {
        let dst = &mut self.linear_rgb[idx];
        let inv = 1.0 - alpha_step;
        dst[0] = dst[0] * inv + paint_linear[0] * alpha_step;
        dst[1] = dst[1] * inv + paint_linear[1] * alpha_step;
        dst[2] = dst[2] * inv + paint_linear[2] * alpha_step;
        self.thickness[idx] =
            (self.thickness[idx] + thickness_dose * THICKNESS_ACCUM_SCALE).min(1.5);
    }

    /// Encodes a tightly packed `rect_w * rect_h * 4` RGBA8 patch for the dirty bounding box.
    pub fn encode_dirty_patch(
        &self,
        min_x: u32,
        min_y: u32,
        max_x: u32,
        max_y: u32,
        reset_epoch: u64,
    ) -> PaintPatch {
        let x0 = min_x.min(self.width.saturating_sub(1));
        let x1 = max_x.min(self.width.saturating_sub(1)).max(x0);
        let y0 = min_y.min(self.height.saturating_sub(1));
        let y1 = max_y.min(self.height.saturating_sub(1)).max(y0);

        let rect_w = (x1 - x0 + 1) as usize;
        let rect_h = (y1 - y0 + 1) as usize;
        let byte_len = rect_w * rect_h * 4;
        let mut albedo_patch = vec![0_u8; byte_len];
        let mut mr_patch = vec![0_u8; byte_len];
        let w = self.width as usize;

        for dy in 0..rect_h {
            let src_row = (y0 as usize + dy) * w;
            let dst_row = dy * rect_w;
            for dx in 0..rect_w {
                let src_idx = src_row + (x0 as usize + dx);
                let dst_base = (dst_row + dx) * 4;

                let lin = self.linear_rgb[src_idx];
                albedo_patch[dst_base] = linear_f32_to_srgb_u8(lin[0]);
                albedo_patch[dst_base + 1] = linear_f32_to_srgb_u8(lin[1]);
                albedo_patch[dst_base + 2] = linear_f32_to_srgb_u8(lin[2]);
                albedo_patch[dst_base + 3] = 255;

                let rough = roughness_from_thickness(self.thickness[src_idx]);
                mr_patch[dst_base] = 0;
                mr_patch[dst_base + 1] = (rough * 255.0).round().clamp(0.0, 255.0) as u8;
                mr_patch[dst_base + 2] = 0;
                mr_patch[dst_base + 3] = 255;
            }
        }

        PaintPatch {
            reset_epoch,
            min_x: x0,
            min_y: y0,
            max_x: x1,
            max_y: y1,
            albedo_patch,
            mr_patch,
        }
    }

    /// Encodes the dirty pixel bounding box into `albedo_bytes` (sRGB RGBA8).
    pub fn sync_dirty_albedo(
        &self,
        min_x: u32,
        min_y: u32,
        max_x: u32,
        max_y: u32,
        albedo_bytes: &mut [u8],
    ) {
        let w = self.width as usize;
        let x0 = min_x.min(self.width.saturating_sub(1));
        let x1 = max_x.min(self.width.saturating_sub(1));
        let y0 = min_y.min(self.height.saturating_sub(1));
        let y1 = max_y.min(self.height.saturating_sub(1));

        for py in y0..=y1 {
            let row = (py as usize) * w;
            for px in x0..=x1 {
                let idx = row + (px as usize);
                let base = idx * 4;
                let lin = self.linear_rgb[idx];
                albedo_bytes[base] = linear_f32_to_srgb_u8(lin[0]);
                albedo_bytes[base + 1] = linear_f32_to_srgb_u8(lin[1]);
                albedo_bytes[base + 2] = linear_f32_to_srgb_u8(lin[2]);
                albedo_bytes[base + 3] = 255;
            }
        }
    }

    /// Encodes the dirty pixel bounding box into `mr_bytes` (linear RGBA8: G=roughness, B=0 metallic).
    pub fn sync_dirty_roughness(
        &self,
        min_x: u32,
        min_y: u32,
        max_x: u32,
        max_y: u32,
        mr_bytes: &mut [u8],
    ) {
        let w = self.width as usize;
        let x0 = min_x.min(self.width.saturating_sub(1));
        let x1 = max_x.min(self.width.saturating_sub(1));
        let y0 = min_y.min(self.height.saturating_sub(1));
        let y1 = max_y.min(self.height.saturating_sub(1));

        for py in y0..=y1 {
            let row = (py as usize) * w;
            for px in x0..=x1 {
                let idx = row + (px as usize);
                let base = idx * 4;
                let rough = roughness_from_thickness(self.thickness[idx]);
                mr_bytes[base] = 0;
                mr_bytes[base + 1] = (rough * 255.0).round().clamp(0.0, 255.0) as u8;
                mr_bytes[base + 2] = 0; // metallic = 0
                mr_bytes[base + 3] = 255;
            }
        }
    }

    /// Encodes the dirty pixel bounding box into `albedo_bytes` (sRGB RGBA8) and `mr_bytes` (linear RGBA8).
    pub fn sync_dirty_rect(
        &self,
        min_x: u32,
        min_y: u32,
        max_x: u32,
        max_y: u32,
        albedo_bytes: &mut [u8],
        mr_bytes: &mut [u8],
    ) {
        self.sync_dirty_albedo(min_x, min_y, max_x, max_y, albedo_bytes);
        self.sync_dirty_roughness(min_x, min_y, max_x, max_y, mr_bytes);
    }
}

/// Tightly packed RGBA8 dirty-rect patch produced by [`CanvasPaintWorker`].
#[derive(Debug, Clone)]
pub struct PaintPatch {
    pub reset_epoch: u64,
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
    pub albedo_patch: Vec<u8>,
    pub mr_patch: Vec<u8>,
}

enum PaintCmd {
    AdvanceTo {
        t_target: f64,
        reset_epoch: u64,
        sync_reply: Option<Sender<()>>,
    },
    Shutdown,
}

/// Persistent background thread owning CPU canvas paint deposition and patch encoding.
pub struct CanvasPaintWorker {
    cmd_tx: Sender<PaintCmd>,
    patch_rx: Receiver<PaintPatch>,
    join_handle: Option<JoinHandle<()>>,
}

impl CanvasPaintWorker {
    /// Spawns the dedicated background paint worker thread.
    pub fn new(
        buffer: CanvasPaintBuffer,
        timeline: CompiledTimeline,
        palette: PalettePlan,
        initial_reset_epoch: u64,
    ) -> Self {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<PaintCmd>();
        let (patch_tx, patch_rx) = crossbeam_channel::unbounded::<PaintPatch>();

        let join_handle = thread::Builder::new()
            .name("canvas-paint-worker".to_string())
            .spawn(move || {
                run_paint_worker(
                    buffer,
                    timeline,
                    palette,
                    initial_reset_epoch,
                    cmd_rx,
                    patch_tx,
                );
            })
            .expect("failed to spawn canvas-paint-worker thread");

        Self {
            cmd_tx,
            patch_rx,
            join_handle: Some(join_handle),
        }
    }

    /// Queues a non-blocking advance request for live playback.
    pub fn advance_async(&self, t_target: f64, reset_epoch: u64) {
        let _ = self.cmd_tx.send(PaintCmd::AdvanceTo {
            t_target,
            reset_epoch,
            sync_reply: None,
        });
    }

    /// Advances synchronously until the worker finishes `[worker_t, t_target]` and queues any patch.
    pub fn advance_sync(&self, t_target: f64, reset_epoch: u64) {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded::<()>(1);
        if self
            .cmd_tx
            .send(PaintCmd::AdvanceTo {
                t_target,
                reset_epoch,
                sync_reply: Some(reply_tx),
            })
            .is_ok()
        {
            let _ = reply_rx.recv();
        }
    }

    /// Drains all currently available encoded dirty-rect patches without blocking.
    pub fn drain_patches(&self) -> impl Iterator<Item = PaintPatch> + '_ {
        self.patch_rx.try_iter()
    }
}

impl Drop for CanvasPaintWorker {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(PaintCmd::Shutdown);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_paint_worker(
    mut buffer: CanvasPaintBuffer,
    timeline: CompiledTimeline,
    palette: PalettePlan,
    initial_epoch: u64,
    cmd_rx: Receiver<PaintCmd>,
    patch_tx: Sender<PaintPatch>,
) {
    let mut worker_epoch = initial_epoch;
    let mut worker_t = 0.0_f64;

    loop {
        let first_cmd = match cmd_rx.recv() {
            Ok(cmd) => cmd,
            Err(_) => break,
        };

        let mut target_t: f64;
        let mut target_epoch: u64;
        let mut pending_replies: Vec<Sender<()>> = Vec::new();
        let mut shutdown = false;

        match first_cmd {
            PaintCmd::Shutdown => break,
            PaintCmd::AdvanceTo {
                t_target,
                reset_epoch,
                sync_reply,
            } => {
                target_t = t_target;
                target_epoch = reset_epoch;
                if let Some(tx) = sync_reply {
                    pending_replies.push(tx);
                }
            }
        }

        // Coalesce queued AdvanceTo commands for the same reset_epoch so we advance
        // continuously across [worker_t, latest_t_target] without falling behind or skipping paint.
        while let Ok(next_cmd) = cmd_rx.try_recv() {
            match next_cmd {
                PaintCmd::Shutdown => {
                    shutdown = true;
                    break;
                }
                PaintCmd::AdvanceTo {
                    t_target,
                    reset_epoch,
                    sync_reply,
                } => {
                    if reset_epoch != target_epoch {
                        for tx in pending_replies.drain(..) {
                            let _ = tx.send(());
                        }
                        target_epoch = reset_epoch;
                        target_t = t_target;
                    } else {
                        target_t = target_t.max(t_target);
                    }
                    if let Some(tx) = sync_reply {
                        pending_replies.push(tx);
                    }
                }
            }
        }

        if shutdown {
            for tx in pending_replies.drain(..) {
                let _ = tx.send(());
            }
            break;
        }

        if target_epoch != worker_epoch {
            buffer.reset();
            worker_epoch = target_epoch;
            worker_t = 0.0;
        }

        if target_t > worker_t + 1e-9 {
            if let Some((min_x, min_y, max_x, max_y)) =
                buffer.deposit_timeline_interval(&timeline, &palette, worker_t, target_t)
            {
                let patch = buffer.encode_dirty_patch(min_x, min_y, max_x, max_y, worker_epoch);
                let _ = patch_tx.send(patch);
            }
            worker_t = target_t;
        }

        for tx in pending_replies.drain(..) {
            let _ = tx.send(());
        }
    }
}

/// Blits a tightly packed `rect_w * rect_h * 4` RGBA8 patch into a full `tex_w * tex_h * 4` buffer.
pub fn blit_patch_to_rgba(
    tex_bytes: &mut [u8],
    tex_w: u32,
    tex_h: u32,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    patch: &[u8],
) {
    if min_x > max_x || min_y > max_y || max_x >= tex_w || max_y >= tex_h {
        return;
    }
    let rect_w = (max_x - min_x + 1) as usize;
    let rect_h = (max_y - min_y + 1) as usize;
    let row_bytes = rect_w * 4;
    if patch.len() != rect_h * row_bytes
        || tex_bytes.len() < (tex_w as usize) * (tex_h as usize) * 4
    {
        return;
    }

    let stride = (tex_w as usize) * 4;
    for dy in 0..rect_h {
        let py = (min_y as usize) + dy;
        let src_start = dy * row_bytes;
        let dst_start = py * stride + (min_x as usize) * 4;
        tex_bytes[dst_start..dst_start + row_bytes]
            .copy_from_slice(&patch[src_start..src_start + row_bytes]);
    }
}

fn setup_canvas(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    app_config: Res<AppConfig>,
    clock: Res<SimClock>,
    spray_job: Res<SprayJobRes>,
) {
    let studio = &app_config.studio;
    let tex_width = app_config.cli.canvas_res.max(1);
    let tex_height =
        ((tex_width as f64 * studio.canvas_height as f64) / studio.canvas_width as f64).max(1.0)
            as u32;

    let mut albedo_image = Image::new_fill(
        Extent3d {
            width: tex_width,
            height: tex_height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[GESSO_SRGB[0], GESSO_SRGB[1], GESSO_SRGB[2], 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    albedo_image.sampler = ImageSampler::linear();
    let albedo_tex = images.add(albedo_image);

    let initial_roughness_byte = (BARE_CANVAS_ROUGHNESS * 255.0).round() as u8;
    let mut mr_image = Image::new_fill(
        Extent3d {
            width: tex_width,
            height: tex_height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, initial_roughness_byte, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mr_image.sampler = ImageSampler::linear();
    let metallic_roughness_tex = images.add(mr_image);

    let normal_w = 1024_u32;
    let normal_h =
        ((normal_w as f64 * studio.canvas_height as f64) / studio.canvas_width as f64).max(1.0)
            as u32;
    let normal_tex = images.add(build_linen_weave_normal_map(normal_w, normal_h));

    let mesh = Mesh::from(Rectangle::new(studio.canvas_width, studio.canvas_height))
        .with_generated_tangents()
        .expect("Rectangle mesh tangent generation should succeed");
    let mesh_handle = meshes.add(mesh);

    let material_handle = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(albedo_tex.clone()),
        normal_map_texture: Some(normal_tex),
        metallic_roughness_texture: Some(metallic_roughness_tex.clone()),
        perceptual_roughness: 1.0,
        metallic: 0.0,
        reflectance: 0.45,
        ..default()
    });

    commands.spawn((
        Name::new("WallCanvas"),
        WallCanvasMesh,
        Mesh3d(mesh_handle),
        MeshMaterial3d(material_handle),
        Transform::from_translation(Vec3::new(
            studio.canvas_center.x,
            studio.canvas_center.y,
            studio.canvas_center.z + 0.001,
        )),
    ));

    let buffer = CanvasPaintBuffer::new(
        tex_width,
        tex_height,
        studio.canvas_center,
        studio.canvas_width,
        studio.canvas_height,
        app_config.cli.seed,
    );
    let worker = CanvasPaintWorker::new(
        buffer,
        spray_job.timeline.clone(),
        spray_job.palette.clone(),
        clock.reset_epoch,
    );

    commands.insert_resource(CanvasResources {
        albedo_tex,
        metallic_roughness_tex,
        tex_width,
        tex_height,
        worker,
        last_reset_epoch: clock.reset_epoch,
        last_dispatched_t_sim: 0.0,
    });
}

fn reset_canvas_textures(
    albedo_tex: &Handle<Image>,
    mr_tex: &Handle<Image>,
    images: &mut Assets<Image>,
) {
    let initial_roughness_byte = (BARE_CANVAS_ROUGHNESS * 255.0).round() as u8;
    if let Some(mut albedo) = images.get_mut(albedo_tex)
        && let Some(data) = albedo.data.as_mut()
    {
        for px in data.chunks_exact_mut(4) {
            px[0] = GESSO_SRGB[0];
            px[1] = GESSO_SRGB[1];
            px[2] = GESSO_SRGB[2];
            px[3] = 255;
        }
    }
    if let Some(mut mr) = images.get_mut(mr_tex)
        && let Some(data) = mr.data.as_mut()
    {
        for px in data.chunks_exact_mut(4) {
            px[0] = 0;
            px[1] = initial_roughness_byte;
            px[2] = 0;
            px[3] = 255;
        }
    }
}

fn blit_worker_patches(canvas_res: &CanvasResources, images: &mut Assets<Image>) {
    let patches: Vec<PaintPatch> = canvas_res
        .worker
        .drain_patches()
        .filter(|patch| patch.reset_epoch == canvas_res.last_reset_epoch)
        .collect();
    if patches.is_empty() {
        return;
    }

    if let Some(mut albedo) = images.get_mut(&canvas_res.albedo_tex)
        && let Some(albedo_bytes) = albedo.data.as_mut()
    {
        for patch in &patches {
            blit_patch_to_rgba(
                albedo_bytes,
                canvas_res.tex_width,
                canvas_res.tex_height,
                patch.min_x,
                patch.min_y,
                patch.max_x,
                patch.max_y,
                &patch.albedo_patch,
            );
        }
    }

    if let Some(mut mr) = images.get_mut(&canvas_res.metallic_roughness_tex)
        && let Some(mr_bytes) = mr.data.as_mut()
    {
        for patch in &patches {
            blit_patch_to_rgba(
                mr_bytes,
                canvas_res.tex_width,
                canvas_res.tex_height,
                patch.min_x,
                patch.min_y,
                patch.max_x,
                patch.max_y,
                &patch.mr_patch,
            );
        }
    }
}

fn update_canvas_deposition(
    app_config: Res<AppConfig>,
    clock: Res<SimClock>,
    mut canvas_res: ResMut<CanvasResources>,
    mut images: ResMut<Assets<Image>>,
) {
    let mut reset_changed = false;
    if clock.reset_epoch != canvas_res.last_reset_epoch {
        reset_changed = true;
        canvas_res.last_reset_epoch = clock.reset_epoch;
        canvas_res.last_dispatched_t_sim = 0.0;
        reset_canvas_textures(
            &canvas_res.albedo_tex,
            &canvas_res.metallic_roughness_tex,
            &mut images,
        );
    }

    let needs_advance = reset_changed || clock.t_sim > canvas_res.last_dispatched_t_sim + 1e-9;

    if app_config.cli.record.is_some() {
        if needs_advance {
            canvas_res
                .worker
                .advance_sync(clock.t_sim, clock.reset_epoch);
            canvas_res.last_dispatched_t_sim = clock.t_sim;
        }
        blit_worker_patches(&canvas_res, &mut images);
    } else {
        blit_worker_patches(&canvas_res, &mut images);
        if needs_advance {
            canvas_res
                .worker
                .advance_async(clock.t_sim, clock.reset_epoch);
            canvas_res.last_dispatched_t_sim = clock.t_sim;
        }
    }
}

/// Generates a subtle procedural linen cross-weave tangent-space normal map (`Rgba8Unorm`).
fn build_linen_weave_normal_map(width: u32, height: u32) -> Image {
    let w = width.max(1);
    let h = height.max(1);
    let mut data = vec![0_u8; (w as usize) * (h as usize) * 4];

    // Smooth orthogonal linen cross-weave kept well below Nyquist at 360p/720p/1080p
    // to avoid diagonal moiré aliasing while preserving subtle tactile gesso relief.
    let cycles_x = 80.0_f32;
    let cycles_y = 48.0_f32;
    let bump_strength = 0.052_f32;

    let weave_height = |u: f32, v: f32| -> f32 {
        let px = u * cycles_x * 2.0 * PI;
        let py = v * cycles_y * 2.0 * PI;
        let warp = px.sin() * (0.75 + 0.25 * py.cos());
        let weft = py.sin() * (0.75 + 0.25 * px.cos());
        0.5 * (warp + weft)
    };

    let du = 1.0 / w as f32;
    let dv = 1.0 / h as f32;

    for py in 0..h {
        let v = (py as f32 + 0.5) * dv;
        for px in 0..w {
            let u = (px as f32 + 0.5) * du;
            let h_l = weave_height(u - du, v);
            let h_r = weave_height(u + du, v);
            let h_d = weave_height(u, v + dv);
            let h_u = weave_height(u, v - dv);

            let dx = (h_r - h_l) * 0.5 * bump_strength;
            let dy = (h_u - h_d) * 0.5 * bump_strength;
            let n = Vec3::new(-dx, -dy, 1.0).normalize();

            let idx = ((py as usize) * (w as usize) + (px as usize)) * 4;
            data[idx] = ((n.x * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            data[idx + 1] = ((n.y * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            data[idx + 2] = ((n.z * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            data[idx + 3] = 255;
        }
    }

    let mut image = Image::new(
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::linear();
    image
}

/// Converts a `[u8; 3]` sRGB triplet to linear RGB `[f32; 3]` (IEC 61966-2-1).
#[inline]
pub fn srgb_u8_to_linear_rgb(srgb: [u8; 3]) -> [f32; 3] {
    [
        srgb_channel_to_linear(srgb[0] as f32 / 255.0),
        srgb_channel_to_linear(srgb[1] as f32 / 255.0),
        srgb_channel_to_linear(srgb[2] as f32 / 255.0),
    ]
}

/// Converts a linear RGB `[f32; 3]` triplet back to `[u8; 3]` sRGB bytes.
#[inline]
pub fn linear_rgb_to_srgb_u8(linear: [f32; 3]) -> [u8; 3] {
    [
        linear_f32_to_srgb_u8(linear[0]),
        linear_f32_to_srgb_u8(linear[1]),
        linear_f32_to_srgb_u8(linear[2]),
    ]
}

#[inline]
fn srgb_channel_to_linear(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_f32_to_srgb_u8(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let s = if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Smoothly interpolates PBR roughness from matte `0.88` to satin acrylic `0.32` as thickness `h` grows.
#[inline]
pub fn roughness_from_thickness(thickness: f32) -> f32 {
    let t = thickness.clamp(0.0, 1.0);
    BARE_CANVAS_ROUGHNESS + (PAINTED_ACRYLIC_ROUGHNESS - BARE_CANVAS_ROUGHNESS) * t
}

/// Deterministic spatial hash in `[0.0, 1.0)` for aerosol rim speckle.
#[inline]
fn spatial_hash_01(px: u32, py: u32, seed: u64) -> f32 {
    let mut z = (px as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (py as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ seed.wrapping_mul(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32) / ((1_u32 << 24) as f32)
}

/// Applies deterministic aerosol rim droplet speckle modulation for `0.75 < u_cone <= 1.35`.
#[inline]
fn apply_rim_modulation(px: u32, py: u32, seed: u64, u_cone: f32, base_falloff: f32) -> f32 {
    if u_cone <= 0.75 {
        base_falloff
    } else {
        let rim_t = ((u_cone - 0.75) / 0.60).clamp(0.0, 1.0);
        let h = spatial_hash_01(px, py, seed);
        let droplet_prob = (1.0 - rim_t).powf(1.2) * 0.90 + 0.05;
        if h < droplet_prob {
            base_falloff * (0.70 + 0.60 * (1.0 - h))
        } else {
            base_falloff * 0.08 * (1.0 - rim_t)
        }
    }
}

#[inline]
fn merge_dirty_rect(
    acc: Option<(u32, u32, u32, u32)>,
    rect: Option<(u32, u32, u32, u32)>,
) -> Option<(u32, u32, u32, u32)> {
    match (acc, rect) {
        (Some((ax0, ay0, ax1, ay1)), Some((bx0, by0, bx1, by1))) => {
            Some((ax0.min(bx0), ay0.min(by0), ax1.max(bx1), ay1.max(by1)))
        }
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Fast Abramowitz & Stegun 7.1.26 approximation of `erf(x)` (max error < 1.5e-7).
#[inline]
fn erf_f32(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = ((((1.061_405_4 * t - 1.453_152) * t + 1.421_413_8) * t - 0.284_496_72) * t
        + 0.254_829_6)
        * t;
    let y = 1.0 - poly * (-ax * ax).exp();
    sign * y
}

/// Verifies mesh UVs match our world-to-pixel mapping on a Bevy `Rectangle`.
pub fn rectangle_uv_matches_world_mapping(
    mesh: &Mesh,
    canvas_center: Vec3,
    canvas_width: f32,
    canvas_height: f32,
    tex_w: u32,
    tex_h: u32,
) -> bool {
    let Some(bevy::mesh::VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return false;
    };
    let Some(bevy::mesh::VertexAttributeValues::Float32x2(uvs)) =
        mesh.attribute(Mesh::ATTRIBUTE_UV_0)
    else {
        return false;
    };
    if positions.len() != uvs.len() || positions.is_empty() {
        return false;
    }

    let _ = Indices::U32(Vec::new()); // keep explicit check compatibility if needed
    let buf = CanvasPaintBuffer::new(tex_w, tex_h, canvas_center, canvas_width, canvas_height, 42);
    for (pos, uv) in positions.iter().zip(uvs.iter()) {
        let world_xy = Vec2::new(canvas_center.x + pos[0], canvas_center.y + pos[1]);
        let Some((px, py)) = buf.world_to_pixel(world_xy) else {
            return false;
        };
        let expected_px = ((uv[0] * tex_w as f32).floor() as i32).clamp(0, tex_w as i32 - 1) as u32;
        let expected_py = ((uv[1] * tex_h as f32).floor() as i32).clamp(0, tex_h as i32 - 1) as u32;
        if px != expected_px || py != expected_py {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::StudioConfig;

    #[test]
    fn world_to_canvas_pixel_uv_mapping() {
        let studio = StudioConfig::default();
        let w = 2048_u32;
        let h = 1228_u32;
        let buf = CanvasPaintBuffer::new(
            w,
            h,
            studio.canvas_center,
            studio.canvas_width,
            studio.canvas_height,
            42,
        );

        // Top-left world corner (-1.25, 2.30) -> pixel (0, 0)
        assert_eq!(buf.world_to_pixel(Vec2::new(-1.25, 2.30)), Some((0, 0)));
        // Bottom-right world corner (+1.25, 0.80) -> pixel (W-1, H-1)
        assert_eq!(
            buf.world_to_pixel(Vec2::new(1.25, 0.80)),
            Some((w - 1, h - 1))
        );
        // Top-right (+1.25, 2.30) -> (W-1, 0)
        assert_eq!(buf.world_to_pixel(Vec2::new(1.25, 2.30)), Some((w - 1, 0)));
        // Bottom-left (-1.25, 0.80) -> (0, H-1)
        assert_eq!(buf.world_to_pixel(Vec2::new(-1.25, 0.80)), Some((0, h - 1)));
        // Center (0.0, 1.55) -> center pixel
        assert_eq!(
            buf.world_to_pixel(Vec2::new(0.0, 1.55)),
            Some((w / 2, h / 2))
        );
        // Outside bounds returns None
        assert_eq!(buf.world_to_pixel(Vec2::new(-1.26, 1.55)), None);
        assert_eq!(buf.world_to_pixel(Vec2::new(0.0, 2.31)), None);

        // Verify directly against Bevy's Rectangle mesh vertex positions & UV_0 attributes
        let mesh = Mesh::from(Rectangle::new(studio.canvas_width, studio.canvas_height))
            .with_generated_tangents()
            .expect("Rectangle tangents");
        assert!(rectangle_uv_matches_world_mapping(
            &mesh,
            studio.canvas_center,
            studio.canvas_width,
            studio.canvas_height,
            w,
            h,
        ));
    }

    #[test]
    fn linear_rgb_blending_and_srgb_roundtrip() {
        // Round-trip exactness across primaries, grays, and gesso
        for c in [
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            GESSO_SRGB,
            [128, 64, 192],
        ] {
            let lin = srgb_u8_to_linear_rgb(c);
            let rt = linear_rgb_to_srgb_u8(lin);
            assert_eq!(rt, c);
        }

        // Verify blending happens in linear space (50% linear blend of black + white is ~0.5 linear -> ~188 sRGB, NOT 128!)
        let studio = StudioConfig::default();
        let mut buf = CanvasPaintBuffer::new(
            200,
            120,
            studio.canvas_center,
            studio.canvas_width,
            studio.canvas_height,
            42,
        );
        // Force black background in linear buffer
        buf.linear_rgb.fill([0.0, 0.0, 0.0]);
        let white_lin = srgb_u8_to_linear_rgb([255, 255, 255]);
        let _ = buf.splat_cone(
            Vec3::new(0.0, 1.55, 0.12),
            Vec3::NEG_Z,
            white_lin,
            1.0,
            0.05,
        );
        let center_px = buf.world_to_pixel(Vec2::new(0.0, 1.55)).unwrap();
        let idx = (center_px.1 as usize) * 200 + (center_px.0 as usize);
        let lin_val = buf.linear_rgb[idx][0];
        assert!(lin_val > 0.1 && lin_val < 0.9, "unexpected linear val {lin_val}");

        let mut albedo = vec![0_u8; 200 * 120 * 4];
        let mut mr = vec![0_u8; 200 * 120 * 4];
        buf.sync_dirty_rect(
            center_px.0,
            center_px.1,
            center_px.0,
            center_px.1,
            &mut albedo,
            &mut mr,
        );
        let encoded_srgb = albedo[idx * 4];
        // Gamma 2.2 encoding of linear value must match linear_f32_to_srgb_u8(lin_val)
        assert_eq!(encoded_srgb, linear_f32_to_srgb_u8(lin_val));
        assert!(encoded_srgb > (lin_val * 255.0) as u8 + 10);
    }

    #[test]
    fn roughness_transitions_from_matte_to_satin_on_deposition() {
        let studio = StudioConfig::default();
        let mut buf = CanvasPaintBuffer::new(
            512,
            307,
            studio.canvas_center,
            studio.canvas_width,
            studio.canvas_height,
            42,
        );
        let center_px = buf.world_to_pixel(Vec2::new(0.0, 1.55)).unwrap();
        assert!((buf.roughness_at(center_px.0, center_px.1) - 0.88).abs() < 1e-6);

        // Simulate a steady straight spray sweep at d = 0.12m, v = 0.333 m/s (1 pass)
        let red_lin = srgb_u8_to_linear_rgb([230, 40, 30]);
        let speed = 0.333_f32;
        let dt_sub = 0.005_f32;
        let mut x = -0.15_f32;
        while x <= 0.15 {
            buf.splat_cone(
                Vec3::new(x, 1.55, 0.12),
                Vec3::NEG_Z,
                red_lin,
                1.0,
                dt_sub,
            );
            x += speed * dt_sub;
        }

        let center_rough = buf.roughness_at(center_px.0, center_px.1);
        assert!(
            (center_rough - 0.32).abs() < 0.02,
            "1 full pass should saturate near satin 0.32, got {center_rough}"
        );

        let far_px = buf.world_to_pixel(Vec2::new(1.0, 1.55)).unwrap();
        assert!((buf.roughness_at(far_px.0, far_px.1) - 0.88).abs() < 1e-6);

        // Reset restores clean gesso and 0.88 matte roughness
        buf.reset();
        assert!((buf.roughness_at(center_px.0, center_px.1) - 0.88).abs() < 1e-6);
    }

    #[test]
    fn skinny_cap_paints_crisp_thin_line_with_high_centerline_opacity() {
        let studio = StudioConfig::default();
        let w = 1024_u32;
        let h = 614_u32;
        let mut fat_buf = CanvasPaintBuffer::new(
            w,
            h,
            studio.canvas_center,
            studio.canvas_width,
            studio.canvas_height,
            42,
        );
        let mut skinny_buf = fat_buf.clone();

        let center_px = skinny_buf.world_to_pixel(Vec2::new(0.0, 1.55)).unwrap();
        let sweep_y = skinny_buf.pixel_to_world(center_px.0, center_px.1).y;
        let paint_black = [0.0_f32, 0.0, 0.0];
        let speed = 0.14_f32;
        let dt_sub = 0.002_f32;
        let standoff = 0.08_f32;
        let mut x = -0.15_f32;
        while x <= 0.15 {
            let pos = Vec3::new(x, sweep_y, standoff);
            fat_buf.splat_cone_with_cap(pos, Vec3::NEG_Z, paint_black, 1.0, dt_sub, NozzleCap::Fat);
            skinny_buf.splat_cone_with_cap(
                pos,
                Vec3::NEG_Z,
                paint_black,
                1.0,
                dt_sub,
                NozzleCap::Skinny,
            );
            x += speed * dt_sub;
        }

        let mut fat_width = 0_u32;
        let mut skinny_width = 0_u32;
        for py in 0..h {
            let idx = (py as usize) * (w as usize) + (center_px.0 as usize);
            if fat_buf.thickness[idx] > 0.02 {
                fat_width += 1;
            }
            if skinny_buf.thickness[idx] > 0.02 {
                skinny_width += 1;
            }
        }

        assert!(skinny_width > 0, "Skinny cap should paint a visible stroke");
        assert!(
            (skinny_width as f32) < 0.35 * (fat_width as f32),
            "Skinny width ({skinny_width}px) must be < 35% of Fat width ({fat_width}px)"
        );

        let center_idx = (center_px.1 as usize) * (w as usize) + (center_px.0 as usize);
        let gesso_luma = skinny_buf.gesso_linear[0];
        let center_luma = skinny_buf.linear_rgb[center_idx][0];
        let coverage = 1.0 - (center_luma / gesso_luma);
        assert!(
            coverage > 0.90,
            "Skinny centerline opacity/coverage should be high (> 0.90), got {coverage:.3}"
        );
    }

    #[test]
    fn swept_segment_matches_fine_discrete_splats() {
        use std::time::Instant;

        let studio = StudioConfig::default();
        let w = 1024_u32;
        let h = 614_u32;
        let paint_red = srgb_u8_to_linear_rgb([220, 45, 35]);
        let paint_blue = srgb_u8_to_linear_rgb([35, 85, 225]);
        let n_discrete = 100_usize;

        for (cap, pos0, pos1, dir, dt, paint, label) in [
            (
                NozzleCap::Fat,
                Vec3::new(-0.08, 1.55, 0.12),
                Vec3::new(0.08, 1.55, 0.12),
                Vec3::new(0.12, 0.0, -1.0).normalize(),
                0.24_f32,
                paint_red,
                "Fat",
            ),
            (
                NozzleCap::Skinny,
                Vec3::new(-0.010, 1.55, 0.08),
                Vec3::new(0.010, 1.55, 0.08),
                Vec3::NEG_Z,
                0.08_f32,
                paint_blue,
                "Skinny",
            ),
        ] {
            let mut swept_buf = CanvasPaintBuffer::new(
                w,
                h,
                studio.canvas_center,
                studio.canvas_width,
                studio.canvas_height,
                42,
            );
            let mut discrete_buf = swept_buf.clone();

            let swept_rect = swept_buf
                .splat_swept_segment_with_cap(pos0, dir, pos1, dir, paint, 1.0, dt, cap)
                .expect("swept segment should paint pixels");

            let dt_sub = dt / (n_discrete as f32);
            for k in 0..n_discrete {
                let frac = (k as f32 + 0.5) / (n_discrete as f32);
                let pos = pos0.lerp(pos1, frac);
                discrete_buf.splat_cone_with_cap(pos, dir, paint, 1.0, dt_sub, cap);
            }

            let mut active_count = 0_usize;
            let mut rgb_abs_sum = 0.0_f64;
            let mut thick_abs_sum = 0.0_f64;
            let mut core_count = 0_usize;
            let mut core_rgb_abs_sum = 0.0_f64;
            for py in swept_rect.1..=swept_rect.3 {
                for px in swept_rect.0..=swept_rect.2 {
                    let idx = (py as usize) * (w as usize) + (px as usize);
                    if discrete_buf.thickness[idx] > 0.01 || swept_buf.thickness[idx] > 0.01 {
                        active_count += 1;
                        let dr = (swept_buf.linear_rgb[idx][0] - discrete_buf.linear_rgb[idx][0]).abs();
                        let dg = (swept_buf.linear_rgb[idx][1] - discrete_buf.linear_rgb[idx][1]).abs();
                        let db = (swept_buf.linear_rgb[idx][2] - discrete_buf.linear_rgb[idx][2]).abs();
                        let mean_ch = ((dr + dg + db) / 3.0) as f64;
                        rgb_abs_sum += mean_ch;
                        thick_abs_sum +=
                            (swept_buf.thickness[idx] - discrete_buf.thickness[idx]).abs() as f64;
                        if discrete_buf.thickness[idx] > 0.15 {
                            core_count += 1;
                            core_rgb_abs_sum += mean_ch;
                        }
                    }
                }
            }

            assert!(active_count > 10, "{label}: expected painted pixels");
            assert!(core_count > 5, "{label}: expected core painted pixels");
            let mae_rgb = rgb_abs_sum / (active_count as f64);
            let core_mae_rgb = core_rgb_abs_sum / (core_count as f64);
            let mae_thick = thick_abs_sum / (active_count as f64);
            assert!(
                core_mae_rgb < 0.015,
                "{label}: swept vs 100-substep core RGB MAE too high: {core_mae_rgb:.5}"
            );
            assert!(
                mae_rgb < 0.03,
                "{label}: swept vs 100-substep RGB MAE too high: {mae_rgb:.5}"
            );
            assert!(
                mae_thick < 0.05,
                "{label}: swept vs 100-substep thickness MAE too high: {mae_thick:.5}"
            );

            // Benchmark swept vs 100-substep discrete deposition over 50 passes
            let bench_iters = 50_usize;
            let mut bench_swept = CanvasPaintBuffer::new(
                w,
                h,
                studio.canvas_center,
                studio.canvas_width,
                studio.canvas_height,
                42,
            );
            let t0 = Instant::now();
            for _ in 0..bench_iters {
                let _ = bench_swept.splat_swept_segment_with_cap(
                    pos0, dir, pos1, dir, paint, 1.0, dt, cap,
                );
            }
            let swept_us = t0.elapsed().as_micros() as f64 / bench_iters as f64;

            let mut bench_disc = CanvasPaintBuffer::new(
                w,
                h,
                studio.canvas_center,
                studio.canvas_width,
                studio.canvas_height,
                42,
            );
            let t1 = Instant::now();
            for _ in 0..bench_iters {
                for k in 0..n_discrete {
                    let frac = (k as f32 + 0.5) / (n_discrete as f32);
                    let pos = pos0.lerp(pos1, frac);
                    let _ = bench_disc.splat_cone_with_cap(pos, dir, paint, 1.0, dt_sub, cap);
                }
            }
            let disc_us = t1.elapsed().as_micros() as f64 / bench_iters as f64;
            let speedup = disc_us / swept_us.max(1e-3);
            println!(
                "[bench {label}] swept={swept_us:.1} us/seg vs discrete({n_discrete}x)={disc_us:.1} us/seg ({speedup:.1}x speedup), core_mae_rgb={core_mae_rgb:.5}, mae_rgb={mae_rgb:.5}, mae_thick={mae_thick:.5}"
            );
            assert!(
                swept_us < disc_us,
                "{label}: swept ({swept_us:.1} us) should be faster than 100x discrete ({disc_us:.1} us)"
            );
        }
    }

    #[test]
    fn canvas_worker_decoupled_advance_coalescing_and_reset() {
        use crate::core::kinematics::KinematicChain;
        use crate::core::types::{JobOp, JobPlan, SprayCanSpec, StrokeWaypoint};

        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);
        let palette = PalettePlan {
            cans: vec![SprayCanSpec {
                slot_index: 0,
                color_srgb: [220, 45, 35],
                label: "Red".to_string(),
            }],
            primitive_color_map: Vec::new(),
        };
        let job = JobPlan {
            ops: vec![
                JobOp::PickCan { slot: 0 },
                JobOp::PaintStroke {
                    points_canvas_3d: vec![
                        StrokeWaypoint {
                            pos: Vec3::new(-0.20, 1.55, 0.0),
                            normal: Vec3::Z,
                            aim_dir: Vec3::NEG_Z,
                            standoff: 0.12,
                            feed_speed: 0.35,
                            valve_on: true,
                            opacity_scale: 1.0,
                            cap: NozzleCap::Fat,
                        },
                        StrokeWaypoint {
                            pos: Vec3::new(0.20, 1.55, 0.0),
                            normal: Vec3::Z,
                            aim_dir: Vec3::NEG_Z,
                            standoff: 0.12,
                            feed_speed: 0.35,
                            valve_on: true,
                            opacity_scale: 1.0,
                            cap: NozzleCap::Fat,
                        },
                    ],
                },
                JobOp::ReturnCan { slot: 0 },
                JobOp::ParkHome,
            ],
        };
        let timeline = CompiledTimeline::compile(&job, &chain, &studio).expect("compile job");
        let total_t = timeline.duration();
        assert!(total_t > 0.5);

        let w = 512_u32;
        let h = 307_u32;
        let mut ref_buf = CanvasPaintBuffer::new(
            w,
            h,
            studio.canvas_center,
            studio.canvas_width,
            studio.canvas_height,
            42,
        );
        let ref_rect = ref_buf
            .deposit_timeline_interval(&timeline, &palette, 0.0, total_t)
            .expect("reference timeline should paint canvas");
        let mut ref_albedo = vec![0_u8; (w as usize) * (h as usize) * 4];
        let mut ref_mr = vec![0_u8; (w as usize) * (h as usize) * 4];
        for px in ref_albedo.chunks_exact_mut(4) {
            px[0] = GESSO_SRGB[0];
            px[1] = GESSO_SRGB[1];
            px[2] = GESSO_SRGB[2];
            px[3] = 255;
        }
        let init_rough = (BARE_CANVAS_ROUGHNESS * 255.0).round() as u8;
        for px in ref_mr.chunks_exact_mut(4) {
            px[0] = 0;
            px[1] = init_rough;
            px[2] = 0;
            px[3] = 255;
        }
        ref_buf.sync_dirty_rect(0, 0, w - 1, h - 1, &mut ref_albedo, &mut ref_mr);

        let worker_buf = CanvasPaintBuffer::new(
            w,
            h,
            studio.canvas_center,
            studio.canvas_width,
            studio.canvas_height,
            42,
        );
        let worker = CanvasPaintWorker::new(worker_buf, timeline, palette, 0);

        // Queue many rapid async advances to exercise worker-side coalescing, then sync at end
        for i in 1..=25 {
            let t = total_t * (i as f64) / 25.0;
            worker.advance_async(t, 0);
        }
        worker.advance_sync(total_t, 0);

        let mut albedo_tex = vec![0_u8; (w as usize) * (h as usize) * 4];
        let mut mr_tex = vec![0_u8; (w as usize) * (h as usize) * 4];
        let reset_tex = |albedo: &mut [u8], mr: &mut [u8]| {
            for px in albedo.chunks_exact_mut(4) {
                px[0] = GESSO_SRGB[0];
                px[1] = GESSO_SRGB[1];
                px[2] = GESSO_SRGB[2];
                px[3] = 255;
            }
            for px in mr.chunks_exact_mut(4) {
                px[0] = 0;
                px[1] = init_rough;
                px[2] = 0;
                px[3] = 255;
            }
        };
        reset_tex(&mut albedo_tex, &mut mr_tex);

        let patches_ep0: Vec<PaintPatch> = worker.drain_patches().collect();
        assert!(!patches_ep0.is_empty(), "expected dirty patches for epoch 0");
        for p in &patches_ep0 {
            assert_eq!(p.reset_epoch, 0);
            let rw = (p.max_x - p.min_x + 1) as usize;
            let rh = (p.max_y - p.min_y + 1) as usize;
            assert_eq!(p.albedo_patch.len(), rw * rh * 4);
            assert_eq!(p.mr_patch.len(), rw * rh * 4);
            blit_patch_to_rgba(
                &mut albedo_tex,
                w,
                h,
                p.min_x,
                p.min_y,
                p.max_x,
                p.max_y,
                &p.albedo_patch,
            );
            blit_patch_to_rgba(
                &mut mr_tex,
                w,
                h,
                p.min_x,
                p.min_y,
                p.max_x,
                p.max_y,
                &p.mr_patch,
            );
        }

        // Verify center stroke pixel painted and blitted textures match reference closely
        let center_px = ref_buf.world_to_pixel(Vec2::new(0.0, 1.55)).unwrap();
        let center_base = ((center_px.1 as usize) * (w as usize) + (center_px.0 as usize)) * 4;
        assert!(
            albedo_tex[center_base] > 180 && albedo_tex[center_base + 1] < 100,
            "center pixel should be painted red, got {:?}",
            &albedo_tex[center_base..center_base + 3]
        );
        let mut max_byte_diff = 0_u8;
        for py in ref_rect.1..=ref_rect.3 {
            for px in ref_rect.0..=ref_rect.2 {
                let base = ((py as usize) * (w as usize) + (px as usize)) * 4;
                for c in 0..3 {
                    max_byte_diff = max_byte_diff.max(albedo_tex[base + c].abs_diff(ref_albedo[base + c]));
                }
                max_byte_diff = max_byte_diff.max(mr_tex[base + 1].abs_diff(ref_mr[base + 1]));
            }
        }
        assert!(
            max_byte_diff <= 2,
            "coalesced worker blit should match reference within 2 sRGB/MR bytes, got {max_byte_diff}"
        );

        // Exercise reset_epoch transition (R key reset): reset to 0.0, advance, and verify epoch filtering
        reset_tex(&mut albedo_tex, &mut mr_tex);
        worker.advance_async(0.0, 1);
        worker.advance_sync(total_t, 1);
        let patches_ep1: Vec<PaintPatch> = worker
            .drain_patches()
            .filter(|p| p.reset_epoch == 1)
            .collect();
        assert!(!patches_ep1.is_empty(), "expected dirty patches after reset_epoch=1");
        for p in &patches_ep1 {
            blit_patch_to_rgba(
                &mut albedo_tex,
                w,
                h,
                p.min_x,
                p.min_y,
                p.max_x,
                p.max_y,
                &p.albedo_patch,
            );
        }
        assert_eq!(albedo_tex[center_base], ref_albedo[center_base]);

        // Reset back to t=0.0 at epoch 2 and ensure no paint is emitted
        worker.advance_sync(0.0, 2);
        assert!(
            worker.drain_patches().next().is_none(),
            "reset to t=0.0 should emit no paint patches"
        );

        drop(worker);
    }
}
