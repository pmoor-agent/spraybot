use std::f32::consts::PI;

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::mesh::Indices;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::core::types::NozzleCap;
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

/// Coarse time step (seconds) for skipping non-spraying intervals while catching short strokes.
const ADAPTIVE_COARSE_DT: f64 = 0.003;

/// Maximum sub-steps per coarse interval to bound work on extreme edge cases.
const MAX_SUBSTEPS_PER_COARSE: usize = 160;

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
    buffer: CanvasPaintBuffer,
    last_reset_epoch: u64,
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

        let min_x_world = hit.x - max_extent;
        let max_x_world = hit.x + max_extent;
        let min_y_world = hit.y - max_extent;
        let max_y_world = hit.y + max_extent;

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

        let px_min = ((u_min * self.width as f32).floor() as i32).clamp(0, self.width as i32 - 1) as u32;
        let px_max = ((u_max * self.width as f32).ceil() as i32).clamp(0, self.width as i32 - 1) as u32;
        let py_min = ((v_min * self.height as f32).floor() as i32).clamp(0, self.height as i32 - 1) as u32;
        let py_max = ((v_max * self.height as f32).ceil() as i32).clamp(0, self.height as i32 - 1) as u32;

        if px_min > px_max || py_min > py_max {
            return None;
        }

        // Scale area by 1 / cos(phi) to conserve paint mass across oblique incidence ellipses.
        let plume_area = (PI * spot_radius * spot_radius / cos_phi).max(1e-6);
        let dose_coeff =
            SPRAY_FLOW_RATE * cap.orifice_flow_factor() * flow_scale * dt_sub / plume_area;

        let mut dirty_x0 = u32::MAX;
        let mut dirty_y0 = u32::MAX;
        let mut dirty_x1 = 0_u32;
        let mut dirty_y1 = 0_u32;

        let w = self.width as usize;
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
                let falloff = if u_cone <= 0.75 {
                    gaussian
                } else {
                    let rim_t = ((u_cone - 0.75) / 0.60).clamp(0.0, 1.0);
                    let h = spatial_hash_01(px, py, self.seed);
                    let droplet_prob = (1.0 - rim_t).powf(1.2) * 0.90 + 0.05;
                    if h < droplet_prob {
                        gaussian * (0.70 + 0.60 * (1.0 - h))
                    } else {
                        gaussian * 0.08 * (1.0 - rim_t)
                    }
                };

                let alpha_step = 1.0 - (-dose_coeff * falloff).exp();
                if alpha_step <= 1e-5 {
                    continue;
                }

                let idx = row_offset + (px as usize);
                let dst = &mut self.linear_rgb[idx];
                let inv = 1.0 - alpha_step;
                dst[0] = dst[0] * inv + paint_linear[0] * alpha_step;
                dst[1] = dst[1] * inv + paint_linear[1] * alpha_step;
                dst[2] = dst[2] * inv + paint_linear[2] * alpha_step;
                self.thickness[idx] =
                    (self.thickness[idx] + alpha_step * THICKNESS_ACCUM_SCALE).min(1.5);

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

fn setup_canvas(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    app_config: Res<AppConfig>,
    clock: Res<SimClock>,
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

    commands.insert_resource(CanvasResources {
        albedo_tex,
        metallic_roughness_tex,
        buffer,
        last_reset_epoch: clock.reset_epoch,
    });
}

fn update_canvas_deposition(
    clock: Res<SimClock>,
    spray_job: Res<SprayJobRes>,
    mut canvas_res: ResMut<CanvasResources>,
    mut images: ResMut<Assets<Image>>,
) {
    if clock.reset_epoch != canvas_res.last_reset_epoch {
        canvas_res.buffer.reset();
        canvas_res.last_reset_epoch = clock.reset_epoch;

        let initial_roughness_byte = (BARE_CANVAS_ROUGHNESS * 255.0).round() as u8;
        if let Some(mut albedo) = images.get_mut(&canvas_res.albedo_tex)
            && let Some(data) = albedo.data.as_mut()
        {
            for px in data.chunks_exact_mut(4) {
                px[0] = GESSO_SRGB[0];
                px[1] = GESSO_SRGB[1];
                px[2] = GESSO_SRGB[2];
                px[3] = 255;
            }
        }
        if let Some(mut mr) = images.get_mut(&canvas_res.metallic_roughness_tex)
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

    if clock.t_sim <= clock.prev_t_sim {
        return;
    }

    let timeline = &spray_job.timeline;
    let palette = &spray_job.palette;
    let z_plane = canvas_res.buffer.canvas_center.z;

    let mut dirty: Option<(u32, u32, u32, u32)> = None;
    let mut t = clock.prev_t_sim;
    let t_end = clock.t_sim;

    while t < t_end - 1e-9 {
        let dt_step = (t_end - t).min(ADAPTIVE_COARSE_DT);
        let s0 = timeline.sample(t);
        let s1 = timeline.sample(t + dt_step);

        if !s0.spray_active && !s1.spray_active {
            t += dt_step;
            continue;
        }

        let active_ref = if s0.spray_active { &s0 } else { &s1 };
        let standoff = (active_ref.nozzle_world_pos.z - z_plane).abs().max(0.025);
        let spot_radius = standoff * active_ref.active_cap.cone_half_angle_rad().tan();
        let max_disp = (0.25 * spot_radius).max(1e-4);

        let nozzle_disp = (s1.nozzle_world_pos - s0.nozzle_world_pos).length();
        let n_sub = (nozzle_disp / max_disp)
            .ceil()
            .clamp(1.0, MAX_SUBSTEPS_PER_COARSE as f32) as usize;
        let dt_sub = dt_step / (n_sub as f64);

        for k in 0..n_sub {
            let t_mid = t + (k as f64 + 0.5) * dt_sub;
            let frame = timeline.sample(t_mid);
            if !frame.spray_active {
                continue;
            }
            let Some(slot) = frame.held_can_slot else {
                continue;
            };
            let Some(can) = palette.cans.get(slot) else {
                continue;
            };
            let paint_linear = srgb_u8_to_linear_rgb(can.color_srgb);

            if let Some(rect) = canvas_res.buffer.splat_cone_with_cap(
                frame.nozzle_world_pos,
                frame.nozzle_world_dir,
                paint_linear,
                frame.flow_scale,
                dt_sub as f32,
                frame.active_cap,
            ) {
                dirty = Some(match dirty {
                    Some((x0, y0, x1, y1)) => {
                        (x0.min(rect.0), y0.min(rect.1), x1.max(rect.2), y1.max(rect.3))
                    }
                    None => rect,
                });
            }
        }

        t += dt_step;
    }

    if let Some((x0, y0, x1, y1)) = dirty {
        if let Some(mut albedo) = images.get_mut(&canvas_res.albedo_tex)
            && let Some(albedo_bytes) = albedo.data.as_mut()
        {
            canvas_res
                .buffer
                .sync_dirty_albedo(x0, y0, x1, y1, albedo_bytes);
        }
        if let Some(mut mr) = images.get_mut(&canvas_res.metallic_roughness_tex)
            && let Some(mr_bytes) = mr.data.as_mut()
        {
            canvas_res
                .buffer
                .sync_dirty_roughness(x0, y0, x1, y1, mr_bytes);
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
}
