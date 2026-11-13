use std::f32::consts::{FRAC_PI_2, PI};

use bevy::{
    app::AppExit,
    camera::{Hdr, RenderTarget},
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    post_process::bloom::Bloom,
    prelude::*,
};

use crate::cli::CameraPreset;
use crate::core::types::JobOp;
use crate::sim::recorder::GracefulShutdownRequested;
use crate::sim::state::{AppConfig, OffscreenRenderTarget, SimClock, SimStepSet, SprayJobRes};

/// Studio interior camera clamp bounds (meters, 1 unit = 1m).
const STUDIO_MIN_X: f32 = -3.85;
const STUDIO_MAX_X: f32 = 3.85;
const STUDIO_MIN_Y: f32 = 0.08;
const STUDIO_MAX_Y: f32 = 4.05;
const STUDIO_MIN_Z: f32 = 0.12;
const STUDIO_MAX_Z: f32 = 6.30;

const MIN_PITCH: f32 = -FRAC_PI_2 + 0.05;
const MAX_PITCH: f32 = FRAC_PI_2 - 0.05;
const MIN_DISTANCE: f32 = 0.45;
const MAX_DISTANCE: f32 = 7.50;
const SPRAY_CONE_HALF_ANGLE_RAD: f32 = 12.0 * PI / 180.0;

/// Marker component for the main 3D studio camera.
#[derive(Component)]
pub struct MainStudioCamera;

/// Orbit/pan/zoom/follow state for the studio camera.
#[derive(Resource, Debug, Clone)]
pub struct CameraRig {
    pub active_preset: Option<CameraPreset>,
    pub desired_target: Vec3,
    pub current_target: Vec3,
    pub desired_yaw: f32,
    pub current_yaw: f32,
    pub desired_pitch: f32,
    pub current_pitch: f32,
    pub desired_distance: f32,
    pub current_distance: f32,
}

impl CameraRig {
    /// Builds an initial rig matching a CLI [`CameraPreset`] and initial TCP position.
    pub fn from_preset(preset: CameraPreset, initial_nozzle: Vec3) -> Self {
        let (eye, target) = preset_eye_target(preset, initial_nozzle);
        let (yaw, pitch, dist) = cartesian_to_spherical(eye, target);
        Self {
            active_preset: Some(preset),
            desired_target: target,
            current_target: target,
            desired_yaw: yaw,
            current_yaw: yaw,
            desired_pitch: pitch,
            current_pitch: pitch,
            desired_distance: dist,
            current_distance: dist,
        }
    }

    /// Transitions smoothly to one of the 4 named camera presets.
    pub fn apply_preset(&mut self, preset: CameraPreset, current_nozzle: Vec3) {
        let (eye, target) = preset_eye_target(preset, current_nozzle);
        let (yaw, pitch, dist) = cartesian_to_spherical(eye, target);
        self.active_preset = Some(preset);
        self.desired_target = target;
        self.desired_yaw = unwrap_angle_near(yaw, self.current_yaw);
        self.desired_pitch = pitch.clamp(MIN_PITCH, MAX_PITCH);
        self.desired_distance = dist.clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    /// Computes the camera world eye position and clamps it inside the studio walls (`Y >= 0.08m`).
    pub fn clamped_eye(&self) -> Vec3 {
        let raw_eye = spherical_to_cartesian(
            self.current_target,
            self.current_yaw,
            self.current_pitch,
            self.current_distance,
        );
        clamp_inside_studio(raw_eye)
    }
}

/// Registers camera spawning, hotkey input handling, mouse orbit/pan/zoom, and 3D gizmos.
pub fn plugin(app: &mut App) {
    app.add_systems(Startup, spawn_camera)
        .add_systems(Update, handle_hotkeys.in_set(SimStepSet::InputAndClock))
        .add_systems(
            Update,
            (update_camera_rig, draw_debug_gizmos)
                .chain()
                .in_set(SimStepSet::CameraAndGizmos),
        );
}

fn spawn_camera(
    mut commands: Commands,
    app_config: Res<AppConfig>,
    offscreen: Res<OffscreenRenderTarget>,
    job: Res<SprayJobRes>,
) {
    let rig = CameraRig::from_preset(app_config.cli.camera, job.current_frame.nozzle_world_pos);
    let eye = rig.clamped_eye();
    let transform = Transform::from_translation(eye).looking_at(rig.current_target, Vec3::Y);

    let mut bloom = Bloom::NATURAL;
    bloom.intensity = 0.08;

    let mut entity = commands.spawn((
        MainStudioCamera,
        Camera3d::default(),
        Hdr,
        Tonemapping::AcesFitted,
        bloom,
        transform,
    ));

    if let Some(image_handle) = offscreen.image.as_ref() {
        entity.insert(RenderTarget::Image(image_handle.clone().into()));
    }

    commands.insert_resource(rig);
}

fn handle_hotkeys(
    keys: Res<ButtonInput<KeyCode>>,
    app_config: Res<AppConfig>,
    mut clock: ResMut<SimClock>,
    mut job: ResMut<SprayJobRes>,
    mut rig: ResMut<CameraRig>,
    mut shutdown_req: ResMut<GracefulShutdownRequested>,
    mut app_exit: MessageWriter<AppExit>,
) {
    let nozzle = job.current_frame.nozzle_world_pos;

    if keys.just_pressed(KeyCode::Digit1) || keys.just_pressed(KeyCode::Numpad1) {
        rig.apply_preset(CameraPreset::Overview, nozzle);
    } else if keys.just_pressed(KeyCode::Digit2) || keys.just_pressed(KeyCode::Numpad2) {
        rig.apply_preset(CameraPreset::Follow, nozzle);
    } else if keys.just_pressed(KeyCode::Digit3) || keys.just_pressed(KeyCode::Numpad3) {
        rig.apply_preset(CameraPreset::Rack, nozzle);
    } else if keys.just_pressed(KeyCode::Digit4) || keys.just_pressed(KeyCode::Numpad4) {
        rig.apply_preset(CameraPreset::Canvas, nozzle);
    }

    if keys.just_pressed(KeyCode::Space) {
        clock.paused = !clock.paused;
    }

    if keys.just_pressed(KeyCode::BracketLeft) {
        clock.speed = (clock.speed * 0.5).clamp(0.25, 32.0);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        clock.speed = (clock.speed * 2.0).clamp(0.25, 32.0);
    }

    if keys.just_pressed(KeyCode::KeyG) {
        clock.show_gizmos = !clock.show_gizmos;
        job.show_gizmos = clock.show_gizmos;
    }

    if keys.just_pressed(KeyCode::KeyR) {
        clock.reset();
        job.current_frame = job.timeline.sample(0.0);
    }

    if keys.just_pressed(KeyCode::Escape) || keys.just_pressed(KeyCode::KeyQ) {
        if app_config.cli.record.is_some() {
            shutdown_req.0 = true;
        } else {
            app_exit.write(AppExit::Success);
        }
    }
}

fn update_camera_rig(
    time: Res<Time>,
    app_config: Res<AppConfig>,
    clock: Res<SimClock>,
    job: Res<SprayJobRes>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion_reader: MessageReader<MouseMotion>,
    mut wheel_reader: MessageReader<MouseWheel>,
    mut rig: ResMut<CameraRig>,
    mut cam_query: Query<&mut Transform, With<MainStudioCamera>>,
) {
    let mut mouse_delta = Vec2::ZERO;
    for ev in motion_reader.read() {
        mouse_delta += ev.delta;
    }

    let mut scroll_y = 0.0_f32;
    for ev in wheel_reader.read() {
        scroll_y += match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y * 0.02,
        };
    }

    // Left-drag: orbit azimuth and elevation.
    if mouse_buttons.pressed(MouseButton::Left) && mouse_delta != Vec2::ZERO {
        rig.desired_yaw -= mouse_delta.x * 0.0055;
        rig.desired_pitch = (rig.desired_pitch + mouse_delta.y * 0.0055).clamp(MIN_PITCH, MAX_PITCH);
    }

    // Right/Middle-drag: pan focal target along camera plane.
    if (mouse_buttons.pressed(MouseButton::Right) || mouse_buttons.pressed(MouseButton::Middle))
        && mouse_delta != Vec2::ZERO
    {
        if rig.active_preset == Some(CameraPreset::Follow) {
            rig.active_preset = None;
        }
        let eye = rig.clamped_eye();
        let forward = (rig.current_target - eye).normalize_or(Vec3::NEG_Z);
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward).normalize_or(Vec3::Y);
        let pan_scale = rig.current_distance * 0.0014;
        rig.desired_target += (-right * mouse_delta.x + up * mouse_delta.y) * pan_scale;
        rig.desired_target = clamp_target_inside_studio(rig.desired_target);
    }

    // Scroll wheel: smooth exponential dolly zoom.
    if scroll_y.abs() > 1e-5 {
        rig.desired_distance =
            (rig.desired_distance * (-scroll_y * 0.12).exp()).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    // Follow preset smoothly tracks the active TCP / nozzle zone.
    if rig.active_preset == Some(CameraPreset::Follow) {
        let nozzle = job.current_frame.nozzle_world_pos;
        rig.desired_target = clamp_target_inside_studio(nozzle);
    }

    // Use deterministic step dt during recording so follow-cam motion is lockstep-smooth.
    let dt = if app_config.cli.record.is_some() {
        let fps = app_config.cli.fps.max(1) as f32;
        if clock.paused {
            0.0
        } else {
            (1.0 / fps) * clock.speed.clamp(0.25, 4.0)
        }
    } else {
        time.delta_secs().min(0.1)
    };

    let target_alpha = 1.0 - (-8.0 * dt).exp();
    let orbit_alpha = 1.0 - (-16.0 * dt).exp();

    rig.current_target = rig.current_target.lerp(rig.desired_target, target_alpha);
    rig.current_yaw = rig.current_yaw + (rig.desired_yaw - rig.current_yaw) * orbit_alpha;
    rig.current_pitch = rig.current_pitch + (rig.desired_pitch - rig.current_pitch) * orbit_alpha;
    rig.current_distance =
        rig.current_distance + (rig.desired_distance - rig.current_distance) * orbit_alpha;

    let clamped_eye = rig.clamped_eye();
    for mut transform in cam_query.iter_mut() {
        *transform = Transform::from_translation(clamped_eye).looking_at(rig.current_target, Vec3::Y);
    }
}

fn draw_debug_gizmos(mut gizmos: Gizmos, clock: Res<SimClock>, job: Res<SprayJobRes>) {
    if !clock.show_gizmos && !job.show_gizmos {
        return;
    }

    // 1. Planned spray strokes on the wall canvas plane (slightly offset in +Z to prevent z-fighting).
    let active_stroke_color = Color::srgba(0.15, 0.90, 1.0, 0.9);
    let lead_color = Color::srgba(1.0, 0.65, 0.15, 0.45);
    for op in &job.job.ops {
        if let JobOp::PaintStroke { points_canvas_3d } = op {
            for pair in points_canvas_3d.windows(2) {
                let a = pair[0];
                let b = pair[1];
                let pa = a.pos + a.normal * 0.004;
                let pb = b.pos + b.normal * 0.004;
                let color = if a.valve_on && b.valve_on {
                    active_stroke_color
                } else {
                    lead_color
                };
                gizmos.line(pa, pb, color);
            }
        }
    }

    // 2. Active nozzle / TCP coordinate axes.
    let frame = &job.current_frame;
    let apex = frame.nozzle_world_pos;
    let dir = frame.nozzle_world_dir.normalize_or(Vec3::NEG_Z);
    let tcp_rot = Quat::from_rotation_arc(Vec3::NEG_Z, dir);
    gizmos.axes(Transform::from_translation(apex).with_rotation(tcp_rot), 0.10);

    // 3. 12-degree spray cone wireframe from nozzle tip toward canvas (Z = 0.0).
    let cone_len = if dir.z < -1e-3 && apex.z > 0.0 {
        (apex.z / -dir.z).clamp(0.04, 0.35)
    } else {
        0.22
    };
    let cone_radius = cone_len * SPRAY_CONE_HALF_ANGLE_RAD.tan();
    let cone_base_center = apex + dir * cone_len;
    let cone_color = if frame.spray_active {
        Color::srgba(0.25, 1.0, 0.40, 0.95)
    } else {
        Color::srgba(0.55, 0.70, 0.90, 0.55)
    };

    let ring_rot = Quat::from_rotation_arc(Vec3::Z, dir);
    gizmos
        .circle(Isometry3d::new(cone_base_center, ring_rot), cone_radius, cone_color)
        .resolution(24);

    let right = ring_rot * Vec3::X;
    let up = ring_rot * Vec3::Y;
    for i in 0..8 {
        let angle = (i as f32) * (2.0 * PI / 8.0);
        let (sin_a, cos_a) = angle.sin_cos();
        let rim_pt = cone_base_center + (right * cos_a + up * sin_a) * cone_radius;
        gizmos.line(apex, rim_pt, cone_color);
    }
}

/// Returns `(eye, target)` world coordinates for each named [`CameraPreset`].
pub fn preset_eye_target(preset: CameraPreset, nozzle: Vec3) -> (Vec3, Vec3) {
    match preset {
        CameraPreset::Overview => (Vec3::new(1.65, 1.85, 3.35), Vec3::new(-0.35, 1.35, 0.25)),
        CameraPreset::Follow => {
            let target = clamp_target_inside_studio(nozzle);
            let eye = clamp_inside_studio(target + Vec3::new(0.45, 0.32, 1.35));
            (eye, target)
        }
        CameraPreset::Rack => (Vec3::new(-1.05, 1.35, 1.85), Vec3::new(-1.85, 1.08, 0.60)),
        CameraPreset::Canvas => (Vec3::new(0.0, 1.55, 2.35), Vec3::new(0.0, 1.55, 0.0)),
    }
}

/// Converts `(eye, target)` into spherical `(yaw, pitch, distance)` around `target`.
pub fn cartesian_to_spherical(eye: Vec3, target: Vec3) -> (f32, f32, f32) {
    let offset = eye - target;
    let distance = offset.length().clamp(MIN_DISTANCE, MAX_DISTANCE);
    let horiz = (offset.x * offset.x + offset.z * offset.z).sqrt();
    let yaw = offset.x.atan2(offset.z);
    let pitch = offset.y.atan2(horiz).clamp(MIN_PITCH, MAX_PITCH);
    (yaw, pitch, distance)
}

/// Converts spherical `(yaw, pitch, distance)` around `target` back into an eye position.
pub fn spherical_to_cartesian(target: Vec3, yaw: f32, pitch: f32, distance: f32) -> Vec3 {
    let (sin_p, cos_p) = pitch.sin_cos();
    let (sin_y, cos_y) = yaw.sin_cos();
    let dir = Vec3::new(cos_p * sin_y, sin_p, cos_p * cos_y);
    target + dir * distance
}

/// Unwraps `target_angle` by multiples of `2π` to stay closest to `reference` (avoids 360° spins).
fn unwrap_angle_near(target_angle: f32, reference: f32) -> f32 {
    let two_pi = 2.0 * PI;
    let delta = (target_angle - reference).rem_euclid(two_pi);
    let shortest = if delta > PI { delta - two_pi } else { delta };
    reference + shortest
}

/// Clamps camera eye position inside the studio room walls and above the floor (`Y >= 0.08m`).
pub fn clamp_inside_studio(eye: Vec3) -> Vec3 {
    Vec3::new(
        eye.x.clamp(STUDIO_MIN_X, STUDIO_MAX_X),
        eye.y.clamp(STUDIO_MIN_Y, STUDIO_MAX_Y),
        eye.z.clamp(STUDIO_MIN_Z, STUDIO_MAX_Z),
    )
}

fn clamp_target_inside_studio(target: Vec3) -> Vec3 {
    Vec3::new(
        target.x.clamp(-3.50, 3.50),
        target.y.clamp(0.05, 3.80),
        target.z.clamp(-0.02, 5.80),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spherical_cartesian_roundtrip_for_all_presets() {
        let nozzle = Vec3::new(0.25, 1.45, 0.14);
        for preset in [
            CameraPreset::Overview,
            CameraPreset::Follow,
            CameraPreset::Rack,
            CameraPreset::Canvas,
        ] {
            let (eye, target) = preset_eye_target(preset, nozzle);
            let (yaw, pitch, dist) = cartesian_to_spherical(eye, target);
            let reconstructed = spherical_to_cartesian(target, yaw, pitch, dist);
            assert_relative_eq!(reconstructed.x, eye.x, epsilon = 1e-4);
            assert_relative_eq!(reconstructed.y, eye.y, epsilon = 1e-4);
            assert_relative_eq!(reconstructed.z, eye.z, epsilon = 1e-4);

            let rig = CameraRig::from_preset(preset, nozzle);
            let clamped = rig.clamped_eye();
            assert!(clamped.y >= STUDIO_MIN_Y);
            assert!(clamped.x >= STUDIO_MIN_X && clamped.x <= STUDIO_MAX_X);
            assert!(clamped.z >= STUDIO_MIN_Z && clamped.z <= STUDIO_MAX_Z);
        }
    }

    #[test]
    fn floor_and_wall_clamping_enforces_studio_bounds() {
        let below_floor = clamp_inside_studio(Vec3::new(0.0, -1.5, 2.0));
        assert_relative_eq!(below_floor.y, 0.08, epsilon = 1e-6);

        let outside_walls = clamp_inside_studio(Vec3::new(10.0, 9.0, -2.0));
        assert_relative_eq!(outside_walls.x, STUDIO_MAX_X, epsilon = 1e-6);
        assert_relative_eq!(outside_walls.y, STUDIO_MAX_Y, epsilon = 1e-6);
        assert_relative_eq!(outside_walls.z, STUDIO_MIN_Z, epsilon = 1e-6);
    }
}
