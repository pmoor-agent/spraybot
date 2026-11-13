use std::f32::consts::FRAC_PI_2;

use bevy::prelude::*;

use crate::core::types::{NozzleCap, RackSlotPose};
use crate::sim::state::{AppConfig, SprayJobRes};

pub fn plugin(app: &mut App) {
    app.init_resource::<HeldCanSlot>()
        .init_resource::<SlotCapVisualState>()
        .init_resource::<StudioRackCache>()
        .add_systems(Startup, spawn_robot_and_cans);
}

/// Identifies a moving link entity driven by [`KinematicChain::fk_links`].
#[derive(Component, Debug, Clone, Copy)]
pub struct RobotLink {
    pub index: usize,
}

/// Identifies a parallel gripper jaw child of Link 8 (`flange_roll`; `side`: `-1.0` left, `+1.0` right).
#[derive(Component, Debug, Clone, Copy)]
pub struct GripperJaw {
    pub side: f32,
}

/// Identifies the servo trigger finger child of Link 8 (`flange_roll`) that depresses the aerosol nozzle cap.
#[derive(Component, Debug, Clone, Copy)]
pub struct TriggerFinger;

/// Identifies a spawned aerosol spray can corresponding to `palette.cans[slot]`.
#[derive(Component, Debug, Clone, Copy)]
pub struct CanEntity {
    pub slot: usize,
}

/// Identifies a switchable nozzle cap visual group attached to a [`CanEntity`].
#[derive(Component, Debug, Clone, Copy)]
pub struct CanCapVisual {
    pub slot: usize,
    pub cap: NozzleCap,
}

/// Tracks which rack slot's spray can is currently held by the gripper.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct HeldCanSlot(pub Option<usize>);

/// Tracks the currently mounted nozzle cap visual state for each rack slot.
#[derive(Resource, Debug, Default, Clone)]
pub struct SlotCapVisualState(pub Vec<NozzleCap>);

/// Cached rack slot poses computed from [`StudioConfig`] at startup.
#[derive(Resource, Debug, Default, Clone)]
pub struct StudioRackCache {
    pub slots: Vec<RackSlotPose>,
}

const TRIGGER_PIVOT_LOCAL: Vec3 = Vec3::new(0.0, 0.114, -0.006);

pub fn jaw_local_translation(side: f32, gripper_open: f32) -> Vec3 {
    let g = gripper_open.clamp(0.0, 1.0);
    // Can radius is 0.033m, jaw thickness is 0.012m -> closed center at ±0.039m, open at ±0.057m
    Vec3::new(side * (0.039 + 0.018 * g), -0.008, -0.025)
}

pub fn trigger_local_transform(trigger_depressed: f32) -> Transform {
    let t = trigger_depressed.clamp(0.0, 1.0);
    // Negative rotation around +X swings the -Z finger tip downward (-Y) onto the nozzle cap.
    Transform::from_translation(TRIGGER_PIVOT_LOCAL)
        .with_rotation(Quat::from_rotation_x(-0.24 * t))
}

fn spawn_robot_and_cans(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    app_config: Res<AppConfig>,
    spray_job: Res<SprayJobRes>,
    mut held_can: ResMut<HeldCanSlot>,
    mut slot_caps: ResMut<SlotCapVisualState>,
    mut rack_cache: ResMut<StudioRackCache>,
) {
    let studio = &app_config.studio;
    let max_slots = app_config
        .cli
        .max_cans
        .max(spray_job.palette.cans.iter().map(|c| c.slot_index + 1).max().unwrap_or(0));
    rack_cache.slots = studio.rack_slots(max_slots);

    let frame = &spray_job.current_frame;
    held_can.0 = frame.held_can_slot;
    slot_caps.0 = vec![NozzleCap::Fat; max_slots];
    if let Some(slot) = frame.held_can_slot
        && let Some(entry) = slot_caps.0.get_mut(slot)
    {
        *entry = frame.active_cap;
    }

    let link_xforms = spray_job.chain.fk_links(&frame.joints);
    let grasp_xform = spray_job.chain.fk_grasp(&frame.joints);

    // Industrial PBR materials
    let cast_iron_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.13, 0.14, 0.16),
        metallic: 0.78,
        perceptual_roughness: 0.48,
        ..default()
    });
    let anodized_dark_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.09, 0.10, 0.12),
        metallic: 0.86,
        perceptual_roughness: 0.30,
        ..default()
    });
    let safety_orange_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#E85D1A").unwrap().into(),
        metallic: 0.18,
        perceptual_roughness: 0.30,
        ..default()
    });
    let graphite_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.20, 0.22, 0.26),
        metallic: 0.58,
        perceptual_roughness: 0.36,
        ..default()
    });
    let aluminum_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.79, 0.83),
        metallic: 0.92,
        perceptual_roughness: 0.22,
        ..default()
    });
    let rubber_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.06, 0.06, 0.07),
        metallic: 0.02,
        perceptual_roughness: 0.86,
        ..default()
    });

    let link_tf = |idx: usize| -> Transform {
        let lt = link_xforms[idx];
        Transform {
            translation: lt.pos,
            rotation: lt.rot,
            scale: Vec3::ONE,
        }
    };

    // Link 0: rail_x cast-iron linear carriage + bearing runners + flanged base ring
    commands.spawn((
        RobotLink { index: 0 },
        link_tf(0),
        Visibility::default(),
        children![
            // Main carriage saddle
            (
                Mesh3d(meshes.add(Cuboid::new(0.38, 0.056, 0.30))),
                MeshMaterial3d(cast_iron_mat.clone()),
                Transform::from_xyz(0.0, -0.022, 0.0),
            ),
            // 4 linear bearing runner blocks riding the twin guideways (Z = ±0.095)
            (
                Mesh3d(meshes.add(Cuboid::new(0.095, 0.036, 0.048))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_xyz(-0.13, -0.052, -0.095),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.095, 0.036, 0.048))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_xyz(0.13, -0.052, -0.095),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.095, 0.036, 0.048))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_xyz(-0.13, -0.052, 0.095),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.095, 0.036, 0.048))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_xyz(0.13, -0.052, 0.095),
            ),
            // Top mounting plate & flanged turret ring
            (
                Mesh3d(meshes.add(Cuboid::new(0.28, 0.034, 0.28))),
                MeshMaterial3d(graphite_mat.clone()),
                Transform::from_xyz(0.0, 0.022, 0.0),
            ),
            (
                Mesh3d(meshes.add(Cylinder::new(0.122, 0.052))),
                MeshMaterial3d(cast_iron_mat.clone()),
                Transform::from_xyz(0.0, 0.062, 0.0),
            ),
            // Rear servo drive pack & drag-chain coupling
            (
                Mesh3d(meshes.add(Cuboid::new(0.11, 0.072, 0.09))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_xyz(0.0, -0.012, -0.17),
            ),
        ],
    ));

    // Link 1: base_yaw rotating riser pedestal & shoulder clevis yoke
    // Frame is at shoulder_dy * 0.5 = 0.485m above Link 0 and 0.485m below Link 2.
    commands.spawn((
        RobotLink { index: 1 },
        link_tf(1),
        Visibility::default(),
        children![
            // Lower bolted flange meeting carriage turret ring
            (
                Mesh3d(meshes.add(Cylinder::new(0.126, 0.044))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_xyz(0.0, -0.382, 0.0),
            ),
            // Tapered graphite riser column
            (
                Mesh3d(
                    meshes.add(
                        ConicalFrustum {
                            radius_top: 0.096,
                            radius_bottom: 0.118,
                            height: 0.70,
                        }
                        .mesh()
                        .resolution(32),
                    ),
                ),
                MeshMaterial3d(graphite_mat.clone()),
                Transform::from_xyz(0.0, -0.02, 0.0),
            ),
            // Safety-orange turret shoulder collar
            (
                Mesh3d(meshes.add(Cylinder::new(0.108, 0.11))),
                MeshMaterial3d(safety_orange_mat.clone()),
                Transform::from_xyz(0.0, 0.345, 0.0),
            ),
            // Twin clevis yoke cheeks up to shoulder pivot (y_local = 0.485)
            (
                Mesh3d(meshes.add(Cuboid::new(0.042, 0.22, 0.155))),
                MeshMaterial3d(cast_iron_mat.clone()),
                Transform::from_xyz(-0.092, 0.435, 0.0),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.042, 0.22, 0.155))),
                MeshMaterial3d(cast_iron_mat.clone()),
                Transform::from_xyz(0.092, 0.435, 0.0),
            ),
        ],
    ));

    // Link 2: shoulder_pitch lower half of upper arm (extends +Y by L2 * 0.5 = 0.34m to upper_arm_roll)
    let half_l2 = studio.upper_arm_l2 * 0.5;
    commands.spawn((
        RobotLink { index: 2 },
        link_tf(2),
        Visibility::default(),
        children![
            // Shoulder joint drum & anodized motor caps
            (
                Mesh3d(meshes.add(Cylinder::new(0.082, 0.235))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ),
            (
                Mesh3d(meshes.add(Cylinder::new(0.068, 0.265))),
                MeshMaterial3d(graphite_mat.clone()),
                Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ),
            // Tapered safety-orange lower upper-arm beam (y = 0 .. half_l2)
            (
                Mesh3d(
                    meshes.add(
                        ConicalFrustum {
                            radius_top: 0.062,
                            radius_bottom: 0.072,
                            height: (half_l2 - 0.04).max(0.05),
                        }
                        .mesh()
                        .resolution(32),
                    ),
                ),
                MeshMaterial3d(safety_orange_mat.clone()),
                Transform::from_xyz(0.0, half_l2 * 0.5, 0.0),
            ),
            // Structural graphite spine rib
            (
                Mesh3d(meshes.add(Cuboid::new(0.046, (half_l2 - 0.08).max(0.05), 0.136))),
                MeshMaterial3d(graphite_mat.clone()),
                Transform::from_xyz(0.0, half_l2 * 0.5, 0.012),
            ),
            // Machined dark-anodized & aluminum rotary swivel ring terminating at y = half_l2 (0.34m)
            (
                Mesh3d(meshes.add(Cylinder::new(0.066, 0.022))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_xyz(0.0, half_l2 - 0.017, 0.0),
            ),
            (
                Mesh3d(meshes.add(Cylinder::new(0.069, 0.006))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(0.0, half_l2 - 0.003, 0.0),
            ),
        ],
    ));

    // Link 3: upper_arm_roll rotary swivel joint at mid-upper-arm (+Y axis, extends +Y by L2 * 0.5 = 0.34m to elbow_pitch)
    commands.spawn((
        RobotLink { index: 3 },
        link_tf(3),
        Visibility::default(),
        children![
            // Sleek rotary bearing collar at y = 0
            (
                Mesh3d(meshes.add(Cylinder::new(0.068, 0.008))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(0.0, 0.004, 0.0),
            ),
            (
                Mesh3d(meshes.add(Cylinder::new(0.066, 0.020))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_xyz(0.0, 0.018, 0.0),
            ),
            // Tapered safety-orange upper segment (y = 0 .. half_l2)
            (
                Mesh3d(
                    meshes.add(
                        ConicalFrustum {
                            radius_top: 0.054,
                            radius_bottom: 0.062,
                            height: (half_l2 - 0.045).max(0.05),
                        }
                        .mesh()
                        .resolution(32),
                    ),
                ),
                MeshMaterial3d(safety_orange_mat.clone()),
                Transform::from_xyz(0.0, half_l2 * 0.5 + 0.005, 0.0),
            ),
            // Structural graphite rib up to elbow drum
            (
                Mesh3d(meshes.add(Cuboid::new(0.044, (half_l2 - 0.09).max(0.05), 0.122))),
                MeshMaterial3d(graphite_mat.clone()),
                Transform::from_xyz(0.0, half_l2 * 0.5 + 0.005, 0.012),
            ),
            // Elbow-end anodized collar
            (
                Mesh3d(meshes.add(Cylinder::new(0.060, 0.036))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_xyz(0.0, half_l2 - 0.036, 0.0),
            ),
        ],
    ));

    // Link 4: elbow_pitch forearm (extends -Z by L3 = 0.60m to wrist_1)
    let l3 = studio.forearm_l3;
    commands.spawn((
        RobotLink { index: 4 },
        link_tf(4),
        Visibility::default(),
        children![
            // Elbow joint drum & anodized caps
            (
                Mesh3d(meshes.add(Cylinder::new(0.068, 0.195))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ),
            // Rear servo housing counterweight
            (
                Mesh3d(meshes.add(Cuboid::new(0.108, 0.118, 0.145))),
                MeshMaterial3d(cast_iron_mat.clone()),
                Transform::from_xyz(0.0, 0.012, 0.078),
            ),
            // Safety-orange tapered forearm tube (-Z direction: top points along -Z via -FRAC_PI_2 around X)
            (
                Mesh3d(
                    meshes.add(
                        ConicalFrustum {
                            radius_top: 0.042,
                            radius_bottom: 0.058,
                            height: l3 - 0.04,
                        }
                        .mesh()
                        .resolution(32),
                    ),
                ),
                MeshMaterial3d(safety_orange_mat.clone()),
                Transform::from_xyz(0.0, 0.0, -l3 * 0.5)
                    .with_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
            ),
            // Top graphite cable dress-pack conduit
            (
                Mesh3d(meshes.add(Cuboid::new(0.028, 0.024, l3 - 0.12))),
                MeshMaterial3d(graphite_mat.clone()),
                Transform::from_xyz(0.0, 0.058, -l3 * 0.5),
            ),
            // Wrist-base anodized collar
            (
                Mesh3d(meshes.add(Cylinder::new(0.046, 0.036))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_xyz(0.0, 0.0, -l3 + 0.02)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ),
        ],
    ));

    // Link 5: wrist_1 machined aluminum roll housing (extends -Z by 0.065m to wrist_2)
    commands.spawn((
        RobotLink { index: 5 },
        link_tf(5),
        Visibility::default(),
        children![
            (
                Mesh3d(meshes.add(Cylinder::new(0.042, 0.046))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(0.0, 0.0, -0.023)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.016, 0.058, 0.044))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(-0.040, 0.0, -0.054),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.016, 0.058, 0.044))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(0.040, 0.0, -0.054),
            ),
        ],
    ));

    // Link 6: wrist_2 pitch clevis (extends -Z by 0.030m to wrist_yaw)
    commands.spawn((
        RobotLink { index: 6 },
        link_tf(6),
        Visibility::default(),
        children![
            (
                Mesh3d(meshes.add(Cylinder::new(0.028, 0.098))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.048, 0.046, 0.024))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(0.0, 0.0, -0.012),
            ),
            (
                Mesh3d(meshes.add(Cylinder::new(0.023, 0.008))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(0.0, 0.0, -0.026)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ),
        ],
    ));

    // Link 7: wrist_yaw vertical rotary drum + machined aluminum U-yoke cheeks (+Y axis, extends -Z by 0.020m to flange_roll)
    commands.spawn((
        RobotLink { index: 7 },
        link_tf(7),
        Visibility::default(),
        children![
            // Vertical dark-anodized rotary drum (+Y axis)
            (
                Mesh3d(meshes.add(Cylinder::new(0.024, 0.064))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::IDENTITY,
            ),
            // Machined aluminum U-yoke cheeks aligned along +Y / -Z
            (
                Mesh3d(meshes.add(Cuboid::new(0.009, 0.056, 0.024))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(-0.026, 0.0, -0.008),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.009, 0.056, 0.024))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(0.026, 0.0, -0.008),
            ),
            // Flange-mount boss bridging yoke cheeks to flange_roll at z = -0.020
            (
                Mesh3d(meshes.add(Cylinder::new(0.028, 0.012))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(0.0, 0.0, -0.014)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ),
        ],
    ));

    // Link 8: flange_roll tool flange + parallel pneumatic gripper + servo trigger finger
    // Grasp center (tcp_offset) is at local (0, 0, -0.025).
    let jaw_mesh = meshes.add(Cuboid::new(0.012, 0.044, 0.040));
    let jaw_pad_mesh = meshes.add(Cuboid::new(0.004, 0.038, 0.034));

    commands.spawn((
        RobotLink { index: 8 },
        link_tf(8),
        Visibility::default(),
        children![
            // Tool flange disc
            (
                Mesh3d(meshes.add(Cylinder::new(0.039, 0.012))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(0.0, 0.0, -0.006)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ),
            // Pneumatic parallel gripper body & linear rail
            (
                Mesh3d(meshes.add(Cuboid::new(0.118, 0.036, 0.018))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_xyz(0.0, 0.0, -0.012),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.112, 0.014, 0.010))),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_xyz(0.0, 0.0, -0.022),
            ),
            // Left parallel gripper jaw (-X)
            (
                GripperJaw { side: -1.0 },
                Mesh3d(jaw_mesh.clone()),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_translation(jaw_local_translation(-1.0, frame.gripper_open)),
                children![(
                    Mesh3d(jaw_pad_mesh.clone()),
                    MeshMaterial3d(rubber_mat.clone()),
                    Transform::from_xyz(0.006, 0.0, 0.0),
                )],
            ),
            // Right parallel gripper jaw (+X)
            (
                GripperJaw { side: 1.0 },
                Mesh3d(jaw_mesh),
                MeshMaterial3d(aluminum_mat.clone()),
                Transform::from_translation(jaw_local_translation(1.0, frame.gripper_open)),
                children![(
                    Mesh3d(jaw_pad_mesh),
                    MeshMaterial3d(rubber_mat.clone()),
                    Transform::from_xyz(-0.006, 0.0, 0.0),
                )],
            ),
            // Vertical trigger-servo mounting tower & micro-servo housing
            (
                Mesh3d(meshes.add(Cuboid::new(0.018, 0.098, 0.014))),
                MeshMaterial3d(graphite_mat),
                Transform::from_xyz(0.0, 0.065, -0.006),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.026, 0.026, 0.022))),
                MeshMaterial3d(anodized_dark_mat.clone()),
                Transform::from_translation(TRIGGER_PIVOT_LOCAL),
            ),
            // Actuating servo trigger finger pivoting down onto nozzle actuator cap
            (
                TriggerFinger,
                trigger_local_transform(frame.trigger_depressed),
                Visibility::default(),
                children![
                    (
                        Mesh3d(meshes.add(Cuboid::new(0.010, 0.009, 0.026))),
                        MeshMaterial3d(aluminum_mat.clone()),
                        Transform::from_xyz(0.0, -0.004, -0.013),
                    ),
                    (
                        Mesh3d(meshes.add(Cylinder::new(0.005, 0.012))),
                        MeshMaterial3d(rubber_mat),
                        Transform::from_xyz(0.0, -0.012, -0.022),
                    ),
                ],
            ),
        ],
    ));

    // Shared PBR meshes & materials for aerosol spray cans
    let can_steel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.78, 0.80, 0.84),
        metallic: 0.85,
        perceptual_roughness: 0.25,
        ..default()
    });
    let label_band_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.12, 0.13, 0.15),
        metallic: 0.08,
        perceptual_roughness: 0.50,
        ..default()
    });
    let white_nozzle_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.94, 0.94, 0.93),
        metallic: 0.02,
        perceptual_roughness: 0.42,
        ..default()
    });
    let orifice_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.08, 0.09),
        metallic: 0.0,
        perceptual_roughness: 0.65,
        ..default()
    });
    let cyan_cap_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#00E5FF").unwrap().into(),
        metallic: 0.08,
        perceptual_roughness: 0.36,
        ..default()
    });
    let needle_metal_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.84, 0.87, 0.91),
        metallic: 0.94,
        perceptual_roughness: 0.16,
        ..default()
    });

    let bottom_rim_mesh = meshes.add(Cylinder::new(0.033, 0.010));
    let body_mesh = meshes.add(Cylinder::new(0.0328, 0.122));
    let label_stripe_mesh = meshes.add(Cylinder::new(0.0331, 0.018));
    let shoulder_mesh = meshes.add(
        ConicalFrustum {
            radius_top: 0.018,
            radius_bottom: 0.0328,
            height: 0.024,
        }
        .mesh()
        .resolution(28),
    );
    let cap_collar_mesh = meshes.add(Cylinder::new(0.020, 0.024));
    let nozzle_button_mesh = meshes.add(Cylinder::new(0.0085, 0.018));
    let nozzle_tip_mesh = meshes.add(Cylinder::new(0.0032, 0.014));
    let skinny_button_mesh = meshes.add(Cylinder::new(0.0080, 0.019));
    let skinny_collar_mesh = meshes.add(Cylinder::new(0.0096, 0.005));
    let skinny_needle_mesh = meshes.add(Cylinder::new(0.0014, 0.015));

    for can in &spray_job.palette.cans {
        let can_color = Color::srgb_u8(can.color_srgb[0], can.color_srgb[1], can.color_srgb[2]);
        let body_mat = materials.add(StandardMaterial {
            base_color: can_color,
            metallic: 0.14,
            perceptual_roughness: 0.34,
            ..default()
        });
        let cap_mat = materials.add(StandardMaterial {
            base_color: can_color,
            metallic: 0.04,
            perceptual_roughness: 0.42,
            ..default()
        });

        let init_tf = if frame.held_can_slot == Some(can.slot_index) {
            Transform {
                translation: grasp_xform.pos,
                rotation: grasp_xform.rot,
                scale: Vec3::ONE,
            }
        } else if let Some(slot_pose) = rack_cache.slots.get(can.slot_index) {
            Transform {
                translation: slot_pose.grasp_pos,
                rotation: slot_pose.grasp_rot,
                scale: Vec3::ONE,
            }
        } else {
            Transform::IDENTITY
        };

        let slot_cap = slot_caps.0.get(can.slot_index).copied().unwrap_or_default();
        let fat_vis = if slot_cap == NozzleCap::Fat {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        let skinny_vis = if slot_cap == NozzleCap::Skinny {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };

        commands.spawn((
            CanEntity {
                slot: can.slot_index,
            },
            init_tf,
            Visibility::default(),
            children![
                // Steel bottom rim (bottom at y = -0.085m resting flush on rack shelf)
                (
                    Mesh3d(bottom_rim_mesh.clone()),
                    MeshMaterial3d(can_steel_mat.clone()),
                    Transform::from_xyz(0.0, -0.080, 0.0),
                ),
                // Main colored aerosol body (-0.075..+0.047)
                (
                    Mesh3d(body_mesh.clone()),
                    MeshMaterial3d(body_mat),
                    Transform::from_xyz(0.0, -0.014, 0.0),
                ),
                // Crisp dark lower collar band
                (
                    Mesh3d(label_stripe_mesh.clone()),
                    MeshMaterial3d(label_band_mat.clone()),
                    Transform::from_xyz(0.0, -0.056, 0.0),
                ),
                // Domed steel shoulder (+0.047..+0.071)
                (
                    Mesh3d(shoulder_mesh.clone()),
                    MeshMaterial3d(can_steel_mat.clone()),
                    Transform::from_xyz(0.0, 0.059, 0.0),
                ),
                // Colored cap rim collar (+0.068..+0.092)
                (
                    Mesh3d(cap_collar_mesh.clone()),
                    MeshMaterial3d(cap_mat),
                    Transform::from_xyz(0.0, 0.078, 0.0),
                ),
                // Switchable Fat cap visual group (white wide actuator button + dark fat orifice)
                (
                    CanCapVisual {
                        slot: can.slot_index,
                        cap: NozzleCap::Fat,
                    },
                    Transform::IDENTITY,
                    fat_vis,
                    children![
                        (
                            Mesh3d(nozzle_button_mesh.clone()),
                            MeshMaterial3d(white_nozzle_mat.clone()),
                            Transform::from_xyz(0.0, 0.089, 0.0),
                        ),
                        (
                            Mesh3d(nozzle_tip_mesh.clone()),
                            MeshMaterial3d(orifice_mat.clone()),
                            Transform::from_xyz(0.0, 0.085, -0.009)
                                .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                        ),
                    ],
                ),
                // Switchable Skinny cap visual group (cyan stencil cap + fine metallic needle orifice)
                (
                    CanCapVisual {
                        slot: can.slot_index,
                        cap: NozzleCap::Skinny,
                    },
                    Transform::IDENTITY,
                    skinny_vis,
                    children![
                        (
                            Mesh3d(skinny_button_mesh.clone()),
                            MeshMaterial3d(cyan_cap_mat.clone()),
                            Transform::from_xyz(0.0, 0.0895, 0.0),
                        ),
                        (
                            Mesh3d(skinny_collar_mesh.clone()),
                            MeshMaterial3d(cyan_cap_mat.clone()),
                            Transform::from_xyz(0.0, 0.082, 0.0),
                        ),
                        (
                            Mesh3d(skinny_needle_mesh.clone()),
                            MeshMaterial3d(needle_metal_mat.clone()),
                            Transform::from_xyz(0.0, 0.085, -0.010)
                                .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                        ),
                    ],
                ),
            ],
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gripper_jaws_clear_can_radius_and_close_flush() {
        let closed_l = jaw_local_translation(-1.0, 0.0);
        let closed_r = jaw_local_translation(1.0, 0.0);
        // Jaw thickness is 0.012m -> inner face is at |x| - 0.006m = 0.033m (exact can radius)
        assert!((closed_r.x - 0.006 - 0.033).abs() < 1e-5);
        assert!((closed_l.x + 0.006 + 0.033).abs() < 1e-5);

        let open_r = jaw_local_translation(1.0, 1.0);
        assert!(open_r.x - 0.006 > 0.045);
    }

    #[test]
    fn trigger_finger_depresses_downward_toward_nozzle() {
        let released = trigger_local_transform(0.0);
        let pressed = trigger_local_transform(1.0);
        let tip_local = Vec3::new(0.0, -0.012, -0.022);
        let y_rel = (released * tip_local).y;
        let y_dep = (pressed * tip_local).y;
        assert!(y_dep < y_rel - 0.004);
    }

    #[test]
    fn spawns_all_nine_robot_links_matching_fk_and_end_effector_on_link_8() {
        use clap::Parser;

        use crate::cli::CliArgs;
        use crate::core::types::StudioConfig;

        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());
        app.insert_resource(AppConfig::new(CliArgs::parse_from(["spraybot"])));
        let studio = StudioConfig::default();
        let job_res = SprayJobRes::empty(&studio);
        let expected_links = job_res.chain.fk_links(&job_res.current_frame.joints);
        app.insert_resource(job_res);
        app.add_plugins(plugin);
        app.update();

        let world = app.world_mut();
        let mut link_q = world.query::<(Entity, &RobotLink, &Transform)>();
        let mut seen = vec![false; 9];
        let mut link8_entity = None;
        for (entity, link, tf) in link_q.iter(world) {
            assert!(link.index < 9, "unexpected RobotLink index {}", link.index);
            assert!(!seen[link.index], "duplicate RobotLink index {}", link.index);
            seen[link.index] = true;
            let expected = expected_links[link.index];
            assert!(
                (tf.translation - expected.pos).length() < 1e-5,
                "link {} pos mismatch: {:?} vs {:?}",
                link.index,
                tf.translation,
                expected.pos
            );
            assert!(
                tf.rotation.dot(expected.rot).abs() > 1.0 - 1e-5,
                "link {} rot mismatch: {:?} vs {:?} (angle_between={:?})",
                link.index,
                tf.rotation,
                expected.rot,
                tf.rotation.angle_between(expected.rot)
            );
            if link.index == 8 {
                link8_entity = Some(entity);
            }
        }
        assert!(seen.iter().all(|&v| v), "missing RobotLink in 0..9: {seen:?}");

        let link8 = link8_entity.expect("link 8 must exist");
        let mut jaw_q = world.query::<(&GripperJaw, &ChildOf)>();
        let jaw_parents: Vec<Entity> = jaw_q.iter(world).map(|(_, p)| p.parent()).collect();
        assert_eq!(jaw_parents, vec![link8, link8]);

        let mut trigger_q = world.query_filtered::<&ChildOf, With<TriggerFinger>>();
        let trigger_parents: Vec<Entity> = trigger_q.iter(world).map(|p| p.parent()).collect();
        assert_eq!(trigger_parents, vec![link8]);
    }
}

