use std::f32::consts::{FRAC_PI_2, PI};

use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    light::{
        CascadeShadowConfigBuilder, GeneratedEnvironmentMapLight, GlobalAmbientLight,
        NotShadowCaster, Skybox,
    },
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
    },
};

use crate::sim::state::AppConfig;

pub fn plugin(app: &mut App) {
    app.add_systems(Startup, setup_studio_environment)
        .add_systems(PreUpdate, attach_studio_env_map_to_cameras);
}

#[derive(Resource)]
struct StudioEnvMap {
    cubemap: Handle<Image>,
}

fn attach_studio_env_map_to_cameras(
    mut commands: Commands,
    env_map: Option<Res<StudioEnvMap>>,
    cameras: Query<Entity, (With<Camera3d>, Without<GeneratedEnvironmentMapLight>)>,
) {
    let Some(env_map) = env_map else {
        return;
    };
    for cam in &cameras {
        commands.entity(cam).insert((
            GeneratedEnvironmentMapLight {
                environment_map: env_map.cubemap.clone(),
                intensity: 450.0,
                ..default()
            },
            Skybox {
                image: Some(env_map.cubemap.clone()),
                brightness: 450.0,
                ..default()
            },
        ));
    }
}

fn setup_studio_environment(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    app_config: Res<AppConfig>,
) {
    let studio = &app_config.studio;

    // 1. Procedural HDR studio reflection cubemap
    let cubemap_handle = images.add(build_studio_hdr_cubemap(64));
    commands.insert_resource(StudioEnvMap {
        cubemap: cubemap_handle,
    });

    // 2. Studio room shell: scuffed concrete floor, brick/concrete back wall, side/ceiling structure
    spawn_studio_room(&mut commands, &mut meshes, &mut materials, &mut images);

    // 3. Wooden canvas stretcher frame + backing behind Z = 0.00m
    spawn_canvas_stretcher(&mut commands, &mut meshes, &mut materials, studio);

    // 4. Machined 3.80m steel linear floor rail at Z = 0.95m
    spawn_floor_rail(&mut commands, &mut meshes, &mut materials, studio);

    // 5. 2-tier 20°-yawed steel spray can rack matching StudioConfig::rack_slots to the millimeter
    spawn_spray_can_rack(
        &mut commands,
        &mut meshes,
        &mut materials,
        studio,
        app_config.cli.max_cans,
    );

    // 6. Studio lighting (directional skylight, 2 warehouse spotlights, subtle ambient)
    spawn_studio_lights(&mut commands, &mut meshes, &mut materials);
}

fn spawn_studio_room(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    images: &mut ResMut<Assets<Image>>,
) {
    let (floor_albedo, floor_roughness, floor_normal) = build_concrete_floor_textures(512);
    let floor_albedo_h = images.add(floor_albedo);
    let floor_roughness_h = images.add(floor_roughness);
    let floor_normal_h = images.add(floor_normal);

    let floor_mesh = meshes.add(
        Plane3d::default()
            .mesh()
            .size(8.0, 6.5)
            .build()
            .with_generated_tangents()
            .expect("floor tangents"),
    );
    let floor_mat = materials.add(StandardMaterial {
        base_color_texture: Some(floor_albedo_h),
        metallic_roughness_texture: Some(floor_roughness_h),
        normal_map_texture: Some(floor_normal_h),
        metallic: 0.0,
        perceptual_roughness: 1.0,
        reflectance: 0.42,
        ..default()
    });
    commands.spawn((
        Mesh3d(floor_mesh),
        MeshMaterial3d(floor_mat),
        Transform::from_xyz(0.0, 0.0, 3.21),
    ));

    // Back wall (Z = -0.04m front surface) with industrial brick/concrete block texture
    let (wall_albedo, wall_normal) = build_industrial_wall_textures(512);
    let wall_albedo_h = images.add(wall_albedo);
    let wall_normal_h = images.add(wall_normal);

    let back_wall_mesh = meshes.add(
        Cuboid::new(8.0, 4.2, 0.16)
            .mesh()
            .build()
            .with_generated_tangents()
            .expect("wall tangents"),
    );
    let back_wall_mat = materials.add(StandardMaterial {
        base_color_texture: Some(wall_albedo_h.clone()),
        normal_map_texture: Some(wall_normal_h.clone()),
        perceptual_roughness: 0.86,
        metallic: 0.02,
        ..default()
    });
    commands.spawn((
        Mesh3d(back_wall_mesh),
        MeshMaterial3d(back_wall_mat),
        // Front face sits at -0.122 + 0.08 = -0.042m (cleanly flush behind stretcher back -0.041m)
        Transform::from_xyz(0.0, 2.1, -0.122),
    ));

    // Side walls, rear wall, and ceiling
    let side_wall_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.24, 0.26, 0.28),
        perceptual_roughness: 0.88,
        metallic: 0.05,
        ..default()
    });
    let side_wall_mesh = meshes.add(Cuboid::new(0.16, 4.2, 6.5));
    commands.spawn((
        Mesh3d(side_wall_mesh.clone()),
        MeshMaterial3d(side_wall_mat.clone()),
        Transform::from_xyz(-4.08, 2.1, 3.21),
    ));
    commands.spawn((
        Mesh3d(side_wall_mesh),
        MeshMaterial3d(side_wall_mat.clone()),
        Transform::from_xyz(4.08, 2.1, 3.21),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.0, 4.2, 0.16))),
        MeshMaterial3d(side_wall_mat),
        Transform::from_xyz(0.0, 2.1, 6.54),
    ));

    // Dark industrial ceiling deck
    let ceiling_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.15, 0.16, 0.18),
        perceptual_roughness: 0.85,
        metallic: 0.25,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.0, 0.16, 6.5))),
        MeshMaterial3d(ceiling_mat),
        Transform::from_xyz(0.0, 4.28, 3.21),
    ));

    // Skylight diffuser light panels recessed in ceiling
    let skylight_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.97, 1.0),
        emissive: LinearRgba::new(3.2, 3.4, 3.8, 1.0),
        unlit: true,
        ..default()
    });
    let panel_mesh = meshes.add(Cuboid::new(2.2, 0.03, 0.95));
    for &(px, pz) in &[(-1.6, 1.5), (1.6, 1.5), (-1.6, 3.8), (1.6, 3.8)] {
        commands.spawn((
            Mesh3d(panel_mesh.clone()),
            MeshMaterial3d(skylight_mat.clone()),
            Transform::from_xyz(px, 4.185, pz),
            NotShadowCaster,
        ));
    }

    // Structural gunmetal steel columns, baseboards & ceiling trusses
    let steel_struct_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.20, 0.23),
        metallic: 0.78,
        perceptual_roughness: 0.44,
        ..default()
    });

    // Baseboards (back & side walls)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.0, 0.12, 0.03))),
        MeshMaterial3d(steel_struct_mat.clone()),
        Transform::from_xyz(0.0, 0.06, -0.027),
    ));
    let side_base_mesh = meshes.add(Cuboid::new(0.03, 0.12, 6.5));
    for &sx in &[-3.985, 3.985] {
        commands.spawn((
            Mesh3d(side_base_mesh.clone()),
            MeshMaterial3d(steel_struct_mat.clone()),
            Transform::from_xyz(sx, 0.06, 3.21),
        ));
    }

    // Vertical structural steel columns along side walls and back wall corners
    let col_mesh = meshes.add(Cuboid::new(0.16, 4.2, 0.16));
    for &z in &[0.06, 2.15, 4.25, 6.25] {
        for &x in &[-3.92, 3.92] {
            commands.spawn((
                Mesh3d(col_mesh.clone()),
                MeshMaterial3d(steel_struct_mat.clone()),
                Transform::from_xyz(x, 2.1, z),
            ));
        }
    }
    for &x in &[-3.65, 3.65] {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.14, 4.2, 0.12))),
            MeshMaterial3d(steel_struct_mat.clone()),
            Transform::from_xyz(x, 2.1, 0.02),
        ));
    }

    // Overhead steel trusses spanning X
    let chord_mesh = meshes.add(Cuboid::new(7.84, 0.08, 0.08));
    let web_mesh = meshes.add(Cuboid::new(0.05, 0.36, 0.05));
    for &tz in &[0.55, 2.20, 3.85, 5.50] {
        commands.spawn((
            Mesh3d(chord_mesh.clone()),
            MeshMaterial3d(steel_struct_mat.clone()),
            Transform::from_xyz(0.0, 4.12, tz),
        ));
        commands.spawn((
            Mesh3d(chord_mesh.clone()),
            MeshMaterial3d(steel_struct_mat.clone()),
            Transform::from_xyz(0.0, 3.74, tz),
        ));
        for ix in -3..=3 {
            commands.spawn((
                Mesh3d(web_mesh.clone()),
                MeshMaterial3d(steel_struct_mat.clone()),
                Transform::from_xyz(ix as f32 * 1.1, 3.93, tz),
            ));
        }
    }

    // Subtle safety zone floor stripe framing the robot linear track envelope
    let hazard_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.82, 0.62, 0.12),
        perceptual_roughness: 0.65,
        metallic: 0.05,
        ..default()
    });
    let stripe_x_mesh = meshes.add(Cuboid::new(4.30, 0.002, 0.035));
    let stripe_z_mesh = meshes.add(Cuboid::new(0.035, 0.002, 1.36));
    commands.spawn((
        Mesh3d(stripe_x_mesh.clone()),
        MeshMaterial3d(hazard_mat.clone()),
        Transform::from_xyz(-0.15, 0.001, 1.56),
    ));
    commands.spawn((
        Mesh3d(stripe_x_mesh),
        MeshMaterial3d(hazard_mat.clone()),
        Transform::from_xyz(-0.15, 0.001, 0.22),
    ));
    commands.spawn((
        Mesh3d(stripe_z_mesh.clone()),
        MeshMaterial3d(hazard_mat.clone()),
        Transform::from_xyz(-2.28, 0.001, 0.89),
    ));
    commands.spawn((
        Mesh3d(stripe_z_mesh),
        MeshMaterial3d(hazard_mat),
        Transform::from_xyz(1.98, 0.001, 0.89),
    ));
}

fn spawn_canvas_stretcher(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    studio: &crate::core::types::StudioConfig,
) {
    let wood_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.68, 0.52, 0.34),
        perceptual_roughness: 0.72,
        metallic: 0.0,
        ..default()
    });
    let backing_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.56, 0.42, 0.27),
        perceptual_roughness: 0.78,
        metallic: 0.0,
        ..default()
    });
    let bracket_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.16, 0.17, 0.19),
        metallic: 0.82,
        perceptual_roughness: 0.38,
        ..default()
    });

    let w = studio.canvas_width; // 2.50m
    let h = studio.canvas_height; // 1.50m
    let d = studio.canvas_depth; // 0.04m
    let cx = studio.canvas_center.x;
    let cy = studio.canvas_center.y; // 1.55m
    let cz = -0.021; // front face at -0.001m, flush behind canvas.rs (Z = 0.001m)

    let frame_bar = 0.055;

    // Top and bottom stretcher rails
    let horiz_mesh = meshes.add(Cuboid::new(w, frame_bar, d));
    commands.spawn((
        Mesh3d(horiz_mesh.clone()),
        MeshMaterial3d(wood_mat.clone()),
        Transform::from_xyz(cx, cy + h * 0.5 - frame_bar * 0.5, cz),
    ));
    commands.spawn((
        Mesh3d(horiz_mesh),
        MeshMaterial3d(wood_mat.clone()),
        Transform::from_xyz(cx, cy - h * 0.5 + frame_bar * 0.5, cz),
    ));

    // Left and right stretcher stiles
    let vert_mesh = meshes.add(Cuboid::new(frame_bar, h - 2.0 * frame_bar, d));
    commands.spawn((
        Mesh3d(vert_mesh.clone()),
        MeshMaterial3d(wood_mat.clone()),
        Transform::from_xyz(cx - w * 0.5 + frame_bar * 0.5, cy, cz),
    ));
    commands.spawn((
        Mesh3d(vert_mesh),
        MeshMaterial3d(wood_mat.clone()),
        Transform::from_xyz(cx + w * 0.5 - frame_bar * 0.5, cy, cz),
    ));

    // Internal cross-braces
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.045, h - 2.0 * frame_bar, d * 0.8))),
        MeshMaterial3d(wood_mat.clone()),
        Transform::from_xyz(cx, cy, cz - 0.003),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(w - 2.0 * frame_bar, 0.045, d * 0.8))),
        MeshMaterial3d(wood_mat),
        Transform::from_xyz(cx, cy, cz - 0.003),
    ));

    // Recessed solid backing board (front face at Z = -0.002m)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(w - 0.006, h - 0.006, 0.014))),
        MeshMaterial3d(backing_mat),
        Transform::from_xyz(cx, cy, -0.009),
    ));

    // Wall-mounting steel cleats
    let cleat_mesh = meshes.add(Cuboid::new(0.10, 0.035, 0.03));
    for &(bx, by) in &[
        (cx - 0.90, cy + h * 0.5 + 0.014),
        (cx + 0.90, cy + h * 0.5 + 0.014),
        (cx - 0.90, cy - h * 0.5 - 0.014),
        (cx + 0.90, cy - h * 0.5 - 0.014),
    ] {
        commands.spawn((
            Mesh3d(cleat_mesh.clone()),
            MeshMaterial3d(bracket_mat.clone()),
            Transform::from_xyz(bx, by, cz - 0.005),
        ));
    }
}

fn spawn_floor_rail(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    studio: &crate::core::types::StudioConfig,
) {
    // 3.80m rail: X in [-2.05, +1.75], center X = -0.15m, Z = studio.rail_z (0.95m)
    let rail_len = studio.rail_length;
    let rail_center_x = -0.15;
    let rz = studio.rail_z;

    let bed_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.14, 0.15, 0.17),
        metallic: 0.72,
        perceptual_roughness: 0.48,
        ..default()
    });
    let machined_steel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.79, 0.82),
        metallic: 0.93,
        perceptual_roughness: 0.18,
        ..default()
    });
    let screw_steel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.54, 0.56, 0.60),
        metallic: 0.88,
        perceptual_roughness: 0.28,
        ..default()
    });
    let bumper_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#E85D1A").unwrap().into(),
        metallic: 0.05,
        perceptual_roughness: 0.55,
        ..default()
    });

    // Main cast-iron rail bed
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(rail_len, 0.086, 0.28))),
        MeshMaterial3d(bed_mat.clone()),
        Transform::from_xyz(rail_center_x, 0.059, rz),
    ));

    // Floor leveling / anchor foot plates
    let foot_mesh = meshes.add(Cuboid::new(0.16, 0.016, 0.38));
    for i in 0..6 {
        let fx = -1.95 + i as f32 * 0.72;
        commands.spawn((
            Mesh3d(foot_mesh.clone()),
            MeshMaterial3d(bed_mat.clone()),
            Transform::from_xyz(fx, 0.008, rz),
        ));
    }

    // Twin precision-ground linear guideways
    let guide_mesh = meshes.add(Cuboid::new(rail_len - 0.04, 0.022, 0.024));
    for &dz in &[-0.095, 0.095] {
        commands.spawn((
            Mesh3d(guide_mesh.clone()),
            MeshMaterial3d(machined_steel_mat.clone()),
            Transform::from_xyz(rail_center_x, 0.113, rz + dz),
        ));
    }

    // Central drive ball-screw shaft
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.014, rail_len - 0.12))),
        MeshMaterial3d(screw_steel_mat),
        Transform::from_xyz(rail_center_x, 0.112, rz)
            .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
    ));

    // Cable drag-chain tray along back edge of rail
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(rail_len - 0.08, 0.036, 0.055))),
        MeshMaterial3d(bed_mat.clone()),
        Transform::from_xyz(rail_center_x, 0.038, rz - 0.172),
    ));

    // End-stop blocks and orange urethane bumpers
    let stop_mesh = meshes.add(Cuboid::new(0.065, 0.072, 0.26));
    let bump_mesh = meshes.add(Cuboid::new(0.020, 0.042, 0.055));
    for &(sx, bx) in &[(-2.015, -1.974), (1.715, 1.674)] {
        commands.spawn((
            Mesh3d(stop_mesh.clone()),
            MeshMaterial3d(bed_mat.clone()),
            Transform::from_xyz(sx, 0.138, rz),
        ));
        commands.spawn((
            Mesh3d(bump_mesh.clone()),
            MeshMaterial3d(bumper_mat.clone()),
            Transform::from_xyz(bx, 0.145, rz),
        ));
    }
}

fn spawn_spray_can_rack(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    studio: &crate::core::types::StudioConfig,
    max_cans: usize,
) {
    let slots = studio.rack_slots(max_cans);
    let upper_count = max_cans.div_ceil(2);
    let lower_count = max_cans - upper_count;
    let tier_slots = upper_count.max(lower_count).max(1);

    let rack_width = tier_slots as f32 * studio.rack_slot_spacing + 0.070;
    let half_w = rack_width * 0.5;
    let can_bottom_offset = -0.085; // matches CanEntity bottom rim in grasp frame

    let rack_steel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.16, 0.18, 0.21),
        metallic: 0.78,
        perceptual_roughness: 0.40,
        ..default()
    });
    let cradle_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.32, 0.35, 0.40),
        metallic: 0.84,
        perceptual_roughness: 0.32,
        ..default()
    });

    let rack_rot = Quat::from_rotation_y(studio.rack_yaw_deg.to_radians());
    let rack_root = commands
        .spawn((
            Transform::from_translation(Vec3::new(
                studio.rack_center_xz.x,
                0.0,
                studio.rack_center_xz.z,
            ))
            .with_rotation(rack_rot),
            Visibility::default(),
        ))
        .id();

    let shelf_mesh = meshes.add(Cuboid::new(rack_width, 0.010, 0.092));
    let back_lip_mesh = meshes.add(Cuboid::new(rack_width, 0.032, 0.008));
    let front_lip_mesh = meshes.add(Cuboid::new(rack_width, 0.015, 0.007));
    let divider_mesh = meshes.add(Cuboid::new(0.007, 0.022, 0.076));

    let tiers = [
        (
            upper_count,
            studio.rack_upper_grasp_y + can_bottom_offset,
            -studio.rack_tier_step_n,
        ),
        (
            lower_count,
            studio.rack_lower_grasp_y + can_bottom_offset,
            studio.rack_tier_step_n,
        ),
    ];

    commands.entity(rack_root).with_children(|parent| {
        for (count, shelf_top_y, tier_z) in tiers {
            if count == 0 {
                continue;
            }
            // Shelf plate: top face sits cleanly at shelf_top_y = grasp_y - 0.085
            parent.spawn((
                Mesh3d(shelf_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(0.0, shelf_top_y - 0.005, tier_z),
            ));
            // Rear backstop rail
            parent.spawn((
                Mesh3d(back_lip_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(0.0, shelf_top_y + 0.016, tier_z - 0.042),
            ));
            // Front retention lip (0.015m high — cleared cleanly by the 0.06m PickCan vertical lift)
            parent.spawn((
                Mesh3d(front_lip_mesh.clone()),
                MeshMaterial3d(cradle_mat.clone()),
                Transform::from_xyz(0.0, shelf_top_y + 0.0075, tier_z + 0.0425),
            ));

            // Cradle divider fins between & outside each slot
            for d in 0..=count {
                let x_div =
                    (d as f32 - count as f32 * 0.5) * studio.rack_slot_spacing;
                parent.spawn((
                    Mesh3d(divider_mesh.clone()),
                    MeshMaterial3d(cradle_mat.clone()),
                    Transform::from_xyz(x_div, shelf_top_y + 0.011, tier_z),
                ));
            }
        }

        // Left & right welded steel side frames + floor baseplates
        let leg_mesh = meshes.add(Cuboid::new(0.036, 1.20, 0.050));
        let front_leg_mesh = meshes.add(Cuboid::new(0.034, 0.86, 0.044));
        let foot_mesh = meshes.add(Cuboid::new(0.090, 0.014, 0.24));
        let gusset_mesh = meshes.add(Cuboid::new(0.034, 0.045, 0.16));

        for &side in &[-1.0_f32, 1.0] {
            let x_leg = side * (half_w - 0.012);
            // Rear upright supporting upper tier
            parent.spawn((
                Mesh3d(leg_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(x_leg, 0.60, -studio.rack_tier_step_n - 0.025),
            ));
            // Front upright supporting lower tier
            parent.spawn((
                Mesh3d(front_leg_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(x_leg, 0.43, studio.rack_tier_step_n + 0.015),
            ));
            // Floor mounting plate
            parent.spawn((
                Mesh3d(foot_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(x_leg, 0.007, 0.0),
            ));
            // Mid tie gusset
            parent.spawn((
                Mesh3d(gusset_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(x_leg, 0.52, 0.0),
            ));
        }

        // Rear cross-braces connecting left & right frames
        let brace_mesh = meshes.add(Cuboid::new(rack_width - 0.02, 0.032, 0.032));
        parent.spawn((
            Mesh3d(brace_mesh.clone()),
            MeshMaterial3d(rack_steel_mat.clone()),
            Transform::from_xyz(0.0, 0.28, -studio.rack_tier_step_n - 0.025),
        ));
        parent.spawn((
            Mesh3d(brace_mesh),
            MeshMaterial3d(rack_steel_mat),
            Transform::from_xyz(0.0, 0.72, -studio.rack_tier_step_n - 0.025),
        ));
    });

    // Subtle slot number verification: ensure every slot from StudioConfig sits above a shelf
    debug_assert_eq!(slots.len(), max_cans);
}

fn spawn_studio_lights(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    // Subtle studio ambient bounce
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.90, 0.92, 0.96),
        brightness: 70.0,
        affects_lightmapped_meshes: true,
    });

    // Key overhead skylight directional light casting crisp soft PCF shadows
    commands.spawn((
        DirectionalLight {
            illuminance: 2_400.0,
            color: Color::srgb(0.98, 0.99, 1.0),
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(2.0, 5.5, 4.2).looking_at(Vec3::new(-0.3, 1.2, 0.4), Vec3::Y),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 5.0,
            maximum_distance: 15.0,
            ..default()
        }
        .build(),
    ));

    let housing_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.12, 0.13, 0.15),
        metallic: 0.80,
        perceptual_roughness: 0.35,
        ..default()
    });
    let lens_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        emissive: LinearRgba::new(6.0, 5.7, 5.2, 1.0),
        unlit: true,
        ..default()
    });
    let can_mesh = meshes.add(Cylinder::new(0.085, 0.22));
    let lens_mesh = meshes.add(Cylinder::new(0.072, 0.015));

    // Spotlight 1: aimed at the wall canvas
    commands.spawn((
        SpotLight {
            intensity: 165_000.0,
            color: Color::srgb(1.0, 0.96, 0.91),
            range: 12.0,
            inner_angle: 0.42,
            outer_angle: 0.72,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.35, 3.66, 2.35).looking_at(Vec3::new(0.0, 1.55, 0.0), Vec3::Y),
        children![
            (
                Mesh3d(can_mesh.clone()),
                MeshMaterial3d(housing_mat.clone()),
                Transform::from_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
            ),
            (
                Mesh3d(lens_mesh.clone()),
                MeshMaterial3d(lens_mat.clone()),
                Transform::from_xyz(0.0, 0.0, -0.105)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
            ),
        ],
    ));

    // Spotlight 2: aimed at the spray can rack and robot home station
    commands.spawn((
        SpotLight {
            intensity: 115_000.0,
            color: Color::srgb(0.95, 0.97, 1.0),
            range: 11.0,
            inner_angle: 0.38,
            outer_angle: 0.68,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-1.20, 3.56, 2.15).looking_at(Vec3::new(-1.65, 1.10, 0.55), Vec3::Y),
        children![
            (
                Mesh3d(can_mesh),
                MeshMaterial3d(housing_mat),
                Transform::from_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
            ),
            (
                Mesh3d(lens_mesh),
                MeshMaterial3d(lens_mat),
                Transform::from_xyz(0.0, 0.0, -0.105)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
            ),
        ],
    ));
}

#[inline]
fn hash2(x: i32, y: i32, seed: u32) -> f32 {
    let mut n = (x.wrapping_mul(374761393))
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(seed as i32);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    ((n ^ (n >> 16)) as u32) as f32 / (u32::MAX as f32)
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - xi as f32;
    let yf = y - yi as f32;
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);
    let a = hash2(xi, yi, seed);
    let b = hash2(xi + 1, yi, seed);
    let c = hash2(xi, yi + 1, seed);
    let d = hash2(xi + 1, yi + 1, seed);
    (a + (b - a) * u) + ((c - a) + (a - b - c + d) * u) * v
}

fn fbm(x: f32, y: f32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for oct in 0..4 {
        sum += amp * value_noise(x * freq, y * freq, seed.wrapping_add(oct * 101));
        freq *= 2.07;
        amp *= 0.5;
    }
    sum
}

fn build_concrete_floor_textures(size: u32) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;

            let macro_n = fbm(u * 10.0, v * 8.0, 17);
            let micro_n = fbm(u * 48.0, v * 40.0, 93);
            let scuff = (fbm(u * 18.0 + 3.1, v * 6.0, 211) - 0.52).clamp(0.0, 1.0);

            // Concrete expansion slab joints every 25% of UV (2.0m x 1.625m slabs)
            let gx = ((u * 4.0).fract() - 0.5).abs();
            let gy = ((v * 4.0).fract() - 0.5).abs();
            let seam = if gx > 0.494 || gy > 0.494 { 1.0 } else { 0.0 };

            let base_luma = (0.34 + 0.12 * macro_n + 0.06 * micro_n - 0.08 * scuff - 0.11 * seam)
                .clamp(0.12, 0.62);

            let r = (base_luma * 1.00 * 255.0).round() as u8;
            let g = (base_luma * 1.01 * 255.0).round() as u8;
            let b = (base_luma * 1.04 * 255.0).round() as u8;
            albedo.extend_from_slice(&[r, g, b, 255]);

            // Roughness in Green channel (glTF/Bevy metallic-roughness convention: G = roughness, B = metallic)
            let r_val = (0.56 + 0.24 * scuff + 0.14 * micro_n + 0.20 * seam).clamp(0.38, 0.94);
            let r_u8 = (r_val * 255.0).round() as u8;
            rough.extend_from_slice(&[0, r_u8, 0, 255]);

            heights[(y * size + x) as usize] = macro_n * 0.4 + micro_n * 0.6 - seam * 0.8;
        }
    }

    let normal = normal_map_from_heights(&heights, size, 1.8);

    let mut albedo_img = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        albedo,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    albedo_img.sampler = ImageSampler::linear();

    let mut rough_img = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rough,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    rough_img.sampler = ImageSampler::linear();

    let mut normal_img = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        normal,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    normal_img.sampler = ImageSampler::linear();

    (albedo_img, rough_img, normal_img)
}

fn build_industrial_wall_textures(size: u32) -> (Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    let cols = 16.0_f32;
    let rows = 20.0_f32;

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;

            let row_idx = (v * rows).floor();
            let row_f = (v * rows).fract();
            let stagger = if (row_idx as i32) % 2 == 0 { 0.0 } else { 0.5 };
            let col_f = (u * cols + stagger).fract();

            let mortar = row_f < 0.07 || row_f > 0.93 || col_f < 0.035 || col_f > 0.965;
            let block_id = ((u * cols + stagger).floor() as i32) * 31 + (row_idx as i32) * 17;
            let block_tint = hash2(block_id, 5, 777) * 0.05;
            let grit = fbm(u * 36.0, v * 36.0, 313);

            let base = if mortar {
                0.20 + 0.04 * grit
            } else {
                0.26 + block_tint + 0.07 * grit
            };

            let r = (base * 1.02 * 255.0).clamp(0.0, 255.0) as u8;
            let g = (base * 1.00 * 255.0).clamp(0.0, 255.0) as u8;
            let b = (base * 1.03 * 255.0).clamp(0.0, 255.0) as u8;
            albedo.extend_from_slice(&[r, g, b, 255]);

            heights[(y * size + x) as usize] = if mortar { -0.45 } else { grit * 0.45 };
        }
    }

    let normal = normal_map_from_heights(&heights, size, 2.2);

    let mut albedo_img = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        albedo,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    albedo_img.sampler = ImageSampler::linear();

    let mut normal_img = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        normal,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    normal_img.sampler = ImageSampler::linear();

    (albedo_img, normal_img)
}

fn normal_map_from_heights(heights: &[f32], size: u32, strength: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    let s = size as i32;
    for y in 0..s {
        for x in 0..s {
            let xm = (x - 1).rem_euclid(s) as usize;
            let xp = (x + 1).rem_euclid(s) as usize;
            let ym = (y - 1).rem_euclid(s) as usize;
            let yp = (y + 1).rem_euclid(s) as usize;
            let row = y as usize * size as usize;
            let dx = heights[row + xp] - heights[row + xm];
            let dy = heights[yp * size as usize + x as usize]
                - heights[ym * size as usize + x as usize];
            let n = Vec3::new(-dx * strength, -dy * strength, 1.0).normalize();
            out.push(((n.x * 0.5 + 0.5) * 255.0).round() as u8);
            out.push(((n.y * 0.5 + 0.5) * 255.0).round() as u8);
            out.push(((n.z * 0.5 + 0.5) * 255.0).round() as u8);
            out.push(255);
        }
    }
    out
}

/// Builds a 6-face `Rgba16Float` HDR studio warehouse cubemap with skylight softboxes,
/// warm industrial wall bounce, and dark concrete floor reflections.
fn build_studio_hdr_cubemap(face_size: u32) -> Image {
    let mut data = Vec::with_capacity((face_size * face_size * 6 * 8) as usize);

    for face in 0..6 {
        for y in 0..face_size {
            let v = 2.0 * ((y as f32 + 0.5) / face_size as f32) - 1.0;
            for x in 0..face_size {
                let u = 2.0 * ((x as f32 + 0.5) / face_size as f32) - 1.0;
                let dir = cubemap_face_dir(face, u, v).normalize();
                let hdr = sample_studio_hdr(dir);
                for ch in hdr {
                    let bits = f32_to_f16_bits(ch);
                    data.extend_from_slice(&bits.to_le_bytes());
                }
            }
        }
    }

    Image {
        texture_view_descriptor: Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        }),
        ..Image::new(
            Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            data,
            TextureFormat::Rgba16Float,
            RenderAssetUsages::RENDER_WORLD,
        )
    }
}

#[inline]
fn cubemap_face_dir(face: usize, u: f32, v: f32) -> Vec3 {
    match face {
        0 => Vec3::new(1.0, -v, -u),  // +X
        1 => Vec3::new(-1.0, -v, u),  // -X
        2 => Vec3::new(u, 1.0, v),    // +Y
        3 => Vec3::new(u, -1.0, -v),  // -Y
        4 => Vec3::new(u, -v, 1.0),   // +Z
        _ => Vec3::new(-u, -v, -1.0), // -Z
    }
}

fn sample_studio_hdr(dir: Vec3) -> [f32; 4] {
    // Base gradient: dark concrete floor -> warm/cool industrial walls -> ceiling deck
    let t_up = (dir.y * 0.5 + 0.5).clamp(0.0, 1.0);
    let mut rgb = if dir.y < 0.0 {
        let f = (-dir.y).clamp(0.0, 1.0);
        Vec3::new(0.24, 0.23, 0.22).lerp(Vec3::new(0.07, 0.075, 0.08), f.sqrt())
    } else {
        Vec3::new(0.26, 0.25, 0.24).lerp(Vec3::new(0.18, 0.21, 0.26), t_up)
    };

    // Bright overhead skylight softbox panels on the ceiling hemisphere
    if dir.y > 0.35 {
        let px = dir.x / dir.y;
        let pz = dir.z / dir.y;
        let in_panel_a = (px.abs() - 0.55).abs() < 0.38 && (pz - 0.25).abs() < 0.42;
        let in_panel_b = px.abs() < 0.45 && (pz + 0.45).abs() < 0.32;
        if in_panel_a || in_panel_b {
            rgb += Vec3::new(6.8, 7.2, 7.8);
        }
    }

    // Warm industrial key reflection strips on side/front upper walls
    let warm_strip = (dir.y - 0.28).abs() < 0.12
        && (dir.z > 0.35 || dir.x.abs() > 0.65);
    if warm_strip {
        let falloff = (1.0 - (dir.y - 0.28).abs() / 0.12).clamp(0.0, 1.0);
        rgb += Vec3::new(2.2, 1.65, 1.15) * falloff;
    }

    // Subtle cool accent rim behind canvas side
    if dir.z < -0.5 && dir.y > -0.1 && dir.y < 0.5 {
        rgb += Vec3::new(0.25, 0.32, 0.42) * (1.0 - (dir.y - 0.2).abs() / 0.3).clamp(0.0, 1.0);
    }

    let _ = PI;
    [rgb.x, rgb.y, rgb.z, 1.0]
}

/// Converts an `f32` into IEEE 754 binary16 (`f16`) bit representation.
#[inline]
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf or NaN
        return sign | 0x7c00 | if frac != 0 { 0x0200 } else { 0 };
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = (frac | 0x0080_0000) >> (14 - new_exp);
        return sign | (m as u16);
    }

    sign | ((new_exp as u16) << 10) | ((frac >> 13) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studio_hdr_cubemap_is_valid_6_face_cube() {
        let img = build_studio_hdr_cubemap(64);
        assert_eq!(img.texture_descriptor.size.width, 64);
        assert_eq!(img.texture_descriptor.size.height, 64);
        assert_eq!(img.texture_descriptor.size.depth_or_array_layers, 6);
        assert_eq!(img.texture_descriptor.dimension, TextureDimension::D2);
        assert_eq!(img.texture_descriptor.format, TextureFormat::Rgba16Float);
        assert_eq!(
            img.texture_view_descriptor.as_ref().and_then(|d| d.dimension),
            Some(TextureViewDimension::Cube)
        );
        assert_eq!(img.data.as_ref().map(|d| d.len()), Some(64 * 64 * 6 * 8));
    }

    #[test]
    fn f16_conversion_handles_zero_one_and_hdr() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(8.0), 0x4800);
    }
}

