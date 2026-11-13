use geo::MultiPolygon;
use glam::{DVec2, Quat, Vec2, Vec3};

/// Normalized 2D vector geometry extracted from SVG in world-canvas meters.
///
/// Canvas coordinates here are 2D world `(x, y)` on the `Z = 0.0` wall plane
/// (`x in [-1.25, +1.25]`, `y in [0.80, 2.30]`, centered at `(0.0, 1.55)`).
#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveGeom {
    FilledPolygon(MultiPolygon<f64>),
    StrokedPath { points: Vec<DVec2>, width: f64 },
}

/// Single drawable SVG element in painter's back-to-front Z-order.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedPrimitive {
    pub id: usize,
    pub color_srgb: [u8; 3],
    pub opacity: f32,
    pub geometry: PrimitiveGeom,
}

/// Physical aerosol spray can assigned to a rack slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SprayCanSpec {
    pub slot_index: usize,
    pub color_srgb: [u8; 3],
    pub label: String,
}

/// Result of perceptual OKLab palette extraction / quantization.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PalettePlan {
    pub cans: Vec<SprayCanSpec>,
    /// Maps `primitive.id` -> `slot_index` in `cans`.
    pub primitive_color_map: Vec<usize>,
}

/// Single 3D toolpath sample on the wall canvas plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeWaypoint {
    /// Target paint point on the canvas surface (meters, typically `Z = 0.0`).
    pub pos: Vec3,
    /// Unit outward surface normal from canvas (`+Z` for front wall).
    pub normal: Vec3,
    /// Nozzle standoff distance from `pos` along `normal` (meters).
    pub standoff: f32,
    /// Desired linear nozzle speed along the path (m/s).
    pub feed_speed: f32,
    /// Whether the spray valve is depressed at this waypoint.
    pub valve_on: bool,
    /// Flow/opacity multiplier in `(0.0, 1.0]`.
    pub opacity_scale: f32,
}

impl StrokeWaypoint {
    /// World-space 3D nozzle tip position (`pos + normal * standoff`).
    #[inline]
    pub fn nozzle_pos(&self) -> Vec3 {
        self.pos + self.normal * self.standoff
    }

    /// World-space spray plume direction (`-normal`).
    #[inline]
    pub fn spray_dir(&self) -> Vec3 {
        -self.normal
    }
}

/// High-level choreography operation produced by the CAM job planner.
#[derive(Debug, Clone, PartialEq)]
pub enum JobOp {
    PickCan { slot: usize },
    ShakeCan,
    PaintStroke { points_canvas_3d: Vec<StrokeWaypoint> },
    ReturnCan { slot: usize },
    ParkHome,
}

/// Ordered sequence of operations to paint an entire SVG mural.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JobPlan {
    pub ops: Vec<JobOp>,
}

/// Continuous sampled robot and spray state at simulation time `t_sim`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobotFrameState {
    /// `joints[0]` = floor rail X (meters); `joints[1..7]` = 6 arm revolute angles (radians).
    pub joints: [f64; 7],
    /// Gripper jaw opening: `0.0` (closed on can) .. `1.0` (open).
    pub gripper_open: f32,
    /// Trigger finger depression: `0.0` (released) .. `1.0` (spraying).
    pub trigger_depressed: f32,
    /// Rack slot index of the can currently held by the gripper, if any.
    pub held_can_slot: Option<usize>,
    /// World-space position of the active nozzle tip (or TCP when no can held).
    pub nozzle_world_pos: Vec3,
    /// World-space unit spray direction vector.
    pub nozzle_world_dir: Vec3,
    /// True while paint should actively deposit onto the canvas and emit mist.
    pub spray_active: bool,
    /// Instantaneous flow/opacity multiplier in `[0.0, 1.0]`.
    pub flow_scale: f32,
}

impl Default for RobotFrameState {
    fn default() -> Self {
        Self {
            joints: [0.0; 7],
            gripper_open: 1.0,
            trigger_depressed: 0.0,
            held_can_slot: None,
            nozzle_world_pos: Vec3::new(0.0, 1.35, 0.50),
            nozzle_world_dir: Vec3::NEG_Z,
            spray_active: false,
            flow_scale: 0.0,
        }
    }
}

/// World-space rigid transform for a single robot link (decoupled from Bevy `Transform`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkTransform {
    pub pos: Vec3,
    pub rot: Quat,
}

impl Default for LinkTransform {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        }
    }
}

/// Target grasp pose for a single spray can slot on the 2-tier rack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RackSlotPose {
    pub slot_index: usize,
    /// World-space center point for grasping the aerosol can in this slot.
    pub grasp_pos: Vec3,
    /// World-space orientation aligned with the rack (`+X = t_rack`, `+Y = +Y`, `+Z = n_rack`).
    pub grasp_rot: Quat,
    /// Unit vector pointing outward from the rack front toward the robot (`n_rack`).
    /// Safe pre-grasp approach pose lies at `grasp_pos + approach_dir * clearance`.
    pub approach_dir: Vec3,
}

/// Physical dimensions and validated studio geometry (right-handed Y-up, 1 unit = 1m).
#[derive(Debug, Clone, PartialEq)]
pub struct StudioConfig {
    // Wall canvas
    pub canvas_width: f32,
    pub canvas_height: f32,
    pub canvas_depth: f32,
    pub canvas_center: Vec3,
    pub canvas_margin: f32,

    // Floor linear rail
    pub rail_length: f32,
    pub rail_z: f32,
    pub rail_q0_min: f32,
    pub rail_q0_max: f32,
    pub carriage_y: f32,

    // Primary 6-DOF arm link dimensions
    pub shoulder_pivot_y: f32,
    pub upper_arm_l2: f32,
    pub forearm_l3: f32,

    // 2-tier stadium-stepped spray can rack
    pub rack_center_xz: Vec3,
    pub rack_yaw_deg: f32,
    pub rack_slots_per_tier_default: usize,
    pub rack_slot_spacing: f32,
    pub rack_upper_grasp_y: f32,
    pub rack_lower_grasp_y: f32,
    pub rack_tier_step_n: f32,
    pub can_diameter: f32,
    pub can_height: f32,
}

impl Default for StudioConfig {
    fn default() -> Self {
        Self {
            canvas_width: 2.50,
            canvas_height: 1.50,
            canvas_depth: 0.04,
            canvas_center: Vec3::new(0.0, 1.55, 0.0),
            canvas_margin: 0.08,

            rail_length: 3.80,
            rail_z: 0.95,
            rail_q0_min: -1.75,
            rail_q0_max: 1.45,
            carriage_y: 0.18,

            shoulder_pivot_y: 1.15,
            upper_arm_l2: 0.68,
            forearm_l3: 0.60,

            rack_center_xz: Vec3::new(-1.95, 0.0, 0.52),
            rack_yaw_deg: 20.0,
            rack_slots_per_tier_default: 8,
            rack_slot_spacing: 0.088,
            rack_upper_grasp_y: 1.24,
            rack_lower_grasp_y: 0.90,
            rack_tier_step_n: 0.04,
            can_diameter: 0.066,
            can_height: 0.19,
        }
    }
}

impl StudioConfig {
    /// Returns the paintable `(min_xy, max_xy)` rectangle on the canvas front plane (`Z = 0.0`)
    /// after subtracting `canvas_margin` on all four sides.
    pub fn canvas_paintable_bounds(&self) -> (Vec2, Vec2) {
        let half_w = (self.canvas_width * 0.5 - self.canvas_margin).max(0.0);
        let half_h = (self.canvas_height * 0.5 - self.canvas_margin).max(0.0);
        let center_xy = Vec2::new(self.canvas_center.x, self.canvas_center.y);
        (
            center_xy - Vec2::new(half_w, half_h),
            center_xy + Vec2::new(half_w, half_h),
        )
    }

    /// Returns `(t_rack, n_rack)` unit basis vectors for the 20°-yawed spray can rack.
    ///
    /// - `n_rack`: outward normal facing toward the robot rail (`≈ normalize(Vec3(0.342, 0, 0.940))`)
    /// - `t_rack`: tangent along the rack shelf (`≈ Vec3(0.940, 0, -0.342)`)
    pub fn rack_axes(&self) -> (Vec3, Vec3) {
        let (sin_y, cos_y) = self.rack_yaw_deg.to_radians().sin_cos();
        let n_rack = Vec3::new(sin_y, 0.0, cos_y).normalize();
        let t_rack = Vec3::new(cos_y, 0.0, -sin_y).normalize();
        (t_rack, n_rack)
    }

    /// Generates `max_cans` slot poses distributed across the 2 stadium-stepped tiers.
    ///
    /// Upper tier (`slots 0..upper_count`) sits at `grasp_y = 1.24m`, stepped `-0.04m * n_rack`;
    /// lower tier (`slots upper_count..max_cans`) sits at `grasp_y = 0.90m`, stepped `+0.04m * n_rack`.
    pub fn rack_slots(&self, max_cans: usize) -> Vec<RackSlotPose> {
        if max_cans == 0 {
            return Vec::new();
        }

        let (t_rack, n_rack) = self.rack_axes();
        let grasp_rot = Quat::from_rotation_y(self.rack_yaw_deg.to_radians());

        let upper_count = max_cans.div_ceil(2);
        let lower_count = max_cans - upper_count;

        let mut slots = Vec::with_capacity(max_cans);
        let tiers = [
            (upper_count, self.rack_upper_grasp_y, -self.rack_tier_step_n),
            (lower_count, self.rack_lower_grasp_y, self.rack_tier_step_n),
        ];

        for (count, grasp_y, step_n) in tiers {
            let tier_center = Vec3::new(self.rack_center_xz.x, grasp_y, self.rack_center_xz.z)
                + n_rack * step_n;
            for i in 0..count {
                let offset = (i as f32 - (count as f32 - 1.0) * 0.5) * self.rack_slot_spacing;
                let grasp_pos = tier_center + t_rack * offset;
                slots.push(RackSlotPose {
                    slot_index: slots.len(),
                    grasp_pos,
                    grasp_rot,
                    approach_dir: n_rack,
                });
            }
        }

        slots
    }

    /// Validates that studio dimensions and limits are physically consistent.
    pub fn is_valid(&self) -> bool {
        self.canvas_width > 0.0
            && self.canvas_height > 0.0
            && self.canvas_depth > 0.0
            && self.canvas_margin >= 0.0
            && self.canvas_margin * 2.0 < self.canvas_width
            && self.canvas_margin * 2.0 < self.canvas_height
            && self.rail_length > 0.0
            && self.rail_q0_min < self.rail_q0_max
            && self.upper_arm_l2 > 0.0
            && self.forearm_l3 > 0.0
            && self.rack_slot_spacing > 0.0
            && self.can_diameter > 0.0
            && self.can_height > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn studio_config_default_is_valid() {
        let cfg = StudioConfig::default();
        assert!(cfg.is_valid());
        let (min_xy, max_xy) = cfg.canvas_paintable_bounds();
        assert_relative_eq!(min_xy.x, -1.17, epsilon = 1e-5);
        assert_relative_eq!(min_xy.y, 0.88, epsilon = 1e-5);
        assert_relative_eq!(max_xy.x, 1.17, epsilon = 1e-5);
        assert_relative_eq!(max_xy.y, 2.22, epsilon = 1e-5);
    }

    #[test]
    fn stroke_waypoint_helpers() {
        let wp = StrokeWaypoint {
            pos: Vec3::new(0.5, 1.5, 0.0),
            normal: Vec3::Z,
            standoff: 0.12,
            feed_speed: 0.35,
            valve_on: true,
            opacity_scale: 0.8,
        };
        assert_relative_eq!(wp.nozzle_pos().x, 0.5, epsilon = 1e-6);
        assert_relative_eq!(wp.nozzle_pos().y, 1.5, epsilon = 1e-6);
        assert_relative_eq!(wp.nozzle_pos().z, 0.12, epsilon = 1e-6);
        assert_relative_eq!(wp.spray_dir().x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(wp.spray_dir().y, 0.0, epsilon = 1e-6);
        assert_relative_eq!(wp.spray_dir().z, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn rack_slots_16_geometry_and_reachability() {
        let cfg = StudioConfig::default();
        let (t_rack, n_rack) = cfg.rack_axes();

        // Check expected 20-deg basis vectors
        let expected_n = Vec3::new(0.342, 0.0, 0.940).normalize();
        let expected_t = Vec3::new(0.940, 0.0, -0.342).normalize();
        assert_relative_eq!(n_rack.x, expected_n.x, epsilon = 5e-4);
        assert_relative_eq!(n_rack.z, expected_n.z, epsilon = 5e-4);
        assert_relative_eq!(t_rack.x, expected_t.x, epsilon = 5e-4);
        assert_relative_eq!(t_rack.z, expected_t.z, epsilon = 5e-4);
        assert_relative_eq!(t_rack.dot(n_rack), 0.0, epsilon = 1e-6);

        let slots = cfg.rack_slots(16);
        assert_eq!(slots.len(), 16);

        // Shoulder pivot when carriage is at leftmost limit q0 = -1.75
        let shoulder_at_min_q0 = Vec3::new(cfg.rail_q0_min, cfg.shoulder_pivot_y, cfg.rail_z);
        let max_arm_reach = cfg.upper_arm_l2 + cfg.forearm_l3;

        for (idx, slot) in slots.iter().enumerate() {
            assert_eq!(slot.slot_index, idx);
            assert_relative_eq!(slot.approach_dir.x, n_rack.x, epsilon = 1e-6);
            assert_relative_eq!(slot.approach_dir.y, n_rack.y, epsilon = 1e-6);
            assert_relative_eq!(slot.approach_dir.z, n_rack.z, epsilon = 1e-6);

            // grasp_rot local axes match (t_rack, +Y, n_rack)
            let local_x = slot.grasp_rot * Vec3::X;
            let local_y = slot.grasp_rot * Vec3::Y;
            let local_z = slot.grasp_rot * Vec3::Z;
            assert_relative_eq!(local_x.dot(t_rack), 1.0, epsilon = 1e-5);
            assert_relative_eq!(local_y.dot(Vec3::Y), 1.0, epsilon = 1e-5);
            assert_relative_eq!(local_z.dot(n_rack), 1.0, epsilon = 1e-5);

            // Tier height & stadium step verification
            if idx < 8 {
                assert_relative_eq!(slot.grasp_pos.y, 1.24, epsilon = 1e-6);
            } else {
                assert_relative_eq!(slot.grasp_pos.y, 0.90, epsilon = 1e-6);
            }

            // Every slot must be comfortably within L2 + L3 arm reach from q0 = -1.75
            let dist = slot.grasp_pos.distance(shoulder_at_min_q0);
            assert!(
                dist < max_arm_reach,
                "slot {idx} distance {dist:.3}m exceeds L2+L3={max_arm_reach:.3}m"
            );
        }

        // Adjacent slots on the same tier are spaced 0.088m apart along t_rack
        for pair in slots[0..8].windows(2) {
            let delta = pair[1].grasp_pos - pair[0].grasp_pos;
            assert_relative_eq!(delta.length(), 0.088, epsilon = 1e-5);
            assert_relative_eq!(delta.normalize().dot(t_rack), 1.0, epsilon = 1e-5);
        }
        for pair in slots[8..16].windows(2) {
            let delta = pair[1].grasp_pos - pair[0].grasp_pos;
            assert_relative_eq!(delta.length(), 0.088, epsilon = 1e-5);
            assert_relative_eq!(delta.normalize().dot(t_rack), 1.0, epsilon = 1e-5);
        }

        // Stadium step: lower tier is 0.08m further forward along n_rack than upper tier
        let upper_to_lower = slots[8].grasp_pos - slots[0].grasp_pos;
        assert_relative_eq!(upper_to_lower.dot(n_rack), 0.08, epsilon = 1e-5);
    }
}
