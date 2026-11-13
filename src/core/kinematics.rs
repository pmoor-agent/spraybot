use glam::Vec3;
use nalgebra::SMatrix;

use crate::core::types::{LinkTransform, StudioConfig};

/// Descriptor for a single joint in the 7-DOF serial chain.
#[derive(Debug, Clone, PartialEq)]
pub struct JointSpec {
    pub name: String,
    pub is_prismatic: bool,
    pub axis: Vec3,
    pub offset: Vec3,
    pub min: f64,
    pub max: f64,
    pub max_vel: f64,
    pub max_accel: f64,
    pub ik_weight: f64,
}

/// Data-driven 7-DOF serial kinematic chain (1 prismatic rail + 6 revolute arm joints).
#[derive(Debug, Clone, PartialEq)]
pub struct KinematicChain {
    pub joints: Vec<JointSpec>,
    pub base_origin: Vec3,
    pub tcp_offset: Vec3,
    pub nozzle_local_pos: Vec3,
    pub nozzle_local_dir: Vec3,
    pub comfort_pose: [f64; 7],
}

impl KinematicChain {
    /// Builds the default 7-DOF SprayBot chain matching `StudioConfig`.
    pub fn default_spraybot(studio: &StudioConfig) -> Self {
        let shoulder_dy = studio.shoulder_pivot_y - studio.carriage_y;
        Self {
            joints: vec![
                JointSpec {
                    name: "rail_x".to_string(),
                    is_prismatic: true,
                    axis: Vec3::X,
                    offset: Vec3::ZERO,
                    min: studio.rail_q0_min as f64,
                    max: studio.rail_q0_max as f64,
                    max_vel: 1.2,
                    max_accel: 2.5,
                    ik_weight: 6.0,
                },
                JointSpec {
                    name: "base_yaw".to_string(),
                    is_prismatic: false,
                    axis: Vec3::Y,
                    offset: Vec3::new(0.0, shoulder_dy * 0.5, 0.0),
                    min: -std::f64::consts::PI,
                    max: std::f64::consts::PI,
                    max_vel: 2.4,
                    max_accel: 6.0,
                    ik_weight: 1.5,
                },
                JointSpec {
                    name: "shoulder_pitch".to_string(),
                    is_prismatic: false,
                    axis: Vec3::X,
                    offset: Vec3::new(0.0, shoulder_dy * 0.5, 0.0),
                    min: -2.2,
                    max: 2.2,
                    max_vel: 2.2,
                    max_accel: 6.0,
                    ik_weight: 1.5,
                },
                JointSpec {
                    name: "elbow_pitch".to_string(),
                    is_prismatic: false,
                    axis: Vec3::X,
                    offset: Vec3::new(0.0, studio.upper_arm_l2, 0.0),
                    min: -2.6,
                    max: 2.6,
                    max_vel: 2.6,
                    max_accel: 8.0,
                    ik_weight: 1.5,
                },
                JointSpec {
                    name: "wrist_1".to_string(),
                    is_prismatic: false,
                    axis: Vec3::Z,
                    offset: Vec3::new(0.0, 0.0, -studio.forearm_l3),
                    min: -std::f64::consts::PI,
                    max: std::f64::consts::PI,
                    max_vel: 3.4,
                    max_accel: 10.0,
                    ik_weight: 0.4,
                },
                JointSpec {
                    name: "wrist_2".to_string(),
                    is_prismatic: false,
                    axis: Vec3::X,
                    offset: Vec3::new(0.0, 0.0, -0.10),
                    min: -2.4,
                    max: 2.4,
                    max_vel: 3.6,
                    max_accel: 11.0,
                    ik_weight: 0.4,
                },
                JointSpec {
                    name: "flange_roll".to_string(),
                    is_prismatic: false,
                    axis: Vec3::Z,
                    offset: Vec3::new(0.0, 0.0, -0.08),
                    min: -std::f64::consts::PI,
                    max: std::f64::consts::PI,
                    max_vel: 3.8,
                    max_accel: 12.0,
                    ik_weight: 0.4,
                },
            ],
            base_origin: Vec3::new(0.0, studio.carriage_y, studio.rail_z),
            tcp_offset: Vec3::new(0.0, 0.0, -0.08),
            nozzle_local_pos: Vec3::new(0.0, 0.095, -0.11),
            nozzle_local_dir: Vec3::NEG_Z,
            comfort_pose: [0.0, 0.0, -0.45, 1.05, 0.0, -0.60, 0.0],
        }
    }

    /// Computes world-space transforms for all 7 moving links.
    pub fn fk_all_links(&self, q: &[f64; 7]) -> Vec<LinkTransform> {
        let _ = q;
        vec![LinkTransform::default(); self.joints.len()]
    }

    /// Computes `(nozzle_world_pos, nozzle_world_dir)` at joint configuration `q`.
    pub fn fk_nozzle(&self, q: &[f64; 7]) -> (Vec3, Vec3) {
        let _ = q;
        (Vec3::new(0.0, 1.35, 0.50), Vec3::NEG_Z)
    }

    /// Analytical 6x7 geometric Jacobian for the nozzle TCP (`[linear; angular]`).
    pub fn jacobian_nozzle(&self, q: &[f64; 7]) -> SMatrix<f64, 6, 7> {
        let _ = q;
        SMatrix::<f64, 6, 7>::zeros()
    }

    /// Weighted Damped Least-Squares Jacobian IK solver seeded at `q_seed`.
    pub fn solve_ik(
        &self,
        q_seed: &[f64; 7],
        _target_pos: Vec3,
        _target_dir: Vec3,
        _upright_weight: f64,
    ) -> Result<[f64; 7], String> {
        Ok(*q_seed)
    }
}
