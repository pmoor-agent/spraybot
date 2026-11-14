use std::f64::consts::PI;

use glam::{DQuat, DVec3, Quat, Vec3};
use nalgebra::{SMatrix, SVector};

use crate::core::types::{LinkTransform, StudioConfig};

/// Descriptor for a single joint in the 9-DOF serial chain.
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

/// IK objective mode controlling roll constraint around the tool/spray axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IkMode {
    /// Painting mode: exact nozzle position and spray direction alignment,
    /// with a soft roll bias keeping can `+Y` near World `+Y` (within `±35°`).
    Spray { roll_weight: f64 },
    /// Exact 6-DOF pose alignment (used for rack pick/place).
    Exact6Dof,
}

impl Default for IkMode {
    fn default() -> Self {
        Self::Spray { roll_weight: 0.15 }
    }
}

/// Data-driven 9-DOF serial kinematic chain (1 prismatic rail + 8 revolute arm joints).
#[derive(Debug, Clone, PartialEq)]
pub struct KinematicChain {
    pub joints: Vec<JointSpec>,
    pub base_origin: Vec3,
    pub tcp_offset: Vec3,
    pub nozzle_local_pos: Vec3,
    pub nozzle_local_dir: Vec3,
    pub comfort_pose: [f64; 9],
}

impl KinematicChain {
    /// Builds the default 9-DOF SprayBot chain matching [`StudioConfig`].
    ///
    /// Link index meanings for [`Self::fk_links`]:
    /// - `0`: `rail_x` carriage at `(q0, carriage_y, rail_z)`, identity rotation.
    /// - `1`: `base_yaw` rotating riser/turret centered at mid-riser height, rotated by `q1` around `+Y`.
    /// - `2`: `shoulder_pitch` lower upper-arm root at `(q0, shoulder_pivot_y, rail_z)` (`+Y` extends `0.5*L2`).
    /// - `3`: `upper_arm_roll` upper-arm roll joint around `+Y` (`+Y` extends `0.5*L2` to elbow).
    /// - `4`: `elbow_pitch` forearm root (`-Z` extends `L3` to `wrist_1`).
    /// - `5`: `wrist_1` roll housing (`-Z` extends `0.065m` to `wrist_2`).
    /// - `6`: `wrist_2` pitch clevis (`-Z` extends `0.030m` to `wrist_yaw`).
    /// - `7`: `wrist_yaw` yaw clevis (`-Z` extends `0.020m` to `flange_roll`).
    /// - `8`: `flange_roll` tool flange (`tcp_offset = (0, 0, -0.025)` reaches grasp center; total wrist-to-grasp `0.14m`).
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
                    ik_weight: 30.0,
                },
                JointSpec {
                    name: "base_yaw".to_string(),
                    is_prismatic: false,
                    axis: Vec3::Y,
                    offset: Vec3::new(0.0, shoulder_dy * 0.5, 0.0),
                    min: -PI,
                    max: PI,
                    max_vel: 2.4,
                    max_accel: 6.0,
                    ik_weight: 1.0,
                },
                JointSpec {
                    name: "shoulder_pitch".to_string(),
                    is_prismatic: false,
                    axis: Vec3::X,
                    offset: Vec3::new(0.0, shoulder_dy * 0.5, 0.0),
                    min: -2.4,
                    max: 2.4,
                    max_vel: 2.2,
                    max_accel: 6.0,
                    ik_weight: 1.6,
                },
                JointSpec {
                    name: "upper_arm_roll".to_string(),
                    is_prismatic: false,
                    axis: Vec3::Y,
                    offset: Vec3::new(0.0, studio.upper_arm_l2 * 0.5, 0.0),
                    min: -PI,
                    max: PI,
                    max_vel: 2.8,
                    max_accel: 8.0,
                    ik_weight: 1.4,
                },
                JointSpec {
                    name: "elbow_pitch".to_string(),
                    is_prismatic: false,
                    axis: Vec3::X,
                    offset: Vec3::new(0.0, studio.upper_arm_l2 * 0.5, 0.0),
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
                    min: -PI,
                    max: PI,
                    max_vel: 3.6,
                    max_accel: 10.0,
                    ik_weight: 0.45,
                },
                JointSpec {
                    name: "wrist_2".to_string(),
                    is_prismatic: false,
                    axis: Vec3::X,
                    offset: Vec3::new(0.0, 0.0, -0.065),
                    min: -2.3,
                    max: 2.3,
                    max_vel: 3.6,
                    max_accel: 10.0,
                    ik_weight: 0.35,
                },
                JointSpec {
                    name: "wrist_yaw".to_string(),
                    is_prismatic: false,
                    axis: Vec3::Y,
                    offset: Vec3::new(0.0, 0.0, -0.030),
                    min: -2.0,
                    max: 2.0,
                    max_vel: 3.8,
                    max_accel: 10.0,
                    ik_weight: 0.35,
                },
                JointSpec {
                    name: "flange_roll".to_string(),
                    is_prismatic: false,
                    axis: Vec3::Z,
                    offset: Vec3::new(0.0, 0.0, -0.020),
                    min: -2.0 * PI,
                    max: 2.0 * PI,
                    max_vel: 3.8,
                    max_accel: 10.0,
                    ik_weight: 0.40,
                },
            ],
            base_origin: Vec3::new(0.0, studio.carriage_y, studio.rail_z),
            tcp_offset: Vec3::new(0.0, 0.0, -0.025),
            nozzle_local_pos: Vec3::new(0.0, 0.085, -0.015),
            nozzle_local_dir: Vec3::NEG_Z,
            comfort_pose: [0.0, 0.0, -0.45, 0.0, 1.05, 0.0, -0.60, 0.0, 0.0],
        }
    }

    /// Clamps a joint vector strictly within `[joint.min, joint.max]`.
    pub fn clamp_joints(&self, q: &[f64; 9]) -> [f64; 9] {
        let mut out = *q;
        for (i, j) in self.joints.iter().enumerate().take(9) {
            out[i] = out[i].clamp(j.min, j.max);
        }
        out
    }

    /// Returns `true` if every joint lies within `[min - tol, max + tol]`.
    pub fn within_limits(&self, q: &[f64; 9], tol: f64) -> bool {
        self.joints
            .iter()
            .enumerate()
            .take(9)
            .all(|(i, j)| q[i] >= j.min - tol && q[i] <= j.max + tol)
    }

    /// Internal f64 forward kinematics returning per-joint `(pos, rot, world_axis)` plus
    /// the grasp frame `(grasp_pos, grasp_rot)`.
    fn fk_internal(&self, q: &[f64; 9]) -> ([(DVec3, DQuat, DVec3); 9], DVec3, DQuat) {
        let mut p = self.base_origin.as_dvec3();
        let mut r = DQuat::IDENTITY;
        let mut frames = [(DVec3::ZERO, DQuat::IDENTITY, DVec3::X); 9];

        for i in 0..9 {
            let j = &self.joints[i];
            p += r * j.offset.as_dvec3();
            let axis_local = j.axis.as_dvec3().normalize();
            let axis_world = (r * axis_local).normalize();
            if j.is_prismatic {
                p += axis_world * q[i];
            } else {
                r = (r * DQuat::from_axis_angle(axis_local, q[i])).normalize();
            }
            frames[i] = (p, r, axis_world);
        }

        let grasp_pos = p + r * self.tcp_offset.as_dvec3();
        let grasp_rot = r;
        (frames, grasp_pos, grasp_rot)
    }

    /// Computes world-space transforms for all 9 moving links.
    ///
    /// Index mapping:
    /// - `0`: `rail_x` carriage
    /// - `1`: `base_yaw` rotating riser
    /// - `2`: `shoulder_pitch` lower upper arm
    /// - `3`: `upper_arm_roll` upper-arm roll joint
    /// - `4`: `elbow_pitch` forearm
    /// - `5`: `wrist_1` roll housing
    /// - `6`: `wrist_2` pitch clevis
    /// - `7`: `wrist_yaw` yaw clevis
    /// - `8`: `flange_roll` tool flange
    pub fn fk_links(&self, q: &[f64; 9]) -> Vec<LinkTransform> {
        let (frames, _, _) = self.fk_internal(q);
        frames
            .iter()
            .map(|(pos, rot, _)| LinkTransform {
                pos: pos.as_vec3(),
                rot: rot.as_quat(),
            })
            .collect()
    }

    /// Alias for [`Self::fk_links`] matching the initial Phase 1 stub name.
    #[inline]
    pub fn fk_all_links(&self, q: &[f64; 9]) -> Vec<LinkTransform> {
        self.fk_links(q)
    }

    /// Computes world-space transform of the gripper grasp center.
    pub fn fk_grasp(&self, q: &[f64; 9]) -> LinkTransform {
        let (_, grasp_pos, grasp_rot) = self.fk_internal(q);
        LinkTransform {
            pos: grasp_pos.as_vec3(),
            rot: grasp_rot.as_quat(),
        }
    }

    /// Computes `(nozzle_world_pos, spray_world_dir, can_up_world)` at joint configuration `q`.
    pub fn fk_nozzle(&self, q: &[f64; 9]) -> (Vec3, Vec3, Vec3) {
        let (_, grasp_pos, grasp_rot) = self.fk_internal(q);
        let nozzle_pos = grasp_pos + grasp_rot * self.nozzle_local_pos.as_dvec3();
        let spray_dir = (grasp_rot * self.nozzle_local_dir.as_dvec3()).normalize();
        let can_up = (grasp_rot * DVec3::Y).normalize();
        (
            nozzle_pos.as_vec3(),
            spray_dir.as_vec3(),
            can_up.as_vec3(),
        )
    }

    /// Analytical 6x9 geometric Jacobian for a point `p_ee` rigidly attached to the tool flange.
    fn jacobian_for_point(&self, frames: &[(DVec3, DQuat, DVec3); 9], p_ee: DVec3) -> SMatrix<f64, 6, 9> {
        let mut jac = SMatrix::<f64, 6, 9>::zeros();
        for i in 0..9 {
            let (p_i, _, a_i) = frames[i];
            if self.joints[i].is_prismatic {
                jac[(0, i)] = a_i.x;
                jac[(1, i)] = a_i.y;
                jac[(2, i)] = a_i.z;
            } else {
                let v = a_i.cross(p_ee - p_i);
                jac[(0, i)] = v.x;
                jac[(1, i)] = v.y;
                jac[(2, i)] = v.z;
                jac[(3, i)] = a_i.x;
                jac[(4, i)] = a_i.y;
                jac[(5, i)] = a_i.z;
            }
        }
        jac
    }

    /// Analytical 6x9 geometric Jacobian for the nozzle TCP (`[linear; angular]`).
    pub fn jacobian_nozzle(&self, q: &[f64; 9]) -> SMatrix<f64, 6, 9> {
        let (frames, grasp_pos, grasp_rot) = self.fk_internal(q);
        let p_nozzle = grasp_pos + grasp_rot * self.nozzle_local_pos.as_dvec3();
        self.jacobian_for_point(&frames, p_nozzle)
    }

    /// Analytical 6x9 geometric Jacobian for the gripper grasp center (`[linear; angular]`).
    pub fn jacobian_grasp(&self, q: &[f64; 9]) -> SMatrix<f64, 6, 9> {
        let (frames, grasp_pos, _) = self.fk_internal(q);
        self.jacobian_for_point(&frames, grasp_pos)
    }

    /// Computes an analytical elbow-up seed for reaching a desired grasp frame `(grasp_pos, grasp_dir)`.
    pub fn smart_seed_for_grasp(&self, grasp_pos: Vec3, grasp_dir: Vec3) -> [f64; 9] {
        self.smart_seed_for_grasp_with_q0_deadband(grasp_pos, grasp_dir, None)
    }

    /// Analytical elbow-up seed for `grasp_pos`, optionally biasing `q0` near `q0_seed` within
    /// `[p_w.x - dx_deadband, p_w.x + dx_deadband]` instead of snapping straight to `p_w.x`.
    fn smart_seed_for_grasp_with_q0_deadband(
        &self,
        grasp_pos: Vec3,
        grasp_dir: Vec3,
        seed_q0_deadband: Option<(f64, f64)>,
    ) -> [f64; 9] {
        let l2 = (self.joints[3].offset.y + self.joints[4].offset.y) as f64;
        let l3 = -self.joints[5].offset.z as f64;
        let d_wrist_j6 = -self.joints[6].offset.z as f64;
        let d_j6_grasp =
            -(self.joints[7].offset.z + self.joints[8].offset.z + self.tcp_offset.z) as f64;
        let l3_eff = l3 + d_wrist_j6;

        let g_pos = grasp_pos.as_dvec3();
        let g_dir = grasp_dir.as_dvec3().normalize();
        // Estimate wrist_2 (J6) center back along -grasp_dir from grasp center
        let p_w = g_pos - g_dir * d_j6_grasp;

        let q0 = match seed_q0_deadband {
            Some((q0_seed, dx_deadband)) => q0_seed
                .clamp(p_w.x - dx_deadband, p_w.x + dx_deadband)
                .clamp(self.joints[0].min, self.joints[0].max),
            None => p_w.x.clamp(self.joints[0].min, self.joints[0].max),
        };
        let shoulder = DVec3::new(
            q0,
            (self.base_origin.y + self.joints[1].offset.y + self.joints[2].offset.y) as f64,
            self.base_origin.z as f64,
        );

        let dx = p_w.x - shoulder.x;
        let dy = p_w.y - shoulder.y;
        let dz = p_w.z - shoulder.z;
        let r_xz = (dx * dx + dz * dz).sqrt();

        let q1 = if r_xz > 1e-6 {
            (-dx).atan2(-dz).clamp(self.joints[1].min, self.joints[1].max)
        } else {
            0.0
        };

        let min_r = (l2 - l3_eff).abs() + 1e-4;
        let max_r = l2 + l3_eff - 1e-4;
        let r_sq = (r_xz * r_xz + dy * dy).clamp(min_r * min_r, max_r * max_r);

        let sin_q4 = ((r_sq - l2 * l2 - l3_eff * l3_eff) / (2.0 * l2 * l3_eff)).clamp(-1.0, 1.0);
        // Elbow-up branch: q4 in [-pi/2, +pi/2]
        let q4 = sin_q4.asin().clamp(self.joints[4].min, self.joints[4].max);

        let a = l2 + l3_eff * q4.sin();
        let b = l3_eff * q4.cos();
        let q2 = (b * dy - a * r_xz)
            .atan2(a * dy + b * r_xz)
            .clamp(self.joints[2].min, self.joints[2].max);

        let target_pitch = g_dir.y.clamp(-1.0, 1.0).asin();
        let q6 = (-(q2 + q4) + target_pitch).clamp(self.joints[6].min, self.joints[6].max);

        // Horizontal yaw angle between Ry(q1)'s -Z and g_dir
        let (sin_q1, cos_q1) = q1.sin_cos();
        let fwd_base = DVec3::new(-sin_q1, 0.0, -cos_q1);
        let right_base = DVec3::new(cos_q1, 0.0, -sin_q1);
        let g_horiz = DVec3::new(g_dir.x, 0.0, g_dir.z);
        let q7 = if g_horiz.length_squared() > 1e-8 {
            (-g_horiz.dot(right_base))
                .atan2(g_horiz.dot(fwd_base))
                .clamp(self.joints[7].min, self.joints[7].max)
        } else {
            0.0
        };

        self.clamp_joints(&[q0, q1, q2, 0.0, q4, 0.0, q6, q7, 0.0])
    }

    /// Computes an analytical elbow-up seed for reaching a desired nozzle target `(nozzle_pos, spray_dir)`.
    pub fn smart_seed_for_nozzle(&self, target_pos: Vec3, target_dir: Vec3) -> [f64; 9] {
        let dir = target_dir.as_dvec3().normalize();
        // Construct upright grasp rotation whose -Z aligns with target_dir
        let rot = upright_rot_from_neg_z(dir, DVec3::Y);
        let grasp_pos = target_pos.as_dvec3() - rot * self.nozzle_local_pos.as_dvec3();
        self.smart_seed_for_grasp(grasp_pos.as_vec3(), target_dir)
    }

    /// Computes an analytical elbow-up nozzle seed while keeping `q0` near `q_seed[0]` inside
    /// the comfortable lateral deadband around the wrist target.
    fn smart_seed_for_nozzle_from_seed(
        &self,
        q_seed: &[f64; 9],
        target_pos: Vec3,
        target_dir: Vec3,
    ) -> [f64; 9] {
        let dir = target_dir.as_dvec3().normalize();
        let rot = upright_rot_from_neg_z(dir, DVec3::Y);
        let grasp_pos = target_pos.as_dvec3() - rot * self.nozzle_local_pos.as_dvec3();
        let dx_seed_deadband = self.comfort_dx_nozzle(target_pos.as_dvec3()) * 0.70;
        self.smart_seed_for_grasp_with_q0_deadband(
            grasp_pos.as_vec3(),
            target_dir,
            Some((q_seed[0], dx_seed_deadband)),
        )
    }

    /// Comfortable lateral reach `dx_comfort` (meters) from the shoulder to a nozzle target
    /// at `(target_pos.y, target_pos.z)`, wide at mid-canvas and tapering near high corners.
    fn comfort_dx_nozzle(&self, target_pos: DVec3) -> f64 {
        let l2 = (self.joints[3].offset.y + self.joints[4].offset.y) as f64;
        let l3 = -self.joints[5].offset.z as f64;
        let d_wrist_j6 = -self.joints[6].offset.z as f64;
        let d_j6_grasp =
            -(self.joints[7].offset.z + self.joints[8].offset.z + self.tcp_offset.z) as f64;
        let l3_eff = l3 + d_wrist_j6;
        let shoulder_y =
            (self.base_origin.y + self.joints[1].offset.y + self.joints[2].offset.y) as f64;
        let shoulder_z = self.base_origin.z as f64;

        let y_w = target_pos.y - self.nozzle_local_pos.y as f64;
        let z_w = target_pos.z - self.nozzle_local_pos.z as f64 + d_j6_grasp;
        let dy = y_w - shoulder_y;
        let dz = z_w - shoulder_z;

        let r_comfort = 0.90 * (l2 + l3_eff);
        let rad_sq = (r_comfort * r_comfort - (dy * dy + dz * dz)).max(0.0);
        rad_sq.sqrt().clamp(0.06, 0.44)
    }

    /// Core Weighted Damped Least-Squares iteration loop for nozzle or grasp targets.
    fn solve_dls_internal(
        &self,
        q_seed: &[f64; 9],
        target_pos: DVec3,
        target_dir: DVec3,
        target_up: DVec3,
        target_rot_exact: Option<DQuat>,
        use_nozzle_tcp: bool,
        mode: IkMode,
        max_iters: usize,
    ) -> ([f64; 9], f64, f64) {
        let mut q = self.clamp_joints(q_seed);
        let d_t = target_dir.normalize();

        // Diagonal joint weights W and inverse W^{-1}
        let mut w_inv = SMatrix::<f64, 9, 9>::zeros();
        for i in 0..9 {
            let w = if i == 0 && !use_nozzle_tcp {
                // Keep rail agile during 6-DOF rack/cap-station grasps
                2.5
            } else {
                self.joints[i].ik_weight
            }
            .max(1e-6);
            w_inv[(i, i)] = 1.0 / w;
        }

        let lambda = 0.025;
        let alpha_null = 0.06;

        for _ in 0..max_iters {
            let (frames, grasp_pos, grasp_rot) = self.fk_internal(&q);
            let (p_cur, d_cur, jac) = if use_nozzle_tcp {
                let p_noz = grasp_pos + grasp_rot * self.nozzle_local_pos.as_dvec3();
                let d_noz = (grasp_rot * self.nozzle_local_dir.as_dvec3()).normalize();
                let j = self.jacobian_for_point(&frames, p_noz);
                (p_noz, d_noz, j)
            } else {
                let d_grasp = (grasp_rot * self.nozzle_local_dir.as_dvec3()).normalize();
                let j = self.jacobian_for_point(&frames, grasp_pos);
                (grasp_pos, d_grasp, j)
            };

            let e_pos = target_pos - p_cur;

            // Compute angular error and task-space weighting S in world frame
            let (e_ang, s_ang) = match (mode, target_rot_exact) {
                (IkMode::Exact6Dof, Some(q_t)) => {
                    let mut q_err = q_t * grasp_rot.conjugate();
                    if q_err.w < 0.0 {
                        q_err = -q_err;
                    }
                    let (axis, angle) = q_err.to_axis_angle();
                    (axis * angle, nalgebra::Matrix3::<f64>::identity())
                }
                (IkMode::Exact6Dof, None) => {
                    let q_t = upright_rot_from_neg_z(d_t, target_up);
                    let mut q_err = q_t * grasp_rot.conjugate();
                    if q_err.w < 0.0 {
                        q_err = -q_err;
                    }
                    let (axis, angle) = q_err.to_axis_angle();
                    (axis * angle, nalgebra::Matrix3::<f64>::identity())
                }
                (IkMode::Spray { roll_weight }, _) => {
                    // Pointing error perpendicular to d_cur
                    let cross = d_cur.cross(d_t);
                    let cross_len = cross.length();
                    let dot = d_cur.dot(d_t).clamp(-1.0, 1.0);
                    let e_point = if cross_len > 1e-9 {
                        let angle = cross_len.atan2(dot);
                        cross * (angle / cross_len)
                    } else if dot < 0.0 {
                        let ortho = if d_cur.x.abs() < 0.9 {
                            DVec3::X
                        } else {
                            DVec3::Y
                        };
                        d_cur.cross(ortho).normalize() * PI
                    } else {
                        DVec3::ZERO
                    };

                    // Roll error around d_t to keep can +Y near target_up
                    let u_proj = target_up - d_t * target_up.dot(d_t);
                    let e_roll = if u_proj.length_squared() > 1e-8 {
                        let u_des = u_proj.normalize();
                        let q_align = DQuat::from_rotation_arc(d_cur, d_t);
                        let u_cur = (q_align * (grasp_rot * DVec3::Y)).normalize();
                        let theta_roll = u_cur.cross(u_des).dot(d_t).atan2(u_cur.dot(u_des));
                        d_t * theta_roll
                    } else {
                        DVec3::ZERO
                    };

                    // Project S onto pointing (weight 1.0) + roll along d_t (roll_weight)
                    let rw = roll_weight.clamp(0.0, 1.0);
                    let d_vec = nalgebra::Vector3::new(d_t.x, d_t.y, d_t.z);
                    let proj_d = d_vec * d_vec.transpose();
                    let s_mat = nalgebra::Matrix3::identity() * 1.0 - proj_d * (1.0 - rw);

                    (e_point + e_roll * rw, s_mat)
                }
            };

            // Check convergence
            let pos_err = e_pos.length();
            let point_err = d_cur.dot(d_t).clamp(-1.0, 1.0).acos();
            let rot_err = match (mode, target_rot_exact) {
                (IkMode::Exact6Dof, Some(q_t)) => {
                    let mut q_err = q_t * grasp_rot.conjugate();
                    if q_err.w < 0.0 {
                        q_err = -q_err;
                    }
                    q_err.to_axis_angle().1
                }
                _ => point_err,
            };
            if pos_err < 1e-6 && rot_err < 1e-5 {
                return (q, pos_err, rot_err);
            }

            // Task-weighted Jacobian J_s = S * J and error e_s
            let mut j_s = jac;
            for r in 0..3 {
                for c in 0..9 {
                    j_s[(3 + r, c)] = s_ang[(r, 0)] * jac[(3, c)]
                        + s_ang[(r, 1)] * jac[(4, c)]
                        + s_ang[(r, 2)] * jac[(5, c)];
                }
            }

            let e_vec = SVector::<f64, 6>::new(
                e_pos.x, e_pos.y, e_pos.z, e_ang.x, e_ang.y, e_ang.z,
            );

            // A = J_s * W^{-1} * J_s^T + lambda^2 * I_6
            let w_inv_jt = w_inv * j_s.transpose();
            let mut a_mat = j_s * w_inv_jt;
            for k in 0..6 {
                a_mat[(k, k)] += lambda * lambda;
            }

            let dq_task = if let Some(chol) = a_mat.cholesky() {
                let y = chol.solve(&e_vec);
                let dq_t = w_inv_jt * y;

                let mut v_null = SVector::<f64, 9>::zeros();
                for i in 0..9 {
                    let j_spec = &self.joints[i];
                    let comfort_pull = if i == 0 {
                        if use_nozzle_tcp {
                            let dx_comfort = self.comfort_dx_nozzle(target_pos);
                            let low = (target_pos.x - dx_comfort).clamp(j_spec.min, j_spec.max);
                            let high = (target_pos.x + dx_comfort).clamp(j_spec.min, j_spec.max);
                            if q[0] < low {
                                let err = low - q[0];
                                0.55 * err * (err / 0.04).min(1.0)
                            } else if q[0] > high {
                                let err = q[0] - high;
                                -0.55 * err * (err / 0.04).min(1.0)
                            } else {
                                0.0
                            }
                        } else {
                            let target_x = target_pos.x.clamp(j_spec.min, j_spec.max);
                            0.28 * (target_x - q[0])
                        }
                    } else {
                        alpha_null * (self.comfort_pose[i] - q[i])
                    };
                    // Soft joint-limit repulsion near endstops (tighter 4% margin on linear rail)
                    let span = (j_spec.max - j_spec.min).max(1e-6);
                    let margin = if j_spec.is_prismatic {
                        0.04 * span
                    } else {
                        0.12 * span
                    };
                    let d_low = q[i] - j_spec.min;
                    let d_high = j_spec.max - q[i];
                    let mut repulse = 0.0;
                    if d_low < margin {
                        repulse += 0.35 * (1.0 - d_low / margin);
                    }
                    if d_high < margin {
                        repulse -= 0.35 * (1.0 - d_high / margin);
                    }
                    v_null[i] = comfort_pull + repulse;
                }
                let mut dq_null = if use_nozzle_tcp {
                    // Project arm comfort pulls into the 8-DOF arm-only null space so they never
                    // drag rail_x inside the deadband, and engage rail_x smoothly outside it.
                    let mut v_arm = v_null;
                    v_arm[0] = 0.0;
                    let mut w_inv_arm = w_inv;
                    w_inv_arm[(0, 0)] = 0.0;
                    let w_inv_arm_jt = w_inv_arm * j_s.transpose();
                    let mut a_arm = j_s * w_inv_arm_jt;
                    for k in 0..6 {
                        a_arm[(k, k)] += lambda * lambda;
                    }
                    let mut dq_arm = if let Some(chol_arm) = a_arm.cholesky() {
                        v_arm - w_inv_arm_jt * chol_arm.solve(&(j_s * v_arm))
                    } else {
                        SVector::<f64, 9>::zeros()
                    };
                    if v_null[0].abs() > 1e-12 {
                        let mut v_rail = SVector::<f64, 9>::zeros();
                        v_rail[0] = v_null[0];
                        dq_arm += v_rail - w_inv_jt * chol.solve(&(j_s * v_rail));
                    }
                    dq_arm
                } else {
                    v_null - w_inv_jt * chol.solve(&(j_s * v_null))
                };
                let max_null = dq_null.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                if max_null > 0.04 {
                    dq_null *= 0.04 / max_null;
                }
                let null_atten = (pos_err.max(rot_err * 0.15) / 0.002).clamp(0.0, 1.0);
                dq_t + dq_null * null_atten
            } else {
                SVector::<f64, 9>::zeros()
            };

            // Step clamp for stability on large errors
            let max_step = dq_task.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let scale = if max_step > 0.35 { 0.35 / max_step } else { 1.0 };

            for i in 0..9 {
                q[i] += dq_task[i] * scale;
            }
            q = self.clamp_joints(&q);
        }

        // Final error check
        let (_, grasp_pos, grasp_rot) = self.fk_internal(&q);
        let (p_cur, d_cur) = if use_nozzle_tcp {
            (
                grasp_pos + grasp_rot * self.nozzle_local_pos.as_dvec3(),
                (grasp_rot * self.nozzle_local_dir.as_dvec3()).normalize(),
            )
        } else {
            (
                grasp_pos,
                (grasp_rot * self.nozzle_local_dir.as_dvec3()).normalize(),
            )
        };
        let pos_err = (target_pos - p_cur).length();
        let rot_err = match (mode, target_rot_exact) {
            (IkMode::Exact6Dof, Some(q_t)) => {
                let mut q_err = q_t * grasp_rot.conjugate();
                if q_err.w < 0.0 {
                    q_err = -q_err;
                }
                q_err.to_axis_angle().1
            }
            _ => d_cur.dot(d_t).clamp(-1.0, 1.0).acos(),
        };
        (q, pos_err, rot_err)
    }

    /// Weighted Damped Least-Squares Jacobian IK solver for a nozzle pose `(target_pos, target_dir)`.
    ///
    /// First runs from `q_seed` (preserving `C¹` continuity during trajectory tracking). If the seed
    /// fails to converge within tight tolerances (e.g., large initial jump from home to rack/canvas),
    /// automatically retries from [`Self::smart_seed_for_nozzle`].
    pub fn solve_ik(
        &self,
        q_seed: &[f64; 9],
        target_pos: Vec3,
        target_dir: Vec3,
        mode: IkMode,
    ) -> Result<[f64; 9], String> {
        self.solve_ik_with_up(q_seed, target_pos, target_dir, Vec3::Y, mode)
    }

    /// Nozzle IK with explicit desired can-up vector (`target_up`, usually `Vec3::Y`).
    pub fn solve_ik_with_up(
        &self,
        q_seed: &[f64; 9],
        target_pos: Vec3,
        target_dir: Vec3,
        target_up: Vec3,
        mode: IkMode,
    ) -> Result<[f64; 9], String> {
        let p_t = target_pos.as_dvec3();
        let d_t = target_dir.as_dvec3().normalize();
        let u_t = target_up.as_dvec3().normalize();

        let seed_noz_dist = (self.fk_nozzle(q_seed).0 - target_pos).length();
        let (q1, pos_err1, rot_err1) =
            self.solve_dls_internal(q_seed, p_t, d_t, u_t, None, true, mode, 16);
        let dx_reach = self.comfort_dx_nozzle(p_t) + 0.03;
        let q1_clean = seed_noz_dist < 0.30
            || ((q1[0] - target_pos.x as f64).abs() <= dx_reach
                && q1[1].abs() <= 1.35
                && q1[3].abs() <= 1.40
                && q1[4].abs() <= 1.65
                && q1[7].abs() <= 1.35
                && self.min_world_z(&q1, true) > 0.012);
        if q1_clean && pos_err1 <= 5e-4 && rot_err1 <= 0.2_f64.to_radians() {
            return Ok(q1);
        }

        // Fallback from analytical elbow-up seed preserving carriage position inside deadband
        let smart = self.smart_seed_for_nozzle_from_seed(q_seed, target_pos, target_dir);
        let (q2, pos_err2, rot_err2) =
            self.solve_dls_internal(&smart, p_t, d_t, u_t, None, true, mode, 24);
        if pos_err2 <= 5e-4 && rot_err2 <= 0.2_f64.to_radians() {
            return Ok(q2);
        }

        // Pick whichever got closer if still slightly above strict threshold, or error
        let (best_q, best_pos, best_rot) = if pos_err2 < pos_err1 {
            (q2, pos_err2, rot_err2)
        } else {
            (q1, pos_err1, rot_err1)
        };
        if best_pos <= 1.5e-3 && best_rot <= 0.5_f64.to_radians() {
            return Ok(best_q);
        }
        Err(format!(
            "Nozzle IK did not converge for target {target_pos:?}: pos_err={best_pos:.6}m, rot_err={:.4}deg",
            best_rot.to_degrees()
        ))
    }

    /// Continuity-preserving nozzle IK solver for dense trajectory tracking.
    ///
    /// Runs `solve_dls_internal` from `q_seed` (`max_iters = 36`) and returns `Ok((q1, false))`
    /// only when tolerances and per-step joint continuity (`max_dof |dq| <= 0.22`) hold.
    /// Otherwise falls back to [`Self::smart_seed_for_nozzle`] (`32` iters) and returns
    /// `Ok((q2, true))` so callers can safely retract and reconfigure with the valve off.
    pub fn solve_ik_continuous(
        &self,
        q_seed: &[f64; 9],
        target_pos: Vec3,
        target_dir: Vec3,
        mode: IkMode,
    ) -> Result<([f64; 9], bool), String> {
        let p_t = target_pos.as_dvec3();
        let d_t = target_dir.as_dvec3().normalize();
        let u_t = DVec3::Y;

        let (q1, pos_err1, rot_err1) =
            self.solve_dls_internal(q_seed, p_t, d_t, u_t, None, true, mode, 36);
        let max_dq1 = (0..9)
            .map(|dof| (q1[dof] - q_seed[dof]).abs())
            .fold(0.0_f64, f64::max);
        if pos_err1 <= 1.0e-3 && rot_err1 <= 0.35_f64.to_radians() && max_dq1 <= 0.22 {
            return Ok((q1, false));
        }

        let smart = self.smart_seed_for_nozzle_from_seed(q_seed, target_pos, target_dir);
        let (q2, pos_err2, rot_err2) =
            self.solve_dls_internal(&smart, p_t, d_t, u_t, None, true, mode, 32);
        if pos_err2 <= 1.0e-3 && rot_err2 <= 0.35_f64.to_radians() {
            return Ok((q2, true));
        }

        let (best_q, best_pos, best_rot, reconfigured) = if pos_err2 < pos_err1 {
            (q2, pos_err2, rot_err2, true)
        } else {
            (q1, pos_err1, rot_err1, max_dq1 > 0.22)
        };
        if best_pos <= 1.5e-3 && best_rot <= 0.5_f64.to_radians() {
            return Ok((best_q, reconfigured));
        }
        Err(format!(
            "Continuous nozzle IK did not converge for target {target_pos:?}: pos_err={best_pos:.6}m, rot_err={:.4}deg",
            best_rot.to_degrees()
        ))
    }

    /// Computes a conservative lower bound on world-space `Z` (meters) across all robot links,
    /// arm midpoints, gripper jaw corners, trigger servo top, and (when `held_can` is true)
    /// the aerosol can cylinder.
    pub fn min_world_z(&self, q: &[f64; 9], held_can: bool) -> f32 {
        const LINK_RADII: [f64; 9] = [0.16, 0.13, 0.08, 0.07, 0.07, 0.05, 0.05, 0.05, 0.05];
        let (frames, grasp_pos, grasp_rot) = self.fk_internal(q);
        let mut min_z = f64::INFINITY;

        for i in 0..9 {
            min_z = min_z.min(frames[i].0.z - LINK_RADII[i]);
        }

        let upper_mid_a = 0.5 * (frames[2].0 + frames[3].0);
        let upper_mid_b = 0.5 * (frames[3].0 + frames[4].0);
        let forearm_mid = 0.5 * (frames[4].0 + frames[5].0);
        min_z = min_z.min(upper_mid_a.z - 0.07);
        min_z = min_z.min(upper_mid_b.z - 0.07);
        min_z = min_z.min(forearm_mid.z - 0.07);

        for sx in [-1.0_f64, 1.0_f64] {
            for sy in [-1.0_f64, 1.0_f64] {
                let corner = grasp_pos + grasp_rot * DVec3::new(sx * 0.058, sy * 0.022, -0.022);
                min_z = min_z.min(corner.z);
            }
        }

        let trigger_top = grasp_pos + grasp_rot * DVec3::new(0.0, 0.13, 0.01);
        min_z = min_z.min(trigger_top.z);

        if held_can {
            let can_radius = 0.034_f64;
            let top = grasp_pos + grasp_rot * DVec3::new(0.0, 0.105, 0.0);
            let bottom = grasp_pos + grasp_rot * DVec3::new(0.0, -0.085, 0.0);
            min_z = min_z.min(top.z - can_radius);
            min_z = min_z.min(bottom.z - can_radius);
        }

        min_z as f32
    }

    /// Weighted Damped Least-Squares 6-DOF IK solver targeting the gripper grasp frame `(target_pos, target_rot)`.
    pub fn solve_ik_grasp(
        &self,
        q_seed: &[f64; 9],
        target_pos: Vec3,
        target_rot: Quat,
    ) -> Result<[f64; 9], String> {
        let p_t = target_pos.as_dvec3();
        let q_t = target_rot.as_dquat().normalize();
        let d_t = (q_t * self.nozzle_local_dir.as_dvec3()).normalize();
        let u_t = (q_t * DVec3::Y).normalize();

        let (q1, pos_err1, rot_err1) = self.solve_dls_internal(
            q_seed,
            p_t,
            d_t,
            u_t,
            Some(q_t),
            false,
            IkMode::Exact6Dof,
            20,
        );
        if pos_err1 <= 5e-4 && rot_err1 <= 0.2_f64.to_radians() {
            return Ok(q1);
        }

        let smart = self.smart_seed_for_grasp(target_pos, d_t.as_vec3());
        let (q2, pos_err2, rot_err2) = self.solve_dls_internal(
            &smart,
            p_t,
            d_t,
            u_t,
            Some(q_t),
            false,
            IkMode::Exact6Dof,
            28,
        );
        if pos_err2 <= 5e-4 && rot_err2 <= 0.2_f64.to_radians() {
            return Ok(q2);
        }

        let (best_q, best_pos, best_rot) = if pos_err2 < pos_err1 {
            (q2, pos_err2, rot_err2)
        } else {
            (q1, pos_err1, rot_err1)
        };
        if best_pos <= 1.5e-3 && best_rot <= 0.5_f64.to_radians() {
            return Ok(best_q);
        }
        Err(format!(
            "Grasp IK did not converge for target {target_pos:?}: pos_err={best_pos:.6}m, rot_err={:.4}deg",
            best_rot.to_degrees()
        ))
    }
}

/// Constructs an upright right-handed rotation whose local `-Z` points along `neg_z_dir`
/// and whose local `+Y` is as close as possible to `up_hint`.
fn upright_rot_from_neg_z(neg_z_dir: DVec3, up_hint: DVec3) -> DQuat {
    let z_axis = (-neg_z_dir).normalize();
    let up = up_hint.normalize();
    let mut x_axis = up.cross(z_axis);
    if x_axis.length_squared() < 1e-8 {
        x_axis = DVec3::X;
    } else {
        x_axis = x_axis.normalize();
    }
    let y_axis = z_axis.cross(x_axis).normalize();
    DQuat::from_mat3(&glam::DMat3::from_cols(x_axis, y_axis, z_axis)).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytical_jacobian_matches_finite_difference() {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);

        let poses: [[f64; 9]; 5] = [
            chain.comfort_pose,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.85, 0.35, -0.60, 0.30, 0.85, 0.25, -0.45, -0.22, 0.50],
            [0.95, -0.40, -0.20, -0.35, 1.30, -0.50, 0.35, 0.28, -0.80],
            [-1.50, 0.42, -0.75, 0.18, -0.35, 0.40, 0.55, -0.15, 0.20],
        ];

        let eps = 1e-6;
        for (idx, q) in poses.iter().enumerate() {
            let j_ana = chain.jacobian_nozzle(q);
            let mut j_num = SMatrix::<f64, 6, 9>::zeros();

            let (_, _, r0) = chain.fk_internal(q);

            for col in 0..9 {
                let mut qp = *q;
                let mut qm = *q;
                qp[col] += eps;
                qm[col] -= eps;

                let (_, gpp, rpp) = chain.fk_internal(&qp);
                let (_, gpm, rpm) = chain.fk_internal(&qm);
                let pp = gpp + rpp * chain.nozzle_local_pos.as_dvec3();
                let pm = gpm + rpm * chain.nozzle_local_pos.as_dvec3();
                let dp = (pp - pm) / (2.0 * eps);

                let mut dq = rpp * rpm.conjugate();
                if dq.w < 0.0 {
                    dq = -dq;
                }
                let (axis, angle) = dq.to_axis_angle();
                let dw = axis * (angle / (2.0 * eps));
                let _ = r0;

                j_num[(0, col)] = dp.x;
                j_num[(1, col)] = dp.y;
                j_num[(2, col)] = dp.z;
                j_num[(3, col)] = dw.x;
                j_num[(4, col)] = dw.y;
                j_num[(5, col)] = dw.z;
            }

            let max_diff = (j_ana - j_num).abs().max();
            assert!(
                max_diff < 1e-5,
                "Jacobian mismatch at pose {idx}: max_diff = {max_diff:.3e}"
            );
        }
    }

    #[test]
    fn canvas_corners_center_and_dense_grid_reachability() {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);

        // Include full physical canvas corners (±1.25, 0.80..2.30), paintable bounds (±1.17, 0.88..2.22),
        // center, and dense grid across d in [0.04, 0.26].
        let xs = [-1.25, -1.17, -0.60, 0.0, 0.60, 1.17, 1.25];
        let ys = [0.80, 0.88, 1.20, 1.55, 1.90, 2.22, 2.30];
        let ds = [0.04, 0.10, 0.16, 0.26];

        let mut q_seed = chain.comfort_pose;
        for &d in &ds {
            for &y in &ys {
                for &x in &xs {
                    let target_pos = Vec3::new(x, y, d);
                    let target_dir = Vec3::NEG_Z;
                    let q = chain
                        .solve_ik(&q_seed, target_pos, target_dir, IkMode::Spray { roll_weight: 0.15 })
                        .unwrap_or_else(|e| panic!("IK failed at ({x}, {y}, {d}): {e}"));

                    assert!(
                        chain.within_limits(&q, 1e-6),
                        "Joint limit violation at ({x}, {y}, {d}): {q:?}"
                    );

                    let (noz_pos, spray_dir, can_up) = chain.fk_nozzle(&q);
                    let pos_err = (noz_pos - target_pos).length();
                    let point_err_deg = spray_dir
                        .normalize()
                        .dot(target_dir)
                        .clamp(-1.0, 1.0)
                        .acos()
                        .to_degrees();
                    let upright_deg = can_up
                        .normalize()
                        .dot(Vec3::Y)
                        .clamp(-1.0, 1.0)
                        .acos()
                        .to_degrees();

                    assert!(
                        pos_err < 0.0005,
                        "Position error {pos_err:.6}m >= 0.5mm at ({x}, {y}, {d})"
                    );
                    assert!(
                        point_err_deg < 0.2,
                        "Pointing error {point_err_deg:.4}deg >= 0.2deg at ({x}, {y}, {d})"
                    );
                    assert!(
                        upright_deg < 35.0,
                        "Can upright angle {upright_deg:.2}deg >= 35deg at ({x}, {y}, {d})"
                    );

                    q_seed = q;
                }
            }
        }
    }

    #[test]
    fn rack_all_16_slots_grasp_approach_and_lift_reachability() {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);
        let slots = studio.rack_slots(16);

        let mut q_seed = chain.comfort_pose;
        for slot in &slots {
            let poses = [
                ("approach", slot.grasp_pos + slot.approach_dir * 0.14),
                ("grasp", slot.grasp_pos),
                ("lift", slot.grasp_pos + Vec3::Y * 0.06),
                ("retract", slot.grasp_pos + Vec3::Y * 0.06 + slot.approach_dir * 0.14),
            ];

            for (label, target_pos) in poses {
                let q = chain
                    .solve_ik_grasp(&q_seed, target_pos, slot.grasp_rot)
                    .unwrap_or_else(|e| {
                        panic!("Rack slot {} {label} IK failed: {e}", slot.slot_index)
                    });

                assert!(
                    chain.within_limits(&q, 1e-6),
                    "Joint limit violation on rack slot {} {label}: {q:?}",
                    slot.slot_index
                );

                let grasp = chain.fk_grasp(&q);
                let pos_err = (grasp.pos - target_pos).length();
                let rot_err_deg = grasp.rot.angle_between(slot.grasp_rot).to_degrees();

                assert!(
                    pos_err < 0.0005,
                    "Rack slot {} {label} pos_err={pos_err:.6}m >= 0.5mm",
                    slot.slot_index
                );
                assert!(
                    rot_err_deg < 0.2,
                    "Rack slot {} {label} rot_err={rot_err_deg:.4}deg >= 0.2deg",
                    slot.slot_index
                );

                q_seed = q;
            }
        }
    }

    #[test]
    fn horizontal_scanline_234cm_continuous_ik_no_reconfig_or_rail_lag() {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);

        let x_start = -1.17_f32;
        let x_end = 1.17_f32;
        let y = 1.55_f32;
        let z = 0.22_f32;
        let steps = (((x_end - x_start) / 0.005).round() as usize).max(1);

        let mut q_prev = chain
            .solve_ik(
                &chain.comfort_pose,
                Vec3::new(x_start, y, z),
                Vec3::NEG_Z,
                IkMode::Spray { roll_weight: 0.15 },
            )
            .expect("Initial scanline IK should succeed");

        for k in 1..=steps {
            let u = k as f32 / steps as f32;
            let x = x_start + (x_end - x_start) * u;
            let (q_k, reconfigured) = chain
                .solve_ik_continuous(
                    &q_prev,
                    Vec3::new(x, y, z),
                    Vec3::NEG_Z,
                    IkMode::Spray { roll_weight: 0.15 },
                )
                .unwrap_or_else(|e| panic!("Scanline step {k} x={x:.3} failed: {e}"));

            assert!(
                !reconfigured,
                "Unexpected reconfiguration at scanline step {k} x={x:.3}"
            );
            let max_dq = (0..9)
                .map(|dof| (q_k[dof] - q_prev[dof]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_dq < 0.06,
                "Joint step too large at step {k} x={x:.3}: max_dq={max_dq:.5}"
            );
            assert!(
                chain.min_world_z(&q_k, true) > 0.012,
                "Canvas clearance violated at step {k} x={x:.3}: min_z={:.4}",
                chain.min_world_z(&q_k, true)
            );
            q_prev = q_k;
        }
    }

    #[test]
    fn oblique_scanline_drag_tilt_continuous_ik_and_wrist_smoothness() {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);

        let x_start = -1.17_f32;
        let x_end = 1.17_f32;
        let y = 1.55_f32;
        let standoff = 0.14_f32;
        let steps = (((x_end - x_start) / 0.005).round() as usize).max(1);

        // Test both +15 deg (rightward drag) and -15 deg (leftward drag) yaw tilt
        for yaw_deg in [15.0_f32, -15.0_f32] {
            let yaw = yaw_deg.to_radians();
            let target_dir = Vec3::new(yaw.sin(), 0.0, -yaw.cos()).normalize();
            let (xs, xe) = if yaw_deg > 0.0 {
                (x_start, x_end)
            } else {
                (x_end, x_start)
            };

            let canvas_pt_0 = Vec3::new(xs, y, 0.0);
            let noz_0 = canvas_pt_0 - target_dir * standoff;
            let seed_0 = chain.smart_seed_for_nozzle(noz_0, target_dir);
            let mut q_prev = chain
                .solve_ik(&seed_0, noz_0, target_dir, IkMode::Spray { roll_weight: 0.15 })
                .unwrap_or_else(|e| panic!("Initial oblique IK (yaw={yaw_deg}deg) failed: {e}"));

            let mut max_wrist_dq = 0.0_f64;
            for k in 1..=steps {
                let u = k as f32 / steps as f32;
                let x = xs + (xe - xs) * u;
                let canvas_pt = Vec3::new(x, y, 0.0);
                let target_noz = canvas_pt - target_dir * standoff;

                let (q_k, reconfigured) = chain
                    .solve_ik_continuous(
                        &q_prev,
                        target_noz,
                        target_dir,
                        IkMode::Spray { roll_weight: 0.15 },
                    )
                    .unwrap_or_else(|e| {
                        panic!("Oblique scanline (yaw={yaw_deg}deg) step {k} x={x:.3} failed: {e}")
                    });

                assert!(
                    !reconfigured,
                    "Unexpected reconfiguration on oblique scanline (yaw={yaw_deg}deg) step {k} x={x:.3}"
                );
                let max_dq = (0..9)
                    .map(|dof| (q_k[dof] - q_prev[dof]).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    max_dq < 0.05,
                    "Oblique joint step too large (yaw={yaw_deg}deg) at step {k} x={x:.3}: max_dq={max_dq:.5}"
                );
                for wrist_dof in 5..9 {
                    max_wrist_dq = max_wrist_dq.max((q_k[wrist_dof] - q_prev[wrist_dof]).abs());
                }

                let (noz_fk, dir_fk, _) = chain.fk_nozzle(&q_k);
                let pos_err = (noz_fk - target_noz).length();
                let dir_err_deg = dir_fk.normalize().dot(target_dir).clamp(-1.0, 1.0).acos().to_degrees();
                assert!(pos_err < 0.001, "pos_err={pos_err:.5}m at x={x:.3}");
                assert!(dir_err_deg < 0.35, "dir_err={dir_err_deg:.3}deg at x={x:.3}");
                assert!(
                    chain.min_world_z(&q_k, true) > 0.012,
                    "Canvas clearance violated on oblique scanline at x={x:.3}: min_z={:.4}",
                    chain.min_world_z(&q_k, true)
                );

                q_prev = q_k;
            }
            assert!(
                max_wrist_dq < 0.03,
                "Wrist delta too large on oblique scanline (yaw={yaw_deg}deg): max_wrist_dq={max_wrist_dq:.5}"
            );
        }
    }

    #[test]
    fn local_36cm_square_and_circle_hatch_arm_first_keeps_rail_planted() {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);

        let center = Vec3::new(0.0, 1.55, 0.14);
        let mode = IkMode::Spray { roll_weight: 0.15 };
        let q_init = chain
            .solve_ik(&chain.comfort_pose, center, Vec3::NEG_Z, mode)
            .expect("Initial mid-canvas IK should succeed");

        // Build a 36cm square contour + horizontal hatch lines + 36cm circle around (0.0, 1.55, 0.14)
        let half = 0.18_f32;
        let mut waypoints: Vec<Vec3> = Vec::new();
        let corners = [
            Vec3::new(-half, center.y - half, center.z),
            Vec3::new(half, center.y - half, center.z),
            Vec3::new(half, center.y + half, center.z),
            Vec3::new(-half, center.y + half, center.z),
            Vec3::new(-half, center.y - half, center.z),
        ];
        // Smooth 5mm lead-in from center to first corner
        let lead_steps = (((corners[0] - center).length() / 0.005).ceil() as usize).max(1);
        for k in 1..=lead_steps {
            waypoints.push(center.lerp(corners[0], k as f32 / lead_steps as f32));
        }
        for w in corners.windows(2) {
            let steps = (((w[1] - w[0]).length() / 0.005).ceil() as usize).max(1);
            for k in 1..=steps {
                waypoints.push(w[0].lerp(w[1], k as f32 / steps as f32));
            }
        }
        // Horizontal boustrophedon hatch lines across the 36cm square
        for row in 0..=6 {
            let y = (center.y - half) + (2.0 * half) * (row as f32 / 6.0);
            let (x0, x1) = if row % 2 == 0 {
                (-half, half)
            } else {
                (half, -half)
            };
            let p0 = Vec3::new(x0, y, center.z);
            let p1 = Vec3::new(x1, y, center.z);
            let prev = *waypoints.last().unwrap();
            let link_steps = (((p0 - prev).length() / 0.005).ceil() as usize).max(1);
            for k in 1..=link_steps {
                waypoints.push(prev.lerp(p0, k as f32 / link_steps as f32));
            }
            let h_steps = (((p1 - p0).length() / 0.005).ceil() as usize).max(1);
            for k in 1..=h_steps {
                waypoints.push(p0.lerp(p1, k as f32 / h_steps as f32));
            }
        }
        // 36cm diameter circle
        let circle_steps = 226usize;
        let c_start = Vec3::new(center.x + half, center.y, center.z);
        let prev = *waypoints.last().unwrap();
        let link_steps = (((c_start - prev).length() / 0.005).ceil() as usize).max(1);
        for k in 1..=link_steps {
            waypoints.push(prev.lerp(c_start, k as f32 / link_steps as f32));
        }
        for k in 1..=circle_steps {
            let theta = 2.0 * std::f32::consts::PI * (k as f32 / circle_steps as f32);
            waypoints.push(Vec3::new(
                center.x + half * theta.cos(),
                center.y + half * theta.sin(),
                center.z,
            ));
        }

        let mut q_prev = q_init;
        let mut min_q = [f64::INFINITY; 9];
        let mut max_q = [f64::NEG_INFINITY; 9];
        for i in 0..9 {
            min_q[i] = min_q[i].min(q_prev[i]);
            max_q[i] = max_q[i].max(q_prev[i]);
        }

        for (step, &target_pos) in waypoints.iter().enumerate() {
            let (q_k, reconfigured) = chain
                .solve_ik_continuous(&q_prev, target_pos, Vec3::NEG_Z, mode)
                .unwrap_or_else(|e| panic!("Local 36cm step {step} failed: {e}"));
            assert!(
                !reconfigured,
                "Unexpected reconfiguration on local 36cm shape at step {step}"
            );
            assert!(chain.within_limits(&q_k, 1e-6));

            let (noz_fk, dir_fk, _) = chain.fk_nozzle(&q_k);
            let pos_err = (noz_fk - target_pos).length();
            let dir_err_deg = dir_fk
                .normalize()
                .dot(Vec3::NEG_Z)
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            assert!(
                pos_err < 0.0005,
                "Local 36cm pos_err={pos_err:.6}m >= 0.5mm at step {step}"
            );
            assert!(
                dir_err_deg < 0.35,
                "Local 36cm dir_err={dir_err_deg:.4}deg >= 0.35deg at step {step}"
            );
            assert!(chain.min_world_z(&q_k, true) > 0.012);

            for i in 0..9 {
                min_q[i] = min_q[i].min(q_k[i]);
                max_q[i] = max_q[i].max(q_k[i]);
            }
            q_prev = q_k;
        }

        let rail_excursion = max_q[0] - min_q[0];
        let yaw_span = max_q[1] - min_q[1];
        let pitch_span = (max_q[2] - min_q[2]).max(max_q[4] - min_q[4]);
        let wrist_span = (5..9).map(|i| max_q[i] - min_q[i]).fold(0.0_f64, f64::max);
        println!(
            "local 36cm shape: rail_excursion={rail_excursion:.4}m, base_yaw_span={yaw_span:.3}rad, pitch_span={pitch_span:.3}rad, wrist_span={wrist_span:.3}rad"
        );
        assert!(
            rail_excursion < 0.025,
            "Expected rail_x excursion < 0.025m for local 36cm hatch/circle, got {rail_excursion:.4}m"
        );
        assert!(
            yaw_span > 0.30,
            "Expected base_yaw to lead horizontal motion (> 0.30 rad), got {yaw_span:.3} rad"
        );
        assert!(
            pitch_span > 0.25,
            "Expected shoulder/elbow pitch articulation (> 0.25 rad), got {pitch_span:.3} rad"
        );
        assert!(
            wrist_span > 0.30,
            "Expected wrist compensation (> 0.30 rad), got {wrist_span:.3} rad"
        );
    }

    #[test]
    fn min_world_z_park_home_is_high_clearance() {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);
        let park = [1.38, -0.25, 0.15, 0.0, 1.10, 0.0, -1.25, 0.0, 0.0];
        let min_z = chain.min_world_z(&park, false);
        assert!(
            min_z >= 0.70,
            "Expected PARK_HOME min_world_z >= 0.70m, got {min_z:.4}m"
        );
    }
}
