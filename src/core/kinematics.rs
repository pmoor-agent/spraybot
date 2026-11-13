use std::f64::consts::PI;

use glam::{DQuat, DVec3, Quat, Vec3};
use nalgebra::{SMatrix, SVector};

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
    /// Builds the default 7-DOF SprayBot chain matching [`StudioConfig`].
    ///
    /// Link index meanings for [`Self::fk_links`]:
    /// - `0`: `rail_x` carriage at `(q0, carriage_y, rail_z)`, identity rotation.
    /// - `1`: `base_yaw` rotating riser/turret centered at mid-riser height, rotated by `q1` around `+Y`.
    /// - `2`: `shoulder_pitch` upper-arm root at `(q0, shoulder_pivot_y, rail_z)` (`+Y` extends `L2` to elbow).
    /// - `3`: `elbow_pitch` forearm root (`-Z` extends `L3` to `wrist_1`).
    /// - `4`: `wrist_1` housing (`-Z` extends `0.09m` to `wrist_2`).
    /// - `5`: `wrist_2` clevis (`-Z` extends `0.025m` to `flange_roll`).
    /// - `6`: `flange_roll` tool flange (`tcp_offset = (0, 0, -0.025)` reaches grasp center; total wrist-to-grasp `0.14m`).
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
                    ik_weight: 2.5,
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
                    ik_weight: 1.5,
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
                    min: -PI,
                    max: PI,
                    max_vel: 3.6,
                    max_accel: 10.0,
                    ik_weight: 0.4,
                },
                JointSpec {
                    name: "wrist_2".to_string(),
                    is_prismatic: false,
                    axis: Vec3::X,
                    offset: Vec3::new(0.0, 0.0, -0.09),
                    min: -2.3,
                    max: 2.3,
                    max_vel: 3.6,
                    max_accel: 10.0,
                    ik_weight: 0.4,
                },
                JointSpec {
                    name: "flange_roll".to_string(),
                    is_prismatic: false,
                    axis: Vec3::Z,
                    offset: Vec3::new(0.0, 0.0, -0.025),
                    min: -2.0 * PI,
                    max: 2.0 * PI,
                    max_vel: 3.6,
                    max_accel: 10.0,
                    ik_weight: 0.4,
                },
            ],
            base_origin: Vec3::new(0.0, studio.carriage_y, studio.rail_z),
            tcp_offset: Vec3::new(0.0, 0.0, -0.025),
            nozzle_local_pos: Vec3::new(0.0, 0.085, -0.015),
            nozzle_local_dir: Vec3::NEG_Z,
            comfort_pose: [0.0, 0.0, -0.45, 1.05, 0.0, -0.60, 0.0],
        }
    }

    /// Clamps a joint vector strictly within `[joint.min, joint.max]`.
    pub fn clamp_joints(&self, q: &[f64; 7]) -> [f64; 7] {
        let mut out = *q;
        for (i, j) in self.joints.iter().enumerate().take(7) {
            out[i] = out[i].clamp(j.min, j.max);
        }
        out
    }

    /// Returns `true` if every joint lies within `[min - tol, max + tol]`.
    pub fn within_limits(&self, q: &[f64; 7], tol: f64) -> bool {
        self.joints
            .iter()
            .enumerate()
            .take(7)
            .all(|(i, j)| q[i] >= j.min - tol && q[i] <= j.max + tol)
    }

    /// Internal f64 forward kinematics returning per-joint `(pos, rot, world_axis)` plus
    /// the grasp frame `(grasp_pos, grasp_rot)`.
    fn fk_internal(&self, q: &[f64; 7]) -> ([(DVec3, DQuat, DVec3); 7], DVec3, DQuat) {
        let mut p = self.base_origin.as_dvec3();
        let mut r = DQuat::IDENTITY;
        let mut frames = [(DVec3::ZERO, DQuat::IDENTITY, DVec3::X); 7];

        for i in 0..7 {
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

    /// Computes world-space transforms for all 7 moving links.
    ///
    /// Index mapping:
    /// - `0`: `rail_x` carriage
    /// - `1`: `base_yaw` rotating riser
    /// - `2`: `shoulder_pitch` upper arm
    /// - `3`: `elbow_pitch` forearm
    /// - `4`: `wrist_1` roll housing
    /// - `5`: `wrist_2` pitch clevis
    /// - `6`: `flange_roll` tool flange
    pub fn fk_links(&self, q: &[f64; 7]) -> Vec<LinkTransform> {
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
    pub fn fk_all_links(&self, q: &[f64; 7]) -> Vec<LinkTransform> {
        self.fk_links(q)
    }

    /// Computes world-space transform of the gripper grasp center.
    pub fn fk_grasp(&self, q: &[f64; 7]) -> LinkTransform {
        let (_, grasp_pos, grasp_rot) = self.fk_internal(q);
        LinkTransform {
            pos: grasp_pos.as_vec3(),
            rot: grasp_rot.as_quat(),
        }
    }

    /// Computes `(nozzle_world_pos, spray_world_dir, can_up_world)` at joint configuration `q`.
    pub fn fk_nozzle(&self, q: &[f64; 7]) -> (Vec3, Vec3, Vec3) {
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

    /// Analytical 6x7 geometric Jacobian for a point `p_ee` rigidly attached to the tool flange.
    fn jacobian_for_point(&self, frames: &[(DVec3, DQuat, DVec3); 7], p_ee: DVec3) -> SMatrix<f64, 6, 7> {
        let mut jac = SMatrix::<f64, 6, 7>::zeros();
        for i in 0..7 {
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

    /// Analytical 6x7 geometric Jacobian for the nozzle TCP (`[linear; angular]`).
    pub fn jacobian_nozzle(&self, q: &[f64; 7]) -> SMatrix<f64, 6, 7> {
        let (frames, grasp_pos, grasp_rot) = self.fk_internal(q);
        let p_nozzle = grasp_pos + grasp_rot * self.nozzle_local_pos.as_dvec3();
        self.jacobian_for_point(&frames, p_nozzle)
    }

    /// Analytical 6x7 geometric Jacobian for the gripper grasp center (`[linear; angular]`).
    pub fn jacobian_grasp(&self, q: &[f64; 7]) -> SMatrix<f64, 6, 7> {
        let (frames, grasp_pos, _) = self.fk_internal(q);
        self.jacobian_for_point(&frames, grasp_pos)
    }

    /// Computes an analytical elbow-up seed for reaching a desired grasp frame `(grasp_pos, grasp_dir)`.
    pub fn smart_seed_for_grasp(&self, grasp_pos: Vec3, grasp_dir: Vec3) -> [f64; 7] {
        let l2 = self.joints[3].offset.y as f64;
        let l3 = -self.joints[4].offset.z as f64;
        let d_wrist_j5 = -self.joints[5].offset.z as f64;
        let d_j5_grasp = -(self.joints[6].offset.z + self.tcp_offset.z) as f64;
        let l3_eff = l3 + d_wrist_j5;

        let g_pos = grasp_pos.as_dvec3();
        let g_dir = grasp_dir.as_dvec3().normalize();
        // Estimate wrist_2 (J5) center back along -grasp_dir from grasp center
        let p_w = g_pos - g_dir * d_j5_grasp;

        let q0 = p_w.x.clamp(self.joints[0].min, self.joints[0].max);
        let shoulder = DVec3::new(q0, (self.base_origin.y + self.joints[1].offset.y + self.joints[2].offset.y) as f64, self.base_origin.z as f64);

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

        let sin_q3 = ((r_sq - l2 * l2 - l3_eff * l3_eff) / (2.0 * l2 * l3_eff)).clamp(-1.0, 1.0);
        // Elbow-up branch: q3 in [-pi/2, +pi/2]
        let q3 = sin_q3.asin().clamp(self.joints[3].min, self.joints[3].max);

        let a = l2 + l3_eff * q3.sin();
        let b = l3_eff * q3.cos();
        let q2 = (b * dy - a * r_xz)
            .atan2(a * dy + b * r_xz)
            .clamp(self.joints[2].min, self.joints[2].max);

        let target_pitch = g_dir.y.clamp(-1.0, 1.0).asin();
        let q5 = (-(q2 + q3) + target_pitch).clamp(self.joints[5].min, self.joints[5].max);

        self.clamp_joints(&[q0, q1, q2, q3, 0.0, q5, 0.0])
    }

    /// Computes an analytical elbow-up seed for reaching a desired nozzle target `(nozzle_pos, spray_dir)`.
    pub fn smart_seed_for_nozzle(&self, target_pos: Vec3, target_dir: Vec3) -> [f64; 7] {
        let dir = target_dir.as_dvec3().normalize();
        // Construct upright grasp rotation whose -Z aligns with target_dir
        let rot = upright_rot_from_neg_z(dir, DVec3::Y);
        let grasp_pos = target_pos.as_dvec3() - rot * self.nozzle_local_pos.as_dvec3();
        self.smart_seed_for_grasp(grasp_pos.as_vec3(), target_dir)
    }

    /// Core Weighted Damped Least-Squares iteration loop for nozzle or grasp targets.
    fn solve_dls_internal(
        &self,
        q_seed: &[f64; 7],
        target_pos: DVec3,
        target_dir: DVec3,
        target_up: DVec3,
        target_rot_exact: Option<DQuat>,
        use_nozzle_tcp: bool,
        mode: IkMode,
        max_iters: usize,
    ) -> ([f64; 7], f64, f64) {
        let mut q = self.clamp_joints(q_seed);
        let d_t = target_dir.normalize();

        // Diagonal joint weights W and inverse W^{-1}
        let mut w_inv = SMatrix::<f64, 7, 7>::zeros();
        for i in 0..7 {
            let w = self.joints[i].ik_weight.max(1e-6);
            w_inv[(i, i)] = 1.0 / w;
        }

        let lambda = 0.025;
        let alpha_null = 0.12;

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
                for c in 0..7 {
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
                // Null-space projector (I - J#_W * J_s)
                let x_mat = chol.solve(&j_s);
                let j_pinv_j = w_inv_jt * x_mat;
                let null_proj = SMatrix::<f64, 7, 7>::identity() - j_pinv_j;

                let mut v_null = SVector::<f64, 7>::zeros();
                for i in 0..7 {
                    let j_spec = &self.joints[i];
                    let comfort_pull = if i == 0 {
                        let target_x = target_pos.x.clamp(j_spec.min, j_spec.max);
                        0.24 * (target_x - q[0])
                    } else {
                        alpha_null * (self.comfort_pose[i] - q[i])
                    };
                    // Soft joint-limit repulsion inside outer 12% of range
                    let span = (j_spec.max - j_spec.min).max(1e-6);
                    let margin = 0.12 * span;
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
                dq_t + null_proj * v_null
            } else {
                SVector::<f64, 7>::zeros()
            };

            // Step clamp for stability on large errors
            let max_step = dq_task.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let scale = if max_step > 0.35 { 0.35 / max_step } else { 1.0 };

            for i in 0..7 {
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
        q_seed: &[f64; 7],
        target_pos: Vec3,
        target_dir: Vec3,
        mode: IkMode,
    ) -> Result<[f64; 7], String> {
        self.solve_ik_with_up(q_seed, target_pos, target_dir, Vec3::Y, mode)
    }

    /// Nozzle IK with explicit desired can-up vector (`target_up`, usually `Vec3::Y`).
    pub fn solve_ik_with_up(
        &self,
        q_seed: &[f64; 7],
        target_pos: Vec3,
        target_dir: Vec3,
        target_up: Vec3,
        mode: IkMode,
    ) -> Result<[f64; 7], String> {
        let p_t = target_pos.as_dvec3();
        let d_t = target_dir.as_dvec3().normalize();
        let u_t = target_up.as_dvec3().normalize();

        let (q1, pos_err1, rot_err1) =
            self.solve_dls_internal(q_seed, p_t, d_t, u_t, None, true, mode, 16);
        if pos_err1 <= 5e-4 && rot_err1 <= 0.2_f64.to_radians() {
            return Ok(q1);
        }

        // Fallback from analytical elbow-up seed for large jumps / branch recovery
        let smart = self.smart_seed_for_nozzle(target_pos, target_dir);
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
        q_seed: &[f64; 7],
        target_pos: Vec3,
        target_dir: Vec3,
        mode: IkMode,
    ) -> Result<([f64; 7], bool), String> {
        let p_t = target_pos.as_dvec3();
        let d_t = target_dir.as_dvec3().normalize();
        let u_t = DVec3::Y;

        let (q1, pos_err1, rot_err1) =
            self.solve_dls_internal(q_seed, p_t, d_t, u_t, None, true, mode, 36);
        let max_dq1 = (0..7)
            .map(|dof| (q1[dof] - q_seed[dof]).abs())
            .fold(0.0_f64, f64::max);
        if pos_err1 <= 1.0e-3 && rot_err1 <= 0.35_f64.to_radians() && max_dq1 <= 0.22 {
            return Ok((q1, false));
        }

        let smart = self.smart_seed_for_nozzle(target_pos, target_dir);
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
    pub fn min_world_z(&self, q: &[f64; 7], held_can: bool) -> f32 {
        const LINK_RADII: [f64; 7] = [0.16, 0.13, 0.08, 0.07, 0.05, 0.05, 0.05];
        let (frames, grasp_pos, grasp_rot) = self.fk_internal(q);
        let mut min_z = f64::INFINITY;

        for i in 0..7 {
            min_z = min_z.min(frames[i].0.z - LINK_RADII[i]);
        }

        let upper_mid = 0.5 * (frames[2].0 + frames[3].0);
        let forearm_mid = 0.5 * (frames[3].0 + frames[4].0);
        min_z = min_z.min(upper_mid.z - 0.07);
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
        q_seed: &[f64; 7],
        target_pos: Vec3,
        target_rot: Quat,
    ) -> Result<[f64; 7], String> {
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

        let poses: [[f64; 7]; 5] = [
            chain.comfort_pose,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.85, 0.35, -0.60, 0.85, 0.25, -0.45, 0.50],
            [0.95, -0.40, -0.20, 1.30, -0.50, 0.35, -0.80],
            [-1.50, 0.42, -0.75, -0.35, 0.40, 0.55, 0.20],
        ];

        let eps = 1e-6;
        for (idx, q) in poses.iter().enumerate() {
            let j_ana = chain.jacobian_nozzle(q);
            let mut j_num = SMatrix::<f64, 6, 7>::zeros();

            let (_, _, r0) = chain.fk_internal(q);

            for col in 0..7 {
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
            let max_dq = (0..7)
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
    fn min_world_z_park_home_is_high_clearance() {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);
        let park = [1.38, -0.25, 0.15, 1.10, 0.0, -1.25, 0.0];
        let min_z = chain.min_world_z(&park, false);
        assert!(
            min_z >= 0.70,
            "Expected PARK_HOME min_world_z >= 0.70m, got {min_z:.4}m"
        );
    }
}
