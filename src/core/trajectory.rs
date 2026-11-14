use std::f64::consts::PI;

use glam::Vec3;

use crate::core::kinematics::{IkMode, KinematicChain};
use crate::core::types::{
    JobOp, JobPlan, NozzleCap, RobotFrameState, StrokeWaypoint, StudioConfig,
};

/// Safe Z standoff from the canvas plane (`Z = 0.0`) during rapid air transits (meters).
const SAFE_TRANSIT_Z: f32 = 0.30;
/// Rack pre-grasp clearance along `approach_dir` (meters).
const RACK_APPROACH_DIST: f32 = 0.14;
/// Vertical lift height to clear the rack shelf pin/lip during pick/return (meters).
const RACK_LIFT_HEIGHT: f32 = 0.06;
/// Duration of gripper open/close actuation (seconds).
const GRIPPER_ACTUATE_DURATION: f64 = 0.25;
/// Duration of pneumatic cap-swap dwell at cap station dock (seconds).
const CAP_SWAP_DWELL_DURATION: f64 = 0.18;
/// Duration of can shake operation (seconds).
const SHAKE_DURATION: f64 = 1.1;
/// Shake oscillation frequency (Hz).
const SHAKE_FREQ_HZ: f64 = 6.0;
/// Shake oscillation amplitude in radians (`±22°`).
const SHAKE_AMP_RAD: f64 = 22.0_f64 * PI / 180.0;
/// Unobstructed side park pose at the right end of the rail (`q0 = 1.38m > +1.25m` canvas edge).
const PARK_HOME_JOINTS: [f64; 9] = [1.38, -0.25, 0.15, 0.0, 1.10, 0.0, -1.25, 0.0, 0.0];

/// Symmetric smooth-ramp (`C²` continuous) S-curve 1D time-scaling profile on `[0, duration]`.
#[derive(Debug, Clone, Copy)]
struct SCurveProfile {
    duration: f64,
    t_accel: f64,
    t_cruise: f64,
    v_peak: f64,
    distance: f64,
}

impl SCurveProfile {
    /// Builds a time-optimal `C²` raised-cosine ramp profile covering `distance`
    /// within `v_lim` and `a_lim`.
    fn new(distance: f64, v_lim: f64, a_lim: f64) -> Self {
        let d = distance.abs();
        if d < 1e-9 {
            return Self {
                duration: 0.0,
                t_accel: 0.0,
                t_cruise: 0.0,
                v_peak: 0.0,
                distance: 0.0,
            };
        }
        let v_max = v_lim.max(1e-4);
        let a_max = a_lim.max(1e-4);
        // Peak acceleration of 0.5 * v_peak * (1 - cos(pi * t / t_a)) is v_peak * pi / (2 * t_a)
        let t_a_full = (PI * 0.5) * v_max / a_max;
        let d_min_full = v_max * t_a_full;

        let (v_peak, t_accel, t_cruise) = if d >= d_min_full {
            (v_max, t_a_full, (d - d_min_full) / v_max)
        } else {
            let v_p = (2.0 * d * a_max / PI).sqrt();
            let t_a = d / v_p.max(1e-9);
            (v_p, t_a, 0.0)
        };

        Self {
            duration: 2.0 * t_accel + t_cruise,
            t_accel,
            t_cruise,
            v_peak,
            distance: d,
        }
    }

    /// Evaluates normalized progress `s(t) ∈ [0.0, 1.0]` at local time `t ∈ [0.0, duration]`.
    fn eval(&self, t: f64) -> f64 {
        if self.distance < 1e-9 || self.duration <= 1e-9 {
            return 1.0;
        }
        let t = t.clamp(0.0, self.duration);
        let s_raw = if t <= self.t_accel {
            if self.t_accel <= 1e-12 {
                0.0
            } else {
                self.v_peak
                    * (0.5 * t - (self.t_accel / (2.0 * PI)) * (PI * t / self.t_accel).sin())
            }
        } else if t <= self.t_accel + self.t_cruise {
            0.5 * self.v_peak * self.t_accel + self.v_peak * (t - self.t_accel)
        } else {
            let t_rem = (self.duration - t).max(0.0);
            let s_rem = if self.t_accel <= 1e-12 {
                0.0
            } else {
                self.v_peak
                    * (0.5 * t_rem
                        - (self.t_accel / (2.0 * PI)) * (PI * t_rem / self.t_accel).sin())
            };
            self.distance - s_rem
        };
        (s_raw / self.distance).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone)]
struct PathSample {
    s: f64,
    t: f64,
    joints: [f64; 9],
    valve_on: bool,
    opacity_scale: f32,
    cap: NozzleCap,
}

#[derive(Debug, Clone)]
enum SegmentKind {
    JointMove {
        q_start: [f64; 9],
        q_end: [f64; 9],
        profile: SCurveProfile,
        gripper_open: f32,
        held_can_slot: Option<usize>,
        active_cap: NozzleCap,
    },
    GripperActuate {
        q: [f64; 9],
        open_start: f32,
        open_end: f32,
        slot_before: Option<usize>,
        slot_after: Option<usize>,
        cap_before: NozzleCap,
        cap_after: NozzleCap,
        switch_at: f32,
    },
    CapSwitchHold {
        q: [f64; 9],
        held_can_slot: Option<usize>,
        cap_before: NozzleCap,
        cap_after: NozzleCap,
        switch_at: f32,
    },
    ShakeMove {
        q_base: [f64; 9],
        pitch_bounds: (f64, f64),
        roll_bounds: (f64, f64),
        held_can_slot: Option<usize>,
        active_cap: NozzleCap,
    },
    ArcScaledPath {
        samples: Vec<PathSample>,
        profile: SCurveProfile,
        gripper_open: f32,
        held_can_slot: Option<usize>,
        active_cap: NozzleCap,
    },
    TimedStroke {
        samples: Vec<PathSample>,
        gripper_open: f32,
        held_can_slot: Option<usize>,
    },
}

#[derive(Debug, Clone)]
struct TimelineSegment {
    t_start: f64,
    t_end: f64,
    kind: SegmentKind,
}

/// Time-parameterized robot trajectory compiled from a [`JobPlan`].
#[derive(Debug, Clone)]
pub struct CompiledTimeline {
    pub total_duration: f64,
    pub home_joints: [f64; 9],
    chain: KinematicChain,
    segments: Vec<TimelineSegment>,
}

impl CompiledTimeline {
    /// Creates an idle parked timeline with zero duration.
    pub fn empty() -> Self {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);
        let home = chain.clamp_joints(&PARK_HOME_JOINTS);
        Self {
            total_duration: 0.0,
            home_joints: home,
            chain,
            segments: Vec::new(),
        }
    }

    /// Returns the robot's home joint configuration.
    #[inline]
    pub fn home_joints(&self) -> [f64; 9] {
        self.home_joints
    }

    /// Total timeline duration in simulation seconds.
    #[inline]
    pub fn duration(&self) -> f64 {
        self.total_duration
    }

    /// Compiles a high-level [`JobPlan`] into a smooth time-parameterized trajectory.
    pub fn compile(
        job: &JobPlan,
        chain: &KinematicChain,
        studio: &StudioConfig,
    ) -> Result<Self, String> {
        let max_slot = job
            .ops
            .iter()
            .filter_map(|op| match op {
                JobOp::PickCan { slot } | JobOp::ReturnCan { slot } => Some(*slot),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let rack_cap = (studio.rack_slots_per_tier_default * 2).max(max_slot + 1);
        let slots = studio.rack_slots(rack_cap);

        let home = chain.clamp_joints(&PARK_HOME_JOINTS);
        let mut q_curr = home;
        let mut gripper_open: f32 = 1.0;
        let mut held_can_slot: Option<usize> = None;
        let mut slot_caps = vec![NozzleCap::Fat; rack_cap];
        let mut active_cap = NozzleCap::Fat;

        let mut segments: Vec<TimelineSegment> = Vec::new();
        let mut t_cursor = 0.0_f64;

        for (op_idx, op) in job.ops.iter().enumerate() {
            match op {
                JobOp::PickCan { slot } => {
                    let slot_pose = slots.get(*slot).ok_or_else(|| {
                        format!("PickCan slot {slot} out of bounds (rack size {})", slots.len())
                    })?;

                    let approach_pos = slot_pose.grasp_pos + slot_pose.approach_dir * RACK_APPROACH_DIST;
                    let grasp_pos = slot_pose.grasp_pos;
                    let lift_pos = slot_pose.grasp_pos + Vec3::Y * RACK_LIFT_HEIGHT;
                    let retract_pos = lift_pos + slot_pose.approach_dir * RACK_APPROACH_DIST;

                    // 1. Move to rack approach
                    let q_approach =
                        chain.solve_ik_grasp(&q_curr, approach_pos, slot_pose.grasp_rot)?;
                    push_safe_joint_move(
                        chain,
                        &mut segments,
                        &mut t_cursor,
                        q_curr,
                        q_approach,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                    );
                    q_curr = q_approach;

                    // 2. Linear insert to grasp_pos
                    let (insert_seg, q_after_insert) = build_cartesian_grasp_segment(
                        chain,
                        q_curr,
                        approach_pos,
                        grasp_pos,
                        slot_pose.grasp_rot,
                        0.35,
                        1.0,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                        t_cursor,
                    )?;
                    t_cursor = insert_seg.t_end;
                    segments.push(insert_seg);
                    q_curr = q_after_insert;

                    // 3. 0.25s gripper close (1.0 -> 0.0, held_can_slot = Some(slot))
                    let next_cap = slot_caps[*slot];
                    let t_end_close = t_cursor + GRIPPER_ACTUATE_DURATION;
                    segments.push(TimelineSegment {
                        t_start: t_cursor,
                        t_end: t_end_close,
                        kind: SegmentKind::GripperActuate {
                            q: q_curr,
                            open_start: gripper_open,
                            open_end: 0.0,
                            slot_before: held_can_slot,
                            slot_after: Some(*slot),
                            cap_before: active_cap,
                            cap_after: next_cap,
                            switch_at: 0.85,
                        },
                    });
                    t_cursor = t_end_close;
                    gripper_open = 0.0;
                    held_can_slot = Some(*slot);
                    active_cap = next_cap;

                    // 4. 0.06m vertical +Y lift
                    let (lift_seg, q_after_lift) = build_cartesian_grasp_segment(
                        chain,
                        q_curr,
                        grasp_pos,
                        lift_pos,
                        slot_pose.grasp_rot,
                        0.25,
                        0.8,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                        t_cursor,
                    )?;
                    t_cursor = lift_seg.t_end;
                    segments.push(lift_seg);
                    q_curr = q_after_lift;

                    // 5. Linear retract 0.14 * approach_dir
                    let (retract_seg, q_after_retract) = build_cartesian_grasp_segment(
                        chain,
                        q_curr,
                        lift_pos,
                        retract_pos,
                        slot_pose.grasp_rot,
                        0.35,
                        1.0,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                        t_cursor,
                    )?;
                    t_cursor = retract_seg.t_end;
                    segments.push(retract_seg);
                    q_curr = q_after_retract;
                }

                JobOp::ShakeCan => {
                    let t_end = t_cursor + SHAKE_DURATION;
                    segments.push(TimelineSegment {
                        t_start: t_cursor,
                        t_end,
                        kind: SegmentKind::ShakeMove {
                            q_base: q_curr,
                            pitch_bounds: (chain.joints[6].min, chain.joints[6].max),
                            roll_bounds: (chain.joints[5].min, chain.joints[5].max),
                            held_can_slot,
                            active_cap,
                        },
                    });
                    t_cursor = t_end;
                }

                JobOp::SwapCap { cap } => {
                    let docks = studio.cap_station_poses(rack_cap);
                    let dock = match cap {
                        NozzleCap::Fat => docks[0],
                        NozzleCap::Skinny => docks[1],
                    };
                    let pre_dock = dock.grasp_pos + dock.approach_dir * 0.12 + Vec3::Y * 0.04;
                    let above_dock = dock.grasp_pos + Vec3::Y * 0.035;

                    let q_pre = chain.solve_ik_grasp(&q_curr, pre_dock, dock.grasp_rot)?;
                    push_safe_joint_move(
                        chain,
                        &mut segments,
                        &mut t_cursor,
                        q_curr,
                        q_pre,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                    );
                    q_curr = q_pre;

                    let (approach_seg, q_above) = build_cartesian_grasp_segment(
                        chain,
                        q_curr,
                        pre_dock,
                        above_dock,
                        dock.grasp_rot,
                        0.32,
                        1.0,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                        t_cursor,
                    )?;
                    t_cursor = approach_seg.t_end;
                    segments.push(approach_seg);
                    q_curr = q_above;

                    let (press_seg, q_dock) = build_cartesian_grasp_segment(
                        chain,
                        q_curr,
                        above_dock,
                        dock.grasp_pos,
                        dock.grasp_rot,
                        0.20,
                        0.8,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                        t_cursor,
                    )?;
                    t_cursor = press_seg.t_end;
                    segments.push(press_seg);
                    q_curr = q_dock;

                    let t_end_dwell = t_cursor + CAP_SWAP_DWELL_DURATION;
                    segments.push(TimelineSegment {
                        t_start: t_cursor,
                        t_end: t_end_dwell,
                        kind: SegmentKind::CapSwitchHold {
                            q: q_curr,
                            held_can_slot,
                            cap_before: active_cap,
                            cap_after: *cap,
                            switch_at: 0.5,
                        },
                    });
                    t_cursor = t_end_dwell;
                    active_cap = *cap;
                    if let Some(slot) = held_can_slot {
                        slot_caps[slot] = *cap;
                    }

                    let (lift_seg, q_lifted) = build_cartesian_grasp_segment(
                        chain,
                        q_curr,
                        dock.grasp_pos,
                        above_dock,
                        dock.grasp_rot,
                        0.22,
                        0.8,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                        t_cursor,
                    )?;
                    t_cursor = lift_seg.t_end;
                    segments.push(lift_seg);
                    q_curr = q_lifted;

                    let (ret_seg, q_ret) = build_cartesian_grasp_segment(
                        chain,
                        q_curr,
                        above_dock,
                        pre_dock,
                        dock.grasp_rot,
                        0.32,
                        1.0,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                        t_cursor,
                    )?;
                    t_cursor = ret_seg.t_end;
                    segments.push(ret_seg);
                    q_curr = q_ret;
                }

                JobOp::PaintStroke { points_canvas_3d } => {
                    if points_canvas_3d.is_empty() {
                        continue;
                    }
                    let wp0 = points_canvas_3d[0];
                    let wp0_noz = wp0.nozzle_pos();
                    let wp0_dir = wp0.spray_dir();

                    // Check if currently facing canvas safely; if so use adaptive-Z Cartesian hop,
                    // otherwise (e.g. coming from rack/home/swap) use safe folded joint-space transit.
                    let (cur_noz, cur_dir, _) = chain.fk_nozzle(&q_curr);
                    let canvas_left = studio.canvas_center.x - studio.canvas_width * 0.6;
                    let facing_wall = cur_dir.z < -0.7
                        && cur_noz.z >= 0.08
                        && cur_noz.x >= canvas_left;

                    let dist_xy = ((cur_noz.x - wp0_noz.x).powi(2) + (cur_noz.y - wp0_noz.y).powi(2)).sqrt();
                    let hop_z = if facing_wall && dist_xy <= 0.45 {
                        cur_noz.z.max((wp0_noz.z + 0.08).clamp(0.12, 0.24))
                    } else {
                        SAFE_TRANSIT_Z
                    };
                    let p_retract_start =
                        Vec3::new(wp0_noz.x, wp0_noz.y, wp0_noz.z.max(hop_z));

                    if facing_wall && (cur_noz - p_retract_start).length() > 1e-3 {
                        let p_cur_safe =
                            Vec3::new(cur_noz.x, cur_noz.y, cur_noz.z.max(hop_z));
                        if (p_cur_safe - cur_noz).length() > 1e-3 {
                            if let Ok((seg, q_next)) = build_cartesian_nozzle_segment(
                                chain,
                                q_curr,
                                cur_noz,
                                p_cur_safe,
                                cur_dir,
                                wp0_dir,
                                1.05,
                                3.2,
                                gripper_open,
                                held_can_slot,
                                active_cap,
                                t_cursor,
                            ) {
                                t_cursor = seg.t_end;
                                segments.push(seg);
                                q_curr = q_next;
                            }
                        }
                        if let Ok((seg, q_next)) = build_cartesian_nozzle_segment(
                            chain,
                            q_curr,
                            p_cur_safe,
                            p_retract_start,
                            cur_dir,
                            wp0_dir,
                            1.45,
                            3.8,
                            gripper_open,
                            held_can_slot,
                            active_cap,
                            t_cursor,
                        ) {
                            t_cursor = seg.t_end;
                            segments.push(seg);
                            q_curr = q_next;
                        } else {
                            let q_retract_start = chain.solve_ik(
                                &q_curr,
                                p_retract_start,
                                wp0_dir,
                                IkMode::Spray { roll_weight: 0.15 },
                            )?;
                            push_safe_joint_move(
                                chain,
                                &mut segments,
                                &mut t_cursor,
                                q_curr,
                                q_retract_start,
                                gripper_open,
                                held_can_slot,
                                active_cap,
                            );
                            q_curr = q_retract_start;
                        }
                    } else if (cur_noz - p_retract_start).length() > 1e-3 {
                        let q_retract_start = chain.solve_ik(
                            &q_curr,
                            p_retract_start,
                            wp0_dir,
                            IkMode::Spray { roll_weight: 0.15 },
                        )?;
                        push_safe_joint_move(
                            chain,
                            &mut segments,
                            &mut t_cursor,
                            q_curr,
                            q_retract_start,
                            gripper_open,
                            held_can_slot,
                            active_cap,
                        );
                        q_curr = q_retract_start;
                    }

                    // Descend from hop/safe Z to first stroke waypoint
                    if (p_retract_start - wp0_noz).length() > 1e-3 {
                        if let Ok((desc_seg, q_at_wp0)) = build_cartesian_nozzle_segment(
                            chain,
                            q_curr,
                            p_retract_start,
                            wp0_noz,
                            wp0_dir,
                            wp0_dir,
                            1.05,
                            3.2,
                            gripper_open,
                            held_can_slot,
                            active_cap,
                            t_cursor,
                        ) {
                            t_cursor = desc_seg.t_end;
                            segments.push(desc_seg);
                            q_curr = q_at_wp0;
                        } else {
                            let q_at_wp0 = chain.solve_ik(
                                &q_curr,
                                wp0_noz,
                                wp0_dir,
                                IkMode::Spray { roll_weight: 0.15 },
                            )?;
                            push_safe_joint_move(
                                chain,
                                &mut segments,
                                &mut t_cursor,
                                q_curr,
                                q_at_wp0,
                                gripper_open,
                                held_can_slot,
                                active_cap,
                            );
                            q_curr = q_at_wp0;
                        }
                    }

                    // Dense continuous IK tracking along stroke waypoints (splitting cleanly on reconfig)
                    if points_canvas_3d.len() >= 2 {
                        q_curr = build_timed_stroke_segments(
                            chain,
                            q_curr,
                            points_canvas_3d,
                            gripper_open,
                            held_can_slot,
                            &mut segments,
                            &mut t_cursor,
                        )?;
                    }

                    // Retract to adaptive hop Z (if next op is a nearby stroke) or SAFE_TRANSIT_Z
                    let wp_last = points_canvas_3d[points_canvas_3d.len() - 1];
                    let p_last = wp_last.nozzle_pos();
                    let d_last = wp_last.spray_dir();
                    let next_retract_z = match job.ops.get(op_idx + 1) {
                        Some(JobOp::PaintStroke { points_canvas_3d: next_pts })
                            if !next_pts.is_empty() =>
                        {
                            let next_wp0 = next_pts[0].nozzle_pos();
                            let next_dist_xy = ((p_last.x - next_wp0.x).powi(2)
                                + (p_last.y - next_wp0.y).powi(2))
                            .sqrt();
                            if next_dist_xy <= 0.45 {
                                (p_last.z.max(next_wp0.z) + 0.08).clamp(0.12, 0.24)
                            } else {
                                SAFE_TRANSIT_Z
                            }
                        }
                        _ => SAFE_TRANSIT_Z,
                    };
                    if p_last.z < next_retract_z - 1e-3 {
                        let p_retract_end = Vec3::new(p_last.x, p_last.y, next_retract_z);
                        if let Ok((ret_seg, q_retracted)) = build_cartesian_nozzle_segment(
                            chain,
                            q_curr,
                            p_last,
                            p_retract_end,
                            d_last,
                            d_last,
                            1.05,
                            3.2,
                            gripper_open,
                            held_can_slot,
                            active_cap,
                            t_cursor,
                        ) {
                            t_cursor = ret_seg.t_end;
                            segments.push(ret_seg);
                            q_curr = q_retracted;
                        } else {
                            let q_retracted = chain.solve_ik(
                                &q_curr,
                                p_retract_end,
                                d_last,
                                IkMode::Spray { roll_weight: 0.15 },
                            )?;
                            push_safe_joint_move(
                                chain,
                                &mut segments,
                                &mut t_cursor,
                                q_curr,
                                q_retracted,
                                gripper_open,
                                held_can_slot,
                                active_cap,
                            );
                            q_curr = q_retracted;
                        }
                    }
                }

                JobOp::ReturnCan { slot } => {
                    let slot_pose = slots.get(*slot).ok_or_else(|| {
                        format!(
                            "ReturnCan slot {slot} out of bounds (rack size {})",
                            slots.len()
                        )
                    })?;

                    let approach_pos =
                        slot_pose.grasp_pos + slot_pose.approach_dir * RACK_APPROACH_DIST;
                    let lift_pos = slot_pose.grasp_pos + Vec3::Y * RACK_LIFT_HEIGHT;
                    let retract_pos = lift_pos + slot_pose.approach_dir * RACK_APPROACH_DIST;
                    let grasp_pos = slot_pose.grasp_pos;

                    // 1. Smooth move to raised approach
                    let q_retract =
                        chain.solve_ik_grasp(&q_curr, retract_pos, slot_pose.grasp_rot)?;
                    push_safe_joint_move(
                        chain,
                        &mut segments,
                        &mut t_cursor,
                        q_curr,
                        q_retract,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                    );
                    q_curr = q_retract;

                    // 2. Linear insert to raised slot (lift_pos)
                    let (insert_seg, q_at_lift) = build_cartesian_grasp_segment(
                        chain,
                        q_curr,
                        retract_pos,
                        lift_pos,
                        slot_pose.grasp_rot,
                        0.35,
                        1.0,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                        t_cursor,
                    )?;
                    t_cursor = insert_seg.t_end;
                    segments.push(insert_seg);
                    q_curr = q_at_lift;

                    // 3. Lower -0.06m to grasp_pos
                    let (lower_seg, q_at_grasp) = build_cartesian_grasp_segment(
                        chain,
                        q_curr,
                        lift_pos,
                        grasp_pos,
                        slot_pose.grasp_rot,
                        0.25,
                        0.8,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                        t_cursor,
                    )?;
                    t_cursor = lower_seg.t_end;
                    segments.push(lower_seg);
                    q_curr = q_at_grasp;

                    // 4. Open gripper 0.25s, held_can_slot = None, active_cap = Fat
                    let t_end_open = t_cursor + GRIPPER_ACTUATE_DURATION;
                    segments.push(TimelineSegment {
                        t_start: t_cursor,
                        t_end: t_end_open,
                        kind: SegmentKind::GripperActuate {
                            q: q_curr,
                            open_start: gripper_open,
                            open_end: 1.0,
                            slot_before: held_can_slot,
                            slot_after: None,
                            cap_before: active_cap,
                            cap_after: NozzleCap::Fat,
                            switch_at: 0.15,
                        },
                    });
                    t_cursor = t_end_open;
                    gripper_open = 1.0;
                    held_can_slot = None;
                    active_cap = NozzleCap::Fat;

                    // 5. Retract 0.14 * approach_dir
                    let (ret_seg, q_at_approach) = build_cartesian_grasp_segment(
                        chain,
                        q_curr,
                        grasp_pos,
                        approach_pos,
                        slot_pose.grasp_rot,
                        0.35,
                        1.0,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                        t_cursor,
                    )?;
                    t_cursor = ret_seg.t_end;
                    segments.push(ret_seg);
                    q_curr = q_at_approach;
                }

                JobOp::ParkHome => {
                    push_safe_joint_move(
                        chain,
                        &mut segments,
                        &mut t_cursor,
                        q_curr,
                        home,
                        gripper_open,
                        held_can_slot,
                        active_cap,
                    );
                    q_curr = home;
                }
            }
        }

        Ok(Self {
            total_duration: t_cursor,
            home_joints: home,
            chain: chain.clone(),
            segments,
        })
    }

    /// Evaluates continuous robot and spray state at simulation time `t_sim`.
    pub fn sample(&self, t_sim: f64) -> RobotFrameState {
        if self.segments.is_empty() || self.total_duration <= 0.0 {
            let (nozzle_world_pos, nozzle_world_dir, _) = self.chain.fk_nozzle(&self.home_joints);
            return RobotFrameState {
                joints: self.home_joints,
                gripper_open: 1.0,
                trigger_depressed: 0.0,
                held_can_slot: None,
                nozzle_world_pos,
                nozzle_world_dir,
                spray_active: false,
                is_shaking: false,
                flow_scale: 0.0,
                active_cap: NozzleCap::Fat,
            };
        }

        let t = t_sim.clamp(0.0, self.total_duration);
        let idx = self
            .segments
            .partition_point(|seg| seg.t_end <= t)
            .min(self.segments.len() - 1);
        let seg = &self.segments[idx];
        let t_local = (t - seg.t_start).clamp(0.0, (seg.t_end - seg.t_start).max(0.0));
        let seg_dur = (seg.t_end - seg.t_start).max(1e-9);

        let (
            joints,
            gripper_open,
            trigger_depressed,
            held_can_slot,
            spray_active,
            is_shaking,
            flow_scale,
            active_cap,
        ) = match &seg.kind {
            SegmentKind::JointMove {
                q_start,
                q_end,
                profile,
                gripper_open,
                held_can_slot,
                active_cap,
            } => {
                let s = profile.eval(t_local);
                let mut q = [0.0; 9];
                for i in 0..9 {
                    q[i] = q_start[i] + (q_end[i] - q_start[i]) * s;
                }
                (q, *gripper_open, 0.0, *held_can_slot, false, false, 0.0, *active_cap)
            }

            SegmentKind::GripperActuate {
                q,
                open_start,
                open_end,
                slot_before,
                slot_after,
                cap_before,
                cap_after,
                switch_at,
            } => {
                let u = (t_local / seg_dur).clamp(0.0, 1.0) as f32;
                let g = open_start + (open_end - open_start) * u;
                let (slot, cap) = if u >= *switch_at {
                    (*slot_after, *cap_after)
                } else {
                    (*slot_before, *cap_before)
                };
                (*q, g, 0.0, slot, false, false, 0.0, cap)
            }

            SegmentKind::CapSwitchHold {
                q,
                held_can_slot,
                cap_before,
                cap_after,
                switch_at,
            } => {
                let u = (t_local / seg_dur).clamp(0.0, 1.0) as f32;
                let cap = if u >= *switch_at {
                    *cap_after
                } else {
                    *cap_before
                };
                (*q, 0.0, 0.0, *held_can_slot, false, false, 0.0, cap)
            }

            SegmentKind::ShakeMove {
                q_base,
                pitch_bounds,
                roll_bounds,
                held_can_slot,
                active_cap,
            } => {
                let ramp = 0.15;
                let env = if t_local < ramp {
                    0.5 * (1.0 - (PI * t_local / ramp).cos())
                } else if t_local > seg_dur - ramp {
                    let rem = (seg_dur - t_local).max(0.0);
                    0.5 * (1.0 - (PI * rem / ramp).cos())
                } else {
                    1.0
                };
                let phase = 2.0 * PI * SHAKE_FREQ_HZ * t_local;
                let mut q = *q_base;
                q[6] = (q_base[6] + env * SHAKE_AMP_RAD * phase.sin())
                    .clamp(pitch_bounds.0, pitch_bounds.1);
                q[5] = (q_base[5] + env * 0.85 * SHAKE_AMP_RAD * (phase + 0.35).sin())
                    .clamp(roll_bounds.0, roll_bounds.1);
                (q, 0.0, 0.0, *held_can_slot, false, true, 0.0, *active_cap)
            }

            SegmentKind::ArcScaledPath {
                samples,
                profile,
                gripper_open,
                held_can_slot,
                active_cap,
            } => {
                let total_s = samples.last().map(|s| s.s).unwrap_or(0.0);
                let target_s = profile.eval(t_local) * total_s;
                let (q, _, _, _) = interpolate_samples_by_s(samples, target_s);
                (q, *gripper_open, 0.0, *held_can_slot, false, false, 0.0, *active_cap)
            }

            SegmentKind::TimedStroke {
                samples,
                gripper_open,
                held_can_slot,
            } => {
                let (q, valve_on, opacity, cap) = interpolate_samples_by_t(samples, t_local);
                let trig = if valve_on { 1.0 } else { 0.0 };
                let flow = if valve_on { opacity } else { 0.0 };
                (q, *gripper_open, trig, *held_can_slot, valve_on, false, flow, cap)
            }
        };

        let (nozzle_world_pos, nozzle_world_dir, _) = self.chain.fk_nozzle(&joints);
        RobotFrameState {
            joints,
            gripper_open,
            trigger_depressed,
            held_can_slot,
            nozzle_world_pos,
            nozzle_world_dir,
            spray_active,
            is_shaking,
            flow_scale,
            active_cap,
        }
    }
}

/// Evaluates minimum world Z clearance along a joint-space linear interpolation `q_a -> q_b`.
fn min_joint_path_z(
    chain: &KinematicChain,
    q_a: &[f64; 9],
    q_b: &[f64; 9],
    held_can: bool,
    steps: usize,
) -> f32 {
    let mut min_z = f32::INFINITY;
    for k in 0..=steps {
        let u = k as f64 / steps as f64;
        let mut q = [0.0; 9];
        for i in 0..9 {
            q[i] = q_a[i] + (q_b[i] - q_a[i]) * u;
        }
        min_z = min_z.min(chain.min_world_z(&q, held_can));
    }
    min_z
}

#[allow(clippy::too_many_arguments)]
fn push_raw_joint_move(
    chain: &KinematicChain,
    segments: &mut Vec<TimelineSegment>,
    t_cursor: &mut f64,
    q_from: [f64; 9],
    q_to: [f64; 9],
    g_open: f32,
    can_slot: Option<usize>,
    active_cap: NozzleCap,
) {
    let profile = joint_move_profile(chain, &q_from, &q_to);
    if profile.duration > 1e-6 {
        let t_end = *t_cursor + profile.duration;
        segments.push(TimelineSegment {
            t_start: *t_cursor,
            t_end,
            kind: SegmentKind::JointMove {
                q_start: q_from,
                q_end: q_to,
                profile,
                gripper_open: g_open,
                held_can_slot: can_slot,
                active_cap,
            },
        });
        *t_cursor = t_end;
    }
}

/// Pushes a joint-space segment, staging through a shoulder-retracted posture (`q2 = +0.45`)
/// whenever direct interpolation would bring any link or held can within `0.06m` of `Z = 0`.
#[allow(clippy::too_many_arguments)]
fn push_checked_joint_move(
    chain: &KinematicChain,
    segments: &mut Vec<TimelineSegment>,
    t_cursor: &mut f64,
    q_from: [f64; 9],
    q_to: [f64; 9],
    g_open: f32,
    can_slot: Option<usize>,
    active_cap: NozzleCap,
) {
    let held_can = can_slot.is_some();
    let end_floor = chain
        .min_world_z(&q_from, held_can)
        .min(chain.min_world_z(&q_to, held_can))
        .min(0.06)
        .max(0.012)
        - 0.002;
    if min_joint_path_z(chain, &q_from, &q_to, held_can, 40) >= end_floor.max(0.04) {
        push_raw_joint_move(
            chain, segments, t_cursor, q_from, q_to, g_open, can_slot, active_cap,
        );
        return;
    }

    // At q2 = +0.45 (shoulder pitched back), wrist Z >= 0.64m regardless of q4/q6.
    // Stage: (optional Cartesian +Z retract) -> q2 = 0.45 -> move other joints at q2 = 0.45 -> move q2 to q_to[2].
    let mut q_cur = q_from;
    let q_retract_a = chain.clamp_joints(&[
        q_cur[0], q_cur[1], 0.45, q_cur[3], q_cur[4], q_cur[5], q_cur[6], q_cur[7], q_cur[8],
    ]);
    if min_joint_path_z(chain, &q_cur, &q_retract_a, held_can, 40) < end_floor {
        let (noz, dir, _) = chain.fk_nozzle(&q_cur);
        if dir.z < -0.4 && noz.z < 0.46 {
            let p_safe = Vec3::new(noz.x, noz.y, 0.50_f32.max(noz.z + 0.14));
            if let Ok((seg, q_safe)) = build_cartesian_nozzle_segment(
                chain, q_cur, noz, p_safe, dir, dir, 0.75, 2.0, g_open, can_slot, active_cap,
                *t_cursor,
            ) {
                *t_cursor = seg.t_end;
                segments.push(seg);
                q_cur = q_safe;
            }
        }
    }

    let q_back_from = chain.clamp_joints(&[
        q_cur[0], q_cur[1], 0.45, q_cur[3], q_cur[4], q_cur[5], q_cur[6], q_cur[7], q_cur[8],
    ]);
    let q_back_to = chain.clamp_joints(&[
        q_to[0], q_to[1], 0.45, q_to[3], q_to[4], q_to[5], q_to[6], q_to[7], q_to[8],
    ]);
    let z_1 = min_joint_path_z(chain, &q_cur, &q_back_from, held_can, 40);
    let z_2 = min_joint_path_z(chain, &q_back_from, &q_back_to, held_can, 40);
    let z_3 = min_joint_path_z(chain, &q_back_to, &q_to, held_can, 40);
    if z_1 >= end_floor && z_2 >= 0.06 && z_3 >= end_floor {
        push_raw_joint_move(
            chain, segments, t_cursor, q_cur, q_back_from, g_open, can_slot, active_cap,
        );
        push_raw_joint_move(
            chain, segments, t_cursor, q_back_from, q_back_to, g_open, can_slot, active_cap,
        );
        push_raw_joint_move(
            chain, segments, t_cursor, q_back_to, q_to, g_open, can_slot, active_cap,
        );
        return;
    }

    // Fallback: tuck via PARK_HOME arm posture at q2 = 0.45
    let q_park_from = chain.clamp_joints(&[
        q_cur[0], q_cur[1], 0.45, 0.0, 1.10, 0.0, -1.25, 0.0, 0.0,
    ]);
    let q_park_to = chain.clamp_joints(&[
        q_to[0], q_to[1], 0.45, 0.0, 1.10, 0.0, -1.25, 0.0, 0.0,
    ]);
    let pz1 = min_joint_path_z(chain, &q_cur, &q_park_from, held_can, 40);
    let pz2 = min_joint_path_z(chain, &q_park_from, &q_park_to, held_can, 40);
    let pz3 = min_joint_path_z(chain, &q_park_to, &q_to, held_can, 40);
    if pz1.min(pz2).min(pz3) > min_joint_path_z(chain, &q_cur, &q_to, held_can, 40) {
        push_raw_joint_move(
            chain, segments, t_cursor, q_cur, q_park_from, g_open, can_slot, active_cap,
        );
        push_raw_joint_move(
            chain, segments, t_cursor, q_park_from, q_park_to, g_open, can_slot, active_cap,
        );
        push_raw_joint_move(
            chain, segments, t_cursor, q_park_to, q_to, g_open, can_slot, active_cap,
        );
        return;
    }

    push_raw_joint_move(
        chain, segments, t_cursor, q_cur, q_to, g_open, can_slot, active_cap,
    );
}

/// Pushes a collision-free joint-space transit, retracting along Cartesian `+Z` when near the wall
/// and folding into a high-clearance posture whenever direct joint interpolation would dip (`min_world_z < 0.18m`).
#[allow(clippy::too_many_arguments)]
fn push_safe_joint_move(
    chain: &KinematicChain,
    segments: &mut Vec<TimelineSegment>,
    t_cursor: &mut f64,
    q_from: [f64; 9],
    q_to: [f64; 9],
    g_open: f32,
    can_slot: Option<usize>,
    active_cap: NozzleCap,
) {
    let held_can = can_slot.is_some();
    if min_joint_path_z(chain, &q_from, &q_to, held_can, 40) >= 0.18 {
        push_raw_joint_move(
            chain, segments, t_cursor, q_from, q_to, g_open, can_slot, active_cap,
        );
        return;
    }

    let tuck_arm = |q: &[f64; 9]| -> [f64; 9] {
        chain.clamp_joints(&[q[0], q[1], 0.15, q[3], 1.10, q[5], -1.25, q[7], q[8]])
    };

    let mut q_cur = q_from;
    let (from_noz, from_dir, _) = chain.fk_nozzle(&q_from);
    if from_dir.z < -0.5 && from_noz.z < 0.46 {
        let safe_z = 0.50_f32.max(from_noz.z + 0.14);
        let p_safe_from = Vec3::new(from_noz.x, from_noz.y, safe_z);
        if let Ok((seg, q_safe_from)) = build_cartesian_nozzle_segment(
            chain,
            q_cur,
            from_noz,
            p_safe_from,
            from_dir,
            from_dir,
            0.75,
            2.0,
            g_open,
            can_slot,
            active_cap,
            *t_cursor,
        ) {
            if min_joint_path_z(chain, &q_cur, &q_safe_from, held_can, 20) >= 0.02 {
                *t_cursor = seg.t_end;
                segments.push(seg);
                q_cur = q_safe_from;
            }
        }
    }

    if min_joint_path_z(chain, &q_cur, &q_to, held_can, 40) >= 0.18 {
        push_raw_joint_move(
            chain, segments, t_cursor, q_cur, q_to, g_open, can_slot, active_cap,
        );
        return;
    }

    // If destination is near the wall, stage through a high-Z pose above it and descend linearly in -Z.
    let (to_noz, to_dir, _) = chain.fk_nozzle(&q_to);
    let mut descend_seg: Option<(TimelineSegment, [f64; 9])> = None;
    let mut q_target = q_to;
    if to_dir.z < -0.5 && to_noz.z < 0.46 {
        let safe_z = 0.50_f32.max(to_noz.z + 0.14);
        let p_safe_to = Vec3::new(to_noz.x, to_noz.y, safe_z);
        if let Ok(q_safe_to) = chain.solve_ik(
            &q_to,
            p_safe_to,
            to_dir,
            IkMode::Spray { roll_weight: 0.15 },
        ) {
            if let Ok((seg, q_desc_end)) = build_cartesian_nozzle_segment(
                chain,
                q_safe_to,
                p_safe_to,
                to_noz,
                to_dir,
                to_dir,
                0.75,
                2.0,
                g_open,
                can_slot,
                active_cap,
                0.0,
            ) {
                let end_floor = chain.min_world_z(&q_to, held_can).min(0.06) - 0.002;
                if min_joint_path_z(chain, &q_desc_end, &q_to, held_can, 16) >= end_floor {
                    q_target = q_safe_to;
                    descend_seg = Some((seg, q_desc_end));
                }
            }
        }
    }

    if min_joint_path_z(chain, &q_cur, &q_target, held_can, 40) >= 0.18 {
        push_raw_joint_move(
            chain, segments, t_cursor, q_cur, q_target, g_open, can_slot, active_cap,
        );
    } else {
        let q_fold_cur = tuck_arm(&q_cur);
        push_checked_joint_move(
            chain, segments, t_cursor, q_cur, q_fold_cur, g_open, can_slot, active_cap,
        );

        let q_sweep = chain.clamp_joints(&[
            q_target[0],
            q_target[1].clamp(-1.35, 1.35),
            0.15,
            0.0,
            1.10,
            0.0,
            -1.25,
            0.0,
            0.0,
        ]);
        push_checked_joint_move(
            chain, segments, t_cursor, q_fold_cur, q_sweep, g_open, can_slot, active_cap,
        );

        push_checked_joint_move(
            chain, segments, t_cursor, q_sweep, q_target, g_open, can_slot, active_cap,
        );
    }

    if let Some((mut seg, q_desc_end)) = descend_seg {
        let dt = seg.t_end - seg.t_start;
        seg.t_start = *t_cursor;
        seg.t_end = *t_cursor + dt;
        *t_cursor = seg.t_end;
        segments.push(seg);
        push_raw_joint_move(
            chain, segments, t_cursor, q_desc_end, q_to, g_open, can_slot, active_cap,
        );
    }
}

/// Computes synchronized joint-space S-curve profile strictly respecting all joint velocity & accel limits.
fn joint_move_profile(chain: &KinematicChain, q_a: &[f64; 9], q_b: &[f64; 9]) -> SCurveProfile {
    let mut max_dq = 0.0_f64;
    let mut s_v_lim = f64::INFINITY;
    let mut s_a_lim = f64::INFINITY;

    for i in 0..9 {
        let dq = (q_b[i] - q_a[i]).abs();
        if dq > 1e-8 {
            max_dq = max_dq.max(dq);
            s_v_lim = s_v_lim.min(chain.joints[i].max_vel * 0.92 / dq);
            s_a_lim = s_a_lim.min(chain.joints[i].max_accel * 0.90 / dq);
        }
    }

    if max_dq < 1e-8 {
        return SCurveProfile::new(0.0, 1.0, 1.0);
    }
    SCurveProfile::new(1.0, s_v_lim, s_a_lim)
}

/// Computes effective normalized S-curve limits along a path from joint deltas and Cartesian limits.
fn path_scurve_profile(
    chain: &KinematicChain,
    samples: &[PathSample],
    v_cart_max: f64,
    a_cart_max: f64,
) -> SCurveProfile {
    let total_s = samples.last().map(|s| s.s).unwrap_or(0.0);
    if total_s < 1e-7 || samples.len() < 2 {
        return SCurveProfile::new(0.0, 1.0, 1.0);
    }

    let mut v_s_lim = v_cart_max;
    let mut a_s_lim = a_cart_max;

    for pair in samples.windows(2) {
        let ds = (pair[1].s - pair[0].s).max(1e-6);
        for i in 0..9 {
            let dq = (pair[1].joints[i] - pair[0].joints[i]).abs();
            if dq > 1e-8 {
                let ratio = dq / ds;
                v_s_lim = v_s_lim.min(chain.joints[i].max_vel * 0.92 / ratio);
                a_s_lim = a_s_lim.min(chain.joints[i].max_accel * 0.90 / ratio);
            }
        }
    }

    SCurveProfile::new(total_s, v_s_lim, a_s_lim)
}

#[allow(clippy::too_many_arguments)]
fn build_cartesian_grasp_segment(
    chain: &KinematicChain,
    q_start: [f64; 9],
    pos_start: Vec3,
    pos_end: Vec3,
    rot: glam::Quat,
    v_cart_max: f64,
    a_cart_max: f64,
    gripper_open: f32,
    held_can_slot: Option<usize>,
    active_cap: NozzleCap,
    t_start: f64,
) -> Result<(TimelineSegment, [f64; 9]), String> {
    let dist = (pos_end - pos_start).length() as f64;
    let steps = ((dist / 0.008).ceil() as usize).clamp(4, 200);

    let mut samples = Vec::with_capacity(steps + 1);
    samples.push(PathSample {
        s: 0.0,
        t: 0.0,
        joints: q_start,
        valve_on: false,
        opacity_scale: 0.0,
        cap: active_cap,
    });
    let mut q_prev = q_start;
    for k in 1..=steps {
        let u = k as f32 / steps as f32;
        let p = pos_start.lerp(pos_end, u);
        let q_k = chain.solve_ik_grasp(&q_prev, p, rot)?;
        samples.push(PathSample {
            s: dist * (u as f64),
            t: 0.0,
            joints: q_k,
            valve_on: false,
            opacity_scale: 0.0,
            cap: active_cap,
        });
        q_prev = q_k;
    }

    let profile = path_scurve_profile(chain, &samples, v_cart_max, a_cart_max);
    let t_end = t_start + profile.duration.max(1e-4);
    Ok((
        TimelineSegment {
            t_start,
            t_end,
            kind: SegmentKind::ArcScaledPath {
                samples,
                profile,
                gripper_open,
                held_can_slot,
                active_cap,
            },
        },
        q_prev,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_cartesian_nozzle_segment(
    chain: &KinematicChain,
    q_start: [f64; 9],
    pos_start: Vec3,
    pos_end: Vec3,
    dir_start: Vec3,
    dir_end: Vec3,
    v_cart_max: f64,
    a_cart_max: f64,
    gripper_open: f32,
    held_can_slot: Option<usize>,
    active_cap: NozzleCap,
    t_start: f64,
) -> Result<(TimelineSegment, [f64; 9]), String> {
    let dist = (pos_end - pos_start).length() as f64;
    let steps = ((dist / 0.004).ceil() as usize).clamp(4, 300);

    let mut samples = Vec::with_capacity(steps + 1);
    samples.push(PathSample {
        s: 0.0,
        t: 0.0,
        joints: q_start,
        valve_on: false,
        opacity_scale: 0.0,
        cap: active_cap,
    });
    let mut q_prev = q_start;
    for k in 1..=steps {
        let u = k as f32 / steps as f32;
        let p = pos_start.lerp(pos_end, u);
        let d = dir_start.lerp(dir_end, u).normalize_or(Vec3::NEG_Z);
        let (q_k, reconfigured) =
            chain.solve_ik_continuous(&q_prev, p, d, IkMode::Spray { roll_weight: 0.15 })?;
        let max_dq = (0..9)
            .map(|dof| (q_k[dof] - q_prev[dof]).abs())
            .fold(0.0_f64, f64::max);
        if reconfigured || max_dq > 0.25 {
            return Err("Cartesian nozzle path requires joint reconfiguration".to_string());
        }
        samples.push(PathSample {
            s: dist * (u as f64),
            t: 0.0,
            joints: q_k,
            valve_on: false,
            opacity_scale: 0.0,
            cap: active_cap,
        });
        q_prev = q_k;
    }

    let profile = path_scurve_profile(chain, &samples, v_cart_max, a_cart_max);
    let t_end = t_start + profile.duration.max(1e-4);
    Ok((
        TimelineSegment {
            t_start,
            t_end,
            kind: SegmentKind::ArcScaledPath {
                samples,
                profile,
                gripper_open,
                held_can_slot,
                active_cap,
            },
        },
        q_prev,
    ))
}

fn finalize_timed_stroke(
    chain: &KinematicChain,
    mut samples: Vec<PathSample>,
    dense_pos: &[Vec3],
    target_speeds: Vec<f64>,
    gripper_open: f32,
    held_can_slot: Option<usize>,
    t_start: f64,
) -> Option<TimelineSegment> {
    if samples.len() < 2 {
        return None;
    }

    // Forward-backward velocity planning respecting joint limits and corners
    let n = samples.len();
    let mut v_lim = target_speeds;
    v_lim[0] = 0.0;
    v_lim[n - 1] = 0.0;

    let mut seg_a_lim = vec![2.2_f64; n - 1];
    for j in 0..n - 1 {
        let ds = (samples[j + 1].s - samples[j].s).max(1e-6);
        for dof in 0..9 {
            let dq = (samples[j + 1].joints[dof] - samples[j].joints[dof]).abs();
            if dq > 1e-8 {
                let ratio = dq / ds;
                let v_j = chain.joints[dof].max_vel * 0.90 / ratio;
                let a_j = chain.joints[dof].max_accel * 0.88 / ratio;
                v_lim[j] = v_lim[j].min(v_j);
                v_lim[j + 1] = v_lim[j + 1].min(v_j);
                seg_a_lim[j] = seg_a_lim[j].min(a_j);
            }
        }
    }

    // Slow smoothly at sharp direction changes (> ~15 deg)
    for j in 1..n - 1 {
        let d1 = (dense_pos[j] - dense_pos[j - 1]).normalize_or_zero();
        let d2 = (dense_pos[j + 1] - dense_pos[j]).normalize_or_zero();
        let cos_a = d1.dot(d2).clamp(-1.0, 1.0) as f64;
        if cos_a < 0.96 {
            let corner_factor = ((1.0 + cos_a) * 0.5).max(0.0).sqrt();
            v_lim[j] *= corner_factor.max(0.28);
        }
    }

    // Forward pass
    let mut v = vec![0.0_f64; n];
    for j in 0..n - 1 {
        let ds = (samples[j + 1].s - samples[j].s).max(1e-7);
        let v_next = (v[j] * v[j] + 2.0 * seg_a_lim[j] * ds).sqrt();
        v[j + 1] = v_lim[j + 1].min(v_next);
    }
    // Backward pass
    for j in (0..n - 1).rev() {
        let ds = (samples[j + 1].s - samples[j].s).max(1e-7);
        let v_prev = (v[j + 1] * v[j + 1] + 2.0 * seg_a_lim[j] * ds).sqrt();
        v[j] = v[j].min(v_prev);
    }

    // Integrate timestamps
    samples[0].t = 0.0;
    for j in 0..n - 1 {
        let ds = (samples[j + 1].s - samples[j].s).max(1e-7);
        let v_avg = (0.5 * (v[j] + v[j + 1])).max(0.02);
        samples[j + 1].t = samples[j].t + ds / v_avg;
    }

    let duration = samples[n - 1].t;
    if duration <= 1e-6 {
        return None;
    }
    Some(TimelineSegment {
        t_start,
        t_end: t_start + duration,
        kind: SegmentKind::TimedStroke {
            samples,
            gripper_open,
            held_can_slot,
        },
    })
}

fn build_timed_stroke_segments(
    chain: &KinematicChain,
    q_start: [f64; 9],
    waypoints: &[StrokeWaypoint],
    gripper_open: f32,
    held_can_slot: Option<usize>,
    segments: &mut Vec<TimelineSegment>,
    t_cursor: &mut f64,
) -> Result<[f64; 9], String> {
    let mut dense_pos: Vec<Vec3> = Vec::new();
    let mut samples: Vec<PathSample> = Vec::new();
    let mut target_speeds: Vec<f64> = Vec::new();

    let mut q_prev = q_start;
    let mut prev_p = waypoints[0].nozzle_pos();
    let mut prev_d = waypoints[0].spray_dir();
    let mut prev_cap = waypoints[0].cap;
    let mut s_acc = 0.0_f64;
    let mut need_initial_sample = true;

    for seg_idx in 0..waypoints.len() - 1 {
        let wp_a = waypoints[seg_idx];
        let wp_b = waypoints[seg_idx + 1];
        let p_a = wp_a.nozzle_pos();
        let p_b = wp_b.nozzle_pos();
        let d_a = wp_a.spray_dir();
        let d_b = wp_b.spray_dir();

        let seg_len = (p_b - p_a).length() as f64;
        if seg_len < 1e-6 {
            continue;
        }

        let sub_steps = ((seg_len / 0.005).ceil() as usize).max(1);
        let seg_valve_on = wp_a.valve_on && wp_b.valve_on;
        let seg_speed = (0.5 * (wp_a.feed_speed + wp_b.feed_speed)).clamp(0.05, 1.2) as f64;
        let seg_opacity = 0.5 * (wp_a.opacity_scale + wp_b.opacity_scale);

        let start_k = if need_initial_sample { 0 } else { 1 };
        for k in start_k..=sub_steps {
            let u = k as f32 / sub_steps as f32;
            let p = p_a.lerp(p_b, u);
            let d = d_a.lerp(d_b, u).normalize_or(Vec3::NEG_Z);
            let cap_k = if k == 0 { wp_a.cap } else { wp_b.cap };

            if need_initial_sample {
                s_acc = 0.0;
                dense_pos.push(p);
                target_speeds.push(seg_speed);
                samples.push(PathSample {
                    s: 0.0,
                    t: 0.0,
                    joints: q_prev,
                    valve_on: seg_valve_on,
                    opacity_scale: seg_opacity,
                    cap: cap_k,
                });
                prev_p = p;
                prev_d = d;
                prev_cap = cap_k;
                need_initial_sample = false;
                continue;
            }

            let (q_k, reconfigured) =
                chain.solve_ik_continuous(&q_prev, p, d, IkMode::Spray { roll_weight: 0.15 })?;
            let max_dq = (0..9)
                .map(|dof| (q_k[dof] - q_prev[dof]).abs())
                .fold(0.0_f64, f64::max);

            // HARD ANTI-SHAKING RULE: never interpolate across a reconfiguration or cap swap while valve_on
            if reconfigured || max_dq > 0.25 || cap_k != prev_cap {
                if let Some(stroke_seg) = finalize_timed_stroke(
                    chain,
                    std::mem::take(&mut samples),
                    &dense_pos,
                    std::mem::take(&mut target_speeds),
                    gripper_open,
                    held_can_slot,
                    *t_cursor,
                ) {
                    *t_cursor = stroke_seg.t_end;
                    segments.push(stroke_seg);
                } else {
                    samples.clear();
                    target_speeds.clear();
                }
                dense_pos.clear();

                // 2. Retract +Z with valve_on = false
                let mut q_curr = q_prev;
                let p_safe_prev = Vec3::new(prev_p.x, prev_p.y, prev_p.z.max(SAFE_TRANSIT_Z));
                if (p_safe_prev - prev_p).length() > 1e-3 {
                    if let Ok((ret_seg, q_ret)) = build_cartesian_nozzle_segment(
                        chain,
                        q_curr,
                        prev_p,
                        p_safe_prev,
                        prev_d,
                        prev_d,
                        0.45,
                        1.4,
                        gripper_open,
                        held_can_slot,
                        prev_cap,
                        *t_cursor,
                    ) {
                        *t_cursor = ret_seg.t_end;
                        segments.push(ret_seg);
                        q_curr = q_ret;
                    } else {
                        let q_ret = chain.solve_ik(
                            &q_curr,
                            p_safe_prev,
                            prev_d,
                            IkMode::Spray { roll_weight: 0.15 },
                        )?;
                        push_safe_joint_move(
                            chain,
                            segments,
                            t_cursor,
                            q_curr,
                            q_ret,
                            gripper_open,
                            held_can_slot,
                            prev_cap,
                        );
                        q_curr = q_ret;
                    }
                }

                // 3. Safe joint move (valve OFF) to reconfigured safe-Z pose
                let p_safe_next = Vec3::new(p.x, p.y, p.z.max(SAFE_TRANSIT_Z));
                let q_safe_next = chain.solve_ik(
                    &q_curr,
                    p_safe_next,
                    d,
                    IkMode::Spray { roll_weight: 0.15 },
                )?;
                push_safe_joint_move(
                    chain,
                    segments,
                    t_cursor,
                    q_curr,
                    q_safe_next,
                    gripper_open,
                    held_can_slot,
                    cap_k,
                );
                q_curr = q_safe_next;

                // 4. Descend -Z (valve OFF) back to (p, d) and resume fresh TimedStroke
                if (p_safe_next - p).length() > 1e-3 {
                    if let Ok((desc_seg, q_desc)) = build_cartesian_nozzle_segment(
                        chain,
                        q_curr,
                        p_safe_next,
                        p,
                        d,
                        d,
                        0.45,
                        1.4,
                        gripper_open,
                        held_can_slot,
                        cap_k,
                        *t_cursor,
                    ) {
                        *t_cursor = desc_seg.t_end;
                        segments.push(desc_seg);
                        q_curr = q_desc;
                    } else {
                        let q_desc = chain.solve_ik(
                            &q_curr,
                            p,
                            d,
                            IkMode::Spray { roll_weight: 0.15 },
                        )?;
                        push_safe_joint_move(
                            chain,
                            segments,
                            t_cursor,
                            q_curr,
                            q_desc,
                            gripper_open,
                            held_can_slot,
                            cap_k,
                        );
                        q_curr = q_desc;
                    }
                }

                q_prev = q_curr;
                prev_p = p;
                prev_d = d;
                prev_cap = cap_k;
                s_acc = 0.0;
                dense_pos.push(p);
                target_speeds.push(seg_speed);
                samples.push(PathSample {
                    s: 0.0,
                    t: 0.0,
                    joints: q_prev,
                    valve_on: seg_valve_on,
                    opacity_scale: seg_opacity,
                    cap: cap_k,
                });
            } else {
                s_acc += (p - prev_p).length() as f64;
                dense_pos.push(p);
                target_speeds.push(seg_speed);
                samples.push(PathSample {
                    s: s_acc,
                    t: 0.0,
                    joints: q_k,
                    valve_on: seg_valve_on,
                    opacity_scale: seg_opacity,
                    cap: cap_k,
                });
                q_prev = q_k;
                prev_p = p;
                prev_d = d;
                prev_cap = cap_k;
            }
        }
    }

    if let Some(stroke_seg) = finalize_timed_stroke(
        chain,
        samples,
        &dense_pos,
        target_speeds,
        gripper_open,
        held_can_slot,
        *t_cursor,
    ) {
        *t_cursor = stroke_seg.t_end;
        segments.push(stroke_seg);
    }

    Ok(q_prev)
}

fn interpolate_samples_by_s(
    samples: &[PathSample],
    target_s: f64,
) -> ([f64; 9], bool, f32, NozzleCap) {
    if samples.is_empty() {
        return ([0.0; 9], false, 0.0, NozzleCap::Fat);
    }
    if samples.len() == 1 || target_s <= samples[0].s {
        return (
            samples[0].joints,
            samples[0].valve_on,
            samples[0].opacity_scale,
            samples[0].cap,
        );
    }
    let last = &samples[samples.len() - 1];
    if target_s >= last.s {
        return (last.joints, last.valve_on, last.opacity_scale, last.cap);
    }
    let idx = samples.partition_point(|smp| smp.s <= target_s);
    let i0 = idx.saturating_sub(1);
    let i1 = idx.min(samples.len() - 1);
    let s0 = samples[i0].s;
    let s1 = samples[i1].s;
    let u = if (s1 - s0).abs() > 1e-9 {
        ((target_s - s0) / (s1 - s0)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut q = [0.0; 9];
    for dof in 0..9 {
        q[dof] = samples[i0].joints[dof] + (samples[i1].joints[dof] - samples[i0].joints[dof]) * u;
    }
    (q, samples[i1].valve_on, samples[i1].opacity_scale, samples[i1].cap)
}

fn interpolate_samples_by_t(
    samples: &[PathSample],
    t_local: f64,
) -> ([f64; 9], bool, f32, NozzleCap) {
    if samples.is_empty() {
        return ([0.0; 9], false, 0.0, NozzleCap::Fat);
    }
    if samples.len() == 1 || t_local <= samples[0].t {
        return (
            samples[0].joints,
            samples[0].valve_on,
            samples[0].opacity_scale,
            samples[0].cap,
        );
    }
    let last = &samples[samples.len() - 1];
    if t_local >= last.t {
        return (last.joints, last.valve_on, last.opacity_scale, last.cap);
    }
    let idx = samples.partition_point(|smp| smp.t <= t_local);
    let i0 = idx.saturating_sub(1);
    let i1 = idx.min(samples.len() - 1);
    let t0 = samples[i0].t;
    let t1 = samples[i1].t;
    let u = if (t1 - t0).abs() > 1e-9 {
        ((t_local - t0) / (t1 - t0)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut q = [0.0; 9];
    for dof in 0..9 {
        q[dof] = samples[i0].joints[dof] + (samples[i1].joints[dof] - samples[i0].joints[dof]) * u;
    }
    (q, samples[i1].valve_on, samples[i1].opacity_scale, samples[i1].cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_full_choreography_continuity_and_state_transitions() {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);

        let stroke_waypoints = vec![
            StrokeWaypoint {
                pos: Vec3::new(-0.435, 1.55, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.10,
                feed_speed: 0.40,
                valve_on: false,
                opacity_scale: 1.0,
                cap: NozzleCap::Fat,
            },
            StrokeWaypoint {
                pos: Vec3::new(-0.40, 1.55, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.10,
                feed_speed: 0.40,
                valve_on: true,
                opacity_scale: 0.9,
                cap: NozzleCap::Fat,
            },
            StrokeWaypoint {
                pos: Vec3::new(0.40, 1.55, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.10,
                feed_speed: 0.40,
                valve_on: true,
                opacity_scale: 0.9,
                cap: NozzleCap::Fat,
            },
            StrokeWaypoint {
                pos: Vec3::new(0.435, 1.55, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.10,
                feed_speed: 0.40,
                valve_on: false,
                opacity_scale: 1.0,
                cap: NozzleCap::Fat,
            },
        ];

        let job = JobPlan {
            ops: vec![
                JobOp::PickCan { slot: 2 },
                JobOp::ShakeCan,
                JobOp::PaintStroke {
                    points_canvas_3d: stroke_waypoints,
                },
                JobOp::ReturnCan { slot: 2 },
                JobOp::ParkHome,
            ],
        };

        let timeline = CompiledTimeline::compile(&job, &chain, &studio)
            .expect("Full choreography should compile cleanly");
        assert!(timeline.duration() > 3.0);

        // Initial and final states
        let s0 = timeline.sample(0.0);
        assert_eq!(s0.held_can_slot, None);
        assert!((s0.gripper_open - 1.0).abs() < 1e-4);
        assert!(!s0.spray_active);
        assert!(!s0.is_shaking);
        for i in 0..9 {
            assert!((s0.joints[i] - timeline.home_joints()[i]).abs() < 1e-6);
        }

        let s_end = timeline.sample(timeline.duration());
        assert_eq!(s_end.held_can_slot, None);
        assert!((s_end.gripper_open - 1.0).abs() < 1e-4);
        assert!(!s_end.spray_active);
        assert!(!s_end.is_shaking);
        for i in 0..9 {
            assert!((s_end.joints[i] - timeline.home_joints()[i]).abs() < 1e-5);
        }

        // Dense sweep verifying C0/C1 continuity, joint limits, and state transitions
        let dt = 0.005;
        let steps = (timeline.duration() / dt).ceil() as usize;
        let mut prev = timeline.sample(0.0);
        let mut saw_held_slot_2 = false;
        let mut saw_spray_on = false;
        let mut saw_shaking = false;
        let mut saw_closed_gripper = false;

        for k in 1..=steps {
            let t = (k as f64 * dt).min(timeline.duration());
            let cur = timeline.sample(t);

            assert!(
                chain.within_limits(&cur.joints, 1e-6),
                "Joint limit exceeded at t={t:.3}: {:?}",
                cur.joints
            );

            // C0 continuity: bounded step per dt
            for i in 0..9 {
                let dq = (cur.joints[i] - prev.joints[i]).abs();
                let max_step = if i == 5 || i == 6 {
                    // During 6 Hz ±22° shake, peak velocity ~ 14.5 rad/s
                    16.0 * dt
                } else {
                    chain.joints[i].max_vel * 1.15 * dt
                };
                assert!(
                    dq <= max_step + 1e-4,
                    "Discontinuity on joint {i} at t={t:.3}: dq={dq:.5} > {max_step:.5}"
                );
            }

            if cur.held_can_slot == Some(2) {
                saw_held_slot_2 = true;
            }
            if cur.gripper_open < 0.05 {
                saw_closed_gripper = true;
            }
            if cur.is_shaking {
                saw_shaking = true;
                assert!(!cur.spray_active, "Cannot spray while shaking");
            }
            if cur.spray_active {
                saw_spray_on = true;
                assert!(!cur.is_shaking, "is_shaking must be false while spraying");
                assert!((cur.trigger_depressed - 1.0).abs() < 1e-5);
                assert_eq!(cur.held_can_slot, Some(2));
                assert!((cur.nozzle_world_pos.z - 0.10).abs() < 0.002);
                assert!(cur.nozzle_world_dir.z < -0.99);
            } else {
                assert!((cur.trigger_depressed - 0.0).abs() < 1e-5);
            }

            prev = cur;
        }

        assert!(saw_held_slot_2, "Never picked up can slot 2");
        assert!(saw_closed_gripper, "Gripper never closed");
        assert!(saw_shaking, "is_shaking was never true during ShakeCan");
        assert!(saw_spray_on, "Spray never activated during PaintStroke");
    }

    #[test]
    fn park_home_pose_clears_wall_canvas_for_unobstructed_view() {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);
        let timeline = CompiledTimeline::empty();
        let home = timeline.home_joints();
        assert!(chain.within_limits(&home, 1e-6));
        assert!(home[0] > studio.canvas_width as f64 * 0.5);
        for (idx, link) in chain.fk_links(&home).iter().enumerate() {
            assert!(
                link.pos.x > studio.canvas_width * 0.5,
                "home link {idx} at x={:.3} overlaps canvas right edge +1.25m",
                link.pos.x
            );
        }
    }

    #[test]
    fn min_world_z_stays_safe_across_pick_swap_strokes_return_park_and_cap_transitions() {
        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);

        let fat_stroke = vec![
            StrokeWaypoint {
                pos: Vec3::new(-1.17, 1.55, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.22,
                feed_speed: 0.45,
                valve_on: false,
                opacity_scale: 1.0,
                cap: NozzleCap::Fat,
            },
            StrokeWaypoint {
                pos: Vec3::new(-1.10, 1.55, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.22,
                feed_speed: 0.45,
                valve_on: true,
                opacity_scale: 0.9,
                cap: NozzleCap::Fat,
            },
            StrokeWaypoint {
                pos: Vec3::new(1.10, 1.55, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.22,
                feed_speed: 0.45,
                valve_on: true,
                opacity_scale: 0.9,
                cap: NozzleCap::Fat,
            },
            StrokeWaypoint {
                pos: Vec3::new(1.17, 1.55, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.22,
                feed_speed: 0.45,
                valve_on: false,
                opacity_scale: 1.0,
                cap: NozzleCap::Fat,
            },
        ];

        let skinny_stroke = vec![
            StrokeWaypoint {
                pos: Vec3::new(0.80, 1.85, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.10,
                feed_speed: 0.35,
                valve_on: false,
                opacity_scale: 1.0,
                cap: NozzleCap::Skinny,
            },
            StrokeWaypoint {
                pos: Vec3::new(0.75, 1.85, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.10,
                feed_speed: 0.35,
                valve_on: true,
                opacity_scale: 1.0,
                cap: NozzleCap::Skinny,
            },
            StrokeWaypoint {
                pos: Vec3::new(-0.75, 1.85, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.10,
                feed_speed: 0.35,
                valve_on: true,
                opacity_scale: 1.0,
                cap: NozzleCap::Skinny,
            },
            StrokeWaypoint {
                pos: Vec3::new(-0.80, 1.85, 0.0),
                normal: Vec3::Z,
                aim_dir: Vec3::NEG_Z,
                standoff: 0.10,
                feed_speed: 0.35,
                valve_on: false,
                opacity_scale: 1.0,
                cap: NozzleCap::Skinny,
            },
        ];

        let job = JobPlan {
            ops: vec![
                JobOp::PickCan { slot: 5 },
                JobOp::ShakeCan,
                JobOp::PaintStroke {
                    points_canvas_3d: fat_stroke,
                },
                JobOp::SwapCap {
                    cap: NozzleCap::Skinny,
                },
                JobOp::PaintStroke {
                    points_canvas_3d: skinny_stroke,
                },
                JobOp::ReturnCan { slot: 5 },
                JobOp::ParkHome,
            ],
        };

        let timeline = CompiledTimeline::compile(&job, &chain, &studio)
            .expect("Pick -> Fat stroke -> SwapCap -> Skinny stroke -> Return -> Park should compile");

        let dt = 0.005;
        let steps = (timeline.duration() / dt).ceil() as usize;
        let mut saw_fat_spray = false;
        let mut saw_skinny_spray = false;
        let mut prev_cap = timeline.sample(0.0).active_cap;
        let mut transitioned_fat_to_skinny = false;
        let mut transitioned_skinny_to_fat = false;

        for k in 0..=steps {
            let t = (k as f64 * dt).min(timeline.duration());
            let frame = timeline.sample(t);
            let min_z = chain.min_world_z(&frame.joints, frame.held_can_slot.is_some());
            assert!(
                min_z > 0.012,
                "Canvas clearance violation at t={t:.3}: min_world_z={min_z:.4}m (held_can={:?}, spray={})",
                frame.held_can_slot,
                frame.spray_active
            );

            if frame.spray_active {
                match frame.active_cap {
                    NozzleCap::Fat => saw_fat_spray = true,
                    NozzleCap::Skinny => saw_skinny_spray = true,
                }
            }
            if prev_cap == NozzleCap::Fat && frame.active_cap == NozzleCap::Skinny {
                assert!(!frame.spray_active, "Cap switched while spraying");
                transitioned_fat_to_skinny = true;
            }
            if prev_cap == NozzleCap::Skinny && frame.active_cap == NozzleCap::Fat {
                assert!(!frame.spray_active, "Cap reset while spraying");
                transitioned_skinny_to_fat = true;
            }
            prev_cap = frame.active_cap;
        }

        assert!(saw_fat_spray, "Expected Fat cap active during first stroke");
        assert!(
            saw_skinny_spray,
            "Expected Skinny cap active during second stroke"
        );
        assert!(
            transitioned_fat_to_skinny,
            "Expected Fat -> Skinny active_cap transition during SwapCap"
        );
        assert!(
            transitioned_skinny_to_fat,
            "Expected Skinny -> Fat active_cap reset on ReturnCan"
        );
        assert_eq!(timeline.sample(timeline.duration()).active_cap, NozzleCap::Fat);
    }
}
