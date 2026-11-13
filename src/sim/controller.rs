use bevy::prelude::*;

use crate::core::types::NozzleCap;
use crate::sim::robot_view::{
    CanCapVisual, CanEntity, GripperJaw, HeldCanSlot, RobotLink, SlotCapVisualState,
    StudioRackCache, TriggerFinger, jaw_local_translation, trigger_local_transform,
};
use crate::sim::state::{AppConfig, SimClock, SimStepSet, SprayJobRes};

pub fn plugin(app: &mut App) {
    app.add_systems(Update, advance_sim_clock.in_set(SimStepSet::InputAndClock))
        .add_systems(
            Update,
            update_robot_kinematics.in_set(SimStepSet::RobotKinematics),
        );
}

fn step_clock(clock: &mut SimClock, dt_wall: f64) {
    if clock.warmup_frames_remaining > 0 {
        clock.warmup_frames_remaining -= 1;
        clock.prev_t_sim = clock.t_sim;
        return;
    }

    if clock.paused || clock.finished {
        return;
    }

    clock.prev_t_sim = clock.t_sim;
    let dt_sim = dt_wall * (clock.speed as f64);

    if clock.t_sim < clock.total_duration {
        clock.t_sim = (clock.t_sim + dt_sim).min(clock.total_duration);
    } else {
        clock.hold_elapsed += dt_wall;
        if clock.hold_elapsed >= clock.hold_secs {
            clock.finished = true;
        }
    }
}

fn advance_sim_clock(
    app_config: Res<AppConfig>,
    time: Res<Time>,
    mut clock: ResMut<SimClock>,
) {
    let dt_wall = if app_config.cli.record.is_some() {
        1.0 / (app_config.cli.fps.max(1) as f64)
    } else {
        time.delta_secs_f64()
    };
    step_clock(&mut clock, dt_wall);
}

#[allow(clippy::too_many_arguments)]
fn update_robot_kinematics(
    clock: Res<SimClock>,
    mut spray_job: ResMut<SprayJobRes>,
    mut held_can: ResMut<HeldCanSlot>,
    mut slot_caps: ResMut<SlotCapVisualState>,
    mut last_reset_epoch: Local<u64>,
    rack_cache: Res<StudioRackCache>,
    mut link_q: Query<(&RobotLink, &mut Transform), (Without<GripperJaw>, Without<TriggerFinger>, Without<CanEntity>)>,
    mut jaw_q: Query<(&GripperJaw, &mut Transform), (Without<RobotLink>, Without<TriggerFinger>, Without<CanEntity>)>,
    mut trigger_q: Query<&mut Transform, (With<TriggerFinger>, Without<RobotLink>, Without<GripperJaw>, Without<CanEntity>)>,
    mut can_q: Query<(&CanEntity, &mut Transform), (Without<RobotLink>, Without<GripperJaw>, Without<TriggerFinger>)>,
    mut cap_vis_q: Query<(&CanCapVisual, &mut Visibility)>,
) {
    if clock.reset_epoch != *last_reset_epoch {
        *last_reset_epoch = clock.reset_epoch;
        slot_caps.0.fill(NozzleCap::Fat);
    }

    let frame = spray_job.timeline.sample(clock.t_sim);
    spray_job.current_frame = frame;
    held_can.0 = frame.held_can_slot;
    if let Some(slot) = frame.held_can_slot
        && let Some(entry) = slot_caps.0.get_mut(slot)
    {
        *entry = frame.active_cap;
    }

    let link_xforms = spray_job.chain.fk_links(&frame.joints);
    for (link, mut tf) in &mut link_q {
        if let Some(lt) = link_xforms.get(link.index) {
            tf.translation = lt.pos;
            tf.rotation = lt.rot;
        }
    }

    for (jaw, mut tf) in &mut jaw_q {
        tf.translation = jaw_local_translation(jaw.side, frame.gripper_open);
    }

    for mut tf in &mut trigger_q {
        *tf = trigger_local_transform(frame.trigger_depressed);
    }

    let grasp_xform = spray_job.chain.fk_grasp(&frame.joints);
    for (can, mut tf) in &mut can_q {
        if frame.held_can_slot == Some(can.slot) {
            tf.translation = grasp_xform.pos;
            tf.rotation = grasp_xform.rot;
        } else if let Some(slot_pose) = rack_cache.slots.get(can.slot) {
            tf.translation = slot_pose.grasp_pos;
            tf.rotation = slot_pose.grasp_rot;
        }
    }

    for (cap_vis, mut vis) in &mut cap_vis_q {
        let mounted_cap = slot_caps.0.get(cap_vis.slot).copied().unwrap_or_default();
        *vis = if cap_vis.cap == mounted_cap {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_waits_warmup_then_advances_and_holds_until_finished() {
        let mut clock = SimClock {
            total_duration: 1.0,
            speed: 2.0,
            hold_secs: 0.5,
            warmup_frames_remaining: 2,
            ..Default::default()
        };

        // Warmup frames do not advance t_sim
        step_clock(&mut clock, 0.25);
        assert_eq!(clock.warmup_frames_remaining, 1);
        assert_eq!(clock.t_sim, 0.0);
        assert_eq!(clock.prev_t_sim, 0.0);

        step_clock(&mut clock, 0.25);
        assert_eq!(clock.warmup_frames_remaining, 0);
        assert_eq!(clock.t_sim, 0.0);

        // First active step: dt_sim = 0.25 * 2.0 = 0.5
        step_clock(&mut clock, 0.25);
        assert!((clock.prev_t_sim - 0.0).abs() < 1e-12);
        assert!((clock.t_sim - 0.5).abs() < 1e-12);
        assert!(!clock.finished);

        // Second active step reaches total_duration = 1.0
        step_clock(&mut clock, 0.30);
        assert!((clock.prev_t_sim - 0.5).abs() < 1e-12);
        assert!((clock.t_sim - 1.0).abs() < 1e-12);
        assert!(!clock.finished);
        assert_eq!(clock.hold_elapsed, 0.0);

        // Hold period accumulates wall dt until hold_secs = 0.5
        step_clock(&mut clock, 0.25);
        assert!((clock.hold_elapsed - 0.25).abs() < 1e-12);
        assert!(!clock.finished);

        step_clock(&mut clock, 0.25);
        assert!((clock.hold_elapsed - 0.50).abs() < 1e-12);
        assert!(clock.finished);
    }

    #[test]
    fn clock_respects_paused_state() {
        let mut clock = SimClock {
            total_duration: 5.0,
            speed: 1.0,
            paused: true,
            warmup_frames_remaining: 0,
            ..Default::default()
        };
        step_clock(&mut clock, 0.1);
        assert_eq!(clock.t_sim, 0.0);
        assert!(!clock.finished);
    }
}
