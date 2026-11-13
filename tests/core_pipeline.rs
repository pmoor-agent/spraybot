use std::path::Path;

use spraybot::core::batching::{BatchConfig, batch_primitives};
use spraybot::core::kinematics::KinematicChain;
use spraybot::core::palette::quantize_palette;
use spraybot::core::planner::{PlannerConfig, plan_job};
use spraybot::core::svg_ingest::{IngestConfig, ingest_svg_file};
use spraybot::core::trajectory::CompiledTimeline;
use spraybot::core::types::{JobOp, StudioConfig};

fn verify_svg_end_to_end(rel_path: &str, max_cans: usize) {
    let studio = StudioConfig::default();
    let chain = KinematicChain::default_spraybot(&studio);

    let primitives = ingest_svg_file(Path::new(rel_path), &IngestConfig::from_studio(&studio))
        .unwrap_or_else(|e| panic!("{rel_path}: ingest failed: {e}"));
    assert!(
        !primitives.is_empty(),
        "{rel_path}: expected non-empty normalized primitives"
    );

    let palette = quantize_palette(&primitives, max_cans, 42);
    assert!(
        !palette.cans.is_empty() && palette.cans.len() <= max_cans,
        "{rel_path}: expected 1..={max_cans} cans, got {}",
        palette.cans.len()
    );
    assert_eq!(palette.primitive_color_map.len(), primitives.len());

    let batches = batch_primitives(&primitives, &palette, &BatchConfig::default());
    assert!(!batches.is_empty(), "{rel_path}: expected non-empty batches");
    let batched_count: usize = batches.iter().map(|b| b.primitive_ids.len()).sum();
    assert_eq!(
        batched_count,
        primitives.len(),
        "{rel_path}: every primitive must be scheduled once"
    );

    let job = plan_job(&primitives, &palette, &batches, &PlannerConfig::default());
    assert!(
        job.ops.iter().any(|op| matches!(op, JobOp::PickCan { .. })),
        "{rel_path}: expected PickCan op"
    );
    assert!(
        job.ops.iter().any(|op| matches!(op, JobOp::ShakeCan)),
        "{rel_path}: expected ShakeCan op"
    );
    assert!(
        job.ops
            .iter()
            .any(|op| matches!(op, JobOp::PaintStroke { .. })),
        "{rel_path}: expected PaintStroke op"
    );
    assert!(
        matches!(job.ops.last(), Some(JobOp::ParkHome)),
        "{rel_path}: expected JobPlan to end with ParkHome"
    );

    let timeline = CompiledTimeline::compile(&job, &chain, &studio)
        .unwrap_or_else(|e| panic!("{rel_path}: trajectory compile failed: {e}"));
    let duration = timeline.duration();
    assert!(
        duration > 1.0,
        "{rel_path}: expected positive trajectory duration > 1s, got {duration}"
    );

    let start = timeline.sample(0.0);
    assert_eq!(start.held_can_slot, None);
    assert!(!start.spray_active);

    let end = timeline.sample(duration);
    assert_eq!(end.held_can_slot, None);
    assert!(!end.spray_active);

    // Sample densely across the full timeline
    let samples = 600usize;
    let mut spray_count = 0usize;
    for i in 0..=samples {
        let t = duration * (i as f64) / (samples as f64);
        let frame = timeline.sample(t);

        for (j, val) in frame.joints.iter().enumerate() {
            assert!(
                val.is_finite(),
                "{rel_path}: non-finite joint[{j}] at t={t:.3}"
            );
        }
        assert!(
            chain.within_limits(&frame.joints, 1e-4),
            "{rel_path}: joint limit violation at t={t:.3}: {:?}",
            frame.joints
        );
        assert!(frame.nozzle_world_pos.is_finite());
        assert!(frame.nozzle_world_dir.is_finite());

        if frame.spray_active {
            spray_count += 1;
            assert!(
                frame.held_can_slot.is_some(),
                "{rel_path}: spraying without held can at t={t:.3}"
            );
            assert!(
                frame.trigger_depressed > 0.5,
                "{rel_path}: trigger not depressed during spray at t={t:.3}"
            );
            assert!(
                frame.flow_scale > 0.0 && frame.flow_scale <= 1.0 + 1e-4,
                "{rel_path}: invalid flow_scale {} at t={t:.3}",
                frame.flow_scale
            );
            assert!(
                frame.nozzle_world_pos.z >= 0.03 && frame.nozzle_world_pos.z <= 0.30,
                "{rel_path}: nozzle Z={} out of spray standoff range at t={t:.3}",
                frame.nozzle_world_pos.z
            );
            assert!(
                frame.nozzle_world_dir.z < -0.90,
                "{rel_path}: nozzle dir {:?} not aimed at canvas (-Z) at t={t:.3}",
                frame.nozzle_world_dir
            );

            // Ray hit onto canvas plane Z = 0.0 must fall within physical canvas bounds
            let t_hit = -frame.nozzle_world_pos.z / frame.nozzle_world_dir.z;
            let hit = frame.nozzle_world_pos + frame.nozzle_world_dir * t_hit;
            assert!(
                hit.x >= -1.25 && hit.x <= 1.25 && hit.y >= 0.80 && hit.y <= 2.30,
                "{rel_path}: spray hit ({:.3}, {:.3}) outside canvas at t={t:.3}",
                hit.x,
                hit.y
            );
        }
    }

    assert!(
        spray_count > 10,
        "{rel_path}: expected substantial active spray samples, got {spray_count}"
    );
}

#[test]
fn core_pipeline_01_geometric_shapes() {
    verify_svg_end_to_end("assets/examples/01_geometric_shapes.svg", 16);
}

#[test]
fn core_pipeline_02_street_art_poster() {
    verify_svg_end_to_end("assets/examples/02_street_art_poster.svg", 16);
}

#[test]
fn core_pipeline_03_palette_stress() {
    verify_svg_end_to_end("assets/examples/03_palette_stress.svg", 16);
}
