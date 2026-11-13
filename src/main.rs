use std::time::Duration;

use bevy::{
    app::ScheduleRunnerPlugin,
    prelude::*,
    window::ExitCondition,
    winit::WinitPlugin,
};
use clap::Parser;

use spraybot::cli::CliArgs;
use spraybot::core::batching::{BatchConfig, batch_primitives};
use spraybot::core::kinematics::KinematicChain;
use spraybot::core::palette::quantize_palette;
use spraybot::core::planner::{PlannerConfig, plan_job};
use spraybot::core::svg_ingest::{IngestConfig, ingest_svg_file};
use spraybot::core::trajectory::CompiledTimeline;
use spraybot::sim::SimPlugin;
use spraybot::sim::state::{AppConfig, OffscreenRenderTarget, SimClock, SprayJobRes};

fn main() {
    let cli = CliArgs::parse();
    if let Err(err) = cli.validate() {
        eprintln!("Invalid CLI options: {err}");
        std::process::exit(2);
    }

    let app_config = AppConfig::new(cli.clone());
    let svg_path = cli.resolved_svg_path();

    let primitives = if svg_path.exists() {
        match ingest_svg_file(&svg_path, &IngestConfig::from_studio(&app_config.studio)) {
            Ok(prims) => prims,
            Err(err) => {
                eprintln!("Failed to load SVG '{}': {err}", svg_path.display());
                std::process::exit(1);
            }
        }
    } else {
        Vec::new()
    };

    let palette = quantize_palette(&primitives, cli.max_cans, cli.seed);
    let batches = batch_primitives(&primitives, &palette, &BatchConfig::default());
    let job = plan_job(&primitives, &palette, &batches, &PlannerConfig::default());
    let chain = KinematicChain::default_spraybot(&app_config.studio);
    let timeline = CompiledTimeline::compile(&job, &chain, &app_config.studio).unwrap_or_else(|err| {
        eprintln!("Failed to compile robot trajectory: {err}");
        std::process::exit(1);
    });
    let total_duration = timeline.duration();
    let current_frame = timeline.sample(0.0);

    let sim_clock = SimClock::from_cli(&cli, total_duration);
    let spray_job = SprayJobRes {
        primitives,
        palette,
        job,
        chain,
        timeline,
        current_frame,
        show_gizmos: false,
    };

    let mut app = App::new();
    app.insert_resource(app_config)
        .insert_resource(sim_clock)
        .insert_resource(spray_job)
        .insert_resource(OffscreenRenderTarget::default());

    if cli.headless {
        let frame_interval = Duration::from_secs_f64(1.0 / cli.fps.max(1) as f64);
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(frame_interval));
    } else {
        let (width, height) = cli.resolution;
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "SprayBot — 3D Mural Simulator".to_string(),
                resolution: (width, height).into(),
                ..default()
            }),
            ..default()
        }));
    }

    app.add_plugins(SimPlugin).run();
}
