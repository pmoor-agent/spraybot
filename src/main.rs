use bevy::{
    app::{RunMode, ScheduleRunnerPlugin},
    prelude::*,
    render::{
        RenderPlugin,
        render_resource::{TextureFormat, TextureUsages},
    },
    window::{ExitCondition, WindowResolution},
    winit::WinitPlugin,
};
use clap::Parser;

use spraybot::cli::{CliArgs, DEFAULT_SVG_PATH};
use spraybot::core::batching::{BatchConfig, batch_primitives};
use spraybot::core::kinematics::KinematicChain;
use spraybot::core::palette::quantize_palette;
use spraybot::core::planner::{PlannerConfig, plan_job};
use spraybot::core::svg_ingest::{IngestConfig, ingest_svg, ingest_svg_file};
use spraybot::core::trajectory::CompiledTimeline;
use spraybot::sim::SimPlugin;
use spraybot::sim::state::{AppConfig, OffscreenRenderTarget, SimClock, SprayJobRes};

/// Embedded fallback showcase SVG used when no path is given and `DEFAULT_SVG_PATH` is absent on disk.
const DEFAULT_SVG_BYTES: &str = include_str!("../assets/examples/02_street_art_poster.svg");

fn main() {
    let cli = CliArgs::parse();
    if let Err(err) = cli.validate() {
        eprintln!("Invalid CLI options: {err}");
        std::process::exit(2);
    }

    let app_config = AppConfig::new(cli.clone());
    let ingest_cfg = IngestConfig::from_studio(&app_config.studio);

    let primitives = match cli.svg_path.as_ref() {
        Some(path) => match ingest_svg_file(path, &ingest_cfg) {
            Ok(prims) => prims,
            Err(err) => {
                eprintln!("Failed to load SVG '{}': {err}", path.display());
                std::process::exit(1);
            }
        },
        None => {
            let default_path = std::path::Path::new(DEFAULT_SVG_PATH);
            let result = if default_path.exists() {
                ingest_svg_file(default_path, &ingest_cfg)
            } else {
                ingest_svg(DEFAULT_SVG_BYTES.as_bytes(), &ingest_cfg)
            };
            match result {
                Ok(prims) => prims,
                Err(err) => {
                    eprintln!("Failed to load default showcase SVG: {err}");
                    std::process::exit(1);
                }
            }
        }
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

    println!(
        "SprayBot job compiled: {} primitives, {} cans, {} ops, {:.2}s duration",
        primitives.len(),
        palette.cans.len(),
        job.ops.len(),
        total_duration
    );

    let mut sim_clock = SimClock::from_cli(&cli, total_duration);
    // Allow sufficient initial frames for Bevy PMREM environment cubemap & shadow pipelines to warm up.
    sim_clock.warmup_frames_remaining = 6;
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
        .insert_resource(spray_job);

    let (width, height) = cli.resolution;

    if cli.headless {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .set(RenderPlugin {
                    synchronous_pipeline_compilation: true,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin {
            run_mode: RunMode::Loop { wait: None },
        });

        let mut target_image =
            Image::new_target_texture(width, height, TextureFormat::Rgba8UnormSrgb, None);
        target_image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let target_handle = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(target_image);
        app.insert_resource(OffscreenRenderTarget {
            image: Some(target_handle),
        });
    } else {
        let resolution = if cli.record.is_some() {
            WindowResolution::new(width, height).with_scale_factor_override(1.0)
        } else {
            WindowResolution::new(width, height)
        };
        app.insert_resource(OffscreenRenderTarget::default())
            .add_plugins(DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "SprayBot".to_string(),
                    resolution,
                    resizable: cli.record.is_none(),
                    ..default()
                }),
                ..default()
            }));
    }

    let exit = app.add_plugins(SimPlugin).run();
    if let bevy::app::AppExit::Error(code) = exit {
        std::process::exit(code.get() as i32);
    }
}
