use bevy::prelude::*;

use crate::cli::CliArgs;
use crate::core::kinematics::KinematicChain;
use crate::core::trajectory::CompiledTimeline;
use crate::core::types::{
    JobPlan, NormalizedPrimitive, PalettePlan, RobotFrameState, StudioConfig,
};

/// Ordered simulation execution stages within Bevy's `Update` schedule.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimStepSet {
    InputAndClock,
    RobotKinematics,
    CanvasAndFx,
    CameraAndGizmos,
    RecordCapture,
}

/// Immutable startup configuration derived from CLI arguments and validated studio dimensions.
#[derive(Resource, Debug, Clone)]
pub struct AppConfig {
    pub cli: CliArgs,
    pub studio: StudioConfig,
}

impl AppConfig {
    pub fn new(cli: CliArgs) -> Self {
        Self {
            cli,
            studio: StudioConfig::default(),
        }
    }
}

/// Deterministic simulation playback clock shared by controller, canvas deposition, and recorder.
#[derive(Resource, Debug, Clone)]
pub struct SimClock {
    pub t_sim: f64,
    pub t_prev: f64,
    pub speed: f32,
    pub paused: bool,
    pub total_duration: f64,
    /// When `Some(dt)`, each rendered frame advances by exact `dt * speed` (lockstep `--record`).
    pub fixed_dt: Option<f64>,
    pub end_hold_elapsed: f64,
    pub finished: bool,
}

impl SimClock {
    pub fn from_cli(cli: &CliArgs, total_duration: f64) -> Self {
        let fixed_dt = cli.record.as_ref().map(|_| 1.0 / (cli.fps.max(1) as f64));
        Self {
            t_sim: 0.0,
            t_prev: 0.0,
            speed: cli.speed,
            paused: false,
            total_duration,
            fixed_dt,
            end_hold_elapsed: 0.0,
            finished: false,
        }
    }

    pub fn reset(&mut self) {
        self.t_sim = 0.0;
        self.t_prev = 0.0;
        self.end_hold_elapsed = 0.0;
        self.finished = false;
    }
}

/// Compiled artwork, palette, kinematic chain, and current sampled robot state.
#[derive(Resource, Debug, Clone)]
pub struct SprayJobRes {
    pub primitives: Vec<NormalizedPrimitive>,
    pub palette: PalettePlan,
    pub job: JobPlan,
    pub chain: KinematicChain,
    pub timeline: CompiledTimeline,
    pub current_frame: RobotFrameState,
    pub show_gizmos: bool,
}

impl SprayJobRes {
    pub fn empty(studio: &StudioConfig) -> Self {
        let chain = KinematicChain::default_spraybot(studio);
        let timeline = CompiledTimeline::empty();
        let current_frame = timeline.sample(0.0);
        Self {
            primitives: Vec::new(),
            palette: PalettePlan::default(),
            job: JobPlan::default(),
            chain,
            timeline,
            current_frame,
            show_gizmos: false,
        }
    }
}

/// Offscreen GPU render target image handle used during headless/video recording.
#[derive(Resource, Debug, Default, Clone)]
pub struct OffscreenRenderTarget {
    pub image: Option<Handle<Image>>,
}
