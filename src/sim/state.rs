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
    pub prev_t_sim: f64,
    pub total_duration: f64,
    pub speed: f32,
    pub paused: bool,
    pub show_gizmos: bool,
    pub reset_epoch: u64,
    pub hold_secs: f64,
    pub hold_elapsed: f64,
    pub finished: bool,
    pub warmup_frames_remaining: u32,
}

impl Default for SimClock {
    fn default() -> Self {
        Self {
            t_sim: 0.0,
            prev_t_sim: 0.0,
            total_duration: 0.0,
            speed: 1.0,
            paused: false,
            show_gizmos: false,
            reset_epoch: 0,
            hold_secs: 2.0,
            hold_elapsed: 0.0,
            finished: false,
            warmup_frames_remaining: 3,
        }
    }
}

impl SimClock {
    pub fn from_cli(cli: &CliArgs, total_duration: f64) -> Self {
        Self {
            total_duration,
            speed: cli.speed,
            ..Self::default()
        }
    }

    pub fn reset(&mut self) {
        self.t_sim = 0.0;
        self.prev_t_sim = 0.0;
        self.hold_elapsed = 0.0;
        self.finished = false;
        self.reset_epoch = self.reset_epoch.wrapping_add(1);
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
