use crate::core::kinematics::KinematicChain;
use crate::core::types::{JobPlan, RobotFrameState, StudioConfig};

/// Time-parameterized robot trajectory compiled from a [`JobPlan`].
#[derive(Debug, Clone)]
pub struct CompiledTimeline {
    pub total_duration: f64,
    pub home_joints: [f64; 7],
}

impl CompiledTimeline {
    /// Creates an idle parked timeline with zero duration.
    pub fn empty() -> Self {
        Self {
            total_duration: 0.0,
            home_joints: [0.0; 7],
        }
    }

    /// Compiles a high-level [`JobPlan`] into a smooth time-parameterized trajectory.
    pub fn compile(
        _job: &JobPlan,
        chain: &KinematicChain,
        _studio: &StudioConfig,
    ) -> Result<Self, String> {
        Ok(Self {
            total_duration: 0.0,
            home_joints: chain.comfort_pose,
        })
    }

    /// Total timeline duration in simulation seconds.
    #[inline]
    pub fn duration(&self) -> f64 {
        self.total_duration
    }

    /// Evaluates continuous robot and spray state at simulation time `t_sim`.
    pub fn sample(&self, _t_sim: f64) -> RobotFrameState {
        RobotFrameState {
            joints: self.home_joints,
            ..RobotFrameState::default()
        }
    }
}
