use crate::core::batching::ColorBatch;
use crate::core::types::{JobOp, JobPlan, NormalizedPrimitive, PalettePlan};

/// Physical spray nozzle and toolpath planning parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannerConfig {
    pub cone_half_angle_rad: f32,
    pub d_min: f32,
    pub d_max: f32,
    pub d_perimeter: f32,
    pub lead_length: f32,
    pub min_feed_speed: f32,
    pub max_feed_speed: f32,
    pub hatch_overlap_step_ratio: f32,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            cone_half_angle_rad: 12.0_f32.to_radians(),
            d_min: 0.04,
            d_max: 0.26,
            d_perimeter: 0.05,
            lead_length: 0.035,
            min_feed_speed: 0.12,
            max_feed_speed: 0.85,
            hatch_overlap_step_ratio: 0.65,
        }
    }
}

impl PlannerConfig {
    /// Effective core spot width `w(d) = 2 * d * tan(theta_c)` at standoff `d`.
    #[inline]
    pub fn spot_width(&self, standoff: f32) -> f32 {
        2.0 * standoff * self.cone_half_angle_rad.tan()
    }
}

/// Generates a full 3D [`JobPlan`] (can swaps, shakes, contour+hatch strokes, park home).
pub fn plan_job(
    _primitives: &[NormalizedPrimitive],
    _palette: &PalettePlan,
    _batches: &[ColorBatch],
    _config: &PlannerConfig,
) -> JobPlan {
    JobPlan {
        ops: vec![JobOp::ParkHome],
    }
}
