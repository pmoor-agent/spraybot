use crate::core::types::{NormalizedPrimitive, PalettePlan};

/// Configuration for overlap-aware topological color batching.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchConfig {
    /// Spray feather expansion margin (meters) added to footprints when checking overlap.
    pub feather_margin_m: f64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            feather_margin_m: 0.008,
        }
    }
}

/// Contiguous sequence of same-color primitives painted before swapping cans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorBatch {
    pub slot_index: usize,
    pub primitive_ids: Vec<usize>,
}

/// Schedules primitives into overlap-safe color batches minimizing rack swaps and air travel.
pub fn batch_primitives(
    _primitives: &[NormalizedPrimitive],
    _palette: &PalettePlan,
    _config: &BatchConfig,
) -> Vec<ColorBatch> {
    Vec::new()
}
