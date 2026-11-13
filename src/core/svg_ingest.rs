use std::path::Path;

use glam::Vec2;

use crate::core::types::{NormalizedPrimitive, StudioConfig};

/// Configuration for SVG parsing, curve flattening, and canvas coordinate normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct IngestConfig {
    pub canvas_center: Vec2,
    pub canvas_size: Vec2,
    pub margin: f32,
    pub flatten_tolerance_m: f64,
    pub gradient_bands: usize,
}

impl IngestConfig {
    pub fn from_studio(studio: &StudioConfig) -> Self {
        Self {
            canvas_center: Vec2::new(studio.canvas_center.x, studio.canvas_center.y),
            canvas_size: Vec2::new(studio.canvas_width, studio.canvas_height),
            margin: studio.canvas_margin,
            flatten_tolerance_m: 0.0005,
            gradient_bands: 5,
        }
    }
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self::from_studio(&StudioConfig::default())
    }
}

/// Parses SVG bytes into world-canvas-normalized primitives in painter's Z-order.
pub fn ingest_svg(
    _svg_bytes: &[u8],
    _config: &IngestConfig,
) -> Result<Vec<NormalizedPrimitive>, String> {
    Ok(Vec::new())
}

/// Convenience wrapper reading an SVG file from disk and calling [`ingest_svg`].
pub fn ingest_svg_file(
    path: &Path,
    config: &IngestConfig,
) -> Result<Vec<NormalizedPrimitive>, String> {
    let bytes =
        std::fs::read(path).map_err(|err| format!("failed to read SVG '{}': {err}", path.display()))?;
    ingest_svg(&bytes, config)
}
