use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// Built-in showcase SVG used when no `[SVG_PATH]` argument is passed.
pub const DEFAULT_SVG_PATH: &str = "assets/examples/02_street_art_poster.svg";

/// Initial camera preset selectable from the CLI (`--camera`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum CameraPreset {
    #[default]
    Overview,
    Follow,
    Rack,
    Canvas,
}

/// Command-line interface for SprayBot.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "spraybot",
    version,
    about = "SprayBot 3D robotics mural simulator and offline video renderer"
)]
pub struct CliArgs {
    /// Input SVG file (defaults to built-in showcase mural if omitted)
    #[arg(value_name = "SVG_PATH")]
    pub svg_path: Option<PathBuf>,

    /// Simulation playback speed multiplier
    #[arg(short = 's', long, default_value_t = 1.0, value_name = "MULTIPLIER")]
    pub speed: f32,

    /// Export video (.mp4/.mkv via ffmpeg) or image sequence (directory/)
    #[arg(short = 'r', long, value_name = "OUTPUT")]
    pub record: Option<PathBuf>,

    /// Target frame rate for video export
    #[arg(long, default_value_t = 60, value_name = "FPS")]
    pub fps: u32,

    /// Viewport / video render resolution
    #[arg(
        long,
        default_value = "1920x1080",
        value_name = "WxH",
        value_parser = parse_resolution
    )]
    pub resolution: (u32, u32),

    /// Horizontal pixel resolution of canvas texture
    #[arg(long, default_value_t = 2048, value_name = "WIDTH")]
    pub canvas_res: u32,

    /// Spray rack slot capacity before k-means quantization
    #[arg(long, default_value_t = 16, value_name = "COUNT")]
    pub max_cans: usize,

    /// Initial camera preset: overview | follow | rack | canvas
    #[arg(long, value_enum, default_value_t = CameraPreset::Overview, value_name = "PRESET")]
    pub camera: CameraPreset,

    /// Render video offscreen without opening an OS window (requires --record)
    #[arg(long, requires = "record")]
    pub headless: bool,

    /// RNG seed for speckle & k-means determinism
    #[arg(long, default_value_t = 42, value_name = "SEED")]
    pub seed: u64,
}

impl CliArgs {
    /// Returns the explicit SVG path or falls back to the built-in showcase mural.
    pub fn resolved_svg_path(&self) -> PathBuf {
        self.svg_path
            .clone()
            .unwrap_or_else(|| PathBuf::from(DEFAULT_SVG_PATH))
    }

    /// Validates numeric ranges that clap's type parsing alone does not constrain.
    pub fn validate(&self) -> Result<(), String> {
        if !self.speed.is_finite() || self.speed <= 0.0 {
            return Err(format!("--speed must be finite and > 0, got {}", self.speed));
        }
        if self.fps == 0 {
            return Err("--fps must be > 0".to_string());
        }
        if self.canvas_res == 0 {
            return Err("--canvas-res must be > 0".to_string());
        }
        if self.max_cans == 0 {
            return Err("--max-cans must be > 0".to_string());
        }
        if self.headless && self.record.is_none() {
            return Err("--headless requires --record <OUTPUT>".to_string());
        }
        Ok(())
    }
}

/// Parses a `<WIDTH>x<HEIGHT>` resolution string into non-zero `(width, height)`.
pub fn parse_resolution(spec: &str) -> Result<(u32, u32), String> {
    let trimmed = spec.trim();
    let (w_str, h_str) = trimmed
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("invalid resolution '{spec}': expected <WIDTH>x<HEIGHT>"))?;

    if h_str.contains(['x', 'X']) {
        return Err(format!(
            "invalid resolution '{spec}': expected a single 'x' separator"
        ));
    }

    let width: u32 = w_str
        .parse()
        .map_err(|_| format!("invalid width '{w_str}' in resolution '{spec}'"))?;
    let height: u32 = h_str
        .parse()
        .map_err(|_| format!("invalid height '{h_str}' in resolution '{spec}'"))?;

    if width == 0 || height == 0 {
        return Err(format!(
            "resolution dimensions must be non-zero, got {width}x{height}"
        ));
    }

    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let args = CliArgs::try_parse_from(["spraybot"]).expect("default args parse");
        assert_eq!(args.svg_path, None);
        assert_eq!(args.resolved_svg_path(), PathBuf::from(DEFAULT_SVG_PATH));
        assert!((args.speed - 1.0).abs() < f32::EPSILON);
        assert_eq!(args.record, None);
        assert_eq!(args.fps, 60);
        assert_eq!(args.resolution, (1920, 1080));
        assert_eq!(args.canvas_res, 2048);
        assert_eq!(args.max_cans, 16);
        assert_eq!(args.camera, CameraPreset::Overview);
        assert!(!args.headless);
        assert_eq!(args.seed, 42);
        assert!(args.validate().is_ok());
    }

    #[test]
    fn parses_all_cli_options() {
        let args = CliArgs::try_parse_from([
            "spraybot",
            "assets/examples/01_geometric_shapes.svg",
            "-s",
            "12.5",
            "-r",
            "/tmp/out.mp4",
            "--fps",
            "30",
            "--resolution",
            "1280x720",
            "--canvas-res",
            "1024",
            "--max-cans",
            "12",
            "--camera",
            "follow",
            "--headless",
            "--seed",
            "999",
        ])
        .expect("full args parse");

        assert_eq!(
            args.svg_path,
            Some(PathBuf::from("assets/examples/01_geometric_shapes.svg"))
        );
        assert!((args.speed - 12.5).abs() < f32::EPSILON);
        assert_eq!(args.record, Some(PathBuf::from("/tmp/out.mp4")));
        assert_eq!(args.fps, 30);
        assert_eq!(args.resolution, (1280, 720));
        assert_eq!(args.canvas_res, 1024);
        assert_eq!(args.max_cans, 12);
        assert_eq!(args.camera, CameraPreset::Follow);
        assert!(args.headless);
        assert_eq!(args.seed, 999);
        assert!(args.validate().is_ok());
    }

    #[test]
    fn headless_requires_record() {
        let err = CliArgs::try_parse_from(["spraybot", "--headless"])
            .expect_err("--headless without --record must fail");
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn camera_presets_parse() {
        for (raw, expected) in [
            ("overview", CameraPreset::Overview),
            ("follow", CameraPreset::Follow),
            ("rack", CameraPreset::Rack),
            ("canvas", CameraPreset::Canvas),
        ] {
            let args = CliArgs::try_parse_from(["spraybot", "--camera", raw]).unwrap();
            assert_eq!(args.camera, expected);
        }
    }

    #[test]
    fn parse_resolution_valid_and_invalid() {
        assert_eq!(parse_resolution("1920x1080"), Ok((1920, 1080)));
        assert_eq!(parse_resolution("1280X720"), Ok((1280, 720)));
        assert_eq!(parse_resolution(" 3840x2160 "), Ok((3840, 2160)));

        assert!(parse_resolution("0x1080").is_err());
        assert!(parse_resolution("1920x0").is_err());
        assert!(parse_resolution("1920").is_err());
        assert!(parse_resolution("1920x1080x2").is_err());
        assert!(parse_resolution("abcxdef").is_err());
        assert!(parse_resolution("-1920x1080").is_err());
        assert!(parse_resolution("").is_err());
    }
}
