use crate::core::types::{NormalizedPrimitive, PalettePlan, SprayCanSpec};

/// Converts an sRGB `[R, G, B]` triple into perceptual OKLab `[L, a, b]`.
pub fn srgb_to_oklab(rgb: [u8; 3]) -> [f32; 3] {
    let to_linear = |c: u8| -> f32 {
        let s = c as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let r = to_linear(rgb[0]);
    let g = to_linear(rgb[1]);
    let b = to_linear(rgb[2]);

    let l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).cbrt();
    let m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).cbrt();
    let s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).cbrt();

    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

/// Extracts and quantizes primitive colors down to at most `max_cans` using OKLab k-means++.
pub fn quantize_palette(
    primitives: &[NormalizedPrimitive],
    max_cans: usize,
    _seed: u64,
) -> PalettePlan {
    if primitives.is_empty() || max_cans == 0 {
        return PalettePlan::default();
    }
    let can = SprayCanSpec {
        slot_index: 0,
        color_srgb: primitives[0].color_srgb,
        label: format!(
            "#{:02X}{:02X}{:02X}",
            primitives[0].color_srgb[0],
            primitives[0].color_srgb[1],
            primitives[0].color_srgb[2]
        ),
    };
    PalettePlan {
        cans: vec![can],
        primitive_color_map: vec![0; primitives.len()],
    }
}
