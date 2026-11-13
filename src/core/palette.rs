use std::collections::BTreeMap;

use geo::algorithm::area::Area;

use crate::core::types::{NormalizedPrimitive, PalettePlan, PrimitiveGeom, SprayCanSpec};

/// Perceptual OKLab distance threshold below which two colors are merged into one can.
pub const OKLAB_DEDUP_THRESHOLD: f32 = 0.015;

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

/// Converts a perceptual OKLab `[L, a, b]` color back to clamped sRGB `[R, G, B]`.
pub fn oklab_to_srgb(lab: [f32; 3]) -> [u8; 3] {
    let l_ = lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2];
    let m_ = lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2];
    let s_ = lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2];

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    let r_lin = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    let g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    let b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

    let to_srgb = |c: f32| -> u8 {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round().clamp(0.0, 255.0) as u8
    };

    [to_srgb(r_lin), to_srgb(g_lin), to_srgb(b_lin)]
}

/// Euclidean distance in perceptual OKLab space (`ΔE_oklab`).
#[inline]
pub fn oklab_delta_e(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dl = a[0] - b[0];
    let da = a[1] - b[1];
    let db = a[2] - b[2];
    (dl * dl + da * da + db * db).sqrt()
}

/// Squared Euclidean distance in OKLab space.
#[inline]
fn oklab_dist_sq(a: [f32; 3], b: [f32; 3]) -> f64 {
    let dl = (a[0] - b[0]) as f64;
    let da = (a[1] - b[1]) as f64;
    let db = (a[2] - b[2]) as f64;
    dl * dl + da * da + db * db
}

/// Computes an effective painted area/arc-length weight for a primitive.
fn primitive_weight(prim: &NormalizedPrimitive) -> f64 {
    let geom_weight = match &prim.geometry {
        PrimitiveGeom::FilledPolygon(mp) => mp.unsigned_area(),
        PrimitiveGeom::StrokedPath { points, width } => {
            let len: f64 = points.windows(2).map(|w| w[0].distance(w[1])).sum();
            len * width.max(0.005)
        }
    };
    (geom_weight * (prim.opacity.clamp(0.05, 1.0) as f64)).max(1e-6)
}

#[derive(Debug, Clone)]
struct WeightedColorCluster {
    lab: [f32; 3],
    weight: f64,
    representative_rgb: [u8; 3],
    member_count: usize,
}

/// Extracts and quantizes primitive colors down to at most `max_cans` using OKLab k-means++.
pub fn quantize_palette(
    primitives: &[NormalizedPrimitive],
    max_cans: usize,
    seed: u64,
) -> PalettePlan {
    if primitives.is_empty() || max_cans == 0 {
        return PalettePlan::default();
    }

    // 1. Aggregate weights per exact sRGB value deterministically (BTreeMap sorted by RGB).
    let mut exact_weights: BTreeMap<[u8; 3], f64> = BTreeMap::new();
    for prim in primitives {
        *exact_weights.entry(prim.color_srgb).or_insert(0.0) += primitive_weight(prim);
    }

    // Sort descending by weight, tie-breaking by RGB so clustering is independent of input order.
    let mut exact_list: Vec<([u8; 3], [f32; 3], f64)> = exact_weights
        .into_iter()
        .map(|(rgb, w)| (rgb, srgb_to_oklab(rgb), w))
        .collect();
    exact_list.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // 2. Deduplicate near-identical colors within ΔE_oklab < 0.015.
    let mut clusters: Vec<WeightedColorCluster> = Vec::new();
    for (rgb, lab, w) in exact_list {
        let mut best_idx: Option<usize> = None;
        let mut best_dist = OKLAB_DEDUP_THRESHOLD;
        for (idx, cluster) in clusters.iter().enumerate() {
            let d = oklab_delta_e(lab, cluster.lab);
            if d < best_dist {
                best_dist = d;
                best_idx = Some(idx);
            }
        }
        if let Some(idx) = best_idx {
            let c = &mut clusters[idx];
            let total_w = c.weight + w;
            let f0 = (c.weight / total_w) as f32;
            let f1 = (w / total_w) as f32;
            c.lab = [
                c.lab[0] * f0 + lab[0] * f1,
                c.lab[1] * f0 + lab[1] * f1,
                c.lab[2] * f0 + lab[2] * f1,
            ];
            c.weight = total_w;
            c.member_count += 1;
        } else {
            clusters.push(WeightedColorCluster {
                lab,
                weight: w,
                representative_rgb: rgb,
                member_count: 1,
            });
        }
    }

    // 3. Build up to max_cans centroids.
    let centroids_rgb_lab: Vec<([u8; 3], [f32; 3], f64)> = if clusters.len() <= max_cans {
        clusters
            .into_iter()
            .map(|c| {
                let rgb = if c.member_count == 1 {
                    c.representative_rgb
                } else {
                    oklab_to_srgb(c.lab)
                };
                (rgb, srgb_to_oklab(rgb), c.weight)
            })
            .collect()
    } else {
        weighted_kmeans_pp(&clusters, max_cans, seed)
    };

    let cans: Vec<SprayCanSpec> = centroids_rgb_lab
        .iter()
        .enumerate()
        .map(|(slot_index, (rgb, _, _))| SprayCanSpec {
            slot_index,
            color_srgb: *rgb,
            label: format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]),
        })
        .collect();

    let can_labs: Vec<[f32; 3]> = centroids_rgb_lab.iter().map(|c| c.1).collect();

    // 4. Map every primitive to its nearest centroid in OKLab.
    // Check whether primitive IDs form a dense 0..N permutation so primitive_color_map[prim.id]
    // and primitive_color_map[idx] are both satisfied cleanly.
    let dense_ids = primitives.iter().all(|p| p.id < primitives.len()) && {
        let mut seen = vec![false; primitives.len()];
        primitives.iter().all(|p| {
            let was = seen[p.id];
            seen[p.id] = true;
            !was
        })
    };

    let mut primitive_color_map = vec![0usize; primitives.len()];
    for (idx, prim) in primitives.iter().enumerate() {
        let lab = srgb_to_oklab(prim.color_srgb);
        let mut best_slot = 0usize;
        let mut best_d2 = f64::INFINITY;
        for (slot, can_lab) in can_labs.iter().enumerate() {
            let d2 = oklab_dist_sq(lab, *can_lab);
            if d2 < best_d2 {
                best_d2 = d2;
                best_slot = slot;
            }
        }
        let target_idx = if dense_ids { prim.id } else { idx };
        primitive_color_map[target_idx] = best_slot;
    }

    PalettePlan {
        cans,
        primitive_color_map,
    }
}

/// Deterministic seeded area-weighted k-means++ over deduplicated OKLab clusters.
fn weighted_kmeans_pp(
    clusters: &[WeightedColorCluster],
    k: usize,
    seed: u64,
) -> Vec<([u8; 3], [f32; 3], f64)> {
    let n = clusters.len();
    let mut rng = SplitMix64::new(seed);

    // k-means++ initialization
    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    let total_w: f64 = clusters.iter().map(|c| c.weight).sum();
    let mut target = rng.next_f64() * total_w;
    let mut first_idx = 0usize;
    for (i, c) in clusters.iter().enumerate() {
        target -= c.weight;
        if target <= 0.0 {
            first_idx = i;
            break;
        }
    }
    centroids.push(clusters[first_idx].lab);

    let mut min_d2 = vec![f64::INFINITY; n];
    for _ in 1..k {
        let last_c = *centroids.last().unwrap();
        let mut sum_wd2 = 0.0_f64;
        for (i, c) in clusters.iter().enumerate() {
            let d2 = oklab_dist_sq(c.lab, last_c);
            if d2 < min_d2[i] {
                min_d2[i] = d2;
            }
            sum_wd2 += c.weight * min_d2[i];
        }

        let chosen = if sum_wd2 > 1e-18 {
            let mut r = rng.next_f64() * sum_wd2;
            let mut idx = n - 1;
            for (i, c) in clusters.iter().enumerate() {
                r -= c.weight * min_d2[i];
                if r <= 0.0 {
                    idx = i;
                    break;
                }
            }
            idx
        } else {
            min_d2
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        centroids.push(clusters[chosen].lab);
    }

    // Lloyd refinement iterations
    let mut assignments = vec![0usize; n];
    for _iter in 0..50 {
        let mut changed = false;
        for (i, c) in clusters.iter().enumerate() {
            let mut best_c = 0usize;
            let mut best_d2 = f64::INFINITY;
            for (ci, cent) in centroids.iter().enumerate() {
                let d2 = oklab_dist_sq(c.lab, *cent);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_c = ci;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        let mut sums = vec![[0.0_f64; 3]; k];
        let mut weights = vec![0.0_f64; k];
        for (i, c) in clusters.iter().enumerate() {
            let ci = assignments[i];
            sums[ci][0] += (c.lab[0] as f64) * c.weight;
            sums[ci][1] += (c.lab[1] as f64) * c.weight;
            sums[ci][2] += (c.lab[2] as f64) * c.weight;
            weights[ci] += c.weight;
        }

        for ci in 0..k {
            if weights[ci] > 0.0 {
                centroids[ci] = [
                    (sums[ci][0] / weights[ci]) as f32,
                    (sums[ci][1] / weights[ci]) as f32,
                    (sums[ci][2] / weights[ci]) as f32,
                ];
            } else {
                // Reseed empty centroid to highest weighted error cluster.
                let farthest = clusters
                    .iter()
                    .enumerate()
                    .max_by(|(i, ca), (j, cb)| {
                        let ea = ca.weight * oklab_dist_sq(ca.lab, centroids[assignments[*i]]);
                        let eb = cb.weight * oklab_dist_sq(cb.lab, centroids[assignments[*j]]);
                        ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                centroids[ci] = clusters[farthest].lab;
            }
        }

        if !changed {
            break;
        }
    }

    // Recompute final cluster weights per centroid and sort descending by total weight for determinism.
    let mut final_weights = vec![0.0_f64; k];
    for c in clusters.iter() {
        let mut best_c = 0usize;
        let mut best_d2 = f64::INFINITY;
        for (ci, cent) in centroids.iter().enumerate() {
            let d2 = oklab_dist_sq(c.lab, *cent);
            if d2 < best_d2 {
                best_d2 = d2;
                best_c = ci;
            }
        }
        final_weights[best_c] += c.weight;
    }

    let mut result: Vec<([u8; 3], [f32; 3], f64)> = centroids
        .into_iter()
        .zip(final_weights)
        .map(|(lab, w)| {
            let rgb = oklab_to_srgb(lab);
            (rgb, srgb_to_oklab(rgb), w)
        })
        .collect();

    result.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    result
}

/// Deterministic 64-bit SplitMix PRNG for reproducible k-means++ initialization.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::Path;

    use geo::{Coord, LineString, MultiPolygon, Polygon};

    use super::*;
    use crate::core::svg_ingest::{IngestConfig, ingest_svg_file};

    fn dummy_square(id: usize, x: f64, color: [u8; 3]) -> NormalizedPrimitive {
        let poly = Polygon::new(
            LineString::new(vec![
                Coord { x, y: 1.0 },
                Coord { x: x + 0.1, y: 1.0 },
                Coord { x: x + 0.1, y: 1.1 },
                Coord { x, y: 1.1 },
                Coord { x, y: 1.0 },
            ]),
            Vec::new(),
        );
        NormalizedPrimitive {
            id,
            color_srgb: color,
            opacity: 1.0,
            geometry: PrimitiveGeom::FilledPolygon(MultiPolygon::new(vec![poly])),
        }
    }

    #[test]
    fn oklab_roundtrip_is_exact_for_primaries_and_grays() {
        let samples = [
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 209, 102],
            [27, 16, 53],
            [128, 128, 128],
        ];
        for rgb in samples {
            let rt = oklab_to_srgb(srgb_to_oklab(rgb));
            for c in 0..3 {
                assert!(
                    (rt[c] as i32 - rgb[c] as i32).abs() <= 1,
                    "roundtrip mismatch for {rgb:?}: got {rt:?}"
                );
            }
        }
    }

    #[test]
    fn deduplicates_near_identical_oklab_colors() {
        // [200, 50, 50] and [201, 50, 51] have ΔE_oklab well below 0.015; [20, 200, 60] is distant.
        let de = oklab_delta_e(srgb_to_oklab([200, 50, 50]), srgb_to_oklab([201, 50, 51]));
        assert!(de < OKLAB_DEDUP_THRESHOLD, "expected ΔE < 0.015, got {de}");

        let prims = vec![
            dummy_square(0, 0.0, [200, 50, 50]),
            dummy_square(1, 0.2, [201, 50, 51]),
            dummy_square(2, 0.4, [20, 200, 60]),
        ];
        let plan = quantize_palette(&prims, 16, 42);
        assert_eq!(plan.cans.len(), 2);
        assert_eq!(plan.primitive_color_map.len(), 3);
        assert_eq!(plan.primitive_color_map[0], plan.primitive_color_map[1]);
        assert_ne!(plan.primitive_color_map[0], plan.primitive_color_map[2]);
    }

    #[test]
    fn deterministic_kmeans_cap_on_palette_stress_svg() {
        let prims = ingest_svg_file(
            Path::new("assets/examples/03_palette_stress.svg"),
            &IngestConfig::default(),
        )
        .expect("ingest 03_palette_stress.svg");

        let unique_raw: HashSet<[u8; 3]> = prims.iter().map(|p| p.color_srgb).collect();
        assert!(
            unique_raw.len() >= 25,
            "expected >= 25 unique raw colors in 03_palette_stress.svg, got {}",
            unique_raw.len()
        );

        let plan_a = quantize_palette(&prims, 16, 42);
        let plan_b = quantize_palette(&prims, 16, 42);
        assert_eq!(plan_a, plan_b, "k-means++ quantization must be deterministic for same seed");
        assert_eq!(plan_a.cans.len(), 16, "expected quantization cap to 16 cans");
        assert_eq!(plan_a.primitive_color_map.len(), prims.len());

        for &slot in &plan_a.primitive_color_map {
            assert!(slot < plan_a.cans.len(), "mapped slot {slot} out of range");
        }
        for (idx, can) in plan_a.cans.iter().enumerate() {
            assert_eq!(can.slot_index, idx);
        }
    }
}
