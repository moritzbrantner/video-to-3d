//! Deterministic feature detection and matching experiments for video-to-3d.
//!
//! The crate deliberately keeps two pipelines side by side:
//! - `BaselineHarrisPatch` mirrors the small, easy-to-debug Harris + normalized patch approach.
//! - `OrbStyle` uses a classical multi-scale FAST + intensity-centroid orientation + rotated BRIEF
//!   descriptor with mutual ratio-checked Hamming matching.
//!
//! This makes the simple path useful as a regression baseline without pretending it is the final
//! feature pipeline for difficult camera motion.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

const FAST_CIRCLE: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];
const ORB_DESCRIPTOR_BITS: usize = 256;
const ORB_WORDS: usize = ORB_DESCRIPTOR_BITS / 64;
const ORB_PATCH_RADIUS: i32 = 8;
const ORB_PYRAMID_SCALE: f32 = 0.75;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureAlgorithm {
    #[default]
    BaselineHarrisPatch,
    OrbStyle,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct FeatureOptions {
    pub max_features: usize,
    pub min_feature_distance: u32,
    pub descriptor_radius: u32,
    pub match_radius: u32,
    pub baseline_max_distance: f32,
    pub orb_max_hamming: u32,
    pub ratio_threshold: f32,
    pub fast_threshold: u8,
    pub pyramid_levels: u8,
}

impl Default for FeatureOptions {
    fn default() -> Self {
        Self {
            max_features: 400,
            min_feature_distance: 6,
            descriptor_radius: 3,
            match_radius: 48,
            baseline_max_distance: 36.0,
            orb_max_hamming: 82,
            ratio_threshold: 0.80,
            fast_threshold: 18,
            pyramid_levels: 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct FeaturePoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
    pub scale: f32,
    pub angle_radians: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct FeatureMatch {
    pub source_index: usize,
    pub target_index: usize,
    pub distance: f32,
    pub dx: f32,
    pub dy: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeatureAnalysis {
    pub algorithm: FeatureAlgorithm,
    pub source_features: Vec<FeaturePoint>,
    pub target_features: Vec<FeaturePoint>,
    pub matches: Vec<FeatureMatch>,
}

#[derive(Clone, Debug)]
struct GrayImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

#[derive(Clone, Debug)]
enum Descriptor {
    Patch(Vec<i16>),
    Binary([u64; ORB_WORDS]),
}

#[derive(Clone, Debug)]
struct Feature {
    point: FeaturePoint,
    descriptor: Descriptor,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    level: usize,
    x: u32,
    y: u32,
    original_x: f32,
    original_y: f32,
    score: f32,
    scale: f32,
}

/// Detect and match features between two RGBA frames with the selected deterministic pipeline.
///
/// `BaselineHarrisPatch` is intentionally close to the original reconstruction feature path.
/// `OrbStyle` is the stronger classical alternative intended for experimentation before promotion
/// into the authoritative reconstruction path.
pub fn analyze_rgba_pair(
    source_rgba: &[u8],
    target_rgba: &[u8],
    width: u32,
    height: u32,
    algorithm: FeatureAlgorithm,
    options: FeatureOptions,
) -> Result<FeatureAnalysis, String> {
    validate_input(source_rgba, target_rgba, width, height, options)?;

    let source = GrayImage {
        width,
        height,
        pixels: rgba_to_luma(source_rgba),
    };
    let target = GrayImage {
        width,
        height,
        pixels: rgba_to_luma(target_rgba),
    };

    let source_features = detect_features(&source, algorithm, options);
    let target_features = detect_features(&target, algorithm, options);
    let matches = match_features(&source_features, &target_features, algorithm, options);

    Ok(FeatureAnalysis {
        algorithm,
        source_features: source_features
            .iter()
            .map(|feature| feature.point)
            .collect(),
        target_features: target_features
            .iter()
            .map(|feature| feature.point)
            .collect(),
        matches,
    })
}

fn validate_input(
    source_rgba: &[u8],
    target_rgba: &[u8],
    width: u32,
    height: u32,
    options: FeatureOptions,
) -> Result<(), String> {
    if width < 32 || height < 24 {
        return Err("feature analysis requires frames of at least 32x24 pixels".into());
    }
    let expected = width as usize * height as usize * 4;
    if source_rgba.len() != expected || target_rgba.len() != expected {
        return Err(format!(
            "invalid RGBA payload: expected {expected} bytes per frame, got {} and {}",
            source_rgba.len(),
            target_rgba.len()
        ));
    }
    if options.max_features == 0 {
        return Err("max_features must be positive".into());
    }
    if options.fast_threshold == 0 {
        return Err("fast_threshold must be positive".into());
    }
    if options.baseline_max_distance <= 0.0 || !options.baseline_max_distance.is_finite() {
        return Err("baseline_max_distance must be finite and positive".into());
    }
    if options.orb_max_hamming == 0 || options.orb_max_hamming > ORB_DESCRIPTOR_BITS as u32 {
        return Err("orb_max_hamming must be between 1 and 256".into());
    }
    if !(0.0..1.0).contains(&options.ratio_threshold) {
        return Err("ratio_threshold must be between 0 and 1".into());
    }
    if options.pyramid_levels == 0 {
        return Err("pyramid_levels must be positive".into());
    }
    Ok(())
}

fn rgba_to_luma(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .map(|pixel| {
            ((77 * u16::from(pixel[0]) + 150 * u16::from(pixel[1]) + 29 * u16::from(pixel[2])) >> 8)
                as u8
        })
        .collect()
}

fn detect_features(
    image: &GrayImage,
    algorithm: FeatureAlgorithm,
    options: FeatureOptions,
) -> Vec<Feature> {
    match algorithm {
        FeatureAlgorithm::BaselineHarrisPatch => detect_baseline(image, options),
        FeatureAlgorithm::OrbStyle => detect_orb_style(image, options),
    }
}

fn detect_baseline(image: &GrayImage, options: FeatureOptions) -> Vec<Feature> {
    let border = options.descriptor_radius.max(3) + 2;
    if image.width <= border * 2 || image.height <= border * 2 {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for y in border..image.height - border {
        for x in border..image.width - border {
            let score = harris_score(image, x, y);
            if score > 1_000_000.0 {
                candidates.push(Candidate {
                    level: 0,
                    x,
                    y,
                    original_x: x as f32,
                    original_y: y as f32,
                    score,
                    scale: 1.0,
                });
            }
        }
    }
    sort_candidates(&mut candidates);

    select_candidates(
        &candidates,
        options.max_features,
        options.min_feature_distance,
    )
    .into_iter()
    .map(|candidate| Feature {
        point: FeaturePoint {
            x: candidate.original_x,
            y: candidate.original_y,
            score: candidate.score,
            scale: 1.0,
            angle_radians: 0.0,
        },
        descriptor: Descriptor::Patch(patch_descriptor(
            image,
            candidate.x,
            candidate.y,
            options.descriptor_radius,
        )),
    })
    .collect()
}

fn detect_orb_style(image: &GrayImage, options: FeatureOptions) -> Vec<Feature> {
    let pyramid = build_pyramid(image, options.pyramid_levels.min(6));
    let mut candidates = Vec::new();

    for (level, level_image) in pyramid.iter().enumerate() {
        let border = u32::try_from(ORB_PATCH_RADIUS + 2).expect("positive ORB border");
        if level_image.width <= border * 2 || level_image.height <= border * 2 {
            continue;
        }
        let scale_x = level_image.width as f32 / image.width as f32;
        let scale_y = level_image.height as f32 / image.height as f32;
        for y in border..level_image.height - border {
            for x in border..level_image.width - border {
                let Some(fast_strength) = fast9_score(level_image, x, y, options.fast_threshold)
                else {
                    continue;
                };
                if !is_fast_local_maximum(level_image, x, y, options.fast_threshold, fast_strength)
                {
                    continue;
                }
                let harris = harris_score(level_image, x, y).max(0.0);
                let score = harris.sqrt() + fast_strength * 100.0;
                candidates.push(Candidate {
                    level,
                    x,
                    y,
                    original_x: x as f32 / scale_x,
                    original_y: y as f32 / scale_y,
                    score,
                    scale: 1.0 / scale_x,
                });
            }
        }
    }
    sort_candidates(&mut candidates);

    select_candidates(
        &candidates,
        options.max_features,
        options.min_feature_distance,
    )
    .into_iter()
    .map(|candidate| {
        let level_image = &pyramid[candidate.level];
        let angle = intensity_centroid_angle(level_image, candidate.x, candidate.y);
        Feature {
            point: FeaturePoint {
                x: candidate.original_x,
                y: candidate.original_y,
                score: candidate.score,
                scale: candidate.scale,
                angle_radians: angle,
            },
            descriptor: Descriptor::Binary(rotated_brief_descriptor(
                level_image,
                candidate.x,
                candidate.y,
                angle,
            )),
        }
    })
    .collect()
}

fn sort_candidates(candidates: &mut [Candidate]) {
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.level.cmp(&right.level))
            .then_with(|| left.y.cmp(&right.y))
            .then_with(|| left.x.cmp(&right.x))
    });
}

fn select_candidates(
    candidates: &[Candidate],
    max_features: usize,
    min_feature_distance: u32,
) -> Vec<Candidate> {
    let minimum_squared = (min_feature_distance as f32).powi(2);
    let mut selected = Vec::with_capacity(max_features);
    for &candidate in candidates {
        if selected.iter().any(|existing: &Candidate| {
            let dx = existing.original_x - candidate.original_x;
            let dy = existing.original_y - candidate.original_y;
            dx * dx + dy * dy < minimum_squared
        }) {
            continue;
        }
        selected.push(candidate);
        if selected.len() >= max_features {
            break;
        }
    }
    selected
}

fn harris_score(image: &GrayImage, x: u32, y: u32) -> f32 {
    let mut sxx = 0.0f32;
    let mut syy = 0.0f32;
    let mut sxy = 0.0f32;
    for wy in y - 1..=y + 1 {
        for wx in x - 1..=x + 1 {
            let gx = f32::from(sample(image, wx + 1, wy)) - f32::from(sample(image, wx - 1, wy));
            let gy = f32::from(sample(image, wx, wy + 1)) - f32::from(sample(image, wx, wy - 1));
            sxx += gx * gx;
            syy += gy * gy;
            sxy += gx * gy;
        }
    }
    let determinant = sxx * syy - sxy * sxy;
    let trace = sxx + syy;
    determinant - 0.04 * trace * trace
}

fn patch_descriptor(image: &GrayImage, x: u32, y: u32, radius: u32) -> Vec<i16> {
    let side = radius * 2 + 1;
    let mut values = Vec::with_capacity((side * side) as usize);
    let mut sum = 0i32;
    for py in y - radius..=y + radius {
        for px in x - radius..=x + radius {
            let value = i16::from(sample(image, px, py));
            values.push(value);
            sum += i32::from(value);
        }
    }
    let mean = sum / i32::try_from(values.len()).expect("patch length fits i32");
    for value in &mut values {
        *value -= mean as i16;
    }
    values
}

fn build_pyramid(image: &GrayImage, level_count: u8) -> Vec<GrayImage> {
    let mut pyramid = Vec::with_capacity(level_count as usize);
    pyramid.push(image.clone());
    for level in 1..level_count {
        let scale = ORB_PYRAMID_SCALE.powi(i32::from(level));
        let width = ((image.width as f32 * scale).round() as u32).max(16);
        let height = ((image.height as f32 * scale).round() as u32).max(12);
        if width == pyramid.last().expect("base pyramid level exists").width
            && height == pyramid.last().expect("base pyramid level exists").height
        {
            continue;
        }
        pyramid.push(resample_bilinear(image, width, height));
    }
    pyramid
}

fn resample_bilinear(source: &GrayImage, width: u32, height: u32) -> GrayImage {
    let mut pixels = vec![0u8; width as usize * height as usize];
    let scale_x = source.width as f32 / width as f32;
    let scale_y = source.height as f32 / height as f32;
    for y in 0..height {
        let source_y = ((y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, source.height as f32 - 1.0);
        let y0 = source_y.floor() as u32;
        let y1 = (y0 + 1).min(source.height - 1);
        let fy = source_y - y0 as f32;
        for x in 0..width {
            let source_x = ((x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, source.width as f32 - 1.0);
            let x0 = source_x.floor() as u32;
            let x1 = (x0 + 1).min(source.width - 1);
            let fx = source_x - x0 as f32;
            let top = f32::from(sample(source, x0, y0)) * (1.0 - fx)
                + f32::from(sample(source, x1, y0)) * fx;
            let bottom = f32::from(sample(source, x0, y1)) * (1.0 - fx)
                + f32::from(sample(source, x1, y1)) * fx;
            pixels[(y * width + x) as usize] =
                (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
        }
    }
    GrayImage {
        width,
        height,
        pixels,
    }
}

fn fast9_score(image: &GrayImage, x: u32, y: u32, threshold: u8) -> Option<f32> {
    let center = i16::from(sample(image, x, y));
    let threshold = i16::from(threshold);
    let mut bright = [false; 16];
    let mut dark = [false; 16];
    let mut strength = 0i32;
    for (index, (dx, dy)) in FAST_CIRCLE.iter().copied().enumerate() {
        let value = i16::from(sample_offset(image, x, y, dx, dy));
        let delta = value - center;
        bright[index] = delta > threshold;
        dark[index] = delta < -threshold;
        if bright[index] || dark[index] {
            strength += i32::from(delta.unsigned_abs());
        }
    }
    if has_circular_run(&bright, 9) || has_circular_run(&dark, 9) {
        Some(strength as f32)
    } else {
        None
    }
}

fn has_circular_run(values: &[bool; 16], required: usize) -> bool {
    let mut run = 0usize;
    for index in 0..(values.len() + required - 1) {
        if values[index % values.len()] {
            run += 1;
            if run >= required {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

fn is_fast_local_maximum(image: &GrayImage, x: u32, y: u32, threshold: u8, score: f32) -> bool {
    for neighbor_y in y - 1..=y + 1 {
        for neighbor_x in x - 1..=x + 1 {
            if neighbor_x == x && neighbor_y == y {
                continue;
            }
            if fast9_score(image, neighbor_x, neighbor_y, threshold)
                .is_some_and(|neighbor_score| neighbor_score > score)
            {
                return false;
            }
        }
    }
    true
}

fn intensity_centroid_angle(image: &GrayImage, x: u32, y: u32) -> f32 {
    let mut m10 = 0.0f32;
    let mut m01 = 0.0f32;
    let radius_squared = ORB_PATCH_RADIUS * ORB_PATCH_RADIUS;
    for dy in -ORB_PATCH_RADIUS..=ORB_PATCH_RADIUS {
        for dx in -ORB_PATCH_RADIUS..=ORB_PATCH_RADIUS {
            if dx * dx + dy * dy > radius_squared {
                continue;
            }
            let intensity = f32::from(sample_offset(image, x, y, dx, dy));
            m10 += dx as f32 * intensity;
            m01 += dy as f32 * intensity;
        }
    }
    m01.atan2(m10)
}

fn rotated_brief_descriptor(image: &GrayImage, x: u32, y: u32, angle: f32) -> [u64; ORB_WORDS] {
    let cosine = angle.cos();
    let sine = angle.sin();
    let mut words = [0u64; ORB_WORDS];
    for bit in 0..ORB_DESCRIPTOR_BITS {
        let (left_x, left_y) =
            brief_offset(0x9e37_79b9u32.wrapping_add((bit as u32).wrapping_mul(2)));
        let (right_x, right_y) =
            brief_offset(0x7f4a_7c15u32.wrapping_add((bit as u32).wrapping_mul(2).wrapping_add(1)));
        let rotated_left = rotate_offset(left_x, left_y, cosine, sine);
        let rotated_right = rotate_offset(right_x, right_y, cosine, sine);
        let left_value = sample_offset(image, x, y, rotated_left.0, rotated_left.1);
        let right_value = sample_offset(image, x, y, rotated_right.0, rotated_right.1);
        if left_value < right_value {
            words[bit / 64] |= 1u64 << (bit % 64);
        }
    }
    words
}

fn brief_offset(seed: u32) -> (i32, i32) {
    let radius_squared = (ORB_PATCH_RADIUS - 1) * (ORB_PATCH_RADIUS - 1);
    let mut state = seed;
    for _ in 0..24 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let x = ((state >> 16) % 15) as i32 - 7;
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let y = ((state >> 16) % 15) as i32 - 7;
        if x * x + y * y <= radius_squared {
            return (x, y);
        }
    }
    (0, 0)
}

fn rotate_offset(x: i32, y: i32, cosine: f32, sine: f32) -> (i32, i32) {
    (
        (cosine * x as f32 - sine * y as f32).round() as i32,
        (sine * x as f32 + cosine * y as f32).round() as i32,
    )
}

fn match_features(
    source: &[Feature],
    target: &[Feature],
    algorithm: FeatureAlgorithm,
    options: FeatureOptions,
) -> Vec<FeatureMatch> {
    let mut matches = Vec::new();
    for (source_index, source_feature) in source.iter().enumerate() {
        let mut best: Option<(usize, f32)> = None;
        let mut second = f32::INFINITY;
        for (target_index, target_feature) in target.iter().enumerate() {
            if !matching_window_allows(source_feature, target_feature, algorithm, options) {
                continue;
            }
            let Some(distance) =
                descriptor_distance(&source_feature.descriptor, &target_feature.descriptor)
            else {
                continue;
            };
            match best {
                None => best = Some((target_index, distance)),
                Some((best_index, best_distance))
                    if distance < best_distance
                        || (distance == best_distance && target_index < best_index) =>
                {
                    second = best_distance;
                    best = Some((target_index, distance));
                }
                Some(_) if distance < second => second = distance,
                _ => {}
            }
        }

        let Some((target_index, best_distance)) = best else {
            continue;
        };
        let max_distance = match algorithm {
            FeatureAlgorithm::BaselineHarrisPatch => options.baseline_max_distance,
            FeatureAlgorithm::OrbStyle => options.orb_max_hamming as f32,
        };
        let ratio_ok = second.is_infinite() || best_distance < second * options.ratio_threshold;
        if !ratio_ok || best_distance > max_distance {
            continue;
        }

        let target_feature = &target[target_index];
        let reverse_best = source
            .iter()
            .enumerate()
            .filter(|(_, reverse_source)| {
                matching_window_allows(reverse_source, target_feature, algorithm, options)
            })
            .filter_map(|(reverse_index, reverse_source)| {
                descriptor_distance(&reverse_source.descriptor, &target_feature.descriptor)
                    .map(|distance| (reverse_index, distance))
            })
            .min_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });
        if !reverse_best.is_some_and(|(reverse_index, _)| reverse_index == source_index) {
            continue;
        }

        matches.push(FeatureMatch {
            source_index,
            target_index,
            distance: best_distance,
            dx: target_feature.point.x - source_feature.point.x,
            dy: target_feature.point.y - source_feature.point.y,
        });
    }
    matches
}

fn matching_window_allows(
    source: &Feature,
    target: &Feature,
    algorithm: FeatureAlgorithm,
    options: FeatureOptions,
) -> bool {
    match algorithm {
        FeatureAlgorithm::BaselineHarrisPatch => {
            let radius = options.match_radius as f32;
            (target.point.x - source.point.x).abs() <= radius
                && (target.point.y - source.point.y).abs() <= radius
        }
        FeatureAlgorithm::OrbStyle => true,
    }
}

fn descriptor_distance(left: &Descriptor, right: &Descriptor) -> Option<f32> {
    match (left, right) {
        (Descriptor::Patch(left), Descriptor::Patch(right)) if left.len() == right.len() => {
            let total = left
                .iter()
                .zip(right)
                .map(|(left, right)| {
                    let difference = i32::from(*left) - i32::from(*right);
                    difference * difference
                })
                .sum::<i32>();
            Some((total as f32 / left.len().max(1) as f32).sqrt())
        }
        (Descriptor::Binary(left), Descriptor::Binary(right)) => Some(
            left.iter()
                .zip(right)
                .map(|(left, right)| (left ^ right).count_ones())
                .sum::<u32>() as f32,
        ),
        _ => None,
    }
}

fn sample(image: &GrayImage, x: u32, y: u32) -> u8 {
    image.pixels[(y * image.width + x) as usize]
}

fn sample_offset(image: &GrayImage, x: u32, y: u32, dx: i32, dy: i32) -> u8 {
    let sample_x = (x as i32 + dx).clamp(0, image.width as i32 - 1) as u32;
    let sample_y = (y as i32 + dy).clamp(0, image.height as i32 - 1) as u32;
    sample(image, sample_x, sample_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured_rgba(width: u32, height: u32) -> Vec<u8> {
        let mut rgba = vec![0u8; width as usize * height as usize * 4];
        for y in 0..height {
            for x in 0..width {
                let base = ((x * 17 + y * 29 + (x * y) % 53) % 180 + 32) as u8;
                let checker = if (x / 9 + y / 7) % 2 == 0 { 42 } else { 0 };
                let value = base.saturating_add(checker);
                let index = ((y * width + x) * 4) as usize;
                rgba[index] = value;
                rgba[index + 1] = value.saturating_add(((x * 3 + y) % 19) as u8);
                rgba[index + 2] = value.saturating_sub(((x + y * 2) % 17) as u8);
                rgba[index + 3] = 255;
            }
        }
        rgba
    }

    fn translated_rgba(source: &[u8], width: u32, height: u32, dx: i32, dy: i32) -> Vec<u8> {
        let mut target = vec![0u8; source.len()];
        for y in 0..height {
            for x in 0..width {
                let target_x = x as i32 + dx;
                let target_y = y as i32 + dy;
                if target_x < 0
                    || target_x >= width as i32
                    || target_y < 0
                    || target_y >= height as i32
                {
                    continue;
                }
                let source_index = ((y * width + x) * 4) as usize;
                let target_index = (((target_y as u32) * width + target_x as u32) * 4) as usize;
                target[target_index..target_index + 4]
                    .copy_from_slice(&source[source_index..source_index + 4]);
            }
        }
        target
    }

    fn median(values: &mut [f32]) -> f32 {
        values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        values[values.len() / 2]
    }

    #[test]
    fn both_pipelines_recover_a_known_translation() {
        let width = 128;
        let height = 96;
        let source = textured_rgba(width, height);
        let target = translated_rgba(&source, width, height, 7, 4);
        let options = FeatureOptions {
            max_features: 300,
            min_feature_distance: 5,
            match_radius: 16,
            ..FeatureOptions::default()
        };

        for algorithm in [
            FeatureAlgorithm::BaselineHarrisPatch,
            FeatureAlgorithm::OrbStyle,
        ] {
            let analysis = analyze_rgba_pair(&source, &target, width, height, algorithm, options)
                .expect("synthetic feature analysis");
            assert!(analysis.source_features.len() >= 30, "{algorithm:?}");
            assert!(analysis.target_features.len() >= 30, "{algorithm:?}");
            assert!(analysis.matches.len() >= 12, "{algorithm:?}");
            let mut dx: Vec<_> = analysis
                .matches
                .iter()
                .map(|feature_match| feature_match.dx)
                .collect();
            let mut dy: Vec<_> = analysis
                .matches
                .iter()
                .map(|feature_match| feature_match.dy)
                .collect();
            assert!((median(&mut dx) - 7.0).abs() <= 1.5, "{algorithm:?}");
            assert!((median(&mut dy) - 4.0).abs() <= 1.5, "{algorithm:?}");
        }
    }

    #[test]
    fn orb_style_is_deterministic() {
        let width = 96;
        let height = 72;
        let source = textured_rgba(width, height);
        let target = translated_rgba(&source, width, height, -5, 3);
        let first = analyze_rgba_pair(
            &source,
            &target,
            width,
            height,
            FeatureAlgorithm::OrbStyle,
            FeatureOptions::default(),
        )
        .expect("first ORB-style analysis");
        let second = analyze_rgba_pair(
            &source,
            &target,
            width,
            height,
            FeatureAlgorithm::OrbStyle,
            FeatureOptions::default(),
        )
        .expect("second ORB-style analysis");
        assert_eq!(first.source_features.len(), second.source_features.len());
        assert_eq!(first.target_features.len(), second.target_features.len());
        assert_eq!(first.matches.len(), second.matches.len());
        for (left, right) in first.matches.iter().zip(&second.matches) {
            assert_eq!(left.source_index, right.source_index);
            assert_eq!(left.target_index, right.target_index);
            assert_eq!(left.distance, right.distance);
            assert_eq!(left.dx, right.dx);
            assert_eq!(left.dy, right.dy);
        }
    }
}
