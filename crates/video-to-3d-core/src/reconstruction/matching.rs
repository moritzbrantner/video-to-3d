//! Descriptor matching, motion-guided matching and global motion compensation.

use super::types::{Feature, FeatureMatch, ReconstructionOptions};
use std::cmp::Ordering;
use std::collections::HashSet;

const MOTION_GUIDED_COARSE_RATIO_THRESHOLD: f32 = 0.72;
const MOTION_GUIDED_MIN_SUPPORT: usize = 4;

pub(crate) fn match_features(
    a: &[Feature],
    b: &[Feature],
    options: ReconstructionOptions,
) -> Vec<FeatureMatch> {
    match_features_around_offset(a, b, options, 0.0, 0.0)
}

pub(crate) fn motion_guided_matches(
    a: &[Feature],
    b: &[Feature],
    options: ReconstructionOptions,
) -> Option<Vec<FeatureMatch>> {
    let coarse = coarse_global_matches(a, b, options);
    let (offset_x, offset_y) = dominant_displacement(a, b, &coarse, options.match_radius)?;
    let guided = match_features_around_offset(a, b, options, offset_x, offset_y);
    (!guided.is_empty()).then_some(guided)
}

fn coarse_global_matches(
    a: &[Feature],
    b: &[Feature],
    options: ReconstructionOptions,
) -> Vec<FeatureMatch> {
    let ratio_threshold = options
        .ratio_threshold
        .min(MOTION_GUIDED_COARSE_RATIO_THRESHOLD);
    let max_descriptor_distance = options.max_descriptor_distance * 0.9;
    let mut proposals = Vec::new();
    for (a_index, feature_a) in a.iter().enumerate() {
        let mut best: Option<(usize, f32)> = None;
        let mut second = f32::INFINITY;
        for (b_index, feature_b) in b.iter().enumerate() {
            let distance = descriptor_distance(&feature_a.descriptor, &feature_b.descriptor);
            match best {
                None => best = Some((b_index, distance)),
                Some((_, best_distance)) if distance < best_distance => {
                    second = best_distance;
                    best = Some((b_index, distance));
                }
                Some(_) if distance < second => second = distance,
                _ => {}
            }
        }
        if let Some((b_index, best_distance)) = best {
            let ratio_ok = second.is_infinite() || best_distance < second * ratio_threshold;
            if ratio_ok && best_distance <= max_descriptor_distance {
                proposals.push(FeatureMatch {
                    a: a_index,
                    b: b_index,
                    distance: best_distance,
                });
            }
        }
    }
    proposals.sort_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b[right.b]
                    .score
                    .partial_cmp(&b[left.b].score)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.a.cmp(&right.a))
            .then_with(|| left.b.cmp(&right.b))
    });
    let mut used_b = HashSet::new();
    proposals
        .into_iter()
        .filter(|proposal| used_b.insert(proposal.b))
        .filter(|proposal| {
            let target = &b[proposal.b];
            let reverse_best = a
                .iter()
                .enumerate()
                .map(|(a_index, source)| {
                    (
                        a_index,
                        descriptor_distance(&source.descriptor, &target.descriptor),
                    )
                })
                .min_by(|left, right| {
                    left.1
                        .partial_cmp(&right.1)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| left.0.cmp(&right.0))
                });
            reverse_best.is_some_and(|(a_index, _)| a_index == proposal.a)
        })
        .collect()
}

fn dominant_displacement(
    a: &[Feature],
    b: &[Feature],
    matches: &[FeatureMatch],
    local_radius: u32,
) -> Option<(f32, f32)> {
    if matches.len() < MOTION_GUIDED_MIN_SUPPORT {
        return None;
    }
    let displacements: Vec<(f32, f32)> = matches
        .iter()
        .map(|feature_match| {
            let source = &a[feature_match.a];
            let target = &b[feature_match.b];
            (
                target.x as f32 - source.x as f32,
                target.y as f32 - source.y as f32,
            )
        })
        .collect();
    let tolerance = (local_radius as f32 * 0.35).clamp(4.0, 18.0);
    let tolerance_sq = tolerance * tolerance;
    let mut best_index = 0usize;
    let mut best_support = 0usize;
    let mut best_residual = f32::INFINITY;
    for (candidate_index, &(candidate_x, candidate_y)) in displacements.iter().enumerate() {
        let mut support = 0usize;
        let mut residual = 0.0f32;
        for &(dx, dy) in &displacements {
            let offset_x = dx - candidate_x;
            let offset_y = dy - candidate_y;
            let residual_sq = offset_x * offset_x + offset_y * offset_y;
            if residual_sq <= tolerance_sq {
                support += 1;
                residual += residual_sq;
            }
        }
        if support > best_support
            || (support == best_support && residual < best_residual)
            || (support == best_support
                && residual == best_residual
                && candidate_index < best_index)
        {
            best_index = candidate_index;
            best_support = support;
            best_residual = residual;
        }
    }
    if best_support < MOTION_GUIDED_MIN_SUPPORT || best_support * 2 < matches.len() {
        return None;
    }
    let (center_x, center_y) = displacements[best_index];
    let mut inlier_dx = Vec::with_capacity(best_support);
    let mut inlier_dy = Vec::with_capacity(best_support);
    for (dx, dy) in displacements {
        let offset_x = dx - center_x;
        let offset_y = dy - center_y;
        if offset_x * offset_x + offset_y * offset_y <= tolerance_sq {
            inlier_dx.push(dx);
            inlier_dy.push(dy);
        }
    }
    Some((median(&mut inlier_dx), median(&mut inlier_dy)))
}

fn match_features_around_offset(
    a: &[Feature],
    b: &[Feature],
    options: ReconstructionOptions,
    offset_x: f32,
    offset_y: f32,
) -> Vec<FeatureMatch> {
    let radius_sq = (options.match_radius * options.match_radius) as f32;
    let mut proposals = Vec::new();
    for (a_index, feature_a) in a.iter().enumerate() {
        let mut best: Option<(usize, f32)> = None;
        let mut second = f32::INFINITY;
        for (b_index, feature_b) in b.iter().enumerate() {
            let dx = feature_b.x as f32 - feature_a.x as f32 - offset_x;
            let dy = feature_b.y as f32 - feature_a.y as f32 - offset_y;
            if dx * dx + dy * dy > radius_sq {
                continue;
            }
            let distance = descriptor_distance(&feature_a.descriptor, &feature_b.descriptor);
            match best {
                None => best = Some((b_index, distance)),
                Some((_, best_distance)) if distance < best_distance => {
                    second = best_distance;
                    best = Some((b_index, distance));
                }
                Some(_) if distance < second => second = distance,
                _ => {}
            }
        }
        if let Some((b_index, best_distance)) = best {
            let ratio_ok = second.is_infinite() || best_distance < second * options.ratio_threshold;
            if ratio_ok && best_distance <= options.max_descriptor_distance {
                proposals.push(FeatureMatch {
                    a: a_index,
                    b: b_index,
                    distance: best_distance,
                });
            }
        }
    }
    proposals.sort_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b[right.b]
                    .score
                    .partial_cmp(&b[left.b].score)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.a.cmp(&right.a))
            .then_with(|| left.b.cmp(&right.b))
    });
    let mut used_b = HashSet::new();
    proposals
        .into_iter()
        .filter(|proposal| used_b.insert(proposal.b))
        .collect()
}

fn descriptor_distance(a: &[i16], b: &[i16]) -> f32 {
    let total: i32 = a
        .iter()
        .zip(b)
        .map(|(left, right)| (*left as i32 - *right as i32).abs())
        .sum();
    total as f32 / a.len() as f32
}

pub(super) fn compensate_global_motion(
    a: &[Feature],
    b: &[Feature],
    matches: &[FeatureMatch],
    width: u32,
    height: u32,
) -> (f32, f32, Vec<f32>) {
    if matches.is_empty() {
        return (0.0, 0.0, Vec::new());
    }

    let center_x = width as f32 * 0.5;
    let center_y = height as f32 * 0.5;
    let mut raw_dx = Vec::with_capacity(matches.len());
    let mut raw_dy = Vec::with_capacity(matches.len());
    for feature_match in matches {
        let source = &a[feature_match.a];
        let target = &b[feature_match.b];
        raw_dx.push(target.x as f32 - source.x as f32);
        raw_dy.push(target.y as f32 - source.y as f32);
    }

    let mut initial_dx = raw_dx.clone();
    let mut initial_dy = raw_dy.clone();
    let initial_tx = median(&mut initial_dx);
    let initial_ty = median(&mut initial_dy);
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (index, feature_match) in matches.iter().enumerate() {
        let source = &a[feature_match.a];
        let x = source.x as f32 - center_x;
        let y = source.y as f32 - center_y;
        let dx = raw_dx[index] - initial_tx;
        let dy = raw_dy[index] - initial_ty;
        numerator += x * dy - y * dx;
        denominator += x * x + y * y;
    }
    let rotation = if denominator > f32::EPSILON {
        (numerator / denominator).clamp(-0.35, 0.35)
    } else {
        0.0
    };

    let mut corrected_dx = Vec::with_capacity(matches.len());
    let mut corrected_dy = Vec::with_capacity(matches.len());
    for (index, feature_match) in matches.iter().enumerate() {
        let source = &a[feature_match.a];
        let x = source.x as f32 - center_x;
        let y = source.y as f32 - center_y;
        corrected_dx.push(raw_dx[index] + rotation * y);
        corrected_dy.push(raw_dy[index] - rotation * x);
    }
    let translation_x = median(&mut corrected_dx);
    let translation_y = median(&mut corrected_dy);

    let residuals = matches
        .iter()
        .enumerate()
        .map(|(index, feature_match)| {
            let source = &a[feature_match.a];
            let x = source.x as f32 - center_x;
            let y = source.y as f32 - center_y;
            let predicted_dx = translation_x - rotation * y;
            let predicted_dy = translation_y + rotation * x;
            (raw_dx[index] - predicted_dx).hypot(raw_dy[index] - predicted_dy)
        })
        .collect();

    (translation_x, translation_y, residuals)
}

pub(super) fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}
