use super::{match_features, pnp, two_view, Feature, FeatureMatch, ReconstructionOptions};
use nalgebra::{Matrix3, Vector3};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

const MIN_REVISIT_MATCHES: usize = 12;
const MIN_REVISIT_OVERLAP: f32 = 0.18;
const MIN_RECOVERY_CORRESPONDENCES: usize = 8;
const MAX_REVISIT_CANDIDATES: usize = 16;
const MAX_REVISIT_PAIR_EVALUATIONS: usize = 12;
const MAX_REVISIT_FEATURES_PER_FRAME: usize = 192;

#[derive(Clone, Debug, Serialize)]
pub struct RevisitCandidateStats {
    pub from_frame: usize,
    pub to_frame: usize,
    pub matches: usize,
    pub overlap_ratio: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct RevisitRecoveryStats {
    pub frame_index: usize,
    pub source_frame_index: usize,
    pub matches: usize,
    pub correspondences: usize,
    pub accepted: bool,
    pub inliers: usize,
    pub median_reprojection_error_pixels: Option<f32>,
}

#[derive(Clone, Debug)]
struct SeedRevisitEvidence {
    target_frame: usize,
    matches: Vec<FeatureMatch>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RevisitStats {
    pub evaluated_pairs: usize,
    pub candidates: Vec<RevisitCandidateStats>,
    pub recoveries: Vec<RevisitRecoveryStats>,
    #[serde(skip)]
    seed_evidence: Vec<SeedRevisitEvidence>,
}

#[derive(Clone, Debug)]
pub(super) struct RecoveredView {
    pub frame_index: usize,
    pub correspondences: usize,
    pub inliers: usize,
    pub median_reprojection_error_pixels: f64,
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub camera_center: Vector3<f64>,
}

#[derive(Clone, Copy)]
pub(super) struct RevisitContext<'a> {
    features: &'a [Vec<Feature>],
    width: u32,
    height: u32,
    focal_pixels: f64,
    options: ReconstructionOptions,
}

impl<'a> RevisitContext<'a> {
    pub(super) fn new(
        features: &'a [Vec<Feature>],
        width: u32,
        height: u32,
        focal_pixels: f64,
        options: ReconstructionOptions,
    ) -> Self {
        Self {
            features,
            width,
            height,
            focal_pixels,
            options,
        }
    }
}

pub(super) fn analyze(
    context: &RevisitContext<'_>,
    keyframes: &[usize],
    seed_pair_index: Option<usize>,
) -> RevisitStats {
    let mut frames = keyframes.to_vec();
    if let Some(seed_pair_index) = seed_pair_index {
        frames.push(seed_pair_index);
        frames.push(seed_pair_index + 1);
    }
    frames.sort_unstable();
    frames.dedup();

    let pairs = preselect_pairs(&frames, seed_pair_index);
    let mut candidates = Vec::new();
    let mut seed_evidence = Vec::new();

    for &(from_frame, to_frame) in &pairs {
        let Some(source) = context.features.get(from_frame) else {
            continue;
        };
        let Some(target) = context.features.get(to_frame) else {
            continue;
        };
        let matches = mutual_unbounded_matches(source, target, context);
        let overlap_ratio = overlap_ratio(source.len(), target.len(), matches.len());
        if matches.len() < MIN_REVISIT_MATCHES || overlap_ratio < MIN_REVISIT_OVERLAP {
            continue;
        }

        candidates.push(RevisitCandidateStats {
            from_frame,
            to_frame,
            matches: matches.len(),
            overlap_ratio,
        });
        if let Some(seed_pair_index) = seed_pair_index {
            if from_frame == seed_pair_index {
                seed_evidence.push(SeedRevisitEvidence {
                    target_frame: to_frame,
                    matches,
                });
            } else if to_frame == seed_pair_index {
                seed_evidence.push(SeedRevisitEvidence {
                    target_frame: from_frame,
                    matches: matches
                        .into_iter()
                        .map(|feature_match| FeatureMatch {
                            a: feature_match.b,
                            b: feature_match.a,
                            distance: feature_match.distance,
                        })
                        .collect(),
                });
            }
        }
    }

    candidates.sort_by(|left, right| {
        right
            .matches
            .cmp(&left.matches)
            .then_with(|| right.overlap_ratio.total_cmp(&left.overlap_ratio))
            .then_with(|| left.from_frame.cmp(&right.from_frame))
            .then_with(|| left.to_frame.cmp(&right.to_frame))
    });
    candidates.truncate(MAX_REVISIT_CANDIDATES);
    seed_evidence.sort_by_key(|evidence| evidence.target_frame);

    RevisitStats {
        evaluated_pairs: pairs.len(),
        candidates,
        recoveries: Vec::new(),
        seed_evidence,
    }
}

pub(super) fn recover_failed_registrations(
    stats: &mut RevisitStats,
    seed_pair_index: usize,
    estimate: &two_view::TwoViewEstimate,
    candidate_frames: &[usize],
    registered_frames: &HashSet<usize>,
    context: &RevisitContext<'_>,
) -> Vec<RecoveredView> {
    let candidate_frames: HashSet<usize> = candidate_frames.iter().copied().collect();
    let seed_points_by_feature: HashMap<usize, Vector3<f64>> = estimate
        .points
        .iter()
        .map(|point| (point.source_feature_index, point.position))
        .collect();
    let mut recovered = Vec::new();

    for evidence in &stats.seed_evidence {
        let frame_index = evidence.target_frame;
        if !candidate_frames.contains(&frame_index) || registered_frames.contains(&frame_index) {
            continue;
        }
        debug_assert!(frame_index.abs_diff(seed_pair_index) > 1);
        let Some(target_features) = context.features.get(frame_index) else {
            continue;
        };

        let correspondences: Vec<pnp::PnpCorrespondence> = evidence
            .matches
            .iter()
            .filter_map(|feature_match| {
                let point = seed_points_by_feature.get(&feature_match.a)?;
                let feature = target_features.get(feature_match.b)?;
                Some(pnp::PnpCorrespondence {
                    point: *point,
                    x_pixels: feature.x as f64,
                    y_pixels: feature.y as f64,
                })
            })
            .collect();

        let pose = if correspondences.len() >= MIN_RECOVERY_CORRESPONDENCES {
            pnp::estimate_pose(
                &correspondences,
                context.width,
                context.height,
                context.focal_pixels,
            )
        } else {
            None
        };
        stats.recoveries.push(RevisitRecoveryStats {
            frame_index,
            source_frame_index: seed_pair_index,
            matches: evidence.matches.len(),
            correspondences: correspondences.len(),
            accepted: pose.is_some(),
            inliers: pose.as_ref().map_or(0, |pose| pose.inliers),
            median_reprojection_error_pixels: pose
                .as_ref()
                .map(|pose| pose.median_reprojection_error_pixels as f32),
        });

        if let Some(pose) = pose {
            recovered.push(RecoveredView {
                frame_index,
                correspondences: correspondences.len(),
                inliers: pose.inliers,
                median_reprojection_error_pixels: pose.median_reprojection_error_pixels,
                rotation: pose.rotation,
                translation: pose.translation,
                camera_center: pose.camera_center,
            });
        }
    }

    recovered.sort_by_key(|view| view.frame_index);
    stats.recoveries.sort_by_key(|attempt| attempt.frame_index);
    recovered
}

fn preselect_pairs(frames: &[usize], seed_pair_index: Option<usize>) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (left_offset, &from_frame) in frames.iter().enumerate() {
        for &to_frame in frames.iter().skip(left_offset + 1) {
            if to_frame <= from_frame + 1 {
                continue;
            }
            pairs.push((from_frame, to_frame));
        }
    }

    pairs.sort_by(|left, right| {
        let left_seed = seed_pair_index.is_some_and(|seed| left.0 == seed || left.1 == seed);
        let right_seed = seed_pair_index.is_some_and(|seed| right.0 == seed || right.1 == seed);
        right_seed
            .cmp(&left_seed)
            .then_with(|| (right.1 - right.0).cmp(&(left.1 - left.0)))
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
    });
    pairs.truncate(MAX_REVISIT_PAIR_EVALUATIONS);
    pairs
}

fn mutual_unbounded_matches(
    source: &[Feature],
    target: &[Feature],
    context: &RevisitContext<'_>,
) -> Vec<FeatureMatch> {
    let source = &source[..source.len().min(MAX_REVISIT_FEATURES_PER_FRAME)];
    let target = &target[..target.len().min(MAX_REVISIT_FEATURES_PER_FRAME)];
    let mut revisit_options = context.options;
    revisit_options.match_radius = context.width.saturating_add(context.height);
    let forward = match_features(source, target, revisit_options);
    let reverse = match_features(target, source, revisit_options);
    let reverse_pairs: HashSet<(usize, usize)> = reverse
        .into_iter()
        .map(|feature_match| (feature_match.a, feature_match.b))
        .collect();

    forward
        .into_iter()
        .filter(|feature_match| reverse_pairs.contains(&(feature_match.b, feature_match.a)))
        .collect()
}

fn overlap_ratio(source_features: usize, target_features: usize, matches: usize) -> f32 {
    let denominator = source_features
        .min(MAX_REVISIT_FEATURES_PER_FRAME)
        .min(target_features.min(MAX_REVISIT_FEATURES_PER_FRAME));
    if denominator == 0 {
        0.0
    } else {
        matches as f32 / denominator as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature(index: usize, x: u32, y: u32) -> Feature {
        Feature {
            x,
            y,
            score: 100.0 - index as f32,
            descriptor: vec![index as i16 * 12, index as i16 * 7, index as i16 * 3],
        }
    }

    #[test]
    fn revisit_matching_is_mutual_and_not_limited_by_adjacent_radius() {
        let source: Vec<Feature> = (0..12)
            .map(|index| feature(index, 20 + index as u32, 25 + index as u32))
            .collect();
        let target: Vec<Feature> = (0..12)
            .map(|index| feature(index, 300 + index as u32, 220 + index as u32))
            .collect();
        let options = ReconstructionOptions {
            match_radius: 8,
            ..ReconstructionOptions::default()
        };
        let features = vec![source.clone(), target.clone()];
        let context = RevisitContext::new(&features, 640, 480, 500.0, options);

        assert!(match_features(&source, &target, options).is_empty());
        let matches = mutual_unbounded_matches(&source, &target, &context);
        assert_eq!(matches.len(), 12);
        assert!(matches
            .iter()
            .enumerate()
            .all(|(index, feature_match)| feature_match.a == index && feature_match.b == index));
    }

    #[test]
    fn revisit_analysis_reports_non_adjacent_keyframe_matches_only() {
        let base: Vec<Feature> = (0..12)
            .map(|index| feature(index, 40 + index as u32, 60 + index as u32))
            .collect();
        let features = vec![base.clone(), base.clone(), base.clone()];
        let context =
            RevisitContext::new(&features, 640, 480, 500.0, ReconstructionOptions::default());
        let stats = analyze(&context, &[0, 1, 2], Some(0));

        assert_eq!(stats.evaluated_pairs, 1);
        assert_eq!(stats.candidates.len(), 1);
        assert_eq!(stats.candidates[0].from_frame, 0);
        assert_eq!(stats.candidates[0].to_frame, 2);
        assert_eq!(stats.candidates[0].matches, 12);
    }

    #[test]
    fn revisit_analysis_bounds_pair_evaluations_before_matching() {
        let frames: Vec<usize> = (0..18).collect();
        let pairs = preselect_pairs(&frames, Some(0));

        assert_eq!(pairs.len(), MAX_REVISIT_PAIR_EVALUATIONS);
        assert!(pairs.iter().all(|(from, to)| *to > *from + 1));
        assert!(pairs.iter().all(|(from, to)| *from == 0 || *to == 0));
    }

    #[test]
    fn direct_seed_revisit_can_recover_a_failed_keyframe_pose() {
        let width = 640;
        let height = 480;
        let focal = 500.0;
        let points = [
            Vector3::new(-0.8, -0.5, 3.8),
            Vector3::new(-0.3, 0.4, 4.4),
            Vector3::new(0.2, -0.6, 5.1),
            Vector3::new(0.7, 0.5, 5.8),
            Vector3::new(-0.6, 0.7, 6.2),
            Vector3::new(0.5, -0.2, 6.8),
            Vector3::new(-0.1, 0.1, 7.4),
            Vector3::new(0.9, -0.7, 8.0),
            Vector3::new(-0.9, 0.2, 8.5),
            Vector3::new(0.4, 0.8, 9.0),
            Vector3::new(0.1, -0.8, 9.6),
            Vector3::new(-0.4, -0.1, 10.2),
        ];
        let camera_center = Vector3::new(0.65, -0.08, 0.12);
        let rotation = Matrix3::identity();
        let translation = -camera_center;
        let seed_features: Vec<Feature> = points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let x = (point.x / point.z * focal + width as f64 * 0.5).round() as u32;
                let y = (point.y / point.z * focal + height as f64 * 0.5).round() as u32;
                feature(index, x, y)
            })
            .collect();
        let target_features: Vec<Feature> = points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let camera_point = rotation * point + translation;
                let x =
                    (camera_point.x / camera_point.z * focal + width as f64 * 0.5).round() as u32;
                let y =
                    (camera_point.y / camera_point.z * focal + height as f64 * 0.5).round() as u32;
                feature(index, x, y)
            })
            .collect();
        let estimate = two_view::TwoViewEstimate {
            rotation: Matrix3::identity(),
            translation: Vector3::new(-0.5, 0.0, 0.0),
            camera_center: Vector3::new(0.5, 0.0, 0.0),
            matches: points.len(),
            inliers: points.len(),
            median_sampson_error_pixels: 0.0,
            median_reprojection_error_pixels: 0.0,
            median_triangulation_angle_degrees: 4.0,
            points: points
                .iter()
                .enumerate()
                .map(|(index, point)| two_view::TriangulatedPoint {
                    source_feature_index: index,
                    descriptor_distance: 0.0,
                    position: *point,
                    reprojection_error_pixels: 0.0,
                    triangulation_angle_degrees: 4.0,
                })
                .collect(),
        };
        let features = vec![
            seed_features.clone(),
            seed_features.clone(),
            seed_features.clone(),
            target_features,
        ];
        let context = RevisitContext::new(
            &features,
            width,
            height,
            focal,
            ReconstructionOptions::default(),
        );
        let mut stats = analyze(&context, &[0, 3], Some(0));
        let recovered = recover_failed_registrations(
            &mut stats,
            0,
            &estimate,
            &[3],
            &HashSet::from([0, 1]),
            &context,
        );

        assert_eq!(stats.recoveries.len(), 1);
        assert!(stats.recoveries[0].accepted);
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].frame_index, 3);
        assert!((recovered[0].camera_center - camera_center).norm() < 0.2);
    }

    #[test]
    fn adjacent_frame_before_seed_is_never_revisit_recovery_evidence() {
        let base: Vec<Feature> = (0..12)
            .map(|index| feature(index, 40 + index as u32, 60 + index as u32))
            .collect();
        let features = vec![base.clone(), base.clone(), base.clone(), base];
        let context =
            RevisitContext::new(&features, 640, 480, 500.0, ReconstructionOptions::default());
        let stats = analyze(&context, &[1, 2, 3], Some(2));

        assert!(stats
            .candidates
            .iter()
            .all(|candidate| candidate.from_frame.abs_diff(candidate.to_frame) > 1));
        assert!(stats
            .seed_evidence
            .iter()
            .all(|evidence| evidence.target_frame.abs_diff(2) > 1));
        assert!(!stats
            .seed_evidence
            .iter()
            .any(|evidence| evidence.target_frame == 1));
    }
}
