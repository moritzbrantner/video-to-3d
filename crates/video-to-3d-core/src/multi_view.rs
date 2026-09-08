use super::{FeatureMatch, PairStats};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

const MIN_KEYFRAME_OVERLAP: f32 = 0.18;
const KEYFRAME_PARALLAX_BUDGET: f32 = 1.25;
const MIN_STRONG_PAIR_PARALLAX: f32 = 0.55;
const MIN_PNP_CORRESPONDENCES: usize = 8;

#[derive(Clone, Debug, Serialize)]
pub struct RegistrationCandidateStats {
    pub frame_index: usize,
    pub seed_landmark_correspondences: usize,
    pub pnp_ready: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct MultiViewStats {
    pub keyframes: Vec<usize>,
    pub track_count: usize,
    pub tracks_three_plus: usize,
    pub longest_track: usize,
    pub observations: usize,
    pub linked_pairs: usize,
    pub registration_candidates: Vec<RegistrationCandidateStats>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SeedLandmark {
    pub point_index: usize,
    pub source_feature_index: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RegistrationCorrespondence {
    pub seed_point_index: usize,
    pub feature_index: usize,
}

#[derive(Clone, Debug)]
pub(super) struct RegistrationCandidate {
    pub frame_index: usize,
    pub correspondences: Vec<RegistrationCorrespondence>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct TrackObservation {
    pub frame_index: usize,
    pub feature_index: usize,
}

#[derive(Clone, Debug)]
pub(super) struct FeatureTrack {
    pub observations: Vec<TrackObservation>,
}

#[derive(Clone, Debug)]
pub(super) struct MultiViewAnalysis {
    pub stats: MultiViewStats,
    pub registration_candidates: Vec<RegistrationCandidate>,
    pub tracks: Vec<FeatureTrack>,
    pub seed_track_indices: HashSet<usize>,
}

pub(super) fn analyze(
    pairs: &[PairStats],
    adjacent_matches: &[Vec<FeatureMatch>],
    seed_pair: Option<(usize, &[SeedLandmark])>,
) -> MultiViewAnalysis {
    let (tracks, membership) = build_feature_tracks(adjacent_matches);
    let keyframes = select_keyframes(pairs);
    let seed_tracks = seed_pair
        .map(|(pair_index, seed_landmarks)| {
            seed_track_pairs(&membership, pair_index, seed_landmarks)
        })
        .unwrap_or_default();
    let registration_candidates = seed_pair
        .map(|(pair_index, _)| {
            registration_candidates(&keyframes, &tracks, pair_index, &seed_tracks)
        })
        .unwrap_or_default();
    let registration_candidate_stats = registration_candidates
        .iter()
        .map(|candidate| RegistrationCandidateStats {
            frame_index: candidate.frame_index,
            seed_landmark_correspondences: candidate.correspondences.len(),
            pnp_ready: candidate.correspondences.len() >= MIN_PNP_CORRESPONDENCES,
        })
        .collect();
    let stats = MultiViewStats {
        keyframes,
        track_count: tracks.len(),
        tracks_three_plus: tracks
            .iter()
            .filter(|track| track.observations.len() >= 3)
            .count(),
        longest_track: tracks
            .iter()
            .map(|track| track.observations.len())
            .max()
            .unwrap_or(0),
        observations: tracks.iter().map(|track| track.observations.len()).sum(),
        linked_pairs: adjacent_matches
            .iter()
            .filter(|matches| !matches.is_empty())
            .count(),
        registration_candidates: registration_candidate_stats,
    };
    let seed_track_indices = seed_tracks
        .iter()
        .map(|&(_, track_index)| track_index)
        .collect();

    MultiViewAnalysis {
        stats,
        registration_candidates,
        tracks,
        seed_track_indices,
    }
}

fn build_feature_tracks(
    adjacent_matches: &[Vec<FeatureMatch>],
) -> (Vec<FeatureTrack>, HashMap<TrackObservation, usize>) {
    let mut tracks: Vec<FeatureTrack> = Vec::new();
    let mut membership: HashMap<TrackObservation, usize> = HashMap::new();

    for (pair_index, matches) in adjacent_matches.iter().enumerate() {
        for feature_match in matches {
            let source = TrackObservation {
                frame_index: pair_index,
                feature_index: feature_match.a,
            };
            let target = TrackObservation {
                frame_index: pair_index + 1,
                feature_index: feature_match.b,
            };

            if let Some(&track_index) = membership.get(&source) {
                match membership.get(&target).copied() {
                    Some(existing_track) if existing_track != track_index => {
                        // Adjacent matching is one-to-one, so this should not occur. Keep the
                        // earlier deterministic assignment rather than merging ambiguous tracks.
                    }
                    Some(_) => {}
                    None => {
                        tracks[track_index].observations.push(target);
                        membership.insert(target, track_index);
                    }
                }
            } else if membership.contains_key(&target) {
                // A target can only be claimed once by the one-to-one matcher. Ignore a
                // conflicting observation rather than creating a non-deterministic merge.
                continue;
            } else {
                let track_index = tracks.len();
                tracks.push(FeatureTrack {
                    observations: vec![source, target],
                });
                membership.insert(source, track_index);
                membership.insert(target, track_index);
            }
        }
    }

    (tracks, membership)
}

fn seed_track_pairs(
    membership: &HashMap<TrackObservation, usize>,
    seed_pair_index: usize,
    seed_landmarks: &[SeedLandmark],
) -> Vec<(usize, usize)> {
    seed_landmarks
        .iter()
        .filter_map(|seed_landmark| {
            membership
                .get(&TrackObservation {
                    frame_index: seed_pair_index,
                    feature_index: seed_landmark.source_feature_index,
                })
                .map(|&track_index| (seed_landmark.point_index, track_index))
        })
        .collect()
}

fn registration_candidates(
    keyframes: &[usize],
    tracks: &[FeatureTrack],
    seed_pair_index: usize,
    seed_tracks: &[(usize, usize)],
) -> Vec<RegistrationCandidate> {
    keyframes
        .iter()
        .copied()
        .filter(|&frame_index| frame_index != seed_pair_index && frame_index != seed_pair_index + 1)
        .map(|frame_index| {
            let correspondences = seed_tracks
                .iter()
                .filter_map(|&(seed_point_index, track_index)| {
                    tracks[track_index]
                        .observations
                        .iter()
                        .find(|observation| observation.frame_index == frame_index)
                        .map(|observation| RegistrationCorrespondence {
                            seed_point_index,
                            feature_index: observation.feature_index,
                        })
                })
                .collect();
            RegistrationCandidate {
                frame_index,
                correspondences,
            }
        })
        .collect()
}

fn select_keyframes(pairs: &[PairStats]) -> Vec<usize> {
    let Some(first_pair) = pairs.first() else {
        return Vec::new();
    };

    let mut keyframes = vec![first_pair.from_frame];
    let mut accumulated_parallax = 0.0f32;
    let mut strongest_pair: Option<(usize, f32)> = None;

    for pair in pairs {
        if pair.matches < 8 || pair.overlap_ratio < MIN_KEYFRAME_OVERLAP {
            accumulated_parallax = 0.0;
            continue;
        }
        accumulated_parallax += pair.median_parallax_residual.max(0.0);
        if !pair.low_parallax && pair.median_parallax_residual >= MIN_STRONG_PAIR_PARALLAX {
            let score = pair.median_parallax_residual * pair.overlap_ratio;
            if strongest_pair.is_none_or(|(_, best_score)| score > best_score) {
                strongest_pair = Some((pair.to_frame, score));
            }
        }

        if accumulated_parallax >= KEYFRAME_PARALLAX_BUDGET {
            if keyframes.last().copied() != Some(pair.to_frame) {
                keyframes.push(pair.to_frame);
            }
            accumulated_parallax = 0.0;
        }
    }

    if keyframes.len() == 1 {
        if let Some((frame_index, _)) = strongest_pair {
            keyframes.push(frame_index);
        }
    }

    keyframes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature_match(a: usize, b: usize) -> FeatureMatch {
        FeatureMatch {
            a,
            b,
            distance: 0.0,
        }
    }

    fn seed_landmarks(count: usize) -> Vec<SeedLandmark> {
        (0..count)
            .map(|index| SeedLandmark {
                point_index: index,
                source_feature_index: index,
            })
            .collect()
    }

    fn pair(
        from_frame: usize,
        overlap_ratio: f32,
        median_parallax_residual: f32,
        low_parallax: bool,
    ) -> PairStats {
        PairStats {
            from_frame,
            to_frame: from_frame + 1,
            features_from: 20,
            features_to: 20,
            matches: 12,
            overlap_ratio,
            median_dx: 0.0,
            median_dy: 0.0,
            median_motion: 2.0,
            median_parallax_residual,
            low_parallax,
        }
    }

    #[test]
    fn chains_adjacent_matches_into_multi_frame_tracks() {
        let matches = vec![
            vec![
                feature_match(0, 0),
                feature_match(1, 1),
                feature_match(2, 2),
            ],
            vec![
                feature_match(0, 0),
                feature_match(1, 1),
                feature_match(3, 3),
            ],
        ];
        let (tracks, _) = build_feature_tracks(&matches);
        let mut lengths: Vec<usize> = tracks
            .iter()
            .map(|track| track.observations.len())
            .collect();
        lengths.sort_unstable();

        assert_eq!(lengths, vec![2, 2, 3, 3]);
    }

    #[test]
    fn reports_registration_ready_keyframes_from_seed_landmark_tracks() {
        let seed_matches: Vec<FeatureMatch> =
            (0..10).map(|index| feature_match(index, index)).collect();
        let next_matches: Vec<FeatureMatch> =
            (0..9).map(|index| feature_match(index, index)).collect();
        let final_matches: Vec<FeatureMatch> =
            (0..8).map(|index| feature_match(index, index)).collect();
        let matches = vec![seed_matches, next_matches, final_matches];
        let (tracks, membership) = build_feature_tracks(&matches);
        let seeds = seed_landmarks(10);
        let seed_tracks = seed_track_pairs(&membership, 0, &seeds);

        let candidates = registration_candidates(&[0, 2, 3], &tracks, 0, &seed_tracks);

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].frame_index, 2);
        assert_eq!(candidates[0].correspondences.len(), 9);
        assert_eq!(candidates[0].correspondences[4].seed_point_index, 4);
        assert_eq!(candidates[0].correspondences[4].feature_index, 4);
        assert_eq!(candidates[1].frame_index, 3);
        assert_eq!(candidates[1].correspondences.len(), 8);
    }

    #[test]
    fn exposes_seed_track_membership_for_new_landmark_exclusion() {
        let matches = vec![
            (0..4).map(|index| feature_match(index, index)).collect(),
            (0..4).map(|index| feature_match(index, index)).collect(),
        ];
        let analysis = analyze(
            &[pair(0, 0.7, 0.7, false), pair(1, 0.7, 0.7, false)],
            &matches,
            Some((0, &seed_landmarks(2))),
        );

        assert_eq!(analysis.tracks.len(), 4);
        assert_eq!(analysis.seed_track_indices.len(), 2);
        assert!(analysis.seed_track_indices.contains(&0));
        assert!(analysis.seed_track_indices.contains(&1));
        assert_eq!(analysis.tracks[2].observations.len(), 3);
    }

    #[test]
    fn does_not_mark_sparse_track_overlap_as_pnp_ready() {
        let seed_matches: Vec<FeatureMatch> =
            (0..8).map(|index| feature_match(index, index)).collect();
        let next_matches: Vec<FeatureMatch> =
            (0..5).map(|index| feature_match(index, index)).collect();
        let matches = vec![seed_matches, next_matches];
        let analysis = analyze(
            &[pair(0, 0.7, 0.7, false), pair(1, 0.7, 0.7, false)],
            &matches,
            Some((0, &seed_landmarks(8))),
        );

        assert_eq!(analysis.stats.registration_candidates.len(), 1);
        assert_eq!(
            analysis.stats.registration_candidates[0].seed_landmark_correspondences,
            5
        );
        assert!(!analysis.stats.registration_candidates[0].pnp_ready);
    }

    #[test]
    fn selects_keyframes_from_overlap_and_accumulated_parallax() {
        let pairs = vec![
            pair(0, 0.65, 0.45, true),
            pair(1, 0.62, 0.46, true),
            pair(2, 0.58, 0.50, true),
            pair(3, 0.55, 0.70, false),
        ];

        assert_eq!(select_keyframes(&pairs), vec![0, 3]);
    }

    #[test]
    fn weak_or_stationary_links_do_not_create_keyframes() {
        let pairs = vec![
            pair(0, 0.70, 0.0, true),
            pair(1, 0.10, 0.8, false),
            pair(2, 0.75, 0.0, true),
        ];

        assert_eq!(select_keyframes(&pairs), vec![0]);
    }
}
