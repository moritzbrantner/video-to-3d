use super::{FeatureMatch, PairStats};
use serde::Serialize;
use std::collections::HashMap;

const MIN_KEYFRAME_OVERLAP: f32 = 0.18;
const KEYFRAME_PARALLAX_BUDGET: f32 = 1.25;
const MIN_STRONG_PAIR_PARALLAX: f32 = 0.55;

#[derive(Clone, Debug, Serialize)]
pub struct MultiViewStats {
    pub keyframes: Vec<usize>,
    pub track_count: usize,
    pub tracks_three_plus: usize,
    pub longest_track: usize,
    pub observations: usize,
    pub linked_pairs: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Observation {
    frame_index: usize,
    feature_index: usize,
}

#[derive(Clone, Debug)]
struct FeatureTrack {
    observations: Vec<Observation>,
}

pub(super) fn analyze(
    pairs: &[PairStats],
    adjacent_matches: &[Vec<FeatureMatch>],
) -> MultiViewStats {
    let tracks = build_feature_tracks(adjacent_matches);
    MultiViewStats {
        keyframes: select_keyframes(pairs),
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
        observations: tracks
            .iter()
            .map(|track| track.observations.len())
            .sum(),
        linked_pairs: adjacent_matches
            .iter()
            .filter(|matches| !matches.is_empty())
            .count(),
    }
}

fn build_feature_tracks(adjacent_matches: &[Vec<FeatureMatch>]) -> Vec<FeatureTrack> {
    let mut tracks: Vec<FeatureTrack> = Vec::new();
    let mut membership: HashMap<Observation, usize> = HashMap::new();

    for (pair_index, matches) in adjacent_matches.iter().enumerate() {
        for feature_match in matches {
            let source = Observation {
                frame_index: pair_index,
                feature_index: feature_match.a,
            };
            let target = Observation {
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

    tracks
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
            vec![feature_match(0, 0), feature_match(1, 1), feature_match(2, 2)],
            vec![feature_match(0, 0), feature_match(1, 1), feature_match(3, 3)],
        ];
        let tracks = build_feature_tracks(&matches);
        let mut lengths: Vec<usize> = tracks
            .iter()
            .map(|track| track.observations.len())
            .collect();
        lengths.sort_unstable();

        assert_eq!(lengths, vec![2, 2, 3, 3]);
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
