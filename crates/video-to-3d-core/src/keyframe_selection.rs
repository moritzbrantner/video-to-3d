//! Motion-aware keyframe selection and clip segmentation.
//!
//! Adjacent links that lose overlap or jump implausibly far between two usable frames
//! split the clip into segments; no parallax accumulates across a split. Inside a
//! segment, frames that are blurred or badly exposed are rejected, and a keyframe is
//! taken whenever accumulated parallax reaches the budget. Every frame gets a recorded
//! decision so the UI can explain the selection.

use serde::Serialize;

use crate::input_readiness::{FrameEvidence, THRESHOLDS};
use crate::PairStats;

pub(crate) const MIN_KEYFRAME_OVERLAP: f32 = 0.18;
const MIN_KEYFRAME_MATCHES: usize = 8;
const KEYFRAME_PARALLAX_BUDGET: f32 = 1.25;
const MIN_STRONG_PAIR_PARALLAX: f32 = 0.55;
/// Median feature motion, as a fraction of the frame diagonal, beyond which an
/// otherwise matched link is a teleport-like discontinuity rather than camera motion.
const MAX_LINK_MOTION_DIAGONAL_FRACTION: f32 = 0.35;
/// A textured frame is blurred when it falls below the readiness sharpness threshold
/// and this fraction of its segment's median sharpness (motion blur, not soft content).
const RELATIVE_BLUR_FRACTION: f32 = 0.8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameDecision {
    Keyframe,
    /// Usable, but too little parallax since the last keyframe.
    Redundant,
    Blurred,
    PoorExposure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentBreak {
    /// Overlap collapsed and exposure jumped: a hard cut.
    HardCut,
    /// Overlap collapsed without an exposure jump: tracking was lost.
    LostOverlap,
    /// Features still match but moved implausibly far in one step.
    MotionJump,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FrameSelectionStats {
    pub frame_index: usize,
    pub segment: usize,
    pub decision: FrameDecision,
    pub sharpness_ratio: f32,
    pub mean_luma: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ClipSegmentStats {
    pub first_frame: usize,
    pub last_frame: usize,
    pub keyframes: usize,
    /// Why this segment ends; `None` for the last segment.
    pub ends_with: Option<SegmentBreak>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct KeyframeSelectionStats {
    pub segments: Vec<ClipSegmentStats>,
    pub frames: Vec<FrameSelectionStats>,
}

pub(crate) struct KeyframeSelection {
    /// Ascending frame indices across all segments.
    pub(crate) keyframes: Vec<usize>,
    pub(crate) stats: KeyframeSelectionStats,
}

impl KeyframeSelection {
    /// Keyframes in the segment that contains `frame_index`.
    pub(crate) fn segment_keyframes(&self, frame_index: usize) -> Vec<usize> {
        let Some(segment) = self
            .stats
            .frames
            .get(frame_index)
            .map(|frame| frame.segment)
        else {
            return Vec::new();
        };
        self.keyframes
            .iter()
            .copied()
            .filter(|&keyframe| self.stats.frames[keyframe].segment == segment)
            .collect()
    }
}

/// `pairs` are the adjacent links `(i, i + 1)`; `frames` has one entry per frame.
pub(crate) fn select(
    pairs: &[PairStats],
    frames: &[FrameEvidence],
    width: u32,
    height: u32,
) -> KeyframeSelection {
    let diagonal = (width as f32).hypot(height as f32);
    let exposure_ok = |frame: &FrameEvidence| {
        frame.mean_luma >= THRESHOLDS.min_mean_luma
            && frame.mean_luma <= THRESHOLDS.max_mean_luma
            && frame.clipped_fraction <= THRESHOLDS.max_clipped_fraction
            && frame.crushed_fraction <= THRESHOLDS.max_clipped_fraction
    };

    // Segments first: a break needs two frames that are usable on exposure, so a single
    // bad frame is rejected inside its segment instead of splitting the clip.
    let mut segment_of = vec![0; frames.len()];
    let mut breaks = Vec::new();
    for pair in pairs {
        let (Some(from), Some(to)) = (frames.get(pair.from_frame), frames.get(pair.to_frame))
        else {
            continue;
        };
        let reason = link_break(pair, from, to, diagonal).filter(|_| exposure_ok(from) && exposure_ok(to));
        if let Some(reason) = reason {
            breaks.push(reason);
        }
        segment_of[pair.to_frame] = breaks.len();
    }

    let mut decisions = vec![FrameDecision::Redundant; frames.len()];
    let mut segments = Vec::with_capacity(breaks.len() + 1);
    let mut keyframes = Vec::new();
    for segment in 0..=breaks.len() {
        let members: Vec<usize> = (0..frames.len())
            .filter(|&frame| segment_of[frame] == segment)
            .collect();
        let (Some(&first_frame), Some(&last_frame)) = (members.first(), members.last()) else {
            continue;
        };
        let median_sharpness = median(members.iter().map(|&frame| frames[frame].sharpness_ratio));
        for &frame in &members {
            let evidence = &frames[frame];
            decisions[frame] = if !exposure_ok(evidence) {
                FrameDecision::PoorExposure
            } else if evidence.texture_density >= THRESHOLDS.min_texture_density
                && evidence.sharpness_ratio < THRESHOLDS.min_sharpness_ratio
                && evidence.sharpness_ratio < RELATIVE_BLUR_FRACTION * median_sharpness
            {
                FrameDecision::Blurred
            } else {
                FrameDecision::Redundant
            };
        }
        let segment_pairs: Vec<&PairStats> = pairs
            .iter()
            .filter(|pair| segment_of[pair.from_frame] == segment && segment_of[pair.to_frame] == segment)
            .collect();
        let selected = select_in_segment(&members, &segment_pairs, &decisions);
        for &frame in &selected {
            decisions[frame] = FrameDecision::Keyframe;
        }
        segments.push(ClipSegmentStats {
            first_frame,
            last_frame,
            keyframes: selected.len(),
            ends_with: breaks.get(segment).copied(),
        });
        keyframes.extend(selected);
    }

    KeyframeSelection {
        keyframes,
        stats: KeyframeSelectionStats {
            segments,
            frames: frames
                .iter()
                .enumerate()
                .map(|(frame_index, evidence)| FrameSelectionStats {
                    frame_index,
                    segment: segment_of[frame_index],
                    decision: decisions[frame_index],
                    sharpness_ratio: evidence.sharpness_ratio,
                    mean_luma: evidence.mean_luma,
                })
                .collect(),
        },
    }
}

fn link_break(
    pair: &PairStats,
    from: &FrameEvidence,
    to: &FrameEvidence,
    diagonal: f32,
) -> Option<SegmentBreak> {
    if pair.matches < MIN_KEYFRAME_MATCHES || pair.overlap_ratio < MIN_KEYFRAME_OVERLAP {
        let brighter = from.mean_luma.max(to.mean_luma).max(1.0);
        let exposure_change = (from.mean_luma - to.mean_luma).abs() / brighter;
        return Some(if exposure_change > THRESHOLDS.max_exposure_change {
            SegmentBreak::HardCut
        } else {
            SegmentBreak::LostOverlap
        });
    }
    (pair.median_motion > MAX_LINK_MOTION_DIAGONAL_FRACTION * diagonal)
        .then_some(SegmentBreak::MotionJump)
}

/// The parallax-budget rule within one segment. A keyframe that would land on a rejected
/// frame moves to the next usable frame; links touching rejected frames still reset the
/// budget, as weak links always have.
fn select_in_segment(
    members: &[usize],
    pairs: &[&PairStats],
    decisions: &[FrameDecision],
) -> Vec<usize> {
    let usable = |frame: usize| decisions[frame] == FrameDecision::Redundant;
    let Some(&first) = members.iter().find(|&&frame| usable(frame)) else {
        // A segment keeps its anchor even when no frame passes the quality gates.
        return members.first().copied().into_iter().collect();
    };
    let mut keyframes = vec![first];
    let mut accumulated_parallax = 0.0f32;
    let mut pending = false;
    let mut strongest_pair: Option<(usize, f32)> = None;

    for pair in pairs {
        let weak = pair.matches < MIN_KEYFRAME_MATCHES || pair.overlap_ratio < MIN_KEYFRAME_OVERLAP;
        if weak || !usable(pair.from_frame) && !usable(pair.to_frame) {
            accumulated_parallax = 0.0;
            pending = false;
            continue;
        }
        accumulated_parallax += pair.median_parallax_residual.max(0.0);
        if !pair.low_parallax
            && pair.median_parallax_residual >= MIN_STRONG_PAIR_PARALLAX
            && usable(pair.to_frame)
        {
            let score = pair.median_parallax_residual * pair.overlap_ratio;
            if strongest_pair.is_none_or(|(_, best_score)| score > best_score) {
                strongest_pair = Some((pair.to_frame, score));
            }
        }
        if accumulated_parallax >= KEYFRAME_PARALLAX_BUDGET || pending {
            if usable(pair.to_frame) {
                if keyframes.last().copied() != Some(pair.to_frame) {
                    keyframes.push(pair.to_frame);
                }
                accumulated_parallax = 0.0;
                pending = false;
            } else {
                pending = true;
            }
        }
    }

    if keyframes.len() == 1 {
        if let Some((frame_index, _)) = strongest_pair {
            keyframes.push(frame_index);
        }
    }
    keyframes
}

/// Selection over sharp, well-exposed frames: for tests that only exercise links.
#[cfg(test)]
pub(crate) fn select_for_pairs(pairs: &[PairStats]) -> KeyframeSelection {
    let count = pairs.iter().map(|pair| pair.to_frame + 1).max().unwrap_or(1);
    let frames: Vec<FrameEvidence> = (0..count)
        .map(|frame_index| FrameEvidence {
            frame_index,
            mean_luma: 120.0,
            crushed_fraction: 0.0,
            clipped_fraction: 0.0,
            texture_density: 0.2,
            sharpness_ratio: 0.9,
            blockiness: 1.0,
        })
        .collect();
    select(pairs, &frames, 320, 240)
}

fn median(values: impl Iterator<Item = f32>) -> f32 {
    let mut values: Vec<f32> = values.collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;

    fn pair(from_frame: usize, overlap_ratio: f32, residual: f32, low_parallax: bool) -> PairStats {
        PairStats {
            from_frame,
            to_frame: from_frame + 1,
            features_from: 20,
            features_to: 20,
            matches: 12,
            overlap_ratio,
            median_dx: 1.0,
            median_dy: 0.0,
            median_motion: 1.0,
            median_parallax_residual: residual,
            low_parallax,
        }
    }

    fn frame(frame_index: usize) -> FrameEvidence {
        FrameEvidence {
            frame_index,
            mean_luma: 120.0,
            crushed_fraction: 0.0,
            clipped_fraction: 0.0,
            texture_density: 0.2,
            sharpness_ratio: 0.9,
            blockiness: 1.0,
        }
    }

    fn frames(count: usize) -> Vec<FrameEvidence> {
        (0..count).map(frame).collect()
    }

    fn decisions(selection: &KeyframeSelection) -> Vec<FrameDecision> {
        selection.stats.frames.iter().map(|frame| frame.decision).collect()
    }

    #[test]
    fn selects_keyframes_from_overlap_and_accumulated_parallax() {
        let pairs = [pair(0, 0.8, 0.4, true), pair(1, 0.8, 0.4, true), pair(2, 0.8, 0.5, false)];

        let selection = select(&pairs, &frames(4), WIDTH, HEIGHT);

        assert_eq!(selection.keyframes, vec![0, 3]);
        assert_eq!(selection.stats.segments.len(), 1);
        use FrameDecision::*;
        assert_eq!(decisions(&selection), vec![Keyframe, Redundant, Redundant, Keyframe]);
    }

    #[test]
    fn weak_or_stationary_links_do_not_create_keyframes() {
        let pairs = [pair(0, 0.8, 0.0, true), pair(1, 0.8, 0.0, true)];

        let selection = select(&pairs, &frames(3), WIDTH, HEIGHT);

        assert_eq!(selection.keyframes, vec![0]);
    }

    #[test]
    fn a_hard_cut_splits_the_clip_and_starts_a_new_segment_keyframe() {
        let mut evidence = frames(6);
        for frame in &mut evidence[3..] {
            frame.mean_luma = 60.0;
        }
        let mut cut = pair(2, 0.02, 0.0, true);
        cut.matches = 1;
        let pairs = [
            pair(0, 0.8, 0.7, false),
            pair(1, 0.8, 0.7, false),
            cut,
            pair(3, 0.8, 0.7, false),
            pair(4, 0.8, 0.7, false),
        ];

        let selection = select(&pairs, &evidence, WIDTH, HEIGHT);

        assert_eq!(
            selection.stats.segments,
            vec![
                ClipSegmentStats {
                    first_frame: 0,
                    last_frame: 2,
                    keyframes: 2,
                    ends_with: Some(SegmentBreak::HardCut),
                },
                ClipSegmentStats {
                    first_frame: 3,
                    last_frame: 5,
                    keyframes: 2,
                    ends_with: None,
                },
            ]
        );
        assert_eq!(selection.keyframes, vec![0, 2, 3, 5]);
        assert_eq!(selection.segment_keyframes(4), vec![3, 5]);
    }

    #[test]
    fn a_matched_teleport_is_a_motion_jump() {
        let mut jump = pair(1, 0.6, 0.2, false);
        jump.median_motion = 0.5 * (WIDTH as f32).hypot(HEIGHT as f32);
        let selection = select(&[pair(0, 0.8, 0.2, true), jump], &frames(3), WIDTH, HEIGHT);

        assert_eq!(selection.stats.segments[0].ends_with, Some(SegmentBreak::MotionJump));
        assert_eq!(selection.keyframes, vec![0, 2]);
    }

    #[test]
    fn a_blurred_frame_is_rejected_and_its_keyframe_moves_to_the_next_sharp_frame() {
        let mut evidence = frames(5);
        evidence[3].sharpness_ratio = 0.3;
        let pairs = [
            pair(0, 0.8, 0.5, false),
            pair(1, 0.8, 0.5, false),
            pair(2, 0.8, 0.5, false),
            pair(3, 0.8, 0.5, false),
        ];

        let selection = select(&pairs, &evidence, WIDTH, HEIGHT);

        use FrameDecision::*;
        assert_eq!(selection.stats.segments.len(), 1, "one bad frame does not split the clip");
        assert_eq!(decisions(&selection), vec![Keyframe, Redundant, Redundant, Blurred, Keyframe]);
    }

    #[test]
    fn soft_but_consistent_footage_is_not_rejected_as_blurred() {
        let mut evidence = frames(3);
        for frame in &mut evidence {
            frame.sharpness_ratio = 0.4;
        }
        let selection = select(&[pair(0, 0.8, 0.7, false), pair(1, 0.8, 0.7, false)], &evidence, WIDTH, HEIGHT);

        assert!(selection.stats.frames.iter().all(|frame| frame.decision != FrameDecision::Blurred));
    }

    #[test]
    fn poorly_exposed_frames_never_become_keyframes() {
        let mut evidence = frames(3);
        evidence[0].mean_luma = 10.0;
        let selection = select(&[pair(0, 0.8, 0.7, false), pair(1, 0.8, 0.7, false)], &evidence, WIDTH, HEIGHT);

        assert_eq!(selection.stats.frames[0].decision, FrameDecision::PoorExposure);
        assert_eq!(selection.keyframes.first(), Some(&1));
    }

    #[test]
    fn a_segment_without_usable_frames_keeps_its_first_frame_as_anchor() {
        let mut evidence = frames(2);
        for frame in &mut evidence {
            frame.mean_luma = 10.0;
        }
        let selection = select(&[pair(0, 0.8, 0.0, true)], &evidence, WIDTH, HEIGHT);

        assert_eq!(selection.keyframes, vec![0]);
        assert_eq!(selection.stats.frames[0].decision, FrameDecision::Keyframe);
    }

    #[test]
    fn selection_is_deterministic() {
        let pairs = [pair(0, 0.8, 0.6, false), pair(1, 0.5, 0.9, false), pair(2, 0.8, 0.3, true)];
        let first = select(&pairs, &frames(4), WIDTH, HEIGHT);
        let second = select(&pairs, &frames(4), WIDTH, HEIGHT);

        assert_eq!(first.keyframes, second.keyframes);
        assert_eq!(first.stats, second.stats);
    }
}
