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
/// and this fraction of the median sharpness around it (motion blur, not soft content).
const RELATIVE_BLUR_FRACTION: f32 = 0.8;
/// Frames on each side that form a frame's local sharpness reference.
const BLUR_WINDOW: usize = 3;

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

/// Per-frame quality against clip-level medians, and the link-break rule built on it.
/// Reconstruction consults the same rule while choosing its seed pair, so a seed never
/// spans two segments.
pub(crate) struct SegmentationContext {
    quality: Vec<FrameDecision>,
    luma: Vec<f32>,
    diagonal: f32,
    /// Breaks found across rejected runs, by the usable frame that ends the run.
    bridge_breaks: std::collections::BTreeMap<usize, SegmentBreak>,
}

impl SegmentationContext {
    pub(crate) fn new(frames: &[FrameEvidence], width: u32, height: u32) -> Self {
        let change = |a: f32, b: f32| (a - b).abs() / a.max(b).max(1.0);
        let quality = frames
            .iter()
            .enumerate()
            .map(|(index, frame)| {
                // A flash or dark frame stands out from both neighbours, which agree with
                // each other; a uniformly bright or dark clip is still usable.
                let exposure_outlier = index > 0
                    && index + 1 < frames.len()
                    && change(frame.mean_luma, frames[index - 1].mean_luma)
                        > THRESHOLDS.max_exposure_change
                    && change(frame.mean_luma, frames[index + 1].mean_luma)
                        > THRESHOLDS.max_exposure_change
                    && change(frames[index - 1].mean_luma, frames[index + 1].mean_luma)
                        <= THRESHOLDS.max_exposure_change;
                let window = index.saturating_sub(BLUR_WINDOW)..(index + BLUR_WINDOW + 1).min(frames.len());
                let local_sharpness = median(frames[window].iter().map(|frame| frame.sharpness_ratio));
                if frame.clipped_fraction > THRESHOLDS.max_clipped_fraction
                    || frame.crushed_fraction > THRESHOLDS.max_clipped_fraction
                    || exposure_outlier
                {
                    FrameDecision::PoorExposure
                } else if frame.texture_density >= THRESHOLDS.min_texture_density
                    && frame.sharpness_ratio < THRESHOLDS.min_sharpness_ratio
                    && frame.sharpness_ratio < RELATIVE_BLUR_FRACTION * local_sharpness
                {
                    FrameDecision::Blurred
                } else {
                    FrameDecision::Redundant
                }
            })
            .collect();
        Self {
            quality,
            luma: frames.iter().map(|frame| frame.mean_luma).collect(),
            diagonal: (width as f32).hypot(height as f32),
            bridge_breaks: std::collections::BTreeMap::new(),
        }
    }

    fn usable(&self, frame: usize) -> bool {
        self.quality.get(frame) == Some(&FrameDecision::Redundant)
    }

    /// Whether the link `from → from + 1` may seed the calibrated solve: both frames pass
    /// the quality gates and the link does not split the clip.
    pub(crate) fn seed_eligible(
        &self,
        from: usize,
        matches: usize,
        overlap_ratio: f32,
        median_motion: f32,
    ) -> bool {
        self.usable(from)
            && self.usable(from + 1)
            && self
                .link_break(from, matches, overlap_ratio, median_motion)
                .is_none()
    }

    /// Whether the link `from → from + 1` splits the clip. Only links between two usable
    /// frames can: a blurred or badly exposed frame is rejected, not treated as a cut.
    pub(crate) fn link_break(
        &self,
        from: usize,
        matches: usize,
        overlap_ratio: f32,
        median_motion: f32,
    ) -> Option<SegmentBreak> {
        self.break_between(from, from + 1, matches, overlap_ratio, median_motion)
    }

    fn break_between(
        &self,
        from: usize,
        to: usize,
        matches: usize,
        overlap_ratio: f32,
        median_motion: f32,
    ) -> Option<SegmentBreak> {
        if !self.usable(from) || !self.usable(to) {
            return None;
        }
        if matches < MIN_KEYFRAME_MATCHES || overlap_ratio < MIN_KEYFRAME_OVERLAP {
            let (a, b) = (self.luma[from], self.luma[to]);
            let exposure_change = (a - b).abs() / a.max(b).max(1.0);
            return Some(if exposure_change > THRESHOLDS.max_exposure_change {
                SegmentBreak::HardCut
            } else {
                SegmentBreak::LostOverlap
            });
        }
        (median_motion > MAX_LINK_MOTION_DIAGONAL_FRACTION * self.diagonal)
            .then_some(SegmentBreak::MotionJump)
    }

    /// Usable frame pairs `(before, after)` with only rejected frames between them. Links
    /// touching rejected frames never split the clip, so these spans need their own match.
    pub(crate) fn rejected_runs(&self) -> Vec<(usize, usize)> {
        let usable: Vec<usize> = (0..self.quality.len()).filter(|&frame| self.usable(frame)).collect();
        usable
            .windows(2)
            .filter(|pair| pair[1] > pair[0] + 1)
            .map(|pair| (pair[0], pair[1]))
            .collect()
    }

    /// Records the direct match across a rejected run: a cut hidden by a blurred or badly
    /// exposed transition frame still splits the clip after the run.
    pub(crate) fn bridge(
        &mut self,
        before: usize,
        after: usize,
        matches: usize,
        overlap_ratio: f32,
        median_motion: f32,
    ) {
        if let Some(reason) =
            self.break_between(before, after, matches, overlap_ratio, median_motion)
        {
            self.bridge_breaks.insert(after, reason);
        }
    }}

/// `pairs` are the adjacent links `(i, i + 1)`; `frames` has one entry per frame.
pub(crate) fn select(
    pairs: &[PairStats],
    frames: &[FrameEvidence],
    context: &SegmentationContext,
) -> KeyframeSelection {
    let mut segment_of = vec![0; frames.len()];
    let mut breaks = Vec::new();
    for pair in pairs {
        if pair.to_frame >= frames.len() {
            continue;
        }
        if let Some(reason) =
            context.link_break(pair.from_frame, pair.matches, pair.overlap_ratio, pair.median_motion)
        {
            breaks.push(reason);
        }
        if let Some(&reason) = context.bridge_breaks.get(&pair.to_frame) {
            breaks.push(reason);
        }
        segment_of[pair.to_frame] = breaks.len();
    }

    let mut decisions = context.quality.clone();
    let mut segments = Vec::with_capacity(breaks.len() + 1);
    let mut keyframes = Vec::new();
    for segment in 0..=breaks.len() {
        let members: Vec<usize> = (0..frames.len())
            .filter(|&frame| segment_of[frame] == segment)
            .collect();
        let (Some(&first_frame), Some(&last_frame)) = (members.first(), members.last()) else {
            continue;
        };
        let segment_pairs: Vec<&PairStats> = pairs
            .iter()
            .filter(|pair| {
                segment_of[pair.from_frame] == segment
                    && segment_of.get(pair.to_frame) == Some(&segment)
            })
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
    select(pairs, &frames, &SegmentationContext::new(&frames, 320, 240))
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

    fn frames_with_blur(count: usize, blurred: usize) -> Vec<FrameEvidence> {
        let mut evidence = frames(count);
        evidence[blurred].sharpness_ratio = 0.3;
        evidence
    }

    fn run(pairs: &[PairStats], frames: &[FrameEvidence]) -> KeyframeSelection {
        select(pairs, frames, &SegmentationContext::new(frames, WIDTH, HEIGHT))
    }

    fn decisions(selection: &KeyframeSelection) -> Vec<FrameDecision> {
        selection.stats.frames.iter().map(|frame| frame.decision).collect()
    }

    #[test]
    fn selects_keyframes_from_overlap_and_accumulated_parallax() {
        let pairs = [pair(0, 0.8, 0.4, true), pair(1, 0.8, 0.4, true), pair(2, 0.8, 0.5, false)];

        let selection = run(&pairs, &frames(4));

        assert_eq!(selection.keyframes, vec![0, 3]);
        assert_eq!(selection.stats.segments.len(), 1);
        use FrameDecision::*;
        assert_eq!(decisions(&selection), vec![Keyframe, Redundant, Redundant, Keyframe]);
    }

    #[test]
    fn weak_or_stationary_links_do_not_create_keyframes() {
        let pairs = [pair(0, 0.8, 0.0, true), pair(1, 0.8, 0.0, true)];

        let selection = run(&pairs, &frames(3));

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

        let selection = run(&pairs, &evidence);

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
        let selection = run(&[pair(0, 0.8, 0.2, true), jump], &frames(3));

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

        let selection = run(&pairs, &evidence);

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
        let selection = run(&[pair(0, 0.8, 0.7, false), pair(1, 0.8, 0.7, false)], &evidence);

        assert!(selection.stats.frames.iter().all(|frame| frame.decision != FrameDecision::Blurred));
    }

    #[test]
    fn poorly_exposed_frames_never_become_keyframes() {
        let mut evidence = frames(3);
        evidence[0].clipped_fraction = 0.6;
        let selection = run(&[pair(0, 0.8, 0.7, false), pair(1, 0.8, 0.7, false)], &evidence);

        assert_eq!(selection.stats.frames[0].decision, FrameDecision::PoorExposure);
        assert_eq!(selection.keyframes.first(), Some(&1));
    }

    #[test]
    fn a_segment_without_usable_frames_keeps_its_first_frame_as_anchor() {
        let mut evidence = frames(2);
        for frame in &mut evidence {
            frame.clipped_fraction = 0.6;
        }
        let selection = run(&[pair(0, 0.8, 0.0, true)], &evidence);

        assert_eq!(selection.keyframes, vec![0]);
        assert_eq!(selection.stats.frames[0].decision, FrameDecision::Keyframe);
    }

    #[test]
    fn an_isolated_blurred_frame_with_weak_links_is_rejected_not_a_cut() {
        let mut evidence = frames(3);
        evidence[1].sharpness_ratio = 0.3;
        let mut into = pair(0, 0.05, 0.0, true);
        into.matches = 3;
        let mut out = pair(1, 0.05, 0.0, true);
        out.matches = 3;

        let selection = run(&[into, out], &evidence);

        use FrameDecision::*;
        assert_eq!(selection.stats.segments.len(), 1);
        assert_eq!(decisions(&selection), vec![Keyframe, Blurred, Redundant]);
    }

    #[test]
    fn a_flash_frame_is_poor_exposure_while_a_bright_clip_is_usable() {
        let mut evidence = frames(5);
        for frame in &mut evidence {
            frame.mean_luma = 232.0;
        }
        evidence[2].mean_luma = 120.0;
        let pairs = [
            pair(0, 0.8, 0.5, false),
            pair(1, 0.8, 0.5, false),
            pair(2, 0.8, 0.5, false),
            pair(3, 0.8, 0.5, false),
        ];

        let selection = run(&pairs, &evidence);

        use FrameDecision::*;
        assert_eq!(selection.stats.segments.len(), 1);
        assert_eq!(decisions(&selection), vec![Keyframe, Redundant, PoorExposure, Keyframe, Redundant]);
    }

    #[test]
    fn segment_breaks_are_reported_for_seed_selection_too() {
        let evidence = frames(3);
        let context = SegmentationContext::new(&evidence, WIDTH, HEIGHT);
        let jump = 0.5 * (WIDTH as f32).hypot(HEIGHT as f32);

        // Reconstruction asks the same question before accepting a seed pair.
        assert_eq!(context.link_break(0, 200, 0.7, jump), Some(SegmentBreak::MotionJump));
        assert_eq!(context.link_break(0, 200, 0.7, 4.0), None);
        assert!(!context.seed_eligible(0, 200, 0.7, jump));
        assert!(context.seed_eligible(0, 200, 0.7, 4.0));
    }

    #[test]
    fn rejected_frames_cannot_seed_even_when_their_link_matches() {
        let mut evidence = frames(3);
        evidence[1].sharpness_ratio = 0.3;
        let context = SegmentationContext::new(&evidence, WIDTH, HEIGHT);

        assert_eq!(context.link_break(0, 200, 0.7, 4.0), None);
        assert!(!context.seed_eligible(0, 200, 0.7, 4.0));
        assert!(!context.seed_eligible(1, 200, 0.7, 4.0));
    }

    #[test]
    fn a_cut_behind_a_blurred_transition_frame_still_splits_the_clip() {
        let mut evidence = frames(5);
        evidence[2].sharpness_ratio = 0.3;
        for frame in &mut evidence[3..] {
            frame.mean_luma = 60.0;
        }
        let weak = |from| {
            let mut link = pair(from, 0.05, 0.0, true);
            link.matches = 2;
            link
        };
        let pairs = [pair(0, 0.8, 0.5, false), weak(1), weak(2), pair(3, 0.8, 0.5, false)];
        let mut context = SegmentationContext::new(&evidence, WIDTH, HEIGHT);
        assert_eq!(context.rejected_runs(), [(1, 3)]);

        // Without a direct match across the blurred frame the shots would stay joined.
        assert_eq!(select(&pairs, &evidence, &context).stats.segments.len(), 1);
        context.bridge(1, 3, 1, 0.01, 0.0);
        let selection = select(&pairs, &evidence, &context);

        assert_eq!(selection.stats.segments.len(), 2);
        assert_eq!(selection.stats.segments[0].ends_with, Some(SegmentBreak::HardCut));
        assert_eq!(selection.stats.segments[1].first_frame, 3);
        assert_eq!(selection.stats.frames[2].decision, FrameDecision::Blurred);

        // The same blurred frame inside one shot matches across and stays one segment.
        let mut same_shot = SegmentationContext::new(&frames_with_blur(5, 2), WIDTH, HEIGHT);
        same_shot.bridge(1, 3, 80, 0.6, 3.0);
        assert_eq!(select(&pairs, &frames_with_blur(5, 2), &same_shot).stats.segments.len(), 1);
    }

    #[test]
    fn selection_is_deterministic() {
        let pairs = [pair(0, 0.8, 0.6, false), pair(1, 0.5, 0.9, false), pair(2, 0.8, 0.3, true)];
        let first = run(&pairs, &frames(4));
        let second = run(&pairs, &frames(4));

        assert_eq!(first.keyframes, second.keyframes);
        assert_eq!(first.stats, second.stats);
    }
}
