//! Diagnosis of the gate that decided a clip's bootstrap.

use super::solve::MIN_TRACK_MATCHES;
use super::types::{
    BootstrapDiagnosis, BootstrapEscalation, BootstrapGate, SeedCandidateStats, SeedGate,
    SeedGateCount,
};
use crate::{two_view, SegmentBreak};

/// What the decisive seed candidate measured against the gate that rejected it.
fn seed_gate_measurement(seed: &SeedCandidateStats, gate: SeedGate) -> String {
    let optional =
        |value: Option<f32>| value.map_or_else(|| "n/a".to_owned(), |v| format!("{v:.2}"));
    match gate {
        SeedGate::FrameQuality => "a frame failed the blur or exposure quality gate".to_owned(),
        SeedGate::SegmentBreak => "the link between the frames splits the clip".to_owned(),
        SeedGate::Parallax => format!(
            "{} matches, median motion {:.2} px and residual parallax {:.2} px (requires at least 6 matches, 1.40 px motion and 0.55 px parallax)",
            seed.matches, seed.median_motion, seed.median_parallax_residual
        ),
        SeedGate::Matches => format!(
            "{} matches (requires {})",
            seed.matches,
            two_view::MIN_SEED_INLIERS
        ),
        SeedGate::EssentialInliers => format!(
            "{} of {} matches are essential-matrix inliers, ratio {} (requires {} inliers and ratio {:.2})",
            seed.inliers.unwrap_or(0),
            seed.matches,
            optional(seed.inlier_ratio),
            two_view::MIN_SEED_INLIERS,
            two_view::MIN_SEED_INLIER_RATIO
        ),
        SeedGate::Baseline => format!(
            "a pure rotation explains the inliers to {} px (requires more than {:.2} px)",
            optional(seed.rotation_only_residual_pixels),
            two_view::MIN_SEED_ROTATION_RESIDUAL_PIXELS
        ),
        SeedGate::Cheirality | SeedGate::Reprojection => format!(
            "{} of {} inliers triangulate in front of both cameras within {:.1} px ({} behind a camera, {} above the reprojection gate, {} without a finite point; requires {})",
            seed.triangulated_points.unwrap_or(0),
            seed.inliers.unwrap_or(0),
            two_view::MAX_SEED_REPROJECTION_ERROR_PIXELS,
            seed.behind_camera.unwrap_or(0),
            seed.high_reprojection.unwrap_or(0),
            seed.failed_triangulations.unwrap_or(0),
            seed.required_points.unwrap_or(0)
        ),
        SeedGate::TriangulationAngle => format!(
            "median triangulation angle {}° (requires {:.2}°; {} inliers without a finite point)",
            optional(seed.median_triangulation_angle_degrees),
            two_view::MIN_SEED_TRIANGULATION_ANGLE_DEGREES,
            seed.failed_triangulations.unwrap_or(0)
        ),
    }
}

/// The seed gate of a link the segmentation will not seed from. Too few matches without an
/// exposure jump is a correspondence limit, not a cut, so it reports the match gate.
pub(super) fn ineligible_link_gate(link_break: Option<SegmentBreak>, matches: usize) -> SeedGate {
    if matches < MIN_TRACK_MATCHES && link_break == Some(SegmentBreak::LostOverlap) {
        SeedGate::Matches
    } else {
        SeedGate::SegmentBreak
    }
}

/// Name the gate that decided the bootstrap. Without a seed pair, that is the gate of the
/// candidate that advanced furthest through the pipeline: every earlier gate passed for
/// it, so the clip's limit is the gate it failed.
pub(super) fn diagnose_bootstrap(
    seeds: &[SeedCandidateStats],
    seeded: bool,
    seed_covers_segment: bool,
    registered_views: usize,
    pnp_ready: usize,
) -> BootstrapDiagnosis {
    let mut rejections: Vec<SeedGateCount> = Vec::new();
    for gate in seeds.iter().filter_map(|seed| seed.rejected_gate) {
        match rejections.iter_mut().find(|known| known.gate == gate) {
            Some(known) => known.candidates += 1,
            None => rejections.push(SeedGateCount {
                gate,
                candidates: 1,
            }),
        }
    }
    rejections.sort_by_key(|known| known.gate);

    if seeded {
        if registered_views > 0 {
            return BootstrapDiagnosis {
                decisive_gate: None,
                decisive_pair: None,
                escalation: None,
                rejections,
                summary: format!(
                    "A calibrated seed pair and {registered_views} further camera(s) were accepted."
                ),
            };
        }
        // A two-frame segment (or clip): the seed pair already covers every frame its
        // solve may register.
        if seed_covers_segment {
            return BootstrapDiagnosis {
                decisive_gate: None,
                decisive_pair: None,
                escalation: None,
                rejections,
                summary:
                    "A calibrated seed pair was accepted and covers every frame of its segment."
                        .into(),
            };
        }
        let summary = if pnp_ready == 0 {
            "A calibrated seed pair was accepted, but no other selected keyframe tracked enough seed landmarks to attempt PnP, so no registration was tried.".to_string()
        } else {
            format!(
                "A calibrated seed pair was accepted, but registration failed: {pnp_ready} other selected keyframe(s) tracked enough seed landmarks for PnP and none passed the inlier and reprojection gates."
            )
        };
        return BootstrapDiagnosis {
            decisive_gate: Some(BootstrapGate::Registration),
            decisive_pair: None,
            escalation: Some(BootstrapEscalation::LearnedMatching),
            rejections,
            summary,
        };
    }

    // Furthest stage first; then more inliers and triangulated points; then the earliest pair.
    let decisive = seeds
        .iter()
        .filter(|seed| seed.rejected_gate.is_some())
        .reduce(|best, seed| {
            if seed.rank() > best.rank() {
                seed
            } else {
                best
            }
        });
    let Some(decisive) = decisive else {
        return BootstrapDiagnosis {
            decisive_gate: None,
            decisive_pair: None,
            escalation: None,
            rejections,
            summary: "No adjacent frame pair was available as a seed-pair candidate.".to_owned(),
        };
    };
    let gate = decisive
        .rejected_gate
        .expect("decisive candidate was rejected");
    let escalation = match gate {
        SeedGate::Matches | SeedGate::EssentialInliers => {
            Some(BootstrapEscalation::LearnedMatching)
        }
        SeedGate::Cheirality | SeedGate::Reprojection => {
            Some(BootstrapEscalation::LearnedCalibration)
        }
        SeedGate::Parallax | SeedGate::Baseline | SeedGate::TriangulationAngle => {
            Some(BootstrapEscalation::LearnedMultiView)
        }
        // Frame selection already reports unusable frames and clip breaks.
        SeedGate::FrameQuality | SeedGate::SegmentBreak => None,
    };
    BootstrapDiagnosis {
        decisive_gate: Some(BootstrapGate::Seed(gate)),
        summary: format!(
            "No calibrated seed pair was accepted. The decisive gate is {}: the furthest candidate, frames {}–{}, passed every earlier gate, but {}.",
            gate.label(),
            decisive.from_frame + 1,
            decisive.to_frame + 1,
            seed_gate_measurement(decisive, gate)
        ),
        decisive_pair: Some(decisive.clone()),
        escalation,
        rejections,
    }
}
