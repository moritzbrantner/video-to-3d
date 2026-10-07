mod dense;
mod keyframe_selection;
mod mesh;
mod multi_view;
mod pnp;
mod revisit;
mod two_view;

pub use dense::{DenseGridSite, DenseStats, DenseWorkingSetEstimate};
pub use keyframe_selection::{
    ClipSegmentStats, FrameDecision, FrameSelectionStats, KeyframeSelectionStats, SegmentBreak,
};
pub use mesh::{MeshStats, MeshTriangle};
pub use multi_view::{
    BundleAdjustmentStats, MultiViewStats, NewLandmarkStats, RegistrationCandidateStats,
};
pub use revisit::{RevisitCandidateStats, RevisitClosureStats, RevisitRecoveryStats, RevisitStats};

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;

const PAN_RECOVERY_WIDTH_FRACTION: f32 = 0.30;
const PAN_RECOVERY_MAX_RADIUS: u32 = 112;
const PAN_RECOVERY_RATIO_THRESHOLD: f32 = 0.78;
const REGISTRATION_RECOVERY_RATIO_THRESHOLD: f32 = 0.74;
const REGISTRATION_RECOVERY_MAX_FEATURES: usize = 640;
const REGISTRATION_RECOVERY_MIN_FEATURE_DISTANCE: u32 = 5;
const MIN_TRACK_MATCHES: usize = 8;
const MIN_TRACK_OVERLAP: f32 = 0.18;
const MOTION_GUIDED_COARSE_RATIO_THRESHOLD: f32 = 0.72;
const MOTION_GUIDED_MIN_SUPPORT: usize = 4;
const DEFAULT_DENSE_WORKING_SET_BUDGET_BYTES: usize = 256 * 1024 * 1024;

const fn default_dense_working_set_budget_bytes() -> usize {
    DEFAULT_DENSE_WORKING_SET_BUDGET_BYTES
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FrameInput {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct ReconstructionOptions {
    pub max_features: usize,
    pub min_feature_distance: u32,
    pub descriptor_radius: u32,
    pub match_radius: u32,
    pub max_descriptor_distance: f32,
    pub ratio_threshold: f32,
    pub focal_length_pixels: Option<f32>,
    #[serde(default = "default_dense_working_set_budget_bytes")]
    pub max_dense_working_set_bytes: usize,
}

impl Default for ReconstructionOptions {
    fn default() -> Self {
        Self {
            max_features: 320,
            min_feature_distance: 7,
            descriptor_radius: 3,
            match_radius: 42,
            max_descriptor_distance: 36.0,
            ratio_threshold: 0.82,
            focal_length_pixels: None,
            max_dense_working_set_bytes: DEFAULT_DENSE_WORKING_SET_BUDGET_BYTES,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReconstructionRequest {
    pub frames: Vec<FrameInput>,
    #[serde(default)]
    pub options: ReconstructionOptions,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CameraPose {
    pub frame_index: usize,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub matched_features: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub confidence: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PairStats {
    pub from_frame: usize,
    pub to_frame: usize,
    pub features_from: usize,
    pub features_to: usize,
    pub matches: usize,
    pub overlap_ratio: f32,
    pub median_dx: f32,
    pub median_dy: f32,
    pub median_motion: f32,
    pub median_parallax_residual: f32,
    pub low_parallax: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CalibratedPairStats {
    pub from_frame: usize,
    pub to_frame: usize,
    pub matches: usize,
    pub inliers: usize,
    pub inlier_ratio: f32,
    pub focal_pixels: f32,
    pub median_sampson_error_pixels: f32,
    pub median_reprojection_error_pixels: f32,
    pub median_triangulation_angle_degrees: f32,
    pub relative_rotation: [f32; 9],
    pub translation_direction: [f32; 3],
}

#[derive(Clone, Debug, Serialize)]
pub struct RegisteredViewStats {
    pub frame_index: usize,
    pub correspondences: usize,
    pub inliers: usize,
    pub inlier_ratio: f32,
    pub median_reprojection_error_pixels: f32,
    pub rotation: [f32; 9],
    pub translation: [f32; 3],
    pub recovered_from_revisit: bool,
}

/// The independent solve of one clip segment. Each segment with a usable seed pair is
/// seeded, registered and bundle-adjusted on its own; its cameras live in that segment's
/// arbitrary monocular frame and are never related to another segment's frame here.
#[derive(Clone, Debug, Serialize)]
pub struct SegmentSolveStats {
    pub segment: usize,
    pub first_frame: usize,
    pub last_frame: usize,
    /// This solve fills the top-level cameras, points, dense depth and mesh.
    pub primary: bool,
    /// The calibrated seed pair chosen inside this segment; `None` when no pair passed.
    pub seed_pair: Option<CalibratedPairStats>,
    pub registration_candidates: usize,
    pub pnp_ready_candidates: usize,
    /// Registered views beyond the seed pair.
    pub registered_views: usize,
    pub recovered_from_revisit: usize,
    /// Seed points plus accepted new landmarks.
    pub sparse_points: usize,
    pub bundle_adjustment_accepted: bool,
    /// Seed and registered camera centers in this segment's own frame.
    pub cameras: Vec<CameraPose>,
}

/// The gate that rejected a calibrated seed-pair candidate, in pipeline order. The
/// first two are frame-selection gates; the rest are calibrated two-view gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedGate {
    /// A frame of the pair failed the blur or exposure quality gate.
    FrameQuality,
    /// The link between the frames splits the clip (hard cut, lost overlap, motion jump).
    SegmentBreak,
    /// Too few matches, too little image motion, or too little residual parallax after
    /// compensating the dominant translation and rotation.
    Parallax,
    /// Fewer matches than the minimal eight-point essential-matrix sample.
    Matches,
    /// The essential-matrix RANSAC consensus is below the inlier count or ratio gate.
    EssentialInliers,
    /// A pure rotation explains the inliers: no measurable translation baseline.
    Baseline,
    /// Too few inliers triangulate in front of both cameras.
    Cheirality,
    /// Triangulated inliers exceed the reprojection-error gate.
    Reprojection,
    /// The median triangulation angle is too small for trustworthy depth.
    TriangulationAngle,
}

impl SeedGate {
    pub fn label(self) -> &'static str {
        match self {
            Self::FrameQuality => "frame quality",
            Self::SegmentBreak => "segment break",
            Self::Parallax => "residual parallax",
            Self::Matches => "feature matches",
            Self::EssentialInliers => "essential-matrix inliers",
            Self::Baseline => "translation baseline",
            Self::Cheirality => "cheirality",
            Self::Reprojection => "reprojection error",
            Self::TriangulationAngle => "triangulation angle",
        }
    }

    fn from_two_view(rejection: two_view::TwoViewRejection) -> Self {
        use two_view::TwoViewRejection as Rejection;
        match rejection {
            Rejection::Matches => Self::Matches,
            Rejection::Inliers => Self::EssentialInliers,
            Rejection::Baseline => Self::Baseline,
            Rejection::Cheirality => Self::Cheirality,
            Rejection::Reprojection => Self::Reprojection,
            Rejection::TriangulationAngle => Self::TriangulationAngle,
        }
    }

    /// Pipeline depth; cheirality and reprojection are the same triangulation stage.
    fn stage(self) -> u8 {
        match self {
            Self::FrameQuality => 0,
            Self::SegmentBreak => 1,
            Self::Parallax => 2,
            Self::Matches => 3,
            Self::EssentialInliers => 4,
            Self::Baseline => 5,
            Self::Cheirality | Self::Reprojection => 6,
            Self::TriangulationAngle => 7,
        }
    }
}

/// Machine-readable evidence of one adjacent pair as a calibrated seed-pair candidate.
/// Calibrated measurements are `None` when an earlier gate stopped the attempt.
#[derive(Clone, Debug, Serialize)]
pub struct SeedCandidateStats {
    pub from_frame: usize,
    pub to_frame: usize,
    pub matches: usize,
    pub median_motion: f32,
    pub median_parallax_residual: f32,
    /// `None` when every gate passed.
    pub rejected_gate: Option<SeedGate>,
    /// This candidate seeded the primary or a segment solve.
    pub selected: bool,
    pub inliers: Option<usize>,
    pub inlier_ratio: Option<f32>,
    /// Median residual of the best pure-rotation fit; at or below the baseline gate the
    /// pair has no measurable translation.
    pub rotation_only_residual_pixels: Option<f32>,
    /// Inliers that must triangulate in front of both cameras within the reprojection gate.
    pub required_points: Option<usize>,
    pub triangulated_points: Option<usize>,
    pub behind_camera: Option<usize>,
    /// Inliers whose rays gave no finite point; not counted as behind a camera.
    pub failed_triangulations: Option<usize>,
    pub high_reprojection: Option<usize>,
    pub median_reprojection_error_pixels: Option<f32>,
    pub median_triangulation_angle_degrees: Option<f32>,
}

impl SeedCandidateStats {
    fn rank(&self) -> (u8, usize, usize) {
        (
            self.rejected_gate.map_or(u8::MAX, SeedGate::stage),
            self.inliers.unwrap_or(0),
            self.triangulated_points.unwrap_or(0),
        )
    }
}

/// The gate that decided the bootstrap: why no calibrated seed pair, or no camera beyond
/// it, was accepted. Serialized as one flat string: a [`SeedGate`] name or `registration`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootstrapGate {
    Seed(SeedGate),
    /// A seed pair was accepted, but no other selected keyframe passed PnP registration.
    Registration,
}

impl Serialize for BootstrapGate {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Seed(gate) => gate.serialize(serializer),
            Self::Registration => serializer.serialize_str("registration"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SeedGateCount {
    pub gate: SeedGate,
    pub candidates: usize,
}

/// Which kind of evidence could lift a bootstrap failure. Reported only; never invoked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapEscalation {
    /// Correspondences are the limit: learned matching may supply more consistent matches.
    LearnedMatching,
    /// Correspondences agree on an essential matrix but not on a calibrated pose: learned
    /// calibration (intrinsics) may resolve it.
    LearnedCalibration,
    /// Consistent correspondences carry too little translation for classical triangulation
    /// (pan-dominant or distant footage): only learned multi-view depth priors can add
    /// geometry, and their output must still pass the Rust gates.
    LearnedMultiView,
}

#[derive(Clone, Debug, Serialize)]
pub struct BootstrapDiagnosis {
    /// `None` when a seed pair and at least one further camera were accepted.
    pub decisive_gate: Option<BootstrapGate>,
    /// The seed candidate that advanced furthest before its gate, for a seed failure.
    pub decisive_pair: Option<SeedCandidateStats>,
    pub escalation: Option<BootstrapEscalation>,
    /// Rejected seed candidates per gate, in pipeline order.
    pub rejections: Vec<SeedGateCount>,
    pub summary: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReconstructionResult {
    pub cameras: Vec<CameraPose>,
    pub points: Vec<Point3>,
    pub dense_points: Vec<Point3>,
    pub dense_grid_sites: Vec<DenseGridSite>,
    pub dense: DenseStats,
    pub mesh_triangles: Vec<MeshTriangle>,
    pub mesh: MeshStats,
    pub pairs: Vec<PairStats>,
    pub calibrated_pair: Option<CalibratedPairStats>,
    pub multi_view: MultiViewStats,
    pub revisits: RevisitStats,
    pub registered_views: Vec<RegisteredViewStats>,
    /// Every adjacent pair as a calibrated seed-pair candidate, with its rejection gate.
    pub seed_candidates: Vec<SeedCandidateStats>,
    pub bootstrap: BootstrapDiagnosis,
    pub warnings: Vec<String>,
    /// Track evidence for recovery ranking, which compares candidates over one frame range.
    #[serde(skip)]
    pub(crate) track_scope: TrackScope,
}

/// Per-frame track evidence of one reconstruction, kept out of the serialized result.
#[derive(Clone, Debug, Default)]
pub(crate) struct TrackScope {
    /// First and last frame of every feature track.
    spans: Vec<(usize, usize)>,
    /// Whether adjacent pair `i` (frames `i`, `i + 1`) has matches.
    linked: Vec<bool>,
    /// The primary segment's frames; the whole clip when it is not split.
    primary: (usize, usize),
}

impl TrackScope {
    /// `[tracks_three_plus, longest_track, linked_pairs]` within frames `first..=last`.
    /// Over the whole clip these equal the clip-wide `MultiViewStats` values.
    fn stats(&self, (first, last): (usize, usize)) -> [usize; 3] {
        let (three_plus, longest) =
            self.spans
                .iter()
                .fold((0, 0), |(three_plus, longest), &(start, end)| {
                    let observed = (end.min(last) + 1).saturating_sub(start.max(first));
                    (three_plus + usize::from(observed >= 3), longest.max(observed))
                });
        let linked = self
            .linked
            .iter()
            .enumerate()
            .filter(|&(pair, &linked)| linked && pair >= first && pair < last)
            .count();
        [three_plus, longest, linked]
    }
}

#[derive(Clone, Debug)]
struct Feature {
    x: u32,
    y: u32,
    score: f32,
    descriptor: Vec<i16>,
}

#[derive(Clone, Copy, Debug)]
struct FeatureMatch {
    a: usize,
    b: usize,
    distance: f32,
}

pub fn reconstruct(request: &ReconstructionRequest) -> Result<ReconstructionResult, String> {
    let initial = reconstruct_once(request)?;
    if !needs_registration_recovery(request, &initial) {
        return finalize_reconstruction(initial);
    }

    let Some(first_frame) = request.frames.first() else {
        return finalize_reconstruction(initial);
    };
    let retry_radius =
        pan_recovery_radius(first_frame.width, request.options.match_radius, &initial);
    // Every candidate's track evidence is compared over the first solve's primary frames,
    // even when a retry segments the clip differently.
    let scope = initial.track_scope.primary;
    let score = |result: &ReconstructionResult| reconstruction_score(result, scope);
    let mut best = initial;
    let mut selected_recovery = None;

    if retry_radius > request.options.match_radius {
        let mut pan_request = request.clone();
        pan_request.options.match_radius = retry_radius;
        pan_request.options.ratio_threshold = pan_request
            .options
            .ratio_threshold
            .min(PAN_RECOVERY_RATIO_THRESHOLD);
        let pan_retry = reconstruct_once(&pan_request)?;
        if score(&pan_retry) > score(&best) {
            best = pan_retry;
            selected_recovery = Some("bounded displacement-informed match-radius recovery");
        }
    }

    let registration_request = needs_registration_recovery(request, &best).then(|| {
        let mut registration_request = request.clone();
        registration_request.options.match_radius = retry_radius;
        registration_request.options.max_features = registration_request
            .options
            .max_features
            .max(REGISTRATION_RECOVERY_MAX_FEATURES);
        registration_request.options.min_feature_distance = registration_request
            .options
            .min_feature_distance
            .min(REGISTRATION_RECOVERY_MIN_FEATURE_DISTANCE);
        registration_request.options.ratio_threshold = registration_request
            .options
            .ratio_threshold
            .min(REGISTRATION_RECOVERY_RATIO_THRESHOLD);
        registration_request
    });

    // A request already at the recovery settings would rerun the same deterministic solve.
    if let Some(registration_request) = registration_request.filter(|retry| {
        let (a, b) = (&retry.options, &request.options);
        (a.match_radius, a.max_features, a.min_feature_distance)
            != (b.match_radius, b.max_features, b.min_feature_distance)
            || a.ratio_threshold != b.ratio_threshold
    }) {
        let registration_retry = reconstruct_once(&registration_request)?;
        if score(&registration_retry) > score(&best) {
            best = registration_retry;
            selected_recovery = Some(
                "bounded registration recovery with denser features and stricter descriptor ambiguity filtering",
            );
        }
    }

    if let Some(recovery) = selected_recovery {
        best.warnings.push(format!(
            "The Rust core selected {recovery} after the ordinary reconstruction lacked sufficient accepted multi-view evidence. Seed-pair, PnP, bundle-adjustment, dense-depth, and mesh acceptance gates remain unchanged."
        ));
    }

    finalize_reconstruction(best)
}

fn finalize_reconstruction(
    mut reconstruction: ReconstructionResult,
) -> Result<ReconstructionResult, String> {
    crate::surface_fusion::consolidate_surface_evidence(&mut reconstruction)?;
    Ok(reconstruction)
}

fn reconstruct_once(request: &ReconstructionRequest) -> Result<ReconstructionResult, String> {
    validate_request(request)?;

    let width = request.frames[0].width;
    let height = request.frames[0].height;
    let options = request.options;
    let luma_frames: Vec<Vec<u8>> = request.frames.iter().map(to_luma).collect();
    let features: Vec<Vec<Feature>> = luma_frames
        .iter()
        .map(|luma| detect_features(luma, width, height, options))
        .collect();

    let focal = options
        .focal_length_pixels
        .unwrap_or(0.86 * width.max(height) as f32);
    let diagonal = (width as f32).hypot(height as f32);
    let mut cameras = Vec::with_capacity(request.frames.len());
    let mut points = Vec::new();
    let mut pairs = Vec::with_capacity(request.frames.len() - 1);
    let mut adjacent_matches = Vec::with_capacity(request.frames.len() - 1);
    let mut warnings = Vec::new();
    let mut motion_guided_pairs = 0usize;
    // Every calibrated pair that passes the seed gates, in pair order. Each segment seeds
    // from its own best candidate.
    let mut seed_candidates: Vec<(usize, two_view::TwoViewEstimate)> = Vec::new();
    let mut seed_evidence: Vec<SeedCandidateStats> = Vec::with_capacity(request.frames.len() - 1);
    let frame_evidence: Vec<_> = luma_frames
        .iter()
        .enumerate()
        .map(|(index, luma)| crate::input_readiness::frame_evidence(index, luma, width, height))
        .collect();
    let mut segmentation =
        keyframe_selection::SegmentationContext::new(&frame_evidence, width, height);
    for (before, after) in segmentation.rejected_runs() {
        let (source, target) = (&features[before], &features[after]);
        let mut matches = match_features(source, target, options);
        if let Some(guided) = motion_guided_matches(source, target, options) {
            if guided.len() > matches.len() {
                matches = guided;
            }
        }
        let overlap = if source.is_empty() || target.is_empty() {
            0.0
        } else {
            matches.len() as f32 / source.len().min(target.len()) as f32
        };
        let mut motion: Vec<f32> = matches
            .iter()
            .map(|m| {
                let (a, b) = (&source[m.a], &target[m.b]);
                (b.x as f32 - a.x as f32).hypot(b.y as f32 - a.y as f32)
            })
            .collect();
        let median_motion = median(&mut motion);
        segmentation.bridge(before, after, matches.len(), overlap, median_motion);
    }

    let mut camera = CameraPose {
        frame_index: 0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        matched_features: 0,
    };
    cameras.push(camera);

    for pair_index in 0..request.frames.len() - 1 {
        let source_features = &features[pair_index];
        let target_features = &features[pair_index + 1];
        let ordinary_matches = match_features(source_features, target_features, options);
        let ordinary_overlap = if source_features.is_empty() || target_features.is_empty() {
            0.0
        } else {
            ordinary_matches.len() as f32
                / source_features.len().min(target_features.len()) as f32
        };
        let (matches, used_motion_guidance) = if ordinary_matches.len() < MIN_TRACK_MATCHES
            || ordinary_overlap < MIN_TRACK_OVERLAP
        {
            match motion_guided_matches(source_features, target_features, options) {
                Some(guided) if guided.len() > ordinary_matches.len() => (guided, true),
                _ => (ordinary_matches, false),
            }
        } else {
            (ordinary_matches, false)
        };
        if used_motion_guidance {
            motion_guided_pairs += 1;
        }
        let overlap_ratio = if source_features.is_empty() || target_features.is_empty() {
            0.0
        } else {
            matches.len() as f32 / source_features.len().min(target_features.len()) as f32
        };

        let mut dx_values = Vec::with_capacity(matches.len());
        let mut dy_values = Vec::with_capacity(matches.len());
        let mut motion_values = Vec::with_capacity(matches.len());
        for feature_match in &matches {
            let a = &source_features[feature_match.a];
            let b = &target_features[feature_match.b];
            let dx = b.x as f32 - a.x as f32;
            let dy = b.y as f32 - a.y as f32;
            dx_values.push(dx);
            dy_values.push(dy);
            motion_values.push(dx.hypot(dy));
        }

        let median_dx = median(&mut dx_values);
        let median_dy = median(&mut dy_values);
        let median_motion = median(&mut motion_values);
        let (translation_x, translation_y, parallax_residuals) =
            compensate_global_motion(source_features, target_features, &matches, width, height);
        let mut residuals_for_median = parallax_residuals.clone();
        let median_parallax_residual = median(&mut residuals_for_median);
        let low_parallax =
            matches.len() < 6 || median_motion < 1.4 || median_parallax_residual < 0.55;

        // The seed uses two frames that pass the quality gates, within one segment.
        let seed_eligible =
            segmentation.seed_eligible(pair_index, matches.len(), overlap_ratio, median_motion);
        let mut seed = SeedCandidateStats {
            from_frame: pair_index,
            to_frame: pair_index + 1,
            matches: matches.len(),
            median_motion,
            median_parallax_residual,
            rejected_gate: None,
            selected: false,
            inliers: None,
            inlier_ratio: None,
            rotation_only_residual_pixels: None,
            required_points: None,
            triangulated_points: None,
            behind_camera: None,
            failed_triangulations: None,
            high_reprojection: None,
            median_reprojection_error_pixels: None,
            median_triangulation_angle_degrees: None,
        };
        if !segmentation.usable(pair_index) || !segmentation.usable(pair_index + 1) {
            seed.rejected_gate = Some(SeedGate::FrameQuality);
        } else if !seed_eligible {
            seed.rejected_gate = Some(ineligible_link_gate(
                segmentation.link_break(pair_index, matches.len(), overlap_ratio, median_motion),
                matches.len(),
            ));
        } else if low_parallax {
            seed.rejected_gate = Some(SeedGate::Parallax);
        } else {
            let outcome = two_view::evaluate_two_view(
                source_features,
                target_features,
                &matches,
                width,
                height,
                focal as f64,
            );
            record_two_view_evidence(&mut seed, &outcome.evidence);
            match outcome.result {
                Ok(estimate) => seed_candidates.push((pair_index, estimate)),
                Err(rejection) => seed.rejected_gate = Some(SeedGate::from_two_view(rejection)),
            }
        }
        seed_evidence.push(seed);

        let observed_baseline = if low_parallax {
            0.0
        } else {
            (median_parallax_residual / diagonal * 12.0).clamp(0.05, 1.0)
        };

        if low_parallax {
            camera = CameraPose {
                frame_index: pair_index + 1,
                matched_features: matches.len(),
                ..camera
            };
        } else {
            camera = CameraPose {
                frame_index: pair_index + 1,
                x: camera.x - translation_x / width as f32 * 1.5,
                y: camera.y + translation_y / height as f32 * 1.5,
                z: camera.z + observed_baseline,
                matched_features: matches.len(),
            };
        }
        cameras.push(camera);

        if !low_parallax {
            for (feature_match, disparity) in matches.iter().zip(&parallax_residuals) {
                if *disparity < 0.35 {
                    continue;
                }
                let a = &source_features[feature_match.a];
                let depth = (focal * observed_baseline / disparity.max(0.5)).clamp(0.35, 18.0);
                let normalized_x = (a.x as f32 - width as f32 * 0.5) / focal;
                let normalized_y = (a.y as f32 - height as f32 * 0.5) / focal;
                let source_camera = cameras[pair_index];
                let (r, g, b) = sample_rgb(&request.frames[pair_index], a.x, a.y);
                let descriptor_confidence = (1.0
                    - feature_match.distance / options.max_descriptor_distance)
                    .clamp(0.0, 1.0);
                let motion_confidence = (disparity / 5.0).clamp(0.15, 1.0);

                points.push(Point3 {
                    x: source_camera.x + normalized_x * depth,
                    y: source_camera.y - normalized_y * depth,
                    z: source_camera.z + depth,
                    confidence: descriptor_confidence * motion_confidence,
                    r,
                    g,
                    b,
                });
            }
        }

        pairs.push(PairStats {
            from_frame: pair_index,
            to_frame: pair_index + 1,
            features_from: source_features.len(),
            features_to: target_features.len(),
            matches: matches.len(),
            overlap_ratio,
            median_dx,
            median_dy,
            median_motion,
            median_parallax_residual,
            low_parallax,
        });
        adjacent_matches.push(matches);
    }

    let keyframe_selection =
        keyframe_selection::select(&pairs, &frame_evidence, &segmentation);
    let revisit_context =
        revisit::RevisitContext::new(&features, width, height, focal as f64, options);
    let segment_of_pair =
        |pair_index: usize| keyframe_selection.stats.frames[pair_index].segment;
    let primary_index = best_seed_index(&seed_candidates, |_| true);
    let primary_segment = primary_index.map(|index| segment_of_pair(seed_candidates[index].0));
    let clip_segments = &keyframe_selection.stats.segments;
    // A split clip solves every segment that has a usable seed on its own, in its own
    // arbitrary frame. The segment with the strongest seed fills the primary output.
    let mut segment_solves: Vec<SegmentSolveStats> = if clip_segments.len() > 1 {
        clip_segments
            .iter()
            .enumerate()
            .filter(|(segment, _)| Some(*segment) != primary_segment)
            .map(|(segment, span)| {
                let seed = best_seed_index(&seed_candidates, |pair_index| {
                    segment_of_pair(pair_index) == segment
                })
                .map(|index| &seed_candidates[index]);
                let solve = seed.map(|seed| {
                    solve_segment(
                        Some(seed),
                        &keyframe_selection,
                        &adjacent_matches,
                        &features,
                        &revisit_context,
                        width,
                        height,
                        focal as f64,
                        true,
                    )
                });
                segment_solve_stats(segment, span, false, seed, solve.as_ref(), focal)
            })
            .collect()
    } else {
        Vec::new()
    };
    for solve in &segment_solves {
        if let Some(seed) = &solve.seed_pair {
            seed_evidence[seed.from_frame].selected = true;
        }
    }
    if let Some(index) = primary_index {
        seed_evidence[seed_candidates[index].0].selected = true;
    }
    let best_two_view = primary_index.map(|index| seed_candidates.swap_remove(index));
    drop(seed_candidates);
    let primary_solve = solve_segment(
        best_two_view.as_ref(),
        &keyframe_selection,
        &adjacent_matches,
        &features,
        &revisit_context,
        width,
        height,
        focal as f64,
        false,
    );
    if let Some(segment) = primary_segment.filter(|_| clip_segments.len() > 1) {
        segment_solves.push(segment_solve_stats(
            segment,
            &clip_segments[segment],
            true,
            best_two_view.as_ref(),
            Some(&primary_solve),
            focal,
        ));
        segment_solves.sort_by_key(|solve| solve.segment);
    }
    let (primary_first, primary_last) = primary_segment
        .filter(|_| clip_segments.len() > 1)
        .map_or((0, request.frames.len() - 1), |segment| {
            (clip_segments[segment].first_frame, clip_segments[segment].last_frame)
        });
    let track_scope = TrackScope {
        spans: primary_solve.multi_view_analysis.track_spans(),
        linked: adjacent_matches.iter().map(|matches| !matches.is_empty()).collect(),
        primary: (primary_first, primary_last),
    };
    let SegmentSolve {
        mut multi_view_analysis,
        revisits,
        registered_views,
        registered_cameras,
        registered_geometry,
        new_landmarks,
        optimized_seed_points,
        optimized_new_landmark_positions,
    } = primary_solve;
    multi_view_analysis.stats.segment_solves = segment_solves;

    let dense_sparse_points: Vec<nalgebra::Vector3<f64>> = optimized_seed_points
        .iter()
        .chain(&optimized_new_landmark_positions)
        .copied()
        .collect();
    let dense_analysis = dense::estimate_depth_points_with_budget(
        &request.frames,
        &registered_geometry,
        &dense_sparse_points,
        focal as f64,
        options.max_dense_working_set_bytes,
        options.max_features,
        options.descriptor_radius,
    );
    let mesh_analysis = mesh::reconstruct_dense_mesh(
        &dense_analysis.points,
        &dense_analysis.grid_sites,
        &dense_analysis.stats,
        &registered_geometry,
        width,
        height,
        focal as f64,
    );
    let mesh = mesh_analysis.stats;
    let mesh_triangles = mesh_analysis.triangles;
    let dense = dense_analysis.stats;
    let dense_grid_sites = dense_analysis.grid_sites;
    let dense_points = dense_analysis.points;
    let multi_view = multi_view_analysis.stats;

    let calibrated_pair = best_two_view.map(|(pair_index, estimate)| {
        let source_features = &features[pair_index];
        let frame = &request.frames[pair_index];
        cameras = seed_pair_cameras(pair_index, &estimate);
        cameras.extend(registered_cameras.iter().copied());
        cameras.sort_by_key(|camera| camera.frame_index);
        points = estimate
            .points
            .iter()
            .enumerate()
            .map(|(point_index, triangulated)| {
                let feature = &source_features[triangulated.source_feature_index];
                let (r, g, b) = sample_rgb(frame, feature.x, feature.y);
                let descriptor_confidence = (1.0
                    - triangulated.descriptor_distance / options.max_descriptor_distance)
                    .clamp(0.0, 1.0);
                let reprojection_confidence = (1.0
                    / (1.0 + triangulated.reprojection_error_pixels as f32 * 0.5))
                    .clamp(0.15, 1.0);
                let angle_confidence =
                    (triangulated.triangulation_angle_degrees as f32 / 3.0).clamp(0.15, 1.0);
                let position = optimized_seed_points
                    .get(point_index)
                    .copied()
                    .unwrap_or(triangulated.position);
                Point3 {
                    x: position.x as f32,
                    y: position.y as f32,
                    z: position.z as f32,
                    confidence: descriptor_confidence * reprojection_confidence * angle_confidence,
                    r,
                    g,
                    b,
                }
            })
            .collect();
        points.extend(
            new_landmarks
                .iter()
                .enumerate()
                .filter_map(|(new_index, landmark)| {
                    let feature = features
                        .get(landmark.source_frame_index)?
                        .get(landmark.source_feature_index)?;
                    let frame = request.frames.get(landmark.source_frame_index)?;
                    let (r, g, b) = sample_rgb(frame, feature.x, feature.y);
                    let support_confidence =
                        (landmark.supporting_observations as f32 / 4.0).clamp(0.5, 1.0);
                    let reprojection_confidence = (1.0
                        / (1.0 + landmark.median_reprojection_error_pixels as f32 * 0.5))
                        .clamp(0.15, 1.0);
                    let angle_confidence =
                        (landmark.triangulation_angle_degrees as f32 / 3.0).clamp(0.15, 1.0);
                    let position = optimized_new_landmark_positions
                        .get(new_index)
                        .copied()
                        .unwrap_or(landmark.position);
                    Some(Point3 {
                        x: position.x as f32,
                        y: position.y as f32,
                        z: position.z as f32,
                        confidence: support_confidence * reprojection_confidence * angle_confidence,
                        r,
                        g,
                        b,
                    })
                }),
        );

        calibrated_pair_stats(pair_index, &estimate, focal)
    });

    if motion_guided_pairs > 0 {
    warnings.push(format!(
        "Motion-guided adjacent matching recovered {motion_guided_pairs} sampled frame pair(s) after the ordinary origin-centered local search was starved. A strict global descriptor consensus only predicts the dominant displacement used to center the existing bounded local search; all calibrated geometry still passes the normal epipolar, PnP, bundle-adjustment, dense-depth, and mesh gates."
    ));
}

    let segments = &multi_view.keyframe_selection.segments;
    if segments.len() > 1 {
        let spans = segments
            .iter()
            .map(|segment| format!("{}–{}", segment.first_frame + 1, segment.last_frame + 1))
            .collect::<Vec<_>>()
            .join(", ");
        let solved = match multi_view
            .segment_solves
            .iter()
            .find(|solve| solve.primary)
        {
            Some(primary) => {
                let seeded = multi_view
                    .segment_solves
                    .iter()
                    .filter(|solve| solve.seed_pair.is_some())
                    .count();
                let unseeded = segments.len() - seeded;
                let unseeded_note = if unseeded > 0 {
                    format!(" {unseeded} segment(s) had no calibrated seed pair and are reported, not solved.")
                } else {
                    String::new()
                };
                format!(
                    "{seeded} segment(s) with a calibrated seed pair are each solved on their own, in their own arbitrary frame; no segment is forced into another's solve. The displayed geometry, dense depth and mesh come from segment {} (frames {}–{}), which has the strongest seed; per-segment seeds and registration counts are listed in the evidence.{unseeded_note}",
                    primary.segment + 1,
                    primary.first_frame + 1,
                    primary.last_frame + 1,
                )
            }
            None => "No segment yielded a calibrated seed pair, so none is solved; the uncalibrated preview spans all segments.".to_owned(),
        };
        warnings.push(format!(
            "The clip splits into {} segments at hard cuts or discontinuities (frames {spans}). {solved}",
            segments.len()
        ));
    }

    let low_pairs = pairs.iter().filter(|pair| pair.low_parallax).count();
    if low_pairs > pairs.len() / 2 {
        warnings.push(
            "Most frame pairs have weak residual parallax after compensating dominant translation and rotation. Move through a textured scene instead of only panning or rotating the camera."
                .into(),
        );
    }
    if points.len() < 80 {
        warnings.push(
            "The sparse cloud is small. Try a more textured, well-lit scene with slower camera motion."
                .into(),
        );
    }

    if let Some(limit) = dense.reference_view_limit {
        warnings.push(format!(
            "The dense working set for every reference view ({} MiB) exceeds the {} MiB budget, so dense reconstruction kept the first {limit} reference view(s) in its usual order ({} MiB). Each kept view passed the same gates as a full run; the remaining views were not reconstructed.",
            dense.full_working_set_bytes / (1024 * 1024),
            dense.working_set_budget_bytes / (1024 * 1024),
            dense.working_set_estimate.total_bytes / (1024 * 1024),
        ));
    }
    if let Some(dense_warning) = dense_warning(&dense) {
        warnings.push(dense_warning);
    }
    if let Some(mesh_warning) = mesh_warning(&mesh) {
        warnings.push(mesh_warning);
    }

    let recovered_from_revisit = revisits
        .recoveries
        .iter()
        .filter(|recovery| recovery.accepted)
        .count();
    let closed_drift = revisits
        .closures
        .iter()
        .filter(|closure| closure.accepted)
        .count();
    if !revisits.candidates.is_empty() {
        if recovered_from_revisit > 0 || closed_drift > 0 {
            warnings.push(format!(
                "Bounded non-adjacent revisit screening found {} strong keyframe-pair candidates, recovered {recovered_from_revisit} previously unregistered selected keyframe poses, and retained {closed_drift} seed-anchored registered-pose drift corrections. Revisit evidence changes geometry only through strict seed-landmark PnP and the existing rollback-safe bundle-adjustment acceptance boundary; broader arbitrary non-seed pose-graph closure is still out of scope.",
                revisits.candidates.len()
            ));
        } else if !revisits.recoveries.is_empty() || !revisits.closures.is_empty() {
            warnings.push(format!(
                "Bounded non-adjacent revisit screening found {} strong keyframe-pair candidates and produced recovery or registered-pose closure attempts, but no additional revisit-driven geometry survived the deterministic PnP, bounded-drift, and bundle-adjustment no-regression gates. No camera pose was invented or half-integrated from revisit evidence.",
                revisits.candidates.len()
            ));
        } else {
            warnings.push(format!(
                "Bounded non-adjacent revisit screening found {} strong keyframe-pair candidates, but none had enough accepted seed geometry to recover or close a selected camera pose. No revisit correction was applied.",
                revisits.candidates.len()
            ));
        }
    }

    // The seed's segment is the only one its solve may register into.
    let seed_covers_segment = calibrated_pair.as_ref().is_some_and(|pair| {
        multi_view
            .keyframe_selection
            .segments
            .iter()
            .find(|segment| segment.first_frame <= pair.from_frame && pair.to_frame <= segment.last_frame)
            .map_or(seed_evidence.len() <= 1, |segment| segment.last_frame - segment.first_frame <= 1)
    });
    let bootstrap = diagnose_bootstrap(
        &seed_evidence,
        calibrated_pair.is_some(),
        seed_covers_segment,
        registered_views.len(),
        multi_view
            .registration_candidates
            .iter()
            .filter(|candidate| candidate.pnp_ready)
            .count(),
    );
    if calibrated_pair.is_some() {
        let pnp_ready = multi_view
            .registration_candidates
            .iter()
            .filter(|candidate| candidate.pnp_ready)
            .count();
        if registered_views.is_empty() {
            warnings.push(format!(
                "Slice 3 finds {pnp_ready} other selected keyframes with enough tracked seed landmarks for a robust PnP attempt, but neither the initial track-based registration nor bounded direct-revisit recovery produced an additional camera pose that passed the deterministic inlier and reprojection gates. The displayed geometry remains the strongest calibrated adjacent pair; translation scale is arbitrary, and bundle adjustment requires at least one accepted additional view."
            ));
        } else if multi_view.bundle_adjustment.accepted {
            let initial_rmse = multi_view
                .bundle_adjustment
                .initial_rmse_reprojection_error_pixels
                .unwrap_or_default();
            let final_rmse = multi_view
                .bundle_adjustment
                .final_rmse_reprojection_error_pixels
                .unwrap_or_default();
            warnings.push(format!(
                "Slice 3 registered {} additional selected keyframes, including {recovered_from_revisit} recovered from bounded direct revisit evidence, triangulated {} new landmarks, retained {closed_drift} seed-anchored loop corrections, and accepted deterministic bundle adjustment across {} cameras, {} landmarks, and {} supported observations. Reprojection RMSE improved from {:.2} px to {:.2} px while the calibrated seed-pair cameras remained fixed to preserve the arbitrary monocular gauge. No metric-scale claim or arbitrary non-seed pose-graph optimization is made.",
                registered_views.len(),
                multi_view.new_landmarks.accepted_landmarks,
                multi_view.bundle_adjustment.optimized_cameras,
                multi_view.bundle_adjustment.optimized_landmarks,
                multi_view.bundle_adjustment.observations,
                initial_rmse,
                final_rmse
            ));
        } else if multi_view.bundle_adjustment.attempted {
            warnings.push(format!(
                "Slice 3 registered {} additional selected keyframes, including {recovered_from_revisit} recovered from bounded direct revisit evidence, and triangulated {} new landmarks. Bundle adjustment ran across {} supported observations but its candidate geometry did not satisfy the no-regression acceptance boundary, so the pre-adjustment cameras and landmarks were retained and no loop correction was committed. The calibrated seed pair remains the fixed arbitrary monocular gauge.",
                registered_views.len(),
                multi_view.new_landmarks.accepted_landmarks,
                multi_view.bundle_adjustment.observations
            ));
        } else {
            warnings.push(format!(
                "Slice 3 registered {} additional selected keyframes, including {recovered_from_revisit} recovered from bounded direct revisit evidence, and triangulated {} new landmarks, but there were not enough mutually supported registered observations to run bundle adjustment. No loop correction can be retained without that downstream gate; all geometry remains in the seed pair's arbitrary monocular coordinate frame.",
                registered_views.len(),
                multi_view.new_landmarks.accepted_landmarks
            ));
        }
    } else {
        warnings.push(format!(
            "{} This result retains the conservative uncalibrated MVP preview.",
            bootstrap.summary
        ));
    }

    Ok(ReconstructionResult {
        cameras,
        points,
        dense_points,
        dense_grid_sites,
        dense,
        mesh_triangles,
        mesh,
        pairs,
        calibrated_pair,
        multi_view,
        revisits,
        registered_views,
        seed_candidates: seed_evidence,
        bootstrap,
        warnings,
        track_scope,
    })
}

fn record_two_view_evidence(seed: &mut SeedCandidateStats, evidence: &two_view::TwoViewEvidence) {
    if evidence.matches < two_view::MIN_SEED_INLIERS {
        return;
    }
    seed.inliers = Some(evidence.inliers);
    seed.inlier_ratio = Some(evidence.inliers as f32 / evidence.matches as f32);
    seed.rotation_only_residual_pixels = evidence
        .rotation_only_residual_pixels
        .map(|value| value as f32);
    if let Some(pose) = evidence.pose {
        seed.required_points = Some(pose.required_points);
        seed.triangulated_points = Some(pose.accepted_points);
        seed.behind_camera = Some(pose.behind_camera);
        seed.failed_triangulations = Some(pose.failed_triangulation);
        seed.high_reprojection = Some(pose.high_reprojection);
        seed.median_reprojection_error_pixels = pose
            .median_reprojection_error_pixels
            .map(|value| value as f32);
        seed.median_triangulation_angle_degrees = pose
            .median_triangulation_angle_degrees
            .map(|value| value as f32);
    }
}

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
fn ineligible_link_gate(link_break: Option<SegmentBreak>, matches: usize) -> SeedGate {
    if matches < MIN_TRACK_MATCHES && link_break == Some(SegmentBreak::LostOverlap) {
        SeedGate::Matches
    } else {
        SeedGate::SegmentBreak
    }
}

/// Name the gate that decided the bootstrap. Without a seed pair, that is the gate of the
/// candidate that advanced furthest through the pipeline: every earlier gate passed for
/// it, so the clip's limit is the gate it failed.
fn diagnose_bootstrap(
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
                summary: "A calibrated seed pair was accepted and covers every frame of its segment.".into(),
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

/// Seed, register, triangulate and bundle-adjust one clip segment in its own frame.
struct SegmentSolve {
    multi_view_analysis: multi_view::MultiViewAnalysis,
    revisits: RevisitStats,
    registered_views: Vec<RegisteredViewStats>,
    registered_cameras: Vec<CameraPose>,
    registered_geometry: Vec<multi_view::RegisteredCamera>,
    new_landmarks: Vec<multi_view::NewLandmark>,
    optimized_seed_points: Vec<nalgebra::Vector3<f64>>,
    optimized_new_landmark_positions: Vec<nalgebra::Vector3<f64>>,
}

#[allow(clippy::too_many_arguments)]
fn solve_segment(
    seed: Option<&(usize, two_view::TwoViewEstimate)>,
    keyframe_selection: &keyframe_selection::KeyframeSelection,
    adjacent_matches: &[Vec<FeatureMatch>],
    features: &[Vec<Feature>],
    revisit_context: &revisit::RevisitContext<'_>,
    width: u32,
    height: u32,
    focal: f64,
    secondary: bool,
) -> SegmentSolve {
    let seed_landmarks = seed.map(|(pair_index, estimate)| {
        (
            *pair_index,
            estimate
                .points
                .iter()
                .enumerate()
                .map(|(point_index, point)| multi_view::SeedLandmark {
                    point_index,
                    source_feature_index: point.source_feature_index,
                })
                .collect::<Vec<_>>(),
        )
    });
    let mut multi_view_analysis = multi_view::analyze(
        keyframe_selection,
        adjacent_matches,
        seed_landmarks
            .as_ref()
            .map(|(pair_index, landmarks)| (*pair_index, landmarks.as_slice())),
    );
    let seed_segment = seed
        .map(|(pair_index, _)| keyframe_selection.segment_keyframes(*pair_index))
        .unwrap_or_default();
    // The primary solve screens the whole clip for its diagnostics; a secondary solve only
    // screens its own segment, since only that segment can recover or close its cameras.
    let revisit_keyframes = if secondary {
        &seed_segment
    } else {
        &multi_view_analysis.stats.keyframes
    };
    let mut revisits = revisit::analyze(
        revisit_context,
        revisit_keyframes,
        seed.map(|(pair_index, _)| *pair_index),
        &seed_segment,
    );

    let mut registered_views = Vec::new();
    let mut registered_cameras = Vec::new();
    let mut registered_geometry = Vec::new();
    if let Some((seed_pair_index, estimate)) = seed {
        registered_geometry.push(multi_view::RegisteredCamera {
            frame_index: *seed_pair_index,
            rotation: nalgebra::Matrix3::identity(),
            translation: nalgebra::Vector3::zeros(),
        });
        registered_geometry.push(multi_view::RegisteredCamera {
            frame_index: *seed_pair_index + 1,
            rotation: estimate.rotation,
            translation: estimate.translation,
        });

        for candidate in &multi_view_analysis.registration_candidates {
            if candidate.correspondences.len() < 8 {
                continue;
            }
            let pnp_correspondences: Vec<pnp::PnpCorrespondence> = candidate
                .correspondences
                .iter()
                .filter_map(|correspondence| {
                    let point = estimate.points.get(correspondence.seed_point_index)?;
                    let feature = features
                        .get(candidate.frame_index)?
                        .get(correspondence.feature_index)?;
                    Some(pnp::PnpCorrespondence {
                        point: point.position,
                        x_pixels: feature.x as f64,
                        y_pixels: feature.y as f64,
                    })
                })
                .collect();
            if pnp_correspondences.len() != candidate.correspondences.len() {
                continue;
            }
            let Some(pose) = pnp::estimate_pose(&pnp_correspondences, width, height, focal)
            else {
                continue;
            };

            registered_geometry.push(multi_view::RegisteredCamera {
                frame_index: candidate.frame_index,
                rotation: pose.rotation,
                translation: pose.translation,
            });
            registered_cameras.push(CameraPose {
                frame_index: candidate.frame_index,
                x: pose.camera_center.x as f32,
                y: pose.camera_center.y as f32,
                z: pose.camera_center.z as f32,
                matched_features: pose.inliers,
            });
            registered_views.push(RegisteredViewStats {
                frame_index: candidate.frame_index,
                correspondences: pnp_correspondences.len(),
                inliers: pose.inliers,
                inlier_ratio: pose.inliers as f32 / pnp_correspondences.len() as f32,
                median_reprojection_error_pixels: pose.median_reprojection_error_pixels as f32,
                rotation: [
                    pose.rotation[(0, 0)] as f32,
                    pose.rotation[(0, 1)] as f32,
                    pose.rotation[(0, 2)] as f32,
                    pose.rotation[(1, 0)] as f32,
                    pose.rotation[(1, 1)] as f32,
                    pose.rotation[(1, 2)] as f32,
                    pose.rotation[(2, 0)] as f32,
                    pose.rotation[(2, 1)] as f32,
                    pose.rotation[(2, 2)] as f32,
                ],
                translation: [
                    pose.translation.x as f32,
                    pose.translation.y as f32,
                    pose.translation.z as f32,
                ],
                recovered_from_revisit: false,
            });
        }

        let registered_frames: HashSet<usize> = registered_geometry
            .iter()
            .map(|camera| camera.frame_index)
            .collect();
        let candidate_frames: Vec<usize> = multi_view_analysis
            .registration_candidates
            .iter()
            .map(|candidate| candidate.frame_index)
            .collect();
        for recovered in revisit::recover_failed_registrations(
            &mut revisits,
            *seed_pair_index,
            estimate,
            &candidate_frames,
            &registered_frames,
            revisit_context,
        ) {
            registered_geometry.push(multi_view::RegisteredCamera {
                frame_index: recovered.frame_index,
                rotation: recovered.rotation,
                translation: recovered.translation,
            });
            registered_cameras.push(CameraPose {
                frame_index: recovered.frame_index,
                x: recovered.camera_center.x as f32,
                y: recovered.camera_center.y as f32,
                z: recovered.camera_center.z as f32,
                matched_features: recovered.inliers,
            });
            registered_views.push(RegisteredViewStats {
                frame_index: recovered.frame_index,
                correspondences: recovered.correspondences,
                inliers: recovered.inliers,
                inlier_ratio: recovered.inliers as f32 / recovered.correspondences as f32,
                median_reprojection_error_pixels: recovered.median_reprojection_error_pixels as f32,
                rotation: [
                    recovered.rotation[(0, 0)] as f32,
                    recovered.rotation[(0, 1)] as f32,
                    recovered.rotation[(0, 2)] as f32,
                    recovered.rotation[(1, 0)] as f32,
                    recovered.rotation[(1, 1)] as f32,
                    recovered.rotation[(1, 2)] as f32,
                    recovered.rotation[(2, 0)] as f32,
                    recovered.rotation[(2, 1)] as f32,
                    recovered.rotation[(2, 2)] as f32,
                ],
                translation: [
                    recovered.translation.x as f32,
                    recovered.translation.y as f32,
                    recovered.translation.z as f32,
                ],
                recovered_from_revisit: true,
            });
        }
    }
    registered_views.sort_by_key(|view| view.frame_index);
    registered_cameras.sort_by_key(|camera| camera.frame_index);
    registered_geometry.sort_by_key(|camera| camera.frame_index);

    let new_landmark_analysis = multi_view::triangulate_new_landmarks(
        &multi_view_analysis,
        &registered_geometry,
        features,
        width,
        height,
        focal,
    );
    multi_view_analysis.stats.new_landmarks = new_landmark_analysis.stats.clone();
    let mut new_landmarks = new_landmark_analysis.landmarks;

    let mut optimized_seed_points = seed
        .map(|(_, estimate)| {
            estimate
                .points
                .iter()
                .map(|point| point.position)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut optimized_new_landmark_positions: Vec<nalgebra::Vector3<f64>> = new_landmarks
        .iter()
        .map(|landmark| landmark.position)
        .collect();
    if let Some((seed_pair_index, estimate)) = seed {
        let adjustment = multi_view::bundle_adjust(
            &multi_view_analysis,
            &optimized_seed_points,
            &new_landmarks,
            &registered_geometry,
            features,
            width,
            height,
            focal,
        );
        multi_view_analysis.stats.bundle_adjustment = adjustment.stats.clone();
        optimized_seed_points = adjustment.seed_points;
        optimized_new_landmark_positions = adjustment.new_landmark_positions;
        registered_geometry = adjustment.cameras;
        for (landmark, position) in new_landmarks
            .iter_mut()
            .zip(&optimized_new_landmark_positions)
        {
            landmark.position = *position;
        }

        let pre_loop_geometry = registered_geometry.clone();
        let pre_loop_seed_points = optimized_seed_points.clone();
        let pre_loop_new_landmark_positions = optimized_new_landmark_positions.clone();
        let pre_loop_adjustment = multi_view_analysis.stats.bundle_adjustment.clone();
        let closures = revisit::close_registered_drift(
            &mut revisits,
            *seed_pair_index,
            estimate,
            &registered_geometry,
            revisit_context,
        );

        if !closures.is_empty() {
            for closure in &closures {
                if let Some(camera) = registered_geometry
                    .iter_mut()
                    .find(|camera| camera.frame_index == closure.frame_index)
                {
                    camera.rotation = closure.rotation;
                    camera.translation = closure.translation;
                }
            }

            let loop_adjustment = multi_view::bundle_adjust(
                &multi_view_analysis,
                &optimized_seed_points,
                &new_landmarks,
                &registered_geometry,
                features,
                width,
                height,
                focal,
            );
            let retained = loop_adjustment.stats.accepted
                && revisit::closure_corrections_retained(
                    &closures,
                    &pre_loop_geometry,
                    &loop_adjustment.cameras,
                );

            if retained {
                multi_view_analysis.stats.bundle_adjustment = loop_adjustment.stats;
                optimized_seed_points = loop_adjustment.seed_points;
                optimized_new_landmark_positions = loop_adjustment.new_landmark_positions;
                registered_geometry = loop_adjustment.cameras;
                for (landmark, position) in new_landmarks
                    .iter_mut()
                    .zip(&optimized_new_landmark_positions)
                {
                    landmark.position = *position;
                }
            } else {
                let closure_frames: HashSet<usize> =
                    closures.iter().map(|closure| closure.frame_index).collect();
                for closure in &mut revisits.closures {
                    if closure_frames.contains(&closure.frame_index) {
                        closure.accepted = false;
                    }
                }
                registered_geometry = pre_loop_geometry;
                optimized_seed_points = pre_loop_seed_points;
                optimized_new_landmark_positions = pre_loop_new_landmark_positions;
                multi_view_analysis.stats.bundle_adjustment = pre_loop_adjustment;
                for (landmark, position) in new_landmarks
                    .iter_mut()
                    .zip(&optimized_new_landmark_positions)
                {
                    landmark.position = *position;
                }
            }
        }

        registered_cameras = registered_geometry
            .iter()
            .filter(|camera| {
                camera.frame_index != *seed_pair_index && camera.frame_index != *seed_pair_index + 1
            })
            .filter_map(|camera| {
                let view = registered_views
                    .iter()
                    .find(|view| view.frame_index == camera.frame_index)?;
                let center = camera.camera_center();
                Some(CameraPose {
                    frame_index: camera.frame_index,
                    x: center.x as f32,
                    y: center.y as f32,
                    z: center.z as f32,
                    matched_features: view.inliers,
                })
            })
            .collect();
    }


    SegmentSolve {
        multi_view_analysis,
        revisits,
        registered_views,
        registered_cameras,
        registered_geometry,
        new_landmarks,
        optimized_seed_points,
        optimized_new_landmark_positions,
    }
}

/// The best seed candidate among the pairs `include` accepts, using the same ordering
/// and tie-break (earliest pair wins) as a single sequential scan.
fn best_seed_index(
    candidates: &[(usize, two_view::TwoViewEstimate)],
    include: impl Fn(usize) -> bool,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (index, (pair_index, estimate)) in candidates.iter().enumerate() {
        if include(*pair_index)
            && best.is_none_or(|current| estimate.is_better_than(&candidates[current].1))
        {
            best = Some(index);
        }
    }
    best
}

fn seed_pair_cameras(pair_index: usize, estimate: &two_view::TwoViewEstimate) -> Vec<CameraPose> {
    vec![
        CameraPose {
            frame_index: pair_index,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            matched_features: 0,
        },
        CameraPose {
            frame_index: pair_index + 1,
            x: estimate.camera_center.x as f32,
            y: estimate.camera_center.y as f32,
            z: estimate.camera_center.z as f32,
            matched_features: estimate.inliers,
        },
    ]
}

fn calibrated_pair_stats(
    pair_index: usize,
    estimate: &two_view::TwoViewEstimate,
    focal: f32,
) -> CalibratedPairStats {
    CalibratedPairStats {
        from_frame: pair_index,
        to_frame: pair_index + 1,
        matches: estimate.matches,
        inliers: estimate.inliers,
        inlier_ratio: estimate.inliers as f32 / estimate.matches as f32,
        focal_pixels: focal,
        median_sampson_error_pixels: estimate.median_sampson_error_pixels as f32,
        median_reprojection_error_pixels: estimate.median_reprojection_error_pixels as f32,
        median_triangulation_angle_degrees: estimate.median_triangulation_angle_degrees as f32,
        relative_rotation: [
            estimate.rotation[(0, 0)] as f32,
            estimate.rotation[(0, 1)] as f32,
            estimate.rotation[(0, 2)] as f32,
            estimate.rotation[(1, 0)] as f32,
            estimate.rotation[(1, 1)] as f32,
            estimate.rotation[(1, 2)] as f32,
            estimate.rotation[(2, 0)] as f32,
            estimate.rotation[(2, 1)] as f32,
            estimate.rotation[(2, 2)] as f32,
        ],
        translation_direction: [
            estimate.translation.x as f32,
            estimate.translation.y as f32,
            estimate.translation.z as f32,
        ],
    }
}

fn segment_solve_stats(
    segment: usize,
    span: &ClipSegmentStats,
    primary: bool,
    seed: Option<&(usize, two_view::TwoViewEstimate)>,
    solve: Option<&SegmentSolve>,
    focal: f32,
) -> SegmentSolveStats {
    let stats = solve.map(|solve| &solve.multi_view_analysis.stats);
    let cameras = match (seed, solve) {
        (Some((pair_index, estimate)), Some(solve)) => {
            let mut cameras = seed_pair_cameras(*pair_index, estimate);
            cameras.extend(solve.registered_cameras.iter().copied());
            cameras.sort_by_key(|camera| camera.frame_index);
            cameras
        }
        _ => Vec::new(),
    };
    SegmentSolveStats {
        segment,
        first_frame: span.first_frame,
        last_frame: span.last_frame,
        primary,
        seed_pair: seed.map(|(pair_index, estimate)| calibrated_pair_stats(*pair_index, estimate, focal)),
        registration_candidates: stats.map_or(0, |stats| stats.registration_candidates.len()),
        pnp_ready_candidates: stats.map_or(0, |stats| {
            stats
                .registration_candidates
                .iter()
                .filter(|candidate| candidate.pnp_ready)
                .count()
        }),
        registered_views: solve.map_or(0, |solve| solve.registered_views.len()),
        recovered_from_revisit: solve.map_or(0, |solve| {
            solve
                .registered_views
                .iter()
                .filter(|view| view.recovered_from_revisit)
                .count()
        }),
        sparse_points: seed.map_or(0, |(_, estimate)| estimate.points.len())
            + solve.map_or(0, |solve| solve.new_landmarks.len()),
        bundle_adjustment_accepted: stats.is_some_and(|stats| stats.bundle_adjustment.accepted),
        cameras,
    }
}

fn mesh_warning(mesh: &MeshStats) -> Option<String> {
    if !mesh.attempted {
        return None;
    }

    if mesh.accepted_triangles > 0 {
        return Some(format!(
            "Slice 4 bounded mesh reconstruction accepted {} triangles from {} candidate triangles across {} reference-grid cells. It rejected {} triangles at depth/spatial discontinuities and {} degenerate or orientation-flipped triangles. This is a bounded multi-reference, non-watertight surface preview. Browser texture projection is appearance-only over accepted per-reference topology; unsupported holes and metric scale are not claimed.",
            mesh.accepted_triangles,
            mesh.candidate_triangles,
            mesh.candidate_cells,
            mesh.rejected_discontinuities,
            mesh.rejected_degenerate
        ));
    }

    Some(format!(
        "Slice 4 bounded mesh reconstruction evaluated {} candidate triangles across {} reference-grid cells, but no triangle survived the continuity and non-degeneracy gates. No surface geometry was invented.",
        mesh.candidate_triangles, mesh.candidate_cells
    ))
}

fn dense_warning(dense: &DenseStats) -> Option<String> {
    if !dense.attempted {
        return None;
    }

    let reference_frame = dense.reference_frame.map_or(0, |frame| frame + 1);
    if dense.accepted_points > 0 {
        return Some(format!(
            "Slice 4 dense point fusion accepted {} fused scene points from reference frame {} using {} registered source views and {} inverse-depth hypotheses. {} primary candidates passed reciprocal depth consistency; spatial fusion rejected {} reverse observations. Fused points remain separate from the sparse map; dense coverage is bounded to accepted reference patches and does not claim unbounded aggregation or metric scale.",
            dense.accepted_points,
            reference_frame,
            dense.source_views,
            dense.depth_hypotheses,
            dense.reciprocal_consistent_points,
            dense.fusion_rejected_observations
        ));
    }

    if dense.reciprocal_consistent_points > 0 {
        return Some(format!(
            "Slice 4 dense point fusion ran from reference frame {} with {} registered source views. {} primary candidates passed reciprocal depth consistency, but no fused point survived the spatial-consistency and minimum-observation gates; {} reverse observations were rejected as spatially inconsistent. No dense geometry was invented.",
            reference_frame,
            dense.source_views,
            dense.reciprocal_consistent_points,
            dense.fusion_rejected_observations
        ));
    }

    if dense.reciprocal_checked_points > 0 {
        return Some(format!(
            "Slice 4 coarse depth estimation ran from reference frame {} with {} registered source views. {} primary candidates passed the texture, photometric-error, and ambiguity gates, but reciprocal depth consistency rejected all of them. No dense geometry was invented.",
            reference_frame,
            dense.source_views,
            dense.reciprocal_checked_points
        ));
    }

    Some(format!(
        "Slice 4 coarse depth estimation ran from reference frame {} with {} registered source views, but no sampled pixel passed the texture, photometric-error, and ambiguity gates. No dense geometry was invented.",
        reference_frame, dense.source_views
    ))
}

fn needs_registration_recovery(
    request: &ReconstructionRequest,
    result: &ReconstructionResult,
) -> bool {
    if request.frames.len() < 3 {
        return false;
    }
    let starved_adjacent_pair = result
        .pairs
        .iter()
        .any(|pair| pair.matches < MIN_TRACK_MATCHES || pair.overlap_ratio < MIN_TRACK_OVERLAP);
    let missing_seed = result.calibrated_pair.is_none();
    let missing_selected_view = result.calibrated_pair.is_some()
        && result.multi_view.registration_candidates.len() > result.registered_views.len();
    let tracks_end_early = request.frames.len() >= 4 && result.multi_view.longest_track < 4;
    // A split clip solves its other segments on their own, so a retry can also help them.
    // A one-frame segment has no pair to seed. Views recovered from revisit evidence need
    // not be PnP-ready candidates, so they cannot cover a failed one.
    let secondary_segment_short = secondary_segment_solves(result).any(|solve| {
        (solve.seed_pair.is_none() && solve.last_frame > solve.first_frame)
            || solve.pnp_ready_candidates
                > solve.registered_views.saturating_sub(solve.recovered_from_revisit)
    });
    missing_seed
        || starved_adjacent_pair
        || missing_selected_view
        || tracks_end_early
        || secondary_segment_short
}

fn secondary_segment_solves(
    result: &ReconstructionResult,
) -> impl Iterator<Item = &SegmentSolveStats> {
    result
        .multi_view
        .segment_solves
        .iter()
        .filter(|solve| !solve.primary)
}

fn pan_recovery_radius(
    width: u32,
    current_radius: u32,
    result: &ReconstructionResult,
) -> u32 {
    if current_radius >= PAN_RECOVERY_MAX_RADIUS {
        return current_radius;
    }
    let width_scaled = (width as f32 * PAN_RECOVERY_WIDTH_FRACTION).round() as u32;
    let observed_motion = result
        .pairs
        .iter()
        .map(|pair| pair.median_motion)
        .filter(|motion| motion.is_finite() && *motion > 0.0)
        .fold(0.0f32, f32::max);
    let motion_scaled = (observed_motion * 2.0 + 16.0).ceil() as u32;
    width_scaled
        .max(motion_scaled)
        .clamp(current_radius, PAN_RECOVERY_MAX_RADIUS)
}

fn accepted_registered_camera_count(result: &ReconstructionResult) -> usize {
    if result.calibrated_pair.is_some() {
        result.cameras.len()
    } else {
        0
    }
}

fn reconstruction_score(result: &ReconstructionResult, scope: (usize, usize)) -> [usize; 10] {
    // Recovery chooses the strongest sparse reconstruction before dense geometry is accepted.
    // Dense output may be suppressed by the working-set budget, so it cannot safely decide which
    // camera/point solution survives. Every field up to the sparse points describes the primary
    // solve, with track statistics counted only within `scope`, the frames every compared
    // candidate shares; the other segments of a split clip only break ties, so a retry never
    // trades primary quality for secondary gains.
    let secondary = |measure: fn(&SegmentSolveStats) -> usize| {
        secondary_segment_solves(result).map(measure).sum::<usize>()
    };
    let [tracks_three_plus, longest_track, linked_pairs] = result.track_scope.stats(scope);
    [
        usize::from(result.calibrated_pair.is_some()),
        accepted_registered_camera_count(result),
        result.registered_views.len(),
        usize::from(result.multi_view.bundle_adjustment.accepted),
        tracks_three_plus,
        longest_track,
        linked_pairs,
        result.points.len(),
        // Cameras rather than seeded-segment counts: a retry may segment the clip differently,
        // and fragmenting one range into more seeded segments must not win by itself.
        secondary(|solve| solve.cameras.len()),
        secondary(|solve| solve.sparse_points),
    ]
}

fn validate_request(request: &ReconstructionRequest) -> Result<(), String> {
    if request.frames.len() < 2 {
        return Err("at least two sampled frames are required".into());
    }
    if request.options.max_descriptor_distance <= 0.0 {
        return Err("max descriptor distance must be positive".into());
    }
    if request.options.max_dense_working_set_bytes == 0 {
        return Err("dense working-set budget must be positive".into());
    }
    if request
        .options
        .focal_length_pixels
        .is_some_and(|focal| !focal.is_finite() || focal <= 0.0)
    {
        return Err("focal length in pixels must be finite and positive".into());
    }

    let width = request.frames[0].width;
    let height = request.frames[0].height;
    if width < 32 || height < 24 {
        return Err("sampled frames are too small for reconstruction".into());
    }

    for frame in &request.frames {
        if frame.width != width || frame.height != height {
            return Err("all sampled frames must share dimensions".into());
        }
        let expected = width as usize * height as usize * 4;
        if frame.rgba.len() != expected {
            return Err(format!(
                "invalid RGBA payload: expected {expected} bytes, got {}",
                frame.rgba.len()
            ));
        }
    }
    Ok(())
}

fn compensate_global_motion(
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

fn to_luma(frame: &FrameInput) -> Vec<u8> {
    frame
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| {
            ((77 * pixel[0] as u16 + 150 * pixel[1] as u16 + 29 * pixel[2] as u16) >> 8) as u8
        })
        .collect()
}

fn detect_features(
    luma: &[u8],
    width: u32,
    height: u32,
    options: ReconstructionOptions,
) -> Vec<Feature> {
    let border = options.descriptor_radius.max(3) + 2;
    let mut candidates = Vec::new();

    for y in border..height.saturating_sub(border) {
        for x in border..width.saturating_sub(border) {
            let mut sxx = 0.0f32;
            let mut syy = 0.0f32;
            let mut sxy = 0.0f32;
            for wy in y - 1..=y + 1 {
                for wx in x - 1..=x + 1 {
                    let gx = sample_luma(luma, width, wx + 1, wy) as f32
                        - sample_luma(luma, width, wx - 1, wy) as f32;
                    let gy = sample_luma(luma, width, wx, wy + 1) as f32
                        - sample_luma(luma, width, wx, wy - 1) as f32;
                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                }
            }
            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            let score = det - 0.04 * trace * trace;
            if score > 1_000_000.0 {
                candidates.push((x, y, score));
            }
        }
    }

    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
    let min_distance_sq = (options.min_feature_distance * options.min_feature_distance) as i64;
    let mut selected: Vec<Feature> = Vec::with_capacity(options.max_features);

    for (x, y, score) in candidates {
        if selected.iter().any(|feature| {
            let dx = feature.x as i64 - x as i64;
            let dy = feature.y as i64 - y as i64;
            dx * dx + dy * dy < min_distance_sq
        }) {
            continue;
        }

        selected.push(Feature {
            x,
            y,
            score,
            descriptor: descriptor(luma, width, x, y, options.descriptor_radius),
        });
        if selected.len() >= options.max_features {
            break;
        }
    }

    selected
}

fn descriptor(luma: &[u8], width: u32, x: u32, y: u32, radius: u32) -> Vec<i16> {
    let side = radius * 2 + 1;
    let mut values = Vec::with_capacity((side * side) as usize);
    let mut sum = 0i32;
    for py in y - radius..=y + radius {
        for px in x - radius..=x + radius {
            let value = sample_luma(luma, width, px, py) as i16;
            values.push(value);
            sum += value as i32;
        }
    }
    let mean = sum / values.len() as i32;
    for value in &mut values {
        *value -= mean as i16;
    }
    values
}

fn match_features(
    a: &[Feature],
    b: &[Feature],
    options: ReconstructionOptions,
) -> Vec<FeatureMatch> {
    match_features_around_offset(a, b, options, 0.0, 0.0)
}

fn motion_guided_matches(
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
                    (a_index, descriptor_distance(&source.descriptor, &target.descriptor))
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
            let ratio_ok =
                second.is_infinite() || best_distance < second * options.ratio_threshold;
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

fn sample_luma(luma: &[u8], width: u32, x: u32, y: u32) -> u8 {
    luma[y as usize * width as usize + x as usize]
}

fn sample_rgb(frame: &FrameInput, x: u32, y: u32) -> (u8, u8, u8) {
    let index = (y as usize * frame.width as usize + x as usize) * 4;
    (
        frame.rgba[index],
        frame.rgba[index + 1],
        frame.rgba[index + 2],
    )
}

fn median(values: &mut [f32]) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_frame(width: u32, height: u32, shift_x: i32) -> FrameInput {
        let mut rgba = vec![18u8; width as usize * height as usize * 4];
        for pixel in rgba.as_chunks_mut::<4>().0 {
            pixel[3] = 255;
        }

        for (index, (base_x, base_y)) in [
            (18, 16),
            (37, 21),
            (58, 15),
            (24, 39),
            (48, 44),
            (70, 36),
            (15, 58),
            (41, 62),
            (66, 57),
        ]
        .into_iter()
        .enumerate()
        {
            let depth_shift = (index as i32 % 3) * shift_x / 3;
            let x = base_x + shift_x + depth_shift;
            if x < 3 || x >= width as i32 - 4 || base_y < 3 || base_y >= height as i32 - 4 {
                continue;
            }
            let intensity = 90 + index as u8 * 16;
            for py in base_y - 2..=base_y + 2 {
                for px in x - 2..=x + 2 {
                    let edge = px == x - 2 || px == x + 2 || py == base_y - 2 || py == base_y + 2;
                    let value = if edge { 245 } else { intensity };
                    let offset = (py as usize * width as usize + px as usize) * 4;
                    rgba[offset] = value;
                    rgba[offset + 1] = value.saturating_sub(index as u8 * 2);
                    rgba[offset + 2] = value / 2;
                }
            }
        }

        FrameInput {
            width,
            height,
            rgba,
        }
    }

    #[test]
    fn detects_repeatable_corners() {
        let frame = synthetic_frame(96, 80, 0);
        let luma = to_luma(&frame);
        let features = detect_features(
            &luma,
            frame.width,
            frame.height,
            ReconstructionOptions::default(),
        );
        assert!(
            features.len() >= 12,
            "found only {} features",
            features.len()
        );
    }

    #[test]
    fn reconstructs_parallax_sequence_into_sparse_output() {
        let request = ReconstructionRequest {
            frames: vec![
                synthetic_frame(96, 80, 0),
                synthetic_frame(96, 80, 3),
                synthetic_frame(96, 80, 6),
            ],
            options: ReconstructionOptions {
                match_radius: 12,
                max_descriptor_distance: 55.0,
                ..ReconstructionOptions::default()
            },
        };

        let result = reconstruct(&request).expect("reconstruction should succeed");
        assert!(result.cameras.len() >= 2);
        assert_eq!(result.pairs.len(), 2);
        assert!(
            result.pairs[0].matches >= 6,
            "too few matches: {}",
            result.pairs[0].matches
        );
        assert!(result.pairs[0].overlap_ratio > 0.0);
        assert!(result.pairs[0].median_dx > 1.0);
        assert!(result.pairs[0].median_parallax_residual > 0.5);
        assert!(!result.pairs[0].low_parallax);
        assert!(result.multi_view.track_count > 0);
        assert!(!result.points.is_empty());
    }

    #[test]
    fn stationary_sequence_does_not_invent_camera_motion() {
        let frame = synthetic_frame(96, 80, 0);
        let request = ReconstructionRequest {
            frames: vec![frame.clone(), frame],
            options: ReconstructionOptions {
                max_descriptor_distance: 55.0,
                ..ReconstructionOptions::default()
            },
        };

        let result = reconstruct(&request).expect("stationary reconstruction should succeed");
        assert!(result.pairs[0].low_parallax);
        assert!(result.calibrated_pair.is_none());
        assert_eq!(result.multi_view.keyframes, vec![0]);
        assert!(result.multi_view.registration_candidates.is_empty());
        assert!(result.revisits.candidates.is_empty());
        assert!(result.revisits.recoveries.is_empty());
        assert!(result.revisits.closures.is_empty());
        assert!(result.registered_views.is_empty());
        assert!(!result.multi_view.bundle_adjustment.attempted);
        assert!(!result.dense.attempted);
        assert!(result.dense_points.is_empty());
        assert!(result.cameras[1].x.abs() < f32::EPSILON);
        assert!(result.cameras[1].y.abs() < f32::EPSILON);
        assert!(result.cameras[1].z.abs() < f32::EPSILON);
        assert!(result.points.is_empty());
    }

    #[test]
    fn motion_guided_matching_recovers_displacement_beyond_local_radius() {
        let source: Vec<Feature> = (0u32..8)
            .map(|index| Feature {
                x: 10 + index * 8,
                y: 16 + (index % 3) * 11,
                score: 10.0 - index as f32 * 0.1,
                descriptor: vec![index as i16 * 32, index as i16 * 32 + 3],
            })
            .collect();
        let target: Vec<Feature> = source
            .iter()
            .map(|feature| Feature {
                x: feature.x + 48,
                y: feature.y + 2,
                score: feature.score,
                descriptor: feature.descriptor.clone(),
            })
            .collect();
        let options = ReconstructionOptions {
            match_radius: 12,
            max_descriptor_distance: 6.0,
            ratio_threshold: 0.8,
            ..ReconstructionOptions::default()
        };
        assert!(match_features(&source, &target, options).is_empty());
        let guided = motion_guided_matches(&source, &target, options)
            .expect("coarse displacement should center the bounded search");
        assert_eq!(guided.len(), source.len());
        let mut dx: Vec<f32> = guided
            .iter()
            .map(|feature_match| {
                target[feature_match.b].x as f32 - source[feature_match.a].x as f32
            })
            .collect();
        let mut dy: Vec<f32> = guided
            .iter()
            .map(|feature_match| {
                target[feature_match.b].y as f32 - source[feature_match.a].y as f32
            })
            .collect();
        assert_eq!(median(&mut dx), 48.0);
        assert_eq!(median(&mut dy), 2.0);
    }

    #[test]
    fn recovery_radius_remains_bounded_and_tracks_observed_motion() {
        let request = ReconstructionRequest {
            frames: vec![synthetic_frame(96, 80, 0), synthetic_frame(96, 80, 0)],
            options: ReconstructionOptions::default(),
        };
        let mut result = reconstruct_once(&request).expect("fixture should reconstruct");
        assert_eq!(pan_recovery_radius(360, 42, &result), 108);
        assert_eq!(pan_recovery_radius(180, 42, &result), 54);
        assert_eq!(pan_recovery_radius(96, 42, &result), 42);
        assert_eq!(pan_recovery_radius(360, 128, &result), 128);
        result.pairs[0].median_motion = 44.0;
        assert_eq!(pan_recovery_radius(240, 42, &result), 104);
    }

    #[test]
    fn recovery_score_is_independent_of_dense_budget_output() {
        let request = ReconstructionRequest {
            frames: vec![synthetic_frame(96, 80, 0), synthetic_frame(96, 80, 0)],
            options: ReconstructionOptions::default(),
        };
        let without_dense = reconstruct_once(&request).expect("fixture should reconstruct");
        let mut with_dense = without_dense.clone();
        with_dense.dense_points.push(Point3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            confidence: 0.9,
            r: 10,
            g: 20,
            b: 30,
        });
        with_dense.mesh_triangles.push(MeshTriangle {
            a: 0,
            b: 0,
            c: 0,
            confidence: 0.8,
        });

        assert_eq!(
            reconstruction_score(&without_dense, without_dense.track_scope.primary),
            reconstruction_score(&with_dense, with_dense.track_scope.primary)
        );
    }

    #[test]
    fn dominant_rotation_is_not_counted_as_parallax() {
        let center_x = 48.0f32;
        let center_y = 40.0f32;
        let angle = 0.05f32;
        let translation_x = 2.0f32;
        let translation_y = -1.0f32;
        let source_positions = [
            (20u32, 18u32),
            (48, 15),
            (75, 20),
            (24, 40),
            (70, 42),
            (18, 64),
            (48, 66),
            (76, 62),
        ];
        let source: Vec<Feature> = source_positions
            .iter()
            .map(|&(x, y)| Feature {
                x,
                y,
                score: 1.0,
                descriptor: vec![0],
            })
            .collect();
        let target: Vec<Feature> = source_positions
            .iter()
            .map(|&(x, y)| {
                let centered_x = x as f32 - center_x;
                let centered_y = y as f32 - center_y;
                Feature {
                    x: (x as f32 + translation_x - angle * centered_y).round() as u32,
                    y: (y as f32 + translation_y + angle * centered_x).round() as u32,
                    score: 1.0,
                    descriptor: vec![0],
                }
            })
            .collect();
        let matches: Vec<FeatureMatch> = (0..source.len())
            .map(|index| FeatureMatch {
                a: index,
                b: index,
                distance: 0.0,
            })
            .collect();

        let (_, _, mut residuals) = compensate_global_motion(&source, &target, &matches, 96, 80);
        let residual = median(&mut residuals);
        assert!(residual < 0.55, "rotation residual was {residual}");
    }

    #[test]
    fn dense_warning_reports_successful_fusion_without_stale_future_claims() {
        let dense = DenseStats {
            attempted: true,
            reference_frame: Some(1),
            source_views: 3,
            depth_hypotheses: 24,
            accepted_points: 17,
            reciprocal_consistent_points: 19,
            fusion_rejected_observations: 4,
            ..DenseStats::default()
        };

        let warning = dense_warning(&dense).expect("dense warning");
        assert!(warning.contains("17 fused scene points"));
        assert!(warning.contains("19 primary candidates passed reciprocal depth consistency"));
        assert!(warning.contains("spatial fusion rejected 4 reverse observations"));
        assert!(!warning.contains("fusion remain future"));
        assert!(!warning.contains("multi-view depth consistency"));
        assert!(!warning.contains("meshing"));
    }

    #[test]
    fn dense_warning_distinguishes_fusion_rejection_from_earlier_gates() {
        let dense = DenseStats {
            attempted: true,
            reference_frame: Some(0),
            source_views: 2,
            reciprocal_checked_points: 9,
            reciprocal_rejected_points: 3,
            reciprocal_consistent_points: 6,
            fusion_input_observations: 14,
            fusion_rejected_observations: 8,
            fusion_rejected_points: 6,
            ..DenseStats::default()
        };

        let warning = dense_warning(&dense).expect("dense warning");
        assert!(warning.contains("6 primary candidates passed reciprocal depth consistency"));
        assert!(warning.contains("no fused point survived"));
        assert!(warning.contains("8 reverse observations were rejected"));
        assert!(!warning.contains("no sampled pixel passed"));
    }

    #[test]
    fn rejects_invalid_focal_length() {
        let request = ReconstructionRequest {
            frames: vec![synthetic_frame(96, 80, 0), synthetic_frame(96, 80, 3)],
            options: ReconstructionOptions {
                focal_length_pixels: Some(0.0),
                ..ReconstructionOptions::default()
            },
        };
        assert!(reconstruct(&request).is_err());
    }

    #[test]
    fn rejects_malformed_rgba_payloads() {
        let request = ReconstructionRequest {
            frames: vec![
                FrameInput {
                    width: 96,
                    height: 80,
                    rgba: vec![0; 4],
                },
                synthetic_frame(96, 80, 0),
            ],
            options: ReconstructionOptions::default(),
        };

        assert!(reconstruct(&request).is_err());
    }

    const SCENE_WIDTH: u32 = 640;
    const SCENE_HEIGHT: u32 = 480;
    const SCENE_FOCAL: f64 = 520.0;

    /// Textured blobs scattered through a box in front of the camera, from a fixed seed.
    fn scene_points(seed: u64) -> Vec<(nalgebra::Vector3<f64>, u8)> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1_u64 << 53) as f64)
        };
        (0..420)
            .map(|index| {
                let point = nalgebra::Vector3::new(
                    (next() - 0.5) * 5.2,
                    (next() - 0.5) * 3.5,
                    4.0 + next() * 6.0,
                );
                (point, 35 + ((index * 47) % 190) as u8)
            })
            .collect()
    }

    /// One frame of a lateral camera move through `points`, on a flat `background`.
    fn render_scene(
        points: &[(nalgebra::Vector3<f64>, u8)],
        camera_x: f64,
        background: u8,
    ) -> FrameInput {
        render_scene_posed(points, camera_x, 0.0, background)
    }

    /// One frame from a camera at `camera_x` turned by `yaw` radians about the vertical axis.
    fn render_scene_posed(
        points: &[(nalgebra::Vector3<f64>, u8)],
        camera_x: f64,
        yaw: f64,
        background: u8,
    ) -> FrameInput {
        let (width, height) = (SCENE_WIDTH, SCENE_HEIGHT);
        let mut rgba = vec![background; (width * height * 4) as usize];
        for pixel in rgba.as_chunks_mut::<4>().0 {
            pixel[3] = 255;
        }
        let (sin, cos) = yaw.sin_cos();
        for (index, (point, shade)) in points.iter().enumerate() {
            let (x, z) = (point.x - camera_x, point.z);
            let (x, z) = (cos * x - sin * z, sin * x + cos * z);
            if z <= 0.5 {
                continue;
            }
            let px = (x / z * SCENE_FOCAL + width as f64 * 0.5).round() as i32;
            let py = (point.y / z * SCENE_FOCAL + height as f64 * 0.5).round() as i32;
            let radius = 3 + (index % 2) as i32;
            if px < radius + 1
                || py < radius + 1
                || px >= width as i32 - radius - 1
                || py >= height as i32 - radius - 1
            {
                continue;
            }
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let offset = (((py + dy) as u32 * width + (px + dx) as u32) * 4) as usize;
                    let value = if ((dx + dy + index as i32) & 1) == 0 {
                        *shade
                    } else {
                        255_u8.saturating_sub(shade / 2)
                    };
                    rgba[offset] = value;
                    rgba[offset + 1] = value.saturating_add((index % 17) as u8);
                    rgba[offset + 2] = value.saturating_sub((index % 13) as u8);
                }
            }
        }
        FrameInput {
            width,
            height,
            rgba,
        }
    }

    fn lateral_span(points: &[(nalgebra::Vector3<f64>, u8)], background: u8) -> Vec<FrameInput> {
        (0..6)
            .map(|frame| render_scene(points, -0.42 + frame as f64 * 0.12, background))
            .collect()
    }

    fn scene_request(frames: Vec<FrameInput>) -> ReconstructionRequest {
        ReconstructionRequest {
            frames,
            options: ReconstructionOptions {
                min_feature_distance: 6,
                match_radius: 80,
                focal_length_pixels: Some(SCENE_FOCAL as f32),
                ..ReconstructionOptions::default()
            },
        }
    }

    #[test]
    fn hard_cut_clip_solves_each_segment_from_its_own_seed() {
        let mut frames = lateral_span(&scene_points(0x5eed), 236);
        frames.extend(lateral_span(&scene_points(0xc0ffee), 64));

        let result = reconstruct(&scene_request(frames)).expect("reconstruction should succeed");
        let segments = &result.multi_view.keyframe_selection.segments;
        assert_eq!(segments.len(), 2, "segments: {segments:?}");
        assert_eq!(segments[0].ends_with, Some(SegmentBreak::HardCut));

        let solves = &result.multi_view.segment_solves;
        assert_eq!(solves.len(), 2);
        assert_eq!(solves.iter().filter(|solve| solve.primary).count(), 1);
        for (segment, solve) in solves.iter().enumerate() {
            assert_eq!(solve.segment, segment);
            let span = &segments[segment];
            let seed = solve
                .seed_pair
                .as_ref()
                .unwrap_or_else(|| panic!("segment {segment} has no seed: {solve:?}"));
            assert!(seed.from_frame >= span.first_frame && seed.to_frame <= span.last_frame);
            assert!(
                solve.registered_views > 0,
                "segment {segment} registered no views: {solve:?}"
            );
            assert_eq!(solve.cameras.len(), 2 + solve.registered_views);
            // Neither camera set reaches into the other segment's frames.
            assert!(solve.cameras.iter().all(|camera| {
                camera.frame_index >= span.first_frame && camera.frame_index <= span.last_frame
            }));
        }

        // The primary solve is the top-level result.
        let primary = solves.iter().find(|solve| solve.primary).unwrap();
        let calibrated = result.calibrated_pair.as_ref().unwrap();
        assert_eq!(primary.seed_pair.as_ref().unwrap().from_frame, calibrated.from_frame);
        assert_eq!(primary.registered_views, result.registered_views.len());
        assert_eq!(
            primary.cameras.iter().map(|camera| camera.frame_index).collect::<Vec<_>>(),
            result.cameras.iter().map(|camera| camera.frame_index).collect::<Vec<_>>()
        );
    }

    #[test]
    fn single_segment_clip_reports_no_segment_solves() {
        let frames = lateral_span(&scene_points(0x5eed), 236);
        let result = reconstruct(&scene_request(frames)).expect("reconstruction should succeed");

        assert_eq!(result.multi_view.keyframe_selection.segments.len(), 1);
        assert!(result.calibrated_pair.is_some());
        assert!(result.multi_view.segment_solves.is_empty());
        let serialized = serde_json::to_string(&result.multi_view).unwrap();
        assert!(!serialized.contains("segment_solves"));
    }


    #[test]
    fn accepted_lateral_clip_records_its_seed_candidates() {
        let frames = lateral_span(&scene_points(0x5eed), 236);
        let result = reconstruct(&scene_request(frames)).expect("reconstruction should succeed");

        let pair = result.calibrated_pair.as_ref().expect("lateral clip seeds");
        assert!(!result.registered_views.is_empty());
        assert_eq!(result.seed_candidates.len(), 5);
        let seed = &result.seed_candidates[pair.from_frame];
        assert!(seed.selected && seed.rejected_gate.is_none());
        assert_eq!(seed.inliers, Some(pair.inliers));
        assert!(seed.rotation_only_residual_pixels.unwrap() > 1.25);
        assert_eq!(result.bootstrap.decisive_gate, None);
        assert_eq!(result.bootstrap.escalation, None);
        assert_eq!(
            result
                .seed_candidates
                .iter()
                .filter(|seed| seed.selected)
                .count(),
            1
        );
    }

    /// The Trevi canary's measured limitation in isolation: a camera that only turns.
    /// Every adjacent pair has abundant consistent matches, but a pure rotation explains
    /// them, so no seed pair and therefore no camera may be accepted.
    #[test]
    fn pure_pan_clip_names_the_baseline_gate_instead_of_inventing_cameras() {
        let points = scene_points(0x7e71);
        let frames = (0..6)
            .map(|frame| render_scene_posed(&points, 0.0, frame as f64 * 0.02, 236))
            .collect();
        let result = reconstruct(&scene_request(frames)).expect("reconstruction should succeed");

        assert!(result.calibrated_pair.is_none());
        assert!(result.registered_views.is_empty());
        assert!(!result.dense.attempted);
        assert_eq!(result.seed_candidates.len(), 5);
        for seed in &result.seed_candidates {
            assert!(!seed.selected);
            assert!(seed.rejected_gate.is_some(), "accepted pan pair: {seed:?}");
        }
        assert_eq!(
            result.bootstrap.decisive_gate,
            Some(BootstrapGate::Seed(SeedGate::Baseline)),
            "bootstrap: {:?}",
            result.bootstrap
        );
        assert_eq!(
            result.bootstrap.escalation,
            Some(BootstrapEscalation::LearnedMultiView)
        );
        let decisive = result.bootstrap.decisive_pair.as_ref().unwrap();
        assert!(decisive.inliers.unwrap() >= 8);
        assert!(decisive.rotation_only_residual_pixels.unwrap() <= 1.25);
        assert!(result
            .warnings
            .iter()
            .any(|warning| warning.contains("decisive gate is translation baseline")));

        let serialized = serde_json::to_value(&result.bootstrap).unwrap();
        assert_eq!(serialized["decisive_gate"], "baseline");
        assert_eq!(serialized["escalation"], "learned_multi_view");
    }

    fn seed_stats(
        from_frame: usize,
        gate: Option<SeedGate>,
        inliers: Option<usize>,
    ) -> SeedCandidateStats {
        SeedCandidateStats {
            from_frame,
            to_frame: from_frame + 1,
            matches: 40,
            median_motion: 6.0,
            median_parallax_residual: 1.0,
            rejected_gate: gate,
            selected: false,
            inliers,
            inlier_ratio: inliers.map(|inliers| inliers as f32 / 40.0),
            rotation_only_residual_pixels: None,
            required_points: None,
            triangulated_points: None,
            behind_camera: None,
            failed_triangulations: None,
            high_reprojection: None,
            median_reprojection_error_pixels: None,
            median_triangulation_angle_degrees: None,
        }
    }

    #[test]
    fn bootstrap_names_the_gate_of_the_furthest_candidate() {
        let seeds = vec![
            seed_stats(0, Some(SeedGate::Parallax), None),
            seed_stats(1, Some(SeedGate::Baseline), Some(30)),
            seed_stats(2, Some(SeedGate::TriangulationAngle), Some(20)),
            seed_stats(3, Some(SeedGate::Baseline), Some(36)),
            seed_stats(4, Some(SeedGate::TriangulationAngle), Some(25)),
        ];
        let diagnosis = diagnose_bootstrap(&seeds, false, false, 0, 0);
        // The furthest stage wins over more inliers at an earlier gate; inliers break ties.
        assert_eq!(
            diagnosis.decisive_gate,
            Some(BootstrapGate::Seed(SeedGate::TriangulationAngle))
        );
        assert_eq!(diagnosis.decisive_pair.as_ref().unwrap().from_frame, 4);
        assert_eq!(
            diagnosis.escalation,
            Some(BootstrapEscalation::LearnedMultiView)
        );
        assert_eq!(
            diagnosis.rejections,
            vec![
                SeedGateCount {
                    gate: SeedGate::Parallax,
                    candidates: 1
                },
                SeedGateCount {
                    gate: SeedGate::Baseline,
                    candidates: 2
                },
                SeedGateCount {
                    gate: SeedGate::TriangulationAngle,
                    candidates: 2
                },
            ]
        );
        assert!(
            diagnosis.summary.contains("frames 5–6"),
            "{}",
            diagnosis.summary
        );

        let correspondence_limited = diagnose_bootstrap(
            &[seed_stats(0, Some(SeedGate::EssentialInliers), Some(9))],
            false,
            false,
            0,
            0,
        );
        assert_eq!(
            correspondence_limited.escalation,
            Some(BootstrapEscalation::LearnedMatching)
        );
        let unusable = diagnose_bootstrap(
            &[seed_stats(0, Some(SeedGate::FrameQuality), None)],
            false,
            false,
            0,
            0,
        );
        assert_eq!(unusable.escalation, None);
    }

    #[test]
    fn too_few_matches_without_a_cut_report_the_match_gate() {
        assert_eq!(
            ineligible_link_gate(Some(SegmentBreak::LostOverlap), MIN_TRACK_MATCHES - 1),
            SeedGate::Matches
        );
        assert_eq!(ineligible_link_gate(Some(SegmentBreak::LostOverlap), 0), SeedGate::Matches);
        // A hard cut, a motion jump, or lost overlap with enough matches is a segment break.
        assert_eq!(ineligible_link_gate(Some(SegmentBreak::HardCut), 0), SeedGate::SegmentBreak);
        assert_eq!(
            ineligible_link_gate(Some(SegmentBreak::MotionJump), 40),
            SeedGate::SegmentBreak
        );
        assert_eq!(
            ineligible_link_gate(Some(SegmentBreak::LostOverlap), MIN_TRACK_MATCHES),
            SeedGate::SegmentBreak
        );
    }

    #[test]
    fn bootstrap_reports_registration_after_an_accepted_seed() {
        let seeds = vec![
            seed_stats(0, None, Some(40)),
            seed_stats(1, Some(SeedGate::Baseline), Some(30)),
        ];
        let stalled = diagnose_bootstrap(&seeds, true, false, 0, 3);
        assert_eq!(stalled.decisive_gate, Some(BootstrapGate::Registration));
        assert!(stalled.decisive_pair.is_none());
        assert!(stalled.summary.contains("3 other selected keyframe(s)"));
        assert_eq!(
            serde_json::to_value(stalled.decisive_gate).unwrap(),
            "registration"
        );

        let registered = diagnose_bootstrap(&seeds, true, false, 2, 3);
        assert_eq!(registered.decisive_gate, None);
        assert_eq!(registered.escalation, None);

        // No keyframe reached PnP: the limit is correspondence, not a rejected pose.
        let untried = diagnose_bootstrap(&seeds, true, false, 0, 0);
        assert_eq!(untried.decisive_gate, Some(BootstrapGate::Registration));
        assert!(untried.summary.contains("no registration was tried"), "{}", untried.summary);
        assert!(!untried.summary.contains("inlier and reprojection gates"));

        // A two-frame clip is fully covered by its seed pair.
        let two_frames = diagnose_bootstrap(&seeds[..1], true, true, 0, 0);
        assert_eq!(two_frames.decisive_gate, None);
        assert_eq!(two_frames.escalation, None);
    }

    /// A split clip whose second segment pans too far for the first pass to seed it.
    fn wide_pan_split_request() -> ReconstructionRequest {
        let mut frames = lateral_span(&scene_points(0x5eed), 236);
        let wide = scene_points(0xc0ffee);
        frames.extend((0..6).map(|frame| render_scene(&wide, -0.42 + frame as f64 * 1.2, 64)));
        scene_request(frames)
    }

    #[test]
    fn recovery_keeps_a_secondary_segment_only_the_retry_solves() {
        let request = wide_pan_split_request();
        let initial = reconstruct_once(&request).expect("first pass should succeed");
        let initial_secondary: Vec<_> = secondary_segment_solves(&initial).collect();
        assert_eq!(initial_secondary.len(), 1);
        assert!(initial_secondary[0].seed_pair.is_none());

        let result = reconstruct(&request).expect("reconstruction should succeed");
        let secondary: Vec<_> = secondary_segment_solves(&result).collect();
        assert_eq!(secondary.len(), 1);
        assert!(secondary[0].seed_pair.is_some(), "{:?}", secondary[0]);
        assert!(secondary[0].registered_views > 0);
        assert!(reconstruction_score(&result, initial.track_scope.primary) > reconstruction_score(&initial, initial.track_scope.primary));
    }

    #[test]
    fn track_scope_counts_only_frames_inside_the_scope() {
        let single = reconstruct_once(&scene_request(lateral_span(&scene_points(0x5eed), 236)))
            .expect("single segment should reconstruct");
        let stats = &single.multi_view;
        assert_eq!(single.track_scope.primary, (0, 5));
        assert_eq!(
            single.track_scope.stats(single.track_scope.primary),
            [stats.tracks_three_plus, stats.longest_track, stats.linked_pairs]
        );

        let split = reconstruct_once(&wide_pan_split_request()).expect("split should reconstruct");
        let stats = &split.multi_view;
        let primary = split.track_scope.primary;
        assert!(stats.keyframe_selection.segments.len() > 1);
        assert!(primary.1 - primary.0 < 11, "the primary scope is one segment");
        let [three_plus, longest, linked] = split.track_scope.stats(primary);
        assert!(linked < stats.linked_pairs, "the other segment's pairs are excluded");
        assert!(three_plus <= stats.tracks_three_plus && longest <= stats.longest_track);
        // Over the whole clip, the scope reproduces the clip-wide statistics.
        assert_eq!(
            split.track_scope.stats((0, 11)),
            [stats.tracks_three_plus, stats.longest_track, stats.linked_pairs]
        );

        let scope = TrackScope {
            spans: vec![(0, 4), (3, 5), (6, 6)],
            linked: vec![true, true, false, true, true],
            primary: (0, 5),
        };
        // Spans observe 5, 3 and 1 frames; within 2..=4 they observe 3, 2 and 0.
        assert_eq!(scope.stats((0, 5)), [2, 5, 4]);
        assert_eq!(scope.stats((2, 4)), [1, 3, 1]);
    }

    #[test]
    fn a_short_secondary_segment_alone_triggers_recovery() {
        let request = scene_request(lateral_span(&scene_points(0x5eed), 236));
        let mut result = reconstruct_once(&request).expect("fixture should reconstruct");
        assert!(!needs_registration_recovery(&request, &result));

        let solve = SegmentSolveStats {
            segment: 1,
            first_frame: 6,
            last_frame: 11,
            primary: false,
            seed_pair: result.calibrated_pair.clone(),
            registration_candidates: 3,
            pnp_ready_candidates: 3,
            registered_views: 3,
            recovered_from_revisit: 0,
            sparse_points: 100,
            bundle_adjustment_accepted: true,
            cameras: Vec::new(),
        };
        result.multi_view.segment_solves = vec![solve.clone()];
        assert!(!needs_registration_recovery(&request, &result));

        result.multi_view.segment_solves[0].registered_views = 2;
        assert!(needs_registration_recovery(&request, &result));
        result.multi_view.segment_solves[0] = SegmentSolveStats {
            seed_pair: None,
            ..solve.clone()
        };
        assert!(needs_registration_recovery(&request, &result));

        // A one-frame segment cannot be seeded, so it never asks for a retry.
        result.multi_view.segment_solves[0] = SegmentSolveStats {
            seed_pair: None,
            first_frame: 6,
            last_frame: 6,
            registration_candidates: 0,
            pnp_ready_candidates: 0,
            registered_views: 0,
            ..solve.clone()
        };
        assert!(!needs_registration_recovery(&request, &result));

        // A revisit-recovered view does not cover a PnP-ready candidate that failed.
        result.multi_view.segment_solves[0] = SegmentSolveStats {
            registered_views: 3,
            recovered_from_revisit: 1,
            ..solve
        };
        assert!(needs_registration_recovery(&request, &result));
    }

    #[test]
    fn secondary_segments_never_outrank_the_primary_solve() {
        let request = scene_request(lateral_span(&scene_points(0x5eed), 236));
        let base = reconstruct_once(&request).expect("fixture should reconstruct");
        let solve = |seeded: bool, registered_views: usize| SegmentSolveStats {
            segment: 1,
            first_frame: 6,
            last_frame: 11,
            primary: false,
            seed_pair: base.calibrated_pair.clone().filter(|_| seeded),
            registration_candidates: 4,
            pnp_ready_candidates: 4,
            registered_views,
            recovered_from_revisit: 0,
            sparse_points: registered_views * 50,
            bundle_adjustment_accepted: seeded,
            cameras: if seeded {
                base.cameras.iter().copied().take(2 + registered_views).collect()
            } else {
                Vec::new()
            },
        };
        let mut unseeded = base.clone();
        unseeded.multi_view.segment_solves = vec![solve(false, 0)];
        let mut seeded = base.clone();
        seeded.multi_view.segment_solves = vec![solve(true, 3)];
        assert!(reconstruction_score(&seeded, seeded.track_scope.primary) > reconstruction_score(&unseeded, unseeded.track_scope.primary));

        let mut stronger_primary = unseeded.clone();
        stronger_primary.points.push(base.points[0]);
        assert!(reconstruction_score(&stronger_primary, stronger_primary.track_scope.primary) > reconstruction_score(&seeded, seeded.track_scope.primary));

        // Ranking reads the primary segment's track statistics, not the clip-wide ones.
        let mut clip_wide = seeded.clone();
        clip_wide.multi_view.tracks_three_plus += 100;
        clip_wide.multi_view.longest_track += 10;
        clip_wide.multi_view.linked_pairs += 10;
        assert_eq!(reconstruction_score(&clip_wide, clip_wide.track_scope.primary), reconstruction_score(&seeded, seeded.track_scope.primary));

        // Two seeded fragments with fewer cameras lose to one segment with more.
        let mut fragmented = base.clone();
        fragmented.multi_view.segment_solves = vec![solve(true, 0), solve(true, 0)];
        let mut merged = base.clone();
        merged.multi_view.segment_solves = vec![solve(true, 3)];
        let scope = base.track_scope.primary;
        assert!(reconstruction_score(&merged, scope) > reconstruction_score(&fragmented, scope));
    }
}
