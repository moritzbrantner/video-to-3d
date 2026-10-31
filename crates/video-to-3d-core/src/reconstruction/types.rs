//! The reconstruction request, result and diagnostics contract, plus the feature
//! records shared across the pipeline.

use crate::{
    two_view, DenseGridSite, DenseStats, EvidencePointAttributes, MeshStats, MeshTriangle,
    MultiViewStats, RevisitStats,
};
use serde::{Deserialize, Serialize};

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

#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct CameraPose {
    pub frame_index: usize,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub matched_features: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub confidence: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, serde::Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, serde::Deserialize)]
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

    pub(super) fn from_two_view(rejection: two_view::TwoViewRejection) -> Self {
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
    pub(super) fn stage(self) -> u8 {
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

/// Machine-readable evidence of one frame pair as a calibrated seed-pair candidate: every
/// adjacent pair, plus wider pairs of a segment none of whose adjacent pairs seeded.
/// Calibrated measurements are `None` when an earlier gate stopped the attempt.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
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
    /// pair has no measurable translation. Adjacent pairs fit the essential inliers; a
    /// non-adjacent pair reports the least-median fit over all its sub-pixel matches,
    /// except when the essential-inlier fit rejected it, which it then reports.
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
    pub(super) fn rank(&self) -> (u8, usize, usize) {
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

impl<'de> Deserialize<'de> for BootstrapGate {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::IntoDeserializer;
        let name = String::deserialize(deserializer)?;
        if name == "registration" {
            return Ok(Self::Registration);
        }
        SeedGate::deserialize(name.as_str().into_deserializer()).map(Self::Seed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct SeedGateCount {
    pub gate: SeedGate,
    pub candidates: usize,
}

/// Which kind of evidence could lift a bootstrap failure. Reported only; never invoked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
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
    /// Per-point reciprocal-consistency and depth-margin evidence, parallel to `dense_points`.
    pub dense_point_attributes: Vec<EvidencePointAttributes>,
    pub dense: DenseStats,
    pub mesh_triangles: Vec<MeshTriangle>,
    /// Sparse pre-canonicalization corners of merged triangles, sorted by triangle: the
    /// reference support overlap canonicalization re-indexed away (see
    /// [`crate::TriangleSourceCorners`]). Empty unless surface consolidation merged patches.
    pub mesh_triangle_sources: Vec<crate::TriangleSourceCorners>,
    /// Lineage of the points surface consolidation derived where it clipped overlap
    /// seams: they follow the observed `dense_points` (and so have no grid site or
    /// point attributes), one record per derived point (see
    /// [`crate::DerivedVertexSource`]).
    pub mesh_derived_vertices: Vec<crate::DerivedVertexSource>,
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
}

/// Per-frame track evidence of one camera solve, for recovery ranking, which compares
/// candidates over one frame range. Not part of [`ReconstructionResult`]; the camera
/// stage artifact keeps it with the rest of the sparse evidence.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct TrackScope {
    /// First and last frame of every feature track.
    pub(super) spans: Vec<(usize, usize)>,
    /// Whether adjacent pair `i` (frames `i`, `i + 1`) has matches.
    pub(super) linked: Vec<bool>,
    /// The primary segment's frames; the whole clip when it is not split.
    pub(super) primary: (usize, usize),
}

impl TrackScope {
    /// `[tracks_three_plus, longest_track, linked_pairs]` within frames `first..=last`.
    /// Over the whole clip these equal the clip-wide `MultiViewStats` values.
    pub(super) fn stats(&self, (first, last): (usize, usize)) -> [usize; 3] {
        let (three_plus, longest) =
            self.spans
                .iter()
                .fold((0, 0), |(three_plus, longest), &(start, end)| {
                    let observed = (end.min(last) + 1).saturating_sub(start.max(first));
                    (
                        three_plus + usize::from(observed >= 3),
                        longest.max(observed),
                    )
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
pub(crate) struct Feature {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) score: f32,
    pub(crate) descriptor: Vec<i16>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FeatureMatch {
    pub(crate) a: usize,
    pub(crate) b: usize,
    pub(crate) distance: f32,
}
