//! Public staged reconstruction: cameras → dense evidence → consolidated mesh.
//!
//! The classic pipeline runs as three stages, each producing its own versioned,
//! serializable artifact that records the content hash of its inputs:
//!
//! ```text
//! reconstruct_cameras(&request)                    -> CameraStageArtifact
//! reconstruct_dense(&request, &cameras)            -> DenseStageArtifact
//! reconstruct_mesh(&cameras, &dense)               -> MeshStageArtifact
//! compose_reconstruction(&cameras, &dense, &mesh)  -> ReconstructionResult
//! ```
//!
//! [`crate::reconstruct`] (and therefore the browser entry point) composes the same three
//! stage cores in memory, so the single call and the composed stages give the same
//! [`ReconstructionResult`], warnings and their order included. Each artifact owns only its
//! stage's output: the camera artifact the sparse registration result, the dense artifact
//! the dense evidence, the mesh artifact the triangles, their provenance and the few dense
//! vertices surface consolidation moved. No artifact repeats another's buffers; a
//! downstream artifact references its input by content hash instead.

use crate::dense::{DenseGridSite, DenseStats};
use crate::multi_view::RegisteredCamera;
use crate::scene_project::ContentHash;
use crate::{
    BootstrapDiagnosis, CalibratedPairStats, CameraPose, EvidencePointAttributes, FrameInput,
    MeshStats, MeshTriangle, MultiViewStats, PairStats, Point3, ReconstructionOptions,
    ReconstructionRequest, ReconstructionResult, RegisteredViewStats, RevisitStats,
    SeedCandidateStats, TrackScope, TriangleSourceCorners,
};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

pub const CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const DENSE_STAGE_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const MESH_STAGE_ARTIFACT_SCHEMA_VERSION: u32 = 1;

const REQUEST_HASH_DOMAIN: &[u8] = b"video-to-3d/reconstruction-request/v1\0";

/// One stage of the classic pipeline; also names the artifact that stage produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionStage {
    Cameras,
    Dense,
    Mesh,
}

/// The input whose recorded identity a stage checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageInput {
    /// The [`ReconstructionRequest`] (frames and options), via [`request_content_hash`].
    Request,
    /// A [`CameraStageArtifact`], via [`CameraStageArtifact::content_hash`].
    Cameras,
    /// A [`DenseStageArtifact`], via [`DenseStageArtifact::content_hash`].
    Dense,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReconstructionStageError {
    /// An artifact (parsed or in memory) carries a schema version this build does not read.
    UnsupportedSchemaVersion {
        artifact: ReconstructionStage,
        found: u32,
        supported: u32,
    },
    /// An input's identity differs from the one an artifact recorded for it
    /// (different request, different or edited upstream artifact). `stage` names the
    /// artifact holding the record (`recorded`); `input` names what was hashed
    /// (`actual`): the dense stage checks `cameras.request_hash` (Cameras, Request),
    /// the mesh stage checks `dense.cameras_hash` (Dense, Cameras), and compose checks
    /// `dense.cameras_hash` then `mesh.dense_hash` (Mesh, Dense), in that order.
    /// Schema versions are checked before any identity.
    InputMismatch {
        stage: ReconstructionStage,
        input: StageInput,
        recorded: ContentHash,
        actual: ContentHash,
    },
    /// A stored artifact document is not valid JSON for its type, or its payload is
    /// inconsistent with the chain it belongs to.
    Malformed {
        artifact: ReconstructionStage,
        message: String,
    },
    /// The reconstruction itself failed (the `String` error of the single-call pipeline).
    Reconstruction(String),
    /// A stage produced a NaN or infinite value at `field` (a dotted path into the
    /// artifact). JSON cannot store it, so the stage returns no artifact rather than one
    /// whose document [`CameraStageArtifact::from_json`] and its siblings would reject.
    NonFinite {
        artifact: ReconstructionStage,
        field: String,
    },
}

impl fmt::Display for ReconstructionStageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion {
                artifact,
                found,
                supported,
            } => write!(
                formatter,
                "{artifact:?} stage artifact has schema version {found}; this build reads version {supported}"
            ),
            Self::InputMismatch {
                stage,
                input,
                recorded,
                actual,
            } => write!(
                formatter,
                "{stage:?} stage artifact was computed from {input:?} {}, not {}",
                recorded.as_str(),
                actual.as_str()
            ),
            Self::Malformed { artifact, message } => {
                write!(formatter, "malformed {artifact:?} stage artifact: {message}")
            }
            Self::Reconstruction(message) => formatter.write_str(message),
            Self::NonFinite { artifact, field } => write!(
                formatter,
                "{artifact:?} stage produced a non-finite value at `{field}`, which its stored document cannot represent"
            ),
        }
    }
}

impl std::error::Error for ReconstructionStageError {}

/// Sparse registration result: seed pair, registered cameras, sparse points, the
/// effective options of the selected (possibly recovered) solve and every
/// camera-stage diagnostic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraStageArtifact {
    pub schema_version: u32,
    /// [`request_content_hash`] of the request this artifact was computed from.
    pub request_hash: ContentHash,
    pub(crate) sparse: SparseReconstruction,
}

/// Dense evidence of one camera artifact: dense points, grid sites, per-point
/// attributes and dense diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseStageArtifact {
    pub schema_version: u32,
    pub request_hash: ContentHash,
    /// [`CameraStageArtifact::content_hash`] of the camera artifact it was computed from.
    pub cameras_hash: ContentHash,
    pub(crate) dense: DenseEvidence,
}

/// Consolidated surface of one dense artifact: what the mesh stage and surface
/// consolidation own (triangles, triangle sources, consolidated vertex positions,
/// mesh and consolidation diagnostics).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshStageArtifact {
    pub schema_version: u32,
    /// [`DenseStageArtifact::content_hash`] of the dense artifact it was computed from.
    pub dense_hash: ContentHash,
    pub(crate) mesh: SurfaceMesh,
}

/// What the camera stage owns. Besides the sparse fields of [`ReconstructionResult`] it
/// keeps exactly what the later stages read: the effective options and image geometry,
/// the registered camera poses at full precision, the sparse points that bound dense
/// depth, and the camera stage's diagnostics before and after the dense/mesh ones.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SparseReconstruction {
    /// Options of the selected solve: the request's, or a recovery retry's.
    pub(crate) options: ReconstructionOptions,
    pub(crate) image_width: u32,
    pub(crate) image_height: u32,
    pub(crate) focal_length_pixels: f32,
    pub(crate) cameras: Vec<CameraPose>,
    pub(crate) points: Vec<Point3>,
    pub(crate) pairs: Vec<PairStats>,
    pub(crate) calibrated_pair: Option<CalibratedPairStats>,
    /// `bundle_adjustment.final_cameras` lives in `final_cameras` here.
    pub(crate) multi_view: MultiViewStats,
    pub(crate) revisits: RevisitStats,
    pub(crate) registered_views: Vec<RegisteredViewStats>,
    pub(crate) seed_candidates: Vec<SeedCandidateStats>,
    pub(crate) bootstrap: BootstrapDiagnosis,
    /// Bundle-adjusted poses the evidence view reads for registered frames.
    pub(crate) final_cameras: Vec<StageCamera>,
    /// Registered geometry the dense and mesh stages project through.
    pub(crate) dense_cameras: Vec<StageCamera>,
    /// Optimized seed points and new landmarks that bound the dense depth search.
    pub(crate) dense_sparse_points: Vec<[f64; 3]>,
    pub(crate) track_scope: TrackScope,
    pub(crate) warnings_before_dense: Vec<String>,
    /// Includes the recovery selection note, which follows every solve diagnostic.
    pub(crate) warnings_after_mesh: Vec<String>,
}

/// A registered world-to-camera pose at full precision.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StageCamera {
    frame_index: usize,
    /// Column-major, as nalgebra stores it.
    rotation: [f64; 9],
    translation: [f64; 3],
}

impl From<&RegisteredCamera> for StageCamera {
    fn from(camera: &RegisteredCamera) -> Self {
        let mut rotation = [0.0; 9];
        rotation.copy_from_slice(camera.rotation.as_slice());
        Self {
            frame_index: camera.frame_index,
            rotation,
            translation: [
                camera.translation.x,
                camera.translation.y,
                camera.translation.z,
            ],
        }
    }
}

impl StageCamera {
    pub(crate) fn registered(&self) -> RegisteredCamera {
        RegisteredCamera {
            frame_index: self.frame_index,
            rotation: Matrix3::from_column_slice(&self.rotation),
            translation: Vector3::from(self.translation),
        }
    }
}

/// What the dense stage owns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DenseEvidence {
    pub(crate) stats: DenseStats,
    pub(crate) points: Vec<Point3>,
    pub(crate) grid_sites: Vec<DenseGridSite>,
    pub(crate) point_attributes: Vec<EvidencePointAttributes>,
    pub(crate) warnings: Vec<String>,
}

/// What the mesh stage and surface consolidation own.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SurfaceMesh {
    pub(crate) stats: MeshStats,
    pub(crate) triangles: Vec<MeshTriangle>,
    pub(crate) triangle_sources: Vec<TriangleSourceCorners>,
    /// The dense points surface fusion moved, by index into the dense artifact's points;
    /// every other dense point is used as the dense stage left it.
    pub(crate) consolidated_points: Vec<ConsolidatedPoint>,
    /// The mesh diagnostic, refreshed after the connected-surface merge.
    pub(crate) mesh_warning: Option<String>,
    /// Fusion and merge diagnostics, which follow every other warning.
    pub(crate) consolidation_warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConsolidatedPoint {
    pub(crate) index: usize,
    pub(crate) point: Point3,
}

macro_rules! stage_artifact_document {
    ($artifact:ty, $stage:expr, $version:expr) => {
        impl $artifact {
            /// The stored document. Deterministic: equal artifacts give equal bytes.
            pub fn to_json(&self) -> String {
                serde_json::to_string(self).expect("stage artifacts serialize")
            }

            /// Read a stored document, rejecting any other `schema_version` with
            /// [`ReconstructionStageError::UnsupportedSchemaVersion`], and invalid JSON,
            /// a missing `schema_version` or another stage's document with
            /// [`ReconstructionStageError::Malformed`].
            pub fn from_json(document: &str) -> Result<Self, ReconstructionStageError> {
                #[derive(Deserialize)]
                struct Version {
                    schema_version: u32,
                }
                let malformed = |error: serde_json::Error| ReconstructionStageError::Malformed {
                    artifact: $stage,
                    message: error.to_string(),
                };
                let Version { schema_version } =
                    serde_json::from_str(document).map_err(malformed)?;
                check_schema_version($stage, schema_version, $version)?;
                serde_json::from_str(document).map_err(malformed)
            }

            /// `ContentHash::of_bytes(self.to_json().as_bytes())`: the identity a
            /// downstream artifact records and the scene store can verify.
            pub fn content_hash(&self) -> ContentHash {
                ContentHash::of_bytes(self.to_json().as_bytes())
            }

            fn check_schema_version(&self) -> Result<(), ReconstructionStageError> {
                check_schema_version($stage, self.schema_version, $version)
            }

            /// Every float of the artifact is finite, so its document reloads.
            fn check_finite(self) -> Result<Self, ReconstructionStageError> {
                finite::first_non_finite(&self).map_or(Ok(self), |field| {
                    Err(ReconstructionStageError::NonFinite {
                        artifact: $stage,
                        field,
                    })
                })
            }
        }
    };
}

stage_artifact_document!(
    CameraStageArtifact,
    ReconstructionStage::Cameras,
    CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION
);
stage_artifact_document!(
    DenseStageArtifact,
    ReconstructionStage::Dense,
    DENSE_STAGE_ARTIFACT_SCHEMA_VERSION
);
stage_artifact_document!(
    MeshStageArtifact,
    ReconstructionStage::Mesh,
    MESH_STAGE_ARTIFACT_SCHEMA_VERSION
);

fn check_schema_version(
    artifact: ReconstructionStage,
    found: u32,
    supported: u32,
) -> Result<(), ReconstructionStageError> {
    if found == supported {
        Ok(())
    } else {
        Err(ReconstructionStageError::UnsupportedSchemaVersion {
            artifact,
            found,
            supported,
        })
    }
}

fn check_identity(
    stage: ReconstructionStage,
    input: StageInput,
    recorded: &ContentHash,
    actual: ContentHash,
) -> Result<(), ReconstructionStageError> {
    if *recorded == actual {
        Ok(())
    } else {
        Err(ReconstructionStageError::InputMismatch {
            stage,
            input,
            recorded: recorded.clone(),
            actual,
        })
    }
}

/// Identity of a request's frames and options. Deterministic; any change to a
/// frame's dimensions or pixels, the frame order or the options changes it.
pub fn request_content_hash(request: &ReconstructionRequest) -> ContentHash {
    // Destructured so that a new option cannot silently fall out of the identity.
    let ReconstructionOptions {
        max_features,
        min_feature_distance,
        descriptor_radius,
        match_radius,
        max_descriptor_distance,
        ratio_threshold,
        focal_length_pixels,
        max_dense_working_set_bytes,
    } = request.options;
    let mut hasher = Sha256::new();
    hasher.update(REQUEST_HASH_DOMAIN);
    hasher.update((max_features as u64).to_le_bytes());
    hasher.update(min_feature_distance.to_le_bytes());
    hasher.update(descriptor_radius.to_le_bytes());
    hasher.update(match_radius.to_le_bytes());
    hasher.update(max_descriptor_distance.to_bits().to_le_bytes());
    hasher.update(ratio_threshold.to_bits().to_le_bytes());
    match focal_length_pixels {
        Some(focal) => {
            hasher.update([1]);
            hasher.update(focal.to_bits().to_le_bytes());
        }
        None => hasher.update([0]),
    }
    hasher.update((max_dense_working_set_bytes as u64).to_le_bytes());
    hasher.update((request.frames.len() as u64).to_le_bytes());
    for frame in &request.frames {
        hasher.update(frame.width.to_le_bytes());
        hasher.update(frame.height.to_le_bytes());
        hasher.update((frame.rgba.len() as u64).to_le_bytes());
        hasher.update(&frame.rgba);
    }
    ContentHash::of_digest(&hasher.finalize())
}

/// Stage 1: frames → cameras, including the bounded registration recovery of
/// [`crate::reconstruct`] (recovery ranks sparse evidence only).
pub fn reconstruct_cameras(
    request: &ReconstructionRequest,
) -> Result<CameraStageArtifact, ReconstructionStageError> {
    let sparse =
        crate::reconstruct_sparse(request).map_err(ReconstructionStageError::Reconstruction)?;
    CameraStageArtifact {
        schema_version: CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION,
        request_hash: request_content_hash(request),
        sparse,
    }
    .check_finite()
}

/// Stage 2: cameras + frames → dense evidence. Rejects a camera artifact of another
/// schema version or one whose `request_hash` is not this request's. Dense depth uses
/// the camera artifact's effective options, which a recovery retry may have changed.
pub fn reconstruct_dense(
    request: &ReconstructionRequest,
    cameras: &CameraStageArtifact,
) -> Result<DenseStageArtifact, ReconstructionStageError> {
    cameras.check_schema_version()?;
    check_identity(
        ReconstructionStage::Cameras,
        StageInput::Request,
        &cameras.request_hash,
        request_content_hash(request),
    )?;
    DenseStageArtifact {
        schema_version: DENSE_STAGE_ARTIFACT_SCHEMA_VERSION,
        request_hash: cameras.request_hash.clone(),
        cameras_hash: cameras.content_hash(),
        dense: estimate_dense_evidence(&request.frames, &cameras.sparse),
    }
    .check_finite()
}

/// Stage 3: dense evidence (+ the cameras it was computed from) → consolidated mesh.
/// No frames: the mesh stage and surface consolidation read no pixels.
pub fn reconstruct_mesh(
    cameras: &CameraStageArtifact,
    dense: &DenseStageArtifact,
) -> Result<MeshStageArtifact, ReconstructionStageError> {
    cameras.check_schema_version()?;
    dense.check_schema_version()?;
    check_identity(
        ReconstructionStage::Dense,
        StageInput::Cameras,
        &dense.cameras_hash,
        cameras.content_hash(),
    )?;
    // Surface consolidation borrows every other stage field; only the points fusion may
    // move need a working buffer, since the dense artifact stays unchanged. The
    // artifact keeps just the points that moved.
    let original = &dense.dense.points;
    let mut points = original.clone();
    let mut surface = build_surface(&cameras.sparse, DenseFields::of(&dense.dense), &mut points)
        .map_err(ReconstructionStageError::Reconstruction)?;
    surface.consolidated_points = points
        .iter()
        .zip(original)
        .enumerate()
        .filter(|(_, (consolidated, original))| consolidated != original)
        .map(|(index, (point, _))| ConsolidatedPoint {
            index,
            point: *point,
        })
        .collect();
    MeshStageArtifact {
        schema_version: MESH_STAGE_ARTIFACT_SCHEMA_VERSION,
        dense_hash: dense.content_hash(),
        mesh: surface,
    }
    .check_finite()
}

/// The single-call [`ReconstructionResult`] from the three artifacts of one chain,
/// after checking every schema version, then `dense.cameras_hash` against `cameras`,
/// then `mesh.dense_hash` against `dense` (the chain is transitive, so the mesh
/// artifact records only its dense input).
pub fn compose_reconstruction(
    cameras: &CameraStageArtifact,
    dense: &DenseStageArtifact,
    mesh: &MeshStageArtifact,
) -> Result<ReconstructionResult, ReconstructionStageError> {
    cameras.check_schema_version()?;
    dense.check_schema_version()?;
    mesh.check_schema_version()?;
    check_identity(
        ReconstructionStage::Dense,
        StageInput::Cameras,
        &dense.cameras_hash,
        cameras.content_hash(),
    )?;
    check_identity(
        ReconstructionStage::Mesh,
        StageInput::Dense,
        &mesh.dense_hash,
        dense.content_hash(),
    )?;
    // The result owns its buffers, so this is the one place the stage outputs are copied.
    let mut dense_evidence = dense.dense.clone();
    for consolidated in &mesh.mesh.consolidated_points {
        let Some(point) = dense_evidence.points.get_mut(consolidated.index) else {
            return Err(ReconstructionStageError::Malformed {
                artifact: ReconstructionStage::Mesh,
                message: format!(
                    "consolidated point {} is outside the {} dense points",
                    consolidated.index,
                    dense.dense.points.len()
                ),
            });
        };
        *point = consolidated.point;
    }
    Ok(into_result(
        cameras.sparse.clone(),
        dense_evidence,
        mesh.mesh.clone(),
    ))
}

/// Dense stage core: dense depth of the camera solve's registered views, with the
/// solve's effective options, and the dense diagnostics.
pub(crate) fn estimate_dense_evidence(
    frames: &[FrameInput],
    sparse: &SparseReconstruction,
) -> DenseEvidence {
    let cameras = registered_cameras(&sparse.dense_cameras);
    let sparse_points: Vec<Vector3<f64>> = sparse
        .dense_sparse_points
        .iter()
        .map(|point| Vector3::from(*point))
        .collect();
    let options = sparse.options;
    let analysis = crate::dense::estimate_depth_points_with_budget(
        frames,
        &cameras,
        &sparse_points,
        sparse.focal_length_pixels as f64,
        options.max_dense_working_set_bytes,
        options.max_features,
        options.descriptor_radius,
    );
    let dense = &analysis.stats;
    let mut warnings = Vec::new();
    if let Some(limit) = dense.reference_view_limit {
        warnings.push(format!(
            "The dense working set for every reference view ({} MiB) exceeds the {} MiB budget, so dense reconstruction kept the first {limit} reference view(s) in its usual order ({} MiB). Each kept view passed the same gates as a full run; the remaining views were not reconstructed.",
            dense.full_working_set_bytes / (1024 * 1024),
            dense.working_set_budget_bytes / (1024 * 1024),
            dense.working_set_estimate.total_bytes / (1024 * 1024),
        ));
    }
    warnings.extend(crate::dense_warning(dense));
    DenseEvidence {
        stats: analysis.stats,
        points: analysis.points,
        grid_sites: analysis.grid_sites,
        point_attributes: analysis.point_attributes,
        warnings,
    }
}

/// Single-call tail: the mesh stage core consolidates the dense stage's own points in
/// place (nothing else needs them unchanged), then the outputs move into the result.
pub(crate) fn finish_reconstruction(
    sparse: SparseReconstruction,
    mut dense: DenseEvidence,
) -> Result<ReconstructionResult, String> {
    let mut points = std::mem::take(&mut dense.points);
    let surface = build_surface(&sparse, DenseFields::of(&dense), &mut points)?;
    dense.points = points;
    Ok(into_result(sparse, dense, surface))
}

/// The dense-stage fields the mesh stage reads; its points are passed separately, as
/// the buffer consolidation may move.
#[derive(Clone, Copy)]
struct DenseFields<'a> {
    stats: &'a DenseStats,
    grid_sites: &'a [DenseGridSite],
    point_attributes: &'a [EvidencePointAttributes],
}

impl<'a> DenseFields<'a> {
    fn of(dense: &'a DenseEvidence) -> Self {
        Self {
            stats: &dense.stats,
            grid_sites: &dense.grid_sites,
            point_attributes: &dense.point_attributes,
        }
    }
}

fn registered_cameras(cameras: &[StageCamera]) -> Vec<RegisteredCamera> {
    cameras.iter().map(StageCamera::registered).collect()
}

/// Mesh stage core: the bounded dense mesh, then cross-reference surface fusion and the
/// connected-surface merge. It borrows the camera and dense outputs and changes only
/// `points` (the seam vertices fusion moves) and the surface it returns. Consolidation
/// sees only the mesh stage's own warning, so it cannot rewrite another stage's.
fn build_surface(
    sparse: &SparseReconstruction,
    dense: DenseFields<'_>,
    points: &mut [Point3],
) -> Result<SurfaceMesh, String> {
    let final_cameras = registered_cameras(&sparse.final_cameras);
    let analysis = crate::mesh::reconstruct_dense_mesh(
        points,
        dense.grid_sites,
        dense.stats,
        &registered_cameras(&sparse.dense_cameras),
        sparse.image_width,
        sparse.image_height,
        sparse.focal_length_pixels as f64,
    );
    let mut surface = SurfaceMesh {
        stats: analysis.stats,
        triangles: analysis.triangles,
        triangle_sources: Vec::new(),
        consolidated_points: Vec::new(),
        mesh_warning: None,
        consolidation_warnings: Vec::new(),
    };
    let mut warnings: Vec<String> = crate::mesh_warning(&surface.stats).into_iter().collect();
    let owned = warnings.len();
    crate::surface_fusion::consolidate_surface(
        crate::surface_fusion::SurfaceSupport {
            calibrated_pair: sparse.calibrated_pair.as_ref(),
            registered_views: &sparse.registered_views,
            final_cameras: &final_cameras,
            reference_patches: &dense.stats.reference_patches,
            point_attributes: dense.point_attributes,
        },
        crate::surface_fusion::SurfaceTarget {
            points,
            triangles: &mut surface.triangles,
            triangle_sources: &mut surface.triangle_sources,
            mesh: &mut surface.stats,
            warnings: &mut warnings,
        },
    )?;
    surface.consolidation_warnings = warnings.split_off(owned);
    surface.mesh_warning = warnings.pop();
    Ok(surface)
}

/// The classic result of one chain's owned outputs. `dense.points` must already be the
/// consolidated points; warnings follow the single-call order: camera warnings before
/// dense, dense, mesh, camera warnings after the mesh (recovery last), consolidation.
fn into_result(
    sparse: SparseReconstruction,
    dense: DenseEvidence,
    surface: SurfaceMesh,
) -> ReconstructionResult {
    let SparseReconstruction {
        options: _,
        image_width: _,
        image_height: _,
        focal_length_pixels: _,
        cameras,
        points,
        pairs,
        calibrated_pair,
        mut multi_view,
        revisits,
        registered_views,
        seed_candidates,
        bootstrap,
        final_cameras,
        dense_cameras: _,
        dense_sparse_points: _,
        track_scope: _,
        warnings_before_dense,
        warnings_after_mesh,
    } = sparse;
    let SurfaceMesh {
        stats: mesh,
        triangles: mesh_triangles,
        triangle_sources: mesh_triangle_sources,
        consolidated_points: _,
        mesh_warning,
        consolidation_warnings,
    } = surface;
    multi_view.bundle_adjustment.final_cameras = registered_cameras(&final_cameras);
    let mut warnings = warnings_before_dense;
    warnings.extend(dense.warnings);
    warnings.extend(mesh_warning);
    warnings.extend(warnings_after_mesh);
    warnings.extend(consolidation_warnings);
    ReconstructionResult {
        cameras,
        points,
        dense_points: dense.points,
        dense_grid_sites: dense.grid_sites,
        dense_point_attributes: dense.point_attributes,
        dense: dense.stats,
        mesh_triangles,
        mesh_triangle_sources,
        mesh,
        pairs,
        calibrated_pair,
        multi_view,
        revisits,
        registered_views,
        seed_candidates,
        bootstrap,
        warnings,
    }
}

/// Finds the first NaN or infinite float of a serializable value without building its
/// document: a serializer that only inspects floats and tracks the field path.
mod finite {
    use serde::ser::{self, Serialize};
    use std::fmt;

    pub(super) fn first_non_finite<T: Serialize>(value: &T) -> Option<String> {
        let mut probe = Probe { path: Vec::new() };
        match value.serialize(&mut probe) {
            Ok(()) => None,
            Err(Found(field)) => Some(field),
        }
    }

    struct Probe {
        path: Vec<String>,
    }

    impl Probe {
        fn check(&self, finite: bool) -> Result<(), Found> {
            if finite {
                Ok(())
            } else {
                Err(Found(self.path.join(".")))
            }
        }

        fn nested<T: ?Sized + Serialize>(&mut self, key: String, value: &T) -> Result<(), Found> {
            self.path.push(key);
            value.serialize(&mut *self)?;
            self.path.pop();
            Ok(())
        }
    }

    #[derive(Debug)]
    struct Found(String);

    impl fmt::Display for Found {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "non-finite value at `{}`", self.0)
        }
    }

    impl std::error::Error for Found {}

    impl ser::Error for Found {
        fn custom<T: fmt::Display>(message: T) -> Self {
            Self(message.to_string())
        }
    }

    macro_rules! ignore {
        ($($method:ident: $type:ty),* $(,)?) => {
            $(fn $method(self, _: $type) -> Result<(), Found> { Ok(()) })*
        };
    }

    impl ser::Serializer for &mut Probe {
        type Ok = ();
        type Error = Found;
        type SerializeSeq = Self;
        type SerializeTuple = Self;
        type SerializeTupleStruct = Self;
        type SerializeTupleVariant = Self;
        type SerializeMap = Self;
        type SerializeStruct = Self;
        type SerializeStructVariant = Self;

        ignore!(
            serialize_bool: bool, serialize_i8: i8, serialize_i16: i16, serialize_i32: i32,
            serialize_i64: i64, serialize_u8: u8, serialize_u16: u16, serialize_u32: u32,
            serialize_u64: u64, serialize_char: char, serialize_str: &str,
            serialize_bytes: &[u8], serialize_unit_struct: &'static str,
        );

        fn serialize_f32(self, value: f32) -> Result<(), Found> {
            self.check(value.is_finite())
        }

        fn serialize_f64(self, value: f64) -> Result<(), Found> {
            self.check(value.is_finite())
        }

        fn serialize_none(self) -> Result<(), Found> {
            Ok(())
        }

        fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<(), Found> {
            value.serialize(self)
        }

        fn serialize_unit(self) -> Result<(), Found> {
            Ok(())
        }

        fn serialize_unit_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
        ) -> Result<(), Found> {
            Ok(())
        }

        fn serialize_newtype_struct<T: ?Sized + Serialize>(
            self,
            _: &'static str,
            value: &T,
        ) -> Result<(), Found> {
            value.serialize(self)
        }

        fn serialize_newtype_variant<T: ?Sized + Serialize>(
            self,
            _: &'static str,
            _: u32,
            variant: &'static str,
            value: &T,
        ) -> Result<(), Found> {
            self.nested(variant.to_owned(), value)
        }

        fn serialize_seq(self, _: Option<usize>) -> Result<Self, Found> {
            self.path.push(String::new());
            Ok(self)
        }

        fn serialize_tuple(self, len: usize) -> Result<Self, Found> {
            self.serialize_seq(Some(len))
        }

        fn serialize_tuple_struct(self, _: &'static str, len: usize) -> Result<Self, Found> {
            self.serialize_seq(Some(len))
        }

        fn serialize_tuple_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            len: usize,
        ) -> Result<Self, Found> {
            self.serialize_seq(Some(len))
        }

        fn serialize_map(self, _: Option<usize>) -> Result<Self, Found> {
            Ok(self)
        }

        fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self, Found> {
            Ok(self)
        }

        fn serialize_struct_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: usize,
        ) -> Result<Self, Found> {
            Ok(self)
        }
    }

    /// Sequences keep a running element index as the last path segment.
    macro_rules! sequence {
        ($($trait:ident :: $method:ident),* $(,)?) => {
            $(impl ser::$trait for &mut Probe {
                type Ok = ();
                type Error = Found;

                fn $method<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Found> {
                    let index = self.path.last_mut().expect("sequence path segment");
                    let current: usize = index.parse().map_or(0, |last: usize| last + 1);
                    *index = current.to_string();
                    value.serialize(&mut **self)
                }

                fn end(self) -> Result<(), Found> {
                    self.path.pop();
                    Ok(())
                }
            })*
        };
    }

    sequence!(
        SerializeSeq::serialize_element,
        SerializeTuple::serialize_element,
        SerializeTupleStruct::serialize_field,
        SerializeTupleVariant::serialize_field,
    );

    impl ser::SerializeMap for &mut Probe {
        type Ok = ();
        type Error = Found;

        fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), Found> {
            key.serialize(&mut **self)
        }

        fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Found> {
            self.nested("[value]".to_owned(), value)
        }

        fn end(self) -> Result<(), Found> {
            Ok(())
        }
    }

    macro_rules! fields {
        ($($trait:ident),* $(,)?) => {
            $(impl ser::$trait for &mut Probe {
                type Ok = ();
                type Error = Found;

                fn serialize_field<T: ?Sized + Serialize>(
                    &mut self,
                    key: &'static str,
                    value: &T,
                ) -> Result<(), Found> {
                    self.nested(key.to_owned(), value)
                }

                fn end(self) -> Result<(), Found> {
                    Ok(())
                }
            })*
        };
    }

    fields!(SerializeStruct, SerializeStructVariant);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_request() -> ReconstructionRequest {
        let flat = FrameInput {
            width: 48,
            height: 32,
            rgba: [120, 120, 120, 255].repeat(48 * 32),
        };
        ReconstructionRequest {
            frames: vec![flat.clone(), flat],
            options: ReconstructionOptions::default(),
        }
    }

    #[test]
    fn non_finite_options_are_rejected_before_any_artifact() {
        type Edit = (&'static str, fn(&mut ReconstructionOptions));
        let edits: [Edit; 6] = [
            ("max descriptor distance", |o| {
                o.max_descriptor_distance = f32::NAN
            }),
            ("max descriptor distance", |o| {
                o.max_descriptor_distance = f32::INFINITY
            }),
            ("max descriptor distance", |o| {
                o.max_descriptor_distance = f32::NEG_INFINITY
            }),
            ("ratio threshold", |o| o.ratio_threshold = f32::NAN),
            ("ratio threshold", |o| o.ratio_threshold = f32::INFINITY),
            ("focal length", |o| o.focal_length_pixels = Some(f32::NAN)),
        ];
        for (option, edit) in edits {
            let mut request = flat_request();
            edit(&mut request.options);
            let single = crate::reconstruct(&request).expect_err("the single call rejects it");
            assert!(
                single.contains(option) && single.contains("finite"),
                "{single}"
            );
            assert_eq!(
                reconstruct_cameras(&request).expect_err("the camera stage rejects it"),
                ReconstructionStageError::Reconstruction(single)
            );
        }
    }

    #[test]
    fn artifacts_with_non_finite_values_are_refused_not_stored() {
        let request = flat_request();
        let cameras = reconstruct_cameras(&request).expect("camera stage");
        let dense = reconstruct_dense(&request, &cameras).expect("dense stage");
        let mesh = reconstruct_mesh(&cameras, &dense).expect("mesh stage");
        assert_eq!(finite::first_non_finite(&cameras), None);
        assert_eq!(finite::first_non_finite(&dense), None);
        assert_eq!(finite::first_non_finite(&mesh), None);

        let mut nan_cameras = cameras.clone();
        nan_cameras.sparse.pairs[0].median_motion = f32::NAN;
        // Why the guard exists: the stored document would not reload.
        assert!(CameraStageArtifact::from_json(&nan_cameras.to_json()).is_err());
        assert_eq!(
            nan_cameras.check_finite(),
            Err(ReconstructionStageError::NonFinite {
                artifact: ReconstructionStage::Cameras,
                field: "sparse.pairs.0.median_motion".into(),
            })
        );

        let mut infinite_dense = dense.clone();
        infinite_dense.dense.stats.median_photometric_error = Some(f32::INFINITY);
        assert_eq!(
            infinite_dense.check_finite(),
            Err(ReconstructionStageError::NonFinite {
                artifact: ReconstructionStage::Dense,
                field: "dense.stats.median_photometric_error".into(),
            })
        );

        let mut nan_mesh = mesh.clone();
        let point = Point3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
            confidence: 1.0,
            r: 0,
            g: 0,
            b: 0,
        };
        nan_mesh.mesh.consolidated_points = vec![
            ConsolidatedPoint { index: 0, point },
            ConsolidatedPoint {
                index: 1,
                point: Point3 {
                    y: f32::NEG_INFINITY,
                    ..point
                },
            },
        ];
        assert_eq!(
            nan_mesh.check_finite(),
            Err(ReconstructionStageError::NonFinite {
                artifact: ReconstructionStage::Mesh,
                field: "mesh.consolidated_points.1.point.y".into(),
            })
        );
        // f64 payloads are checked too.
        let mut nan_pose = cameras;
        nan_pose.sparse.dense_sparse_points = vec![[0.0, f64::NAN, 1.0]];
        assert!(matches!(
            nan_pose.check_finite(),
            Err(ReconstructionStageError::NonFinite { field, .. })
                if field == "sparse.dense_sparse_points.0.1"
        ));
    }
}

// Independently authored acceptance only; shared-stage API translation is allowed.
#[cfg(test)]
#[path = "native_stage_acceptance_tests.rs"]
mod native_stage_acceptance_tests;
