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
    fn registered(&self) -> RegisteredCamera {
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
    Ok(CameraStageArtifact {
        schema_version: CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION,
        request_hash: request_content_hash(request),
        sparse,
    })
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
    Ok(DenseStageArtifact {
        schema_version: DENSE_STAGE_ARTIFACT_SCHEMA_VERSION,
        request_hash: cameras.request_hash.clone(),
        cameras_hash: cameras.content_hash(),
        dense: estimate_dense_evidence(&request.frames, &cameras.sparse),
    })
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
    // Consolidation reads the whole classic result and moves dense vertices in place,
    // so it runs on a working copy; the artifact keeps only what it changed.
    let (mut result, surface_inputs, _) = into_result(cameras.sparse.clone(), dense.dense.clone());
    let (mesh_warning, consolidation_warnings) = build_surface(&mut result, &surface_inputs)
        .map_err(ReconstructionStageError::Reconstruction)?;
    let consolidated_points = result
        .dense_points
        .iter()
        .zip(&dense.dense.points)
        .enumerate()
        .filter(|(_, (consolidated, original))| consolidated != original)
        .map(|(index, (point, _))| ConsolidatedPoint {
            index,
            point: *point,
        })
        .collect();
    Ok(MeshStageArtifact {
        schema_version: MESH_STAGE_ARTIFACT_SCHEMA_VERSION,
        dense_hash: dense.content_hash(),
        mesh: SurfaceMesh {
            stats: std::mem::take(&mut result.mesh),
            triangles: std::mem::take(&mut result.mesh_triangles),
            triangle_sources: std::mem::take(&mut result.mesh_triangle_sources),
            consolidated_points,
            mesh_warning,
            consolidation_warnings,
        },
    })
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
    let (mut result, _, warnings) = into_result(cameras.sparse.clone(), dense.dense.clone());
    let surface = &mesh.mesh;
    for consolidated in &surface.consolidated_points {
        let Some(point) = result.dense_points.get_mut(consolidated.index) else {
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
    result.mesh = surface.stats.clone();
    result.mesh_triangles = surface.triangles.clone();
    result.mesh_triangle_sources = surface.triangle_sources.clone();
    result.warnings = ordered_warnings(
        warnings,
        surface.mesh_warning.clone(),
        surface.consolidation_warnings.clone(),
    );
    Ok(result)
}

/// Dense stage core: dense depth of the camera solve's registered views, with the
/// solve's effective options, and the dense diagnostics.
pub(crate) fn estimate_dense_evidence(
    frames: &[FrameInput],
    sparse: &SparseReconstruction,
) -> DenseEvidence {
    let cameras: Vec<RegisteredCamera> = sparse
        .dense_cameras
        .iter()
        .map(StageCamera::registered)
        .collect();
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

/// Single-call tail: the mesh stage core on the owned stage outputs, without the working
/// copy the public mesh stage needs to keep its dense input unchanged.
pub(crate) fn finish_reconstruction(
    sparse: SparseReconstruction,
    dense: DenseEvidence,
) -> Result<ReconstructionResult, String> {
    let (mut result, surface_inputs, warnings) = into_result(sparse, dense);
    let (mesh_warning, consolidation_warnings) = build_surface(&mut result, &surface_inputs)?;
    result.warnings = ordered_warnings(warnings, mesh_warning, consolidation_warnings);
    Ok(result)
}

/// What the mesh stage reads from the camera stage besides the assembled result.
struct SurfaceInputs {
    cameras: Vec<RegisteredCamera>,
    width: u32,
    height: u32,
    focal: f64,
}

/// The camera and dense stage warnings, in result order around the mesh stage's.
struct StageWarnings {
    before_dense: Vec<String>,
    dense: Vec<String>,
    after_mesh: Vec<String>,
}

/// The result of the camera and dense outputs, with no mesh and no warnings yet.
fn into_result(
    sparse: SparseReconstruction,
    dense: DenseEvidence,
) -> (ReconstructionResult, SurfaceInputs, StageWarnings) {
    let SparseReconstruction {
        options: _,
        image_width,
        image_height,
        focal_length_pixels,
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
        dense_cameras,
        dense_sparse_points: _,
        track_scope: _,
        warnings_before_dense,
        warnings_after_mesh,
    } = sparse;
    multi_view.bundle_adjustment.final_cameras =
        final_cameras.iter().map(StageCamera::registered).collect();
    let result = ReconstructionResult {
        cameras,
        points,
        dense_points: dense.points,
        dense_grid_sites: dense.grid_sites,
        dense_point_attributes: dense.point_attributes,
        dense: dense.stats,
        mesh_triangles: Vec::new(),
        mesh_triangle_sources: Vec::new(),
        mesh: MeshStats::default(),
        pairs,
        calibrated_pair,
        multi_view,
        revisits,
        registered_views,
        seed_candidates,
        bootstrap,
        warnings: Vec::new(),
    };
    let surface_inputs = SurfaceInputs {
        cameras: dense_cameras.iter().map(StageCamera::registered).collect(),
        width: image_width,
        height: image_height,
        focal: focal_length_pixels as f64,
    };
    let warnings = StageWarnings {
        before_dense: warnings_before_dense,
        dense: dense.warnings,
        after_mesh: warnings_after_mesh,
    };
    (result, surface_inputs, warnings)
}

/// Mesh stage core: the bounded dense mesh, then cross-reference surface fusion and the
/// connected-surface merge, on `result` in place. Returns the mesh diagnostic (refreshed
/// by the merge) and the consolidation diagnostics; the other stages' warnings are not
/// in `result` meanwhile, so consolidation can only rewrite the mesh stage's own.
fn build_surface(
    result: &mut ReconstructionResult,
    inputs: &SurfaceInputs,
) -> Result<(Option<String>, Vec<String>), String> {
    let analysis = crate::mesh::reconstruct_dense_mesh(
        &result.dense_points,
        &result.dense_grid_sites,
        &result.dense,
        &inputs.cameras,
        inputs.width,
        inputs.height,
        inputs.focal,
    );
    result.mesh = analysis.stats;
    result.mesh_triangles = analysis.triangles;
    result.mesh_triangle_sources = Vec::new();
    result.warnings = crate::mesh_warning(&result.mesh).into_iter().collect();
    let owned = result.warnings.len();
    crate::surface_fusion::consolidate_surface_evidence(result)?;
    let mut mesh_warning = std::mem::take(&mut result.warnings);
    let consolidation_warnings = mesh_warning.split_off(owned);
    Ok((mesh_warning.pop(), consolidation_warnings))
}

fn ordered_warnings(
    stages: StageWarnings,
    mesh_warning: Option<String>,
    consolidation_warnings: Vec<String>,
) -> Vec<String> {
    let StageWarnings {
        before_dense,
        dense,
        after_mesh,
    } = stages;
    let mut warnings = before_dense;
    warnings.extend(dense);
    warnings.extend(mesh_warning);
    warnings.extend(after_mesh);
    warnings.extend(consolidation_warnings);
    warnings
}
