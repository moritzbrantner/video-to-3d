//! Public staged reconstruction: cameras → dense evidence → consolidated mesh.
//!
//! contract: implemented by #172. This file only fixes the public API that the
//! acceptance tests in `tests/reconstruction_stages.rs` exercise. Every stage
//! returns [`ReconstructionStageError::NotImplemented`] until #172 replaces the
//! bodies; the artifact structs carry only their identity fields so far and gain
//! their owned stage payload in the implementation. Keep the field names, the
//! function signatures, the error variants and the schema-version constants;
//! everything else (payload layout, hashing of the request) is the implementer's
//! choice within the contract that the tests and the doc comments below state.

use crate::scene_project::ContentHash;
use crate::{ReconstructionRequest, ReconstructionResult};
use serde::{Deserialize, Serialize};
use std::fmt;

// contract: implemented by #172
pub const CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION: u32 = 1;
// contract: implemented by #172
pub const DENSE_STAGE_ARTIFACT_SCHEMA_VERSION: u32 = 1;
// contract: implemented by #172
pub const MESH_STAGE_ARTIFACT_SCHEMA_VERSION: u32 = 1;

/// One stage of the classic pipeline; also names the artifact that stage produces.
// contract: implemented by #172
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionStage {
    Cameras,
    Dense,
    Mesh,
}

/// The input whose recorded identity a stage checks.
// contract: implemented by #172
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageInput {
    /// The [`ReconstructionRequest`] (frames and options), via [`request_content_hash`].
    Request,
    /// A [`CameraStageArtifact`], via [`CameraStageArtifact::content_hash`].
    Cameras,
    /// A [`DenseStageArtifact`], via [`DenseStageArtifact::content_hash`].
    Dense,
}

// contract: implemented by #172
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
    /// A stored artifact document is not valid JSON for its type.
    Malformed {
        artifact: ReconstructionStage,
        message: String,
    },
    /// The reconstruction itself failed (the `String` error of the single-call pipeline).
    Reconstruction(String),
    /// Placeholder until #172 lands. Must not exist after the implementation.
    NotImplemented,
}

impl fmt::Display for ReconstructionStageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ReconstructionStageError {}

/// Sparse registration result: seed pair, registered cameras, sparse points, the
/// effective options of the selected (possibly recovered) solve and every
/// camera-stage diagnostic.
// contract: implemented by #172
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraStageArtifact {
    pub schema_version: u32,
    /// [`request_content_hash`] of the request this artifact was computed from.
    pub request_hash: ContentHash,
}

/// Dense evidence of one camera artifact: dense points, grid sites, per-point
/// attributes and dense diagnostics.
// contract: implemented by #172
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DenseStageArtifact {
    pub schema_version: u32,
    pub request_hash: ContentHash,
    /// [`CameraStageArtifact::content_hash`] of the camera artifact it was computed from.
    pub cameras_hash: ContentHash,
}

/// Consolidated surface of one dense artifact: what the mesh stage and surface
/// consolidation own (triangles, triangle sources, consolidated vertex positions,
/// mesh and consolidation diagnostics).
// contract: implemented by #172
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshStageArtifact {
    pub schema_version: u32,
    /// [`DenseStageArtifact::content_hash`] of the dense artifact it was computed from.
    pub dense_hash: ContentHash,
}

macro_rules! stage_artifact_document {
    ($artifact:ty) => {
        impl $artifact {
            /// The stored document. Deterministic: equal artifacts give equal bytes.
            // contract: implemented by #172
            pub fn to_json(&self) -> String {
                serde_json::to_string(self).expect("stage artifacts serialize")
            }

            /// Read a stored document, rejecting any other `schema_version` with
            /// [`ReconstructionStageError::UnsupportedSchemaVersion`], and invalid JSON,
            /// a missing `schema_version` or another stage's document with
            /// [`ReconstructionStageError::Malformed`].
            // contract: implemented by #172
            pub fn from_json(_document: &str) -> Result<Self, ReconstructionStageError> {
                Err(ReconstructionStageError::NotImplemented)
            }

            /// `ContentHash::of_bytes(self.to_json().as_bytes())`: the identity a
            /// downstream artifact records and the scene store can verify.
            // contract: implemented by #172
            pub fn content_hash(&self) -> ContentHash {
                ContentHash::of_bytes(self.to_json().as_bytes())
            }
        }
    };
}

stage_artifact_document!(CameraStageArtifact);
stage_artifact_document!(DenseStageArtifact);
stage_artifact_document!(MeshStageArtifact);

/// Identity of a request's frames and options. Deterministic; any change to a
/// frame's dimensions or pixels, the frame order or the options changes it.
// contract: implemented by #172
pub fn request_content_hash(_request: &ReconstructionRequest) -> ContentHash {
    ContentHash::of_bytes(b"contract: implemented by #172")
}

/// Stage 1: frames → cameras, including the bounded registration recovery of
/// [`crate::reconstruct`] (recovery ranks sparse evidence only).
// contract: implemented by #172
pub fn reconstruct_cameras(
    _request: &ReconstructionRequest,
) -> Result<CameraStageArtifact, ReconstructionStageError> {
    Err(ReconstructionStageError::NotImplemented)
}

/// Stage 2: cameras + frames → dense evidence. Rejects a camera artifact of another
/// schema version or one whose `request_hash` is not this request's.
// contract: implemented by #172
pub fn reconstruct_dense(
    _request: &ReconstructionRequest,
    _cameras: &CameraStageArtifact,
) -> Result<DenseStageArtifact, ReconstructionStageError> {
    Err(ReconstructionStageError::NotImplemented)
}

/// Stage 3: dense evidence (+ the cameras it was computed from) → consolidated mesh.
/// No frames: the mesh stage and surface consolidation read no pixels.
// contract: implemented by #172
pub fn reconstruct_mesh(
    _cameras: &CameraStageArtifact,
    _dense: &DenseStageArtifact,
) -> Result<MeshStageArtifact, ReconstructionStageError> {
    Err(ReconstructionStageError::NotImplemented)
}

/// The single-call [`ReconstructionResult`] from the three artifacts of one chain,
/// after checking every schema version, then `dense.cameras_hash` against `cameras`,
/// then `mesh.dense_hash` against `dense` (the chain is transitive, so the mesh
/// artifact records only its dense input).
// contract: implemented by #172
pub fn compose_reconstruction(
    _cameras: &CameraStageArtifact,
    _dense: &DenseStageArtifact,
    _mesh: &MeshStageArtifact,
) -> Result<ReconstructionResult, ReconstructionStageError> {
    Err(ReconstructionStageError::NotImplemented)
}
