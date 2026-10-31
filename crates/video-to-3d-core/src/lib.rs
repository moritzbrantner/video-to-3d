//! Reusable reconstruction semantics and interchange contracts.
//!
//! The classic pipeline is available as one call ([`reconstruct`], [`reconstruct_browser`])
//! and as three public stages with versioned artifacts ([`reconstruct_cameras`] →
//! [`reconstruct_dense`] → [`reconstruct_mesh`], joined by [`compose_reconstruction`]);
//! both give the same [`ReconstructionResult`].

pub mod clean_plate;
pub mod coarse_collision;
pub mod colmap;
pub mod evidence_interchange;
pub mod feature_analysis;
pub mod geometry_confidence;
pub mod input_readiness;
pub mod object_generation;
pub mod object_identity;
pub mod object_masks;
pub mod object_placement;
pub mod object_references;
pub mod placement_validation;
pub mod scene_artifacts;
#[cfg(not(target_arch = "wasm32"))]
pub mod scene_export;
pub mod scene_model;
pub mod scene_project;
pub mod scene_runner;
pub mod scene_store;
pub mod semantic_inventory;
pub mod surface_materials;
pub mod textured_glb;

mod dense;
mod keyframe_selection;
mod mesh;
mod multi_view;
mod pnp;
mod reconstruction;
mod revisit;
mod two_view;

pub use dense::{DenseGridSite, DenseReferencePatchStats, DenseStats, DenseWorkingSetEstimate};
pub use keyframe_selection::{
    ClipSegmentStats, FrameDecision, FrameSelectionStats, KeyframeSelectionStats, SegmentBreak,
};
pub use mesh::{MeshStats, MeshTriangle};
pub use multi_view::{
    BundleAdjustmentStats, MultiViewStats, NewLandmarkStats, RegistrationCandidateStats,
};
pub(crate) use reconstruction::{
    dense_warning, detect_features, match_features, mesh_warning, motion_guided_matches,
    reconstruct_sparse_with_control, to_luma, Feature, FeatureMatch, ReconstructionControl,
    TrackScope, MIN_TRACK_MATCHES, MIN_TRACK_OVERLAP,
};
pub use reconstruction::{
    reconstruct, BootstrapDiagnosis, BootstrapEscalation, BootstrapGate, CalibratedPairStats,
    CameraPose, FrameInput, PairStats, Point3, ReconstructionOptions, ReconstructionRequest,
    ReconstructionResult, RegisteredViewStats, SeedCandidateStats, SeedGate, SeedGateCount,
    SegmentSolveStats,
};
#[cfg(test)]
pub(crate) use reconstruction::{reconstruct_once, reconstruct_sparse};
pub use revisit::{RevisitCandidateStats, RevisitClosureStats, RevisitRecoveryStats, RevisitStats};

mod browser_contract;
mod controlled_json;
mod learned_depth;
mod reconstruction_evidence;
mod reconstruction_stages;
mod surface_fusion;
mod surface_merge;
pub use browser_contract::{
    BrowserReconstructionResult, BundleAdjustmentStatus, CameraKind, CameraPipelineState,
    DenseCameraRole, FrameCameraState, RegistrationStatus,
};
pub use learned_depth::{
    evaluate_relative_depth, LearnedDepthCamera, RelativeDepthEvaluation, RelativeDepthFitKind,
    RelativeDepthFrame, RelativeDepthFrameDiagnostics,
};
pub use reconstruction_evidence::{
    classic_reference_patch_regions, DerivedVertexSource, EvidenceCamera, EvidenceCameraAuthority,
    EvidenceOrigin, EvidencePointAttributes, EvidenceRange, EvidenceScale,
    ReconstructionEvidenceSummary, ReconstructionEvidenceView, ReconstructionProviderClass,
    ReconstructionProviderDescriptor, SurfaceEvidenceRegion, TriangleSourceCorners,
    RECONSTRUCTION_EVIDENCE_SCHEMA_VERSION,
};
pub use reconstruction_stages::{
    compose_reconstruction, reconstruct_cameras, reconstruct_cameras_with_cancel,
    reconstruct_dense, reconstruct_dense_with_cancel, reconstruct_mesh, reconstruct_mesh_owned,
    reconstruct_mesh_owned_with_cancel, reconstruct_mesh_with_cancel, request_content_hash,
    CameraStageArtifact, ConsumedDenseGeometry, DenseStageArtifact, MeshStageArtifact,
    ReconstructionStage, ReconstructionStageError, StageInput,
    CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION, DENSE_STAGE_ARTIFACT_SCHEMA_VERSION,
    MESH_STAGE_ARTIFACT_SCHEMA_VERSION,
};

pub fn reconstruct_browser(
    request: &ReconstructionRequest,
) -> Result<BrowserReconstructionResult, String> {
    let mut result = browser_contract::reconstruct_browser(request)?;
    let evidence = ReconstructionEvidenceView::from_classic(&result.reconstruction)?;
    result.reconstruction.warnings.push(evidence.diagnostic());
    let confidence = result
        .geometry_confidence
        .diagnostic(&result.reconstruction.mesh_triangles);
    result.reconstruction.warnings.push(confidence);
    result
        .reconstruction
        .warnings
        .push(result.collision.diagnostic());
    Ok(result)
}

#[cfg(test)]
mod geometry_confidence_tests;
#[cfg(test)]
mod surface_completion_tests;
