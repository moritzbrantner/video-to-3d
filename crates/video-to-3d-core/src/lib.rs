//! Reusable reconstruction semantics and interchange contracts.

pub mod coarse_collision;
pub mod colmap;
pub mod feature_analysis;
pub mod geometry_confidence;
pub mod input_readiness;
pub mod scene_model;
pub mod scene_project;
pub mod scene_runner;
pub mod scene_store;
pub mod surface_materials;
pub mod textured_glb;

// Keep the established reconstruction API at crate root while the large kernel body
// remains isolated from format/interchange modules.
include!("reconstruction.rs");

mod browser_contract;
mod learned_depth;
mod reconstruction_evidence;
mod surface_fusion;
pub use browser_contract::{
    BrowserReconstructionResult, BundleAdjustmentStatus, CameraKind, CameraPipelineState,
    DenseCameraRole, FrameCameraState, RegistrationStatus,
};
pub use learned_depth::{
    evaluate_relative_depth, LearnedDepthCamera, RelativeDepthEvaluation, RelativeDepthFitKind,
    RelativeDepthFrame, RelativeDepthFrameDiagnostics,
};
pub use reconstruction_evidence::{
    classic_reference_patch_regions, EvidenceCamera, EvidenceCameraAuthority, EvidenceOrigin,
    EvidenceRange, EvidenceScale, ReconstructionEvidenceSummary, ReconstructionEvidenceView,
    ReconstructionProviderClass, ReconstructionProviderDescriptor, SurfaceEvidenceRegion,
    RECONSTRUCTION_EVIDENCE_SCHEMA_VERSION,
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
