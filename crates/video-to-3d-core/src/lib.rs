//! Reusable reconstruction semantics and interchange contracts.

pub mod colmap;
pub mod feature_analysis;

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
    EvidenceCamera, EvidenceCameraAuthority, EvidenceOrigin, EvidenceRange, EvidenceScale,
    ReconstructionEvidenceSummary, ReconstructionEvidenceView, ReconstructionProviderClass,
    ReconstructionProviderDescriptor, SurfaceEvidenceRegion,
    RECONSTRUCTION_EVIDENCE_SCHEMA_VERSION,
};

pub fn reconstruct_browser(
    request: &ReconstructionRequest,
) -> Result<BrowserReconstructionResult, String> {
    let mut result = browser_contract::reconstruct_browser(request)?;
    let evidence = ReconstructionEvidenceView::from_classic(&result.reconstruction)?;
    result.reconstruction.warnings.push(evidence.diagnostic());
    Ok(result)
}

#[cfg(test)]
mod surface_completion_tests;
