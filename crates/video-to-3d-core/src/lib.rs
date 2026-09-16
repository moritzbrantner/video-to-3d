//! Reusable reconstruction semantics and interchange contracts.

pub mod colmap;
pub mod feature_analysis;

// Keep the established reconstruction API at crate root while the large kernel body
// remains isolated from format/interchange modules.
include!("reconstruction.rs");

mod browser_contract;
mod surface_fusion;
pub use browser_contract::{
    BrowserReconstructionResult, BundleAdjustmentStatus, CameraKind, CameraPipelineState,
    DenseCameraRole, FrameCameraState, RegistrationStatus,
};

pub fn reconstruct_browser(
    request: &ReconstructionRequest,
) -> Result<BrowserReconstructionResult, String> {
    let mut result = browser_contract::reconstruct_browser(request)?;
    surface_fusion::consolidate_surface_patches(&mut result.reconstruction);
    Ok(result)
}

#[cfg(test)]
mod surface_completion_tests;
