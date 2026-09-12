//! Reusable reconstruction semantics and interchange contracts.

pub mod colmap;

// Keep the established reconstruction API at crate root while the large kernel body
// remains isolated from format/interchange modules.
include!("reconstruction.rs");

mod browser_contract;
pub use browser_contract::{
    reconstruct_browser, BrowserReconstructionResult, BundleAdjustmentStatus, CameraKind,
    CameraPipelineState, DenseCameraRole, FrameCameraState, RegistrationStatus,
};

#[cfg(test)]
mod surface_completion_tests;
