//! Reusable reconstruction semantics and interchange contracts.

pub mod colmap;

// Keep the established reconstruction API at crate root while the large kernel body
// remains isolated from format/interchange modules.
include!("reconstruction.rs");

#[cfg(test)]
mod surface_completion_tests;
