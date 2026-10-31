//! The classic reconstruction kernel: request and result contract, the camera stage with
//! its recovery retries, and the shared feature pipeline. The crate root re-exports this
//! API, so these modules are an internal layout rather than public paths.

mod bootstrap;
mod control;
mod features;
mod matching;
mod recovery;
mod seed;
mod segment;
mod solve;
mod types;
mod warnings;

#[cfg(test)]
mod tests;

pub(crate) use control::ReconstructionControl;
pub(crate) use features::{detect_features, to_luma};
pub(crate) use matching::{match_features, motion_guided_matches};
pub use recovery::reconstruct;
#[cfg(test)]
pub(crate) use recovery::reconstruct_sparse;
pub(crate) use recovery::reconstruct_sparse_with_control;
#[cfg(test)]
pub(crate) use solve::reconstruct_once;
pub(crate) use solve::{MIN_TRACK_MATCHES, MIN_TRACK_OVERLAP};
pub use types::*;
pub(crate) use warnings::{dense_warning, mesh_warning};
