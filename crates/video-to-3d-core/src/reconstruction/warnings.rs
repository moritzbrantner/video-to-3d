//! User-facing dense and mesh diagnostics.

use crate::{DenseStats, MeshStats};

pub(crate) fn mesh_warning(mesh: &MeshStats) -> Option<String> {
    if !mesh.attempted {
        return None;
    }

    if mesh.accepted_triangles > 0 {
        return Some(format!(
            "Slice 4 bounded mesh reconstruction accepted {} triangles from {} candidate triangles across {} reference-grid cells. It rejected {} triangles at depth/spatial discontinuities and {} degenerate or orientation-flipped triangles. This is a bounded multi-reference, non-watertight surface preview. Browser texture projection is appearance-only over accepted per-reference topology; unsupported holes and metric scale are not claimed.",
            mesh.accepted_triangles,
            mesh.candidate_triangles,
            mesh.candidate_cells,
            mesh.rejected_discontinuities,
            mesh.rejected_degenerate
        ));
    }

    Some(format!(
        "Slice 4 bounded mesh reconstruction evaluated {} candidate triangles across {} reference-grid cells, but no triangle survived the continuity and non-degeneracy gates. No surface geometry was invented.",
        mesh.candidate_triangles, mesh.candidate_cells
    ))
}

pub(crate) fn dense_warning(dense: &DenseStats) -> Option<String> {
    if !dense.attempted {
        return None;
    }

    let reference_frame = dense.reference_frame.map_or(0, |frame| frame + 1);
    if dense.accepted_points > 0 {
        return Some(format!(
            "Slice 4 dense point fusion accepted {} fused scene points from reference frame {} using {} registered source views and {} inverse-depth hypotheses. {} primary candidates passed reciprocal depth consistency; spatial fusion rejected {} reverse observations. Fused points remain separate from the sparse map; dense coverage is bounded to accepted reference patches and does not claim unbounded aggregation or metric scale.",
            dense.accepted_points,
            reference_frame,
            dense.source_views,
            dense.depth_hypotheses,
            dense.reciprocal_consistent_points,
            dense.fusion_rejected_observations
        ));
    }

    if dense.reciprocal_consistent_points > 0 {
        return Some(format!(
            "Slice 4 dense point fusion ran from reference frame {} with {} registered source views. {} primary candidates passed reciprocal depth consistency, but no fused point survived the spatial-consistency and minimum-observation gates; {} reverse observations were rejected as spatially inconsistent. No dense geometry was invented.",
            reference_frame,
            dense.source_views,
            dense.reciprocal_consistent_points,
            dense.fusion_rejected_observations
        ));
    }

    if dense.reciprocal_checked_points > 0 {
        return Some(format!(
            "Slice 4 coarse depth estimation ran from reference frame {} with {} registered source views. {} primary candidates passed the texture, photometric-error, and ambiguity gates, but reciprocal depth consistency rejected all of them. No dense geometry was invented.",
            reference_frame,
            dense.source_views,
            dense.reciprocal_checked_points
        ));
    }

    Some(format!(
        "Slice 4 coarse depth estimation ran from reference frame {} with {} registered source views, but no sampled pixel passed the texture, photometric-error, and ambiguity gates. No dense geometry was invented.",
        reference_frame, dense.source_views
    ))
}
