use crate::{
    dense::{DenseReferencePatchStats, DenseStats},
    multi_view::RegisteredCamera,
    Point3,
};
use serde::Serialize;

mod legacy {
    include!("mesh_legacy.rs");
}

pub use legacy::MeshTriangle;

#[derive(Clone, Debug, Default, Serialize)]
pub struct MeshReferencePatchStats {
    pub reference_frame: usize,
    pub point_count: usize,
    pub attempted: bool,
    pub skip_reason: Option<String>,
    pub grid_vertices: usize,
    pub rejected_grid_vertices: usize,
    pub candidate_cells: usize,
    pub candidate_triangles: usize,
    pub accepted_triangles: usize,
    pub rejected_discontinuities: usize,
    pub rejected_degenerate: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MeshStats {
    pub attempted: bool,
    pub skip_reason: Option<String>,
    pub reference_frame: Option<usize>,
    pub grid_vertices: usize,
    pub rejected_grid_vertices: usize,
    pub candidate_cells: usize,
    pub candidate_triangles: usize,
    pub accepted_triangles: usize,
    pub rejected_discontinuities: usize,
    pub rejected_degenerate: usize,
    pub reference_frames: Vec<usize>,
    pub reference_patches: Vec<MeshReferencePatchStats>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct MeshAnalysis {
    pub stats: MeshStats,
    pub triangles: Vec<MeshTriangle>,
}

pub(super) fn reconstruct_dense_mesh(
    dense_points: &[Point3],
    grid_sites: &[crate::dense::DenseGridSite],
    dense: &DenseStats,
    cameras: &[RegisteredCamera],
    width: u32,
    height: u32,
    focal: f64,
) -> MeshAnalysis {
    if dense.reference_patches.is_empty() {
        return from_legacy(legacy::reconstruct_dense_mesh(
            dense_points,
            grid_sites,
            dense,
            cameras,
            width,
            height,
            focal,
        ));
    }

    let mut triangles = Vec::new();
    let mut patch_stats = Vec::with_capacity(dense.reference_patches.len());
    let mut attempted = false;
    let mut grid_vertices = 0usize;
    let mut rejected_grid_vertices = 0usize;
    let mut candidate_cells = 0usize;
    let mut candidate_triangles = 0usize;
    let mut rejected_discontinuities = 0usize;
    let mut rejected_degenerate = 0usize;

    for patch in &dense.reference_patches {
        let Some(global_indices) = patch_point_indices(patch, dense_points.len(), grid_sites.len())
        else {
            patch_stats.push(MeshReferencePatchStats {
                reference_frame: patch.reference_frame,
                point_count: patch.accepted_points,
                skip_reason: Some("dense patch ranges exceed the accepted point evidence".into()),
                ..MeshReferencePatchStats::default()
            });
            continue;
        };

        let local_points = global_indices
            .iter()
            .map(|index| dense_points[*index])
            .collect::<Vec<_>>();
        let local_sites = global_indices
            .iter()
            .map(|index| grid_sites[*index])
            .collect::<Vec<_>>();
        let mut patch_dense = dense.clone();
        patch_dense.attempted = true;
        patch_dense.skip_reason = None;
        patch_dense.reference_frame = Some(patch.reference_frame);
        patch_dense.source_views = patch.source_frames.len();
        patch_dense.source_frames = patch.source_frames.clone();
        patch_dense.accepted_points = local_points.len();
        patch_dense.surface_completed_points = patch.completed_points;
        patch_dense.grid_stride = patch.grid_stride;
        patch_dense.grid_border = patch.grid_border;
        patch_dense.search_min_depth = patch.search_min_depth;
        patch_dense.search_max_depth = patch.search_max_depth;
        patch_dense.reference_frames = vec![patch.reference_frame];
        patch_dense.reference_patches.clear();

        let result = legacy::reconstruct_dense_mesh(
            &local_points,
            &local_sites,
            &patch_dense,
            cameras,
            width,
            height,
            focal,
        );
        attempted |= result.stats.attempted;
        grid_vertices += result.stats.grid_vertices;
        rejected_grid_vertices += result.stats.rejected_grid_vertices;
        candidate_cells += result.stats.candidate_cells;
        candidate_triangles += result.stats.candidate_triangles;
        rejected_discontinuities += result.stats.rejected_discontinuities;
        rejected_degenerate += result.stats.rejected_degenerate;

        for triangle in &result.triangles {
            let (Some(&a), Some(&b), Some(&c)) = (
                global_indices.get(triangle.a),
                global_indices.get(triangle.b),
                global_indices.get(triangle.c),
            ) else {
                continue;
            };
            triangles.push(MeshTriangle {
                a,
                b,
                c,
                confidence: triangle.confidence,
            });
        }

        patch_stats.push(MeshReferencePatchStats {
            reference_frame: patch.reference_frame,
            point_count: local_points.len(),
            attempted: result.stats.attempted,
            skip_reason: result.stats.skip_reason,
            grid_vertices: result.stats.grid_vertices,
            rejected_grid_vertices: result.stats.rejected_grid_vertices,
            candidate_cells: result.stats.candidate_cells,
            candidate_triangles: result.stats.candidate_triangles,
            accepted_triangles: result.triangles.len(),
            rejected_discontinuities: result.stats.rejected_discontinuities,
            rejected_degenerate: result.stats.rejected_degenerate,
        });
    }

    let reference_frames = patch_stats
        .iter()
        .map(|patch| patch.reference_frame)
        .collect::<Vec<_>>();
    let skip_reason = if triangles.is_empty() {
        Some("no reference patch produced an accepted surface triangle".into())
    } else {
        None
    };

    MeshAnalysis {
        stats: MeshStats {
            attempted,
            skip_reason,
            reference_frame: reference_frames.first().copied(),
            grid_vertices,
            rejected_grid_vertices,
            candidate_cells,
            candidate_triangles,
            accepted_triangles: triangles.len(),
            rejected_discontinuities,
            rejected_degenerate,
            reference_frames,
            reference_patches: patch_stats,
        },
        triangles,
    }
}

fn patch_point_indices(
    patch: &DenseReferencePatchStats,
    point_count: usize,
    grid_site_count: usize,
) -> Option<Vec<usize>> {
    if point_count != grid_site_count {
        return None;
    }
    let primary_end = patch.primary_start.checked_add(patch.primary_points)?;
    let completion_end = patch
        .completion_start
        .checked_add(patch.completed_points)?;
    if primary_end > point_count || completion_end > point_count {
        return None;
    }

    let mut indices = Vec::with_capacity(patch.primary_points + patch.completed_points);
    indices.extend(patch.primary_start..primary_end);
    indices.extend(patch.completion_start..completion_end);
    Some(indices)
}

fn from_legacy(result: legacy::MeshAnalysis) -> MeshAnalysis {
    let stats = result.stats;
    let reference_frames = stats.reference_frame.into_iter().collect::<Vec<_>>();
    let reference_patches = stats
        .reference_frame
        .map(|reference_frame| {
            vec![MeshReferencePatchStats {
                reference_frame,
                point_count: stats.grid_vertices + stats.rejected_grid_vertices,
                attempted: stats.attempted,
                skip_reason: stats.skip_reason.clone(),
                grid_vertices: stats.grid_vertices,
                rejected_grid_vertices: stats.rejected_grid_vertices,
                candidate_cells: stats.candidate_cells,
                candidate_triangles: stats.candidate_triangles,
                accepted_triangles: stats.accepted_triangles,
                rejected_discontinuities: stats.rejected_discontinuities,
                rejected_degenerate: stats.rejected_degenerate,
            }]
        })
        .unwrap_or_default();

    MeshAnalysis {
        stats: MeshStats {
            attempted: stats.attempted,
            skip_reason: stats.skip_reason,
            reference_frame: stats.reference_frame,
            grid_vertices: stats.grid_vertices,
            rejected_grid_vertices: stats.rejected_grid_vertices,
            candidate_cells: stats.candidate_cells,
            candidate_triangles: stats.candidate_triangles,
            accepted_triangles: stats.accepted_triangles,
            rejected_discontinuities: stats.rejected_discontinuities,
            rejected_degenerate: stats.rejected_degenerate,
            reference_frames,
            reference_patches,
        },
        triangles: result.triangles,
    }
}

#[cfg(test)]
mod multi_reference_tests {
    use super::*;
    use crate::dense::DenseReferencePatchStats;
    use nalgebra::Matrix3;

    fn camera(frame_index: usize) -> RegisteredCamera {
        RegisteredCamera {
            frame_index,
            rotation: Matrix3::identity(),
            translation: nalgebra::Vector3::zeros(),
        }
    }

    fn point_at_pixel(x: f64, y: f64, width: u32, height: u32, focal: f64) -> Point3 {
        let depth = 4.0;
        Point3 {
            x: ((x - width as f64 * 0.5) / focal * depth) as f32,
            y: ((y - height as f64 * 0.5) / focal * depth) as f32,
            z: depth as f32,
            confidence: 0.8,
            r: 120,
            g: 140,
            b: 160,
        }
    }

    #[test]
    fn combines_reference_meshes_using_global_dense_point_indices() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let sites = [
            crate::dense::DenseGridSite { x: 3, y: 3 },
            crate::dense::DenseGridSite { x: 7, y: 3 },
            crate::dense::DenseGridSite { x: 3, y: 7 },
            crate::dense::DenseGridSite { x: 7, y: 7 },
        ];
        let mut points = sites
            .iter()
            .map(|site| point_at_pixel(site.x as f64, site.y as f64, width, height, focal))
            .collect::<Vec<_>>();
        let first_patch_points = points.clone();
        points.extend(first_patch_points);
        let mut grid_sites = sites.to_vec();
        grid_sites.extend(sites);
        let dense = DenseStats {
            attempted: true,
            reference_frame: Some(0),
            accepted_points: points.len(),
            grid_stride: 4,
            grid_border: 3,
            reference_frames: vec![0, 1],
            reference_patches: vec![
                DenseReferencePatchStats {
                    reference_frame: 0,
                    primary_start: 0,
                    primary_points: 4,
                    completion_start: 4,
                    grid_stride: 4,
                    grid_border: 3,
                    accepted_points: 4,
                    ..DenseReferencePatchStats::default()
                },
                DenseReferencePatchStats {
                    reference_frame: 1,
                    primary_start: 4,
                    primary_points: 4,
                    completion_start: 8,
                    grid_stride: 4,
                    grid_border: 3,
                    accepted_points: 4,
                    ..DenseReferencePatchStats::default()
                },
            ],
            ..DenseStats::default()
        };

        let result = reconstruct_dense_mesh(
            &points,
            &grid_sites,
            &dense,
            &[camera(0), camera(1)],
            width,
            height,
            focal,
        );

        assert!(result.stats.attempted);
        assert_eq!(result.stats.reference_frames, vec![0, 1]);
        assert_eq!(result.stats.reference_patches.len(), 2);
        assert_eq!(result.stats.accepted_triangles, 4);
        assert_eq!(result.triangles.len(), 4);
        assert!(result
            .triangles
            .iter()
            .any(|triangle| triangle.a >= 4 && triangle.b >= 4 && triangle.c >= 4));
    }
}
