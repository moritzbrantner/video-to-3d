use crate::{dense::{DenseGridSite, DenseStats}, multi_view::RegisteredCamera, Point3};
use nalgebra::Vector3;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

const MAX_GRID_REPROJECTION_OFFSET_STRIDES: f64 = 0.5;
const MAX_MESH_RELATIVE_DEPTH_JUMP: f64 = 0.15;
const MAX_BRIDGE_RELATIVE_DEPTH_JUMP: f64 = 0.08;
const MAX_MESH_EDGE_FOOTPRINT_MULTIPLIER: f64 = 3.0;
const MIN_MESH_AREA_FOOTPRINT_RATIO: f64 = 0.05;
const MIN_MESH_PROJECTED_AREA_FOOTPRINT_RATIO: f64 = 0.05;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct MeshTriangle {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub confidence: f32,
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
}

#[derive(Clone, Debug, Default)]
pub(super) struct MeshAnalysis {
    pub stats: MeshStats,
    pub triangles: Vec<MeshTriangle>,
}

impl MeshAnalysis {
    fn skipped(reference_frame: Option<usize>, reason: impl Into<String>) -> Self {
        Self {
            stats: MeshStats {
                reference_frame,
                skip_reason: Some(reason.into()),
                ..MeshStats::default()
            },
            triangles: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GridVertex {
    point_index: usize,
    position: Vector3<f64>,
    projected_x: f64,
    projected_y: f64,
    reference_depth: f64,
    confidence: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TriangleRejection {
    Discontinuity,
    Degenerate,
}

#[derive(Default)]
struct MeshBuildState {
    triangle_keys: BTreeSet<(usize, usize, usize)>,
    triangles: Vec<MeshTriangle>,
    candidate_triangles: usize,
    rejected_discontinuities: usize,
    rejected_degenerate: usize,
}

impl MeshBuildState {
    fn evaluate_candidate(
        &mut self,
        candidate: [GridVertex; 3],
        stride: f64,
        focal: f64,
        max_relative_depth_jump: f64,
    ) {
        let key = triangle_key(candidate);
        if !self.triangle_keys.insert(key) {
            return;
        }
        self.candidate_triangles += 1;
        match accepted_triangle(
            candidate[0],
            candidate[1],
            candidate[2],
            stride,
            focal,
            max_relative_depth_jump,
        ) {
            Ok(triangle) => self.triangles.push(triangle),
            Err(TriangleRejection::Discontinuity) => self.rejected_discontinuities += 1,
            Err(TriangleRejection::Degenerate) => self.rejected_degenerate += 1,
        }
    }
}

pub(super) fn reconstruct_dense_mesh(
    dense_points: &[Point3],
    grid_sites: &[DenseGridSite],
    dense: &DenseStats,
    cameras: &[RegisteredCamera],
    width: u32,
    height: u32,
    focal: f64,
) -> MeshAnalysis {
    if !dense.attempted {
        return MeshAnalysis::skipped(
            dense.reference_frame,
            "dense reconstruction was not attempted",
        );
    }
    if dense_points.len() < 3 {
        return MeshAnalysis::skipped(
            dense.reference_frame,
            "fewer than three accepted fused dense points are available",
        );
    }
    if dense_points.len() != grid_sites.len() {
        return MeshAnalysis::skipped(
            dense.reference_frame,
            "dense point and reference-grid evidence counts disagree",
        );
    }
    if dense.grid_stride == 0 {
        return MeshAnalysis::skipped(
            dense.reference_frame,
            "dense reconstruction did not expose a valid reference-grid stride",
        );
    }
    if width == 0 || height == 0 || !focal.is_finite() || focal <= 0.0 {
        return MeshAnalysis::skipped(
            dense.reference_frame,
            "the reference camera does not provide finite positive projection parameters",
        );
    }

    let Some(reference_frame) = dense.reference_frame else {
        return MeshAnalysis::skipped(None, "dense reconstruction did not select a reference frame");
    };
    let Some(reference_camera) = cameras
        .iter()
        .find(|camera| camera.frame_index == reference_frame)
    else {
        return MeshAnalysis::skipped(
            Some(reference_frame),
            "the dense reference frame has no accepted registered camera",
        );
    };

    let stride = dense.grid_stride as f64;
    let mut grid = BTreeMap::<(i32, i32), GridVertex>::new();

    for (point_index, (point, grid_site)) in dense_points.iter().zip(grid_sites).enumerate() {
        let position = Vector3::new(point.x as f64, point.y as f64, point.z as f64);
        let Some((projected_x, projected_y, depth)) =
            project(reference_camera, position, width, height, focal)
        else {
            continue;
        };
        if !within_original_grid_footprint(
            projected_x,
            projected_y,
            *grid_site,
            dense.grid_stride,
        ) {
            continue;
        }
        let Some(gx) = original_grid_coordinate(grid_site.x, width, dense.grid_border, dense.grid_stride) else {
            continue;
        };
        let Some(gy) = original_grid_coordinate(grid_site.y, height, dense.grid_border, dense.grid_stride) else {
            continue;
        };
        let vertex = GridVertex {
            point_index,
            position,
            projected_x,
            projected_y,
            reference_depth: depth,
            confidence: point.confidence,
        };
        grid.entry((gx, gy))
            .and_modify(|current| {
                if better_grid_vertex(vertex, *current) {
                    *current = vertex;
                }
            })
            .or_insert(vertex);
    }

    if grid.len() < 3 {
        return MeshAnalysis {
            stats: MeshStats {
                attempted: true,
                skip_reason: Some(
                    "fewer than three fused dense points align with the reference sampling grid"
                        .into(),
                ),
                reference_frame: Some(reference_frame),
                grid_vertices: grid.len(),
                rejected_grid_vertices: dense_points.len().saturating_sub(grid.len()),
                ..MeshStats::default()
            },
            triangles: Vec::new(),
        };
    }

    let mut cell_origins = BTreeSet::new();
    for &(gx, gy) in grid.keys() {
        for dx in [-1, 0] {
            for dy in [-1, 0] {
                cell_origins.insert((gx + dx, gy + dy));
            }
        }
    }

    let mut build = MeshBuildState::default();
    let mut candidate_cells = 0usize;

    for (gx, gy) in cell_origins {
        let corners = [
            grid.get(&(gx, gy)).copied(),
            grid.get(&(gx + 1, gy)).copied(),
            grid.get(&(gx, gy + 1)).copied(),
            grid.get(&(gx + 1, gy + 1)).copied(),
        ];
        let present = corners.iter().flatten().count();
        if present < 3 {
            continue;
        }
        candidate_cells += 1;

        for candidate in cell_triangles(corners) {
            build.evaluate_candidate(candidate, stride, focal, MAX_MESH_RELATIVE_DEPTH_JUMP);
        }
    }

    let minimum_x = grid.keys().map(|(x, _)| *x).min().expect("non-empty grid");
    let maximum_x = grid.keys().map(|(x, _)| *x).max().expect("non-empty grid");
    let minimum_y = grid.keys().map(|(_, y)| *y).min().expect("non-empty grid");
    let maximum_y = grid.keys().map(|(_, y)| *y).max().expect("non-empty grid");

    if maximum_x - minimum_x >= 2 {
        for gx in minimum_x..=maximum_x - 2 {
            for gy in minimum_y..=maximum_y {
                if let Some(candidate) = horizontal_gap_bridge(&grid, gx, gy) {
                    candidate_cells += 1;
                    build.evaluate_candidate(
                        candidate,
                        stride,
                        focal,
                        MAX_BRIDGE_RELATIVE_DEPTH_JUMP,
                    );
                }
            }
        }
    }
    if maximum_y - minimum_y >= 2 {
        for gx in minimum_x..=maximum_x {
            for gy in minimum_y..=maximum_y - 2 {
                if let Some(candidate) = vertical_gap_bridge(&grid, gx, gy) {
                    candidate_cells += 1;
                    build.evaluate_candidate(
                        candidate,
                        stride,
                        focal,
                        MAX_BRIDGE_RELATIVE_DEPTH_JUMP,
                    );
                }
            }
        }
    }

    MeshAnalysis {
        stats: MeshStats {
            attempted: true,
            skip_reason: None,
            reference_frame: Some(reference_frame),
            grid_vertices: grid.len(),
            rejected_grid_vertices: dense_points.len().saturating_sub(grid.len()),
            candidate_cells,
            candidate_triangles: build.candidate_triangles,
            accepted_triangles: build.triangles.len(),
            rejected_discontinuities: build.rejected_discontinuities,
            rejected_degenerate: build.rejected_degenerate,
        },
        triangles: build.triangles,
    }
}

fn triangle_key(vertices: [GridVertex; 3]) -> (usize, usize, usize) {
    let mut indices = [
        vertices[0].point_index,
        vertices[1].point_index,
        vertices[2].point_index,
    ];
    indices.sort_unstable();
    (indices[0], indices[1], indices[2])
}

fn horizontal_gap_bridge(
    grid: &BTreeMap<(i32, i32), GridVertex>,
    gx: i32,
    gy: i32,
) -> Option<[GridVertex; 3]> {
    let top_left = grid.get(&(gx, gy)).copied();
    let top_right = grid.get(&(gx + 2, gy)).copied();
    let bottom_left = grid.get(&(gx, gy + 1)).copied();
    let bottom_right = grid.get(&(gx + 2, gy + 1)).copied();
    let top_middle = grid.get(&(gx + 1, gy)).copied();
    let bottom_middle = grid.get(&(gx + 1, gy + 1)).copied();

    match (
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        top_middle,
        bottom_middle,
    ) {
        (Some(tl), Some(tr), _, _, None, Some(middle)) => Some([tl, tr, middle]),
        (_, _, Some(bl), Some(br), Some(middle), None) => Some([bl, middle, br]),
        _ => None,
    }
}

fn vertical_gap_bridge(
    grid: &BTreeMap<(i32, i32), GridVertex>,
    gx: i32,
    gy: i32,
) -> Option<[GridVertex; 3]> {
    let top_left = grid.get(&(gx, gy)).copied();
    let top_right = grid.get(&(gx + 1, gy)).copied();
    let bottom_left = grid.get(&(gx, gy + 2)).copied();
    let bottom_right = grid.get(&(gx + 1, gy + 2)).copied();
    let left_middle = grid.get(&(gx, gy + 1)).copied();
    let right_middle = grid.get(&(gx + 1, gy + 1)).copied();

    match (
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        left_middle,
        right_middle,
    ) {
        (Some(tl), _, Some(bl), _, None, Some(middle)) => Some([tl, middle, bl]),
        (_, Some(tr), _, Some(br), Some(middle), None) => Some([tr, br, middle]),
        _ => None,
    }
}

fn better_grid_vertex(candidate: GridVertex, current: GridVertex) -> bool {
    candidate.confidence > current.confidence
        || (candidate.confidence == current.confidence && candidate.point_index < current.point_index)
}

fn within_original_grid_footprint(
    projected_x: f64,
    projected_y: f64,
    grid_site: DenseGridSite,
    stride: usize,
) -> bool {
    if stride == 0 || !projected_x.is_finite() || !projected_y.is_finite() {
        return false;
    }
    let limit = stride as f64 * MAX_GRID_REPROJECTION_OFFSET_STRIDES;
    (projected_x - grid_site.x as f64).abs() < limit
        && (projected_y - grid_site.y as f64).abs() < limit
}

fn original_grid_coordinate(
    value: u32,
    limit: u32,
    border: u32,
    stride: usize,
) -> Option<i32> {
    if stride == 0 || value < border || value >= limit.saturating_sub(border) {
        return None;
    }
    let offset = (value - border) as usize;
    if !offset.is_multiple_of(stride) {
        return None;
    }
    i32::try_from(offset / stride).ok()
}

fn cell_triangles(corners: [Option<GridVertex>; 4]) -> Vec<[GridVertex; 3]> {
    let [top_left, top_right, bottom_left, bottom_right] = corners;
    match (top_left, top_right, bottom_left, bottom_right) {
        (Some(tl), Some(tr), Some(bl), Some(br)) => {
            let tl_br = (tl.position - br.position).norm_squared();
            let tr_bl = (tr.position - bl.position).norm_squared();
            if tl_br <= tr_bl {
                vec![[tl, tr, br], [tl, br, bl]]
            } else {
                vec![[tl, tr, bl], [tr, br, bl]]
            }
        }
        (Some(tl), Some(tr), Some(bl), None) => vec![[tl, tr, bl]],
        (Some(tl), Some(tr), None, Some(br)) => vec![[tl, tr, br]],
        (Some(tl), None, Some(bl), Some(br)) => vec![[tl, br, bl]],
        (None, Some(tr), Some(bl), Some(br)) => vec![[tr, br, bl]],
        _ => Vec::new(),
    }
}

fn accepted_triangle(
    a: GridVertex,
    b: GridVertex,
    c: GridVertex,
    stride: f64,
    focal: f64,
    max_relative_depth_jump: f64,
) -> Result<MeshTriangle, TriangleRejection> {
    let depths = [a.reference_depth, b.reference_depth, c.reference_depth];
    if depths
        .iter()
        .any(|depth| !depth.is_finite() || *depth <= 0.0)
    {
        return Err(TriangleRejection::Discontinuity);
    }
    let minimum_depth = depths.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum_depth = depths
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if (maximum_depth - minimum_depth) / maximum_depth > max_relative_depth_jump {
        return Err(TriangleRejection::Discontinuity);
    }

    let signed_projected_double_area =
        (b.projected_x - a.projected_x) * (c.projected_y - a.projected_y)
            - (b.projected_y - a.projected_y) * (c.projected_x - a.projected_x);
    let minimum_projected_double_area =
        stride * stride * MIN_MESH_PROJECTED_AREA_FOOTPRINT_RATIO;
    if !signed_projected_double_area.is_finite()
        || signed_projected_double_area <= minimum_projected_double_area
    {
        return Err(TriangleRejection::Degenerate);
    }

    let mean_depth = depths.iter().sum::<f64>() / 3.0;
    let local_step = mean_depth * stride / focal;
    let maximum_edge = local_step
        * std::f64::consts::SQRT_2
        * MAX_MESH_EDGE_FOOTPRINT_MULTIPLIER;
    let edges = [
        (a.position - b.position).norm(),
        (b.position - c.position).norm(),
        (c.position - a.position).norm(),
    ];
    if edges
        .iter()
        .any(|edge| !edge.is_finite() || *edge > maximum_edge)
    {
        return Err(TriangleRejection::Discontinuity);
    }

    let double_area = (b.position - a.position)
        .cross(&(c.position - a.position))
        .norm();
    let minimum_double_area = local_step * local_step * MIN_MESH_AREA_FOOTPRINT_RATIO;
    if !double_area.is_finite() || double_area <= minimum_double_area {
        return Err(TriangleRejection::Degenerate);
    }

    Ok(MeshTriangle {
        a: a.point_index,
        b: b.point_index,
        c: c.point_index,
        confidence: a.confidence.min(b.confidence).min(c.confidence).clamp(0.0, 1.0),
    })
}

fn project(
    camera: &RegisteredCamera,
    point: Vector3<f64>,
    width: u32,
    height: u32,
    focal: f64,
) -> Option<(f64, f64, f64)> {
    let camera_point = camera.rotation * point + camera.translation;
    if !camera_point.iter().all(|value| value.is_finite()) || camera_point.z <= 1.0e-4 {
        return None;
    }
    Some((
        focal * camera_point.x / camera_point.z + width as f64 * 0.5,
        focal * camera_point.y / camera_point.z + height as f64 * 0.5,
        camera_point.z,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    fn camera() -> RegisteredCamera {
        RegisteredCamera {
            frame_index: 0,
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        }
    }

    fn point_at_pixel(
        x: f64,
        y: f64,
        depth: f64,
        width: u32,
        height: u32,
        focal: f64,
        confidence: f32,
    ) -> Point3 {
        Point3 {
            x: ((x - width as f64 * 0.5) / focal * depth) as f32,
            y: ((y - height as f64 * 0.5) / focal * depth) as f32,
            z: depth as f32,
            confidence,
            r: 120,
            g: 140,
            b: 160,
        }
    }

    fn grid_sites() -> Vec<DenseGridSite> {
        vec![
            DenseGridSite { x: 3, y: 3 },
            DenseGridSite { x: 7, y: 3 },
            DenseGridSite { x: 3, y: 7 },
            DenseGridSite { x: 7, y: 7 },
        ]
    }

    fn dense_stats() -> DenseStats {
        DenseStats {
            attempted: true,
            reference_frame: Some(0),
            grid_stride: 4,
            grid_border: 3,
            ..DenseStats::default()
        }
    }

    #[test]
    fn preserves_admission_stats_when_too_few_grid_vertices_survive() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let points = vec![
            point_at_pixel(3.0, 3.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(7.0, 3.0, 4.0, width, height, focal, 0.7),
            point_at_pixel(6.0, 7.0, 4.0, width, height, focal, 0.9),
        ];
        let sites = vec![
            DenseGridSite { x: 3, y: 3 },
            DenseGridSite { x: 7, y: 3 },
            DenseGridSite { x: 3, y: 7 },
        ];

        let result = reconstruct_dense_mesh(
            &points,
            &sites,
            &dense_stats(),
            &[camera()],
            width,
            height,
            focal,
        );

        assert!(result.stats.attempted);
        assert!(result.stats.skip_reason.is_some());
        assert_eq!(result.stats.grid_vertices, 2);
        assert_eq!(result.stats.rejected_grid_vertices, 1);
        assert!(result.triangles.is_empty());
    }

    #[test]
    fn triangulates_a_smooth_reference_grid_quad() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let points = vec![
            point_at_pixel(3.0, 3.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(7.0, 3.0, 4.0, width, height, focal, 0.7),
            point_at_pixel(3.0, 7.0, 4.0, width, height, focal, 0.9),
            point_at_pixel(7.0, 7.0, 4.0, width, height, focal, 0.6),
        ];

        let result = reconstruct_dense_mesh(
            &points,
            &grid_sites(),
            &dense_stats(),
            &[camera()],
            width,
            height,
            focal,
        );

        assert!(result.stats.attempted);
        assert_eq!(result.stats.grid_vertices, 4);
        assert_eq!(result.stats.candidate_cells, 1);
        assert_eq!(result.stats.candidate_triangles, 2);
        assert_eq!(result.stats.accepted_triangles, 2);
        assert_eq!(result.stats.rejected_discontinuities, 0);
        assert_eq!(result.stats.rejected_degenerate, 0);
        assert_eq!(result.triangles.len(), 2);
        assert!(result
            .triangles
            .iter()
            .all(|triangle| triangle.a < 4 && triangle.b < 4 && triangle.c < 4));
        assert!(result
            .triangles
            .iter()
            .all(|triangle| triangle.confidence <= 0.8));
    }

    #[test]
    fn bridges_one_missing_grid_sample_on_a_smooth_surface() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let sites = vec![
            DenseGridSite { x: 3, y: 3 },
            DenseGridSite { x: 11, y: 3 },
            DenseGridSite { x: 3, y: 7 },
            DenseGridSite { x: 11, y: 7 },
            DenseGridSite { x: 7, y: 7 },
        ];
        let points = sites
            .iter()
            .map(|site| point_at_pixel(site.x as f64, site.y as f64, 4.0, width, height, focal, 0.8))
            .collect::<Vec<_>>();

        let result = reconstruct_dense_mesh(
            &points,
            &sites,
            &dense_stats(),
            &[camera()],
            width,
            height,
            focal,
        );

        assert!(result.stats.attempted);
        assert_eq!(result.stats.grid_vertices, 5);
        assert_eq!(result.stats.accepted_triangles, 3);
        assert!(result.triangles.iter().any(|triangle| {
            let mut indices = [triangle.a, triangle.b, triangle.c];
            indices.sort_unstable();
            indices == [0, 1, 4]
        }));
    }

    #[test]
    fn does_not_bridge_a_missing_sample_across_a_depth_discontinuity() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let sites = vec![
            DenseGridSite { x: 3, y: 3 },
            DenseGridSite { x: 11, y: 3 },
            DenseGridSite { x: 3, y: 7 },
            DenseGridSite { x: 11, y: 7 },
            DenseGridSite { x: 7, y: 7 },
        ];
        let points = vec![
            point_at_pixel(3.0, 3.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(11.0, 3.0, 5.0, width, height, focal, 0.8),
            point_at_pixel(3.0, 7.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(11.0, 7.0, 5.0, width, height, focal, 0.8),
            point_at_pixel(7.0, 7.0, 4.0, width, height, focal, 0.8),
        ];

        let result = reconstruct_dense_mesh(
            &points,
            &sites,
            &dense_stats(),
            &[camera()],
            width,
            height,
            focal,
        );

        assert!(result.stats.attempted);
        assert!(result.stats.rejected_discontinuities > 0);
        assert!(!result.triangles.iter().any(|triangle| {
            let mut indices = [triangle.a, triangle.b, triangle.c];
            indices.sort_unstable();
            indices == [0, 1, 4]
        }));
    }

    #[test]
    fn rejects_triangles_that_bridge_a_depth_discontinuity() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let points = vec![
            point_at_pixel(3.0, 3.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(7.0, 3.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(3.0, 7.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(7.0, 7.0, 7.0, width, height, focal, 0.8),
        ];

        let result = reconstruct_dense_mesh(
            &points,
            &grid_sites(),
            &dense_stats(),
            &[camera()],
            width,
            height,
            focal,
        );

        assert!(result.stats.attempted);
        assert_eq!(result.stats.candidate_triangles, 2);
        assert!(result.stats.accepted_triangles < result.stats.candidate_triangles);
        assert!(result.stats.rejected_discontinuities > 0);
        assert!(result.stats.accepted_triangles > 0);
    }

    #[test]
    fn preserves_original_grid_site_after_fusion_shifts_reprojection() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let points = vec![
            point_at_pixel(4.8, 3.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(7.0, 3.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(3.0, 7.0, 4.0, width, height, focal, 0.8),
        ];
        let sites = vec![
            DenseGridSite { x: 3, y: 3 },
            DenseGridSite { x: 7, y: 3 },
            DenseGridSite { x: 3, y: 7 },
        ];

        let result = reconstruct_dense_mesh(
            &points,
            &sites,
            &dense_stats(),
            &[camera()],
            width,
            height,
            focal,
        );

        assert!(result.stats.attempted);
        assert_eq!(result.stats.grid_vertices, 3);
        assert_eq!(result.stats.accepted_triangles, 1);
    }

    #[test]
    fn rejects_fused_vertex_that_leaves_original_grid_footprint() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let points = vec![
            point_at_pixel(5.4, 3.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(7.0, 3.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(3.0, 7.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(7.0, 7.0, 4.0, width, height, focal, 0.8),
        ];

        let result = reconstruct_dense_mesh(
            &points,
            &grid_sites(),
            &dense_stats(),
            &[camera()],
            width,
            height,
            focal,
        );

        assert!(result.stats.attempted);
        assert_eq!(result.stats.grid_vertices, 3);
        assert_eq!(result.stats.accepted_triangles, 1);
        assert!(result.triangles.iter().all(|triangle| {
            triangle.a != 0 && triangle.b != 0 && triangle.c != 0
        }));
    }

    #[test]
    fn rejects_triangle_that_flips_reference_projection_winding() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let points = vec![
            point_at_pixel(4.9, 1.1, 4.0, width, height, focal, 0.8),
            point_at_pixel(5.1, 4.9, 4.0, width, height, focal, 0.8),
            point_at_pixel(8.9, 5.1, 4.0, width, height, focal, 0.8),
        ];
        let sites = vec![
            DenseGridSite { x: 3, y: 3 },
            DenseGridSite { x: 7, y: 3 },
            DenseGridSite { x: 7, y: 7 },
        ];

        let result = reconstruct_dense_mesh(
            &points,
            &sites,
            &dense_stats(),
            &[camera()],
            width,
            height,
            focal,
        );

        assert!(result.stats.attempted);
        assert_eq!(result.stats.grid_vertices, 3);
        assert_eq!(result.stats.candidate_triangles, 1);
        assert_eq!(result.stats.accepted_triangles, 0);
        assert_eq!(result.stats.rejected_degenerate, 1);
    }

    #[test]
    fn does_not_attempt_mesh_without_enough_dense_points() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let points = vec![
            point_at_pixel(3.0, 3.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(7.0, 3.0, 4.0, width, height, focal, 0.8),
        ];

        let result = reconstruct_dense_mesh(
            &points,
            &grid_sites(),
            &dense_stats(),
            &[camera()],
            width,
            height,
            focal,
        );

        assert!(!result.stats.attempted);
        assert!(result.triangles.is_empty());
        assert!(result
            .stats
            .skip_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("three accepted fused dense points")));
    }
}
