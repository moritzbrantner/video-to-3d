from pathlib import Path


def replace_once(path_str, old, new, label):
    path = Path(path_str)
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1))

# Preserve the authoritative reference-grid identity from dense sampling through fusion.
replace_once(
    "crates/video-to-3d-core/src/dense.rs",
    """#[derive(Clone, Debug, Default)]
pub(super) struct DenseAnalysis {
    pub stats: DenseStats,
    pub points: Vec<Point3>,
}
""",
    """#[derive(Clone, Copy, Debug)]
pub(super) struct DenseGridSite {
    pub x: u32,
    pub y: u32,
}

#[derive(Clone, Debug, Default)]
pub(super) struct DenseAnalysis {
    pub stats: DenseStats,
    pub points: Vec<Point3>,
    pub grid_sites: Vec<DenseGridSite>,
}
""",
    "dense grid site contract",
)
replace_once(
    "crates/video-to-3d-core/src/dense.rs",
    """            points: Vec::new(),
        }
""",
    """            points: Vec::new(),
            grid_sites: Vec::new(),
        }
""",
    "dense skipped grid sites",
)
replace_once(
    "crates/video-to-3d-core/src/dense.rs",
    """    let mut points = Vec::new();
    let mut errors = Vec::new();
""",
    """    let mut points = Vec::new();
    let mut grid_sites = Vec::new();
    let mut errors = Vec::new();
""",
    "dense grid site storage",
)
replace_once(
    "crates/video-to-3d-core/src/dense.rs",
    """            errors.push(best.error);
            supports.push(best.support as f64);
""",
    """            grid_sites.push(DenseGridSite { x, y });
            errors.push(best.error);
            supports.push(best.support as f64);
""",
    "dense accepted grid site",
)
replace_once(
    "crates/video-to-3d-core/src/dense.rs",
    """        points,
    }
}
""",
    """        points,
        grid_sites,
    }
}
""",
    "dense result grid sites",
)

# Mesh consumes original grid identity; fused position is used only for geometry/continuity.
replace_once(
    "crates/video-to-3d-core/src/mesh.rs",
    "use crate::{dense::DenseStats, multi_view::RegisteredCamera, Point3};",
    "use crate::{dense::{DenseGridSite, DenseStats}, multi_view::RegisteredCamera, Point3};",
    "mesh dense imports",
)
replace_once(
    "crates/video-to-3d-core/src/mesh.rs",
    "const MAX_GRID_REPROJECTION_OFFSET_STRIDES: f64 = 0.75;\n",
    "",
    "remove inferred grid tolerance",
)
replace_once(
    "crates/video-to-3d-core/src/mesh.rs",
    """    dense_points: &[Point3],
    dense: &DenseStats,
""",
    """    dense_points: &[Point3],
    grid_sites: &[DenseGridSite],
    dense: &DenseStats,
""",
    "mesh grid sites argument",
)
replace_once(
    "crates/video-to-3d-core/src/mesh.rs",
    """    if dense_points.len() < 3 {
        return MeshAnalysis::skipped(
            dense.reference_frame,
            "fewer than three accepted fused dense points are available",
        );
    }
""",
    """    if dense_points.len() < 3 {
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
""",
    "mesh aligned grid evidence",
)
replace_once(
    "crates/video-to-3d-core/src/mesh.rs",
    """    for (point_index, point) in dense_points.iter().enumerate() {
        let position = Vector3::new(point.x as f64, point.y as f64, point.z as f64);
        let Some((x, y, depth)) = project(reference_camera, position, width, height, focal) else {
            continue;
        };
        let Some(gx) = nearest_grid_coordinate(x, width, border, stride) else {
            continue;
        };
        let Some(gy) = nearest_grid_coordinate(y, height, border, stride) else {
            continue;
        };
""",
    """    for (point_index, (point, grid_site)) in dense_points.iter().zip(grid_sites).enumerate() {
        let position = Vector3::new(point.x as f64, point.y as f64, point.z as f64);
        let Some((_, _, depth)) = project(reference_camera, position, width, height, focal) else {
            continue;
        };
        let Some(gx) = original_grid_coordinate(grid_site.x, width, dense.grid_border, dense.grid_stride) else {
            continue;
        };
        let Some(gy) = original_grid_coordinate(grid_site.y, height, dense.grid_border, dense.grid_stride) else {
            continue;
        };
""",
    "mesh original grid mapping",
)
path = Path("crates/video-to-3d-core/src/mesh.rs")
text = path.read_text()
start = text.index("fn nearest_grid_coordinate(")
end = text.index("\nfn cell_triangles(", start)
replacement = """fn original_grid_coordinate(
    value: u32,
    limit: u32,
    border: u32,
    stride: usize,
) -> Option<i32> {
    if stride == 0 || value < border || value >= limit.saturating_sub(border) {
        return None;
    }
    let offset = (value - border) as usize;
    if offset % stride != 0 {
        return None;
    }
    i32::try_from(offset / stride).ok()
}
"""
path.write_text(text[:start] + replacement + text[end:])

# Wire aligned evidence through reconstruction.
replace_once(
    "crates/video-to-3d-core/src/reconstruction.rs",
    """        &dense_analysis.points,
        &dense_analysis.stats,
""",
    """        &dense_analysis.points,
        &dense_analysis.grid_sites,
        &dense_analysis.stats,
""",
    "mesh aligned evidence call",
)
replace_once(
    "crates/video-to-3d-core/src/reconstruction.rs",
    "Fused points remain separate from the sparse map; meshing, general multi-reference depth aggregation, and metric scale are not claimed yet.",
    "Fused points remain separate from the sparse map; general multi-reference depth aggregation and metric scale are not claimed yet.",
    "dense warning mesh wording",
)
replace_once(
    "crates/video-to-3d-core/src/reconstruction.rs",
    """        assert!(!warning.contains("multi-view depth consistency"));
""",
    """        assert!(!warning.contains("multi-view depth consistency"));
        assert!(!warning.contains("meshing"));
""",
    "dense warning regression assertion",
)

# Update existing mesh tests to provide original sites.
path = Path("crates/video-to-3d-core/src/mesh.rs")
text = path.read_text()
marker = """    fn dense_stats() -> DenseStats {
"""
if text.count(marker) != 1:
    raise SystemExit("mesh test helper marker mismatch")
helper = """    fn grid_sites() -> Vec<DenseGridSite> {
        vec![
            DenseGridSite { x: 3, y: 3 },
            DenseGridSite { x: 7, y: 3 },
            DenseGridSite { x: 3, y: 7 },
            DenseGridSite { x: 7, y: 7 },
        ]
    }

    fn dense_stats() -> DenseStats {
"""
text = text.replace(marker, helper, 1)
text = text.replace(
    """            &points,
            &dense_stats(),
""",
    """            &points,
            &grid_sites(),
            &dense_stats(),
""",
)
# Insert regression before insufficient-evidence test.
marker = """    #[test]
    fn does_not_attempt_mesh_without_enough_dense_points() {
"""
if text.count(marker) != 1:
    raise SystemExit("mesh regression insertion marker mismatch")
regression = """    #[test]
    fn preserves_original_grid_site_after_fusion_shifts_reprojection() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let points = vec![
            point_at_pixel(5.4, 3.0, 4.0, width, height, focal, 0.8),
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
    fn does_not_attempt_mesh_without_enough_dense_points() {
"""
path.write_text(text.replace(marker, regression, 1))
