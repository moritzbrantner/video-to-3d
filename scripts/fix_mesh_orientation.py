from pathlib import Path


def replace_once(path_str, old, new, label):
    path = Path(path_str)
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1))

mesh = "crates/video-to-3d-core/src/mesh.rs"

replace_once(
    mesh,
    "const MIN_MESH_AREA_FOOTPRINT_RATIO: f64 = 0.05;\n",
    "const MIN_MESH_AREA_FOOTPRINT_RATIO: f64 = 0.05;\nconst MIN_MESH_PROJECTED_AREA_FOOTPRINT_RATIO: f64 = 0.05;\n",
    "projected area threshold",
)
replace_once(
    mesh,
    """struct GridVertex {
    point_index: usize,
    position: Vector3<f64>,
    reference_depth: f64,
    confidence: f32,
}
""",
    """struct GridVertex {
    point_index: usize,
    position: Vector3<f64>,
    projected_x: f64,
    projected_y: f64,
    reference_depth: f64,
    confidence: f32,
}
""",
    "projected vertex coordinates",
)
replace_once(
    mesh,
    """        let vertex = GridVertex {
            point_index,
            position,
            reference_depth: depth,
            confidence: point.confidence,
        };
""",
    """        let vertex = GridVertex {
            point_index,
            position,
            projected_x,
            projected_y,
            reference_depth: depth,
            confidence: point.confidence,
        };
""",
    "store projected vertex coordinates",
)
replace_once(
    mesh,
    """    let mean_depth = depths.iter().sum::<f64>() / 3.0;
    let local_step = mean_depth * stride / focal;
""",
    """    let signed_projected_double_area =
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
""",
    "signed projected orientation gate",
)

path = Path(mesh)
text = path.read_text()
marker = """    #[test]
    fn does_not_attempt_mesh_without_enough_dense_points() {
"""
if text.count(marker) != 1:
    raise SystemExit("orientation regression insertion marker mismatch")
regression = """    #[test]
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
"""
path.write_text(text.replace(marker, regression, 1))

replace_once(
    "crates/video-to-3d-core/src/reconstruction.rs",
    "It rejected {} triangles at depth/spatial discontinuities and {} degenerate triangles.",
    "It rejected {} triangles at depth/spatial discontinuities and {} degenerate or orientation-flipped triangles.",
    "mesh warning orientation wording",
)
replace_once(
    "apps/web/src/SceneCanvas.tsx",
    "rejected ${reconstruction.mesh.rejected_discontinuities} discontinuity bridges and ${reconstruction.mesh.rejected_degenerate} degenerate candidates",
    "rejected ${reconstruction.mesh.rejected_discontinuities} discontinuity bridges and ${reconstruction.mesh.rejected_degenerate} degenerate/orientation-flipped candidates",
    "viewer orientation diagnostic",
)
