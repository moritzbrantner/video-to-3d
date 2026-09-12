from pathlib import Path


def replace_once(path_str, old, new, label):
    path = Path(path_str)
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1))

path_str = "crates/video-to-3d-core/src/mesh.rs"

replace_once(
    path_str,
    "const MAX_MESH_RELATIVE_DEPTH_JUMP: f64 = 0.15;\n",
    "const MAX_GRID_REPROJECTION_OFFSET_STRIDES: f64 = 0.5;\nconst MAX_MESH_RELATIVE_DEPTH_JUMP: f64 = 0.15;\n",
    "mesh footprint constant",
)

replace_once(
    path_str,
    """        let Some((_, _, depth)) = project(reference_camera, position, width, height, focal) else {
            continue;
        };
        let Some(gx) = original_grid_coordinate(grid_site.x, width, dense.grid_border, dense.grid_stride) else {
            continue;
        };
        let Some(gy) = original_grid_coordinate(grid_site.y, height, dense.grid_border, dense.grid_stride) else {
            continue;
        };
""",
    """        let Some((projected_x, projected_y, depth)) =
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
""",
    "mesh footprint admission",
)

path = Path(path_str)
text = path.read_text()
marker = "fn original_grid_coordinate(\n"
if text.count(marker) != 1:
    raise SystemExit("original_grid_coordinate marker mismatch")
helper = """fn within_original_grid_footprint(
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
"""
path.write_text(text.replace(marker, helper, 1))

# The prior regression now proves a fused point can move substantially while remaining
# inside its original half-stride footprint and preserving topology identity.
replace_once(
    path_str,
    "point_at_pixel(5.4, 3.0, 4.0, width, height, focal, 0.8),",
    "point_at_pixel(4.8, 3.0, 4.0, width, height, focal, 0.8),",
    "safe fusion-shift regression",
)

path = Path(path_str)
text = path.read_text()
marker = """    #[test]
    fn does_not_attempt_mesh_without_enough_dense_points() {
"""
if text.count(marker) != 1:
    raise SystemExit("mesh footprint regression insertion marker mismatch")
regression = """    #[test]
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
    fn does_not_attempt_mesh_without_enough_dense_points() {
"""
path.write_text(text.replace(marker, regression, 1))
