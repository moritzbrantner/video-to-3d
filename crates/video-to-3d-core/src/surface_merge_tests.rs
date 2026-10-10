use super::*;
use nalgebra::Matrix3;

fn point(x: f32, y: f32, z: f32) -> Point3 {
    Point3 {
        x,
        y,
        z,
        confidence: 1.0,
        r: 40,
        g: 80,
        b: 120,
    }
}

fn triangle(a: usize, b: usize, c: usize) -> MeshTriangle {
    MeshTriangle {
        a,
        b,
        c,
        confidence: 0.9,
    }
}

fn geometry(corners: [(f64, f64, f64); 3]) -> TriangleGeometry {
    TriangleGeometry::new(corners.map(|(x, y, z)| Vector3::new(x, y, z))).unwrap()
}

fn signature(triangles: &[MeshTriangle]) -> Vec<(usize, usize, usize)> {
    triangles.iter().map(|t| (t.a, t.b, t.c)).collect()
}

#[test]
fn union_coverage_handles_a_different_diagonal_and_large_triangles() {
    let left = geometry([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]);
    let right = geometry([(2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]);
    // The whole triangle is inside the square, but neither kept triangle
    // contains its centroid plus all corners. One triangle may be much larger
    // than the spatial cell, which must not hide its interior.
    let inner = geometry([(0.2, 0.2, 0.05), (1.8, 0.3, 0.05), (1.4, 1.7, 0.05)]);
    assert_eq!(
        inner.covered_by([&left, &right].into_iter(), 0.1),
        (true, true)
    );
}

#[test]
fn never_discards_an_unsupported_hole_or_partial_boundary() {
    let left = geometry([(0.0, 0.0, 0.0), (0.9, 0.0, 0.0), (0.0, 1.0, 0.0)]);
    let right = geometry([(1.1, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0)]);
    let spanning = geometry([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.8, 0.0)]);
    assert!(!spanning.covered_by([&left, &right].into_iter(), 0.1).0);
    let accepted = geometry([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]);
    let outside = geometry([(0.7, 0.1, 0.0), (1.4, 0.1, 0.0), (1.1, 0.8, 0.0)]);
    assert!(!outside.covered_by([&accepted].into_iter(), 0.1).0);
}

#[test]
fn normal_and_depth_disagreement_are_not_coverage() {
    let kept = geometry([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]);
    let distant = geometry([(0.1, 0.1, 1.0), (1.0, 0.1, 1.0), (0.1, 1.0, 1.0)]);
    assert_eq!(distant.covered_by([&kept].into_iter(), 0.1), (false, false));
    let orthogonal = geometry([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]);
    assert_eq!(
        orthogonal.covered_by([&kept].into_iter(), 0.1),
        (false, false)
    );
}

#[test]
fn no_accepted_fusion_pair_means_no_dedup_even_for_coincident_sheets() {
    let points = vec![
        point(0.0, 0.0, 0.0),
        point(1.0, 0.0, 0.0),
        point(0.0, 1.0, 0.0),
        point(0.0, 0.0, 0.0),
        point(1.0, 0.0, 0.0),
        point(0.0, 1.0, 0.0),
    ];
    let original = vec![triangle(0, 1, 2), triangle(3, 4, 5)];
    let mut triangles = original.clone();
    let stats = merge_fused_patches(
        &points,
        &mut triangles,
        &[Some(0); 3]
            .into_iter()
            .chain([Some(1); 3])
            .collect::<Vec<_>>(),
        &[],
        &[],
        &[],
    );
    assert_eq!(stats.removed_duplicate_triangles, 0);
    assert_eq!(stats.cross_reference_triangles, 0);
    assert_eq!(stats.patch_components, 2);
    assert_eq!(signature(&triangles), signature(&original));
}

#[test]
fn a_fusion_link_does_not_erase_an_unrelated_island_of_the_same_camera_pair() {
    let points = vec![
        point(0.0, 0.0, 0.0),
        point(1.0, 0.0, 0.0),
        point(0.0, 1.0, 0.0),
        point(10.0, 0.0, 0.0),
        point(11.0, 0.0, 0.0),
        point(10.0, 1.0, 0.0),
        point(0.0, 0.0, 0.0),
        point(1.0, 0.0, 0.0),
        point(0.0, 1.0, 0.0),
        point(10.0, 0.0, 0.0),
        point(11.0, 0.0, 0.0),
        point(10.0, 1.0, 0.0),
    ];
    let mut triangles = vec![
        triangle(0, 1, 2),
        triangle(3, 4, 5),
        triangle(6, 7, 8),
        triangle(9, 10, 11),
    ];
    let membership = [vec![Some(0); 6], vec![Some(1); 6]].concat();

    let stats = merge_fused_patches(&points, &mut triangles, &membership, &[(0, 6)], &[], &[]);

    assert_eq!(stats.fused_pairs, 1);
    assert_eq!(stats.removed_duplicate_triangles, 1);
    assert!(signature(&triangles).contains(&(9, 10, 11)));
    assert!(signature(&triangles).contains(&(3, 4, 5)));
}

/// Two adjacent camera grids: A covers x=0..2, B x=1..3, fused at x=2.
fn adjacent_grids_fixture() -> (Vec<Point3>, Vec<MeshTriangle>, Vec<Option<usize>>) {
    let mut points = Vec::new();
    let mut membership = Vec::new();
    for (start, z, patch) in [(0.0_f32, 0.0_f32, 0), (1.0_f32, 0.06_f32, 1)] {
        for y in 0..=1 {
            for x in 0..=2 {
                let shared = patch == 1 && x == 1;
                points.push(point(
                    start + x as f32,
                    y as f32,
                    if shared { 0.0 } else { z },
                ));
                membership.push(Some(patch));
            }
        }
    }
    let mut triangles = Vec::new();
    for offset in [0, 6] {
        for x in 0..2 {
            let start = offset + x;
            triangles.push(triangle(start, start + 1, start + 3));
            triangles.push(triangle(start + 1, start + 4, start + 3));
        }
    }
    (points, triangles, membership)
}

#[test]
fn a_confirmed_seam_deduplicates_overlap_but_keeps_the_uncovered_extension() {
    // Two adjacent camera grids: A covers x=0..2, B x=1..3.
    // The last column of B joins A at x=2; x=1..2 is the redundant sheet,
    // but x=2..3 is real, uniquely supported surface that must stay.
    let (points, mut triangles, membership) = adjacent_grids_fixture();
    let original_points = points
        .iter()
        .map(|p| (p.x, p.y, p.z, p.r, p.g, p.b))
        .collect::<Vec<_>>();
    let original = triangles.clone();
    let stats = merge_fused_patches(
        &points,
        &mut triangles,
        &membership,
        &[(2, 7), (5, 10)],
        &[],
        &[],
    );
    assert_eq!(stats.fused_pairs, 2);
    assert_eq!(stats.patch_components, 1);
    assert!(stats.cross_reference_triangles > 0);
    assert_eq!(stats.removed_duplicate_triangles, 2);
    assert_eq!(triangles.len(), original.len() - 2);
    assert_eq!(
        points
            .iter()
            .map(|p| (p.x, p.y, p.z, p.r, p.g, p.b))
            .collect::<Vec<_>>(),
        original_points
    );
    assert!(triangles.iter().any(|t| [t.a, t.b, t.c].contains(&8)));
    let first = signature(&triangles);
    let again = merge_fused_patches(
        &points,
        &mut original.clone(),
        &membership,
        &[(2, 7), (5, 10)],
        &[],
        &[],
    );
    assert_eq!(stats, again);
    let mut replay = original;
    merge_fused_patches(
        &points,
        &mut replay,
        &membership,
        &[(2, 7), (5, 10)],
        &[],
        &[],
    );
    assert_eq!(first, signature(&replay));
}

#[test]
fn large_triangles_are_indexed_across_their_interior() {
    let surface = geometry([(0.0, 0.0, 0.0), (60.0, 0.0, 0.0), (0.0, 60.0, 0.0)]);
    // A large triangle is indexed once, at a coarse level, and found by a
    // small query inside it.
    let mut grid = LevelGrid::<3>::new(1.0);
    let (low, high) = surface.aabb(0.0);
    grid.insert(7, low, high);
    assert_eq!(grid.cells.len(), 1);
    assert!(grid.cells.keys().all(|(level, _)| *level > 0));
    let mut found = BTreeSet::new();
    assert!(grid.query([4.0, 4.0, 0.0], [5.0, 5.0, 0.0], MAX_GRID_CELLS, &mut found));
    assert!(found.contains(&7));
    let small = geometry([(4.0, 4.0, 0.0), (5.0, 4.0, 0.0), (4.0, 5.0, 0.0)]);
    assert!(small.covered_by([&surface].into_iter(), 0.1).0);
}

fn identity_camera(x: f64) -> ReferenceCamera {
    ReferenceCamera {
        rotation: Matrix3::identity(),
        translation: Vector3::new(-x, 0.0, 0.0),
    }
}

fn corners(values: [(f64, f64, f64); 3]) -> [Vector3<f64>; 3] {
    values.map(|(x, y, z)| Vector3::new(x, y, z))
}

/// A 2 × 2 square at depth 5 split into two kept triangles.
fn square_footprint(camera: ReferenceCamera) -> Footprint {
    Footprint::new(
        0,
        camera,
        &[
            (
                corners([(-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (1.0, 1.0, 5.0)]),
                0,
                [0, 1, 2],
                101,
            ),
            (
                corners([(-1.0, -1.0, 5.0), (1.0, 1.0, 5.0), (-1.0, 1.0, 5.0)]),
                0,
                [0, 2, 3],
                102,
            ),
        ],
    )
    .unwrap()
}

#[test]
fn image_ownership_tolerates_reference_scale_but_not_another_surface() {
    let footprint = square_footprint(identity_camera(0.0));
    // Inside both kept triangles, 1 % farther: the same surface.
    let same = corners([(-0.5, -0.5, 5.05), (0.6, -0.4, 5.05), (0.1, 0.7, 5.05)]);
    assert!(
        (footprint
            .ownership(&same, 1.0, &|_| true)
            .map(|(owned, _)| owned)
            .unwrap_or_default()
            .image_share
            - 1.0)
            .abs()
            < 1.0e-9
    );
    // A layer 4 % in front (0.2 m before a wall 5 m away) is a separate surface.
    let layer = corners([(-0.5, -0.5, 4.8), (0.6, -0.4, 4.8), (0.1, 0.7, 4.8)]);
    assert_eq!(
        footprint
            .ownership(&layer, 1.0, &|_| true)
            .map(|(owned, _)| owned)
            .unwrap_or_default()
            .image_share,
        0.0
    );
    // Unless the whole patch is 4 % nearer: a relative depth scale, removed
    // before the bound applies.
    let ratios = footprint.depth_ratios(&layer, &|_| true, &|_| true, usize::MAX);
    assert!(!ratios.is_empty());
    assert!(ratios.iter().all(|ratio| (ratio - 0.96).abs() < 1.0e-9));
    assert!(
        (footprint
            .ownership(&layer, 0.96, &|_| true)
            .map(|(owned, _)| owned)
            .unwrap_or_default()
            .image_share
            - 1.0)
            .abs()
            < 1.0e-9
    );
    // 12 % farther: a separate surface, never owned.
    let behind = corners([(-0.5, -0.5, 5.6), (0.6, -0.4, 5.6), (0.1, 0.7, 5.6)]);
    assert_eq!(
        footprint
            .ownership(&behind, 1.0, &|_| true)
            .map(|(owned, _)| owned)
            .unwrap_or_default()
            .image_share,
        0.0
    );
    // Behind the camera the image cannot judge it at all.
    let flipped = corners([(-0.5, -0.5, -5.0), (0.6, -0.4, -5.0), (0.1, 0.7, -5.0)]);
    assert_eq!(
        footprint
            .ownership(&flipped, 1.0, &|_| true)
            .map(|(owned, _)| owned),
        None
    );
}

#[test]
fn image_ownership_measures_the_share_on_the_footprint() {
    let footprint = square_footprint(identity_camera(0.0));
    // Half of this triangle's projected area lies beyond x = 1.
    let seam = corners([(0.0, -0.5, 5.0), (2.0, -0.5, 5.0), (1.0, 0.5, 5.0)]);
    let owned = footprint
        .ownership(&seam, 1.0, &|_| true)
        .map(|(owned, _)| owned)
        .unwrap();
    assert!((owned.image_share - 0.5).abs() < 1.0e-9);
    // Fronto-parallel, so the owned surface area is half of the 3D area too.
    assert!((owned.surface_area - 0.5).abs() < 1.0e-9);
    // Tilted in depth, the image share and the surface share differ; the
    // surface area is measured on the triangle's own plane.
    let tilted = corners([(0.0, -0.5, 4.9), (2.0, -0.5, 5.1), (1.0, 0.5, 5.0)]);
    let owned = footprint
        .ownership(&tilted, 1.0, &|_| true)
        .map(|(owned, _)| owned)
        .unwrap();
    let full = (tilted[1] - tilted[0])
        .cross(&(tilted[2] - tilted[0]))
        .norm()
        * 0.5;
    assert!(owned.image_share > 0.4 && owned.image_share < 0.6);
    assert!(owned.surface_area > 0.0 && owned.surface_area < full);
    let outside = corners([(1.5, -0.5, 5.0), (2.5, -0.5, 5.0), (2.0, 0.5, 5.0)]);
    assert_eq!(
        footprint
            .ownership(&outside, 1.0, &|_| true)
            .map(|(owned, _)| owned)
            .unwrap_or_default()
            .image_share,
        0.0
    );
}

/// Points, membership, triangles, fused link and cameras of
/// [`owned_overlap_is_removed_and_unique_surface_is_kept`].
type OwnedOverlapFixture = (
    Vec<Point3>,
    Vec<Option<usize>>,
    Vec<MeshTriangle>,
    [(usize, usize); 1],
    [Option<ReferenceCamera>; 2],
);

fn owned_overlap_fixture() -> OwnedOverlapFixture {
    // Patch 0 (four triangles, higher priority): the square at depth 5, one
    // triangle elsewhere, and a triangle carrying the seam vertex 12.
    // Patch 1: component X, linked to the square by the fused pair (12, 7),
    // holds a triangle inside the square at 1 % depth disagreement and one
    // reaching far beyond it; island Y lies inside the square at the same
    // depth but no accepted correspondence links it.
    let points = vec![
        point(-1.0, -1.0, 5.0),
        point(1.0, -1.0, 5.0),
        point(1.0, 1.0, 5.0),
        point(-1.0, 1.0, 5.0),
        point(-4.0, -1.0, 5.0),
        point(-3.0, -1.0, 5.0),
        point(-3.5, 0.0, 5.0),
        point(-0.5, -0.5, 5.05),
        point(0.6, -0.4, 5.05),
        point(0.1, 0.7, 5.05),
        point(3.0, -0.5, 5.0),
        point(3.5, 0.5, 5.0),
        point(-0.5, -0.5, 5.05),
        point(-0.6, 0.2, 5.0),
        point(-0.2, 0.2, 5.0),
        point(-0.4, 0.6, 5.0),
    ];
    let membership: Vec<Option<usize>> = (0..points.len())
        .map(|index| Some(usize::from((7..12).contains(&index) || index >= 13)))
        .collect();
    let original = vec![
        triangle(0, 1, 2),
        triangle(0, 2, 3),
        triangle(4, 5, 6),
        triangle(0, 1, 12),
        triangle(7, 8, 9),
        triangle(8, 10, 11),
        triangle(13, 14, 15),
    ];
    let link = [(12, 7)];
    let cameras = [Some(identity_camera(0.0)), Some(identity_camera(0.4))];
    (points, membership, original, link, cameras)
}

#[test]
fn owned_overlap_is_removed_and_unique_surface_is_kept() {
    let (points, membership, original, link, cameras) = owned_overlap_fixture();
    let mut triangles = original.clone();
    let stats = merge_fused_patches(&points, &mut triangles, &membership, &link, &cameras, &[]);
    // Only the linked inner triangle is owned; the far-reaching one and the
    // unlinked island stay, and nothing of patch 0 is touched.
    assert_eq!(stats.removed_duplicate_triangles, 1);
    assert_eq!(stats.retained_by_group, vec![4, 2]);
    assert!(triangles.iter().any(|t| (t.a, t.b, t.c) == (13, 14, 15)));
    assert!(triangles.iter().any(|t| (t.b, t.c) == (10, 11)));
    assert!(!triangles.iter().any(|t| (t.b, t.c) == (8, 9)));

    // Without a fusion-accepted pair nothing is owned, with or without
    // reference cameras (the fail-closed 3D path is used without them).
    for cameras in [&cameras[..], &[]] {
        let mut triangles = original.clone();
        let stats = merge_fused_patches(&points, &mut triangles, &membership, &[], cameras, &[]);
        assert_eq!(stats.removed_duplicate_triangles, 0);
        assert_eq!(triangles.len(), original.len());
    }
}

#[test]
fn a_depth_rejection_in_the_image_is_not_overturned_by_the_3d_test() {
    // Patch 1 sees the square wall (three triangles, which fix its depth
    // scale) and, linked as well, a parallel layer 4 % in front of it. The
    // image gate rejects the layer as another surface; the 3D test (tolerance
    // ~0.9 edge, here about 1.8) must not delete it anyway.
    let points = vec![
        point(-1.0, -1.0, 5.0),
        point(1.0, -1.0, 5.0),
        point(1.0, 1.0, 5.0),
        point(-1.0, 1.0, 5.0),
        point(-4.0, -1.0, 5.0),
        point(-3.0, -1.0, 5.0),
        point(-3.5, 0.0, 5.0),
        // Seam vertices of patch 0, coincident with patch-1 vertices.
        point(-0.5, -0.5, 4.8),
        point(0.9, -0.9, 5.0),
        // Patch 1: the layer, then a strip on the wall.
        point(-0.5, -0.5, 4.8),
        point(0.0, -0.4, 4.8),
        point(-0.3, 0.0, 4.8),
        point(0.9, -0.9, 5.0),
        point(0.9, 0.9, 5.0),
        point(0.2, -0.9, 5.0),
        point(0.2, 0.9, 5.0),
        point(0.5, 0.0, 5.0),
    ];
    let membership: Vec<Option<usize>> = (0..points.len())
        .map(|index| Some(usize::from(index >= 9)))
        .collect();
    let original = vec![
        triangle(0, 1, 2),
        triangle(0, 2, 3),
        triangle(4, 5, 6),
        triangle(0, 1, 7),
        triangle(1, 2, 8),
        triangle(9, 10, 11),
        triangle(12, 13, 16),
        triangle(12, 16, 14),
        triangle(13, 15, 16),
    ];
    let cameras = [Some(identity_camera(0.0)), Some(identity_camera(0.4))];
    let mut triangles = original.clone();
    let stats = merge_fused_patches(
        &points,
        &mut triangles,
        &membership,
        &[(7, 9), (8, 12)],
        &cameras,
        &[],
    );
    // The wall strip is owned; the layer stays.
    assert_eq!(stats.removed_duplicate_triangles, 3, "{stats:?}");
    assert!(triangles.iter().any(|t| (t.b, t.c) == (10, 11)));
}

#[test]
fn the_relative_scale_is_anchored_at_fusion_accepted_seam_vertices() {
    // One patch-1 component: two wall triangles at depth 5 (one carries the
    // fused seam vertex), joined by a connector to a densely tessellated
    // layer at 4.8 that supplies most overlap samples. An unanchored median
    // would take the layer's 0.96 as the patch scale and delete the layer.
    let mut points = vec![
        point(-1.0, -1.0, 5.0),
        point(1.0, -1.0, 5.0),
        point(1.0, 1.0, 5.0),
        point(-1.0, 1.0, 5.0),
        point(-4.0, -1.0, 5.0),
        point(-3.0, -1.0, 5.0),
        point(-3.5, 0.0, 5.0),
        // 7: patch-0 seam vertex, coincident with 8.
        point(0.9, -0.9, 5.0),
        // Patch 1 wall: 8..=11.
        point(0.9, -0.9, 5.0),
        point(0.9, -0.2, 5.0),
        point(0.4, -0.9, 5.0),
        point(0.4, -0.2, 5.0),
    ];
    // Layer grid 12..=20 (3 × 3 vertices) at 4.8 over x, y ∈ [-0.9, 0.1].
    for row in 0..3 {
        for column in 0..3 {
            points.push(point(
                -0.9 + column as f32 * 0.5,
                -0.9 + row as f32 * 0.5,
                4.8,
            ));
        }
    }
    let mut membership: Vec<Option<usize>> = (0..points.len())
        .map(|index| Some(usize::from(index >= 8)))
        .collect();
    let grid = |row: usize, column: usize| 12 + row * 3 + column;
    let mut original = vec![
        triangle(0, 1, 2),
        triangle(0, 2, 3),
        triangle(4, 5, 6),
        triangle(1, 2, 7),
        triangle(8, 9, 10),
        triangle(9, 11, 10),
        // Connector from the wall to the layer.
        triangle(11, 9, grid(2, 2)),
    ];
    // Patch 0 keeps priority: more triangles, far from the square.
    original.extend((0..12).map(|_| triangle(4, 5, 6)));
    for row in 0..2 {
        for column in 0..2 {
            original.push(triangle(
                grid(row, column),
                grid(row, column + 1),
                grid(row + 1, column + 1),
            ));
            original.push(triangle(
                grid(row, column),
                grid(row + 1, column + 1),
                grid(row + 1, column),
            ));
        }
    }
    let cameras = [Some(identity_camera(0.0)), Some(identity_camera(0.4))];
    let mut triangles = original.clone();
    let stats = merge_fused_patches(
        &points,
        &mut triangles,
        &membership,
        &[(7, 8)],
        &cameras,
        &[],
    );
    let layer_kept = |triangles: &[MeshTriangle]| {
        triangles
            .iter()
            .filter(|t| {
                [t.a, t.b, t.c]
                    .iter()
                    .all(|vertex| (12..21).contains(vertex))
            })
            .count()
    };
    assert_eq!(layer_kept(&triangles), 8, "{stats:?}");
    // Too few anchored samples: the pair is compared uncalibrated, and the
    // diagnostic says so.
    assert_eq!(stats.uncalibrated_patch_pairs, vec![(0, 1)]);
    assert!(stats
        .diagnostic(2)
        .contains("1 linked patch pair(s) had too few anchored overlap samples"));

    // A third patch fused to the layer: its seam vertices must not calibrate
    // patch 1 against patch 0.
    let third = points.len();
    let mut pairs = vec![(7, 8)];
    for row in 0..3 {
        for column in 0..3 {
            pairs.push((grid(row, column), points.len()));
            points.push(point(
                -0.9 + column as f32 * 0.5,
                -0.9 + row as f32 * 0.5,
                4.8,
            ));
            membership.push(Some(2));
        }
    }
    points.push(point(-6.0, 3.0, 4.8));
    membership.push(Some(2));
    original.push(triangle(third, third + 1, points.len() - 1));
    let cameras = [
        Some(identity_camera(0.0)),
        Some(identity_camera(0.4)),
        Some(identity_camera(0.8)),
    ];
    let mut triangles = original.clone();
    let stats = merge_fused_patches(&points, &mut triangles, &membership, &pairs, &cameras, &[]);
    assert_eq!(layer_kept(&triangles), 8, "{stats:?}");
}

#[test]
fn pairs_across_evidence_origins_link_patches_but_keep_their_vertices() {
    use crate::EvidenceOrigin::{GeometricMultiView, RevalidatedCompletion};
    // The square (patch 0, geometric) and a patch-1 triangle inside it whose
    // seam vertex 7 is revalidated completion, paired with patch-0 vertex 12.
    let points = vec![
        point(-1.0, -1.0, 5.0),
        point(1.0, -1.0, 5.0),
        point(1.0, 1.0, 5.0),
        point(-1.0, 1.0, 5.0),
        point(-4.0, -1.0, 5.0),
        point(-3.0, -1.0, 5.0),
        point(-3.5, 0.0, 5.0),
        point(-0.5, -0.5, 5.0),
        point(0.6, -0.4, 5.0),
        point(0.1, 0.7, 5.0),
        point(-3.0, 1.0, 5.0),
        point(-2.0, 1.0, 5.0),
        point(-0.5, -0.5, 5.0),
    ];
    let membership: Vec<Option<usize>> = (0..points.len())
        .map(|index| Some(usize::from((7..10).contains(&index))))
        .collect();
    let mut origins = vec![Some(GeometricMultiView); points.len()];
    origins[7] = Some(RevalidatedCompletion);
    let original = vec![
        triangle(0, 1, 2),
        triangle(0, 2, 3),
        triangle(4, 5, 6),
        triangle(4, 10, 11),
        triangle(0, 1, 12),
        triangle(7, 8, 9),
    ];
    let cameras = [Some(identity_camera(0.0)), Some(identity_camera(0.4))];
    let mut triangles = original.clone();
    let stats = merge_fused_patches(
        &points,
        &mut triangles,
        &membership,
        &[(12, 7)],
        &cameras,
        &origins,
    );
    assert_eq!(stats.fused_pairs, 0);
    assert_eq!(stats.provenance_separated_pairs, 1);
    assert_eq!(stats.shared_vertices, 0);
    // The pair still links the patches, so the owned triangle is removed
    // without any corner having been relabelled.
    assert_eq!(stats.removed_duplicate_triangles, 1, "{stats:?}");
    assert!(triangles.iter().all(|t| (t.b, t.c) != (8, 9)));

    // With equal origins the same pair becomes a shared vertex.
    let same = vec![Some(GeometricMultiView); points.len()];
    let mut triangles = original.clone();
    let stats = merge_fused_patches(
        &points,
        &mut triangles,
        &membership,
        &[(12, 7)],
        &cameras,
        &same,
    );
    assert_eq!(stats.fused_pairs, 1);
    assert_eq!(stats.provenance_separated_pairs, 0);
}

#[test]
fn only_vertices_referenced_by_two_patches_count_as_shared() {
    let points = vec![
        point(0.0, 0.0, 0.0),
        point(1.0, 0.0, 0.0),
        point(0.0, 1.0, 0.0),
        point(0.0, 1.0, 0.0),
        point(-1.0, 2.0, 0.0),
        point(-1.0, 1.0, 0.0),
    ];
    let mut triangles = vec![triangle(0, 1, 2), triangle(3, 4, 5)];
    let membership: Vec<Option<usize>> = [Some(0); 3].into_iter().chain([Some(1); 3]).collect();
    let stats = merge_fused_patches(&points, &mut triangles, &membership, &[(2, 3)], &[], &[]);
    assert_eq!(stats.fused_pairs, 1);
    assert_eq!(stats.cross_reference_triangles, 1);
    // Only the fused seam corner is referenced by both patches.
    assert_eq!(stats.shared_vertices, 1);
}

#[test]
fn oversized_owner_triangles_are_checked_exactly() {
    // One kept triangle far larger than the rest of the footprint's grid.
    let mut kept = vec![(
        corners([(-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (1.0, 1.0, 5.0)]),
        0,
        [0, 1, 2],
        103,
    )];
    for step in 0..4 {
        let x = 3.0 + f64::from(step) * 0.02;
        kept.push((
            corners([(x, 3.0, 5.0), (x + 0.01, 3.0, 5.0), (x, 3.01, 5.0)]),
            0,
            [3 + step as usize, 9, 10],
            104,
        ));
    }
    let footprint = Footprint::new(0, identity_camera(0.0), &kept).unwrap();
    // The large triangle sits on a coarser grid level than the small ones.
    assert!(
        footprint.grid.levels > 1,
        "levels={}",
        footprint.grid.levels
    );
    let inside = corners([(0.5, -0.8, 5.0), (0.8, -0.8, 5.0), (0.8, -0.5, 5.0)]);
    let owned = footprint
        .ownership(&inside, 1.0, &|_| true)
        .map(|(owned, _)| owned)
        .unwrap();
    assert!((owned.image_share - 1.0).abs() < 1.0e-9, "{owned:?}");
}

#[test]
fn scale_samples_need_an_anchored_owner_triangle() {
    let footprint = square_footprint(identity_camera(0.0));
    let layer = corners([(-0.5, -0.5, 4.8), (0.6, -0.4, 4.8), (0.1, 0.7, 4.8)]);
    // No owner triangle touches an anchor: no sample.
    assert!(footprint
        .depth_ratios(
            &layer,
            &|_| true,
            &|vertices| vertices.contains(&99),
            usize::MAX
        )
        .is_empty());
    // Only the owner triangle with vertex 1 is anchored: one sample.
    assert_eq!(
        footprint
            .depth_ratios(
                &layer,
                &|_| true,
                &|vertices| vertices.contains(&1),
                usize::MAX
            )
            .len(),
        1
    );
}

#[test]
fn unprojectable_linked_owners_leave_the_image_unable_to_judge() {
    let mut kept = vec![
        (
            corners([(-1.0, -1.0, 5.0), (1.0, -1.0, 5.0), (1.0, 1.0, 5.0)]),
            0,
            [0, 1, 2],
            105,
        ),
        (
            corners([(-1.0, -1.0, 5.0), (1.0, 1.0, 5.0), (-1.0, 1.0, 5.0)]),
            0,
            [0, 2, 3],
            106,
        ),
    ];
    // Behind the reference camera: absent from the image.
    kept.push((
        corners([(3.0, 3.0, -1.0), (4.0, 3.0, -1.0), (3.0, 4.0, -1.0)]),
        1,
        [4, 5, 6],
        107,
    ));
    let footprint = Footprint::new(0, identity_camera(0.0), &kept).unwrap();
    let outside = corners([(2.0, 2.0, 5.0), (2.5, 2.0, 5.0), (2.0, 2.5, 5.0)]);
    // The image judges the projected component and leaves the omitted one
    // to the 3D test.
    let (judged, unjudged) = footprint.ownership(&outside, 1.0, &|_| true).unwrap();
    assert_eq!(judged.image_share, 0.0);
    // The omitted triangle is reported by its mesh index.
    assert_eq!(unjudged.len(), 1);
    // Without a link to the omitted component nothing is left unjudged.
    let (_, unjudged) = footprint
        .ownership(&outside, 1.0, &|component| component == 0)
        .unwrap();
    assert!(unjudged.is_empty());
}

#[test]
fn scale_samples_stop_at_the_limit() {
    let footprint = square_footprint(identity_camera(0.0));
    let layer = corners([(-0.5, -0.5, 4.8), (0.6, -0.4, 4.8), (0.1, 0.7, 4.8)]);
    assert_eq!(
        footprint
            .depth_ratios(&layer, &|_| true, &|_| true, usize::MAX)
            .len(),
        2
    );
    assert_eq!(
        footprint
            .depth_ratios(&layer, &|_| true, &|_| true, 1)
            .len(),
        1
    );
    assert!(footprint
        .depth_ratios(&layer, &|_| true, &|_| true, 0)
        .is_empty());
}

#[test]
fn an_owner_without_an_image_judgement_still_gets_the_3d_test() {
    use crate::EvidenceOrigin::{GeometricMultiView, RevalidatedCompletion};
    let points = vec![
        // Patch 0 (no reference camera): a square at depth 5 ...
        point(-1.0, -1.0, 5.0),
        point(1.0, -1.0, 5.0),
        point(1.0, 1.0, 5.0),
        point(-1.0, 1.0, 5.0),
        // ... and filler for priority.
        point(-9.0, -9.0, 5.0),
        point(-8.0, -9.0, 5.0),
        point(-9.0, -8.0, 5.0),
        // Patch 1 (with a camera), elsewhere: it judges the candidate unowned.
        point(10.0, 0.0, 5.0),
        point(11.0, 0.0, 5.0),
        point(10.0, 1.0, 5.0),
        // Patch 2: a candidate lying on patch 0's square.
        point(-1.0, -1.0, 5.0),
        point(0.0, -1.0, 5.0),
        point(-1.0, 0.0, 5.0),
    ];
    let membership: Vec<Option<usize>> = (0..points.len())
        .map(|index| {
            Some(if index < 7 {
                0
            } else if index < 10 {
                1
            } else {
                2
            })
        })
        .collect();
    let mut origins = vec![Some(GeometricMultiView); points.len()];
    // The patch-1 link keeps separate vertices, so the candidate stays put.
    origins[7] = Some(RevalidatedCompletion);
    let mut triangles = vec![triangle(0, 1, 2), triangle(0, 2, 3)];
    triangles.extend((0..12).map(|_| triangle(4, 5, 6)));
    triangles.extend([triangle(7, 8, 9), triangle(7, 8, 9)]);
    triangles.push(triangle(10, 11, 12));
    let cameras = [None, Some(identity_camera(0.0)), None];
    let stats = merge_fused_patches(
        &points,
        &mut triangles,
        &membership,
        &[(0, 10), (7, 11)],
        &cameras,
        &origins,
    );
    assert!(
        !triangles.iter().any(|t| [t.a, t.b, t.c]
            .iter()
            .any(|vertex| (11..13).contains(vertex))),
        "the duplicate candidate survived: {stats:?}"
    );
}

#[test]
fn large_items_far_away_are_not_scanned_by_small_queries() {
    let mut grid = LevelGrid::<2>::new(1.0);
    for item in 0..100 {
        // Long strips, all far from the origin.
        let y = 1000.0 + f64::from(item) * 50.0;
        grid.insert(item as usize, [0.0, y], [500.0, y + 1.0]);
    }
    grid.insert(500, [0.2, 0.2], [0.8, 0.8]);
    let mut found = BTreeSet::new();
    assert!(grid.query([0.0, 0.0], [1.0, 1.0], MAX_IMAGE_GRID_CELLS, &mut found));
    assert_eq!(found, BTreeSet::from([500]));
}

// ---------------------------------------------------------------------------
// Acceptance for #145: bounded grid postings and observable clipping-budget
// exhaustion. Written by an independent acceptance agent before the
// implementation; the observable API below is chosen here and is not yet
// implemented on the baseline.
//
// API choices (kept minimal):
// - `MAX_POSTINGS_PER_QUERY: usize` (module constant in `surface_merge.rs`):
//   the most grid postings (owner triangles retrieved from `LevelGrid`
//   buckets, including the exact-list fallback) one candidate lookup may
//   process. It must stay at least `2 * MAX_UNCOVERED_PIECES`, so the posting
//   bound never makes the clipping budget unreachable, and at most 1024, so
//   these fixtures (sized from the bound) stay fast.
// - `MergeStats::max_postings_per_query: usize`: the most postings any single
//   candidate lookup of this merge processed (image ownership, scale samples
//   and the 3D test alike). It makes per-query work observable without timing.
// - `MergeStats::posting_bound_hits: usize`: candidate lookups that needed more
//   than `MAX_POSTINGS_PER_QUERY` postings and were cut off (fail-closed: such
//   a lookup may only under-estimate ownership, never invent it).
// - `MergeStats::clipping_budget_exhausted: usize`: candidate triangles whose
//   subtraction needed more than `MAX_UNCOVERED_PIECES` uncovered fragments.
//   They are retained (fail-closed) but counted here and NOT in
//   `remaining_overlap_triangles`.
// - `MergeStats::diagnostic` mentions both counters with the phrases
//   "{posting_bound_hits} candidate lookup(s) hit the posting bound" and
//   "{clipping_budget_exhausted} triangle(s) exhausted the clipping budget".
// ---------------------------------------------------------------------------

fn bound_hits_phrase(stats: &MergeStats) -> String {
    format!(
        "{} candidate lookup(s) hit the posting bound",
        stats.posting_bound_hits
    )
}

fn clipping_budget_phrase(stats: &MergeStats) -> String {
    format!(
        "{} triangle(s) exhausted the clipping budget",
        stats.clipping_budget_exhausted
    )
}

#[test]
fn the_posting_bound_leaves_the_clipping_budget_reachable() {
    assert!(
        MAX_POSTINGS_PER_QUERY >= 2 * MAX_UNCOVERED_PIECES,
        "MAX_POSTINGS_PER_QUERY={MAX_POSTINGS_PER_QUERY}"
    );
    assert!(
        MAX_POSTINGS_PER_QUERY <= 1024,
        "MAX_POSTINGS_PER_QUERY={MAX_POSTINGS_PER_QUERY}"
    );
}

/// Two crossing strip-like triangulations at depth 5: patch 0 is a comb of
/// `2 * bound + 4` tall, thin columns over x ∈ [-1, 1], y ∈ [-1, 1] (two
/// rows); patch 1 a thin horizontal strip over x ∈ [-1.5, 1.5], y ∈ [0, 0.05]
/// with segment breaks at x = -1, 0, 1. Every comb triangle lands in one of
/// four coarse buckets (its long edges set the cell size), and each inner
/// strip triangle genuinely overlaps more than `bound` comb triangles. The
/// patches are fused at the coincident vertex (-1, 0, 5). Returns the fixture
/// and the strip's two outer end vertices.
fn crossing_strips(
    bound: usize,
) -> (
    Vec<Point3>,
    Vec<Option<usize>>,
    Vec<MeshTriangle>,
    Vec<(usize, usize)>,
    [usize; 2],
) {
    let columns = 2 * bound + 4;
    let comb = |row: usize, column: usize| row * (columns + 1) + column;
    let mut points = Vec::new();
    for row in 0..3 {
        for column in 0..=columns {
            points.push(point(
                -1.0 + 2.0 * column as f32 / columns as f32,
                row as f32 - 1.0,
                5.0,
            ));
        }
    }
    let mut triangles = Vec::new();
    for row in 0..2 {
        for column in 0..columns {
            triangles.push(triangle(
                comb(row, column),
                comb(row, column + 1),
                comb(row + 1, column + 1),
            ));
            triangles.push(triangle(
                comb(row, column),
                comb(row + 1, column + 1),
                comb(row + 1, column),
            ));
        }
    }
    let base = points.len();
    let xs = [-1.5_f32, -1.0, 0.0, 1.0, 1.5];
    for y in [0.0_f32, 0.05] {
        for x in xs {
            points.push(point(x, y, 5.0));
        }
    }
    let (bottom, top) = (|k: usize| base + k, |k: usize| base + xs.len() + k);
    for k in 0..xs.len() - 1 {
        triangles.push(triangle(bottom(k), bottom(k + 1), top(k + 1)));
        triangles.push(triangle(bottom(k), top(k + 1), top(k)));
    }
    let membership = (0..points.len())
        .map(|index| Some(usize::from(index >= base)))
        .collect();
    // Comb vertex (-1, 0) and strip vertex (-1, 0) coincide.
    let pairs = vec![(comb(1, 0), bottom(1))];
    (
        points,
        membership,
        triangles,
        pairs,
        [bottom(0), top(xs.len() - 1)],
    )
}

#[test]
fn crossing_strip_triangulations_keep_per_query_postings_bounded() {
    let (points, membership, original, pairs, [left_end, right_end]) =
        crossing_strips(MAX_POSTINGS_PER_QUERY);
    let with_cameras = [Some(identity_camera(0.0)), Some(identity_camera(0.0))];
    // The image ownership path and the 3D fallback (no reference cameras).
    for cameras in [&with_cameras[..], &[]] {
        let mut triangles = original.clone();
        let stats = merge_fused_patches(&points, &mut triangles, &membership, &pairs, cameras, &[]);
        assert_eq!(stats.fused_pairs, 1, "{stats:?}");
        // Work is counted, and no single lookup exceeds the stated bound,
        // although one dense bucket holds about 2 * bound comb triangles.
        assert!(stats.max_postings_per_query > 0, "{stats:?}");
        assert!(
            stats.max_postings_per_query <= MAX_POSTINGS_PER_QUERY,
            "max_postings_per_query={} > {MAX_POSTINGS_PER_QUERY}",
            stats.max_postings_per_query
        );
        // Each inner strip triangle overlaps more than `bound` owner
        // triangles, so its lookup must hit the bound, and that is recorded.
        assert!(stats.posting_bound_hits > 0, "{stats:?}");
        assert!(
            stats.diagnostic(2).contains(&bound_hits_phrase(&stats)),
            "{}",
            stats.diagnostic(2)
        );
        // Fail-closed: strip surface beyond the comb is never removed.
        let ends = triangles
            .iter()
            .filter(|t| {
                [t.a, t.b, t.c]
                    .iter()
                    .any(|v| *v == left_end || *v == right_end)
            })
            .count();
        assert_eq!(ends, 4, "{stats:?}");
        // The comb (higher priority) is untouched.
        assert_eq!(stats.retained_by_group[0], original.len() - 8);
    }
}

/// Patch 0: `slivers` thin triangles fanning out from a hub far below,
/// each crossing patch 1's single candidate triangle (-2,0)-(2,0)-(0,2) on
/// the plane z = 0 as a narrow, near-vertical band, with gaps between them;
/// plus one triangle at the hub touching the candidate only at its corner
/// (-2, 0), whose vertex is fused with that corner. Every crossing sliver
/// splits one uncovered fragment in two, so the candidate's 3D subtraction
/// needs `slivers + 1` fragments and never covers it fully.
fn sliver_fan(slivers: usize) -> (Vec<Point3>, Vec<Option<usize>>, Vec<MeshTriangle>, usize) {
    let mut points = vec![
        point(0.0, -100.0, 0.0),
        point(-2.0, 0.0, 0.0),
        point(-2.0, -1.0, 0.0),
    ];
    let mut triangles = vec![triangle(0, 1, 2)];
    for sliver in 0..slivers {
        let x = -2.0 + 4.0 * (sliver as f32 + 0.5) / slivers as f32;
        let index = points.len();
        points.push(point(x, 100.0, 0.0));
        points.push(point(x + 0.01, 100.0, 0.0));
        triangles.push(triangle(0, index, index + 1));
    }
    let candidate = points.len();
    points.extend([
        point(-2.0, 0.0, 0.0),
        point(2.0, 0.0, 0.0),
        point(0.0, 2.0, 0.0),
    ]);
    triangles.push(triangle(candidate, candidate + 1, candidate + 2));
    let membership = (0..points.len())
        .map(|index| Some(usize::from(index >= candidate)))
        .collect();
    (points, membership, triangles, candidate)
}

#[test]
fn a_seam_beyond_the_fragment_budget_is_reported_as_budget_exhausted() {
    let slivers = MAX_UNCOVERED_PIECES + 32;
    let (points, membership, mut triangles, candidate) = sliver_fan(slivers);
    let stats = merge_fused_patches(
        &points,
        &mut triangles,
        &membership,
        &[(1, candidate)],
        &[],
        &[],
    );
    assert_eq!(stats.fused_pairs, 1, "{stats:?}");
    // The postings fit the bound, so only the clipping budget stops it.
    assert_eq!(stats.posting_bound_hits, 0, "{stats:?}");
    // Fail-closed: the gaps between slivers are real surface, so the
    // candidate stays ...
    assert_eq!(stats.removed_duplicate_triangles, 0, "{stats:?}");
    assert_eq!(stats.retained_by_group, vec![slivers + 1, 1]);
    // ... but it is reported as budget-exhausted, not as an ordinary partial
    // overlap, and the diagnostic says the bounded algorithm stopped early.
    assert_eq!(stats.clipping_budget_exhausted, 1, "{stats:?}");
    assert_eq!(stats.remaining_overlap_triangles, 0, "{stats:?}");
    let diagnostic = stats.diagnostic(2);
    assert!(
        diagnostic.contains(&clipping_budget_phrase(&stats)),
        "{diagnostic}"
    );
}

#[test]
fn ordinary_overlaps_report_no_bound_hits_and_keep_their_results() {
    // A seam within the fragment budget stays an ordinary partial overlap.
    let (points, membership, mut triangles, candidate) = sliver_fan(8);
    let stats = merge_fused_patches(
        &points,
        &mut triangles,
        &membership,
        &[(1, candidate)],
        &[],
        &[],
    );
    assert_eq!(stats.removed_duplicate_triangles, 0, "{stats:?}");
    assert_eq!(stats.remaining_overlap_triangles, 1, "{stats:?}");
    assert_eq!(stats.clipping_budget_exhausted, 0, "{stats:?}");
    assert_eq!(stats.posting_bound_hits, 0, "{stats:?}");
    assert!(stats.max_postings_per_query > 0, "{stats:?}");
    assert!(stats
        .diagnostic(2)
        .contains(&clipping_budget_phrase(&stats)));

    // The image-ownership fixture keeps its result and hits neither bound.
    let (points, membership, original, link, cameras) = owned_overlap_fixture();
    let mut triangles = original.clone();
    let stats = merge_fused_patches(&points, &mut triangles, &membership, &link, &cameras, &[]);
    assert_eq!(stats.removed_duplicate_triangles, 1, "{stats:?}");
    assert_eq!(stats.retained_by_group, vec![4, 2]);
    assert_eq!(stats.posting_bound_hits, 0, "{stats:?}");
    assert_eq!(stats.clipping_budget_exhausted, 0, "{stats:?}");
    assert!(stats.max_postings_per_query > 0, "{stats:?}");
    assert!(stats.max_postings_per_query <= original.len(), "{stats:?}");

    // The 3D seam fixture likewise.
    let (points, mut triangles, membership) = adjacent_grids_fixture();
    let total = triangles.len();
    let stats = merge_fused_patches(
        &points,
        &mut triangles,
        &membership,
        &[(2, 7), (5, 10)],
        &[],
        &[],
    );
    assert_eq!(stats.removed_duplicate_triangles, 2, "{stats:?}");
    assert_eq!(triangles.len(), total - 2);
    assert_eq!(stats.posting_bound_hits, 0, "{stats:?}");
    assert_eq!(stats.clipping_budget_exhausted, 0, "{stats:?}");
    assert!(stats.max_postings_per_query > 0, "{stats:?}");
    assert!(stats.max_postings_per_query <= total, "{stats:?}");
    assert!(stats.diagnostic(2).contains(&bound_hits_phrase(&stats)));
}
