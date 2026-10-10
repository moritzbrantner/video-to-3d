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

    let stats = merge_fused_patches(&points, &mut triangles, &membership, &[(0, 6)], &[]);

    assert_eq!(stats.fused_pairs, 1);
    assert_eq!(stats.removed_duplicate_triangles, 1);
    assert!(signature(&triangles).contains(&(9, 10, 11)));
    assert!(signature(&triangles).contains(&(3, 4, 5)));
}

#[test]
fn a_confirmed_seam_deduplicates_overlap_but_keeps_the_uncovered_extension() {
    // Two adjacent camera grids: A covers x=0..2, B x=1..3.
    // The last column of B joins A at x=2; x=1..2 is the redundant sheet,
    // but x=2..3 is real, uniquely supported surface that must stay.
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
    );
    assert_eq!(stats, again);
    let mut replay = original;
    merge_fused_patches(&points, &mut replay, &membership, &[(2, 7), (5, 10)], &[]);
    assert_eq!(first, signature(&replay));
}

#[test]
fn large_triangles_are_indexed_across_their_interior() {
    let surface = geometry([(0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (0.0, 30.0, 0.0)]);
    let grid = surface.grid_bounds(1.0, 0.0);
    assert!(grid_cells(&grid).is_none());
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
            ),
            (
                corners([(-1.0, -1.0, 5.0), (1.0, 1.0, 5.0), (-1.0, 1.0, 5.0)]),
                0,
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
    assert!((footprint.ownership(&same, 1.0, &|_| true).image_share - 1.0).abs() < 1.0e-9);
    // A layer 4 % in front (0.2 m before a wall 5 m away) is a separate surface.
    let layer = corners([(-0.5, -0.5, 4.8), (0.6, -0.4, 4.8), (0.1, 0.7, 4.8)]);
    assert_eq!(footprint.ownership(&layer, 1.0, &|_| true).image_share, 0.0);
    // Unless the whole patch is 4 % nearer: a relative depth scale, removed
    // before the bound applies.
    let ratios = footprint.depth_ratios(&layer, &|_| true);
    assert!(!ratios.is_empty());
    assert!(ratios.iter().all(|ratio| (ratio - 0.96).abs() < 1.0e-9));
    assert!((footprint.ownership(&layer, 0.96, &|_| true).image_share - 1.0).abs() < 1.0e-9);
    // 12 % farther: a separate surface, never owned.
    let behind = corners([(-0.5, -0.5, 5.6), (0.6, -0.4, 5.6), (0.1, 0.7, 5.6)]);
    assert_eq!(
        footprint.ownership(&behind, 1.0, &|_| true).image_share,
        0.0
    );
    // Behind the camera.
    let flipped = corners([(-0.5, -0.5, -5.0), (0.6, -0.4, -5.0), (0.1, 0.7, -5.0)]);
    assert_eq!(
        footprint.ownership(&flipped, 1.0, &|_| true).image_share,
        0.0
    );
}

#[test]
fn image_ownership_measures_the_share_on_the_footprint() {
    let footprint = square_footprint(identity_camera(0.0));
    // Half of this triangle's projected area lies beyond x = 1.
    let seam = corners([(0.0, -0.5, 5.0), (2.0, -0.5, 5.0), (1.0, 0.5, 5.0)]);
    let owned = footprint.ownership(&seam, 1.0, &|_| true);
    assert!((owned.image_share - 0.5).abs() < 1.0e-9);
    // Fronto-parallel, so the owned surface area is half of the 3D area too.
    assert!((owned.surface_area - 0.5).abs() < 1.0e-9);
    // Tilted in depth, the image share and the surface share differ; the
    // surface area is measured on the triangle's own plane.
    let tilted = corners([(0.0, -0.5, 4.9), (2.0, -0.5, 5.1), (1.0, 0.5, 5.0)]);
    let owned = footprint.ownership(&tilted, 1.0, &|_| true);
    let full = (tilted[1] - tilted[0])
        .cross(&(tilted[2] - tilted[0]))
        .norm()
        * 0.5;
    assert!(owned.image_share > 0.4 && owned.image_share < 0.6);
    assert!(owned.surface_area > 0.0 && owned.surface_area < full);
    let outside = corners([(1.5, -0.5, 5.0), (2.5, -0.5, 5.0), (2.0, 0.5, 5.0)]);
    assert_eq!(
        footprint.ownership(&outside, 1.0, &|_| true).image_share,
        0.0
    );
}

#[test]
fn owned_overlap_is_removed_and_unique_surface_is_kept() {
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

    let mut triangles = original.clone();
    let stats = merge_fused_patches(&points, &mut triangles, &membership, &link, &cameras);
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
        let stats = merge_fused_patches(&points, &mut triangles, &membership, &[], cameras);
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
    );
    // The wall strip is owned; the layer stays.
    assert_eq!(stats.removed_duplicate_triangles, 3, "{stats:?}");
    assert!(triangles.iter().any(|t| (t.b, t.c) == (10, 11)));
}
