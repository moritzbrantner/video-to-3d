use super::*;

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
    MeshTriangle { a, b, c, confidence: 0.9 }
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
    assert_eq!(inner.covered_by([&left, &right].into_iter(), 0.1), (true, true));
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
    assert_eq!(orthogonal.covered_by([&kept].into_iter(), 0.1), (false, false));
}

#[test]
fn no_accepted_fusion_pair_means_no_dedup_even_for_coincident_sheets() {
    let points = vec![
        point(0.0, 0.0, 0.0), point(1.0, 0.0, 0.0), point(0.0, 1.0, 0.0),
        point(0.0, 0.0, 0.0), point(1.0, 0.0, 0.0), point(0.0, 1.0, 0.0),
    ];
    let original = vec![triangle(0, 1, 2), triangle(3, 4, 5)];
    let mut triangles = original.clone();
    let stats = merge_fused_patches(&points, &mut triangles, &[Some(0); 3].into_iter()
        .chain([Some(1); 3]).collect::<Vec<_>>(), &[]);
    assert_eq!(stats.removed_duplicate_triangles, 0);
    assert_eq!(stats.cross_reference_triangles, 0);
    assert_eq!(stats.patch_components, 2);
    assert_eq!(signature(&triangles), signature(&original));
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
                points.push(point(start + x as f32, y as f32, if shared { 0.0 } else { z }));
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
    let original_points = points.iter().map(|p| (p.x, p.y, p.z, p.r, p.g, p.b)).collect::<Vec<_>>();
    let original = triangles.clone();
    let stats = merge_fused_patches(&points, &mut triangles, &membership, &[(2, 7), (5, 10)]);
    assert_eq!(stats.fused_pairs, 2);
    assert_eq!(stats.patch_components, 1);
    assert!(stats.cross_reference_triangles > 0);
    assert_eq!(stats.removed_duplicate_triangles, 2);
    assert_eq!(triangles.len(), original.len() - 2);
    assert_eq!(points.iter().map(|p| (p.x, p.y, p.z, p.r, p.g, p.b)).collect::<Vec<_>>(), original_points);
    assert!(triangles.iter().any(|t| [t.a, t.b, t.c].contains(&8)));
    let first = signature(&triangles);
    let again = merge_fused_patches(&points, &mut original.clone(), &membership, &[(2, 7), (5, 10)]);
    assert_eq!(stats, again);
    let mut replay = original;
    merge_fused_patches(&points, &mut replay, &membership, &[(2, 7), (5, 10)]);
    assert_eq!(first, signature(&replay));
}

#[test]
fn large_triangles_are_indexed_across_their_interior() {
    let surface = geometry([(0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (0.0, 20.0, 0.0)]);
    let grid = surface.grid_bounds(1.0, 0.0);
    assert!(grid_cells(&grid).is_none());
    let small = geometry([(4.0, 4.0, 0.0), (5.0, 4.0, 0.0), (4.0, 5.0, 0.0)]);
    assert!(small.covered_by([&surface].into_iter(), 0.1).0);
}
