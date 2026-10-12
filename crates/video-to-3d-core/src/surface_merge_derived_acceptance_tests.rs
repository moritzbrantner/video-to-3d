//! Independent owned-result contract. Routine names/record spelling may be
//! translated by the acceptance author; geometric expectations are protected.
use super::*;

fn partial_overlap(sloped: bool) -> SeamFixture {
    let mut fixture = SeamFixture::default();
    let owner = push_grid(&mut fixture, &OWNER_XS, &SEAM_ROWS, 5.0, 0);
    let lower = push_grid(
        &mut fixture,
        &[-0.46, -0.04, 0.46, 0.96],
        &SEAM_ROWS,
        5.0,
        1,
    );
    if sloped {
        for p in &mut fixture.points {
            p.z = 5.0 + 0.25 * p.x + 0.15 * p.y;
        }
    }
    fixture.pairs = (0..SEAM_ROWS.len())
        .map(|row| (owner[row][3], lower[row][0]))
        .collect();
    fixture.cameras = vec![Some(identity_camera(0.0)), Some(identity_camera(0.4))];
    fixture
}

fn xyz(p: &Point3) -> Vector3<f64> {
    Vector3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z))
}

fn assert_edge_conforming(points: &[Point3], triangles: &[MeshTriangle]) {
    let referenced: BTreeSet<_> = triangles.iter().flat_map(|t| [t.a, t.b, t.c]).collect();
    for t in triangles {
        for (a, b) in [(t.a, t.b), (t.b, t.c), (t.c, t.a)] {
            let start = xyz(&points[a]);
            let direction = xyz(&points[b]) - start;
            assert!(direction.norm_squared() > 1.0e-14);
            for &v in &referenced {
                if v == a || v == b {
                    continue;
                }
                let delta = xyz(&points[v]) - start;
                let fraction = delta.dot(&direction) / direction.norm_squared();
                assert!(
                    !(fraction > 1.0e-6
                        && fraction < 1.0 - 1.0e-6
                        && (delta - direction * fraction).norm() < 1.0e-6),
                    "vertex {v} is a T-junction inside retained edge {a}..{b}"
                );
            }
        }
    }
}

fn owned_clip_contract(sloped: bool) {
    let mut fixture = partial_overlap(sloped);
    // Tiny test oracle only. Production owns the input buffers and must not
    // duplicate a complete source mesh merely to represent sparse lineage.
    let original_points = fixture.points.clone();
    let original_triangles = fixture.triangles.clone();
    let source_image: Vec<_> = original_triangles
        .iter()
        .map(|t| fixture.projected(t))
        .collect();
    let mut union: f64 = source_image.iter().map(|c| signed_area_2d(c)).sum();
    for (first, a) in original_triangles.iter().enumerate() {
        for (second, b) in original_triangles.iter().enumerate().skip(first + 1) {
            if fixture.membership[a.a] != fixture.membership[b.a] {
                union -= intersection_area(source_image[first], source_image[second]);
            }
        }
    }
    fixture.points.reserve(1024);
    let allocation = fixture.points.as_ptr();
    let output = merge_fused_patches_owned(
        fixture.points,
        fixture.triangles,
        &fixture.membership,
        &fixture.pairs,
        &fixture.cameras,
        &fixture.origins,
    )
    .expect("supported linked observed overlap");
    assert_eq!(
        output.points.as_ptr(),
        allocation,
        "available owned capacity must be reused"
    );
    assert!(
        output.points.len() > original_points.len(),
        "this boundary crosses source interiors"
    );
    assert_eq!(
        output.derived_vertices.len(),
        output.points.len() - original_points.len()
    );
    for (before, after) in original_points.iter().zip(&output.points) {
        assert_eq!(
            (before.x, before.y, before.z, before.r, before.g, before.b),
            (after.x, after.y, after.z, after.r, after.g, after.b)
        );
    }
    let mut derived = BTreeSet::new();
    for record in &output.derived_vertices {
        assert!(derived.insert(record.point_index), "duplicate lineage");
        assert!(
            record.point_index >= original_points.len() && record.point_index < output.points.len()
        );
        let parent = original_triangles
            .get(record.source_triangle)
            .expect("immutable pre-clip source identity");
        assert!(record
            .barycentric
            .iter()
            .all(|w| w.is_finite() && *w >= -1.0e-7 && *w <= 1.0 + 1.0e-7));
        assert!((record.barycentric.iter().sum::<f64>() - 1.0).abs() < 1.0e-7);
        let expected: Vector3<f64> = [parent.a, parent.b, parent.c]
            .into_iter()
            .zip(record.barycentric)
            .map(|(v, w)| xyz(&original_points[v]) * w)
            .sum();
        assert!((expected - xyz(&output.points[record.point_index])).norm() < 1.0e-6);
    }
    let camera = fixture.cameras[0].unwrap();
    let projected: Vec<_> = output
        .triangles
        .iter()
        .map(|t| {
            [t.a, t.b, t.c].map(|v| {
                camera
                    .project(xyz(&output.points[v]))
                    .expect("observed in owner image")
                    .0
            })
        })
        .collect();
    let area: f64 = projected
        .iter()
        .map(|c| {
            assert!(signed_area_2d(c) > 0.0);
            signed_area_2d(c)
        })
        .sum();
    assert!(
        (area - union).abs() < 1.0e-6 * union,
        "observed union coverage {area} vs {union}"
    );
    for first in 0..projected.len() {
        for second in first + 1..projected.len() {
            assert!(intersection_area(projected[first], projected[second]) < 1.0e-9);
        }
    }
    assert_eq!(
        vertex_components(&output.triangles, output.points.len()).0,
        1
    );
    assert_edge_conforming(&output.points, &output.triangles);
}

#[test]
fn flat_overlap_uses_owned_buffers_and_sparse_source_triangle_lineage() {
    owned_clip_contract(false);
}

#[test]
fn sloped_overlap_reconstructs_derived_world_positions_from_immutable_parents() {
    owned_clip_contract(true);
}
