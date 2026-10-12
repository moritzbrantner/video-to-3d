//! Independent repair acceptance for #174 review4243558328, frozen9f7e485.
use super::*;

fn triangle_work() -> (Vec<Point3>, Vec<MeshTriangle>) {
    let points = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)]
        .into_iter()
        .map(|(x, y, z)| Point3 {
            x,
            y,
            z,
            confidence: 0.9,
            r: 128,
            g: 128,
            b: 128,
        })
        .collect();
    // Actual prepass triangle operations, not a staged reconstruction substitute.
    // Repeated topology keeps the independent geometric oracle simple.
    let triangles = vec![
        MeshTriangle {
            a: 0,
            b: 1,
            c: 2,
            confidence: 0.9
        };
        4096
    ];
    (points, triangles)
}

#[test]
fn controlled_fusion_prepass_preserves_actual_geometry_evidence() {
    let (points, triangles) = triangle_work();
    let reference = vertex_surface_evidence(&points, &triangles);
    let mut polls = 0;
    let mut poll = || {
        polls += 1;
        false
    };
    let actual = vertex_surface_evidence_with_control(
        &points,
        &triangles,
        &mut ReconstructionControl::new(&mut poll),
    )
    .unwrap();
    assert!(polls > 4, "no work-boundary polling");
    assert_eq!(actual.normals, reference.normals);
    assert_eq!(actual.local_scales, reference.local_scales);
    assert_eq!(actual.incident_triangles, reference.incident_triangles);
    assert_eq!(actual.global_edge_scale, reference.global_edge_scale);
    assert_eq!(actual.global_edge_scale, Some(1.));
    assert_eq!(actual.normals, vec![Some(Vector3::new(0., 0., 1.)); 3]);
    assert_eq!(
        actual.incident_triangles,
        vec![(0..triangles.len()).collect::<Vec<_>>(); 3]
    );
}

#[test]
fn fusion_prepass_aborts_after_entry_before_accepting_partial_evidence() {
    let (points, triangles) = triangle_work();
    for cancel_at in [2, 4] {
        let mut polls = 0;
        let mut poll = || {
            polls += 1;
            polls >= cancel_at
        };
        let result = vertex_surface_evidence_with_control(
            &points,
            &triangles,
            &mut ReconstructionControl::new(&mut poll),
        );
        assert!(
            result.is_err(),
            "prepass returned evidence after cancellation"
        );
        assert!(polls >= cancel_at);
    }
}
