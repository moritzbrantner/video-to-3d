//! Independent serialization cancellation contract, before producer repair.
use super::*;

use super::native_stage_acceptance_tests::golden;

// Independent before serialization repair; real multiview stage outputs.
fn serialization_fixture() -> (DenseStageArtifact, MeshStageArtifact) {
    let request = golden::request();
    let cameras = reconstruct_cameras(&request).unwrap();
    let dense = reconstruct_dense(&request, &cameras).unwrap();
    let mesh = reconstruct_mesh(&cameras, &dense).unwrap();
    assert!(!dense.dense.points.is_empty());
    assert!(!dense.dense.grid_sites.is_empty());
    assert!(!dense.dense.point_attributes.is_empty());
    assert!(!mesh.mesh.triangles.is_empty());
    // Frozen9f independently reproduces 629 points/9 triangles and zero source
    // corners: this Lateral fixture has no canonicalized corner replacements.
    assert!(mesh.mesh.triangle_sources.is_empty());
    (dense, mesh)
}

#[test]
fn dense_serialization_false_cancel_preserves_actual_multiarray_bytes() {
    let (dense, _) = serialization_fixture();
    let expected = dense.to_json();
    let mut polls = 0;
    let actual = dense
        .to_json_with_cancel(|| {
            polls += 1;
            false
        })
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(DenseStageArtifact::from_json(&actual).unwrap(), dense);
    assert!(
        polls > dense.dense.points.len(),
        "pre/post checks cannot cancel geometry serialization"
    );
}

#[test]
fn mesh_serialization_false_cancel_preserves_actual_multiarray_bytes() {
    let (_, mesh) = serialization_fixture();
    let expected = mesh.to_json();
    let mut polls = 0;
    let actual = mesh
        .to_json_with_cancel(|| {
            polls += 1;
            false
        })
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(MeshStageArtifact::from_json(&actual).unwrap(), mesh);
    assert!(
        polls > mesh.mesh.triangles.len(),
        "pre/post checks cannot cancel geometry serialization"
    );
}

#[test]
fn dense_serialization_cancels_during_real_emission_without_output() {
    let (dense, _) = serialization_fixture();
    let original = dense.to_json();
    let mut complete_polls = 0;
    dense
        .to_json_with_cancel(|| {
            complete_polls += 1;
            false
        })
        .unwrap();
    assert!(complete_polls > 64);
    for cancel_at in [2, 4, 32, complete_polls / 2, complete_polls - 1] {
        let mut polls = 0;
        let result = dense.to_json_with_cancel(|| {
            polls += 1;
            polls >= cancel_at
        });
        assert!(
            result.is_err(),
            "dense returned output after cancellation at {cancel_at}"
        );
        assert!(result.unwrap_err().to_string().contains("canceled"));
        assert!(
            polls >= cancel_at && polls < complete_polls,
            "stop before finishing the document"
        );
        assert_eq!(dense.to_json(), original);
    }
}

#[test]
fn mesh_serialization_cancels_during_real_emission_without_output() {
    let (_, mesh) = serialization_fixture();
    let original = mesh.to_json();
    let mut complete_polls = 0;
    mesh.to_json_with_cancel(|| {
        complete_polls += 1;
        false
    })
    .unwrap();
    assert!(complete_polls > 64);
    for cancel_at in [2, 4, 32, complete_polls / 2, complete_polls - 1] {
        let mut polls = 0;
        let result = mesh.to_json_with_cancel(|| {
            polls += 1;
            polls >= cancel_at
        });
        assert!(
            result.is_err(),
            "mesh returned output after cancellation at {cancel_at}"
        );
        assert!(result.unwrap_err().to_string().contains("canceled"));
        assert!(
            polls >= cancel_at && polls < complete_polls,
            "stop before finishing the document"
        );
        assert_eq!(mesh.to_json(), original);
    }
}

#[test]
fn dense_serialization_validation_rejects_nonfinite_payload_without_output() {
    let (mut dense, _) = serialization_fixture();
    dense.dense.points[0].x = f32::NAN;
    let result = dense.to_json_with_cancel(|| false);
    assert!(
        result.is_err(),
        "NaN-to-null conversion is not validated artifact output"
    );
    assert!(result.unwrap_err().to_string().contains("finite"));
}

#[test]
fn mesh_serialization_validation_rejects_nonfinite_payload_without_output() {
    let (_, mut mesh) = serialization_fixture();
    mesh.mesh.triangles[0].confidence = f32::INFINITY;
    let result = mesh.to_json_with_cancel(|| false);
    assert!(
        result.is_err(),
        "nonfinite geometry cannot become successful JSON"
    );
    assert!(result.unwrap_err().to_string().contains("finite"));
}
