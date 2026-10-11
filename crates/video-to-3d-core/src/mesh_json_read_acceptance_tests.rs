// Independent large-input parse cancellation, authored before its repair.
use super::*;

fn large_mesh_document() -> String {
    let mut mesh = Fixture::new().mesh();
    // Keep existing accepted vertices/provenance; repeat their valid triangles
    // for a genuinely large geometry JSON payload, not whitespace/poll padding.
    mesh.triangles = mesh.triangles.repeat(8192);
    mesh.validate().unwrap();
    serde_json::to_string(&mesh).unwrap()
}

#[test]
fn mesh_json_noop_cancel_parse_preserves_existing_document_semantics() {
    let document = large_mesh_document();
    let expected = SurfaceMeshArtifact::from_json(&document).unwrap();
    let actual = SurfaceMeshArtifact::from_json_with_cancel(&document, || Ok(())).unwrap();
    assert_eq!(
        serde_json::to_string(&actual).unwrap(),
        serde_json::to_string(&expected).unwrap()
    );
    assert_eq!(actual.points, expected.points);
    assert_eq!(actual.triangles, expected.triangles);
}

#[test]
fn mesh_json_cancellation_during_actual_reads_precedes_trailing_syntax_error() {
    let mut document = large_mesh_document();
    document.push_str(" trailing-invalid-json");
    assert!(SurfaceMeshArtifact::from_json(&document).is_err());
    for after in [2, 4] {
        let mut polls = 0;
        let error = SurfaceMeshArtifact::from_json_with_cancel(&document, || {
            polls += 1;
            if polls > after {
                Err("acceptance mesh JSON read canceled".into())
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert!(
            error.contains("canceled"),
            "Cancellation did not precede trailing parse error; polls={polls}: {error}"
        );
        assert!(polls > after);
    }
}
