// Independently protected actual native SurfaceMeshArtifact format, before repair.
use super::*;

fn mesh() -> SurfaceMeshArtifact {
    let mut mesh = Fixture::new().mesh();
    mesh.triangles = mesh.triangles.repeat(8192);
    mesh.validate().unwrap();
    mesh
}

#[test]
fn native_surface_mesh_controlled_writer_preserves_actual_pretty_newline_bytes() {
    let mesh = mesh();
    let expected = mesh.to_json();
    assert!(expected.ends_with('\n'));
    let actual = mesh.to_json_with_cancel(|| Ok(())).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn native_surface_mesh_writer_cancels_inside_actual_geometry_output() {
    let mesh = mesh();
    for after in [2, 4] {
        let mut polls = 0;
        let result = mesh.to_json_with_cancel(|| {
            polls += 1;
            if polls > after {
                Err("acceptance native mesh write canceled".into())
            } else {
                Ok(())
            }
        });
        assert!(
            result.is_err(),
            "Native mesh returned output despite cancellation"
        );
        assert!(result.unwrap_err().contains("canceled"));
        assert!(polls > after);
    }
}

#[test]
fn native_surface_mesh_writer_rejects_nonfinite_geometry() {
    let mut mesh = mesh();
    mesh.points[0].x = f32::NAN;
    assert!(mesh.to_json_with_cancel(|| Ok(())).is_err());
}
