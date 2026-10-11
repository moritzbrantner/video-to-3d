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

#[test]
fn native_surface_mesh_cancels_after_validation_inside_json_emission() {
    let mesh = mesh();
    let before = mesh.to_json();
    let mut validation_polls = 0;
    mesh.validate_with_cancel(|| {
        validation_polls += 1;
        Ok(())
    })
    .unwrap();
    let mut total_polls = 0;
    assert_eq!(
        mesh.to_json_with_cancel(|| {
            total_polls += 1;
            Ok(())
        })
        .unwrap(),
        before
    );
    assert!(total_polls > validation_polls + 4);
    for cancel_at in [
        validation_polls + (total_polls - validation_polls) / 2,
        total_polls - 1,
    ] {
        assert!(cancel_at > validation_polls);
        let mut polls = 0;
        let result = mesh.to_json_with_cancel(|| {
            polls += 1;
            if polls >= cancel_at {
                Err("acceptance native mesh emission canceled".into())
            } else {
                Ok(())
            }
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("canceled"));
        assert!(polls >= cancel_at && polls < total_polls);
        assert_eq!(
            mesh.to_json(),
            before,
            "Canceled emission mutated the source mesh"
        );
    }
}
