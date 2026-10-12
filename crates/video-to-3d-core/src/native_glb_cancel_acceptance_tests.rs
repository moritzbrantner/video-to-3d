//! Supplemental independent coverage authored after production repair 88ce75d.
use super::*;
use crate::textured_glb::encode_textured_glb_with_cancel;

fn native_fallback(fixture: &Fixture) -> SurfaceMaterialBake {
    SurfaceMaterialBake {
        schema_version: SURFACE_MATERIAL_BAKE_SCHEMA_VERSION,
        point_count: fixture.points.len(),
        triangle_count: fixture.triangles.len(),
        materials: Vec::new(),
        fallback: FallbackSurface {
            triangles: (0..fixture.triangles.len()).collect(),
            reasons: FallbackReasons {
                missing_reference_image: fixture.triangles.len(),
                ..Default::default()
            },
        },
        canonicalized_same_reference_triangles: 0,
    }
}

#[test]
fn native_untextured_cancel_adapter_preserves_public_glb_bytes() {
    let fixture = Fixture::new();
    let evidence = fixture.evidence();
    evidence.validate().unwrap();
    let bake = native_fallback(&fixture);
    let expected = encode_textured_glb(&evidence, &bake).unwrap();
    let mut polls = 0;
    let actual = encode_textured_glb_with_cancel(&evidence, &bake, || {
        polls += 1;
        false
    })
    .unwrap();
    assert_eq!(actual, expected);
    let (json, _) = glb_parts(&actual);
    assert_eq!(
        json["asset"]["extras"]["video_to_3d"]["fallback_triangles"],
        fixture.triangles.len()
    );
    assert!(polls > fixture.triangles.len() * 2 + 1);
}

#[test]
fn native_untextured_export_cancels_inside_triangle_and_attribute_work() {
    let fixture = Fixture::new();
    let evidence = fixture.evidence();
    evidence.validate().unwrap();
    let bake = native_fallback(&fixture);
    // Source-audited phases: entry + assignment traversal, fallback triangle
    // traversal, then position components and actual accessor writes. These
    // boundaries are semantic test inputs, not a fixed total callback count.
    let triangles = fixture.triangles.len();
    for after in [
        2,
        4,
        1 + triangles + 2,
        1 + triangles * 2 + 4,
        1 + triangles * 2 + fixture.points.len() * 3 + 2,
    ] {
        let mut polls = 0;
        let result = encode_textured_glb_with_cancel(&evidence, &bake, || {
            polls += 1;
            polls > after
        });
        assert!(
            result.is_err(),
            "Export accepted output after cancellation at {after}"
        );
        assert!(result.unwrap_err().contains("canceled"));
        assert!(polls > after);
    }
}
