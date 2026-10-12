//! Synthetic source correspondence at the existing validated artifact/bake
//! boundary. API/record names are routine proposals, independently translatable.
use super::*;
use crate::{DerivedVertexSource, EvidenceOrigin, EvidenceRange};

fn lineage() -> DerivedVertexSource {
    DerivedVertexSource {
        point_index: 8,
        source_triangle: 0,
        source_corners: [0, 1, 2],
        barycentric: [0.5, 0.5, 0.0],
    }
}

fn derived_fixture() -> Fixture {
    let mut fixture = Fixture::new();
    // The source triangle has constant camera depth: midpoint source UV is
    // unambiguous without prescribing a projective interpolation algorithm.
    for (index, (x, y)) in [(0, (-0.25, -0.25)), (1, (0.25, -0.25)), (2, (-0.25, 0.25))] {
        fixture.points[index].x = x;
        fixture.points[index].y = y;
        fixture.points[index].z = 2.0;
    }
    let (a, b) = (fixture.points[0], fixture.points[1]);
    fixture.points.push(Point3 {
        x: (a.x + b.x) * 0.5,
        y: (a.y + b.y) * 0.5,
        z: (a.z + b.z) * 0.5,
        confidence: a.confidence.min(b.confidence),
        r: a.r / 2 + b.r / 2,
        g: a.g,
        b: a.b,
    });
    fixture.triangles[0] = triangle(0, 8, 2);
    fixture.triangles.push(triangle(8, 1, 2));
    // Eight original grid sites and provenance points stay eight: the ninth
    // vertex has parent-derived correspondence, not invented pixel evidence.
    fixture
}

fn view<'a>(
    fixture: &'a Fixture,
    records: &'a [DerivedVertexSource],
) -> Result<ReconstructionEvidenceView<'a>, String> {
    ReconstructionEvidenceView::new_with_derived_vertices(
        ReconstructionProviderDescriptor::classic(),
        EvidenceScale::ArbitraryMonocular,
        fixture.cameras.clone(),
        fixture.regions.clone(),
        &fixture.points,
        &fixture.triangles,
        records,
    )
}

#[test]
fn stored_derived_mesh_preserves_lineage_and_independent_observation_counts() {
    let fixture = derived_fixture();
    let records = [lineage()];
    let evidence = view(&fixture, &records).expect("valid derived geometry");
    assert_eq!(evidence.points.as_ptr(), fixture.points.as_ptr());
    assert_eq!(evidence.triangles.as_ptr(), fixture.triangles.as_ptr());
    assert_eq!(evidence.summary().geometric_points, 7);
    assert_eq!(evidence.summary().revalidated_completion_points, 1);
    let artifact = SurfaceMeshArtifact::from_evidence(&evidence, &fixture.sites).unwrap();
    let document = artifact.to_json();
    let restored = SurfaceMeshArtifact::from_json(&document).unwrap();
    assert_eq!(restored.to_json(), document);
    let restored_evidence = restored.evidence().unwrap();
    assert_eq!(
        serde_json::to_value(&restored_evidence.provider).unwrap(),
        serde_json::to_value(evidence.provider.clone()).unwrap()
    );
    assert_eq!(restored_evidence.scale, evidence.scale);
    assert_eq!(
        serde_json::to_value(&restored_evidence.cameras).unwrap(),
        serde_json::to_value(&evidence.cameras).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&restored_evidence.regions).unwrap(),
        serde_json::to_value(&fixture.regions).unwrap()
    );
    assert_eq!(
        serde_json::to_value(restored_evidence.derived_vertices).unwrap(),
        serde_json::to_value(&records).unwrap()
    );
    assert_eq!(restored_evidence.summary().geometric_points, 7);
    assert_eq!(restored_evidence.summary().revalidated_completion_points, 1);
    let mut interchange = Vec::new();
    crate::evidence_interchange::encode_evidence(&restored.evidence().unwrap(), &mut interchange)
        .unwrap();
    let decoded = crate::evidence_interchange::decode_evidence(
        &mut interchange.as_slice(),
        crate::evidence_interchange::InterchangeLimits::default(),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(decoded.view().unwrap().derived_vertices).unwrap(),
        serde_json::to_value(&records).unwrap()
    );
}

#[test]
fn invalid_derived_parent_geometry_and_cycles_fail_shared_validation() {
    let fixture = derived_fixture();
    for (label, record) in [
        (
            "parent index",
            DerivedVertexSource {
                source_corners: [0, 1, usize::MAX],
                ..lineage()
            },
        ),
        (
            "cycle",
            DerivedVertexSource {
                source_corners: [0, 8, 2],
                ..lineage()
            },
        ),
        (
            "outside barycentric",
            DerivedVertexSource {
                barycentric: [-0.5, 1.5, 0.0],
                ..lineage()
            },
        ),
        (
            "wrong sum",
            DerivedVertexSource {
                barycentric: [0.5, 0.25, 0.0],
                ..lineage()
            },
        ),
        (
            "nonfinite",
            DerivedVertexSource {
                barycentric: [f64::NAN, 0.5, 0.0],
                ..lineage()
            },
        ),
        (
            "wrong position",
            DerivedVertexSource {
                barycentric: [0.25, 0.75, 0.0],
                ..lineage()
            },
        ),
    ] {
        assert!(
            view(&fixture, &[record]).is_err(),
            "{label} must reject before bake/export"
        );
    }
    assert!(
        view(&fixture, &[lineage(), lineage()]).is_err(),
        "duplicate lineage"
    );
    let mut relabeled = derived_fixture();
    relabeled.regions.push(SurfaceEvidenceRegion {
        origin: EvidenceOrigin::GeometricMultiView,
        reference_frame: Some(0),
        source_frames: vec![1, 2],
        points: EvidenceRange::new(8, 1),
    });
    assert!(
        view(&relabeled, &[lineage()]).is_err(),
        "a derived vertex cannot be a new independent observation"
    );
}

#[test]
fn stored_derived_mesh_bakes_parent_correspondence_without_fabricated_grid_sites() {
    let dir = TempDir::new("derived-lineage-137");
    let fixture = derived_fixture();
    let records = [lineage()];
    let artifact =
        SurfaceMeshArtifact::from_evidence(&view(&fixture, &records).unwrap(), &fixture.sites)
            .unwrap();
    let frames: Vec<_> = fixture
        .images
        .iter()
        .map(|(frame_index, rgba)| SampledFrame {
            frame_index: *frame_index,
            timestamp_seconds: *frame_index as f64 / 30.0,
            width: WIDTH,
            height: HEIGHT,
            rgba,
        })
        .collect();
    let keyframes = path("inputs/keyframes.json");
    write_keyframes_artifact(&dir.0, &keyframes, &frames).unwrap();
    let mesh = path("inputs/mesh.json");
    fs::write(mesh.resolve(&dir.0), artifact.to_json()).unwrap();
    let inputs = (keyframes, mesh);
    let first = bake(&dir.0, &inputs);
    assert_eq!(
        first
            .artifact
            .materials
            .iter()
            .map(|m| m.triangles.len())
            .sum::<usize>(),
        5
    );
    assert_eq!(first.artifact.fallback_triangles.len(), 1);
    let restored =
        SurfaceMeshArtifact::from_json(&fs::read_to_string(inputs.1.resolve(&dir.0)).unwrap())
            .unwrap();
    let images: Vec<_> = fixture
        .images
        .iter()
        .map(
            |(frame_index, rgba)| crate::surface_materials::ReferenceImage {
                frame_index: *frame_index,
                width: WIDTH,
                height: HEIGHT,
                rgba,
            },
        )
        .collect();
    let material_bake = crate::surface_materials::bake_surface_materials(
        &restored.evidence().unwrap(),
        &fixture.sites,
        &images,
    )
    .unwrap();
    let material = material_bake
        .materials
        .iter()
        .find(|m| m.reference_frame == 0)
        .unwrap();
    for (ordinal, &triangle_index) in material.triangles.iter().enumerate() {
        let t = &fixture.triangles[triangle_index];
        for (corner, v) in [t.a, t.b, t.c].into_iter().enumerate() {
            if v != 8 {
                continue;
            }
            let uv = material.corner_uvs[ordinal][corner];
            let pixel = [
                uv[0] * material.texture.width as f32 + material.texture.crop_origin[0] as f32,
                uv[1] * material.texture.height as f32 + material.texture.crop_origin[1] as f32,
            ];
            assert!(
                (pixel[0] - 4.5).abs() < 1.0e-5 && (pixel[1] - 2.5).abs() < 1.0e-5,
                "midpoint source pixel must interpolate original sites, got {pixel:?}"
            );
        }
    }
    let glb =
        crate::textured_glb::encode_textured_glb(&restored.evidence().unwrap(), &material_bake)
            .unwrap();
    assert_exported_geometry(&glb, &fixture.points, fixture.triangles.len());
    let again = bake(&dir.0, &inputs);
    assert_eq!(first.content_hash, again.content_hash);
    assert_eq!(
        fs::read(inputs.1.resolve(&dir.0)).unwrap(),
        artifact.to_json().as_bytes()
    );
    assert_eq!(
        glb,
        crate::textured_glb::encode_textured_glb(&restored.evidence().unwrap(), &material_bake)
            .unwrap()
    );
}

#[test]
fn stored_lineage_tampering_rejects_before_material_baking() {
    let fixture = derived_fixture();
    let records = [lineage()];
    let artifact =
        SurfaceMeshArtifact::from_evidence(&view(&fixture, &records).unwrap(), &fixture.sites)
            .unwrap();
    let document = serde_json::to_value(artifact).unwrap();
    for (field, value) in [
        ("source_corners", serde_json::json!([0, 1, 9999])),
        ("barycentric", serde_json::json!([0.25, 0.75, 0.0])),
        ("point_index", serde_json::json!(0)),
    ] {
        let mut tampered = document.clone();
        tampered["derived_vertices"][0][field] = value;
        let result = SurfaceMeshArtifact::from_json(&tampered.to_string());
        assert!(
            result.is_err(),
            "stored {field} tamper must not be accepted"
        );
    }
}

fn assert_exported_geometry(glb: &[u8], points: &[Point3], triangle_count: usize) {
    assert_eq!(&glb[..4], b"glTF");
    let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    let json: serde_json::Value = serde_json::from_slice(&glb[20..20 + json_len]).unwrap();
    let bin = &glb[20 + json_len + 8..];
    let mut count = 0;
    let mut derived_seen = false;
    for primitive in json["meshes"][0]["primitives"].as_array().unwrap() {
        let indices = &json["accessors"][primitive["indices"].as_u64().unwrap() as usize];
        count += indices["count"].as_u64().unwrap() as usize / 3;
        let accessor =
            &json["accessors"][primitive["attributes"]["POSITION"].as_u64().unwrap() as usize];
        assert_eq!(accessor["componentType"], 5126);
        assert_eq!(accessor["type"], "VEC3");
        let buffer = &json["bufferViews"][accessor["bufferView"].as_u64().unwrap() as usize];
        let offset = buffer["byteOffset"].as_u64().unwrap_or(0) as usize
            + accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
        let vertices = accessor["count"].as_u64().unwrap() as usize;
        for raw in bin[offset..offset + vertices * 12].chunks_exact(12) {
            let position: [f32; 3] = std::array::from_fn(|axis| {
                f32::from_le_bytes(raw[axis * 4..axis * 4 + 4].try_into().unwrap())
            });
            assert!(
                points.iter().any(|p| position == [p.x, p.y, p.z]),
                "export invented {position:?}"
            );
            derived_seen |= position == [points[8].x, points[8].y, points[8].z];
        }
    }
    assert_eq!(count, triangle_count);
    assert!(
        derived_seen,
        "derived accepted position missing from actual GLB"
    );
    assert_eq!(
        json["asset"]["extras"]["video_to_3d"]["provider"]["id"],
        "video-to-3d-core/classic"
    );
}
