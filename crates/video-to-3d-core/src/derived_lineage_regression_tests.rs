//! Implementation regressions for the #137 review findings: derived (clipped) points
//! inherit their source corner's provenance in confidence/collision consumers, lineage
//! cannot point across reference patches, and derived streams use their own
//! interchange format version.
use super::*;
use crate::coarse_collision::{CoarseCollider, CoarseColliderOptions};
use crate::evidence_interchange::{
    decode_evidence, encode_evidence, InterchangeLimits,
    EVIDENCE_INTERCHANGE_DERIVED_FORMAT_VERSION, EVIDENCE_INTERCHANGE_FORMAT_VERSION,
};
use crate::geometry_confidence::GeometryConfidenceField;
use crate::DerivedVertexSource;

/// Point 8 at the midpoint of points 0 and 1 (patch of frame 0), splitting triangle 0.
fn derived() -> (Fixture, [DerivedVertexSource; 1]) {
    let mut fixture = Fixture::new();
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
        ..a
    });
    fixture.triangles[0] = triangle(0, 8, 2);
    fixture.triangles.push(triangle(8, 1, 2));
    let record = DerivedVertexSource {
        point_index: 8,
        source_triangle: 0,
        source_corners: [0, 1, 2],
        barycentric: [0.5, 0.5, 0.0],
    };
    (fixture, [record])
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
fn derived_points_inherit_source_confidence_and_collision_provenance() {
    let (fixture, records) = derived();
    let evidence = view(&fixture, &records).unwrap();
    let field = GeometryConfidenceField::from_evidence(&evidence);
    assert_eq!(field.point_confidence(8), field.point_confidence(0));
    assert!(field.point_confidence(8) > 0.0);
    let collider =
        CoarseCollider::from_evidence(&evidence, &field, CoarseColliderOptions::default());
    // Only provenance decides here: derived corners must not make a triangle unobserved.
    assert_eq!(collider.excluded.unobserved_provenance, 0, "{collider:?}");
}

#[test]
fn lineage_across_reference_patches_is_rejected() {
    let (fixture, mut records) = derived();
    // Points 0 and 4 belong to different reference patches (frames 0 and 1).
    records[0].source_corners = [0, 4, 2];
    let mut moved = fixture;
    let (a, b) = (moved.points[0], moved.points[4]);
    moved.points[8].x = (a.x + b.x) * 0.5;
    moved.points[8].y = (a.y + b.y) * 0.5;
    moved.points[8].z = (a.z + b.z) * 0.5;
    let error = view(&moved, &records).unwrap_err();
    assert!(error.contains("one reference patch"), "{error}");
}

#[test]
fn derived_streams_use_their_own_interchange_version() {
    let (fixture, records) = derived();
    let mut bytes = Vec::new();
    encode_evidence(&view(&fixture, &records).unwrap(), &mut bytes).unwrap();
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        EVIDENCE_INTERCHANGE_DERIVED_FORMAT_VERSION
    );
    // Relabeling a derived stream as the original layout fails closed.
    let mut relabeled = bytes.clone();
    relabeled[8..12].copy_from_slice(&EVIDENCE_INTERCHANGE_FORMAT_VERSION.to_le_bytes());
    assert!(decode_evidence(&mut relabeled.as_slice(), InterchangeLimits::default()).is_err());
    let decoded = decode_evidence(&mut bytes.as_slice(), InterchangeLimits::default()).unwrap();
    assert_eq!(decoded.derived_vertices, records);

    // Without derived points the stream stays version 1.
    let plain = Fixture::new();
    let mut bytes = Vec::new();
    encode_evidence(&view(&plain, &[]).unwrap(), &mut bytes).unwrap();
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        EVIDENCE_INTERCHANGE_FORMAT_VERSION
    );
}
