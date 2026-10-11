//! Acceptance contract for the owned native-process evidence interchange (#78).
//!
//! Binary layout assumed by these tests (the contract the implementation must satisfy):
//!
//! | offset              | bytes      | content                                          |
//! |---------------------|------------|--------------------------------------------------|
//! | 0                   | 8          | `EVIDENCE_INTERCHANGE_MAGIC`                     |
//! | 8                   | 4          | format version, u32 LE                           |
//! | 12                  | 4          | JSON header length `h`, u32 LE                   |
//! | 16                  | h          | JSON header object (top-level `points` and       |
//! |                     |            | `triangles` integer counts, regions carrying     |
//! |                     |            | `origin` as the snake_case `EvidenceOrigin`)     |
//! | 16 + h              | 19 * n     | points: x, y, z, confidence f32 LE; r, g, b u8   |
//! | ... + 19 * n        | 16 * t     | triangles: a, b, c u32 LE; confidence f32 LE     |
//! | ... + 16 * t        | 8 * n      | optional attributes: reciprocal_consistency,     |
//! |                     |            | depth_margin f32 LE (only when present)          |
//!
//! Error keywords relied on: "magic", "version", "limit", "truncated", "trailing".
//! Invalid provider output is expected to surface the message of the shared
//! `ReconstructionEvidenceView::validate` rule it violates.

use super::*;
use crate::{
    EvidenceCamera, EvidenceCameraAuthority, EvidenceOrigin, EvidencePointAttributes,
    EvidenceRange, EvidenceScale, MeshTriangle, Point3, ReconstructionEvidenceView,
    ReconstructionProviderClass, ReconstructionProviderDescriptor, SurfaceEvidenceRegion,
};

const PREFIX_BYTES: usize = 16;
const POINT_BYTES: usize = 19;
const TRIANGLE_BYTES: usize = 16;
const ATTRIBUTE_BYTES: usize = 8;

fn point(x: f32, y: f32, z: f32, confidence: f32, rgb: [u8; 3]) -> Point3 {
    Point3 {
        x,
        y,
        z,
        confidence,
        r: rgb[0],
        g: rgb[1],
        b: rgb[2],
    }
}

fn triangle(a: usize, b: usize, c: usize, confidence: f32) -> MeshTriangle {
    MeshTriangle {
        a,
        b,
        c,
        confidence,
    }
}

fn camera(
    frame_index: usize,
    authority: EvidenceCameraAuthority,
    rotation: [f32; 9],
    translation: [f32; 3],
    confidence: Option<f32>,
    error: Option<f32>,
) -> EvidenceCamera {
    EvidenceCamera {
        frame_index,
        authority,
        rotation,
        translation,
        confidence,
        median_reprojection_error_pixels: error,
    }
}

const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
// Values whose shortest decimal form is not short, to prove exact f32 round trips.
const ROTATED: [f32; 9] = [
    0.70710677,
    -0.70710677,
    0.0,
    0.70710677,
    0.70710677,
    0.0,
    0.0,
    0.0,
    1.0,
];

struct Buffers {
    points: Vec<Point3>,
    triangles: Vec<MeshTriangle>,
    attributes: Vec<EvidencePointAttributes>,
}

fn classic_buffers() -> Buffers {
    Buffers {
        points: vec![
            point(0.0, 0.0, 1.0, 0.9, [255, 0, 0]),
            point(1.0, 0.0, 1.0, 0.8, [0, 255, 0]),
            point(0.0, 1.0, 1.25, 0.1, [0, 0, 255]),
            point(-0.333_333_34, 2.5, 3.75, 1.0, [12, 34, 56]),
            point(1e-7, -4.0, 0.5, 0.0, [200, 100, 50]),
        ],
        triangles: vec![triangle(0, 1, 2, 0.75), triangle(2, 3, 4, 0.333_333_34)],
        attributes: vec![
            EvidencePointAttributes {
                reciprocal_consistency: 1.0,
                depth_margin: 0.5,
            },
            EvidencePointAttributes {
                reciprocal_consistency: 0.25,
                depth_margin: 0.0,
            },
            EvidencePointAttributes {
                reciprocal_consistency: 0.666_666_7,
                depth_margin: 0.125,
            },
            EvidencePointAttributes {
                reciprocal_consistency: 0.0,
                depth_margin: 1.0,
            },
            EvidencePointAttributes {
                reciprocal_consistency: 0.5,
                depth_margin: 0.999,
            },
        ],
    }
}

fn classic_view(buffers: &Buffers) -> ReconstructionEvidenceView<'_> {
    ReconstructionEvidenceView::new(
        ReconstructionProviderDescriptor::classic(),
        EvidenceScale::ArbitraryMonocular,
        vec![
            camera(
                0,
                EvidenceCameraAuthority::CalibratedSeed,
                IDENTITY,
                [0.0, 0.0, 0.0],
                Some(0.9),
                Some(0.5),
            ),
            camera(
                1,
                EvidenceCameraAuthority::CalibratedSeed,
                ROTATED,
                [0.6, -0.8, 0.0],
                Some(0.9),
                Some(0.5),
            ),
            camera(
                4,
                EvidenceCameraAuthority::RegisteredGeometry,
                ROTATED,
                [1.5, 0.25, -3.0],
                Some(0.42),
                None,
            ),
        ],
        vec![
            SurfaceEvidenceRegion {
                origin: EvidenceOrigin::GeometricMultiView,
                reference_frame: Some(0),
                source_frames: vec![1, 4],
                points: EvidenceRange::new(0, 3),
            },
            SurfaceEvidenceRegion {
                origin: EvidenceOrigin::RevalidatedCompletion,
                reference_frame: Some(1),
                source_frames: vec![4, 0],
                points: EvidenceRange::new(3, 2),
            },
        ],
        &buffers.points,
        &buffers.triangles,
    )
    .expect("valid classic evidence")
    .with_point_attributes(&buffers.attributes)
    .expect("valid point attributes")
}

fn learned_provider() -> ReconstructionProviderDescriptor {
    ReconstructionProviderDescriptor::new(
        "test/native-learned",
        ReconstructionProviderClass::LearnedMultiView,
        Some("0.3.1".into()),
    )
    .expect("provider")
}

fn learned_buffers() -> Buffers {
    Buffers {
        points: vec![
            point(0.0, 0.0, 2.0, 0.6, [10, 20, 30]),
            point(1.0, 0.0, 2.0, 0.7, [40, 50, 60]),
            point(0.0, 1.0, 2.0, 0.8, [70, 80, 90]),
        ],
        triangles: vec![triangle(0, 1, 2, 0.5)],
        attributes: Vec::new(),
    }
}

fn learned_view(buffers: &Buffers) -> ReconstructionEvidenceView<'_> {
    ReconstructionEvidenceView::new(
        learned_provider(),
        EvidenceScale::ProviderLocal,
        vec![
            camera(
                0,
                EvidenceCameraAuthority::ProviderEstimated,
                IDENTITY,
                [0.0, 0.0, 0.0],
                Some(0.8),
                None,
            ),
            camera(
                1,
                EvidenceCameraAuthority::ProviderEstimated,
                ROTATED,
                [0.1, 0.2, 0.3],
                None,
                Some(1.25),
            ),
        ],
        vec![SurfaceEvidenceRegion {
            origin: EvidenceOrigin::LearnedMultiView,
            reference_frame: Some(0),
            source_frames: vec![1],
            points: EvidenceRange::new(0, 3),
        }],
        &buffers.points,
        &buffers.triangles,
    )
    .expect("valid learned evidence")
}

fn encode(view: &ReconstructionEvidenceView<'_>) -> Vec<u8> {
    let mut bytes = Vec::new();
    encode_evidence(view, &mut bytes).expect("encode valid evidence");
    bytes
}

fn decode(bytes: &[u8]) -> Result<OwnedReconstructionEvidence, String> {
    decode_evidence(&mut &bytes[..], InterchangeLimits::default())
}

fn decode_with(
    bytes: &[u8],
    limits: InterchangeLimits,
) -> Result<OwnedReconstructionEvidence, String> {
    decode_evidence(&mut &bytes[..], limits)
}

fn header_len(bytes: &[u8]) -> usize {
    u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize
}

fn header_json(bytes: &[u8]) -> serde_json::Value {
    let len = header_len(bytes);
    serde_json::from_slice(&bytes[PREFIX_BYTES..PREFIX_BYTES + len]).expect("JSON header")
}

/// Rebuild a stream with a replacement JSON header, keeping magic, version and buffers.
fn with_header(bytes: &[u8], header: &serde_json::Value) -> Vec<u8> {
    let payload = &bytes[PREFIX_BYTES + header_len(bytes)..];
    let header = serde_json::to_vec(header).expect("serialize header");
    let mut out = bytes[..12].to_vec();
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(payload);
    out
}

fn points_offset(bytes: &[u8]) -> usize {
    PREFIX_BYTES + header_len(bytes)
}

fn triangles_offset(bytes: &[u8], points: usize) -> usize {
    points_offset(bytes) + POINT_BYTES * points
}

fn attributes_offset(bytes: &[u8], points: usize, triangles: usize) -> usize {
    triangles_offset(bytes, points) + TRIANGLE_BYTES * triangles
}

fn patch_f32(bytes: &mut [u8], offset: usize, value: f32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn assert_err_contains(result: Result<OwnedReconstructionEvidence, String>, needle: &str) {
    match result {
        Ok(_) => panic!("decode must fail closed (expected an error mentioning {needle:?})"),
        Err(error) => assert!(
            error.to_lowercase().contains(&needle.to_lowercase()),
            "error {error:?} should mention {needle:?}"
        ),
    }
}

/// Replace every `"origin": from` in the header (wherever regions are nested).
fn relabel_origins(value: &mut serde_json::Value, from: &str, to: &str) -> usize {
    match value {
        serde_json::Value::Object(map) => {
            let mut changed = 0;
            for (key, child) in map.iter_mut() {
                if key == "origin" && child.as_str() == Some(from) {
                    *child = serde_json::Value::String(to.into());
                    changed += 1;
                } else {
                    changed += relabel_origins(child, from, to);
                }
            }
            changed
        }
        serde_json::Value::Array(items) => items
            .iter_mut()
            .map(|item| relabel_origins(item, from, to))
            .sum(),
        _ => 0,
    }
}

#[test]
fn interchange_constants_pin_magic_and_format_version() {
    assert_eq!(EVIDENCE_INTERCHANGE_FORMAT_VERSION, 1);
    assert_eq!(&EVIDENCE_INTERCHANGE_MAGIC, b"V3DEVID\0");
}

#[test]
fn classic_evidence_round_trips_losslessly_and_deterministically() {
    let buffers = classic_buffers();
    let original = classic_view(&buffers);
    let bytes = encode(&original);

    let decoded = decode(&bytes).expect("decode valid stream");
    assert_eq!(decoded, OwnedReconstructionEvidence::from_view(&original));
    assert_eq!(
        decoded.provider,
        ReconstructionProviderDescriptor::classic()
    );
    assert_eq!(decoded.scale, EvidenceScale::ArbitraryMonocular);
    assert_eq!(decoded.cameras.len(), 3);
    assert_eq!(decoded.regions.len(), 2);
    assert_eq!(
        decoded.regions[1].origin,
        EvidenceOrigin::RevalidatedCompletion
    );
    assert_eq!(decoded.cameras[1].rotation, ROTATED);
    assert_eq!(decoded.points[3].x.to_bits(), (-0.333_333_34f32).to_bits());
    assert_eq!(
        (
            decoded.points[4].r,
            decoded.points[4].g,
            decoded.points[4].b
        ),
        (200, 100, 50)
    );
    assert_eq!(decoded.triangles[1].c, 4);
    assert_eq!(
        decoded.point_attributes.as_deref(),
        Some(buffers.attributes.as_slice())
    );

    let view = decoded.view().expect("decoded evidence validates");
    assert_eq!(view.summary(), original.summary());
    assert_eq!(view.schema_version, original.schema_version);

    let mut again = Vec::new();
    encode_evidence(&view, &mut again).expect("re-encode decoded evidence");
    assert_eq!(
        again, bytes,
        "encoding must be deterministic and byte-identical"
    );
}

#[test]
fn learned_evidence_without_attributes_round_trips() {
    let buffers = learned_buffers();
    let original = learned_view(&buffers);
    assert!(original.point_attributes.is_none());
    let bytes = encode(&original);

    let decoded = decode(&bytes).expect("decode learned stream");
    assert_eq!(decoded, OwnedReconstructionEvidence::from_view(&original));
    assert_eq!(decoded.point_attributes, None);
    assert_eq!(decoded.scale, EvidenceScale::ProviderLocal);
    assert_eq!(decoded.provider, learned_provider());
    assert_eq!(decoded.cameras[1].confidence, None);
    assert_eq!(
        decoded.cameras[1].median_reprojection_error_pixels,
        Some(1.25)
    );

    let view = decoded.view().expect("decoded learned evidence validates");
    assert!(view.point_attributes.is_none());
    assert_eq!(view.summary(), original.summary());
    assert_eq!(view.summary().learned_points, 3);

    // Without attributes no attribute buffer is written.
    assert_eq!(bytes.len(), triangles_offset(&bytes, 3) + TRIANGLE_BYTES);
}

#[test]
fn camera_free_generative_and_empty_evidence_round_trip() {
    let points = vec![point(0.5, 0.5, 0.5, 0.3, [1, 2, 3])];
    let generative = ReconstructionEvidenceView::new(
        ReconstructionProviderDescriptor::new(
            "test/generative",
            ReconstructionProviderClass::GenerativeCompletion,
            None,
        )
        .unwrap(),
        EvidenceScale::ProviderLocal,
        Vec::new(),
        vec![SurfaceEvidenceRegion {
            origin: EvidenceOrigin::GenerativeCompletion,
            reference_frame: None,
            source_frames: Vec::new(),
            points: EvidenceRange::new(0, 1),
        }],
        &points,
        &[],
    )
    .unwrap();
    let decoded = decode(&encode(&generative)).expect("decode generative stream");
    assert_eq!(decoded, OwnedReconstructionEvidence::from_view(&generative));
    assert_eq!(decoded.provider.version, None);
    assert_eq!(
        decoded
            .view()
            .unwrap()
            .summary()
            .generative_completion_points,
        1
    );

    let empty = ReconstructionEvidenceView::new(
        ReconstructionProviderDescriptor::classic(),
        EvidenceScale::Metric,
        Vec::new(),
        Vec::new(),
        &[],
        &[],
    )
    .unwrap();
    let decoded = decode(&encode(&empty)).expect("decode empty stream");
    assert!(decoded.points.is_empty() && decoded.triangles.is_empty());
    assert_eq!(decoded.scale, EvidenceScale::Metric);
    decoded.view().expect("empty evidence validates");
}

#[test]
fn owned_view_borrows_owned_buffers_without_copying() {
    // In-process borrowed views are covered by
    // `reconstruction_evidence::tests::provider_boundary_borrows_geometry_buffers`.
    let buffers = classic_buffers();
    let owned = OwnedReconstructionEvidence::from_view(&classic_view(&buffers));
    let view = owned.view().expect("owned evidence validates");
    assert!(std::ptr::eq(view.points.as_ptr(), owned.points.as_ptr()));
    assert!(std::ptr::eq(
        view.triangles.as_ptr(),
        owned.triangles.as_ptr()
    ));
    assert!(std::ptr::eq(
        view.point_attributes.expect("attributes").as_ptr(),
        owned
            .point_attributes
            .as_ref()
            .expect("owned attributes")
            .as_ptr()
    ));
    assert_eq!(view.points.len(), 5);
    assert_eq!(view.triangles.len(), 2);
}

#[test]
fn stream_layout_is_header_then_raw_little_endian_buffers() {
    let buffers = classic_buffers();
    let bytes = encode(&classic_view(&buffers));

    assert_eq!(&bytes[..8], &EVIDENCE_INTERCHANGE_MAGIC);
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        EVIDENCE_INTERCHANGE_FORMAT_VERSION
    );
    let header = header_json(&bytes);
    assert!(header.is_object(), "header must be a JSON object");
    assert_eq!(header["points"].as_u64(), Some(5));
    assert_eq!(header["triangles"].as_u64(), Some(2));

    // Geometry lives only in the raw buffers, not duplicated into the header.
    assert_eq!(
        bytes.len(),
        attributes_offset(&bytes, 5, 2) + ATTRIBUTE_BYTES * 5,
        "stream must be prefix + header + exactly the raw point/triangle/attribute buffers"
    );

    let p = points_offset(&bytes) + POINT_BYTES * 3;
    let expected = buffers.points[3];
    assert_eq!(&bytes[p..p + 4], &expected.x.to_le_bytes());
    assert_eq!(&bytes[p + 4..p + 8], &expected.y.to_le_bytes());
    assert_eq!(&bytes[p + 8..p + 12], &expected.z.to_le_bytes());
    assert_eq!(&bytes[p + 12..p + 16], &expected.confidence.to_le_bytes());
    assert_eq!(&bytes[p + 16..p + 19], &[12, 34, 56]);

    let t = triangles_offset(&bytes, 5) + TRIANGLE_BYTES;
    assert_eq!(&bytes[t..t + 4], &2u32.to_le_bytes());
    assert_eq!(&bytes[t + 4..t + 8], &3u32.to_le_bytes());
    assert_eq!(&bytes[t + 8..t + 12], &4u32.to_le_bytes());
    assert_eq!(&bytes[t + 12..t + 16], &0.333_333_34f32.to_le_bytes());

    let a = attributes_offset(&bytes, 5, 2) + ATTRIBUTE_BYTES * 2;
    assert_eq!(&bytes[a..a + 4], &0.666_666_7f32.to_le_bytes());
    assert_eq!(&bytes[a + 4..a + 8], &0.125f32.to_le_bytes());
}

#[test]
fn decode_rejects_wrong_magic_and_unknown_format_version() {
    let buffers = classic_buffers();
    let bytes = encode(&classic_view(&buffers));

    let mut wrong_magic = bytes.clone();
    wrong_magic[0] ^= 0xff;
    assert_err_contains(decode(&wrong_magic), "magic");

    for version in [0u32, 2, u32::MAX] {
        let mut wrong_version = bytes.clone();
        wrong_version[8..12].copy_from_slice(&version.to_le_bytes());
        assert_err_contains(decode(&wrong_version), "version");
    }
}

#[test]
fn decode_rejects_truncated_streams() {
    let buffers = classic_buffers();
    let bytes = encode(&classic_view(&buffers));

    assert!(decode(&[]).is_err(), "empty input must fail");
    assert!(
        decode(&bytes[..10]).is_err(),
        "cut inside the prefix must fail"
    );

    let cut_in_header = PREFIX_BYTES + header_len(&bytes) / 2;
    assert_err_contains(decode(&bytes[..cut_in_header]), "truncated");

    let cut_in_points = points_offset(&bytes) + POINT_BYTES + 5;
    assert_err_contains(decode(&bytes[..cut_in_points]), "truncated");

    let cut_in_triangles = triangles_offset(&bytes, 5) + 3;
    assert_err_contains(decode(&bytes[..cut_in_triangles]), "truncated");

    assert_err_contains(decode(&bytes[..bytes.len() - 1]), "truncated");
}

#[test]
fn decode_rejects_trailing_bytes() {
    let buffers = classic_buffers();
    let mut bytes = encode(&classic_view(&buffers));
    bytes.push(0);
    assert_err_contains(decode(&bytes), "trailing");
}

#[test]
fn decode_rejects_unknown_header_fields() {
    let buffers = classic_buffers();
    let bytes = encode(&classic_view(&buffers));
    let header = header_json(&bytes);
    // Control: rebuilding with the unchanged header still decodes.
    decode(&with_header(&bytes, &header)).expect("unchanged rebuilt header decodes");

    let mut extended = header;
    extended
        .as_object_mut()
        .unwrap()
        .insert("unexpected_extra".into(), serde_json::Value::Bool(true));
    assert!(
        decode(&with_header(&bytes, &extended)).is_err(),
        "unknown header field must fail closed"
    );
}

#[test]
fn declared_counts_are_bounded_before_allocation() {
    let buffers = classic_buffers();
    let bytes = encode(&classic_view(&buffers));
    let header = header_json(&bytes);

    // A tiny stream claiming astronomically many elements must be refused by the
    // limit check, not by an allocation failure or a slow read loop.
    for key in ["points", "triangles"] {
        let mut lying = header.clone();
        lying[key] = serde_json::json!(1u64 << 40);
        assert_err_contains(decode(&with_header(&bytes, &lying)), "limit");
    }

    // A declared header length above the header limit is refused before reading it.
    let mut huge_header = bytes.clone();
    huge_header[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_err_contains(
        decode_with(
            &huge_header,
            InterchangeLimits {
                max_header_bytes: 1024,
                ..InterchangeLimits::default()
            },
        ),
        "limit",
    );
}

#[test]
fn caller_limits_bound_every_declared_count() {
    let buffers = classic_buffers();
    let bytes = encode(&classic_view(&buffers));
    let exact = InterchangeLimits {
        max_header_bytes: header_len(&bytes),
        max_cameras: 3,
        max_regions: 2,
        max_points: 5,
        max_triangles: 2,
    };
    decode_with(&bytes, exact).expect("counts equal to the limits are accepted");

    for limits in [
        InterchangeLimits {
            max_header_bytes: header_len(&bytes) - 1,
            ..exact
        },
        InterchangeLimits {
            max_cameras: 2,
            ..exact
        },
        InterchangeLimits {
            max_regions: 1,
            ..exact
        },
        InterchangeLimits {
            max_points: 4,
            ..exact
        },
        InterchangeLimits {
            max_triangles: 1,
            ..exact
        },
    ] {
        assert_err_contains(decode_with(&bytes, limits), "limit");
    }
}

#[test]
fn invalid_provider_buffers_fail_at_decode_through_shared_validation() {
    let buffers = classic_buffers();
    let bytes = encode(&classic_view(&buffers));

    let mut out_of_buffer = bytes.clone();
    let t = triangles_offset(&bytes, 5);
    out_of_buffer[t..t + 4].copy_from_slice(&99u32.to_le_bytes());
    assert_err_contains(decode(&out_of_buffer), "outside the shared evidence buffer");

    let mut degenerate = bytes.clone();
    degenerate[t + 4..t + 8].copy_from_slice(&0u32.to_le_bytes());
    assert_err_contains(decode(&degenerate), "topologically degenerate");

    for invalid in [f32::NAN, f32::INFINITY] {
        let mut non_finite = bytes.clone();
        patch_f32(
            &mut non_finite,
            points_offset(&bytes) + POINT_BYTES,
            invalid,
        );
        assert_err_contains(decode(&non_finite), "invalid geometry/confidence");
    }

    let mut bad_confidence = bytes.clone();
    patch_f32(&mut bad_confidence, points_offset(&bytes) + 12, 1.5);
    assert_err_contains(decode(&bad_confidence), "invalid geometry/confidence");

    let mut bad_attribute = bytes.clone();
    patch_f32(&mut bad_attribute, attributes_offset(&bytes, 5, 2), 1.5);
    assert_err_contains(decode(&bad_attribute), "invalid reciprocal/depth-margin");
}

#[test]
fn learned_provider_cannot_relabel_provenance_across_the_interchange() {
    let buffers = learned_buffers();
    let bytes = encode(&learned_view(&buffers));
    let mut header = header_json(&bytes);
    assert_eq!(
        relabel_origins(&mut header, "learned_multi_view", "geometric_multi_view"),
        1,
        "header must carry each region's origin as the snake_case EvidenceOrigin"
    );
    assert_err_contains(
        decode(&with_header(&bytes, &header)),
        "cannot emit GeometricMultiView provenance",
    );
}

#[test]
fn owned_view_applies_shared_validation_rules() {
    let buffers = learned_buffers();
    let valid = OwnedReconstructionEvidence::from_view(&learned_view(&buffers));
    valid.view().expect("valid owned evidence");

    let mut relabeled = valid.clone();
    relabeled.regions[0].origin = EvidenceOrigin::GeometricMultiView;
    let error = relabeled.view().expect_err("learned geometric provenance");
    assert!(
        error.contains("cannot emit GeometricMultiView provenance"),
        "{error}"
    );

    let mut out_of_buffer = valid.clone();
    out_of_buffer.triangles[0].c = 3;
    let error = out_of_buffer.view().expect_err("triangle outside buffer");
    assert!(
        error.contains("outside the shared evidence buffer"),
        "{error}"
    );

    let mut non_finite = valid.clone();
    non_finite.points[1].z = f32::NAN;
    let error = non_finite.view().expect_err("non-finite point");
    assert!(error.contains("invalid geometry/confidence"), "{error}");

    let mut short_attributes = valid.clone();
    short_attributes.point_attributes = Some(vec![EvidencePointAttributes::default(); 2]);
    let error = short_attributes
        .view()
        .expect_err("attributes must cover every point");
    assert!(error.contains("cover 2 of 3 dense points"), "{error}");

    let mut bad_attribute = valid.clone();
    bad_attribute.point_attributes = Some(vec![
        EvidencePointAttributes {
            reciprocal_consistency: 0.5,
            depth_margin: -0.25,
        };
        3
    ]);
    let error = bad_attribute.view().expect_err("out-of-range attribute");
    assert!(error.contains("invalid reciprocal/depth-margin"), "{error}");

    let mut stolen_authority = valid;
    stolen_authority.cameras[0].authority = EvidenceCameraAuthority::RegisteredGeometry;
    let error = stolen_authority
        .view()
        .expect_err("learned camera authority");
    assert!(error.contains("camera authority"), "{error}");
}
