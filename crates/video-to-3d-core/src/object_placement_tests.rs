use super::*;
use crate::object_generation::GenerationRequest;
use crate::semantic_inventory::{MaskEncoding, SemanticMask};

/// Identity camera at the origin looking down +z; scene up is −y.
fn camera() -> FrameCamera {
    FrameCamera {
        frame_index: 0,
        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation: [0.0, 0.0, 0.0],
        focal_pixels: 300.0,
        principal_point: [160.0, 120.0],
    }
}

fn request() -> GenerationRequest {
    GenerationRequest {
        request_id: "gen-1".into(),
        object: "obj-1".into(),
        label: "chair".into(),
        observation: "o".into(),
        frame_index: 0,
        source_frame_hash: ContentHash::of_bytes(b"f"),
        width: 4,
        height: 4,
        crop: [0, 0, 4, 4],
        mask_hash: ContentHash::of_bytes(b"m"),
        mask: SemanticMask {
            encoding: MaskEncoding::RleRowMajor,
            width: 4,
            height: 4,
            counts: vec![0, 16],
        },
        observed_points: 0,
    }
}

/// A box 0.4 wide (x), 0.3 tall (along −y) and 0.2 deep (z) centred at
/// `centre`; only its camera-facing surface is observed.
fn front_face(centre: [f64; 3]) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for i in 0..=8 {
        for j in 0..=6 {
            points.push([
                centre[0] - 0.2 + 0.05 * i as f64,
                centre[1] - 0.15 + 0.05 * j as f64,
                centre[2] - 0.1,
            ]);
        }
    }
    points
}

/// The generator's unit-ish mesh: depth along x, width along y, height along z.
fn bounds() -> MeshBounds {
    MeshBounds {
        min: [-0.25, -0.5, -0.375],
        max: [0.25, 0.5, 0.375],
    }
}

fn floor(height: f64) -> SupportPlane {
    // Floor plane y = height, normal pointing up (−y).
    SupportPlane {
        object: "obj-floor".into(),
        label: "floor".into(),
        normal: [0.0, -1.0, 0.0],
        offset: -height,
        points: 50,
    }
}

fn apply(placement: &ObjectPlacement, point: [f64; 3]) -> [f64; 3] {
    let r = placement.rotation;
    let rotated = [0, 1, 2]
        .map(|row| r[row * 3] * point[0] + r[row * 3 + 1] * point[1] + r[row * 3 + 2] * point[2]);
    add(scaled(rotated, placement.scale), placement.translation)
}

#[test]
fn a_known_object_transform_is_recovered_and_rests_on_the_floor() {
    let centre = [0.3, 0.85, 3.0];
    let placement = place_one(
        &request(),
        &front_face(centre),
        bounds(),
        &camera(),
        [0.0, -1.0, 0.0],
        &[floor(1.0)],
    )
    .unwrap();
    assert!((placement.scale - 0.4).abs() < 1e-9, "{}", placement.scale);
    let placed_centre = apply(&placement, [0.0, 0.0, 0.0]);
    for axis in 0..3 {
        assert!(
            (placed_centre[axis] - centre[axis]).abs() < 1e-9,
            "{placed_centre:?}"
        );
    }
    // The mesh's +z (up) is scene up, +x faces the camera, +y is image right.
    let up = sub(apply(&placement, [0.0, 0.0, 1.0]), placed_centre);
    assert!(dot(normalize(up), [0.0, -1.0, 0.0]) > 0.999);
    let toward = sub(apply(&placement, [1.0, 0.0, 0.0]), placed_centre);
    assert!(dot(normalize(toward), [0.0, 0.0, -1.0]) > 0.999);
    assert_eq!(
        placement.contact,
        ContactState::InContact,
        "{}",
        placement.reason
    );
    assert_eq!(placement.support.as_deref(), Some("obj-floor"));
}

#[test]
fn floating_and_sunken_placements_are_flagged_against_contact_evidence() {
    let centre = [0.3, 0.85, 3.0];
    let floating = place_one(
        &request(),
        &front_face(centre),
        bounds(),
        &camera(),
        [0.0, -1.0, 0.0],
        &[floor(1.3)],
    )
    .unwrap();
    assert_eq!(
        floating.contact,
        ContactState::Floating,
        "{}",
        floating.reason
    );
    assert!(floating.contact_gap.unwrap() > 0.2);
    let sunken = place_one(
        &request(),
        &front_face(centre),
        bounds(),
        &camera(),
        [0.0, -1.0, 0.0],
        &[floor(0.94)],
    )
    .unwrap();
    assert_eq!(sunken.contact, ContactState::Sunken, "{}", sunken.reason);
    let unsupported = place_one(
        &request(),
        &front_face(centre),
        bounds(),
        &camera(),
        [0.0, -1.0, 0.0],
        &[],
    )
    .unwrap();
    assert_eq!(unsupported.contact, ContactState::NoContactEvidence);
}

#[test]
fn a_tilted_camera_still_places_the_object_upright_and_planes_fit() {
    // Camera pitched down by 20° about x: the object must stay upright.
    let angle = 20f64.to_radians();
    let mut tilted = camera();
    tilted.rotation = [
        1.0,
        0.0,
        0.0,
        0.0,
        angle.cos(),
        -angle.sin(),
        0.0,
        angle.sin(),
        angle.cos(),
    ];
    let placement = place_one(
        &request(),
        &front_face([0.3, 0.85, 3.0]),
        bounds(),
        &tilted,
        [0.0, -1.0, 0.0],
        &[],
    )
    .unwrap();
    let up = sub(
        apply(&placement, [0.0, 0.0, 1.0]),
        apply(&placement, [0.0; 3]),
    );
    assert!(dot(normalize(up), [0.0, -1.0, 0.0]) > 0.999);
    let floor_points: Vec<[f64; 3]> = (0..40)
        .map(|i| [(i % 8) as f64 * 0.1, 1.0, 2.0 + (i / 8) as f64 * 0.2])
        .collect();
    let (normal, offset) = fit_plane(&floor_points).unwrap();
    assert!(dot(normal, [0.0, 1.0, 0.0]).abs() > 0.999);
    assert!((offset.abs() - 1.0).abs() < 1e-9);
}

#[test]
fn too_little_observed_extent_is_not_placed_and_priors_never_scale() {
    let error = place_one(
        &request(),
        &[[0.0, 0.0, 3.0]],
        bounds(),
        &camera(),
        [0.0, -1.0, 0.0],
        &[],
    )
    .unwrap_err();
    assert!(error.contains("no observed scale"), "{error}");
    assert_eq!(size_prior("Office chair"), Some(0.9));
    assert_eq!(size_prior("spaceship"), None);
}
