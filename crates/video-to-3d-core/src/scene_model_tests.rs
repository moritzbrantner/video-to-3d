use super::*;
use serde_json::{json, Value};

const HASH_A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HASH_B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const HASH_C: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const HASH_D: &str = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const HASH_E: &str = "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const HASH_F: &str = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

fn identity() -> Value {
    json!({ "translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0] })
}

fn scene_document() -> Value {
    json!({
        "schema_version": 1,
        "frame": { "unit": "arbitrary_monocular", "handedness": "right_handed", "up_axis": "positive_y" },
        "cameras": [
            { "id": "cam-0", "source_frame": 0, "authority": "calibrated_seed", "transform": identity(),
              "intrinsics": { "width": 640, "height": 480, "focal_length_pixels": 500.0, "principal_point": [320.0, 240.0] } },
            { "id": "cam-4", "source_frame": 4, "authority": "registered_geometry",
              "transform": { "translation": [0.5, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0] } }
        ],
        "resources": [
            { "id": "room-mesh", "kind": "mesh", "path": "scene/room.mesh", "content_hash": HASH_A,
              "provenance": "geometric_multi_view", "confidence": 0.8, "source_frames": [4, 0] },
            { "id": "room-splat", "kind": "splat_field", "path": "scene/room.splat", "content_hash": HASH_B,
              "provenance": "learned_multi_view", "source_frames": [0, 4] },
            { "id": "room-albedo", "kind": "texture", "path": "scene/room.png", "content_hash": HASH_C,
              "provenance": "geometric_multi_view", "source_frames": [0] },
            { "id": "chair-mesh", "kind": "mesh", "path": "objects/chair.glb", "content_hash": HASH_D,
              "provenance": "generative_completion" },
            { "id": "sky", "kind": "environment_map", "path": "scene/sky.hdr", "content_hash": HASH_E,
              "provenance": "estimated" },
            { "id": "hum", "kind": "audio_clip", "path": "audio/hum.ogg", "content_hash": HASH_F,
              "provenance": "authored" }
        ],
        "materials": [
            { "id": "room-material", "base_color": [1.0, 1.0, 1.0, 1.0], "base_color_texture": "room-albedo",
              "metallic": 0.0, "roughness": 0.9 }
        ],
        "assets": [
            { "id": "environment", "role": "environment", "transform": identity(),
              "visual": { "mesh": { "mesh": "room-mesh", "material": "room-material" }, "splat": "room-splat" },
              "collision": { "mesh": { "mesh": "room-mesh" } } },
            { "id": "floor-collider", "role": "environment", "parent": "environment", "transform": identity(),
              "collision": { "box": { "half_extents": [2.0, 0.05, 2.0] } }, "collision_provenance": "geometric_multi_view" },
            { "id": "chair", "role": "editable_object", "parent": "environment",
              "transform": { "translation": [0.2, 0.0, 1.0], "rotation": [0.0, 0.6, 0.0, 0.8], "scale": [1.0, 1.0, 1.0] },
              "visual": { "mesh": { "mesh": "chair-mesh" } },
              "labels": [{ "name": "chair", "confidence": 0.9 }] }
        ],
        "lights": [
            { "id": "sky-light", "kind": { "environment": { "map": "sky" } }, "transform": identity(),
              "color": [1.0, 1.0, 1.0], "intensity": 1.0, "provenance": "estimated" }
        ],
        "audio": [
            { "id": "chair-hum", "clip": "hum", "ambient": false, "attached_to": "chair", "provenance": "authored" }
        ]
    })
}

fn parse(document: &Value) -> Result<AssembledScene, SceneProjectError> {
    AssembledScene::from_json(&document.to_string())
}

fn with(mut document: Value, pointer: &str, value: Value) -> Value {
    let (parent, key) = pointer.rsplit_once('/').expect("fixture pointer");
    match document.pointer_mut(parent).expect("fixture parent") {
        Value::Object(object) => {
            object.insert(key.to_owned(), value);
        }
        Value::Array(array) => array[key.parse::<usize>().expect("index")] = value,
        _ => panic!("fixture parent must be a container"),
    }
    document
}

fn without(mut document: Value, pointer: &str) -> Value {
    let (parent, key) = pointer.rsplit_once('/').expect("fixture pointer");
    document
        .pointer_mut(parent)
        .and_then(Value::as_object_mut)
        .expect("object parent")
        .remove(key);
    document
}

fn expect_invalid(document: &Value, needle: &str) {
    match parse(document) {
        Err(SceneProjectError::Invalid(message)) => assert!(
            message.contains(needle),
            "expected `{needle}` in `{message}`"
        ),
        other => panic!("expected invalid scene containing `{needle}`, got {other:?}"),
    }
}

#[test]
fn full_scene_round_trips_canonically() {
    let scene = parse(&scene_document()).expect("valid scene");
    let canonical = scene.to_canonical_json().unwrap();
    let reloaded = AssembledScene::from_json(&canonical).unwrap();
    assert_eq!(reloaded, scene);
    assert_eq!(reloaded.to_canonical_json().unwrap(), canonical);
}

#[test]
fn minimal_scene_round_trips_without_renderer_state() {
    let mut scene = AssembledScene::new(SceneFrame {
        unit: SceneUnit::Meters,
        handedness: Handedness::RightHanded,
        up_axis: UpAxis::PositiveY,
    });
    scene.assets.push(SceneAsset {
        id: "probe".into(),
        role: AssetRole::Environment,
        parent: None,
        transform: Transform::IDENTITY,
        visual: None,
        collision: Some(CollisionShape::Sphere { radius: 0.5 }),
        labels: Vec::new(),
        collision_provenance: Some(SceneProvenance::Authored),
    });
    let document = scene.to_canonical_json().unwrap();
    assert_eq!(AssembledScene::from_json(&document).unwrap(), scene);
}

#[test]
fn provenance_survives_serialization_and_cannot_be_relabeled() {
    let scene = parse(&scene_document()).unwrap();
    let reloaded = AssembledScene::from_json(&scene.to_canonical_json().unwrap()).unwrap();
    let chair = reloaded.asset_provenance("chair").unwrap();
    assert!(chair.contains_generative());
    assert_eq!(
        chair.visual,
        BTreeSet::from([SceneProvenance::GenerativeCompletion])
    );
    let environment = reloaded.asset_provenance("environment").unwrap();
    assert!(!environment.contains_generative());
    assert_eq!(
        environment.visual,
        BTreeSet::from([
            SceneProvenance::GeometricMultiView,
            SceneProvenance::LearnedMultiView
        ])
    );
    assert_eq!(
        environment.collision,
        Some(SceneProvenance::GeometricMultiView)
    );

    // An asset has no provenance field of its own, so it cannot relabel a
    // generative resource as observed.
    let relabel = with(
        scene_document(),
        "/assets/2/provenance",
        json!("geometric_multi_view"),
    );
    assert!(matches!(
        parse(&relabel),
        Err(SceneProjectError::Malformed(_))
    ));
    assert_eq!(
        SceneProvenance::from(EvidenceOrigin::GenerativeCompletion),
        SceneProvenance::GenerativeCompletion
    );
}

#[test]
fn visual_only_collision_only_and_combined_assets_are_representable() {
    let scene = parse(&scene_document()).unwrap();
    let by_id = |id: &str| scene.assets.iter().find(|asset| asset.id == id).unwrap();
    let combined = by_id("environment");
    assert!(combined.visual.is_some() && combined.collision.is_some());
    let visual = combined.visual.as_ref().unwrap();
    assert!(visual.mesh.is_some() && visual.splat.is_some());
    assert!(by_id("chair").collision.is_none());
    assert!(by_id("floor-collider").visual.is_none());

    expect_invalid(
        &without(
            without(scene_document(), "/assets/2/visual"),
            "/assets/2/labels",
        ),
        "visual representation, a collision shape, or both",
    );
    expect_invalid(
        &with(scene_document(), "/assets/2/visual", json!({})),
        "empty visual representation",
    );
}

#[test]
fn invalid_cross_references_fail_closed() {
    expect_invalid(
        &with(
            scene_document(),
            "/assets/0/visual/splat",
            json!("room-mesh"),
        ),
        "expects SplatField resource",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/assets/2/visual/mesh/mesh",
            json!("missing"),
        ),
        "unknown resource `missing`",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/assets/0/visual/mesh/material",
            json!("missing"),
        ),
        "unknown material",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/materials/0/base_color_texture",
            json!("sky"),
        ),
        "expects Texture resource",
    );
    expect_invalid(
        &with(scene_document(), "/assets/2/parent", json!("ghost")),
        "unknown parent",
    );
    expect_invalid(
        &with(scene_document(), "/audio/0/attached_to", json!("ghost")),
        "unknown asset",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/lights/0/kind",
            json!({ "environment": { "map": "hum" } }),
        ),
        "expects EnvironmentMap resource",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/resources/0/source_frames",
            json!([0, 7]),
        ),
        "no accepted camera",
    );
    expect_invalid(
        &with(scene_document(), "/resources/1/id", json!("room-mesh")),
        "declared more than once",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/resources/1/path",
            json!("Scene/Room.mesh"),
        ),
        "used more than once",
    );
    let mut cyclic = with(scene_document(), "/assets/0/parent", json!("chair"));
    cyclic = with(cyclic, "/assets/2/parent", json!("environment"));
    expect_invalid(&cyclic, "cyclic parent chain");
}

#[test]
fn invalid_transforms_and_factors_fail_closed() {
    expect_invalid(
        &with(
            scene_document(),
            "/assets/2/transform/rotation",
            json!([0.0, 1.0, 0.0, 1.0]),
        ),
        "unit quaternion",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/assets/2/transform/scale",
            json!([1.0, 0.0, 1.0]),
        ),
        "scale must be positive",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/cameras/1/transform/scale",
            json!([2.0, 2.0, 2.0]),
        ),
        "unit scale",
    );
    expect_invalid(
        &with(scene_document(), "/cameras/1/source_frame", json!(0)),
        "more than one accepted camera",
    );
    expect_invalid(
        &with(scene_document(), "/resources/0/confidence", json!(1.5)),
        "confidence",
    );
    expect_invalid(
        &with(scene_document(), "/materials/0/roughness", json!(-0.1)),
        "within [0, 1]",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/assets/1/collision",
            json!({ "box": { "half_extents": [1.0, 0.0, 1.0] } }),
        ),
        "positive finite extents",
    );
}

#[test]
fn provenance_rules_fail_closed() {
    expect_invalid(
        &without(scene_document(), "/resources/0/source_frames"),
        "must list its supporting source frames",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/resources/3/provenance",
            json!("estimated"),
        ),
        "cannot have estimated provenance",
    );
    expect_invalid(
        &without(scene_document(), "/assets/1/collision_provenance"),
        "requires collision_provenance",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/assets/0/collision_provenance",
            json!("authored"),
        ),
        "comes from its mesh resource",
    );
    expect_invalid(
        &with(
            scene_document(),
            "/lights/0/provenance",
            json!("geometric_multi_view"),
        ),
        "cannot claim observed-geometry provenance",
    );
}

#[test]
fn audio_anchors_are_either_ambient_or_positioned() {
    let ambient = with(
        without(scene_document(), "/audio/0/attached_to"),
        "/audio/0/ambient",
        json!(true),
    );
    parse(&ambient).expect("unpositioned ambient audio is valid");
    expect_invalid(
        &with(scene_document(), "/audio/0/ambient", json!(true)),
        "must not be positioned",
    );
    expect_invalid(
        &without(scene_document(), "/audio/0/attached_to"),
        "requires a transform or an attached asset",
    );
}

#[test]
fn unknown_versions_and_fields_fail_closed() {
    assert_eq!(
        parse(&with(scene_document(), "/schema_version", json!(9))),
        Err(SceneProjectError::UnsupportedSchemaVersion(9))
    );
    assert!(matches!(
        parse(&with(scene_document(), "/renderer", json!({ "three": {} }))),
        Err(SceneProjectError::Malformed(_))
    ));
    assert!(matches!(
        parse(&with(scene_document(), "/frame/unit", json!("feet"))),
        Err(SceneProjectError::Malformed(_))
    ));
}

#[test]
fn learned_resources_require_accepted_camera_support() {
    expect_invalid(
        &without(scene_document(), "/resources/1/source_frames"),
        "must list its supporting source frames",
    );
}

#[test]
fn generated_textures_surface_in_asset_provenance() {
    let document = with(
        scene_document(),
        "/resources/2/provenance",
        json!("generative_completion"),
    );
    let document = without(document, "/resources/2/source_frames");
    let scene = parse(&document).unwrap();
    let environment = scene.asset_provenance("environment").unwrap();
    assert!(environment.contains_generative());
    assert!(environment
        .visual
        .contains(&SceneProvenance::GenerativeCompletion));
}
