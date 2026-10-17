//! Independent acceptance contract of #60 (first slice, Decision Atlas 8.8):
//! the versioned, provider-neutral semantic-inventory evidence artifact and its
//! manifest operation. Authored before any production code; see
//! `docs/specs/semantic-inventory-acceptance.md` for the contract, what is
//! protected and what the implementer may translate.
//!
//! Proposed public API (names translatable, semantics protected):
//!
//! ```text
//! video_to_3d_core::semantic_inventory::{
//!     SemanticInventoryArtifact,            // from_json / to_json / validate / validate_against_keyframes
//!     SEMANTIC_INVENTORY_SCHEMA_VERSION,    // == 1
//!     default_room_vocabulary,              // -> Vec<String>
//! }
//! OperationKind::SemanticInventory  ("semantic_inventory", provider operation)
//! ArtifactKind::SemanticInventory   ("semantic_inventory")
//! OperationDeclaration.vocabulary   (optional per-project override, semantic only)
//! ```
//!
//! Semantics are evidence only: no field of the artifact may carry cameras,
//! poses, 3D positions or geometry, and no geometry producer may consume it.

use serde_json::{json, Value};
use video_to_3d_core::scene_artifacts::{KeyframeRecord, KeyframesArtifact};
use video_to_3d_core::scene_project::{
    ArtifactKind, ContentHash, OperationExecution, OperationKind, ProjectPath, SceneProjectManifest,
};
use video_to_3d_core::semantic_inventory::{
    default_room_vocabulary, SemanticInventoryArtifact, SEMANTIC_INVENTORY_SCHEMA_VERSION,
};

const W: u32 = 64;
const H: u32 = 48;

fn frame_hash(index: usize) -> String {
    ContentHash::of_bytes(format!("frame-{index}").as_bytes())
        .as_str()
        .to_owned()
}

/// Row-major run lengths, alternating, starting with background (`false`).
fn rle(width: u32, height: u32, inside: impl Fn(u32, u32) -> bool) -> Vec<u64> {
    let mut counts = Vec::new();
    let mut current = false;
    let mut run = 0_u64;
    for y in 0..height {
        for x in 0..width {
            let value = inside(x, y);
            if value != current {
                counts.push(run);
                run = 0;
                current = value;
            }
            run += 1;
        }
    }
    counts.push(run);
    counts
}

fn rect_mask(x0: u32, y0: u32, x1: u32, y1: u32) -> Value {
    json!({
        "encoding": "rle_row_major",
        "width": W,
        "height": H,
        "counts": rle(W, H, |x, y| x >= x0 && x < x1 && y >= y0 && y < y1),
    })
}

/// A valid available inventory over keyframes 0, 1 and 2 (64x48).
fn valid() -> Value {
    json!({
        "schema_version": 1,
        "status": "available",
        "provider": {
            "id": "local-grounded-sam2",
            "revision": "gdino-tiny+sam2.1-hiera-tiny/1",
            "models": [
                {"role": "detector", "id": "IDEA-Research/grounding-dino-tiny",
                 "version": "a2bb814", "url": "https://huggingface.co/IDEA-Research/grounding-dino-tiny"},
                {"role": "segmenter", "id": "facebook/sam2.1-hiera-tiny",
                 "version": "sam2.1", "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"}
            ],
            "runtime": {"name": "pytorch", "version": "2.5.1", "device": "cuda",
                        "device_name": "NVIDIA GeForce RTX 3060 Ti"}
        },
        "vocabulary": {"source": "default", "terms": ["wall", "door", "chair"]},
        "frames": ([0_usize, 1, 2].iter().map(|index| json!({
            "frame_index": index, "width": W, "height": H, "content_hash": frame_hash(*index)
        })).collect::<Vec<_>>()),
        "observations": [
            {"id": "obs-0", "frame_index": 0, "label": "chair", "bbox": [10.0, 8.0, 20.0, 18.0],
             "detection_confidence": 0.72, "mask": rect_mask(10, 8, 20, 18), "mask_confidence": 0.91},
            {"id": "obs-1", "frame_index": 1, "label": "chair", "bbox": [12.0, 8.0, 22.0, 18.0],
             "detection_confidence": 0.64, "mask": rect_mask(12, 9, 21, 17), "mask_confidence": 0.85},
            {"id": "obs-2", "frame_index": 2, "label": "wall", "bbox": [0.0, 0.0, 64.0, 30.0],
             "detection_confidence": 0.31, "mask": null, "mask_confidence": null}
        ],
        "tracks": [
            {"id": "track-0", "label": "chair", "observations": ["obs-0", "obs-1"],
             "association_confidence": 0.8}
        ],
        "uncertainties": [
            {"kind": "missing", "frame_index": 2, "label": "chair", "observation": null,
             "reason": "tracked chair not detected in frame 2"},
            {"kind": "weak", "frame_index": 2, "label": "wall", "observation": "obs-2",
             "reason": "mask prediction failed; box only"}
        ],
        "measurements": {"elapsed_seconds": 3.5, "peak_gpu_memory_bytes": 1_500_000_000_u64}
    })
}

fn unavailable() -> Value {
    json!({
        "schema_version": 1,
        "status": "unavailable",
        "unavailable_reason": "no CUDA device available; segmentation requires a local GPU",
        "provider": {"id": "local-grounded-sam2", "revision": "gdino-tiny+sam2.1-hiera-tiny/1",
                     "models": [], "runtime": null},
        "vocabulary": {"source": "default", "terms": ["wall", "door", "chair"]},
        "frames": [{"frame_index": 0, "width": W, "height": H, "content_hash": frame_hash(0)}],
        "observations": [],
        "tracks": [],
        "uncertainties": [],
        "measurements": null
    })
}

fn parse(value: &Value) -> Result<SemanticInventoryArtifact, String> {
    SemanticInventoryArtifact::from_json(&value.to_string())
}

#[track_caller]
fn rejects(value: Value, why: &str) {
    assert!(parse(&value).is_err(), "must fail closed: {why}");
}

fn keyframes(frames: &[(usize, u32, u32)]) -> KeyframesArtifact {
    KeyframesArtifact {
        schema_version: 1,
        frames: frames
            .iter()
            .map(|(index, width, height)| KeyframeRecord {
                frame_index: *index,
                timestamp_seconds: *index as f64 * 0.5,
                width: *width,
                height: *height,
                pixels: ProjectPath::new(format!("artifacts/browser/frames/{index}.rgba")).unwrap(),
                content_hash: ContentHash::new(frame_hash(*index)).unwrap(),
            })
            .collect(),
    }
}

#[test]
fn schema_version_is_one_and_valid_fixtures_parse() {
    assert_eq!(SEMANTIC_INVENTORY_SCHEMA_VERSION, 1);
    parse(&valid()).expect("available fixture is valid");
    parse(&unavailable()).expect("unavailable fixture is valid");
    let mut other = valid();
    other["schema_version"] = json!(2);
    rejects(other, "unknown schema version");
    let mut missing = valid();
    missing.as_object_mut().unwrap().remove("schema_version");
    rejects(missing, "missing schema version");
}

#[test]
fn json_round_trip_preserves_evidence_exactly() {
    let artifact = parse(&valid()).unwrap();
    let document = artifact.to_json();
    let reparsed = SemanticInventoryArtifact::from_json(&document).unwrap();
    assert_eq!(reparsed, artifact);
    assert_eq!(reparsed.to_json(), document, "serialization is canonical");
    let value: Value = serde_json::from_str(&document).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["status"], "available");
    assert_eq!(value["vocabulary"], valid()["vocabulary"]);
    assert_eq!(value["provider"]["models"], valid()["provider"]["models"]);
    assert_eq!(value["provider"]["runtime"], valid()["provider"]["runtime"]);
    assert_eq!(value["frames"], valid()["frames"]);
    assert_eq!(value["tracks"], valid()["tracks"]);
    for (index, observation) in value["observations"].as_array().unwrap().iter().enumerate() {
        let original = &valid()["observations"][index];
        for key in ["id", "frame_index", "label", "bbox", "detection_confidence"] {
            assert_eq!(observation[key], original[key], "{key}");
        }
        if original["mask"].is_null() {
            assert!(observation.get("mask").is_none_or(Value::is_null));
        } else {
            assert_eq!(observation["mask"], original["mask"]);
            assert_eq!(observation["mask_confidence"], original["mask_confidence"]);
        }
    }
    assert_eq!(value["uncertainties"].as_array().unwrap().len(), 2);
    assert_eq!(
        value["measurements"]["peak_gpu_memory_bytes"],
        1_500_000_000_u64
    );
    let unavailable_document = parse(&unavailable()).unwrap().to_json();
    let unavailable_value: Value = serde_json::from_str(&unavailable_document).unwrap();
    assert_eq!(unavailable_value["status"], "unavailable");
    assert_eq!(
        unavailable_value["unavailable_reason"],
        unavailable()["unavailable_reason"]
    );
}

#[test]
fn default_room_vocabulary_covers_architecture_and_furniture() {
    let terms = default_room_vocabulary();
    for required in ["wall", "door", "window", "floor", "ceiling", "chair"] {
        assert!(
            terms.iter().any(|term| term == required),
            "default vocabulary lacks `{required}`: {terms:?}"
        );
    }
    assert!(
        terms.iter().any(|term| term == "table" || term == "desk"),
        "default vocabulary lacks a table/desk furniture term"
    );
    let unique: std::collections::BTreeSet<_> = terms.iter().collect();
    assert_eq!(unique.len(), terms.len(), "terms are unique");
    assert!(terms.iter().all(|term| !term.trim().is_empty()));
}

#[test]
fn vocabulary_identity_is_required_and_labels_come_from_it() {
    let mut missing = valid();
    missing.as_object_mut().unwrap().remove("vocabulary");
    rejects(missing, "missing vocabulary");
    let mut empty = valid();
    empty["vocabulary"]["terms"] = json!([]);
    rejects(empty, "empty vocabulary");
    let mut source = valid();
    source["vocabulary"]["source"] = json!("guessed");
    rejects(source, "unknown vocabulary source");
    let mut duplicate = valid();
    duplicate["vocabulary"]["terms"] = json!(["wall", "chair", "chair"]);
    rejects(duplicate, "duplicate vocabulary term");
    let mut label = valid();
    label["observations"][0]["label"] = json!("sofa");
    label["tracks"][0]["label"] = json!("sofa");
    rejects(label, "observation label outside the recorded vocabulary");

    // A project override is recorded verbatim (order, spelling and spacing).
    let mut overridden = valid();
    overridden["vocabulary"] =
        json!({"source": "project", "terms": ["Yellow die", "chair", "wall", "door"]});
    let artifact = parse(&overridden).expect("project vocabulary override is valid");
    let value: Value = serde_json::from_str(&artifact.to_json()).unwrap();
    assert_eq!(value["vocabulary"], overridden["vocabulary"]);
}

#[test]
fn provider_and_model_identity_are_required() {
    for path in [&["provider", "id"][..], &["provider", "revision"][..]] {
        let mut removed = valid();
        removed[path[0]].as_object_mut().unwrap().remove(path[1]);
        rejects(removed, &format!("missing {path:?}"));
        let mut blank = valid();
        blank[path[0]][path[1]] = json!("");
        rejects(blank, &format!("blank {path:?}"));
    }
    let mut no_provider = valid();
    no_provider.as_object_mut().unwrap().remove("provider");
    rejects(no_provider, "missing provider");
    let mut no_models = valid();
    no_models["provider"]["models"] = json!([]);
    rejects(no_models, "available evidence without model identity");
    for key in ["id", "version", "url"] {
        let mut model = valid();
        model["provider"]["models"][0]
            .as_object_mut()
            .unwrap()
            .remove(key);
        rejects(model, &format!("model without {key}"));
    }
    let mut runtime = valid();
    runtime["provider"]["runtime"] = Value::Null;
    rejects(runtime, "available evidence without runtime identity");
    let mut unavailable_identity = unavailable();
    unavailable_identity["provider"]["revision"] = json!("");
    rejects(
        unavailable_identity,
        "unavailable result still names its provider revision",
    );
}

#[test]
fn every_frame_reference_must_be_a_recorded_keyframe() {
    let mut observation = valid();
    observation["observations"][1]["frame_index"] = json!(7);
    rejects(observation, "observation frame not among frames");
    let mut uncertainty = valid();
    uncertainty["uncertainties"][0]["frame_index"] = json!(9);
    rejects(uncertainty, "uncertainty frame not among frames");
    let mut duplicated = valid();
    duplicated["frames"][2]["frame_index"] = json!(1);
    rejects(duplicated, "duplicate frame record");
    let mut hash = valid();
    hash["frames"][0]["content_hash"] = json!("not-a-hash");
    rejects(hash, "frame without content identity");

    let artifact = parse(&valid()).unwrap();
    artifact
        .validate_against_keyframes(&keyframes(&[(0, W, H), (1, W, H), (2, W, H), (3, W, H)]))
        .expect("frames are a subset of the source keyframes with identical identity");
    assert!(
        artifact
            .validate_against_keyframes(&keyframes(&[(0, W, H), (1, W, H)]))
            .is_err(),
        "frame 2 is not among the keyframes"
    );
    assert!(
        artifact
            .validate_against_keyframes(&keyframes(&[(0, W, H), (1, W, H), (2, W + 1, H)]))
            .is_err(),
        "frame size differs from its keyframe"
    );
    let mut other_pixels = keyframes(&[(0, W, H), (1, W, H), (2, W, H)]);
    other_pixels.frames[1].content_hash = ContentHash::of_bytes(b"different pixels");
    assert!(
        artifact.validate_against_keyframes(&other_pixels).is_err(),
        "frame content identity differs from its keyframe"
    );
}

#[test]
fn boxes_must_be_inside_their_frame_and_nondegenerate() {
    for (bbox, why) in [
        (json!([10.0, 8.0, 65.0, 18.0]), "x_max beyond width"),
        (json!([10.0, 8.0, 20.0, 49.0]), "y_max beyond height"),
        (json!([-1.0, 8.0, 20.0, 18.0]), "negative x_min"),
        (json!([20.0, 8.0, 20.0, 18.0]), "zero width"),
        (json!([10.0, 18.0, 20.0, 8.0]), "inverted y"),
        (json!([10.0, 8.0, 20.0]), "three coordinates"),
    ] {
        let mut value = valid();
        value["observations"][2]["bbox"] = bbox;
        rejects(value, why);
    }
    // Non-finite numbers are not representable; an overflowing literal must
    // not be accepted as infinity.
    let document = valid()
        .to_string()
        .replace("[0.0,0.0,64.0,30.0]", "[0.0,0.0,1e400,30.0]");
    assert!(document.contains("1e400"));
    assert!(SemanticInventoryArtifact::from_json(&document).is_err());
    let mut edge = valid();
    edge["observations"][2]["bbox"] = json!([0.0, 0.0, 64.0, 48.0]);
    parse(&edge).expect("a box covering the whole frame is valid");
}

#[test]
fn masks_must_match_their_frame_and_stay_inside_their_box() {
    let mut size = valid();
    size["observations"][0]["mask"]["width"] = json!(W * 2);
    rejects(size, "mask width differs from frame");
    let mut height = valid();
    height["observations"][0]["mask"]["height"] = json!(H - 1);
    rejects(height, "mask height differs from frame");
    let mut short = valid();
    let counts = short["observations"][0]["mask"]["counts"]
        .as_array_mut()
        .unwrap();
    *counts.last_mut().unwrap() = json!(counts.last().unwrap().as_u64().unwrap() - 1);
    rejects(short, "run lengths do not cover the frame");
    let mut long = valid();
    long["observations"][0]["mask"]["counts"]
        .as_array_mut()
        .unwrap()
        .push(json!(5));
    rejects(long, "run lengths exceed the frame");
    let mut outside = valid();
    outside["observations"][0]["mask"] = rect_mask(10, 8, 30, 18);
    rejects(outside, "mask foreground outside its box");
    let mut empty = valid();
    empty["observations"][0]["mask"]["counts"] = json!([W * H]);
    rejects(empty, "mask without foreground");
    let mut encoding = valid();
    encoding["observations"][0]["mask"]["encoding"] = json!("png");
    rejects(encoding, "unknown mask encoding");
}

#[test]
fn confidences_must_be_finite_probabilities() {
    for (pointer, value) in [
        ("/observations/0/detection_confidence", json!(1.2)),
        ("/observations/0/detection_confidence", json!(-0.1)),
        ("/observations/0/mask_confidence", json!(1.01)),
        ("/tracks/0/association_confidence", json!(-0.5)),
        ("/tracks/0/association_confidence", json!(2.0)),
    ] {
        let mut document = valid();
        *document.pointer_mut(pointer).unwrap() = value.clone();
        rejects(document, &format!("{pointer} = {value}"));
    }
    let mut missing = valid();
    missing["observations"][0]["mask_confidence"] = Value::Null;
    rejects(missing, "mask without mask confidence");
    let mut orphan = valid();
    orphan["observations"][2]["mask_confidence"] = json!(0.4);
    rejects(orphan, "mask confidence without a mask");
    let mut detection = valid();
    detection["observations"][0]
        .as_object_mut()
        .unwrap()
        .remove("detection_confidence");
    rejects(detection, "observation without detection confidence");
}

#[test]
fn tracks_reference_existing_observations_with_one_label() {
    let mut missing = valid();
    missing["tracks"][0]["observations"] = json!(["obs-0", "obs-9"]);
    rejects(missing, "track references a missing observation");
    let mut conflicting = valid();
    conflicting["tracks"][0]["observations"] = json!(["obs-0", "obs-2"]);
    rejects(
        conflicting,
        "track links observations with different labels",
    );
    let mut label = valid();
    label["tracks"][0]["label"] = json!("door");
    rejects(label, "track label differs from its observations");
    let mut empty = valid();
    empty["tracks"][0]["observations"] = json!([]);
    rejects(empty, "empty track");
    let mut repeated = valid();
    repeated["tracks"][0]["observations"] = json!(["obs-0", "obs-0"]);
    rejects(repeated, "track repeats an observation");
    let mut duplicate_track = valid();
    let track = duplicate_track["tracks"][0].clone();
    duplicate_track["tracks"]
        .as_array_mut()
        .unwrap()
        .push(track);
    rejects(duplicate_track, "duplicate track id");
    let mut duplicate_observation = valid();
    duplicate_observation["observations"][1]["id"] = json!("obs-0");
    rejects(duplicate_observation, "duplicate observation id");
}

#[test]
fn uncertainty_records_are_explicit_and_referential() {
    for kind in ["missing", "weak", "ambiguous"] {
        let mut document = valid();
        document["uncertainties"][0]["kind"] = json!(kind);
        parse(&document).unwrap_or_else(|error| panic!("{kind}: {error}"));
    }
    let mut kind = valid();
    kind["uncertainties"][0]["kind"] = json!("fine");
    rejects(kind, "unknown uncertainty kind");
    let mut reason = valid();
    reason["uncertainties"][0]["reason"] = json!("");
    rejects(reason, "uncertainty without a reason");
    let mut observation = valid();
    observation["uncertainties"][1]["observation"] = json!("obs-9");
    rejects(observation, "uncertainty references a missing observation");
    let mut label = valid();
    label["uncertainties"][0]["label"] = json!("sofa");
    rejects(label, "uncertainty label outside the vocabulary");
}

#[test]
fn unavailable_results_carry_a_reason_and_no_invented_evidence() {
    let mut no_reason = unavailable();
    no_reason
        .as_object_mut()
        .unwrap()
        .remove("unavailable_reason");
    rejects(no_reason, "unavailable without a reason");
    let mut blank = unavailable();
    blank["unavailable_reason"] = json!(" ");
    rejects(blank, "unavailable with a blank reason");
    let mut observations = unavailable();
    observations["observations"] = json!([valid()["observations"][2].clone()]);
    observations["observations"][0]["frame_index"] = json!(0);
    rejects(
        observations,
        "unavailable result still carries observations",
    );
    let mut tracks = unavailable();
    tracks["tracks"] = valid()["tracks"].clone();
    rejects(tracks, "unavailable result still carries tracks");
    let mut reason = valid();
    reason["unavailable_reason"] = json!("should not be here");
    rejects(reason, "available result with an unavailable reason");
    let mut status = valid();
    status["status"] = json!("partial");
    rejects(status, "unknown status");
}

#[test]
fn measurements_are_finite_and_nonnegative() {
    let mut elapsed = valid();
    elapsed["measurements"]["elapsed_seconds"] = json!(-1.0);
    rejects(elapsed, "negative elapsed time");
    let mut memory = valid();
    memory["measurements"]["peak_gpu_memory_bytes"] = json!(-5);
    rejects(memory, "negative peak memory");
}

#[test]
fn the_artifact_cannot_carry_geometry_or_camera_authority() {
    let cases: [(&str, Value); 8] = [
        ("/cameras", json!([{"frame_index": 0, "pose": [1, 0, 0]}])),
        ("/points", json!([[0.0, 0.0, 1.0]])),
        ("/mesh", json!({"triangles": []})),
        ("/provenance", json!("geometric_multi_view")),
        ("/observations/0/world_position", json!([0.1, 0.2, 1.5])),
        ("/observations/0/depth", json!(1.2)),
        ("/tracks/0/mesh_resource", json!("resources/chair.glb")),
        ("/frames/0/pose", json!([1, 0, 0, 0])),
    ];
    for (pointer, value) in cases {
        let mut document = valid();
        let (parent, key) = pointer.rsplit_once('/').unwrap();
        let target = if parent.is_empty() {
            &mut document
        } else {
            document.pointer_mut(parent).unwrap()
        };
        target
            .as_object_mut()
            .unwrap()
            .insert(key.to_owned(), value);
        rejects(document, &format!("geometry authority field {pointer}"));
    }
}

// ---------------------------------------------------------------------------
// Manifest operation

fn manifest(semantic: Value, providers: Value) -> Value {
    json!({
        "schema_version": 1,
        "project_id": "semantic-contract",
        "quality_mode": "standard",
        "inputs": [{"id": "clip", "path": "media/clip.mp4",
                    "content_hash": ContentHash::of_bytes(b"clip").as_str(), "byte_length": 4}],
        "provider_policy": {"execution": "local_only", "providers": providers},
        "operations": [
            {"id": "ingest", "kind": "ingest_video", "inputs": [{"media": "clip"}]},
            {"id": "sparse", "kind": "sparse_reconstruction", "inputs": [{"operation": "ingest"}]},
            {"id": "dense", "kind": "dense_reconstruction",
             "inputs": [{"operation": "ingest"}, {"operation": "sparse"}]},
            semantic
        ],
        "requested_outputs": ["semantic_inventory"]
    })
}

fn local_provider(revision: &str) -> Value {
    json!([{"id": "local-grounded-sam2", "location": "local",
            "capabilities": ["semantic_inventory"], "revision": revision}])
}

fn semantic_operation(inputs: Value) -> Value {
    json!({"id": "semantic", "kind": "semantic_inventory", "provider": "local-grounded-sam2",
           "inputs": inputs})
}

fn load(value: &Value) -> Result<SceneProjectManifest, String> {
    SceneProjectManifest::from_json(&value.to_string()).map_err(|error| error.to_string())
}

#[test]
fn semantic_inventory_is_a_provider_operation_over_keyframes() {
    let kind: OperationKind = serde_json::from_value(json!("semantic_inventory")).unwrap();
    assert_eq!(kind, OperationKind::SemanticInventory);
    assert!(OperationKind::ALL.contains(&OperationKind::SemanticInventory));
    assert_eq!(kind.execution(), OperationExecution::Provider);
    assert_eq!(kind.output(), ArtifactKind::SemanticInventory);
    assert_eq!(
        serde_json::to_value(ArtifactKind::SemanticInventory).unwrap(),
        json!("semantic_inventory")
    );
    let upstream = kind.allowed_upstream();
    assert!(upstream.contains(&OperationKind::IngestVideo));
    assert!(
        upstream
            .iter()
            .all(|kind| matches!(kind, OperationKind::IngestVideo | OperationKind::SparseReconstruction)),
        "semantic inventory consumes only browser keyframes and optional registered-frame identity: {upstream:?}"
    );

    load(&manifest(
        semantic_operation(json!([{"operation": "ingest"}])),
        local_provider("r1"),
    ))
    .expect("keyframes-only semantic inventory");
    load(&manifest(
        semantic_operation(json!([{"operation": "ingest"}, {"operation": "sparse"}])),
        local_provider("r1"),
    ))
    .expect("semantic inventory with registered-frame identity");
    assert!(
        load(&manifest(
            semantic_operation(json!([{"operation": "sparse"}])),
            local_provider("r1")
        ))
        .is_err(),
        "semantic inventory without keyframes"
    );
    assert!(
        load(&manifest(
            semantic_operation(json!([{"operation": "ingest"}, {"operation": "dense"}])),
            local_provider("r1")
        ))
        .is_err(),
        "semantic inventory consuming dense geometry"
    );
    let mut builtin = semantic_operation(json!([{"operation": "ingest"}]));
    builtin.as_object_mut().unwrap().remove("provider");
    assert!(
        load(&manifest(builtin, local_provider("r1"))).is_err(),
        "semantic inventory requires a declared provider"
    );
}

#[test]
fn geometry_and_camera_producers_cannot_consume_semantics() {
    for kind in [
        OperationKind::SparseReconstruction,
        OperationKind::DenseReconstruction,
        OperationKind::SurfaceMesh,
        OperationKind::LearnedReconstruction,
    ] {
        assert!(
            !kind
                .allowed_upstream()
                .contains(&OperationKind::SemanticInventory),
            "{kind:?} must not take geometry or camera authority from semantic evidence"
        );
    }
    let mut document = manifest(
        semantic_operation(json!([{"operation": "ingest"}])),
        local_provider("r1"),
    );
    document["operations"][2]["inputs"]
        .as_array_mut()
        .unwrap()
        .push(json!({"operation": "semantic"}));
    assert!(
        load(&document).is_err(),
        "dense consuming semantic evidence"
    );
}

#[test]
fn vocabulary_override_is_validated_and_part_of_semantic_identity_only() {
    let with = |vocabulary: Value| {
        let mut operation = semantic_operation(json!([{"operation": "ingest"}]));
        operation["vocabulary"] = vocabulary;
        manifest(operation, local_provider("r1"))
    };
    let base = load(&with(json!(["yellow die", "chair"]))).expect("override accepted");
    let declared = base
        .operations
        .iter()
        .find(|operation| operation.id == "semantic")
        .unwrap();
    assert_eq!(
        declared.vocabulary.as_deref(),
        Some(&["yellow die".to_owned(), "chair".to_owned()][..]),
        "override kept verbatim and in order"
    );
    for (value, why) in [
        (json!([]), "empty override"),
        (json!(["chair", ""]), "blank term"),
        (json!(["chair", "chair"]), "duplicate term"),
    ] {
        assert!(load(&with(value)).is_err(), "{why}");
    }
    let mut misplaced = with(json!(["chair"]));
    misplaced["operations"][1]["vocabulary"] = json!(["chair"]);
    assert!(
        load(&misplaced).is_err(),
        "vocabulary on a non-semantic operation"
    );

    let identities = |document: &Value| load(document).unwrap().operation_identities().unwrap();
    let default_vocabulary = identities(&manifest(
        semantic_operation(json!([{"operation": "ingest"}])),
        local_provider("r1"),
    ));
    let first = identities(&with(json!(["yellow die", "chair"])));
    let reordered = identities(&with(json!(["chair", "yellow die"])));
    let revised = identities(&manifest(
        semantic_operation(json!([{"operation": "ingest"}])),
        local_provider("r2"),
    ));
    assert_ne!(first["semantic"], default_vocabulary["semantic"]);
    assert_ne!(
        first["semantic"], reordered["semantic"],
        "the recorded prompt order is part of the evidence identity"
    );
    assert_ne!(revised["semantic"], default_vocabulary["semantic"]);
    for other in ["ingest", "sparse", "dense"] {
        assert_eq!(first[other], default_vocabulary[other], "{other}");
        assert_eq!(revised[other], default_vocabulary[other], "{other}");
    }
}
