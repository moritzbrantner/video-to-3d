//! Real-video object identity and mask consistency (#221, for #61/#62): the
//! actual semantic sidecar on the approved 18-frame TUM capture (#179), then
//! the built-in `object_identity` and `object_masks` operations through the
//! real CLI. Explicit tier (local CUDA GPU):
//!
//! ```sh
//! VIDEO_TO_3D_REAL_ROOM_INPUT=/path/to/browser-input \
//! VIDEO_TO_3D_REAL_ROOM_WITNESS_HASH=sha256:... \
//! VIDEO_TO_3D_SEMANTIC_SIDECAR=tools/semantic-sidecar/video-to-3d-semantic-sidecar \
//! [VIDEO_TO_3D_OBJECT_REPORT=/path/to/report.json] \
//! cargo test --release -p video-to-3d-cli --test object_identity_real -- --ignored
//! ```
#![cfg(unix)]

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use video_to_3d_core::object_identity::ObjectIdentityArtifact;
use video_to_3d_core::object_masks::ObjectMasksArtifact;
use video_to_3d_core::object_references::ObjectReferencesArtifact;
use video_to_3d_core::scene_artifacts::{
    write_project_keyframes_artifact, KeyframesArtifact, SampledFrame,
};
use video_to_3d_core::scene_project::{ContentHash, OperationKind, ProjectPath};
use video_to_3d_core::scene_runner::{
    AttemptOutcome, CancellationToken, OperationExecutor, OperationRequest, ProducedArtifact,
    RunOptions,
};
use video_to_3d_core::scene_store::{ProjectStore, Reproducibility};
use video_to_3d_core::semantic_inventory::SemanticInventoryArtifact;

const VOCABULARY: [&str; 9] = [
    "yellow die",
    "teddy bear",
    "globe",
    "chair",
    "desk",
    "monitor",
    "keyboard",
    "wall",
    "floor",
];

fn die_samples() -> Vec<u64> {
    let expected: Value = serde_json::from_str(include_str!(
        "../../../tools/tests/fixtures/approved-controlled-room-expectations.json"
    ))
    .unwrap();
    let text = expected.to_string();
    assert!(text.contains("yellow-die"));
    fn find(value: &Value) -> Option<Vec<u64>> {
        match value {
            Value::Object(map) if map.get("id") == Some(&json!("yellow-die")) => Some(
                map["visible_samples"]
                    .as_array()?
                    .iter()
                    .filter_map(Value::as_u64)
                    .collect(),
            ),
            Value::Object(map) => map.values().find_map(find),
            Value::Array(items) => items.iter().find_map(find),
            _ => None,
        }
    }
    let samples = find(&expected).expect("independent yellow-die inventory");
    assert_eq!(samples, vec![0, 1, 2, 15, 16, 17], "inventory unchanged");
    samples
}

fn is_die(label: &str) -> bool {
    let label = label.to_ascii_lowercase();
    label.contains("die") || label.contains("dice") || label.contains("cube")
}

/// The pinned #179 browser capture: witness identity plus per-frame pixels.
fn capture() -> (Value, Vec<Vec<u8>>) {
    let root = PathBuf::from(
        std::env::var_os("VIDEO_TO_3D_REAL_ROOM_INPUT")
            .expect("run the #179 browser capture first"),
    );
    let witness = std::env::var("VIDEO_TO_3D_REAL_ROOM_WITNESS_HASH")
        .expect("pin the independently captured transfer identity");
    let bytes = fs::read(root.join("browser-frames.json")).unwrap();
    assert_eq!(ContentHash::of_bytes(&bytes).as_str(), witness);
    let document: Value = serde_json::from_slice(&bytes).unwrap();
    let frames = document["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 18);
    let pixels = frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            assert_eq!(frame["frame_index"], index);
            let bytes = fs::read(root.join(frame["path"].as_str().unwrap())).unwrap();
            assert_eq!(bytes.len(), 360 * 270 * 4);
            assert_eq!(
                frame["content_hash"],
                ContentHash::of_bytes(&bytes).as_str()
            );
            bytes
        })
        .collect();
    (document, pixels)
}

struct Import<'a> {
    root: &'a Path,
    document: &'a Value,
    pixels: &'a [Vec<u8>],
}
impl OperationExecutor for Import<'_> {
    fn execute(&self, operation: &OperationRequest, _: &CancellationToken) -> AttemptOutcome {
        if operation.operation.kind != OperationKind::IngestVideo {
            return AttemptOutcome::Unsupported("actual CLI owns the rest".into());
        }
        let (_, manifest) = ProjectStore::open(&self.root.join("project.json")).unwrap();
        let frames: Vec<_> = self
            .pixels
            .iter()
            .enumerate()
            .map(|(index, rgba)| SampledFrame {
                frame_index: index,
                timestamp_seconds: self.document["frames"][index]["presented_time"]
                    .as_f64()
                    .unwrap(),
                width: 360,
                height: 270,
                rgba,
            })
            .collect();
        let path = ProjectPath::new("artifacts/browser/keyframes.json").unwrap();
        let (_, hash) =
            write_project_keyframes_artifact(self.root, &manifest, &path, &frames).unwrap();
        AttemptOutcome::Succeeded(ProducedArtifact::new(
            path,
            hash,
            Reproducibility::Deterministic,
        ))
    }
}

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn project(document: &Value, pixels: &[Vec<u8>]) -> Temporary {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "objects-real-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst)
    ));
    fs::create_dir_all(root.join("media")).unwrap();
    // The capture metadata stands in as the media identity; no native decoding.
    let source = fs::read(
        PathBuf::from(std::env::var_os("VIDEO_TO_3D_REAL_ROOM_INPUT").unwrap())
            .join("browser-frames.json"),
    )
    .unwrap();
    fs::write(root.join("media/browser-frames.json"), &source).unwrap();
    fs::write(root.join("project.json"), json!({
        "schema_version": 1, "project_id": "approved-real-room-objects", "quality_mode": "standard",
        "inputs": [{"id": "clip", "path": "media/browser-frames.json",
                    "content_hash": ContentHash::of_bytes(&source).as_str(), "byte_length": source.len()}],
        "provider_policy": {"execution": "local_only", "providers": [{
            "id": "local-grounded-sam2", "location": "local",
            "capabilities": ["semantic_inventory"], "revision": "real-acceptance"
        }]},
        "operations": [
            {"id": "ingest", "kind": "ingest_video", "inputs": [{"media": "clip"}]},
            {"id": "sparse", "kind": "sparse_reconstruction", "inputs": [{"operation": "ingest"}]},
            {"id": "dense", "kind": "dense_reconstruction", "inputs": [{"operation": "ingest"}, {"operation": "sparse"}]},
            {"id": "mesh", "kind": "surface_mesh", "inputs": [{"operation": "dense"}]},
            {"id": "assemble", "kind": "scene_assembly", "inputs": [{"operation": "sparse"}, {"operation": "mesh"}]},
            {"id": "semantic", "kind": "semantic_inventory", "provider": "local-grounded-sam2",
             "vocabulary": VOCABULARY, "inputs": [{"operation": "ingest"}]},
            {"id": "objects", "kind": "object_identity",
             "inputs": [{"operation": "semantic"}, {"operation": "sparse"}, {"operation": "mesh"}]},
            {"id": "masks", "kind": "object_masks",
             "inputs": [{"operation": "semantic"}, {"operation": "objects"}, {"operation": "sparse"}, {"operation": "mesh"}]},
            {"id": "references", "kind": "object_references",
             "inputs": [{"operation": "semantic"}, {"operation": "objects"}, {"operation": "masks"}, {"operation": "sparse"}, {"operation": "mesh"}]}
        ],
        "requested_outputs": ["assembled_scene", "object_identity", "object_masks", "object_references"]
    }).to_string()).unwrap();
    let (store, mut manifest) = ProjectStore::open(&root.join("project.json")).unwrap();
    let report = store
        .build(
            &mut manifest,
            &Import {
                root: &root,
                document,
                pixels,
            },
            RunOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
    assert!(report.run.states["ingest"].is_complete());
    Temporary(root)
}

fn build(root: &Path, hide_gpu: bool) -> (i32, Value, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_video-to-3d"));
    command
        .args(["build", root.join("project.json").to_str().unwrap()])
        .current_dir(std::env::temp_dir());
    if hide_gpu {
        command.env("CUDA_VISIBLE_DEVICES", "");
    }
    let output = command.output().unwrap();
    (
        output.status.code().unwrap(),
        serde_json::from_slice(&output.stdout).expect("actual CLI must emit JSON"),
        String::from_utf8(output.stderr).unwrap(),
    )
}

fn semantic(root: &Path) -> (SemanticInventoryArtifact, Value) {
    let (_, manifest) = ProjectStore::open(&root.join("project.json")).unwrap();
    let read = |id: &str| {
        let record = manifest
            .artifacts
            .iter()
            .find(|artifact| artifact.produced_by == id)
            .unwrap_or_else(|| panic!("{id} artifact recorded"));
        fs::read_to_string(record.path.resolve(root)).unwrap()
    };
    let artifact = SemanticInventoryArtifact::from_json(&read("semantic")).unwrap();
    artifact
        .validate_against_keyframes(&KeyframesArtifact::from_json(&read("ingest")).unwrap())
        .unwrap();
    let value = serde_json::from_str(&artifact.to_json()).unwrap();
    (artifact, value)
}

fn report(entry: &str, value: Value) {
    eprintln!("{entry}: {value:#}");
    if let Some(path) = std::env::var_os("VIDEO_TO_3D_OBJECT_REPORT") {
        let path = PathBuf::from(path);
        let mut document: Value = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_else(|| json!({}));
        document[entry] = value;
        fs::write(path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    }
}

fn read(root: &Path, id: &str) -> String {
    let (_, manifest) = ProjectStore::open(&root.join("project.json")).unwrap();
    let record = manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.produced_by == id)
        .unwrap_or_else(|| panic!("{id} artifact recorded"));
    fs::read_to_string(record.path.resolve(root)).unwrap()
}

#[test]
#[ignore = "explicit #221 real-video tier: approved TUM capture and a local CUDA GPU required"]
fn real_yellow_die_keeps_one_identity_and_its_masks_are_diagnosed() {
    let samples = die_samples();
    let (document, pixels) = capture();
    let root = project(&document, &pixels);
    let (code, result, stderr) = build(&root.0, false);
    let (_, semantic_value) = semantic(&root.0);
    let identities = ObjectIdentityArtifact::from_json(&read(&root.0, "objects")).unwrap();
    let masks = ObjectMasksArtifact::from_json(&read(&root.0, "masks")).unwrap();
    let frame_of = |id: &str| {
        semantic_value["observations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|observation| observation["id"] == id)
            .and_then(|observation| observation["frame_index"].as_u64())
            .unwrap()
    };
    let dice: Vec<_> = identities
        .objects
        .iter()
        .filter(|object| is_die(&object.label))
        .collect();
    let best = dice
        .iter()
        .max_by_key(|object| {
            object
                .observations
                .iter()
                .filter(|id| samples.contains(&frame_of(id)))
                .count()
        })
        .expect("a die identity");
    let best_samples: std::collections::BTreeSet<u64> = best
        .observations
        .iter()
        .map(|id| frame_of(id))
        .filter(|frame| samples.contains(frame))
        .collect();
    let die_masks: Vec<_> = masks
        .masks
        .iter()
        .filter(|mask| mask.object == best.id)
        .collect();
    let references = ObjectReferencesArtifact::from_json(&read(&root.0, "references")).unwrap();
    let die_package = references
        .packages
        .iter()
        .find(|package| package.object == best.id);
    report(
        "objects",
        json!({
            "exit_code": code,
            "operations": result["operations"],
            "registered_cameras": result["operations"]["sparse"],
            "objects": identities.objects.len(),
            "ambiguities": identities.ambiguities.len(),
            "die_identities": dice.len(),
            "best_die": {"id": best.id, "label": best.label, "tracks": best.source_tracks,
                          "samples": best_samples, "motion": best.motion, "reason": best.motion_reason,
                          "locations": best.locations.len()},
            "die_masks": die_masks.iter().map(|mask| json!({
                "frame": mask.frame_index, "consistency": mask.consistency,
                "support": mask.support, "quality": mask.quality, "reason": mask.reason})).collect::<Vec<_>>(),
            "die_package": die_package.map(|package| json!({
                "primary": package.primary.frame_index,
                "additional": package.additional.iter().map(|view| view.frame_index).collect::<Vec<_>>(),
                "rejected": package.rejected.iter().map(|view| json!({"frame": view.frame_index, "reason": view.reason})).collect::<Vec<_>>(),
                "observed_points": package.observed_points.len(),
                "extent": package.extent,
            })),
            "packages": references.packages.len(),
            "unpackaged": references.unpackaged.len(),
        }),
    );
    assert_eq!(
        result["operations"]["objects"]["state"], "succeeded",
        "{stderr}"
    );
    assert_eq!(
        result["operations"]["masks"]["state"], "succeeded",
        "{stderr}"
    );
    assert!(
        best_samples.iter().any(|frame| *frame <= 2)
            && best_samples.iter().any(|frame| *frame >= 15),
        "one die identity spans the separated sample groups 0-2 and 15-17: {best_samples:?}"
    );
    assert!(!best.motion_reason.is_empty());
    assert_eq!(
        die_masks.len(),
        best.observations.len(),
        "every die mask is diagnosed"
    );
}
