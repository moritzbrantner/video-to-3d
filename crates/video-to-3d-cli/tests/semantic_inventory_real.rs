//! Independent real-model acceptance of #60 (Decision Atlas 8.7/8.8): the
//! actual pinned Grounding DINO + SAM 2.1 sidecar on the approved 18-frame TUM
//! browser capture (#179), through the real CLI. Explicit tier: CI has no GPU.
//!
//! ```sh
//! VIDEO_TO_3D_REAL_ROOM_INPUT=/path/to/browser-input \
//! VIDEO_TO_3D_REAL_ROOM_WITNESS_HASH=sha256:... \
//! [VIDEO_TO_3D_SEMANTIC_SIDECAR=/path/to/sidecar] \
//! [VIDEO_TO_3D_SEMANTIC_REPORT=/path/to/report.json] \
//! cargo test --release -p video-to-3d-cli --test semantic_inventory_real -- --ignored --test-threads 1
//! ```
//!
//! Without `VIDEO_TO_3D_SEMANTIC_SIDECAR` the implementation's pinned default
//! sidecar is used. The independently authored inventory reports the yellow
//! die visible in samples 0/1/2/15/16/17; those are not pixel annotations, so
//! no box position is asserted here.
#![cfg(unix)]

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
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
        "semantic-real-{}-{}",
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
        "schema_version": 1, "project_id": "approved-real-room-semantic", "quality_mode": "standard",
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
             "vocabulary": VOCABULARY, "inputs": [{"operation": "ingest"}]}
        ],
        "requested_outputs": ["assembled_scene", "semantic_inventory"]
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
    if let Some(path) = std::env::var_os("VIDEO_TO_3D_SEMANTIC_REPORT") {
        let path = PathBuf::from(path);
        let mut document: Value = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_else(|| json!({}));
        document[entry] = value;
        fs::write(path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    }
}

#[test]
#[ignore = "explicit #60 real-model acceptance: approved TUM capture and a local CUDA GPU required"]
fn real_sidecar_finds_and_tracks_the_yellow_die_on_the_gpu() {
    let samples = die_samples();
    let (document, pixels) = capture();
    let root = project(&document, &pixels);
    let (code, result, stderr) = build(&root.0, false);
    let (_, value) = semantic(&root.0);
    let die_observations: Vec<&Value> = value["observations"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|observation| is_die(observation["label"].as_str().unwrap()))
        .collect();
    let masked_frames: Vec<u64> = samples
        .iter()
        .copied()
        .filter(|frame| {
            die_observations.iter().any(|observation| {
                observation["frame_index"] == *frame && !observation["mask"].is_null()
            })
        })
        .collect();
    report(
        "gpu",
        json!({
            "exit_code": code,
            "operations": result["operations"],
            "status": value["status"],
            "unavailable_reason": value["unavailable_reason"],
            "provider": value["provider"],
            "vocabulary": value["vocabulary"],
            "measurements": value["measurements"],
            "observation_count": value["observations"].as_array().unwrap().len(),
            "track_count": value["tracks"].as_array().unwrap().len(),
            "die_observations": die_observations.iter().map(|observation| json!({
                "frame_index": observation["frame_index"], "label": observation["label"],
                "bbox": observation["bbox"], "detection_confidence": observation["detection_confidence"],
                "mask_confidence": observation["mask_confidence"], "masked": !observation["mask"].is_null(),
            })).collect::<Vec<_>>(),
            "die_masked_visible_samples": masked_frames,
            "uncertainties": value["uncertainties"],
        }),
    );

    assert_eq!(
        result["operations"]["semantic"]["state"], "succeeded",
        "{stderr}"
    );
    assert_eq!(
        value["status"], "available",
        "{}",
        value["unavailable_reason"]
    );
    assert_eq!(
        value["vocabulary"],
        json!({"source": "project", "terms": VOCABULARY})
    );
    let runtime = &value["provider"]["runtime"];
    assert_eq!(runtime["device"], "cuda");
    let card = runtime["device_name"].as_str().unwrap_or_default();
    assert!(
        card.contains("3060"),
        "exact 3060-class card recorded: {card}"
    );
    let models = value["provider"]["models"].as_array().unwrap();
    for role in ["detector", "segmenter"] {
        assert!(
            models.iter().any(|model| model["role"] == role),
            "{role} identity recorded"
        );
    }
    let measurements = &value["measurements"];
    let elapsed = measurements["elapsed_seconds"].as_f64().unwrap();
    let peak = measurements["peak_gpu_memory_bytes"].as_u64().unwrap();
    assert!(elapsed > 0.0);
    assert!(
        peak > 0 && peak <= 8 * 1024 * 1024 * 1024,
        "measured 8 GB fit: {peak}"
    );

    assert!(!value["observations"].as_array().unwrap().is_empty());
    assert!(
        masked_frames.len() >= 2,
        "die-like masks on at least two inventory-visible samples: {masked_frames:?}"
    );
    let tracked = value["tracks"].as_array().unwrap().iter().any(|track| {
        is_die(track["label"].as_str().unwrap())
            && track["observations"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|id| {
                    die_observations
                        .iter()
                        .find(|observation| observation["id"] == *id)
                })
                .filter(|observation| {
                    samples.contains(&observation["frame_index"].as_u64().unwrap())
                })
                .count()
                >= 2
    });
    assert!(
        tracked,
        "one die-like track links at least two visible samples"
    );
    for frame in samples
        .iter()
        .filter(|frame| !masked_frames.contains(frame))
    {
        assert!(
            value["uncertainties"]
                .as_array()
                .unwrap()
                .iter()
                .any(|record| {
                    record["frame_index"] == *frame && record["label"].as_str().is_some_and(is_die)
                }),
            "visible sample {frame} without a die mask must be recorded as missing/weak/ambiguous"
        );
    }
    for id in ["sparse", "dense", "mesh", "assemble"] {
        assert!(
            ["succeeded", "reused"].contains(&result["operations"][id]["state"].as_str().unwrap()),
            "{id}"
        );
    }
}

#[test]
#[ignore = "explicit #60 CPU-only smoke: approved TUM capture and the real sidecar required"]
fn real_sidecar_without_gpu_reports_unavailable_and_build_stays_usable() {
    let (document, pixels) = capture();
    let root = project(&document, &pixels);
    let (code, result, stderr) = build(&root.0, true);
    let (_, value) = semantic(&root.0);
    report(
        "cpu_only",
        json!({
            "exit_code": code,
            "operations": result["operations"],
            "status": value["status"],
            "unavailable_reason": value["unavailable_reason"],
        }),
    );
    assert!(![1, 2, 64].contains(&code), "{stderr}");
    assert_eq!(value["status"], "unavailable");
    assert!(!value["unavailable_reason"]
        .as_str()
        .unwrap()
        .trim()
        .is_empty());
    assert_eq!(value["observations"], json!([]));
    assert_eq!(value["tracks"], json!([]));
    for id in ["sparse", "dense", "mesh", "assemble"] {
        assert!(
            ["succeeded", "reused"].contains(&result["operations"][id]["state"].as_str().unwrap()),
            "{id}: {stderr}"
        );
    }
}
