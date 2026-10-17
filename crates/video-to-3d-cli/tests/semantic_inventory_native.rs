//! Independent acceptance of #60 (first slice, Decision Atlas 8.8): the native
//! `semantic_inventory` provider operation in the real CLI/ProjectStore.
//! Authored before production code; contract and protocol in
//! `docs/specs/semantic-inventory-acceptance.md`.
//!
//! The model runtime is an external sidecar program. These tests replace it
//! with a deterministic test double (a POSIX shell script) through the
//! documented sidecar protocol `video-to-3d-semantic-sidecar/1`:
//!
//! - the program is resolved from `BuiltInExecutor::with_semantic_sidecar`,
//!   else the `VIDEO_TO_3D_SEMANTIC_SIDECAR` environment variable (an
//!   executable path), else the implementation's pinned default sidecar;
//! - it is invoked as `<program> <request.json> <response.json>` (absolute
//!   paths), writes the response file and exits 0;
//! - a program that cannot be started yields a recorded *unavailable* result.
//!
//! The browser keyframe handoff is imported by a test-only executor (as in
//! `native_build.rs`); every other operation runs in the actual CLI/core.
#![cfg(unix)]

use serde_json::{json, Value};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use video_to_3d_core::scene_artifacts::{
    write_keyframes_artifact, BuiltInExecutor, KeyframesArtifact, SampledFrame,
};
use video_to_3d_core::scene_model::AssembledScene;
use video_to_3d_core::scene_project::{
    ArtifactKind, ContentHash, OperationKind, ProjectPath, SceneProjectError, SceneProjectManifest,
};
use video_to_3d_core::scene_runner::{
    AttemptOutcome, CancellationToken, OperationExecutor, OperationRequest, OperationState,
    ProducedArtifact, RunOptions,
};
use video_to_3d_core::scene_store::{hash_file, ProjectStore, Reproducibility, VerifyMode};
use video_to_3d_core::semantic_inventory::{default_room_vocabulary, SemanticInventoryArtifact};
use video_to_3d_core::ReconstructionRequest;

const PROTOCOL: &str = "video-to-3d-semantic-sidecar/1";
const PROVIDER: &str = "local-grounded-sam2";
const GEOMETRY: [&str; 4] = ["sparse", "dense", "mesh", "assemble"];

#[allow(dead_code)]
mod golden {
    pub fn request() -> ReconstructionRequest {
        let points = scene();
        ReconstructionRequest {
            frames: (0..FRAME_COUNT)
                .map(|frame| render_frame(frame, &points, Scenario::Lateral))
                .collect(),
            options: ReconstructionOptions::default(),
        }
    }

    include!("../../video-to-3d-core/examples/golden_fixture.rs");
}

fn request() -> &'static ReconstructionRequest {
    static REQUEST: OnceLock<ReconstructionRequest> = OnceLock::new();
    REQUEST.get_or_init(golden::request)
}

/// Test-only browser keyframe handoff. It has no reconstruction or semantics.
struct ImportFrames(PathBuf);
impl OperationExecutor for ImportFrames {
    fn execute(&self, operation: &OperationRequest, _: &CancellationToken) -> AttemptOutcome {
        if operation.operation.kind != OperationKind::IngestVideo {
            return AttemptOutcome::Unsupported("left to the real CLI".into());
        }
        let frames: Vec<_> = request()
            .frames
            .iter()
            .enumerate()
            .map(|(index, frame)| SampledFrame {
                frame_index: index,
                timestamp_seconds: index as f64 * 0.5,
                width: frame.width,
                height: frame.height,
                rgba: &frame.rgba,
            })
            .collect();
        let path = ProjectPath::new("artifacts/browser/keyframes.json").unwrap();
        let (_, hash) = write_keyframes_artifact(&self.0, &path, &frames).unwrap();
        AttemptOutcome::Succeeded(ProducedArtifact::new(
            path,
            hash,
            Reproducibility::Deterministic,
        ))
    }
}

struct Project {
    root: PathBuf,
    /// Outside the project: the double's script, response and request log.
    double: PathBuf,
}

impl Project {
    fn new(name: &str, semantic: bool) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let base = std::env::temp_dir().join(format!(
            "video-to-3d-semantic-{name}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let root = base.join("project");
        let double = base.join("double");
        fs::create_dir_all(root.join("media")).unwrap();
        fs::create_dir_all(double.join("requests")).unwrap();
        let source: &[u8] = b"golden_fixture.rs/lateral: deterministic generated sampled frames";
        fs::write(root.join("media/generated-source.txt"), source).unwrap();
        let mut operations = vec![
            json!({ "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] }),
            json!({ "id": "sparse", "kind": "sparse_reconstruction", "inputs": [{ "operation": "ingest" }] }),
            json!({ "id": "dense", "kind": "dense_reconstruction", "inputs": [{ "operation": "ingest" }, { "operation": "sparse" }] }),
            json!({ "id": "mesh", "kind": "surface_mesh", "inputs": [{ "operation": "dense" }] }),
            json!({ "id": "assemble", "kind": "scene_assembly", "inputs": [{ "operation": "sparse" }, { "operation": "mesh" }] }),
        ];
        let mut outputs = vec![json!("assembled_scene")];
        let mut providers = json!([]);
        if semantic {
            operations.push(json!({
                "id": "semantic", "kind": "semantic_inventory", "provider": PROVIDER,
                "inputs": [{ "operation": "ingest" }]
            }));
            outputs.push(json!("semantic_inventory"));
            providers = json!([{
                "id": PROVIDER, "location": "local",
                "capabilities": ["semantic_inventory"], "revision": "acceptance-r1"
            }]);
        }
        let document = json!({
            "schema_version": 1,
            "project_id": "native-semantic-inventory",
            "quality_mode": "standard",
            "inputs": [{
                "id": "clip", "path": "media/generated-source.txt",
                "content_hash": ContentHash::of_bytes(source).as_str(),
                "byte_length": source.len()
            }],
            "provider_policy": { "execution": "local_only", "providers": providers },
            "operations": operations,
            "requested_outputs": outputs
        });
        fs::write(root.join("project.json"), document.to_string()).unwrap();
        let project = Self { root, double };
        project.write_double("cp \"$D/response.json\" \"$2\"");
        project.set_response(&available_response(&["chair", "wall"]));
        project.import_frames();
        project
    }

    /// Install the double: it logs every request, then runs `body`.
    fn write_double(&self, body: &str) {
        let script = format!(
            "#!/bin/sh\nset -eu\nD='{}'\nn=$(ls \"$D/requests\" | wc -l)\ncp \"$1\" \"$D/requests/$n.json\"\n{body}\n",
            self.double.display()
        );
        let path = self.sidecar();
        fs::write(&path, script).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    fn sidecar(&self) -> PathBuf {
        self.double.join("sidecar.sh")
    }

    fn set_response(&self, response: &Value) {
        fs::write(self.double.join("response.json"), response.to_string()).unwrap();
    }

    fn requests(&self) -> Vec<Value> {
        let mut requests = Vec::new();
        for index in 0.. {
            let path = self.double.join(format!("requests/{index}.json"));
            let Ok(bytes) = fs::read(path) else { break };
            requests.push(serde_json::from_slice(&bytes).unwrap());
        }
        requests
    }

    fn import_frames(&self) {
        let (store, mut manifest) = self.open();
        let report = store
            .build(
                &mut manifest,
                &ImportFrames(self.root.clone()),
                RunOptions::default(),
                &CancellationToken::new(),
            )
            .unwrap();
        assert!(report.run.states["ingest"].is_complete());
    }

    fn manifest(&self) -> PathBuf {
        self.root.join("project.json")
    }

    fn open(&self) -> (ProjectStore, SceneProjectManifest) {
        ProjectStore::open(&self.manifest()).unwrap()
    }

    fn record(&self, id: &str) -> Option<video_to_3d_core::scene_project::ArtifactRecord> {
        self.open()
            .1
            .artifacts
            .into_iter()
            .find(|artifact| artifact.produced_by == id)
    }

    fn document(&self, id: &str) -> String {
        let record = self
            .record(id)
            .unwrap_or_else(|| panic!("{id} has no artifact"));
        fs::read_to_string(record.path.resolve(&self.root)).unwrap()
    }

    fn keyframes(&self) -> KeyframesArtifact {
        KeyframesArtifact::from_json(&self.document("ingest")).unwrap()
    }

    fn semantic(&self) -> SemanticInventoryArtifact {
        let record = self.record("semantic").expect("semantic artifact recorded");
        assert_eq!(record.kind, ArtifactKind::SemanticInventory);
        assert_eq!(record.provider.as_deref(), Some(PROVIDER));
        let artifact = SemanticInventoryArtifact::from_json(&self.document("semantic"))
            .expect("recorded artifact passes shared validation");
        artifact
            .validate_against_keyframes(&self.keyframes())
            .expect("evidence frames are the source keyframes");
        artifact
    }

    fn semantic_value(&self) -> Value {
        serde_json::from_str(&self.semantic().to_json()).unwrap()
    }

    /// Edit the persisted manifest document (keeping recorded artifacts).
    fn edit(&self, change: impl FnOnce(&mut Value)) {
        let mut document: Value =
            serde_json::from_slice(&fs::read(self.manifest()).unwrap()).unwrap();
        change(&mut document);
        fs::write(self.manifest(), document.to_string()).unwrap();
    }

    fn build_with(&self, sidecar: &Path) -> Output {
        let output = Command::new(env!("CARGO_BIN_EXE_video-to-3d"))
            .arg("build")
            .arg(self.manifest())
            .env("VIDEO_TO_3D_SEMANTIC_SIDECAR", sidecar)
            .current_dir(std::env::temp_dir())
            .output()
            .unwrap();
        Output {
            code: output.status.code().unwrap(),
            json: serde_json::from_slice(&output.stdout).expect("one JSON stdout document"),
            stderr: String::from_utf8(output.stderr).unwrap(),
        }
    }

    fn build(&self) -> Output {
        self.build_with(&self.sidecar())
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        if let Some(base) = self.root.parent() {
            let _ = fs::remove_dir_all(base);
        }
    }
}

struct Output {
    code: i32,
    json: Value,
    stderr: String,
}

impl Output {
    fn state(&self, id: &str) -> &str {
        self.json["operations"][id]["state"]
            .as_str()
            .unwrap_or_else(|| panic!("{id} has no state: {}", self.json))
    }

    #[track_caller]
    fn assert_geometry_completed(&self) {
        for id in GEOMETRY {
            assert!(
                ["succeeded", "reused"].contains(&self.state(id)),
                "{id}: {} / {}",
                self.json,
                self.stderr
            );
        }
    }
}

/// Golden keyframes are 640x480.
const W: u32 = 640;
const H: u32 = 480;

fn rect_mask(x0: u32, y0: u32, x1: u32, y1: u32) -> Value {
    let mut counts = Vec::new();
    let (mut current, mut run) = (false, 0_u64);
    for y in 0..H {
        for x in 0..W {
            let inside = x >= x0 && x < x1 && y >= y0 && y < y1;
            if inside != current {
                counts.push(run);
                (current, run) = (inside, 0);
            }
            run += 1;
        }
    }
    counts.push(run);
    json!({"encoding": "rle_row_major", "width": W, "height": H, "counts": counts})
}

/// Fixed detections: `labels[0]` tracked over frames 0 and 1 with masks,
/// `labels[1]` a weak box-only detection on frame 2, and an explicit gap.
fn available_response(labels: &[&str; 2]) -> Value {
    let [tracked, weak] = labels;
    json!({
        "protocol": PROTOCOL,
        "status": "available",
        "models": [
            {"role": "detector", "id": "test-double/detector", "version": "1",
             "url": "https://example.invalid/test-double/detector"},
            {"role": "segmenter", "id": "test-double/segmenter", "version": "1",
             "url": "https://example.invalid/test-double/segmenter"}
        ],
        "runtime": {"name": "test-double", "version": "1", "device": "cpu", "device_name": null},
        "observations": [
            {"id": "obs-0", "frame_index": 0, "label": tracked, "bbox": [100.0, 100.0, 140.0, 130.0],
             "detection_confidence": 0.7, "mask": rect_mask(100, 100, 140, 130), "mask_confidence": 0.9},
            {"id": "obs-1", "frame_index": 1, "label": tracked, "bbox": [110.0, 100.0, 150.0, 130.0],
             "detection_confidence": 0.6, "mask": rect_mask(112, 102, 148, 128), "mask_confidence": 0.8},
            {"id": "obs-2", "frame_index": 2, "label": weak, "bbox": [0.0, 0.0, 640.0, 200.0],
             "detection_confidence": 0.3, "mask": null, "mask_confidence": null}
        ],
        "tracks": [
            {"id": "track-0", "label": tracked, "observations": ["obs-0", "obs-1"],
             "association_confidence": 0.75}
        ],
        "uncertainties": [
            {"kind": "missing", "frame_index": 2, "label": tracked, "observation": null,
             "reason": "test double: tracked object not found"},
            {"kind": "weak", "frame_index": 2, "label": weak, "observation": "obs-2",
             "reason": "test double: no mask"}
        ],
        "measurements": {"elapsed_seconds": 0.25, "peak_gpu_memory_bytes": null}
    })
}

fn unavailable_response(reason: &str) -> Value {
    json!({"protocol": PROTOCOL, "status": "unavailable", "reason": reason})
}

#[track_caller]
fn assert_request(request: &Value, project: &Project, vocabulary: &[String]) {
    assert_eq!(request["protocol"], PROTOCOL);
    assert_eq!(request["provider"]["id"], PROVIDER);
    assert!(request["provider"]["revision"].is_string());
    assert_eq!(
        request["vocabulary"],
        json!(vocabulary),
        "request carries the recorded vocabulary verbatim"
    );
    let keyframes = project.keyframes();
    let frames = request["frames"].as_array().unwrap();
    assert_eq!(
        frames.len(),
        keyframes.frames.len(),
        "every keyframe is offered"
    );
    for (frame, keyframe) in frames.iter().zip(&keyframes.frames) {
        assert_eq!(frame["frame_index"], keyframe.frame_index);
        assert_eq!(frame["width"], keyframe.width);
        assert_eq!(frame["height"], keyframe.height);
        assert_eq!(frame["content_hash"], keyframe.content_hash.as_str());
        let path = PathBuf::from(frame["rgba_path"].as_str().unwrap());
        assert!(path.is_absolute(), "sidecar receives absolute local paths");
        let (hash, length) = hash_file(&path).expect("pixels are readable by the sidecar");
        assert_eq!(hash, keyframe.content_hash);
        assert_eq!(length, u64::from(keyframe.width * keyframe.height * 4));
    }
    let text = request.to_string();
    assert!(
        !text.contains("http://") && !text.contains("https://"),
        "no remote endpoints"
    );
}

#[test]
fn local_provider_records_validated_evidence_with_receipt_and_reuse() {
    let project = Project::new("available", true);
    let result = project.build();
    assert_eq!(result.code, 0, "{}", result.stderr);
    assert_eq!(result.json["complete"], true);
    result.assert_geometry_completed();
    assert_eq!(result.state("semantic"), "succeeded");
    assert_eq!(result.json["operations"]["semantic"]["provider"], PROVIDER);
    assert_eq!(
        result.json["operations"]["semantic"]["observations"]["segmentation"],
        "available"
    );

    let requests = project.requests();
    assert_eq!(requests.len(), 1, "the sidecar runs once");
    assert_request(&requests[0], &project, &default_room_vocabulary());

    let value = project.semantic_value();
    let response = available_response(&["chair", "wall"]);
    assert_eq!(value["status"], "available");
    assert_eq!(value["provider"]["id"], PROVIDER);
    assert_eq!(value["provider"]["revision"], "acceptance-r1");
    assert_eq!(value["provider"]["models"], response["models"]);
    assert_eq!(value["provider"]["runtime"], response["runtime"]);
    assert_eq!(
        value["vocabulary"],
        json!({"source": "default", "terms": default_room_vocabulary()})
    );
    assert_eq!(value["tracks"], response["tracks"]);
    let observations = value["observations"].as_array().unwrap();
    assert_eq!(observations.len(), 3);
    for (observation, expected) in observations
        .iter()
        .zip(response["observations"].as_array().unwrap())
    {
        for key in ["id", "frame_index", "label", "bbox", "detection_confidence"] {
            assert_eq!(observation[key], expected[key], "{key}");
        }
        if !expected["mask"].is_null() {
            assert_eq!(observation["mask"], expected["mask"]);
        }
    }
    assert!(
        observations[2].get("mask").is_none_or(Value::is_null),
        "no invented mask"
    );
    assert_eq!(value["uncertainties"].as_array().unwrap().len(), 2);
    let frames: Vec<u64> = value["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|frame| frame["frame_index"].as_u64().unwrap())
        .collect();
    assert_eq!(frames, (0..8).collect::<Vec<_>>());

    let (store, manifest) = project.open();
    let record = project.record("semantic").unwrap();
    let receipt = store.read_receipt("semantic").unwrap().expect("receipt");
    assert_eq!(receipt.output.content_hash, record.content_hash);
    assert_eq!(receipt.output.path, record.path);
    assert_eq!(receipt.operation_identity, record.operation_identity);
    assert_eq!(
        hash_file(&record.path.resolve(&project.root)).unwrap(),
        (
            receipt.output.content_hash.clone(),
            receipt.output.byte_length
        )
    );
    let mut verified = manifest.clone();
    assert!(store
        .reconcile(&mut verified, VerifyMode::Full)
        .unwrap()
        .invalidated
        .is_empty());

    let rebuilt = project.build();
    assert_eq!(rebuilt.code, 0, "{}", rebuilt.stderr);
    assert_eq!(rebuilt.json["executed"], json!([]));
    assert_eq!(rebuilt.state("semantic"), "reused");
    assert_eq!(
        project.requests().len(),
        1,
        "reuse does not rerun the sidecar"
    );
}

#[test]
fn vocabulary_override_and_provider_revision_invalidate_only_semantics() {
    let project = Project::new("invalidate", true);
    assert_eq!(project.build().code, 0);
    let geometry: Vec<String> = GEOMETRY.iter().map(|id| project.document(id)).collect();

    let override_terms = vec![
        "Yellow die".to_owned(),
        "chair".to_owned(),
        "wall".to_owned(),
    ];
    project.edit(|document| {
        let operations = document["operations"].as_array_mut().unwrap();
        let semantic = operations
            .iter_mut()
            .find(|operation| operation["id"] == "semantic")
            .unwrap();
        semantic["vocabulary"] = json!(override_terms);
    });
    project.set_response(&available_response(&["Yellow die", "wall"]));
    let result = project.build();
    assert_eq!(result.code, 0, "{}", result.stderr);
    assert_eq!(result.json["executed"], json!(["semantic"]));
    let requests = project.requests();
    assert_eq!(requests.len(), 2);
    assert_request(&requests[1], &project, &override_terms);
    let value = project.semantic_value();
    assert_eq!(
        value["vocabulary"],
        json!({"source": "project", "terms": override_terms}),
        "override recorded verbatim with the evidence"
    );
    assert_eq!(value["tracks"][0]["label"], "Yellow die");

    project.edit(|document| {
        document["provider_policy"]["providers"][0]["revision"] = json!("acceptance-r2");
    });
    let revised = project.build();
    assert_eq!(revised.code, 0, "{}", revised.stderr);
    assert_eq!(revised.json["executed"], json!(["semantic"]));
    assert_eq!(
        project.requests()[2]["provider"]["revision"],
        "acceptance-r2"
    );
    assert_eq!(
        project.semantic_value()["provider"]["revision"],
        "acceptance-r2"
    );
    for (id, before) in GEOMETRY.iter().zip(&geometry) {
        assert_eq!(&project.document(id), before, "{id} unchanged by semantics");
    }
}

#[test]
fn semantic_evidence_never_changes_cameras_or_geometry() {
    let with = Project::new("with-semantics", true);
    let without = Project::new("without-semantics", false);
    assert_eq!(with.build().code, 0);
    let plain = without.build();
    assert_eq!(plain.code, 0, "{}", plain.stderr);
    for id in GEOMETRY {
        assert_eq!(
            with.document(id),
            without.document(id),
            "{id}: semantic inventory must not alter cameras, geometry or assembly"
        );
    }
    let scene = AssembledScene::from_json(&with.document("assemble")).unwrap();
    scene.validate().unwrap();
}

#[test]
fn missing_sidecar_records_unavailable_evidence_and_build_stays_usable() {
    let project = Project::new("no-sidecar", true);
    let absent = project.double.join("not-installed");
    let result = project.build_with(&absent);
    assert!(
        ![1, 2, 64].contains(&result.code),
        "an unavailable provider is not an internal/project error: {} {}",
        result.code,
        result.stderr
    );
    result.assert_geometry_completed();
    AssembledScene::from_json(&project.document("assemble"))
        .unwrap()
        .validate()
        .unwrap();
    assert_eq!(result.state("semantic"), "succeeded", "{}", result.json);
    let observations = &result.json["operations"]["semantic"]["observations"];
    assert_eq!(observations["segmentation"], "unavailable");
    assert!(!observations["segmentation_reason"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .is_empty());
    let value = project.semantic_value();
    assert_eq!(value["status"], "unavailable");
    assert!(!value["unavailable_reason"]
        .as_str()
        .unwrap()
        .trim()
        .is_empty());
    assert_eq!(value["observations"], json!([]));
    assert_eq!(value["tracks"], json!([]));
    assert!(
        !project.document("semantic").contains("\"counts\""),
        "no masks"
    );
    assert!(project.open().0.read_receipt("semantic").unwrap().is_some());

    // Availability is re-probed: once the provider runs, real evidence replaces
    // the unavailable record without editing the project.
    let recovered = project.build();
    assert_eq!(recovered.code, 0, "{}", recovered.stderr);
    assert_eq!(recovered.json["executed"], json!(["semantic"]));
    assert_eq!(project.semantic_value()["status"], "available");
}

#[test]
fn sidecar_reported_unavailability_is_recorded_with_its_reason() {
    let project = Project::new("no-gpu", true);
    let reason = "CUDA device not available (test double)";
    project.set_response(&unavailable_response(reason));
    let result = project.build();
    assert!(![1, 2, 64].contains(&result.code), "{}", result.stderr);
    result.assert_geometry_completed();
    assert_eq!(result.state("semantic"), "succeeded");
    let value = project.semantic_value();
    assert_eq!(value["status"], "unavailable");
    assert!(
        value["unavailable_reason"]
            .as_str()
            .unwrap()
            .contains(reason),
        "the sidecar's own reason is preserved: {}",
        value["unavailable_reason"]
    );
    assert_eq!(value["observations"], json!([]));
    assert_eq!(
        value["vocabulary"]["terms"],
        json!(default_room_vocabulary()),
        "vocabulary identity is recorded even when unavailable"
    );
}

#[test]
fn invalid_sidecar_evidence_fails_closed_without_publishing() {
    let project = Project::new("invalid-response", true);
    let mut response = available_response(&["chair", "wall"]);
    response["observations"][0]["bbox"] = json!([100.0, 100.0, 700.0, 130.0]);
    project.set_response(&response);
    let result = project.build();
    assert_eq!(result.state("semantic"), "failed", "{}", result.json);
    assert_eq!(result.code, 4);
    result.assert_geometry_completed();
    assert!(project.record("semantic").is_none());
    assert!(project.open().0.read_receipt("semantic").unwrap().is_none());

    let mut unknown_label = available_response(&["chair", "wall"]);
    unknown_label["observations"][2]["label"] = json!("spaceship");
    project.set_response(&unknown_label);
    // A changed attempt bound is a new identity with a fresh budget.
    project.edit(|document| {
        document["operations"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|operation| operation["id"] == "semantic")
            .unwrap()["max_attempts"] = json!(2);
    });
    let labelled = project.build();
    assert_eq!(labelled.state("semantic"), "failed", "{}", labelled.json);
    assert!(
        project.record("semantic").is_none(),
        "label outside vocabulary"
    );
}

#[test]
fn crashing_sidecar_never_yields_available_evidence() {
    let project = Project::new("crash", true);
    project.write_double("echo 'CUDA out of memory (test double)' >&2\nexit 7");
    let result = project.build();
    result.assert_geometry_completed();
    match project.record("semantic") {
        None => assert_eq!(result.state("semantic"), "failed", "{}", result.json),
        Some(_) => {
            let value = project.semantic_value();
            assert_eq!(value["status"], "unavailable");
            assert_eq!(value["observations"], json!([]));
        }
    }
}

struct CancelWhenSidecarStarts {
    inner: BuiltInExecutor,
    started: PathBuf,
}

impl OperationExecutor for CancelWhenSidecarStarts {
    fn prepare_wave(&self, manifest: &SceneProjectManifest) -> Result<(), SceneProjectError> {
        self.inner.prepare_wave(manifest)
    }

    fn execute(&self, request: &OperationRequest, cancel: &CancellationToken) -> AttemptOutcome {
        if request.operation.kind != OperationKind::SemanticInventory {
            return self.inner.execute(request, cancel);
        }
        let done = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let deadline = Instant::now() + Duration::from_secs(30);
                while !self.started.exists()
                    && !done.load(Ordering::SeqCst)
                    && Instant::now() < deadline
                {
                    std::thread::sleep(Duration::from_millis(20));
                }
                cancel.cancel();
            });
            let outcome = self.inner.execute(request, cancel);
            done.store(true, Ordering::SeqCst);
            outcome
        })
    }
}

#[test]
fn cancellation_stops_the_sidecar_and_publishes_nothing() {
    let project = Project::new("cancel", true);
    // Remove geometry from the graph so only semantics remain pending.
    project.edit(|document| {
        document["operations"]
            .as_array_mut()
            .unwrap()
            .retain(|operation| {
                ["ingest", "semantic"].contains(&operation["id"].as_str().unwrap())
            });
        document["requested_outputs"] = json!(["semantic_inventory"]);
    });
    let started = project.double.join("started");
    let pid = project.double.join("pid");
    project.write_double(&format!(
        "echo $$ > '{}'\ntouch '{}'\nexec sleep 60",
        pid.display(),
        started.display()
    ));
    let (store, mut manifest) = project.open();
    let executor = CancelWhenSidecarStarts {
        inner: BuiltInExecutor::for_project(&project.root, project.manifest())
            .with_semantic_sidecar(project.sidecar()),
        started: started.clone(),
    };
    let cancel = CancellationToken::new();
    let begun = Instant::now();
    let report = store
        .build(&mut manifest, &executor, RunOptions::default(), &cancel)
        .unwrap();
    assert!(started.exists(), "the sidecar was actually started");
    assert!(
        begun.elapsed() < Duration::from_secs(20),
        "cancellation must not wait for the sidecar to finish"
    );
    assert!(cancel.is_canceled());
    assert!(
        matches!(
            report.run.states["semantic"],
            OperationState::Canceled { .. }
        ),
        "{:?}",
        report.run.states["semantic"]
    );
    assert!(store.read_receipt("semantic").unwrap().is_none());
    assert!(project.record("semantic").is_none());
    let pid: u32 = fs::read_to_string(&pid).unwrap().trim().parse().unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while Path::new(&format!("/proc/{pid}")).exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        !Path::new(&format!("/proc/{pid}")).exists(),
        "the canceled sidecar process is terminated and reaped"
    );

    project.write_double("cp \"$D/response.json\" \"$2\"");
    let resumed = project.build();
    assert_eq!(resumed.code, 0, "{}", resumed.stderr);
    assert_eq!(resumed.state("semantic"), "succeeded");
    assert_eq!(project.semantic_value()["status"], "available");
}
