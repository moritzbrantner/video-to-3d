//! Independent #179 real-input execution contract, baseline 6cf3e4e.
//! Browser capture is external; the test importer never implements reconstruction.
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
use video_to_3d_core::scene_artifacts::{
    write_project_keyframes_artifact, BuiltInExecutor, SampledFrame, SurfaceMeshArtifact,
};
use video_to_3d_core::scene_model::{AssembledScene, AssetRole};
use video_to_3d_core::scene_project::{
    ContentHash, OperationKind, ProjectPath, SceneProjectError, SceneProjectManifest,
};
use video_to_3d_core::scene_runner::{
    AttemptOutcome, CancellationToken, OperationExecutor, OperationRequest, OperationState,
    ProducedArtifact, RunOptions,
};
use video_to_3d_core::scene_store::{
    hash_file, BuildReport, ProjectStore, Reproducibility, VerifyMode,
};
use video_to_3d_core::CameraStageArtifact;

/// The approved dense TUM browser plan (#230).
const DENSE_SAMPLES: usize = 87;

fn expected() -> Value {
    serde_json::from_str::<Value>(include_str!(
        "../../../tools/tests/fixtures/approved-controlled-room-expectations.json"
    ))
    .unwrap()["cases"][0]
        .clone()
}
fn input_root() -> PathBuf {
    std::env::var_os("VIDEO_TO_3D_REAL_ROOM_INPUT")
        .expect("run independent browser capture first")
        .into()
}
fn witness_hash() -> String {
    std::env::var("VIDEO_TO_3D_REAL_ROOM_WITNESS_HASH")
        .expect("pin independently captured transfer identity")
}
fn capture(root: &Path, witness: &str) -> Result<(Value, Vec<Vec<u8>>), String> {
    let bytes = fs::read(root.join("browser-frames.json")).map_err(|e| e.to_string())?;
    if ContentHash::of_bytes(&bytes).as_str() != witness {
        return Err("browser witness identity changed".into());
    }
    let document: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let wanted = expected();
    let identity: Value = serde_json::from_str(include_str!(
        "../../../tools/tests/fixtures/approved-real-room-capture-identity.json"
    ))
    .unwrap();
    if document["schema_version"] != 1
        || document["case_id"] != wanted["id"]
        || document["original"]["content_hash"]
            != format!("sha256:{}", wanted["asset"]["sha256"].as_str().unwrap())
        || document["original"]["byte_length"] != wanted["asset"]["byte_length"]
        || document["approved_plan"] != wanted["preparation"]["plan"]
        || document["carrier"]["content_hash"]
            != format!("sha256:{}", identity["carrier"]["sha256"].as_str().unwrap())
        || document["carrier"]["byte_length"] != identity["carrier"]["byte_length"]
        || document["sampling_receipt_hash"]
            != format!(
                "sha256:{}",
                identity["sampling_receipt"]["sha256"].as_str().unwrap()
            )
        || document["browser_offline_pixel_equivalence"] != false
        || document["browser_version"]
            .as_str()
            .is_none_or(str::is_empty)
    {
        return Err("capture differs from approved real input".into());
    }
    let frames = document["frames"]
        .as_array()
        .ok_or("missing browser frames")?;
    if frames.len() != DENSE_SAMPLES {
        return Err("expected the approved dense browser frame count".into());
    }
    let mut pixels = Vec::new();
    let mut previous = -1.;
    for (index, frame) in frames.iter().enumerate() {
        let path = format!("frames/{index:03}.rgba");
        let requested = frame["requested_time"]
            .as_f64()
            .ok_or("missing requested time")?;
        let presented = frame["presented_time"]
            .as_f64()
            .ok_or("missing decoded time")?;
        let wanted_time = wanted["preparation"]["plan"]["times"][index]
            .as_f64()
            .unwrap();
        if frame["frame_index"] != index
            || frame["path"] != path
            || frame["width"] != 360
            || frame["height"] != 270
            || frame["byte_length"] != 360 * 270 * 4
            || frame["presented_time_decoded"] != true
            || !requested.is_finite()
            || (requested - wanted_time).abs() > 1. / 30.
            || !presented.is_finite()
            || presented <= previous
            || presented >= 86.166667
        {
            return Err(format!("invalid fixed browser sample {index}"));
        }
        previous = presented;
        let bytes = fs::read(root.join(path)).map_err(|e| e.to_string())?;
        if bytes.len() != 360 * 270 * 4
            || !bytes.iter().any(|b| *b != 0)
            || frame["content_hash"] != ContentHash::of_bytes(&bytes).as_str()
        {
            return Err(format!("browser pixels changed: {index}"));
        }
        pixels.push(bytes);
    }
    Ok((document, pixels))
}
struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "real-room-native-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct Import<'a> {
    root: &'a Path,
    document: &'a Value,
    pixels: &'a [Vec<u8>],
}
impl OperationExecutor for Import<'_> {
    fn execute(&self, operation: &OperationRequest, _: &CancellationToken) -> AttemptOutcome {
        if operation.operation.kind != OperationKind::IngestVideo {
            return AttemptOutcome::Unsupported("actual CLI owns reconstruction".into());
        }
        let (_, manifest) = ProjectStore::open(&self.root.join("project.json")).unwrap();
        let frames: Vec<_> = self
            .pixels
            .iter()
            .enumerate()
            .map(|(index, pixels)| SampledFrame {
                frame_index: index,
                timestamp_seconds: self.document["frames"][index]["presented_time"]
                    .as_f64()
                    .unwrap(),
                width: 360,
                height: 270,
                rgba: pixels,
            })
            .collect();
        let path = ProjectPath::new("artifacts/browser/keyframes.json").unwrap();
        let (_, hash) =
            write_project_keyframes_artifact(self.root, &manifest, &path, &frames).unwrap();
        let mut output = ProducedArtifact::new(path, hash, Reproducibility::Deterministic);
        output
            .observations
            .insert("browser_witness".into(), witness_hash());
        output.observations.insert(
            "browser_version".into(),
            self.document["browser_version"].as_str().unwrap().into(),
        );
        output.observations.insert(
            "original".into(),
            self.document["original"]["content_hash"]
                .as_str()
                .unwrap()
                .into(),
        );
        AttemptOutcome::Succeeded(output)
    }
}
fn project(document: &Value, pixels: &[Vec<u8>]) -> Temporary {
    let root = Temporary::new();
    fs::create_dir(root.0.join("media")).unwrap();
    let original = PathBuf::from(
        std::env::var_os("VIDEO_TO_3D_REAL_ROOM_SOURCE")
            .expect("actual approved original required"),
    );
    let (hash, length) = hash_file(&original).unwrap();
    assert_eq!(json!(hash.as_str()), document["original"]["content_hash"]);
    assert_eq!(json!(length), document["original"]["byte_length"]);
    fs::copy(original, root.0.join("media/original.avi")).unwrap();
    fs::write(root.0.join("project.json"), json!({
        "schema_version":1,"project_id":"approved-real-room-native","quality_mode":"preview",
        "inputs":[{"id":"clip","path":"media/original.avi","content_hash":hash.as_str(),"byte_length":length}],
        "provider_policy":{"execution":"local_only"},
        "operations":[
            {"id":"ingest","kind":"ingest_video","inputs":[{"media":"clip"}]},
            {"id":"sparse","kind":"sparse_reconstruction","inputs":[{"operation":"ingest"}]},
            {"id":"dense","kind":"dense_reconstruction","inputs":[{"operation":"ingest"},{"operation":"sparse"}]},
            {"id":"mesh","kind":"surface_mesh","inputs":[{"operation":"dense"}]},
            {"id":"assemble","kind":"scene_assembly","inputs":[{"operation":"sparse"},{"operation":"mesh"}]}
        ],"requested_outputs":["assembled_scene"]
    }).to_string()).unwrap();
    let (store, mut manifest) = ProjectStore::open(&root.0.join("project.json")).unwrap();
    let report = store
        .build(
            &mut manifest,
            &Import {
                root: &root.0,
                document,
                pixels,
            },
            RunOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
    assert!(report.run.states["ingest"].is_complete());
    assert!(!report.run.is_complete());
    root
}
fn cli(root: &Path, command: &str) -> (i32, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_video-to-3d"))
        .args([command, root.join("project.json").to_str().unwrap()])
        .current_dir(std::env::temp_dir())
        .output()
        .unwrap();
    let document: Value =
        serde_json::from_slice(&output.stdout).expect("actual CLI must emit JSON");
    (output.status.code().unwrap(), document)
}
fn artifact_bytes(root: &Path) -> Vec<(String, Vec<u8>)> {
    let (_, manifest) = ProjectStore::open(&root.join("project.json")).unwrap();
    manifest
        .artifacts
        .iter()
        .map(|a| {
            (
                a.produced_by.clone(),
                fs::read(a.path.resolve(root)).unwrap(),
            )
        })
        .collect()
}
struct CancelAfterSparse<E>(E);
impl<E: OperationExecutor> OperationExecutor for CancelAfterSparse<E> {
    fn prepare_wave(&self, manifest: &SceneProjectManifest) -> Result<(), SceneProjectError> {
        self.0.prepare_wave(manifest)
    }
    fn execute(&self, operation: &OperationRequest, cancel: &CancellationToken) -> AttemptOutcome {
        let outcome = self.0.execute(operation, cancel);
        if operation.operation.kind == OperationKind::SparseReconstruction
            && matches!(&outcome, AttemptOutcome::Succeeded(_))
        {
            cancel.cancel();
        }
        outcome
    }
}

fn sparse_boundary(report: &BuildReport, store: &ProjectStore) -> Result<(), String> {
    if !matches!(
        report.run.states.get("sparse"),
        Some(OperationState::Succeeded { .. })
    ) {
        return Err("cancellation/resume requires a successful sparse attempt".into());
    }
    let receipt = store
        .read_receipt("sparse")
        .map_err(|error| error.to_string())?
        .ok_or("successful sparse attempt has no receipt")?;
    let actual = hash_file(&receipt.output.path.resolve(store.root()))
        .ok_or("sparse receipt artifact is unavailable")?;
    if actual != (receipt.output.content_hash, receipt.output.byte_length) {
        return Err("sparse receipt does not match its retained artifact".into());
    }
    Ok(())
}

fn outcomes(result: &Value) -> Value {
    Value::Object(
        result["operations"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, operation)| {
                let state = operation["state"].as_str().unwrap();
                (
                    id.clone(),
                    json!(if ["reused", "succeeded"].contains(&state) {
                        "complete"
                    } else {
                        state
                    }),
                )
            })
            .collect(),
    )
}

fn quality_metrics(artifacts: &[(String, Vec<u8>)], exit_code: i32) -> Value {
    let find = |id: &str| {
        artifacts
            .iter()
            .find(|(operation, _)| operation == id)
            .map(|(_, bytes)| std::str::from_utf8(bytes).unwrap())
    };
    let mut editable_assets = None;
    let (registered_cameras, camera_source) = if let Some(document) = find("assemble") {
        let scene = AssembledScene::from_json(document).unwrap();
        scene.validate().unwrap();
        editable_assets = Some(
            scene
                .assets
                .iter()
                .filter(|asset| asset.role == AssetRole::EditableObject)
                .count(),
        );
        (Some(scene.cameras.len()), Some("assemble"))
    } else if let Some(document) = find("mesh") {
        let mesh = SurfaceMeshArtifact::from_json(document).unwrap();
        (Some(mesh.cameras.len()), Some("mesh"))
    } else if let Some(document) = find("sparse") {
        CameraStageArtifact::from_json(document).unwrap();
        let stage: Value = serde_json::from_str(document).unwrap();
        (
            Some(stage["sparse"]["final_cameras"].as_array().unwrap().len()),
            Some("sparse"),
        )
    } else {
        (None, None)
    };
    let minimum = expected()["preparation"]["inventory"]["minimum_recognizable_editable_objects"]
        .as_u64()
        .unwrap();
    json!({
        "registered_cameras": registered_cameras,
        "registered_cameras_source": camera_source,
        "native_editable_assets": editable_assets,
        "native_editable_assets_source": editable_assets.map(|_| "assemble"),
        "required_minimum_recognizable_editable_objects": minimum,
        "status": if exit_code != 0 || editable_assets.is_none_or(|count| (count as u64) < minimum) {"nonpassing"} else {"independent_recognition_and_blender_unverified"},
        "limitation": "null metrics are unavailable; native roles do not establish recognizability or independent Blender editability"
    })
}

// Controlled failure of one stage tests acceptance classification, never product quality.
struct FailStage {
    delegate: BuiltInExecutor,
    kind: OperationKind,
}
impl OperationExecutor for FailStage {
    fn prepare_wave(&self, manifest: &SceneProjectManifest) -> Result<(), SceneProjectError> {
        self.delegate.prepare_wave(manifest)
    }
    fn execute(&self, operation: &OperationRequest, cancel: &CancellationToken) -> AttemptOutcome {
        if operation.operation.kind == self.kind {
            AttemptOutcome::Failed("controlled acceptance failure".into())
        } else {
            self.delegate.execute(operation, cancel)
        }
    }
}

fn reject_false_cancellation_and_unavailable_quality(
    document: &Value,
    pixels: &[Vec<u8>],
) -> Vec<Value> {
    let mut controls = Vec::new();
    for kind in [
        OperationKind::SparseReconstruction,
        OperationKind::DenseReconstruction,
        OperationKind::SceneAssembly,
    ] {
        let project = project(document, pixels);
        let (store, mut manifest) = ProjectStore::open(&project.0.join("project.json")).unwrap();
        let cancel = CancellationToken::new();
        let executor = FailStage {
            delegate: BuiltInExecutor::for_project(&project.0, project.0.join("project.json")),
            kind,
        };
        let report = if kind == OperationKind::SparseReconstruction {
            store
                .build(
                    &mut manifest,
                    &CancelAfterSparse(executor),
                    RunOptions::default(),
                    &cancel,
                )
                .unwrap()
        } else {
            store
                .build(&mut manifest, &executor, RunOptions::default(), &cancel)
                .unwrap()
        };
        assert!(
            !cancel.is_canceled(),
            "a failed sparse stage must not manufacture cancellation"
        );
        assert!(!report.run.is_complete());
        let (code, result) = cli(&project.0, "build");
        assert_ne!(code, 0);
        assert_eq!(result["complete"], false);
        let metrics = quality_metrics(&artifact_bytes(&project.0), code);
        assert_eq!(metrics["status"], "nonpassing");
        assert!(metrics["native_editable_assets"].is_null());
        if kind == OperationKind::SparseReconstruction {
            assert!(sparse_boundary(&report, &store).is_err());
            assert!(store.read_receipt("sparse").unwrap().is_none());
            assert!(metrics["registered_cameras"].is_null());
        } else {
            sparse_boundary(&report, &store).unwrap();
            assert_eq!(
                metrics["registered_cameras_source"],
                if kind == OperationKind::DenseReconstruction {
                    "sparse"
                } else {
                    "mesh"
                }
            );
            assert!(metrics["registered_cameras"].as_u64().is_some());
        }
        controls.push(json!({
            "injected_failure_stage": format!("{kind:?}"),
            "exit_code": code, "complete": result["complete"],
            "canceled": cancel.is_canceled(),
            "sparse_success_and_receipt_verified": sparse_boundary(&report, &store).is_ok(),
            "quality": metrics
        }));
    }
    controls
}

#[test]
#[ignore = "explicit #179 acceptance: independently captured browser input and original required"]
fn approved_room_browser_pixels_reach_actual_cli_and_verified_receipt_reuse() {
    let (document, pixels) = capture(&input_root(), &witness_hash()).unwrap();
    let cold = project(&document, &pixels);
    let (code, first) = cli(&cold.0, "build");
    assert!(
        [0, 3, 4, 5].contains(&code),
        "unexpected CLI error: {first}"
    );
    assert_eq!(first["complete"], code == 0);
    assert_eq!(first["operations"]["ingest"]["state"], "reused");
    assert_eq!(
        first["operations"]["sparse"]["state"], "succeeded",
        "post-success cancellation acceptance requires actual cold sparse success: {first}"
    );
    assert!(
        first["executed"]
            .as_array()
            .unwrap()
            .contains(&json!("sparse")),
        "must execute actual sparse stage: {first}"
    );
    if code != 0 {
        assert!(
            first["operations"].as_object().unwrap().values().any(|s| [
                "failed",
                "retryable_failure",
                "unsupported"
            ]
            .iter()
            .any(|name| s["state"] == *name)
                && s["message"].as_str().is_some_and(|m| !m.is_empty())),
            "earliest actual limitation must be useful: {first}"
        );
    }
    let (store, mut manifest) = ProjectStore::open(&cold.0.join("project.json")).unwrap();
    assert!(store
        .reconcile(&mut manifest, VerifyMode::Full)
        .unwrap()
        .invalidated
        .is_empty());
    for artifact in &manifest.artifacts {
        let receipt = store.read_receipt(&artifact.produced_by).unwrap().unwrap();
        assert_eq!(
            hash_file(&artifact.path.resolve(&cold.0)).unwrap(),
            (receipt.output.content_hash, receipt.output.byte_length)
        );
    }
    assert!(store.read_receipt("sparse").unwrap().is_some());
    let before = artifact_bytes(&cold.0);
    let (replay_code, replay) = cli(&cold.0, "build");
    assert_eq!(replay_code, code);
    assert_eq!(replay["complete"], first["complete"]);
    assert_eq!(outcomes(&replay), outcomes(&first));
    assert_eq!(replay["executed"], json!([]));
    for (id, _) in &before {
        assert_eq!(replay["operations"][id]["state"], "reused");
    }
    assert_eq!(artifact_bytes(&cold.0), before);
    let (_, inspect) = cli(&cold.0, "inspect");
    let stages: Vec<_> = before
        .iter()
        .filter(|(id, _)| id != "ingest")
        .map(|(id, bytes)| (id, serde_json::from_slice::<Value>(bytes).unwrap()))
        .collect();
    let quality = quality_metrics(&before, code);
    let classification_controls =
        reject_false_cancellation_and_unavailable_quality(&document, &pixels);
    let interrupted = project(&document, &pixels);
    let (interrupted_store, mut interrupted_manifest) =
        ProjectStore::open(&interrupted.0.join("project.json")).unwrap();
    let cancel = CancellationToken::new();
    let report = interrupted_store
        .build(
            &mut interrupted_manifest,
            &CancelAfterSparse(BuiltInExecutor::for_project(
                &interrupted.0,
                interrupted.0.join("project.json"),
            )),
            RunOptions::default(),
            &cancel,
        )
        .unwrap();
    sparse_boundary(&report, &interrupted_store).unwrap();
    let sparse_before_resume = artifact_bytes(&interrupted.0);
    assert!(
        cancel.is_canceled(),
        "actual sparse boundary must be reached"
    );
    assert!(!report.run.is_complete());
    for id in ["dense", "mesh", "assemble"] {
        assert!(interrupted_store.read_receipt(id).unwrap().is_none());
    }
    let (resumed_code, resumed) = cli(&interrupted.0, "build");
    assert_eq!(resumed_code, code);
    assert_eq!(resumed["complete"], first["complete"]);
    assert_eq!(outcomes(&resumed), outcomes(&first));
    assert_eq!(resumed["operations"]["sparse"]["state"], "reused");
    assert!(
        resumed["executed"]
            .as_array()
            .unwrap()
            .contains(&json!("dense")),
        "resumption must actually continue downstream work: {resumed}"
    );
    let resumed_artifacts = artifact_bytes(&interrupted.0);
    for (id, bytes) in &sparse_before_resume {
        assert_eq!(
            resumed_artifacts
                .iter()
                .find(|(operation, _)| operation == id)
                .unwrap()
                .1,
            *bytes
        );
    }
    assert_eq!(
        resumed_artifacts, before,
        "resume must preserve actual cold artifacts"
    );
    if let Some(path) = std::env::var_os("VIDEO_TO_3D_REAL_ROOM_REPORT") {
        fs::write(path, serde_json::to_vec_pretty(&json!({"input":document,"cold_exit_code":code,"cold":first,"replay":replay,"inspect":inspect,"stages":stages,"resumed_exit_code":resumed_code,"resumed":resumed,
            "cancellation_resume":{"sparse_success_and_receipt_verified":true,"downstream_outcome":if code == 0 {"complete"} else {"nonpassing"}},"quality":quality,"acceptance_failure_controls":classification_controls,"three_video_acceptance":"unrun","blender_acceptance":"unrun"})).unwrap()).unwrap();
    }
}

#[test]
#[ignore = "explicit #179 acceptance: independently captured browser input required"]
fn real_browser_transfer_rejects_pixel_corruption_and_self_consistent_manifest_changes() {
    let original = input_root();
    capture(&original, &witness_hash()).unwrap();
    let copy = Temporary::new();
    fs::create_dir(copy.0.join("frames")).unwrap();
    fs::copy(
        original.join("browser-frames.json"),
        copy.0.join("browser-frames.json"),
    )
    .unwrap();
    for index in 0..DENSE_SAMPLES {
        let path = format!("frames/{index:03}.rgba");
        fs::copy(original.join(&path), copy.0.join(path)).unwrap();
    }
    let path = copy.0.join("frames/000.rgba");
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] ^= 1;
    fs::write(&path, &bytes).unwrap();
    assert!(capture(&copy.0, &witness_hash())
        .unwrap_err()
        .contains("pixels changed"));
    let mut changed: Value =
        serde_json::from_slice(&fs::read(copy.0.join("browser-frames.json")).unwrap()).unwrap();
    changed["frames"][0]["content_hash"] = json!(ContentHash::of_bytes(&bytes).as_str());
    fs::write(copy.0.join("browser-frames.json"), changed.to_string()).unwrap();
    assert!(capture(&copy.0, &witness_hash())
        .unwrap_err()
        .contains("witness identity changed"));
}

#[test]
#[ignore = "explicit #179 acceptance: independently captured browser input and original required"]
fn actual_cli_rejects_corrupted_imported_browser_pixels_before_reconstruction() {
    let (document, pixels) = capture(&input_root(), &witness_hash()).unwrap();
    let project = project(&document, &pixels);
    let (store, manifest) = ProjectStore::open(&project.0.join("project.json")).unwrap();
    let ingest = manifest
        .artifacts
        .iter()
        .find(|a| a.produced_by == "ingest")
        .unwrap();
    let index: Value =
        serde_json::from_slice(&fs::read(ingest.path.resolve(&project.0)).unwrap()).unwrap();
    let path = ProjectPath::new(index["frames"][0]["pixels"].as_str().unwrap())
        .unwrap()
        .resolve(&project.0);
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] ^= 1;
    fs::write(path, bytes).unwrap();
    let (code, result) = cli(&project.0, "build");
    assert_eq!(code, 3, "{result}");
    assert_eq!(result["complete"], false);
    assert_eq!(result["operations"]["ingest"]["state"], "unsupported");
    for id in ["sparse", "dense", "mesh", "assemble"] {
        assert_eq!(result["operations"][id]["state"], "blocked");
        assert!(store.read_receipt(id).unwrap().is_none());
    }
    assert!(result["reconcile"]["invalidated"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| entry["operation"] == "ingest" && entry["reason"] == "corrupt_sidecar"));
}
