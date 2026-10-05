use super::*;
use crate::scene_runner::{AttemptOutcome, OperationRequest, ProducedArtifact, ResolvedInput};
use serde_json::json;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const CLIP: &[u8] = b"pretend this is a webm clip";

struct TempProject {
    root: PathBuf,
}

impl TempProject {
    fn new(name: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "video-to-3d-store-{name}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("media")).unwrap();
        fs::write(root.join("media/clip.webm"), CLIP).unwrap();
        let (hash, length) = hash_file(&root.join("media/clip.webm")).unwrap();
        let document = json!({
            "schema_version": 1,
            "project_id": "store",
            "quality_mode": "preview",
            "inputs": [
                { "id": "clip", "path": "media/clip.webm", "content_hash": hash.as_str(), "byte_length": length }
            ],
            "provider_policy": { "execution": "local_only" },
            "operations": [
                { "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] },
                { "id": "sparse", "kind": "sparse_reconstruction", "inputs": [{ "operation": "ingest" }] },
                { "id": "dense", "kind": "dense_reconstruction", "inputs": [{ "operation": "sparse" }] },
                { "id": "mesh", "kind": "surface_mesh", "inputs": [{ "operation": "dense" }] },
                { "id": "assemble", "kind": "scene_assembly", "inputs": [{ "operation": "sparse" }, { "operation": "mesh" }] }
            ],
            "requested_outputs": ["assembled_scene"]
        });
        fs::write(root.join("project.json"), document.to_string()).unwrap();
        Self { root }
    }

    fn manifest_path(&self) -> PathBuf {
        self.root.join("project.json")
    }

    fn open(&self) -> (ProjectStore, SceneProjectManifest) {
        ProjectStore::open(&self.manifest_path()).unwrap()
    }

    fn file(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }
}

impl Drop for TempProject {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// Writes deterministic outputs derived from the request identity and input
/// hashes into the project directory.
struct WritingExecutor {
    root: PathBuf,
    executed: Mutex<Vec<String>>,
    cancel_after: Option<(&'static str, CancellationToken)>,
}

impl WritingExecutor {
    fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            executed: Mutex::new(Vec::new()),
            cancel_after: None,
        }
    }

    fn executed(&self) -> BTreeSet<String> {
        self.executed.lock().unwrap().iter().cloned().collect()
    }
}

impl OperationExecutor for WritingExecutor {
    fn execute(&self, request: &OperationRequest, _cancel: &CancellationToken) -> AttemptOutcome {
        self.executed
            .lock()
            .unwrap()
            .push(request.operation.id.clone());
        let mut bytes = request.identity.as_str().as_bytes().to_vec();
        for input in &request.inputs {
            let hash = match input {
                ResolvedInput::Media(media) => &media.content_hash,
                ResolvedInput::Artifact(artifact) => &artifact.content_hash,
            };
            bytes.extend_from_slice(hash.as_str().as_bytes());
        }
        let path = ProjectPath::new(format!("artifacts/{}.bin", request.operation.id)).unwrap();
        let file = path.resolve(&self.root);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, &bytes).unwrap();
        if let Some((operation, token)) = &self.cancel_after {
            if request.operation.id == *operation {
                token.cancel();
            }
        }
        let mut produced = ProducedArtifact::new(
            path,
            ContentHash::of_bytes(&bytes),
            Reproducibility::Deterministic,
        );
        produced
            .observations
            .insert("bytes".into(), bytes.len().to_string());
        AttemptOutcome::Succeeded(produced)
    }
}

fn build(project: &TempProject, executor: &WritingExecutor) -> BuildReport {
    let (store, mut manifest) = project.open();
    store
        .build(
            &mut manifest,
            executor,
            RunOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap()
}

fn all() -> BTreeSet<String> {
    ["ingest", "sparse", "dense", "mesh", "assemble"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[test]
fn cold_build_writes_receipts_and_rebuild_reuses_everything() {
    let project = TempProject::new("cold");
    let executor = WritingExecutor::new(&project.root);
    let report = build(&project, &executor);
    assert!(report.run.is_complete());
    assert_eq!(executor.executed(), all());

    let (store, manifest) = project.open();
    assert_eq!(manifest.artifacts.len(), 5);
    for artifact in &manifest.artifacts {
        let receipt = store.read_receipt(&artifact.produced_by).unwrap().unwrap();
        assert_eq!(receipt.operation_identity, artifact.operation_identity);
        assert_eq!(receipt.output.content_hash, artifact.content_hash);
        assert_eq!(receipt.reproducibility, Reproducibility::Deterministic);
        assert!(receipt.observations.contains_key("bytes"));
        assert!(!receipt.inputs.is_empty());
    }

    let again = WritingExecutor::new(&project.root);
    let report = build(&project, &again);
    assert!(again.executed().is_empty());
    assert_eq!(report.reconcile.verified.len(), 5);
    assert!(report.reconcile.invalidated.is_empty());
}

#[test]
fn tampered_output_invalidates_exactly_the_affected_work() {
    let project = TempProject::new("tamper");
    build(&project, &WritingExecutor::new(&project.root));
    let (_, cold) = project.open();

    fs::write(project.file("artifacts/dense.bin"), b"tampered").unwrap();
    let executor = WritingExecutor::new(&project.root);
    let report = build(&project, &executor);
    assert!(matches!(
        report.reconcile.invalidated.as_slice(),
        [(operation, Invalidation::CorruptOutput { .. })] if operation == "dense"
    ));
    // `dense` is deterministic, so re-producing it restores the original
    // content and `mesh`/`assemble` remain current.
    assert_eq!(executor.executed(), BTreeSet::from(["dense".to_owned()]));
    let (_, resumed) = project.open();
    assert_eq!(
        resumed.to_canonical_json().unwrap(),
        cold.to_canonical_json().unwrap()
    );
}

#[test]
fn changed_upstream_content_rebuilds_descendants() {
    let project = TempProject::new("upstream");
    build(&project, &WritingExecutor::new(&project.root));
    // Simulate a non-deterministic upstream rerun: replace `sparse` with
    // different bytes and a matching record and receipt.
    let (store, mut manifest) = project.open();
    fs::write(project.file("artifacts/sparse.bin"), b"new sparse").unwrap();
    let new_hash = ContentHash::of_bytes(b"new sparse");
    let mut receipt = store.read_receipt("sparse").unwrap().unwrap();
    receipt.output.content_hash = new_hash.clone();
    receipt.output.byte_length = 10;
    store.write_receipt(&receipt).unwrap();
    for artifact in &mut manifest.artifacts {
        if artifact.produced_by == "sparse" {
            artifact.content_hash = new_hash.clone();
        }
    }
    store.save_manifest(&manifest).unwrap();

    let executor = WritingExecutor::new(&project.root);
    let report = build(&project, &executor);
    // Receipts of direct consumers no longer match their inputs.
    let invalidated: BTreeSet<&str> = report
        .reconcile
        .invalidated
        .iter()
        .map(|(operation, _)| operation.as_str())
        .collect();
    assert_eq!(invalidated, BTreeSet::from(["assemble", "dense"]));
    assert_eq!(
        executor.executed(),
        BTreeSet::from(["assemble".into(), "dense".into(), "mesh".into()])
    );
}

#[test]
fn missing_output_or_receipt_and_corrupt_receipts_are_rebuilt() {
    let project = TempProject::new("missing");
    build(&project, &WritingExecutor::new(&project.root));
    fs::remove_file(project.file("artifacts/mesh.bin")).unwrap();
    fs::remove_file(project.file(".video-to-3d/receipts/sparse.json")).unwrap();
    fs::write(project.file(".video-to-3d/receipts/assemble.json"), b"{").unwrap();

    let executor = WritingExecutor::new(&project.root);
    let report = build(&project, &executor);
    let reasons: BTreeMap<&str, &Invalidation> = report
        .reconcile
        .invalidated
        .iter()
        .map(|(operation, reason)| (operation.as_str(), reason))
        .collect();
    assert_eq!(reasons["mesh"], &Invalidation::MissingOutput);
    assert_eq!(reasons["sparse"], &Invalidation::MissingReceipt);
    assert!(matches!(
        reasons["assemble"],
        Invalidation::UnreadableReceipt(_)
    ));
    assert_eq!(
        executor.executed(),
        BTreeSet::from(["assemble".into(), "mesh".into(), "sparse".into()])
    );
    assert!(report.run.is_complete());
    assert!(!report.reconcile.diagnostics().is_empty());
}

#[test]
fn interrupted_build_resumes_to_the_cold_result() {
    let cold_project = TempProject::new("cold-reference");
    build(&cold_project, &WritingExecutor::new(&cold_project.root));
    let (_, cold) = cold_project.open();

    let project = TempProject::new("interrupted");
    let cancel = CancellationToken::new();
    let mut interrupted = WritingExecutor::new(&project.root);
    interrupted.cancel_after = Some(("sparse", cancel.clone()));
    let (store, mut manifest) = project.open();
    let report = store
        .build(&mut manifest, &interrupted, RunOptions::default(), &cancel)
        .unwrap();
    assert!(!report.run.is_complete());
    // The persisted manifest holds the verified prefix.
    let (_, persisted) = project.open();
    let recorded: BTreeSet<&str> = persisted
        .artifacts
        .iter()
        .map(|artifact| artifact.produced_by.as_str())
        .collect();
    assert_eq!(recorded, BTreeSet::from(["ingest", "sparse"]));

    let resume = WritingExecutor::new(&project.root);
    build(&project, &resume);
    assert_eq!(
        resume.executed(),
        BTreeSet::from(["assemble".into(), "dense".into(), "mesh".into()])
    );
    let (_, resumed) = project.open();
    assert_eq!(
        resumed.to_canonical_json().unwrap(),
        cold.to_canonical_json().unwrap()
    );
}

#[test]
fn receipts_without_records_are_orphans_and_get_rebuilt() {
    let project = TempProject::new("orphan");
    build(&project, &WritingExecutor::new(&project.root));
    // Crash between writing receipts and saving the manifest: the manifest
    // lost the `assemble` record while its receipt remains.
    let (store, mut manifest) = project.open();
    manifest
        .artifacts
        .retain(|artifact| artifact.produced_by != "assemble");
    store.save_manifest(&manifest).unwrap();

    let executor = WritingExecutor::new(&project.root);
    let report = build(&project, &executor);
    assert_eq!(report.reconcile.orphan_receipts, ["assemble"]);
    assert_eq!(executor.executed(), BTreeSet::from(["assemble".to_owned()]));
}

#[test]
fn cache_is_acceleration_only() {
    let project = TempProject::new("cache");
    build(&project, &WritingExecutor::new(&project.root));
    let cache = project.file(".video-to-3d/cache/content-hashes.json");
    assert!(cache.exists());

    // A corrupt cache is ignored.
    fs::write(&cache, b"not json").unwrap();
    let executor = WritingExecutor::new(&project.root);
    let report = build(&project, &executor);
    assert!(executor.executed().is_empty());
    assert_eq!(report.reconcile.verified.len(), 5);

    // A forged cache entry cannot hide tampering in full verification.
    let (store, mut manifest) = project.open();
    let original = fs::metadata(project.file("artifacts/mesh.bin")).unwrap();
    let mut bytes = fs::read(project.file("artifacts/mesh.bin")).unwrap();
    bytes[0] ^= 0xff;
    fs::write(project.file("artifacts/mesh.bin"), &bytes).unwrap();
    let file = fs::File::options()
        .write(true)
        .open(project.file("artifacts/mesh.bin"))
        .unwrap();
    file.set_modified(original.modified().unwrap()).unwrap();
    drop(file);

    let cached = store
        .reconcile(&mut manifest.clone(), VerifyMode::Cached)
        .unwrap();
    assert!(
        cached.invalidated.is_empty(),
        "cached status trusts size+mtime"
    );
    let full = store.reconcile(&mut manifest, VerifyMode::Full).unwrap();
    assert!(matches!(
        full.invalidated.as_slice(),
        [(operation, Invalidation::CorruptOutput { .. })] if operation == "mesh"
    ));
}

#[test]
fn cached_reconcile_does_not_write_state() {
    let project = TempProject::new("readonly");
    let (store, mut manifest) = project.open();
    store.reconcile(&mut manifest, VerifyMode::Cached).unwrap();
    assert!(!project.file(".video-to-3d").exists());
}

#[test]
fn changed_media_fails_closed() {
    let project = TempProject::new("media");
    build(&project, &WritingExecutor::new(&project.root));
    fs::write(project.file("media/clip.webm"), b"another clip").unwrap();
    let (store, mut manifest) = project.open();
    let error = store
        .build(
            &mut manifest,
            &WritingExecutor::new(&project.root),
            RunOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("does not match its declared"));
}

#[test]
fn manifest_paths_cannot_enter_the_state_directory() {
    let project = TempProject::new("reserved");
    let (_, manifest) = project.open();
    let mut document: serde_json::Value =
        serde_json::from_str(&manifest.to_canonical_json().unwrap()).unwrap();
    document["inputs"][0]["path"] = json!(".video-to-3d/receipts/x.json");
    let error = SceneProjectManifest::from_json(&document.to_string()).unwrap_err();
    assert!(error.to_string().contains("reserved"));
}

#[test]
fn receipts_with_wrong_provenance_are_rejected() {
    let project = TempProject::new("provenance");
    build(&project, &WritingExecutor::new(&project.root));
    let (store, _) = project.open();
    let mut receipt = store.read_receipt("mesh").unwrap().unwrap();
    receipt.kind = OperationKind::GenerativeCompletion;
    store.write_receipt(&receipt).unwrap();
    let mut receipt = store.read_receipt("sparse").unwrap().unwrap();
    receipt.provider = Some(ReceiptProvider {
        id: "someone-else".into(),
        revision: None,
    });
    store.write_receipt(&receipt).unwrap();
    let mut receipt = store.read_receipt("dense").unwrap().unwrap();
    receipt.inputs[0].content_hash = ContentHash::of_bytes(b"other");
    store.write_receipt(&receipt).unwrap();

    let executor = WritingExecutor::new(&project.root);
    let report = build(&project, &executor);
    let invalidated: BTreeSet<&str> = report
        .reconcile
        .invalidated
        .iter()
        .filter(|(_, reason)| matches!(reason, Invalidation::ReceiptMismatch(_)))
        .map(|(operation, _)| operation.as_str())
        .collect();
    assert_eq!(invalidated, BTreeSet::from(["dense", "mesh", "sparse"]));
    assert!(report.run.is_complete());
}

struct LyingExecutor(WritingExecutor);

impl OperationExecutor for LyingExecutor {
    fn execute(&self, request: &OperationRequest, cancel: &CancellationToken) -> AttemptOutcome {
        match self.0.execute(request, cancel) {
            AttemptOutcome::Succeeded(mut produced) if request.operation.id == "dense" => {
                produced.content_hash = ContentHash::of_bytes(b"claimed");
                AttemptOutcome::Succeeded(produced)
            }
            outcome => outcome,
        }
    }
}

#[test]
fn outputs_are_verified_before_their_receipt_is_written() {
    let project = TempProject::new("lying");
    let (store, mut manifest) = project.open();
    let report = store
        .build(
            &mut manifest,
            &LyingExecutor(WritingExecutor::new(&project.root)),
            RunOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
    assert!(matches!(
        &report.run.states["dense"],
        crate::scene_runner::OperationState::Failed { message, .. }
            if message.contains("output verification failed")
    ));
    assert!(matches!(
        report.run.states["mesh"],
        crate::scene_runner::OperationState::Blocked { .. }
    ));
    assert!(store.read_receipt("dense").unwrap().is_none());
    let (_, persisted) = project.open();
    assert!(!persisted
        .artifacts
        .iter()
        .any(|artifact| artifact.produced_by == "dense"));
}

#[test]
fn reservations_are_persisted_before_dispatch() {
    let project = TempProject::new("reserve");
    struct Checking(PathBuf, WritingExecutor);
    impl OperationExecutor for Checking {
        fn execute(
            &self,
            request: &OperationRequest,
            cancel: &CancellationToken,
        ) -> AttemptOutcome {
            let manifest = SceneProjectManifest::load(&self.0).unwrap();
            let usage = manifest
                .attempt_usage
                .iter()
                .find(|usage| usage.operation == request.operation.id)
                .expect("reservation persisted before dispatch");
            assert_eq!(usage.attempts, request.attempt);
            self.1.execute(request, cancel)
        }
    }
    let (store, mut manifest) = project.open();
    let report = store
        .build(
            &mut manifest,
            &Checking(project.manifest_path(), WritingExecutor::new(&project.root)),
            RunOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
    assert!(report.run.is_complete());
    let (_, persisted) = project.open();
    assert!(
        persisted.attempt_usage.is_empty(),
        "success clears reservations"
    );
}

#[test]
fn receipts_must_name_the_recorded_provider() {
    let project = TempProject::new("bound-provider");
    build(&project, &WritingExecutor::new(&project.root));
    let (store, mut manifest) = project.open();
    for artifact in &mut manifest.artifacts {
        if artifact.produced_by == "mesh" {
            artifact.provider = Some("someone".into());
        }
    }
    store.save_manifest(&manifest).unwrap();
    let report = build(&project, &WritingExecutor::new(&project.root));
    assert!(matches!(
        report.reconcile.invalidated.as_slice(),
        [(operation, Invalidation::ReceiptMismatch(message))]
            if operation == "mesh" && message.contains("recorded with the artifact")
    ));
}
