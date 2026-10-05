use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use video_to_3d_core::scene_project::{ContentHash, ProjectPath};
use video_to_3d_core::scene_runner::{
    AttemptOutcome, CancellationToken, OperationExecutor, OperationRequest, ProducedArtifact,
    RunOptions,
};
use video_to_3d_core::scene_store::{hash_file, ProjectStore, Reproducibility};

struct Project {
    root: PathBuf,
}

impl Project {
    fn new(name: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "video-to-3d-cli-{name}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("media")).unwrap();
        fs::write(root.join("media/clip.webm"), b"clip bytes").unwrap();
        let (hash, length) = hash_file(&root.join("media/clip.webm")).unwrap();
        let document = json!({
            "schema_version": 1,
            "project_id": "cli",
            "quality_mode": "preview",
            "inputs": [
                { "id": "clip", "path": "media/clip.webm", "content_hash": hash.as_str(), "byte_length": length }
            ],
            "provider_policy": { "execution": "local_only" },
            "operations": [
                { "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] },
                { "id": "sparse", "kind": "sparse_reconstruction", "inputs": [{ "operation": "ingest" }] },
                { "id": "assemble", "kind": "scene_assembly", "inputs": [{ "operation": "sparse" }] }
            ],
            "requested_outputs": ["assembled_scene"],
            "exports": [
                { "id": "web", "format": "web_bundle", "path": "exports/web", "includes": ["assembled_scene"] }
            ]
        });
        fs::write(root.join("project.json"), document.to_string()).unwrap();
        Self { root }
    }

    fn manifest(&self) -> PathBuf {
        self.root.join("project.json")
    }

    /// Complete the build through the core API with a test executor that
    /// writes deterministic outputs (the CLI has no native executors yet).
    fn complete(&self) {
        let (store, mut manifest) = ProjectStore::open(&self.manifest()).unwrap();
        let report = store
            .build(
                &mut manifest,
                &Writer(self.root.clone()),
                RunOptions::default(),
                &CancellationToken::new(),
            )
            .unwrap();
        assert!(report.run.is_complete());
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

struct Writer(PathBuf);

const SCENE: &str = r#"{
  "schema_version": 1,
  "frame": { "unit": "arbitrary_monocular", "handedness": "right_handed", "up_axis": "positive_y" },
  "resources": [
    { "id": "chair-mesh", "kind": "mesh", "path": "objects/chair.glb",
      "content_hash": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
      "provenance": ["generative_completion"] }
  ],
  "assets": [
    { "id": "chair", "role": "editable_object",
      "transform": { "translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1] },
      "visual": { "mesh": { "mesh": "chair-mesh" } } }
  ]
}"#;

impl OperationExecutor for Writer {
    fn execute(&self, request: &OperationRequest, _cancel: &CancellationToken) -> AttemptOutcome {
        let id = &request.operation.id;
        let bytes = if id == "assemble" {
            SCENE.as_bytes().to_vec()
        } else {
            request.identity.as_str().as_bytes().to_vec()
        };
        let path = ProjectPath::new(format!("artifacts/{id}.json")).unwrap();
        let file = path.resolve(&self.0);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, &bytes).unwrap();
        AttemptOutcome::Succeeded(ProducedArtifact::new(
            path,
            ContentHash::of_bytes(&bytes),
            Reproducibility::Deterministic,
        ))
    }
}

struct Output {
    code: i32,
    json: Value,
    stderr: String,
}

/// Run the binary from an unrelated working directory to prove commands do
/// not depend on it.
fn cli(args: &[&str]) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_video-to-3d"))
        .args(args)
        .current_dir(std::env::temp_dir())
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    Output {
        code: output.status.code().unwrap(),
        json: serde_json::from_str(&stdout).unwrap_or(Value::Null),
        stderr: String::from_utf8(output.stderr).unwrap(),
    }
}

fn path(project: &Project) -> String {
    project.manifest().to_str().unwrap().to_owned()
}

fn snapshot(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut entries = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let bytes = fs::read(&path).unwrap();
                entries.push((path, bytes));
            }
        }
    }
    entries.sort();
    entries
}

#[test]
fn build_without_native_executors_reports_unsupported() {
    let project = Project::new("unsupported");
    let output = cli(&["build", &path(&project)]);
    assert_eq!(output.code, 3, "{}", output.stderr);
    assert_eq!(output.json["operations"]["ingest"]["state"], "unsupported");
    assert_eq!(output.json["operations"]["sparse"]["state"], "blocked");
    assert!(output.stderr.contains("ingest: unsupported"));
}

#[test]
fn status_is_non_mutating_and_reports_pending_work() {
    let project = Project::new("status");
    let before = snapshot(&project.root);
    let output = cli(&["status", &path(&project)]);
    assert_eq!(output.code, 5);
    assert_eq!(output.json["complete"], false);
    assert_eq!(output.json["operations"]["ingest"]["state"], "pending");
    assert_eq!(snapshot(&project.root), before);
}

#[test]
fn completed_project_status_build_inspect_and_export() {
    let project = Project::new("complete");
    project.complete();

    let before = snapshot(&project.root);
    let status = cli(&["status", &path(&project)]);
    assert_eq!(status.code, 0, "{}", status.stderr);
    assert_eq!(status.json["operations"]["assemble"]["state"], "current");
    assert_eq!(snapshot(&project.root), before);

    let build = cli(&["build", &path(&project), "--jobs", "2"]);
    assert_eq!(build.code, 0, "{}", build.stderr);
    assert_eq!(build.json["executed"], json!([]));
    assert_eq!(build.json["operations"]["sparse"]["state"], "reused");

    let inspect = cli(&["inspect", &path(&project)]);
    assert_eq!(inspect.code, 0, "{}", inspect.stderr);
    let sparse = &inspect.json["operations"]["sparse"];
    assert_eq!(sparse["execution"], "built_in");
    assert_eq!(sparse["receipt"]["reproducibility"], "deterministic");
    assert!(sparse["artifact"]["content_hash"]
        .as_str()
        .unwrap()
        .starts_with("sha256:"));
    let chair = &inspect.json["assembled_scene"]["assets"][0];
    assert_eq!(chair["contains_generative"], true);
    assert_eq!(chair["visual_provenance"], json!(["generative_completion"]));

    let export = cli(&["export", &path(&project), "--target", "web"]);
    assert_eq!(export.code, 3);
    assert!(export.stderr.contains("not available"));
}

#[test]
fn tampered_outputs_show_up_in_status_and_inspect() {
    let project = Project::new("tamper");
    project.complete();
    fs::write(project.root.join("artifacts/sparse.json"), b"tampered").unwrap();
    // Full verification (inspect/status use the cache only for unchanged
    // size and mtime, which this write changes).
    let status = cli(&["status", &path(&project)]);
    assert_eq!(status.code, 5);
    assert_eq!(status.json["operations"]["sparse"]["state"], "invalid");
    assert_eq!(
        status.json["operations"]["sparse"]["reason"],
        "corrupt_output"
    );
    assert_eq!(status.json["operations"]["assemble"]["state"], "stale");
    assert!(status
        .stderr
        .contains("sparse: recorded artifact invalidated"));
}

#[test]
fn exit_codes_distinguish_failure_classes() {
    let project = Project::new("codes");
    let usage = cli(&[]);
    assert_eq!(usage.code, 64);
    assert_eq!(usage.json["exit_code"], 64);
    assert_eq!(cli(&["frobnicate", &path(&project)]).code, 64);
    assert_eq!(
        cli(&["export", &path(&project), "--target", "usd"]).code,
        64
    );
    assert_eq!(cli(&["status", "/nonexistent/project.json"]).code, 1);

    // Incomplete build cannot be exported.
    assert_eq!(cli(&["export", &path(&project), "--target", "web"]).code, 5);

    fs::write(project.manifest(), r#"{"schema_version": 9}"#).unwrap();
    let invalid = cli(&["status", &path(&project)]);
    assert_eq!(invalid.code, 2);
    assert_eq!(invalid.json["command"], "status");
    assert_eq!(invalid.json["exit_code"], 2);
    assert!(invalid.json["error"]
        .as_str()
        .unwrap()
        .contains("schema version 9"));
    assert!(invalid
        .stderr
        .contains("unsupported scene-project schema version 9"));
}

#[test]
fn exhausted_budget_outranks_a_stale_artifact() {
    use video_to_3d_core::scene_project::AttemptUsage;
    let project = Project::new("exhausted");
    project.complete();
    let (store, mut manifest) = ProjectStore::open(&project.manifest()).unwrap();
    // A new declaration (new identity) whose whole budget was spent while
    // the old artifact is still recorded.
    for operation in &mut manifest.operations {
        if operation.id == "assemble" {
            operation.max_attempts = 2;
        }
    }
    let identity = manifest.operation_identities().unwrap()["assemble"].clone();
    manifest.attempt_usage.push(AttemptUsage {
        operation: "assemble".into(),
        operation_identity: identity,
        attempts: 2,
    });
    store.save_manifest(&manifest).unwrap();

    let status = cli(&["status", &path(&project)]);
    assert_eq!(status.json["operations"]["assemble"]["state"], "exhausted");
    assert_eq!(status.code, 4);
}
