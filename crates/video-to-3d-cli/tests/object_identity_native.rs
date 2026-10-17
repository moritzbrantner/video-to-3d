//! The built-in `object_identity` operation (#61) in the real CLI/store, over
//! semantic evidence from a deterministic test-double sidecar.
#![cfg(unix)]

use serde_json::{json, Value};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use video_to_3d_core::object_identity::{IdentityStatus, MotionClass, ObjectIdentityArtifact};
use video_to_3d_core::scene_artifacts::{write_keyframes_artifact, SampledFrame};
use video_to_3d_core::scene_project::{ContentHash, OperationKind, ProjectPath};
use video_to_3d_core::scene_runner::{
    AttemptOutcome, CancellationToken, OperationExecutor, OperationRequest, ProducedArtifact,
    RunOptions,
};
use video_to_3d_core::scene_store::{ProjectStore, Reproducibility};
use video_to_3d_core::ReconstructionRequest;

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

fn project(response: &Value) -> PathBuf {
    let base = std::env::temp_dir().join(format!(
        "video-to-3d-object-identity-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&base);
    let root = base.join("project");
    fs::create_dir_all(root.join("media")).unwrap();
    let source: &[u8] = b"golden_fixture.rs/lateral";
    fs::write(root.join("media/source.txt"), source).unwrap();
    fs::write(base.join("response.json"), response.to_string()).unwrap();
    let sidecar = base.join("sidecar.sh");
    fs::write(
        &sidecar,
        format!(
            "#!/bin/sh\ncp '{}' \"$2\"\n",
            base.join("response.json").display()
        ),
    )
    .unwrap();
    fs::set_permissions(&sidecar, fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(root.join("project.json"), json!({
        "schema_version": 1, "project_id": "object-identity", "quality_mode": "standard",
        "inputs": [{"id": "clip", "path": "media/source.txt",
                    "content_hash": ContentHash::of_bytes(source).as_str(), "byte_length": source.len()}],
        "provider_policy": {"execution": "local_only", "providers": [{
            "id": "local-grounded-sam2", "location": "local", "capabilities": ["semantic_inventory"], "revision": "r1"}]},
        "operations": [
            {"id": "ingest", "kind": "ingest_video", "inputs": [{"media": "clip"}]},
            {"id": "sparse", "kind": "sparse_reconstruction", "inputs": [{"operation": "ingest"}]},
            {"id": "dense", "kind": "dense_reconstruction", "inputs": [{"operation": "ingest"}, {"operation": "sparse"}]},
            {"id": "mesh", "kind": "surface_mesh", "inputs": [{"operation": "dense"}]},
            {"id": "semantic", "kind": "semantic_inventory", "provider": "local-grounded-sam2",
             "vocabulary": ["toy", "wall"], "inputs": [{"operation": "ingest"}]},
            {"id": "objects", "kind": "object_identity",
             "inputs": [{"operation": "semantic"}, {"operation": "sparse"}, {"operation": "mesh"}]},
            {"id": "masks", "kind": "object_masks",
             "inputs": [{"operation": "semantic"}, {"operation": "objects"}, {"operation": "sparse"}, {"operation": "mesh"}]},
            {"id": "references", "kind": "object_references",
             "inputs": [{"operation": "semantic"}, {"operation": "objects"}, {"operation": "masks"}, {"operation": "sparse"}, {"operation": "mesh"}]}
        ],
        "requested_outputs": ["object_identity", "object_masks", "object_references"]
    }).to_string()).unwrap();
    let (store, mut manifest) = ProjectStore::open(&root.join("project.json")).unwrap();
    store
        .build(
            &mut manifest,
            &ImportFrames(root.clone()),
            RunOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
    std::env::set_var("VIDEO_TO_3D_SEMANTIC_SIDECAR", &sidecar);
    root
}

fn build(root: &std::path::Path) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_video-to-3d"))
        .arg("build")
        .arg(root.join("project.json"))
        .output()
        .unwrap();
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn object_identity_runs_over_semantic_evidence_without_touching_geometry() {
    let observation = |id: &str, frame: usize, label: &str, bbox: [f64; 4]| {
        json!({
            "id": id, "frame_index": frame, "label": label, "bbox": bbox,
            "detection_confidence": 0.7, "mask": null, "mask_confidence": null
        })
    };
    let response = json!({
        "protocol": "video-to-3d-semantic-sidecar/1", "status": "available",
        "models": [{"role": "detector", "id": "double", "version": "1", "url": "https://example.invalid/d"}],
        "runtime": {"name": "double", "version": "1", "device": "cpu", "device_name": null},
        "observations": [
            observation("obs-0", 0, "toy", [200.0, 150.0, 440.0, 330.0]),
            observation("obs-1", 1, "toy", [200.0, 150.0, 440.0, 330.0]),
            observation("obs-2", 2, "wall", [0.0, 0.0, 640.0, 480.0])
        ],
        "tracks": [{"id": "track-0", "label": "toy", "observations": ["obs-0", "obs-1"], "association_confidence": 0.8}],
        "uncertainties": [], "measurements": null
    });
    let root = project(&response);
    let result = build(&root);
    assert_eq!(
        result["operations"]["objects"]["state"], "succeeded",
        "{result}"
    );
    assert_eq!(
        result["operations"]["objects"]["observations"]["objects"],
        "2"
    );
    let (_, manifest) = ProjectStore::open(&root.join("project.json")).unwrap();
    let record = manifest
        .artifacts
        .iter()
        .find(|a| a.produced_by == "objects")
        .unwrap();
    let artifact =
        ObjectIdentityArtifact::from_json(&fs::read_to_string(record.path.resolve(&root)).unwrap())
            .unwrap();
    assert_eq!(artifact.status, IdentityStatus::Available);
    let wall = artifact.objects.iter().find(|o| o.label == "wall").unwrap();
    assert_eq!(wall.motion, MotionClass::StaticStructure);
    let toy = artifact.objects.iter().find(|o| o.label == "toy").unwrap();
    assert_eq!(toy.observations, ["obs-0", "obs-1"]);
    assert_eq!(toy.source_tracks, ["track-0"]);
    assert!(!toy.motion_reason.is_empty());
    // Box-only test evidence: every mask is reported missing, never elevated.
    assert_eq!(
        result["operations"]["masks"]["state"], "succeeded",
        "{result}"
    );
    assert_eq!(
        result["operations"]["masks"]["observations"]["missing_masks"],
        "3"
    );
    // Box-only evidence packages nothing, and says why for the movable toy.
    assert_eq!(
        result["operations"]["references"]["state"], "succeeded",
        "{result}"
    );
    assert_eq!(
        result["operations"]["references"]["observations"]["unpackaged"],
        "1"
    );
    let rebuilt = build(&root);
    assert_eq!(
        rebuilt["operations"]["objects"]["state"], "reused",
        "{rebuilt}"
    );
}
