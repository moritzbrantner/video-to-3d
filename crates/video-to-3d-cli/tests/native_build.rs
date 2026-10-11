//! Independent pre-implementation acceptance of #150, baseline 3553210.
//!
//! This is a local geometric smoke, not three-real-video/object/Blender acceptance.
//! The fixture handoff stands in for browser sampling, and writes only keyframes;
//! all reconstruction/assembly operations must run through the actual CLI/core.

use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use video_to_3d_core::scene_artifacts::{
    write_keyframes_artifact, BuiltInExecutor, SampledFrame, SurfaceMeshArtifact,
};
use video_to_3d_core::scene_model::{AssembledScene, AssetRole, ResourceKind, SceneProvenance};
use video_to_3d_core::scene_project::{ArtifactKind, ContentHash, OperationKind, ProjectPath};
use video_to_3d_core::scene_runner::{
    AttemptOutcome, CancellationToken, OperationExecutor, OperationRequest, OperationState,
    ProducedArtifact, RunOptions,
};
use video_to_3d_core::scene_store::{hash_file, ProjectStore, Reproducibility, VerifyMode};
use video_to_3d_core::{
    reconstruct, CameraStageArtifact, DenseStageArtifact, ReconstructionRequest,
};

// Reuse the actual deterministic golden renderer, without copying its geometry
// or inventing object ground truth. The example's private command/scoring helpers
// are intentionally unused by this integration harness.
#[allow(dead_code)]
mod golden {
    pub fn request() -> ReconstructionRequest {
        let points = scene();
        ReconstructionRequest {
            frames: (0..FRAME_COUNT)
                .map(|frame| render_frame(frame, &points, Scenario::Lateral))
                .collect(),
            // Native sampling uses the existing browser/default options: the
            // manifest has no invented per-operation reconstruction settings.
            options: ReconstructionOptions::default(),
        }
    }

    include!("../../video-to-3d-core/examples/golden_fixture.rs");
}

fn request() -> &'static ReconstructionRequest {
    static REQUEST: OnceLock<ReconstructionRequest> = OnceLock::new();
    REQUEST.get_or_init(|| {
        let Some(root) = std::env::var_os("VIDEO_TO_3D_ROOM_FIXTURE") else {
            return golden::request();
        };
        let root = PathBuf::from(root);
        let cameras: Value =
            serde_json::from_slice(&fs::read(root.join("truth/cameras.json")).unwrap()).unwrap();
        assert_eq!(cameras["case"], "controlled-room");
        let frames = cameras["frames"]
            .as_array()
            .unwrap()
            .iter()
            .map(|camera| {
                let image = camera["image"].as_str().unwrap();
                let bytes = fs::read(root.join(image)).unwrap();
                let pinned = include_str!("controlled-room-inputs.sha256")
                    .lines()
                    .find_map(|line| {
                        let (hash, path) = line.split_once("  ")?;
                        (path == image).then_some(hash)
                    })
                    .expect("canonical frame identity is pinned");
                assert_eq!(
                    ContentHash::of_bytes(&bytes).as_str(),
                    format!("sha256:{pinned}")
                );
                let header = b"P6\n360 203\n255\n";
                assert!(
                    bytes.starts_with(header),
                    "canonical #75 exported PPM required"
                );
                let pixels = &bytes[header.len()..];
                assert_eq!(pixels.len(), 360 * 203 * 3);
                video_to_3d_core::FrameInput {
                    width: 360,
                    height: 203,
                    rgba: pixels
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(frames.len(), 18);
        ReconstructionRequest {
            frames,
            options: Default::default(),
        }
    })
}

struct Project {
    root: PathBuf,
}

impl Project {
    fn new(name: &str, import: bool) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "video-to-3d-native-{name}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir_all(root.join("media")).unwrap();
        // Honest generated-input provenance; this is not a fake encoded video
        // and the native implementation must never decode it.
        let source: &[u8] = if std::env::var_os("VIDEO_TO_3D_ROOM_FIXTURE").is_some() {
            b"synthetic_room_fixture.rs/controlled-room: seed 0x00755ce9e0ac1e00, canonical generated sampled frames"
        } else {
            b"golden_fixture.rs/lateral: deterministic generated sampled frames"
        };
        fs::write(root.join("media/generated-source.txt"), source).unwrap();
        let document = json!({
            "schema_version": 1,
            "project_id": "native-geometric-smoke",
            "quality_mode": "preview",
            "inputs": [{
                "id": "clip", "path": "media/generated-source.txt",
                "content_hash": ContentHash::of_bytes(source).as_str(),
                "byte_length": source.len()
            }],
            "provider_policy": { "execution": "local_only" },
            "operations": [
                { "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] },
                { "id": "sparse", "kind": "sparse_reconstruction", "inputs": [{ "operation": "ingest" }] },
                { "id": "dense", "kind": "dense_reconstruction", "inputs": [{ "operation": "ingest" }, { "operation": "sparse" }] },
                { "id": "mesh", "kind": "surface_mesh", "inputs": [{ "operation": "dense" }] },
                { "id": "assemble", "kind": "scene_assembly", "inputs": [{ "operation": "sparse" }, { "operation": "mesh" }] }
            ],
            "requested_outputs": ["assembled_scene"]
        });
        fs::write(root.join("project.json"), document.to_string()).unwrap();
        let project = Self { root };
        if import {
            project.import_frames();
        }
        project
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
        assert!(store.read_receipt("ingest").unwrap().is_some());
        assert!(!report.run.is_complete());
    }

    fn manifest(&self) -> PathBuf {
        self.root.join("project.json")
    }
    fn open(
        &self,
    ) -> (
        ProjectStore,
        video_to_3d_core::scene_project::SceneProjectManifest,
    ) {
        ProjectStore::open(&self.manifest()).unwrap()
    }
    fn output(&self, id: &str) -> PathBuf {
        let (_, manifest) = self.open();
        manifest
            .artifacts
            .iter()
            .find(|artifact| artifact.produced_by == id)
            .unwrap()
            .path
            .resolve(&self.root)
    }
    fn document(&self, id: &str) -> String {
        fs::read_to_string(self.output(id)).unwrap()
    }
    fn build(&self) -> Output {
        cli(self, "build")
    }
    fn assert_complete(&self) {
        let output = self.build();
        assert_eq!(output.code, 0, "{}", output.stderr);
        assert_eq!(output.json["complete"], true);
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// Test-only external browser handoff. It has no reconstruction implementation.
struct ImportFrames(PathBuf);
impl OperationExecutor for ImportFrames {
    fn execute(&self, operation: &OperationRequest, _: &CancellationToken) -> AttemptOutcome {
        if operation.operation.kind != OperationKind::IngestVideo {
            return AttemptOutcome::Unsupported("native operations left to the real CLI".into());
        }
        let frames: Vec<_> = request()
            .frames
            .iter()
            .enumerate()
            .map(|(index, frame)| SampledFrame {
                frame_index: index,
                timestamp_seconds: if std::env::var_os("VIDEO_TO_3D_ROOM_FIXTURE").is_some() {
                    // Canonical sampling_plan: 18 inclusive samples, 0.8..15.2s.
                    0.8 + index as f64 * (14.4 / 17.0)
                } else {
                    index as f64 * 0.5
                },
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

struct Output {
    code: i32,
    json: Value,
    stderr: String,
}
fn cli(project: &Project, command: &str) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_video-to-3d"))
        .arg(command)
        .arg(project.manifest())
        .current_dir(std::env::temp_dir())
        .output()
        .unwrap();
    Output {
        code: output.status.code().unwrap(),
        json: serde_json::from_slice(&output.stdout).expect("one JSON stdout document"),
        stderr: String::from_utf8(output.stderr).unwrap(),
    }
}

#[test]
fn golden_handoff_has_real_supported_geometry_without_native_decoding() {
    let result = reconstruct(request()).unwrap();
    assert!(result.calibrated_pair.is_some());
    assert!(!result.dense_points.is_empty());
    assert!(!result.mesh_triangles.is_empty());
}

#[test]
fn native_build_computes_real_stages_and_environment_with_verified_receipts() {
    let project = Project::new("cold", true);
    let result = project.build();
    assert_eq!(result.code, 0, "{}", result.stderr);
    assert_eq!(result.json["complete"], true);
    assert_eq!(result.json["operations"]["ingest"]["state"], "reused");
    for id in ["sparse", "dense", "mesh", "assemble"] {
        assert_eq!(result.json["operations"][id]["state"], "succeeded", "{id}");
    }
    let (store, manifest) = project.open();
    for artifact in &manifest.artifacts {
        let receipt = store.read_receipt(&artifact.produced_by).unwrap().unwrap();
        assert_eq!(receipt.schema_version, 1);
        assert_eq!(receipt.output.content_hash, artifact.content_hash);
        assert_eq!(receipt.output.path, artifact.path);
        assert_eq!(receipt.operation_identity, artifact.operation_identity);
        assert_eq!(receipt.reproducibility, Reproducibility::Deterministic);
        let (hash, length) = hash_file(&artifact.path.resolve(&project.root)).unwrap();
        assert_eq!(
            (hash, length),
            (receipt.output.content_hash, receipt.output.byte_length)
        );
    }
    let cameras = CameraStageArtifact::from_json(&project.document("sparse")).unwrap();
    let dense = DenseStageArtifact::from_json(&project.document("dense")).unwrap();
    assert_eq!(dense.cameras_hash, cameras.content_hash());
    assert_eq!(dense.request_hash, cameras.request_hash);
    let mesh = SurfaceMeshArtifact::from_json(&project.document("mesh")).unwrap();
    let reference = reconstruct(request()).unwrap();
    let evidence = video_to_3d_core::ReconstructionEvidenceView::from_classic(&reference).unwrap();
    let expected =
        SurfaceMeshArtifact::from_evidence(&evidence, &reference.dense_grid_sites).unwrap();
    assert_eq!(
        mesh.to_json(),
        expected.to_json(),
        "native output retains exact classic evidence"
    );
    assert!(!mesh.triangles.is_empty());
    assert!(mesh
        .regions
        .iter()
        .all(|region| region.origin == video_to_3d_core::EvidenceOrigin::GeometricMultiView));

    let scene = AssembledScene::from_json(&project.document("assemble")).unwrap();
    scene.validate().unwrap();
    assert!(!scene.cameras.is_empty());
    assert!(scene
        .assets
        .iter()
        .any(|asset| asset.role == AssetRole::Environment && asset.visual.is_some()));
    assert!(
        scene
            .assets
            .iter()
            .all(|asset| asset.role != AssetRole::EditableObject),
        "smoke does not fabricate decomposition"
    );
    let resource = scene
        .resources
        .iter()
        .find(|resource| resource.kind == ResourceKind::Mesh)
        .unwrap();
    assert_eq!(
        resource.provenance,
        vec![SceneProvenance::GeometricMultiView]
    );
    let bytes = fs::read(resource.path.resolve(&project.root)).unwrap();
    assert_eq!(ContentHash::of_bytes(&bytes), resource.content_hash);
    assert_eq!(
        &bytes[..4],
        b"glTF",
        "scene visual is a consumable mesh resource"
    );
    let rebuilt = project.build();
    assert_eq!(rebuilt.code, 0, "{}", rebuilt.stderr);
    assert_eq!(rebuilt.json["executed"], json!([]));
    for id in ["ingest", "sparse", "dense", "mesh", "assemble"] {
        assert_eq!(rebuilt.json["operations"][id]["state"], "reused");
    }
    assert_eq!(cli(&project, "status").code, 0);
}

#[test]
fn corrupt_dense_output_rebuilds_only_dense_and_descendants() {
    let project = Project::new("dense-corruption", true);
    project.assert_complete();
    let sparse = project.document("sparse");
    fs::write(project.output("dense"), b"corrupt dense artifact").unwrap();
    let result = project.build();
    assert_eq!(result.code, 0, "{}", result.stderr);
    assert_eq!(
        result.json["executed"],
        json!(["assemble", "dense", "mesh"])
    );
    assert_eq!(result.json["operations"]["ingest"]["state"], "reused");
    assert_eq!(result.json["operations"]["sparse"]["state"], "reused");
    assert_eq!(project.document("sparse"), sparse);
}

#[test]
fn corrupt_scene_resource_invalidates_assembly_instead_of_reporting_current() {
    let project = Project::new("resource-corruption", true);
    project.assert_complete();
    let scene = AssembledScene::from_json(&project.document("assemble")).unwrap();
    let resource = scene
        .resources
        .iter()
        .find(|resource| resource.kind == ResourceKind::Mesh)
        .unwrap();
    fs::write(resource.path.resolve(&project.root), b"corrupt scene mesh").unwrap();
    let status = cli(&project, "status");
    assert_ne!(status.code, 0);
    let rebuilt = project.build();
    assert_eq!(rebuilt.code, 0, "{}", rebuilt.stderr);
    assert_eq!(rebuilt.json["executed"], json!(["assemble"]));
}

struct CancelAfterSparse(BuiltInExecutor);
impl OperationExecutor for CancelAfterSparse {
    fn execute(&self, operation: &OperationRequest, cancel: &CancellationToken) -> AttemptOutcome {
        let outcome = self.0.execute(operation, cancel);
        if operation.operation.kind == OperationKind::SparseReconstruction
            && matches!(outcome, AttemptOutcome::Succeeded(_))
        {
            cancel.cancel();
        }
        outcome
    }
}

#[test]
fn cancellation_keeps_sparse_receipt_and_resume_matches_a_cold_build() {
    let interrupted = Project::new("cancel", true);
    let (store, mut manifest) = interrupted.open();
    let cancel = CancellationToken::new();
    let executor = CancelAfterSparse(BuiltInExecutor::for_project(
        &interrupted.root,
        interrupted.manifest(),
    ));
    let report = store
        .build(&mut manifest, &executor, RunOptions::default(), &cancel)
        .unwrap();
    assert!(!report.run.is_complete());
    assert!(cancel.is_canceled());
    assert!(report.run.states["sparse"].is_complete());
    assert!(store.read_receipt("sparse").unwrap().is_some());
    for id in ["dense", "mesh", "assemble"] {
        assert!(matches!(
            report.run.states[id],
            OperationState::Canceled { .. } | OperationState::Blocked { .. }
        ));
        assert!(store.read_receipt(id).unwrap().is_none());
    }
    let resumed = interrupted.build();
    assert_eq!(resumed.code, 0, "{}", resumed.stderr);
    assert_eq!(resumed.json["operations"]["sparse"]["state"], "reused");
    let cold = Project::new("cold-cancel-oracle", true);
    cold.assert_complete();
    for id in ["sparse", "dense", "mesh", "assemble"] {
        assert_eq!(
            interrupted.document(id),
            cold.document(id),
            "{id}: resumed output differs"
        );
    }
    let (_, mut current) = interrupted.open();
    assert!(store
        .reconcile(&mut current, VerifyMode::Full)
        .unwrap()
        .invalidated
        .is_empty());
}

#[test]
fn missing_keyframe_handoff_fails_closed_at_ingest() {
    let project = Project::new("missing", false);
    let output = project.build();
    assert_eq!(output.code, 3);
    assert_eq!(output.json["complete"], false);
    assert_eq!(output.json["operations"]["ingest"]["state"], "unsupported");
    for id in ["sparse", "dense", "mesh", "assemble"] {
        assert_eq!(output.json["operations"][id]["state"], "blocked");
    }
    assert!(project.open().1.artifacts.is_empty());
}

#[test]
fn unavailable_local_provider_remains_nonpassing_without_fabricated_scene() {
    let project = Project::new("provider", false);
    let mut document: Value =
        serde_json::from_str(&fs::read_to_string(project.manifest()).unwrap()).unwrap();
    document["quality_mode"] = json!("standard");
    document["provider_policy"]["providers"] = json!([
        {"id":"absent-depth", "location":"local", "capabilities":["learned_reconstruction"]}
    ]);
    document["operations"].as_array_mut().unwrap().insert(
        2,
        json!({
            "id":"learned", "kind":"learned_reconstruction", "provider":"absent-depth",
            "inputs":[{"operation":"ingest"}, {"operation":"sparse"}]
        }),
    );
    document["operations"][3]["inputs"]
        .as_array_mut()
        .unwrap()
        .push(json!({"operation":"learned"}));
    fs::write(project.manifest(), document.to_string()).unwrap();
    project.import_frames();
    let output = project.build();
    assert_eq!(output.code, 3, "{}", output.stderr);
    assert_eq!(output.json["complete"], false);
    assert_eq!(output.json["operations"]["sparse"]["state"], "succeeded");
    assert_eq!(output.json["operations"]["learned"]["state"], "unsupported");
    assert_eq!(output.json["operations"]["assemble"]["state"], "blocked");
    assert!(!project
        .open()
        .1
        .artifacts
        .iter()
        .any(|artifact| artifact.kind == ArtifactKind::AssembledScene));
}
