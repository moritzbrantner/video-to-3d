//! Actual canonical reference frames -> existing native CLI/store. The only
//! test adapter writes the browser-owned keyframe handoff; it owns no geometry.
use super::*;

#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../../../video-to-3d-core/examples/reference_fixture.rs"]
mod reference;

struct ReferenceFrames(PathBuf);
impl OperationExecutor for ReferenceFrames {
    fn execute(&self, request: &OperationRequest, _: &CancellationToken) -> AttemptOutcome {
        if request.operation.kind != OperationKind::IngestVideo {
            return AttemptOutcome::Unsupported(
                "native stages remain with the canonical executor".into(),
            );
        }
        let reference = reference::sampled_request("overlapping-references");
        let frames: Vec<_> = reference
            .frames
            .iter()
            .enumerate()
            .map(|(index, frame)| SampledFrame {
                frame_index: index,
                timestamp_seconds: 0.8 + index as f64 * 14.4 / 17.0,
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

fn reference_project(name: &str) -> Project {
    let project = Project::new(name, false);
    let source = b"reference_fixture.rs/overlapping-references: seed 0x01287e71f00d5eed, canonical synthetic sampled frames";
    fs::write(project.root.join("media/generated-source.txt"), source).unwrap();
    let mut document: Value =
        serde_json::from_slice(&fs::read(project.manifest()).unwrap()).unwrap();
    document["inputs"][0]["content_hash"] = json!(ContentHash::of_bytes(source).as_str());
    document["inputs"][0]["byte_length"] = json!(source.len());
    fs::write(project.manifest(), document.to_string()).unwrap();
    let (store, mut manifest) = project.open();
    let report = store
        .build(
            &mut manifest,
            &ReferenceFrames(project.root.clone()),
            RunOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
    assert!(report.run.states["ingest"].is_complete());
    assert!(!report.run.is_complete());
    project
}

struct CancelAfterMesh(BuiltInExecutor);
impl OperationExecutor for CancelAfterMesh {
    fn execute(&self, request: &OperationRequest, cancel: &CancellationToken) -> AttemptOutcome {
        let outcome = self.0.execute(request, cancel);
        if request.operation.kind == OperationKind::SurfaceMesh
            && matches!(outcome, AttemptOutcome::Succeeded(_))
        {
            cancel.cancel();
        }
        outcome
    }
}

#[test]
fn real_overlap_derived_mesh_receipt_resumes_without_losing_lineage_or_geometry() {
    let interrupted = reference_project("overlap-lineage-resume-137");
    let (store, mut manifest) = interrupted.open();
    let cancel = CancellationToken::new();
    let executor = CancelAfterMesh(BuiltInExecutor::for_project(
        &interrupted.root,
        interrupted.manifest(),
    ));
    let report = store
        .build(&mut manifest, &executor, RunOptions::default(), &cancel)
        .unwrap();
    assert!(cancel.is_canceled());
    assert!(report.run.states["mesh"].is_complete());
    assert!(!report.run.is_complete());
    let mesh_bytes = interrupted.document("mesh");
    let mesh = SurfaceMeshArtifact::from_json(&mesh_bytes).unwrap();
    assert!(
        !mesh.derived_vertices.is_empty(),
        "actual reference must exercise derived geometry, not just an unchanged receipt"
    );
    let receipt = store.read_receipt("mesh").unwrap().unwrap();
    assert_eq!(
        receipt.output.content_hash,
        ContentHash::of_bytes(mesh_bytes.as_bytes())
    );
    assert_eq!(receipt.output.byte_length, mesh_bytes.len() as u64);
    let resumed = interrupted.build();
    assert_eq!(resumed.code, 0, "{}", resumed.stderr);
    assert_eq!(resumed.json["operations"]["mesh"]["state"], "reused");
    assert_eq!(interrupted.document("mesh"), mesh_bytes);
    let cold = reference_project("overlap-lineage-cold-137");
    cold.assert_complete();
    for id in ["sparse", "dense", "mesh", "assemble"] {
        assert_eq!(
            interrupted.document(id),
            cold.document(id),
            "{id}: resumed differs from cold canonical execution"
        );
    }
    let scene = AssembledScene::from_json(&interrupted.document("assemble")).unwrap();
    for resource in &scene.resources {
        let cold_scene = AssembledScene::from_json(&cold.document("assemble")).unwrap();
        let matching = cold_scene
            .resources
            .iter()
            .find(|other| other.id == resource.id)
            .unwrap();
        assert_eq!(resource.content_hash, matching.content_hash);
        assert_eq!(
            fs::read(resource.path.resolve(&interrupted.root)).unwrap(),
            fs::read(matching.path.resolve(&cold.root)).unwrap()
        );
    }
    let (_, mut current) = interrupted.open();
    assert!(store
        .reconcile(&mut current, VerifyMode::Full)
        .unwrap()
        .invalidated
        .is_empty());
}
