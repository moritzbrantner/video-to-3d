use super::*;
use crate::scene_model::{AssembledScene, CameraAuthority, SceneCamera, SceneFrame, Transform};
use crate::scene_model::{Handedness, SceneUnit, UpAxis};
use crate::scene_project::SceneProjectManifest;
use crate::scene_runner::OperationState;
use crate::scene_runner::RunOptions;
use crate::scene_store::ProjectStore;
use crate::{
    classic_reference_patch_regions, DenseReferencePatchStats, ReconstructionProviderDescriptor,
};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 12;

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "video-to-3d-artifacts-{name}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        Self(root)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn path(value: &str) -> ProjectPath {
    ProjectPath::new(value).unwrap()
}

fn point(index: usize) -> Point3 {
    Point3 {
        x: index as f32 * 0.25,
        y: 0.5 - index as f32 * 0.125,
        z: 2.0 + index as f32 * 0.01,
        confidence: 0.9,
        r: (index * 30) as u8,
        g: 200,
        b: 10,
    }
}

fn camera(frame_index: usize, x: f32) -> EvidenceCamera {
    EvidenceCamera {
        frame_index,
        authority: EvidenceCameraAuthority::CalibratedSeed,
        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation: [x, 0.0, 0.0],
        confidence: Some(0.9),
        median_reprojection_error_pixels: Some(0.4),
    }
}

fn image(seed: u8) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            rgba.extend_from_slice(&[
                (x as u8 * 13).wrapping_add(seed),
                (y as u8 * 17).wrapping_add(seed),
                seed,
                255,
            ]);
        }
    }
    rgba
}

fn triangle(a: usize, b: usize, c: usize) -> MeshTriangle {
    MeshTriangle {
        a,
        b,
        c,
        confidence: 0.8,
    }
}

/// Two reference patches: frame 0 owns points 0..4 (point 3 is revalidated
/// completion), frame 1 owns points 4..8; triangle 3 mixes both references.
struct Fixture {
    points: Vec<Point3>,
    sites: Vec<DenseGridSite>,
    triangles: Vec<MeshTriangle>,
    cameras: Vec<EvidenceCamera>,
    regions: Vec<SurfaceEvidenceRegion>,
    images: Vec<(usize, Vec<u8>)>,
}

impl Fixture {
    fn new() -> Self {
        let patches = vec![
            DenseReferencePatchStats {
                reference_frame: 0,
                source_frames: vec![1, 2],
                primary_start: 0,
                primary_points: 3,
                completion_start: 3,
                completed_points: 1,
                ..DenseReferencePatchStats::default()
            },
            DenseReferencePatchStats {
                reference_frame: 1,
                source_frames: vec![0],
                primary_start: 4,
                primary_points: 4,
                completion_start: 8,
                completed_points: 0,
                ..DenseReferencePatchStats::default()
            },
        ];
        let site = |x, y| DenseGridSite { x, y };
        Self {
            points: (0..8).map(point).collect(),
            sites: vec![
                site(2, 2),
                site(6, 2),
                site(2, 6),
                site(6, 6),
                site(3, 3),
                site(9, 3),
                site(3, 9),
                site(9, 9),
            ],
            triangles: vec![
                triangle(0, 1, 2),
                triangle(1, 3, 2),
                triangle(4, 5, 6),
                triangle(2, 3, 4),
                triangle(5, 7, 6),
            ],
            cameras: vec![camera(0, 0.0), camera(1, 0.3), camera(2, 0.6)],
            regions: classic_reference_patch_regions(&patches),
            images: vec![(0, image(5)), (1, image(90)), (2, image(170))],
        }
    }

    fn mesh(&self) -> SurfaceMeshArtifact {
        let evidence = ReconstructionEvidenceView::new(
            ReconstructionProviderDescriptor::classic(),
            EvidenceScale::ArbitraryMonocular,
            self.cameras.clone(),
            self.regions.clone(),
            &self.points,
            &self.triangles,
        )
        .unwrap();
        SurfaceMeshArtifact::from_evidence(&evidence, &self.sites).unwrap()
    }

    /// Write `keyframes.json` and `mesh.json` under `inputs/`.
    fn write(&self, root: &Path) -> (ProjectPath, ProjectPath) {
        let frames: Vec<SampledFrame<'_>> = self
            .images
            .iter()
            .map(|(frame_index, rgba)| SampledFrame {
                frame_index: *frame_index,
                timestamp_seconds: *frame_index as f64 / 30.0,
                width: WIDTH,
                height: HEIGHT,
                rgba,
            })
            .collect();
        let keyframes = path("inputs/keyframes.json");
        write_keyframes_artifact(root, &keyframes, &frames).unwrap();
        let mesh = path("inputs/mesh.json");
        fs::write(mesh.resolve(root), self.mesh().to_json()).unwrap();
        (keyframes, mesh)
    }
}

fn bake(root: &Path, inputs: &(ProjectPath, ProjectPath)) -> TextureBakeOutcome {
    bake_texture_artifact(
        root,
        &path("artifacts/bake/surface-textures.json"),
        &inputs.0,
        &inputs.1,
        &CancellationToken::new(),
    )
    .unwrap()
}

fn texture_file(root: &Path, outcome: &TextureBakeOutcome, frame: usize) -> PathBuf {
    outcome
        .artifact
        .materials
        .iter()
        .find(|material| material.reference_frame == frame)
        .unwrap()
        .texture
        .path
        .resolve(root)
}

#[test]
fn keyframes_round_trip_and_verify_their_pixels() {
    let dir = TempDir::new("keyframes");
    let fixture = Fixture::new();
    let (keyframes, _) = fixture.write(&dir.0);
    let document = fs::read_to_string(keyframes.resolve(&dir.0)).unwrap();
    let artifact = KeyframesArtifact::from_json(&document).unwrap();
    assert_eq!(artifact.to_json(), document);
    for frame in &artifact.frames {
        let digest = &frame.content_hash.as_str()["sha256:".len()..];
        assert_eq!(
            frame.pixels.as_str(),
            format!("inputs/frames/{digest}.rgba")
        );
    }
    assert_eq!(artifact.load_pixels(&dir.0, 1).unwrap(), image(90));

    // A rejected rewrite leaves the existing artifact intact.
    let bad = [SampledFrame {
        frame_index: 0,
        timestamp_seconds: 0.0,
        width: WIDTH,
        height: HEIGHT,
        rgba: &[1, 2, 3],
    }];
    assert!(write_keyframes_artifact(&dir.0, &keyframes, &bad).is_err());
    let duplicate_pixels = image(1);
    let duplicate = [0, 0].map(|frame_index| SampledFrame {
        frame_index,
        timestamp_seconds: 0.0,
        width: WIDTH,
        height: HEIGHT,
        rgba: &duplicate_pixels,
    });
    assert!(write_keyframes_artifact(&dir.0, &keyframes, &duplicate).is_err());
    assert_eq!(
        fs::read_to_string(keyframes.resolve(&dir.0)).unwrap(),
        document
    );
    for frame in 0..3 {
        artifact.load_pixels(&dir.0, frame).unwrap();
    }

    fs::write(artifact.frames[1].pixels.resolve(&dir.0), image(91)).unwrap();
    let error = artifact.load_pixels(&dir.0, 1).unwrap_err();
    assert!(
        error.contains("does not match its recorded content hash"),
        "{error}"
    );

    let future = document.replace("\"schema_version\": 1", "\"schema_version\": 2");
    let error = KeyframesArtifact::from_json(&future).unwrap_err();
    assert!(error.contains("unsupported schema version 2"), "{error}");
}

#[test]
fn surface_mesh_round_trips_the_evidence_view() {
    let fixture = Fixture::new();
    let mesh = fixture.mesh();
    let document = mesh.to_json();
    let parsed = SurfaceMeshArtifact::from_json(&document).unwrap();
    assert_eq!(parsed.to_json(), document);
    let evidence = parsed.evidence().unwrap();
    assert_eq!(
        evidence.summary(),
        fixture.mesh().evidence().unwrap().summary()
    );
    assert_eq!(evidence.points.len(), 8);

    let mut short = parsed.clone();
    short.grid_sites.pop();
    let error = SurfaceMeshArtifact::from_json(&short.to_json()).unwrap_err();
    assert!(error.contains("7 grid sites for 8 points"), "{error}");

    let mut unknown: serde_json::Value = serde_json::from_str(&document).unwrap();
    unknown["extra"] = json!(true);
    assert!(SurfaceMeshArtifact::from_json(&unknown.to_string()).is_err());
}

#[test]
fn bake_writes_one_texture_per_reference_and_matches_the_shared_bake() {
    let dir = TempDir::new("bake");
    let fixture = Fixture::new();
    let inputs = fixture.write(&dir.0);
    let outcome = bake(&dir.0, &inputs);

    assert_eq!(outcome.written, vec![0, 1]);
    assert!(outcome.reused.is_empty());
    let artifact = &outcome.artifact;
    assert_eq!(artifact.materials.len(), 2);
    assert_eq!(artifact.materials[0].triangles, vec![0, 1]);
    assert_eq!(artifact.materials[1].triangles, vec![2, 4]);
    assert_eq!(artifact.fallback_triangles, vec![3]);
    assert_eq!(artifact.fallback_reasons.mixed_reference, 1);
    for material in &artifact.materials {
        let digest = &material.appearance_key.as_str()["sha256:".len()..];
        assert_eq!(
            material.texture.path.as_str(),
            format!("artifacts/bake/textures/{digest}.png")
        );
        let png = fs::read(material.texture.path.resolve(&dir.0)).unwrap();
        assert!(png.starts_with(b"\x89PNG"));
        assert_eq!(ContentHash::of_bytes(&png), material.texture.content_hash);
    }
    artifact.verify_textures(&dir.0).unwrap();

    // The recorded manifest is exactly what was written and parses back.
    let document = fs::read_to_string(dir.0.join("artifacts/bake/surface-textures.json")).unwrap();
    assert_eq!(
        ContentHash::of_bytes(document.as_bytes()),
        outcome.content_hash
    );
    assert_eq!(
        &SurfaceTexturesArtifact::from_json(&document).unwrap(),
        artifact
    );
    assert!(outcome
        .diagnostic
        .contains("4 of 5 accepted triangles textured"));
}

#[test]
fn reframing_one_reference_rewrites_only_its_texture() {
    let dir = TempDir::new("reframe");
    let mut fixture = Fixture::new();
    let inputs = fixture.write(&dir.0);
    let first = bake(&dir.0, &inputs);
    let kept = texture_file(&dir.0, &first, 0);
    let replaced = texture_file(&dir.0, &first, 1);
    let kept_modified = fs::metadata(&kept).unwrap().modified().unwrap();

    // Unchanged inputs reuse every texture.
    let again = bake(&dir.0, &inputs);
    assert_eq!(again.written, Vec::<usize>::new());
    assert_eq!(again.reused, vec![0, 1]);
    assert_eq!(again.content_hash, first.content_hash);

    // Reproject reference 1: new camera pose and shifted grid sites.
    fixture.cameras[1].translation = [0.35, 0.0, 0.0];
    for site in &mut fixture.sites[4..8] {
        site.x += 1;
    }
    let inputs = fixture.write(&dir.0);
    let second = bake(&dir.0, &inputs);
    assert_eq!(second.written, vec![1]);
    assert_eq!(second.reused, vec![0]);
    assert_eq!(second.invalidation.reused, vec![0]);
    assert_eq!(second.invalidation.invalidated, vec![1]);
    assert_eq!(
        fs::metadata(&kept).unwrap().modified().unwrap(),
        kept_modified
    );
    assert!(
        !replaced.exists(),
        "the stale texture of reference 1 is removed"
    );
    assert!(texture_file(&dir.0, &second, 1).exists());
    second.artifact.verify_textures(&dir.0).unwrap();
}

#[test]
fn edited_texture_sidecars_are_detected_and_rewritten() {
    let dir = TempDir::new("tamper");
    let fixture = Fixture::new();
    let inputs = fixture.write(&dir.0);
    let first = bake(&dir.0, &inputs);
    fs::write(texture_file(&dir.0, &first, 0), b"not a png").unwrap();
    let error = first.artifact.verify_textures(&dir.0).unwrap_err();
    assert!(
        error.contains("does not match its recorded content hash"),
        "{error}"
    );

    let repaired = bake(&dir.0, &inputs);
    assert_eq!(repaired.written, vec![0]);
    assert_eq!(repaired.reused, vec![1]);
    // The appearance diagnostic reports the repair, not a reuse.
    assert_eq!(repaired.invalidation.reused, vec![1]);
    assert_eq!(repaired.invalidation.invalidated, vec![0]);
    repaired.artifact.verify_textures(&dir.0).unwrap();
}

fn scene_with(artifact: &SurfaceTexturesArtifact) -> AssembledScene {
    let mut scene = AssembledScene::new(SceneFrame {
        unit: SceneUnit::ArbitraryMonocular,
        handedness: Handedness::RightHanded,
        up_axis: UpAxis::PositiveY,
    });
    scene.cameras = (0..3)
        .map(|frame| SceneCamera {
            id: format!("camera-{frame}"),
            source_frame: frame,
            authority: CameraAuthority::CalibratedSeed,
            transform: Transform::IDENTITY,
            intrinsics: None,
        })
        .collect();
    let (resources, materials) = artifact.scene_entries("bake").unwrap();
    scene.resources = resources;
    scene.materials = materials;
    scene
}

#[test]
fn baked_textures_map_to_camera_backed_scene_resources() {
    let dir = TempDir::new("scene");
    let fixture = Fixture::new();
    let inputs = fixture.write(&dir.0);
    let artifact = bake(&dir.0, &inputs).artifact;
    let scene = scene_with(&artifact);
    scene.validate().unwrap();

    let texture = &scene.resources[0];
    assert_eq!(texture.id, "bake.texture-f0");
    assert_eq!(texture.kind, ResourceKind::Texture);
    assert_eq!(texture.source_frames, vec![0, 1, 2]);
    assert_eq!(
        texture.provenance,
        vec![
            SceneProvenance::GeometricMultiView,
            SceneProvenance::RevalidatedCompletion
        ]
    );
    assert_eq!(texture.path, artifact.materials[0].texture.path);
    assert_eq!(scene.materials[0].id, "bake.material-f0");
    assert_eq!(
        scene.materials[0].base_color_texture.as_deref(),
        Some("bake.texture-f0")
    );
    // The canonical scene document round-trips.
    let document = scene.to_canonical_json().unwrap();
    assert_eq!(
        AssembledScene::from_json(&document)
            .unwrap()
            .resources
            .len(),
        2
    );

    // Relabeling a baked texture as camera-free content is rejected.
    let mut relabeled = scene.clone();
    relabeled.resources[0].provenance = vec![SceneProvenance::GenerativeCompletion];
    assert!(relabeled.validate().is_err());
    let mut unlinked = scene.clone();
    unlinked.resources[0].source_frames.clear();
    assert!(unlinked.validate().is_err());
    // Cited frames need accepted cameras.
    let mut missing_camera = scene;
    missing_camera
        .cameras
        .retain(|camera| camera.source_frame != 2);
    assert!(missing_camera.validate().is_err());
}

#[test]
fn builtin_executor_bakes_texture_operations_in_a_project_build() {
    let dir = TempDir::new("build");
    let fixture = Fixture::new();
    fs::create_dir_all(dir.0.join("media")).unwrap();
    fs::write(dir.0.join("media/clip.webm"), b"clip").unwrap();
    let document = json!({
        "schema_version": 1,
        "project_id": "bake",
        "quality_mode": "standard",
        "inputs": [{
            "id": "clip", "path": "media/clip.webm",
            "content_hash": ContentHash::of_bytes(b"clip").as_str(), "byte_length": 4
        }],
        "provider_policy": { "execution": "local_only" },
        "operations": [
            { "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] },
            { "id": "sparse", "kind": "sparse_reconstruction", "inputs": [{ "operation": "ingest" }] },
            { "id": "dense", "kind": "dense_reconstruction", "inputs": [{ "operation": "sparse" }] },
            { "id": "mesh", "kind": "surface_mesh", "inputs": [{ "operation": "dense" }] },
            { "id": "bake", "kind": "texture_bake", "inputs": [{ "operation": "mesh" }, { "operation": "ingest" }] }
        ],
        "requested_outputs": ["surface_textures"]
    });
    fs::write(dir.0.join("project.json"), document.to_string()).unwrap();

    /// Stands in for the browser-produced upstream artifacts.
    struct Upstream<'a> {
        fixture: &'a Fixture,
        root: PathBuf,
        builtin: BuiltInExecutor,
    }
    impl OperationExecutor for Upstream<'_> {
        fn execute(
            &self,
            request: &OperationRequest,
            cancel: &CancellationToken,
        ) -> AttemptOutcome {
            let (keyframes, mesh) = self.fixture.write(&self.root);
            let output = match request.operation.id.as_str() {
                "ingest" => keyframes,
                "mesh" => mesh,
                "sparse" | "dense" => {
                    let output = path(&format!("artifacts/{}.bin", request.operation.id));
                    fs::create_dir_all(output.resolve(&self.root).parent().unwrap()).unwrap();
                    fs::write(output.resolve(&self.root), request.identity.as_str()).unwrap();
                    output
                }
                _ => return self.builtin.execute(request, cancel),
            };
            let bytes = fs::read(output.resolve(&self.root)).unwrap();
            AttemptOutcome::Succeeded(ProducedArtifact::new(
                output,
                ContentHash::of_bytes(&bytes),
                Reproducibility::Deterministic,
            ))
        }
    }

    let (store, mut manifest) = ProjectStore::open(&dir.0.join("project.json")).unwrap();
    let executor = Upstream {
        fixture: &fixture,
        root: dir.0.clone(),
        builtin: BuiltInExecutor::new(&dir.0, &manifest),
    };
    let report = store
        .build(
            &mut manifest,
            &executor,
            RunOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
    assert!(report.run.is_complete(), "{:?}", report.run.diagnostics());
    let artifact = manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.produced_by == "bake")
        .unwrap();
    assert_eq!(artifact.kind, ArtifactKind::SurfaceTextures);
    assert_eq!(
        artifact.path.as_str(),
        "artifacts/bake/surface-textures.json"
    );
    let receipt = store.read_receipt("bake").unwrap().unwrap();
    assert_eq!(receipt.observations["textures_written"], "0,1");
    assert!(receipt.observations["diagnostic"].contains("4 of 5 accepted triangles textured"));
    assert!(receipt.observations["appearance"].contains("added 0, 1"));

    // A rebuild reuses the recorded artifact without executing anything.
    let reloaded = SceneProjectManifest::load(&dir.0.join("project.json")).unwrap();
    let mut manifest = reloaded;
    let report = store
        .build(
            &mut manifest,
            &executor,
            RunOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
    assert_eq!(report.run.states["bake"], OperationState::Reused);

    // A deleted texture sidecar invalidates the recorded bake, which reruns.
    let document = fs::read_to_string(dir.0.join("artifacts/bake/surface-textures.json")).unwrap();
    let textures = SurfaceTexturesArtifact::from_json(&document).unwrap();
    fs::remove_file(textures.materials[0].texture.path.resolve(&dir.0)).unwrap();
    let mut manifest = SceneProjectManifest::load(&dir.0.join("project.json")).unwrap();
    let report = store
        .build(
            &mut manifest,
            &executor,
            RunOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap();
    assert!(matches!(
        report.reconcile.invalidated.as_slice(),
        [(operation, crate::scene_store::Invalidation::CorruptSidecar(_))] if operation == "bake"
    ));
    assert!(matches!(
        report.run.states["bake"],
        OperationState::Succeeded { .. }
    ));
    textures.verify_textures(&dir.0).unwrap();
}

#[test]
fn builtin_executor_reports_other_operations_as_unsupported() {
    let dir = TempDir::new("unsupported");
    let manifest_document = json!({
        "schema_version": 1,
        "project_id": "unsupported",
        "quality_mode": "preview",
        "inputs": [{
            "id": "clip", "path": "media/clip.webm",
            "content_hash": ContentHash::of_bytes(b"clip").as_str(), "byte_length": 4
        }],
        "provider_policy": { "execution": "local_only" },
        "operations": [
            { "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] }
        ],
        "requested_outputs": ["keyframes"]
    });
    let mut manifest = SceneProjectManifest::from_json(&manifest_document.to_string()).unwrap();
    let executor = BuiltInExecutor::new(&dir.0, &manifest);
    let report = crate::scene_runner::run(
        &mut manifest,
        &executor,
        RunOptions::default(),
        &CancellationToken::new(),
    )
    .unwrap();
    assert_eq!(
        report.states["ingest"].diagnostic("ingest"),
        "ingest: unsupported by every reachable provider after 1 attempt(s): ingest_video has no native executor in this build"
    );
}

#[test]
fn texture_artifacts_must_partition_the_triangles() {
    let dir = TempDir::new("partition");
    let fixture = Fixture::new();
    let inputs = fixture.write(&dir.0);
    let artifact = bake(&dir.0, &inputs).artifact;
    artifact.validate().unwrap();

    let mut duplicated = artifact.clone();
    duplicated.fallback_triangles = vec![0];
    assert!(duplicated
        .validate()
        .unwrap_err()
        .contains("more than once"));
    let mut out_of_range = artifact.clone();
    out_of_range.fallback_triangles = vec![9];
    assert!(out_of_range
        .validate()
        .unwrap_err()
        .contains("triangle 9 of 5"));
    let mut missing = artifact.clone();
    missing.fallback_triangles.clear();
    missing.fallback_reasons = FallbackReasons::default();
    assert!(missing
        .validate()
        .unwrap_err()
        .contains("assigns 4 triangle(s) for 5"));
    let mut huge = artifact.clone();
    huge.triangle_count = usize::MAX;
    assert!(huge.validate().is_err());
    assert!(artifact.scene_entries(&"x".repeat(60)).is_err());
}

#[test]
fn texture_artifacts_reject_incompatible_versions_provenance_and_fallbacks() {
    let dir = TempDir::new("contract");
    let inputs = Fixture::new().write(&dir.0);
    let artifact = bake(&dir.0, &inputs).artifact;

    let mut old = artifact.clone();
    old.bake_schema_version += 1;
    assert!(old
        .validate()
        .unwrap_err()
        .contains("unsupported bake schema"));
    assert!(SurfaceTexturesArtifact::from_json(&old.to_json()).is_err());

    // A material texturing no triangle would export observed provenance
    // without supporting geometry.
    let mut empty = artifact.clone();
    let moved = std::mem::take(&mut empty.materials[0].triangles);
    empty.materials[0].corner_uvs.clear();
    empty.fallback_triangles.extend(moved);
    empty.fallback_triangles.sort_unstable();
    assert!(empty
        .validate()
        .unwrap_err()
        .contains("textures no triangle"));
    assert!(SurfaceTexturesArtifact::from_json(&empty.to_json()).is_err());

    // Two materials may not name one texture path (ASCII case folded).
    let mut shared = artifact.clone();
    shared.materials[1].texture.path = ProjectPath::new(
        shared.materials[0]
            .texture
            .path
            .as_str()
            .to_ascii_uppercase(),
    )
    .unwrap();
    assert!(shared.validate().unwrap_err().contains("twice"));

    let mut learned = artifact.clone();
    learned.materials[0].provenance = vec![EvidenceOrigin::LearnedMultiView];
    assert!(learned
        .validate()
        .unwrap_err()
        .contains("non-observed provenance"));
    assert!(learned.scene_entries("bake").is_err());

    let mut repeated = artifact.clone();
    repeated.materials[0]
        .provenance
        .push(EvidenceOrigin::GeometricMultiView);
    assert!(repeated
        .validate()
        .unwrap_err()
        .contains("repeats provenance"));
    assert!(SurfaceTexturesArtifact::from_json(&repeated.to_json()).is_err());

    let mut no_reasons = artifact.clone();
    no_reasons.fallback_reasons = FallbackReasons::default();
    assert!(no_reasons
        .validate()
        .unwrap_err()
        .contains("fallback reason(s)"));
    let mut overflowing = artifact.clone();
    overflowing.fallback_reasons.mixed_reference = usize::MAX;
    overflowing.fallback_reasons.unobserved_reference = 1;
    assert!(overflowing
        .validate()
        .unwrap_err()
        .contains("count overflows"));

    assert!(artifact
        .scene_entries("Bad")
        .unwrap_err()
        .contains("scene identifier"));
    assert!(artifact
        .scene_entries("-")
        .unwrap_err()
        .contains("scene identifier"));
    scene_with(&artifact).validate().unwrap();
}

#[test]
fn texture_artifacts_reject_incorrect_camera_source_associations() {
    let dir = TempDir::new("sources");
    let fixture = Fixture::new();
    let inputs = fixture.write(&dir.0);
    let original = bake(&dir.0, &inputs).artifact;

    // The reference frame is itself a source, even if other frames supported
    // the observed region. Validation also protects direct scene entry calls.
    let mut missing = original.clone();
    missing.materials[0].source_frames = vec![1, 2];
    let error = missing.validate().unwrap_err();
    assert!(error.contains("must cite its reference frame"), "{error}");
    assert!(SurfaceTexturesArtifact::from_json(&missing.to_json()).is_err());
    assert!(missing.scene_entries("bake").is_err());

    let mut duplicate = original.clone();
    duplicate.materials[0].source_frames = vec![0, 1, 1];
    let error = duplicate.validate().unwrap_err();
    assert!(error.contains("source frame 1 more than once"), "{error}");
    assert!(SurfaceTexturesArtifact::from_json(&duplicate.to_json()).is_err());
    assert!(duplicate.scene_entries("bake").is_err());

    original.validate().unwrap();
    scene_with(&original).validate().unwrap();
}

#[test]
fn cancellation_before_baking_preserves_existing_inputs_and_writes_no_output() {
    let dir = TempDir::new("cancel");
    let fixture = Fixture::new();
    let inputs = fixture.write(&dir.0);
    let original = fs::read(inputs.0.resolve(&dir.0)).unwrap();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let output = path("artifacts/bake/surface-textures.json");
    let error = bake_texture_artifact(&dir.0, &output, &inputs.0, &inputs.1, &cancel).unwrap_err();
    assert!(error.contains("canceled"), "{error}");
    assert!(!output.resolve(&dir.0).exists());
    assert!(!dir.0.join("artifacts/bake/textures").exists());
    assert_eq!(fs::read(inputs.0.resolve(&dir.0)).unwrap(), original);
}

#[test]
fn builtin_executor_rejects_ambiguous_inputs_and_foreign_paths() {
    let dir = TempDir::new("guards");
    let fixture = Fixture::new();
    let (keyframes, mesh) = fixture.write(&dir.0);
    let hash = |path: &ProjectPath| ContentHash::of_bytes(&fs::read(path.resolve(&dir.0)).unwrap());
    let record =
        |id: &str, kind: ArtifactKind, path: &ProjectPath| crate::scene_project::ArtifactRecord {
            id: format!("{id}.output"),
            kind,
            produced_by: id.into(),
            operation_identity: ContentHash::of_bytes(id.as_bytes()),
            path: path.clone(),
            content_hash: hash(path),
            provider: None,
        };
    let manifest_document = json!({
        "schema_version": 1,
        "project_id": "guards",
        "quality_mode": "standard",
        "inputs": [{
            "id": "clip", "path": "artifacts/bake/clip.webm",
            "content_hash": ContentHash::of_bytes(b"clip").as_str(), "byte_length": 4
        }],
        "provider_policy": { "execution": "local_only" },
        "operations": [
            { "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] }
        ],
        "requested_outputs": ["keyframes"]
    });
    let manifest = SceneProjectManifest::from_json(&manifest_document.to_string()).unwrap();
    let request = |inputs: Vec<ResolvedInput>| OperationRequest {
        operation: crate::scene_project::OperationDeclaration {
            id: "bake".into(),
            kind: OperationKind::TextureBake,
            inputs: Vec::new(),
            provider: None,
            max_attempts: 1,
        },
        identity: ContentHash::of_bytes(b"bake"),
        provider: None,
        attempt: 1,
        inputs,
    };
    let keyframes_input =
        ResolvedInput::Artifact(record("ingest", ArtifactKind::Keyframes, &keyframes));
    let mesh_input = ResolvedInput::Artifact(record("mesh", ArtifactKind::SurfaceMesh, &mesh));

    let executor = BuiltInExecutor::new(&dir.0, &manifest);
    let outcome = executor.execute(
        &request(vec![
            keyframes_input.clone(),
            mesh_input.clone(),
            mesh_input.clone(),
        ]),
        &CancellationToken::new(),
    );
    assert!(
        matches!(&outcome, AttemptOutcome::Failed(message) if message.contains("exactly one keyframes and one surface_mesh")),
        "{outcome:?}"
    );

    // The media input lives inside the bake's output directory: refuse to write.
    let outcome = executor.execute(
        &request(vec![keyframes_input.clone(), mesh_input.clone()]),
        &CancellationToken::new(),
    );
    assert!(
        matches!(&outcome, AttemptOutcome::Failed(message) if message.contains("would overwrite the declared path `artifacts/bake/clip.webm`")),
        "{outcome:?}"
    );
    assert!(!dir.0.join("artifacts/bake").exists());

    // Manifest paths are ASCII case-folded for uniqueness. Collision
    // preflight must apply the same rule even when tested on Linux, before
    // a case-insensitive filesystem could overwrite media or exports.
    for declared in [
        "Artifacts/Bake/clip.webm",
        "ARTIFACTS/BAKE",
        "ARTIFACTS/BAKE/SURFACE-TEXTURES.JSON",
        "ARTIFACTS",
    ] {
        let mut document = manifest_document.clone();
        document["inputs"][0]["path"] = json!(declared);
        let manifest = SceneProjectManifest::from_json(&document.to_string()).unwrap();
        let executor = BuiltInExecutor::new(&dir.0, &manifest);
        let outcome = executor.execute(
            &request(vec![keyframes_input.clone(), mesh_input.clone()]),
            &CancellationToken::new(),
        );
        assert!(
            matches!(&outcome, AttemptOutcome::Failed(message) if message.contains(declared)),
            "{declared}: {outcome:?}"
        );
        assert!(!dir.0.join("artifacts/bake").exists());
    }
}

#[test]
fn builtin_executor_reserves_sidecars_of_recorded_artifacts() {
    let dir = TempDir::new("sidecar-reservation");
    let fixture = Fixture::new();
    let (keyframes, mesh) = fixture.write(&dir.0);
    // Move one keyframe sidecar into the bake's output directory, as an
    // unconstrained sidecar path of a recorded upstream artifact may be.
    let keyframes_file = keyframes.resolve(&dir.0);
    let mut document: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&keyframes_file).unwrap()).unwrap();
    let original = path(document["frames"][0]["pixels"].as_str().unwrap());
    let moved = path("artifacts/bake/textures/frame-0.rgba");
    fs::create_dir_all(moved.resolve(&dir.0).parent().unwrap()).unwrap();
    fs::rename(original.resolve(&dir.0), moved.resolve(&dir.0)).unwrap();
    document["frames"][0]["pixels"] = json!(moved.as_str());
    fs::write(&keyframes_file, document.to_string()).unwrap();
    let sidecar_bytes = fs::read(moved.resolve(&dir.0)).unwrap();

    let hash = |path: &ProjectPath| ContentHash::of_bytes(&fs::read(path.resolve(&dir.0)).unwrap());
    let record =
        |id: &str, kind: ArtifactKind, path: &ProjectPath| crate::scene_project::ArtifactRecord {
            id: format!("{id}.output"),
            kind,
            produced_by: id.into(),
            operation_identity: ContentHash::of_bytes(id.as_bytes()),
            path: path.clone(),
            content_hash: hash(path),
            provider: None,
        };
    let mut manifest = SceneProjectManifest::from_json(
        &json!({
            "schema_version": 1,
            "project_id": "sidecar-reservation",
            "quality_mode": "standard",
            "inputs": [{
                "id": "clip", "path": "media/clip.webm",
                "content_hash": ContentHash::of_bytes(b"clip").as_str(), "byte_length": 4
            }],
            "provider_policy": { "execution": "local_only" },
            "operations": [
                { "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] }
            ],
            "requested_outputs": ["keyframes"]
        })
        .to_string(),
    )
    .unwrap();
    let keyframes_record = record("ingest", ArtifactKind::Keyframes, &keyframes);
    manifest.artifacts.push(keyframes_record.clone());
    let request = OperationRequest {
        operation: crate::scene_project::OperationDeclaration {
            id: "bake".into(),
            kind: OperationKind::TextureBake,
            inputs: Vec::new(),
            provider: None,
            max_attempts: 1,
        },
        identity: ContentHash::of_bytes(b"bake"),
        provider: None,
        attempt: 1,
        inputs: vec![
            ResolvedInput::Artifact(keyframes_record),
            ResolvedInput::Artifact(record("mesh", ArtifactKind::SurfaceMesh, &mesh)),
        ],
    };
    let outcome =
        BuiltInExecutor::new(&dir.0, &manifest).execute(&request, &CancellationToken::new());
    assert!(
        matches!(&outcome, AttemptOutcome::Failed(message) if message.contains(moved.as_str())),
        "{outcome:?}"
    );
    assert_eq!(fs::read(moved.resolve(&dir.0)).unwrap(), sidecar_bytes);
    assert!(!dir.0.join("artifacts/bake/surface-textures.json").exists());
}

#[test]
fn surface_mesh_parsing_polls_for_cancellation_during_validation() {
    let document = Fixture::new().mesh().to_json();
    let mut polls = 0;
    SurfaceMeshArtifact::from_json_with_cancel(&document, || {
        polls += 1;
        Ok(())
    })
    .unwrap();
    assert!(polls > 2, "validation polled only {polls} time(s)");

    // Cancellation after the first poll stops the traversal mid-way.
    let mut seen = 0;
    let error = SurfaceMeshArtifact::from_json_with_cancel(&document, || {
        seen += 1;
        if seen > 1 {
            Err("texture bake was canceled".into())
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    assert!(error.contains("canceled"), "{error}");
    assert!(seen < polls, "cancellation did not stop validation early");
}

#[test]
fn legacy_inputs_are_reported_unsupported_without_charging_the_bake() {
    let dir = TempDir::new("legacy-inputs");
    let fixture = Fixture::new();
    let (keyframes, mesh) = fixture.write(&dir.0);
    let legacy = path("inputs/legacy-keyframes.bin");
    fs::write(legacy.resolve(&dir.0), [0xff_u8, 0x00, 0x7b, 0x80]).unwrap();
    let legacy_mesh = path("inputs/legacy-mesh.json");
    fs::write(legacy_mesh.resolve(&dir.0), r#"{"vertices":[]}"#).unwrap();
    let record = |id: &str, kind: ArtifactKind, path: &ProjectPath| {
        ResolvedInput::Artifact(crate::scene_project::ArtifactRecord {
            id: format!("{id}.output"),
            kind,
            produced_by: id.into(),
            operation_identity: ContentHash::of_bytes(id.as_bytes()),
            path: path.clone(),
            content_hash: ContentHash::of_bytes(&fs::read(path.resolve(&dir.0)).unwrap()),
            provider: None,
        })
    };
    let manifest = SceneProjectManifest::from_json(
        &json!({
            "schema_version": 1,
            "project_id": "legacy-inputs",
            "quality_mode": "standard",
            "inputs": [{
                "id": "clip", "path": "media/clip.webm",
                "content_hash": ContentHash::of_bytes(b"clip").as_str(), "byte_length": 4
            }],
            "provider_policy": { "execution": "local_only" },
            "operations": [
                { "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] }
            ],
            "requested_outputs": ["keyframes"]
        })
        .to_string(),
    )
    .unwrap();
    let executor = BuiltInExecutor::new(&dir.0, &manifest);
    let request = |keyframes: &ProjectPath, mesh: &ProjectPath| OperationRequest {
        operation: crate::scene_project::OperationDeclaration {
            id: "bake".into(),
            kind: OperationKind::TextureBake,
            inputs: Vec::new(),
            provider: None,
            max_attempts: 1,
        },
        identity: ContentHash::of_bytes(b"bake"),
        provider: None,
        attempt: 1,
        inputs: vec![
            record("ingest", ArtifactKind::Keyframes, keyframes),
            record("mesh", ArtifactKind::SurfaceMesh, mesh),
        ],
    };
    for (keyframes, mesh, named) in [
        (&legacy, &mesh, &legacy),
        (&keyframes, &legacy_mesh, &legacy_mesh),
    ] {
        let outcome = executor.execute(&request(keyframes, mesh), &CancellationToken::new());
        assert!(
            matches!(&outcome, AttemptOutcome::Unsupported(message)
                if message.contains(named.as_str()) && message.contains("not a versioned")),
            "{outcome:?}"
        );
    }
    assert!(!dir.0.join("artifacts/bake").exists());
}

#[test]
fn keyframe_pixels_load_in_cancelable_chunks() {
    let dir = TempDir::new("cancel-pixels");
    let (keyframes, _) = Fixture::new().write(&dir.0);
    let artifact =
        KeyframesArtifact::from_json(&fs::read_to_string(keyframes.resolve(&dir.0)).unwrap())
            .unwrap();
    let mut polls = 0;
    let rgba = artifact
        .load_pixels_with_cancel(&dir.0, 1, || {
            polls += 1;
            false
        })
        .unwrap();
    assert_eq!(rgba, artifact.load_pixels(&dir.0, 1).unwrap());
    assert!(polls >= 1);
    let error = artifact
        .load_pixels_with_cancel(&dir.0, 1, || true)
        .unwrap_err();
    assert!(error.contains("canceled"), "{error}");
}

#[test]
fn texture_sidecars_must_be_the_declared_png() {
    let dir = TempDir::new("png-sidecar");
    let inputs = Fixture::new().write(&dir.0);
    let outcome = bake(&dir.0, &inputs);
    let header = fs::read(texture_file(&dir.0, &outcome, 0)).unwrap();
    let texture = &outcome.artifact.materials[0].texture;
    assert!(is_png_rgb8(&header, texture.width, texture.height));
    assert!(!is_png_rgb8(&header, texture.width + 1, texture.height));
    assert!(!is_png_rgb8(b"not a png", texture.width, texture.height));
    let sidecars =
        artifact_sidecars(ArtifactKind::SurfaceTextures, &outcome.artifact.to_json()).unwrap();
    assert_eq!(sidecars[0].png_rgb8, Some((texture.width, texture.height)));
}

#[test]
fn rewriting_keyframes_removes_only_unreferenced_superseded_sidecars() {
    let dir = TempDir::new("superseded");
    let index = path("inputs/keyframes.json");
    let frame = |frame_index: usize, rgba: &'static [u8]| SampledFrame {
        frame_index,
        timestamp_seconds: frame_index as f64,
        width: 1,
        height: 1,
        rgba,
    };
    let (first, _) = write_keyframes_artifact(
        &dir.0,
        &index,
        &[frame(0, &[1, 2, 3, 255]), frame(1, &[4, 5, 6, 255])],
    )
    .unwrap();
    // Another recorded keyframes artifact, anywhere in the project, lists
    // frame 1's content-addressed sidecar.
    let other = path("elsewhere/other.idx");
    fs::create_dir_all(dir.0.join("elsewhere")).unwrap();
    let (mut other_artifact, _) =
        write_keyframes_artifact(&dir.0, &other, &[frame(1, &[4, 5, 6, 255])]).unwrap();
    other_artifact.frames[0].pixels = first.frames[1].pixels.clone();
    fs::write(other.resolve(&dir.0), other_artifact.to_json()).unwrap();
    let manifest_document = json!({
        "schema_version": 1,
        "project_id": "superseded",
        "quality_mode": "standard",
        "inputs": [{
            "id": "clip", "path": "media/clip.webm",
            "content_hash": ContentHash::of_bytes(b"clip").as_str(), "byte_length": 4
        }],
        "provider_policy": { "execution": "local_only" },
        "operations": [
            { "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] },
            { "id": "ingest2", "kind": "ingest_video", "inputs": [{ "media": "clip" }] }
        ],
        "requested_outputs": ["keyframes"]
    });
    let mut manifest = SceneProjectManifest::from_json(&manifest_document.to_string()).unwrap();
    manifest
        .artifacts
        .push(crate::scene_project::ArtifactRecord {
            id: "ingest2.output".into(),
            kind: ArtifactKind::Keyframes,
            produced_by: "ingest2".into(),
            operation_identity: ContentHash::of_bytes(b"ingest2"),
            path: other.clone(),
            content_hash: ContentHash::of_bytes(&fs::read(other.resolve(&dir.0)).unwrap()),
            provider: None,
        });

    // The plain writer never deletes.
    write_keyframes_artifact(&dir.0, &index, &[frame(2, &[7, 8, 9, 255])]).unwrap();
    let exists = |artifact: &KeyframesArtifact, at: usize| {
        artifact.frames[at].pixels.resolve(&dir.0).exists()
    };
    assert!(exists(&first, 0) && exists(&first, 1));

    // The project writer removes what no recorded artifact still lists.
    let (_, _) = write_keyframes_artifact(&dir.0, &index, &first_frames()).unwrap();
    let (second, _) =
        write_project_keyframes_artifact(&dir.0, &manifest, &index, &[frame(2, &[7, 8, 9, 255])])
            .unwrap();
    assert!(!exists(&first, 0), "superseded sidecar was kept");
    assert!(
        exists(&first, 1),
        "a sidecar another recorded artifact lists was removed"
    );
    assert!(exists(&second, 0));
    second.load_pixels(&dir.0, 2).unwrap();
}

fn first_frames() -> Vec<SampledFrame<'static>> {
    let frame = |frame_index: usize, rgba: &'static [u8]| SampledFrame {
        frame_index,
        timestamp_seconds: frame_index as f64,
        width: 1,
        height: 1,
        rgba,
    };
    vec![frame(0, &[1, 2, 3, 255]), frame(1, &[4, 5, 6, 255])]
}

#[test]
fn png_sidecars_are_validated_completely() {
    let rgba: Vec<u8> = (0..6 * 4).map(|value| value as u8 * 9).collect();
    let png = crate::textured_glb::encode_png_rgb(3, 2, &rgba).unwrap();
    validate_png_rgb8(&png, 3, 2).unwrap();
    assert!(validate_png_rgb8(&png, 2, 3).is_err());
    // Header only: no pixel stream.
    assert!(validate_png_rgb8(&png[..33], 3, 2).is_err());
    // A flipped byte inside IDAT breaks its CRC.
    let mut corrupt = png.clone();
    corrupt[45] ^= 0xff;
    assert!(validate_png_rgb8(&corrupt, 3, 2)
        .unwrap_err()
        .contains("CRC"));
    // Trailing data after IEND.
    let mut trailing = png.clone();
    trailing.push(0);
    assert!(validate_png_rgb8(&trailing, 3, 2).is_err());
}

#[test]
fn surface_mesh_rejects_mismatched_grid_sites_before_copying() {
    let fixture = Fixture::new();
    let evidence = ReconstructionEvidenceView::new(
        ReconstructionProviderDescriptor::classic(),
        EvidenceScale::ArbitraryMonocular,
        fixture.cameras.clone(),
        fixture.regions.clone(),
        &fixture.points,
        &fixture.triangles,
    )
    .unwrap();
    let error = SurfaceMeshArtifact::from_evidence(&evidence, &fixture.sites[..3]).unwrap_err();
    assert!(error.contains("grid sites"), "{error}");
}

#[test]
fn case_aliased_previous_textures_are_not_deleted() {
    let dir = TempDir::new("texture-alias");
    let inputs = Fixture::new().write(&dir.0);
    let output = path("artifacts/bake/surface-textures.json");
    let first = bake(&dir.0, &inputs);
    // The previous artifact named the same texture with different case.
    let mut previous = first.artifact.clone();
    let lower = previous.materials[0].texture.path.clone();
    let upper = ProjectPath::new(lower.as_str().replace(".png", ".PNG")).unwrap();
    fs::copy(lower.resolve(&dir.0), upper.resolve(&dir.0)).unwrap();
    previous.materials[0].texture.path = upper.clone();
    fs::write(output.resolve(&dir.0), previous.to_json()).unwrap();
    bake(&dir.0, &inputs);
    // On a case-insensitive filesystem these are one file; deleting the old
    // spelling would delete the newly referenced texture.
    assert!(upper.resolve(&dir.0).exists());
    assert!(lower.resolve(&dir.0).exists());
}
