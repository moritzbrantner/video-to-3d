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
    assert_eq!(frames(&second.invalidation.reused), vec![0]);
    assert_eq!(frames(&second.invalidation.invalidated), vec![1]);
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
    assert_eq!(frames(&repaired.invalidation.reused), vec![1]);
    assert_eq!(frames(&repaired.invalidation.invalidated), vec![0]);
    repaired.artifact.verify_textures(&dir.0).unwrap();
}

#[test]
fn a_coherently_replaced_texture_is_not_reused() {
    let dir = TempDir::new("coherent-tamper");
    let fixture = Fixture::new();
    let inputs = fixture.write(&dir.0);
    let first = bake(&dir.0, &inputs);
    // Replace texture 0 with another well-formed PNG of the same size and
    // record its hash, so document and sidecar agree.
    let material = &first.artifact.materials[0];
    let (width, height) = (material.texture.width, material.texture.height);
    let other = vec![200_u8; width as usize * height as usize * 4];
    let png = crate::textured_glb::encode_png_rgb(width, height, &other).unwrap();
    fs::write(texture_file(&dir.0, &first, material.reference_frame), &png).unwrap();
    let mut tampered = first.artifact.clone();
    tampered.materials[0].texture.content_hash = ContentHash::of_bytes(&png);
    let output = path("artifacts/bake/surface-textures.json");
    fs::write(output.resolve(&dir.0), tampered.to_json()).unwrap();

    let rebuilt = bake(&dir.0, &inputs);
    assert!(rebuilt.written.contains(&material.reference_frame));
    assert_eq!(
        rebuilt.artifact.materials[0].texture.content_hash,
        material.texture.content_hash
    );
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

    // The crop must lie inside a non-empty source image.
    for (origin, source) in [
        ([1000, 1000], (100, 100)),
        ([0, 0], (0, 0)),
        ([u32::MAX, 0], (100, 100)),
    ] {
        let mut cropped = original.clone();
        cropped.materials[0].texture.crop_origin = origin;
        cropped.materials[0].texture.source_image_width = source.0;
        cropped.materials[0].texture.source_image_height = source.1;
        let error = cropped.validate().unwrap_err();
        assert!(error.contains("outside its"), "{error}");
    }

    // Appearance keys are unique.
    if original.materials.len() > 1 {
        let mut shared = original.clone();
        shared.materials[1].appearance_key = shared.materials[0].appearance_key.clone();
        assert!(shared.validate().unwrap_err().contains("appearance key"));
    }

    // Triangles need at least three points.
    let mut pointless = original.clone();
    pointless.point_count = 2;
    assert!(pointless
        .validate()
        .unwrap_err()
        .contains("over 2 point(s)"));

    // Observed provenance needs a supporting view besides the reference.
    let mut alone = original.clone();
    alone.materials[0].source_frames = vec![alone.materials[0].reference_frame];
    let error = alone.validate().unwrap_err();
    assert!(error.contains("no supporting frame"), "{error}");
    assert!(alone.scene_entries("bake").is_err());

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
        // Not charged: a declaration conflict is not a bake attempt.
        matches!(&outcome, AttemptOutcome::Unsupported(message) if message.contains("would overwrite the declared path `artifacts/bake/clip.webm`")),
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
            matches!(&outcome, AttemptOutcome::Unsupported(message) if message.contains(declared)),
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
        matches!(&outcome, AttemptOutcome::Unsupported(message) if message.contains(moved.as_str())),
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

    // A superseded sidecar that a media input (or any other project entry)
    // owns is never removed either.
    let (_, _) = write_keyframes_artifact(&dir.0, &index, &first_frames()).unwrap();
    manifest.inputs[0].path = first.frames[0].pixels.clone();
    write_project_keyframes_artifact(&dir.0, &manifest, &index, &[frame(2, &[7, 8, 9, 255])])
        .unwrap();
    assert!(
        exists(&first, 0),
        "a sidecar a media input owns was removed"
    );
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
    // Chunks inserted right after IHDR: a repeated IHDR and an unknown
    // critical chunk are rejected; an unknown ancillary chunk is fine.
    let with_chunk = |kind: &[u8; 4], data: &[u8]| {
        let mut chunk = (data.len() as u32).to_be_bytes().to_vec();
        chunk.extend_from_slice(kind);
        chunk.extend_from_slice(data);
        let crc = crc32fast::hash(&chunk[4..]);
        chunk.extend_from_slice(&crc.to_be_bytes());
        let mut out = png[..33].to_vec();
        out.extend_from_slice(&chunk);
        out.extend_from_slice(&png[33..]);
        out
    };
    let ihdr = png[16..29].to_vec();
    assert!(validate_png_rgb8(&with_chunk(b"IHDR", &ihdr), 3, 2)
        .unwrap_err()
        .contains("repeats IHDR"));
    assert!(validate_png_rgb8(&with_chunk(b"ZZZZ", b"x"), 3, 2)
        .unwrap_err()
        .contains("unknown critical"));
    // Unknown ancillary chunks may affect rendering in some decoders, so a
    // sidecar carrying one is not trusted (#143 static-image allowlist).
    assert!(validate_png_rgb8(&with_chunk(b"zzZz", b"x"), 3, 2).is_err());
    validate_png_rgb8(&with_chunk(b"PLTE", &[0, 0, 0]), 3, 2).unwrap();
    // PLTE must hold 1..=256 RGB entries.
    for palette in [&[][..], &[0, 0][..], &[0; 257 * 3][..]] {
        assert!(validate_png_rgb8(&with_chunk(b"PLTE", palette), 3, 2)
            .unwrap_err()
            .contains("PLTE"));
    }
    // Decoding inverts the encoder's filters exactly.
    let rgb = decode_png_rgb8_with_cancel(&png, 3, 2, || false).unwrap();
    let expected: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|pixel| pixel[..3].to_vec())
        .collect();
    assert_eq!(rgb, expected);
    assert!(decode_png_rgb8_with_cancel(&png, 3, 2, || true)
        .unwrap_err()
        .contains("canceled"));
    // Declared sizes beyond the texture limit are rejected before decoding.
    assert!(
        decode_png_rgb8_with_cancel(&png, 1 << 13, 1 << 13, || false)
            .unwrap_err()
            .contains("pixel limit")
    );
    // Exactly one zlib stream: trailing bytes inside IDAT are rejected.
    let idat_length = u32::from_be_bytes(png[33..37].try_into().unwrap()) as usize;
    let mut idat = png[41..41 + idat_length].to_vec();
    idat.push(0);
    let mut trailing_zlib = png[..33].to_vec();
    trailing_zlib.extend_from_slice(&(idat.len() as u32).to_be_bytes());
    let mut body = b"IDAT".to_vec();
    body.extend_from_slice(&idat);
    trailing_zlib.extend_from_slice(&body);
    trailing_zlib.extend_from_slice(&crc32fast::hash(&body).to_be_bytes());
    trailing_zlib.extend_from_slice(&png[41 + idat_length + 4..]);
    assert!(validate_png_rgb8(&trailing_zlib, 3, 2).is_err());
    // IEND carries no data.
    let mut iend = png[..png.len() - 12].to_vec();
    iend.extend_from_slice(&1_u32.to_be_bytes());
    iend.extend_from_slice(b"IEND");
    iend.push(0);
    iend.extend_from_slice(&crc32fast::hash(b"IEND\0").to_be_bytes());
    assert!(validate_png_rgb8(&iend, 3, 2).unwrap_err().contains("IEND"));
    // PNG dimensions are never zero.
    assert!(!is_png_rgb8(&png, 0, 2));
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

#[test]
fn an_interrupted_bake_removes_the_textures_it_created() {
    let dir = TempDir::new("interrupted-bake");
    let fixture = Fixture::new();
    let inputs = fixture.write(&dir.0);
    // The document path is occupied by a directory, so committing fails
    // after the textures were written.
    let output = path("artifacts/bake/surface-textures.json");
    fs::create_dir_all(output.resolve(&dir.0).join("blocker")).unwrap();
    let result = bake_texture_artifact(
        &dir.0,
        &output,
        &inputs.0,
        &inputs.1,
        &CancellationToken::new(),
    );
    assert!(result.is_err());
    let textures = dir.0.join("artifacts/bake/textures");
    // Textures were written (their directory exists) before the failure.
    assert!(textures.is_dir());
    let left: Vec<_> = fs::read_dir(&textures)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .collect()
        })
        .unwrap_or_default();
    assert!(left.is_empty(), "orphaned textures: {left:?}");
}

/// Reference frames of same-reference appearance artifacts (the only rule a
/// surface-textures artifact records).
fn frames(artifacts: &[crate::surface_materials::AppearanceArtifact]) -> Vec<usize> {
    artifacts
        .iter()
        .map(|artifact| {
            assert_eq!(
                artifact.rule,
                crate::surface_materials::MaterialRule::SameReferenceGrid
            );
            artifact.reference_frame
        })
        .collect()
}

// Acceptance for issue #143, written before the implementation from the issue
// and the public PNG/artifact seams (`validate_png_rgb8`,
// `SurfaceTexturesArtifact::from_json`, `bake_texture_artifact`).

/// `png` with one `kind` chunk inserted right after IHDR.
fn png_with_chunk_after_ihdr(png: &[u8], kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut chunk = (data.len() as u32).to_be_bytes().to_vec();
    chunk.extend_from_slice(kind);
    chunk.extend_from_slice(data);
    let crc = crc32fast::hash(&chunk[4..]);
    chunk.extend_from_slice(&crc.to_be_bytes());
    let mut out = png[..33].to_vec();
    out.extend_from_slice(&chunk);
    out.extend_from_slice(&png[33..]);
    out
}

/// A `width` x `height` 8-bit RGB PNG whose single IDAT holds `zlib` verbatim.
fn png_with_idat(width: u32, height: u32, zlib: &[u8]) -> Vec<u8> {
    let chunk = |png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]| {
        png.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = png.len();
        png.extend_from_slice(kind);
        png.extend_from_slice(data);
        let crc = crc32fast::hash(&png[start..]);
        png.extend_from_slice(&crc.to_be_bytes());
    };
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = width.to_be_bytes().to_vec();
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", zlib);
    chunk(&mut png, b"IEND", &[]);
    png
}

/// A valid zlib stream of `raw` preceded by `padding` empty, non-final stored
/// deflate blocks: it inflates to exactly `raw` however large it is.
fn padded_zlib(raw: &[u8], padding: usize) -> Vec<u8> {
    assert!(raw.len() <= 0xffff);
    let mut zlib = vec![0x78, 0x01];
    for _ in 0..padding {
        // BFINAL = 0, BTYPE = stored, LEN = 0, NLEN = !0.
        zlib.extend_from_slice(&[0x00, 0x00, 0x00, 0xff, 0xff]);
    }
    let length = raw.len() as u16;
    zlib.push(0x01);
    zlib.extend_from_slice(&length.to_le_bytes());
    zlib.extend_from_slice(&(!length).to_le_bytes());
    zlib.extend_from_slice(raw);
    let (mut a, mut b) = (1_u32, 0_u32);
    for byte in raw {
        a = (a + u32::from(*byte)) % 65521;
        b = (b + a) % 65521;
    }
    zlib.extend_from_slice(&((b << 16) | a).to_be_bytes());
    zlib
}

#[test]
fn rendering_affecting_png_chunks_are_rejected() {
    let rgba: Vec<u8> = (0..6 * 4).map(|value| value as u8 * 9).collect();
    let png = crate::textured_glb::encode_png_rgb(3, 2, &rgba).unwrap();
    validate_png_rgb8(&png, 3, 2).unwrap();
    // tRNS for color type 2: one 16-bit RGB sample that renders transparent.
    let transparent = png_with_chunk_after_ihdr(&png, b"tRNS", &[0, 0, 0, 9, 0, 18]);
    assert!(
        validate_png_rgb8(&transparent, 3, 2).is_err(),
        "a PNG with tRNS renders differently from its RGB samples"
    );
    // Color-management chunks change how the same samples render.
    let gamma = 45_455_u32.to_be_bytes();
    let chromaticities = [0_u8; 32];
    let profile = b"icc\0\0x\x9c\x03\0\0\0\0\x01";
    for (kind, data) in [
        (b"gAMA", &gamma[..]),
        (b"cHRM", &chromaticities[..]),
        (b"sRGB", &[0][..]),
        (b"iCCP", &profile[..]),
    ] {
        assert!(
            validate_png_rgb8(&png_with_chunk_after_ihdr(&png, kind, data), 3, 2).is_err(),
            "a PNG with {} must not verify as plain RGB",
            String::from_utf8_lossy(kind)
        );
    }
    // Ancillary chunks that do not affect rendering stay accepted.
    validate_png_rgb8(
        &png_with_chunk_after_ihdr(&png, b"tEXt", b"Comment\0x"),
        3,
        2,
    )
    .unwrap();
}

#[test]
fn only_text_and_time_ancillary_chunks_are_trusted() {
    // Codex finding on 4559b45 (eXIf orientation): a denylist keeps missing
    // chunks that change how the same samples render. Sidecars therefore
    // accept only the textual and timestamp ancillary chunks, besides the
    // suggested PLTE, and reject every other ancillary chunk, known or not.
    let rgba: Vec<u8> = (0..6 * 4).map(|value| value as u8 * 9).collect();
    let png = crate::textured_glb::encode_png_rgb(3, 2, &rgba).unwrap();
    let time = [0x07, 0xea, 10, 10, 12, 0, 0];
    for (kind, data) in [
        (b"tEXt", &b"Comment\0x"[..]),
        (b"zTXt", &b"Comment\0\0x\x9c\x03\0\0\0\0\x01"[..]),
        (b"iTXt", &b"Comment\0\0\0\0\0x"[..]),
        (b"tIME", &time[..]),
    ] {
        validate_png_rgb8(&png_with_chunk_after_ihdr(&png, kind, data), 3, 2).unwrap_or_else(
            |error| {
                panic!(
                    "{} must stay accepted: {error}",
                    String::from_utf8_lossy(kind)
                )
            },
        );
    }
    let exif = b"MM\0\x2a\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01\0\x06\0\0\0\0\0\0";
    for (kind, data) in [
        (b"eXIf", &exif[..]),
        (b"pHYs", &[0, 0, 0, 1, 0, 0, 0, 2, 0][..]),
        (b"sBIT", &[8, 8, 8][..]),
        (b"bKGD", &[0, 0, 0, 0, 0, 0][..]),
        (b"zzZz", &b"x"[..]),
    ] {
        assert!(
            validate_png_rgb8(&png_with_chunk_after_ihdr(&png, kind, data), 3, 2).is_err(),
            "a PNG with {} must not verify as a static RGB texture",
            String::from_utf8_lossy(kind)
        );
    }
}

#[test]
fn a_cached_texture_with_transparency_is_not_reused() {
    let dir = TempDir::new("trns-tamper");
    let fixture = Fixture::new();
    let inputs = fixture.write(&dir.0);
    let first = bake(&dir.0, &inputs);
    let material = &first.artifact.materials[0];
    let file = texture_file(&dir.0, &first, material.reference_frame);
    // Same RGB samples, plus a tRNS chunk that makes one color transparent;
    // the document records the new hash, so document and sidecar agree.
    let original = fs::read(&file).unwrap();
    let transparent = png_with_chunk_after_ihdr(&original, b"tRNS", &[0, 0, 0, 0, 0, 0]);
    fs::write(&file, &transparent).unwrap();
    let mut tampered = first.artifact.clone();
    tampered.materials[0].texture.content_hash = ContentHash::of_bytes(&transparent);
    let output = path("artifacts/bake/surface-textures.json");
    fs::write(output.resolve(&dir.0), tampered.to_json()).unwrap();

    let rebuilt = bake(&dir.0, &inputs);
    assert!(
        rebuilt.written.contains(&material.reference_frame),
        "written {:?}, reused {:?}",
        rebuilt.written,
        rebuilt.reused
    );
    assert!(!rebuilt.reused.contains(&material.reference_frame));
    assert_eq!(
        rebuilt.artifact.materials[0].texture.content_hash,
        material.texture.content_hash
    );
    assert_eq!(
        fs::read(texture_file(&dir.0, &rebuilt, material.reference_frame)).unwrap(),
        original
    );
    rebuilt.artifact.verify_textures(&dir.0).unwrap();
}

#[test]
fn texture_artifacts_reject_zero_crop_extents() {
    let dir = TempDir::new("zero-extent");
    let inputs = Fixture::new().write(&dir.0);
    let outcome = bake(&dir.0, &inputs);
    SurfaceTexturesArtifact::from_json(&outcome.artifact.to_json()).unwrap();
    for (zero_width, zero_height) in [(true, false), (false, true), (true, true)] {
        let mut artifact = outcome.artifact.clone();
        let texture = &mut artifact.materials[0].texture;
        if zero_width {
            texture.width = 0;
        }
        if zero_height {
            texture.height = 0;
        }
        let (width, height) = (texture.width, texture.height);
        assert!(
            SurfaceTexturesArtifact::from_json(&artifact.to_json()).is_err(),
            "a {width}x{height} texture crop must not validate"
        );
        assert!(artifact.validate().is_err());
    }
}

#[test]
fn encoded_png_data_is_bounded_by_the_declared_image() {
    // 1x1 RGB: one filter byte plus one pixel.
    let raw = [0_u8, 10, 20, 30];
    let compact = padded_zlib(&raw, 0);
    validate_png_rgb8(&png_with_idat(1, 1, &compact), 1, 1).unwrap();

    // The same pixels behind 1 MiB of empty stored blocks: still a valid zlib
    // stream that inflates to exactly the image, but its encoded size is far
    // beyond anything a 1x1 image needs.
    let padded = padded_zlib(&raw, (1 << 20) / 5);
    assert_eq!(
        miniz_oxide::inflate::decompress_to_vec_zlib(&padded).unwrap(),
        raw
    );
    assert!(
        validate_png_rgb8(&png_with_idat(1, 1, &padded), 1, 1).is_err(),
        "a 1x1 sidecar with {} encoded bytes must be rejected",
        padded.len()
    );

    // Uncompressed (stored) encodings of real images stay within the bound.
    let (width, height) = (256_u32, 256_u32);
    let mut stored_raw = Vec::new();
    for y in 0..height {
        stored_raw.push(0);
        for x in 0..width {
            stored_raw.extend_from_slice(&[x as u8, y as u8, (x ^ y) as u8]);
        }
    }
    let stored = miniz_oxide::deflate::compress_to_vec_zlib(&stored_raw, 0);
    assert!(stored.len() > stored_raw.len());
    validate_png_rgb8(&png_with_idat(width, height, &stored), width, height).unwrap();
}

/// A `width` x `height` 8-bit RGB PNG whose IDAT chunks hold `parts` in order.
fn png_with_idat_parts(width: u32, height: u32, parts: &[&[u8]]) -> Vec<u8> {
    let single = png_with_idat(width, height, &[]);
    // Signature + IHDR, then one IDAT per part, then IEND.
    let mut png = single[..33].to_vec();
    for part in parts {
        png.extend_from_slice(&(part.len() as u32).to_be_bytes());
        let start = png.len();
        png.extend_from_slice(b"IDAT");
        png.extend_from_slice(part);
        let crc = crc32fast::hash(&png[start..]);
        png.extend_from_slice(&crc.to_be_bytes());
    }
    png.extend_from_slice(&single[single.len() - 12..]);
    png
}

#[test]
fn split_idat_chunks_stream_into_one_zlib_stream() {
    let (width, height) = (300_u32, 200_u32);
    let mut raw = Vec::new();
    for y in 0..height {
        raw.push(0);
        for x in 0..width {
            raw.extend_from_slice(&[x as u8, y as u8, (x * y) as u8]);
        }
    }
    // Stored so the stream spans several inflate input chunks.
    let zlib = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 0);
    let (head, tail) = zlib.split_at(zlib.len() / 3);
    let (middle, tail) = tail.split_at(70_000);
    let parts: [&[u8]; 6] = [&[], &head[..1], &head[1..], middle, &[], tail];
    let pixels = decode_png_rgb8_with_cancel(
        &png_with_idat_parts(width, height, &parts),
        width,
        height,
        || false,
    )
    .unwrap();
    assert_eq!(
        pixels,
        raw.chunks_exact(width as usize * 3 + 1)
            .flat_map(|line| line[1..].to_vec())
            .collect::<Vec<_>>()
    );
    // Trailing data in a later IDAT is still rejected.
    let trailing: [&[u8]; 3] = [&zlib, &[], &[0]];
    assert!(validate_png_rgb8(
        &png_with_idat_parts(width, height, &trailing),
        width,
        height
    )
    .is_err());
    // Only empty IDAT chunks hold no stream.
    let empty: [&[u8]; 2] = [&[], &[]];
    assert!(validate_png_rgb8(&png_with_idat_parts(width, height, &empty), width, height).is_err());
}

#[test]
fn animated_png_chunks_are_rejected() {
    // Codex finding on ceeaa61: APNG control/data chunks let extra frames
    // render differently while the IDAT samples match the expected texture.
    let rgba: Vec<u8> = (0..6 * 4).map(|value| value as u8 * 9).collect();
    let png = crate::textured_glb::encode_png_rgb(3, 2, &rgba).unwrap();
    let mut actl = 2_u32.to_be_bytes().to_vec();
    actl.extend_from_slice(&0_u32.to_be_bytes());
    let mut fctl = [0_u8; 26];
    fctl[7] = 3; // width
    fctl[11] = 2; // height
    let mut fdat = 1_u32.to_be_bytes().to_vec();
    fdat.extend_from_slice(&[
        0x78, 0x01, 0x01, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
    ]);
    for (kind, data) in [
        (b"acTL", &actl[..]),
        (b"fcTL", &fctl[..]),
        (b"fdAT", &fdat[..]),
    ] {
        assert!(
            validate_png_rgb8(&png_with_chunk_after_ihdr(&png, kind, data), 3, 2).is_err(),
            "a PNG with {} must not verify as a static RGB texture",
            String::from_utf8_lossy(kind)
        );
    }
}

// Acceptance for issue #144, written before the implementation from the issue
// and the public bake / keyframe-writer seams.

#[test]
fn baseline_fixture_bake_output_is_unchanged() {
    // Pins the bake output of the existing two-reference fixture, captured at
    // the pre-#144 baseline, so incremental reference loading stays
    // byte-identical.
    let dir = TempDir::new("bake-baseline-144");
    let inputs = Fixture::new().write(&dir.0);
    let outcome = bake(&dir.0, &inputs);
    assert_eq!(outcome.content_hash.as_str(), BASELINE_144_BAKE_HASH);
}

#[test]
fn a_reference_without_textureable_triangles_is_never_loaded() {
    let dir = TempDir::new("unused-reference");
    let mut fixture = Fixture::new();
    // Reference 3 owns one observed point that no triangle uses.
    let start = fixture.points.len();
    fixture.points.push(point(start));
    fixture.sites.push(DenseGridSite { x: 4, y: 4 });
    fixture.cameras.push(camera(3, 0.9));
    fixture.images.push((3, image(33)));
    let mut patches = vec![
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
    patches.push(DenseReferencePatchStats {
        reference_frame: 3,
        source_frames: vec![0],
        primary_start: start,
        primary_points: 1,
        completion_start: start + 1,
        completed_points: 0,
        ..DenseReferencePatchStats::default()
    });
    fixture.regions = classic_reference_patch_regions(&patches);
    let inputs = fixture.write(&dir.0);
    let reference = bake(&dir.0, &inputs);
    assert!(reference
        .artifact
        .materials
        .iter()
        .all(|material| material.reference_frame != 3));

    // Corrupt reference 3's keyframe sidecar. A bake that loaded it would fail
    // hash verification; one that skips non-contributing references succeeds
    // with the same output.
    let keyframes =
        KeyframesArtifact::from_json(&fs::read_to_string(inputs.0.resolve(&dir.0)).unwrap())
            .unwrap();
    let unused = keyframes
        .frames
        .iter()
        .find(|frame| frame.frame_index == 3)
        .unwrap();
    fs::write(unused.pixels.resolve(&dir.0), b"corrupt").unwrap();
    fs::remove_file(path("artifacts/bake/surface-textures.json").resolve(&dir.0)).unwrap();
    let rebuilt = bake_texture_artifact(
        &dir.0,
        &path("artifacts/bake/surface-textures.json"),
        &inputs.0,
        &inputs.1,
        &CancellationToken::new(),
    )
    .expect("a reference that textures nothing must not be loaded");
    assert_eq!(rebuilt.content_hash, reference.content_hash);
}

#[test]
fn a_failed_keyframe_index_write_leaves_no_new_sidecars() {
    let dir = TempDir::new("keyframe-rollback");
    let frame = |frame_index: usize, rgba: &'static [u8]| SampledFrame {
        frame_index,
        timestamp_seconds: frame_index as f64,
        width: 1,
        height: 1,
        rgba,
    };
    // Another index already owns frame 0's content-addressed sidecar.
    let (existing, _) =
        write_keyframes_artifact(&dir.0, &path("inputs/a.json"), &[frame(0, &[1, 2, 3, 255])])
            .unwrap();
    let kept = existing.frames[0].pixels.resolve(&dir.0);
    assert!(kept.exists());

    // The replacement index path is occupied by a directory, so committing it
    // fails after the new sidecars would have been written.
    let blocked = path("inputs/b.json");
    fs::create_dir_all(blocked.resolve(&dir.0).join("blocker")).unwrap();
    let result = write_keyframes_artifact(
        &dir.0,
        &blocked,
        &[
            frame(0, &[1, 2, 3, 255]),
            frame(1, &[4, 5, 6, 255]),
            frame(2, &[7, 8, 9, 255]),
        ],
    );
    assert!(result.is_err());

    let frames_dir = dir.0.join("inputs/frames");
    let mut left: Vec<_> = fs::read_dir(&frames_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect();
    left.sort();
    // Only the pre-existing sidecar another index owns survives.
    assert_eq!(left, vec![kept]);
}
const BASELINE_144_BAKE_HASH: &str =
    "sha256:4561a455e335e787b7f4655e8c9652aaaba2b4e248c9d82dd0ce1ca0f252f8ea";
