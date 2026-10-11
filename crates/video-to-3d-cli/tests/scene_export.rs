//! Independent #84 export acceptance, baseline 301d393.
//! Explicit acceptance tier: generate unchanged #75 truth resources first.
//! The test importer records a synthetic canonical scene through ProjectStore;
//! it does not execute or claim reconstruction, semantic decomposition or #77.
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
use video_to_3d_core::scene_model::AssembledScene;
use video_to_3d_core::scene_project::{ContentHash, ProjectPath};
use video_to_3d_core::scene_runner::{
    AttemptOutcome, CancellationToken, OperationExecutor, OperationRequest, ProducedArtifact,
    RunOptions,
};
use video_to_3d_core::scene_store::{hash_file, ProjectStore, Reproducibility};

struct Project {
    root: PathBuf,
    scene: Value,
}
impl Project {
    fn new() -> Self {
        Self::with_observed(false)
    }
    fn with_observed(observed: bool) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let input = PathBuf::from(
            std::env::var_os("VIDEO_TO_3D_EXPORT_FIXTURE")
                .expect("run the documented #84 fixture preparation first"),
        );
        let root = input.parent().unwrap().join(format!(
            "export-case-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir_all(root.join("resources")).unwrap();
        for entry in fs::read_dir(input.join("resources")).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), root.join("resources").join(entry.file_name())).unwrap();
        }
        let input_scene = fs::read_to_string(input.join("scene.json")).unwrap();
        let input_scene = if observed {
            observed_scene(&input, &root, &input_scene)
        } else {
            input_scene
        };
        let canonical = AssembledScene::from_json(&input_scene)
            .expect("independently prepared canonical scene is valid")
            .to_canonical_json()
            .unwrap();
        let scene: Value = serde_json::from_str(&canonical).unwrap();
        let source = fs::read(input.join("provenance.txt")).unwrap();
        fs::write(root.join("synthetic-source.txt"), &source).unwrap();
        fs::write(root.join("project.json"), json!({
            "schema_version":1,"project_id":"synthetic-export-acceptance","quality_mode":"preview",
            "inputs":[{"id":"source","path":"synthetic-source.txt","content_hash":ContentHash::of_bytes(&source).as_str(),"byte_length":source.len()}],
            "provider_policy":{"execution":"local_only"},
            "operations":[
                {"id":"ingest","kind":"ingest_video","inputs":[{"media":"source"}]},
                {"id":"sparse","kind":"sparse_reconstruction","inputs":[{"operation":"ingest"}]},
                {"id":"assemble","kind":"scene_assembly","inputs":[{"operation":"sparse"}]}
            ],"requested_outputs":["assembled_scene"],
            "exports":[{"id":"blender","format":"glb","path":"exports/scene.glb","includes":["assembled_scene"]}]
        }).to_string()).unwrap();
        let project = Self { root, scene };
        project.record_scene();
        project
    }
    fn record_scene(&self) {
        let (store, mut manifest) = ProjectStore::open(&self.root.join("project.json")).unwrap();
        let result = store
            .build(
                &mut manifest,
                &ImportScene(self.root.clone(), self.scene.clone()),
                RunOptions::default(),
                &CancellationToken::new(),
            )
            .unwrap();
        assert!(result.run.is_complete());
    }
    fn export(&self, extra: &[&str]) -> std::process::Output {
        Command::new(env!("CARGO_BIN_EXE_video-to-3d"))
            .args([
                "export",
                self.root.join("project.json").to_str().unwrap(),
                "--target",
                "glb",
            ])
            .args(extra)
            .current_dir(self.root.parent().unwrap())
            .output()
            .unwrap()
    }
    fn export_ok(&self, extra: &[&str]) -> Value {
        let result = self.export(extra);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        glb_document(&fs::read(self.root.join("exports/scene.glb")).unwrap())
    }
    fn source_snapshot(&self) -> Vec<(PathBuf, Vec<u8>)> {
        snapshot(&self.root.join("resources"))
    }
}
impl Drop for Project {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
struct ImportScene(PathBuf, Value);
impl OperationExecutor for ImportScene {
    fn execute(&self, request: &OperationRequest, _: &CancellationToken) -> AttemptOutcome {
        let bytes = if request.operation.id == "assemble" {
            self.1.to_string().into_bytes()
        } else {
            request.identity.as_str().as_bytes().to_vec()
        };
        let path = ProjectPath::new(format!("artifacts/{}.json", request.operation.id)).unwrap();
        fs::create_dir_all(self.0.join("artifacts")).unwrap();
        fs::write(path.resolve(&self.0), &bytes).unwrap();
        AttemptOutcome::Succeeded(ProducedArtifact::new(
            path,
            ContentHash::of_bytes(&bytes),
            Reproducibility::Deterministic,
        ))
    }
}
fn snapshot(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut files = vec![];
    if !root.exists() {
        return files;
    }
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(snapshot(&path));
        } else {
            let bytes = fs::read(&path).unwrap();
            files.push((path, bytes));
        }
    }
    files.sort();
    files
}
fn glb_document(bytes: &[u8]) -> Value {
    assert_eq!(&bytes[..4], b"glTF");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
        bytes.len()
    );
    let length = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    assert_eq!(&bytes[16..20], b"JSON");
    serde_json::from_slice(&bytes[20..20 + length]).unwrap()
}
fn identified<'a>(document: &'a Value, id: &str) -> &'a Value {
    let found = document["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|node| node["extras"]["video_to_3d_id"] == id)
        .collect::<Vec<_>>();
    assert_eq!(found.len(), 1, "stable asset {id} occurs exactly once");
    found[0]
}
/// A real observed mesh from the unchanged #75 sampled frames, computed by the
/// existing canonical core. This is not fabricated object decomposition.
fn observed_scene(input: &Path, root: &Path, scene: &str) -> String {
    let frames = (0..18)
        .map(|index| {
            let bytes = fs::read(input.join(format!("sampled/frame-{index:02}.ppm"))).unwrap();
            let header = b"P6\n360 203\n255\n";
            assert!(bytes.starts_with(header));
            let pixels = &bytes[header.len()..];
            assert_eq!(pixels.len(), 360 * 203 * 3);
            video_to_3d_core::FrameInput {
                width: 360,
                height: 203,
                rgba: pixels
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
                    .collect(),
            }
        })
        .collect();
    let result = video_to_3d_core::reconstruct(&video_to_3d_core::ReconstructionRequest {
        frames,
        options: Default::default(),
    })
    .unwrap();
    let evidence = video_to_3d_core::ReconstructionEvidenceView::from_classic(&result).unwrap();
    assert!(!evidence.triangles.is_empty());
    let bake = video_to_3d_core::surface_materials::bake_surface_materials(
        &evidence,
        &result.dense_grid_sites,
        &[],
    )
    .unwrap();
    let bytes = video_to_3d_core::textured_glb::encode_textured_glb(&evidence, &bake).unwrap();
    fs::write(root.join("resources/observed-smoke.glb"), &bytes).unwrap();
    let mut scene: Value = serde_json::from_str(scene).unwrap();
    scene["frame"]["unit"] = json!("arbitrary_monocular");
    scene["cameras"]=json!(evidence.cameras.iter().map(|camera| {
        let r=&camera.rotation; let t=&camera.translation; let q=[1.,-1.,-1.];
        let rotation=(0..9).map(|index| { let row=index/3;let col=index%3; q[row]*r[col*3+row]*q[col] }).collect::<Vec<_>>();
        let center=(0..3).map(|axis| -q[axis]*(0..3).map(|row|r[row*3+axis]*t[row]).sum::<f32>()).collect::<Vec<_>>();
        json!({"id":format!("observed-camera-{}",camera.frame_index),"source_frame":camera.frame_index,"authority":camera.authority,"transform":{"translation":center,"rotation":quaternion(&rotation),"scale":[1,1,1]}})
    }).collect::<Vec<_>>());
    scene["resources"].as_array_mut().unwrap().push(json!({"id":"observed-smoke-mesh","kind":"mesh","path":"resources/observed-smoke.glb","content_hash":ContentHash::of_bytes(&bytes).as_str(),"provenance":["geometric_multi_view"],"confidence":0.8,"source_frames":evidence.cameras.iter().map(|camera|camera.frame_index).collect::<Vec<_>>()}));
    scene["assets"].as_array_mut().unwrap().push(json!({"id":"observed-smoke","role":"environment","transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},"visual":{"mesh":{"mesh":"observed-smoke-mesh"}}}));
    scene.to_string()
}
fn quaternion(r: &[f32]) -> [f32; 4] {
    // Standard row-major rotation -> unit xyzw; fixture-only coordinate oracle.
    let trace = r[0] + r[4] + r[8];
    if trace > 0. {
        let s = (trace + 1.).sqrt() * 2.;
        return [
            (r[7] - r[5]) / s,
            (r[2] - r[6]) / s,
            (r[3] - r[1]) / s,
            s / 4.,
        ];
    }
    let i = if r[0] > r[4] && r[0] > r[8] {
        0
    } else if r[4] > r[8] {
        1
    } else {
        2
    };
    let j = (i + 1) % 3;
    let k = (j + 1) % 3;
    let scale = (1. + r[i * 3 + i] - r[j * 3 + j] - r[k * 3 + k]).sqrt() * 2.;
    let mut q = [0.; 4];
    q[i] = scale / 4.;
    q[j] = (r[j * 3 + i] + r[i * 3 + j]) / scale;
    q[k] = (r[k * 3 + i] + r[i * 3 + k]) / scale;
    q[3] = (r[k * 3 + j] - r[j * 3 + k]) / scale;
    q
}

fn local_matrix(node: &Value) -> Vec<f64> {
    if let Some(matrix) = node["matrix"].as_array() {
        assert_eq!(matrix.len(), 16);
        return matrix.iter().map(|v| v.as_f64().unwrap()).collect();
    }
    let vector = |key: &str, default: &[f64]| {
        node[key].as_array().map_or_else(
            || default.to_vec(),
            |v| v.iter().map(|x| x.as_f64().unwrap()).collect(),
        )
    };
    let t = vector("translation", &[0., 0., 0.]);
    let q = vector("rotation", &[0., 0., 0., 1.]);
    let s = vector("scale", &[1., 1., 1.]);
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    vec![
        (1. - 2. * (y * y + z * z)) * s[0],
        2. * (x * y + z * w) * s[0],
        2. * (x * z - y * w) * s[0],
        0.,
        2. * (x * y - z * w) * s[1],
        (1. - 2. * (x * x + z * z)) * s[1],
        2. * (y * z + x * w) * s[1],
        0.,
        2. * (x * z + y * w) * s[2],
        2. * (y * z - x * w) * s[2],
        (1. - 2. * (x * x + y * y)) * s[2],
        0.,
        t[0],
        t[1],
        t[2],
        1.,
    ]
}

fn sidecar(root: &Path) -> Value {
    // Do not lock a speculative filename or Rust serializer API. The versioned
    // sidecar must carry canonical scene metadata, either root or `scene` envelope.
    let found = snapshot(&if root.join("exports").is_dir() {
        root.join("exports")
    } else {
        root.to_path_buf()
    })
    .into_iter()
    .filter_map(|(path, bytes)| {
        if path.extension().is_none_or(|extension| extension != "json") {
            return None;
        }
        let value: Value = serde_json::from_slice(&bytes).ok()?;
        let scene = if value.get("assets").is_some() {
            value
        } else {
            value.get("scene")?.clone()
        };
        (scene.get("resources").is_some() && scene.get("frame").is_some()).then_some(scene)
    })
    .collect::<Vec<_>>();
    assert_eq!(found.len(), 1, "one canonical scene provenance sidecar");
    found[0].clone()
}

#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn synthetic_inputs_have_real_valid_meshes_and_nontrivial_hierarchy() {
    let project = Project::new();
    assert_eq!(
        project.scene["assets"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|asset| asset["parent"] == "room-group")
            .count(),
        12
    );
    for resource in project.scene["resources"].as_array().unwrap() {
        let path = project.root.join(resource["path"].as_str().unwrap());
        assert_eq!(
            hash_file(&path).unwrap().0.as_str(),
            resource["content_hash"].as_str().unwrap()
        );
        if resource["kind"] == "mesh" {
            let mesh = glb_document(&fs::read(path).unwrap());
            assert!(!mesh["meshes"].as_array().unwrap().is_empty());
        }
    }
}
#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn glb_preserves_nodes_parent_transforms_materials_and_embedded_textures() {
    let project = Project::new();
    let before = project.source_snapshot();
    let document = project.export_ok(&[]);
    let nodes = document["nodes"].as_array().unwrap();
    let group = identified(&document, "room-group");
    for asset in project.scene["assets"].as_array().unwrap() {
        let node = identified(&document, asset["id"].as_str().unwrap());
        let actual = local_matrix(node);
        let expected = local_matrix(&asset["transform"]);
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "canonical local transform preserved");
        }
        if asset["parent"] == "room-group" {
            let index = nodes
                .iter()
                .position(|candidate| candidate == node)
                .unwrap();
            assert!(group["children"]
                .as_array()
                .unwrap()
                .contains(&json!(index)));
        }
        if asset.get("visual").is_some() {
            assert!(node["mesh"].is_u64());
        }
    }
    assert!(document["buffers"]
        .as_array()
        .unwrap()
        .iter()
        .all(|buffer| buffer.get("uri").is_none()));
    let images = document["images"]
        .as_array()
        .expect("real canonical texture exported");
    assert!(!images.is_empty());
    assert!(images
        .iter()
        .all(|image| image.get("uri").is_none() && image["bufferView"].is_u64()));
    for expected in project.scene["materials"].as_array().unwrap() {
        let matches = document["materials"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|material| material["name"] == expected["id"])
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1);
        let pbr = &matches[0]["pbrMetallicRoughness"];
        for (a, b) in pbr["baseColorFactor"]
            .as_array()
            .unwrap()
            .iter()
            .zip(expected["base_color"].as_array().unwrap())
        {
            assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-6);
        }
        for (actual, wanted) in [
            ("metallicFactor", "metallic"),
            ("roughnessFactor", "roughness"),
        ] {
            assert!(
                (pbr[actual].as_f64().unwrap() - expected[wanted].as_f64().unwrap()).abs() < 1e-6
            );
        }
        assert!(pbr["baseColorTexture"]["index"].is_u64());
    }
    assert_eq!(project.source_snapshot(), before);
    let first = snapshot(&project.root.join("exports"));
    project.export_ok(&[]);
    assert_eq!(
        snapshot(&project.root.join("exports")),
        first,
        "deterministic repeat export"
    );
}
#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn provenance_and_collision_metadata_survive_without_relabeling() {
    let project = Project::new();
    project.export_ok(&[]);
    let metadata = sidecar(&project.root);
    assert_eq!(metadata["frame"], project.scene["frame"]);
    for expected in project.scene["resources"].as_array().unwrap() {
        let actual = metadata["resources"]
            .as_array()
            .unwrap()
            .iter()
            .find(|resource| resource["id"] == expected["id"])
            .unwrap();
        for field in [
            "kind",
            "path",
            "content_hash",
            "provenance",
            "confidence",
            "source_frames",
        ] {
            assert_eq!(actual[field], expected[field]);
        }
    }
    for expected in project.scene["assets"].as_array().unwrap() {
        let actual = metadata["assets"]
            .as_array()
            .unwrap()
            .iter()
            .find(|asset| asset["id"] == expected["id"])
            .unwrap();
        for field in [
            "role",
            "parent",
            "transform",
            "labels",
            "collision",
            "collision_provenance",
        ] {
            assert_eq!(actual[field], expected[field]);
        }
    }
}
#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn corrupt_or_missing_mesh_and_texture_resources_fail_closed() {
    for (path, remove) in [
        ("resources/crate.glb", false),
        ("resources/checker.png", false),
        ("resources/door.glb", true),
    ] {
        let project = Project::new();
        if remove {
            fs::remove_file(project.root.join(path)).unwrap();
        } else {
            fs::write(project.root.join(path), b"tampered").unwrap();
        }
        let result = project.export(&[]);
        assert_eq!(
            result.status.code(),
            Some(5),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!project.root.join("exports/scene.glb").exists());
    }
}
#[cfg(unix)]
#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn resource_symlink_escape_is_not_exported_even_with_matching_hash() {
    let project = Project::new();
    let resource = project.root.join("resources/crate.glb");
    let outside = project
        .root
        .parent()
        .unwrap()
        .join(format!("outside-{}.glb", std::process::id()));
    fs::copy(&resource, &outside).unwrap();
    fs::remove_file(&resource).unwrap();
    std::os::unix::fs::symlink(&outside, &resource).unwrap();
    let result = project.export(&[]);
    let _ = fs::remove_file(outside);
    assert_eq!(
        result.status.code(),
        Some(5),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!project.root.join("exports/scene.glb").exists());
}

#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn observation_only_export_excludes_labeled_completion() {
    let project = Project::with_observed(true);
    let completed = project.export_ok(&["--scene-mode", "completed"]);
    identified(&completed, "completion-sentinel");
    let metadata = sidecar(&project.root);
    assert_eq!(metadata["cameras"], project.scene["cameras"]);
    let resource = metadata["resources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|resource| resource["id"] == "observed-smoke-mesh")
        .unwrap();
    let source = project.scene["resources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|resource| resource["id"] == "observed-smoke-mesh")
        .unwrap();
    assert!(!source["source_frames"].as_array().unwrap().is_empty());
    assert_eq!(resource["source_frames"], source["source_frames"]);
    assert_eq!(resource["provenance"], json!(["geometric_multi_view"]));
    let observed = project.export_ok(&["--scene-mode", "observed"]);
    assert!(observed["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .all(|node| node["extras"]["video_to_3d_id"] != "completion-sentinel"));
    identified(&observed, "observed-smoke");
    // Existing canonical classification governs authored content; do not force
    // unrelated imported truth assets into the observation-only selection.
    assert!(
        video_to_3d_core::scene_model::SceneProvenance::GeometricMultiView.is_observed_geometry()
    );
    assert!(
        video_to_3d_core::scene_model::SceneProvenance::RevalidatedCompletion
            .is_observed_geometry()
    );
    assert!(!video_to_3d_core::scene_model::SceneProvenance::Authored.is_observed_geometry());
}

#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE and VIDEO_TO_3D_EXPORT_PORTABLE required"]
fn portable_bundle_survives_removal_of_its_project_workspace() {
    let project = Project::new();
    project.export_ok(&["--scene-mode", "completed"]);
    let destination = PathBuf::from(
        std::env::var_os("VIDEO_TO_3D_EXPORT_PORTABLE")
            .expect("set a disposable output directory for Blender acceptance"),
    );
    assert!(!destination.exists(), "portable destination must be fresh");
    fs::create_dir_all(&destination).unwrap();
    for (path, bytes) in snapshot(&project.root.join("exports")) {
        let relative = path.strip_prefix(project.root.join("exports")).unwrap();
        let target = destination.join(relative);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(target, bytes).unwrap();
    }
    let root = project.root.clone();
    drop(project);
    assert!(
        !root.exists(),
        "original provider/project workspace is absent"
    );
    let document = glb_document(&fs::read(destination.join("scene.glb")).unwrap());
    identified(&document, "cabinet");
    identified(&document, "floor");
    let metadata = sidecar(
        &destination
            .parent()
            .unwrap()
            .join(destination.file_name().unwrap()),
    );
    assert_eq!(metadata["frame"]["unit"], "meters");
}
