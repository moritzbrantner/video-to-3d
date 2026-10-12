// Independent retrospective adjacency acceptance before further #84 repair.
fn rebuild_adjacent_scene(project: &Project) {
    let (_, manifest) = ProjectStore::open(&project.root.join("project.json")).unwrap();
    for artifact in &manifest.artifacts {
        if artifact.kind == video_to_3d_core::scene_project::ArtifactKind::AssembledScene {
            fs::remove_file(artifact.path.resolve(&project.root)).unwrap();
        }
    }
    project.record_scene();
}
fn rewrite_resource_glb(project: &mut Project, id: &str, edit: impl FnOnce(&mut Value)) {
    let resource = project.scene["resources"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|r| r["id"] == id)
        .unwrap();
    let path = project.root.join(resource["path"].as_str().unwrap());
    let bytes = fs::read(&path).unwrap();
    let mut document = glb_document(&bytes);
    edit(&mut document);
    let json_length = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let suffix = &bytes[20 + json_length..];
    let mut json = serde_json::to_vec(&document).unwrap();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let mut output = Vec::new();
    output.extend_from_slice(b"glTF");
    output.extend_from_slice(&2_u32.to_le_bytes());
    output.extend_from_slice(&((20 + json.len() + suffix.len()) as u32).to_le_bytes());
    output.extend_from_slice(&(json.len() as u32).to_le_bytes());
    output.extend_from_slice(&0x4e4f534a_u32.to_le_bytes());
    output.extend_from_slice(&json);
    output.extend_from_slice(suffix);
    fs::write(path, &output).unwrap();
    resource["content_hash"] = json!(ContentHash::of_bytes(&output).as_str());
}

#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn valid_materialless_canonical_scene_exports_without_inventing_materials() {
    let mut project = Project::new();
    project.scene["materials"] = json!([]);
    for asset in project.scene["assets"].as_array_mut().unwrap() {
        if let Some(mesh) = asset
            .get_mut("visual")
            .and_then(|v| v.get_mut("mesh"))
            .and_then(Value::as_object_mut)
        {
            mesh.remove("material");
        }
    }
    let ids = project.scene["resources"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["kind"] == "mesh")
        .map(|r| r["id"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    for id in ids {
        rewrite_resource_glb(&mut project, &id, |doc| {
            doc.as_object_mut().unwrap().remove("materials");
            for mesh in doc["meshes"].as_array_mut().unwrap() {
                for primitive in mesh["primitives"].as_array_mut().unwrap() {
                    primitive.as_object_mut().unwrap().remove("material");
                }
            }
        });
    }
    AssembledScene::from_json(&project.scene.to_string())
        .expect("public canonical model admits materialless visuals");
    rebuild_adjacent_scene(&project);
    let document = project.export_ok(&[]);
    assert!(document
        .get("materials")
        .is_none_or(|v| v.as_array().is_some_and(Vec::is_empty)));
}

#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn default_empty_source_scene_cannot_silently_export_alternate_scene_geometry() {
    let mut project = Project::new();
    rewrite_resource_glb(&mut project, "crate-mesh", |doc| {
        let old = doc["scenes"].clone();
        doc["scenes"] = json!([{"nodes":[]}, old[0].clone()]);
        doc["scene"] = json!(0);
    });
    AssembledScene::from_json(&project.scene.to_string())
        .expect("canonical resource is hash-correct and model-valid");
    rebuild_adjacent_scene(&project);
    let result = project.export(&[]);
    assert!(
        !result.status.success(),
        "empty declared default scene must not become mesh-only geometry"
    );
    let reason = String::from_utf8_lossy(&result.stderr).to_lowercase();
    assert!(
        reason.contains("scene")
            && (reason.contains("unsupported")
                || reason.contains("default")
                || reason.contains("layout")),
        "useful source layout rejection: {reason}"
    );
    assert!(!project.root.join("exports/scene.glb").exists());
}

#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn normalized_mesh_order_preserves_real_portable_resource_mapping() {
    let mut project = Project::new();
    for resource in project.scene["resources"].as_array_mut().unwrap() {
        if resource["id"] == "crate-mesh" {
            resource["id"] = json!("a");
        } else if resource["id"] == "door-mesh" {
            resource["id"] = json!("a-");
        }
    }
    for asset in project.scene["assets"].as_array_mut().unwrap() {
        if let Some(mesh) = asset
            .get_mut("visual")
            .and_then(|v| v.get_mut("mesh"))
            .and_then(Value::as_object_mut)
        {
            if mesh["mesh"] == "crate-mesh" {
                mesh.insert("mesh".into(), json!("a"));
            } else if mesh["mesh"] == "door-mesh" {
                mesh.insert("mesh".into(), json!("a-"));
            }
        }
    }
    AssembledScene::from_json(&project.scene.to_string())
        .expect("valid renamed resource identities");
    rebuild_adjacent_scene(&project);
    project.export_ok(&[]);
    let metadata: Value =
        serde_json::from_slice(&fs::read(project.root.join("exports/scene.scene.json")).unwrap())
            .unwrap();
    receipt_consumer::verify_bundle(&project.root.join("exports"), &metadata)
        .expect("independent portable consumer resolves actual normalized mesh bindings");
}
