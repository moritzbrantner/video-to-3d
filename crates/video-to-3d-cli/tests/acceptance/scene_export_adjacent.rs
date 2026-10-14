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
    while !json.len().is_multiple_of(4) {
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
    assert_materialless_scene_export(false);
}

#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn omitted_default_material_table_exports_without_inventing_materials() {
    assert_materialless_scene_export(true);
}

fn assert_materialless_scene_export(omit_table: bool) {
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
    if omit_table {
        project.scene.as_object_mut().unwrap().remove("materials");
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
fn export_directory_cannot_alias_protected_project_subtrees() {
    // ProjectPath deliberately uses portable '/' syntax. Its public resolver
    // creates native components, so this same CLI contract covers Windows '\\'
    // filesystem paths when executed there; it does not pass invalid manifest paths.
    for path in [
        "resources/export/scene.glb",
        "artifacts/export/scene.glb",
        "scene.glb",
    ] {
        let project = Project::new();
        let manifest_path = project.root.join("project.json");
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["exports"][0]["path"] = json!(path);
        fs::write(&manifest_path, manifest.to_string()).unwrap();
        ProjectStore::open(&manifest_path).expect("valid portable manifest path");
        let before = snapshot(&project.root);
        let result = project.export(&[]);
        assert!(!result.status.success(), "protected destination accepted: {path}");
        let reason = String::from_utf8_lossy(&result.stderr).to_lowercase();
        assert!(
            reason.contains("export")
                && (reason.contains("overlap")
                    || reason.contains("protected")
                    || reason.contains("contain")
                    || reason.contains("alias")),
            "useful protected-path rejection for {path}: {reason}"
        );
        assert_eq!(snapshot(&project.root), before, "rejection preserves project bytes: {path}");
    }
}

#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn isolated_sibling_exports_share_a_parent_without_overwriting_other_output() {
    let project = Project::new();
    let manifest_path = project.root.join("project.json");
    let mut manifest: Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["exports"].as_array_mut().unwrap().push(json!({
        "id":"web", "format":"web_bundle", "path":"exports/web", "includes":["assembled_scene"]
    }));
    fs::write(&manifest_path, manifest.to_string()).unwrap();
    ProjectStore::open(&manifest_path).expect("valid sibling export declaration");
    fs::create_dir_all(project.root.join("exports/web")).unwrap();
    fs::write(project.root.join("exports/web/sentinel.txt"), b"other export bytes").unwrap();
    let other_before = snapshot(&project.root.join("exports/web"));
    let source_before = project.source_snapshot();
    let document = project.export_ok(&[]);
    identified(&document, "crate");
    assert_eq!(snapshot(&project.root.join("exports/web")), other_before);
    assert_eq!(project.source_snapshot(), source_before);
    receipt_consumer::verify_bundle(&project.root.join("exports"), &sidecar_document(&project.root))
        .expect("actual sibling-compatible portable export");
}

#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn concrete_export_subtrees_cannot_overlap_another_declared_export() {
    for (glb, web) in [
        ("exports/web/nested/scene.glb", "exports/web"),
        ("exports/scene.glb", "exports/scene.glb.resources/web"),
    ] {
        let project = Project::new();
        let manifest_path = project.root.join("project.json");
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["exports"][0]["path"] = json!(glb);
        manifest["exports"].as_array_mut().unwrap().push(json!({
            "id":"web", "format":"web_bundle", "path":web, "includes":["assembled_scene"]
        }));
        fs::write(&manifest_path, manifest.to_string()).unwrap();
        ProjectStore::open(&manifest_path).expect("valid portable overlapping declaration");
        let before = snapshot(&project.root);
        let result = project.export(&[]);
        assert!(!result.status.success(), "overlapping exports accepted: {glb}, {web}");
        assert_eq!(snapshot(&project.root), before, "rejection preserves project bytes");
    }
}

#[cfg(unix)]
fn assert_sibling_physical_alias_rejected(target: &str, directory: bool, missing_child: bool) {
    let project = Project::new();
    let exports = project.root.join("exports");
    fs::create_dir_all(&exports).unwrap();
    if directory {
        fs::create_dir_all(exports.join(target)).unwrap();
        fs::write(exports.join(target).join("owned-web.txt"), b"other export bytes").unwrap();
    } else {
        fs::write(exports.join(target), b"other export bytes").unwrap();
    }
    std::os::unix::fs::symlink(target, exports.join("web")).unwrap();
    let web = if missing_child { "exports/web/future" } else { "exports/web" };
    let manifest_path = project.root.join("project.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["exports"].as_array_mut().unwrap().push(json!({
        "id":"web", "format":"web_bundle", "path":web, "includes":["assembled_scene"]
    }));
    fs::write(&manifest_path, manifest.to_string()).unwrap();
    ProjectStore::open(&manifest_path).expect("valid sibling declaration with a physical alias");
    let before = snapshot(&project.root);
    let result = project.export(&[]);
    assert!(
        !result.status.success(),
        "physical sibling alias accepted: {web} -> {target}"
    );
    let reason = String::from_utf8_lossy(&result.stderr).to_lowercase();
    assert!(
        reason.contains("export")
            && ["overlap", "protected", "contain", "alias", "symlink"]
                .iter().any(|word| reason.contains(word)),
        "useful physical-alias rejection: {reason}"
    );
    assert_eq!(snapshot(&project.root), before, "rejection preserves every project byte");
}

#[cfg(unix)]
#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn sibling_resource_symlink_alias_is_rejected_before_writing() {
    assert_sibling_physical_alias_rejected("scene.glb.resources", true, false);
}

#[cfg(unix)]
#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn sibling_glb_symlink_alias_is_rejected_before_writing() {
    assert_sibling_physical_alias_rejected("scene.glb", false, false);
}

#[cfg(unix)]
#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn sibling_sidecar_symlink_alias_is_rejected_before_writing() {
    assert_sibling_physical_alias_rejected("scene.scene.json", false, false);
}

#[cfg(unix)]
#[test]
#[ignore = "explicit #84 acceptance: VIDEO_TO_3D_EXPORT_FIXTURE required"]
fn sibling_missing_child_beneath_symlink_is_rejected_before_writing() {
    assert_sibling_physical_alias_rejected("scene.glb.resources", true, true);
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
