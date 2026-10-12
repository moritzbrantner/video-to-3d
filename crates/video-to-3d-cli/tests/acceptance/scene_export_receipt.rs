//! Independent #84 portable receipt consumer, authored before production.
//! The self-test uses real prepared GLB payloads, never claims product export.
use serde_json::Value;
use std::{collections::BTreeSet, fs, path::Path};
use video_to_3d_core::scene_model::AssembledScene;
use video_to_3d_core::scene_project::ContentHash;

pub fn verify_bundle(root: &Path, document: &Value) -> Result<(), String> {
    let scene = document.get("scene").cloned().unwrap_or_else(|| {
        let mut scene = document.clone();
        scene.as_object_mut().unwrap().remove("export_receipt");
        scene
    });
    AssembledScene::from_json(&scene.to_string()).map_err(|e| e.to_string())?;
    let receipt = &document["export_receipt"];
    ensure(
        receipt["schema_version"] == 1 && receipt["target"] == "glb",
        "receipt version/target",
    )?;
    let mode = receipt["scene_mode"].as_str().ok_or("receipt scene_mode")?;
    ensure(
        ["observed", "completed"].contains(&mode),
        "receipt scene_mode",
    )?;
    let bytes = verified_payload(root, &receipt["output"])?;
    ensure(
        bytes.len() >= 20 && &bytes[..4] == b"glTF",
        "output GLB header",
    )?;
    ensure(
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()) == 2,
        "output GLB version",
    )?;
    ensure(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize == bytes.len(),
        "output GLB length",
    )?;
    let json_length = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    ensure(
        &bytes[16..20] == b"JSON" && json_length <= bytes.len() - 20,
        "output GLB JSON",
    )?;
    let glb: Value =
        serde_json::from_slice(&bytes[20..20 + json_length]).map_err(|e| e.to_string())?;
    let nodes = glb["nodes"].as_array().ok_or("GLB nodes")?;
    let assets = receipt["assets"].as_object().ok_or("receipt assets")?;
    let canonical_assets = scene["assets"].as_array().ok_or("canonical assets")?;
    let mut mapped = BTreeSet::new();
    for (id, mapping) in assets {
        let canonical = canonical_assets
            .iter()
            .find(|asset| asset["id"] == id.as_str())
            .ok_or("unknown mapped asset")?;
        let index = mapping["node_index"].as_u64().ok_or("asset node_index")? as usize;
        let node = nodes.get(index).ok_or("asset node range")?;
        ensure(mapped.insert(index), "duplicate mapped node")?;
        ensure(
            node["extras"]["video_to_3d_id"] == id.as_str(),
            "asset node identity",
        )?;
        let selection = mapping["selection"].as_str().ok_or("asset selection")?;
        ensure(
            ["visual", "ancestor", "collision"].contains(&selection),
            "asset selection",
        )?;
        if selection == "visual" {
            ensure(
                canonical["visual"]["mesh"].is_object() && node["mesh"].is_u64(),
                "visual mesh mapping",
            )?;
            let mesh_index = node["mesh"].as_u64().unwrap() as usize;
            let mesh = glb["meshes"]
                .as_array()
                .and_then(|meshes| meshes.get(mesh_index))
                .ok_or("visual mesh range")?;
            if let Some(material_id) = canonical["visual"]["mesh"]["material"].as_str() {
                for primitive in mesh["primitives"].as_array().ok_or("mesh primitives")? {
                    let index = primitive["material"]
                        .as_u64()
                        .ok_or("visual material mapping")? as usize;
                    let material = glb["materials"]
                        .as_array()
                        .and_then(|materials| materials.get(index))
                        .ok_or("material range")?;
                    ensure(
                        material["name"] == material_id,
                        "canonical material identity",
                    )?;
                }
            }
        }
        if let Some(parent) = canonical["parent"].as_str() {
            let parent_index = assets.get(parent).ok_or("missing ancestor mapping")?["node_index"]
                .as_u64()
                .ok_or("parent node_index")? as usize;
            ensure(
                nodes
                    .get(parent_index)
                    .and_then(|node| node["children"].as_array())
                    .is_some_and(|children| children.contains(&Value::from(index))),
                "mapped parent edge",
            )?;
        }
    }
    for (index, node) in nodes.iter().enumerate() {
        if node["extras"]["video_to_3d_id"].is_string() {
            ensure(mapped.contains(&index), "unmapped canonical node")?;
        }
    }
    if mode == "completed" {
        for asset in canonical_assets {
            if asset["visual"]["mesh"].is_object() {
                ensure(
                    assets
                        .get(asset["id"].as_str().unwrap())
                        .is_some_and(|mapping| mapping["selection"] == "visual"),
                    "missing completed visual",
                )?;
            }
        }
    }
    let resources = receipt["resources"]
        .as_object()
        .ok_or("receipt resources")?;
    let canonical_resources = scene["resources"].as_array().ok_or("canonical resources")?;
    ensure(
        resources.len() == canonical_resources.len(),
        "portable resource closure",
    )?;
    for resource in canonical_resources {
        let id = resource["id"].as_str().unwrap();
        let mapping = resources.get(id).ok_or("missing portable resource")?;
        ensure(
            mapping["content_hash"] == resource["content_hash"],
            "changed canonical resource hash",
        )?;
        verified_payload(root, mapping)?;
        let mesh_indices = mapping["mesh_indices"]
            .as_array()
            .ok_or("resource mesh_indices")?;
        let expected: BTreeSet<u64> = canonical_assets
            .iter()
            .filter(|asset| asset["visual"]["mesh"]["mesh"] == id)
            .filter_map(|asset| assets.get(asset["id"].as_str().unwrap()))
            .filter(|mapping| mapping["selection"] == "visual")
            .filter_map(|mapping| {
                nodes[mapping["node_index"].as_u64().unwrap() as usize]["mesh"].as_u64()
            })
            .collect();
        let actual: BTreeSet<u64> = mesh_indices
            .iter()
            .map(|index| index.as_u64().ok_or("resource mesh index"))
            .collect::<Result<_, _>>()?;
        ensure(
            actual == expected && actual.len() == mesh_indices.len(),
            "resource mesh mapping",
        )?;
        let material_indices = mapping["material_indices"]
            .as_array()
            .ok_or("resource material_indices")?;
        let expected_names: BTreeSet<&str> = scene["materials"]
            .as_array()
            .ok_or("canonical materials")?
            .iter()
            .filter(|material| material["base_color_texture"] == id)
            .filter_map(|material| material["id"].as_str())
            .collect();
        let mut expected_materials: BTreeSet<u64> = BTreeSet::new();
        if resource["kind"] == "mesh" {
            for index in &expected {
                for primitive in glb["meshes"][*index as usize]["primitives"]
                    .as_array()
                    .ok_or("mesh primitives")?
                {
                    if let Some(index) = primitive["material"].as_u64() {
                        expected_materials.insert(index);
                    }
                }
            }
        } else if resource["kind"] == "texture" {
            for (index, material) in glb["materials"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                if material["name"]
                    .as_str()
                    .is_some_and(|name| expected_names.contains(name))
                {
                    expected_materials.insert(index as u64);
                }
            }
        }
        let actual_materials: BTreeSet<u64> = material_indices
            .iter()
            .map(|index| index.as_u64().ok_or("resource material index"))
            .collect::<Result<_, _>>()?;
        ensure(
            actual_materials == expected_materials
                && actual_materials.len() == material_indices.len(),
            "resource material mapping",
        )?;
    }
    Ok(())
}
fn ensure(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn verified_payload(root: &Path, metadata: &Value) -> Result<Vec<u8>, String> {
    let relative = metadata["path"].as_str().ok_or("payload path")?;
    let path =
        video_to_3d_core::scene_project::ProjectPath::new(relative).map_err(|e| e.to_string())?;
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    let resolved = path
        .resolve(&root)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    ensure(resolved.starts_with(&root), "portable path escape")?;
    let bytes = fs::read(resolved).map_err(|e| e.to_string())?;
    ensure(
        metadata["byte_length"].as_u64() == Some(bytes.len() as u64),
        "payload byte length",
    )?;
    ensure(
        metadata["content_hash"] == ContentHash::of_bytes(&bytes).as_str(),
        "payload hash",
    )?;
    Ok(bytes)
}
