//! Portable canonical-scene interchange. The shared 3d-lab asset/scene/export
//! boundaries own geometry, material validation and normalization; this adapter
//! owns project containment, canonical identity and the portable receipt.
use crate::scene_model::{AssembledScene, ResourceKind, SceneAsset};
use crate::scene_project::{
    ArtifactRecord, ContentHash, ExportFormat, ProjectPath, SceneProjectManifest, STATE_DIRECTORY,
};
use crate::scene_store::{hash_file, ProjectStore};
use serde::Serialize;
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use three_d_animation::{Quat, Transform};
use three_d_assets::{
    Asset, AssetMesh, BaseColorFactor, EncodedImage, Material, MaterialTextureBinding,
    MeshPrimitive, NormalTextureBinding, Texture,
};
use three_d_core::Vec3;
use three_d_scene::{SceneNode, SceneSnapshot};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneMode {
    Completed,
    Observed,
}

#[derive(Debug)]
pub enum ExportError {
    Incomplete(String),
    Unsupported(String),
    Invalid(String),
    Io(std::io::Error),
}
impl std::fmt::Display for ExportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Incomplete(message) | Self::Unsupported(message) | Self::Invalid(message) => {
                formatter.write_str(message)
            }
            Self::Io(error) => write!(formatter, "cannot write portable export: {error}"),
        }
    }
}
impl std::error::Error for ExportError {}
impl From<std::io::Error> for ExportError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
fn unsupported(error: impl std::fmt::Display) -> ExportError {
    ExportError::Unsupported(error.to_string())
}

#[derive(Serialize)]
pub struct ExportReceipt {
    pub schema_version: u32,
    pub target: &'static str,
    pub scene_mode: SceneMode,
    pub output: PayloadReceipt,
    pub assets: BTreeMap<String, AssetReceipt>,
    pub resources: BTreeMap<String, ResourceReceipt>,
}
#[derive(Serialize)]
pub struct PayloadReceipt {
    pub path: String,
    pub content_hash: ContentHash,
    pub byte_length: u64,
}
#[derive(Serialize)]
pub struct AssetReceipt {
    pub node_index: usize,
    pub selection: &'static str,
}
#[derive(Serialize)]
pub struct ResourceReceipt {
    #[serde(flatten)]
    pub payload: PayloadReceipt,
    pub mesh_indices: Vec<usize>,
    pub material_indices: Vec<usize>,
}

/// Export exactly one manifest-declared GLB from a current assembled artifact.
/// No source file is changed, and no output is opened before complete preflight.
pub fn export_project_glb(
    store: &ProjectStore,
    manifest: &SceneProjectManifest,
    manifest_path: &Path,
    artifact: &ArtifactRecord,
    mode: SceneMode,
) -> Result<ExportReceipt, ExportError> {
    let declarations = manifest
        .exports
        .iter()
        .filter(|export| export.format == ExportFormat::Glb)
        .collect::<Vec<_>>();
    let [declaration] = declarations.as_slice() else {
        return Err(ExportError::Invalid(
            "GLB export requires exactly one manifest-declared glb destination".into(),
        ));
    };
    if !declaration
        .includes
        .contains(&crate::scene_project::ArtifactKind::AssembledScene)
    {
        return Err(ExportError::Invalid(
            "GLB export must include assembled_scene".into(),
        ));
    }
    let root = store.root().canonicalize().map_err(|error| {
        ExportError::Incomplete(format!("cannot resolve project root: {error}"))
    })?;
    let scene_bytes = verified_bytes(&root, &artifact.path, &artifact.content_hash)?;
    let scene = AssembledScene::from_json(std::str::from_utf8(&scene_bytes).map_err(unsupported)?)
        .map_err(unsupported)?;
    // Retain the verified document's scalar representation. Converting typed
    // f32 fields through serde_json::Value would expand them to different f64
    // decimals even though validation and renderer precision are unchanged.
    let source_document: Value = serde_json::from_slice(&scene_bytes).map_err(unsupported)?;
    let mut payloads = BTreeMap::new();
    for resource in &scene.resources {
        payloads.insert(
            resource.id.clone(),
            verified_bytes(&root, &resource.path, &resource.content_hash)?,
        );
    }
    let selected = select_assets(&scene, mode)?;
    let snapshot = adapt_scene(&scene, &payloads, &selected)?;
    let (glb_bytes, _) =
        three_d_export::export_scene_glb_with_observations(&snapshot).map_err(unsupported)?;
    let mut document: Value = serde_json::from_slice(
        &gltf::binary::Glb::from_slice(&glb_bytes)
            .map_err(unsupported)?
            .json,
    )
    .map_err(unsupported)?;
    let assets = annotate_nodes(&scene, &selected, &mut document)?;
    annotate_material_identity(&source_document, &mut document)?;
    // Resource indices belong to the normalized output, not the input snapshot.
    let mut mesh_resources = BTreeMap::new();
    for (index, mesh) in document["meshes"]
        .as_array()
        .ok_or_else(|| unsupported("shared exporter omitted mesh table"))?
        .iter()
        .enumerate()
    {
        let id = mesh["name"]
            .as_str()
            .ok_or_else(|| unsupported("shared exporter omitted mesh resource identity"))?;
        if mesh_resources.insert(id.to_owned(), index).is_some() {
            return Err(unsupported(
                "shared exporter duplicated mesh resource identity",
            ));
        }
    }
    let destination = declaration.path.resolve(&root);
    let directory = destination
        .parent()
        .ok_or_else(|| ExportError::Invalid("GLB destination has no directory".into()))?;
    let filename = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ExportError::Invalid("GLB destination is not portable".into()))?;
    let metadata_path = destination.with_extension("scene.json");
    let mut resources = BTreeMap::new();
    let mut copies = Vec::new();
    for (index, resource) in scene.resources.iter().enumerate() {
        let extension = Path::new(resource.path.as_str())
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("bin");
        let path = format!("{filename}.resources/{index:04}.{extension}");
        let mesh_indices = mesh_resources
            .get(&resource.id)
            .copied()
            .into_iter()
            .collect::<Vec<_>>();
        let mut material_indices = BTreeSet::new();
        for &index in &mesh_indices {
            for primitive in document["meshes"][index]["primitives"]
                .as_array()
                .ok_or_else(|| unsupported("shared exporter omitted mesh primitives"))?
            {
                if let Some(index) = primitive["material"].as_u64() {
                    material_indices.insert(index as usize);
                }
            }
        }
        if resource.kind == ResourceKind::Texture {
            for material in &scene.materials {
                if material.base_color_texture.as_deref() == Some(&resource.id) {
                    if let Some(index) = document["materials"].as_array().and_then(|materials| {
                        materials
                            .iter()
                            .position(|entry| entry["name"] == material.id)
                    }) {
                        material_indices.insert(index);
                    }
                }
            }
        }
        let bytes = &payloads[&resource.id];
        copies.push((directory.join(&path), bytes.as_slice()));
        resources.insert(
            resource.id.clone(),
            ResourceReceipt {
                payload: PayloadReceipt {
                    path,
                    content_hash: resource.content_hash.clone(),
                    byte_length: bytes.len() as u64,
                },
                mesh_indices,
                material_indices: material_indices.into_iter().collect(),
            },
        );
    }
    preflight_destinations(
        &root,
        manifest,
        manifest_path,
        &scene,
        &destination,
        &metadata_path,
        &copies,
    )?;
    // The standard GLB container borrows the exporter's BIN. Metadata annotation
    // neither decodes nor duplicates the geometry buffer.
    let parsed = gltf::binary::Glb::from_slice(&glb_bytes).map_err(unsupported)?;
    let annotated = gltf::binary::Glb {
        header: parsed.header,
        json: Cow::Owned(serde_json::to_vec(&document).map_err(unsupported)?),
        bin: parsed.bin,
    };
    let temporary = temporary_path(&destination);
    fs::create_dir_all(directory)?;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    annotated
        .to_writer(&mut output)
        .map_err(|error| ExportError::Io(std::io::Error::other(error.to_string())))?;
    output.sync_all()?;
    drop(output);
    let (content_hash, byte_length) = hash_file(&temporary).ok_or_else(|| {
        ExportError::Io(std::io::Error::other("cannot hash completed GLB export"))
    })?;
    let receipt = ExportReceipt {
        schema_version: 1,
        target: "glb",
        scene_mode: mode,
        output: PayloadReceipt {
            path: filename.to_owned(),
            content_hash,
            byte_length,
        },
        assets,
        resources,
    };
    let metadata = serde_json::to_vec_pretty(&json!({
        "scene": source_document,
        "export_receipt": receipt,
    }))
    .map_err(unsupported)?;
    for (path, bytes) in copies {
        write_export_file(&path, bytes)?;
    }
    fs::rename(temporary, &destination)?;
    // Receipt is published last. Partial replacement fails its output/resource
    // hashes instead of declaring a valid bundle.
    write_export_file(&metadata_path, &metadata)?;
    Ok(receipt)
}

fn verified_bytes(
    root: &Path,
    path: &ProjectPath,
    expected: &ContentHash,
) -> Result<Vec<u8>, ExportError> {
    let resolved = path.resolve(root).canonicalize().map_err(|error| {
        ExportError::Incomplete(format!("missing resource {}: {error}", path.as_str()))
    })?;
    if !resolved.starts_with(root) {
        return Err(ExportError::Incomplete(format!(
            "resource {} escapes the verified project root",
            path.as_str()
        )));
    }
    let bytes = fs::read(&resolved).map_err(|error| {
        ExportError::Incomplete(format!("cannot read resource {}: {error}", path.as_str()))
    })?;
    if ContentHash::of_bytes(&bytes) != *expected {
        return Err(ExportError::Incomplete(format!(
            "resource hash mismatch: {}",
            path.as_str()
        )));
    }
    Ok(bytes)
}

fn select_assets(
    scene: &AssembledScene,
    mode: SceneMode,
) -> Result<BTreeMap<String, &'static str>, ExportError> {
    let mut selected = BTreeMap::new();
    for asset in &scene.assets {
        let mesh = asset
            .visual
            .as_ref()
            .and_then(|visual| visual.mesh.as_ref());
        let include = match (mode, mesh) {
            (SceneMode::Completed, Some(_)) => true,
            (SceneMode::Observed, Some(mesh)) => {
                let resource = scene
                    .resources
                    .iter()
                    .find(|resource| resource.id == mesh.mesh)
                    .ok_or_else(|| unsupported("canonical mesh resource is absent"))?;
                let observed = resource
                    .provenance
                    .iter()
                    .any(|origin| origin.is_observed_geometry());
                if observed
                    && !resource
                        .provenance
                        .iter()
                        .all(|origin| origin.is_observed_geometry())
                {
                    return Err(unsupported(
                        "observed export cannot separate mixed geometry provenance in one resource",
                    ));
                }
                observed
            }
            _ => false,
        };
        if asset
            .visual
            .as_ref()
            .is_some_and(|visual| visual.splat.is_some())
        {
            return Err(unsupported("splat companion export is not supported; mesh geometry cannot silently replace its companion"));
        }
        if include {
            selected.insert(asset.id.clone(), "visual");
        } else if mode == SceneMode::Completed && asset.collision.is_some() {
            selected.insert(asset.id.clone(), "collision");
        }
    }
    let initial = selected.keys().cloned().collect::<Vec<_>>();
    for id in initial {
        let mut asset = scene
            .assets
            .iter()
            .find(|asset| asset.id == id)
            .ok_or_else(|| unsupported("selected canonical asset is absent"))?;
        while let Some(parent) = &asset.parent {
            selected.entry(parent.clone()).or_insert("ancestor");
            asset = scene
                .assets
                .iter()
                .find(|asset| &asset.id == parent)
                .ok_or_else(|| unsupported("canonical parent is absent"))?;
        }
    }
    if !selected.values().any(|selection| *selection == "visual") {
        return Err(unsupported(
            "selected scene contains no supported mesh visuals",
        ));
    }
    Ok(selected)
}

type MaterialResources = (Vec<Material>, Vec<EncodedImage>, Vec<Texture>);

fn canonical_materials(
    scene: &AssembledScene,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<MaterialResources, ExportError> {
    let mut materials = Vec::new();
    let mut images = Vec::new();
    let mut textures = Vec::new();
    let mut texture_indices = BTreeMap::new();
    for material in &scene.materials {
        let [r, g, b, a] = material.base_color;
        let mut value = Material::pbr_metallic_roughness(
            Some(material.id.clone()),
            BaseColorFactor::new(r, g, b, a).map_err(unsupported)?,
            material.metallic,
            material.roughness,
            false,
        )
        .map_err(unsupported)?;
        if let Some(id) = &material.base_color_texture {
            let index = if let Some(&index) = texture_indices.get(id) {
                index
            } else {
                let bytes = payloads
                    .get(id)
                    .ok_or_else(|| unsupported("canonical texture is absent"))?;
                let mime = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
                    "image/png"
                } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
                    "image/jpeg"
                } else {
                    return Err(unsupported(
                        "canonical base-color texture requires encoded PNG or JPEG",
                    ));
                };
                let index = textures.len();
                images.push(
                    EncodedImage::new(Some(id.clone()), mime.into(), bytes.clone())
                        .map_err(unsupported)?,
                );
                textures.push(Texture::new(Some(id.clone()), images.len() - 1, None));
                texture_indices.insert(id.clone(), index);
                index
            };
            value = value.with_base_color_texture(MaterialTextureBinding::new(index, 0));
        }
        materials.push(value);
    }
    Ok((materials, images, textures))
}

fn adapt_scene(
    scene: &AssembledScene,
    payloads: &BTreeMap<String, Vec<u8>>,
    selected: &BTreeMap<String, &'static str>,
) -> Result<SceneSnapshot, ExportError> {
    let (mut materials, mut images, mut textures) = canonical_materials(scene, payloads)?;
    let mut samplers = Vec::new();
    let mut groups = Vec::new();
    let mut mesh_resources = BTreeMap::new();
    let mut bindings = BTreeMap::<String, Option<String>>::new();
    for asset in &scene.assets {
        if selected.get(&asset.id) != Some(&"visual") {
            continue;
        }
        let mesh = asset
            .visual
            .as_ref()
            .and_then(|visual| visual.mesh.as_ref())
            .ok_or_else(|| unsupported("selected asset has no mesh"))?;
        if let Some(previous) = bindings.insert(mesh.mesh.clone(), mesh.material.clone()) {
            if previous != mesh.material {
                return Err(unsupported("one mesh resource instanced with different canonical material overrides is unsupported"));
            }
        }
    }
    for (id, override_id) in bindings {
        let bytes = &payloads[&id];
        supported_source_layout(bytes)?;
        let parts = three_d_formats::load_gltf(bytes)
            .map_err(unsupported)?
            .into_parts();
        if parts.meshes.len() != 1 {
            return Err(unsupported(
                "canonical mesh resource must identify one local mesh group",
            ));
        }
        let material_offset = materials.len();
        let texture_offset = textures.len();
        let image_offset = images.len();
        let sampler_offset = samplers.len();
        for (index, material) in parts.materials.iter().enumerate() {
            materials.push(remap_material(
                material,
                format!("{id}::material-{index}"),
                texture_offset,
            )?);
        }
        images.extend(parts.images);
        samplers.extend(parts.samplers);
        textures.extend(parts.textures.into_iter().map(|texture| {
            Texture::new(
                texture.name().map(str::to_owned),
                texture.image() + image_offset,
                texture.sampler().map(|index| index + sampler_offset),
            )
        }));
        let override_index = override_id
            .as_ref()
            .map(|id| {
                scene
                    .materials
                    .iter()
                    .position(|material| &material.id == id)
                    .ok_or_else(|| unsupported("canonical material override is absent"))
            })
            .transpose()?;
        let mut primitives = Vec::new();
        for group in parts.meshes {
            let (_, source) = group.into_parts();
            for primitive in source {
                let (mesh, material, skin) = primitive.into_parts();
                if skin.is_some() {
                    return Err(unsupported(
                        "animated/skinned canonical scene export is unsupported",
                    ));
                }
                primitives.push(MeshPrimitive::new(
                    mesh,
                    override_index.or(material.map(|index| index + material_offset)),
                ));
            }
        }
        mesh_resources.insert(id.clone(), groups.len());
        groups.push(AssetMesh::new(Some(id), primitives).map_err(unsupported)?);
    }
    let asset = Asset::with_resources(groups, materials, images, samplers, textures)
        .map_err(unsupported)?;
    let mut nodes = Vec::new();
    let mut node_indices = BTreeMap::new();
    let mut remaining = scene
        .assets
        .iter()
        .filter(|asset| selected.contains_key(&asset.id))
        .collect::<Vec<_>>();
    while !remaining.is_empty() {
        let Some(index) = remaining.iter().position(|asset| {
            asset
                .parent
                .as_ref()
                .is_none_or(|parent| node_indices.contains_key(parent))
        }) else {
            return Err(unsupported("canonical hierarchy cannot be ordered"));
        };
        let asset = remaining.remove(index);
        let parent = asset.parent.as_ref().map(|parent| node_indices[parent]);
        let mesh = if selected[&asset.id] == "visual" {
            asset
                .visual
                .as_ref()
                .and_then(|visual| visual.mesh.as_ref())
                .map(|mesh| mesh_resources[&mesh.mesh])
        } else {
            None
        };
        let transform = asset.transform;
        let [x, y, z, w] = transform.rotation;
        let [tx, ty, tz] = transform.translation;
        let [sx, sy, sz] = transform.scale;
        node_indices.insert(asset.id.clone(), nodes.len());
        nodes.push(SceneNode::new(
            &asset.id,
            parent,
            mesh,
            Transform {
                translation: Vec3::new(tx, ty, tz),
                rotation: Quat::new(x, y, z, w),
                scale: Vec3::new(sx, sy, sz),
            },
        ));
    }
    SceneSnapshot::from_asset(asset, nodes).map_err(unsupported)
}

fn remap_material(
    material: &Material,
    name: String,
    offset: usize,
) -> Result<Material, ExportError> {
    let mut value = Material::pbr_metallic_roughness(
        Some(name),
        material.base_color_factor(),
        material.metallic_factor(),
        material.roughness_factor(),
        material.double_sided(),
    )
    .map_err(unsupported)?;
    if let Some(binding) = material.base_color_texture() {
        value = value.with_base_color_texture(MaterialTextureBinding::new(
            binding.texture() + offset,
            binding.tex_coord(),
        ));
    }
    if let Some(binding) = material.metallic_roughness_texture() {
        value = value.with_metallic_roughness_texture(MaterialTextureBinding::new(
            binding.texture() + offset,
            binding.tex_coord(),
        ));
    }
    if let Some(binding) = material.normal_texture() {
        value = value.with_normal_texture(
            NormalTextureBinding::new(
                binding.texture() + offset,
                binding.tex_coord(),
                binding.scale(),
            )
            .map_err(unsupported)?,
        );
    }
    Ok(value)
}

fn supported_source_layout(bytes: &[u8]) -> Result<(), ExportError> {
    // The established mesh loader owns attributes/materials, but does not lower
    // source node placement. Reject unsupported layouts rather than losing it.
    let source = gltf::Gltf::from_slice(bytes).map_err(unsupported)?;
    if source.animations().next().is_some() || source.skins().next().is_some() {
        return Err(unsupported("animated/skinned source layout is unsupported"));
    }
    let mesh_nodes = source
        .nodes()
        .filter(|node| node.mesh().is_some())
        .collect::<Vec<_>>();
    if mesh_nodes.len() != 1 {
        return Err(unsupported(
            "source resource must contain one unambiguous local mesh node",
        ));
    }
    let default_scene = source.default_scene().or_else(|| {
        let mut scenes = source.scenes();
        let first = scenes.next()?;
        scenes.next().is_none().then_some(first)
    });
    let default_scene = default_scene
        .ok_or_else(|| unsupported("source layout requires an unambiguous default scene"))?;
    let mut pending = default_scene.nodes().collect::<Vec<_>>();
    let mut reachable = BTreeSet::new();
    while let Some(node) = pending.pop() {
        if reachable.insert(node.index()) {
            pending.extend(node.children());
        }
    }
    if !reachable.contains(&mesh_nodes[0].index()) {
        return Err(unsupported(
            "source mesh is not part of its default scene layout",
        ));
    }
    for node in source.nodes() {
        let (translation, rotation, scale) = node.transform().decomposed();
        if translation != [0.0; 3] || rotation != [0.0, 0.0, 0.0, 1.0] || scale != [1.0; 3] {
            return Err(unsupported("non-identity source node placement requires explicit source-to-canonical adaptation"));
        }
    }
    Ok(())
}

fn annotate_nodes(
    scene: &AssembledScene,
    selected: &BTreeMap<String, &'static str>,
    document: &mut Value,
) -> Result<BTreeMap<String, AssetReceipt>, ExportError> {
    let mut assets = BTreeMap::new();
    for (index, node) in document["nodes"]
        .as_array_mut()
        .ok_or_else(|| unsupported("shared exporter omitted nodes"))?
        .iter_mut()
        .enumerate()
    {
        let id = node["name"]
            .as_str()
            .ok_or_else(|| unsupported("shared exporter omitted stable node name"))?
            .to_owned();
        let asset: &SceneAsset = scene
            .assets
            .iter()
            .find(|asset| asset.id == id)
            .ok_or_else(|| unsupported("shared exporter emitted an unknown canonical node"))?;
        node["extras"] = json!({"video_to_3d_id":id,"role":asset.role,"labels":asset.labels,
            "provenance":scene.asset_provenance(&id).map(|provenance| json!({"visual":provenance.visual,"collision":provenance.collision}))});
        assets.insert(
            id.clone(),
            AssetReceipt {
                node_index: index,
                selection: selected[&id],
            },
        );
    }
    Ok(assets)
}

fn annotate_material_identity(source: &Value, document: &mut Value) -> Result<(), ExportError> {
    let Some(original_materials) = source.get("materials") else {
        // The validated canonical model defaults an omitted table to empty.
        return Ok(());
    };
    let original_materials = original_materials
        .as_array()
        .ok_or_else(|| unsupported("canonical material table is absent"))?;
    if original_materials.is_empty() {
        return Ok(());
    }
    let materials = document["materials"]
        .as_array_mut()
        .ok_or_else(|| unsupported("shared exporter omitted material table"))?;
    for original in original_materials {
        let material = materials
            .iter_mut()
            .find(|material| material["name"] == original["id"])
            .ok_or_else(|| unsupported("shared exporter omitted canonical material identity"))?;
        // These factors crossed the shared typed material validator unchanged.
        // Preserve their original JSON representation, not a new material model.
        for (target, source) in [
            ("baseColorFactor", "base_color"),
            ("metallicFactor", "metallic"),
            ("roughnessFactor", "roughness"),
        ] {
            material["pbrMetallicRoughness"][target] = original[source].clone();
        }
        material["extras"] = json!({
            "video_to_3d_material_id":original["id"],
            "provenance":original["provenance"],
        });
    }
    Ok(())
}

fn folded(path: &Path) -> PathBuf {
    path.components()
        .map(|component| component.as_os_str().to_ascii_lowercase())
        .collect()
}
fn aliases(a: &Path, b: &Path) -> bool {
    let a = folded(a);
    let b = folded(b);
    a.starts_with(&b) || b.starts_with(&a)
}
fn temporary_path(path: &Path) -> PathBuf {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    PathBuf::from(temporary)
}
fn preflight_destinations(
    root: &Path,
    manifest: &SceneProjectManifest,
    manifest_path: &Path,
    scene: &AssembledScene,
    destination: &Path,
    metadata: &Path,
    copies: &[(PathBuf, &[u8])],
) -> Result<(), ExportError> {
    let mut protected = vec![manifest_path.canonicalize()?, root.join(STATE_DIRECTORY)];
    let directory = destination
        .parent()
        .ok_or_else(|| ExportError::Invalid("GLB destination has no directory".into()))?;
    for path in manifest
        .artifacts
        .iter()
        .map(|artifact| &artifact.path)
        .chain(scene.resources.iter().map(|resource| &resource.path))
    {
        let path = path.resolve(root);
        if let Some(parent) = path.parent().filter(|parent| *parent != root) {
            if aliases(directory, parent) {
                return Err(ExportError::Invalid(
                    "export directory aliases protected resource or artifact subtree".into(),
                ));
            }
        }
    }
    for path in manifest
        .inputs
        .iter()
        .map(|input| &input.path)
        .chain(manifest.artifacts.iter().map(|artifact| &artifact.path))
        .chain(scene.resources.iter().map(|resource| &resource.path))
    {
        protected.push(path.resolve(root));
        if let Ok(actual) = path.resolve(root).canonicalize() {
            protected.push(actual);
        }
    }
    if protected
        .iter()
        .any(|source| folded(source).starts_with(folded(directory)))
    {
        return Err(ExportError::Invalid(
            "export directory contains protected project source or metadata".into(),
        ));
    }
    // Other exports own their concrete output subtrees, not a shared parent.
    for export in &manifest.exports {
        let path = export.path.resolve(root);
        if path != destination {
            if copies
                .iter()
                .filter_map(|(copy, _)| copy.parent())
                .any(|resources| aliases(&path, resources))
            {
                return Err(ExportError::Invalid(
                    "export resources subtree aliases another declared export".into(),
                ));
            }
            protected.push(path);
        }
    }
    let outputs = std::iter::once(destination.to_path_buf())
        .chain(std::iter::once(metadata.to_path_buf()))
        .chain(copies.iter().map(|(path, _)| path.clone()))
        .collect::<Vec<_>>();
    let mut checked: Vec<PathBuf> = Vec::new();
    for output in &outputs {
        for path in [output.clone(), temporary_path(output)] {
            if protected.iter().any(|source| aliases(&path, source))
                || checked.iter().any(|other| aliases(&path, other))
            {
                return Err(ExportError::Invalid(
                    "export output aliases protected source, metadata or another output".into(),
                ));
            }
            let relative = path.strip_prefix(root).map_err(|_| {
                ExportError::Invalid("export destination escapes project root".into())
            })?;
            let mut current = root.to_path_buf();
            for component in relative.components() {
                current.push(component);
                match fs::symlink_metadata(&current) {
                    Ok(metadata) if metadata.file_type().is_symlink() => {
                        return Err(ExportError::Invalid(
                            "export destination contains a symlink".into(),
                        ))
                    }
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
            checked.push(path);
        }
    }
    Ok(())
}
fn write_export_file(path: &Path, bytes: &[u8]) -> Result<(), ExportError> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = temporary_path(path);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(temporary, path)?;
    Ok(())
}
