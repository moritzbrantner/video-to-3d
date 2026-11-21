//! Native handoff between object generation, evidence-based placement, and the
//! browser's textured GLB. The collateral scene and all observed accessors
//! retain their original bytes. Each accepted completion gets its own node.
use crate::object_generation::{GeneratedMesh, ObjectGenerationArtifact};
use crate::object_identity::{FrameCamera, ObjectIdentityArtifact};
use crate::object_placement::{place_objects_with_faces, ObjectPlacement, ObjectPlacementArtifact};
use crate::object_references::ObjectReferencesArtifact;
use crate::placement_validation::{PlacementValidation, PlacementVerdict};
use crate::scene_artifacts::read_verified;
use crate::scene_export::supported_source_layout;
use crate::scene_project::ContentHash;
use crate::semantic_inventory::SemanticInventoryArtifact;
use crate::{
    EvidenceCamera, EvidenceOrigin, EvidenceRange, EvidenceScale, MeshTriangle, Point3,
    ReconstructionEvidenceView, ReconstructionProviderClass, ReconstructionProviderDescriptor,
    SurfaceEvidenceRegion,
};
#[path = "generated_object_library.rs"]
mod library;
use gltf::{binary::Glb, mesh::Mode};
pub use library::export_object_library;
use serde_json::{json, Value};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

/// Adapt the local candidate into the shared provider evidence boundary before
/// its positions participate in the scene's placement calculation. TripoSR
/// supplies vertices in its canonical (+x toward camera, +z up) object frame.
struct CandidateGeometry {
    positions: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
}

fn candidate_geometry(
    bytes: &[u8],
    receipt: &GeneratedMesh,
    generation: &ObjectGenerationArtifact,
) -> Result<CandidateGeometry, String> {
    supported_source_layout(bytes).map_err(|e| e.to_string())?;
    let model = gltf::Gltf::from_slice(bytes).map_err(|e| e.to_string())?;
    let mut meshes = model.meshes();
    let mesh = meshes.next().ok_or("candidate has no mesh")?;
    let mut primitives = mesh.primitives();
    let prim = primitives.next().ok_or("candidate has no primitive")?;
    if meshes.next().is_some() || primitives.next().is_some() || prim.mode() != Mode::Triangles {
        return Err("object generator must provide one indexed triangle mesh".into());
    }
    let reader = prim.reader(|b| (b.index() == 0).then_some(model.blob.as_deref()).flatten());
    let positions = reader
        .read_positions()
        .ok_or("missing candidate positions")?;
    let points: Vec<_> = positions
        .map(|[x, y, z]| Point3 {
            x,
            y,
            z,
            r: 0,
            g: 0,
            b: 0,
            confidence: 0.,
        })
        .collect();
    let mut indices = reader
        .read_indices()
        .ok_or("missing candidate triangle indices")?
        .into_u32();
    let mut triangles = Vec::with_capacity(indices.len() / 3);
    while let Some(a) = indices.next() {
        let b = indices.next().ok_or("incomplete candidate triangle")?;
        let c = indices.next().ok_or("incomplete candidate triangle")?;
        triangles.push(MeshTriangle {
            a: a as usize,
            b: b as usize,
            c: c as usize,
            confidence: 0.,
        });
    }
    if points.len() as u64 != receipt.vertices || triangles.len() as u64 != receipt.triangles {
        return Err("candidate mesh counts differ from its generation receipt".into());
    }
    let view = ReconstructionEvidenceView::new(
        ReconstructionProviderDescriptor::new(
            &generation.provider.id,
            ReconstructionProviderClass::GenerativeCompletion,
            Some(generation.provider.revision.clone()),
        )?,
        EvidenceScale::ProviderLocal,
        vec![],
        vec![SurfaceEvidenceRegion {
            origin: EvidenceOrigin::GenerativeCompletion,
            reference_frame: None,
            source_frames: vec![],
            points: EvidenceRange::new(0, points.len()),
        }],
        &points,
        &triangles,
    )?;
    Ok(CandidateGeometry {
        positions: view
            .points
            .iter()
            .map(|p| [p.x, p.y, p.z].map(f64::from))
            .collect(),
        faces: view
            .triangles
            .iter()
            .map(|t| [t.a as u32, t.b as u32, t.c as u32])
            .collect(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn place_generated_meshes(
    inventory: &SemanticInventoryArtifact,
    identities: &ObjectIdentityArtifact,
    references: &ObjectReferencesArtifact,
    generation: &ObjectGenerationArtifact,
    root: &Path,
    cameras: &[FrameCamera],
    points: &[[f64; 3]],
    frames: &[crate::multi_view_objects::FramePixels],
    mut is_canceled: impl FnMut() -> bool,
) -> Result<ObjectPlacementArtifact, String> {
    let mut vertices = BTreeMap::new();
    let mut faces = BTreeMap::new();
    let mut rejected = BTreeMap::new();
    for generated in &generation.objects {
        if let Some(mesh) = &generated.output {
            let data = crate::scene_artifacts::read_verified_with_cancel(
                root,
                &mesh.path,
                &mesh.content_hash,
                None,
                &mut is_canceled,
            )?;
            match candidate_geometry(&data, mesh, generation) {
                Ok(shape) => {
                    faces.insert(generated.request.request_id.clone(), shape.faces);
                    vertices.insert(generated.request.request_id.clone(), shape.positions);
                }
                Err(error) => {
                    rejected.insert(
                        generated.request.object.clone(),
                        format!("invalid generated mesh: {error}"),
                    );
                }
            }
        }
    }
    let mut placement = place_objects_with_faces(
        inventory,
        identities,
        references,
        generation,
        ContentHash::of_bytes(generation.to_json().as_bytes()),
        &vertices,
        Some(&faces),
        cameras,
        points,
        frames,
    );
    for row in &mut placement.unplaced {
        if let Some(reason) = rejected.remove(&row.object) {
            row.reason = reason;
        }
    }
    placement.unplaced.extend(
        rejected
            .into_iter()
            .map(|(object, reason)| crate::object_placement::UnplacedObject { object, reason }),
    );
    placement.validate()?;
    Ok(placement)
}

fn list_mut<'a>(document: &'a mut Value, name: &str) -> Result<&'a mut Vec<Value>, String> {
    document[name]
        .as_array_mut()
        .ok_or_else(|| format!("missing GLB {name} array"))
}

fn shift(index: &mut Value, offset: usize) -> Result<(), String> {
    let local = index.as_u64().ok_or("invalid GLB resource index")?;
    *index = json!(local
        .checked_add(offset as u64)
        .ok_or("GLB index out of range")?);
    Ok(())
}

fn append_mesh(base: &mut Value, bin: &mut Vec<u8>, bytes: &[u8]) -> Result<usize, String> {
    let parsed = Glb::from_slice(bytes).map_err(|e| e.to_string())?;
    let mut source: Value = serde_json::from_slice(&parsed.json).map_err(|e| e.to_string())?;
    if source["buffers"].as_array().map(Vec::len) != Some(1)
        || !source["buffers"][0]["uri"].is_null()
        || [
            "textures",
            "images",
            "skins",
            "animations",
            "extensionsRequired",
        ]
        .iter()
        .any(|k| source[k].as_array().is_some_and(|v| !v.is_empty()))
    {
        return Err(
            "unsupported generated mesh resources; expected a self-contained vertex-colored mesh"
                .into(),
        );
    }
    let local_bin = parsed.bin.ok_or("candidate has no binary data")?;
    let view_base = list_mut(base, "bufferViews")?.len();
    let accessor_base = list_mut(base, "accessors")?.len();
    let material_base = list_mut(base, "materials")?.len();
    let aligned = bin.len().next_multiple_of(4);
    bin.resize(aligned, 0);
    bin.extend_from_slice(&local_bin);
    for mut view in list_mut(&mut source, "bufferViews")?.drain(..) {
        if view["buffer"] != 0 {
            return Err("candidate view references an external buffer".into());
        }
        view["byteOffset"] = json!(aligned as u64 + view["byteOffset"].as_u64().unwrap_or(0));
        list_mut(base, "bufferViews")?.push(view);
    }
    for mut accessor in list_mut(&mut source, "accessors")?.drain(..) {
        if !accessor["sparse"].is_null() {
            return Err("sparse generated mesh accessors are unsupported".into());
        }
        shift(&mut accessor["bufferView"], view_base)?;
        list_mut(base, "accessors")?.push(accessor);
    }
    if let Some(materials) = source["materials"].as_array_mut() {
        list_mut(base, "materials")?.append(materials);
    }
    let mut meshes = list_mut(&mut source, "meshes")?.drain(..);
    let mut mesh = meshes.next().ok_or("missing generated mesh")?;
    let prim = &mut mesh["primitives"][0];
    for index in prim["attributes"]
        .as_object_mut()
        .ok_or("missing attributes")?
        .values_mut()
    {
        shift(index, accessor_base)?;
    }
    shift(&mut prim["indices"], accessor_base)?;
    if let Some(index) = prim.get_mut("material") {
        shift(index, material_base)?;
    }
    let index = list_mut(base, "meshes")?.len();
    list_mut(base, "meshes")?.push(mesh);
    Ok(index)
}

/// Column-major glTF matrix for the accepted generator-to-reconstruction
/// similarity followed by the proper CV-to-glTF rotation, diag(1,-1,-1).
fn gltf_transform(p: &ObjectPlacement) -> [f64; 16] {
    let mut matrix = [0.; 16];
    for (r, sign) in [1., -1., -1.].into_iter().enumerate() {
        for c in 0..3 {
            matrix[c * 4 + r] = sign * p.scale * p.rotation[r * 3 + c];
        }
        matrix[12 + r] = sign * p.translation[r];
    }
    matrix[15] = 1.;
    matrix
}

/// Selection metadata for silhouettes carved from registered views. Positions in
/// these GLBs are already in the reconstruction world before the GLB root transform.
#[derive(serde::Deserialize)]
pub struct MultiViewObjectReport {
    pub schema_version: u32,
    pub semantic_inventory: ContentHash,
    pub object_identity: ContentHash,
    pub objects: Vec<MultiViewCandidate>,
}
#[derive(serde::Deserialize)]
pub struct MultiViewCandidate {
    pub object: String,
    pub label: String,
    pub views: Vec<usize>,
    pub reason: String,
    pub holdout_frame: Option<usize>,
    pub holdout_mask_iou: Option<f64>,
    pub output: Option<GeneratedMesh>,
    pub validation: Option<PlacementValidation>,
}

/// Compare accepted candidates on the same source views, leaving each model's
/// independent validation camera out of the selection. Box overlap measures
/// projected extent, mask inclusion measures foreground precision, and the
/// depth/landmark residuals penalize disagreement with registered geometry.
#[derive(serde::Serialize)]
struct CandidateComparison {
    fitting_frames: Vec<usize>,
    silhouette_score: f64,
    generated_score: f64,
}

impl CandidateComparison {
    fn prefers_generated(&self) -> bool {
        self.generated_score > self.silhouette_score
    }
}

fn compare_candidates(
    hull: &MultiViewCandidate,
    placement: &ObjectPlacement,
) -> Option<CandidateComparison> {
    let silhouette = hull.validation.as_ref()?;
    if silhouette.verdict != PlacementVerdict::Accepted
        || placement.validation.verdict != PlacementVerdict::Accepted
    {
        return None;
    }
    let registration = placement.silhouette_registration.as_ref();
    let mut comparison = CandidateComparison {
        fitting_frames: Vec::new(),
        silhouette_score: 0.,
        generated_score: 0.,
    };
    let score = |r: &crate::placement_validation::ViewResidual| {
        Some(
            (r.inside_share? * r.box_iou?).sqrt() * (1. - r.depth_violation_share.unwrap_or(0.))
                / (1. + r.landmark_residual.unwrap_or(0.)),
        )
    };
    for view in &silhouette.views {
        let frame = view.frame_index;
        if Some(frame) == hull.holdout_frame
            || registration.is_some_and(|r| !r.fitting_frames.contains(&frame))
        {
            continue;
        }
        let Some(other) = placement
            .validation
            .views
            .iter()
            .find(|v| v.frame_index == frame)
        else {
            continue;
        };
        if let (Some(a), Some(b)) = (score(view), score(other)) {
            comparison.fitting_frames.push(frame);
            comparison.silhouette_score += a;
            comparison.generated_score += b;
        }
    }
    let n = comparison.fitting_frames.len();
    if n < 2 {
        return None;
    }
    comparison.generated_score /= n as f64;
    comparison.silhouette_score /= n as f64;
    Some(comparison)
}

fn add_node(
    document: &mut Value,
    bin: &mut Vec<u8>,
    bytes: &[u8],
    visual: usize,
    mut node: Value,
) -> Result<(), String> {
    node["mesh"] = json!(append_mesh(document, bin, bytes)?);
    let nodes = list_mut(document, "nodes")?;
    let index = nodes.len();
    nodes.push(node);
    list_mut(&mut document["scenes"][visual], "nodes")?.push(json!(index));
    Ok(())
}

pub fn append_validated_objects(
    base: &[u8],
    root: &Path,
    generation: &ObjectGenerationArtifact,
    placement: &ObjectPlacementArtifact,
    inventory_hash: &ContentHash,
    cameras: &[FrameCamera],
    silhouettes: &MultiViewObjectReport,
) -> Result<Vec<u8>, String> {
    placement.validate()?;
    generation.validate()?;
    if placement.object_generation != ContentHash::of_bytes(generation.to_json().as_bytes()) {
        return Err("placement belongs to another generation".into());
    }
    let parsed = Glb::from_slice(base).map_err(|e| e.to_string())?;
    let mut document: Value = serde_json::from_slice(&parsed.json).map_err(|e| e.to_string())?;
    let metadata = &document["asset"]["extras"]["video_to_3d"];
    // PatchMatch exports an observation-only surface before semantic processing.
    // It joins the inventory only after the shared camera gauge and object receipts agree.
    let observation_only = metadata["provider"]["id"] == "colmap-patch-match"
        && metadata["semantic_inventory"].is_null();
    if metadata["schema"] != crate::textured_glb::TEXTURED_SURFACE_SCHEMA
        || (!observation_only && metadata["semantic_inventory"] != inventory_hash.as_str())
        || !metadata["completion"].is_null()
    {
        return Err("expected the observed scene baked from the same object detections".into());
    }
    let observed: Vec<EvidenceCamera> =
        serde_json::from_value(metadata["accepted_camera_evidence"].clone())
            .map_err(|e| e.to_string())?;
    if observed.len() != cameras.len()
        || !observed.iter().all(|c| {
            cameras.iter().any(|r| {
                r.frame_index == c.frame_index
                    // The browser canonicalizes the seed pose to identity;
                    // COLMAP's independently validated pose keeps roundoff.
                    // Allow only exporter-level numerical differences.
                    && r.rotation.iter().zip(c.rotation).all(|(a, b)| (a - f64::from(b)).abs() <= 1e-6)
                    && r.translation.iter().zip(c.translation).all(|(a, b)| (a - f64::from(b)).abs() <= 1e-6)
            })
        })
    {
        return Err("object placements belong to a different camera reconstruction".into());
    }
    let mut bin = parsed
        .bin
        .ok_or("environment has no binary payload")?
        .into_owned();
    let visual = document["scene"].as_u64().ok_or("missing active scene")? as usize;
    let observed_scene = document["scenes"]
        .get(visual)
        .cloned()
        .ok_or("invalid visual scene index")?;
    let candidates: BTreeMap<_, _> = generation
        .objects
        .iter()
        .map(|o| (&o.request.request_id, o))
        .collect();
    if silhouettes.schema_version != 1 || &silhouettes.semantic_inventory != inventory_hash {
        return Err("multi-view report version or source inventory does not match".into());
    }
    let comparisons: BTreeMap<_, _> = silhouettes
        .objects
        .iter()
        .filter_map(|hull| {
            let p = placement
                .placements
                .iter()
                .find(|p| p.object == hull.object)?;
            Some((hull.object.as_str(), compare_candidates(hull, p)?))
        })
        .collect();
    let mut seen_hulls = BTreeSet::new();
    let mut completed = BTreeSet::new();
    for hull in &silhouettes.objects {
        let (Some(mesh), Some(validation)) = (&hull.output, &hull.validation) else {
            continue;
        };
        if validation.verdict != PlacementVerdict::Accepted {
            continue;
        }
        let views: BTreeSet<_> = hull.views.iter().copied().collect();
        if validation.needs_review
            || views.len() < 2
            || !views
                .iter()
                .all(|i| observed.iter().any(|c| c.frame_index == *i))
        {
            return Err("accepted silhouette has insufficient registered camera support".into());
        }
        if !seen_hulls.insert(&hull.object) {
            return Err("duplicate accepted silhouette identity".into());
        }
        let comparison = comparisons.get(hull.object.as_str());
        if comparison.is_some_and(CandidateComparison::prefers_generated) {
            continue;
        }
        completed.insert(&hull.object);
        let bytes = read_verified(root, &mesh.path, &mesh.content_hash)?;
        let node = json!({
            "name":format!("{} {} (silhouette)",hull.label,hull.object),"rotation":[1,0,0,0],
            "extras":{"video_to_3d":{"role":"completed_object_surface","geometry_origin":"revalidated_completion","method":"multi_view_silhouette",
                "object_id":hull.object,"label":hull.label,"source_frames":hull.views,
                "mesh_hash":mesh.content_hash,"validation":validation,"candidate_comparison":comparison,"reason":hull.reason,
                "holdout_frame":hull.holdout_frame,"holdout_mask_iou":hull.holdout_mask_iou}}
        });
        add_node(&mut document, &mut bin, &bytes, visual, node)?;
    }
    for p in placement
        .placements
        .iter()
        .filter(|p| p.validation.verdict == PlacementVerdict::Accepted)
    {
        if !completed.insert(&p.object) {
            continue;
        }
        let candidate = candidates
            .get(&p.generation_request)
            .ok_or("object generation request missing")?;
        let req = &candidate.request;
        if req.object != p.object || req.label != p.label {
            return Err("placed object differs from its generation request".into());
        }
        let mesh = candidate
            .output
            .as_ref()
            .ok_or("accepted placement has no generated mesh")?;
        let bytes = read_verified(root, &mesh.path, &mesh.content_hash)?;
        let node = json!({
            "name":format!("{} {} (completed)",p.label,p.object),"matrix":gltf_transform(p),
            "extras":{"video_to_3d":{"role":"completed_object_surface","geometry_origin":"generative_completion","method":"triposr","object_id":p.object,
                "label":p.label,"request":req.request_id,"source_frame":req.frame_index,"source_observation":req.observation,
                "mask_hash": req.mask_hash,"mesh_hash":mesh.content_hash,"validation":p.validation,"candidate_comparison":comparisons.get(p.object.as_str()),"contact":p.contact}}
        });
        add_node(&mut document, &mut bin, &bytes, visual, node)?;
    }
    if completed.is_empty() {
        return Err("no object has an accepted multi-view placement".into());
    }
    let replaced: BTreeSet<_> = list_mut(&mut document, "nodes")?
        .iter()
        .enumerate()
        .filter_map(|(i, n)| {
            let meta = &n["extras"]["video_to_3d"];
            (meta["role"] == "observed_object_surface"
                && meta["object_id"]
                    .as_str()
                    .is_some_and(|id| completed.iter().any(|kept| kept.as_str() == id)))
            .then_some(i as u64)
        })
        .collect();
    list_mut(&mut document["scenes"][visual], "nodes")?
        .retain(|n| !n.as_u64().is_some_and(|i| replaced.contains(&i)));
    // Keep source fragments reachable in the observed scene while the default
    // completed scene has just one selectable representation of each identity.
    document["scenes"][visual]["name"] = json!("completed");
    let mut observed_scene = observed_scene;
    observed_scene["name"] = json!("observed");
    list_mut(&mut document, "scenes")?.push(observed_scene);
    document["asset"]["extras"]["video_to_3d"]["semantic_inventory"] = json!(inventory_hash);
    document["asset"]["extras"]["video_to_3d"]["completion"] = json!({"accepted_objects":completed.len(),"provider":generation.provider,"generation":placement.object_generation});
    document["buffers"][0]["byteLength"] = json!(bin.len());
    let reassembled = Glb {
        header: parsed.header,
        json: Cow::Owned(serde_json::to_vec(&document).map_err(|e| e.to_string())?),
        bin: Some(Cow::Owned(bin)),
    };
    reassembled.to_vec().map_err(|e| e.to_string())
}

#[cfg(test)]
#[path = "generated_scene_tests.rs"]
mod tests;
