//! One selectable mesh per reconstructed identity, in a regular inspection grid.
//! Original local vertex buffers stay intact; the node transform is solely a
//! presentation transform and makes no claim about world pose or metric scale.
use super::*;
use gltf::binary::Header;

struct Item<'a> {
    object: &'a str,
    label: &'a str,
    mesh: &'a GeneratedMesh,
    hull: bool,
    validation: Option<&'a PlacementValidation>,
    provenance: Value,
}

impl Item<'_> {
    fn rank(&self) -> (u8, bool) {
        let quality = match self.validation.map(|v| v.verdict) {
            Some(PlacementVerdict::Accepted) => 3,
            Some(PlacementVerdict::Approximate) => 2,
            None => 1,
            Some(PlacementVerdict::Contradicted) => 0,
        };
        // Prefer genuinely constrained multi-view shapes. When both candidates
        // fail placement, the image-conditioned model remains a separate local
        // shape guess; a contradicted world-space hull adds no pose advantage.
        (quality, self.hull == (quality >= 2))
    }
}

/// A presentation-only unit cell with the object's bottom at grid height zero.
fn display_matrix(item: &Item, bytes: &[u8], x: f64, z: f64) -> Result<[f64; 16], String> {
    let parsed = Glb::from_slice(bytes).map_err(|e| e.to_string())?;
    let data: Value = serde_json::from_slice(&parsed.json).map_err(|e| e.to_string())?;
    let p = data["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
        .as_u64()
        .ok_or("missing positions")? as usize;
    let a = &data["accessors"][p];
    let lo: [f64; 3] = serde_json::from_value(a["min"].clone()).map_err(|e| e.to_string())?;
    let hi: [f64; 3] = serde_json::from_value(a["max"].clone()).map_err(|e| e.to_string())?;
    let extent = [0, 1, 2].map(|k| hi[k] - lo[k]);
    let size = extent.into_iter().fold(0.0, f64::max);
    if size <= 0. || !size.is_finite() || extent.iter().any(|e| *e < 0.) {
        return Err("candidate has invalid position bounds".into());
    }
    let s = 1. / size;
    let c = [0, 1, 2].map(|k| (lo[k] + hi[k]) / 2.);
    Ok(if item.hull {
        // Hull vertices are in the world camera convention, +y down, +z forward.
        [
            s,
            0.,
            0.,
            0.,
            0.,
            -s,
            0.,
            0.,
            0.,
            0.,
            -s,
            0.,
            x - c[0] * s,
            hi[1] * s,
            z + c[2] * s,
            1.,
        ]
    } else {
        // TripoSR canonical +z up, preview conversion (x,y,z) -> (-y,z,-x).
        [
            0.,
            0.,
            -s,
            0.,
            -s,
            0.,
            0.,
            0.,
            0.,
            s,
            0.,
            0.,
            x + c[1] * s,
            -lo[2] * s,
            z + c[0] * s,
            1.,
        ]
    })
}

pub fn export_object_library(
    root: &Path,
    generation: &ObjectGenerationArtifact,
    placement: &ObjectPlacementArtifact,
    hulls: &MultiViewObjectReport,
) -> Result<Vec<u8>, String> {
    placement.validate()?;
    generation.validate()?;
    if hulls.schema_version != 1
        || placement.object_generation != ContentHash::of_bytes(generation.to_json().as_bytes())
    {
        return Err("object library inputs belong to different reconstruction results".into());
    }
    let placements: BTreeMap<_, _> = placement
        .placements
        .iter()
        .map(|p| (p.object.as_str(), p))
        .collect();
    let invalid: BTreeSet<_> = placement
        .unplaced
        .iter()
        .filter(|p| p.reason.starts_with("invalid generated mesh:"))
        .map(|p| p.object.as_str())
        .collect();
    let mut items = BTreeMap::new();
    for o in &generation.objects {
        let Some(mesh) = &o.output else { continue };
        if invalid.contains(o.request.object.as_str()) {
            continue;
        }
        let r = &o.request;
        let item = Item {
            object: &r.object,
            label: &r.label,
            mesh,
            hull: false,
            validation: placements.get(r.object.as_str()).map(|p| &p.validation),
            provenance: json!({"method":"triposr","geometry_origin":"generative_completion","source_frame":r.frame_index,"source_observation":r.observation,"mask_hash":r.mask_hash,"generation_request":r.request_id}),
        };
        items.insert(item.object, item);
    }
    for h in &hulls.objects {
        let Some(mesh) = &h.output else { continue };
        let accepted = h
            .validation
            .as_ref()
            .is_some_and(|v| v.verdict == PlacementVerdict::Accepted);
        let item = Item {
            object: &h.object,
            label: &h.label,
            mesh,
            hull: true,
            validation: h.validation.as_ref(),
            provenance: json!({"method":"multi_view_silhouette","geometry_origin":if accepted {"revalidated_completion"}else{"generative_completion"},"source_frames":h.views,
                "holdout_frame":h.holdout_frame,"holdout_mask_iou":h.holdout_mask_iou,"reason":h.reason}),
        };
        let prefer_generated = placements
            .get(item.object)
            .and_then(|p| compare_candidates(h, p))
            .is_some_and(|c| c.prefers_generated());
        if !prefer_generated
            && items
                .get(item.object)
                .is_none_or(|prior| item.rank() > prior.rank())
        {
            items.insert(item.object, item);
        }
    }
    if items.is_empty() {
        return Err("no object has a candidate mesh to inspect".into());
    }
    let mut doc = json!({"asset":{"version":"2.0","generator":"video-to-3d","extras":{"video_to_3d":{
        "schema":"video-to-3d/object-library","schema_version":1,"layout":"normalized_inspection_grid","world_placement":false,"semantic_inventory":hulls.semantic_inventory,
        "object_identity":hulls.object_identity,"object_count":items.len(),"provider":generation.provider}}},
        "scene":0,"scenes":[{"name":"Object library (inspection grid)","nodes":[]}],"nodes":[],"meshes":[],"materials":[],"accessors":[],"bufferViews":[]});
    let mut bin = Vec::new();
    let columns = (items.len() as f64).sqrt().ceil() as usize;
    for (index, (_, item)) in items.into_iter().enumerate() {
        let bytes = read_verified(root, &item.mesh.path, &item.mesh.content_hash)?;
        let matrix = display_matrix(
            &item,
            &bytes,
            (index % columns) as f64 * 1.4,
            (index / columns) as f64 * 1.4,
        )?;
        let mut meta = item.provenance;
        meta["role"] = json!("object_library_candidate");
        meta["object_id"] = json!(item.object);
        meta["label"] = json!(item.label);
        meta["placement_verdict"] = item
            .validation
            .map_or_else(|| json!("unplaced"), |v| json!(v.verdict));
        meta["mesh_hash"] = json!(item.mesh.content_hash);
        meta["inspection_layout"] = json!(true);
        let node = json!({"name":format!("{} {}",item.label,item.object),"matrix":matrix,"extras":{"video_to_3d":meta}});
        add_node(&mut doc, &mut bin, &bytes, 0, node)?;
    }
    doc["buffers"] = json!([{"byteLength":bin.len()}]);
    Glb {
        header: Header {
            magic: *b"glTF",
            version: 2,
            length: 0,
        },
        json: Cow::Owned(serde_json::to_vec(&doc).map_err(|e| e.to_string())?),
        bin: Some(Cow::Owned(bin)),
    }
    .to_vec()
    .map_err(|e| e.to_string())
}
