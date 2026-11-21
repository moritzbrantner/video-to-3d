use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use video_to_3d_core::object_generation::ObjectGenerationArtifact;
use video_to_3d_core::object_identity::{identify_objects, RegisteredObjectGeometry};
use video_to_3d_core::object_masks::check_object_masks;
use video_to_3d_core::object_references::package_object_references;
use video_to_3d_core::scene_artifacts::{
    generate_object_candidates, KeyframesArtifact, DEFAULT_GENERATION_SIDECAR,
    GENERATION_ENVIRONMENT,
};
use video_to_3d_core::scene_project::ContentHash;
use video_to_3d_core::scene_runner::CancellationToken;
use video_to_3d_core::semantic_inventory::SemanticInventoryArtifact;
use video_to_3d_core::EvidenceScale;

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))
}

fn save(root: &Path, name: &str, document: String) -> Result<ContentHash, String> {
    let path = root.join(name);
    fs::write(&path, &document).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(ContentHash::of_bytes(document.as_bytes()))
}

fn multiview(
    frames: &[video_to_3d_core::multi_view_objects::FramePixels],
    inventory: &SemanticInventoryArtifact,
    identities: &video_to_3d_core::object_identity::ObjectIdentityArtifact,
    geometry: &RegisteredObjectGeometry,
    output: &Path,
) -> Result<(), String> {
    use video_to_3d_core::multi_view_objects::{hull_glb, reconstruct_hulls_with_frames};
    let outcomes = reconstruct_hulls_with_frames(
        inventory,
        identities,
        &geometry.cameras,
        &geometry.points,
        frames,
    );
    let mut objects = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        let mut entry = json!({"object":outcome.object,"label":outcome.label,"reason":outcome.reason,
            "views":outcome.views,"holdout_frame":outcome.holdout_frame,"holdout_mask_iou":outcome.holdout_mask_iou});
        if let Some(mesh) = &outcome.mesh {
            let bytes = hull_glb(
                mesh,
                &outcome,
                frames,
                inventory,
                identities,
                &geometry.cameras,
            )?;
            let path = format!("multiview/{}.glb", outcome.object);
            fs::create_dir_all(output.join("multiview")).map_err(|e| e.to_string())?;
            fs::write(output.join(&path), &bytes).map_err(|e| e.to_string())?;
            entry["output"] = json!({"path":path,"content_hash":ContentHash::of_bytes(&bytes).as_str(),
                "vertices":mesh.vertices.len(),"triangles":mesh.triangles.len()});
            entry["validation"] = json!(mesh.validation);
        }
        objects.push(entry);
    }
    let report = json!({
        "schema_version": 1, "semantic_inventory": identities.semantic_inventory,
        "object_identity": ContentHash::of_bytes(identities.to_json().as_bytes()), "objects": objects,
    });
    let document = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    save(output, "multiview.json", document)?;
    Ok(())
}

pub(super) fn run(
    keyframes_path: &Path,
    inventory_path: &Path,
    geometry: &Path,
    output: &Path,
) -> Result<Value, String> {
    let keyframes = KeyframesArtifact::from_json(&read(keyframes_path)?)?;
    let inventory_document = read(inventory_path)?;
    let inventory = SemanticInventoryArtifact::from_json(&inventory_document)?;
    inventory.validate_against_keyframes(&keyframes)?;
    let inventory_hash = ContentHash::of_bytes(inventory_document.as_bytes());
    let geometry = RegisteredObjectGeometry::from_json(&read(geometry)?)?;
    let RegisteredObjectGeometry { cameras, points } = &geometry;
    if cameras.iter().any(|camera| {
        !keyframes
            .frames
            .iter()
            .any(|f| f.frame_index == camera.frame_index)
    }) {
        return Err("registered object camera has no corresponding keyframe".into());
    }
    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let identities = identify_objects(&inventory, inventory_hash.clone(), cameras, points);
    identities.validate()?;
    let identity_hash = save(output, "identity.json", identities.to_json())?;
    let root = keyframes_path.parent().unwrap_or_else(|| Path::new("."));
    let images = keyframes
        .frames
        .iter()
        .map(|f| keyframes.load_pixels(root, f.frame_index))
        .collect::<Result<Vec<_>, _>>()?;
    let frames = keyframes
        .frames
        .iter()
        .zip(&images)
        .map(|(f, bytes)| {
            (
                f.frame_index,
                f.width as usize,
                f.height as usize,
                bytes.as_slice(),
            )
        })
        .collect::<Vec<_>>();
    multiview(&frames, &inventory, &identities, &geometry, output)?;
    let masks = check_object_masks(
        &inventory,
        inventory_hash.clone(),
        &identities,
        identity_hash.clone(),
        cameras,
        points,
    );
    masks.validate()?;
    let masks_hash = save(output, "masks.json", masks.to_json())?;
    let references = package_object_references(
        &inventory,
        inventory_hash,
        &identities,
        identity_hash,
        &masks,
        masks_hash,
        cameras,
        points,
        EvidenceScale::ArbitraryMonocular,
    );
    references.validate()?;
    let refs_hash = save(output, "references.json", references.to_json())?;
    let sidecar = std::env::var_os(GENERATION_ENVIRONMENT)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_GENERATION_SIDECAR));
    let generation_cache = std::env::var_os("VIDEO_TO_3D_OBJECT_CANDIDATE_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| output.to_owned());
    let cached = fs::read_to_string(generation_cache.join("generation.json"))
        .ok()
        .and_then(|document| ObjectGenerationArtifact::from_json(&document).ok());
    let generation = generate_object_candidates(
        keyframes_path.parent().unwrap_or_else(|| Path::new(".")),
        &keyframes,
        &inventory,
        &references,
        refs_hash,
        output,
        "objects",
        ("triposr", "1"),
        &sidecar,
        cached
            .as_ref()
            .map(|prior| (generation_cache.as_path(), prior)),
        &CancellationToken::new(),
    )?
    .ok_or("generation canceled")?;
    save(output, "generation.json", generation.to_json())?;
    let placement = video_to_3d_core::generated_scene::place_generated_meshes(
        &inventory,
        &identities,
        &references,
        &generation,
        output,
        cameras,
        points,
        &frames,
        || false,
    )?;
    save(output, "placement.json", placement.to_json())?;
    Ok(
        json!({"generation":"generation.json","references":"references.json","identity":"identity.json","placement":"placement.json","multiview":"multiview.json","objects":generation.objects.len()}),
    )
}
