use serde_json::{json, Value};
use std::{fs, io::Write, path::Path};
use video_to_3d_core::generated_scene::{
    append_validated_objects, export_object_library, MultiViewObjectReport,
};
use video_to_3d_core::object_generation::ObjectGenerationArtifact;
use video_to_3d_core::object_identity::RegisteredObjectGeometry;
use video_to_3d_core::object_placement::ObjectPlacementArtifact;
use video_to_3d_core::object_references::ObjectReferencesArtifact;
use video_to_3d_core::scene_project::ContentHash;

pub(super) fn run(
    observed: Option<&Path>,
    root: &Path,
    destination: &Path,
) -> Result<Value, String> {
    let read = |name: &str| fs::read_to_string(root.join(name)).map_err(|e| format!("{name}: {e}"));
    let references = read("references.json")?;
    let generation = ObjectGenerationArtifact::from_json(&read("generation.json")?)?;
    if generation.object_references != ContentHash::of_bytes(references.as_bytes()) {
        return Err("generation belongs to another reference package".into());
    }
    let references = ObjectReferencesArtifact::from_json(&references)?;
    let placement = ObjectPlacementArtifact::from_json(&read("placement.json")?)?;

    let multiview: MultiViewObjectReport =
        serde_json::from_str(&read("multiview.json")?).map_err(|e| e.to_string())?;
    if multiview.object_identity != references.object_identity {
        return Err("multi-view objects belong to different source identities".into());
    }
    let (command, bytes) = if let Some(observed) = observed {
        let geometry = RegisteredObjectGeometry::from_json(&read("geometry.json")?)?;
        let input = fs::read(observed).map_err(|e| format!("{}: {e}", observed.display()))?;
        (
            "complete-scene",
            append_validated_objects(
                &input,
                root,
                &generation,
                &placement,
                &references.semantic_inventory,
                &geometry.cameras,
                &multiview,
            )?,
        )
    } else {
        (
            "object-library",
            export_object_library(root, &generation, &placement, &multiview)?,
        )
    };
    // A completed export is immutable. Publish the fully written sibling file
    // without overwriting an earlier export, including under concurrent runs.
    let temporary =
        destination.with_file_name(format!(".video-to-3d-complete-{}.tmp", std::process::id()));
    let mut file = fs::File::create_new(&temporary).map_err(|e| e.to_string())?;
    let result = (|| {
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::hard_link(&temporary, destination)
    })();
    drop(file);
    let _ = fs::remove_file(temporary);
    result.map_err(|e| format!("{}: {e}", destination.display()))?;
    Ok(json!({"command":command,"output":destination,"bytes":bytes.len()}))
}
