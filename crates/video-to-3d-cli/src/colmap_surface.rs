use serde_json::Value;
use std::{fs, io::Write, path::Path};
use video_to_3d_core::{scene_artifacts::KeyframesArtifact, FrameInput, ReconstructionRequest};

pub(super) fn run(
    manifest: &Path,
    registration: &Path,
    grids: &Path,
    output: &Path,
    inventory_path: Option<&Path>,
) -> Result<Value, String> {
    let read = |p: &Path| fs::read_to_string(p).map_err(|e| e.to_string());
    let frames = KeyframesArtifact::from_json(&read(manifest)?)?;
    let inventory = inventory_path
        .map(|path| {
            let doc = read(path)?;
            let inventory =
                video_to_3d_core::semantic_inventory::SemanticInventoryArtifact::from_json(&doc)?;
            inventory.validate_against_keyframes(&frames)?;
            let hash = video_to_3d_core::scene_project::ContentHash::of_bytes(doc.as_bytes());
            Ok::<_, String>((inventory, hash))
        })
        .transpose()?;
    let root = manifest.parent().ok_or("missing keyframe directory")?;
    let request = ReconstructionRequest {
        frames: frames
            .frames
            .iter()
            .map(|f| {
                Ok(FrameInput {
                    width: f.width,
                    height: f.height,
                    rgba: frames.load_pixels(root, f.frame_index)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        options: Default::default(),
    };
    let registration = serde_json::from_str(&read(registration)?).map_err(|e| e.to_string())?;
    let grids = serde_json::from_str(&read(grids)?).map_err(|e| e.to_string())?;
    let (glb, report) = video_to_3d_core::colmap_surface::export(
        &request,
        registration,
        grids,
        inventory
            .as_ref()
            .map(|(inventory, hash)| (inventory, hash.clone())),
    )?;
    let mut partial = output.as_os_str().to_owned();
    partial.push(format!(".partial-{}", std::process::id()));
    let partial = std::path::PathBuf::from(partial);
    let mut destination = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)
        .map_err(|e| e.to_string())?;
    let publish = destination.write_all(&glb).and_then(|_| {
        drop(destination);
        fs::hard_link(&partial, output)
    });
    let _ = fs::remove_file(partial);
    publish.map_err(|e| e.to_string())?;
    serde_json::to_value(report).map_err(|e| e.to_string())
}
