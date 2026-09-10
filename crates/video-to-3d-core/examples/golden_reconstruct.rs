use std::{env, fs, path::Path};
use video_to_3d_core::{reconstruct, FrameInput, ReconstructionOptions, ReconstructionRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let fixture = args.next().unwrap_or_else(|| "target/golden-scene".into());
    let output = args
        .next()
        .unwrap_or_else(|| "target/golden-scene/ours-cameras.tsv".into());

    let manifest = fs::read_to_string(Path::new(&fixture).join("ground_truth.tsv"))?;
    let mut frames = Vec::new();
    let mut focal = None;
    let mut width = None;
    let mut height = None;

    for line in manifest.lines().skip(1) {
        let columns: Vec<&str> = line.split('\t').collect();
        if columns.len() != 9 {
            return Err(format!("invalid golden-scene row: {line}").into());
        }
        let row_focal: f32 = columns[6].parse()?;
        let row_width: u32 = columns[7].parse()?;
        let row_height: u32 = columns[8].parse()?;
        focal.get_or_insert(row_focal);
        width.get_or_insert(row_width);
        height.get_or_insert(row_height);
        if focal != Some(row_focal) || width != Some(row_width) || height != Some(row_height) {
            return Err("golden-scene camera intrinsics must be constant".into());
        }
        frames.push(FrameInput {
            width: row_width,
            height: row_height,
            rgba: fs::read(Path::new(&fixture).join(columns[2]))?,
        });
    }

    let result = reconstruct(&ReconstructionRequest {
        frames,
        options: ReconstructionOptions {
            max_features: 320,
            min_feature_distance: 5,
            descriptor_radius: 3,
            match_radius: 72,
            max_descriptor_distance: 48.0,
            ratio_threshold: 0.88,
            focal_length_pixels: focal,
        },
    })?;

    let mut text = String::from("frame_index\tcx\tcy\tcz\n");
    for camera in &result.cameras {
        text.push_str(&format!(
            "{}\t{:.9}\t{:.9}\t{:.9}\n",
            camera.frame_index, camera.x, camera.y, camera.z
        ));
    }
    fs::write(output, text)?;

    eprintln!(
        "golden reconstruction: cameras={} points={} calibrated={} closures={}",
        result.cameras.len(),
        result.points.len(),
        result.calibrated_pair.is_some(),
        result
            .revisits
            .closures
            .iter()
            .filter(|closure| closure.accepted)
            .count()
    );
    Ok(())
}
