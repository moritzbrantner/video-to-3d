use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use video_to_3d_core::{reconstruct, FrameInput, Point3, ReconstructionOptions, ReconstructionRequest};

fn next_header_token(bytes: &[u8], cursor: &mut usize) -> Result<String, String> {
    loop {
        while *cursor < bytes.len() && bytes[*cursor].is_ascii_whitespace() {
            *cursor += 1;
        }
        if *cursor < bytes.len() && bytes[*cursor] == b'#' {
            while *cursor < bytes.len() && bytes[*cursor] != b'\n' {
                *cursor += 1;
            }
            continue;
        }
        break;
    }

    let start = *cursor;
    while *cursor < bytes.len() && !bytes[*cursor].is_ascii_whitespace() {
        *cursor += 1;
    }
    if start == *cursor {
        return Err("unexpected end of PPM header".to_string());
    }
    std::str::from_utf8(&bytes[start..*cursor])
        .map(str::to_owned)
        .map_err(|error| format!("invalid UTF-8 in PPM header: {error}"))
}

fn read_ppm(path: &Path) -> Result<FrameInput, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut cursor = 0;
    let magic = next_header_token(&bytes, &mut cursor)?;
    if magic != "P6" {
        return Err(format!("{}: expected binary P6 PPM", path.display()));
    }
    let width: u32 = next_header_token(&bytes, &mut cursor)?
        .parse()
        .map_err(|_| format!("{}: invalid width", path.display()))?;
    let height: u32 = next_header_token(&bytes, &mut cursor)?
        .parse()
        .map_err(|_| format!("{}: invalid height", path.display()))?;
    let max_value: u32 = next_header_token(&bytes, &mut cursor)?
        .parse()
        .map_err(|_| format!("{}: invalid max value", path.display()))?;
    if max_value != 255 {
        return Err(format!("{}: only 8-bit PPM is supported", path.display()));
    }
    if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
        return Err(format!("{}: missing PPM pixel separator", path.display()));
    }
    cursor += 1;

    let expected = width as usize * height as usize * 3;
    if bytes.len().saturating_sub(cursor) != expected {
        return Err(format!(
            "{}: expected {expected} RGB bytes, got {}",
            path.display(),
            bytes.len().saturating_sub(cursor)
        ));
    }

    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for rgb in bytes[cursor..].chunks_exact(3) {
        rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    Ok(FrameInput { width, height, rgba })
}

fn ppm_paths(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = fs::read_dir(directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "ppm"))
        .collect::<Vec<_>>();
    paths.sort();
    if paths.len() < 4 {
        return Err(format!(
            "{}: expected at least four sampled PPM frames, got {}",
            directory.display(),
            paths.len()
        ));
    }
    Ok(paths)
}

fn write_ply(path: &Path, points: &[Point3]) -> Result<(), String> {
    let mut file = fs::File::create(path).map_err(|error| format!("{}: {error}", path.display()))?;
    writeln!(file, "ply").map_err(|error| error.to_string())?;
    writeln!(file, "format ascii 1.0").map_err(|error| error.to_string())?;
    writeln!(file, "element vertex {}", points.len()).map_err(|error| error.to_string())?;
    writeln!(file, "property float x").map_err(|error| error.to_string())?;
    writeln!(file, "property float y").map_err(|error| error.to_string())?;
    writeln!(file, "property float z").map_err(|error| error.to_string())?;
    writeln!(file, "property uchar red").map_err(|error| error.to_string())?;
    writeln!(file, "property uchar green").map_err(|error| error.to_string())?;
    writeln!(file, "property uchar blue").map_err(|error| error.to_string())?;
    writeln!(file, "property float confidence").map_err(|error| error.to_string())?;
    writeln!(file, "end_header").map_err(|error| error.to_string())?;
    for point in points {
        writeln!(
            file,
            "{} {} {} {} {} {} {}",
            point.x, point.y, point.z, point.r, point.g, point.b, point.confidence
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[middle - 1] + values[middle]) * 0.5)
    } else {
        Some(values[middle])
    }
}

fn json_number(value: Option<f32>) -> String {
    value
        .filter(|number| number.is_finite())
        .map(|number| format!("{number:.6}"))
        .unwrap_or_else(|| "null".to_string())
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let frames_directory = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: real_video_fixture <frames-directory> <output-directory>".to_string())?;
    let output_directory = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: real_video_fixture <frames-directory> <output-directory>".to_string())?;
    if arguments.next().is_some() {
        return Err("usage: real_video_fixture <frames-directory> <output-directory>".to_string());
    }

    let paths = ppm_paths(&frames_directory)?;
    let frames = paths
        .iter()
        .map(|path| read_ppm(path))
        .collect::<Result<Vec<_>, _>>()?;
    let width = frames[0].width;
    let height = frames[0].height;
    if frames
        .iter()
        .any(|frame| frame.width != width || frame.height != height)
    {
        return Err("sampled frames do not share one analysis resolution".to_string());
    }

    let result = reconstruct(&ReconstructionRequest {
        frames,
        options: ReconstructionOptions::default(),
    })?;

    fs::create_dir_all(&output_directory)
        .map_err(|error| format!("{}: {error}", output_directory.display()))?;
    write_ply(&output_directory.join("sparse.ply"), &result.points)?;
    write_ply(&output_directory.join("dense.ply"), &result.dense_points)?;

    let mut cameras = fs::File::create(output_directory.join("cameras.csv"))
        .map_err(|error| error.to_string())?;
    writeln!(cameras, "frame_index,x,y,z,matched_features").map_err(|error| error.to_string())?;
    if result.calibrated_pair.is_some() {
        for camera in &result.cameras {
            writeln!(
                cameras,
                "{},{},{},{},{}",
                camera.frame_index, camera.x, camera.y, camera.z, camera.matched_features
            )
            .map_err(|error| error.to_string())?;
        }
    }

    let mut registered_reprojection = result
        .registered_views
        .iter()
        .map(|view| view.median_reprojection_error_pixels)
        .collect::<Vec<_>>();
    let median_registered_reprojection = median(&mut registered_reprojection);
    let calibrated_seed_cameras = usize::from(result.calibrated_pair.is_some()) * 2;
    let accepted_cameras = calibrated_seed_cameras + result.registered_views.len();
    let bundle = &result.multi_view.bundle_adjustment;

    let metrics = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"video-to-3d/real-video-result/v1\",\n",
            "  \"frames\": {},\n",
            "  \"width\": {},\n",
            "  \"height\": {},\n",
            "  \"calibrated\": {},\n",
            "  \"accepted_cameras\": {},\n",
            "  \"registered_cameras\": {},\n",
            "  \"selected_keyframes\": {},\n",
            "  \"sparse_points\": {},\n",
            "  \"dense_attempted\": {},\n",
            "  \"dense_points\": {},\n",
            "  \"mesh_attempted\": {},\n",
            "  \"mesh_triangles\": {},\n",
            "  \"bundle_adjustment_attempted\": {},\n",
            "  \"bundle_adjustment_accepted\": {},\n",
            "  \"median_registered_reprojection_error_pixels\": {},\n",
            "  \"warnings\": {}\n",
            "}}\n"
        ),
        paths.len(),
        width,
        height,
        result.calibrated_pair.is_some(),
        accepted_cameras,
        result.registered_views.len(),
        result.multi_view.keyframes.len(),
        result.points.len(),
        result.dense.attempted,
        result.dense_points.len(),
        result.mesh.attempted,
        result.mesh_triangles.len(),
        bundle.attempted,
        bundle.accepted,
        json_number(median_registered_reprojection),
        result.warnings.len(),
    );
    fs::write(output_directory.join("metrics.json"), metrics)
        .map_err(|error| error.to_string())?;

    println!(
        "real-video-reconstruction frames={} accepted-cameras={} sparse={} dense={} triangles={}",
        paths.len(),
        accepted_cameras,
        result.points.len(),
        result.dense_points.len(),
        result.mesh_triangles.len()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("real-video-fixture: {error}");
        std::process::exit(1);
    }
}
