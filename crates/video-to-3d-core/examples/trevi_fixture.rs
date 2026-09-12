use std::{env, fs, path::Path};
use video_to_3d_core::{
    reconstruct, reconstruct_browser, FrameInput, ReconstructionOptions, ReconstructionRequest,
};

fn next_ppm_token<'a>(bytes: &'a [u8], offset: &mut usize) -> Result<&'a [u8], String> {
    loop {
        while *offset < bytes.len() && bytes[*offset].is_ascii_whitespace() {
            *offset += 1;
        }
        if *offset < bytes.len() && bytes[*offset] == b'#' {
            while *offset < bytes.len() && bytes[*offset] != b'\n' {
                *offset += 1;
            }
            continue;
        }
        break;
    }

    let start = *offset;
    while *offset < bytes.len()
        && !bytes[*offset].is_ascii_whitespace()
        && bytes[*offset] != b'#'
    {
        *offset += 1;
    }
    if start == *offset {
        return Err("unexpected end of PPM header".into());
    }
    Ok(&bytes[start..*offset])
}

fn parse_usize(token: &[u8], label: &str) -> Result<usize, String> {
    std::str::from_utf8(token)
        .map_err(|_| format!("invalid UTF-8 in PPM {label}"))?
        .parse::<usize>()
        .map_err(|_| format!("invalid PPM {label}"))
}

fn load_ppm(path: &Path) -> Result<FrameInput, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut offset = 0usize;
    if next_ppm_token(&bytes, &mut offset)? != b"P6" {
        return Err(format!("{} is not a binary P6 PPM", path.display()));
    }
    let width = parse_usize(next_ppm_token(&bytes, &mut offset)?, "width")?;
    let height = parse_usize(next_ppm_token(&bytes, &mut offset)?, "height")?;
    let max_value = parse_usize(next_ppm_token(&bytes, &mut offset)?, "max value")?;
    if max_value != 255 {
        return Err(format!("{} uses unsupported max value {max_value}", path.display()));
    }

    if offset >= bytes.len() || !bytes[offset].is_ascii_whitespace() {
        return Err(format!("{} has no PPM header terminator", path.display()));
    }
    if bytes[offset] == b'\r' && bytes.get(offset + 1) == Some(&b'\n') {
        offset += 2;
    } else {
        offset += 1;
    }

    let expected_rgb = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| format!("{} dimensions overflow", path.display()))?;
    let rgb = &bytes[offset..];
    if rgb.len() != expected_rgb {
        return Err(format!(
            "{} has {} RGB bytes, expected {expected_rgb}",
            path.display(),
            rgb.len()
        ));
    }

    let mut rgba = Vec::with_capacity(width * height * 4);
    for pixel in rgb.as_chunks::<3>().0 {
        rgba.extend_from_slice(pixel);
        rgba.push(255);
    }
    Ok(FrameInput {
        width: u32::try_from(width).map_err(|_| "PPM width does not fit u32")?,
        height: u32::try_from(height).map_err(|_| "PPM height does not fit u32")?,
        rgba,
    })
}

fn main() -> Result<(), String> {
    let directory = env::args()
        .nth(1)
        .ok_or_else(|| "usage: trevi_fixture <sampled-frame-directory>".to_string())?;
    let directory = Path::new(&directory);
    let mut paths = fs::read_dir(directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    paths.retain(|path| path.extension().is_some_and(|extension| extension == "ppm"));
    paths.sort();
    if paths.len() < 4 {
        return Err(format!("expected at least four Trevi frames, got {}", paths.len()));
    }

    let frames = paths
        .iter()
        .map(|path| load_ppm(path))
        .collect::<Result<Vec<_>, _>>()?;
    let request = ReconstructionRequest {
        frames,
        options: ReconstructionOptions::default(),
    };
    let result = match reconstruct_browser(&request) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("trevi.browser_contract_error={error}");
            let raw = reconstruct(&request)?;
            let camera_frames: Vec<_> = raw.cameras.iter().map(|camera| camera.frame_index).collect();
            let registered_view_frames: Vec<_> = raw
                .registered_views
                .iter()
                .map(|view| view.frame_index)
                .collect();
            let seed_frames = raw
                .calibrated_pair
                .as_ref()
                .map(|pair| vec![pair.from_frame, pair.to_frame])
                .unwrap_or_default();
            eprintln!("trevi.raw_camera_frames={camera_frames:?}");
            eprintln!("trevi.raw_seed_frames={seed_frames:?}");
            eprintln!("trevi.raw_registered_view_frames={registered_view_frames:?}");
            return Err(error);
        }
    };
    let reconstruction = &result.reconstruction;
    let camera_state = &result.camera_state;

    println!("trevi.frames={}", camera_state.frames.len());
    println!(
        "trevi.accepted_cameras={}",
        camera_state.calibrated_seed_cameras.len() + camera_state.registered_cameras.len()
    );
    println!(
        "trevi.seed_cameras={}",
        camera_state.calibrated_seed_cameras.len()
    );
    println!(
        "trevi.registered_cameras={}",
        camera_state.registered_cameras.len()
    );
    println!(
        "trevi.dense_eligible_cameras={}",
        camera_state.dense_eligible_cameras.len()
    );
    println!("trevi.sparse_points={}", reconstruction.points.len());
    println!("trevi.dense_points={}", reconstruction.dense_points.len());
    println!("trevi.mesh_triangles={}", reconstruction.mesh_triangles.len());

    if let Some(pair) = &reconstruction.calibrated_pair {
        println!(
            "trevi.seed_pair={}->{} matches={} inliers={} inlier_ratio={:.4} reprojection_px={:.4} triangulation_deg={:.4}",
            pair.from_frame,
            pair.to_frame,
            pair.matches,
            pair.inliers,
            pair.inlier_ratio,
            pair.median_reprojection_error_pixels,
            pair.median_triangulation_angle_degrees
        );
    } else {
        println!("trevi.seed_pair=none");
    }

    println!(
        "trevi.bundle_adjustment=attempted:{} accepted:{} iterations:{} observations:{} optimized_cameras:{} optimized_landmarks:{} initial_median_px:{:?} final_median_px:{:?}",
        reconstruction.multi_view.bundle_adjustment.attempted,
        reconstruction.multi_view.bundle_adjustment.accepted,
        reconstruction.multi_view.bundle_adjustment.iterations,
        reconstruction.multi_view.bundle_adjustment.observations,
        reconstruction.multi_view.bundle_adjustment.optimized_cameras,
        reconstruction.multi_view.bundle_adjustment.optimized_landmarks,
        reconstruction.multi_view.bundle_adjustment.initial_median_reprojection_error_pixels,
        reconstruction.multi_view.bundle_adjustment.final_median_reprojection_error_pixels
    );
    println!(
        "trevi.dense=attempted:{} reference:{:?} sources:{:?} accepted_points:{} skip:{:?}",
        reconstruction.dense.attempted,
        reconstruction.dense.reference_frame,
        reconstruction.dense.source_frames,
        reconstruction.dense.accepted_points,
        reconstruction.dense.skip_reason
    );
    println!(
        "trevi.mesh=attempted:{} reference:{:?} accepted_triangles:{} skip:{:?}",
        reconstruction.mesh.attempted,
        reconstruction.mesh.reference_frame,
        reconstruction.mesh.accepted_triangles,
        reconstruction.mesh.skip_reason
    );

    for pair in &reconstruction.pairs {
        println!(
            "trevi.pair={}->{} features={}/{} matches={} overlap={:.4} median_motion={:.3} parallax_residual={:.3} low_parallax={}",
            pair.from_frame,
            pair.to_frame,
            pair.features_from,
            pair.features_to,
            pair.matches,
            pair.overlap_ratio,
            pair.median_motion,
            pair.median_parallax_residual,
            pair.low_parallax
        );
    }
    for frame in &camera_state.frames {
        println!(
            "trevi.frame={} kind={:?} registration={:?} correspondences={} inliers={} reprojection_px={:?} revisit={} bundle={:?} dense_role={:?} dense_skip={:?}",
            frame.frame_index,
            frame.camera_kind,
            frame.registration_status,
            frame.correspondences,
            frame.inliers,
            frame.median_reprojection_error_pixels,
            frame.recovered_from_revisit,
            frame.bundle_adjustment,
            frame.dense_role,
            frame.dense_ineligibility_reason
        );
    }
    for warning in &reconstruction.warnings {
        println!("trevi.warning={warning}");
    }

    Ok(())
}
