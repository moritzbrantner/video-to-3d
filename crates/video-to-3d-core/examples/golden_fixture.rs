use std::{env, fs, path::Path};

use video_to_3d_core::{reconstruct, FrameInput, ReconstructionOptions, ReconstructionRequest};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FOCAL: f64 = 520.0;
const FRAME_COUNT: usize = 8;
const POINT_COUNT: usize = 420;

#[derive(Clone, Copy)]
struct ScenePoint {
    x: f64,
    y: f64,
    z: f64,
    shade: u8,
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
}

fn scene() -> Vec<ScenePoint> {
    let mut state = 0x5eed_3d_u64;
    (0..POINT_COUNT)
        .map(|index| ScenePoint {
            x: (lcg(&mut state) - 0.5) * 5.2,
            y: (lcg(&mut state) - 0.5) * 3.5,
            z: 4.0 + lcg(&mut state) * 6.0,
            shade: 35 + ((index * 47) % 190) as u8,
        })
        .collect()
}

fn camera_x(frame: usize) -> f64 {
    -0.42 + frame as f64 * 0.12
}

fn render_frame(frame: usize, points: &[ScenePoint]) -> FrameInput {
    let mut rgba = vec![238_u8; (WIDTH * HEIGHT * 4) as usize];
    for pixel in rgba.chunks_exact_mut(4) {
        pixel[3] = 255;
    }

    let center_x = WIDTH as f64 * 0.5;
    let center_y = HEIGHT as f64 * 0.5;
    let cx = camera_x(frame);
    for (index, point) in points.iter().enumerate() {
        let px = ((point.x - cx) / point.z * FOCAL + center_x).round() as i32;
        let py = (point.y / point.z * FOCAL + center_y).round() as i32;
        if px < 5 || py < 5 || px >= WIDTH as i32 - 5 || py >= HEIGHT as i32 - 5 {
            continue;
        }
        let radius = 3 + (index % 2) as i32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = px + dx;
                let y = py + dy;
                let offset = ((y as u32 * WIDTH + x as u32) * 4) as usize;
                let checker = ((dx + dy + index as i32) & 1) == 0;
                let value = if checker {
                    point.shade
                } else {
                    255_u8.saturating_sub(point.shade / 2)
                };
                rgba[offset] = value;
                rgba[offset + 1] = value.saturating_add((index % 17) as u8);
                rgba[offset + 2] = value.saturating_sub((index % 13) as u8);
            }
        }
    }

    FrameInput {
        width: WIDTH,
        height: HEIGHT,
        rgba,
    }
}

fn write_ppm(path: &Path, frame: &FrameInput) {
    let mut bytes = format!("P6\n{} {}\n255\n", frame.width, frame.height).into_bytes();
    bytes.reserve((frame.width * frame.height * 3) as usize);
    for pixel in frame.rgba.chunks_exact(4) {
        bytes.extend_from_slice(&pixel[..3]);
    }
    fs::write(path, bytes).expect("write PPM fixture");
}

fn main() {
    let output_dir = env::args().nth(1);
    let points = scene();
    let frames: Vec<FrameInput> = (0..FRAME_COUNT)
        .map(|frame| render_frame(frame, &points))
        .collect();

    if let Some(output_dir) = output_dir.as_deref().filter(|value| *value != "-") {
        fs::create_dir_all(output_dir).expect("create fixture directory");
        for (index, frame) in frames.iter().enumerate() {
            write_ppm(
                Path::new(output_dir)
                    .join(format!("frame-{index:02}.ppm"))
                    .as_path(),
                frame,
            );
        }
    }

    let result = reconstruct(&ReconstructionRequest {
        frames,
        options: ReconstructionOptions {
            max_features: 320,
            min_feature_distance: 6,
            descriptor_radius: 3,
            match_radius: 80,
            focal_length_pixels: Some(FOCAL as f32),
            ..ReconstructionOptions::default()
        },
    })
    .expect("golden fixture reconstruction");

    let registered_images = result
        .calibrated_pair
        .as_ref()
        .map_or(0, |_| 2 + result.registered_views.len());
    let reprojection = result
        .multi_view
        .bundle_adjustment
        .final_median_reprojection_error_pixels
        .or_else(|| {
            result
                .calibrated_pair
                .as_ref()
                .map(|pair| pair.median_reprojection_error_pixels)
        });

    println!(
        "golden-rust registered_images={} points={} median_reprojection_error_pixels={}",
        registered_images,
        result.points.len(),
        reprojection.map_or_else(|| "nan".to_owned(), |value| format!("{value:.6}"))
    );
}
