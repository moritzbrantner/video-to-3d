use std::{env, fs, path::Path};

use nalgebra::Vector3;
use video_to_3d_core::{
    reconstruct, CameraPose, FrameInput, ReconstructionOptions, ReconstructionRequest,
};

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

#[derive(Clone, Copy)]
struct CameraCenter {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy)]
enum Scenario {
    Lateral,
    Revisit,
    Forward,
}

impl Scenario {
    fn parse(value: &str) -> Self {
        match value {
            "lateral" => Self::Lateral,
            "revisit" => Self::Revisit,
            "forward" => Self::Forward,
            _ => panic!("unknown golden scenario: {value}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Lateral => "lateral",
            Self::Revisit => "revisit",
            Self::Forward => "forward",
        }
    }

    fn camera_center(self, frame: usize) -> CameraCenter {
        match self {
            Self::Lateral => CameraCenter {
                x: -0.42 + frame as f64 * 0.12,
                y: 0.0,
                z: 0.0,
            },
            Self::Revisit => {
                const X: [f64; FRAME_COUNT] = [-0.42, -0.25, -0.08, 0.10, 0.29, 0.14, -0.03, -0.21];
                const Y: [f64; FRAME_COUNT] = [0.00, 0.02, 0.04, 0.05, 0.04, 0.02, 0.00, -0.02];
                CameraCenter {
                    x: X[frame],
                    y: Y[frame],
                    z: 0.0,
                }
            }
            Self::Forward => CameraCenter {
                x: -0.21 + frame as f64 * 0.06,
                y: 0.0,
                z: frame as f64 * 0.08,
            },
        }
    }
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
}

fn scene() -> Vec<ScenePoint> {
    let mut state = 0x005e_ed3d_u64;
    (0..POINT_COUNT)
        .map(|index| ScenePoint {
            x: (lcg(&mut state) - 0.5) * 5.2,
            y: (lcg(&mut state) - 0.5) * 3.5,
            z: 4.0 + lcg(&mut state) * 6.0,
            shade: 35 + ((index * 47) % 190) as u8,
        })
        .collect()
}

fn render_frame(frame: usize, points: &[ScenePoint], scenario: Scenario) -> FrameInput {
    let mut rgba = vec![238_u8; (WIDTH * HEIGHT * 4) as usize];
    for pixel in rgba.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }

    let center_x = WIDTH as f64 * 0.5;
    let center_y = HEIGHT as f64 * 0.5;
    let camera = scenario.camera_center(frame);
    for (index, point) in points.iter().enumerate() {
        let relative_x = point.x - camera.x;
        let relative_y = point.y - camera.y;
        let relative_z = point.z - camera.z;
        if relative_z <= 0.25 {
            continue;
        }
        let px = (relative_x / relative_z * FOCAL + center_x).round() as i32;
        let py = (relative_y / relative_z * FOCAL + center_y).round() as i32;
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
    for pixel in frame.rgba.as_chunks::<4>().0 {
        bytes.extend_from_slice(&pixel[..3]);
    }
    fs::write(path, bytes).expect("write PPM fixture");
}

fn normalized_pose_rmse(cameras: &[CameraPose], scenario: Scenario) -> Option<f64> {
    let matched: Vec<(Vector3<f64>, Vector3<f64>)> = cameras
        .iter()
        .map(|camera| {
            let truth = scenario.camera_center(camera.frame_index);
            (
                Vector3::new(camera.x as f64, camera.y as f64, camera.z as f64),
                Vector3::new(truth.x, truth.y, truth.z),
            )
        })
        .collect();
    if matched.len() < 3 {
        return None;
    }

    let count = matched.len() as f64;
    let estimated_centroid = matched
        .iter()
        .map(|(estimated, _)| *estimated)
        .sum::<Vector3<f64>>()
        / count;
    let truth_centroid = matched
        .iter()
        .map(|(_, truth)| *truth)
        .sum::<Vector3<f64>>()
        / count;

    let mut covariance = nalgebra::Matrix3::zeros();
    let mut estimated_variance = 0.0;
    for (estimated, truth) in &matched {
        let estimated_centered = estimated - estimated_centroid;
        let truth_centered = truth - truth_centroid;
        covariance += estimated_centered * truth_centered.transpose();
        estimated_variance += estimated_centered.norm_squared();
    }
    if estimated_variance <= 1e-12 {
        return None;
    }

    let svd = covariance.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let v = v_t.transpose();
    let mut correction = nalgebra::Matrix3::identity();
    if (v * u.transpose()).determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    let rotation = v * correction * u.transpose();
    let signed_singular_sum = svd.singular_values[0]
        + svd.singular_values[1]
        + correction[(2, 2)] * svd.singular_values[2];
    let scale = signed_singular_sum / estimated_variance;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let translation = truth_centroid - scale * rotation * estimated_centroid;

    let squared_error = matched
        .iter()
        .map(|(estimated, truth)| {
            let aligned = scale * rotation * estimated + translation;
            (aligned - truth).norm_squared()
        })
        .sum::<f64>();
    let rmse = (squared_error / count).sqrt();

    let trajectory_span = (0..FRAME_COUNT)
        .flat_map(|left| {
            ((left + 1)..FRAME_COUNT).map(move |right| {
                let a = scenario.camera_center(left);
                let b = scenario.camera_center(right);
                Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z).norm()
            })
        })
        .fold(0.0_f64, f64::max);
    (trajectory_span > 1e-12).then_some(rmse / trajectory_span)
}

fn main() {
    let output_dir = env::args().nth(1);
    let scenario = Scenario::parse(env::args().nth(2).as_deref().unwrap_or("lateral"));
    let points = scene();
    let frames: Vec<FrameInput> = (0..FRAME_COUNT)
        .map(|frame| render_frame(frame, &points, scenario))
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
    let pose_rmse = normalized_pose_rmse(&result.cameras, scenario);

    println!(
        "golden-rust case={} registered_images={} points={} median_reprojection_error_pixels={} normalized_pose_rmse={}",
        scenario.name(),
        registered_images,
        result.points.len(),
        reprojection.map_or_else(|| "nan".to_owned(), |value| format!("{value:.6}")),
        pose_rmse.map_or_else(|| "nan".to_owned(), |value| format!("{value:.6}"))
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};

    fn pose(frame_index: usize, position: Vector3<f64>) -> CameraPose {
        CameraPose {
            frame_index,
            x: position.x as f32,
            y: position.y as f32,
            z: position.z as f32,
            matched_features: 0,
        }
    }

    #[test]
    fn pose_alignment_removes_similarity_gauge() {
        let angle = 0.47_f64;
        let rotation = Matrix3::new(
            angle.cos(),
            -angle.sin(),
            0.0,
            angle.sin(),
            angle.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let scale = 2.3;
        let offset = Vector3::new(1.2, -0.7, 0.4);
        let cameras: Vec<CameraPose> = (0..FRAME_COUNT)
            .map(|frame_index| {
                let truth = Scenario::Revisit.camera_center(frame_index);
                let truth = Vector3::new(truth.x, truth.y, truth.z);
                pose(frame_index, scale * rotation * truth + offset)
            })
            .collect();

        let error = normalized_pose_rmse(&cameras, Scenario::Revisit).expect("valid alignment");
        assert!(error < 1e-5, "similarity gauge should align exactly, got {error}");
    }

    #[test]
    fn pose_alignment_preserves_real_drift() {
        let mut cameras: Vec<CameraPose> = (0..FRAME_COUNT)
            .map(|frame_index| {
                let truth = Scenario::Revisit.camera_center(frame_index);
                pose(frame_index, Vector3::new(truth.x, truth.y, truth.z))
            })
            .collect();
        cameras[4].y += 0.35;

        let error = normalized_pose_rmse(&cameras, Scenario::Revisit).expect("valid alignment");
        assert!(error > 0.1, "pose drift should survive Sim(3) alignment, got {error}");
    }

    #[test]
    fn pose_alignment_rejects_degenerate_camera_centers() {
        let cameras: Vec<CameraPose> = (0..FRAME_COUNT)
            .map(|frame_index| pose(frame_index, Vector3::new(1.0, 1.0, 1.0)))
            .collect();

        assert!(normalized_pose_rmse(&cameras, Scenario::Revisit).is_none());
    }
}
