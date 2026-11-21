//! A camera-only oracle for the feature-poor asset-tooling storage house. The
//! fixture holds the 18 browser-decoded 360x270 luma frames of a Blender orbit
//! of asset-tooling’s raid-defense/v4/models/storage_house.glb (SHA-256
//! 7639ad48e50aa9566319b9bc25f4083e8f9f924a6336df331c04d8d1119cf431).
//! The render camera moved 3° per frame around a level circle, looking at the
//! house center with elevation atan(0.40). Only pixels enter reconstruction.

use nalgebra::{Matrix3, Vector3};
use video_to_3d_core::{reconstruct, FrameInput, ReconstructionOptions, ReconstructionRequest};

#[test]
fn sparse_house_recovers_the_direction_of_the_rendered_orbit() {
    let (width, height) = (360, 270);
    let pixels = (width * height) as usize;
    let luma = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(
        include_bytes!("fixtures/storage-house-luma.zlib"),
        18 * pixels,
    )
    .unwrap();
    assert_eq!(luma.len(), 18 * pixels);
    let frames = luma
        .chunks_exact(pixels)
        .map(|frame| FrameInput {
            width,
            height,
            rgba: frame.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        })
        .collect();
    let scene = reconstruct(&ReconstructionRequest {
        frames,
        options: ReconstructionOptions::default(),
    })
    .unwrap();
    let seed = scene.calibrated_pair.expect("calibrated house cameras");
    let yaw = ((seed.to_frame - seed.from_frame) as f64 * 3.0).to_radians();
    let elevation = 0.40_f64.atan();
    // Ground-truth displacement in the source camera's right/down/forward axes.
    let expected = Vector3::new(
        yaw.sin(),
        elevation.sin() * (yaw.cos() - 1.0),
        elevation.cos() * (1.0 - yaw.cos()),
    );
    let rotation = Matrix3::from_row_slice(&seed.relative_rotation).cast::<f64>();
    let translation = Vector3::from_column_slice(&seed.translation_direction).cast::<f64>();
    let recovered = -rotation.transpose() * translation;
    let error = recovered.angle(&expected).to_degrees();
    assert!(error < 20.0, "seed translation direction error {error}°");
    assert!(
        !scene.registered_views.is_empty(),
        "no confirming third view"
    );
}
