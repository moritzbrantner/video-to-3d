use crate::{dense::estimate_depth_points, multi_view::RegisteredCamera, FrameInput};
use nalgebra::{Matrix3, Vector3};

fn camera(frame_index: usize, center_x: f64) -> RegisteredCamera {
    RegisteredCamera {
        frame_index,
        rotation: Matrix3::identity(),
        translation: Vector3::new(-center_x, 0.0, 0.0),
    }
}

fn texture(world_x: f64, world_y: f64) -> u8 {
    let value = 128.0
        + 55.0 * (world_x * 15.0 + world_y * 2.7).sin()
        + 40.0 * (world_y * 17.0 - world_x * 3.1).cos()
        + 25.0 * ((world_x + world_y) * 11.0).sin();
    value.round().clamp(0.0, 255.0) as u8
}

fn plane_frame(width: u32, height: u32, focal: f64, center_x: f64, depth: f64) -> FrameInput {
    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    for y in 0..height {
        for x in 0..width {
            let world_x = center_x + (x as f64 - width as f64 * 0.5) / focal * depth;
            let world_y = (y as f64 - height as f64 * 0.5) / focal * depth;
            let value = texture(world_x, world_y);
            let index = (y as usize * width as usize + x as usize) * 4;
            rgba[index] = value;
            rgba[index + 1] = value;
            rgba[index + 2] = value;
            rgba[index + 3] = 255;
        }
    }
    FrameInput {
        width,
        height,
        rgba,
    }
}

fn plane_sparse_points(width: u32, height: u32, focal: f64, depth: f64) -> Vec<Vector3<f64>> {
    let mut points = Vec::new();
    for y in [10u32, 18, 26, 34] {
        for x in [12u32, 24, 36, 48, 56] {
            points.push(Vector3::new(
                (x as f64 - width as f64 * 0.5) / focal * depth,
                (y as f64 - height as f64 * 0.5) / focal * depth,
                depth,
            ));
        }
    }
    points
}

#[test]
fn verified_surface_completion_adds_expected_plane_geometry() {
    let width = 68;
    let height = 48;
    let focal = 60.0;
    let expected_depth = 4.0;
    let frames = vec![
        plane_frame(width, height, focal, 0.0, expected_depth),
        plane_frame(width, height, focal, 0.18, expected_depth),
        plane_frame(width, height, focal, -0.16, expected_depth),
    ];
    let cameras = vec![camera(0, 0.0), camera(1, 0.18), camera(2, -0.16)];
    let sparse = plane_sparse_points(width, height, focal, expected_depth);

    let result = estimate_depth_points(&frames, &cameras, &sparse, focal);

    assert!(result.stats.attempted);
    assert!(
        result.stats.surface_completion_proposals > 0,
        "fixture must create at least one coherent surface proposal",
    );
    assert!(
        result.stats.surface_completed_points > 0,
        "fixture must exercise successful verified surface completion",
    );
    assert_eq!(
        result.stats.surface_completion_proposals,
        result.stats.surface_completed_points
            + result.stats.surface_completion_rejected_texture
            + result.stats.surface_completion_rejected_cross_view
            + result.stats.surface_completion_rejected_reciprocal
            + result.stats.surface_completion_rejected_fusion
            + result.stats.surface_completion_rejected_footprint,
    );

    let primary_points = result
        .points
        .len()
        .checked_sub(result.stats.surface_completed_points)
        .expect("completion count cannot exceed accepted points");
    let completed_points = &result.points[primary_points..];
    let completed_sites = &result.grid_sites[primary_points..];
    assert_eq!(completed_points.len(), completed_sites.len());
    assert_eq!(completed_points.len(), result.stats.surface_completed_points);

    for (point, site) in completed_points.iter().zip(completed_sites) {
        assert!(
            (point.z as f64 - expected_depth).abs() < 0.55,
            "completed site ({}, {}) has unexpected depth {}",
            site.x,
            site.y,
            point.z,
        );
        assert!(point.confidence >= 0.05 && point.confidence <= 0.9);
    }
}
