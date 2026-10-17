use super::*;
use crate::semantic_inventory::{MaskEncoding, SemanticMask};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

/// Mesh frame: depth along x, width along y, height along z.
const LOW: [f64; 3] = [-0.25, -0.5, -0.375];
const HIGH: [f64; 3] = [0.25, 0.5, 0.375];
/// Mesh +x toward camera 0 (−z), +y image right (+x), +z scene up (−y).
const ROTATION: [f64; 9] = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0];
const CENTRE: [f64; 3] = [0.3, 0.85, 3.0];
const SCALE: f64 = 0.4;

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let length = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    a.map(|value| value / length)
}

/// Camera at `centre` looking at `target`, image down along scene +y.
fn look_at(frame_index: usize, centre: [f64; 3], target: [f64; 3]) -> FrameCamera {
    let forward = unit([0, 1, 2].map(|axis| target[axis] - centre[axis]));
    let right = unit(cross([0.0, 1.0, 0.0], forward));
    let down = cross(forward, right);
    let rotation = [
        right[0], right[1], right[2], down[0], down[1], down[2], forward[0], forward[1], forward[2],
    ];
    let translation = [0, 1, 2].map(|row| {
        -(0..3)
            .map(|k| rotation[row * 3 + k] * centre[k])
            .sum::<f64>()
    });
    FrameCamera {
        frame_index,
        rotation,
        translation,
        focal_pixels: 300.0,
        principal_point: [160.0, 120.0],
    }
}

fn cameras() -> Vec<FrameCamera> {
    vec![
        look_at(0, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        look_at(1, [1.6, 0.4, 0.8], CENTRE),
    ]
}

/// Points on the surface of the mesh box, in the mesh frame.
fn box_surface(step: f64) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    let steps = |axis: usize| ((HIGH[axis] - LOW[axis]) / step).round() as usize;
    for fixed in 0..3 {
        let (a, b) = ((fixed + 1) % 3, (fixed + 2) % 3);
        for side in [LOW[fixed], HIGH[fixed]] {
            for i in 0..=steps(a) {
                for j in 0..=steps(b) {
                    let mut point = [0.0; 3];
                    point[fixed] = side;
                    point[a] = LOW[a] + (HIGH[a] - LOW[a]) * i as f64 / steps(a) as f64;
                    point[b] = LOW[b] + (HIGH[b] - LOW[b]) * j as f64 / steps(b) as f64;
                    points.push(point);
                }
            }
        }
    }
    points
}

fn placed<'a>(vertices: &'a [[f64; 3]], scale: f64, translation: [f64; 3]) -> PlacedMesh<'a> {
    PlacedMesh {
        rotation: ROTATION,
        scale,
        translation,
        vertices,
        reference_frame: 0,
    }
}

/// The true object's surface in the scene.
fn true_surface(step: f64) -> Vec<[f64; 3]> {
    let vertices = box_surface(step);
    let truth = placed(&vertices, SCALE, CENTRE);
    vertices.iter().map(|v| truth.to_scene(*v)).collect()
}

/// A back wall at z = 4.5 behind the object.
fn wall() -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for i in 0..=90 {
        for j in 0..=110 {
            points.push([-1.2 + 0.03 * i as f64, -0.8 + 0.03 * j as f64, 4.5]);
        }
    }
    points
}

/// The true object's mask in `camera`, as an original semantic observation.
fn observation(camera: &FrameCamera) -> SemanticObservation {
    let mut grid = vec![false; (WIDTH * HEIGHT) as usize];
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, 0.0_f64, 0.0_f64);
    for point in true_surface(0.01) {
        let Some((pixel, _)) = project(camera, point) else {
            continue;
        };
        let (column, row) = (pixel[0].floor() as i64, pixel[1].floor() as i64);
        if column < 0 || row < 0 || column >= WIDTH as i64 || row >= HEIGHT as i64 {
            continue;
        }
        grid[(row as u32 * WIDTH + column as u32) as usize] = true;
        x0 = x0.min(column as f64);
        y0 = y0.min(row as f64);
        x1 = x1.max(column as f64 + 1.0);
        y1 = y1.max(row as f64 + 1.0);
    }
    let mut counts = Vec::new();
    let (mut current, mut run) = (false, 0u64);
    for value in grid {
        if value != current {
            counts.push(run);
            current = value;
            run = 0;
        }
        run += 1;
    }
    counts.push(run);
    SemanticObservation {
        id: format!("obs-{}", camera.frame_index),
        frame_index: camera.frame_index,
        label: "chair".into(),
        bbox: [x0, y0, x1, y1],
        detection_confidence: 0.9,
        mask: Some(SemanticMask {
            encoding: MaskEncoding::RleRowMajor,
            width: WIDTH,
            height: HEIGHT,
            counts,
        }),
        mask_confidence: Some(0.9),
    }
}

fn validate(scale: f64, translation: [f64; 3], view_count: usize) -> PlacementValidation {
    let cameras = cameras();
    let observations: Vec<SemanticObservation> = cameras.iter().map(observation).collect();
    let views: Vec<(&SemanticObservation, &FrameCamera)> =
        observations.iter().zip(&cameras).take(view_count).collect();
    let vertices = box_surface(0.05);
    let mut points = true_surface(0.05);
    points.extend(wall());
    validate_placement(&placed(&vertices, scale, translation), &views, &points)
}

#[test]
fn the_true_placement_is_accepted_by_both_views() {
    let validation = validate(SCALE, CENTRE, 2);
    for view in &validation.views {
        assert_eq!(view.verdict, ViewVerdict::Consistent, "{view:?}");
        assert!(view.landmarks > 0, "{view:?}");
        assert!(view.landmark_residual.unwrap() < 0.05, "{view:?}");
    }
    assert_eq!(
        validation.verdict,
        PlacementVerdict::Accepted,
        "{}",
        validation.reason
    );
    assert!(!validation.needs_review);
}

#[test]
fn a_single_reference_view_leaves_the_placement_approximate() {
    let validation = validate(SCALE, CENTRE, 1);
    assert_eq!(validation.views[0].verdict, ViewVerdict::Consistent);
    assert_eq!(validation.verdict, PlacementVerdict::Approximate);
    assert!(validation.needs_review);
}

#[test]
fn a_wrong_scale_is_contradicted() {
    let validation = validate(SCALE * 2.0, CENTRE, 2);
    assert_eq!(
        validation.verdict,
        PlacementVerdict::Contradicted,
        "{validation:?}"
    );
    assert!(validation.needs_review);
    assert!(validation.views[0].inside_share.unwrap() < MIN_INSIDE_SHARE);
}

#[test]
fn a_wrong_position_is_contradicted() {
    let shifted = [CENTRE[0] + 0.25, CENTRE[1], CENTRE[2]];
    let validation = validate(SCALE, shifted, 2);
    assert_eq!(
        validation.verdict,
        PlacementVerdict::Contradicted,
        "{validation:?}"
    );
    assert!(validation.views[0].box_iou.unwrap() < MIN_BOX_IOU);
}

#[test]
fn a_placement_behind_the_observed_wall_violates_depth_ordering() {
    // Twice as far and twice as large: the same silhouette in the reference
    // view, but behind the wall that is seen where the object is.
    let far = CENTRE.map(|value| value * 2.0);
    let validation = validate(SCALE * 2.0, far, 1);
    let reference = &validation.views[0];
    assert_eq!(reference.verdict, ViewVerdict::Contradicts, "{reference:?}");
    assert!(
        reference.depth_violation_share.unwrap() > MAX_DEPTH_VIOLATION_SHARE,
        "{reference:?}"
    );
    assert_eq!(validation.verdict, PlacementVerdict::Contradicted);
}

#[test]
fn a_wrong_orientation_is_contradicted() {
    // Turned 90° about scene up: the box's long side now faces away from
    // the reference camera.
    let cameras = cameras();
    let observations: Vec<SemanticObservation> = cameras.iter().map(observation).collect();
    let views: Vec<(&SemanticObservation, &FrameCamera)> =
        observations.iter().zip(&cameras).collect();
    let vertices = box_surface(0.05);
    let mut points = true_surface(0.05);
    points.extend(wall());
    // Rotation about −y by 90° applied after ROTATION: world x → −z, z → x.
    let turn = [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let rotation: [f64; 9] = std::array::from_fn(|index| {
        let (row, column) = (index / 3, index % 3);
        (0..3)
            .map(|k| turn[row * 3 + k] * ROTATION[k * 3 + column])
            .sum()
    });
    let turned = PlacedMesh {
        rotation,
        ..placed(&vertices, SCALE, CENTRE)
    };
    let validation = validate_placement(&turned, &views, &points);
    assert_eq!(
        validation.verdict,
        PlacementVerdict::Contradicted,
        "{validation:#?}"
    );
}
