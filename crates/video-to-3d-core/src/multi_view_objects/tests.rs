use super::*;
use crate::semantic_inventory::{MaskEncoding, SemanticMask};

fn camera(x: f64, frame_index: usize) -> FrameCamera {
    FrameCamera {
        frame_index,
        rotation: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        translation: [-x, 0., 0.],
        focal_pixels: 100.,
        principal_point: [48.; 2],
    }
}

fn sphere_mask(cam: &FrameCamera, shift: i32) -> SemanticObservation {
    let eye = camera_center(cam);
    let center = Vector3::new(0., 0., 3.);
    let mut counts = vec![0];
    let mut foreground = false;
    let mut bbox = [96., 96., 0., 0.];
    for y in 0..96 {
        for x in 0..96 {
            let ray = Vector3::new(
                (f64::from(x - shift) + 0.5 - 48.) / 100.,
                (f64::from(y) + 0.5 - 48.) / 100.,
                1.,
            )
            .normalize();
            let hit = (center - eye - (center - eye).dot(&ray) * ray).norm() < 0.45;
            if hit {
                bbox[0] = f64::min(bbox[0], x as f64);
                bbox[1] = f64::min(bbox[1], y as f64);
                bbox[2] = f64::max(bbox[2], x as f64 + 1.);
                bbox[3] = f64::max(bbox[3], y as f64 + 1.);
            }
            if hit != foreground {
                counts.push(0);
                foreground = hit;
            }
            *counts.last_mut().unwrap() += 1;
        }
    }
    SemanticObservation {
        id: format!("obs-{}", cam.frame_index),
        frame_index: cam.frame_index,
        label: "ball".into(),
        bbox,
        detection_confidence: 1.,
        mask_confidence: Some(1.),
        mask: Some(SemanticMask {
            encoding: MaskEncoding::RleRowMajor,
            width: 96,
            height: 96,
            counts,
        }),
    }
}

fn trial(baseline: f64, heldout_shift: i32) -> HullOutcome {
    let cams: Vec<_> = [-baseline, 0., baseline]
        .into_iter()
        .enumerate()
        .map(|(i, x)| camera(x, i))
        .collect();
    let obs: Vec<_> = cams
        .iter()
        .map(|c| sphere_mask(c, if c.frame_index == 1 { heldout_shift } else { 0 }))
        .collect();
    let views: Vec<_> = cams
        .iter()
        .zip(&obs)
        .map(|(c, o)| View::new(c, o).unwrap())
        .collect();
    let mut out = HullOutcome {
        object: "ball-1".into(),
        label: "ball".into(),
        views: vec![0, 1, 2],
        holdout_frame: None,
        holdout_mask_iou: None,
        mesh: None,
        reason: String::new(),
    };
    let points = [
        [-0.15, 0., 2.60],
        [0.15, 0., 2.60],
        [0., -0.15, 2.60],
        [0., 0.15, 2.60],
    ];
    if let Err(e) = reconstruct_one(&views, &points, false, &mut out, &[]) {
        out.reason = e.into();
    }
    out
}

#[test]
fn changing_only_the_heldout_mask_can_reject_but_never_shapes_the_hull() {
    let aligned = trial(0.9, 0);
    let shifted = trial(0.9, 32);
    let mesh = aligned.mesh.unwrap();
    assert!(aligned.holdout_mask_iou.unwrap() > 0.75);
    assert_eq!(
        mesh.validation.verdict,
        PlacementVerdict::Accepted,
        "{:?}",
        mesh.validation
    );
    let other = shifted.mesh.unwrap();
    assert_eq!(mesh.vertices, other.vertices);
    assert_eq!(mesh.triangles, other.triangles);
    assert_eq!(other.validation.verdict, PlacementVerdict::Contradicted);
    assert!(shifted.holdout_mask_iou.unwrap() < 0.15);
}

#[test]
fn identical_views_do_not_invent_object_depth() {
    let result = trial(0., 0);
    assert!(result.mesh.is_none());
    assert!(result.reason.contains("angular separation"));
}

#[test]
fn surface_nets_faces_point_out_of_the_volume_and_share_edges() {
    let n = 8;
    let field: Vec<_> = (0..n * n * n)
        .map(|i| {
            [i % n, i / n % n, i / n / n]
                .iter()
                .all(|k| (2..=5).contains(k))
        })
        .collect();
    let (vertices, triangles) = surface_nets(&field, n, |p| p);
    let mut edges = BTreeMap::new();
    for triangle in triangles {
        let [a, b, c] = triangle.map(|i| Vector3::from(vertices[i as usize]));
        assert!((b - a).cross(&(c - a)).dot(&(a - Vector3::repeat(3.5))) > 0.);
        for [i, j] in [[0, 1], [1, 2], [2, 0]] {
            let edge = (triangle[i].min(triangle[j]), triangle[i].max(triangle[j]));
            *edges.entry(edge).or_insert(0) += 1;
        }
    }
    assert!(edges.values().all(|count| *count == 2));
}

#[test]
fn calibrated_silhouette_rays_resolve_an_object_without_surface_landmarks() {
    let a = camera(-0.8, 0);
    let b = camera(0.8, 1);
    let obs_a = sphere_mask(&a, 0);
    let obs_b = sphere_mask(&b, 0);
    let va = View::new(&a, &obs_a).unwrap();
    let vb = View::new(&b, &obs_b).unwrap();
    let depth = triangulated_depth(&[&va, &vb], &a).unwrap();
    assert!((depth - 3.).abs() < 0.15);
    assert!(triangulated_depth(&[&va, &va], &a).is_none());
    let supports: Vec<_> = (-3..=3)
        .flat_map(|i| (-3..=3).map(move |j| [i as f64 * 0.05, j as f64 * 0.05, 3.]))
        .collect();
    let (vertices, faces) = plane_patch(&va, &[&va, &vb], 2, &supports).unwrap();
    assert!(faces.len() > 100);
    assert!(vertices.iter().all(|p| (p[2] - 3.).abs() < 1e-8));
    let line: Vec<_> = (-10..10).map(|i| [i as f64 * 0.02, 0., 3.]).collect();
    assert!(plane_patch(&va, &[&va, &vb], 2, &line).is_err());
}

#[test]
fn two_masks_create_an_editable_candidate_without_verified_placement() {
    let cameras = [camera(-0.9, 0), camera(0.9, 1)];
    let observations = cameras.each_ref().map(|cam| sphere_mask(cam, 0));
    let views: Vec<_> = cameras
        .iter()
        .zip(&observations)
        .map(|(cam, obs)| View::new(cam, obs).unwrap())
        .collect();
    let mut outcome = HullOutcome {
        object: "ball".into(),
        label: "ball".into(),
        views: vec![0, 1],
        holdout_frame: None,
        holdout_mask_iou: None,
        mesh: None,
        reason: String::new(),
    };
    reconstruct_one(&views, &[], false, &mut outcome, &[]).unwrap();
    assert!(outcome.holdout_mask_iou.is_none());
    let mesh = outcome.mesh.unwrap();
    assert!(!mesh.triangles.is_empty());
    assert_ne!(mesh.validation.verdict, PlacementVerdict::Accepted);
}

#[test]
fn textured_foreground_anchors_depth_without_using_the_heldout_image() {
    let cams = [camera(-0.6, 0), camera(0., 1), camera(0.6, 2)];
    let obs = cams.each_ref().map(|cam| sphere_mask(cam, 0));
    let views: Vec<_> = cams
        .iter()
        .zip(&obs)
        .map(|(c, o)| View::new(c, o).unwrap())
        .collect();
    // Translations shift the same textured plane by exactly 20 pixels between cameras.
    let mut images = Vec::new();
    for i in 0..3 {
        let mut pixels = vec![0; 96 * 96 * 4];
        for y in 0..96 {
            for x in 0..96 {
                let u = x + i * 20;
                let value = ((u * 37 + y * 71 + u * y * 11) % 231) as u8;
                pixels[4 * (y * 96 + x)..4 * (y * 96 + x) + 3].fill(value);
            }
        }
        images.push(pixels);
    }
    let frames: Vec<_> = images
        .iter()
        .enumerate()
        .map(|(i, p)| (i, 96, 96, p.as_slice()))
        .collect();
    let fit = [&views[0], &views[2]];
    let band = stereo::depth_band(&views[0], &fit, &frames, 3., 1.).unwrap();
    assert!((0.5 * (band.near + band.far) - 3.).abs() < 0.1);
    let flat = vec![127; 96 * 96 * 4];
    let flat_frames = [(0, 96, 96, flat.as_slice()), (2, 96, 96, flat.as_slice())];
    assert!(stereo::depth_band(&views[0], &fit, &flat_frames, 3., 1.).is_none());
}

// A full chair provides a stronger regression target than a solid sphere:
// seats, backs, empty space and individual thin legs all affect the novel view.
#[test]
fn a_well_observed_chair_preserves_its_silhouette_from_an_unused_orbit_angle() {
    let mut boxes = vec![
        (
            Vector3::new(-0.45, -0.04, -0.40),
            Vector3::new(0.45, 0.04, 0.40),
        ),
        (
            Vector3::new(-0.45, -0.68, 0.30),
            Vector3::new(0.45, -0.04, 0.38),
        ),
    ];
    for x in [-0.40, 0.40] {
        for z in [-0.35, 0.35] {
            boxes.push((
                Vector3::new(x - 0.027, 0.03, z - 0.027),
                Vector3::new(x + 0.027, 0.68, z + 0.027),
            ));
        }
    }
    let chair_view = |degrees: f64, index| {
        let angle = degrees.to_radians();
        let eye = Vector3::new(2.7 * angle.sin(), -0.55, -2.7 * angle.cos());
        let forward = (-eye).normalize();
        let right = Vector3::y().cross(&forward).normalize();
        let down = forward.cross(&right);
        let axes = [right, down, forward];
        let cam = FrameCamera {
            frame_index: index,
            rotation: std::array::from_fn(|i| axes[i / 3][i % 3]),
            translation: axes.map(|axis| -axis.dot(&eye)),
            focal_pixels: 142.,
            principal_point: [48.; 2],
        };
        let mut counts = vec![0];
        let mut was_inside = false;
        let mut bbox = [96., 96., 0., 0.];
        for y in 0..96 {
            for x in 0..96 {
                let ray = ray_at(&cam, [x as f64 + 0.5, y as f64 + 0.5]);
                let inside = boxes.iter().any(|(lo, hi)| {
                    let mut front = 0.0f64;
                    let mut back = f64::INFINITY;
                    for axis in 0..3 {
                        if ray[axis].abs() < 1e-10 {
                            if eye[axis] < lo[axis] || eye[axis] > hi[axis] {
                                return false;
                            }
                        } else {
                            let a = (lo[axis] - eye[axis]) / ray[axis];
                            let b = (hi[axis] - eye[axis]) / ray[axis];
                            front = front.max(a.min(b));
                            back = back.min(a.max(b));
                        }
                    }
                    front <= back
                });
                if inside {
                    bbox = [
                        f64::min(bbox[0], x as f64),
                        f64::min(bbox[1], y as f64),
                        f64::max(bbox[2], (x + 1) as f64),
                        f64::max(bbox[3], (y + 1) as f64),
                    ];
                }
                if inside != was_inside {
                    counts.push(0);
                    was_inside = inside;
                }
                *counts.last_mut().unwrap() += 1;
            }
        }
        let obs = SemanticObservation {
            id: format!("chair-{index}"),
            frame_index: index,
            label: "chair".into(),
            bbox,
            detection_confidence: 1.,
            mask_confidence: Some(1.),
            mask: Some(SemanticMask {
                width: 96,
                height: 96,
                encoding: MaskEncoding::RleRowMajor,
                counts,
            }),
        };
        (cam, obs)
    };
    let shots: Vec<_> = [-125., -94., -68., -38., -9., 19., 49., 81., 113., 145.]
        .into_iter()
        .enumerate()
        .map(|(i, a)| chair_view(a, i))
        .collect();
    let views: Vec<_> = shots
        .iter()
        .map(|(c, o)| View::new(c, o).unwrap())
        .collect();
    let points: Vec<_> = boxes
        .iter()
        .flat_map(|(lo, hi)| [(*lo).into(), (*hi).into()])
        .collect();
    let mut result = HullOutcome {
        object: "chair".into(),
        label: "chair".into(),
        reason: String::new(),
        views: vec![],
        holdout_frame: None,
        holdout_mask_iou: None,
        mesh: None,
    };
    reconstruct_one(&views, &points, false, &mut result, &[]).unwrap();
    let mesh = result.mesh.unwrap();
    let (cam, obs) = chair_view(65., 22);
    let iou = heldout_mesh_iou(&cam, &obs, &mesh.vertices, &mesh.triangles).unwrap();
    assert!(iou > 0.73, "unused 65 degree chair silhouette IoU {iou:.3}");
}
