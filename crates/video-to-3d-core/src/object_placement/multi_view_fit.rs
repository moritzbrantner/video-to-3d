//! Fit a generated shape to calibrated masks. A frame excluded from the fit
//! checks the final full-resolution triangle silhouette independently.
use super::*;
use crate::object_generation::GenerationRequest;
use crate::object_identity::{front_points, project};
use crate::placement_validation::{
    DEPTH_CELL_PIXELS, DEPTH_MARGIN, MAX_DEPTH_VIOLATION_SHARE, MAX_LANDMARK_RESIDUAL,
    MIN_INSIDE_SHARE,
};
use crate::semantic_inventory::SemanticObservation;
use nalgebra::{Matrix3, Rotation3, Unit, Vector3};

struct Target<'a> {
    camera: &'a FrameCamera,
    cell: usize,
    width: usize,
    height: usize,
    occupied: Vec<bool>,
    area: usize,
    stereo_samples: Vec<([usize; 2], f64)>,
    pixels: Vec<bool>,
    image_width: usize,
    image_height: usize,
    nearest_depth: Vec<f64>,
    front: Vec<[f64; 3]>,
    centre: [f64; 2],
    span: [f64; 2],
}

impl<'a> Target<'a> {
    fn new(
        obs: &SemanticObservation,
        camera: &'a FrameCamera,
        points: &[[f64; 3]],
    ) -> Option<Self> {
        let mask = obs.mask.as_ref()?;
        let w = mask.width as usize;
        let h = mask.height as usize;
        let span = (obs.bbox[2] - obs.bbox[0]).max(obs.bbox[3] - obs.bbox[1]);
        // Use cells small enough to preserve narrow objects and coarse enough
        // that the bounded fitting vertex sample covers the visible shape.
        let cell = (span / 22.).ceil().clamp(1., 6.) as usize;
        let width = w.div_ceil(cell);
        let height = h.div_ceil(cell);
        let mut counts = vec![0u32; width * height];
        let mut foreground = vec![false; w * h];
        let spacing = DEPTH_CELL_PIXELS as usize;
        let columns = w.div_ceil(spacing);
        let mut nearest_depth = vec![f64::INFINITY; columns * h.div_ceil(spacing)];
        for point in points {
            if let Some(([x, y], z)) = project(camera, *point) {
                if x >= 0. && y >= 0. && x < w as f64 && y < h as f64 {
                    let cell = (y as usize / spacing) * columns + (x as usize / spacing);
                    nearest_depth[cell] = nearest_depth[cell].min(z);
                }
            }
        }
        let front = front_points(obs, camera, points)
            .into_iter()
            .map(|(_, p)| p)
            .collect();

        let mut centre = [0.; 2];
        let mut pixels = 0;
        let mut lo = [w, h];
        let mut hi = [0, 0];
        let mut offset = 0;
        for (i, run) in mask.counts.iter().enumerate() {
            let end = offset + *run as usize;
            if i % 2 == 1 {
                foreground[offset..end].fill(true);
                for n in offset..end {
                    let x = n % w;
                    let y = n / w;
                    counts[(y / cell) * width + x / cell] += 1;
                    centre[0] += x as f64 + 0.5;
                    centre[1] += y as f64 + 0.5;
                    pixels += 1;
                    lo = [lo[0].min(x), lo[1].min(y)];
                    hi = [hi[0].max(x), hi[1].max(y)];
                }
            }
            offset = end;
        }
        if pixels == 0 {
            return None;
        }
        let occupied: Vec<_> = counts
            .into_iter()
            .map(|n| n as usize * 2 >= cell * cell)
            .collect();
        let area = occupied.iter().filter(|x| **x).count();
        if area == 0 {
            return None;
        }
        Some(Self {
            camera,
            width,
            height,
            cell,
            occupied,
            area,
            stereo_samples: Vec::new(),
            pixels: foreground,
            image_width: w,
            image_height: h,
            nearest_depth,
            front,
            centre: centre.map(|p| p / pixels as f64),
            span: [(hi[0] + 1 - lo[0]) as f64, (hi[1] + 1 - lo[1]) as f64],
        })
    }

    fn ray(&self) -> (Vector3<f64>, Vector3<f64>) {
        let (eye, right, down, forward) = camera_axes(self.camera);
        let [u, v] = [0, 1]
            .map(|i| (self.centre[i] - self.camera.principal_point[i]) / self.camera.focal_pixels);
        (
            Vector3::from(eye),
            (Vector3::from(forward) + Vector3::from(right) * u + Vector3::from(down) * v)
                .normalize(),
        )
    }

    fn score(
        &self,
        vertices: &[[f64; 3]],
        centre: Vector3<f64>,
        rotation: &Matrix3<f64>,
        half_size: Vector3<f64>,
    ) -> f64 {
        let mut raster = vec![false; self.occupied.len()];
        let mut surface = if self.stereo_samples.is_empty() {
            Vec::new()
        } else {
            vec![f64::INFINITY; self.occupied.len()]
        };
        let mut hits = 0;
        let mut covered = 0;
        let mut visible = 0usize;
        let mut inside = 0usize;
        let mut violations = 0usize;
        let step = DEPTH_CELL_PIXELS as usize;
        let cols = self.image_width.div_ceil(step);
        for v in vertices {
            if let Some(([x, y], depth)) = project(self.camera, *v) {
                if x < 0. || y < 0. {
                    continue;
                }
                if x < self.image_width as f64 && y < self.image_height as f64 {
                    let hit = self.pixels[y as usize * self.image_width + x as usize];
                    let nearest =
                        self.nearest_depth[(y as usize / step) * cols + x as usize / step];
                    let behind = nearest.is_finite() && depth > nearest * (1. + DEPTH_MARGIN);
                    let before = nearest.is_finite() && depth < nearest * (1. - DEPTH_MARGIN);
                    violations += usize::from(hit && behind || !hit && before);
                    if !behind {
                        visible += 1;
                        inside += usize::from(hit);
                    }
                }
                let u = x as usize / self.cell;
                let v = y as usize / self.cell;
                if u >= self.width || v >= self.height {
                    continue;
                }
                let k = v * self.width + u;
                if !surface.is_empty() {
                    surface[k] = surface[k].min(depth);
                }
                if !raster[k] {
                    raster[k] = true;
                    hits += usize::from(self.occupied[k]);
                    covered += 1;
                }
            }
        }
        let silhouette = hits as f64 / (self.area + covered - hits).max(1) as f64;
        let depth_penalty =
            (violations as f64 / vertices.len() as f64 - MAX_DEPTH_VIOLATION_SHARE).max(0.);
        let mask_penalty = (MIN_INSIDE_SHARE - inside as f64 / visible.max(1) as f64).max(0.);
        let mut distances: Vec<_> = self
            .front
            .iter()
            .map(|p| {
                let local = rotation.transpose() * (Vector3::from(*p) - centre);
                (local.abs() - half_size).sup(&Vector3::zeros()).norm()
                    / (2. * half_size.z).max(1e-9)
            })
            .collect();
        distances.sort_by(f64::total_cmp);
        let landmark_penalty = (distances.get(distances.len() / 2).copied().unwrap_or(0.)
            - MAX_LANDMARK_RESIDUAL)
            .clamp(0., 1.);
        // Compare the observed feature depth against the first candidate
        // surface in that pixel cell. Missing coverage receives a bounded cost.
        let stereo_error = self
            .stereo_samples
            .iter()
            .map(|([x, y], depth)| {
                let z = surface[(y / self.cell) * self.width + x / self.cell];
                ((z - depth).abs() / half_size.norm().max(1e-9)).min(1.)
            })
            .sum::<f64>()
            / self.stereo_samples.len().max(1) as f64;
        silhouette - depth_penalty - mask_penalty - landmark_penalty - stereo_error
    }
}

fn triangulate(targets: &[Target]) -> Option<Vector3<f64>> {
    let mut system = Matrix3::zeros();
    let mut right = Vector3::zeros();
    let rays: Vec<_> = targets.iter().map(Target::ray).collect();
    // Parallel views do not supply depth, even if their masks agree perfectly.
    if !rays.iter().any(|(_, a)| {
        rays.iter()
            .any(|(_, b)| a.cross(b).norm() > 1f64.to_radians().sin())
    }) {
        return None;
    }
    for (eye, direction) in &rays {
        let projection = Matrix3::identity() - direction * direction.transpose();
        system += projection;
        right += projection * eye;
    }
    let centre = system.lu().solve(&right)?;
    rays.iter()
        .all(|(eye, direction)| (centre - eye).dot(direction) > 0.)
        .then_some(centre)
}

#[derive(Clone, Copy)]
struct Pose {
    centre: Vector3<f64>,
    scale: f64,
    yaw: f64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fit(
    request: &GenerationRequest,
    bounds: MeshBounds,
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
    observations: &[(&SemanticObservation, &FrameCamera)],
    up: [f64; 3],
    supports: &[SupportPlane],
    points: &[[f64; 3]],
    baseline: Option<&ObjectPlacement>,
    frames: &[crate::multi_view_objects::FramePixels],
) -> Option<(ObjectPlacement, usize, f64)> {
    let mut views: Vec<_> = observations
        .iter()
        .copied()
        .filter(|(obs, _)| obs.mask.is_some() && obs.mask_confidence.unwrap_or_default() >= 0.5)
        .collect();
    views.sort_by_key(|(_, cam)| cam.frame_index);
    views.dedup_by_key(|(_, cam)| cam.frame_index);
    if views.len() < 3 {
        return None;
    }
    let candidate_holdouts: Vec<_> = views
        .iter()
        .enumerate()
        .filter(|(_, (_, cam))| cam.frame_index != request.frame_index)
        .map(|(index, _)| index)
        .collect();
    let heldout = views.remove(candidate_holdouts[candidate_holdouts.len() / 2]);
    let primary = views
        .iter()
        .find(|(_, cam)| cam.frame_index == request.frame_index)?
        .1;
    let mut targets = Vec::new();
    // Bound the optimizer cost while retaining time-separated evidence.
    let count = views.len().min(8);
    let mut fit_views = Vec::new();
    for i in 0..count {
        let (obs, cam) = views[i * (views.len() - 1) / (count - 1)];
        fit_views.push((obs, cam));
        targets.push(Target::new(obs, cam, points)?);
    }
    let start = triangulate(&targets)?;
    let mut ranked: Vec<_> = (0..targets.len()).collect();
    ranked.sort_by_key(|&i| std::cmp::Reverse(targets[i].area * targets[i].cell * targets[i].cell));
    let mut stereo_factors = Vec::new();
    let mut patch_count = 0usize;
    // A partly occluded object may have no distinctive texture in its largest
    // mask. Consult up to three fitting views instead of relying on one crop.
    for &reference in ranked.iter().take(3) {
        let target = &targets[reference];
        let initial = project(target.camera, start.into())?.1;
        let span = target.span[0].max(target.span[1]) * initial / target.camera.focal_pixels;
        let reference_ray = (start - Vector3::from(camera_axes(target.camera).0)).normalize();
        // Choose nearby views that offer measurable parallax but keep patch
        // appearance stable under rotation, occlusion and foreshortening.
        let mut neighbors: Vec<_> = fit_views
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if i == reference {
                    return None;
                }
                let ray = (start - Vector3::from(camera_axes(v.1).0)).normalize();
                let angle = reference_ray.dot(&ray).clamp(-1., 1.).acos().to_degrees();
                (angle > 1. && angle < 35.).then_some(((angle - 8.).abs(), v))
            })
            .collect();
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0));
        let subset = std::iter::once(fit_views[reference])
            .chain(neighbors.into_iter().take(2).map(|(_, v)| v))
            .collect::<Vec<_>>();
        let band = crate::multi_view_objects::stereo::registered_depths(
            target.camera.frame_index,
            &subset,
            frames,
            initial,
            span,
        );
        if let Some(band) = band {
            stereo_factors.push((band.near + band.far) * 0.5 / initial);
            patch_count += band.samples.len();
            targets[reference].stereo_samples = band.samples;
        }
    }
    let (eye, right, down, forward) = camera_axes(primary);
    let base = rotation_between(scaled(down, -1.), up)
        * Matrix3::from_columns(&[
            Vector3::from(scaled(forward, -1.)),
            Vector3::from(right),
            Vector3::from(scaled(down, -1.)),
        ]);
    let up_axis = Unit::new_normalize(Vector3::from(up));
    let centre = Vector3::from([0, 1, 2].map(|i| (bounds.min[i] + bounds.max[i]) * 0.5));
    let sampled: Vec<_> = vertices
        .iter()
        .step_by(vertices.len().div_ceil(1536).max(1))
        .map(|p| base * (Vector3::from(*p) - centre))
        .collect();
    let dimensions = Vector3::from(bounds.max) - Vector3::from(bounds.min);
    let mut scales = Vec::new();
    for t in &targets {
        let (_, depth) = project(t.camera, start.into())?;
        for ratio in [t.span[0] / dimensions.y, t.span[1] / dimensions.z] {
            let value = ratio * depth / t.camera.focal_pixels;
            if value.is_finite() && value > 0. {
                scales.push(value);
            }
        }
    }
    scales.sort_by(f64::total_cmp);
    let scale = *scales.get(scales.len() / 2)?;
    let rotation = |yaw| Rotation3::from_axis_angle(&up_axis, yaw);
    let score = |p: Pose| {
        let r = rotation(p.yaw);
        let world: Vec<_> = sampled
            .iter()
            .map(|v| (p.centre + r * v * p.scale).into())
            .collect();
        let matrix = r.matrix() * base;
        let scores: Vec<_> = targets
            .iter()
            .map(|t| t.score(&world, p.centre, &matrix, dimensions * p.scale * 0.5))
            .collect();
        (scores.iter().sum::<f64>() / scores.len() as f64
            + scores.iter().copied().fold(f64::INFINITY, f64::min))
            * 0.5
    };
    let mut best = Pose {
        centre: start,
        scale,
        yaw: 0.,
    };
    let mut best_score = score(best);
    if let Some(existing) = baseline {
        let pose = Pose {
            centre: Vector3::from(existing.translation) + base * centre * existing.scale,
            scale: existing.scale,
            yaw: 0.,
        };
        let value = score(pose);
        if value > best_score {
            best = pose;
            best_score = value;
        }
    }
    let direction = (start - Vector3::from(eye)).normalize();
    let view_distance = (start - Vector3::from(eye)).norm();
    let mut depth_factors = vec![0.8, 1., 1.25];
    depth_factors.extend(stereo_factors);
    for target in &targets {
        if target.front.len() < 3 {
            continue;
        }
        let mut depth: Vec<_> = target
            .front
            .iter()
            .map(|p| (Vector3::from(*p) - Vector3::from(eye)).dot(&direction) / view_distance)
            .collect();
        depth.sort_by(f64::total_cmp);
        let middle = depth[depth.len() / 2];
        if middle > 0. && middle.is_finite() {
            depth_factors.push(middle);
        }
    }
    for depth_factor in depth_factors {
        for size in [0.8, 1., 1.25] {
            for n in 0..8 {
                let p = Pose {
                    centre: start + direction * (depth_factor - 1.) * view_distance,
                    scale: scale * depth_factor * size,
                    yaw: n as f64 * std::f64::consts::FRAC_PI_4,
                };
                let value = score(p);
                if value > best_score {
                    best_score = value;
                    best = p;
                }
            }
        }
    }
    for fraction in [0.16, 0.08, 0.04, 0.02] {
        for _ in 0..2 {
            let previous = best;
            for sign in [-1., 1.] {
                for axis in 0..5 {
                    let mut p = previous;
                    let f = sign * fraction;
                    match axis {
                        0 => p.centre += Vector3::from(right) * scale * dimensions.y * f,
                        1 => p.centre += Vector3::from(down) * scale * dimensions.z * f,
                        2 => {
                            p.centre += direction * (view_distance * f);
                            p.scale *= 1. + f;
                        }
                        3 => p.scale *= 1. + f,
                        _ => p.yaw += f * std::f64::consts::PI,
                    }
                    let s = score(p);
                    if s > best_score {
                        best_score = s;
                        best = p;
                    }
                }
            }
        }
    }
    let matrix = rotation(best.yaw).matrix() * base;
    let t = best.centre - matrix * centre * best.scale;
    let world: Vec<_> = vertices
        .iter()
        .map(|v| (matrix * Vector3::from(*v) * best.scale + t).into())
        .collect();
    let iou = crate::multi_view_objects::heldout_mesh_iou(heldout.1, heldout.0, &world, faces)?;
    let frames = targets
        .iter()
        .map(|t| t.camera.frame_index.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let reason=format!("Registered candidate to silhouettes in frames {frames}; fitting silhouette/depth score {best_score:.2} with {patch_count} matched image patch depths, independent frame {} silhouette IoU {iou:.2}",heldout.1.frame_index);
    let mut placement = placed_transform(
        request,
        matrix,
        best.scale,
        t.into(),
        best.centre.into(),
        up,
        dimensions.z,
        supports,
        ScaleSource::MultiViewSilhouettes,
        0,
        reason,
    );
    placement.silhouette_registration = Some(SilhouetteRegistration {
        matched_patch_depths: patch_count,
        fitting_frames: targets.iter().map(|t| t.camera.frame_index).collect(),
        heldout_frame: heldout.1.frame_index,
        heldout_mask_iou: iou,
    });
    Some((placement, heldout.1.frame_index, iou))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_project::ContentHash;
    use crate::semantic_inventory::{MaskEncoding, SemanticMask};

    fn rectangle(frame: usize, x: usize) -> SemanticObservation {
        let mut counts = vec![0u64];
        let mut value = false;
        for y in 0..100 {
            for column in 0..100 {
                let is_foreground = (x..x + 40).contains(&column) && (30..70).contains(&y);
                if is_foreground != value {
                    counts.push(0);
                    value = is_foreground;
                }
                *counts.last_mut().unwrap() += 1;
            }
        }
        SemanticObservation {
            id: format!("obs{frame}"),
            label: "panel".into(),
            frame_index: frame,
            bbox: [x as f64, 30., (x + 40) as f64, 70.],
            detection_confidence: 0.9,
            mask_confidence: Some(0.9),
            mask: Some(SemanticMask {
                encoding: MaskEncoding::RleRowMajor,
                width: 100,
                height: 100,
                counts,
            }),
        }
    }

    #[test]
    fn stereo_texture_distinguishes_equally_projected_surfaces_at_different_depths() {
        let cameras = [0, 1].map(|i| FrameCamera {
            frame_index: i,
            rotation: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
            translation: [-0.5 * i as f64, 0., 0.],
            focal_pixels: 100.,
            principal_point: [50., 50.],
        });
        let obs = [rectangle(0, 40), rectangle(1, 30)];
        let views = obs.iter().zip(&cameras).collect::<Vec<_>>();
        let images = [0, 1].map(|i| {
            (0..10_000usize)
                .flat_map(|p| {
                    let x = p % 100 + 10 * i;
                    let y = p / 100;
                    let v = ((x * x + 11 * x * y + 3 * y * y) % 251) as u8;
                    [v, v, v, 255]
                })
                .collect::<Vec<_>>()
        });
        let frames = [
            (0, 100, 100, images[0].as_slice()),
            (1, 100, 100, images[1].as_slice()),
        ];
        let band = crate::multi_view_objects::stereo::registered_depths(0, &views, &frames, 7., 2.)
            .unwrap();
        assert!(band.samples.len() > 8);
        let mut t = Target::new(&obs[0], &cameras[0], &[]).unwrap();
        t.stereo_samples = band.samples;
        let score = |z: f64| {
            let vertices = (30..70)
                .flat_map(|y| {
                    (40..80).map(move |x| {
                        [
                            (x as f64 + 0.5 - 50.) * z / 100.,
                            (y as f64 + 0.5 - 50.) * z / 100.,
                            z,
                        ]
                    })
                })
                .collect::<Vec<_>>();
            t.score(
                &vertices,
                Vector3::new(0., 0., z),
                &Matrix3::identity(),
                Vector3::new(1., 1., 1.),
            )
        };
        assert!(score(5.) > score(7.) + 0.6);
    }

    #[test]
    fn heldout_silhouette_cannot_move_a_fitted_mesh() {
        let cameras = [-0.5, 0., 0.5].map(|x| FrameCamera {
            frame_index: ((x + 0.5) * 2.) as usize,
            rotation: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
            translation: [-x, 0., 0.],
            focal_pixels: 100.,
            principal_point: [50., 50.],
        });
        let mut observations = [rectangle(0, 40), rectangle(1, 30), rectangle(2, 20)];
        let request = GenerationRequest {
            request_id: "r".into(),
            object: "panel".into(),
            label: "panel".into(),
            observation: "obs0".into(),
            frame_index: 0,
            timestamp_seconds: None,
            source_frame_hash: ContentHash::of_bytes(b"pixels"),
            width: 100,
            height: 100,
            crop: [40, 30, 80, 70],
            mask_hash: ContentHash::of_bytes(b"mask"),
            mask: observations[0].mask.clone().unwrap(),
            observed_points: 0,
        };
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for j in 0..=25 {
            for i in 0..=25 {
                vertices.push([0., i as f64 / 12.5 - 1., j as f64 / 12.5 - 1.]);
                let n = j * 26 + i;
                if j > 0 && i > 0 {
                    faces.extend([[n - 27, n - 26, n], [n - 27, n, n - 1]]);
                }
            }
        }
        let solve = |obs: &[SemanticObservation]| {
            let views = obs.iter().zip(&cameras).collect::<Vec<_>>();
            fit(
                &request,
                MeshBounds::of(&vertices).unwrap(),
                &vertices,
                &faces,
                &views,
                [0., -1., 0.],
                &[],
                &[],
                None,
                &[],
            )
            .unwrap()
        };
        let (first, holdout, correct) = solve(&observations);
        assert_eq!(holdout, 2);
        assert!(correct > 0.8, "{correct}");
        observations[2] = rectangle(2, 60);
        let (second, _, wrong) = solve(&observations);
        assert_eq!(first.translation, second.translation);
        assert_eq!(first.rotation, second.rotation);
        assert_eq!(first.scale, second.scale);
        assert!(wrong < 0.1, "{wrong}");
    }
}
