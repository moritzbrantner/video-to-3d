//! A masked visual hull in the gauge of the accepted cameras. Multiple source
//! silhouettes carve the volume; the unobserved interior is a shape hypothesis.
use crate::object_identity::{
    project, FrameCamera, MotionClass, ObjectCategory, ObjectIdentityArtifact,
};
use crate::placement_validation::{
    validate_placement, PlacedMesh, PlacementValidation, PlacementVerdict,
};
use crate::semantic_inventory::{SemanticInventoryArtifact, SemanticObservation};
use nalgebra::Vector3;
use std::collections::{BTreeMap, BTreeSet};

type TriangleSurface = (Vec<[f64; 3]>, Vec<[u32; 3]>);

mod export;
mod plane;
pub(crate) mod stereo;
pub use export::{hull_glb, FramePixels};
use plane::plane_patch;

pub struct HullOutcome {
    pub object: String,
    pub label: String,
    pub reason: String,
    pub views: Vec<usize>,
    pub holdout_frame: Option<usize>,
    pub holdout_mask_iou: Option<f64>,
    pub mesh: Option<HullMesh>,
}

pub struct HullMesh {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub validation: PlacementValidation,
}

struct View<'a> {
    cam: &'a FrameCamera,
    obs: &'a SemanticObservation,
    mask: Vec<bool>,
    width: usize,
    height: usize,
}

impl<'a> View<'a> {
    fn new(cam: &'a FrameCamera, obs: &'a SemanticObservation) -> Option<Self> {
        let rle = obs.mask.as_ref()?;
        let mask: Vec<_> = rle
            .counts
            .iter()
            .enumerate()
            .flat_map(|(i, n)| std::iter::repeat_n(i % 2 == 1, *n as usize))
            .collect();
        Some(Self {
            cam,
            obs,
            mask,
            width: rle.width as usize,
            height: rle.height as usize,
        })
    }

    fn pixel(&self, xyz: [f64; 3]) -> Option<usize> {
        let ([u, v], _) = project(self.cam, xyz)?;
        if !(u >= 0. && v >= 0. && u < self.width as f64 && v < self.height as f64) {
            return None;
        }
        Some(v as usize * self.width + u as usize)
    }

    fn contains(&self, xyz: [f64; 3]) -> bool {
        self.pixel(xyz).is_some_and(|i| self.mask[i])
    }
}

fn camera_axes(cam: &FrameCamera) -> [Vector3<f64>; 3] {
    [0, 1, 2].map(|i| Vector3::from_row_slice(&cam.rotation[3 * i..3 * i + 3]))
}

fn camera_center(cam: &FrameCamera) -> Vector3<f64> {
    let [right, down, forward] = camera_axes(cam);
    -(right * cam.translation[0] + down * cam.translation[1] + forward * cam.translation[2])
}

pub fn reconstruct_hulls(
    inventory: &SemanticInventoryArtifact,
    identities: &ObjectIdentityArtifact,
    cameras: &[FrameCamera],
    points: &[[f64; 3]],
) -> Vec<HullOutcome> {
    reconstruct_hulls_with_frames(inventory, identities, cameras, points, &[])
}

pub fn reconstruct_hulls_with_frames(
    inventory: &SemanticInventoryArtifact,
    identities: &ObjectIdentityArtifact,
    cameras: &[FrameCamera],
    points: &[[f64; 3]],
    frames: &[FramePixels<'_>],
) -> Vec<HullOutcome> {
    let by_id: BTreeMap<_, _> = inventory
        .observations
        .iter()
        .map(|o| (o.id.as_str(), o))
        .collect();
    let by_frame: BTreeMap<_, _> = cameras.iter().map(|c| (c.frame_index, c)).collect();
    let ambiguous: BTreeSet<_> = identities
        .ambiguities
        .iter()
        .flat_map(|a| &a.objects)
        .collect();
    identities
        .objects
        .iter()
        .map(|obj| {
            let mut outcome = HullOutcome {
                object: obj.id.clone(),
                label: obj.label.clone(),
                reason: String::new(),
                views: Vec::new(),
                holdout_frame: None,
                holdout_mask_iou: None,
                mesh: None,
            };
            let mut seen_frames = BTreeSet::new();
            let views: Vec<_> = obj
                .observations
                .iter()
                .filter_map(|id| {
                    let obs = *by_id.get(id.as_str())?;
                    let camera = *by_frame.get(&obs.frame_index)?;
                    if !seen_frames.insert(obs.frame_index) {
                        return None;
                    }
                    View::new(camera, obs)
                })
                .collect();
            outcome.views = views.iter().map(|v| v.cam.frame_index).collect();
            let attempt = if obj.motion == MotionClass::Moving {
                Err("The track moves relative to the registered cameras.")
            } else {
                reconstruct_one(
                    &views,
                    points,
                    obj.category == ObjectCategory::Structure,
                    &mut outcome,
                    frames,
                )
            };
            if let Err(reason) = attempt {
                outcome.reason = reason.into();
            }
            if ambiguous.contains(&obj.id) {
                if let Some(mesh) = &mut outcome.mesh {
                    if mesh.validation.verdict == PlacementVerdict::Accepted {
                        mesh.validation.verdict = PlacementVerdict::Approximate;
                    }
                    mesh.validation.needs_review = true;
                    mesh.validation
                        .reason
                        .push_str(" Object identity is disputed.");
                }
                outcome
                    .reason
                    .push_str(" Object identity is disputed; review the source track.");
            }
            outcome
        })
        .collect()
}

fn ray_at(camera: &FrameCamera, pixel: [f64; 2]) -> Vector3<f64> {
    let [right, down, forward] = camera_axes(camera);
    forward
        + right * ((pixel[0] - camera.principal_point[0]) / camera.focal_pixels)
        + down * ((pixel[1] - camera.principal_point[1]) / camera.focal_pixels)
}

// When SfM misses a thin or untextured object, intersect its measured silhouette
// centre rays. An ill-conditioned or visibly inconsistent intersection supplies no depth.
fn triangulated_depth(views: &[&View<'_>], reference: &FrameCamera) -> Option<f64> {
    let mut a = nalgebra::Matrix3::zeros();
    let mut b = Vector3::zeros();
    for v in views {
        let mut xy = [0.; 2];
        let mut count = 0;
        for i in (0..v.mask.len()).filter(|i| v.mask[*i]) {
            xy[0] += (i % v.width) as f64 + 0.5;
            xy[1] += (i / v.width) as f64 + 0.5;
            count += 1;
        }
        if count == 0 {
            return None;
        }
        let ray = ray_at(v.cam, xy.map(|sum| sum / count as f64)).normalize();
        let constraint = nalgebra::Matrix3::identity() - ray * ray.transpose();
        a += constraint;
        b += constraint * camera_center(v.cam);
    }
    let eig = a.symmetric_eigen().eigenvalues;
    if eig.min() < 1e-4 * eig.max() {
        return None;
    }
    let p: [f64; 3] = a.lu().solve(&b)?.into();
    for view in views {
        let ([x, y], _) = project(view.cam, p)?;
        let [x0, y0, x1, y1] = view.obs.bbox;
        if x < x0 || x > x1 || y < y0 || y > y1 {
            return None;
        }
    }
    Some(project(reference, p)?.1)
}

fn reconstruct_one(
    views: &[View<'_>],
    points: &[[f64; 3]],
    architecture: bool,
    outcome: &mut HullOutcome,
    frames: &[FramePixels<'_>],
) -> Result<(), &'static str> {
    // Reserve one view in temporal order, away from the endpoints used for baseline.
    // It constrains neither the volume bounds nor the carved shape.
    if views.len() < if architecture { 1 } else { 2 } {
        return Err("Too few registered masks to reconstruct this surface.");
    }
    let holdout = (views.len() >= 3).then_some(views.len() / 2);
    let fit: Vec<_> = views
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (Some(i) != holdout).then_some(v))
        .collect();
    let reference = fit
        .iter()
        .max_by_key(|v| v.mask.iter().filter(|p| **p).count())
        .ok_or("No reference mask.")?;
    let required = (fit.len() * 4).div_ceil(5);
    let support: Vec<_> = points
        .iter()
        .copied()
        .filter(|p| fit.iter().filter(|v| v.contains(*p)).count() >= required)
        .collect();
    let (vertices, triangles, method) = if architecture {
        let (vertices, triangles) = plane_patch(reference, &fit, required, &support)?;
        (vertices, triangles, String::from("mask-trimmed plane"))
    } else {
        let mut depths: Vec<_> = support
            .iter()
            .filter_map(|p| project(reference.cam, *p).map(|(_, d)| d))
            .collect();
        let (depth, method) = if depths.len() >= 3 {
            depths.sort_by(f64::total_cmp);
            (
                depths[depths.len() / 2],
                "visual hull anchored by observed depth",
            )
        } else {
            let depth = triangulated_depth(&fit, reference.cam).ok_or(
                "Neither observed landmarks nor a stable intersection of silhouette rays determines object depth.",
            )?;
            (depth, "visual hull anchored by triangulated silhouettes")
        };
        let [x0, y0, x1, y1] = reference.obs.bbox;
        let f = reference.cam.focal_pixels;
        let span = (x1 - x0).max(y1 - y0) * depth / f;
        if !span.is_finite() || span <= 0. {
            return Err("Mask and camera have no finite object extent.");
        }
        let stereo = stereo::depth_band(reference, &fit, frames, depth, span);
        let depth = stereo
            .as_ref()
            .map_or(depth, |band| (band.near + band.far) * 0.5);
        let span = (x1 - x0).max(y1 - y0) * depth / f;
        let thickness = stereo
            .as_ref()
            .map_or(span * 1.5, |band| band.far - band.near);
        let method = stereo.as_ref().map_or_else(
            || method.to_string(),
            |band| {
                format!(
                    "visual hull bounded by {} multi-view image patch depths",
                    band.matches
                )
            },
        );
        let [right, down, forward] = camera_axes(reference.cam);
        let [cx, cy] = reference.cam.principal_point;
        let center = camera_center(reference.cam)
            + depth
                * (forward
                    + right * ((x0 + x1) * 0.5 - cx) / f
                    + down * ((y0 + y1) * 0.5 - cy) / f);
        let rays: Vec<_> = fit
            .iter()
            .map(|v| (center - camera_center(v.cam)).normalize())
            .collect();
        let min_cos = rays
            .iter()
            .enumerate()
            .flat_map(|(i, a)| rays[..i].iter().map(move |b| a.dot(b)))
            .fold(1.0, f64::min);
        let separation = min_cos.clamp(-1., 1.).acos().to_degrees();
        if separation < 2. {
            return Err(
                "The registered cameras have too little angular separation to constrain shape.",
            );
        }
        // A monocular arc may leave the cone intersection unbounded behind the object.
        // Restrict depth to a region around the corroborated landmarks and apparent size.
        // A fixed coarse grid can discard a thin leg that spans several camera pixels.
        // Keep roughly a source pixel per cell where the available image detail allows it.
        let mask_bounds = reference
            .obs
            .mask
            .as_ref()
            .and_then(|mask| mask.foreground_bounds())
            .unwrap_or(reference.obs.bbox);
        let pixel_extent = (mask_bounds[2] - mask_bounds[0]).max(mask_bounds[3] - mask_bounds[1]);
        let resolution = (pixel_extent * 1.5).ceil().clamp(40., 88.) as usize;
        let step = span * 1.5 / (resolution as f64 - 3.);
        let offset = (resolution - 1) as f64 / 2.;
        let world = |p: [f64; 3]| -> [f64; 3] {
            (center
                + step
                    * (right * (p[0] - offset)
                        + down * (p[1] - offset)
                        + forward * (p[2] - offset) * (thickness / (span * 1.5))))
                .into()
        };
        let mut occupied = vec![false; resolution * resolution * resolution];
        for z in 1..resolution - 1 {
            for y in 1..resolution - 1 {
                for x in 1..resolution - 1 {
                    let point = world([x as f64, y as f64, z as f64]);
                    let mut available = 0;
                    let mut inside = 0;
                    for view in &fit {
                        if let Some(pixel) = view.pixel(point) {
                            let vote = view.mask[pixel];
                            available += 1;
                            inside += usize::from(vote);
                        }
                    }
                    occupied[x + resolution * (y + resolution * z)] =
                        inside >= 2 && inside * 5 >= available * 4;
                }
            }
        }
        let (vertices, triangles) = surface_nets(&occupied, resolution, world);
        (
            vertices,
            triangles,
            format!("{method} (camera rays span {separation:.1}°)"),
        )
    };
    if triangles.is_empty() {
        return Err(
            "The registered silhouettes have no shared volume in the observed depth range.",
        );
    }
    let checked: Vec<_> = views.iter().map(|v| (v.obs, v.cam)).collect();
    let placed = PlacedMesh {
        rotation: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        scale: 1.,
        translation: [0.; 3],
        vertices: &vertices,
        reference_frame: reference.cam.frame_index,
    };
    let mut validation = validate_placement(&placed, &checked, points);
    if let Some(index) = holdout {
        let test = &views[index];
        let iou = heldout_mesh_iou(test.cam, test.obs, &vertices, &triangles)
            .ok_or("Missing held-out mask.")?;
        outcome.holdout_frame = Some(test.cam.frame_index);
        outcome.holdout_mask_iou = Some(iou);
        if iou < 0.4 {
            validation.verdict = PlacementVerdict::Contradicted;
            validation.needs_review = true;
            validation.reason = format!("Held-out silhouette IoU {iou:.2} is below 0.40.");
        }
        outcome.reason = format!(
            "{method} from {} views; held-out frame {} mask IoU {iou:.2}. {}",
            fit.len(),
            test.cam.frame_index,
            validation.reason
        );
    } else {
        if validation.verdict == PlacementVerdict::Accepted {
            validation.verdict = PlacementVerdict::Approximate;
        }
        validation.needs_review = true;
        validation
            .reason
            .push_str(" No independent mask remains to validate this object.");
        outcome.reason = format!("{method} from {} views. {}", fit.len(), validation.reason);
    }
    outcome.mesh = Some(HullMesh {
        vertices,
        triangles,
        validation,
    });
    Ok(())
}

// One surface vertex per boundary cell at the mean of its crossed edges.
// The four cells around each sign-changing lattice edge form a quad.
fn surface_nets(
    field: &[bool],
    n: usize,
    world: impl Fn([f64; 3]) -> [f64; 3],
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let stride = [1, n, n * n];
    let mut ids = vec![u32::MAX; field.len()];
    let mut vertices = Vec::new();
    for z in 0..n - 1 {
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let base = x + n * (y + n * z);
                let mut sum = [0.; 3];
                let mut count = 0;
                for corner in 0..8 {
                    let d = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
                    let i = base + d[0] + n * (d[1] + n * d[2]);
                    for axis in 0..3 {
                        if d[axis] == 0 && field[i] != field[i + stride[axis]] {
                            for k in 0..3 {
                                sum[k] += d[k] as f64;
                            }
                            sum[axis] += 0.5;
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    ids[base] = vertices.len() as u32;
                    let origin = [x, y, z];
                    vertices.push(world(
                        [0, 1, 2].map(|k| origin[k] as f64 + sum[k] / count as f64),
                    ));
                }
            }
        }
    }
    let mut faces = Vec::new();
    for z in 1..n - 1 {
        for y in 1..n - 1 {
            for x in 1..n - 1 {
                let i = x + n * (y + n * z);
                for axis in 0..3 {
                    if field[i] == field[i + stride[axis]] {
                        continue;
                    }
                    let a = stride[(axis + 1) % 3];
                    let b = stride[(axis + 2) % 3];
                    let mut quad = [ids[i], ids[i - a], ids[i - a - b], ids[i - b]];
                    if !field[i] {
                        quad.reverse();
                    }
                    faces.push([quad[0], quad[1], quad[2]]);
                    faces.push([quad[0], quad[2], quad[3]]);
                }
            }
        }
    }
    (vertices, faces)
}

/// Raster overlap against a mask that was excluded from fitting.
pub(crate) fn heldout_mesh_iou(
    camera: &FrameCamera,
    observation: &SemanticObservation,
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
) -> Option<f64> {
    Some(mask_iou(&View::new(camera, observation)?, vertices, faces))
}

fn mask_iou(view: &View<'_>, vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> f64 {
    let uv: Vec<_> = vertices
        .iter()
        .map(|p| project(view.cam, *p).map(|(uv, _)| uv))
        .collect();
    let mut raster = vec![false; view.mask.len()];
    let edge = |a: [f64; 2], b: [f64; 2], x: f64, y: f64| {
        (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
    };
    for face in faces {
        let [Some(a), Some(b), Some(c)] = face.map(|i| uv[i as usize]) else {
            continue;
        };
        let signed_area = edge(a, b, c[0], c[1]);
        if signed_area.abs() < 1e-9 {
            continue;
        }
        let sign = signed_area.signum();
        let xs = a[0].min(b[0]).min(c[0]).floor().max(0.) as usize
            ..(a[0].max(b[0]).max(c[0]).ceil() as usize).min(view.width);
        let ys = a[1].min(b[1]).min(c[1]).floor().max(0.) as usize
            ..(a[1].max(b[1]).max(c[1]).ceil() as usize).min(view.height);
        for y in ys {
            for x in xs.clone() {
                let (u, v) = (x as f64 + 0.5, y as f64 + 0.5);
                if [edge(a, b, u, v), edge(b, c, u, v), edge(c, a, u, v)]
                    .iter()
                    .all(|e| e * sign >= 0.)
                {
                    raster[y * view.width + x] = true;
                }
            }
        }
    }
    let mut intersection = 0;
    let mut union = 0;
    for (a, b) in raster.into_iter().zip(&view.mask) {
        intersection += usize::from(a && *b);
        union += usize::from(a || *b);
    }
    intersection as f64 / union.max(1) as f64
}

#[cfg(test)]
mod tests;
