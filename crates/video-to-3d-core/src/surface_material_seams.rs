//! Single-camera projection rule for mixed-reference seam triangles.
//!
//! A seam triangle has observed vertices that belong to different reference
//! views, so no single reference grid addresses all three corners. The rule
//! admits such a triangle into a texture only when one accepted camera, with a
//! supplied reference image, provably sees all three vertices:
//!
//! 1. The camera is a seam candidate: its accepted pose and the supplied
//!    pinhole intrinsics reproduce the camera's own accepted grid sites (median
//!    residual at most [`POSE_CONSISTENCY_PIXELS`]). A camera without own
//!    observed points has no such evidence and is never a candidate.
//! 2. Every vertex lies in front of the camera.
//! 3. Every vertex projects inside the reference image.
//! 4. The projected footprint is non-degenerate and the triangle is not seen
//!    at a grazing angle.
//! 5. Depth test against all accepted triangles: every vertex, edge midpoint
//!    and the centroid is the closest accepted surface along its pixel ray.
//!    A sample is occluded when accepted geometry is clearly in front of it,
//!    and ambiguous when the depth difference falls inside the tolerance band
//!    or an occluding depth lies in its one-pixel neighbourhood.
//!
//! The rule never blends cameras and never creates, moves or repairs
//! geometry. Among passing cameras the one owning most vertices wins, then the
//! larger projected footprint, then the lower frame index. A triangle no
//! camera admits keeps the vertex-color fallback; it is counted under the
//! furthest check any candidate reached.

use super::{Ownership, ReferenceImage};
use crate::{DenseGridSite, EvidenceCamera, Point3, ReconstructionEvidenceView};
use serde::Serialize;
use std::collections::BTreeMap;

/// Revision of the seam rule; part of every seam appearance key.
pub(super) const SEAM_RULE_REVISION: u32 = 1;
const NEAR_DEPTH: f64 = 1.0e-4;
/// Maximum median distance between the projection of a camera's own observed
/// points and their accepted grid sites.
pub const POSE_CONSISTENCY_PIXELS: f64 = 2.0;
/// Relative depth difference still treated as the same surface.
const VISIBLE_DEPTH_TOLERANCE: f64 = 0.02;
/// Relative depth difference from which accepted geometry clearly occludes.
const OCCLUDED_DEPTH_TOLERANCE: f64 = 0.05;
/// Minimum |cos| between the triangle normal and the viewing ray.
const MIN_VIEW_COSINE: f64 = 0.2;
/// Minimum projected double area, in square pixels.
const MIN_FOOTPRINT_DOUBLE_AREA: f64 = 1.0;

/// Pinhole intrinsics shared by the accepted reconstruction cameras: the
/// principal point is the image center, as everywhere in the core.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SeamProjection {
    pub focal_pixels: f32,
}

/// Why no accepted camera admitted a seam triangle, ordered by how far the
/// rule progressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum SeamRejection {
    NoCandidateCamera,
    PoseInconsistent,
    BehindCamera,
    OutOfFrame,
    GrazingView,
    Occluded,
    Ambiguous,
}

/// A seam triangle admitted by one accepted camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SeamAdmission {
    pub camera_frame: usize,
    /// Continuous pixel coordinates of corners `[a, b, c]` in the camera's
    /// reference image (pixel `x` is centered at coordinate `x`).
    pub corner_pixels: [[f64; 2]; 3],
}

#[derive(Clone, Copy)]
struct Candidate {
    owned: usize,
    double_area: f64,
    admission: SeamAdmission,
}

/// Evaluate the rule for every triangle in `seams` (indices into the evidence
/// triangles). Results are in the order of `seams`.
pub(super) fn evaluate_seams(
    evidence: &ReconstructionEvidenceView<'_>,
    ownership: &[Ownership],
    grid_sites: Option<&[DenseGridSite]>,
    images: &BTreeMap<usize, ReferenceImage<'_>>,
    projection: Option<SeamProjection>,
    seams: &[usize],
    check_canceled: &mut impl FnMut() -> Result<(), String>,
) -> Result<Vec<Result<SeamAdmission, SeamRejection>>, String> {
    let mut rejection = vec![SeamRejection::NoCandidateCamera; seams.len()];
    let mut best: Vec<Option<Candidate>> = vec![None; seams.len()];
    let (Some(projection), Some(grid_sites)) = (projection, grid_sites) else {
        return Ok(rejection.into_iter().map(Err).collect());
    };
    if seams.is_empty() {
        return Ok(Vec::new());
    }
    let focal = projection.focal_pixels as f64;
    let mut cameras = evidence.cameras.iter().collect::<Vec<_>>();
    cameras.sort_by_key(|camera| camera.frame_index);
    for camera in cameras {
        check_canceled()?;
        let Some(image) = images.get(&camera.frame_index) else {
            continue;
        };
        let view = CameraView::new(camera, focal, image.width, image.height);
        let Some(consistent) = view.reproduces_own_grid(evidence.points, ownership, grid_sites)
        else {
            continue;
        };
        if !consistent {
            for reason in &mut rejection {
                *reason = (*reason).max(SeamRejection::PoseInconsistent);
            }
            continue;
        }
        let depth = DepthBuffer::rasterize(&view, evidence, check_canceled)?;
        for (slot, triangle_index) in seams.iter().enumerate() {
            if slot % 1024 == 0 {
                check_canceled()?;
            }
            let triangle = &evidence.triangles[*triangle_index];
            let corners = [triangle.a, triangle.b, triangle.c];
            match view.admit(evidence, &depth, corners) {
                Ok((corner_pixels, double_area)) => {
                    let owned = corners
                        .iter()
                        .filter(|index| {
                            matches!(ownership[**index], Ownership::Observed { reference, .. } if reference == camera.frame_index)
                        })
                        .count();
                    let candidate = Candidate {
                        owned,
                        double_area,
                        admission: SeamAdmission {
                            camera_frame: camera.frame_index,
                            corner_pixels,
                        },
                    };
                    // Cameras are visited in ascending frame order, so only a
                    // strictly better candidate replaces the current one.
                    let better = best[slot].is_none_or(|current| {
                        (owned, double_area) > (current.owned, current.double_area)
                    });
                    if better {
                        best[slot] = Some(candidate);
                    }
                }
                Err(reason) => rejection[slot] = rejection[slot].max(reason),
            }
        }
    }
    Ok(best
        .into_iter()
        .zip(rejection)
        .map(|(best, reason)| best.map(|candidate| candidate.admission).ok_or(reason))
        .collect())
}

struct CameraView<'a> {
    camera: &'a EvidenceCamera,
    focal: f64,
    width: u32,
    height: u32,
}

impl<'a> CameraView<'a> {
    fn new(camera: &'a EvidenceCamera, focal: f64, width: u32, height: u32) -> Self {
        Self {
            camera,
            focal,
            width,
            height,
        }
    }

    fn to_camera(&self, point: [f64; 3]) -> [f64; 3] {
        let r = self.camera.rotation.map(f64::from);
        let t = self.camera.translation.map(f64::from);
        [0, 1, 2].map(|row| {
            r[row * 3] * point[0] + r[row * 3 + 1] * point[1] + r[row * 3 + 2] * point[2] + t[row]
        })
    }

    /// Pixel coordinates of a camera-space point in front of the camera.
    fn pixel(&self, camera_point: [f64; 3]) -> [f64; 2] {
        [
            self.focal * camera_point[0] / camera_point[2] + self.width as f64 * 0.5,
            self.focal * camera_point[1] / camera_point[2] + self.height as f64 * 0.5,
        ]
    }

    fn in_front(camera_point: [f64; 3]) -> bool {
        camera_point.iter().all(|value| value.is_finite()) && camera_point[2] > NEAR_DEPTH
    }

    /// `None` when the camera owns no observed point to check against.
    fn reproduces_own_grid(
        &self,
        points: &[Point3],
        ownership: &[Ownership],
        grid_sites: &[DenseGridSite],
    ) -> Option<bool> {
        let frame = self.camera.frame_index;
        let mut residuals = Vec::new();
        for (index, owner) in ownership.iter().enumerate() {
            if !matches!(owner, Ownership::Observed { reference, .. } if *reference == frame) {
                continue;
            }
            let site = grid_sites[index];
            let camera_point = self.to_camera(position(&points[index]));
            let residual = if Self::in_front(camera_point) {
                let [u, v] = self.pixel(camera_point);
                (u - site.x as f64).hypot(v - site.y as f64)
            } else {
                f64::INFINITY
            };
            residuals.push(if residual.is_nan() {
                f64::INFINITY
            } else {
                residual
            });
        }
        if residuals.is_empty() {
            return None;
        }
        residuals.sort_by(f64::total_cmp);
        let middle = residuals.len() / 2;
        let median = if residuals.len() % 2 == 0 {
            0.5 * (residuals[middle - 1] + residuals[middle])
        } else {
            residuals[middle]
        };
        Some(median <= POSE_CONSISTENCY_PIXELS)
    }

    /// Apply checks 2-5 of the rule for one camera.
    fn admit(
        &self,
        evidence: &ReconstructionEvidenceView<'_>,
        depth: &DepthBuffer,
        corners: [usize; 3],
    ) -> Result<([[f64; 2]; 3], f64), SeamRejection> {
        let camera = corners.map(|index| self.to_camera(position(&evidence.points[index])));
        if !camera.iter().all(|point| Self::in_front(*point)) {
            return Err(SeamRejection::BehindCamera);
        }
        let pixels = camera.map(|point| self.pixel(point));
        let (max_x, max_y) = ((self.width - 1) as f64, (self.height - 1) as f64);
        if pixels
            .iter()
            .any(|[u, v]| !(0.0..=max_x).contains(u) || !(0.0..=max_y).contains(v))
        {
            return Err(SeamRejection::OutOfFrame);
        }
        let double_area = ((pixels[1][0] - pixels[0][0]) * (pixels[2][1] - pixels[0][1])
            - (pixels[1][1] - pixels[0][1]) * (pixels[2][0] - pixels[0][0]))
            .abs();
        let normal = cross(sub(camera[1], camera[0]), sub(camera[2], camera[0]));
        let centroid =
            [0, 1, 2].map(|axis| (camera[0][axis] + camera[1][axis] + camera[2][axis]) / 3.0);
        let cosine = dot(normal, centroid).abs() / (norm(normal) * norm(centroid));
        if !(double_area >= MIN_FOOTPRINT_DOUBLE_AREA && cosine >= MIN_VIEW_COSINE) {
            return Err(SeamRejection::GrazingView);
        }
        let mid = |a: [f64; 3], b: [f64; 3]| [0, 1, 2].map(|axis| (a[axis] + b[axis]) * 0.5);
        let samples = [
            camera[0],
            camera[1],
            camera[2],
            mid(camera[0], camera[1]),
            mid(camera[1], camera[2]),
            mid(camera[2], camera[0]),
            centroid,
        ];
        let plane = InverseDepthPlane::through(pixels, camera.map(|point| 1.0 / point[2]))
            .ok_or(SeamRejection::Ambiguous)?;
        let mut ambiguous = false;
        for sample in samples {
            let [u, v] = self.pixel(sample);
            match depth.classify(u, v, &plane) {
                Visibility::Visible => {}
                Visibility::Ambiguous => ambiguous = true,
                Visibility::Occluded => return Err(SeamRejection::Occluded),
            }
        }
        if ambiguous {
            return Err(SeamRejection::Ambiguous);
        }
        Ok((pixels, double_area))
    }
}

fn position(point: &Point3) -> [f64; 3] {
    [point.x as f64, point.y as f64, point.z as f64]
}

enum Visibility {
    Visible,
    Ambiguous,
    Occluded,
}

/// Inverse camera depth of the seam triangle's plane as an affine function of
/// pixel coordinates (exact under the pinhole projection):
/// `1 / z = a u + b v + c`.
struct InverseDepthPlane {
    a: f64,
    b: f64,
    c: f64,
}

impl InverseDepthPlane {
    fn through(pixels: [[f64; 2]; 3], inverse_depth: [f64; 3]) -> Option<Self> {
        let [[u0, v0], [u1, v1], [u2, v2]] = pixels;
        let [w0, w1, w2] = inverse_depth;
        let det = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0);
        let a = ((w1 - w0) * (v2 - v0) - (w2 - w0) * (v1 - v0)) / det;
        let b = ((u1 - u0) * (w2 - w0) - (u2 - u0) * (w1 - w0)) / det;
        let c = w0 - a * u0 - b * v0;
        [a, b, c]
            .iter()
            .all(|value| value.is_finite())
            .then_some(Self { a, b, c })
    }

    /// Nearest depth the plane reaches inside the pixel cell centered at
    /// `(x, y)`. Every depth the buffer records for that cell was taken within
    /// half a pixel of its center, so the triangle's own surface (and coplanar
    /// neighbours) can never be nearer than this. `None` when the plane does
    /// not stay in front of the camera over the cell.
    fn nearest_depth_in_cell(&self, x: isize, y: isize) -> Option<f64> {
        let inverse =
            self.a * x as f64 + self.b * y as f64 + self.c + 0.5 * (self.a.abs() + self.b.abs());
        (inverse.is_finite() && inverse > 0.0).then(|| 1.0 / inverse)
    }
}

/// Closest accepted surface depth per pixel (centers plus conservatively
/// stamped triangle edges); `INFINITY` is empty.
struct DepthBuffer {
    width: usize,
    height: usize,
    depth: Vec<f64>,
    /// Pixel and edge samples written since the last cancellation poll.
    unpolled_work: usize,
}

/// Rasterization work (pixels tested plus edge samples) between two
/// cancellation polls.
const RASTER_POLL_INTERVAL: usize = 1 << 16;

impl DepthBuffer {
    fn rasterize(
        view: &CameraView<'_>,
        evidence: &ReconstructionEvidenceView<'_>,
        check_canceled: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<Self, String> {
        let (width, height) = (view.width as usize, view.height as usize);
        let mut buffer = Self {
            width,
            height,
            depth: vec![f64::INFINITY; width * height],
            unpolled_work: 0,
        };
        for (ordinal, triangle) in evidence.triangles.iter().enumerate() {
            if ordinal % 1024 == 0 {
                check_canceled()?;
            }
            let camera = [triangle.a, triangle.b, triangle.c]
                .map(|index| view.to_camera(position(&evidence.points[index])));
            let polygon = clip_near(&camera);
            for fan in 1..polygon.len().saturating_sub(1) {
                buffer.fill(
                    view,
                    [polygon[0], polygon[fan], polygon[fan + 1]],
                    check_canceled,
                )?;
            }
        }
        Ok(buffer)
    }

    /// Count `work` units and poll cancellation once enough accumulated.
    fn charge(
        &mut self,
        work: usize,
        check_canceled: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<(), String> {
        self.unpolled_work += work;
        if self.unpolled_work >= RASTER_POLL_INTERVAL {
            self.unpolled_work = 0;
            check_canceled()?;
        }
        Ok(())
    }

    fn fill(
        &mut self,
        view: &CameraView<'_>,
        corners: [[f64; 3]; 3],
        check_canceled: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<(), String> {
        let pixels = corners.map(|point| view.pixel(point));
        let inverse_depth = corners.map(|point| 1.0 / point[2]);
        let edge = |a: [f64; 2], b: [f64; 2], p: [f64; 2]| {
            (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        };
        self.stamp_edges(pixels, inverse_depth, check_canceled)?;
        let area = edge(pixels[0], pixels[1], pixels[2]);
        if !area.is_finite() || area.abs() < 1.0e-12 {
            return Ok(());
        }
        let bound = |axis: usize, limit: usize| {
            let low = pixels.iter().map(|p| p[axis]).fold(f64::INFINITY, f64::min);
            let high = pixels
                .iter()
                .map(|p| p[axis])
                .fold(f64::NEG_INFINITY, f64::max);
            let low = low.ceil().max(0.0);
            let high = high.floor().min(limit as f64 - 1.0);
            (low <= high).then_some((low as usize, high as usize))
        };
        let (Some((x0, x1)), Some((y0, y1))) = (bound(0, self.width), bound(1, self.height)) else {
            return Ok(());
        };
        for y in y0..=y1 {
            self.charge(x1 - x0 + 1, check_canceled)?;
            for x in x0..=x1 {
                let p = [x as f64, y as f64];
                let weights = [
                    edge(pixels[1], pixels[2], p) / area,
                    edge(pixels[2], pixels[0], p) / area,
                    edge(pixels[0], pixels[1], p) / area,
                ];
                if weights.iter().any(|weight| *weight < -1.0e-9) {
                    continue;
                }
                let inverse = weights[0] * inverse_depth[0]
                    + weights[1] * inverse_depth[1]
                    + weights[2] * inverse_depth[2];
                if inverse <= 0.0 {
                    continue;
                }
                let slot = &mut self.depth[y * self.width + x];
                *slot = slot.min(1.0 / inverse);
            }
        }
        Ok(())
    }

    /// Conservative coverage: sampling only pixel centers misses thin or
    /// subpixel triangles, which would let a hidden seam pass as visible.
    /// Record each edge's perspective-correct depth at the pixel nearest to
    /// every sample (spacing at most a quarter pixel), so any accepted triangle
    /// reaches the depth buffer within half a pixel of its footprint.
    fn stamp_edges(
        &mut self,
        pixels: [[f64; 2]; 3],
        inverse_depth: [f64; 3],
        check_canceled: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<(), String> {
        const SPACING_PIXELS: f64 = 0.25;
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let (start, end) = (pixels[a], pixels[b]);
            // Only the part of the edge over the image (plus a pixel) can
            // stamp, which bounds the work by the image size.
            let Some((t0, t1)) = clip_segment(
                start,
                end,
                [-1.0, -1.0],
                [self.width as f64, self.height as f64],
            ) else {
                continue;
            };
            let length = (t1 - t0) * (end[0] - start[0]).hypot(end[1] - start[1]);
            if !length.is_finite() {
                continue;
            }
            let steps = (length / SPACING_PIXELS).ceil().max(1.0) as usize;
            self.charge(steps + 1, check_canceled)?;
            for step in 0..=steps {
                let t = t0 + (t1 - t0) * step as f64 / steps as f64;
                let u = start[0] + (end[0] - start[0]) * t;
                let v = start[1] + (end[1] - start[1]) * t;
                let inverse = inverse_depth[a] + (inverse_depth[b] - inverse_depth[a]) * t;
                if inverse <= 0.0 {
                    continue;
                }
                let (x, y) = (u.round(), v.round());
                if x < 0.0 || y < 0.0 || x >= self.width as f64 || y >= self.height as f64 {
                    continue;
                }
                let slot = &mut self.depth[y as usize * self.width + x as usize];
                *slot = slot.min(1.0 / inverse);
            }
        }
        Ok(())
    }

    fn at(&self, x: isize, y: isize) -> f64 {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return f64::INFINITY;
        }
        self.depth[y as usize * self.width + x as usize]
    }

    /// Classify the sample at pixel `(u, v)` on the seam triangle's `plane`.
    /// Each buffer cell is compared with the nearest depth the plane reaches
    /// inside that cell, so the triangle's own (and coplanar neighbours')
    /// depth recorded off-center never reads as an occluder on a slanted
    /// surface, while geometry clearly in front of the plane still does.
    fn classify(&self, u: f64, v: f64, plane: &InverseDepthPlane) -> Visibility {
        let (x, y) = (u.round() as isize, v.round() as isize);
        let Some(own) = plane.nearest_depth_in_cell(x, y) else {
            return Visibility::Ambiguous;
        };
        let center = self.at(x, y);
        if center < own * (1.0 - OCCLUDED_DEPTH_TOLERANCE) {
            return Visibility::Occluded;
        }
        if center < own * (1.0 - VISIBLE_DEPTH_TOLERANCE) {
            return Visibility::Ambiguous;
        }
        for dy in -1..=1 {
            for dx in -1..=1 {
                let Some(own) = plane.nearest_depth_in_cell(x + dx, y + dy) else {
                    return Visibility::Ambiguous;
                };
                if self.at(x + dx, y + dy) < own * (1.0 - OCCLUDED_DEPTH_TOLERANCE) {
                    return Visibility::Ambiguous;
                }
            }
        }
        Visibility::Visible
    }
}

/// Parameter range `[t0, t1]` of the segment `start + t (end - start)`,
/// `t` in `[0, 1]`, inside the box `[low, high]` (Liang-Barsky).
fn clip_segment(
    start: [f64; 2],
    end: [f64; 2],
    low: [f64; 2],
    high: [f64; 2],
) -> Option<(f64, f64)> {
    if !start.iter().chain(&end).all(|value| value.is_finite()) {
        return None;
    }
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for axis in 0..2 {
        let delta = end[axis] - start[axis];
        for (p, q) in [
            (-delta, start[axis] - low[axis]),
            (delta, high[axis] - start[axis]),
        ] {
            if p == 0.0 {
                if q < 0.0 {
                    return None;
                }
            } else {
                let r = q / p;
                if p < 0.0 {
                    t0 = t0.max(r);
                } else {
                    t1 = t1.min(r);
                }
            }
        }
    }
    (t0 <= t1).then_some((t0, t1))
}

/// Clip a camera-space triangle against the near plane (Sutherland-Hodgman).
fn clip_near(triangle: &[[f64; 3]; 3]) -> Vec<[f64; 3]> {
    if !triangle.iter().flatten().all(|value| value.is_finite()) {
        return Vec::new();
    }
    let mut polygon = Vec::with_capacity(4);
    for index in 0..3 {
        let current = triangle[index];
        let next = triangle[(index + 1) % 3];
        let (current_in, next_in) = (current[2] > NEAR_DEPTH, next[2] > NEAR_DEPTH);
        if current_in {
            polygon.push(current);
        }
        if current_in != next_in {
            let t = (NEAR_DEPTH - current[2]) / (next[2] - current[2]);
            polygon.push([0, 1, 2].map(|axis| current[axis] + (next[axis] - current[axis]) * t));
        }
    }
    polygon
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
