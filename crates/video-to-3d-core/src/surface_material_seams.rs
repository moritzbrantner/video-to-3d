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
//!    A sample is occluded when accepted geometry covering its own ray is
//!    clearly in front of the triangle's plane there, and ambiguous when the
//!    depth difference falls inside the tolerance band, or when accepted
//!    geometry in its one-pixel neighbourhood is clearly in front of the plane
//!    or lies where the plane is beyond its horizon.
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
        let coverage = CoverageIndex::build(&view, evidence, check_canceled)?;
        for (slot, triangle_index) in seams.iter().enumerate() {
            if slot % 1024 == 0 {
                check_canceled()?;
            }
            let triangle = &evidence.triangles[*triangle_index];
            let corners = [triangle.a, triangle.b, triangle.c];
            match view.admit(evidence, &coverage, corners) {
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
        coverage: &CoverageIndex,
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
            match coverage.classify(u, v, &plane) {
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

    /// Inverse depth of the plane along the ray of pixel point `(u, v)`; not
    /// positive beyond the plane's horizon.
    fn at(&self, u: f64, v: f64) -> f64 {
        self.a * u + self.b * v + self.c
    }
}

/// A near-clipped accepted triangle in camera pixel space. Each corner is
/// `[u, v, 1 / z]`; inverse depth is affine in pixel coordinates, so clipping
/// interpolates it exactly.
struct RasterTriangle {
    corners: [[f64; 3]; 3],
}

/// Registration margin, in pixels, so a triangle touching a cell boundary is
/// listed in both cells.
const CELL_MARGIN: f64 = 1.0e-6;

/// Rasterization work (cells registered or counted, rows scanned) between
/// two cancellation polls.
const RASTER_POLL_INTERVAL: usize = 1 << 16;

/// The cells of one row a triangle's footprint touches.
struct Span {
    row: usize,
    first: usize,
    last: usize,
    triangle: u32,
}

/// Every accepted triangle registered in each pixel cell
/// (`[x - 0.5, x + 0.5] x [y - 0.5, y + 0.5]`) its footprint touches, so the
/// visibility test compares the actual geometry at its own positions with the
/// seam plane instead of one depth per cell. Thin and subpixel triangles are registered in every cell they touch.
struct CoverageIndex {
    width: usize,
    height: usize,
    triangles: Vec<RasterTriangle>,
    /// Triangles of cell `i` are `entries[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<usize>,
    entries: Vec<u32>,
    /// Work done since the last cancellation poll.
    unpolled_work: usize,
}

impl CoverageIndex {
    fn build(
        view: &CameraView<'_>,
        evidence: &ReconstructionEvidenceView<'_>,
        check_canceled: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<Self, String> {
        let (width, height) = (view.width as usize, view.height as usize);
        let cells = width * height;
        let mut index = Self {
            width,
            height,
            triangles: Vec::new(),
            offsets: vec![0; cells + 1],
            entries: Vec::new(),
            unpolled_work: 0,
        };
        let mut spans = Vec::new();
        for (ordinal, triangle) in evidence.triangles.iter().enumerate() {
            if ordinal % 1024 == 0 {
                check_canceled()?;
            }
            let camera = [triangle.a, triangle.b, triangle.c]
                .map(|index| view.to_camera(position(&evidence.points[index])));
            let polygon = clip_near(&camera);
            for fan in 1..polygon.len().saturating_sub(1) {
                let corners = [polygon[0], polygon[fan], polygon[fan + 1]].map(|point| {
                    let [u, v] = view.pixel(point);
                    [u, v, 1.0 / point[2]]
                });
                if !corners.iter().flatten().all(|value| value.is_finite()) {
                    continue;
                }
                let ordinal = u32::try_from(index.triangles.len()).map_err(|_| {
                    "too many accepted triangles for the seam visibility index".to_owned()
                })?;
                index.push_spans(&corners, ordinal, &mut spans, check_canceled)?;
                index.triangles.push(RasterTriangle { corners });
            }
        }

        // Count per cell, turn counts into inclusive ends (the last one
        // carries the total into the sentinel), then fill each cell from its
        // end so `offsets[i]` becomes its start.
        for span in &spans {
            index.for_cells(span, check_canceled, |offsets, _, cell| offsets[cell] += 1)?;
        }
        for row in 0..height {
            index.charge(width, check_canceled)?;
            for cell in row * width..(row + 1) * width {
                index.offsets[cell + 1] += index.offsets[cell];
            }
        }
        index.entries = vec![0; index.offsets[cells]];
        for span in &spans {
            index.for_cells(span, check_canceled, |offsets, entries, cell| {
                offsets[cell] -= 1;
                entries[offsets[cell]] = span.triangle;
            })?;
        }
        Ok(index)
    }

    /// Record the cells each row of `corners`' footprint touches.
    fn push_spans(
        &mut self,
        corners: &[[f64; 3]; 3],
        triangle: u32,
        spans: &mut Vec<Span>,
        check_canceled: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<(), String> {
        let (low, high) = corners
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), p| {
                (low.min(p[1]), high.max(p[1]))
            });
        let first_row = (low - 0.5 - CELL_MARGIN).ceil().max(0.0);
        let last_row = (high + 0.5 + CELL_MARGIN)
            .floor()
            .min(self.height as f64 - 1.0);
        if first_row > last_row {
            return Ok(());
        }
        let last_column = self.width as f64 - 1.0;
        for row in first_row as usize..=last_row as usize {
            self.charge(1, check_canceled)?;
            let strip = clip_axis(corners, 1, row as f64 - 0.5 - CELL_MARGIN, 1.0);
            let strip = clip_axis(&strip, 1, row as f64 + 0.5 + CELL_MARGIN, -1.0);
            let (left, right) = strip
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(left, right), p| {
                    (left.min(p[0]), right.max(p[0]))
                });
            let first = (left - 0.5 - CELL_MARGIN).ceil().max(0.0);
            let last = (right + 0.5 + CELL_MARGIN).floor().min(last_column);
            if first <= last {
                spans.push(Span {
                    row,
                    first: first as usize,
                    last: last as usize,
                    triangle,
                });
            }
        }
        Ok(())
    }

    /// Visit the cells of `span` in chunks of at most one poll interval,
    /// charging each chunk before its work so oversized spans still poll at
    /// bounded intervals.
    fn for_cells(
        &mut self,
        span: &Span,
        check_canceled: &mut impl FnMut() -> Result<(), String>,
        mut visit: impl FnMut(&mut [usize], &mut [u32], usize),
    ) -> Result<(), String> {
        let base = span.row * self.width;
        let mut first = span.first;
        while first <= span.last {
            let last = span.last.min(first + RASTER_POLL_INTERVAL - 1);
            self.charge(last - first + 1, check_canceled)?;
            for column in first..=last {
                visit(&mut self.offsets, &mut self.entries, base + column);
            }
            first = last + 1;
        }
        Ok(())
    }

    /// Count `work` units and poll cancellation once per full interval.
    fn charge(
        &mut self,
        work: usize,
        check_canceled: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<(), String> {
        self.unpolled_work += work;
        while self.unpolled_work >= RASTER_POLL_INTERVAL {
            self.unpolled_work -= RASTER_POLL_INTERVAL;
            check_canceled()?;
        }
        Ok(())
    }

    fn cell(&self, x: isize, y: isize) -> impl Iterator<Item = &RasterTriangle> {
        let range = if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            0..0
        } else {
            let cell = y as usize * self.width + x as usize;
            self.offsets[cell]..self.offsets[cell + 1]
        };
        self.entries[range]
            .iter()
            .map(|triangle| &self.triangles[*triangle as usize])
    }

    /// Classify the sample at pixel `(u, v)` on the seam triangle's `plane`.
    ///
    /// Each accepted triangle registered in a cell is clipped to that cell and
    /// compared with the plane at the clipped part's own positions, not with
    /// one depth per cell: coplanar neighbours meet the plane exactly there,
    /// so a steep plane never occludes itself, while geometry clearly in front
    /// of the plane anywhere on the sample's ray (which lies in its own cell)
    /// is always seen. Inverse depths are affine in pixel coordinates, so the
    /// clipped part's vertices bound the comparison over the whole part.
    ///
    /// In the sample's own cell, geometry clearly in front of the plane
    /// occludes and geometry inside the tolerance band is ambiguous; in the
    /// neighbouring cells geometry clearly in front is ambiguous. Geometry
    /// where the plane lies beyond its horizon has no plane depth to compare
    /// with and fails closed as ambiguous.
    fn classify(&self, u: f64, v: f64, plane: &InverseDepthPlane) -> Visibility {
        let (x, y) = (u.round() as isize, v.round() as isize);
        let mut ambiguous = false;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let own_cell = dx == 0 && dy == 0;
                let (cx, cy) = (x + dx, y + dy);
                let (low, high) = (
                    [cx as f64 - 0.5, cy as f64 - 0.5],
                    [cx as f64 + 0.5, cy as f64 + 0.5],
                );
                for triangle in self.cell(cx, cy) {
                    let mut part = triangle.corners.to_vec();
                    for axis in 0..2 {
                        part = clip_axis(&part, axis, low[axis], 1.0);
                        part = clip_axis(&part, axis, high[axis], -1.0);
                    }
                    for [pu, pv, inverse] in part {
                        let plane_inverse = plane.at(pu, pv);
                        if !(plane_inverse.is_finite() && plane_inverse > 0.0) {
                            ambiguous = true;
                        } else if inverse * (1.0 - OCCLUDED_DEPTH_TOLERANCE) > plane_inverse {
                            if own_cell {
                                return Visibility::Occluded;
                            }
                            ambiguous = true;
                        } else if own_cell
                            && inverse * (1.0 - VISIBLE_DEPTH_TOLERANCE) > plane_inverse
                        {
                            ambiguous = true;
                        }
                    }
                }
            }
        }
        if ambiguous {
            Visibility::Ambiguous
        } else {
            Visibility::Visible
        }
    }
}

/// The part of a convex pixel-space polygon (vertices `[u, v, 1 / z]`) where
/// `sign * (vertex[axis] - bound) >= 0` (Sutherland-Hodgman, one boundary).
fn clip_axis(polygon: &[[f64; 3]], axis: usize, bound: f64, sign: f64) -> Vec<[f64; 3]> {
    let mut clipped = Vec::with_capacity(polygon.len() + 1);
    for (index, current) in polygon.iter().enumerate() {
        let next = polygon[(index + 1) % polygon.len()];
        let (inside, next_inside) = (sign * (current[axis] - bound), sign * (next[axis] - bound));
        if inside >= 0.0 {
            clipped.push(*current);
        }
        if (inside >= 0.0) != (next_inside >= 0.0) {
            let t = inside / (inside - next_inside);
            clipped.push([0, 1, 2].map(|k| current[k] + (next[k] - current[k]) * t));
        }
    }
    clipped
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
