//! Multi-view validation of object placements (#70).
//!
//! A placed candidate mesh (#69) is projected into every registered source
//! view of its object and compared there with the **original** semantic
//! observation (its SAM mask, or its box when no mask exists) and with the
//! accepted scene geometry. Clean plates are never consulted.
//!
//! Per view:
//!
//! - **Silhouette.** The share of projected mesh vertices inside the mask
//!   (vertices hidden behind nearer scene geometry are left out, so partial
//!   occlusion does not count against it) and the IoU of the projected
//!   bounding box with the observation box.
//! - **Landmarks.** The object's own visible accepted points in this view
//!   (its front depth band inside the mask) and their median distance from
//!   the placed mesh's bounding box, as a share of the placed height.
//! - **Depth ordering.** The share of projected vertices that would hide
//!   accepted geometry seen *outside* the mask (the object is placed too
//!   near) or that sit behind accepted geometry *inside* the mask, where the
//!   object is seen (it is placed too far).
//!
//! A view that fails any threshold contradicts the placement. A placement is
//! accepted only with no contradicting view and at least one consistent view
//! besides its reference frame (the reference frame fixed the scale, so it
//! cannot confirm it alone); otherwise it stays approximate and flagged for
//! review. Nothing here changes the placement transform.

use crate::object_identity::{front_points, inside, project, FrameCamera};
use crate::semantic_inventory::SemanticObservation;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Share of visible projected vertices that must land inside the mask.
pub const MIN_INSIDE_SHARE: f64 = 0.6;
/// IoU the projected box must reach with the observation box.
pub const MIN_BOX_IOU: f64 = 0.4;
/// Highest share of projected vertices in the wrong depth order.
pub const MAX_DEPTH_VIOLATION_SHARE: f64 = 0.2;
/// Highest median landmark distance from the placed box, as a share of the
/// placed height.
pub const MAX_LANDMARK_RESIDUAL: f64 = 0.25;
/// Fewest projected vertices that can judge a view.
pub const MIN_PROJECTED_VERTICES: usize = 12;
/// Most mesh vertices projected per view.
pub const MAX_VALIDATION_VERTICES: usize = 4000;
/// Depth grid cell in pixels and relative depth margin for ordering.
pub const DEPTH_CELL_PIXELS: f64 = 3.0;
pub const DEPTH_MARGIN: f64 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewVerdict {
    Consistent,
    Contradicts,
    /// Too little projected mesh in this view to judge it.
    Unverified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementVerdict {
    /// Consistent with the reference view and at least one other view; no
    /// view contradicts it.
    Accepted,
    /// No view contradicts it, but no view besides the reference confirms
    /// it: needs review.
    Approximate,
    /// At least one view contradicts it: needs review.
    Contradicted,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewResidual {
    pub observation: String,
    pub frame_index: usize,
    pub verdict: ViewVerdict,
    /// Mesh vertices in front of this camera.
    pub projected_vertices: usize,
    /// Visible projected vertices inside the original mask (or box).
    pub inside_share: Option<f64>,
    /// IoU of the projected mesh box with the observation box.
    pub box_iou: Option<f64>,
    /// Object landmarks visible in this view and their median distance from
    /// the placed box, as a share of the placed height.
    pub landmarks: usize,
    pub landmark_residual: Option<f64>,
    /// Projected vertices in the wrong depth order against accepted geometry.
    pub depth_violation_share: Option<f64>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementValidation {
    pub verdict: PlacementVerdict,
    pub needs_review: bool,
    pub views: Vec<ViewResidual>,
    pub reason: String,
}

/// A placed mesh in the scene: `x_scene = scale · rotation · x_mesh + translation`.
pub struct PlacedMesh<'a> {
    pub rotation: [f64; 9],
    pub scale: f64,
    pub translation: [f64; 3],
    pub vertices: &'a [[f64; 3]],
    pub reference_frame: usize,
}

impl PlacedMesh<'_> {
    fn to_scene(&self, point: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        [0, 1, 2].map(|row| {
            self.scale
                * (r[row * 3] * point[0] + r[row * 3 + 1] * point[1] + r[row * 3 + 2] * point[2])
                + self.translation[row]
        })
    }

    fn to_mesh(&self, point: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        let d = [0, 1, 2].map(|axis| point[axis] - self.translation[axis]);
        // Rᵀ d / scale
        [0, 1, 2].map(|column| {
            (r[column] * d[0] + r[3 + column] * d[1] + r[6 + column] * d[2]) / self.scale
        })
    }
}

fn bounds(points: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    points.iter().fold(
        ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
        |(low, high), point| {
            (
                [0, 1, 2].map(|axis| low[axis].min(point[axis])),
                [0, 1, 2].map(|axis| high[axis].max(point[axis])),
            )
        },
    )
}

fn box_iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = width * height;
    let area = |r: [f64; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let union = area(a) + area(b) - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

fn cell(pixel: [f64; 2]) -> (i64, i64) {
    (
        (pixel[0] / DEPTH_CELL_PIXELS).floor() as i64,
        (pixel[1] / DEPTH_CELL_PIXELS).floor() as i64,
    )
}

/// Validate one placement against its object's registered observations.
pub fn validate_placement(
    placed: &PlacedMesh,
    views: &[(&SemanticObservation, &FrameCamera)],
    points: &[[f64; 3]],
) -> PlacementValidation {
    let (mesh_low, mesh_high) = bounds(placed.vertices);
    let height = placed.scale * (mesh_high[2] - mesh_low[2]);
    // Scene geometry that is not the placed object itself.
    let margin = [0, 1, 2].map(|axis| (mesh_high[axis] - mesh_low[axis]) * 0.05);
    let outside_object = |point: &[f64; 3]| {
        let local = placed.to_mesh(*point);
        (0..3).any(|axis| {
            local[axis] < mesh_low[axis] - margin[axis]
                || local[axis] > mesh_high[axis] + margin[axis]
        })
    };
    let scene: Vec<[f64; 3]> = points.iter().copied().filter(outside_object).collect();
    // A bounded, deterministic vertex sample keeps large meshes cheap.
    let step = placed
        .vertices
        .len()
        .div_ceil(MAX_VALIDATION_VERTICES)
        .max(1);
    let scene_vertices: Vec<[f64; 3]> = placed
        .vertices
        .iter()
        .step_by(step)
        .map(|v| placed.to_scene(*v))
        .collect();

    let mut residuals = Vec::new();
    for (observation, camera) in views {
        let mut grid: BTreeMap<(i64, i64), f64> = BTreeMap::new();
        for point in &scene {
            if let Some((pixel, depth)) = project(camera, *point) {
                let nearest = grid.entry(cell(pixel)).or_insert(depth);
                *nearest = nearest.min(depth);
            }
        }
        let projected: Vec<([f64; 2], f64)> = scene_vertices
            .iter()
            .filter_map(|vertex| project(camera, *vertex))
            .collect();
        let landmarks = front_points(observation, camera, points);
        let landmark_residual = (height > 0.0)
            .then(|| {
                median(
                    landmarks
                        .iter()
                        .map(|(_, point)| {
                            let local = placed.to_mesh(*point);
                            let outside = [0, 1, 2].map(|axis| {
                                (mesh_low[axis] - local[axis])
                                    .max(local[axis] - mesh_high[axis])
                                    .max(0.0)
                            });
                            placed.scale
                                * (outside[0].powi(2) + outside[1].powi(2) + outside[2].powi(2))
                                    .sqrt()
                                / height
                        })
                        .collect(),
                )
            })
            .flatten();
        let mut residual = ViewResidual {
            observation: observation.id.clone(),
            frame_index: observation.frame_index,
            verdict: ViewVerdict::Unverified,
            projected_vertices: projected.len(),
            inside_share: None,
            box_iou: None,
            landmarks: landmarks.len(),
            landmark_residual,
            depth_violation_share: None,
            reason: String::new(),
        };
        if projected.len() < MIN_PROJECTED_VERTICES {
            residual.reason = format!(
                "only {} mesh vertices project in front of this camera",
                projected.len()
            );
            residuals.push(residual);
            continue;
        }
        let (mut visible, mut visible_inside, mut violations) = (0usize, 0usize, 0usize);
        let mut pixel_box = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for (pixel, depth) in &projected {
            pixel_box = [
                pixel_box[0].min(pixel[0]),
                pixel_box[1].min(pixel[1]),
                pixel_box[2].max(pixel[0]),
                pixel_box[3].max(pixel[1]),
            ];
            let in_mask = inside(observation, *pixel);
            let scene_depth = grid.get(&cell(*pixel)).copied();
            let behind = scene_depth.is_some_and(|scene| *depth > scene * (1.0 + DEPTH_MARGIN));
            let in_front = scene_depth.is_some_and(|scene| *depth < scene * (1.0 - DEPTH_MARGIN));
            if (in_mask && behind) || (!in_mask && in_front) {
                violations += 1;
            }
            if !behind {
                visible += 1;
                if in_mask {
                    visible_inside += 1;
                }
            }
        }
        let inside_share = (visible > 0).then(|| visible_inside as f64 / visible as f64);
        let iou = box_iou(pixel_box, observation.bbox);
        let violation_share = violations as f64 / projected.len() as f64;
        residual.inside_share = inside_share;
        residual.box_iou = Some(iou);
        residual.depth_violation_share = Some(violation_share);
        let mut failures = Vec::new();
        if inside_share.is_none_or(|share| share < MIN_INSIDE_SHARE) {
            failures.push(format!(
                "{:.0}% of visible vertices inside the mask (< {:.0}%)",
                inside_share.unwrap_or(0.0) * 100.0,
                MIN_INSIDE_SHARE * 100.0
            ));
        }
        if iou < MIN_BOX_IOU {
            failures.push(format!("box IoU {iou:.2} (< {MIN_BOX_IOU:.2})"));
        }
        if violation_share > MAX_DEPTH_VIOLATION_SHARE {
            failures.push(format!(
                "{:.0}% of vertices in the wrong depth order (> {:.0}%)",
                violation_share * 100.0,
                MAX_DEPTH_VIOLATION_SHARE * 100.0
            ));
        }
        if landmark_residual.is_some_and(|value| value > MAX_LANDMARK_RESIDUAL) {
            failures.push(format!(
                "landmarks lie {:.2} heights from the placed box (> {MAX_LANDMARK_RESIDUAL:.2})",
                landmark_residual.unwrap()
            ));
        }
        if failures.is_empty() {
            residual.verdict = ViewVerdict::Consistent;
            residual.reason = format!(
                "{:.0}% inside the mask, box IoU {iou:.2}, {:.0}% depth-order violations",
                inside_share.unwrap_or(0.0) * 100.0,
                violation_share * 100.0
            );
        } else {
            residual.verdict = ViewVerdict::Contradicts;
            residual.reason = failures.join("; ");
        }
        residuals.push(residual);
    }

    let contradicting: Vec<usize> = residuals
        .iter()
        .filter(|view| view.verdict == ViewVerdict::Contradicts)
        .map(|view| view.frame_index)
        .collect();
    let confirming = residuals
        .iter()
        .filter(|view| {
            view.verdict == ViewVerdict::Consistent && view.frame_index != placed.reference_frame
        })
        .count();
    let (verdict, reason) = if !contradicting.is_empty() {
        (
            PlacementVerdict::Contradicted,
            format!("contradicted by frame(s) {contradicting:?}"),
        )
    } else if confirming > 0 {
        (
            PlacementVerdict::Accepted,
            format!(
                "consistent in {confirming} view(s) besides reference frame {}",
                placed.reference_frame
            ),
        )
    } else {
        (
            PlacementVerdict::Approximate,
            format!(
                "no registered view besides reference frame {} confirms it",
                placed.reference_frame
            ),
        )
    };
    PlacementValidation {
        needs_review: verdict != PlacementVerdict::Accepted,
        verdict,
        views: residuals,
        reason,
    }
}

#[cfg(test)]
#[path = "placement_validation_tests.rs"]
mod tests;
