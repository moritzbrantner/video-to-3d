//! Cross-view mask consistency for scene objects (#62).
//!
//! Every semantic mask of an identified object (#61) is checked against the
//! other views of the same object: the accepted points that localize the
//! object in its other registered frames are reprojected into this frame,
//! and the share landing inside this mask is its *support*. A mask whose
//! support is too low contradicts the other views and is reported
//! inconsistent. Points hidden behind other accepted geometry in this view
//! (partial occlusion) do not count against it. A box-only observation stays `missing_mask` and a low SAM
//! confidence stays `weak` — neither is elevated. Masks are referenced by the
//! semantic inventory's observation ids and a content hash of their encoding,
//! so consumers never re-run segmentation. Nothing here changes cameras,
//! geometry, masks or identities.

use crate::object_identity::{front_points, inside, project, FrameCamera, ObjectIdentityArtifact};
use crate::scene_project::ContentHash;
use crate::semantic_inventory::{SegmentationStatus, SemanticInventoryArtifact};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const OBJECT_MASKS_SCHEMA_VERSION: u32 = 1;
/// Share of the object's other-view points a mask must contain to agree.
pub const MIN_MASK_SUPPORT: f64 = 0.5;
/// SAM mask confidence below which a mask is reported weak.
pub const WEAK_MASK_CONFIDENCE: f64 = 0.5;
/// Fewest reprojected points that can judge a mask.
pub const MIN_SUPPORT_POINTS: usize = 3;
/// Occlusion grid cell, in pixels: a reprojected object point behind another
/// accepted point in its cell (by more than [`OCCLUSION_MARGIN`]) is hidden
/// in this view and does not count against the mask.
pub const OCCLUSION_CELL_PIXELS: f64 = 3.0;
pub const OCCLUSION_MARGIN: f64 = 0.05;

/// Nearest accepted depth per occlusion cell of one camera.
fn depth_grid(camera: &FrameCamera, points: &[[f64; 3]]) -> BTreeMap<(i64, i64), f64> {
    let mut grid = BTreeMap::new();
    for point in points {
        if let Some((pixel, depth)) = project(camera, *point) {
            let cell = (
                (pixel[0] / OCCLUSION_CELL_PIXELS).floor() as i64,
                (pixel[1] / OCCLUSION_CELL_PIXELS).floor() as i64,
            );
            let nearest = grid.entry(cell).or_insert(depth);
            if depth < *nearest {
                *nearest = depth;
            }
        }
    }
    grid
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskConsistency {
    Consistent,
    Inconsistent,
    /// No other registered view localizes the object, or this frame has no
    /// registered camera.
    Unverified,
    /// Box-only observation: no mask to check.
    MissingMask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskQuality {
    Confident,
    Weak,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectMask {
    pub object: String,
    pub source_tracks: Vec<String>,
    pub observation: String,
    pub frame_index: usize,
    /// Content identity of the source keyframe.
    pub frame_hash: ContentHash,
    pub label: String,
    /// Hash of the semantic mask's encoding; `None` for a box-only observation.
    pub mask_hash: Option<ContentHash>,
    pub mask_confidence: Option<f64>,
    pub quality: Option<MaskQuality>,
    pub consistency: MaskConsistency,
    /// Share of the object's other-view points inside this mask.
    pub support: Option<f64>,
    pub compared_frames: Vec<usize>,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MasksStatus {
    Available,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectMasksArtifact {
    pub schema_version: u32,
    pub status: MasksStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
    pub semantic_inventory: ContentHash,
    pub object_identity: ContentHash,
    pub masks: Vec<ObjectMask>,
}

impl ObjectMasksArtifact {
    pub fn from_json(document: &str) -> Result<Self, String> {
        let artifact: Self = serde_json::from_str(document)
            .map_err(|error| format!("object masks artifact: {error}"))?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn to_json(&self) -> String {
        let mut document = serde_json::to_string_pretty(self).expect("object masks serialize");
        document.push('\n');
        document
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != OBJECT_MASKS_SCHEMA_VERSION {
            return Err(format!(
                "unsupported object masks schema version {}",
                self.schema_version
            ));
        }
        if (self.status == MasksStatus::Unavailable)
            != self
                .unavailable_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
            || (self.status == MasksStatus::Unavailable && !self.masks.is_empty())
        {
            return Err("object masks: unavailable needs a reason and no masks".into());
        }
        for mask in &self.masks {
            if mask.reason.trim().is_empty() {
                return Err(format!("object mask {} has no reason", mask.observation));
            }
            let missing = mask.mask_hash.is_none();
            if missing != (mask.consistency == MaskConsistency::MissingMask)
                || missing != mask.quality.is_none()
            {
                return Err(format!(
                    "object mask {} disagrees on whether it has a mask",
                    mask.observation
                ));
            }
            match (mask.consistency, mask.support) {
                (MaskConsistency::Consistent | MaskConsistency::Inconsistent, Some(value))
                    if value.is_finite() && (0.0..=1.0).contains(&value) => {}
                (MaskConsistency::Consistent | MaskConsistency::Inconsistent, _) => {
                    return Err(format!(
                        "object mask {} is judged without a valid support",
                        mask.observation
                    ))
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Cross-view consistency of every mask of every identified object.
pub fn check_object_masks(
    inventory: &SemanticInventoryArtifact,
    inventory_hash: ContentHash,
    identities: &ObjectIdentityArtifact,
    identity_hash: ContentHash,
    cameras: &[FrameCamera],
    points: &[[f64; 3]],
) -> ObjectMasksArtifact {
    let mut artifact = ObjectMasksArtifact {
        schema_version: OBJECT_MASKS_SCHEMA_VERSION,
        status: MasksStatus::Available,
        unavailable_reason: None,
        semantic_inventory: inventory_hash,
        object_identity: identity_hash,
        masks: Vec::new(),
    };
    if inventory.status == SegmentationStatus::Unavailable
        || identities.status == crate::object_identity::IdentityStatus::Unavailable
    {
        artifact.status = MasksStatus::Unavailable;
        artifact.unavailable_reason = Some(
            identities
                .unavailable_reason
                .clone()
                .or_else(|| inventory.unavailable_reason.clone())
                .unwrap_or_else(|| "no semantic evidence".into()),
        );
        return artifact;
    }
    let cameras: BTreeMap<usize, &FrameCamera> = cameras
        .iter()
        .map(|camera| (camera.frame_index, camera))
        .collect();
    let observations: BTreeMap<&str, _> = inventory
        .observations
        .iter()
        .map(|observation| (observation.id.as_str(), observation))
        .collect();
    let frame_hashes: BTreeMap<usize, &ContentHash> = inventory
        .frames
        .iter()
        .map(|frame| (frame.frame_index, &frame.content_hash))
        .collect();
    for object in &identities.objects {
        // Object points seen in each frame, from its localizing observations.
        let views: Vec<(usize, Vec<[f64; 3]>)> = object
            .observations
            .iter()
            .filter_map(|id| {
                let observation = observations.get(id.as_str())?;
                let camera = cameras.get(&observation.frame_index)?;
                let front = front_points(observation, camera, points);
                (!front.is_empty()).then(|| {
                    (
                        observation.frame_index,
                        front.into_iter().map(|(_, point)| point).collect(),
                    )
                })
            })
            .collect();
        for id in &object.observations {
            let Some(observation) = observations.get(id.as_str()) else {
                continue;
            };
            let frame = observation.frame_index;
            let mask_hash = observation.mask.as_ref().map(|mask| {
                ContentHash::of_bytes(&serde_json::to_vec(mask).expect("semantic masks serialize"))
            });
            let quality = observation.mask.as_ref().map(|_| {
                if observation.mask_confidence.unwrap_or(0.0) < WEAK_MASK_CONFIDENCE {
                    MaskQuality::Weak
                } else {
                    MaskQuality::Confident
                }
            });
            let others: Vec<&(usize, Vec<[f64; 3]>)> =
                views.iter().filter(|(view, _)| *view != frame).collect();
            let compared_frames: Vec<usize> = others.iter().map(|(view, _)| *view).collect();
            let (consistency, support, reason) = if observation.mask.is_none() {
                (
                    MaskConsistency::MissingMask,
                    None,
                    "box-only observation: no mask to check".to_owned(),
                )
            } else if let Some(camera) = cameras.get(&frame) {
                let grid = depth_grid(camera, points);
                let visible = |pixel: [f64; 2], depth: f64| {
                    let cell = (
                        (pixel[0] / OCCLUSION_CELL_PIXELS).floor() as i64,
                        (pixel[1] / OCCLUSION_CELL_PIXELS).floor() as i64,
                    );
                    grid.get(&cell)
                        .is_none_or(|nearest| *nearest >= depth * (1.0 - OCCLUSION_MARGIN))
                };
                let reprojected: Vec<[f64; 2]> = others
                    .iter()
                    .flat_map(|(_, view)| view.iter())
                    .filter_map(|point| project(camera, *point))
                    .filter(|(pixel, depth)| visible(*pixel, *depth))
                    .map(|(pixel, _)| pixel)
                    .filter(|pixel| {
                        pixel[0] >= 0.0
                            && pixel[1] >= 0.0
                            && pixel[0] < f64::from(observation.mask.as_ref().unwrap().width)
                            && pixel[1] < f64::from(observation.mask.as_ref().unwrap().height)
                    })
                    .collect();
                if reprojected.len() < MIN_SUPPORT_POINTS {
                    (
                        MaskConsistency::Unverified,
                        None,
                        format!(
                            "{} point(s) from {} other localized view(s) reproject into this frame; {MIN_SUPPORT_POINTS} are needed",
                            reprojected.len(),
                            compared_frames.len()
                        ),
                    )
                } else {
                    let inside_count = reprojected
                        .iter()
                        .filter(|pixel| inside(observation, **pixel))
                        .count();
                    let value = inside_count as f64 / reprojected.len() as f64;
                    let consistency = if value >= MIN_MASK_SUPPORT {
                        MaskConsistency::Consistent
                    } else {
                        MaskConsistency::Inconsistent
                    };
                    (
                        consistency,
                        Some(value),
                        format!(
                            "{inside_count} of {} points of this object seen in frame(s) {compared_frames:?} reproject inside the mask (need {:.0}%)",
                            reprojected.len(),
                            MIN_MASK_SUPPORT * 100.0
                        ),
                    )
                }
            } else {
                (
                    MaskConsistency::Unverified,
                    None,
                    format!("frame {frame} has no registered camera"),
                )
            };
            let Some(frame_hash) = frame_hashes.get(&frame) else {
                continue;
            };
            artifact.masks.push(ObjectMask {
                object: object.id.clone(),
                source_tracks: object.source_tracks.clone(),
                observation: id.clone(),
                frame_index: frame,
                frame_hash: (*frame_hash).clone(),
                label: object.label.clone(),
                mask_hash,
                mask_confidence: observation.mask_confidence,
                quality,
                consistency,
                support,
                compared_frames,
                reason,
            });
        }
    }
    artifact
}

#[cfg(test)]
#[path = "object_masks_tests.rs"]
mod tests;
