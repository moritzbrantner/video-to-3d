//! Multi-view object reference packages (#63).
//!
//! For every identified movable object (#61), choose a primary reference view
//! and well-separated additional views from its accepted, cross-view-checked
//! masks (#62), and package them with exact frame/mask provenance, the
//! observed points that localize it, a scale/extent estimate in the scene's
//! own scale, and its semantic description. The package references pixels by
//! keyframe content hash and crop box rather than copying them, so any local
//! or remote generation adapter can resolve them from the project. Views that
//! contradict the object's other views, are box-only or weak, or are heavily
//! occluded/truncated are rejected with a reason. Nothing here rewrites the
//! reconstruction.

use crate::object_identity::{
    front_points, FrameCamera, ObjectCategory, ObjectIdentityArtifact, SceneObject,
};
use crate::object_masks::{MaskConsistency, MaskQuality, ObjectMask, ObjectMasksArtifact};
use crate::scene_project::ContentHash;
use crate::semantic_inventory::{SemanticInventoryArtifact, SemanticObservation};
use crate::EvidenceScale;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const OBJECT_REFERENCES_SCHEMA_VERSION: u32 = 1;
/// Most additional views beyond the primary.
pub const MAX_ADDITIONAL_VIEWS: usize = 4;
/// A view whose mask covers less than this share of the object's largest
/// mask is treated as too occluded (or too distant) to serve as a reference.
pub const MIN_RELATIVE_MASK_AREA: f64 = 0.5;
/// Most observed points kept per package.
pub const MAX_PACKAGE_POINTS: usize = 2048;
/// Crop margin around a reference box, as a share of its larger side.
pub const CROP_MARGIN: f64 = 0.15;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceView {
    pub observation: String,
    pub frame_index: usize,
    /// Content identity of the source keyframe.
    pub frame_hash: ContentHash,
    /// Pixel crop `[x0, y0, x1, y1]` around the object, inside the frame.
    pub crop: [u32; 4],
    pub bbox: [f64; 4],
    pub mask_hash: ContentHash,
    pub mask_area_pixels: u64,
    pub detection_confidence: f64,
    pub mask_confidence: f64,
    pub consistency: MaskConsistency,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectedView {
    pub observation: String,
    pub frame_index: usize,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectExtent {
    /// Axis-aligned extent of the observed points, in scene units.
    pub size: [f64; 3],
    pub centre: [f64; 3],
    /// The scene's scale: arbitrary monocular unless metric evidence exists.
    pub scale: EvidenceScale,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectReferencePackage {
    pub object: String,
    pub label: String,
    pub source_tracks: Vec<String>,
    pub description: String,
    pub primary: ReferenceView,
    pub additional: Vec<ReferenceView>,
    pub rejected: Vec<RejectedView>,
    /// Observed accepted points localizing the object (scene coordinates).
    pub observed_points: Vec<[f64; 3]>,
    pub extent: Option<ObjectExtent>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnpackagedObject {
    pub object: String,
    pub label: String,
    pub reason: String,
    pub rejected: Vec<RejectedView>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferencesStatus {
    Available,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectReferencesArtifact {
    pub schema_version: u32,
    pub status: ReferencesStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
    pub semantic_inventory: ContentHash,
    pub object_identity: ContentHash,
    pub object_masks: ContentHash,
    pub packages: Vec<ObjectReferencePackage>,
    pub unpackaged: Vec<UnpackagedObject>,
}

impl ObjectReferencesArtifact {
    pub fn from_json(document: &str) -> Result<Self, String> {
        let artifact: Self = serde_json::from_str(document)
            .map_err(|error| format!("object references artifact: {error}"))?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn to_json(&self) -> String {
        let mut document = serde_json::to_string_pretty(self).expect("object references serialize");
        document.push('\n');
        document
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != OBJECT_REFERENCES_SCHEMA_VERSION {
            return Err(format!(
                "unsupported object references schema version {}",
                self.schema_version
            ));
        }
        let unavailable = self.status == ReferencesStatus::Unavailable;
        if unavailable
            != self
                .unavailable_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
            || (unavailable && !(self.packages.is_empty() && self.unpackaged.is_empty()))
        {
            return Err("object references: unavailable needs a reason and no packages".into());
        }
        let mut objects = BTreeSet::new();
        for package in &self.packages {
            if !objects.insert(package.object.as_str()) {
                return Err(format!("object {} is packaged twice", package.object));
            }
            let mut frames = BTreeSet::new();
            for view in std::iter::once(&package.primary).chain(&package.additional) {
                if !frames.insert(view.frame_index) {
                    return Err(format!(
                        "object {} references frame {} twice",
                        package.object, view.frame_index
                    ));
                }
                if view.consistency == MaskConsistency::Inconsistent
                    || view.consistency == MaskConsistency::MissingMask
                {
                    return Err(format!(
                        "object {} references a rejected view",
                        package.object
                    ));
                }
                let [x0, y0, x1, y1] = view.crop;
                if x0 >= x1 || y0 >= y1 {
                    return Err(format!("object {} has an empty crop", package.object));
                }
            }
            if package.additional.len() > MAX_ADDITIONAL_VIEWS {
                return Err(format!("object {} has too many views", package.object));
            }
        }
        for object in &self.unpackaged {
            if object.reason.trim().is_empty() || !objects.insert(object.object.as_str()) {
                return Err(format!(
                    "unpackaged object {} has no reason or is repeated",
                    object.object
                ));
            }
        }
        Ok(())
    }
}

fn mask_area(observation: &SemanticObservation) -> u64 {
    observation
        .mask
        .as_ref()
        .map_or(0, |mask| mask.counts.iter().skip(1).step_by(2).sum())
}

fn crop(observation: &SemanticObservation, width: u32, height: u32) -> [u32; 4] {
    let [x0, y0, x1, y1] = observation.bbox;
    let margin = (x1 - x0).max(y1 - y0) * CROP_MARGIN;
    [
        (x0 - margin).floor().max(0.0) as u32,
        (y0 - margin).floor().max(0.0) as u32,
        ((x1 + margin).ceil() as u32).min(width),
        ((y1 + margin).ceil() as u32).min(height),
    ]
}

/// Reference packages for every identified movable object.
#[allow(clippy::too_many_arguments)]
pub fn package_object_references(
    inventory: &SemanticInventoryArtifact,
    inventory_hash: ContentHash,
    identities: &ObjectIdentityArtifact,
    identity_hash: ContentHash,
    masks: &ObjectMasksArtifact,
    masks_hash: ContentHash,
    cameras: &[FrameCamera],
    points: &[[f64; 3]],
    scale: EvidenceScale,
) -> ObjectReferencesArtifact {
    let mut artifact = ObjectReferencesArtifact {
        schema_version: OBJECT_REFERENCES_SCHEMA_VERSION,
        status: ReferencesStatus::Available,
        unavailable_reason: None,
        semantic_inventory: inventory_hash,
        object_identity: identity_hash,
        object_masks: masks_hash,
        packages: Vec::new(),
        unpackaged: Vec::new(),
    };
    if masks.status == crate::object_masks::MasksStatus::Unavailable {
        artifact.status = ReferencesStatus::Unavailable;
        artifact.unavailable_reason = masks.unavailable_reason.clone();
        return artifact;
    }
    let observations: BTreeMap<&str, &SemanticObservation> = inventory
        .observations
        .iter()
        .map(|observation| (observation.id.as_str(), observation))
        .collect();
    let frame_sizes: BTreeMap<usize, (u32, u32)> = inventory
        .frames
        .iter()
        .map(|frame| (frame.frame_index, (frame.width, frame.height)))
        .collect();
    let checks: BTreeMap<&str, &ObjectMask> = masks
        .masks
        .iter()
        .map(|mask| (mask.observation.as_str(), mask))
        .collect();
    let cameras: BTreeMap<usize, &FrameCamera> = cameras
        .iter()
        .map(|camera| (camera.frame_index, camera))
        .collect();
    for object in identities
        .objects
        .iter()
        .filter(|object| object.category == ObjectCategory::Object)
    {
        package_one(
            object,
            &observations,
            &frame_sizes,
            &checks,
            &cameras,
            points,
            scale,
            &mut artifact,
        );
    }
    artifact
}

#[allow(clippy::too_many_arguments)]
fn package_one(
    object: &SceneObject,
    observations: &BTreeMap<&str, &SemanticObservation>,
    frame_sizes: &BTreeMap<usize, (u32, u32)>,
    checks: &BTreeMap<&str, &ObjectMask>,
    cameras: &BTreeMap<usize, &FrameCamera>,
    points: &[[f64; 3]],
    scale: EvidenceScale,
    artifact: &mut ObjectReferencesArtifact,
) {
    let largest = object
        .observations
        .iter()
        .filter_map(|id| observations.get(id.as_str()))
        .map(|observation| mask_area(observation))
        .max()
        .unwrap_or(0);
    let mut accepted: Vec<(f64, ReferenceView)> = Vec::new();
    let mut rejected = Vec::new();
    for id in &object.observations {
        let (Some(observation), Some(check)) =
            (observations.get(id.as_str()), checks.get(id.as_str()))
        else {
            continue;
        };
        let reject = |reason: String| RejectedView {
            observation: id.clone(),
            frame_index: observation.frame_index,
            reason,
        };
        let area = mask_area(observation);
        // A heavily occluded or truncated view also fails cross-view support,
        // so its size is reported first as the more specific reason.
        let reason = match (check.consistency, check.quality) {
            (MaskConsistency::MissingMask, _) => Some("box-only observation: no mask".into()),
            _ if (area as f64) < MIN_RELATIVE_MASK_AREA * largest as f64 => Some(format!(
                "mask covers {area} px, under {:.0}% of the object's largest view ({largest} px): too occluded, truncated or distant",
                MIN_RELATIVE_MASK_AREA * 100.0
            )),
            (MaskConsistency::Inconsistent, _) => Some(format!(
                "contradicts the object's other views: {}",
                check.reason
            )),
            (_, Some(MaskQuality::Weak)) => Some("weak mask (low SAM confidence)".into()),
            _ => None,
        };
        if let Some(reason) = reason {
            rejected.push(reject(reason));
            continue;
        }
        let (width, height) = frame_sizes
            .get(&observation.frame_index)
            .copied()
            .unwrap_or((0, 0));
        let confidence =
            observation.detection_confidence * observation.mask_confidence.unwrap_or(0.0);
        let verified = if check.consistency == MaskConsistency::Consistent {
            1.0
        } else {
            0.5
        };
        let score = verified * confidence * (area as f64 / largest.max(1) as f64);
        accepted.push((
            score,
            ReferenceView {
                observation: id.clone(),
                frame_index: observation.frame_index,
                frame_hash: check.frame_hash.clone(),
                crop: crop(observation, width, height),
                bbox: observation.bbox,
                mask_hash: check.mask_hash.clone().expect("checked masks carry a hash"),
                mask_area_pixels: area,
                detection_confidence: observation.detection_confidence,
                mask_confidence: observation.mask_confidence.unwrap_or(0.0),
                consistency: check.consistency,
            },
        ));
    }
    // Primary: best score, earliest frame on ties.
    accepted.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.1.frame_index.cmp(&b.1.frame_index))
    });
    if accepted.is_empty() {
        artifact.unpackaged.push(UnpackagedObject {
            object: object.id.clone(),
            label: object.label.clone(),
            reason: "no accepted reference view: every mask was rejected or missing".into(),
            rejected,
        });
        return;
    }
    let primary = accepted.remove(0).1;
    // Additional views: farthest in frame index from those already chosen.
    let mut chosen_frames = vec![primary.frame_index];
    let mut additional = Vec::new();
    while additional.len() < MAX_ADDITIONAL_VIEWS && !accepted.is_empty() {
        let (slot, _) = accepted
            .iter()
            .enumerate()
            .map(|(slot, (_, view))| {
                let separation = chosen_frames
                    .iter()
                    .map(|frame| frame.abs_diff(view.frame_index))
                    .min()
                    .unwrap_or(0);
                (slot, (separation, std::cmp::Reverse(view.frame_index)))
            })
            .max_by_key(|(_, key)| *key)
            .expect("non-empty");
        let view = accepted.remove(slot).1;
        if view.frame_index == primary.frame_index
            || additional
                .iter()
                .any(|other: &ReferenceView| other.frame_index == view.frame_index)
        {
            continue;
        }
        chosen_frames.push(view.frame_index);
        additional.push(view);
    }
    additional.sort_by_key(|view| view.frame_index);
    // Observed points from every chosen view with a registered camera.
    let mut observed: Vec<[f64; 3]> = Vec::new();
    let mut seen = BTreeSet::new();
    for view in std::iter::once(&primary).chain(&additional) {
        let (Some(observation), Some(camera)) = (
            observations.get(view.observation.as_str()),
            cameras.get(&view.frame_index),
        ) else {
            continue;
        };
        for (_, point) in front_points(observation, camera, points) {
            if observed.len() >= MAX_PACKAGE_POINTS {
                break;
            }
            if seen.insert(point.map(f64::to_bits)) {
                observed.push(point);
            }
        }
    }
    let extent = (!observed.is_empty()).then(|| {
        let mut low = [f64::INFINITY; 3];
        let mut high = [f64::NEG_INFINITY; 3];
        for point in &observed {
            for axis in 0..3 {
                low[axis] = low[axis].min(point[axis]);
                high[axis] = high[axis].max(point[axis]);
            }
        }
        ObjectExtent {
            size: [0, 1, 2].map(|axis| high[axis] - low[axis]),
            centre: [0, 1, 2].map(|axis| (high[axis] + low[axis]) / 2.0),
            scale,
        }
    });
    let description = format!(
        "{} ({} views selected from {} observation(s), motion: {:?}; {})",
        object.label,
        1 + additional.len(),
        object.observations.len(),
        object.motion,
        object.motion_reason
    );
    artifact.packages.push(ObjectReferencePackage {
        object: object.id.clone(),
        label: object.label.clone(),
        source_tracks: object.source_tracks.clone(),
        description,
        primary,
        additional,
        rejected,
        observed_points: observed,
        extent,
    });
}

#[cfg(test)]
#[path = "object_references_tests.rs"]
mod tests;
