//! Stable scene-object identities and motion classes over semantic evidence (#61).
//!
//! Consumes the versioned semantic inventory (#60) and the accepted
//! reconstruction read-only: registered cameras and accepted points. Model
//! track indices are proposals; this pass turns them into stable object
//! identities and classifies how each object may participate in
//! reconstruction:
//!
//! - **Localization.** In every frame with a registered camera, the accepted
//!   points projecting inside an observation's mask, nearest the camera (the
//!   front depth band, so the wall behind an object does not count), give the
//!   object's evidence position in that frame.
//! - **Identity.** Each semantic track (or untracked observation) is a
//!   candidate. Candidates of one label that never share a frame and whose
//!   evidence positions agree are one object across occlusions and viewpoint
//!   gaps; candidates that would merge but share a frame stay separate and are
//!   reported as an ambiguous merge. Identities are content-derived, so the
//!   same evidence always yields the same ids.
//! - **Motion.** Architecture labels are static structure. Other objects are
//!   movable-but-stationary when their evidence positions agree across
//!   frames, moving when they do not, and unknown with a reason when fewer
//!   than two frames localize them.
//!
//! Nothing here changes cameras or geometry: positions are reported evidence,
//! and an unavailable inventory yields an unavailable result.

use crate::scene_project::ContentHash;
use crate::semantic_inventory::{
    SegmentationStatus, SemanticInventoryArtifact, SemanticMask, SemanticObservation,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const OBJECT_IDENTITY_SCHEMA_VERSION: u32 = 1;
/// Labels treated as fixed scene structure rather than movable objects.
pub const STRUCTURE_LABELS: [&str; 6] = ["wall", "floor", "ceiling", "door", "window", "ground"];
/// Fewest accepted points inside a mask that localize an observation.
pub const MIN_LOCALIZING_POINTS: usize = 3;
/// Points within this share of the nearest localizing depth form the front
/// band that localizes an observation.
pub const FRONT_DEPTH_BAND: f64 = 0.1;
/// Largest distance between evidence positions, as a share of their viewing
/// depth, at which two frames see one stationary object.
pub const STATIONARY_TOLERANCE: f64 = 0.08;

/// A registered pinhole camera of one keyframe (world-to-camera, row-major).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameCamera {
    pub frame_index: usize,
    pub rotation: [f64; 9],
    pub translation: [f64; 3],
    pub focal_pixels: f64,
    pub principal_point: [f64; 2],
}

impl FrameCamera {
    fn camera_point(&self, point: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        [0, 1, 2].map(|row| {
            r[row * 3] * point[0]
                + r[row * 3 + 1] * point[1]
                + r[row * 3 + 2] * point[2]
                + self.translation[row]
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectCategory {
    /// Architecture: walls, floors, ceilings, doors, windows.
    Structure,
    /// Furniture and other meaningful objects.
    Object,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionClass {
    StaticStructure,
    MovableStationary,
    Moving,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityStatus {
    Available,
    Unavailable,
}

/// Where accepted geometry places an observation in one frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectLocation {
    pub observation: String,
    pub frame_index: usize,
    /// Mean of the front-band points, in reconstruction coordinates.
    pub position: [f64; 3],
    /// Mean camera depth of those points.
    pub depth: f64,
    pub points: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneObject {
    pub id: String,
    pub label: String,
    pub category: ObjectCategory,
    /// Semantic observation ids, in frame order.
    pub observations: Vec<String>,
    /// Semantic tracks this identity joins.
    pub source_tracks: Vec<String>,
    pub frames: Vec<usize>,
    pub locations: Vec<ObjectLocation>,
    pub motion: MotionClass,
    pub motion_reason: String,
    /// Largest spread of the evidence positions, relative to viewing depth.
    pub spread: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmbiguityKind {
    /// Positions agree but the candidates share a frame.
    Merge,
    /// One track whose positions disagree within itself.
    Split,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdentityAmbiguity {
    pub kind: AmbiguityKind,
    pub objects: Vec<String>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectIdentityArtifact {
    pub schema_version: u32,
    pub status: IdentityStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
    /// Identity of the semantic inventory document consumed.
    pub semantic_inventory: ContentHash,
    pub objects: Vec<SceneObject>,
    pub ambiguities: Vec<IdentityAmbiguity>,
}

impl ObjectIdentityArtifact {
    pub fn from_json(document: &str) -> Result<Self, String> {
        let artifact: Self = serde_json::from_str(document)
            .map_err(|error| format!("object identity artifact: {error}"))?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn to_json(&self) -> String {
        let mut document = serde_json::to_string_pretty(self).expect("object identities serialize");
        document.push('\n');
        document
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != OBJECT_IDENTITY_SCHEMA_VERSION {
            return Err(format!(
                "unsupported object identity schema version {}",
                self.schema_version
            ));
        }
        match self.status {
            IdentityStatus::Unavailable => {
                if self
                    .unavailable_reason
                    .as_deref()
                    .is_none_or(|r| r.trim().is_empty())
                    || !self.objects.is_empty()
                {
                    return Err("unavailable object identities need a reason and no objects".into());
                }
            }
            IdentityStatus::Available => {
                if self.unavailable_reason.is_some() {
                    return Err("available object identities cannot carry a reason".into());
                }
            }
        }
        let mut ids = BTreeSet::new();
        let mut observations = BTreeSet::new();
        for object in &self.objects {
            if !ids.insert(object.id.as_str()) {
                return Err(format!("object identity {} is repeated", object.id));
            }
            if object.observations.is_empty() {
                return Err(format!("object {} has no observations", object.id));
            }
            for observation in &object.observations {
                if !observations.insert(observation.as_str()) {
                    return Err(format!("observation {observation} belongs to two objects"));
                }
            }
            if object.motion_reason.trim().is_empty() {
                return Err(format!("object {} has no motion reason", object.id));
            }
            let structure = object.category == ObjectCategory::Structure;
            if structure != (object.motion == MotionClass::StaticStructure) {
                return Err(format!(
                    "object {} motion class disagrees with its category",
                    object.id
                ));
            }
        }
        for ambiguity in &self.ambiguities {
            if ambiguity
                .objects
                .iter()
                .any(|id| !ids.contains(id.as_str()))
                || ambiguity.reason.trim().is_empty()
            {
                return Err("object ambiguity references a missing object or has no reason".into());
            }
        }
        Ok(())
    }
}

fn mask_contains(mask: &SemanticMask, column: u32, row: u32) -> bool {
    if column >= mask.width || row >= mask.height {
        return false;
    }
    let pixel = u64::from(row) * u64::from(mask.width) + u64::from(column);
    let mut position = 0_u64;
    for (index, &run) in mask.counts.iter().enumerate() {
        if pixel < position + run {
            return index % 2 == 1;
        }
        position += run;
    }
    false
}

fn inside(observation: &SemanticObservation, pixel: [f64; 2]) -> bool {
    let [x0, y0, x1, y1] = observation.bbox;
    let [x, y] = pixel;
    if !(x >= x0 && x < x1 && y >= y0 && y < y1) {
        return false;
    }
    match &observation.mask {
        Some(mask) => mask_contains(mask, x.floor() as u32, y.floor() as u32),
        None => true,
    }
}

/// Localize an observation with the accepted points that project inside it.
fn locate(
    observation: &SemanticObservation,
    camera: &FrameCamera,
    points: &[[f64; 3]],
) -> Option<ObjectLocation> {
    let mut hits: Vec<(f64, [f64; 3])> = Vec::new();
    for point in points {
        let local = camera.camera_point(*point);
        if !(local[2] > 1.0e-9 && local.iter().all(|value| value.is_finite())) {
            continue;
        }
        let pixel = [
            camera.focal_pixels * local[0] / local[2] + camera.principal_point[0],
            camera.focal_pixels * local[1] / local[2] + camera.principal_point[1],
        ];
        if inside(observation, pixel) {
            hits.push((local[2], *point));
        }
    }
    let nearest = hits
        .iter()
        .map(|(depth, _)| *depth)
        .fold(f64::INFINITY, f64::min);
    let front: Vec<&(f64, [f64; 3])> = hits
        .iter()
        .filter(|(depth, _)| *depth <= nearest * (1.0 + FRONT_DEPTH_BAND))
        .collect();
    if front.len() < MIN_LOCALIZING_POINTS {
        return None;
    }
    let count = front.len() as f64;
    let mut position = [0.0; 3];
    let mut depth = 0.0;
    for (point_depth, point) in &front {
        depth += point_depth / count;
        for axis in 0..3 {
            position[axis] += point[axis] / count;
        }
    }
    Some(ObjectLocation {
        observation: observation.id.clone(),
        frame_index: observation.frame_index,
        position,
        depth,
        points: front.len(),
    })
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3)
        .map(|axis| (a[axis] - b[axis]).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Largest pairwise distance relative to the mean viewing depth.
fn spread(locations: &[&ObjectLocation]) -> Option<f64> {
    if locations.len() < 2 {
        return None;
    }
    let depth = locations.iter().map(|l| l.depth).sum::<f64>() / locations.len() as f64;
    let mut largest = 0.0_f64;
    for (index, first) in locations.iter().enumerate() {
        for second in &locations[index + 1..] {
            largest = largest.max(distance(first.position, second.position));
        }
    }
    Some(largest / depth.max(1.0e-9))
}

struct Candidate {
    label: String,
    observations: Vec<String>,
    tracks: Vec<String>,
    frames: BTreeSet<usize>,
    locations: Vec<ObjectLocation>,
}

fn stable_id(label: &str, observations: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(label.as_bytes());
    for observation in observations {
        hasher.update([0]);
        hasher.update(observation.as_bytes());
    }
    let digest = hasher.finalize();
    let hex: String = digest[..6]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("obj-{hex}")
}

/// Stable object identities and motion classes for `inventory`, localized
/// with registered `cameras` and accepted `points`.
pub fn identify_objects(
    inventory: &SemanticInventoryArtifact,
    inventory_hash: ContentHash,
    cameras: &[FrameCamera],
    points: &[[f64; 3]],
) -> ObjectIdentityArtifact {
    if inventory.status == SegmentationStatus::Unavailable {
        return ObjectIdentityArtifact {
            schema_version: OBJECT_IDENTITY_SCHEMA_VERSION,
            status: IdentityStatus::Unavailable,
            unavailable_reason: Some(format!(
                "no semantic evidence: {}",
                inventory
                    .unavailable_reason
                    .as_deref()
                    .unwrap_or("segmentation unavailable")
            )),
            semantic_inventory: inventory_hash,
            objects: Vec::new(),
            ambiguities: Vec::new(),
        };
    }
    let cameras: BTreeMap<usize, &FrameCamera> = cameras
        .iter()
        .map(|camera| (camera.frame_index, camera))
        .collect();
    let observations: BTreeMap<&str, &SemanticObservation> = inventory
        .observations
        .iter()
        .map(|observation| (observation.id.as_str(), observation))
        .collect();
    let location_of = |id: &str| {
        let observation = observations[id];
        cameras
            .get(&observation.frame_index)
            .and_then(|camera| locate(observation, camera, points))
    };

    // Candidates: every track, then every untracked observation.
    let mut tracked = BTreeSet::new();
    let mut candidates: Vec<Candidate> = Vec::new();
    for track in &inventory.tracks {
        let mut members: Vec<&SemanticObservation> = track
            .observations
            .iter()
            .map(|id| observations[id.as_str()])
            .collect();
        members.sort_by(|a, b| a.frame_index.cmp(&b.frame_index).then(a.id.cmp(&b.id)));
        tracked.extend(track.observations.iter().map(String::as_str));
        candidates.push(Candidate {
            label: track.label.clone(),
            observations: members.iter().map(|o| o.id.clone()).collect(),
            tracks: vec![track.id.clone()],
            frames: members.iter().map(|o| o.frame_index).collect(),
            locations: members.iter().filter_map(|o| location_of(&o.id)).collect(),
        });
    }
    for observation in &inventory.observations {
        if tracked.contains(observation.id.as_str()) {
            continue;
        }
        candidates.push(Candidate {
            label: observation.label.clone(),
            observations: vec![observation.id.clone()],
            tracks: Vec::new(),
            frames: BTreeSet::from([observation.frame_index]),
            locations: location_of(&observation.id).into_iter().collect(),
        });
    }

    let structure = |label: &str| {
        STRUCTURE_LABELS
            .iter()
            .any(|term| label.eq_ignore_ascii_case(term))
    };
    let consistent = |first: &Candidate, second: &Candidate| {
        let locations: Vec<&ObjectLocation> =
            first.locations.iter().chain(&second.locations).collect();
        !first.locations.is_empty()
            && !second.locations.is_empty()
            && spread(&locations).is_some_and(|value| value <= STATIONARY_TOLERANCE)
    };

    // Merge same-label, frame-disjoint, position-consistent movable candidates
    // (largest first, deterministic order); structure is not merged by
    // position (one wall plane spans several walls' views).
    candidates.sort_by(|a, b| {
        b.observations
            .len()
            .cmp(&a.observations.len())
            .then(a.observations.cmp(&b.observations))
    });
    let mut merged: Vec<Candidate> = Vec::new();
    let mut pending_ambiguities: Vec<(usize, usize)> = Vec::new();
    for candidate in candidates {
        let target = if structure(&candidate.label) {
            None
        } else {
            merged.iter().position(|existing| {
                existing.label == candidate.label
                    && existing.frames.is_disjoint(&candidate.frames)
                    && consistent(existing, &candidate)
            })
        };
        match target {
            Some(index) => {
                let existing = &mut merged[index];
                existing.observations.extend(candidate.observations);
                existing.tracks.extend(candidate.tracks);
                existing.frames.extend(candidate.frames);
                existing.locations.extend(candidate.locations);
            }
            None => {
                if !structure(&candidate.label) {
                    for (index, existing) in merged.iter().enumerate() {
                        if existing.label == candidate.label
                            && !existing.frames.is_disjoint(&candidate.frames)
                            && consistent(existing, &candidate)
                        {
                            pending_ambiguities.push((index, merged.len()));
                        }
                    }
                }
                merged.push(candidate);
            }
        }
    }

    let mut objects: Vec<SceneObject> = Vec::new();
    let mut ambiguities = Vec::new();
    for candidate in &mut merged {
        candidate
            .observations
            .sort_by_key(|id| (observations[id.as_str()].frame_index, id.clone()));
        candidate.locations.sort_by(|a, b| {
            a.frame_index
                .cmp(&b.frame_index)
                .then(a.observation.cmp(&b.observation))
        });
        candidate.tracks.sort();
    }
    for candidate in &merged {
        let id = stable_id(&candidate.label, &candidate.observations);
        let category = if structure(&candidate.label) {
            ObjectCategory::Structure
        } else {
            ObjectCategory::Object
        };
        let locations: Vec<&ObjectLocation> = candidate.locations.iter().collect();
        let spread_value = spread(&locations);
        let localized_frames: BTreeSet<usize> = candidate
            .locations
            .iter()
            .map(|location| location.frame_index)
            .collect();
        let (motion, reason) = match (category, spread_value) {
            (ObjectCategory::Structure, _) => (
                MotionClass::StaticStructure,
                format!("`{}` is architecture: fixed scene structure", candidate.label),
            ),
            (_, Some(value)) if localized_frames.len() >= 2 && value <= STATIONARY_TOLERANCE => (
                MotionClass::MovableStationary,
                format!(
                    "positions in {} localized frames agree within {:.1}% of viewing depth (bound {:.1}%)",
                    localized_frames.len(),
                    value * 100.0,
                    STATIONARY_TOLERANCE * 100.0
                ),
            ),
            (_, Some(value)) if localized_frames.len() >= 2 => (
                MotionClass::Moving,
                format!(
                    "positions in {} localized frames differ by {:.1}% of viewing depth (bound {:.1}%)",
                    localized_frames.len(),
                    value * 100.0,
                    STATIONARY_TOLERANCE * 100.0
                ),
            ),
            _ => (
                MotionClass::Unknown,
                format!(
                    "accepted geometry localizes this object in {} registered frame(s); two are needed to judge motion",
                    localized_frames.len()
                ),
            ),
        };
        if candidate.tracks.len() == 1 && motion == MotionClass::Moving {
            ambiguities.push(IdentityAmbiguity {
                kind: AmbiguityKind::Split,
                objects: vec![id.clone()],
                reason: "one semantic track whose evidence positions disagree: either a moving object or two objects linked by the tracker".into(),
            });
        }
        objects.push(SceneObject {
            id,
            label: candidate.label.clone(),
            category,
            observations: candidate.observations.clone(),
            source_tracks: candidate.tracks.clone(),
            frames: candidate.frames.iter().copied().collect(),
            locations: candidate.locations.clone(),
            motion,
            motion_reason: reason,
            spread: spread_value,
        });
    }
    for (first, second) in pending_ambiguities {
        ambiguities.push(IdentityAmbiguity {
            kind: AmbiguityKind::Merge,
            objects: vec![objects[first].id.clone(), objects[second].id.clone()],
            reason: "same label at an agreeing position, but both appear in one frame: kept as two objects".into(),
        });
    }
    objects.sort_by(|a, b| a.id.cmp(&b.id));
    ObjectIdentityArtifact {
        schema_version: OBJECT_IDENTITY_SCHEMA_VERSION,
        status: IdentityStatus::Available,
        unavailable_reason: None,
        semantic_inventory: inventory_hash,
        objects,
        ambiguities,
    }
}

#[cfg(test)]
#[path = "object_identity_tests.rs"]
mod tests;
