//! Clean-plate references (#64, Decision Atlas 8.9).
//!
//! For each representative source view of the packaged movable objects
//! (#63), the masks of every packaged object visible in that frame are
//! united, slightly dilated, and submitted with the original keyframe to a
//! local inpainting provider (LaMa today). The result is **generated/edited
//! appearance**: the artifact records the source frame and mask identities,
//! the deterministic request identity, and the edited pixels as a separate
//! sidecar, and never replaces the original keyframe, which stays the
//! validation source. A failed edit is reported and changes nothing else.

use crate::object_references::ObjectReferencesArtifact;
use crate::scene_project::{ContentHash, ProjectPath};
use crate::semantic_inventory::{MaskEncoding, SemanticInventoryArtifact, SemanticMask};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CLEAN_PLATE_SCHEMA_VERSION: u32 = 1;
/// Square dilation of the removal mask, in pixels: inpainting needs the
/// object's soft edge and shadow fringe inside the hole.
pub const MASK_DILATION_PIXELS: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlateStatus {
    Available,
    Unavailable,
}

/// Always `generated_edit`: a clean plate is never camera evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlateClassification {
    GeneratedEdit,
}

/// One deterministic edit request: remove `objects` from `frame_index`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlateRequest {
    /// Content-derived: equal sources, masks and provider give equal ids.
    pub edit_id: String,
    pub frame_index: usize,
    pub source_frame_hash: ContentHash,
    pub width: u32,
    pub height: u32,
    /// Objects removed, sorted.
    pub objects: Vec<String>,
    /// Semantic observations whose masks were united, sorted.
    pub observations: Vec<String>,
    /// Content hashes of those masks, in observation order.
    pub mask_hashes: Vec<ContentHash>,
    /// The dilated union actually submitted.
    pub removal_mask: SemanticMask,
    /// Packaged objects visible here that have no mask and were not removed.
    pub unremoved: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlateOutput {
    pub path: ProjectPath,
    pub content_hash: ContentHash,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanPlate {
    pub request: PlateRequest,
    pub classification: PlateClassification,
    /// Raw RGBA8 of the edited frame; `None` when the edit failed.
    pub output: Option<PlateOutput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlateProvider {
    pub id: String,
    pub revision: String,
    pub models: Vec<crate::semantic_inventory::SemanticModel>,
    pub runtime: Option<crate::semantic_inventory::SemanticRuntime>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanPlateArtifact {
    pub schema_version: u32,
    pub status: PlateStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
    pub provider: PlateProvider,
    pub object_references: ContentHash,
    pub plates: Vec<CleanPlate>,
}

impl CleanPlateArtifact {
    pub fn from_json(document: &str) -> Result<Self, String> {
        let artifact: Self = serde_json::from_str(document)
            .map_err(|error| format!("clean plate artifact: {error}"))?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn to_json(&self) -> String {
        let mut document = serde_json::to_string_pretty(self).expect("clean plates serialize");
        document.push('\n');
        document
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != CLEAN_PLATE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported clean plate schema version {}",
                self.schema_version
            ));
        }
        let unavailable = self.status == PlateStatus::Unavailable;
        if unavailable
            != self
                .unavailable_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
        {
            return Err("clean plates: unavailable needs a reason (and only then)".into());
        }
        let mut ids = BTreeSet::new();
        for plate in &self.plates {
            let request = &plate.request;
            if !ids.insert(request.edit_id.as_str()) {
                return Err(format!("clean plate {} is repeated", request.edit_id));
            }
            if request.objects.is_empty() || request.observations.len() != request.mask_hashes.len()
            {
                return Err(format!(
                    "clean plate {} removes nothing or misses mask identities",
                    request.edit_id
                ));
            }
            let mask = &request.removal_mask;
            let total: u64 = mask.counts.iter().sum();
            if mask.width != request.width
                || mask.height != request.height
                || total != u64::from(request.width) * u64::from(request.height)
            {
                return Err(format!(
                    "clean plate {} removal mask does not match its frame",
                    request.edit_id
                ));
            }
            match (&plate.output, &plate.failure) {
                (Some(_), None) => {}
                (None, Some(reason)) if !reason.trim().is_empty() => {}
                _ => {
                    return Err(format!(
                        "clean plate {} needs exactly an output or a failure reason",
                        request.edit_id
                    ))
                }
            }
            if unavailable && plate.output.is_some() {
                return Err("unavailable clean plates cannot carry outputs".into());
            }
        }
        Ok(())
    }
}

fn decode(mask: &SemanticMask) -> Vec<bool> {
    let mut pixels = Vec::with_capacity((mask.width * mask.height) as usize);
    for (index, &run) in mask.counts.iter().enumerate() {
        pixels.extend(std::iter::repeat_n(index % 2 == 1, run as usize));
    }
    pixels
}

/// Row-major run lengths, background first.
pub(crate) fn encode(width: u32, height: u32, pixels: &[bool]) -> SemanticMask {
    let mut counts = Vec::new();
    let (mut current, mut run) = (false, 0_u64);
    for &value in pixels {
        if value != current {
            counts.push(run);
            (current, run) = (value, 0);
        }
        run += 1;
    }
    counts.push(run);
    SemanticMask {
        encoding: MaskEncoding::RleRowMajor,
        width,
        height,
        counts,
    }
}

fn dilate(width: u32, height: u32, pixels: &[bool], radius: u32) -> Vec<bool> {
    let (w, h, r) = (width as i64, height as i64, radius as i64);
    // Separable square dilation: rows, then columns.
    let mut rows = vec![false; pixels.len()];
    for y in 0..h {
        let mut last: i64 = i64::MIN / 2;
        let mut next = vec![i64::MAX / 2; w as usize];
        let mut upcoming = i64::MAX / 2;
        for x in (0..w).rev() {
            if pixels[(y * w + x) as usize] {
                upcoming = x;
            }
            next[x as usize] = upcoming;
        }
        for x in 0..w {
            if pixels[(y * w + x) as usize] {
                last = x;
            }
            rows[(y * w + x) as usize] = x - last <= r || next[x as usize] - x <= r;
        }
    }
    let mut out = vec![false; pixels.len()];
    for x in 0..w {
        let mut last: i64 = i64::MIN / 2;
        let mut next = vec![i64::MAX / 2; h as usize];
        let mut upcoming = i64::MAX / 2;
        for y in (0..h).rev() {
            if rows[(y * w + x) as usize] {
                upcoming = y;
            }
            next[y as usize] = upcoming;
        }
        for y in 0..h {
            if rows[(y * w + x) as usize] {
                last = y;
            }
            out[(y * w + x) as usize] = y - last <= r || next[y as usize] - y <= r;
        }
    }
    out
}

/// One deterministic request per representative frame of the packages.
pub fn plan_clean_plates(
    inventory: &SemanticInventoryArtifact,
    references: &ObjectReferencesArtifact,
    provider: (&str, &str),
) -> Vec<PlateRequest> {
    let observations: BTreeMap<&str, _> = inventory
        .observations
        .iter()
        .map(|observation| (observation.id.as_str(), observation))
        .collect();
    let frames: BTreeMap<usize, _> = inventory
        .frames
        .iter()
        .map(|frame| (frame.frame_index, frame))
        .collect();
    // frame -> (object -> observation ids with that object in the frame)
    let mut by_frame: BTreeMap<usize, BTreeMap<String, Vec<String>>> = BTreeMap::new();
    let mut representative: BTreeSet<usize> = BTreeSet::new();
    for package in &references.packages {
        for view in std::iter::once(&package.primary).chain(&package.additional) {
            representative.insert(view.frame_index);
            by_frame
                .entry(view.frame_index)
                .or_default()
                .entry(package.object.clone())
                .or_default()
                .push(view.observation.clone());
        }
        for view in &package.rejected {
            by_frame
                .entry(view.frame_index)
                .or_default()
                .entry(package.object.clone())
                .or_default()
                .push(view.observation.clone());
        }
    }
    let mut requests = Vec::new();
    for frame_index in representative {
        let Some(frame) = frames.get(&frame_index) else {
            continue;
        };
        let mut union = vec![false; (frame.width * frame.height) as usize];
        let mut objects = Vec::new();
        let mut used = Vec::new();
        let mut unremoved = Vec::new();
        for (object, ids) in &by_frame[&frame_index] {
            let masked: Vec<&String> = ids
                .iter()
                .filter(|id| {
                    observations
                        .get(id.as_str())
                        .is_some_and(|observation| observation.mask.is_some())
                })
                .collect();
            if masked.is_empty() {
                unremoved.push(object.clone());
                continue;
            }
            objects.push(object.clone());
            for id in masked {
                let mask = observations[id.as_str()].mask.as_ref().unwrap();
                for (pixel, value) in union.iter_mut().zip(decode(mask)) {
                    *pixel |= value;
                }
                used.push(id.clone());
            }
        }
        if objects.is_empty() {
            continue;
        }
        used.sort();
        used.dedup();
        objects.sort();
        let mask_hashes: Vec<ContentHash> = used
            .iter()
            .map(|id| {
                ContentHash::of_bytes(
                    &serde_json::to_vec(observations[id.as_str()].mask.as_ref().unwrap())
                        .expect("masks serialize"),
                )
            })
            .collect();
        let removal_mask = encode(
            frame.width,
            frame.height,
            &dilate(frame.width, frame.height, &union, MASK_DILATION_PIXELS),
        );
        let identity = ContentHash::of_bytes(
            &serde_json::to_vec(&serde_json::json!({
                "frame": frame.content_hash,
                "mask": removal_mask,
                "provider": provider.0,
                "revision": provider.1,
            }))
            .expect("plate identity serializes"),
        );
        requests.push(PlateRequest {
            edit_id: format!(
                "plate-{frame_index}-{}",
                &identity.as_str()["sha256:".len().."sha256:".len() + 12]
            ),
            frame_index,
            source_frame_hash: frame.content_hash.clone(),
            width: frame.width,
            height: frame.height,
            objects,
            observations: used,
            mask_hashes,
            removal_mask,
            unremoved,
        });
    }
    requests
}

#[cfg(test)]
#[path = "clean_plate_tests.rs"]
mod tests;
