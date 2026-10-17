//! Semantic inventory evidence (#60, Decision Atlas 8.7/8.8).
//!
//! A versioned, provider-neutral record of what a local segmentation provider
//! proposed over the browser-owned keyframes: labeled boxes and masks in
//! keyframe pixel space, cross-frame tracks, explicit missing/weak/ambiguous
//! gaps, the provider/model/runtime identity and the vocabulary actually used.
//! It is evidence only: it carries no cameras, poses, depth or geometry, and
//! no geometry producer consumes it. An `unavailable` result records honestly
//! that the provider could not run (for example without a CUDA GPU, Atlas
//! 4.6), with its reason and no invented observations.
//!
//! The model runtime is a separate concern (a local sidecar today, in-process
//! ONNX and browser inference later); every runtime produces this artifact,
//! which [`SemanticInventoryArtifact::validate`] checks fail-closed.

use crate::scene_artifacts::KeyframesArtifact;
use crate::scene_project::ContentHash;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SEMANTIC_INVENTORY_SCHEMA_VERSION: u32 = 1;

/// Default open-vocabulary prompts for room scenes: individual architecture
/// elements and meaningful furniture/objects (owner granularity decision on
/// #60). Individual instances are observations and tracks, not terms.
pub fn default_room_vocabulary() -> Vec<String> {
    [
        "wall", "floor", "ceiling", "door", "window", "table", "desk", "chair", "sofa", "bed",
        "shelf", "cabinet", "lamp", "monitor", "keyboard", "plant", "box", "toy", "person",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentationStatus {
    Available,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticInventoryArtifact {
    pub schema_version: u32,
    pub status: SegmentationStatus,
    /// Required exactly when `status` is `unavailable`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
    pub provider: SemanticProvider,
    pub vocabulary: SemanticVocabulary,
    /// The keyframes the provider was offered, with their identity.
    pub frames: Vec<SemanticFrame>,
    pub observations: Vec<SemanticObservation>,
    pub tracks: Vec<SemanticTrack>,
    pub uncertainties: Vec<SemanticUncertainty>,
    pub measurements: Option<SemanticMeasurements>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticProvider {
    /// The manifest provider id and revision.
    pub id: String,
    pub revision: String,
    pub models: Vec<SemanticModel>,
    pub runtime: Option<SemanticRuntime>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticModelRole {
    Detector,
    Segmenter,
    /// Image inpainting (clean plates, #64).
    Inpainter,
    /// Image-to-3D object generation (#68).
    Generator,
}

/// A pinned model identity: id, version and where it comes from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticModel {
    pub role: SemanticModelRole,
    pub id: String,
    pub version: String,
    pub url: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticRuntime {
    pub name: String,
    pub version: String,
    /// `cuda` or `cpu` (or another local device class).
    pub device: String,
    /// The exact device, e.g. `NVIDIA GeForce RTX 3060 Ti`.
    pub device_name: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VocabularySource {
    /// [`default_room_vocabulary`].
    Default,
    /// The project's per-operation override, verbatim.
    Project,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticVocabulary {
    pub source: VocabularySource,
    /// Prompt terms in their recorded order; every label is one of them.
    pub terms: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticFrame {
    pub frame_index: usize,
    pub width: u32,
    pub height: u32,
    pub content_hash: ContentHash,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticObservation {
    pub id: String,
    pub frame_index: usize,
    pub label: String,
    /// `[x_min, y_min, x_max, y_max]` in continuous keyframe pixel
    /// coordinates, origin top-left.
    pub bbox: [f64; 4],
    pub detection_confidence: f64,
    pub mask: Option<SemanticMask>,
    pub mask_confidence: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskEncoding {
    /// Row-major run lengths alternating background/foreground, starting
    /// with background (the first run may be 0).
    RleRowMajor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticMask {
    pub encoding: MaskEncoding,
    pub width: u32,
    pub height: u32,
    pub counts: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticTrack {
    pub id: String,
    pub label: String,
    /// Observation ids, one object candidate across frames.
    pub observations: Vec<String>,
    pub association_confidence: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyKind {
    Missing,
    Weak,
    Ambiguous,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticUncertainty {
    pub kind: UncertaintyKind,
    pub frame_index: usize,
    pub label: String,
    pub observation: Option<String>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticMeasurements {
    pub elapsed_seconds: f64,
    pub peak_gpu_memory_bytes: Option<u64>,
}

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn probability(value: f64, what: &str) -> Result<(), String> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(format!(
            "semantic inventory {what} {value} is not in [0, 1]"
        ))
    }
}

impl SemanticVocabulary {
    pub(crate) fn validate(&self) -> Result<(), String> {
        validate_terms(&self.terms)
    }
}

/// A vocabulary is non-empty, without blank or duplicate terms.
pub(crate) fn validate_terms(terms: &[String]) -> Result<(), String> {
    if terms.is_empty() {
        return Err("semantic vocabulary must not be empty".into());
    }
    let mut seen = BTreeSet::new();
    for term in terms {
        if blank(term) {
            return Err("semantic vocabulary has a blank term".into());
        }
        if !seen.insert(term.as_str()) {
            return Err(format!("semantic vocabulary repeats `{term}`"));
        }
    }
    Ok(())
}

impl SemanticMask {
    fn validate(&self, frame: &SemanticFrame, bbox: &[f64; 4], id: &str) -> Result<(), String> {
        if self.width != frame.width || self.height != frame.height {
            return Err(format!(
                "semantic observation {id} mask is {}x{} for a {}x{} frame",
                self.width, self.height, frame.width, frame.height
            ));
        }
        let total = u64::from(self.width) * u64::from(self.height);
        let width = u64::from(self.width);
        let [x0, y0, x1, y1] = *bbox;
        let mut position = 0_u64;
        let mut foreground_pixels = 0_u64;
        for (index, &run) in self.counts.iter().enumerate() {
            let end = position
                .checked_add(run)
                .filter(|end| *end <= total)
                .ok_or_else(|| format!("semantic observation {id} mask runs exceed the frame"))?;
            if index % 2 == 1 && run > 0 {
                foreground_pixels += run;
                // Every foreground pixel (col, row) must intersect the box.
                let mut start = position;
                while start < end {
                    let row = start / width;
                    let row_end = end.min((row + 1) * width);
                    let (first, last) = (start % width, (row_end - 1) % width);
                    let (row, first, last) = (row as f64, first as f64, last as f64);
                    if !(row + 1.0 > y0 && row < y1 && first + 1.0 > x0 && last < x1) {
                        return Err(format!(
                            "semantic observation {id} mask has foreground outside its box"
                        ));
                    }
                    start = row_end;
                }
            }
            position = end;
        }
        if position != total {
            return Err(format!(
                "semantic observation {id} mask runs cover {position} of {total} pixels"
            ));
        }
        if foreground_pixels == 0 {
            return Err(format!("semantic observation {id} mask has no foreground"));
        }
        Ok(())
    }
}

impl SemanticInventoryArtifact {
    pub fn from_json(document: &str) -> Result<Self, String> {
        let artifact: Self = serde_json::from_str(document)
            .map_err(|error| format!("semantic inventory artifact: {error}"))?;
        artifact.validate()?;
        Ok(artifact)
    }

    /// The canonical stored document.
    pub fn to_json(&self) -> String {
        let mut document =
            serde_json::to_string_pretty(self).expect("semantic inventory serializes");
        document.push('\n');
        document
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SEMANTIC_INVENTORY_SCHEMA_VERSION {
            return Err(format!(
                "unsupported semantic inventory schema version {}",
                self.schema_version
            ));
        }
        if blank(&self.provider.id) || blank(&self.provider.revision) {
            return Err("semantic inventory provider needs an id and a revision".into());
        }
        for model in &self.provider.models {
            if blank(&model.id) || blank(&model.version) || blank(&model.url) {
                return Err("semantic inventory model needs an id, version and url".into());
            }
        }
        if let Some(runtime) = &self.provider.runtime {
            if blank(&runtime.name) || blank(&runtime.version) || blank(&runtime.device) {
                return Err("semantic inventory runtime needs a name, version and device".into());
            }
        }
        self.vocabulary.validate()?;
        match self.status {
            SegmentationStatus::Unavailable => {
                if self.unavailable_reason.as_deref().is_none_or(blank) {
                    return Err("unavailable semantic inventory needs a reason".into());
                }
                if !self.observations.is_empty() || !self.tracks.is_empty() {
                    return Err(
                        "unavailable semantic inventory cannot carry observations or tracks".into(),
                    );
                }
            }
            SegmentationStatus::Available => {
                if self.unavailable_reason.is_some() {
                    return Err("available semantic inventory cannot carry a reason".into());
                }
                if self.provider.models.is_empty() || self.provider.runtime.is_none() {
                    return Err(
                        "available semantic inventory needs model and runtime identity".into(),
                    );
                }
            }
        }

        let mut frames = BTreeMap::new();
        for frame in &self.frames {
            if frame.width == 0 || frame.height == 0 {
                return Err(format!(
                    "semantic inventory frame {} is empty",
                    frame.frame_index
                ));
            }
            if frames.insert(frame.frame_index, frame).is_some() {
                return Err(format!(
                    "semantic inventory repeats frame {}",
                    frame.frame_index
                ));
            }
        }
        let terms: BTreeSet<&str> = self.vocabulary.terms.iter().map(String::as_str).collect();
        let mut labels = BTreeMap::new();
        for observation in &self.observations {
            let id = &observation.id;
            if blank(id) || labels.insert(id.as_str(), &observation.label).is_some() {
                return Err(format!(
                    "semantic observation id `{id}` is blank or repeated"
                ));
            }
            let frame = frames.get(&observation.frame_index).ok_or_else(|| {
                format!(
                    "semantic observation {id} names frame {} that is not recorded",
                    observation.frame_index
                )
            })?;
            if !terms.contains(observation.label.as_str()) {
                return Err(format!(
                    "semantic observation {id} label `{}` is not in the vocabulary",
                    observation.label
                ));
            }
            let [x0, y0, x1, y1] = observation.bbox;
            let (width, height) = (f64::from(frame.width), f64::from(frame.height));
            if !observation.bbox.iter().all(|value| value.is_finite())
                || !(0.0 <= x0 && x0 < x1 && x1 <= width && 0.0 <= y0 && y0 < y1 && y1 <= height)
            {
                return Err(format!(
                    "semantic observation {id} box {:?} is not inside its {}x{} frame",
                    observation.bbox, frame.width, frame.height
                ));
            }
            probability(observation.detection_confidence, "detection confidence")?;
            match (&observation.mask, observation.mask_confidence) {
                (Some(mask), Some(confidence)) => {
                    probability(confidence, "mask confidence")?;
                    mask.validate(frame, &observation.bbox, id)?;
                }
                (None, None) => {}
                _ => {
                    return Err(format!(
                        "semantic observation {id} has a mask without its confidence or the reverse"
                    ))
                }
            }
        }
        let mut tracks = BTreeSet::new();
        for track in &self.tracks {
            if blank(&track.id) || !tracks.insert(track.id.as_str()) {
                return Err(format!(
                    "semantic track id `{}` is blank or repeated",
                    track.id
                ));
            }
            if track.observations.is_empty() {
                return Err(format!("semantic track {} is empty", track.id));
            }
            let mut members = BTreeSet::new();
            for member in &track.observations {
                if !members.insert(member.as_str()) {
                    return Err(format!(
                        "semantic track {} repeats observation {member}",
                        track.id
                    ));
                }
                match labels.get(member.as_str()) {
                    Some(label) if **label == track.label => {}
                    Some(_) => {
                        return Err(format!(
                            "semantic track {} links observations of another label",
                            track.id
                        ))
                    }
                    None => {
                        return Err(format!(
                            "semantic track {} references missing observation {member}",
                            track.id
                        ))
                    }
                }
            }
            probability(track.association_confidence, "association confidence")?;
        }
        for uncertainty in &self.uncertainties {
            if !frames.contains_key(&uncertainty.frame_index) {
                return Err(format!(
                    "semantic uncertainty names frame {} that is not recorded",
                    uncertainty.frame_index
                ));
            }
            if !terms.contains(uncertainty.label.as_str()) {
                return Err(format!(
                    "semantic uncertainty label `{}` is not in the vocabulary",
                    uncertainty.label
                ));
            }
            if blank(&uncertainty.reason) {
                return Err("semantic uncertainty needs a reason".into());
            }
            if let Some(observation) = &uncertainty.observation {
                if !labels.contains_key(observation.as_str()) {
                    return Err(format!(
                        "semantic uncertainty references missing observation {observation}"
                    ));
                }
            }
        }
        if let Some(measurements) = &self.measurements {
            if !(measurements.elapsed_seconds.is_finite() && measurements.elapsed_seconds >= 0.0) {
                return Err(
                    "semantic inventory elapsed time must be finite and non-negative".into(),
                );
            }
        }
        Ok(())
    }

    /// Every recorded frame is a source keyframe with the same size and pixels.
    pub fn validate_against_keyframes(&self, keyframes: &KeyframesArtifact) -> Result<(), String> {
        for frame in &self.frames {
            let keyframe = keyframes
                .frames
                .iter()
                .find(|keyframe| keyframe.frame_index == frame.frame_index)
                .ok_or_else(|| {
                    format!(
                        "semantic inventory frame {} is not a source keyframe",
                        frame.frame_index
                    )
                })?;
            if keyframe.width != frame.width
                || keyframe.height != frame.height
                || keyframe.content_hash != frame.content_hash
            {
                return Err(format!(
                    "semantic inventory frame {} differs from its source keyframe",
                    frame.frame_index
                ));
            }
        }
        Ok(())
    }
}
