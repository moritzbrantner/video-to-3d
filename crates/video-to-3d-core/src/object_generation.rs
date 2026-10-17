//! Generated object assets from multi-view reference packages (#68, Atlas 8.10).
//!
//! Every packaged movable object (#63) becomes one deterministic generation
//! request built from its primary reference view: the source keyframe, the
//! crop and the exact mask. A local image-to-3D provider (TripoSR today)
//! answers with a mesh, which is recorded as a **generative candidate** with
//! its request, provider and output identity. Generated geometry never
//! becomes scene authority: reconstructed evidence stays authoritative, and
//! the package's observed points travel with each candidate so placement
//! (#69/#70) can prefer reconstruction where evidence is sufficient. A failed
//! generation affects only that object.

use crate::object_references::ObjectReferencesArtifact;
use crate::scene_project::{ContentHash, ProjectPath};
use crate::semantic_inventory::{
    SemanticInventoryArtifact, SemanticMask, SemanticModel, SemanticRuntime,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const OBJECT_GENERATION_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationStatus {
    Available,
    Unavailable,
}

/// Always `generative_candidate`: never observed geometry or scene authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneratedClassification {
    GenerativeCandidate,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationRequest {
    /// Content-derived from the inputs and provider.
    pub request_id: String,
    pub object: String,
    pub label: String,
    pub observation: String,
    pub frame_index: usize,
    pub source_frame_hash: ContentHash,
    pub width: u32,
    pub height: u32,
    pub crop: [u32; 4],
    pub mask_hash: ContentHash,
    pub mask: SemanticMask,
    /// Observed accepted points of the object in its package: reconstruction
    /// evidence that placement weighs against the candidate.
    pub observed_points: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedMesh {
    /// Binary glTF of the candidate mesh, in the generator's own frame.
    pub path: ProjectPath,
    pub content_hash: ContentHash,
    pub vertices: u64,
    pub triangles: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedObject {
    pub request: GenerationRequest,
    pub classification: GeneratedClassification,
    pub output: Option<GeneratedMesh>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationProvider {
    pub id: String,
    pub revision: String,
    pub models: Vec<SemanticModel>,
    pub runtime: Option<SemanticRuntime>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectGenerationArtifact {
    pub schema_version: u32,
    pub status: GenerationStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
    pub provider: GenerationProvider,
    pub object_references: ContentHash,
    pub objects: Vec<GeneratedObject>,
}

impl ObjectGenerationArtifact {
    pub fn from_json(document: &str) -> Result<Self, String> {
        let artifact: Self = serde_json::from_str(document)
            .map_err(|error| format!("object generation artifact: {error}"))?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn to_json(&self) -> String {
        let mut document =
            serde_json::to_string_pretty(self).expect("object generation serializes");
        document.push('\n');
        document
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != OBJECT_GENERATION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported object generation schema version {}",
                self.schema_version
            ));
        }
        let unavailable = self.status == GenerationStatus::Unavailable;
        if unavailable
            != self
                .unavailable_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
        {
            return Err("object generation: unavailable needs a reason (and only then)".into());
        }
        let mut requests = BTreeSet::new();
        let mut objects = BTreeSet::new();
        for object in &self.objects {
            let request = &object.request;
            if !requests.insert(request.request_id.as_str())
                || !objects.insert(request.object.as_str())
            {
                return Err(format!(
                    "object generation request {} is repeated",
                    request.request_id
                ));
            }
            let [x0, y0, x1, y1] = request.crop;
            if x0 >= x1 || y0 >= y1 || x1 > request.width || y1 > request.height {
                return Err(format!(
                    "generation request {} has an invalid crop",
                    request.request_id
                ));
            }
            match (&object.output, &object.failure) {
                (Some(mesh), None) if mesh.triangles > 0 && !unavailable => {}
                (None, Some(reason)) if !reason.trim().is_empty() => {}
                _ => {
                    return Err(format!(
                        "generation request {} needs exactly a mesh or a failure reason",
                        request.request_id
                    ))
                }
            }
        }
        Ok(())
    }
}

/// One deterministic request per packaged object, from its primary view.
pub fn plan_generation(
    inventory: &SemanticInventoryArtifact,
    references: &ObjectReferencesArtifact,
    provider: (&str, &str),
) -> Vec<GenerationRequest> {
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
    let mut requests = Vec::new();
    for package in &references.packages {
        let view = &package.primary;
        let (Some(observation), Some(frame)) = (
            observations.get(view.observation.as_str()),
            frames.get(&view.frame_index),
        ) else {
            continue;
        };
        let Some(mask) = &observation.mask else {
            continue;
        };
        let identity = ContentHash::of_bytes(
            &serde_json::to_vec(&serde_json::json!({
                "object": package.object,
                "frame": frame.content_hash,
                "crop": view.crop,
                "mask": view.mask_hash,
                "provider": provider.0,
                "revision": provider.1,
            }))
            .expect("generation identity serializes"),
        );
        requests.push(GenerationRequest {
            request_id: format!(
                "gen-{}",
                &identity.as_str()["sha256:".len().."sha256:".len() + 12]
            ),
            object: package.object.clone(),
            label: package.label.clone(),
            observation: view.observation.clone(),
            frame_index: view.frame_index,
            source_frame_hash: frame.content_hash.clone(),
            width: frame.width,
            height: frame.height,
            crop: view.crop,
            mask_hash: view.mask_hash.clone(),
            mask: mask.clone(),
            observed_points: package.observed_points.len(),
        });
    }
    requests.sort_by(|a, b| a.object.cmp(&b.object));
    requests
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_masks::MaskConsistency;
    use crate::object_references::{
        ObjectReferencePackage, ReferenceView, ReferencesStatus, OBJECT_REFERENCES_SCHEMA_VERSION,
    };
    use crate::semantic_inventory::{
        MaskEncoding, SegmentationStatus, SemanticFrame, SemanticObservation, SemanticProvider,
        SemanticVocabulary, VocabularySource, SEMANTIC_INVENTORY_SCHEMA_VERSION,
    };

    fn inputs() -> (SemanticInventoryArtifact, ObjectReferencesArtifact) {
        let mask = SemanticMask {
            encoding: MaskEncoding::RleRowMajor,
            width: 4,
            height: 4,
            counts: vec![5, 2, 2, 2, 5],
        };
        let hash = ContentHash::of_bytes(b"frame");
        let inventory = SemanticInventoryArtifact {
            schema_version: SEMANTIC_INVENTORY_SCHEMA_VERSION,
            status: SegmentationStatus::Available,
            unavailable_reason: None,
            provider: SemanticProvider {
                id: "t".into(),
                revision: "1".into(),
                models: Vec::new(),
                runtime: None,
            },
            vocabulary: SemanticVocabulary {
                source: VocabularySource::Project,
                terms: vec!["toy".into()],
            },
            frames: vec![SemanticFrame {
                frame_index: 3,
                width: 4,
                height: 4,
                content_hash: hash.clone(),
            }],
            observations: vec![SemanticObservation {
                id: "o".into(),
                frame_index: 3,
                label: "toy".into(),
                bbox: [1.0, 1.0, 3.0, 3.0],
                detection_confidence: 0.9,
                mask: Some(mask),
                mask_confidence: Some(0.9),
            }],
            tracks: Vec::new(),
            uncertainties: Vec::new(),
            measurements: None,
        };
        let view = ReferenceView {
            observation: "o".into(),
            frame_index: 3,
            frame_hash: hash,
            crop: [0, 0, 4, 4],
            bbox: [1.0, 1.0, 3.0, 3.0],
            mask_hash: ContentHash::of_bytes(b"mask"),
            mask_area_pixels: 4,
            detection_confidence: 0.9,
            mask_confidence: 0.9,
            consistency: MaskConsistency::Consistent,
        };
        let references = ObjectReferencesArtifact {
            schema_version: OBJECT_REFERENCES_SCHEMA_VERSION,
            status: ReferencesStatus::Available,
            unavailable_reason: None,
            semantic_inventory: ContentHash::of_bytes(b"i"),
            object_identity: ContentHash::of_bytes(b"o"),
            object_masks: ContentHash::of_bytes(b"m"),
            packages: vec![ObjectReferencePackage {
                object: "obj-1".into(),
                label: "toy".into(),
                source_tracks: Vec::new(),
                description: "toy".into(),
                primary: view,
                additional: Vec::new(),
                rejected: Vec::new(),
                observed_points: vec![[0.0, 0.0, 1.0]; 7],
                extent: None,
            }],
            unpackaged: Vec::new(),
        };
        (inventory, references)
    }

    #[test]
    fn requests_are_deterministic_and_keep_their_source_identity() {
        let (inventory, references) = inputs();
        let requests = plan_generation(&inventory, &references, ("triposr", "r1"));
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.object, "obj-1");
        assert_eq!(request.frame_index, 3);
        assert_eq!(request.source_frame_hash, inventory.frames[0].content_hash);
        assert_eq!(request.observed_points, 7);
        assert_eq!(
            plan_generation(&inventory, &references, ("triposr", "r1")),
            requests
        );
        let other = plan_generation(&inventory, &references, ("other", "r1"));
        assert_ne!(
            other[0].request_id, request.request_id,
            "provider is identity"
        );
    }

    #[test]
    fn a_generation_has_exactly_a_mesh_or_a_failure_and_stays_a_candidate() {
        let (inventory, references) = inputs();
        let request = plan_generation(&inventory, &references, ("triposr", "r1")).remove(0);
        let mut artifact = ObjectGenerationArtifact {
            schema_version: OBJECT_GENERATION_SCHEMA_VERSION,
            status: GenerationStatus::Available,
            unavailable_reason: None,
            provider: GenerationProvider {
                id: "triposr".into(),
                revision: "r1".into(),
                models: Vec::new(),
                runtime: None,
            },
            object_references: ContentHash::of_bytes(b"r"),
            objects: vec![GeneratedObject {
                request,
                classification: GeneratedClassification::GenerativeCandidate,
                output: None,
                failure: Some("out of memory".into()),
            }],
        };
        artifact.validate().unwrap();
        let value: serde_json::Value = serde_json::from_str(&artifact.to_json()).unwrap();
        assert_eq!(
            value["objects"][0]["classification"],
            "generative_candidate"
        );
        artifact.objects[0].failure = None;
        assert!(artifact.validate().is_err());
        artifact.objects[0].output = Some(GeneratedMesh {
            path: ProjectPath::new("artifacts/native/g/objects/x.glb").unwrap(),
            content_hash: ContentHash::of_bytes(b"glb"),
            vertices: 3,
            triangles: 1,
        });
        artifact.validate().unwrap();
        artifact.status = GenerationStatus::Unavailable;
        artifact.unavailable_reason = Some("no GPU".into());
        assert!(
            artifact.validate().is_err(),
            "unavailable generations carry no meshes"
        );
    }
}
