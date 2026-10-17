use super::*;
use crate::object_masks::MaskConsistency;
use crate::object_references::{
    ObjectReferencePackage, ReferenceView, ReferencesStatus, RejectedView,
    OBJECT_REFERENCES_SCHEMA_VERSION,
};
use crate::semantic_inventory::{
    SegmentationStatus, SemanticFrame, SemanticObservation, SemanticProvider, SemanticVocabulary,
    VocabularySource, SEMANTIC_INVENTORY_SCHEMA_VERSION,
};

const W: u32 = 40;
const H: u32 = 30;

fn rect(x0: u32, y0: u32, x1: u32, y1: u32) -> SemanticMask {
    let pixels: Vec<bool> = (0..H)
        .flat_map(|y| (0..W).map(move |x| x >= x0 && x < x1 && y >= y0 && y < y1))
        .collect();
    encode(W, H, &pixels)
}

fn observation(id: &str, frame: usize, mask: Option<SemanticMask>) -> SemanticObservation {
    SemanticObservation {
        id: id.into(),
        frame_index: frame,
        label: "toy".into(),
        bbox: [0.0, 0.0, f64::from(W), f64::from(H)],
        detection_confidence: 0.9,
        mask_confidence: mask.as_ref().map(|_| 0.9),
        mask,
    }
}

fn view(id: &str, frame: usize) -> ReferenceView {
    ReferenceView {
        observation: id.into(),
        frame_index: frame,
        frame_hash: ContentHash::of_bytes(format!("{frame}").as_bytes()),
        crop: [0, 0, W, H],
        bbox: [0.0, 0.0, f64::from(W), f64::from(H)],
        mask_hash: ContentHash::of_bytes(b"m"),
        mask_area_pixels: 1,
        detection_confidence: 0.9,
        mask_confidence: 0.9,
        consistency: MaskConsistency::Consistent,
    }
}

fn package(
    object: &str,
    primary: ReferenceView,
    rejected: Vec<RejectedView>,
) -> ObjectReferencePackage {
    ObjectReferencePackage {
        object: object.into(),
        label: "toy".into(),
        source_tracks: Vec::new(),
        description: "toy".into(),
        primary,
        additional: Vec::new(),
        rejected,
        observed_points: Vec::new(),
        extent: None,
    }
}

fn inputs() -> (SemanticInventoryArtifact, ObjectReferencesArtifact) {
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
        frames: (0..2)
            .map(|frame| SemanticFrame {
                frame_index: frame,
                width: W,
                height: H,
                content_hash: ContentHash::of_bytes(format!("{frame}").as_bytes()),
            })
            .collect(),
        observations: vec![
            observation("a0", 0, Some(rect(5, 5, 10, 10))),
            observation("b0", 0, Some(rect(20, 10, 25, 20))),
            observation("c0", 0, None),
            observation("a1", 1, Some(rect(6, 5, 11, 10))),
        ],
        tracks: Vec::new(),
        uncertainties: Vec::new(),
        measurements: None,
    };
    let references = ObjectReferencesArtifact {
        schema_version: OBJECT_REFERENCES_SCHEMA_VERSION,
        status: ReferencesStatus::Available,
        unavailable_reason: None,
        semantic_inventory: ContentHash::of_bytes(b"i"),
        object_identity: ContentHash::of_bytes(b"o"),
        object_masks: ContentHash::of_bytes(b"m"),
        packages: vec![
            package("obj-a", view("a0", 0), Vec::new()),
            package("obj-b", view("b0", 0), Vec::new()),
            package(
                "obj-c",
                view("a1", 1),
                vec![RejectedView {
                    observation: "c0".into(),
                    frame_index: 0,
                    reason: "box-only".into(),
                }],
            ),
        ],
        unpackaged: Vec::new(),
    };
    (inventory, references)
}

#[test]
fn every_representative_frame_removes_all_masked_packaged_objects_deterministically() {
    let (inventory, references) = inputs();
    let requests = plan_clean_plates(&inventory, &references, ("lama", "r1"));
    assert_eq!(requests.len(), 2);
    let first = &requests[0];
    assert_eq!(first.frame_index, 0);
    assert_eq!(first.objects, ["obj-a", "obj-b"]);
    assert_eq!(first.observations, ["a0", "b0"]);
    assert_eq!(
        first.unremoved,
        ["obj-c"],
        "a box-only object is not removed"
    );
    assert_eq!(first.source_frame_hash, inventory.frames[0].content_hash);
    assert_eq!(first.mask_hashes.len(), 2);
    // The union is dilated by MASK_DILATION_PIXELS around both rectangles.
    let pixels = decode(&first.removal_mask);
    let at = |x: u32, y: u32| pixels[(y * W + x) as usize];
    assert!(at(5, 5) && at(22, 15), "both objects are removed");
    assert!(
        at(1, 5) && at(13, 13),
        "dilated by {MASK_DILATION_PIXELS} px"
    );
    assert!(!at(0, 0) && !at(15, 5) && !at(39, 29));
    assert_eq!(
        plan_clean_plates(&inventory, &references, ("lama", "r1")),
        requests
    );
    let revised = plan_clean_plates(&inventory, &references, ("lama", "r2"));
    assert_ne!(
        revised[0].edit_id, first.edit_id,
        "the provider is part of the identity"
    );
}

#[test]
fn artifacts_need_an_output_or_a_failure_and_never_both_when_unavailable() {
    let (inventory, references) = inputs();
    let request = plan_clean_plates(&inventory, &references, ("lama", "r1")).remove(0);
    let provider = PlateProvider {
        id: "lama".into(),
        revision: "r1".into(),
        models: Vec::new(),
        runtime: None,
    };
    let mut artifact = CleanPlateArtifact {
        schema_version: CLEAN_PLATE_SCHEMA_VERSION,
        status: PlateStatus::Available,
        unavailable_reason: None,
        provider,
        object_references: ContentHash::of_bytes(b"r"),
        plates: vec![CleanPlate {
            request,
            classification: PlateClassification::GeneratedEdit,
            output: None,
            failure: Some("inpainting failed".into()),
        }],
    };
    artifact.validate().unwrap();
    assert_eq!(
        CleanPlateArtifact::from_json(&artifact.to_json()).unwrap(),
        artifact
    );
    artifact.plates[0].failure = None;
    assert!(artifact.validate().is_err());
    artifact.plates[0].output = Some(PlateOutput {
        path: ProjectPath::new("artifacts/native/plates/x.rgba").unwrap(),
        content_hash: ContentHash::of_bytes(b"x"),
    });
    artifact.validate().unwrap();
    artifact.status = PlateStatus::Unavailable;
    artifact.unavailable_reason = Some("no sidecar".into());
    assert!(
        artifact.validate().is_err(),
        "unavailable plates carry no outputs"
    );
}
