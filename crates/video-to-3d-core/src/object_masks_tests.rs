use super::*;
use crate::object_identity::{
    IdentityStatus, MotionClass, ObjectCategory, SceneObject, OBJECT_IDENTITY_SCHEMA_VERSION,
};
use crate::semantic_inventory::{
    MaskEncoding, SemanticFrame, SemanticMask, SemanticModel, SemanticModelRole,
    SemanticObservation, SemanticProvider, SemanticRuntime, SemanticVocabulary, VocabularySource,
    SEMANTIC_INVENTORY_SCHEMA_VERSION,
};

const W: u32 = 320;
const H: u32 = 240;
const FOCAL: f64 = 300.0;

fn camera(frame: usize) -> FrameCamera {
    FrameCamera {
        frame_index: frame,
        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation: [-(frame as f64) * 0.15, 0.0, 0.0],
        focal_pixels: FOCAL,
        principal_point: [f64::from(W) / 2.0, f64::from(H) / 2.0],
    }
}

fn grid(x: (f64, f64), y: (f64, f64), z: f64, step: f64) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    let mut px = x.0;
    while px <= x.1 + 1e-9 {
        let mut py = y.0;
        while py <= y.1 + 1e-9 {
            points.push([px, py, z]);
            py += step;
        }
        px += step;
    }
    points
}

fn pixel(frame: usize, point: [f64; 3]) -> [f64; 2] {
    project(&camera(frame), point).unwrap().0
}

/// A rectangular mask (and its box) around the projection of `points`.
fn mask_around(frame: usize, points: &[[f64; 3]]) -> ([f64; 4], SemanticMask) {
    let pixels: Vec<[f64; 2]> = points.iter().map(|p| pixel(frame, *p)).collect();
    let x0 = pixels
        .iter()
        .map(|p| p[0])
        .fold(f64::INFINITY, f64::min)
        .floor()
        .max(0.0) as u32;
    let y0 = pixels
        .iter()
        .map(|p| p[1])
        .fold(f64::INFINITY, f64::min)
        .floor()
        .max(0.0) as u32;
    let x1 = (pixels.iter().map(|p| p[0]).fold(0.0, f64::max).ceil() as u32 + 1).min(W);
    let y1 = (pixels.iter().map(|p| p[1]).fold(0.0, f64::max).ceil() as u32 + 1).min(H);
    let mut counts = Vec::new();
    let (mut current, mut run) = (false, 0_u64);
    for row in 0..H {
        for column in 0..W {
            let value = column >= x0 && column < x1 && row >= y0 && row < y1;
            if value != current {
                counts.push(run);
                (current, run) = (value, 0);
            }
            run += 1;
        }
    }
    counts.push(run);
    (
        [f64::from(x0), f64::from(y0), f64::from(x1), f64::from(y1)],
        SemanticMask {
            encoding: MaskEncoding::RleRowMajor,
            width: W,
            height: H,
            counts,
        },
    )
}

struct Case {
    inventory: SemanticInventoryArtifact,
    points: Vec<[f64; 3]>,
}

impl Case {
    fn new(points: Vec<[f64; 3]>) -> Self {
        Self {
            inventory: SemanticInventoryArtifact {
                schema_version: SEMANTIC_INVENTORY_SCHEMA_VERSION,
                status: SegmentationStatus::Available,
                unavailable_reason: None,
                provider: SemanticProvider {
                    id: "test".into(),
                    revision: "1".into(),
                    models: vec![SemanticModel {
                        role: SemanticModelRole::Segmenter,
                        id: "test/segmenter".into(),
                        version: "1".into(),
                        url: "https://example.invalid/segmenter".into(),
                    }],
                    runtime: Some(SemanticRuntime {
                        name: "test".into(),
                        version: "1".into(),
                        device: "cpu".into(),
                        device_name: None,
                    }),
                },
                vocabulary: SemanticVocabulary {
                    source: VocabularySource::Project,
                    terms: vec!["toy".into(), "pole".into()],
                },
                frames: (0..4)
                    .map(|frame| SemanticFrame {
                        frame_index: frame,
                        width: W,
                        height: H,
                        content_hash: ContentHash::of_bytes(format!("{frame}").as_bytes()),
                    })
                    .collect(),
                observations: Vec::new(),
                tracks: Vec::new(),
                uncertainties: Vec::new(),
                measurements: None,
            },
            points,
        }
    }

    fn observe(
        &mut self,
        label: &str,
        frame: usize,
        region: &[[f64; 3]],
        confidence: Option<f64>,
    ) -> String {
        let (bbox, mask) = mask_around(frame, region);
        let id = format!("obs-{}", self.inventory.observations.len());
        self.inventory.observations.push(SemanticObservation {
            id: id.clone(),
            frame_index: frame,
            label: label.into(),
            bbox,
            detection_confidence: 0.8,
            mask: confidence.map(|_| mask),
            mask_confidence: confidence,
        });
        id
    }

    fn check(&self, objects: &[(&str, Vec<String>)]) -> ObjectMasksArtifact {
        self.inventory.validate().unwrap();
        let identities = ObjectIdentityArtifact {
            schema_version: OBJECT_IDENTITY_SCHEMA_VERSION,
            status: IdentityStatus::Available,
            unavailable_reason: None,
            semantic_inventory: ContentHash::of_bytes(b"inventory"),
            objects: objects
                .iter()
                .enumerate()
                .map(|(index, (label, observations))| SceneObject {
                    id: format!("obj-{index}"),
                    label: (*label).into(),
                    category: ObjectCategory::Object,
                    observations: observations.clone(),
                    source_tracks: vec![format!("track-{index}")],
                    frames: Vec::new(),
                    locations: Vec::new(),
                    motion: MotionClass::Unknown,
                    motion_reason: "test".into(),
                    spread: None,
                })
                .collect(),
            ambiguities: Vec::new(),
        };
        let cameras: Vec<FrameCamera> = (0..4).map(camera).collect();
        let artifact = check_object_masks(
            &self.inventory,
            ContentHash::of_bytes(b"inventory"),
            &identities,
            ContentHash::of_bytes(b"identities"),
            &cameras,
            &self.points,
        );
        artifact.validate().unwrap();
        assert_eq!(
            ObjectMasksArtifact::from_json(&artifact.to_json()).unwrap(),
            artifact
        );
        artifact
    }
}

fn mask<'a>(artifact: &'a ObjectMasksArtifact, observation: &str) -> &'a ObjectMask {
    artifact
        .masks
        .iter()
        .find(|m| m.observation == observation)
        .unwrap()
}

#[test]
fn masks_that_agree_across_viewpoint_changes_are_consistent_and_keep_identity() {
    let toy = grid((0.0, 0.3), (0.0, 0.3), 3.0, 0.05);
    let mut case = Case::new(toy.clone());
    let ids: Vec<String> = (0..4)
        .map(|frame| case.observe("toy", frame, &toy, Some(0.9)))
        .collect();
    let artifact = case.check(&[("toy", ids.clone())]);
    for (frame, id) in ids.iter().enumerate() {
        let m = mask(&artifact, id);
        assert_eq!(m.consistency, MaskConsistency::Consistent, "{}", m.reason);
        assert_eq!(m.frame_index, frame);
        assert_eq!(m.frame_hash, case.inventory.frames[frame].content_hash);
        assert_eq!(m.object, "obj-0");
        assert_eq!(m.source_tracks, ["track-0"]);
        assert!(m.mask_hash.is_some());
        assert_eq!(m.quality, Some(MaskQuality::Confident));
    }
}

#[test]
fn a_mask_contradicting_the_other_views_is_diagnosed() {
    let toy = grid((0.0, 0.3), (0.0, 0.3), 3.0, 0.05);
    let elsewhere = grid((-0.9, -0.6), (-0.3, 0.0), 3.0, 0.05);
    let mut points = toy.clone();
    points.extend(elsewhere.clone());
    let mut case = Case::new(points);
    let mut ids: Vec<String> = (0..3)
        .map(|frame| case.observe("toy", frame, &toy, Some(0.9)))
        .collect();
    ids.push(case.observe("toy", 3, &elsewhere, Some(0.9)));
    let artifact = case.check(&[("toy", ids.clone())]);
    let wrong = mask(&artifact, &ids[3]);
    assert_eq!(
        wrong.consistency,
        MaskConsistency::Inconsistent,
        "{}",
        wrong.reason
    );
    assert!(wrong.support.unwrap() < MIN_MASK_SUPPORT);
}

#[test]
fn partial_occlusion_does_not_count_hidden_points_against_a_mask() {
    let toy = grid((0.0, 0.3), (0.0, 0.3), 3.0, 0.05);
    // An occluder hides all but the right edge of the toy in frame 3.
    let left: Vec<[f64; 3]> = toy.iter().copied().filter(|p| p[0] < 0.24).collect();
    let right: Vec<[f64; 3]> = toy.iter().copied().filter(|p| p[0] >= 0.24).collect();
    let camera3 = camera(3);
    let occluder: Vec<[f64; 3]> = left
        .iter()
        .map(|p| {
            // On the ray from camera 3 through p, at depth 1.5.
            let centre = [-camera3.translation[0], 0.0, 0.0];
            let t = 1.5 / p[2];
            [centre[0] + (p[0] - centre[0]) * t, p[1] * t, 1.5]
        })
        .collect();
    let mut points = toy.clone();
    points.extend(occluder);
    let mut case = Case::new(points);
    // The unoccluded reference view is frame 0 (the occluder lies outside it there).
    let ids = vec![
        case.observe("toy", 0, &toy, Some(0.9)),
        case.observe("toy", 3, &right, Some(0.9)),
    ];
    let artifact = case.check(&[("toy", ids.clone())]);
    let partial = mask(&artifact, &ids[1]);
    assert_eq!(
        partial.consistency,
        MaskConsistency::Consistent,
        "{}",
        partial.reason
    );
}

#[test]
fn thin_structures_weak_and_missing_masks_are_never_elevated() {
    let pole = grid((0.2, 0.2), (-0.4, 0.4), 3.0, 0.02);
    let mut case = Case::new(pole.clone());
    let a = case.observe("pole", 0, &pole, Some(0.9));
    let b = case.observe("pole", 1, &pole, Some(0.3));
    let c = case.observe("pole", 2, &pole, None);
    let artifact = case.check(&[("pole", vec![a.clone(), b.clone(), c.clone()])]);
    assert_eq!(
        mask(&artifact, &a).consistency,
        MaskConsistency::Consistent,
        "{}",
        mask(&artifact, &a).reason
    );
    let weak = mask(&artifact, &b);
    assert_eq!(weak.quality, Some(MaskQuality::Weak));
    let missing = mask(&artifact, &c);
    assert_eq!(missing.consistency, MaskConsistency::MissingMask);
    assert!(missing.mask_hash.is_none() && missing.quality.is_none());
}

#[test]
fn a_single_view_is_unverified_and_unavailable_semantics_propagate() {
    let toy = grid((0.0, 0.3), (0.0, 0.3), 3.0, 0.05);
    let mut case = Case::new(toy.clone());
    let only = case.observe("toy", 0, &toy, Some(0.9));
    let artifact = case.check(&[("toy", vec![only.clone()])]);
    assert_eq!(
        mask(&artifact, &only).consistency,
        MaskConsistency::Unverified
    );

    let mut unavailable = Case::new(Vec::new());
    unavailable.inventory.status = SegmentationStatus::Unavailable;
    unavailable.inventory.unavailable_reason = Some("no CUDA device".into());
    unavailable.inventory.provider.models.clear();
    unavailable.inventory.provider.runtime = None;
    let identities = ObjectIdentityArtifact {
        schema_version: OBJECT_IDENTITY_SCHEMA_VERSION,
        status: IdentityStatus::Unavailable,
        unavailable_reason: Some("no semantic evidence: no CUDA device".into()),
        semantic_inventory: ContentHash::of_bytes(b"x"),
        objects: Vec::new(),
        ambiguities: Vec::new(),
    };
    let artifact = check_object_masks(
        &unavailable.inventory,
        ContentHash::of_bytes(b"x"),
        &identities,
        ContentHash::of_bytes(b"y"),
        &[camera(0)],
        &[],
    );
    assert_eq!(artifact.status, MasksStatus::Unavailable);
    artifact.validate().unwrap();
}
