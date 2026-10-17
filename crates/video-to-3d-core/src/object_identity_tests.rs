use super::*;
use crate::semantic_inventory::{
    SemanticFrame, SemanticModel, SemanticModelRole, SemanticProvider, SemanticRuntime,
    SemanticTrack, SemanticVocabulary, VocabularySource, SEMANTIC_INVENTORY_SCHEMA_VERSION,
};

const W: u32 = 320;
const H: u32 = 240;
const FOCAL: f64 = 300.0;

/// Cameras translate along x, looking down +z (identity rotation).
fn camera(frame: usize) -> FrameCamera {
    FrameCamera {
        frame_index: frame,
        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation: [-(frame as f64) * 0.1, 0.0, 0.0],
        focal_pixels: FOCAL,
        principal_point: [f64::from(W) / 2.0, f64::from(H) / 2.0],
    }
}

/// A small cube of points centred at `centre`.
fn cube(centre: [f64; 3]) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..2 {
                points.push([
                    centre[0] - 0.1 + x as f64 * 0.066,
                    centre[1] - 0.1 + y as f64 * 0.066,
                    centre[2] + z as f64 * 0.05,
                ]);
            }
        }
    }
    points
}

/// A wall plane behind everything at z = 6.
fn wall() -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for x in -20..30 {
        for y in -15..15 {
            points.push([x as f64 * 0.1, y as f64 * 0.1, 6.0]);
        }
    }
    points
}

/// Box of the projection of `points` in `frame`, slightly padded.
fn bbox(points: &[[f64; 3]], frame: usize) -> [f64; 4] {
    let camera = camera(frame);
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for point in points {
        let local = camera.camera_point(*point);
        let x = FOCAL * local[0] / local[2] + f64::from(W) / 2.0;
        let y = FOCAL * local[1] / local[2] + f64::from(H) / 2.0;
        bounds = [
            bounds[0].min(x),
            bounds[1].min(y),
            bounds[2].max(x),
            bounds[3].max(y),
        ];
    }
    [
        bounds[0] - 1.0,
        bounds[1] - 1.0,
        bounds[2] + 1.0,
        bounds[3] + 1.0,
    ]
}

struct Scene {
    inventory: SemanticInventoryArtifact,
    points: Vec<[f64; 3]>,
}

impl Scene {
    fn new() -> Self {
        Self {
            inventory: SemanticInventoryArtifact {
                schema_version: SEMANTIC_INVENTORY_SCHEMA_VERSION,
                status: SegmentationStatus::Available,
                unavailable_reason: None,
                provider: SemanticProvider {
                    id: "test".into(),
                    revision: "1".into(),
                    models: vec![SemanticModel {
                        role: SemanticModelRole::Detector,
                        id: "test/detector".into(),
                        version: "1".into(),
                        url: "https://example.invalid/detector".into(),
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
                    terms: vec!["toy".into(), "wall".into()],
                },
                frames: (0..6)
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
            points: wall(),
        }
    }

    /// Observe `object` (points per frame) as `label` in `frames`; returns ids.
    fn observe(
        &mut self,
        label: &str,
        frames: &[usize],
        object: impl Fn(usize) -> Vec<[f64; 3]>,
    ) -> Vec<String> {
        let mut ids = Vec::new();
        for &frame in frames {
            let points = object(frame);
            let id = format!("obs-{}", self.inventory.observations.len());
            self.inventory.observations.push(SemanticObservation {
                id: id.clone(),
                frame_index: frame,
                label: label.into(),
                bbox: bbox(&points, frame),
                detection_confidence: 0.8,
                mask: None,
                mask_confidence: None,
            });
            ids.push(id);
        }
        ids
    }

    fn track(&mut self, label: &str, observations: Vec<String>) {
        let id = format!("track-{}", self.inventory.tracks.len());
        self.inventory.tracks.push(SemanticTrack {
            id,
            label: label.into(),
            observations,
            association_confidence: 0.9,
        });
    }

    fn identify(&self) -> ObjectIdentityArtifact {
        self.inventory.validate().unwrap();
        let cameras: Vec<FrameCamera> = (0..6).map(camera).collect();
        let artifact = identify_objects(
            &self.inventory,
            ContentHash::of_bytes(b"inventory"),
            &cameras,
            &self.points,
        );
        artifact.validate().unwrap();
        let reparsed = ObjectIdentityArtifact::from_json(&artifact.to_json()).unwrap();
        assert_eq!(reparsed, artifact);
        artifact
    }
}

fn object_with<'a>(artifact: &'a ObjectIdentityArtifact, observation: &str) -> &'a SceneObject {
    artifact
        .objects
        .iter()
        .find(|object| object.observations.iter().any(|id| id == observation))
        .unwrap()
}

#[test]
fn a_stationary_prop_keeps_one_identity_across_an_occlusion_gap() {
    let mut scene = Scene::new();
    let toy = cube([0.2, 0.1, 3.0]);
    scene.points.extend(toy.clone());
    // Tracked in frames 0-1, occluded in 2, tracked again in 3-5.
    let early = scene.observe("toy", &[0, 1], |_| toy.clone());
    let late = scene.observe("toy", &[3, 4, 5], |_| toy.clone());
    scene.track("toy", early.clone());
    scene.track("toy", late.clone());
    let artifact = scene.identify();
    let object = object_with(&artifact, &early[0]);
    assert_eq!(object.observations.len(), 5, "{artifact:#?}");
    assert_eq!(object.source_tracks.len(), 2);
    assert_eq!(
        object.motion,
        MotionClass::MovableStationary,
        "{}",
        object.motion_reason
    );
    assert_eq!(object.locations.len(), 5);
    assert!(
        artifact.ambiguities.is_empty(),
        "{:?}",
        artifact.ambiguities
    );
    // Identities are stable across runs.
    assert_eq!(scene.identify(), artifact);
}

#[test]
fn a_genuinely_moving_object_is_classified_moving_and_flagged() {
    let mut scene = Scene::new();
    let position = |frame: usize| [-0.4 + frame as f64 * 0.3, 0.0, 3.0];
    let all: Vec<[f64; 3]> = (0..6).flat_map(|frame| cube(position(frame))).collect();
    scene.points.extend(all);
    let observations = scene.observe("toy", &[0, 2, 4], |frame| cube(position(frame)));
    scene.track("toy", observations.clone());
    let artifact = scene.identify();
    let object = object_with(&artifact, &observations[0]);
    assert_eq!(
        object.motion,
        MotionClass::Moving,
        "{}",
        object.motion_reason
    );
    assert!(artifact
        .ambiguities
        .iter()
        .any(|ambiguity| ambiguity.kind == AmbiguityKind::Split
            && ambiguity.objects == [object.id.clone()]));
}

#[test]
fn structure_is_static_and_unlocalized_objects_are_unknown() {
    let mut scene = Scene::new();
    let wall_points: Vec<[f64; 3]> = scene
        .points
        .iter()
        .copied()
        .filter(|point| point[0] > 0.5 && point[0] < 1.0 && point[1].abs() < 0.3)
        .collect();
    let wall_ids = scene.observe("wall", &[0, 1], |_| wall_points.clone());
    scene.track("wall", wall_ids.clone());
    // A detection whose box (above the wall's image) holds no accepted geometry.
    let lonely = ["obs-lonely".to_owned()];
    scene.inventory.observations.push(SemanticObservation {
        id: lonely[0].clone(),
        frame_index: 2,
        label: "toy".into(),
        bbox: [10.0, 5.0, 40.0, 30.0],
        detection_confidence: 0.5,
        mask: None,
        mask_confidence: None,
    });
    let artifact = scene.identify();
    let wall = object_with(&artifact, &wall_ids[0]);
    assert_eq!(wall.category, ObjectCategory::Structure);
    assert_eq!(wall.motion, MotionClass::StaticStructure);
    let toy = object_with(&artifact, &lonely[0]);
    assert_eq!(toy.motion, MotionClass::Unknown);
    assert!(
        toy.motion_reason.contains("0 registered frame"),
        "{}",
        toy.motion_reason
    );
}

#[test]
fn overlapping_same_label_candidates_stay_separate_as_an_explicit_merge_ambiguity() {
    let mut scene = Scene::new();
    let toy = cube([0.2, 0.1, 3.0]);
    scene.points.extend(toy.clone());
    let first = scene.observe("toy", &[0, 1, 2], |_| toy.clone());
    let second = scene.observe("toy", &[2, 3], |_| toy.clone());
    scene.track("toy", first.clone());
    scene.track("toy", second.clone());
    let artifact = scene.identify();
    let a = object_with(&artifact, &first[0]);
    let b = object_with(&artifact, &second[0]);
    assert_ne!(a.id, b.id);
    assert!(artifact.ambiguities.iter().any(|ambiguity| {
        ambiguity.kind == AmbiguityKind::Merge
            && ambiguity.objects.contains(&a.id)
            && ambiguity.objects.contains(&b.id)
    }));
}

#[test]
fn unavailable_semantics_yield_an_unavailable_result() {
    let mut scene = Scene::new();
    scene.inventory.status = SegmentationStatus::Unavailable;
    scene.inventory.unavailable_reason = Some("no CUDA device".into());
    scene.inventory.provider.models.clear();
    scene.inventory.provider.runtime = None;
    let artifact = identify_objects(
        &scene.inventory,
        ContentHash::of_bytes(b"x"),
        &[camera(0)],
        &scene.points,
    );
    assert_eq!(artifact.status, IdentityStatus::Unavailable);
    assert!(artifact
        .unavailable_reason
        .unwrap()
        .contains("no CUDA device"));
    assert!(artifact.objects.is_empty());
}
