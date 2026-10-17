use super::*;
use crate::object_identity::identify_objects;
use crate::object_masks::check_object_masks;
use crate::semantic_inventory::{
    MaskEncoding, SegmentationStatus, SemanticFrame, SemanticMask, SemanticModel,
    SemanticModelRole, SemanticProvider, SemanticRuntime, SemanticTrack, SemanticVocabulary,
    VocabularySource, SEMANTIC_INVENTORY_SCHEMA_VERSION,
};

const W: u32 = 320;
const H: u32 = 240;

fn camera(frame: usize) -> FrameCamera {
    FrameCamera {
        frame_index: frame,
        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation: [-(frame as f64) * 0.1, 0.0, 0.0],
        focal_pixels: 300.0,
        principal_point: [f64::from(W) / 2.0, f64::from(H) / 2.0],
    }
}

fn grid(x: (f64, f64), y: (f64, f64), z: f64) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    let mut px = x.0;
    while px <= x.1 + 1e-9 {
        let mut py = y.0;
        while py <= y.1 + 1e-9 {
            points.push([px, py, z]);
            py += 0.05;
        }
        px += 0.05;
    }
    points
}

fn rect(frame: usize, region: &[[f64; 3]]) -> ([f64; 4], SemanticMask) {
    let pixels: Vec<[f64; 2]> = region
        .iter()
        .map(|p| {
            crate::object_identity::project(&camera(frame), *p)
                .unwrap()
                .0
        })
        .collect();
    let x0 = pixels
        .iter()
        .map(|p| p[0])
        .fold(f64::INFINITY, f64::min)
        .floor() as u32;
    let y0 = pixels
        .iter()
        .map(|p| p[1])
        .fold(f64::INFINITY, f64::min)
        .floor() as u32;
    let x1 = pixels.iter().map(|p| p[0]).fold(0.0, f64::max).ceil() as u32 + 1;
    let y1 = pixels.iter().map(|p| p[1]).fold(0.0, f64::max).ceil() as u32 + 1;
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

struct Scene {
    inventory: SemanticInventoryArtifact,
    points: Vec<[f64; 3]>,
}

impl Scene {
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
            points,
        }
    }

    fn observe(
        &mut self,
        frame: usize,
        region: &[[f64; 3]],
        mask_confidence: Option<f64>,
    ) -> String {
        let (bbox, mask) = rect(frame, region);
        let id = format!("obs-{}", self.inventory.observations.len());
        self.inventory.observations.push(SemanticObservation {
            id: id.clone(),
            frame_index: frame,
            label: "toy".into(),
            bbox,
            detection_confidence: 0.8,
            mask: mask_confidence.map(|_| mask),
            mask_confidence,
        });
        id
    }

    fn track(&mut self, observations: Vec<String>) {
        self.inventory.tracks.push(SemanticTrack {
            id: format!("track-{}", self.inventory.tracks.len()),
            label: "toy".into(),
            observations,
            association_confidence: 0.9,
        });
    }

    fn package(&self) -> ObjectReferencesArtifact {
        self.inventory.validate().unwrap();
        let cameras: Vec<FrameCamera> = (0..6).map(camera).collect();
        let identities = identify_objects(
            &self.inventory,
            ContentHash::of_bytes(b"inventory"),
            &cameras,
            &self.points,
        );
        let masks = check_object_masks(
            &self.inventory,
            ContentHash::of_bytes(b"inventory"),
            &identities,
            ContentHash::of_bytes(b"identities"),
            &cameras,
            &self.points,
        );
        let artifact = package_object_references(
            &self.inventory,
            ContentHash::of_bytes(b"inventory"),
            &identities,
            ContentHash::of_bytes(b"identities"),
            &masks,
            ContentHash::of_bytes(b"masks"),
            &cameras,
            &self.points,
            EvidenceScale::ArbitraryMonocular,
        );
        artifact.validate().unwrap();
        assert_eq!(
            ObjectReferencesArtifact::from_json(&artifact.to_json()).unwrap(),
            artifact
        );
        artifact
    }
}

fn toy() -> Vec<[f64; 3]> {
    grid((0.0, 0.3), (0.0, 0.3), 3.0)
}

#[test]
fn a_tracked_object_gets_a_deterministic_multi_view_package_with_provenance() {
    let mut scene = Scene::new(toy());
    let ids: Vec<String> = (0..6)
        .map(|frame| scene.observe(frame, &toy(), Some(0.9)))
        .collect();
    scene.track(ids.clone());
    let artifact = scene.package();
    assert_eq!(artifact.packages.len(), 1, "{artifact:#?}");
    let package = &artifact.packages[0];
    assert_eq!(package.label, "toy");
    assert_eq!(package.source_tracks, ["track-0"]);
    let views: Vec<usize> = std::iter::once(&package.primary)
        .chain(&package.additional)
        .map(|view| view.frame_index)
        .collect();
    assert_eq!(views.len(), 1 + MAX_ADDITIONAL_VIEWS);
    assert_eq!(
        package.primary.frame_index, 0,
        "ties go to the earliest frame"
    );
    assert!(
        views.contains(&5),
        "additional views are spread out: {views:?}"
    );
    for view in std::iter::once(&package.primary).chain(&package.additional) {
        assert_eq!(
            view.frame_hash,
            scene.inventory.frames[view.frame_index].content_hash
        );
        assert_eq!(view.consistency, MaskConsistency::Consistent);
        assert!(view.crop[2] <= W && view.crop[3] <= H);
    }
    assert!(!package.observed_points.is_empty());
    let extent = package.extent.as_ref().unwrap();
    assert!((extent.size[0] - 0.3).abs() < 1e-9, "{extent:?}");
    assert_eq!(extent.scale, EvidenceScale::ArbitraryMonocular);
    assert_eq!(scene.package(), artifact, "deterministic");
}

#[test]
fn contradicting_weak_box_only_and_occluded_views_are_rejected() {
    let elsewhere = grid((-0.9, -0.6), (-0.3, 0.0), 3.0);
    let mut points = toy();
    points.extend(elsewhere.clone());
    let mut scene = Scene::new(points);
    let corner: Vec<[f64; 3]> = toy()
        .into_iter()
        .filter(|p| p[0] < 0.1 && p[1] < 0.1)
        .collect();
    let ids = vec![
        scene.observe(0, &toy(), Some(0.9)),
        scene.observe(1, &toy(), Some(0.9)),
        scene.observe(2, &elsewhere, Some(0.9)),
        scene.observe(3, &toy(), Some(0.2)),
        scene.observe(4, &toy(), None),
        scene.observe(5, &corner, Some(0.9)),
    ];
    // One track holds every view, as a tracker might have linked them.
    scene.track(ids.clone());
    let artifact = scene.package();
    let package = &artifact.packages[0];
    let used: BTreeSet<usize> = std::iter::once(&package.primary)
        .chain(&package.additional)
        .map(|view| view.frame_index)
        .collect();
    assert_eq!(used, BTreeSet::from([0, 1]), "{package:#?}");
    let reasons: BTreeMap<usize, &str> = package
        .rejected
        .iter()
        .map(|view| (view.frame_index, view.reason.as_str()))
        .collect();
    assert!(reasons[&2].contains("contradicts"), "{reasons:?}");
    assert!(reasons[&3].contains("weak"), "{reasons:?}");
    assert!(reasons[&4].contains("box-only"), "{reasons:?}");
    assert!(reasons[&5].contains("occluded"), "{reasons:?}");
}

#[test]
fn objects_without_an_acceptable_view_are_reported_and_structure_is_not_packaged() {
    let mut scene = Scene::new(toy());
    let only = scene.observe(0, &toy(), None);
    let second = scene.observe(1, &toy(), None);
    scene.track(vec![only.clone(), second]);
    let wall = grid((0.5, 1.0), (0.0, 0.3), 6.0);
    scene.points.extend(wall.clone());
    let (bbox, mask) = rect(0, &wall);
    scene.inventory.observations.push(SemanticObservation {
        id: "obs-wall".into(),
        frame_index: 0,
        label: "wall".into(),
        bbox,
        detection_confidence: 0.9,
        mask: Some(mask),
        mask_confidence: Some(0.9),
    });
    let artifact = scene.package();
    assert!(artifact.packages.is_empty());
    assert_eq!(artifact.unpackaged.len(), 1);
    assert_eq!(artifact.unpackaged[0].label, "toy");
    assert_eq!(artifact.unpackaged[0].rejected.len(), 2);
}

#[test]
fn unavailable_masks_yield_an_unavailable_package_set() {
    let masks = crate::object_masks::ObjectMasksArtifact {
        schema_version: crate::object_masks::OBJECT_MASKS_SCHEMA_VERSION,
        status: crate::object_masks::MasksStatus::Unavailable,
        unavailable_reason: Some("no semantic evidence: no CUDA device".into()),
        semantic_inventory: ContentHash::of_bytes(b"i"),
        object_identity: ContentHash::of_bytes(b"o"),
        masks: Vec::new(),
    };
    let mut scene = Scene::new(Vec::new());
    scene.inventory.status = SegmentationStatus::Unavailable;
    scene.inventory.unavailable_reason = Some("no CUDA device".into());
    scene.inventory.provider.models.clear();
    scene.inventory.provider.runtime = None;
    let identities = identify_objects(&scene.inventory, ContentHash::of_bytes(b"i"), &[], &[]);
    let artifact = package_object_references(
        &scene.inventory,
        ContentHash::of_bytes(b"i"),
        &identities,
        ContentHash::of_bytes(b"o"),
        &masks,
        ContentHash::of_bytes(b"m"),
        &[],
        &[],
        EvidenceScale::ArbitraryMonocular,
    );
    assert_eq!(artifact.status, ReferencesStatus::Unavailable);
    artifact.validate().unwrap();
}
