use crate::coarse_collision::{CoarseCollider, CoarseColliderOptions, ColliderBox};
use crate::geometry_confidence::{
    AgreementSource, ConfidenceBand, GeometryConfidenceField, RegionConfidence,
};
use crate::surface_materials::bake_surface_materials;
use crate::textured_glb::{encode_textured_glb, encode_textured_glb_with_collider};
use crate::{
    DenseGridSite, EvidenceCamera, EvidenceCameraAuthority, EvidenceOrigin,
    EvidencePointAttributes, EvidenceRange, EvidenceScale, MeshTriangle, Point3,
    ReconstructionEvidenceView, ReconstructionProviderClass, ReconstructionProviderDescriptor,
    SurfaceEvidenceRegion,
};
use serde_json::Value;

const DEPTH: f32 = 4.0;

fn camera(frame_index: usize, center_x: f32, error: Option<f32>) -> EvidenceCamera {
    EvidenceCamera {
        frame_index,
        authority: EvidenceCameraAuthority::RegisteredGeometry,
        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation: [-center_x, 0.0, 0.0],
        confidence: Some(0.9),
        median_reprojection_error_pixels: error,
    }
}

/// `n x n` points on `z = DEPTH + tilt_x * x + tilt_y * y` over `[-1, 1]^2`,
/// row-major, with two triangles per grid quad except where `keep` refuses.
struct Plane {
    points: Vec<Point3>,
    triangles: Vec<MeshTriangle>,
    sites: Vec<DenseGridSite>,
}

fn plane(n: usize, tilt: [f32; 2], confidence: f32, keep: impl Fn(f32, f32) -> bool) -> Plane {
    let coordinate = |i: usize| -1.0 + 2.0 * i as f32 / (n - 1) as f32;
    let mut points = Vec::with_capacity(n * n);
    let mut sites = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let (x, y) = (coordinate(i), coordinate(j));
            points.push(Point3 {
                x,
                y,
                z: DEPTH + tilt[0] * x + tilt[1] * y,
                confidence,
                r: (i * 5) as u8,
                g: (j * 5) as u8,
                b: 90,
            });
            sites.push(DenseGridSite {
                x: 4 + i as u32 * 2,
                y: 4 + j as u32 * 2,
            });
        }
    }
    let mut triangles = Vec::new();
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let corner = |di: usize, dj: usize| (j + dj) * n + i + di;
            let center = (
                (coordinate(i) + coordinate(i + 1)) * 0.5,
                (coordinate(j) + coordinate(j + 1)) * 0.5,
            );
            if !keep(center.0, center.1) {
                continue;
            }
            for (a, b, c) in [
                (corner(0, 0), corner(1, 0), corner(0, 1)),
                (corner(1, 0), corner(1, 1), corner(0, 1)),
            ] {
                triangles.push(MeshTriangle {
                    a,
                    b,
                    c,
                    confidence: confidence.min(0.9),
                });
            }
        }
    }
    Plane {
        points,
        triangles,
        sites,
    }
}

fn region(
    origin: EvidenceOrigin,
    reference: Option<usize>,
    sources: &[usize],
    range: EvidenceRange,
) -> SurfaceEvidenceRegion {
    SurfaceEvidenceRegion {
        origin,
        reference_frame: reference,
        source_frames: sources.to_vec(),
        points: range,
    }
}

fn provider(class: ReconstructionProviderClass) -> ReconstructionProviderDescriptor {
    match class {
        ReconstructionProviderClass::GeometricMultiView => {
            ReconstructionProviderDescriptor::classic()
        }
        _ => ReconstructionProviderDescriptor::new("test/provider", class, None).unwrap(),
    }
}

struct Scenario {
    origin: EvidenceOrigin,
    /// `(center_x, median reprojection error)` per source camera.
    sources: Vec<(f32, Option<f32>)>,
    reference_error: Option<f32>,
    point_confidence: f32,
    /// Per-point reciprocal/depth-margin attributes, uniform over the region.
    attributes: Option<EvidencePointAttributes>,
}

impl Scenario {
    fn strong() -> Self {
        Self {
            origin: EvidenceOrigin::GeometricMultiView,
            sources: vec![(0.6, Some(0.4)), (-0.6, Some(0.4)), (0.7, Some(0.4))],
            reference_error: Some(0.4),
            point_confidence: 0.9,
            attributes: None,
        }
    }

    fn strong_with_attributes() -> Self {
        Self {
            attributes: Some(EvidencePointAttributes {
                reciprocal_consistency: 0.9,
                depth_margin: 0.9,
            }),
            ..Self::strong()
        }
    }

    fn evaluate(&self) -> RegionConfidence {
        let plane = plane(9, [0.0, 0.0], self.point_confidence, |_, _| true);
        let class = match self.origin {
            EvidenceOrigin::GeometricMultiView | EvidenceOrigin::RevalidatedCompletion => {
                ReconstructionProviderClass::GeometricMultiView
            }
            EvidenceOrigin::LearnedMultiView => ReconstructionProviderClass::LearnedMultiView,
            EvidenceOrigin::GenerativeCompletion => {
                ReconstructionProviderClass::GenerativeCompletion
            }
        };
        let generative = self.origin == EvidenceOrigin::GenerativeCompletion;
        let mut cameras = Vec::new();
        if !generative {
            cameras.push(camera(0, 0.0, self.reference_error));
            for (index, (center, error)) in self.sources.iter().enumerate() {
                cameras.push(camera(index + 1, *center, *error));
            }
            if class != ReconstructionProviderClass::GeometricMultiView {
                for camera in &mut cameras {
                    camera.authority = EvidenceCameraAuthority::ProviderEstimated;
                }
            }
        }
        let sources: Vec<usize> = (1..=self.sources.len()).collect();
        let regions = vec![if generative {
            region(
                self.origin,
                None,
                &[],
                EvidenceRange::new(0, plane.points.len()),
            )
        } else {
            region(
                self.origin,
                Some(0),
                &sources,
                EvidenceRange::new(0, plane.points.len()),
            )
        }];
        let attributes = self
            .attributes
            .map(|attribute| vec![attribute; plane.points.len()]);
        let mut evidence = ReconstructionEvidenceView::new(
            provider(class),
            EvidenceScale::ArbitraryMonocular,
            cameras,
            regions,
            &plane.points,
            &plane.triangles,
        )
        .expect("valid scenario evidence");
        if let Some(attributes) = &attributes {
            evidence = evidence
                .with_point_attributes(attributes)
                .expect("valid scenario attributes");
        }
        GeometryConfidenceField::from_evidence(&evidence).regions[0].clone()
    }
}

#[test]
fn confidence_orders_each_evidence_factor() {
    let strong = Scenario::strong().evaluate();
    assert_eq!(strong.band, ConfidenceBand::High, "{strong:?}");
    assert_eq!(strong.factors.camera_support, 1.0);
    assert_eq!(strong.factors.triangulation, 1.0);
    assert_eq!(strong.factors.reference_exclusivity, 1.0);
    assert_eq!(strong.worst_reprojection_error_pixels, Some(0.4));

    let mut weaker = Vec::new();
    let mut one_source = Scenario::strong();
    one_source.sources.truncate(1);
    weaker.push(("camera support", one_source.evaluate()));
    let mut reprojection = Scenario::strong();
    reprojection.sources[1].1 = Some(3.0);
    weaker.push(("reprojection", reprojection.evaluate()));
    let mut narrow = Scenario::strong();
    for (center, _) in &mut narrow.sources {
        *center *= 0.15;
    }
    weaker.push(("triangulation", narrow.evaluate()));
    let mut disagreement = Scenario::strong();
    disagreement.point_confidence = 0.3;
    weaker.push(("reciprocal agreement", disagreement.evaluate()));
    let mut revalidated = Scenario::strong();
    revalidated.origin = EvidenceOrigin::RevalidatedCompletion;
    weaker.push(("revalidated provenance", revalidated.evaluate()));

    for (factor, region) in &weaker {
        assert!(
            region.confidence < strong.confidence && region.confidence > 0.0,
            "weaker {factor} must lower confidence: {} vs {}",
            region.confidence,
            strong.confidence
        );
    }
    // Revalidated completion stays below geometric evidence but above learned
    // evidence with identical camera geometry.
    let mut learned = Scenario::strong();
    learned.origin = EvidenceOrigin::LearnedMultiView;
    let learned = learned.evaluate();
    let revalidated = &weaker[4].1;
    assert!(learned.confidence < revalidated.confidence);
    assert!(learned.confidence > 0.0);

    let mut unknown_error = Scenario::strong();
    unknown_error.reference_error = None;
    let unknown_error = unknown_error.evaluate();
    assert_eq!(unknown_error.factors.reprojection, 0.5);
    assert!(unknown_error.confidence < strong.confidence);

    let mut generative = Scenario::strong();
    generative.origin = EvidenceOrigin::GenerativeCompletion;
    let generative = generative.evaluate();
    assert_eq!(generative.confidence, 0.0);
    assert_eq!(generative.band, ConfidenceBand::Unsupported);
}

#[test]
fn providers_without_point_attributes_keep_the_combined_factor() {
    let strong = Scenario::strong().evaluate();
    let factors = strong.factors;
    assert_eq!(
        factors.agreement_source,
        AgreementSource::CombinedPointConfidence
    );
    assert_eq!(factors.depth_ambiguity, None);
    assert_eq!(factors.reciprocal_agreement, 0.9);
    // The geometric mean still runs over the five factors of schema v1.
    let five = [
        factors.camera_support,
        factors.reprojection,
        factors.reciprocal_agreement,
        factors.triangulation,
        factors.reference_exclusivity,
    ];
    let expected = (five.iter().map(|f| f64::from(*f).ln()).sum::<f64>() / 5.0).exp() as f32;
    assert_eq!(strong.confidence.to_bits(), expected.to_bits());
}

#[test]
fn reciprocal_and_depth_ambiguity_order_confidence_on_their_own() {
    let strong = Scenario::strong_with_attributes().evaluate();
    assert_eq!(
        strong.factors.agreement_source,
        AgreementSource::PointAttributes
    );
    assert_eq!(strong.factors.reciprocal_agreement, 0.9);
    assert_eq!(strong.factors.depth_ambiguity, Some(0.9));
    assert_eq!(strong.band, ConfidenceBand::High, "{strong:?}");

    // The combined point confidence no longer stands in for either factor.
    let mut weak_points = Scenario::strong_with_attributes();
    weak_points.point_confidence = 0.2;
    let weak_points = weak_points.evaluate();
    assert_eq!(
        weak_points.confidence.to_bits(),
        strong.confidence.to_bits()
    );

    // Only reciprocal consistency drops: only the reciprocal factor follows.
    let mut previous = strong.clone();
    for reciprocal in [0.6, 0.35, 0.1] {
        let mut scenario = Scenario::strong_with_attributes();
        scenario.attributes.as_mut().unwrap().reciprocal_consistency = reciprocal;
        let region = scenario.evaluate();
        assert_eq!(region.factors.reciprocal_agreement, reciprocal);
        assert_eq!(region.factors.depth_ambiguity, Some(0.9));
        assert!(
            region.confidence < previous.confidence && region.confidence > 0.0,
            "reciprocal {reciprocal}: {} vs {}",
            region.confidence,
            previous.confidence
        );
        previous = region;
    }

    // Only the depth margin drops: only the depth-ambiguity factor follows.
    let mut previous = strong.clone();
    for margin in [0.6, 0.35, 0.1] {
        let mut scenario = Scenario::strong_with_attributes();
        scenario.attributes.as_mut().unwrap().depth_margin = margin;
        let region = scenario.evaluate();
        assert_eq!(region.factors.reciprocal_agreement, 0.9);
        assert_eq!(region.factors.depth_ambiguity, Some(margin));
        assert!(
            region.confidence < previous.confidence && region.confidence > 0.0,
            "depth margin {margin}: {} vs {}",
            region.confidence,
            previous.confidence
        );
        previous = region;
    }

    // Either factor at zero leaves the region unsupported.
    for attributes in [
        EvidencePointAttributes {
            reciprocal_consistency: 0.0,
            depth_margin: 0.9,
        },
        EvidencePointAttributes {
            reciprocal_consistency: 0.9,
            depth_margin: 0.0,
        },
    ] {
        let mut scenario = Scenario::strong_with_attributes();
        scenario.attributes = Some(attributes);
        assert_eq!(scenario.evaluate().band, ConfidenceBand::Unsupported);
    }
}

#[test]
fn split_factors_reach_the_summary_and_diagnostic() {
    let plane = plane(5, [0.0, 0.0], 0.8, |_, _| true);
    let attributes: Vec<EvidencePointAttributes> = (0..plane.points.len())
        .map(|index| EvidencePointAttributes {
            reciprocal_consistency: 0.75,
            depth_margin: if index % 2 == 0 { 0.4 } else { 0.8 },
        })
        .collect();
    let combined = single_reference_evidence(&plane);
    let combined_field = GeometryConfidenceField::from_evidence(&combined);
    let combined_summary = combined_field.summary(combined.triangles);
    assert_eq!(combined_summary.combined_agreement_regions, 1);
    assert_eq!(combined_summary.split_agreement_regions, 0);
    assert_eq!(combined_summary.depth_ambiguity, None);
    let diagnostic = combined_field.diagnostic(combined.triangles);
    assert!(
        diagnostic.contains("with depth ambiguity folded in"),
        "{diagnostic}"
    );

    let split = single_reference_evidence(&plane)
        .with_point_attributes(&attributes)
        .unwrap();
    let field = GeometryConfidenceField::from_evidence(&split);
    let summary = field.summary(split.triangles);
    assert_eq!(field.schema_version, 2);
    assert_eq!(summary.split_agreement_regions, 1);
    assert_eq!(summary.combined_agreement_regions, 0);
    assert_eq!(summary.reciprocal_agreement.unwrap().min, 0.75);
    // 13 of 25 points carry the 0.4 margin, so the lower median is 0.4.
    assert_eq!(summary.depth_ambiguity.unwrap().min, 0.4);
    let diagnostic = field.diagnostic(split.triangles);
    assert!(
        diagnostic.contains("reciprocal agreement 0.75-0.75 and depth ambiguity 0.40-0.40"),
        "{diagnostic}"
    );

    // A field from the combined evidence does not describe the split evidence, and a
    // changed attribute is a changed field, so a stale collider fails closed.
    assert!(!combined_field.describes(&split));
    let mut changed = attributes.clone();
    for attribute in &mut changed {
        attribute.reciprocal_consistency = 0.5;
    }
    let changed_evidence = single_reference_evidence(&plane)
        .with_point_attributes(&changed)
        .unwrap();
    assert!(!field.describes(&changed_evidence));
    assert!(field.describes(&split));
}

/// Points `0..split` belong to reference 0 and the rest to reference 1; the
/// triangles that straddle the split are reference-ambiguous.
fn two_reference_evidence<'a>(
    plane: &'a Plane,
    split: usize,
    reversed: bool,
) -> ReconstructionEvidenceView<'a> {
    let mut regions = vec![
        region(
            EvidenceOrigin::GeometricMultiView,
            Some(0),
            &[1, 2],
            EvidenceRange::new(0, split),
        ),
        region(
            EvidenceOrigin::GeometricMultiView,
            Some(1),
            &[0, 2],
            EvidenceRange::new(split, plane.points.len() - split),
        ),
    ];
    if reversed {
        regions.reverse();
    }
    ReconstructionEvidenceView::new(
        ReconstructionProviderDescriptor::classic(),
        EvidenceScale::ArbitraryMonocular,
        vec![
            camera(0, 0.0, Some(0.4)),
            camera(1, 0.6, Some(0.4)),
            camera(2, -0.6, Some(0.4)),
        ],
        regions,
        &plane.points,
        &plane.triangles,
    )
    .expect("valid two-reference evidence")
}

#[test]
fn reference_ambiguity_lowers_confidence_and_queries_resolve_by_region() {
    let n = 9;
    // Split mid-row so triangles straddle the reference boundary.
    let split = 4 * n + 4;
    let plane_with_seam = plane(n, [0.0, 0.0], 0.9, |_, _| true);
    let evidence = two_reference_evidence(&plane_with_seam, split, false);
    let field = GeometryConfidenceField::from_evidence(&evidence);
    let mixed: Vec<&MeshTriangle> = plane_with_seam
        .triangles
        .iter()
        .filter(|t| {
            [t.a, t.b, t.c].iter().any(|p| *p < split)
                && [t.a, t.b, t.c].iter().any(|p| *p >= split)
        })
        .collect();
    assert!(!mixed.is_empty());
    assert_eq!(field.regions[0].mixed_reference_triangles, mixed.len());
    assert!(field.regions[0].factors.reference_exclusivity < 1.0);

    // The same evidence without the ambiguous triangles is more confident.
    let mut separated = plane(n, [0.0, 0.0], 0.9, |_, _| true);
    separated.triangles.retain(|t| {
        let below = [t.a, t.b, t.c].iter().filter(|p| **p < split).count();
        below == 0 || below == 3
    });
    let separated_evidence = two_reference_evidence(&separated, split, false);
    let separated_field = GeometryConfidenceField::from_evidence(&separated_evidence);
    for index in 0..2 {
        assert_eq!(
            separated_field.regions[index].factors.reference_exclusivity,
            1.0
        );
        assert!(field.regions[index].confidence < separated_field.regions[index].confidence);
    }

    // Region queries.
    assert_eq!(field.region_of_point(split - 1).unwrap().region, 0);
    assert_eq!(field.region_of_point(split).unwrap().region, 1);
    assert!(field
        .region_of_point(plane_with_seam.points.len())
        .is_none());
    assert_eq!(field.point_confidence(plane_with_seam.points.len()), 0.0);
    let triangle = mixed[0];
    let expected = [triangle.a, triangle.b, triangle.c]
        .iter()
        .map(|p| field.point_confidence(*p))
        .fold(1.0, f32::min);
    assert_eq!(field.triangle_confidence(triangle), expected);
    let ceiling = field.regions[0].confidence.max(field.regions[1].confidence);
    let below: Vec<usize> = field.regions_below(ceiling).map(|r| r.region).collect();
    assert_eq!(below.len(), 1);
    assert!(field.regions_below(0.0).next().is_none());
    assert_eq!(
        field.summary(&plane_with_seam.triangles).points.high
            + field.summary(&plane_with_seam.triangles).points.medium
            + field.summary(&plane_with_seam.triangles).points.low,
        plane_with_seam.points.len()
    );
    assert!(field
        .diagnostic(&plane_with_seam.triangles)
        .starts_with("Geometry confidence v2: 2 regions"));
}

#[test]
fn confidence_is_stable_for_identical_accepted_evidence() {
    let n = 9;
    let split = 4 * n + 4;
    let plane_a = plane(n, [0.1, 0.0], 0.8, |_, _| true);
    let first =
        GeometryConfidenceField::from_evidence(&two_reference_evidence(&plane_a, split, false));
    let again =
        GeometryConfidenceField::from_evidence(&two_reference_evidence(&plane_a, split, false));
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&again).unwrap()
    );

    // Appearance is not geometric evidence.
    let mut recolored = plane(n, [0.1, 0.0], 0.8, |_, _| true);
    for point in &mut recolored.points {
        (point.r, point.g, point.b) = (1, 2, 3);
    }
    let recolored =
        GeometryConfidenceField::from_evidence(&two_reference_evidence(&recolored, split, false));
    assert_eq!(first, recolored);

    // Region order is not evidence either: each region keeps its confidence.
    let reversed =
        GeometryConfidenceField::from_evidence(&two_reference_evidence(&plane_a, split, true));
    for region in &first.regions {
        let twin = reversed
            .regions
            .iter()
            .find(|other| other.points == region.points)
            .unwrap();
        assert_eq!(twin.confidence.to_bits(), region.confidence.to_bits());
        assert_eq!(twin.factors, region.factors);
    }
}

fn single_reference_evidence(plane: &Plane) -> ReconstructionEvidenceView<'_> {
    ReconstructionEvidenceView::new(
        ReconstructionProviderDescriptor::classic(),
        EvidenceScale::ArbitraryMonocular,
        vec![
            camera(0, 0.0, Some(0.4)),
            camera(1, 0.6, Some(0.4)),
            camera(2, -0.6, Some(0.4)),
        ],
        vec![region(
            EvidenceOrigin::GeometricMultiView,
            Some(0),
            &[1, 2],
            EvidenceRange::new(0, plane.points.len()),
        )],
        &plane.points,
        &plane.triangles,
    )
    .expect("valid plane evidence")
}

fn inside_any(boxes: &[ColliderBox], point: [f32; 3]) -> bool {
    boxes
        .iter()
        .any(|collider_box| collider_box.contains(point))
}

fn assert_contains_triangles(collider: &CoarseCollider, plane: &Plane) {
    for triangle in &plane.triangles {
        let corners = [triangle.a, triangle.b, triangle.c].map(|index| {
            let p = &plane.points[index];
            [p.x as f64, p.y as f64, p.z as f64]
        });
        for i in 0..=4 {
            for j in 0..=(4 - i) {
                let (u, v) = (i as f64 / 4.0, j as f64 / 4.0);
                let w = 1.0 - u - v;
                let sample = [0, 1, 2].map(|axis| {
                    (corners[0][axis] * w + corners[1][axis] * u + corners[2][axis] * v) as f32
                });
                assert!(
                    inside_any(&collider.boxes, sample),
                    "accepted surface sample {sample:?} escapes the collider"
                );
            }
        }
    }
}

#[test]
fn collider_contains_accepted_surface_with_fewer_primitives() {
    for tilt in [[0.0, 0.0], [0.3, 0.2]] {
        let plane = plane(41, tilt, 0.9, |_, _| true);
        let evidence = single_reference_evidence(&plane);
        let field = GeometryConfidenceField::from_evidence(&evidence);
        let collider =
            CoarseCollider::from_evidence(&evidence, &field, CoarseColliderOptions::default());

        assert_eq!(collider.role, "collision");
        assert_eq!(collider.source_triangles, plane.triangles.len());
        assert_eq!(collider.box_count, collider.boxes.len());
        assert!(
            collider.boxes.len() * 10 < plane.triangles.len(),
            "collider must be lower detail: {} boxes for {} triangles",
            collider.boxes.len(),
            plane.triangles.len()
        );
        let cell = collider.cell_size.unwrap();
        // Twice the median edge: 0.05 in the image plane, longer when tilted.
        assert!((0.1 - 1e-5..0.11).contains(&cell), "cell {cell}");
        for collider_box in &collider.boxes {
            for axis in 0..3 {
                assert!(collider_box.min[axis].is_finite() && collider_box.max[axis].is_finite());
                assert!(collider_box.min[axis] < collider_box.max[axis]);
            }
            // Every box touches accepted geometry within one cell.
            assert!(plane.points.iter().any(|p| (0..3).all(|axis| {
                let value = [p.x, p.y, p.z][axis];
                collider_box.min[axis] - cell <= value && value <= collider_box.max[axis] + cell
            })));
        }
        assert_contains_triangles(&collider, &plane);
        assert!(collider
            .diagnostic()
            .contains("unsupported holes stay open"));

        let again =
            CoarseCollider::from_evidence(&evidence, &field, CoarseColliderOptions::default());
        assert_eq!(collider, again);
        assert_eq!(collider.boxes, again.boxes);
    }
    // A flat plane collapses into one slab.
    let flat = plane(41, [0.0, 0.0], 0.9, |_, _| true);
    let evidence = single_reference_evidence(&flat);
    let field = GeometryConfidenceField::from_evidence(&evidence);
    let collider =
        CoarseCollider::from_evidence(&evidence, &field, CoarseColliderOptions::default());
    assert_eq!(collider.boxes.len(), 1);
}

#[test]
fn unsupported_holes_stay_open_in_the_collider() {
    let hole = |x: f32, y: f32| x.abs() > 0.3 || y.abs() > 0.3;
    let plane = plane(41, [0.0, 0.0], 0.9, hole);
    let evidence = single_reference_evidence(&plane);
    let field = GeometryConfidenceField::from_evidence(&evidence);
    let collider =
        CoarseCollider::from_evidence(&evidence, &field, CoarseColliderOptions::default());

    assert_contains_triangles(&collider, &plane);
    for x in [-0.15f32, 0.0, 0.15] {
        for y in [-0.15f32, 0.0, 0.15] {
            assert!(
                !inside_any(&collider.boxes, [x, y, DEPTH + 0.01]),
                "unsupported hole at ({x}, {y}) was filled by the collider"
            );
        }
    }
    // The hole's points still carry their region confidence; the field adds
    // nothing outside the accepted points.
    assert_eq!(field.regions.len(), 1);
    assert_eq!(field.point_confidence(plane.points.len()), 0.0);
}

#[test]
fn collider_excludes_unobserved_and_low_confidence_evidence() {
    let plane = plane(9, [0.0, 0.0], 0.9, |_, _| true);
    let mut cameras = vec![camera(0, 0.0, None), camera(1, 0.6, None)];
    for camera in &mut cameras {
        camera.authority = EvidenceCameraAuthority::ProviderEstimated;
    }
    let learned = ReconstructionEvidenceView::new(
        provider(ReconstructionProviderClass::LearnedMultiView),
        EvidenceScale::ProviderLocal,
        cameras,
        vec![region(
            EvidenceOrigin::LearnedMultiView,
            Some(0),
            &[1],
            EvidenceRange::new(0, plane.points.len()),
        )],
        &plane.points,
        &plane.triangles,
    )
    .unwrap();
    let field = GeometryConfidenceField::from_evidence(&learned);
    let collider =
        CoarseCollider::from_evidence(&learned, &field, CoarseColliderOptions::default());
    assert!(collider.boxes.is_empty());
    assert_eq!(collider.cell_size, None);
    assert_eq!(
        collider.excluded.unobserved_provenance,
        plane.triangles.len()
    );
    assert!(collider.diagnostic().starts_with("Coarse collider: none."));

    let observed = single_reference_evidence(&plane);
    let field = GeometryConfidenceField::from_evidence(&observed);
    let strict = CoarseColliderOptions {
        min_confidence: 0.99,
        ..CoarseColliderOptions::default()
    };
    let collider = CoarseCollider::from_evidence(&observed, &field, strict);
    assert!(collider.boxes.is_empty());
    assert_eq!(collider.excluded.low_confidence, plane.triangles.len());

    // A field built from geometric evidence never admits the learned evidence it is paired
    // with: the mismatch excludes everything rather than trusting the stale origins.
    let stale = GeometryConfidenceField::from_evidence(&observed);
    assert!(!stale.describes(&learned));
    let collider =
        CoarseCollider::from_evidence(&learned, &stale, CoarseColliderOptions::default());
    assert!(collider.boxes.is_empty());
    assert_eq!(collider.source_triangles, 0);
    assert_eq!(
        collider.excluded.mismatched_confidence,
        plane.triangles.len()
    );
    assert!(field.describes(&observed));

    // A locally weak triangle stays out even when its region scores well.
    let mut local = plane.triangles.clone();
    local[0].confidence = 0.0;
    let locally_weak = ReconstructionEvidenceView::new(
        observed.provider.clone(),
        observed.scale,
        observed.cameras.clone(),
        observed.regions.clone(),
        &plane.points,
        &local,
    )
    .unwrap();
    let local_field = GeometryConfidenceField::from_evidence(&locally_weak);
    let collider = CoarseCollider::from_evidence(
        &locally_weak,
        &local_field,
        CoarseColliderOptions::default(),
    );
    assert_eq!(collider.excluded.low_confidence, 1);
    assert_eq!(collider.source_triangles, plane.triangles.len() - 1);

    // Same regions and ranges, weaker points: the field no longer describes the evidence.
    let mut weaker = plane.points.clone();
    for point in &mut weaker {
        point.confidence *= 0.1;
    }
    let weakened = ReconstructionEvidenceView::new(
        observed.provider.clone(),
        observed.scale,
        observed.cameras.clone(),
        observed.regions.clone(),
        &weaker,
        &plane.triangles,
    )
    .unwrap();
    assert!(!field.describes(&weakened));
    let collider =
        CoarseCollider::from_evidence(&weakened, &field, CoarseColliderOptions::default());
    assert_eq!(
        collider.excluded.mismatched_confidence,
        plane.triangles.len()
    );
}

fn glb_json(glb: &[u8]) -> (Value, Vec<u8>) {
    let json_length = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    let json = serde_json::from_slice(&glb[20..20 + json_length]).unwrap();
    let bin = glb[20 + json_length + 8..].to_vec();
    (json, bin)
}

#[test]
fn glb_carries_the_collider_as_a_separate_collision_scene() {
    let plane = plane(17, [0.2, 0.0], 0.9, |x, y| x.abs() > 0.4 || y.abs() > 0.4);
    let evidence = single_reference_evidence(&plane);
    let field = GeometryConfidenceField::from_evidence(&evidence);
    let collider =
        CoarseCollider::from_evidence(&evidence, &field, CoarseColliderOptions::default());
    assert!(!collider.boxes.is_empty());
    let bake = bake_surface_materials(&evidence, &plane.sites, &[]).unwrap();

    // A collider built from other evidence is refused rather than paired with this surface.
    let other_plane = self::plane(9, [0.0, 0.0], 0.9, |_, _| true);
    let other = single_reference_evidence(&other_plane);
    let other_field = GeometryConfidenceField::from_evidence(&other);
    let foreign =
        CoarseCollider::from_evidence(&other, &other_field, CoarseColliderOptions::default());
    assert!(encode_textured_glb_with_collider(&evidence, &bake, Some(&foreign)).is_err());

    let visual_only = encode_textured_glb(&evidence, &bake).unwrap();
    let with_collider =
        encode_textured_glb_with_collider(&evidence, &bake, Some(&collider)).unwrap();
    let (visual, visual_bin) = glb_json(&visual_only);
    let (json, bin) = glb_json(&with_collider);

    // The visual surface is untouched: same mesh, same leading binary data.
    assert_eq!(json["meshes"][0], visual["meshes"][0]);
    assert_eq!(json["materials"], visual["materials"]);
    assert_eq!(&bin[..visual_bin.len()], &visual_bin[..]);
    assert_eq!(json["scene"], 0);
    assert_eq!(json["scenes"][0]["name"], "visual");
    assert_eq!(json["scenes"][0]["nodes"], serde_json::json!([0]));

    assert_eq!(json["scenes"][1]["name"], "collision");
    let node = &json["nodes"][json["scenes"][1]["nodes"][0].as_u64().unwrap() as usize];
    assert_eq!(node["name"], "coarse-collider");
    assert_eq!(node["extras"]["video_to_3d"]["role"], "collision");
    let mesh = &json["meshes"][node["mesh"].as_u64().unwrap() as usize];
    assert!(mesh["primitives"][0].get("material").is_none());
    let indices = &json["accessors"][mesh["primitives"][0]["indices"].as_u64().unwrap() as usize];
    assert_eq!(
        indices["count"].as_u64().unwrap() as usize,
        collider.boxes.len() * 36
    );
    let positions = &json["accessors"][mesh["primitives"][0]["attributes"]["POSITION"]
        .as_u64()
        .unwrap() as usize];
    let low = [0, 1, 2].map(|axis| {
        collider
            .boxes
            .iter()
            .map(|b| b.min[axis])
            .fold(f32::INFINITY, f32::min)
    });
    let high = [0, 1, 2].map(|axis| {
        collider
            .boxes
            .iter()
            .map(|b| b.max[axis])
            .fold(f32::NEG_INFINITY, f32::max)
    });
    assert_eq!(
        positions["min"],
        serde_json::json!([low[0], -high[1], -high[2]])
    );
    assert_eq!(
        positions["max"],
        serde_json::json!([high[0], -low[1], -low[2]])
    );
    let extras = &json["asset"]["extras"]["video_to_3d"]["collision"];
    assert_eq!(extras["boxes"], collider.box_count);
    assert_eq!(
        visual["asset"]["extras"]["video_to_3d"]["collision"],
        Value::Null
    );
}
