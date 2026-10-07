//! Deterministic two-reference seam fixture for the single-camera rule.
//!
//! Both cameras look down +Z at a fronto-parallel plane at depth 4; camera 1
//! sits one unit to the right. Reference 0 owns points 0..4, reference 1 owns
//! points 4..8, and every point lies exactly on the ray of its grid site.
//! Triangles 4 `(1, 4, 3)` and 5 `(4, 6, 3)` are mixed-reference seams
//! between the two patches.

use super::*;
use crate::textured_glb::encode_textured_glb;
use crate::{
    classic_reference_patch_regions, DenseReferencePatchStats, EvidenceCamera, EvidenceScale,
    MeshTriangle, Point3, ReconstructionProviderDescriptor,
};
use serde_json::Value;

const WIDTH: u32 = 32;
const HEIGHT: u32 = 24;
const FOCAL: f32 = 20.0;
const DEPTH: f32 = 4.0;
const SEAM_A: usize = 4;
const SEAM_B: usize = 5;

fn on_ray(camera_x: f32, site: DenseGridSite, depth: f32) -> Point3 {
    Point3 {
        x: (site.x as f32 - WIDTH as f32 * 0.5) / FOCAL * depth + camera_x,
        y: (site.y as f32 - HEIGHT as f32 * 0.5) / FOCAL * depth,
        z: depth,
        confidence: 0.9,
        r: 120,
        g: 80,
        b: 40,
    }
}

fn world(x: f32, y: f32, z: f32) -> Point3 {
    Point3 {
        x,
        y,
        z,
        confidence: 0.9,
        r: 0,
        g: 0,
        b: 0,
    }
}

fn camera(frame_index: usize, x: f32) -> EvidenceCamera {
    EvidenceCamera {
        frame_index,
        authority: EvidenceCameraAuthority::RegisteredGeometry,
        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation: [-x, 0.0, 0.0],
        confidence: Some(0.9),
        median_reprojection_error_pixels: Some(0.5),
    }
}

fn image_pixels(seed: u8) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            rgba.extend_from_slice(&[
                (x as u8 * 7).wrapping_add(seed),
                (y as u8 * 11).wrapping_add(seed),
                seed,
                255,
            ]);
        }
    }
    rgba
}

fn triangle(a: usize, b: usize, c: usize) -> MeshTriangle {
    MeshTriangle {
        a,
        b,
        c,
        confidence: 0.8,
    }
}

fn patch(
    reference_frame: usize,
    source: usize,
    start: usize,
    count: usize,
) -> DenseReferencePatchStats {
    DenseReferencePatchStats {
        reference_frame,
        source_frames: vec![source],
        primary_start: start,
        primary_points: count,
        completion_start: start + count,
        completed_points: 0,
        ..DenseReferencePatchStats::default()
    }
}

struct Seams {
    points: Vec<Point3>,
    sites: Vec<DenseGridSite>,
    triangles: Vec<MeshTriangle>,
    cameras: Vec<EvidenceCamera>,
    patches: Vec<DenseReferencePatchStats>,
    images: Vec<(usize, Vec<u8>)>,
    focal: Option<f32>,
}

impl Seams {
    fn new() -> Self {
        let site = |x, y| DenseGridSite { x, y };
        let reference_0 = [site(10, 8), site(14, 8), site(10, 14), site(14, 14)];
        let reference_1 = [site(14, 8), site(18, 8), site(14, 14), site(18, 14)];
        let mut points = reference_0
            .iter()
            .map(|site| on_ray(0.0, *site, DEPTH))
            .collect::<Vec<_>>();
        points.extend(reference_1.iter().map(|site| on_ray(1.0, *site, DEPTH)));
        Self {
            points,
            sites: reference_0.into_iter().chain(reference_1).collect(),
            triangles: vec![
                triangle(0, 1, 2),
                triangle(1, 3, 2),
                triangle(4, 5, 6),
                triangle(5, 7, 6),
                triangle(1, 4, 3),
                triangle(4, 6, 3),
            ],
            cameras: vec![camera(0, 0.0), camera(1, 1.0)],
            patches: vec![patch(0, 1, 0, 4), patch(1, 0, 4, 4)],
            images: vec![(0, image_pixels(5)), (1, image_pixels(90))],
            focal: Some(FOCAL),
        }
    }

    /// Add an accepted occluder triangle owned by reference 2, whose camera
    /// supplies no image (so it is never a texturing camera itself).
    fn with_occluder(mut self, corners: [Point3; 3]) -> Self {
        let start = self.points.len();
        self.points.extend(corners);
        self.sites.extend([
            DenseGridSite { x: 0, y: 0 },
            DenseGridSite { x: 1, y: 0 },
            DenseGridSite { x: 0, y: 1 },
        ]);
        self.triangles.push(triangle(start, start + 1, start + 2));
        self.cameras.push(camera(2, 0.5));
        self.patches.push(patch(2, 0, start, 3));
        self
    }

    fn only_image_0(mut self) -> Self {
        self.images.truncate(1);
        self
    }

    fn evidence(&self) -> ReconstructionEvidenceView<'_> {
        ReconstructionEvidenceView::new(
            ReconstructionProviderDescriptor::classic(),
            EvidenceScale::ArbitraryMonocular,
            self.cameras.clone(),
            classic_reference_patch_regions(&self.patches),
            &self.points,
            &self.triangles,
        )
        .expect("seam fixture evidence is valid")
    }

    fn bake(&self) -> SurfaceMaterialBake {
        let images = self
            .images
            .iter()
            .map(|(frame_index, rgba)| ReferenceImage {
                frame_index: *frame_index,
                width: WIDTH,
                height: HEIGHT,
                rgba,
            })
            .collect::<Vec<_>>();
        let projection = self
            .focal
            .map(|focal_pixels| SeamProjection { focal_pixels });
        bake_surface_materials_with_seams(&self.evidence(), &self.sites, &images, projection)
            .expect("seam bake succeeds")
    }
}

fn seam_material(bake: &SurfaceMaterialBake, frame: usize) -> &BakedReferenceMaterial {
    bake.materials
        .iter()
        .find(|material| {
            material.rule == MaterialRule::SeamSingleCameraProjection
                && material.reference_frame == frame
        })
        .expect("seam material exists")
}

#[test]
fn a_camera_seeing_all_three_vertices_textures_the_seam() {
    let fixture = Seams::new();
    let bake = fixture.bake();
    assert_eq!(bake, fixture.bake(), "the bake is deterministic");

    // Same-reference materials are unchanged; each seam goes to the passing
    // camera that owns most of its vertices.
    assert_eq!(bake.materials.len(), 4);
    assert_eq!(bake.materials[0].rule, MaterialRule::SameReferenceGrid);
    assert_eq!(bake.materials[0].triangles, vec![0, 1]);
    assert_eq!(bake.materials[1].triangles, vec![2, 3]);
    let seam_0 = seam_material(&bake, 0);
    let seam_1 = seam_material(&bake, 1);
    assert_eq!(seam_0.triangles, vec![SEAM_A]);
    assert_eq!(seam_1.triangles, vec![SEAM_B]);
    assert!(bake.fallback.triangles.is_empty());
    assert_eq!(bake.fallback.reasons, FallbackReasons::default());
    assert_eq!(bake.seam_triangles(), 2);
    assert_eq!(
        bake.textured_triangles() + bake.fallback.triangles.len(),
        fixture.triangles.len()
    );

    // Seam materials name their camera, rule, intrinsics and both references.
    assert_eq!(seam_0.seam_reference_frames, vec![0, 1]);
    assert_eq!(seam_0.focal_pixels, Some(FOCAL));
    assert_eq!(
        seam_0.camera_authority,
        EvidenceCameraAuthority::RegisteredGeometry
    );
    assert_eq!(seam_0.source_frames, vec![0, 1]);
    assert_eq!(seam_0.provenance, vec![EvidenceOrigin::GeometricMultiView]);
    assert_eq!(bake.materials[0].focal_pixels, None);
    assert!(bake.materials[0].seam_reference_frames.is_empty());

    // Corners (1, 4, 3) project to pixels (14, 8), (19, 8), (14, 14) of
    // camera 0; the crop keeps one texel of margin.
    let texture = &seam_0.texture;
    assert_eq!(texture.crop_origin, [13, 7]);
    assert_eq!((texture.width, texture.height), (8, 9));
    let expected = [[14.0, 8.0], [19.0, 8.0], [14.0, 14.0]];
    for (uv, [x, y]) in seam_0.corner_uvs[0].iter().zip(expected) {
        assert!((uv[0] - (x - 13.0 + 0.5) / 8.0).abs() < 1e-4, "{uv:?}");
        assert!((uv[1] - (y - 7.0 + 0.5) / 9.0).abs() < 1e-4, "{uv:?}");
    }
    let source = &fixture.images[0].1;
    for y in 0..texture.height {
        for x in 0..texture.width {
            let crop = ((y * texture.width + x) * 4) as usize;
            let full = (((y + 7) * WIDTH + x + 13) * 4) as usize;
            assert_eq!(texture.rgba[crop..crop + 4], source[full..full + 4]);
        }
    }

    let diagnostic = bake.diagnostic();
    assert!(diagnostic.contains("6 of 6 accepted triangles textured"));
    assert!(diagnostic.contains(
        "2 seam triangle(s) in 2 seam material(s) from camera frame(s) 0, 1 by the single-camera projection rule"
    ));
}

#[test]
fn a_seam_falls_back_to_another_camera_before_the_vertex_color_fallback() {
    // Without image 1, camera 0 is the only candidate and sees both seams.
    let bake = Seams::new().only_image_0().bake();
    assert_eq!(seam_material(&bake, 0).triangles, vec![SEAM_A, SEAM_B]);
    assert_eq!(bake.fallback.reasons.missing_reference_image, 2);
    assert_eq!(bake.fallback.reasons.mixed_reference, 0);
}

#[test]
fn occluded_seams_keep_the_fallback() {
    // An accepted triangle at depth 2 covers both images completely.
    let fixture = Seams::new().with_occluder([
        world(-6.0, -6.0, 2.0),
        world(6.0, -6.0, 2.0),
        world(0.0, 6.0, 2.0),
    ]);
    let bake = fixture.bake();
    assert!(bake
        .materials
        .iter()
        .all(|material| material.rule == MaterialRule::SameReferenceGrid));
    let reasons = &bake.fallback.reasons;
    assert_eq!(reasons.mixed_reference, 2);
    assert_eq!(reasons.seam.occluded, 2);
    assert_eq!(reasons.total(), 3, "occluder itself has no image");
    assert!(bake.fallback.triangles.contains(&SEAM_A));
    assert!(bake.diagnostic().contains("2 occluded"));
}

#[test]
fn seams_behind_or_outside_every_camera_keep_the_fallback() {
    // Point 6 behind camera 0 (the only camera with an image). Seam B is
    // behind the camera; the near-plane-clipped part of the folded accepted
    // triangles through point 6 now lies in front of seam A, so A is occluded.
    let mut behind = Seams::new().only_image_0();
    behind.points[6].z = -1.0;
    let bake = behind.bake();
    assert_eq!(bake.seam_triangles(), 0);
    assert_eq!(bake.fallback.reasons.seam.behind_camera, 1);
    assert_eq!(bake.fallback.reasons.seam.occluded, 1);
    assert_eq!(bake.fallback.reasons.mixed_reference, 2);

    // Point 6 on the ray of camera 1 pixel (30, 14) projects to u = 35 in
    // camera 0, outside its 32-pixel-wide image.
    let mut outside = Seams::new().only_image_0();
    outside.sites[6] = DenseGridSite { x: 30, y: 14 };
    outside.points[6] = on_ray(1.0, outside.sites[6], DEPTH);
    let bake = outside.bake();
    assert_eq!(seam_material(&bake, 0).triangles, vec![SEAM_A]);
    assert_eq!(bake.fallback.reasons.seam.out_of_frame, 1);
    assert_eq!(bake.fallback.triangles, vec![2, 3, SEAM_B]);
}

#[test]
fn ambiguous_visibility_is_rejected() {
    // Accepted geometry 3% in front of the plane: inside the tolerance band,
    // neither clearly the same surface nor clearly occluding.
    let band = Seams::new().only_image_0().with_occluder([
        world(-6.0, -6.0, DEPTH * 0.97),
        world(6.0, -6.0, DEPTH * 0.97),
        world(0.0, 6.0, DEPTH * 0.97),
    ]);
    let bake = band.bake();
    assert_eq!(bake.fallback.reasons.seam.ambiguous, 2);
    assert_eq!(bake.seam_triangles(), 0);

    // An occluder whose edge ends one pixel left of the seam vertices at
    // u = 14 in camera 0: the vertices are visible, but an occluding depth
    // lies in their pixel neighbourhood.
    let edge_x = (13.4 - WIDTH as f32 * 0.5) / FOCAL * 2.0;
    let edge = Seams::new().only_image_0().with_occluder([
        world(-3.0, -3.0, 2.0),
        world(edge_x, -3.0, 2.0),
        world(edge_x, 3.0, 2.0),
    ]);
    let bake = edge.bake();
    assert_eq!(bake.fallback.reasons.seam.ambiguous, 2);

    // Camera 1 sees the same seams far from that occluder edge and admits them.
    let mut both = edge;
    both.images.push((1, image_pixels(90)));
    let bake = both.bake();
    assert_eq!(bake.seam_triangles(), 2);
    assert_eq!(seam_material(&bake, 1).triangles, vec![SEAM_A, SEAM_B]);
}

#[test]
fn seams_need_intrinsics_that_reproduce_the_accepted_grid() {
    let mut no_intrinsics = Seams::new();
    no_intrinsics.focal = None;
    let bake = no_intrinsics.bake();
    assert_eq!(bake.fallback.reasons.seam.no_candidate_camera, 2);
    assert_eq!(bake.seam_triangles(), 0);

    // A focal length the accepted grid does not support.
    let mut wrong = Seams::new();
    wrong.focal = Some(FOCAL * 1.5);
    let bake = wrong.bake();
    assert_eq!(bake.fallback.reasons.seam.pose_inconsistent, 2);
    assert!(bake.diagnostic().contains("2 pose inconsistent"));

    let mut invalid = Seams::new();
    invalid.focal = Some(f32::NAN);
    let evidence = invalid.evidence();
    assert!(bake_surface_materials_with_seams(
        &evidence,
        &invalid.sites,
        &[],
        Some(SeamProjection {
            focal_pixels: f32::NAN
        })
    )
    .is_err());
}

#[test]
fn seam_inputs_invalidate_only_the_affected_seam_material() {
    let artifact = |reference_frame, rule| AppearanceArtifact {
        reference_frame,
        rule,
    };
    let reference = |frame| artifact(frame, MaterialRule::SameReferenceGrid);
    let seam = |frame| artifact(frame, MaterialRule::SeamSingleCameraProjection);
    let recorded = Seams::new().bake().recorded_appearance();
    assert_eq!(
        Seams::new().bake().invalidation_against(&recorded).reused,
        vec![reference(0), reference(1), seam(0), seam(1)]
    );

    // Moving a vertex of seam A only (point 1) changes its projection.
    let mut moved = Seams::new();
    moved.points[1].x -= 0.01;
    let diff = moved.bake().invalidation_against(&recorded);
    assert_eq!(diff.invalidated, vec![seam(0)]);
    assert_eq!(diff.reused, vec![reference(0), reference(1), seam(1)]);
    assert!(diff.diagnostic().contains("invalidated seam 0"));

    // The intrinsics are a seam input but not a same-reference input.
    let mut refocused = Seams::new();
    refocused.focal = Some(FOCAL + 0.01);
    let diff = refocused.bake().invalidation_against(&recorded);
    assert_eq!(diff.invalidated, vec![seam(0), seam(1)]);
    assert_eq!(diff.reused, vec![reference(0), reference(1)]);

    // Re-sampled pixels inside camera 1's seam crop (outside its reference
    // crop) invalidate only seam 1.
    let mut resampled = Seams::new();
    let pixel = ((10 * WIDTH + 12) * 4) as usize;
    resampled.images[1].1[pixel] ^= 0xff;
    let diff = resampled.bake().invalidation_against(&recorded);
    assert_eq!(diff.invalidated, vec![seam(1)]);

    // An occluder that rejects the seams removes their artifacts only.
    let occluded = Seams::new().with_occluder([
        world(-6.0, -6.0, 2.0),
        world(6.0, -6.0, 2.0),
        world(0.0, 6.0, 2.0),
    ]);
    let diff = occluded.bake().invalidation_against(&recorded);
    assert_eq!(diff.removed, vec![seam(0), seam(1)]);
    assert_eq!(diff.reused, vec![reference(0), reference(1)]);
}

#[test]
fn recorded_appearance_without_a_rule_is_a_same_reference_record() {
    let key = Seams::new().bake().materials[0].appearance_key.clone();
    let record: RecordedAppearance = serde_json::from_value(serde_json::json!({
        "reference_frame": 0,
        "appearance_key": key.as_str(),
    }))
    .unwrap();
    assert_eq!(record.rule, MaterialRule::SameReferenceGrid);
    let diff = Seams::new().bake().invalidation_against(&[record]);
    assert_eq!(diff.reused.len(), 1);
    assert_eq!(diff.added.len(), 3);
}

#[test]
fn seam_materials_export_their_camera_and_rule_without_new_geometry() {
    let fixture = Seams::new();
    let evidence = fixture.evidence();
    let bake = fixture.bake();
    let glb = encode_textured_glb(&evidence, &bake).unwrap();
    let json_length = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    let json: Value = serde_json::from_slice(&glb[20..20 + json_length]).unwrap();
    let extras = &json["asset"]["extras"]["video_to_3d"];
    assert_eq!(extras["textured_triangles"], 6);
    assert_eq!(extras["seam_textured_triangles"], 2);
    assert_eq!(extras["triangles"], fixture.triangles.len());
    assert_eq!(extras["points"], fixture.points.len());

    let materials = json["materials"].as_array().unwrap();
    assert_eq!(materials.len(), 4);
    assert_eq!(
        materials[0]["extras"]["video_to_3d"]["rule"],
        "same_reference_grid"
    );
    let seam = &materials[2];
    assert_eq!(seam["name"], "seam-camera-frame-0");
    let provenance = &seam["extras"]["video_to_3d"];
    assert_eq!(provenance["rule"], "seam_single_camera_projection");
    assert_eq!(provenance["reference_frame"], 0);
    assert_eq!(
        provenance["seam_reference_frames"],
        serde_json::json!([0, 1])
    );
    assert_eq!(provenance["focal_pixels"], FOCAL);
    assert_eq!(provenance["camera_authority"], "registered_geometry");
    assert_eq!(
        provenance["appearance_key"],
        seam_material(&bake, 0).appearance_key.as_str()
    );

    let exported: u64 = json["meshes"][0]["primitives"]
        .as_array()
        .unwrap()
        .iter()
        .map(|primitive| {
            json["accessors"][primitive["indices"].as_u64().unwrap() as usize]["count"]
                .as_u64()
                .unwrap()
                / 3
        })
        .sum();
    assert_eq!(exported as usize, fixture.triangles.len());
}
