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

    // A focal length the accepted grid does not support: at twice the focal
    // length every residual equals the site's distance from the principal
    // point, so the true medians are about 5.39 px (camera 0) and 3.65 px
    // (camera 1), both above the 2 px bound.
    let mut wrong = Seams::new();
    wrong.focal = Some(FOCAL * 2.0);
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

/// Shift the accepted grid sites of camera 0's points 2 and 3 by `pixels`
/// along x, so camera 0's own residuals are `[0, 0, pixels, pixels]`.
fn with_camera_0_residuals(pixels: u32) -> Seams {
    let mut fixture = Seams::new().only_image_0();
    for index in [2, 3] {
        fixture.sites[index].x += pixels;
    }
    fixture
}

#[test]
fn an_even_residual_count_uses_the_true_median() {
    // Camera 0 owns four points. Residuals [0, 0, 3, 3] have median 1.5 px,
    // within the 2 px bound, although the upper-middle residual is 3 px.
    let bake = with_camera_0_residuals(3).bake();
    assert_eq!(
        bake.fallback.reasons.seam.pose_inconsistent, 0,
        "median 1.5 px must not reject camera 0 as pose inconsistent"
    );
    assert_eq!(seam_material(&bake, 0).triangles, vec![SEAM_A, SEAM_B]);

    // Residuals [0, 0, 5, 5] have median 2.5 px: still rejected, even though
    // the lower-middle residual is 0 px.
    let bake = with_camera_0_residuals(5).bake();
    assert_eq!(bake.fallback.reasons.seam.pose_inconsistent, 2);
    assert_eq!(bake.seam_triangles(), 0);
}

#[test]
fn a_subpixel_occluder_in_front_of_a_seam_sample_is_not_visible() {
    // A 0.2 px accepted triangle at depth 2 straddles the ray of seam A's
    // edge midpoint (16.5, 8) in camera 0, between pixel centers: its
    // footprint contains no integer pixel center at all.
    let at = |u: f32, v: f32| {
        let depth = 2.0;
        world(
            (u - WIDTH as f32 * 0.5) / FOCAL * depth,
            (v - HEIGHT as f32 * 0.5) / FOCAL * depth,
            depth,
        )
    };
    let occluder = [at(16.4, 7.9), at(16.6, 7.9), at(16.5, 8.1)];

    // Camera 0 alone: seam A is hidden at one of its sample rays and must
    // not be textured; seam B is unaffected.
    let bake = Seams::new().only_image_0().with_occluder(occluder).bake();
    let seam = &bake.fallback.reasons.seam;
    assert_eq!(
        seam.occluded + seam.ambiguous,
        1,
        "seam A must be occluded or ambiguous in camera 0: {seam:?}"
    );
    assert_eq!(seam_material(&bake, 0).triangles, vec![SEAM_B]);
    assert!(bake.fallback.triangles.contains(&SEAM_A));

    // With both images, camera 1 sees seam A far from the occluder and may
    // admit it; camera 0 (owning more of its vertices) must not.
    let mut both = Seams::new().with_occluder(occluder);
    both.images.truncate(2);
    let bake = both.bake();
    let admitting = bake
        .materials
        .iter()
        .filter(|material| material.rule == MaterialRule::SeamSingleCameraProjection)
        .filter(|material| material.triangles.contains(&SEAM_A))
        .map(|material| material.reference_frame)
        .collect::<Vec<_>>();
    assert_eq!(
        admitting,
        vec![1],
        "only the unoccluded camera 1 may texture seam A"
    );
}

/// Point on the plane `z = DEPTH + slope * x` along the ray of camera-`camera_x`
/// pixel `(u, v)`: `t (1 - slope s) = DEPTH + slope camera_x`.
fn on_slanted_ray(camera_x: f32, u: f32, v: f32, slope: f32) -> Point3 {
    let s = (u - WIDTH as f32 * 0.5) / FOCAL;
    let r = (v - HEIGHT as f32 * 0.5) / FOCAL;
    let t = (DEPTH + slope * camera_x) / (1.0 - slope * s);
    Point3 {
        x: camera_x + s * t,
        y: r * t,
        z: t,
        confidence: 0.9,
        r: 120,
        g: 80,
        b: 40,
    }
}

/// The seam fixture on a plane slanted about the vertical axis: every point
/// stays on the ray of its own grid site (so both poses reproduce their grids
/// exactly), and all eight triangles are coplanar. With the fixture's short
/// focal length the plane's depth changes by several percent per pixel,
/// although it is far from grazing.
fn slanted(slope: f32) -> Seams {
    let mut fixture = Seams::new();
    for (index, site) in fixture.sites.iter().enumerate().take(8) {
        let camera_x = if index < 4 { 0.0 } else { 1.0 };
        fixture.points[index] = on_slanted_ray(camera_x, site.x as f32, site.y as f32, slope);
    }
    fixture
}

/// `|cos|` between the slanted plane's normal and camera 0's ray through the
/// centroid of seam A, as the rule measures the grazing angle.
fn seam_a_view_cosine(fixture: &Seams) -> f64 {
    let triangle = &fixture.triangles[SEAM_A];
    let [a, b, c] = [triangle.a, triangle.b, triangle.c].map(|index| {
        let point = &fixture.points[index];
        [point.x as f64, point.y as f64, point.z as f64]
    });
    let sub = |p: [f64; 3], q: [f64; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let (u, w) = (sub(b, a), sub(c, a));
    let normal = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];
    let centroid = [0, 1, 2].map(|axis| (a[axis] + b[axis] + c[axis]) / 3.0);
    let dot = |p: [f64; 3], q: [f64; 3]| p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    dot(normal, centroid).abs() / (dot(normal, normal) * dot(centroid, centroid)).sqrt()
}

/// Slopes of 45 and 60 degrees between the plane normal and the optical axis.
const NON_GRAZING_SLOPES: [f32; 2] = [1.0, 1.732_050_8];

#[test]
fn a_visible_seam_on_a_slanted_connected_plane_is_admitted() {
    for slope in NON_GRAZING_SLOPES {
        let fixture = slanted(slope).only_image_0();
        let cosine = seam_a_view_cosine(&fixture);
        assert!(
            cosine > 0.45,
            "slope {slope}: the fixture must be clearly non-grazing (|cos| {cosine})"
        );
        // Nothing but the plane itself is accepted geometry, so neither the
        // seams nor their coplanar neighbours can occlude a seam sample.
        let bake = fixture.bake();
        assert_eq!(
            bake.fallback.reasons.seam,
            SeamRejections::default(),
            "slope {slope}: a visible seam on its own plane must not be rejected"
        );
        assert_eq!(seam_material(&bake, 0).triangles, vec![SEAM_A, SEAM_B]);
        assert_eq!(bake.fallback.triangles, vec![2, 3], "slope {slope}");

        // With both images every seam is still textured by some camera.
        let bake = slanted(slope).bake();
        assert_eq!(bake.seam_triangles(), 2, "slope {slope}: {bake:?}");
        assert!(bake.fallback.triangles.is_empty(), "slope {slope}");
    }
}

#[test]
fn a_lone_visible_slanted_seam_is_not_occluded_by_its_own_depth() {
    for slope in NON_GRAZING_SLOPES {
        // Only seam A is accepted topology: its own depth is the only depth
        // in camera 0's buffer.
        let mut fixture = slanted(slope).only_image_0();
        fixture.triangles = vec![fixture.triangles[SEAM_A]];
        let bake = fixture.bake();
        assert_eq!(
            bake.fallback.reasons.seam,
            SeamRejections::default(),
            "slope {slope}: a seam must not occlude itself"
        );
        assert_eq!(seam_material(&bake, 0).triangles, vec![0]);
        assert!(bake.fallback.triangles.is_empty(), "slope {slope}");
    }
}

#[test]
fn a_parallel_occluder_in_front_of_a_slanted_seam_still_occludes() {
    for slope in NON_GRAZING_SLOPES {
        // A quad parallel to the slanted plane at 88 % of its depth along
        // every ray of camera 0 (scaling about the camera center keeps it
        // parallel): 12 % clearly in front of each seam sample, beyond the
        // 5 % occlusion tolerance, and covering the seams with margin.
        let in_front = |u: f32, v: f32| {
            let mut point = on_slanted_ray(0.0, u, v, slope);
            point.x *= 0.88;
            point.y *= 0.88;
            point.z *= 0.88;
            point
        };
        let mut fixture = slanted(slope).only_image_0();
        let start = fixture.points.len();
        fixture.points.extend([
            in_front(6.0, 2.0),
            in_front(25.0, 2.0),
            in_front(25.0, 22.0),
            in_front(6.0, 22.0),
        ]);
        fixture
            .sites
            .extend((0..4).map(|x| DenseGridSite { x, y: 0 }));
        fixture.triangles.extend([
            triangle(start, start + 1, start + 2),
            triangle(start, start + 2, start + 3),
        ]);
        fixture.cameras.push(camera(2, 0.5));
        fixture.patches.push(patch(2, 0, start, 4));

        let bake = fixture.bake();
        assert_eq!(
            bake.fallback.reasons.seam.occluded, 2,
            "slope {slope}: {:?}",
            bake.fallback.reasons.seam
        );
        assert_eq!(bake.seam_triangles(), 0, "slope {slope}");
        assert!(bake.fallback.triangles.contains(&SEAM_A));
        assert!(bake.fallback.triangles.contains(&SEAM_B));
    }
}

/// Bake with seam intrinsics through the cancellation entry point, counting
/// how often cancellation is polled; images are `width` x `height`.
fn polled_bake(fixture: &Seams, width: u32, height: u32) -> (SurfaceMaterialBake, usize) {
    let images = fixture
        .images
        .iter()
        .map(|(frame_index, rgba)| ReferenceImage {
            frame_index: *frame_index,
            width,
            height,
            rgba,
        })
        .collect::<Vec<_>>();
    let mut polls = 0usize;
    let bake = bake_with_cancel(
        &fixture.evidence(),
        &fixture.sites,
        &images,
        fixture
            .focal
            .map(|focal_pixels| SeamProjection { focal_pixels }),
        || {
            polls += 1;
            false
        },
    )
    .expect("uncanceled bake succeeds");
    (bake, polls)
}

/// The seam fixture centered in a `size` x `size` image (same focal length,
/// sites shifted with the principal point), plus an accepted background
/// triangle at depth 10 behind the plane. `cover` selects a background that
/// covers the whole image or one a few pixels wide.
fn large_image_with_background(size: u32, cover: bool) -> Seams {
    let mut fixture = Seams::new().only_image_0();
    let (dx, dy) = ((size - WIDTH) / 2, (size - HEIGHT) / 2);
    for site in &mut fixture.sites {
        site.x += dx;
        site.y += dy;
    }
    let rgba = (0..size * size)
        .flat_map(|index| [(index % 251) as u8, (index % 241) as u8, 7, 255])
        .collect();
    fixture.images = vec![(0, rgba)];
    let extent = if cover { 100_000.0 } else { 0.5 };
    fixture.with_occluder([
        world(-extent, -extent, 10.0),
        world(extent, -extent, 10.0),
        world(0.0, extent, 10.0),
    ])
}

#[test]
fn seam_depth_rasterization_polls_cancellation_at_bounded_pixel_intervals() {
    const SIZE: u32 = 2048;
    // The background lies behind the seams, so both bakes admit the same
    // seams and crop the same textures; they differ only in how many pixels
    // camera 0's seam depth buffer rasterizes for the background triangle.
    let (small, small_polls) = polled_bake(&large_image_with_background(SIZE, false), SIZE, SIZE);
    let (large, large_polls) = polled_bake(&large_image_with_background(SIZE, true), SIZE, SIZE);
    assert_eq!(small.seam_triangles(), 2, "{}", small.diagnostic());
    assert_eq!(large.seam_triangles(), 2, "{}", large.diagnostic());
    assert_eq!(small.materials.len(), large.materials.len());

    // One accepted triangle covering all 4,194,304 pixels must poll at least
    // once per 262,144 rasterized pixels.
    let pixels = (SIZE * SIZE) as usize;
    assert!(
        large_polls >= small_polls + pixels / 262_144,
        "rasterizing {pixels} pixels added only {} cancellation polls ({small_polls} -> {large_polls})",
        large_polls.saturating_sub(small_polls)
    );
}

#[test]
fn crop_corner_passes_poll_cancellation_per_bounded_triangle_count() {
    // A 1x1 image: one row to copy and one chunk to hash, so all other work
    // is the per-triangle corner bounds and corner UV passes.
    let rgba = [9u8, 8, 7, 255];
    let image = ReferenceImage {
        frame_index: 0,
        width: 1,
        height: 1,
        rgba: &rgba,
    };
    const TRIANGLES: usize = 262_144;
    let corners = vec![[[0.0f64, 0.0]; 3]; TRIANGLES];

    let mut polls = 0usize;
    let (texture, uvs) = crop_texture(&image, &corners, &mut || {
        polls += 1;
        Ok(())
    })
    .expect("uncanceled crop succeeds");
    assert_eq!((texture.width, texture.height), (1, 1));
    assert_eq!(uvs.len(), TRIANGLES);
    // Each of the two per-triangle passes polls at least every 4,096 triangles.
    assert!(
        polls >= 2 * (TRIANGLES / 4096),
        "{TRIANGLES} triangles polled cancellation only {polls} times"
    );

    // A cancellation requested after the copy and hash polls must still abort
    // the crop instead of finishing the corner passes.
    let mut seen = 0usize;
    let result = crop_texture(&image, &corners, &mut || {
        seen += 1;
        if seen > 2 {
            Err("texture bake was canceled".to_owned())
        } else {
            Ok(())
        }
    });
    assert_eq!(
        result.map(|_| ()),
        Err("texture bake was canceled".to_owned())
    );
}

/// Camera-0 samples of seam A as the rule tests them: vertices, edge
/// midpoints and the centroid (camera 0 is the world frame).
fn seam_a_samples(fixture: &Seams) -> [[f64; 3]; 7] {
    let triangle = &fixture.triangles[SEAM_A];
    let [a, b, c] = [triangle.a, triangle.b, triangle.c].map(|index| {
        let point = &fixture.points[index];
        [point.x as f64, point.y as f64, point.z as f64]
    });
    let mid = |p: [f64; 3], q: [f64; 3]| [0, 1, 2].map(|axis| (p[axis] + q[axis]) * 0.5);
    let centroid = [0, 1, 2].map(|axis| (a[axis] + b[axis] + c[axis]) / 3.0);
    [a, b, c, mid(a, b), mid(b, c), mid(c, a), centroid]
}

/// A 0.2 px fronto-parallel triangle centered on camera 0's ray through
/// `sample`, at `fraction` of the sample's depth.
fn thin_occluder_on_ray(sample: [f64; 3], fraction: f64) -> [Point3; 3] {
    let u = FOCAL as f64 * sample[0] / sample[2] + WIDTH as f64 * 0.5;
    let v = FOCAL as f64 * sample[1] / sample[2] + HEIGHT as f64 * 0.5;
    let depth = sample[2] * fraction;
    let at = |u: f64, v: f64| {
        world(
            ((u - WIDTH as f64 * 0.5) / FOCAL as f64 * depth) as f32,
            ((v - HEIGHT as f64 * 0.5) / FOCAL as f64 * depth) as f32,
            depth as f32,
        )
    };
    [at(u - 0.1, v - 0.1), at(u + 0.1, v - 0.1), at(u, v + 0.1)]
}

/// Steep but non-grazing slopes (|cos| about 0.71, 0.51, 0.38 and 0.33 at
/// seam A, all above the 0.2 grazing bound).
const STEEP_SLOPES: [f32; 4] = [1.0, 1.732_050_8, 2.5, 3.0];

#[test]
fn a_thin_occluder_clearly_in_front_along_a_slanted_seam_sample_ray_rejects_the_seam() {
    // On a steep plane the seam's own depth varies by several percent inside
    // one pixel cell. Accepted geometry 7 or 8 % in front of a sample along
    // that sample's own ray is clearly in front (beyond the 5 % tolerance) and
    // must never read as the seam's own surface, wherever the sample falls in
    // its cell.
    let mut admitted = Vec::new();
    for slope in STEEP_SLOPES {
        let clean = slanted(slope).only_image_0();
        assert!(
            seam_a_view_cosine(&clean) > 0.3,
            "slope {slope} must not be grazing"
        );
        // Without the occluder the seam is visible and admitted, so a
        // rejection below is caused by the occluder alone.
        assert!(
            seam_material(&clean.bake(), 0).triangles.contains(&SEAM_A),
            "slope {slope}: the unoccluded slanted seam A must be admitted"
        );
        for (ordinal, sample) in seam_a_samples(&clean).into_iter().enumerate() {
            for fraction in [0.92, 0.93] {
                let bake = slanted(slope)
                    .only_image_0()
                    .with_occluder(thin_occluder_on_ray(sample, fraction))
                    .bake();
                if !bake.fallback.triangles.contains(&SEAM_A) {
                    admitted.push((slope, ordinal, fraction));
                }
            }
        }
    }
    assert!(
        admitted.is_empty(),
        "seam A was textured although accepted geometry lies clearly in front along one of \
         its sample rays; (slope, sample 0-2 vertices / 3-5 midpoints / 6 centroid, \
         occluder depth fraction): {admitted:?}"
    );
}

// No bake-level test pins the horizon-crossing neighbour cell (Codex finding on
// 9a5391b): a seam sample whose 3x3 neighbourhood straddles the seam plane's
// horizon lies within about 1.5 px of it, and the 3-D centroid of a triangle on
// the plane projects at roughly the harmonic mean of its vertices' pixel
// distances to the horizon, so that centroid stays within about 4.5 px of it.
// The view cosine there is about 4.5 / (focal |ray|) * k / sqrt(1 + k^2), below
// the 0.2 grazing bound for this fixture's 20 px focal length, so every such
// seam is already rejected as grazing before visibility is classified.

/// The seam fixture with camera 0's image `width` x 1 pixels: the sites keep
/// camera 0's grid exactly reproducible (x shifted with the principal point,
/// camera 0 moved so v = site.y), so camera 0 is a seam candidate and its seam
/// depth buffer is rasterized, while every seam is out of its frame. The extra
/// accepted triangle at depth 10 has camera-0 pixel corners `corners`.
fn strip_image(width: u32, corners: [[f64; 2]; 3]) -> Seams {
    let mut fixture = Seams::new().only_image_0();
    for site in &mut fixture.sites {
        site.x += (width - WIDTH) / 2;
    }
    // v = FOCAL (y + t_y) / DEPTH + 0.5 equals the old v = FOCAL y / DEPTH + 12.
    let lift = (HEIGHT as f32 * 0.5 - 0.5) * DEPTH / FOCAL;
    fixture.cameras[0].translation[1] = lift;
    fixture.images = vec![(0, [3, 2, 1, 255].repeat(width as usize))];
    let depth = 10.0;
    fixture.with_occluder(corners.map(|[u, v]| {
        world(
            ((u - width as f64 * 0.5) / FOCAL as f64 * depth) as f32,
            ((v - 0.5) / FOCAL as f64 * depth) as f32 - lift,
            depth as f32,
        )
    }))
}

/// Polls of a bake of `strip_image(width, corners)` minus those of the same
/// bake whose extra triangle lies entirely outside the image.
fn extra_raster_polls(width: u32, corners: [[f64; 2]; 3]) -> usize {
    let outside = [[-10.0, -10.0], [-9.0, -10.0], [-10.0, -9.0]];
    let (quiet, quiet_polls) = polled_bake(&strip_image(width, outside), width, 1);
    let (busy, busy_polls) = polled_bake(&strip_image(width, corners), width, 1);
    for bake in [&quiet, &busy] {
        assert_eq!(
            bake.fallback.reasons.seam.out_of_frame,
            2,
            "{}",
            bake.diagnostic()
        );
    }
    busy_polls.saturating_sub(quiet_polls)
}

const STRIP_WIDTH: u32 = 1 << 20;

#[test]
fn a_raster_row_wider_than_a_poll_batch_still_polls_at_bounded_intervals() {
    // One row of 1,048,576 pixels, covered by a triangle whose edges all stay
    // outside the strip (no edge samples): the work is the row fill alone.
    let w = STRIP_WIDTH as f64;
    let extra = extra_raster_polls(
        STRIP_WIDTH,
        [[-4.0 * w, -5.0], [5.0 * w, -5.0], [0.5 * w, 1.0e4]],
    );
    let pixels = STRIP_WIDTH as usize;
    assert!(
        extra >= pixels / 262_144,
        "filling one {pixels}-pixel row polled cancellation only {extra} extra times"
    );
}

#[test]
fn an_edge_with_more_samples_than_a_poll_batch_still_polls_at_bounded_intervals() {
    // A horizontal edge at v = 0.3 across the whole strip (about 4,194,304
    // quarter-pixel samples); the triangle lies above it, so no pixel center
    // of the strip is filled and the other edges stay outside the strip.
    let w = STRIP_WIDTH as f64;
    let extra = extra_raster_polls(
        STRIP_WIDTH,
        [[-3.0 * w, 0.3], [4.0 * w, 0.3], [0.5 * w, 1.0e4]],
    );
    let samples = 4 * STRIP_WIDTH as usize;
    assert!(
        extra >= samples / 262_144,
        "stamping one edge of about {samples} samples polled cancellation only {extra} extra times"
    );
}
