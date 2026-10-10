use super::*;
use crate::textured_glb::encode_textured_glb;
use crate::{
    classic_reference_patch_regions, DenseReferencePatchStats, EvidenceCamera, EvidenceRange,
    EvidenceScale, MeshTriangle, Point3, ReconstructionProviderClass,
    ReconstructionProviderDescriptor, SurfaceEvidenceRegion,
};
use serde_json::Value;

const WIDTH: u32 = 16;
const HEIGHT: u32 = 12;

fn point(index: usize) -> Point3 {
    Point3 {
        x: index as f32 * 0.25,
        y: 0.5 - index as f32 * 0.125,
        z: 2.0 + index as f32 * 0.01,
        confidence: 0.9,
        r: (index * 30) as u8,
        g: 200,
        b: 10,
    }
}

fn camera(frame_index: usize, x: f32) -> EvidenceCamera {
    EvidenceCamera {
        frame_index,
        authority: EvidenceCameraAuthority::CalibratedSeed,
        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation: [x, 0.0, 0.0],
        confidence: Some(0.9),
        median_reprojection_error_pixels: Some(0.4),
    }
}

fn image_pixels(seed: u8) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            rgba.extend_from_slice(&[
                (x as u8 * 13).wrapping_add(seed),
                (y as u8 * 17).wrapping_add(seed),
                seed,
                255,
            ]);
        }
    }
    rgba
}

fn site(x: u32, y: u32) -> DenseGridSite {
    DenseGridSite { x, y }
}

fn triangle(a: usize, b: usize, c: usize) -> MeshTriangle {
    MeshTriangle {
        a,
        b,
        c,
        confidence: 0.8,
    }
}

/// Two reference patches: frame 0 owns points 0..4 (point 3 is revalidated
/// completion), frame 1 owns points 4..8.
struct Fixture {
    points: Vec<Point3>,
    sites: Vec<DenseGridSite>,
    triangles: Vec<MeshTriangle>,
    cameras: Vec<EvidenceCamera>,
    regions: Vec<SurfaceEvidenceRegion>,
    images: Vec<(usize, Vec<u8>)>,
}

impl Fixture {
    fn new() -> Self {
        let patches = vec![
            DenseReferencePatchStats {
                reference_frame: 0,
                source_frames: vec![1, 2],
                primary_start: 0,
                primary_points: 3,
                completion_start: 3,
                completed_points: 1,
                ..DenseReferencePatchStats::default()
            },
            DenseReferencePatchStats {
                reference_frame: 1,
                source_frames: vec![0],
                primary_start: 4,
                primary_points: 4,
                completion_start: 8,
                completed_points: 0,
                ..DenseReferencePatchStats::default()
            },
        ];
        Self {
            points: (0..8).map(point).collect(),
            sites: vec![
                site(2, 2),
                site(6, 2),
                site(2, 6),
                site(6, 6),
                site(3, 3),
                site(9, 3),
                site(3, 9),
                site(9, 9),
            ],
            triangles: vec![
                triangle(0, 1, 2),
                triangle(1, 3, 2),
                triangle(4, 5, 6),
                triangle(2, 3, 4),
                triangle(5, 7, 6),
            ],
            cameras: vec![camera(0, 0.0), camera(1, 0.3), camera(2, 0.6)],
            regions: classic_reference_patch_regions(&patches),
            images: vec![(0, image_pixels(5)), (1, image_pixels(90))],
        }
    }

    fn evidence(&self) -> ReconstructionEvidenceView<'_> {
        ReconstructionEvidenceView::new(
            ReconstructionProviderDescriptor::classic(),
            EvidenceScale::ArbitraryMonocular,
            self.cameras.clone(),
            self.regions.clone(),
            &self.points,
            &self.triangles,
        )
        .expect("fixture evidence is valid")
    }

    fn bake(&self) -> SurfaceMaterialBake {
        bake_surface_materials(&self.evidence(), &self.sites, &self.reference_images())
            .expect("bake succeeds")
    }

    fn reference_images(&self) -> Vec<ReferenceImage<'_>> {
        self.images
            .iter()
            .map(|(frame_index, rgba)| ReferenceImage {
                frame_index: *frame_index,
                width: WIDTH,
                height: HEIGHT,
                rgba,
            })
            .collect()
    }
}

#[test]
fn single_reference_triangles_are_textured_and_mixed_ones_fall_back() {
    let fixture = Fixture::new();
    let bake = fixture.bake();

    assert_eq!(bake.materials.len(), 2);
    let first = &bake.materials[0];
    assert_eq!(first.reference_frame, 0);
    assert_eq!(first.triangles, vec![0, 1]);
    assert_eq!(first.source_frames, vec![0, 1, 2]);
    assert_eq!(
        first.provenance,
        vec![
            EvidenceOrigin::GeometricMultiView,
            EvidenceOrigin::RevalidatedCompletion
        ]
    );
    assert_eq!(
        first.camera_authority,
        EvidenceCameraAuthority::CalibratedSeed
    );
    let second = &bake.materials[1];
    assert_eq!(second.reference_frame, 1);
    assert_eq!(second.triangles, vec![2, 4]);
    assert_eq!(second.provenance, vec![EvidenceOrigin::GeometricMultiView]);
    assert_eq!(second.camera_translation, [0.3, 0.0, 0.0]);

    assert_eq!(bake.fallback.triangles, vec![3]);
    assert_eq!(bake.fallback.reasons.mixed_reference, 1);
    assert_eq!(bake.fallback.reasons.total(), 1);
    assert_eq!(bake.textured_triangles() + bake.fallback.triangles.len(), 5);
    assert!(bake
        .diagnostic()
        .contains("4 of 5 accepted triangles textured"));
}

#[test]
fn texture_crop_and_uvs_follow_the_reference_grid() {
    let fixture = Fixture::new();
    let bake = fixture.bake();
    let texture = &bake.materials[0].texture;
    // Sites span 2..=6 in both axes; one texel of margin on every side.
    assert_eq!(texture.crop_origin, [1, 1]);
    assert_eq!((texture.width, texture.height), (7, 7));
    assert_eq!(
        (texture.source_image_width, texture.source_image_height),
        (WIDTH, HEIGHT)
    );
    let source = &fixture.images[0].1;
    for y in 0..texture.height {
        for x in 0..texture.width {
            let crop = ((y * texture.width + x) * 4) as usize;
            let full = (((y + 1) * WIDTH + x + 1) * 4) as usize;
            assert_eq!(texture.rgba[crop..crop + 4], source[full..full + 4]);
        }
    }
    // Corner a of triangle 0 is site (2, 2): texel (1, 1) of the crop, center.
    let uv = bake.materials[0].corner_uvs[0][0];
    assert!((uv[0] - 1.5 / 7.0).abs() < 1e-6 && (uv[1] - 1.5 / 7.0).abs() < 1e-6);
}

#[test]
fn missing_images_unobserved_sites_and_degenerate_footprints_fall_back() {
    let mut fixture = Fixture::new();
    fixture.images.truncate(1);
    let bake = fixture.bake();
    assert_eq!(bake.materials.len(), 1);
    assert_eq!(bake.fallback.reasons.missing_reference_image, 2);
    assert_eq!(bake.fallback.reasons.mixed_reference, 1);

    let mut fixture = Fixture::new();
    fixture.sites[5] = site(WIDTH, 3);
    fixture.sites[0] = site(2, 6);
    fixture.sites[1] = site(4, 4);
    let bake = fixture.bake();
    // Triangle 0 sites (2,6), (4,4), (2,6) are collinear; triangles 2 and 4 use
    // the out-of-image site of point 5.
    assert_eq!(bake.fallback.reasons.degenerate_footprint, 1);
    assert_eq!(bake.fallback.reasons.missing_grid_site, 2);

    let mut fixture = Fixture::new();
    fixture.sites.pop();
    let bake = fixture.bake();
    assert!(bake.materials.is_empty());
    assert_eq!(bake.fallback.reasons.missing_grid_site, 4);
    assert_eq!(bake.fallback.reasons.mixed_reference, 1);
}

#[test]
fn learned_evidence_is_never_textured_as_observed_reference_appearance() {
    let fixture = Fixture::new();
    let regions = vec![SurfaceEvidenceRegion {
        origin: EvidenceOrigin::LearnedMultiView,
        reference_frame: Some(0),
        source_frames: vec![1],
        points: EvidenceRange::new(0, fixture.points.len()),
    }];
    let cameras = fixture
        .cameras
        .iter()
        .cloned()
        .map(|mut camera| {
            camera.authority = EvidenceCameraAuthority::ProviderEstimated;
            camera
        })
        .collect();
    let provider = ReconstructionProviderDescriptor::new(
        "test/learned",
        ReconstructionProviderClass::LearnedMultiView,
        None,
    )
    .unwrap();
    let evidence = ReconstructionEvidenceView::new(
        provider,
        EvidenceScale::ProviderLocal,
        cameras,
        regions,
        &fixture.points,
        &fixture.triangles,
    )
    .unwrap();
    let bake =
        bake_surface_materials(&evidence, &fixture.sites, &fixture.reference_images()).unwrap();
    assert!(bake.materials.is_empty());
    assert_eq!(bake.fallback.reasons.unobserved_reference, 5);
}

#[test]
fn malformed_or_duplicate_reference_images_fail_closed() {
    let fixture = Fixture::new();
    let evidence = fixture.evidence();
    let short = ReferenceImage {
        frame_index: 0,
        width: WIDTH,
        height: HEIGHT,
        rgba: &fixture.images[0].1[4..],
    };
    assert!(bake_surface_materials(&evidence, &fixture.sites, &[short]).is_err());
    let images = fixture.reference_images();
    let duplicate = [images[0], images[0]];
    assert!(bake_surface_materials(&evidence, &fixture.sites, &duplicate).is_err());
}

#[test]
fn cancellation_interrupts_evidence_validation_before_traversal_finishes() {
    let mut small = Fixture::new();
    small.triangles.truncate(1);
    let mut small_polls = 0;
    small
        .evidence()
        .validate_with_cancel(|| {
            small_polls += 1;
            Ok(())
        })
        .unwrap();

    let mut large = Fixture::new();
    large.triangles = (0..4096).map(|_| triangle(0, 1, 2)).collect();
    let mut polls = 0;
    let error = large
        .evidence()
        .validate_with_cancel(|| {
            polls += 1;
            if polls >= small_polls + 2 {
                Err("validation canceled".into())
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert!(error.contains("canceled"), "{error}");
    assert_eq!(polls, small_polls + 2);
}

#[test]
fn cancellation_interrupts_inner_triangle_work() {
    let mut fixture = Fixture::new();
    // More than two polling windows; a cancel delivered mid-traversal must
    // stop before the full bake or any of the texture crops are produced.
    fixture.triangles = (0..4096).map(|_| triangle(0, 1, 2)).collect();
    let evidence = fixture.evidence();
    let images = fixture.reference_images();
    let mut validation_polls = 0;
    evidence
        .validate_with_cancel(|| {
            validation_polls += 1;
            Ok(())
        })
        .unwrap();
    // Beyond validation: check the image and region setup, then cancel in
    // the large triangle traversal before materials are allocated.
    let threshold = validation_polls + 11;
    let mut checks = 0;
    let error = bake_surface_materials_with_cancel(&evidence, &fixture.sites, &images, || {
        checks += 1;
        checks >= threshold
    })
    .unwrap_err();
    assert!(error.contains("canceled"), "{error}");
    assert_eq!(checks, threshold, "the work should poll during triangle traversal");
}

#[test]
fn cancellation_rejects_all_fallback_bakes() {
    let mut fixture = Fixture::new();
    // No grid mapping means every triangle falls back; there are no materials
    // and hence no PNG loop in which an executor could notice cancellation.
    fixture.sites.clear();
    let evidence = fixture.evidence();
    let images = fixture.reference_images();
    let mut full_polls = 0;
    let full = bake_surface_materials_with_cancel(&evidence, &fixture.sites, &images, || {
        full_polls += 1;
        false
    })
    .unwrap();
    assert!(full.materials.is_empty());
    let mut checks = 0;
    let error = bake_surface_materials_with_cancel(&evidence, &fixture.sites, &images, || {
        checks += 1;
        checks == full_polls
    })
    .unwrap_err();
    assert!(error.contains("canceled"), "{error}");
    assert_eq!(
        checks, full_polls,
        "cancellation must be checked after fallback processing"
    );
}

#[test]
fn reprojection_and_framing_changes_invalidate_only_the_affected_reference() {
    let fixture = Fixture::new();
    let baseline = fixture.bake();
    let recorded = baseline.recorded_appearance();
    let unchanged = fixture.bake().invalidation_against(&recorded);
    assert_eq!(unchanged.reused, vec![0, 1]);
    assert!(unchanged.invalidated.is_empty());

    // Reprojecting reference 1 (new camera pose) leaves reference 0 valid.
    let mut moved = Fixture::new();
    moved.cameras[1].translation = [0.31, 0.0, 0.0];
    let diff = moved.bake().invalidation_against(&recorded);
    assert_eq!((diff.reused, diff.invalidated), (vec![0], vec![1]));

    // Re-sampling pixels inside reference 0's crop invalidates only it.
    let mut reframed = Fixture::new();
    let pixel = ((3 * WIDTH + 3) * 4) as usize;
    reframed.images[0].1[pixel] ^= 0xff;
    let diff = reframed.bake().invalidation_against(&recorded);
    assert_eq!((diff.reused, diff.invalidated), (vec![1], vec![0]));

    // Pixels outside every crop and point positions do not affect appearance.
    let mut unrelated = Fixture::new();
    let outside = ((11 * WIDTH + 15) * 4) as usize;
    unrelated.images[0].1[outside] ^= 0xff;
    unrelated.points[4].z += 0.5;
    let diff = unrelated.bake().invalidation_against(&recorded);
    assert_eq!(diff.reused, vec![0, 1]);

    // Changing reference 1's footprint shifts nothing in reference 0.
    let mut regridded = Fixture::new();
    regridded.sites[7] = site(10, 10);
    let diff = regridded.bake().invalidation_against(&recorded);
    assert_eq!((diff.reused, diff.invalidated), (vec![0], vec![1]));

    // Losing a reference image removes only that artifact.
    let mut dropped = Fixture::new();
    dropped.images.truncate(1);
    let diff = dropped.bake().invalidation_against(&recorded);
    assert!(diff.diagnostic().contains("removed 1"));
    assert_eq!((diff.reused, diff.removed), (vec![0], vec![1]));
}

fn glb_parts(glb: &[u8]) -> (Value, &[u8]) {
    assert_eq!(&glb[0..4], b"glTF");
    assert_eq!(u32::from_le_bytes(glb[4..8].try_into().unwrap()), 2);
    assert_eq!(
        u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize,
        glb.len()
    );
    let json_length = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    assert_eq!(&glb[16..20], b"JSON");
    let json = serde_json::from_slice(&glb[20..20 + json_length]).unwrap();
    let bin_header = 20 + json_length;
    let bin_length =
        u32::from_le_bytes(glb[bin_header..bin_header + 4].try_into().unwrap()) as usize;
    assert_eq!(&glb[bin_header + 4..bin_header + 8], b"BIN\0");
    let bin = &glb[bin_header + 8..bin_header + 8 + bin_length];
    assert_eq!(bin_header + 8 + bin_length, glb.len());
    (json, bin)
}

fn view_bytes<'a>(json: &Value, bin: &'a [u8], view: &Value) -> &'a [u8] {
    let view = &json["bufferViews"][view.as_u64().unwrap() as usize];
    let offset = view["byteOffset"].as_u64().unwrap() as usize;
    let length = view["byteLength"].as_u64().unwrap() as usize;
    &bin[offset..offset + length]
}

fn decode_png_rgb(png: &[u8]) -> (u32, u32, Vec<u8>) {
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    let mut cursor = 8;
    let (mut width, mut height, mut idat) = (0, 0, Vec::new());
    while cursor < png.len() {
        let length = u32::from_be_bytes(png[cursor..cursor + 4].try_into().unwrap()) as usize;
        let kind = &png[cursor + 4..cursor + 8];
        let data = &png[cursor + 8..cursor + 8 + length];
        let crc = u32::from_be_bytes(
            png[cursor + 8 + length..cursor + 12 + length]
                .try_into()
                .unwrap(),
        );
        assert_eq!(crc, crc32fast::hash(&png[cursor + 4..cursor + 8 + length]));
        match kind {
            b"IHDR" => {
                width = u32::from_be_bytes(data[0..4].try_into().unwrap());
                height = u32::from_be_bytes(data[4..8].try_into().unwrap());
                assert_eq!(&data[8..], &[8, 2, 0, 0, 0]);
            }
            b"IDAT" => idat.extend_from_slice(data),
            _ => {}
        }
        cursor += 12 + length;
    }
    let raw = miniz_oxide::inflate::decompress_to_vec_zlib(&idat).unwrap();
    let row = width as usize * 3;
    let mut rgb = Vec::with_capacity(row * height as usize);
    for line in raw.chunks_exact(row + 1) {
        assert_eq!(line[0], 1, "rows use the Sub filter");
        let start = rgb.len();
        for (index, byte) in line[1..].iter().enumerate() {
            let left = if index >= 3 {
                rgb[start + index - 3]
            } else {
                0
            };
            rgb.push(byte.wrapping_add(left));
        }
    }
    (width, height, rgb)
}

#[test]
fn textured_glb_is_self_contained_and_preserves_accepted_geometry() {
    let fixture = Fixture::new();
    let evidence = fixture.evidence();
    let bake = fixture.bake();
    let glb = encode_textured_glb(&evidence, &bake).unwrap();
    assert_eq!(
        glb,
        encode_textured_glb(&evidence, &fixture.bake()).unwrap()
    );
    let (json, bin) = glb_parts(&glb);

    assert_eq!(json["asset"]["version"], "2.0");
    let extras = &json["asset"]["extras"]["video_to_3d"];
    assert_eq!(extras["schema"], "video-to-3d/textured-surface");
    assert_eq!(extras["provider"]["id"], "video-to-3d-core/classic");
    assert_eq!(extras["textured_triangles"], 4);
    assert_eq!(extras["fallback_triangles"], 1);
    assert!(json.get("uri").is_none() && json["buffers"][0].get("uri").is_none());

    let primitives = json["meshes"][0]["primitives"].as_array().unwrap();
    assert_eq!(primitives.len(), 3);
    let mut exported_triangles = 0;
    for (primitive_index, primitive) in primitives.iter().enumerate() {
        let indices = &json["accessors"][primitive["indices"].as_u64().unwrap() as usize];
        exported_triangles += indices["count"].as_u64().unwrap() / 3;
        let positions =
            &json["accessors"][primitive["attributes"]["POSITION"].as_u64().unwrap() as usize];
        let floats = view_bytes(&json, bin, &positions["bufferView"])
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        // Every exported vertex is an accepted point re-expressed as (x, -y, -z).
        for vertex in floats.as_chunks::<3>().0 {
            assert!(fixture
                .points
                .iter()
                .any(|p| [p.x, -p.y, -p.z] == [vertex[0], vertex[1], vertex[2]]));
        }
        let material = &json["materials"][primitive["material"].as_u64().unwrap() as usize];
        if primitive_index < 2 {
            assert!(primitive["attributes"]["TEXCOORD_0"].is_u64());
            let provenance = &material["extras"]["video_to_3d"];
            let reference = &bake.materials[primitive_index];
            assert_eq!(provenance["reference_frame"], reference.reference_frame);
            assert_eq!(
                provenance["appearance_key"],
                reference.appearance_key.as_str()
            );
            assert_eq!(provenance["camera_authority"], "calibrated_seed");
            let texture = &json["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]
                ["index"]
                .as_u64()
                .unwrap() as usize];
            let image = &json["images"][texture["source"].as_u64().unwrap() as usize];
            assert_eq!(image["mimeType"], "image/png");
            let (width, height, rgb) = decode_png_rgb(view_bytes(&json, bin, &image["bufferView"]));
            assert_eq!(
                (width, height),
                (reference.texture.width, reference.texture.height)
            );
            let expected = reference
                .texture
                .rgba
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|pixel| pixel[..3].to_vec())
                .collect::<Vec<_>>();
            assert_eq!(rgb, expected);
        } else {
            assert!(primitive["attributes"]["COLOR_0"].is_u64());
            assert_eq!(
                material["extras"]["video_to_3d"]["fallback"],
                "vertex_color"
            );
            assert_eq!(
                material["extras"]["video_to_3d"]["reasons"]["mixed_reference"],
                1
            );
        }
    }
    assert_eq!(exported_triangles as usize, fixture.triangles.len());
}

#[test]
fn glb_export_rejects_a_bake_for_different_topology() {
    let fixture = Fixture::new();
    let mut bake = fixture.bake();
    bake.fallback.triangles.clear();
    assert!(encode_textured_glb(&fixture.evidence(), &bake).is_err());
    let mut bake = fixture.bake();
    bake.fallback.triangles.push(0);
    assert!(encode_textured_glb(&fixture.evidence(), &bake).is_err());
}
