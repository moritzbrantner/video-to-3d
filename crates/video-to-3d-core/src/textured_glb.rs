//! Self-contained binary glTF 2.0 export of an accepted textured surface.
//!
//! The GLB embeds every baked reference texture as a PNG and needs no browser
//! state to load. Each reference material becomes one primitive with
//! `TEXCOORD_0`; the untextured fallback becomes one primitive with vertex
//! colors. Positions are exported exactly as accepted, only re-expressed in the
//! glTF frame. Material, image, and asset `extras.video_to_3d` carry the
//! texturing camera, the rule that admitted the material's triangles
//! (`same_reference_grid` or `seam_single_camera_projection`, the latter with
//! the seam's reference frames and focal length), provenance, appearance key,
//! and fallback reasons.
//!
//! Frame: reconstruction coordinates follow the camera convention of the seed
//! camera (+X right, +Y down, +Z forward). glTF is +Y up with cameras looking
//! along -Z, so positions are rotated 180 degrees about X: `(x, -y, -z)`. The
//! rotation is proper, so it preserves handedness; accepted triangles have
//! positive signed area in their (y-down) reference image, i.e. they are
//! clockwise as seen from the reference camera, so corners are emitted as
//! `(a, c, b)` to face that camera under the glTF counter-clockwise rule.
//!
//! Collision: when a [`CoarseCollider`] is supplied it is written as its own
//! mesh and node (`coarse-collider`, `extras.video_to_3d.role = "collision"`)
//! in a second glTF scene named `collision`. The default scene stays the
//! visual surface, so viewers render exactly the accepted geometry while
//! engines and tools can load the collider by scene, node name or role.

use crate::coarse_collision::CoarseCollider;
use crate::geometry_confidence::GeometryConfidenceField;
use crate::surface_materials::{
    MaterialRule, SurfaceMaterialBake, SURFACE_MATERIAL_BAKE_SCHEMA_VERSION,
};
use crate::ReconstructionEvidenceView;
use serde_json::{json, Value};
use std::collections::HashMap;

const GLB_MAGIC: u32 = 0x4654_6C67;
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;
const LINEAR: u32 = 9729;
const LINEAR_MIPMAP_LINEAR: u32 = 9987;
const CLAMP_TO_EDGE: u32 = 33071;
pub const TEXTURED_SURFACE_SCHEMA: &str = "video-to-3d/textured-surface";

#[derive(Default)]
struct GltfBuilder {
    bin: Vec<u8>,
    buffer_views: Vec<Value>,
    accessors: Vec<Value>,
}

impl GltfBuilder {
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        let mut view = json!({
            "buffer": 0,
            "byteOffset": self.bin.len(),
            "byteLength": bytes.len(),
        });
        if let Some(target) = target {
            view["target"] = json!(target);
        }
        self.bin.extend_from_slice(bytes);
        self.buffer_views.push(view);
        self.buffer_views.len() - 1
    }

    fn float_accessor(&mut self, values: &[f32], kind: &str, width: usize, bounds: bool) -> usize {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view = self.view(&bytes, Some(ARRAY_BUFFER));
        let mut accessor = json!({
            "bufferView": view,
            "componentType": FLOAT,
            "count": values.len() / width,
            "type": kind,
        });
        if bounds {
            let mut min = vec![f32::INFINITY; width];
            let mut max = vec![f32::NEG_INFINITY; width];
            for chunk in values.chunks_exact(width) {
                for (axis, value) in chunk.iter().enumerate() {
                    min[axis] = min[axis].min(*value);
                    max[axis] = max[axis].max(*value);
                }
            }
            accessor["min"] = json!(min);
            accessor["max"] = json!(max);
        }
        self.accessors.push(accessor);
        self.accessors.len() - 1
    }

    fn index_accessor(&mut self, indices: &[u32]) -> usize {
        let bytes = indices
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view = self.view(&bytes, Some(ELEMENT_ARRAY_BUFFER));
        self.accessors.push(json!({
            "bufferView": view,
            "componentType": UNSIGNED_INT,
            "count": indices.len(),
            "type": "SCALAR",
        }));
        self.accessors.len() - 1
    }
}

/// Collects the vertices of one primitive, keyed by their dense point index.
#[derive(Default)]
struct PrimitiveVertices {
    local: HashMap<usize, u32>,
    order: Vec<usize>,
    indices: Vec<u32>,
}

impl PrimitiveVertices {
    /// Returns whether the point was newly added.
    fn push(&mut self, point: usize) -> bool {
        let next = self.order.len() as u32;
        let mut added = false;
        let local = *self.local.entry(point).or_insert_with(|| {
            added = true;
            next
        });
        if added {
            self.order.push(point);
        }
        self.indices.push(local);
        added
    }
}

fn srgb_to_linear(channel: u8) -> f32 {
    let value = channel as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Encode the accepted surface of `evidence` with the materials of `bake`.
pub fn encode_textured_glb(
    evidence: &ReconstructionEvidenceView<'_>,
    bake: &SurfaceMaterialBake,
) -> Result<Vec<u8>, String> {
    encode_textured_glb_with_collider(evidence, bake, None)
}

/// Like [`encode_textured_glb`], plus an optional separately identified
/// collision scene.
pub fn encode_textured_glb_with_collider(
    evidence: &ReconstructionEvidenceView<'_>,
    bake: &SurfaceMaterialBake,
    collider: Option<&CoarseCollider>,
) -> Result<Vec<u8>, String> {
    // A collider is exported only when it is exactly what this evidence yields, so stale
    // or foreign collision geometry can never be paired with this surface.
    if let Some(collider) = collider {
        let field = GeometryConfidenceField::from_evidence(evidence);
        if CoarseCollider::from_evidence(evidence, &field, collider.options) != *collider {
            return Err("the coarse collider was not built from this evidence".into());
        }
    }
    let points = evidence.points;
    let triangles = evidence.triangles;
    if bake.point_count != points.len() || bake.triangle_count != triangles.len() {
        return Err(format!(
            "surface material bake covers {} points and {} triangles, but the evidence has {} and {}",
            bake.point_count,
            bake.triangle_count,
            points.len(),
            triangles.len()
        ));
    }
    if triangles.is_empty() {
        return Err("no accepted surface triangles to export".into());
    }
    let mut covered = vec![false; triangles.len()];
    let all_triangles = bake
        .materials
        .iter()
        .flat_map(|material| &material.triangles)
        .chain(&bake.fallback.triangles);
    for index in all_triangles {
        match covered.get_mut(*index) {
            Some(seen @ false) => *seen = true,
            _ => {
                return Err(format!(
                    "surface material bake assigns triangle {index} zero or several times"
                ))
            }
        }
    }
    if covered.iter().any(|seen| !seen) {
        return Err("surface material bake leaves an accepted triangle unassigned".into());
    }

    let position = |index: usize| {
        let point = &points[index];
        [point.x, -point.y, -point.z]
    };
    let mut gltf = GltfBuilder::default();
    let mut primitives = Vec::new();
    let mut materials = Vec::new();
    let mut images = Vec::new();
    let mut textures = Vec::new();

    for material in &bake.materials {
        if material.corner_uvs.len() != material.triangles.len() {
            return Err(format!(
                "reference material {} has {} corner UV sets for {} triangles",
                material.reference_frame,
                material.corner_uvs.len(),
                material.triangles.len()
            ));
        }
        let mut vertices = PrimitiveVertices::default();
        let mut uvs = Vec::new();
        for (triangle, corner_uvs) in material.triangles.iter().zip(&material.corner_uvs) {
            let triangle = &triangles[*triangle];
            for (point, uv) in [
                (triangle.a, corner_uvs[0]),
                (triangle.c, corner_uvs[2]),
                (triangle.b, corner_uvs[1]),
            ] {
                if vertices.push(point) {
                    uvs.extend_from_slice(&uv);
                }
            }
        }
        let positions = vertices
            .order
            .iter()
            .flat_map(|index| position(*index))
            .collect::<Vec<_>>();
        let position_accessor = gltf.float_accessor(&positions, "VEC3", 3, true);
        let uv_accessor = gltf.float_accessor(&uvs, "VEC2", 2, false);
        let index_accessor = gltf.index_accessor(&vertices.indices);

        let texture = &material.texture;
        let png = encode_png_rgb(texture.width, texture.height, &texture.rgba)?;
        let image_view = gltf.view(&png, None);
        let name = match material.rule {
            MaterialRule::SameReferenceGrid => {
                format!("reference-frame-{}", material.reference_frame)
            }
            MaterialRule::SeamSingleCameraProjection => {
                format!("seam-camera-frame-{}", material.reference_frame)
            }
        };
        let mut provenance = json!({
            "reference_frame": material.reference_frame,
            "rule": material.rule,
            "camera_authority": material.camera_authority,
            "camera_rotation_world_to_camera": material.camera_rotation,
            "camera_translation_world_to_camera": material.camera_translation,
            "source_frames": material.source_frames,
            "provenance": material.provenance,
            "appearance_key": material.appearance_key,
            "texture_content_hash": texture.content_hash,
            "crop_origin": texture.crop_origin,
            "crop_size": [texture.width, texture.height],
            "source_image_size": [texture.source_image_width, texture.source_image_height],
            "triangles": material.triangles.len(),
        });
        if material.rule == MaterialRule::SeamSingleCameraProjection {
            provenance["seam_reference_frames"] = json!(material.seam_reference_frames);
            provenance["focal_pixels"] = json!(material.focal_pixels);
            provenance["principal_point"] = json!("image_center");
        }
        images.push(json!({
            "name": name,
            "bufferView": image_view,
            "mimeType": "image/png",
            "extras": { "video_to_3d": {
                "reference_frame": material.reference_frame,
                "rule": material.rule,
                "texture_content_hash": texture.content_hash,
            }},
        }));
        textures.push(json!({ "sampler": 0, "source": images.len() - 1 }));
        materials.push(json!({
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorTexture": { "index": textures.len() - 1 },
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": true,
            "extensions": { "KHR_materials_unlit": {} },
            "extras": { "video_to_3d": provenance },
        }));
        primitives.push(json!({
            "attributes": { "POSITION": position_accessor, "TEXCOORD_0": uv_accessor },
            "indices": index_accessor,
            "material": materials.len() - 1,
        }));
    }

    if !bake.fallback.triangles.is_empty() {
        let mut vertices = PrimitiveVertices::default();
        for triangle in &bake.fallback.triangles {
            let triangle = &triangles[*triangle];
            for point in [triangle.a, triangle.c, triangle.b] {
                vertices.push(point);
            }
        }
        let positions = vertices
            .order
            .iter()
            .flat_map(|index| position(*index))
            .collect::<Vec<_>>();
        let colors = vertices
            .order
            .iter()
            .flat_map(|index| {
                let point = &points[*index];
                [point.r, point.g, point.b].map(srgb_to_linear)
            })
            .collect::<Vec<_>>();
        let position_accessor = gltf.float_accessor(&positions, "VEC3", 3, true);
        let color_accessor = gltf.float_accessor(&colors, "VEC3", 3, false);
        let index_accessor = gltf.index_accessor(&vertices.indices);
        materials.push(json!({
            "name": "vertex-color-fallback",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": true,
            "extensions": { "KHR_materials_unlit": {} },
            "extras": { "video_to_3d": {
                "fallback": "vertex_color",
                "triangles": bake.fallback.triangles.len(),
                "reasons": bake.fallback.reasons,
            }},
        }));
        primitives.push(json!({
            "attributes": { "POSITION": position_accessor, "COLOR_0": color_accessor },
            "indices": index_accessor,
            "material": materials.len() - 1,
        }));
    }

    let mut meshes = vec![json!({ "name": "accepted-surface", "primitives": primitives })];
    let mut nodes = vec![json!({ "name": "accepted-surface", "mesh": 0 })];
    let mut scenes = vec![json!({ "name": "visual", "nodes": [0] })];
    let collision = collider.map(|collider| {
        json!({
            "role": collider.role,
            "schema_version": collider.schema_version,
            "scene": (!collider.boxes.is_empty()).then_some("collision"),
            "boxes": collider.box_count,
            "source_triangles": collider.source_triangles,
            "cell_size": collider.cell_size,
            "max_surface_offset": collider.max_surface_offset,
            "min_confidence": collider.options.min_confidence,
            "excluded": collider.excluded,
        })
    });
    if let Some(collider) = collider.filter(|collider| !collider.boxes.is_empty()) {
        let (positions, indices) = collider_mesh(collider);
        let position_accessor = gltf.float_accessor(&positions, "VEC3", 3, true);
        let index_accessor = gltf.index_accessor(&indices);
        let extras = json!({ "video_to_3d": collision });
        meshes.push(json!({
            "name": "coarse-collider",
            "primitives": [{
                "attributes": { "POSITION": position_accessor },
                "indices": index_accessor,
            }],
            "extras": extras,
        }));
        nodes.push(json!({ "name": "coarse-collider", "mesh": 1, "extras": extras }));
        scenes.push(json!({ "name": "collision", "nodes": [1] }));
    }

    let mut document = json!({
        "asset": {
            "version": "2.0",
            "generator": format!("video-to-3d-core {}", env!("CARGO_PKG_VERSION")),
            "extras": { "video_to_3d": {
                "schema": TEXTURED_SURFACE_SCHEMA,
                "schema_version": SURFACE_MATERIAL_BAKE_SCHEMA_VERSION,
                "provider": evidence.provider,
                "scale": evidence.scale,
                "frame": "reconstruction camera frame (+Y down, +Z forward) rotated 180 degrees about X into glTF (+Y up, -Z forward)",
                "points": points.len(),
                "triangles": triangles.len(),
                "textured_triangles": bake.textured_triangles(),
                "seam_textured_triangles": bake.seam_triangles(),
                "fallback_triangles": bake.fallback.triangles.len(),
                "collision": collision,
            }},
        },
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": scenes,
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": gltf.accessors,
        "bufferViews": gltf.buffer_views,
    });
    if !images.is_empty() {
        document["images"] = json!(images);
        document["textures"] = json!(textures);
        document["samplers"] = json!([{
            "magFilter": LINEAR,
            "minFilter": LINEAR_MIPMAP_LINEAR,
            "wrapS": CLAMP_TO_EDGE,
            "wrapT": CLAMP_TO_EDGE,
        }]);
    }
    let mut bin = gltf.bin;
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    document["buffers"] = json!([{ "byteLength": bin.len() }]);

    let mut json_chunk = serde_json::to_vec(&document).map_err(|error| error.to_string())?;
    while !json_chunk.len().is_multiple_of(4) {
        json_chunk.push(b' ');
    }
    let total = 12 + 8 + json_chunk.len() + 8 + bin.len();
    let total = u32::try_from(total).map_err(|_| "textured GLB exceeds 4 GiB".to_owned())?;
    let mut glb = Vec::with_capacity(total as usize);
    glb.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&total.to_le_bytes());
    glb.extend_from_slice(&(json_chunk.len() as u32).to_le_bytes());
    glb.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    glb.extend_from_slice(&json_chunk);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    glb.extend_from_slice(&bin);
    Ok(glb)
}

/// Box corners and outward counter-clockwise faces in the glTF frame.
fn collider_mesh(collider: &CoarseCollider) -> (Vec<f32>, Vec<u32>) {
    const FACES: [[u32; 4]; 6] = [
        [0, 2, 3, 1], // -Z
        [4, 5, 7, 6], // +Z
        [0, 1, 5, 4], // -Y
        [2, 6, 7, 3], // +Y
        [0, 4, 6, 2], // -X
        [1, 3, 7, 5], // +X
    ];
    let mut positions = Vec::with_capacity(collider.boxes.len() * 24);
    let mut indices = Vec::with_capacity(collider.boxes.len() * 36);
    for collider_box in &collider.boxes {
        // (x, y, z) -> (x, -y, -z) swaps the min/max of Y and Z.
        let low = [
            collider_box.min[0],
            -collider_box.max[1],
            -collider_box.max[2],
        ];
        let high = [
            collider_box.max[0],
            -collider_box.min[1],
            -collider_box.min[2],
        ];
        let base = (positions.len() / 3) as u32;
        for corner in 0..8 {
            positions.push(if corner & 1 == 0 { low[0] } else { high[0] });
            positions.push(if corner & 2 == 0 { low[1] } else { high[1] });
            positions.push(if corner & 4 == 0 { low[2] } else { high[2] });
        }
        for [a, b, c, d] in FACES {
            indices.extend_from_slice(&[
                base + a,
                base + b,
                base + c,
                base + a,
                base + c,
                base + d,
            ]);
        }
    }
    (positions, indices)
}

/// Deterministic 8-bit RGB PNG (alpha dropped; reference frames are opaque).
pub fn encode_png_rgb(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    encode_png_rgb_with_cancel(width, height, rgba, || false)
}

/// Input bytes compressed between cancellation polls.
const PNG_COMPRESSION_CHUNK: usize = 64 * 1024;

/// [`encode_png_rgb`] with cooperative cancellation, polled every row while
/// filtering and every [`PNG_COMPRESSION_CHUNK`] input bytes while
/// compressing. The output is byte-identical to the uncancelable encoder.
pub(crate) fn encode_png_rgb_with_cancel(
    width: u32,
    height: u32,
    rgba: &[u8],
    mut is_canceled: impl FnMut() -> bool,
) -> Result<Vec<u8>, String> {
    let canceled = || "texture encoding was canceled".to_owned();
    if width == 0 || height == 0 || rgba.len() != width as usize * height as usize * 4 {
        return Err(format!("texture pixels do not match {width}x{height} RGBA"));
    }
    let row_bytes = width as usize * 3;
    let mut raw = Vec::with_capacity((row_bytes + 1) * height as usize);
    for row in rgba.chunks_exact(width as usize * 4) {
        if is_canceled() {
            return Err(canceled());
        }
        // Sub filter: each byte minus the same channel of the previous pixel.
        raw.push(1);
        let start = raw.len();
        for pixel in row.as_chunks::<4>().0 {
            raw.extend_from_slice(&pixel[..3]);
        }
        for index in (start + 3..raw.len()).rev() {
            raw[index] = raw[index].wrapping_sub(raw[index - 3]);
        }
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    png_chunk(&mut png, b"IHDR", &header);
    let compressed = compress_zlib_with_cancel(&raw, 6, &mut is_canceled)?.ok_or_else(canceled)?;
    png_chunk(&mut png, b"IDAT", &compressed);
    png_chunk(&mut png, b"IEND", &[]);
    Ok(png)
}

/// zlib compression in bounded input chunks; `Ok(None)` once canceled.
fn compress_zlib_with_cancel(
    input: &[u8],
    level: u8,
    is_canceled: &mut impl FnMut() -> bool,
) -> Result<Option<Vec<u8>>, String> {
    use miniz_oxide::deflate::core::{
        compress, create_comp_flags_from_zip_params, CompressorOxide, TDEFLFlush, TDEFLStatus,
    };
    let mut compressor =
        CompressorOxide::new(create_comp_flags_from_zip_params(level.into(), 1, 0));
    let mut output = Vec::with_capacity(input.len() / 2 + 64);
    let mut buffer = vec![0_u8; PNG_COMPRESSION_CHUNK];
    let mut consumed = 0;
    loop {
        if is_canceled() {
            return Ok(None);
        }
        let end = (consumed + PNG_COMPRESSION_CHUNK).min(input.len());
        let flush = if end == input.len() {
            TDEFLFlush::Finish
        } else {
            TDEFLFlush::None
        };
        let (status, read, written) =
            compress(&mut compressor, &input[consumed..end], &mut buffer, flush);
        consumed += read;
        output.extend_from_slice(&buffer[..written]);
        match status {
            TDEFLStatus::Done => return Ok(Some(output)),
            TDEFLStatus::Okay => {}
            status => return Err(format!("PNG compression failed: {status:?}")),
        }
    }
}

fn png_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let crc = crc32fast::hash(&png[start..]);
    png.extend_from_slice(&crc.to_be_bytes());
}

#[cfg(test)]
mod png_tests {
    use super::*;

    fn pixels(width: u32, height: u32) -> Vec<u8> {
        (0..width * height)
            .flat_map(|index| {
                let (x, y) = (index % width, index / width);
                [(x * 7 + y) as u8, (y * 3) as u8, (x ^ y) as u8, 255]
            })
            .collect()
    }

    #[test]
    fn chunked_compression_matches_the_one_shot_encoder() {
        // More than one compression chunk of filtered rows.
        let (width, height) = (300, 200);
        let rgba = pixels(width, height);
        let png = encode_png_rgb(width, height, &rgba).unwrap();
        let mut raw = Vec::new();
        for row in rgba.chunks_exact(width as usize * 4) {
            raw.push(1);
            let start = raw.len();
            for pixel in row.as_chunks::<4>().0 {
                raw.extend_from_slice(&pixel[..3]);
            }
            for index in (start + 3..raw.len()).rev() {
                raw[index] = raw[index].wrapping_sub(raw[index - 3]);
            }
        }
        assert!(raw.len() > PNG_COMPRESSION_CHUNK);
        let one_shot = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6);
        let idat_length = u32::from_be_bytes(png[33..37].try_into().unwrap()) as usize;
        assert_eq!(&png[37..41], b"IDAT");
        assert_eq!(&png[41..41 + idat_length], one_shot.as_slice());
        assert_eq!(
            miniz_oxide::inflate::decompress_to_vec_zlib(&png[41..41 + idat_length]).unwrap(),
            raw
        );
    }

    #[test]
    fn encoding_stops_when_canceled_while_filtering_or_compressing() {
        let (width, height) = (300, 200);
        let rgba = pixels(width, height);
        for after in [1, height as usize + 1] {
            let mut polls = 0;
            let error = encode_png_rgb_with_cancel(width, height, &rgba, || {
                polls += 1;
                polls > after
            })
            .unwrap_err();
            assert!(error.contains("canceled"), "{error}");
            assert_eq!(polls, after + 1);
        }
    }
}
