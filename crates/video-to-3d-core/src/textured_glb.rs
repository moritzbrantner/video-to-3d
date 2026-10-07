//! Self-contained binary glTF 2.0 export of an accepted textured surface.
//!
//! The GLB embeds every baked reference texture as a PNG and needs no browser
//! state to load. Each reference material becomes one primitive with
//! `TEXCOORD_0`; the untextured fallback becomes one primitive with vertex
//! colors. Positions are exported exactly as accepted, only re-expressed in the
//! glTF frame. Material, image, and asset `extras.video_to_3d` carry the
//! reference camera, provenance, appearance key, and fallback reasons.
//!
//! Frame: reconstruction coordinates follow the camera convention of the seed
//! camera (+X right, +Y down, +Z forward). glTF is +Y up with cameras looking
//! along -Z, so positions are rotated 180 degrees about X: `(x, -y, -z)`. The
//! rotation is proper, so it preserves handedness; accepted triangles have
//! positive signed area in their (y-down) reference image, i.e. they are
//! clockwise as seen from the reference camera, so corners are emitted as
//! `(a, c, b)` to face that camera under the glTF counter-clockwise rule.

use crate::surface_materials::{SurfaceMaterialBake, SURFACE_MATERIAL_BAKE_SCHEMA_VERSION};
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
        let provenance = json!({
            "reference_frame": material.reference_frame,
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
        images.push(json!({
            "name": format!("reference-frame-{}", material.reference_frame),
            "bufferView": image_view,
            "mimeType": "image/png",
            "extras": { "video_to_3d": {
                "reference_frame": material.reference_frame,
                "texture_content_hash": texture.content_hash,
            }},
        }));
        textures.push(json!({ "sampler": 0, "source": images.len() - 1 }));
        materials.push(json!({
            "name": format!("reference-frame-{}", material.reference_frame),
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
                "fallback_triangles": bake.fallback.triangles.len(),
            }},
        },
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "name": "accepted-surface", "mesh": 0 }],
        "meshes": [{ "name": "accepted-surface", "primitives": primitives }],
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

/// Deterministic 8-bit RGB PNG (alpha dropped; reference frames are opaque).
pub fn encode_png_rgb(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 || rgba.len() != width as usize * height as usize * 4 {
        return Err(format!("texture pixels do not match {width}x{height} RGBA"));
    }
    let row_bytes = width as usize * 3;
    let mut raw = Vec::with_capacity((row_bytes + 1) * height as usize);
    for row in rgba.chunks_exact(width as usize * 4) {
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
    png_chunk(
        &mut png,
        b"IDAT",
        &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6),
    );
    png_chunk(&mut png, b"IEND", &[]);
    Ok(png)
}

fn png_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let crc = crc32fast::hash(&png[start..]);
    png.extend_from_slice(&crc.to_be_bytes());
}
