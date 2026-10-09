//! On-disk formats of the scene-project artifacts that built-in operations
//! exchange, and the built-in `texture_bake` executor that reads and writes
//! them.
//!
//! The formats are intentionally minimal and versioned so they can change
//! without migration machinery: a reader rejects any other `schema_version`,
//! and the producing operation simply runs again.
//!
//! - **Keyframes** (`ingest_video` → `keyframes`): a JSON index of sampled
//!   frames. Each frame names its source frame index, timestamp, dimensions,
//!   and a sidecar file of raw row-major RGBA8 pixels with its content hash.
//! - **Surface mesh** (`surface_mesh` → `surface_mesh`): one JSON document
//!   holding the owned [`ReconstructionEvidenceView`] (provider, scale,
//!   cameras, provenance regions, points, triangles, optional per-point
//!   attributes) plus the reference-grid site of every dense point.
//! - **Surface textures** (`texture_bake` → `surface_textures`): a JSON
//!   material manifest of the Rust-owned bake ([`bake_surface_materials`]) with
//!   one sidecar PNG (8-bit RGB) per textured reference. Sidecar files are
//!   named by the material's appearance key, so a rebuild rewrites only the
//!   references whose appearance changed.
//!
//! Recorded artifacts and receipts cover the JSON documents. Their sidecar
//! files are covered by the content hashes inside those documents and are
//! verified whenever they are read, so a missing or edited sidecar fails
//! loudly instead of being used.

use crate::scene_model::{Material, ResourceKind, SceneProvenance, SceneResource};
use crate::scene_project::{ArtifactKind, ContentHash, OperationKind, ProjectPath};
use crate::scene_runner::{
    AttemptOutcome, CancellationToken, OperationExecutor, OperationRequest, ProducedArtifact,
    ResolvedInput,
};
use crate::scene_store::Reproducibility;
use crate::surface_materials::{
    bake_surface_materials, AppearanceInvalidation, FallbackReasons, RecordedAppearance,
    ReferenceImage, SURFACE_MATERIAL_BAKE_SCHEMA_VERSION,
};
use crate::textured_glb::encode_png_rgb;
use crate::{
    DenseGridSite, EvidenceCamera, EvidenceCameraAuthority, EvidenceOrigin,
    EvidencePointAttributes, EvidenceScale, MeshTriangle, Point3, ReconstructionEvidenceView,
    ReconstructionProviderDescriptor, SurfaceEvidenceRegion,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const KEYFRAMES_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const SURFACE_MESH_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const SURFACE_TEXTURES_ARTIFACT_SCHEMA_VERSION: u32 = 1;
/// File name of the material manifest inside a texture-bake output directory.
pub const SURFACE_TEXTURES_FILE: &str = "surface-textures.json";

fn parse_versioned<T: DeserializeOwned>(
    document: &str,
    label: &str,
    expected: u32,
) -> Result<T, String> {
    let value: serde_json::Value =
        serde_json::from_str(document).map_err(|error| format!("{label}: {error}"))?;
    match value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
    {
        Some(version) if version == u64::from(expected) => {}
        Some(version) => {
            return Err(format!(
                "{label}: unsupported schema version {version}; this build reads version {expected}"
            ))
        }
        None => return Err(format!("{label}: `schema_version` is missing")),
    }
    serde_json::from_value(value).map_err(|error| format!("{label}: {error}"))
}

fn to_document<T: Serialize>(value: &T) -> String {
    let mut document = serde_json::to_string_pretty(value).expect("artifact documents serialize");
    document.push('\n');
    document
}

/// Directory part of a project path (`None` for a top-level file).
fn parent(path: &ProjectPath) -> Option<&str> {
    path.as_str().rsplit_once('/').map(|(parent, _)| parent)
}

fn sibling(path: &ProjectPath, relative: &str) -> Result<ProjectPath, String> {
    let joined = match parent(path) {
        Some(parent) => format!("{parent}/{relative}"),
        None => relative.to_owned(),
    };
    ProjectPath::new(joined).map_err(|error| error.to_string())
}

/// Write through a temporary sibling and rename, so readers never observe a
/// partially written file.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".partial");
    let temporary = PathBuf::from(temporary);
    fs::write(&temporary, bytes)
        .map_err(|error| format!("cannot write {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

/// Read a sidecar file and check it against its recorded hash.
fn read_verified(root: &Path, path: &ProjectPath, hash: &ContentHash) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path.resolve(root))
        .map_err(|error| format!("cannot read `{}`: {error}", path.as_str()))?;
    if ContentHash::of_bytes(&bytes) != *hash {
        return Err(format!(
            "`{}` does not match its recorded content hash {}",
            path.as_str(),
            hash.as_str()
        ));
    }
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// Keyframes

/// One sampled frame of a keyframes artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyframeRecord {
    /// Index of the sampled frame; evidence cameras and regions refer to it.
    pub frame_index: usize,
    /// Presentation time in the source video.
    pub timestamp_seconds: f64,
    pub width: u32,
    pub height: u32,
    /// Raw row-major RGBA8 pixels, `width * height * 4` bytes.
    pub pixels: ProjectPath,
    pub content_hash: ContentHash,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyframesArtifact {
    pub schema_version: u32,
    /// Sorted by `frame_index`, which is unique.
    pub frames: Vec<KeyframeRecord>,
}

/// A sampled frame handed to [`write_keyframes_artifact`].
#[derive(Clone, Copy, Debug)]
pub struct SampledFrame<'a> {
    pub frame_index: usize,
    pub timestamp_seconds: f64,
    pub width: u32,
    pub height: u32,
    pub rgba: &'a [u8],
}

fn rgba_length(width: u32, height: u32) -> Option<usize> {
    (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(4)
        .filter(|length| *length > 0)
}

impl KeyframesArtifact {
    pub fn from_json(document: &str) -> Result<Self, String> {
        let artifact: Self = parse_versioned(
            document,
            "keyframes artifact",
            KEYFRAMES_ARTIFACT_SCHEMA_VERSION,
        )?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut previous = None;
        for frame in &self.frames {
            if previous.is_some_and(|index| index >= frame.frame_index) {
                return Err(format!(
                    "keyframes artifact: frame {} is duplicated or out of order",
                    frame.frame_index
                ));
            }
            previous = Some(frame.frame_index);
            if !frame.timestamp_seconds.is_finite() || frame.timestamp_seconds < 0.0 {
                return Err(format!(
                    "keyframes artifact: frame {} has an invalid timestamp",
                    frame.frame_index
                ));
            }
            if rgba_length(frame.width, frame.height).is_none() {
                return Err(format!(
                    "keyframes artifact: frame {} has invalid dimensions {}x{}",
                    frame.frame_index, frame.width, frame.height
                ));
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> String {
        to_document(self)
    }

    /// Read and verify the pixels of one frame.
    pub fn load_pixels(&self, root: &Path, frame_index: usize) -> Result<Vec<u8>, String> {
        let frame = self
            .frames
            .iter()
            .find(|frame| frame.frame_index == frame_index)
            .ok_or_else(|| format!("keyframes artifact has no frame {frame_index}"))?;
        let rgba = read_verified(root, &frame.pixels, &frame.content_hash)?;
        if Some(rgba.len()) != rgba_length(frame.width, frame.height) {
            return Err(format!(
                "keyframe {frame_index} pixels are not {}x{} RGBA",
                frame.width, frame.height
            ));
        }
        Ok(rgba)
    }
}

/// Write a keyframes artifact at `index` with one pixel sidecar per frame under
/// `frames/` next to it. Returns the written index and its content hash.
pub fn write_keyframes_artifact(
    root: &Path,
    index: &ProjectPath,
    frames: &[SampledFrame<'_>],
) -> Result<(KeyframesArtifact, ContentHash), String> {
    let mut records = Vec::with_capacity(frames.len());
    for frame in frames {
        if Some(frame.rgba.len()) != rgba_length(frame.width, frame.height) {
            return Err(format!(
                "sampled frame {} must be non-empty RGBA of {}x{} pixels",
                frame.frame_index, frame.width, frame.height
            ));
        }
        let pixels = sibling(
            index,
            &format!("frames/frame-{:06}.rgba", frame.frame_index),
        )?;
        write_atomically(&pixels.resolve(root), frame.rgba)?;
        records.push(KeyframeRecord {
            frame_index: frame.frame_index,
            timestamp_seconds: frame.timestamp_seconds,
            width: frame.width,
            height: frame.height,
            pixels,
            content_hash: ContentHash::of_bytes(frame.rgba),
        });
    }
    records.sort_by_key(|record| record.frame_index);
    let artifact = KeyframesArtifact {
        schema_version: KEYFRAMES_ARTIFACT_SCHEMA_VERSION,
        frames: records,
    };
    artifact.validate()?;
    let document = artifact.to_json();
    write_atomically(&index.resolve(root), document.as_bytes())?;
    Ok((artifact, ContentHash::of_bytes(document.as_bytes())))
}

// ---------------------------------------------------------------------------
// Surface mesh

/// Owned, serialized form of a validated [`ReconstructionEvidenceView`] plus
/// the reference-grid site of every dense point.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceMeshArtifact {
    pub schema_version: u32,
    pub provider: ReconstructionProviderDescriptor,
    pub scale: EvidenceScale,
    pub cameras: Vec<EvidenceCamera>,
    pub regions: Vec<SurfaceEvidenceRegion>,
    pub points: Vec<Point3>,
    pub triangles: Vec<MeshTriangle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_attributes: Option<Vec<EvidencePointAttributes>>,
    /// `grid_sites[i]` is the pixel of dense point `i` in its own reference
    /// image; parallel to `points`.
    pub grid_sites: Vec<DenseGridSite>,
}

impl SurfaceMeshArtifact {
    /// Materialize the evidence for interchange. This is the one place the
    /// borrowed buffers are copied, because the artifact must own them.
    pub fn from_evidence(
        evidence: &ReconstructionEvidenceView<'_>,
        grid_sites: &[DenseGridSite],
    ) -> Result<Self, String> {
        evidence.validate()?;
        let artifact = Self {
            schema_version: SURFACE_MESH_ARTIFACT_SCHEMA_VERSION,
            provider: evidence.provider.clone(),
            scale: evidence.scale,
            cameras: evidence.cameras.clone(),
            regions: evidence.regions.clone(),
            points: evidence.points.to_vec(),
            triangles: evidence.triangles.to_vec(),
            point_attributes: evidence.point_attributes.map(<[_]>::to_vec),
            grid_sites: grid_sites.to_vec(),
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn from_json(document: &str) -> Result<Self, String> {
        let artifact: Self = parse_versioned(
            document,
            "surface mesh artifact",
            SURFACE_MESH_ARTIFACT_SCHEMA_VERSION,
        )?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.evidence()?;
        if self.grid_sites.len() != self.points.len() {
            return Err(format!(
                "surface mesh artifact has {} grid sites for {} points",
                self.grid_sites.len(),
                self.points.len()
            ));
        }
        Ok(())
    }

    /// Borrowed, validated evidence view over the artifact's buffers.
    pub fn evidence(&self) -> Result<ReconstructionEvidenceView<'_>, String> {
        let view = ReconstructionEvidenceView::new(
            self.provider.clone(),
            self.scale,
            self.cameras.clone(),
            self.regions.clone(),
            &self.points,
            &self.triangles,
        )?;
        match &self.point_attributes {
            Some(attributes) => view.with_point_attributes(attributes),
            None => Ok(view),
        }
    }

    pub fn to_json(&self) -> String {
        to_document(self)
    }
}

// ---------------------------------------------------------------------------
// Surface textures

/// PNG sidecar of one baked reference material.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceTextureFile {
    /// 8-bit RGB PNG of the cropped reference pixels.
    pub path: ProjectPath,
    pub content_hash: ContentHash,
    pub width: u32,
    pub height: u32,
    /// Crop origin in the reference image, in pixels.
    pub crop_origin: [u32; 2],
    pub source_image_width: u32,
    pub source_image_height: u32,
}

/// One baked reference material, as recorded in a surface-textures artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceTextureMaterial {
    pub reference_frame: usize,
    pub appearance_key: ContentHash,
    pub camera_authority: EvidenceCameraAuthority,
    /// World-to-camera rotation (row-major) of the reference camera.
    pub camera_rotation: [f32; 9],
    pub camera_translation: [f32; 3],
    /// Reference frame plus every supporting source frame.
    pub source_frames: Vec<usize>,
    pub provenance: Vec<EvidenceOrigin>,
    /// Indices into the surface mesh triangle list, in mesh order.
    pub triangles: Vec<usize>,
    /// Per triangle corner `[a, b, c]`, normalized to the texture with `(0, 0)`
    /// at its top-left corner.
    pub corner_uvs: Vec<[[f32; 2]; 3]>,
    pub texture: SurfaceTextureFile,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceTexturesArtifact {
    pub schema_version: u32,
    /// Schema version of the bake whose output this records.
    pub bake_schema_version: u32,
    pub point_count: usize,
    pub triangle_count: usize,
    /// Sorted by `reference_frame`, which is unique.
    pub materials: Vec<SurfaceTextureMaterial>,
    /// Accepted triangles that keep the untextured vertex-color fallback.
    pub fallback_triangles: Vec<usize>,
    pub fallback_reasons: FallbackReasons,
}

impl SurfaceTexturesArtifact {
    pub fn from_json(document: &str) -> Result<Self, String> {
        let artifact: Self = parse_versioned(
            document,
            "surface textures artifact",
            SURFACE_TEXTURES_ARTIFACT_SCHEMA_VERSION,
        )?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut frames = BTreeSet::new();
        for material in &self.materials {
            if !frames.insert(material.reference_frame) {
                return Err(format!(
                    "surface textures artifact lists reference frame {} twice",
                    material.reference_frame
                ));
            }
            if material.corner_uvs.len() != material.triangles.len() {
                return Err(format!(
                    "surface textures material for frame {} has {} UV triples for {} triangles",
                    material.reference_frame,
                    material.corner_uvs.len(),
                    material.triangles.len()
                ));
            }
            if material.source_frames.is_empty() || material.provenance.is_empty() {
                return Err(format!(
                    "surface textures material for frame {} must cite source frames and provenance",
                    material.reference_frame
                ));
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> String {
        to_document(self)
    }

    pub fn recorded_appearance(&self) -> Vec<RecordedAppearance> {
        self.materials
            .iter()
            .map(|material| RecordedAppearance {
                reference_frame: material.reference_frame,
                appearance_key: material.appearance_key.clone(),
            })
            .collect()
    }

    /// Check every texture sidecar against its recorded hash.
    pub fn verify_textures(&self, root: &Path) -> Result<(), String> {
        for material in &self.materials {
            read_verified(root, &material.texture.path, &material.texture.content_hash)?;
        }
        Ok(())
    }

    /// Assembled-scene entries for the baked materials: one camera-backed
    /// `Texture` resource and one `Material` per reference.
    ///
    /// The texture resource carries the observed provenance and supporting
    /// source frames of the geometry it was baked for. The material factors
    /// are neutral constants (white, non-metallic, fully rough) and are marked
    /// `Estimated`, since material factors carry no evidence links of their
    /// own. The scene must declare an accepted camera for every cited frame.
    pub fn scene_entries(&self) -> (Vec<SceneResource>, Vec<Material>) {
        let mut resources = Vec::with_capacity(self.materials.len());
        let mut materials = Vec::with_capacity(self.materials.len());
        for material in &self.materials {
            let texture_id = surface_texture_id(material.reference_frame);
            let mut provenance: Vec<SceneProvenance> = material
                .provenance
                .iter()
                .map(|origin| SceneProvenance::from(*origin))
                .collect();
            provenance.sort();
            provenance.dedup();
            resources.push(SceneResource {
                id: texture_id.clone(),
                kind: ResourceKind::Texture,
                path: material.texture.path.clone(),
                content_hash: material.texture.content_hash.clone(),
                provenance,
                confidence: None,
                source_frames: material.source_frames.clone(),
            });
            materials.push(Material {
                id: surface_material_id(material.reference_frame),
                base_color: [1.0; 4],
                base_color_texture: Some(texture_id),
                metallic: 0.0,
                roughness: 1.0,
                provenance: SceneProvenance::Estimated,
            });
        }
        (resources, materials)
    }

    pub fn diagnostic(&self) -> String {
        let textured: usize = self
            .materials
            .iter()
            .map(|material| material.triangles.len())
            .sum();
        format!(
            "Surface textures v{}: {} reference texture(s) cover {textured} of {} accepted triangles; {} keep the vertex-color fallback.",
            self.schema_version,
            self.materials.len(),
            self.triangle_count,
            self.fallback_triangles.len(),
        )
    }
}

/// Scene resource id of the baked texture of one reference frame.
pub fn surface_texture_id(reference_frame: usize) -> String {
    format!("surface-texture-f{reference_frame}")
}

/// Scene material id of the baked material of one reference frame.
pub fn surface_material_id(reference_frame: usize) -> String {
    format!("surface-material-f{reference_frame}")
}

/// What a texture bake wrote.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureBakeOutcome {
    pub artifact: SurfaceTexturesArtifact,
    pub content_hash: ContentHash,
    /// Bake diagnostic (textured/fallback counts).
    pub diagnostic: String,
    /// Appearance comparison against the previous artifact at the same path.
    pub invalidation: AppearanceInvalidation,
    /// Reference frames whose texture sidecar was (re)written.
    pub written: Vec<usize>,
    /// Reference frames whose existing texture sidecar was kept.
    pub reused: Vec<usize>,
}

/// Bake surface textures from a keyframes and a surface mesh artifact and
/// write the surface-textures artifact at `output`.
///
/// A texture sidecar is rewritten only when its reference's appearance key
/// changed (or the previous sidecar no longer verifies); sidecars of dropped
/// or invalidated references are removed after the new manifest is written.
pub fn bake_texture_artifact(
    root: &Path,
    output: &ProjectPath,
    keyframes: &ProjectPath,
    surface_mesh: &ProjectPath,
    cancel: &CancellationToken,
) -> Result<TextureBakeOutcome, String> {
    let read = |path: &ProjectPath| {
        fs::read_to_string(path.resolve(root))
            .map_err(|error| format!("cannot read `{}`: {error}", path.as_str()))
    };
    let keyframes = KeyframesArtifact::from_json(&read(keyframes)?)?;
    let mesh = SurfaceMeshArtifact::from_json(&read(surface_mesh)?)?;
    let evidence = mesh.evidence()?;

    // Only reference frames of observed regions can be textured.
    let references: BTreeSet<usize> = evidence
        .regions
        .iter()
        .filter(|region| {
            matches!(
                region.origin,
                EvidenceOrigin::GeometricMultiView | EvidenceOrigin::RevalidatedCompletion
            )
        })
        .filter_map(|region| region.reference_frame)
        .collect();
    let mut pixels = Vec::new();
    for frame in &keyframes.frames {
        if references.contains(&frame.frame_index) {
            pixels.push((frame, keyframes.load_pixels(root, frame.frame_index)?));
        }
    }
    let images: Vec<ReferenceImage<'_>> = pixels
        .iter()
        .map(|(frame, rgba)| ReferenceImage {
            frame_index: frame.frame_index,
            width: frame.width,
            height: frame.height,
            rgba,
        })
        .collect();
    let bake = bake_surface_materials(&evidence, &mesh.grid_sites, &images)?;

    let previous = fs::read_to_string(output.resolve(root))
        .ok()
        .and_then(|document| SurfaceTexturesArtifact::from_json(&document).ok());
    let invalidation = bake.invalidation_against(
        &previous
            .as_ref()
            .map(SurfaceTexturesArtifact::recorded_appearance)
            .unwrap_or_default(),
    );
    let previous_files: BTreeMap<&ContentHash, &SurfaceTextureFile> = previous
        .iter()
        .flat_map(|artifact| &artifact.materials)
        .map(|material| (&material.appearance_key, &material.texture))
        .collect();

    let mut materials = Vec::with_capacity(bake.materials.len());
    let (mut written, mut reused) = (Vec::new(), Vec::new());
    for material in &bake.materials {
        if cancel.is_canceled() {
            return Err("texture bake was canceled".into());
        }
        let digest = &material.appearance_key.as_str()["sha256:".len()..];
        let path = sibling(output, &format!("textures/{digest}.png"))?;
        let kept = previous_files.get(&material.appearance_key).filter(|file| {
            file.path == path && read_verified(root, &file.path, &file.content_hash).is_ok()
        });
        let content_hash = match kept {
            Some(file) => {
                reused.push(material.reference_frame);
                file.content_hash.clone()
            }
            None => {
                let png = encode_png_rgb(
                    material.texture.width,
                    material.texture.height,
                    &material.texture.rgba,
                )?;
                write_atomically(&path.resolve(root), &png)?;
                written.push(material.reference_frame);
                ContentHash::of_bytes(&png)
            }
        };
        materials.push(SurfaceTextureMaterial {
            reference_frame: material.reference_frame,
            appearance_key: material.appearance_key.clone(),
            camera_authority: material.camera_authority,
            camera_rotation: material.camera_rotation,
            camera_translation: material.camera_translation,
            source_frames: material.source_frames.clone(),
            provenance: material.provenance.clone(),
            triangles: material.triangles.clone(),
            corner_uvs: material.corner_uvs.clone(),
            texture: SurfaceTextureFile {
                path,
                content_hash,
                width: material.texture.width,
                height: material.texture.height,
                crop_origin: material.texture.crop_origin,
                source_image_width: material.texture.source_image_width,
                source_image_height: material.texture.source_image_height,
            },
        });
    }
    let artifact = SurfaceTexturesArtifact {
        schema_version: SURFACE_TEXTURES_ARTIFACT_SCHEMA_VERSION,
        bake_schema_version: SURFACE_MATERIAL_BAKE_SCHEMA_VERSION,
        point_count: bake.point_count,
        triangle_count: bake.triangle_count,
        materials,
        fallback_triangles: bake.fallback.triangles.clone(),
        fallback_reasons: bake.fallback.reasons.clone(),
    };
    artifact.validate()?;
    let document = artifact.to_json();
    write_atomically(&output.resolve(root), document.as_bytes())?;

    // Drop sidecars the new manifest no longer references. Only files inside
    // this operation's own texture directory are ever removed.
    let texture_directory = format!("{}/", sibling(output, "textures")?.as_str());
    let current: BTreeSet<&ProjectPath> = artifact
        .materials
        .iter()
        .map(|material| &material.texture.path)
        .collect();
    for file in previous_files.values() {
        if !current.contains(&file.path) && file.path.as_str().starts_with(&texture_directory) {
            let _ = fs::remove_file(file.path.resolve(root));
        }
    }

    Ok(TextureBakeOutcome {
        content_hash: ContentHash::of_bytes(document.as_bytes()),
        diagnostic: bake.diagnostic(),
        invalidation,
        written,
        reused,
        artifact,
    })
}

/// Project-relative output path of a texture-bake operation.
pub fn texture_bake_output(operation_id: &str) -> Result<ProjectPath, String> {
    ProjectPath::new(format!("artifacts/{operation_id}/{SURFACE_TEXTURES_FILE}"))
        .map_err(|error| error.to_string())
}

/// Native executor for the built-in operations that have an implementation.
/// Today that is `texture_bake`; every other kind reports an explicit
/// unsupported outcome instead of fabricating output (video decoding is
/// browser-owned and provider adapters are not integrated).
#[derive(Clone, Debug)]
pub struct BuiltInExecutor {
    root: PathBuf,
}

impl BuiltInExecutor {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            root: project_root.into(),
        }
    }

    fn texture_bake(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> AttemptOutcome {
        let input = |kind: ArtifactKind| {
            request.inputs.iter().find_map(|input| match input {
                ResolvedInput::Artifact(artifact) if artifact.kind == kind => {
                    Some(artifact.path.clone())
                }
                _ => None,
            })
        };
        let (Some(keyframes), Some(mesh)) = (
            input(ArtifactKind::Keyframes),
            input(ArtifactKind::SurfaceMesh),
        ) else {
            return AttemptOutcome::Failed(
                "texture_bake needs a keyframes and a surface_mesh input".into(),
            );
        };
        let output = match texture_bake_output(&request.operation.id) {
            Ok(output) => output,
            Err(error) => return AttemptOutcome::Failed(error),
        };
        match bake_texture_artifact(&self.root, &output, &keyframes, &mesh, cancel) {
            Ok(outcome) => {
                let list = |frames: &[usize]| {
                    frames
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                };
                let mut produced = ProducedArtifact::new(
                    output,
                    outcome.content_hash,
                    Reproducibility::Deterministic,
                );
                let observations = &mut produced.observations;
                observations.insert("diagnostic".into(), outcome.diagnostic);
                observations.insert("appearance".into(), outcome.invalidation.diagnostic());
                observations.insert("textures_written".into(), list(&outcome.written));
                observations.insert("textures_reused".into(), list(&outcome.reused));
                AttemptOutcome::Succeeded(produced)
            }
            Err(_) if cancel.is_canceled() => AttemptOutcome::Canceled,
            Err(error) => AttemptOutcome::Failed(error),
        }
    }
}

impl OperationExecutor for BuiltInExecutor {
    fn execute(&self, request: &OperationRequest, cancel: &CancellationToken) -> AttemptOutcome {
        let kind = request.operation.kind;
        if kind == OperationKind::TextureBake && request.provider.is_none() {
            return self.texture_bake(request, cancel);
        }
        let name = serde_json::to_value(kind)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_else(|| "operation".into());
        AttemptOutcome::Unsupported(match &request.provider {
            Some(provider) => format!(
                "provider `{}` for {name} has no native adapter in this build",
                provider.id
            ),
            None => format!("{name} has no native executor in this build"),
        })
    }
}

#[cfg(test)]
#[path = "scene_artifacts_tests.rs"]
mod tests;
