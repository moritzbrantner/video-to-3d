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
//! files are covered by the content hashes inside those documents: build
//! reconciliation checks them (a missing or edited sidecar invalidates the
//! artifact, so it is rebuilt) and readers verify them again on use.

use crate::scene_model::{Material, ResourceKind, SceneProvenance, SceneResource};
use crate::scene_project::{
    ArtifactKind, ContentHash, OperationKind, ProjectPath, SceneProjectManifest,
};
use crate::scene_runner::{
    AttemptOutcome, CancellationToken, OperationExecutor, OperationRequest, ProducedArtifact,
    ResolvedInput,
};
use crate::scene_store::Reproducibility;
use crate::surface_materials::{
    bake_surface_materials_with_cancel, AppearanceInvalidation, FallbackReasons,
    RecordedAppearance, ReferenceImage, SURFACE_MATERIAL_BAKE_SCHEMA_VERSION,
};
use crate::textured_glb::encode_png_rgb_with_cancel;
use crate::{
    DenseGridSite, EvidenceCamera, EvidenceCameraAuthority, EvidenceOrigin,
    EvidencePointAttributes, EvidenceScale, MeshTriangle, Point3, ReconstructionEvidenceView,
    ReconstructionProviderDescriptor, SurfaceEvidenceRegion,
    RECONSTRUCTION_EVIDENCE_SCHEMA_VERSION,
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
/// Identifier limit of the assembled scene model.
const MAX_SCENE_IDENTIFIER_LENGTH: usize = 64;

fn parse_versioned<T: DeserializeOwned>(
    document: &str,
    label: &str,
    expected: u32,
) -> Result<T, String> {
    // Only the version is read first; every other field is skipped without
    // building a generic JSON tree, so a dense mesh is materialized once.
    #[derive(Deserialize)]
    struct Version {
        #[serde(default)]
        schema_version: Option<serde_json::Value>,
    }
    let header: Version =
        serde_json::from_str(document).map_err(|error| format!("{label}: {error}"))?;
    match header
        .schema_version
        .as_ref()
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
    serde_json::from_str(document).map_err(|error| format!("{label}: {error}"))
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
    // Validate the complete replacement first. Sidecars are content-addressed,
    // so writing them never changes a file an existing index refers to.
    let mut records = Vec::with_capacity(frames.len());
    for frame in frames {
        if Some(frame.rgba.len()) != rgba_length(frame.width, frame.height) {
            return Err(format!(
                "sampled frame {} must be non-empty RGBA of {}x{} pixels",
                frame.frame_index, frame.width, frame.height
            ));
        }
        let content_hash = ContentHash::of_bytes(frame.rgba);
        let digest = &content_hash.as_str()["sha256:".len()..];
        let pixels = sibling(index, &format!("frames/{digest}.rgba"))?;
        records.push(KeyframeRecord {
            frame_index: frame.frame_index,
            timestamp_seconds: frame.timestamp_seconds,
            width: frame.width,
            height: frame.height,
            pixels,
            content_hash,
        });
    }
    records.sort_by_key(|record| record.frame_index);
    let artifact = KeyframesArtifact {
        schema_version: KEYFRAMES_ARTIFACT_SCHEMA_VERSION,
        frames: records,
    };
    artifact.validate()?;
    for (record, frame) in artifact.frames.iter().zip({
        let mut sorted: Vec<&SampledFrame<'_>> = frames.iter().collect();
        sorted.sort_by_key(|frame| frame.frame_index);
        sorted
    }) {
        let target = record.pixels.resolve(root);
        if read_verified(root, &record.pixels, &record.content_hash).is_err() {
            write_atomically(&target, frame.rgba)?;
        }
    }
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
        Self::from_json_with_cancel(document, || Ok(()))
    }

    /// Parse and validate with cooperative polling during the evidence
    /// traversal, so a large artifact can be canceled while it is checked.
    pub(crate) fn from_json_with_cancel(
        document: &str,
        poll: impl FnMut() -> Result<(), String>,
    ) -> Result<Self, String> {
        let artifact: Self = parse_versioned(
            document,
            "surface mesh artifact",
            SURFACE_MESH_ARTIFACT_SCHEMA_VERSION,
        )?;
        artifact.validate_with_cancel(poll)?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.validate_with_cancel(|| Ok(()))
    }

    fn validate_with_cancel(&self, poll: impl FnMut() -> Result<(), String>) -> Result<(), String> {
        self.evidence_view().validate_with_cancel(poll)?;
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
        let view = self.evidence_view();
        view.validate()?;
        Ok(view)
    }

    /// The evidence view without validating it; callers validate it once
    /// (cancel-aware where the caller is cancelable).
    fn evidence_view(&self) -> ReconstructionEvidenceView<'_> {
        ReconstructionEvidenceView {
            schema_version: RECONSTRUCTION_EVIDENCE_SCHEMA_VERSION,
            provider: self.provider.clone(),
            scale: self.scale,
            cameras: self.cameras.clone(),
            regions: self.regions.clone(),
            points: &self.points,
            triangles: &self.triangles,
            point_attributes: self.point_attributes.as_deref(),
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
        if self.bake_schema_version != SURFACE_MATERIAL_BAKE_SCHEMA_VERSION {
            return Err(format!(
                "surface textures artifact: unsupported bake schema version {} (expected {})",
                self.bake_schema_version, SURFACE_MATERIAL_BAKE_SCHEMA_VERSION
            ));
        }
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
            let mut sources = BTreeSet::new();
            for &source in &material.source_frames {
                if !sources.insert(source) {
                    return Err(format!(
                        "surface textures material for frame {} cites source frame {source} more than once",
                        material.reference_frame
                    ));
                }
            }
            if !sources.contains(&material.reference_frame) {
                return Err(format!(
                    "surface textures material for frame {} must cite its reference frame",
                    material.reference_frame
                ));
            }
            let mut origins = Vec::new();
            for &origin in &material.provenance {
                if !matches!(
                    origin,
                    EvidenceOrigin::GeometricMultiView | EvidenceOrigin::RevalidatedCompletion
                ) {
                    return Err(format!(
                        "surface textures material for frame {} cannot claim non-observed provenance {origin:?}",
                        material.reference_frame
                    ));
                }
                if origins.contains(&origin) {
                    return Err(format!(
                        "surface textures material for frame {} repeats provenance {origin:?}",
                        material.reference_frame
                    ));
                }
                origins.push(origin);
            }
        }
        // These counts are deserialized from untrusted JSON: do not overflow
        // before checking the claimed partition.
        let reasons = [
            self.fallback_reasons.mixed_reference,
            self.fallback_reasons.unobserved_reference,
            self.fallback_reasons.missing_reference_image,
            self.fallback_reasons.missing_grid_site,
            self.fallback_reasons.degenerate_footprint,
        ]
        .into_iter()
        .try_fold(0usize, |sum, count| sum.checked_add(count))
        .ok_or_else(|| "surface textures fallback reason count overflows".to_owned())?;
        if reasons != self.fallback_triangles.len() {
            return Err(format!(
                "surface textures artifact has {reasons} fallback reason(s) for {} fallback triangle(s)",
                self.fallback_triangles.len()
            ));
        }
        // Textured and fallback triangles partition the accepted triangles.
        // Check the count first so an untrusted `triangle_count` never sizes
        // an allocation beyond the listed indices.
        let listed_count = self
            .materials
            .iter()
            .map(|material| material.triangles.len())
            .sum::<usize>()
            + self.fallback_triangles.len();
        if listed_count != self.triangle_count {
            return Err(format!(
                "surface textures artifact assigns {listed_count} triangle(s) for {} accepted triangles",
                self.triangle_count
            ));
        }
        let mut seen = vec![false; self.triangle_count];
        let listed = self
            .materials
            .iter()
            .flat_map(|material| &material.triangles)
            .chain(&self.fallback_triangles);
        for &triangle in listed {
            match seen.get_mut(triangle) {
                Some(slot) if !*slot => *slot = true,
                Some(_) => {
                    return Err(format!(
                        "surface textures artifact lists triangle {triangle} more than once"
                    ))
                }
                None => {
                    return Err(format!(
                        "surface textures artifact names triangle {triangle} of {}",
                        self.triangle_count
                    ))
                }
            }
        }
        if let Some(missing) = seen.iter().position(|seen| !seen) {
            return Err(format!(
                "surface textures artifact does not assign triangle {missing}"
            ));
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
    /// `namespace` (for example the producing operation id) prefixes every
    /// generated identifier, so entries of several bakes can share one scene.
    pub fn scene_entries(
        &self,
        namespace: &str,
    ) -> Result<(Vec<SceneResource>, Vec<Material>), String> {
        self.validate()?;
        // Scene identifiers are at most 64 bytes; reject a namespace that
        // cannot carry the longest generated suffix.
        if let Some(longest) = self
            .materials
            .iter()
            .map(|material| {
                surface_material_id(namespace, material.reference_frame)
                    .len()
                    .max(surface_texture_id(namespace, material.reference_frame).len())
            })
            .max()
        {
            if longest > MAX_SCENE_IDENTIFIER_LENGTH {
                return Err(format!(
                    "scene entry namespace `{namespace}` makes identifiers of {longest} bytes (limit {MAX_SCENE_IDENTIFIER_LENGTH})"
                ));
            }
        }
        let mut resources = Vec::with_capacity(self.materials.len());
        let mut materials = Vec::with_capacity(self.materials.len());
        for material in &self.materials {
            let texture_id = surface_texture_id(namespace, material.reference_frame);
            crate::scene_model::validate_identifier(&texture_id)
                .map_err(|error| error.to_string())?;
            crate::scene_model::validate_identifier(&surface_material_id(
                namespace,
                material.reference_frame,
            ))
            .map_err(|error| error.to_string())?;
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
                id: surface_material_id(namespace, material.reference_frame),
                base_color: [1.0; 4],
                base_color_texture: Some(texture_id),
                metallic: 0.0,
                roughness: 1.0,
                provenance: SceneProvenance::Estimated,
            });
        }
        Ok((resources, materials))
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

/// One sidecar file a recorded artifact document refers to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactSidecar {
    pub path: ProjectPath,
    pub content_hash: ContentHash,
    /// Exact byte length the document's format requires (RGBA keyframes),
    /// when it fixes one.
    pub byte_length: Option<u64>,
}

/// Sidecar files a recorded artifact document refers to, with their hashes.
/// Reconciliation verifies them next to the document itself; artifact kinds
/// without sidecars return an empty list.
pub fn artifact_sidecars(
    kind: ArtifactKind,
    document: &str,
) -> Result<Vec<ArtifactSidecar>, String> {
    Ok(match kind {
        ArtifactKind::Keyframes => KeyframesArtifact::from_json(document)?
            .frames
            .into_iter()
            .map(|frame| ArtifactSidecar {
                byte_length: rgba_length(frame.width, frame.height).map(|length| length as u64),
                path: frame.pixels,
                content_hash: frame.content_hash,
            })
            .collect(),
        ArtifactKind::SurfaceTextures => SurfaceTexturesArtifact::from_json(document)?
            .materials
            .into_iter()
            .map(|material| ArtifactSidecar {
                path: material.texture.path,
                content_hash: material.texture.content_hash,
                byte_length: None,
            })
            .collect(),
        _ => Vec::new(),
    })
}

/// Scene resource id of the baked texture of one reference frame.
pub fn surface_texture_id(namespace: &str, reference_frame: usize) -> String {
    format!("{namespace}.texture-f{reference_frame}")
}

/// Scene material id of the baked material of one reference frame.
pub fn surface_material_id(namespace: &str, reference_frame: usize) -> String {
    format!("{namespace}.material-f{reference_frame}")
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
    bake_classified(root, output, keyframes, surface_mesh, cancel).map_err(
        |failure| match failure {
            BakeFailure::IncompatibleInput(message) | BakeFailure::Other(message) => message,
        },
    )
}

/// Why a texture bake did not complete.
enum BakeFailure {
    /// An input is not in the versioned format the bake reads (for example an
    /// opaque or legacy output of an earlier producer). Nothing was baked.
    IncompatibleInput(String),
    Other(String),
}

fn bake_classified(
    root: &Path,
    output: &ProjectPath,
    keyframes: &ProjectPath,
    surface_mesh: &ProjectPath,
    cancel: &CancellationToken,
) -> Result<TextureBakeOutcome, BakeFailure> {
    // Inputs that are not the versioned documents the bake reads.
    let incompatible = |path: &ProjectPath, label: &str, error: String| {
        if cancel.is_canceled() {
            BakeFailure::Other("texture bake was canceled".into())
        } else {
            BakeFailure::IncompatibleInput(format!(
                "`{}` is not a versioned {label} artifact ({error}); rebuild it with a producer of that format",
                path.as_str()
            ))
        }
    };
    let read = |path: &ProjectPath, label: &str| -> Result<String, BakeFailure> {
        let bytes = fs::read(path.resolve(root)).map_err(|error| {
            BakeFailure::Other(format!("cannot read `{}`: {error}", path.as_str()))
        })?;
        String::from_utf8(bytes)
            .map_err(|_| incompatible(path, label, "the document is not UTF-8".into()))
    };
    let keyframes_artifact = KeyframesArtifact::from_json(&read(keyframes, "keyframes")?)
        .map_err(|error| incompatible(keyframes, "keyframes", error))?;
    let mesh_document = read(surface_mesh, "surface mesh")?;
    let check = || {
        if cancel.is_canceled() {
            Err("texture bake was canceled".to_owned())
        } else {
            Ok(())
        }
    };
    check().map_err(BakeFailure::Other)?;
    let mesh = SurfaceMeshArtifact::from_json_with_cancel(&mesh_document, check)
        .map_err(|error| incompatible(surface_mesh, "surface mesh", error))?;
    drop(mesh_document);
    bake_parsed(root, output, &keyframes_artifact, &mesh, cancel).map_err(BakeFailure::Other)
}

fn bake_parsed(
    root: &Path,
    output: &ProjectPath,
    keyframes: &KeyframesArtifact,
    mesh: &SurfaceMeshArtifact,
    cancel: &CancellationToken,
) -> Result<TextureBakeOutcome, String> {
    let check_canceled = || {
        if cancel.is_canceled() {
            Err("texture bake was canceled".to_owned())
        } else {
            Ok(())
        }
    };
    check_canceled()?;
    // The mesh was validated once while parsing, cancel-aware; the bake below
    // re-checks the same view with the same polling before it relies on it.
    let evidence = mesh.evidence_view();

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
        check_canceled()?;
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
    check_canceled()?;
    let bake = bake_surface_materials_with_cancel(&evidence, &mesh.grid_sites, &images, || {
        cancel.is_canceled()
    })?;
    check_canceled()?;

    let previous = fs::read_to_string(output.resolve(root))
        .ok()
        .and_then(|document| SurfaceTexturesArtifact::from_json(&document).ok());
    let mut invalidation = bake.invalidation_against(
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
        check_canceled()?;
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
                let png = encode_png_rgb_with_cancel(
                    material.texture.width,
                    material.texture.height,
                    &material.texture.rgba,
                    || cancel.is_canceled(),
                )
                .map_err(|error| {
                    if cancel.is_canceled() {
                        "texture bake was canceled".to_owned()
                    } else {
                        error
                    }
                })?;
                check_canceled()?;
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
    // A reference whose appearance key is unchanged but whose recorded
    // sidecar no longer verified was repaired, not reused.
    let repaired: Vec<usize> = invalidation
        .reused
        .iter()
        .copied()
        .filter(|frame| written.contains(frame))
        .collect();
    if !repaired.is_empty() {
        invalidation
            .reused
            .retain(|frame| !repaired.contains(frame));
        invalidation.invalidated.extend(repaired);
        invalidation.invalidated.sort_unstable();
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
    check_canceled()?;
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
    reservations: Reservations,
}

#[derive(Clone, Debug)]
enum Reservations {
    Fixed(Vec<(String, ProjectPath)>),
    /// Re-read at execution, so the reconciled manifest a build persisted
    /// before running decides (dropped stale records do not block an output).
    ManifestFile(PathBuf),
}

/// Declared paths and their owners (media inputs, exports, artifacts
/// recorded by operations and the sidecars those artifacts list), which an
/// output of another owner must never overwrite.
fn reserved_paths(root: &Path, manifest: &SceneProjectManifest) -> Vec<(String, ProjectPath)> {
    let mut reserved: Vec<(String, ProjectPath)> = manifest
        .inputs
        .iter()
        .map(|input| (String::new(), input.path.clone()))
        .collect();
    reserved.extend(
        manifest
            .exports
            .iter()
            .map(|export| (String::new(), export.path.clone())),
    );
    reserved.extend(
        manifest
            .artifacts
            .iter()
            .map(|artifact| (artifact.produced_by.clone(), artifact.path.clone())),
    );
    // Sidecar paths are unconstrained, so a recorded artifact may own files
    // anywhere in the project, including another bake's directory. An
    // unreadable or opaque document lists none (reconciliation judges it).
    for artifact in &manifest.artifacts {
        reserved.extend(
            recorded_sidecar_paths(root, artifact)
                .into_iter()
                .map(|path| (artifact.produced_by.clone(), path)),
        );
    }
    reserved
}

/// Sidecar paths a recorded artifact's document lists. An unreadable,
/// opaque (non-UTF-8) or unparsable document lists none; reconciliation
/// judges its validity separately.
pub(crate) fn recorded_sidecar_paths(
    root: &Path,
    artifact: &crate::scene_project::ArtifactRecord,
) -> Vec<ProjectPath> {
    if !matches!(
        artifact.kind,
        ArtifactKind::Keyframes | ArtifactKind::SurfaceTextures
    ) {
        return Vec::new();
    }
    let Ok(bytes) = fs::read(artifact.path.resolve(root)) else {
        return Vec::new();
    };
    let Ok(document) = std::str::from_utf8(&bytes) else {
        return Vec::new();
    };
    artifact_sidecars(artifact.kind, document)
        .map(|sidecars| sidecars.into_iter().map(|sidecar| sidecar.path).collect())
        .unwrap_or_default()
}

impl BuiltInExecutor {
    /// Reservations from an in-memory manifest.
    pub fn new(project_root: impl Into<PathBuf>, manifest: &SceneProjectManifest) -> Self {
        let root = project_root.into();
        let reservations = Reservations::Fixed(reserved_paths(&root, manifest));
        Self { root, reservations }
    }

    /// Reservations re-read from the project manifest at each execution; use
    /// this with [`crate::scene_store::ProjectStore::build`], which reconciles
    /// and saves the manifest before running.
    pub fn for_project(
        project_root: impl Into<PathBuf>,
        manifest_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            root: project_root.into(),
            reservations: Reservations::ManifestFile(manifest_path.into()),
        }
    }

    /// A declared path of another owner inside `directory` (or the directory
    /// itself, or one of its ancestors).
    fn collision(&self, operation: &str, directory: &str) -> Result<Option<ProjectPath>, String> {
        let loaded;
        let reserved = match &self.reservations {
            Reservations::Fixed(reserved) => reserved,
            Reservations::ManifestFile(path) => {
                loaded = reserved_paths(
                    &self.root,
                    &SceneProjectManifest::load(path).map_err(|error| error.to_string())?,
                );
                &loaded
            }
        };
        // ProjectPath manifest uniqueness folds ASCII case. Match it here,
        // including ancestor/descendant collisions, before any sidecar write.
        let folded_directory = directory.to_ascii_lowercase();
        Ok(reserved
            .iter()
            .filter(|(owner, _)| owner != operation)
            .map(|(_, path)| path)
            .find(|path| {
                let folded_path = path.as_str().to_ascii_lowercase();
                folded_path == folded_directory
                    || folded_path.starts_with(&format!("{folded_directory}/"))
                    || folded_directory.starts_with(&format!("{folded_path}/"))
            })
            .cloned())
    }

    fn texture_bake(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> AttemptOutcome {
        // Exactly one input of each kind: every input is part of the
        // operation identity, so an ignored extra input is not allowed.
        let inputs = |kind: ArtifactKind| -> Vec<ProjectPath> {
            request
                .inputs
                .iter()
                .filter_map(|input| match input {
                    ResolvedInput::Artifact(artifact) if artifact.kind == kind => {
                        Some(artifact.path.clone())
                    }
                    _ => None,
                })
                .collect()
        };
        let (keyframes, mesh) = (
            inputs(ArtifactKind::Keyframes),
            inputs(ArtifactKind::SurfaceMesh),
        );
        let ([keyframes], [mesh]) = (keyframes.as_slice(), mesh.as_slice()) else {
            return AttemptOutcome::Failed(format!(
                "texture_bake needs exactly one keyframes and one surface_mesh input (got {} and {})",
                keyframes.len(),
                mesh.len()
            ));
        };
        if request.inputs.len() != 2 {
            return AttemptOutcome::Failed(
                "texture_bake takes only a keyframes and a surface_mesh input".into(),
            );
        }
        let output = match texture_bake_output(&request.operation.id) {
            Ok(output) => output,
            Err(error) => return AttemptOutcome::Failed(error),
        };
        let directory = parent(&output).unwrap_or_default();
        let collision = match self.collision(&request.operation.id, directory) {
            Ok(collision) => collision,
            Err(error) => return AttemptOutcome::Failed(error),
        };
        if let Some(path) = collision {
            return AttemptOutcome::Failed(format!(
                "texture_bake output directory `{directory}` would overwrite the declared path `{}`",
                path.as_str()
            ));
        }
        match bake_classified(&self.root, &output, keyframes, mesh, cancel) {
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
            // Not charged against the bake's attempt budget: the input, not
            // the bake, is wrong.
            Err(BakeFailure::IncompatibleInput(message)) => AttemptOutcome::Unsupported(message),
            Err(BakeFailure::Other(message)) => AttemptOutcome::Failed(message),
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
