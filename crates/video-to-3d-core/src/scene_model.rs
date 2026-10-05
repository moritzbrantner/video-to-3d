//! Canonical assembled-scene artifact model.
//!
//! This is the runtime-neutral output contract of a completed media-to-scene
//! build. It is deliberately not a renderer scene graph: it records what was
//! accepted (cameras, surfaces, splat fields, editable assets, materials,
//! collision proxies, semantic labels, lights, audio anchors) and where each
//! piece of content came from. Renderers and exporters derive their own state
//! from it.
//!
//! Provenance is attached to every content resource and cannot be overridden by
//! the assets that reference it, so generative completion stays distinguishable
//! from observed geometry through serialization and export.

use crate::scene_project::{ContentHash, ProjectPath, SceneProjectError};
use crate::{EvidenceCameraAuthority, EvidenceOrigin};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ASSEMBLED_SCENE_SCHEMA_VERSION: u32 = 1;
const MAX_IDENTIFIER_LENGTH: usize = 64;
const UNIT_QUATERNION_TOLERANCE: f32 = 1e-4;

fn invalid<T>(message: impl Into<String>) -> Result<T, SceneProjectError> {
    Err(SceneProjectError::Invalid(message.into()))
}

/// Unit policy of the shared scene frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneUnit {
    /// Monocular reconstruction without recovered metric scale.
    ArbitraryMonocular,
    /// One scene unit is one meter.
    Meters,
}

/// The single coordinate frame shared by every camera, asset, light, and audio
/// anchor. The frame is right-handed and Y-up; only the unit policy varies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneFrame {
    pub unit: SceneUnit,
    pub handedness: Handedness,
    pub up_axis: UpAxis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Handedness {
    RightHanded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpAxis {
    PositiveY,
}

/// Where a piece of scene content came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneProvenance {
    /// Directly accepted multi-view geometric evidence.
    GeometricMultiView,
    /// Locally proposed completion that re-passed image and multi-view checks.
    RevalidatedCompletion,
    /// Evidence supplied by a learned multi-view reconstruction provider.
    LearnedMultiView,
    /// Content inferred without direct multi-view observation support.
    GenerativeCompletion,
    /// Estimated from observations but not geometry (for example lighting).
    Estimated,
    /// Explicitly authored or imported by a user.
    Authored,
}

impl SceneProvenance {
    pub fn is_observed_geometry(self) -> bool {
        matches!(self, Self::GeometricMultiView | Self::RevalidatedCompletion)
    }

    /// Provenance classes whose evidence must stay linked to accepted cameras,
    /// matching the `ReconstructionEvidenceView` camera-support rule.
    pub fn requires_camera_evidence(self) -> bool {
        self.is_observed_geometry() || self == Self::LearnedMultiView
    }
}

impl From<EvidenceOrigin> for SceneProvenance {
    fn from(origin: EvidenceOrigin) -> Self {
        match origin {
            EvidenceOrigin::GeometricMultiView => Self::GeometricMultiView,
            EvidenceOrigin::RevalidatedCompletion => Self::RevalidatedCompletion,
            EvidenceOrigin::LearnedMultiView => Self::LearnedMultiView,
            EvidenceOrigin::GenerativeCompletion => Self::GenerativeCompletion,
        }
    }
}

/// Rigid placement plus positive scale, in the parent's frame (or the scene
/// frame for root assets). Rotation is a unit quaternion `[x, y, z, w]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transform {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
    };

    fn validate(&self, owner: &str) -> Result<(), SceneProjectError> {
        let values = self
            .translation
            .iter()
            .chain(&self.rotation)
            .chain(&self.scale);
        if values.clone().any(|value| !value.is_finite()) {
            return invalid(format!("`{owner}` transform must be finite"));
        }
        let norm = self
            .rotation
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if (norm - 1.0).abs() > UNIT_QUATERNION_TOLERANCE {
            return invalid(format!(
                "`{owner}` rotation must be a unit quaternion (norm {norm})"
            ));
        }
        if self.scale.iter().any(|value| *value <= 0.0) {
            return invalid(format!("`{owner}` scale must be positive"));
        }
        Ok(())
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraIntrinsics {
    pub width: u32,
    pub height: u32,
    pub focal_length_pixels: f32,
    pub principal_point: [f32; 2],
}

/// Accepted camera in the scene frame. The transform maps camera to scene.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneCamera {
    pub id: String,
    pub source_frame: usize,
    pub authority: CameraAuthority,
    pub transform: Transform,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intrinsics: Option<CameraIntrinsics>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraAuthority {
    CalibratedSeed,
    RegisteredGeometry,
    ProviderEstimated,
}

impl From<EvidenceCameraAuthority> for CameraAuthority {
    fn from(authority: EvidenceCameraAuthority) -> Self {
        match authority {
            EvidenceCameraAuthority::CalibratedSeed => Self::CalibratedSeed,
            EvidenceCameraAuthority::RegisteredGeometry => Self::RegisteredGeometry,
            EvidenceCameraAuthority::ProviderEstimated => Self::ProviderEstimated,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    Mesh,
    SplatField,
    Texture,
    EnvironmentMap,
    AudioClip,
}

impl ResourceKind {
    fn is_geometry(self) -> bool {
        matches!(self, Self::Mesh | Self::SplatField)
    }
}

/// Content-addressed payload stored next to the scene document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneResource {
    pub id: String,
    pub kind: ResourceKind,
    pub path: ProjectPath,
    pub content_hash: ContentHash,
    /// Every provenance class present in the payload. A fused resource whose
    /// regions have different origins lists all of them; region-level detail
    /// stays in the referenced payload.
    pub provenance: Vec<SceneProvenance>,
    /// Aggregate acceptance confidence in `[0, 1]`, when the producer has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Accepted source frames supporting the camera-backed part of the
    /// resource. Required for camera-backed classes, forbidden when every class
    /// is camera-free (generative completion must stay camera-free).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_frames: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Material {
    pub id: String,
    pub base_color: [f32; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_color_texture: Option<String>,
    pub metallic: f32,
    pub roughness: f32,
    /// Origin of the material factors themselves (texture provenance lives on
    /// the texture resource).
    pub provenance: SceneProvenance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshVisual {
    pub mesh: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
}

/// Visual representation. Mesh and splat may coexist for one asset.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisualRepresentation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<MeshVisual>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub splat: Option<String>,
}

/// Collision proxy; independent of the visual representation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum CollisionShape {
    Mesh { mesh: String },
    ConvexHull { mesh: String },
    Box { half_extents: [f32; 3] },
    Sphere { radius: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticLabel {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetRole {
    /// Static environment (observed surface, completed background).
    Environment,
    /// Individually editable object.
    EditableObject,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneAsset {
    pub id: String,
    pub role: AssetRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    pub transform: Transform,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visual: Option<VisualRepresentation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collision: Option<CollisionShape>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<SemanticLabel>,
    /// Provenance of a primitive collision proxy (box/sphere), which has no
    /// resource of its own. Required exactly when such a primitive is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collision_provenance: Option<SceneProvenance>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum LightKind {
    Ambient,
    Directional,
    Point { range: Option<f32> },
    Spot { inner_cone: f32, outer_cone: f32 },
    Environment { map: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneLight {
    pub id: String,
    pub kind: LightKind,
    pub transform: Transform,
    pub color: [f32; 3],
    pub intensity: f32,
    pub provenance: SceneProvenance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioAnchor {
    pub id: String,
    pub clip: String,
    /// Ambient audio has no position; positional audio is placed by transform,
    /// optionally relative to an asset.
    pub ambient: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attached_to: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<Transform>,
    pub provenance: SceneProvenance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssembledScene {
    pub schema_version: u32,
    pub frame: SceneFrame,
    #[serde(default)]
    pub cameras: Vec<SceneCamera>,
    #[serde(default)]
    pub resources: Vec<SceneResource>,
    #[serde(default)]
    pub materials: Vec<Material>,
    #[serde(default)]
    pub assets: Vec<SceneAsset>,
    #[serde(default)]
    pub lights: Vec<SceneLight>,
    #[serde(default)]
    pub audio: Vec<AudioAnchor>,
}

/// Which representations an asset carries and where they came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetProvenance {
    pub visual: BTreeSet<SceneProvenance>,
    pub collision: BTreeSet<SceneProvenance>,
}

impl AssetProvenance {
    /// True when any part of the asset is generative completion.
    pub fn contains_generative(&self) -> bool {
        self.visual.contains(&SceneProvenance::GenerativeCompletion)
            || self
                .collision
                .contains(&SceneProvenance::GenerativeCompletion)
    }
}

impl AssembledScene {
    pub fn new(frame: SceneFrame) -> Self {
        Self {
            schema_version: ASSEMBLED_SCENE_SCHEMA_VERSION,
            frame,
            cameras: Vec::new(),
            resources: Vec::new(),
            materials: Vec::new(),
            assets: Vec::new(),
            lights: Vec::new(),
            audio: Vec::new(),
        }
    }

    pub fn from_json(document: &str) -> Result<Self, SceneProjectError> {
        let value: serde_json::Value = serde_json::from_str(document)
            .map_err(|error| SceneProjectError::Malformed(error.to_string()))?;
        match value.get("schema_version").map(serde_json::Value::as_u64) {
            Some(Some(version)) if version == u64::from(ASSEMBLED_SCENE_SCHEMA_VERSION) => {}
            Some(Some(version)) => {
                return Err(SceneProjectError::UnsupportedSchemaVersion(version))
            }
            _ => {
                return Err(SceneProjectError::Malformed(
                    "`schema_version` must be a non-negative integer".into(),
                ))
            }
        }
        let mut scene: Self = serde_json::from_value(value)
            .map_err(|error| SceneProjectError::Malformed(error.to_string()))?;
        scene.validate()?;
        scene.canonicalize();
        Ok(scene)
    }

    pub fn to_canonical_json(&self) -> Result<String, SceneProjectError> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.canonicalize();
        let mut document = serde_json::to_string_pretty(&canonical)
            .map_err(|error| SceneProjectError::Malformed(error.to_string()))?;
        document.push('\n');
        Ok(document)
    }

    pub fn canonicalize(&mut self) {
        self.cameras.sort_by(|a, b| a.id.cmp(&b.id));
        self.resources.sort_by(|a, b| a.id.cmp(&b.id));
        for resource in &mut self.resources {
            resource.provenance.sort();
            resource.source_frames.sort_unstable();
        }
        self.materials.sort_by(|a, b| a.id.cmp(&b.id));
        self.assets.sort_by(|a, b| a.id.cmp(&b.id));
        for asset in &mut self.assets {
            asset.labels.sort_by(|a, b| a.name.cmp(&b.name));
        }
        self.lights.sort_by(|a, b| a.id.cmp(&b.id));
        self.audio.sort_by(|a, b| a.id.cmp(&b.id));
    }

    pub fn validate(&self) -> Result<(), SceneProjectError> {
        if self.schema_version != ASSEMBLED_SCENE_SCHEMA_VERSION {
            return Err(SceneProjectError::UnsupportedSchemaVersion(u64::from(
                self.schema_version,
            )));
        }
        let mut ids = BTreeSet::new();
        let mut declare = |id: &str| -> Result<(), SceneProjectError> {
            validate_identifier(id)?;
            if !ids.insert(id.to_owned()) {
                return invalid(format!(
                    "scene identifier `{id}` is declared more than once"
                ));
            }
            Ok(())
        };

        let mut frames = BTreeSet::new();
        for camera in &self.cameras {
            declare(&camera.id)?;
            camera.transform.validate(&camera.id)?;
            if camera.transform.scale != [1.0; 3] {
                return invalid(format!("camera `{}` must have unit scale", camera.id));
            }
            if !frames.insert(camera.source_frame) {
                return invalid(format!(
                    "source frame {} has more than one accepted camera",
                    camera.source_frame
                ));
            }
            if let Some(intrinsics) = &camera.intrinsics {
                let finite = intrinsics.focal_length_pixels.is_finite()
                    && intrinsics.principal_point.iter().all(|v| v.is_finite());
                if intrinsics.width == 0
                    || intrinsics.height == 0
                    || !finite
                    || intrinsics.focal_length_pixels <= 0.0
                {
                    return invalid(format!("camera `{}` has invalid intrinsics", camera.id));
                }
            }
        }

        let mut resources: BTreeMap<&str, &SceneResource> = BTreeMap::new();
        let mut paths = BTreeSet::new();
        for resource in &self.resources {
            declare(&resource.id)?;
            if !paths.insert(resource.path.as_str().to_ascii_lowercase()) {
                return invalid(format!(
                    "resource path `{}` is used more than once",
                    resource.path.as_str()
                ));
            }
            validate_confidence(&resource.id, resource.confidence)?;
            if resource.provenance.is_empty() {
                return invalid(format!("resource `{}` declares no provenance", resource.id));
            }
            let mut classes = BTreeSet::new();
            for provenance in &resource.provenance {
                if !classes.insert(*provenance) {
                    return invalid(format!(
                        "resource `{}` lists provenance {provenance:?} more than once",
                        resource.id
                    ));
                }
            }
            if resource.kind.is_geometry() && classes.contains(&SceneProvenance::Estimated) {
                return invalid(format!(
                    "geometry resource `{}` cannot have estimated provenance",
                    resource.id
                ));
            }
            let camera_backed = classes
                .iter()
                .any(|provenance| provenance.requires_camera_evidence());
            if camera_backed && resource.source_frames.is_empty() {
                return invalid(format!(
                    "camera-supported resource `{}` must list its supporting source frames",
                    resource.id
                ));
            }
            if !camera_backed && !resource.source_frames.is_empty() {
                return invalid(format!(
                    "resource `{}` has only camera-free provenance and must not cite source frames",
                    resource.id
                ));
            }
            let mut unique_frames = BTreeSet::new();
            for frame in &resource.source_frames {
                if !unique_frames.insert(*frame) {
                    return invalid(format!(
                        "resource `{}` lists source frame {frame} more than once",
                        resource.id
                    ));
                }
                if !frames.contains(frame) {
                    return invalid(format!(
                        "resource `{}` cites source frame {frame}, which has no accepted camera",
                        resource.id
                    ));
                }
            }
            resources.insert(resource.id.as_str(), resource);
        }
        let resource_of = |owner: &str, id: &str, kind: ResourceKind| match resources.get(id) {
            Some(resource) if resource.kind == kind => Ok(*resource),
            Some(resource) => invalid(format!(
                "`{owner}` expects {kind:?} resource but `{id}` is {:?}",
                resource.kind
            )),
            None => invalid(format!("`{owner}` references unknown resource `{id}`")),
        };

        let mut materials = BTreeSet::new();
        for material in &self.materials {
            declare(&material.id)?;
            let factors = material
                .base_color
                .iter()
                .chain([&material.metallic, &material.roughness]);
            if factors
                .clone()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            {
                return invalid(format!(
                    "material `{}` factors must be finite and within [0, 1]",
                    material.id
                ));
            }
            if let Some(texture) = &material.base_color_texture {
                resource_of(&material.id, texture, ResourceKind::Texture)?;
            }
            materials.insert(material.id.as_str());
        }

        let assets: BTreeMap<&str, &SceneAsset> = self
            .assets
            .iter()
            .map(|asset| (asset.id.as_str(), asset))
            .collect();
        for asset in &self.assets {
            declare(&asset.id)?;
            asset.transform.validate(&asset.id)?;
            if asset.visual.is_none() && asset.collision.is_none() {
                return invalid(format!(
                    "asset `{}` must carry a visual representation, a collision shape, or both",
                    asset.id
                ));
            }
            if let Some(visual) = &asset.visual {
                if visual.mesh.is_none() && visual.splat.is_none() {
                    return invalid(format!(
                        "asset `{}` declares an empty visual representation",
                        asset.id
                    ));
                }
                if let Some(mesh) = &visual.mesh {
                    resource_of(&asset.id, &mesh.mesh, ResourceKind::Mesh)?;
                    if let Some(material) = &mesh.material {
                        if !materials.contains(material.as_str()) {
                            return invalid(format!(
                                "asset `{}` references unknown material `{material}`",
                                asset.id
                            ));
                        }
                    }
                }
                if let Some(splat) = &visual.splat {
                    resource_of(&asset.id, splat, ResourceKind::SplatField)?;
                }
            }
            match &asset.collision {
                Some(CollisionShape::Mesh { mesh } | CollisionShape::ConvexHull { mesh }) => {
                    resource_of(&asset.id, mesh, ResourceKind::Mesh)?;
                    if asset.collision_provenance.is_some() {
                        return invalid(format!(
                            "asset `{}` collision provenance comes from its mesh resource",
                            asset.id
                        ));
                    }
                }
                Some(primitive) => {
                    let valid = match primitive {
                        CollisionShape::Box { half_extents } => half_extents
                            .iter()
                            .all(|value| value.is_finite() && *value > 0.0),
                        CollisionShape::Sphere { radius } => radius.is_finite() && *radius > 0.0,
                        _ => unreachable!(),
                    };
                    if !valid {
                        return invalid(format!(
                            "asset `{}` collision primitive must have positive finite extents",
                            asset.id
                        ));
                    }
                    match asset.collision_provenance {
                        None => {
                            return invalid(format!(
                                "asset `{}` primitive collision shape requires collision_provenance",
                                asset.id
                            ))
                        }
                        // A primitive proxy has no evidence linkage of its own;
                        // camera-backed proxies must be mesh resources instead.
                        Some(provenance) if provenance.requires_camera_evidence() => {
                            return invalid(format!(
                                "asset `{}` primitive collision proxy cannot claim camera-backed provenance {provenance:?}",
                                asset.id
                            ))
                        }
                        Some(_) => {}
                    }
                }
                None if asset.collision_provenance.is_some() => {
                    return invalid(format!(
                        "asset `{}` declares collision provenance without a collision shape",
                        asset.id
                    ))
                }
                None => {}
            }
            let mut labels = BTreeSet::new();
            for label in &asset.labels {
                if label.name.trim().is_empty() || label.name.chars().any(char::is_control) {
                    return invalid(format!("asset `{}` has an empty semantic label", asset.id));
                }
                if !labels.insert(label.name.as_str()) {
                    return invalid(format!(
                        "asset `{}` repeats semantic label `{}`",
                        asset.id, label.name
                    ));
                }
                validate_confidence(&asset.id, label.confidence)?;
            }
            if let Some(parent) = &asset.parent {
                if !assets.contains_key(parent.as_str()) {
                    return invalid(format!(
                        "asset `{}` references unknown parent `{parent}`",
                        asset.id
                    ));
                }
            }
        }
        for asset in &self.assets {
            let mut visited = BTreeSet::new();
            let mut current = Some(asset);
            while let Some(node) = current {
                if !visited.insert(node.id.as_str()) {
                    return invalid(format!("asset `{}` has a cyclic parent chain", asset.id));
                }
                current = node
                    .parent
                    .as_deref()
                    .and_then(|parent| assets.get(parent).copied());
            }
        }

        for light in &self.lights {
            declare(&light.id)?;
            light.transform.validate(&light.id)?;
            if light
                .color
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                || !light.intensity.is_finite()
                || light.intensity < 0.0
            {
                return invalid(format!(
                    "light `{}` color must be within [0, 1] and intensity non-negative",
                    light.id
                ));
            }
            match &light.kind {
                LightKind::Environment { map } => {
                    resource_of(&light.id, map, ResourceKind::EnvironmentMap)?;
                }
                LightKind::Point { range: Some(range) } if !(range.is_finite() && *range > 0.0) => {
                    return invalid(format!("light `{}` range must be positive", light.id));
                }
                LightKind::Spot {
                    inner_cone,
                    outer_cone,
                } if !(inner_cone.is_finite()
                    && outer_cone.is_finite()
                    && 0.0 <= *inner_cone
                    && inner_cone <= outer_cone
                    && *outer_cone <= std::f32::consts::FRAC_PI_2) =>
                {
                    return invalid(format!(
                        "light `{}` cone angles must satisfy 0 <= inner <= outer <= pi/2",
                        light.id
                    ));
                }
                _ => {}
            }
            if light.provenance.requires_camera_evidence() {
                return invalid(format!(
                    "light `{}` cannot claim observed-geometry provenance",
                    light.id
                ));
            }
        }

        for anchor in &self.audio {
            declare(&anchor.id)?;
            resource_of(&anchor.id, &anchor.clip, ResourceKind::AudioClip)?;
            if anchor.ambient && (anchor.attached_to.is_some() || anchor.transform.is_some()) {
                return invalid(format!(
                    "ambient audio `{}` must not be positioned",
                    anchor.id
                ));
            }
            if !anchor.ambient && anchor.attached_to.is_none() && anchor.transform.is_none() {
                return invalid(format!(
                    "positional audio `{}` requires a transform or an attached asset",
                    anchor.id
                ));
            }
            if let Some(asset) = &anchor.attached_to {
                if !assets.contains_key(asset.as_str()) {
                    return invalid(format!(
                        "audio `{}` references unknown asset `{asset}`",
                        anchor.id
                    ));
                }
            }
            if let Some(transform) = &anchor.transform {
                transform.validate(&anchor.id)?;
            }
            if anchor.provenance.requires_camera_evidence() {
                return invalid(format!(
                    "audio `{}` cannot claim observed-geometry provenance",
                    anchor.id
                ));
            }
        }
        Ok(())
    }

    /// Provenance of an asset's representations, derived from the resources it
    /// references. Assets cannot relabel resource provenance.
    pub fn asset_provenance(&self, asset_id: &str) -> Option<AssetProvenance> {
        let asset = self.assets.iter().find(|asset| asset.id == asset_id)?;
        let provenance_of = |id: &str| {
            self.resources
                .iter()
                .find(|resource| resource.id == id)
                .map(|resource| resource.provenance.clone())
                .unwrap_or_default()
        };
        let mut visual = BTreeSet::new();
        if let Some(representation) = &asset.visual {
            if let Some(mesh) = &representation.mesh {
                visual.extend(provenance_of(&mesh.mesh));
                // Appearance is part of the visual: a generated texture on an
                // observed mesh must still surface as generative content.
                let material = mesh.material.as_deref().and_then(|material| {
                    self.materials
                        .iter()
                        .find(|candidate| candidate.id == material)
                });
                if let Some(material) = material {
                    visual.insert(material.provenance);
                    if let Some(texture) = &material.base_color_texture {
                        visual.extend(provenance_of(texture));
                    }
                }
            }
            if let Some(splat) = &representation.splat {
                visual.extend(provenance_of(splat));
            }
        }
        let collision = match &asset.collision {
            Some(CollisionShape::Mesh { mesh } | CollisionShape::ConvexHull { mesh }) => {
                provenance_of(mesh)
            }
            Some(_) => asset.collision_provenance.into_iter().collect(),
            None => Vec::new(),
        };
        let collision = collision.into_iter().collect();
        Some(AssetProvenance { visual, collision })
    }
}

fn validate_identifier(id: &str) -> Result<(), SceneProjectError> {
    let valid = !id.is_empty()
        && id.len() <= MAX_IDENTIFIER_LENGTH
        && id
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        && id.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
        });
    if !valid {
        return invalid(format!(
            "scene identifier `{id}` must be 1..={MAX_IDENTIFIER_LENGTH} lowercase ASCII letters, digits, `-`, `_` or `.`"
        ));
    }
    Ok(())
}

fn validate_confidence(owner: &str, confidence: Option<f32>) -> Result<(), SceneProjectError> {
    if let Some(value) = confidence {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return invalid(format!("`{owner}` confidence must be within [0, 1]"));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "scene_model_tests.rs"]
mod tests;
