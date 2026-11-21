//! Scene placement of generated object candidates (#69).
//!
//! Each generated candidate mesh (#68) is placed back into the reconstructed
//! scene from observed evidence, in the scene's own (monocular) gauge:
//!
//! - **Up direction.** The normal of a fitted floor plane when the scene shows
//!   enough floor; otherwise the reference cameras' mean up axis. The source
//!   is recorded.
//! - **Rotation.** The generator's canonical frame (TripoSR: the input camera
//!   on +x, +y right, +z up) is mapped onto the object's primary reference
//!   camera, then tilted so the object's up matches the scene up.
//! - **Scale and position.** The observed points of the object (its front
//!   surface) fix the width and height across the view and the visible
//!   surface's position; the mesh's own depth is placed behind that surface.
//!   Observed geometry always wins; weak semantic size priors are used only
//!   to report a metric scale hint, never to override observed scale.
//! - **Support.** Horizontal planes fitted to floor and furniture (table,
//!   desk, shelf, cabinet) points under the object are support candidates; the
//!   gap between the placed object's bottom and the nearest support below
//!   classifies it as in contact, floating or sunken, or reports that no
//!   contact evidence exists.
//!
//! - **Validation (#70).** Every placement is projected back into the
//!   object's registered source views and checked against the original masks
//!   and accepted geometry ([`crate::placement_validation`]); contradicted or
//!   single-view placements are marked for review.
//!
//! Nothing here changes cameras, the reconstructed geometry or the candidate
//! meshes: a placement is a transform plus its evidence.

mod multi_view_fit;

use crate::object_generation::ObjectGenerationArtifact;
use crate::object_identity::{front_points, FrameCamera, ObjectCategory, ObjectIdentityArtifact};
use crate::object_references::ObjectReferencesArtifact;
use crate::placement_validation::{
    validate_placement, PlacedMesh, PlacementValidation, PlacementVerdict,
};
use crate::scene_project::ContentHash;
use crate::semantic_inventory::SemanticInventoryArtifact;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const OBJECT_PLACEMENT_SCHEMA_VERSION: u32 = 2;
/// Floor points needed to take the scene up direction from a fitted floor.
pub const MIN_FLOOR_POINTS: usize = 20;
/// Points needed to fit a support plane.
pub const MIN_SUPPORT_POINTS: usize = 8;
/// A plane counts as horizontal when its normal is this close to up.
pub const HORIZONTAL_COSINE: f64 = 0.9;
/// Contact tolerance, as a share of the placed object's height.
pub const CONTACT_TOLERANCE: f64 = 0.1;
/// Labels whose planes can support objects.
pub const SUPPORT_LABELS: [&str; 6] = ["floor", "ground", "table", "desk", "shelf", "cabinet"];

/// Typical heights in metres: weak priors that only produce a metric scale
/// hint for the monocular scene, never an object's scale.
pub fn size_prior(label: &str) -> Option<f64> {
    let label = label.to_ascii_lowercase();
    [
        ("chair", 0.9),
        ("table", 0.75),
        ("desk", 0.75),
        ("sofa", 0.85),
        ("bed", 0.6),
        ("monitor", 0.45),
        ("lamp", 0.5),
        ("teddy bear", 0.3),
        ("globe", 0.4),
        ("plant", 0.5),
        ("die", 0.03),
    ]
    .into_iter()
    .find(|(name, _)| label.contains(name))
    .map(|(_, height)| height)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpSource {
    FloorPlane,
    CameraUp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContactState {
    InContact,
    Floating,
    Sunken,
    NoContactEvidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleSource {
    /// Width and height across the view from observed points.
    Observed,
    /// A segmented reference silhouette back-projected at observed front depth.
    MaskAtObservedDepth,
    /// Metric-free scale from calibrated separation of corresponding object masks.
    MultiViewSilhouettes,
    /// Too little observed extent to scale the object.
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupportPlane {
    pub object: String,
    pub label: String,
    /// Unit normal (pointing up) and offset: `normal · x = offset`.
    pub normal: [f64; 3],
    pub offset: f64,
    pub points: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SilhouetteRegistration {
    #[serde(default)]
    pub matched_patch_depths: usize,
    pub fitting_frames: Vec<usize>,
    pub heldout_frame: usize,
    pub heldout_mask_iou: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectPlacement {
    pub object: String,
    pub label: String,
    pub generation_request: String,
    /// Mesh-to-scene transform: `x_scene = scale · rotation · x_mesh + translation`
    /// (rotation row-major).
    pub rotation: [f64; 9],
    pub scale: f64,
    pub translation: [f64; 3],
    pub scale_source: ScaleSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub silhouette_registration: Option<SilhouetteRegistration>,
    pub observed_points: usize,
    pub reference_frame: usize,
    pub contact: ContactState,
    pub support: Option<String>,
    /// Distance from the placed bottom to the support, along up (scene units).
    pub contact_gap: Option<f64>,
    pub reason: String,
    /// Multi-view check against the original observations.
    pub validation: PlacementValidation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnplacedObject {
    pub object: String,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementStatus {
    Available,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectPlacementArtifact {
    pub schema_version: u32,
    pub status: PlacementStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
    pub object_generation: ContentHash,
    pub up: [f64; 3],
    pub up_source: UpSource,
    pub supports: Vec<SupportPlane>,
    pub placements: Vec<ObjectPlacement>,
    pub unplaced: Vec<UnplacedObject>,
    /// Scene units per metre suggested by semantic size priors, with the
    /// labels used; reported only, never applied to the gauge.
    pub metric_scale_hint: Option<f64>,
    pub metric_hint_labels: Vec<String>,
}

impl ObjectPlacementArtifact {
    pub fn from_json(document: &str) -> Result<Self, String> {
        let artifact: Self = serde_json::from_str(document)
            .map_err(|error| format!("object placement artifact: {error}"))?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn to_json(&self) -> String {
        let mut document = serde_json::to_string_pretty(self).expect("placements serialize");
        document.push('\n');
        document
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != OBJECT_PLACEMENT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported object placement schema version {}",
                self.schema_version
            ));
        }
        let unavailable = self.status == PlacementStatus::Unavailable;
        if unavailable
            != self
                .unavailable_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
            || (unavailable && !self.placements.is_empty())
        {
            return Err("object placements: unavailable needs a reason and no placements".into());
        }
        if (norm(self.up) - 1.0).abs() > 1e-6 {
            return Err("object placement up direction is not a unit vector".into());
        }
        for placement in &self.placements {
            if let Some(fit) = &placement.silhouette_registration {
                if fit.fitting_frames.len() < 2
                    || fit.fitting_frames.windows(2).any(|w| w[0] >= w[1])
                    || fit.fitting_frames.contains(&fit.heldout_frame)
                    || !(0.0..=1.0).contains(&fit.heldout_mask_iou)
                    || placement.scale_source != ScaleSource::MultiViewSilhouettes
                {
                    return Err(format!(
                        "object {} has invalid silhouette registration evidence",
                        placement.object
                    ));
                }
            } else if placement.scale_source == ScaleSource::MultiViewSilhouettes {
                return Err(format!(
                    "object {} is missing its silhouette registration evidence",
                    placement.object
                ));
            }
            let finite = placement
                .rotation
                .iter()
                .chain(&placement.translation)
                .chain(std::iter::once(&placement.scale))
                .all(|value| value.is_finite());
            if !finite
                || placement.scale <= 0.0
                || placement.reason.trim().is_empty()
                || placement.validation.needs_review
                    != (placement.validation.verdict != PlacementVerdict::Accepted)
            {
                return Err(format!(
                    "object placement of {} is not a valid transform",
                    placement.object
                ));
            }
        }
        Ok(())
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let length = norm(a).max(1e-12);
    a.map(|value| value / length)
}

fn scaled(a: [f64; 3], factor: f64) -> [f64; 3] {
    a.map(|value| value * factor)
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Least-squares plane through `points`: (unit normal, offset).
pub(crate) fn fit_plane(points: &[[f64; 3]]) -> Option<([f64; 3], f64)> {
    if points.len() < 3 {
        return None;
    }
    let count = points.len() as f64;
    let centre = points
        .iter()
        .fold([0.0; 3], |sum, point| add(sum, *point))
        .map(|value| value / count);
    let mut covariance = nalgebra::Matrix3::<f64>::zeros();
    for point in points {
        let d = nalgebra::Vector3::from(sub(*point, centre));
        covariance += d * d.transpose();
    }
    let eigen = covariance.symmetric_eigen();
    let (index, _) = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))?;
    let normal = eigen.eigenvectors.column(index);
    let normal = normalize([normal[0], normal[1], normal[2]]);
    Some((normal, dot(normal, centre)))
}

/// Camera centre, right, down and forward axes in the scene.
fn camera_axes(camera: &FrameCamera) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    let r = camera.rotation;
    let row = |index: usize| [r[index * 3], r[index * 3 + 1], r[index * 3 + 2]];
    let (right, down, forward) = (row(0), row(1), row(2));
    let t = camera.translation;
    // centre = -Rᵀ t
    let centre = [0, 1, 2].map(|axis| -(r[axis] * t[0] + r[3 + axis] * t[1] + r[6 + axis] * t[2]));
    (centre, right, down, forward)
}

/// Rotation taking `from` onto `to` (unit vectors), row-major.
fn rotation_between(from: [f64; 3], to: [f64; 3]) -> nalgebra::Matrix3<f64> {
    let a = nalgebra::Vector3::from(from);
    let b = nalgebra::Vector3::from(to);
    nalgebra::Rotation3::rotation_between(&a, &b)
        .map(|rotation| *rotation.matrix())
        .unwrap_or_else(nalgebra::Matrix3::identity)
}

/// Axis-aligned bounds of a candidate mesh in the generator's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshBounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl MeshBounds {
    pub fn of(vertices: &[[f64; 3]]) -> Option<Self> {
        let first = *vertices.first()?;
        Some(vertices.iter().fold(
            Self {
                min: first,
                max: first,
            },
            |bounds, vertex| Self {
                min: [0, 1, 2].map(|axis| bounds.min[axis].min(vertex[axis])),
                max: [0, 1, 2].map(|axis| bounds.max[axis].max(vertex[axis])),
            },
        ))
    }
}

/// Place every generated candidate in the scene.
#[allow(clippy::too_many_arguments)]
pub fn place_objects(
    inventory: &SemanticInventoryArtifact,
    identities: &ObjectIdentityArtifact,
    references: &ObjectReferencesArtifact,
    generation: &ObjectGenerationArtifact,
    generation_hash: ContentHash,
    mesh_vertices: &BTreeMap<String, Vec<[f64; 3]>>,
    cameras: &[FrameCamera],
    points: &[[f64; 3]],
) -> ObjectPlacementArtifact {
    place_objects_with_faces(
        inventory,
        identities,
        references,
        generation,
        generation_hash,
        mesh_vertices,
        None,
        cameras,
        points,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn place_objects_with_faces(
    inventory: &SemanticInventoryArtifact,
    identities: &ObjectIdentityArtifact,
    references: &ObjectReferencesArtifact,
    generation: &ObjectGenerationArtifact,
    generation_hash: ContentHash,
    mesh_vertices: &BTreeMap<String, Vec<[f64; 3]>>,
    mesh_faces: Option<&BTreeMap<String, Vec<[u32; 3]>>>,
    cameras: &[FrameCamera],
    points: &[[f64; 3]],
    frames: &[crate::multi_view_objects::FramePixels],
) -> ObjectPlacementArtifact {
    let cameras_by_frame: BTreeMap<usize, &FrameCamera> = cameras
        .iter()
        .map(|camera| (camera.frame_index, camera))
        .collect();
    let observations: BTreeMap<&str, _> = inventory
        .observations
        .iter()
        .map(|observation| (observation.id.as_str(), observation))
        .collect();
    let mut artifact = ObjectPlacementArtifact {
        schema_version: OBJECT_PLACEMENT_SCHEMA_VERSION,
        status: PlacementStatus::Available,
        unavailable_reason: None,
        object_generation: generation_hash,
        up: [0.0, 0.0, 1.0],
        up_source: UpSource::CameraUp,
        supports: Vec::new(),
        placements: Vec::new(),
        unplaced: Vec::new(),
        metric_scale_hint: None,
        metric_hint_labels: Vec::new(),
    };
    if generation.status == crate::object_generation::GenerationStatus::Unavailable {
        artifact.status = PlacementStatus::Unavailable;
        artifact.unavailable_reason = generation.unavailable_reason.clone();
        return artifact;
    }
    // Points of one identified object, from every observation with a camera.
    let object_points = |object: &crate::object_identity::SceneObject| -> Vec<[f64; 3]> {
        object
            .observations
            .iter()
            .filter_map(|id| {
                let observation = observations.get(id.as_str())?;
                let camera = cameras_by_frame.get(&observation.frame_index)?;
                Some(front_points(observation, camera, points))
            })
            .flatten()
            .map(|(_, point)| point)
            .collect()
    };

    // Up direction: the floor normal when there is enough floor, else the
    // cameras' mean up axis.
    let camera_up = normalize(cameras.iter().fold([0.0; 3], |sum, camera| {
        let (_, _, down, _) = camera_axes(camera);
        sub(sum, down)
    }));
    let floor_points: Vec<[f64; 3]> = identities
        .objects
        .iter()
        .filter(|object| {
            matches!(
                object.label.to_ascii_lowercase().as_str(),
                "floor" | "ground"
            )
        })
        .flat_map(&object_points)
        .collect();
    let mut up = if cameras.is_empty() {
        [0.0, 0.0, 1.0]
    } else {
        camera_up
    };
    if floor_points.len() >= MIN_FLOOR_POINTS {
        if let Some((normal, _)) = fit_plane(&floor_points) {
            up = if dot(normal, camera_up) >= 0.0 {
                normal
            } else {
                scaled(normal, -1.0)
            };
            artifact.up_source = UpSource::FloorPlane;
        }
    }
    artifact.up = normalize(up);
    let up = artifact.up;

    // Horizontal support planes from floor and furniture.
    for object in &identities.objects {
        let label = object.label.to_ascii_lowercase();
        if !SUPPORT_LABELS.iter().any(|support| label.contains(support)) {
            continue;
        }
        let support_points = object_points(object);
        if support_points.len() < MIN_SUPPORT_POINTS {
            continue;
        }
        if let Some((normal, offset)) = fit_plane(&support_points) {
            let (normal, offset) = if dot(normal, up) >= 0.0 {
                (normal, offset)
            } else {
                (scaled(normal, -1.0), -offset)
            };
            if dot(normal, up) >= HORIZONTAL_COSINE {
                artifact.supports.push(SupportPlane {
                    object: object.id.clone(),
                    label: object.label.clone(),
                    normal,
                    offset,
                    points: support_points.len(),
                });
            }
        }
    }

    let packages: BTreeMap<&str, _> = references
        .packages
        .iter()
        .map(|package| (package.object.as_str(), package))
        .collect();
    let categories: BTreeMap<&str, ObjectCategory> = identities
        .objects
        .iter()
        .map(|object| (object.id.as_str(), object.category))
        .collect();
    let mut hints = Vec::new();
    for generated in &generation.objects {
        let request = &generated.request;
        let unplaced = |reason: String| UnplacedObject {
            object: request.object.clone(),
            reason,
        };
        if generated.output.is_none() {
            artifact
                .unplaced
                .push(unplaced("no generated mesh (generation failed)".into()));
            continue;
        }
        if categories.get(request.object.as_str()) == Some(&ObjectCategory::Structure) {
            continue;
        }
        let vertices = mesh_vertices.get(&request.request_id);
        let (Some(bounds), Some(package), Some(camera)) = (
            vertices.and_then(|vertices| MeshBounds::of(vertices)),
            packages.get(request.object.as_str()),
            cameras_by_frame.get(&request.frame_index),
        ) else {
            artifact.unplaced.push(unplaced(
                "the primary reference frame has no registered camera or the mesh has no bounds"
                    .into(),
            ));
            continue;
        };
        let views: Vec<_> = identities
            .objects
            .iter()
            .find(|object| object.id == request.object)
            .map(|object| {
                object
                    .observations
                    .iter()
                    .filter_map(|id| {
                        let observation = *observations.get(id.as_str())?;
                        let camera = *cameras_by_frame.get(&observation.frame_index)?;
                        Some((observation, camera))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let validate = |placement: &mut ObjectPlacement| {
            placement.validation = validate_placement(
                &PlacedMesh {
                    rotation: placement.rotation,
                    scale: placement.scale,
                    translation: placement.translation,
                    vertices: vertices.map(Vec::as_slice).unwrap_or_default(),
                    reference_frame: placement.reference_frame,
                },
                &views,
                points,
            );
        };
        let mut proposal = place_one(
            request,
            &package.observed_points,
            bounds,
            camera,
            up,
            &artifact.supports,
        );
        if let Ok(p) = &mut proposal {
            validate(p);
        }
        // Sparse landmarks can cover only one chair leg. If that hypothesis
        // misses the *source* silhouette, estimate extent from the measured
        // mask at the depth of observed foreground landmarks instead.
        let fits_source = |p: &ObjectPlacement| {
            p.validation.views.iter().any(|v| {
                v.frame_index == request.frame_index
                    && v.verdict == crate::placement_validation::ViewVerdict::Consistent
            })
        };
        if !proposal.as_ref().is_ok_and(fits_source) {
            if let Some(obs) = observations.get(request.observation.as_str()) {
                if let Some(mut fitted) =
                    place_from_mask(request, obs, bounds, camera, up, &artifact.supports, points)
                {
                    validate(&mut fitted);
                    if proposal.is_err() || fits_source(&fitted) {
                        proposal = Ok(fitted);
                    }
                }
            }
        }
        let eligible = identities.objects.iter().any(|o| {
            o.id == request.object && o.motion != crate::object_identity::MotionClass::Moving
        }) && !identities
            .ambiguities
            .iter()
            .any(|a| a.objects.contains(&request.object));
        if eligible {
            if let (Some(vertices), Some(faces)) = (
                vertices,
                mesh_faces.and_then(|m| m.get(&request.request_id)),
            ) {
                if let Some((mut fitted, heldout, iou)) = multi_view_fit::fit(
                    request,
                    bounds,
                    vertices,
                    faces,
                    &views,
                    up,
                    &artifact.supports,
                    points,
                    proposal.as_ref().ok(),
                    frames,
                ) {
                    validate(&mut fitted);
                    let check = fitted
                        .validation
                        .views
                        .iter()
                        .find(|v| v.frame_index == heldout);
                    let agrees = check.is_some_and(|v| {
                        v.verdict == crate::placement_validation::ViewVerdict::Consistent
                    }) && iou >= 0.4;
                    if !agrees {
                        fitted.validation.verdict = PlacementVerdict::Contradicted;
                        fitted.validation.needs_review = true;
                    }
                    let matched_depths = fitted
                        .silhouette_registration
                        .as_ref()
                        .map_or(0, |r| r.matched_patch_depths);
                    fitted.validation.reason = format!(
                        "Held-out frame {heldout} silhouette IoU {iou:.2}. {matched_depths} image patch depths constrained fitting. {}",
                        fitted.validation.reason
                    );
                    proposal = Ok(fitted);
                }
            }
        }
        let placement = match proposal {
            Ok(p) => p,
            Err(reason) => {
                artifact.unplaced.push(unplaced(reason));
                continue;
            }
        };
        let height = placement.scale * (bounds.max[2] - bounds.min[2]);
        if let Some(prior) = size_prior(&request.label) {
            if height > 0.0 {
                hints.push((height / prior, request.label.clone()));
            }
        }
        artifact.placements.push(placement);
    }
    if !hints.is_empty() {
        hints.sort_by(|a, b| a.0.total_cmp(&b.0));
        artifact.metric_scale_hint = Some(hints[hints.len() / 2].0);
        let mut labels: Vec<String> = hints.into_iter().map(|(_, label)| label).collect();
        labels.sort();
        labels.dedup();
        artifact.metric_hint_labels = labels;
    }
    artifact
}

// Only the primary frame participates in this fit. Other registered views
// continue to test placement, including possible occlusion/depth conflicts.
fn place_from_mask(
    request: &crate::object_generation::GenerationRequest,
    obs: &crate::semantic_inventory::SemanticObservation,
    bounds: MeshBounds,
    camera: &FrameCamera,
    up: [f64; 3],
    supports: &[SupportPlane],
    scene_points: &[[f64; 3]],
) -> Option<ObjectPlacement> {
    let mask = obs.mask.as_ref()?;
    let width = u64::from(mask.width);
    let (mut x0, mut y0, mut x1, mut y1) = (width, u64::from(mask.height), 0, 0);
    let mut start = 0;
    for (index, count) in mask.counts.iter().enumerate() {
        let end = start + count;
        if index % 2 == 1 && end > start {
            let first_row = start / width;
            let last_row = (end - 1) / width;
            if first_row != last_row {
                x0 = 0;
                x1 = width;
            } else {
                x0 = x0.min(start % width);
                x1 = x1.max((end - 1) % width + 1);
            }
            y0 = y0.min(first_row);
            y1 = y1.max(last_row + 1);
        }
        start = end;
    }
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let mut front = front_points(obs, camera, scene_points);
    if front.is_empty() {
        return None;
    }
    front.sort_by(|a, b| a.0.total_cmp(&b.0));
    let z = front[front.len() / 2].0;
    let (eye, right, down, forward) = camera_axes(camera);
    let footprint = [[x0, y0], [x0, y1], [x1, y0], [x1, y1]].map(|[x, y]| {
        let u = (x as f64 - camera.principal_point[0]) / camera.focal_pixels;
        let v = (y as f64 - camera.principal_point[1]) / camera.focal_pixels;
        add(
            eye,
            scaled(add(forward, add(scaled(right, u), scaled(down, v))), z),
        )
    });
    let mut placed = place_one(request, &footprint, bounds, camera, up, supports).ok()?;
    placed.observed_points = front.len();
    placed.scale_source = ScaleSource::MaskAtObservedDepth;
    placed.reason = format!(
        "Reference mask extent at depth {z:.3} supported by {} foreground landmarks; {:?}",
        front.len(),
        placed.contact
    );
    Some(placed)
}

/// Place one candidate mesh from its object's observed points and reference
/// camera; `Err` explains why it cannot be placed.
pub(crate) fn place_one(
    request: &crate::object_generation::GenerationRequest,
    observed: &[[f64; 3]],
    bounds: MeshBounds,
    camera: &FrameCamera,
    up: [f64; 3],
    supports: &[SupportPlane],
) -> Result<ObjectPlacement, String> {
    let (_, right, down, forward) = camera_axes(camera);
    // Mesh frame (camera on +x, +y right, +z up) onto the reference camera.
    let base = nalgebra::Matrix3::from_columns(&[
        nalgebra::Vector3::from(scaled(forward, -1.0)),
        nalgebra::Vector3::from(right),
        nalgebra::Vector3::from(scaled(down, -1.0)),
    ]);
    // Tilt so the object's up is the scene up.
    let tilt = rotation_between(scaled(down, -1.0), up);
    let rotation = tilt * base;
    let world = |axis: usize| {
        let column = rotation.column(axis);
        [column[0], column[1], column[2]]
    };
    let (toward_camera, across, object_up) = (world(0), world(1), world(2));
    let extent_along = |axis: [f64; 3]| {
        let values: Vec<f64> = observed.iter().map(|point| dot(*point, axis)).collect();
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (low, high)
    };
    let mesh_size = [0, 1, 2].map(|axis| bounds.max[axis] - bounds.min[axis]);
    let mesh_centre = [0, 1, 2].map(|axis| (bounds.max[axis] + bounds.min[axis]) / 2.0);
    let (across_low, across_high) = extent_along(across);
    let (up_low, up_high) = extent_along(object_up);
    let (near_low, near_high) = extent_along(toward_camera);
    let ratios: Vec<f64> = [
        (across_high - across_low, mesh_size[1]),
        (up_high - up_low, mesh_size[2]),
    ]
    .into_iter()
    .filter(|(seen, model)| seen.is_finite() && *seen > 0.0 && *model > 1e-9)
    .map(|(seen, model)| seen / model)
    .collect();
    if ratios.is_empty() || observed.len() < 3 {
        return Err(format!(
            "{} observed point(s) do not span the object: no observed scale",
            observed.len()
        ));
    }
    let scale = ratios.iter().sum::<f64>() / ratios.len() as f64;
    // Centre across and up from the observed span; the visible surface is
    // the side facing the camera, so the mesh centre sits half its depth
    // behind the nearest observed surface.
    let surface = near_high.max(near_low);
    let depth_centre = surface - scale * mesh_size[0] / 2.0;
    let centre_scene = add(
        add(
            scaled(across, (across_low + across_high) / 2.0),
            scaled(object_up, (up_low + up_high) / 2.0),
        ),
        scaled(toward_camera, depth_centre),
    );
    // translation = centre_scene − scale·R·mesh_centre
    let rotated_centre = rotation * nalgebra::Vector3::from(mesh_centre);
    let translation = sub(
        centre_scene,
        scaled(
            [rotated_centre[0], rotated_centre[1], rotated_centre[2]],
            scale,
        ),
    );
    Ok(placed_transform(
        request,
        rotation,
        scale,
        translation,
        centre_scene,
        object_up,
        mesh_size[2],
        supports,
        ScaleSource::Observed,
        observed.len(),
        format!(
            "Scale {scale:.4} from {} observed width/height samples in frame {}",
            observed.len(),
            request.frame_index
        ),
    ))
}

#[allow(clippy::too_many_arguments)]
fn placed_transform(
    request: &crate::object_generation::GenerationRequest,
    rotation: nalgebra::Matrix3<f64>,
    scale: f64,
    translation: [f64; 3],
    centre_scene: [f64; 3],
    object_up: [f64; 3],
    model_height: f64,
    supports: &[SupportPlane],
    scale_source: ScaleSource,
    observed_points: usize,
    reason: String,
) -> ObjectPlacement {
    let height = scale * model_height;
    let bottom = sub(centre_scene, scaled(object_up, height / 2.0));
    // Nearest support below the object whose points lie under it.
    let mut support = None;
    for plane in supports {
        if plane.object == request.object {
            continue;
        }
        let gap = dot(plane.normal, bottom) - plane.offset;
        if gap >= -CONTACT_TOLERANCE * height * 3.0
            && support
                .as_ref()
                .is_none_or(|(_, best): &(&SupportPlane, f64)| gap.abs() < best.abs())
        {
            support = Some((plane, gap));
        }
    }
    let tolerance = CONTACT_TOLERANCE * height;
    let (contact, support_object, contact_gap, contact_reason) = match support {
        Some((plane, gap)) if gap.abs() <= tolerance => (
            ContactState::InContact,
            Some(plane.object.clone()),
            Some(gap),
            format!(
                "rests on {} ({}) within {:.0}% of its height",
                plane.label,
                plane.object,
                CONTACT_TOLERANCE * 100.0
            ),
        ),
        Some((plane, gap)) if gap > 0.0 => (
            ContactState::Floating,
            Some(plane.object.clone()),
            Some(gap),
            format!(
                "floats {:.3} scene units above {} ({})",
                gap, plane.label, plane.object
            ),
        ),
        Some((plane, gap)) => (
            ContactState::Sunken,
            Some(plane.object.clone()),
            Some(gap),
            format!(
                "sinks {:.3} scene units into {} ({})",
                -gap, plane.label, plane.object
            ),
        ),
        None => (
            ContactState::NoContactEvidence,
            None,
            None,
            "no horizontal floor or furniture support was observed below it".into(),
        ),
    };
    ObjectPlacement {
        object: request.object.clone(),
        label: request.label.clone(),
        generation_request: request.request_id.clone(),
        rotation: [
            rotation[(0, 0)],
            rotation[(0, 1)],
            rotation[(0, 2)],
            rotation[(1, 0)],
            rotation[(1, 1)],
            rotation[(1, 2)],
            rotation[(2, 0)],
            rotation[(2, 1)],
            rotation[(2, 2)],
        ],
        scale,
        translation,
        scale_source,
        silhouette_registration: None,
        observed_points,
        reference_frame: request.frame_index,
        contact,
        support: support_object,
        contact_gap,
        reason: format!("{reason}; {contact_reason}"),
        validation: PlacementValidation {
            verdict: PlacementVerdict::Approximate,
            needs_review: true,
            views: Vec::new(),
            reason: "not validated against other views".into(),
        },
    }
}

#[cfg(test)]
#[path = "object_placement_tests.rs"]
mod tests;
