//! Native adapter of the `semantic_inventory` provider operation (#60).
//!
//! The model runtime is an external local program (the semantic sidecar,
//! protocol [`SIDECAR_PROTOCOL`]): Rust offers it the browser keyframes and
//! the vocabulary, the program answers with detections, masks and tracks (or
//! an honest `unavailable`), and Rust validates the answer with the shared
//! [`SemanticInventoryArtifact`] rules before anything is recorded. Swapping
//! the runtime (in-process ONNX, browser inference) changes only how the
//! answer is produced, never the artifact.
//!
//! The program is resolved from [`BuiltInExecutor::with_semantic_sidecar`],
//! else the [`SIDECAR_ENVIRONMENT`] variable, else [`DEFAULT_SIDECAR`] on the
//! `PATH`. A program that cannot be started records an unavailable result:
//! segmentation is an optional capability (Atlas 4.6), so the rest of the
//! build stays usable.
use super::*;
use crate::semantic_inventory::{
    default_room_vocabulary, SegmentationStatus, SemanticFrame, SemanticInventoryArtifact,
    SemanticMeasurements, SemanticModel, SemanticObservation, SemanticProvider, SemanticRuntime,
    SemanticTrack, SemanticUncertainty, SemanticVocabulary, VocabularySource,
    SEMANTIC_INVENTORY_SCHEMA_VERSION,
};
use serde::Deserialize;
use std::process::{Command, Stdio};
use std::time::Duration;

pub const SIDECAR_PROTOCOL: &str = "video-to-3d-semantic-sidecar/1";
/// Environment variable naming the sidecar executable.
pub const SIDECAR_ENVIRONMENT: &str = "VIDEO_TO_3D_SEMANTIC_SIDECAR";
/// The sidecar looked up on the `PATH` when nothing else names one; the
/// repository ships it as `tools/semantic-sidecar/video-to-3d-semantic-sidecar`.
pub const DEFAULT_SIDECAR: &str = "video-to-3d-semantic-sidecar";
/// Largest sidecar response accepted (masks of many frames are large but
/// bounded).
const MAX_RESPONSE_BYTES: u64 = 512 << 20;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AvailableResponse {
    protocol: String,
    #[allow(dead_code)]
    status: String,
    models: Vec<SemanticModel>,
    runtime: SemanticRuntime,
    observations: Vec<SemanticObservation>,
    tracks: Vec<SemanticTrack>,
    uncertainties: Vec<SemanticUncertainty>,
    measurements: Option<SemanticMeasurements>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UnavailableResponse {
    protocol: String,
    #[allow(dead_code)]
    status: String,
    reason: String,
}

/// A scratch directory for one sidecar exchange, removed on drop.
struct Exchange(PathBuf);

impl Exchange {
    fn new(operation: &str) -> Result<Self, String> {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "video-to-3d-semantic-{operation}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        fs::create_dir_all(&path).map_err(|error| {
            format!("cannot create sidecar exchange {}: {error}", path.display())
        })?;
        Ok(Self(path))
    }
}

impl Drop for Exchange {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

enum SidecarRun {
    Response(Vec<u8>),
    NotStarted(String),
    Canceled,
}

impl BuiltInExecutor {
    /// Run semantic inventory with this sidecar program instead of the
    /// environment variable or the default.
    pub fn with_semantic_sidecar(mut self, program: impl Into<PathBuf>) -> Self {
        self.semantic_sidecar = Some(program.into());
        self
    }

    fn sidecar_program(&self) -> PathBuf {
        self.semantic_sidecar
            .clone()
            .or_else(|| std::env::var_os(SIDECAR_ENVIRONMENT).map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(DEFAULT_SIDECAR))
    }

    pub(super) fn semantic_inventory(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> AttemptOutcome {
        if cancel.is_canceled() {
            return AttemptOutcome::Canceled;
        }
        let directory = format!("artifacts/native/{}", request.operation.id);
        match self.collision(&request.operation.id, &directory) {
            Ok(Some(path)) => {
                return AttemptOutcome::Unsupported(format!(
                    "semantic output would overwrite declared path {}",
                    path.as_str()
                ))
            }
            Err(message) => return AttemptOutcome::Failed(message),
            Ok(None) => {}
        }
        match self.run_semantic_inventory(request, cancel, &directory) {
            Ok(Some(produced)) => AttemptOutcome::Succeeded(produced),
            Ok(None) => AttemptOutcome::Canceled,
            Err(_) if cancel.is_canceled() => AttemptOutcome::Canceled,
            Err(message) => AttemptOutcome::Failed(message),
        }
    }

    fn run_semantic_inventory(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
        directory: &str,
    ) -> Result<Option<ProducedArtifact>, String> {
        let provider = request
            .provider
            .as_ref()
            .ok_or("semantic inventory needs its declared provider")?;
        let keyframes = KeyframesArtifact::from_json_with_cancel(
            &self.document(self.input(request, ArtifactKind::Keyframes)?, cancel)?,
            || {
                if cancel.is_canceled() {
                    Err("semantic keyframes read canceled".into())
                } else {
                    Ok(())
                }
            },
        )?;
        let (source, terms) = match &request.operation.vocabulary {
            Some(terms) => (VocabularySource::Project, terms.clone()),
            None => (VocabularySource::Default, default_room_vocabulary()),
        };
        let revision = provider.revision.clone().unwrap_or_default();
        let mut frames = Vec::with_capacity(keyframes.frames.len());
        let mut offered = Vec::with_capacity(keyframes.frames.len());
        for frame in &keyframes.frames {
            if cancel.is_canceled() {
                return Ok(None);
            }
            let path = frame.pixels.resolve(&self.root);
            let expected = u64::from(frame.width) * u64::from(frame.height) * 4;
            match crate::scene_store::hash_file(&path) {
                Some((hash, length)) if hash == frame.content_hash && length == expected => {}
                _ => {
                    return Err(format!(
                        "keyframe {} pixels do not match their recorded identity",
                        frame.frame_index
                    ))
                }
            }
            let absolute = fs::canonicalize(&path).map_err(|error| error.to_string())?;
            offered.push(serde_json::json!({
                "frame_index": frame.frame_index,
                "width": frame.width,
                "height": frame.height,
                "content_hash": frame.content_hash.as_str(),
                "rgba_path": absolute,
            }));
            frames.push(SemanticFrame {
                frame_index: frame.frame_index,
                width: frame.width,
                height: frame.height,
                content_hash: frame.content_hash.clone(),
            });
        }
        let exchange = Exchange::new(&request.operation.id)?;
        let request_path = exchange.0.join("request.json");
        let response_path = exchange.0.join("response.json");
        let document = serde_json::json!({
            "protocol": SIDECAR_PROTOCOL,
            "provider": {"id": provider.id, "revision": revision},
            "vocabulary": terms,
            "frames": offered,
        });
        fs::write(&request_path, document.to_string()).map_err(|error| error.to_string())?;

        let program = self.sidecar_program();
        let mut artifact = SemanticInventoryArtifact {
            schema_version: SEMANTIC_INVENTORY_SCHEMA_VERSION,
            status: SegmentationStatus::Unavailable,
            unavailable_reason: None,
            provider: SemanticProvider {
                id: provider.id.clone(),
                revision,
                models: Vec::new(),
                runtime: None,
            },
            vocabulary: SemanticVocabulary { source, terms },
            frames,
            observations: Vec::new(),
            tracks: Vec::new(),
            uncertainties: Vec::new(),
            measurements: None,
        };
        match run_sidecar(&program, &request_path, &response_path, &exchange.0, cancel)? {
            SidecarRun::Canceled => return Ok(None),
            SidecarRun::NotStarted(reason) => artifact.unavailable_reason = Some(reason),
            SidecarRun::Response(bytes) => {
                let value: serde_json::Value = serde_json::from_slice(&bytes)
                    .map_err(|error| format!("semantic sidecar response is not JSON: {error}"))?;
                match value.get("status").and_then(serde_json::Value::as_str) {
                    Some("available") => {
                        let response: AvailableResponse = serde_json::from_value(value)
                            .map_err(|error| format!("semantic sidecar response: {error}"))?;
                        check_protocol(&response.protocol)?;
                        artifact.status = SegmentationStatus::Available;
                        artifact.provider.models = response.models;
                        artifact.provider.runtime = Some(response.runtime);
                        artifact.observations = response.observations;
                        artifact.tracks = response.tracks;
                        artifact.uncertainties = response.uncertainties;
                        artifact.measurements = response.measurements;
                    }
                    Some("unavailable") => {
                        let response: UnavailableResponse = serde_json::from_value(value)
                            .map_err(|error| format!("semantic sidecar response: {error}"))?;
                        check_protocol(&response.protocol)?;
                        artifact.unavailable_reason = Some(response.reason);
                    }
                    _ => return Err("semantic sidecar response has no valid status".into()),
                }
            }
        }
        artifact.validate()?;
        artifact.validate_against_keyframes(&keyframes)?;
        if cancel.is_canceled() {
            return Ok(None);
        }
        let output =
            ProjectPath::new(format!("{directory}/artifact.json")).map_err(|e| e.to_string())?;
        let document = artifact.to_json();
        write_atomically(&output.resolve(&self.root), document.as_bytes())?;
        let available = artifact.status == SegmentationStatus::Available;
        let mut produced = ProducedArtifact::new(
            output,
            ContentHash::of_bytes(document.as_bytes()),
            if available {
                Reproducibility::NonDeterministic
            } else {
                Reproducibility::Provisional
            },
        );
        produced.observations.insert(
            "segmentation".into(),
            if available {
                "available"
            } else {
                "unavailable"
            }
            .into(),
        );
        if let Some(reason) = &artifact.unavailable_reason {
            produced
                .observations
                .insert("segmentation_reason".into(), reason.clone());
        } else {
            produced.observations.insert(
                "semantic_observations".into(),
                artifact.observations.len().to_string(),
            );
            produced
                .observations
                .insert("semantic_tracks".into(), artifact.tracks.len().to_string());
        }
        Ok(Some(produced))
    }
}

fn check_protocol(protocol: &str) -> Result<(), String> {
    if protocol == SIDECAR_PROTOCOL {
        Ok(())
    } else {
        Err(format!(
            "semantic sidecar speaks `{protocol}`, expected `{SIDECAR_PROTOCOL}`"
        ))
    }
}

/// Run the sidecar to completion, killing and reaping it on cancellation.
fn run_sidecar(
    program: &Path,
    request: &Path,
    response: &Path,
    scratch: &Path,
    cancel: &CancellationToken,
) -> Result<SidecarRun, String> {
    let stderr_path = scratch.join("stderr.log");
    let stderr = fs::File::create(&stderr_path).map_err(|error| error.to_string())?;
    let mut child = match Command::new(program)
        .arg(request)
        .arg(response)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr)
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return Ok(SidecarRun::NotStarted(format!(
                "semantic segmentation is unavailable: the local sidecar `{}` could not be started ({error})",
                program.display()
            )))
        }
    };
    let status = loop {
        if cancel.is_canceled() {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(SidecarRun::Canceled);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("semantic sidecar wait failed: {error}"));
            }
        }
    };
    if !status.success() {
        let log = fs::read_to_string(&stderr_path).unwrap_or_default();
        let tail: String = log
            .lines()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" / ");
        return Err(format!("semantic sidecar exited with {status}: {tail}"));
    }
    let length = fs::metadata(response)
        .map_err(|error| format!("semantic sidecar wrote no response: {error}"))?
        .len();
    if length > MAX_RESPONSE_BYTES {
        return Err(format!(
            "semantic sidecar response of {length} bytes exceeds the {MAX_RESPONSE_BYTES}-byte limit"
        ));
    }
    fs::read(response)
        .map(SidecarRun::Response)
        .map_err(|error| format!("semantic sidecar response is unreadable: {error}"))
}

impl BuiltInExecutor {
    /// Built-in object identities (#61): semantic evidence localized with the
    /// accepted cameras and surface, read-only.
    pub(super) fn object_identity(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> AttemptOutcome {
        if cancel.is_canceled() {
            return AttemptOutcome::Canceled;
        }
        let directory = format!("artifacts/native/{}", request.operation.id);
        match self.collision(&request.operation.id, &directory) {
            Ok(Some(path)) => {
                return AttemptOutcome::Unsupported(format!(
                    "object identity output would overwrite declared path {}",
                    path.as_str()
                ))
            }
            Err(message) => return AttemptOutcome::Failed(message),
            Ok(None) => {}
        }
        match self.run_object_identity(request, cancel, &directory) {
            Ok(produced) => AttemptOutcome::Succeeded(produced),
            Err(_) if cancel.is_canceled() => AttemptOutcome::Canceled,
            Err(message) => AttemptOutcome::Failed(message),
        }
    }

    fn run_object_identity(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
        directory: &str,
    ) -> Result<ProducedArtifact, String> {
        let semantic_record = self.input(request, ArtifactKind::SemanticInventory)?;
        let inventory =
            SemanticInventoryArtifact::from_json(&self.document(semantic_record, cancel)?)?;
        let (frame_cameras, points) = self.registered_geometry(request, cancel)?;
        if cancel.is_canceled() {
            return Err("object identity canceled".into());
        }
        let artifact = crate::object_identity::identify_objects(
            &inventory,
            semantic_record.content_hash.clone(),
            &frame_cameras,
            &points,
        );
        artifact.validate()?;
        let output =
            ProjectPath::new(format!("{directory}/artifact.json")).map_err(|e| e.to_string())?;
        let document = artifact.to_json();
        write_atomically(&output.resolve(&self.root), document.as_bytes())?;
        let mut produced = ProducedArtifact::new(
            output,
            ContentHash::of_bytes(document.as_bytes()),
            Reproducibility::Deterministic,
        );
        let count = |motion: crate::object_identity::MotionClass| {
            artifact
                .objects
                .iter()
                .filter(|object| object.motion == motion)
                .count()
                .to_string()
        };
        use crate::object_identity::MotionClass;
        produced
            .observations
            .insert("objects".into(), artifact.objects.len().to_string());
        produced.observations.insert(
            "static_structure".into(),
            count(MotionClass::StaticStructure),
        );
        produced.observations.insert(
            "movable_stationary".into(),
            count(MotionClass::MovableStationary),
        );
        produced
            .observations
            .insert("moving".into(), count(MotionClass::Moving));
        produced
            .observations
            .insert("motion_unknown".into(), count(MotionClass::Unknown));
        produced
            .observations
            .insert("ambiguities".into(), artifact.ambiguities.len().to_string());
        if let Some(reason) = &artifact.unavailable_reason {
            produced
                .observations
                .insert("identity_reason".into(), reason.clone());
        }
        Ok(produced)
    }
}

impl BuiltInExecutor {
    /// Registered pinhole cameras and accepted surface points, read-only.
    fn registered_geometry(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> Result<(Vec<crate::object_identity::FrameCamera>, Vec<[f64; 3]>), String> {
        let (cameras, points, _) = self.registered_geometry_with_scale(request, cancel)?;
        Ok((cameras, points))
    }

    /// [`Self::registered_geometry`] plus the scene's evidence scale.
    #[allow(clippy::type_complexity)]
    fn registered_geometry_with_scale(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> Result<
        (
            Vec<crate::object_identity::FrameCamera>,
            Vec<[f64; 3]>,
            crate::EvidenceScale,
        ),
        String,
    > {
        let cameras = crate::CameraStageArtifact::from_json_with_cancel(
            &self.document(
                self.input(request, ArtifactKind::SparseReconstruction)?,
                cancel,
            )?,
            || cancel.is_canceled(),
        )
        .map_err(|error| error.to_string())?;
        let mesh = SurfaceMeshArtifact::from_json_with_cancel(
            &self.document(self.input(request, ArtifactKind::SurfaceMesh)?, cancel)?,
            || {
                if cancel.is_canceled() {
                    Err("object geometry read canceled".into())
                } else {
                    Ok(())
                }
            },
        )?;
        let sparse = &cameras.sparse;
        let principal = [
            f64::from(sparse.image_width) / 2.0,
            f64::from(sparse.image_height) / 2.0,
        ];
        let frame_cameras = mesh
            .cameras
            .iter()
            .map(|camera| crate::object_identity::FrameCamera {
                frame_index: camera.frame_index,
                rotation: camera.rotation.map(f64::from),
                translation: camera.translation.map(f64::from),
                focal_pixels: f64::from(sparse.focal_length_pixels),
                principal_point: principal,
            })
            .collect();
        let points = mesh
            .points
            .iter()
            .map(|point| [f64::from(point.x), f64::from(point.y), f64::from(point.z)])
            .collect();
        Ok((frame_cameras, points, mesh.scale))
    }

    /// Built-in cross-view mask consistency (#62), read-only.
    pub(super) fn object_masks(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> AttemptOutcome {
        if cancel.is_canceled() {
            return AttemptOutcome::Canceled;
        }
        let directory = format!("artifacts/native/{}", request.operation.id);
        match self.collision(&request.operation.id, &directory) {
            Ok(Some(path)) => {
                return AttemptOutcome::Unsupported(format!(
                    "object masks output would overwrite declared path {}",
                    path.as_str()
                ))
            }
            Err(message) => return AttemptOutcome::Failed(message),
            Ok(None) => {}
        }
        let result = (|| -> Result<ProducedArtifact, String> {
            let semantic_record = self.input(request, ArtifactKind::SemanticInventory)?;
            let inventory =
                SemanticInventoryArtifact::from_json(&self.document(semantic_record, cancel)?)?;
            let identity_record = self.input(request, ArtifactKind::ObjectIdentity)?;
            let identities = crate::object_identity::ObjectIdentityArtifact::from_json(
                &self.document(identity_record, cancel)?,
            )?;
            if identities.semantic_inventory != semantic_record.content_hash {
                return Err(
                    "object identities were derived from another semantic inventory".into(),
                );
            }
            let (cameras, points) = self.registered_geometry(request, cancel)?;
            if cancel.is_canceled() {
                return Err("object masks canceled".into());
            }
            let artifact = crate::object_masks::check_object_masks(
                &inventory,
                semantic_record.content_hash.clone(),
                &identities,
                identity_record.content_hash.clone(),
                &cameras,
                &points,
            );
            artifact.validate()?;
            let output = ProjectPath::new(format!("{directory}/artifact.json"))
                .map_err(|e| e.to_string())?;
            let document = artifact.to_json();
            write_atomically(&output.resolve(&self.root), document.as_bytes())?;
            let mut produced = ProducedArtifact::new(
                output,
                ContentHash::of_bytes(document.as_bytes()),
                Reproducibility::Deterministic,
            );
            use crate::object_masks::MaskConsistency;
            for (key, class) in [
                ("consistent_masks", MaskConsistency::Consistent),
                ("inconsistent_masks", MaskConsistency::Inconsistent),
                ("unverified_masks", MaskConsistency::Unverified),
                ("missing_masks", MaskConsistency::MissingMask),
            ] {
                produced.observations.insert(
                    key.into(),
                    artifact
                        .masks
                        .iter()
                        .filter(|mask| mask.consistency == class)
                        .count()
                        .to_string(),
                );
            }
            Ok(produced)
        })();
        match result {
            Ok(produced) => AttemptOutcome::Succeeded(produced),
            Err(_) if cancel.is_canceled() => AttemptOutcome::Canceled,
            Err(message) => AttemptOutcome::Failed(message),
        }
    }
}

impl BuiltInExecutor {
    /// Built-in multi-view object reference packages (#63), read-only.
    pub(super) fn object_references(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> AttemptOutcome {
        if cancel.is_canceled() {
            return AttemptOutcome::Canceled;
        }
        let directory = format!("artifacts/native/{}", request.operation.id);
        match self.collision(&request.operation.id, &directory) {
            Ok(Some(path)) => {
                return AttemptOutcome::Unsupported(format!(
                    "object references output would overwrite declared path {}",
                    path.as_str()
                ))
            }
            Err(message) => return AttemptOutcome::Failed(message),
            Ok(None) => {}
        }
        let result = (|| -> Result<ProducedArtifact, String> {
            let semantic_record = self.input(request, ArtifactKind::SemanticInventory)?;
            let inventory =
                SemanticInventoryArtifact::from_json(&self.document(semantic_record, cancel)?)?;
            let identity_record = self.input(request, ArtifactKind::ObjectIdentity)?;
            let identities = crate::object_identity::ObjectIdentityArtifact::from_json(
                &self.document(identity_record, cancel)?,
            )?;
            let masks_record = self.input(request, ArtifactKind::ObjectMasks)?;
            let masks = crate::object_masks::ObjectMasksArtifact::from_json(
                &self.document(masks_record, cancel)?,
            )?;
            if identities.semantic_inventory != semantic_record.content_hash
                || masks.semantic_inventory != semantic_record.content_hash
                || masks.object_identity != identity_record.content_hash
            {
                return Err(
                    "object references inputs come from different semantic evidence".into(),
                );
            }
            let (cameras, points, scale) = self.registered_geometry_with_scale(request, cancel)?;
            if cancel.is_canceled() {
                return Err("object references canceled".into());
            }
            let artifact = crate::object_references::package_object_references(
                &inventory,
                semantic_record.content_hash.clone(),
                &identities,
                identity_record.content_hash.clone(),
                &masks,
                masks_record.content_hash.clone(),
                &cameras,
                &points,
                scale,
            );
            artifact.validate()?;
            let output = ProjectPath::new(format!("{directory}/artifact.json"))
                .map_err(|e| e.to_string())?;
            let document = artifact.to_json();
            write_atomically(&output.resolve(&self.root), document.as_bytes())?;
            let mut produced = ProducedArtifact::new(
                output,
                ContentHash::of_bytes(document.as_bytes()),
                Reproducibility::Deterministic,
            );
            produced
                .observations
                .insert("packages".into(), artifact.packages.len().to_string());
            produced
                .observations
                .insert("unpackaged".into(), artifact.unpackaged.len().to_string());
            Ok(produced)
        })();
        match result {
            Ok(produced) => AttemptOutcome::Succeeded(produced),
            Err(_) if cancel.is_canceled() => AttemptOutcome::Canceled,
            Err(message) => AttemptOutcome::Failed(message),
        }
    }
}

/// Protocol of the local clean-plate (inpainting) sidecar.
pub const CLEAN_PLATE_PROTOCOL: &str = "video-to-3d-clean-plate-sidecar/1";
/// Environment variable naming the clean-plate sidecar executable.
pub const CLEAN_PLATE_ENVIRONMENT: &str = "VIDEO_TO_3D_CLEAN_PLATE_SIDECAR";
/// The clean-plate sidecar looked up on the `PATH` by default; the repository
/// ships it as `tools/semantic-sidecar/video-to-3d-clean-plate-sidecar`.
pub const DEFAULT_CLEAN_PLATE_SIDECAR: &str = "video-to-3d-clean-plate-sidecar";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlateEditResult {
    edit_id: String,
    status: String,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlateResponse {
    protocol: String,
    status: String,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    models: Vec<SemanticModel>,
    #[serde(default)]
    runtime: Option<SemanticRuntime>,
    #[serde(default)]
    edits: Vec<PlateEditResult>,
}

impl BuiltInExecutor {
    /// Run clean plates with this sidecar program instead of the environment
    /// variable or the default.
    pub fn with_clean_plate_sidecar(mut self, program: impl Into<PathBuf>) -> Self {
        self.clean_plate_sidecar = Some(program.into());
        self
    }

    /// Clean-plate references (#64) from a local inpainting sidecar.
    pub(super) fn clean_plate(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> AttemptOutcome {
        if cancel.is_canceled() {
            return AttemptOutcome::Canceled;
        }
        let directory = format!("artifacts/native/{}", request.operation.id);
        match self.collision(&request.operation.id, &directory) {
            Ok(Some(path)) => {
                return AttemptOutcome::Unsupported(format!(
                    "clean plate output would overwrite declared path {}",
                    path.as_str()
                ))
            }
            Err(message) => return AttemptOutcome::Failed(message),
            Ok(None) => {}
        }
        match self.run_clean_plate(request, cancel, &directory) {
            Ok(Some(produced)) => AttemptOutcome::Succeeded(produced),
            Ok(None) => AttemptOutcome::Canceled,
            Err(_) if cancel.is_canceled() => AttemptOutcome::Canceled,
            Err(message) => AttemptOutcome::Failed(message),
        }
    }

    fn run_clean_plate(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
        directory: &str,
    ) -> Result<Option<ProducedArtifact>, String> {
        use crate::clean_plate::{
            plan_clean_plates, CleanPlate, CleanPlateArtifact, PlateClassification, PlateOutput,
            PlateProvider, PlateStatus, CLEAN_PLATE_SCHEMA_VERSION,
        };
        let provider = request
            .provider
            .as_ref()
            .ok_or("clean plates need their declared provider")?;
        let revision = provider.revision.clone().unwrap_or_default();
        let keyframes = KeyframesArtifact::from_json_with_cancel(
            &self.document(self.input(request, ArtifactKind::Keyframes)?, cancel)?,
            || {
                if cancel.is_canceled() {
                    Err("clean plate keyframes read canceled".into())
                } else {
                    Ok(())
                }
            },
        )?;
        let inventory = SemanticInventoryArtifact::from_json(&self.document(
            self.input(request, ArtifactKind::SemanticInventory)?,
            cancel,
        )?)?;
        inventory.validate_against_keyframes(&keyframes)?;
        let references_record = self.input(request, ArtifactKind::ObjectReferences)?;
        let references = crate::object_references::ObjectReferencesArtifact::from_json(
            &self.document(references_record, cancel)?,
        )?;
        let requests = plan_clean_plates(&inventory, &references, (&provider.id, &revision));
        let mut artifact = CleanPlateArtifact {
            schema_version: CLEAN_PLATE_SCHEMA_VERSION,
            status: PlateStatus::Available,
            unavailable_reason: None,
            provider: PlateProvider {
                id: provider.id.clone(),
                revision: revision.clone(),
                models: Vec::new(),
                runtime: None,
            },
            object_references: references_record.content_hash.clone(),
            plates: Vec::new(),
        };
        let exchange = Exchange::new(&request.operation.id)?;
        let mut edits = Vec::new();
        for plan in &requests {
            let keyframe = keyframes
                .frames
                .iter()
                .find(|frame| frame.frame_index == plan.frame_index)
                .ok_or("clean plate frame is not a keyframe")?;
            let source = fs::canonicalize(keyframe.pixels.resolve(&self.root))
                .map_err(|error| error.to_string())?;
            edits.push(serde_json::json!({
                "edit_id": plan.edit_id,
                "width": plan.width,
                "height": plan.height,
                "rgba_path": source,
                "mask_counts": plan.removal_mask.counts,
                "output_path": exchange.0.join(format!("{}.rgba", plan.edit_id)),
            }));
        }
        let mut available = true;
        if !requests.is_empty() {
            let request_path = exchange.0.join("request.json");
            let response_path = exchange.0.join("response.json");
            fs::write(
                &request_path,
                serde_json::json!({
                    "protocol": CLEAN_PLATE_PROTOCOL,
                    "provider": {"id": provider.id, "revision": revision},
                    "edits": edits,
                })
                .to_string(),
            )
            .map_err(|error| error.to_string())?;
            let program = self
                .clean_plate_sidecar
                .clone()
                .or_else(|| std::env::var_os(CLEAN_PLATE_ENVIRONMENT).map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from(DEFAULT_CLEAN_PLATE_SIDECAR));
            let results: BTreeMap<String, PlateEditResult> =
                match run_sidecar(&program, &request_path, &response_path, &exchange.0, cancel)? {
                    SidecarRun::Canceled => return Ok(None),
                    SidecarRun::NotStarted(reason) => {
                        available = false;
                        artifact.unavailable_reason = Some(reason.replace(
                            "semantic segmentation is unavailable",
                            "clean plates are unavailable",
                        ));
                        BTreeMap::new()
                    }
                    SidecarRun::Response(bytes) => {
                        let response: PlateResponse = serde_json::from_slice(&bytes)
                            .map_err(|error| format!("clean plate sidecar response: {error}"))?;
                        if response.protocol != CLEAN_PLATE_PROTOCOL {
                            return Err(format!(
                            "clean plate sidecar speaks `{}`, expected `{CLEAN_PLATE_PROTOCOL}`",
                            response.protocol
                        ));
                        }
                        match response.status.as_str() {
                            "available" => {
                                artifact.provider.models = response.models;
                                artifact.provider.runtime = response.runtime;
                                response
                                    .edits
                                    .into_iter()
                                    .map(|edit| (edit.edit_id.clone(), edit))
                                    .collect()
                            }
                            "unavailable" => {
                                available = false;
                                artifact.unavailable_reason = Some(
                                    response
                                        .reason
                                        .filter(|reason| !reason.trim().is_empty())
                                        .unwrap_or_else(|| "clean plates are unavailable".into()),
                                );
                                BTreeMap::new()
                            }
                            other => return Err(format!("clean plate sidecar status `{other}`")),
                        }
                    }
                };
            for plan in requests {
                if cancel.is_canceled() {
                    return Ok(None);
                }
                let (output, failure) = if !available {
                    (None, Some("clean plates were unavailable".into()))
                } else {
                    match results.get(&plan.edit_id) {
                        Some(result) if result.status == "ok" => {
                            let produced = exchange.0.join(format!("{}.rgba", plan.edit_id));
                            let bytes = fs::read(&produced).map_err(|error| {
                                format!(
                                    "clean plate {} output is unreadable: {error}",
                                    plan.edit_id
                                )
                            })?;
                            if bytes.len() as u64
                                != u64::from(plan.width) * u64::from(plan.height) * 4
                            {
                                return Err(format!(
                                    "clean plate {} output has the wrong size",
                                    plan.edit_id
                                ));
                            }
                            let path = ProjectPath::new(format!(
                                "{directory}/plates/{}.rgba",
                                plan.edit_id
                            ))
                            .map_err(|e| e.to_string())?;
                            write_atomically(&path.resolve(&self.root), &bytes)?;
                            (
                                Some(PlateOutput {
                                    path,
                                    content_hash: ContentHash::of_bytes(&bytes),
                                }),
                                None,
                            )
                        }
                        Some(result) => (
                            None,
                            Some(
                                result
                                    .reason
                                    .clone()
                                    .filter(|reason| !reason.trim().is_empty())
                                    .unwrap_or_else(|| "inpainting failed".into()),
                            ),
                        ),
                        None => (
                            None,
                            Some("the sidecar returned no result for this edit".into()),
                        ),
                    }
                };
                artifact.plates.push(CleanPlate {
                    request: plan,
                    classification: PlateClassification::GeneratedEdit,
                    output,
                    failure,
                });
            }
        }
        if !available {
            artifact.status = PlateStatus::Unavailable;
        }
        artifact.validate()?;
        let output =
            ProjectPath::new(format!("{directory}/artifact.json")).map_err(|e| e.to_string())?;
        let document = artifact.to_json();
        write_atomically(&output.resolve(&self.root), document.as_bytes())?;
        let mut produced = ProducedArtifact::new(
            output,
            ContentHash::of_bytes(document.as_bytes()),
            if available {
                Reproducibility::NonDeterministic
            } else {
                Reproducibility::Provisional
            },
        );
        let succeeded = artifact
            .plates
            .iter()
            .filter(|plate| plate.output.is_some())
            .count();
        produced
            .observations
            .insert("clean_plates".into(), succeeded.to_string());
        produced.observations.insert(
            "failed_plates".into(),
            (artifact.plates.len() - succeeded).to_string(),
        );
        if let Some(reason) = &artifact.unavailable_reason {
            produced
                .observations
                .insert("clean_plate_reason".into(), reason.clone());
        }
        Ok(Some(produced))
    }
}

/// Protocol of the local object-generation (image-to-3D) sidecar.
pub const GENERATION_PROTOCOL: &str = "video-to-3d-object-generation-sidecar/1";
/// Environment variable naming the object-generation sidecar executable.
pub const GENERATION_ENVIRONMENT: &str = "VIDEO_TO_3D_OBJECT_GENERATION_SIDECAR";
/// The object-generation sidecar looked up on the `PATH` by default; the
/// repository ships it as `tools/semantic-sidecar/video-to-3d-object-generation-sidecar`.
pub const DEFAULT_GENERATION_SIDECAR: &str = "video-to-3d-object-generation-sidecar";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationResult {
    request_id: String,
    status: String,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    vertices: Option<u64>,
    #[serde(default)]
    triangles: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationResponse {
    protocol: String,
    status: String,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    models: Vec<SemanticModel>,
    #[serde(default)]
    runtime: Option<SemanticRuntime>,
    #[serde(default)]
    results: Vec<GenerationResult>,
}

impl BuiltInExecutor {
    /// Run object generation with this sidecar program instead of the
    /// environment variable or the default.
    pub fn with_object_generation_sidecar(mut self, program: impl Into<PathBuf>) -> Self {
        self.generation_sidecar = Some(program.into());
        self
    }

    /// Generated candidate meshes (#68) from a local image-to-3D sidecar.
    pub(super) fn object_generation(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> AttemptOutcome {
        if cancel.is_canceled() {
            return AttemptOutcome::Canceled;
        }
        let directory = format!("artifacts/native/{}", request.operation.id);
        match self.collision(&request.operation.id, &directory) {
            Ok(Some(path)) => {
                return AttemptOutcome::Unsupported(format!(
                    "object generation output would overwrite declared path {}",
                    path.as_str()
                ))
            }
            Err(message) => return AttemptOutcome::Failed(message),
            Ok(None) => {}
        }
        match self.run_object_generation(request, cancel, &directory) {
            Ok(Some(produced)) => AttemptOutcome::Succeeded(produced),
            Ok(None) => AttemptOutcome::Canceled,
            Err(_) if cancel.is_canceled() => AttemptOutcome::Canceled,
            Err(message) => AttemptOutcome::Failed(message),
        }
    }

    fn run_object_generation(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
        directory: &str,
    ) -> Result<Option<ProducedArtifact>, String> {
        use crate::object_generation::{
            plan_generation, GeneratedClassification, GeneratedMesh, GeneratedObject,
            GenerationProvider, GenerationStatus, ObjectGenerationArtifact,
            OBJECT_GENERATION_SCHEMA_VERSION,
        };
        let provider = request
            .provider
            .as_ref()
            .ok_or("object generation needs its declared provider")?;
        let revision = provider.revision.clone().unwrap_or_default();
        let keyframes = KeyframesArtifact::from_json_with_cancel(
            &self.document(self.input(request, ArtifactKind::Keyframes)?, cancel)?,
            || {
                if cancel.is_canceled() {
                    Err("object generation keyframes read canceled".into())
                } else {
                    Ok(())
                }
            },
        )?;
        let inventory = SemanticInventoryArtifact::from_json(&self.document(
            self.input(request, ArtifactKind::SemanticInventory)?,
            cancel,
        )?)?;
        inventory.validate_against_keyframes(&keyframes)?;
        let references_record = self.input(request, ArtifactKind::ObjectReferences)?;
        let references = crate::object_references::ObjectReferencesArtifact::from_json(
            &self.document(references_record, cancel)?,
        )?;
        let plans = plan_generation(&inventory, &references, (&provider.id, &revision));
        let mut artifact = ObjectGenerationArtifact {
            schema_version: OBJECT_GENERATION_SCHEMA_VERSION,
            status: GenerationStatus::Available,
            unavailable_reason: None,
            provider: GenerationProvider {
                id: provider.id.clone(),
                revision: revision.clone(),
                models: Vec::new(),
                runtime: None,
            },
            object_references: references_record.content_hash.clone(),
            objects: Vec::new(),
        };
        let exchange = Exchange::new(&request.operation.id)?;
        let mut available = true;
        if !plans.is_empty() {
            let mut entries = Vec::new();
            for plan in &plans {
                let keyframe = keyframes
                    .frames
                    .iter()
                    .find(|frame| frame.frame_index == plan.frame_index)
                    .ok_or("generation frame is not a keyframe")?;
                let source = fs::canonicalize(keyframe.pixels.resolve(&self.root))
                    .map_err(|error| error.to_string())?;
                entries.push(serde_json::json!({
                    "request_id": plan.request_id,
                    "label": plan.label,
                    "width": plan.width,
                    "height": plan.height,
                    "rgba_path": source,
                    "crop": plan.crop,
                    "mask_counts": plan.mask.counts,
                    "output_path": exchange.0.join(format!("{}.glb", plan.request_id)),
                }));
            }
            let request_path = exchange.0.join("request.json");
            let response_path = exchange.0.join("response.json");
            fs::write(
                &request_path,
                serde_json::json!({
                    "protocol": GENERATION_PROTOCOL,
                    "provider": {"id": provider.id, "revision": revision},
                    "requests": entries,
                })
                .to_string(),
            )
            .map_err(|error| error.to_string())?;
            let program = self
                .generation_sidecar
                .clone()
                .or_else(|| std::env::var_os(GENERATION_ENVIRONMENT).map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from(DEFAULT_GENERATION_SIDECAR));
            let results: BTreeMap<String, GenerationResult> = match run_sidecar(
                &program,
                &request_path,
                &response_path,
                &exchange.0,
                cancel,
            )? {
                SidecarRun::Canceled => return Ok(None),
                SidecarRun::NotStarted(reason) => {
                    available = false;
                    artifact.unavailable_reason = Some(reason.replace(
                        "semantic segmentation is unavailable",
                        "object generation is unavailable",
                    ));
                    BTreeMap::new()
                }
                SidecarRun::Response(bytes) => {
                    let response: GenerationResponse = serde_json::from_slice(&bytes)
                        .map_err(|error| format!("object generation sidecar response: {error}"))?;
                    if response.protocol != GENERATION_PROTOCOL {
                        return Err(format!(
                                "object generation sidecar speaks `{}`, expected `{GENERATION_PROTOCOL}`",
                                response.protocol
                            ));
                    }
                    match response.status.as_str() {
                        "available" => {
                            artifact.provider.models = response.models;
                            artifact.provider.runtime = response.runtime;
                            response
                                .results
                                .into_iter()
                                .map(|result| (result.request_id.clone(), result))
                                .collect()
                        }
                        "unavailable" => {
                            available = false;
                            artifact.unavailable_reason = Some(
                                response
                                    .reason
                                    .filter(|reason| !reason.trim().is_empty())
                                    .unwrap_or_else(|| "object generation is unavailable".into()),
                            );
                            BTreeMap::new()
                        }
                        other => return Err(format!("object generation sidecar status `{other}`")),
                    }
                }
            };
            for plan in plans {
                if cancel.is_canceled() {
                    return Ok(None);
                }
                let (output, failure) = if !available {
                    (None, Some("object generation was unavailable".into()))
                } else {
                    match results.get(&plan.request_id) {
                        Some(result) if result.status == "ok" => {
                            let produced = exchange.0.join(format!("{}.glb", plan.request_id));
                            let bytes = fs::read(&produced).map_err(|error| {
                                format!("generated mesh {} is unreadable: {error}", plan.request_id)
                            })?;
                            if bytes.len() < 12 || &bytes[..4] != b"glTF" {
                                (None, Some("the sidecar wrote no binary glTF".to_owned()))
                            } else if result.triangles.unwrap_or(0) == 0 {
                                (None, Some("the generated mesh is empty".to_owned()))
                            } else {
                                let path = ProjectPath::new(format!(
                                    "{directory}/objects/{}.glb",
                                    plan.request_id
                                ))
                                .map_err(|e| e.to_string())?;
                                write_atomically(&path.resolve(&self.root), &bytes)?;
                                (
                                    Some(GeneratedMesh {
                                        path,
                                        content_hash: ContentHash::of_bytes(&bytes),
                                        vertices: result.vertices.unwrap_or(0),
                                        triangles: result.triangles.unwrap_or(0),
                                    }),
                                    None,
                                )
                            }
                        }
                        Some(result) => (
                            None,
                            Some(
                                result
                                    .reason
                                    .clone()
                                    .filter(|reason| !reason.trim().is_empty())
                                    .unwrap_or_else(|| "generation failed".into()),
                            ),
                        ),
                        None => (
                            None,
                            Some("the sidecar returned no result for this object".into()),
                        ),
                    }
                };
                artifact.objects.push(GeneratedObject {
                    request: plan,
                    classification: GeneratedClassification::GenerativeCandidate,
                    output,
                    failure,
                });
            }
        }
        if !available {
            artifact.status = GenerationStatus::Unavailable;
        }
        artifact.validate()?;
        let output =
            ProjectPath::new(format!("{directory}/artifact.json")).map_err(|e| e.to_string())?;
        let document = artifact.to_json();
        write_atomically(&output.resolve(&self.root), document.as_bytes())?;
        let mut produced = ProducedArtifact::new(
            output,
            ContentHash::of_bytes(document.as_bytes()),
            if available {
                Reproducibility::NonDeterministic
            } else {
                Reproducibility::Provisional
            },
        );
        let succeeded = artifact
            .objects
            .iter()
            .filter(|object| object.output.is_some())
            .count();
        produced
            .observations
            .insert("generated_objects".into(), succeeded.to_string());
        produced.observations.insert(
            "failed_generations".into(),
            (artifact.objects.len() - succeeded).to_string(),
        );
        if let Some(reason) = &artifact.unavailable_reason {
            produced
                .observations
                .insert("generation_reason".into(), reason.clone());
        }
        Ok(Some(produced))
    }
}

/// POSITION bounds of the first mesh primitive of a binary glTF.
fn glb_position_bounds(bytes: &[u8]) -> Result<crate::object_placement::MeshBounds, String> {
    if bytes.len() < 20 || &bytes[..4] != b"glTF" {
        return Err("not a binary glTF".into());
    }
    let length = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let json: serde_json::Value = serde_json::from_slice(
        bytes
            .get(20..20 + length)
            .ok_or("binary glTF JSON chunk is truncated")?,
    )
    .map_err(|error| format!("binary glTF JSON chunk: {error}"))?;
    let accessor = json["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
        .as_u64()
        .ok_or("binary glTF has no POSITION accessor")?;
    let accessor = &json["accessors"][accessor as usize];
    let read = |key: &str| -> Result<[f64; 3], String> {
        let values = accessor[key]
            .as_array()
            .filter(|values| values.len() == 3)
            .ok_or("POSITION accessor has no bounds")?;
        let mut out = [0.0; 3];
        for (slot, value) in out.iter_mut().zip(values) {
            *slot = value.as_f64().ok_or("POSITION bound is not a number")?;
        }
        Ok(out)
    };
    Ok(crate::object_placement::MeshBounds {
        min: read("min")?,
        max: read("max")?,
    })
}

impl BuiltInExecutor {
    /// Built-in scene placement of generated candidates (#69), read-only.
    pub(super) fn object_placement(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> AttemptOutcome {
        if cancel.is_canceled() {
            return AttemptOutcome::Canceled;
        }
        let directory = format!("artifacts/native/{}", request.operation.id);
        match self.collision(&request.operation.id, &directory) {
            Ok(Some(path)) => {
                return AttemptOutcome::Unsupported(format!(
                    "object placement output would overwrite declared path {}",
                    path.as_str()
                ))
            }
            Err(message) => return AttemptOutcome::Failed(message),
            Ok(None) => {}
        }
        let result = (|| -> Result<ProducedArtifact, String> {
            let inventory = SemanticInventoryArtifact::from_json(&self.document(
                self.input(request, ArtifactKind::SemanticInventory)?,
                cancel,
            )?)?;
            let identities = crate::object_identity::ObjectIdentityArtifact::from_json(
                &self.document(self.input(request, ArtifactKind::ObjectIdentity)?, cancel)?,
            )?;
            let references = crate::object_references::ObjectReferencesArtifact::from_json(
                &self.document(self.input(request, ArtifactKind::ObjectReferences)?, cancel)?,
            )?;
            let generation_record = self.input(request, ArtifactKind::GeneratedObjects)?;
            let generation = crate::object_generation::ObjectGenerationArtifact::from_json(
                &self.document(generation_record, cancel)?,
            )?;
            let mut bounds = BTreeMap::new();
            for object in &generation.objects {
                if let Some(mesh) = &object.output {
                    let bytes = crate::scene_artifacts::read_verified_with_cancel(
                        &self.root,
                        &mesh.path,
                        &mesh.content_hash,
                        None,
                        || cancel.is_canceled(),
                    )?;
                    bounds.insert(
                        object.request.request_id.clone(),
                        glb_position_bounds(&bytes)?,
                    );
                }
            }
            let (cameras, points) = self.registered_geometry(request, cancel)?;
            if cancel.is_canceled() {
                return Err("object placement canceled".into());
            }
            let artifact = crate::object_placement::place_objects(
                &inventory,
                &identities,
                &references,
                &generation,
                generation_record.content_hash.clone(),
                &bounds,
                &cameras,
                &points,
            );
            artifact.validate()?;
            let output = ProjectPath::new(format!("{directory}/artifact.json"))
                .map_err(|e| e.to_string())?;
            let document = artifact.to_json();
            write_atomically(&output.resolve(&self.root), document.as_bytes())?;
            let mut produced = ProducedArtifact::new(
                output,
                ContentHash::of_bytes(document.as_bytes()),
                Reproducibility::Deterministic,
            );
            use crate::object_placement::ContactState;
            produced.observations.insert(
                "placed_objects".into(),
                artifact.placements.len().to_string(),
            );
            produced.observations.insert(
                "unplaced_objects".into(),
                artifact.unplaced.len().to_string(),
            );
            for (key, state) in [
                ("in_contact", ContactState::InContact),
                ("floating", ContactState::Floating),
                ("sunken", ContactState::Sunken),
                ("no_contact_evidence", ContactState::NoContactEvidence),
            ] {
                produced.observations.insert(
                    key.into(),
                    artifact
                        .placements
                        .iter()
                        .filter(|placement| placement.contact == state)
                        .count()
                        .to_string(),
                );
            }
            produced
                .observations
                .insert("up_source".into(), format!("{:?}", artifact.up_source));
            Ok(produced)
        })();
        match result {
            Ok(produced) => AttemptOutcome::Succeeded(produced),
            Err(_) if cancel.is_canceled() => AttemptOutcome::Canceled,
            Err(message) => AttemptOutcome::Failed(message),
        }
    }
}
