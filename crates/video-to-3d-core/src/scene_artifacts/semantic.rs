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
        Ok((frame_cameras, points))
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
