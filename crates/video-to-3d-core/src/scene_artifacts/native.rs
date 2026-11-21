//! Native adapters for the existing classic stages. Scheduling and persistence
//! remain owned by scene_runner and ProjectStore; video sampling stays external.
use super::*;
use crate::scene_model::*;
use crate::{CameraStageArtifact, DenseStageArtifact, FrameInput, ReconstructionRequest};

impl BuiltInExecutor {
    /// Prefer the repository's pinned local camera provider for staged native builds.
    /// `VIDEO_TO_3D_CAMERA_SIDECAR` can select an installed runtime.
    pub fn with_local_camera_registration(mut self) -> Self {
        let bundled =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/colmap-sparse/run.sh");
        self.camera_sidecar = Some(
            std::env::var_os("VIDEO_TO_3D_CAMERA_SIDECAR")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    if bundled.is_file() {
                        bundled
                    } else {
                        "video-to-3d-camera-sidecar".into()
                    }
                }),
        );
        self
    }

    fn register_cameras(
        &self,
        program: &Path,
        frames: &ReconstructionRequest,
        mut keyframes: KeyframesArtifact,
        cancel: &CancellationToken,
    ) -> Result<CameraStageArtifact, String> {
        use super::sidecar::{run_sidecar, Exchange, SidecarRun};
        if frames.frames.len() < 3 {
            return Err("COLMAP registration needs at least three sampled frames".into());
        }
        let exchange = Exchange::new("camera-registration")?;
        for (index, (record, frame)) in keyframes.frames.iter_mut().zip(&frames.frames).enumerate()
        {
            if cancel.is_canceled() {
                return Err("native registration canceled".into());
            }
            // Pass the exact verified bytes, with paths relative to the private exchange.
            record.pixels = ProjectPath::new(format!("{index}.rgba")).map_err(|e| e.to_string())?;
            fs::write(record.pixels.resolve(&exchange.0), &frame.rgba)
                .map_err(|e| e.to_string())?;
        }
        let request = exchange.0.join("keyframes.json");
        let response = exchange.0.join("registered.json");
        fs::write(&request, keyframes.to_json()).map_err(|e| e.to_string())?;
        match run_sidecar("COLMAP", program, &request, &response, &exchange.0, cancel)? {
            SidecarRun::Response(bytes) => {
                let registered: crate::registered_sparse::RegisteredSparseInput =
                    serde_json::from_slice(&bytes)
                        .map_err(|e| format!("invalid COLMAP response: {e}"))?;
                registered.camera_stage(frames, || cancel.is_canceled())
            }
            SidecarRun::NotStarted(reason) => Err(reason),
            SidecarRun::Canceled => Err("native registration canceled".into()),
        }
    }

    fn manifest(&self) -> Result<SceneProjectManifest, String> {
        match &self.reservations {
            Reservations::Fixed { manifest } => manifest
                .read()
                .map(|snapshot| snapshot.clone())
                .map_err(|_| "executor manifest lock poisoned".into()),
            Reservations::ManifestFile(path) => {
                SceneProjectManifest::load(path).map_err(|e| e.to_string())
            }
        }
    }

    /// Placement consumes the same browser frames as its generated-object input.
    /// Follow that operation's declared parent even if the project has other clips.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn generation_keyframes(
        &self,
        generation: &crate::scene_project::ArtifactRecord,
        cancel: &CancellationToken,
    ) -> Result<KeyframesArtifact, String> {
        let manifest = self.manifest()?;
        let identities = manifest.operation_identities().map_err(|e| e.to_string())?;
        if identities.get(&generation.produced_by) != Some(&generation.operation_identity) {
            return Err("generated objects have stale operation identity".into());
        }
        let operation = manifest
            .operations
            .iter()
            .find(|op| op.id == generation.produced_by)
            .ok_or("object generation producer is missing")?;
        let mut inputs = operation.inputs.iter().filter_map(|input| match input {
            crate::scene_project::OperationInput::Operation(id) => manifest
                .artifacts
                .iter()
                .find(|record| record.produced_by == *id && record.kind == ArtifactKind::Keyframes),
            _ => None,
        });
        let record = inputs
            .next()
            .ok_or("generated objects have no keyframes ancestor")?;
        if inputs.next().is_some()
            || identities.get(&record.produced_by) != Some(&record.operation_identity)
        {
            return Err("generated objects require one current keyframes ancestor".into());
        }
        KeyframesArtifact::from_json_with_cancel(&self.document(record, cancel)?, || {
            if cancel.is_canceled() {
                Err("object placement canceled".into())
            } else {
                Ok(())
            }
        })
    }

    pub(super) fn input<'a>(
        &self,
        request: &'a OperationRequest,
        kind: ArtifactKind,
    ) -> Result<&'a crate::scene_project::ArtifactRecord, String> {
        let mut found = request.inputs.iter().filter_map(|input| match input {
            ResolvedInput::Artifact(record) if record.kind == kind => Some(record),
            _ => None,
        });
        let record = found
            .next()
            .ok_or_else(|| format!("native stage needs one {kind:?} input"))?;
        if found.next().is_some() {
            return Err(format!("native stage has multiple {kind:?} inputs"));
        }
        Ok(record)
    }

    pub(super) fn document(
        &self,
        record: &crate::scene_project::ArtifactRecord,
        cancel: &CancellationToken,
    ) -> Result<String, String> {
        let bytes = read_verified_with_cancel(
            &self.root,
            &record.path,
            &record.content_hash,
            None,
            || cancel.is_canceled(),
        )?;
        String::from_utf8(bytes).map_err(|e| e.to_string())
    }

    fn frames(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> Result<(ReconstructionRequest, KeyframesArtifact), String> {
        let keyframes = KeyframesArtifact::from_json_with_cancel(
            &self.document(self.input(request, ArtifactKind::Keyframes)?, cancel)?,
            || {
                if cancel.is_canceled() {
                    Err("native keyframes read canceled".into())
                } else {
                    Ok(())
                }
            },
        )?;
        let mut frames = Vec::with_capacity(keyframes.frames.len());
        for (index, frame) in keyframes.frames.iter().enumerate() {
            if cancel.is_canceled() {
                return Err("native frame loading canceled".into());
            }
            // The classic request uses contiguous sampled-frame ordinals.
            if frame.frame_index != index {
                return Err(
                    "classic keyframes must have contiguous frame indices starting at zero".into(),
                );
            }
            frames.push(FrameInput {
                width: frame.width,
                height: frame.height,
                rgba: keyframes
                    .load_pixels_with_cancel(&self.root, index, || cancel.is_canceled())?,
            });
        }
        Ok((
            ReconstructionRequest {
                frames,
                options: Default::default(),
            },
            keyframes,
        ))
    }

    /// Resolve the declared sparse parent of dense through the reconciled
    /// manifest, verifying both its operation identity and content identity.
    /// Never scan unrelated files or substitute an independently solved camera.
    fn dense_cameras(
        &self,
        dense_record: &crate::scene_project::ArtifactRecord,
        cameras_hash: &ContentHash,
        cancel: &CancellationToken,
    ) -> Result<CameraStageArtifact, String> {
        let manifest = self.manifest()?;
        let identities = manifest.operation_identities().map_err(|e| e.to_string())?;
        let operation = manifest
            .operations
            .iter()
            .find(|op| op.id == dense_record.produced_by)
            .ok_or("dense producer is missing from manifest")?;
        let mut cameras = Vec::new();
        for input in &operation.inputs {
            if let crate::scene_project::OperationInput::Operation(id) = input {
                if let Some(record) = manifest
                    .artifacts
                    .iter()
                    .find(|r| r.produced_by == *id && r.kind == ArtifactKind::SparseReconstruction)
                {
                    if identities.get(id) != Some(&record.operation_identity) {
                        return Err("camera ancestor has stale operation identity".into());
                    }
                    let camera = CameraStageArtifact::from_json_with_cancel(
                        &self.document(record, cancel)?,
                        || cancel.is_canceled(),
                    )
                    .map_err(|e| e.to_string())?;
                    if camera
                        .content_hash_with_cancel(|| cancel.is_canceled())
                        .map_err(|error| error.to_string())?
                        != *cameras_hash
                    {
                        return Err("dense camera ancestor content hash does not match".into());
                    }
                    cameras.push(camera);
                }
            }
        }
        if cameras.len() != 1 {
            return Err("dense needs exactly one verified sparse camera ancestor".into());
        }
        Ok(cameras.remove(0))
    }

    pub(super) fn native_stage(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> Option<AttemptOutcome> {
        let expected: &[ArtifactKind] = match request.operation.kind {
            OperationKind::SparseReconstruction => &[ArtifactKind::Keyframes],
            OperationKind::DenseReconstruction => {
                &[ArtifactKind::Keyframes, ArtifactKind::SparseReconstruction]
            }
            OperationKind::SurfaceMesh => &[ArtifactKind::DenseEvidence],
            OperationKind::SceneAssembly => &[
                ArtifactKind::SparseReconstruction,
                ArtifactKind::SurfaceMesh,
            ],
            _ => return None,
        };
        if request.inputs.len() != expected.len()
            || expected.iter().any(|kind| {
                request.inputs.iter().filter(|input| {
                matches!(input, ResolvedInput::Artifact(record) if record.kind == *kind)
            }).count() != 1
            })
        {
            return Some(AttemptOutcome::Unsupported(
                "native classic stage has unsupported extra or missing inputs".into(),
            ));
        }
        if cancel.is_canceled() {
            return Some(AttemptOutcome::Canceled);
        }
        let directory = format!("artifacts/native/{}", request.operation.id);
        match self.collision(&request.operation.id, &directory) {
            Ok(Some(path)) => {
                return Some(AttemptOutcome::Unsupported(format!(
                    "native output would overwrite declared path {}",
                    path.as_str()
                )))
            }
            Err(message) => return Some(AttemptOutcome::Failed(message)),
            Ok(None) => {}
        }
        let result = self.run_native_stage(request, cancel);
        Some(match result {
            Ok(produced) => AttemptOutcome::Succeeded(produced),
            Err(_) if cancel.is_canceled() => AttemptOutcome::Canceled,
            Err(message) => AttemptOutcome::Failed(message),
        })
    }

    fn run_native_stage(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> Result<ProducedArtifact, String> {
        let directory = format!("artifacts/native/{}", request.operation.id);
        let output =
            ProjectPath::new(format!("{directory}/artifact.json")).map_err(|e| e.to_string())?;
        let mut observations = BTreeMap::new();
        let mut reproducibility = Reproducibility::Deterministic;
        let document = match request.operation.kind {
            OperationKind::SparseReconstruction => {
                let (frames, keyframes) = self.frames(request, cancel)?;
                let classic = || {
                    crate::reconstruct_cameras_with_cancel(&frames, || cancel.is_canceled())
                        .map_err(|e| e.to_string())
                };
                let cameras = if let Some(program) = &self.camera_sidecar {
                    match self.register_cameras(program, &frames, keyframes, cancel) {
                        Ok(cameras) => {
                            reproducibility = Reproducibility::SeededStochastic;
                            observations.insert("random_seed".into(), "0".into());
                            observations.insert("camera_provider".into(), "colmap".into());
                            observations.insert(
                                "diagnostic".into(),
                                cameras.sparse.bootstrap.summary.clone(),
                            );
                            cameras
                        }
                        Err(reason) if cancel.is_canceled() => return Err(reason),
                        Err(reason) => {
                            observations.insert(
                                "camera_provider".into(),
                                crate::ReconstructionProviderDescriptor::classic().id,
                            );
                            observations.insert(
                                "diagnostic".into(),
                                format!("Using Rust camera registration. Local COLMAP: {reason}"),
                            );
                            classic()?
                        }
                    }
                } else {
                    classic()?
                };
                cameras
                    .to_json_with_cancel(|| cancel.is_canceled())
                    .map_err(|error| error.to_string())?
            }
            OperationKind::DenseReconstruction => {
                let cameras = CameraStageArtifact::from_json_with_cancel(
                    &self.document(
                        self.input(request, ArtifactKind::SparseReconstruction)?,
                        cancel,
                    )?,
                    || cancel.is_canceled(),
                )
                .map_err(|e| e.to_string())?;
                crate::reconstruct_dense_with_cancel(
                    &self.frames(request, cancel)?.0,
                    &cameras,
                    || cancel.is_canceled(),
                )
                .map_err(|e| e.to_string())?
                .to_json_with_cancel(|| cancel.is_canceled())
                .map_err(|error| error.to_string())?
            }
            OperationKind::SurfaceMesh => {
                let record = self.input(request, ArtifactKind::DenseEvidence)?;
                let dense = DenseStageArtifact::from_json_with_cancel(
                    &self.document(record, cancel)?,
                    || cancel.is_canceled(),
                )
                .map_err(|e| e.to_string())?;
                let cameras = self.dense_cameras(record, &dense.cameras_hash, cancel)?;
                let (mut dense, mesh) =
                    crate::reconstruct_mesh_owned_with_cancel(&cameras, dense, || {
                        cancel.is_canceled()
                    })
                    .map_err(|e| e.to_string())?;
                // Derived (clipped) vertices follow the observed dense points.
                dense
                    .dense
                    .points
                    .extend_from_slice(&mesh.mesh.derived_points);
                let final_cameras: Vec<_> = cameras
                    .sparse
                    .final_cameras
                    .iter()
                    .map(|camera| camera.registered())
                    .collect();
                let evidence = ReconstructionEvidenceView::from_classic_parts_with_cancel(
                    &crate::reconstruction_evidence::ClassicEvidenceParts {
                        calibrated_pair: cameras.sparse.calibrated_pair.as_ref(),
                        registered_views: &cameras.sparse.registered_views,
                        final_cameras: &final_cameras,
                        reference_patches: &dense.dense.stats.reference_patches,
                        points: &dense.dense.points,
                        point_attributes: &dense.dense.point_attributes,
                        triangles: &mesh.mesh.triangles,
                        triangle_sources: &mesh.mesh.triangle_sources,
                        derived_vertices: &mesh.mesh.derived_vertices,
                    },
                    || {
                        if cancel.is_canceled() {
                            Err("native mesh canceled".into())
                        } else {
                            Ok(())
                        }
                    },
                )?;
                let provider = evidence.provider;
                let scale = evidence.scale;
                let mut evidence_cameras = evidence.cameras;
                if cameras.registration_provider.is_some() {
                    for camera in &mut evidence_cameras {
                        camera.authority = EvidenceCameraAuthority::ProviderEstimated;
                    }
                }
                let regions = evidence.regions;
                let artifact = SurfaceMeshArtifact {
                    schema_version: SURFACE_MESH_ARTIFACT_SCHEMA_VERSION,
                    provider,
                    scale,
                    cameras: evidence_cameras,
                    regions,
                    points: dense.dense.points,
                    triangles: mesh.mesh.triangles,
                    point_attributes: Some(dense.dense.point_attributes),
                    triangle_sources: mesh.mesh.triangle_sources,
                    derived_vertices: mesh.mesh.derived_vertices,
                    grid_sites: dense.dense.grid_sites,
                };
                artifact.to_json_with_cancel(|| {
                    if cancel.is_canceled() {
                        Err("native surface mesh write canceled".into())
                    } else {
                        Ok(())
                    }
                })?
            }
            OperationKind::SceneAssembly => crate::controlled_json::to_string(
                &self.assemble(request, &directory, cancel)?,
                true,
                true,
                &mut || {
                    if cancel.is_canceled() {
                        Err("native assembly write canceled".into())
                    } else {
                        Ok(())
                    }
                },
            )?,
            _ => return Err("unsupported native stage".into()),
        };
        if cancel.is_canceled() {
            return Err("native stage canceled".into());
        }
        write_atomically(&output.resolve(&self.root), document.as_bytes())?;
        let mut produced = ProducedArtifact::new(
            output,
            ContentHash::of_bytes(document.as_bytes()),
            reproducibility,
        );
        produced.observations = observations;
        Ok(produced)
    }

    fn assemble(
        &self,
        request: &OperationRequest,
        directory: &str,
        cancel: &CancellationToken,
    ) -> Result<AssembledScene, String> {
        let sparse_record = self.input(request, ArtifactKind::SparseReconstruction)?;
        let sparse = CameraStageArtifact::from_json_with_cancel(
            &self.document(sparse_record, cancel)?,
            || cancel.is_canceled(),
        )
        .map_err(|e| e.to_string())?;
        let mesh_record = self.input(request, ArtifactKind::SurfaceMesh)?;
        // Validate the full declared mesh -> dense -> sparse ancestry rather
        // than pairing arbitrary camera and surface artifacts.
        let manifest = self.manifest()?;
        let mesh_op = manifest
            .operations
            .iter()
            .find(|op| op.id == mesh_record.produced_by)
            .ok_or("mesh producer missing")?;
        let dense_id = match mesh_op.inputs.as_slice() {
            [crate::scene_project::OperationInput::Operation(id)] => id,
            _ => return Err("classic mesh needs one dense ancestor".into()),
        };
        let dense_record = manifest
            .artifacts
            .iter()
            .find(|r| r.produced_by == *dense_id && r.kind == ArtifactKind::DenseEvidence)
            .ok_or("dense ancestor missing")?;
        // Assembly needs identity metadata, not a second dense geometry buffer.
        // The document/receipt already crossed store verification; deserialize
        // only this versioned ancestry header, skipping dense arrays.
        #[derive(Deserialize)]
        struct DenseIdentity {
            schema_version: u32,
            request_hash: ContentHash,
            cameras_hash: ContentHash,
        }
        let dense: DenseIdentity =
            crate::controlled_json::from_str(&self.document(dense_record, cancel)?, &mut || {
                if cancel.is_canceled() {
                    Err("native dense ancestry read canceled".into())
                } else {
                    Ok(())
                }
            })?;
        if dense.schema_version != crate::DENSE_STAGE_ARTIFACT_SCHEMA_VERSION {
            return Err("unsupported dense ancestor schema version".into());
        }
        let identities = manifest
            .operation_identities()
            .map_err(|error| error.to_string())?;
        if identities.get(&dense_record.produced_by) != Some(&dense_record.operation_identity) {
            return Err("dense ancestor has stale operation identity".into());
        }
        if dense.request_hash != sparse.request_hash
            || self
                .dense_cameras(dense_record, &dense.cameras_hash, cancel)?
                .content_hash_with_cancel(|| cancel.is_canceled())
                .map_err(|error| error.to_string())?
                != sparse
                    .content_hash_with_cancel(|| cancel.is_canceled())
                    .map_err(|error| error.to_string())?
        {
            return Err("scene camera does not match mesh ancestry".into());
        }
        let check = || {
            if cancel.is_canceled() {
                Err("native assembly canceled".into())
            } else {
                Ok(())
            }
        };
        let mesh = SurfaceMeshArtifact::from_json_with_cancel(
            &self.document(mesh_record, cancel)?,
            check,
        )?;
        // Parsing validated this exact owned artifact once with the active token.
        let evidence = mesh.evidence_view();
        if mesh.triangles.is_empty() {
            return Err("no observed surface triangles available for scene assembly".into());
        }
        // Existing GLB encoder's untextured vertex-color fallback preserves
        // accepted geometry; this stage does not invent a texture bake.
        let bake = crate::surface_materials::SurfaceMaterialBake {
            schema_version: SURFACE_MATERIAL_BAKE_SCHEMA_VERSION,
            point_count: mesh.points.len(),
            triangle_count: mesh.triangles.len(),
            materials: Vec::new(),
            fallback: crate::surface_materials::FallbackSurface {
                triangles: (0..mesh.triangles.len()).collect(),
                reasons: FallbackReasons {
                    missing_reference_image: mesh.triangles.len(),
                    ..Default::default()
                },
            },
            canonicalized_same_reference_triangles: 0,
        };
        let bytes = crate::textured_glb::encode_textured_glb_with_cancel(&evidence, &bake, || {
            cancel.is_canceled()
        })?;
        let path =
            ProjectPath::new(format!("{directory}/environment.glb")).map_err(|e| e.to_string())?;
        let content_hash = ContentHash::of_bytes(&bytes);
        let cameras = evidence.cameras.iter().map(scene_camera).collect();
        let provenance: Vec<_> = evidence
            .regions
            .iter()
            .map(|region| SceneProvenance::from(region.origin))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let source_frames = evidence
            .cameras
            .iter()
            .map(|camera| camera.frame_index)
            .collect();
        let scene = AssembledScene {
            schema_version: ASSEMBLED_SCENE_SCHEMA_VERSION,
            frame: SceneFrame {
                unit: SceneUnit::ArbitraryMonocular,
                handedness: Handedness::RightHanded,
                up_axis: UpAxis::PositiveY,
            },
            cameras,
            resources: vec![SceneResource {
                id: "observed-environment".into(),
                kind: ResourceKind::Mesh,
                path: path.clone(),
                content_hash,
                provenance,
                confidence: None,
                source_frames,
            }],
            materials: Vec::new(),
            assets: vec![SceneAsset {
                id: "environment".into(),
                role: AssetRole::Environment,
                parent: None,
                transform: Transform::IDENTITY,
                visual: Some(VisualRepresentation {
                    mesh: Some(MeshVisual {
                        mesh: "observed-environment".into(),
                        material: None,
                    }),
                    splat: None,
                }),
                collision: None,
                labels: Vec::new(),
                collision_provenance: None,
            }],
            lights: Vec::new(),
            audio: Vec::new(),
        };
        scene
            .validate_with_cancel(&mut || {
                if cancel.is_canceled() {
                    Err("native scene validation canceled".into())
                } else {
                    Ok(())
                }
            })
            .map_err(|e| e.to_string())?;
        if cancel.is_canceled() {
            return Err("scene assembly canceled".into());
        }
        write_atomically(&path.resolve(&self.root), &bytes)?;
        Ok(scene)
    }
}

/// Evidence stores world-to-camera R,t in the seed's Y-down frame. The
/// canonical scene and existing GLB use C=diag(1,-1,-1); camera-to-scene is
/// C R^T C and position -C R^T t.
fn scene_camera(camera: &EvidenceCamera) -> SceneCamera {
    use nalgebra::{Matrix3, UnitQuaternion, Vector3};
    let r = Matrix3::from_row_slice(&camera.rotation);
    let c = Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0));
    let rotation = UnitQuaternion::from_matrix(&(c * r.transpose() * c));
    let translation = -c * r.transpose() * Vector3::from(camera.translation);
    SceneCamera {
        id: format!("camera-{}", camera.frame_index),
        source_frame: camera.frame_index,
        authority: camera.authority.into(),
        transform: Transform {
            translation: translation.into(),
            rotation: [rotation.i, rotation.j, rotation.k, rotation.w],
            scale: [1.0; 3],
        },
        intrinsics: None,
    }
}
