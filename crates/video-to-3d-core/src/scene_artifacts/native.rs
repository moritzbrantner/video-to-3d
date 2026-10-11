//! Native adapters for the existing classic stages. Scheduling and persistence
//! remain owned by scene_runner and ProjectStore; video sampling stays external.
use super::*;
use crate::scene_model::*;
use crate::{CameraStageArtifact, DenseStageArtifact, FrameInput, ReconstructionRequest};

impl BuiltInExecutor {
    fn manifest(&self) -> Result<SceneProjectManifest, String> {
        match &self.reservations {
            Reservations::Fixed { manifest, .. } => Ok((**manifest).clone()),
            Reservations::ManifestFile(path) => {
                SceneProjectManifest::load(path).map_err(|e| e.to_string())
            }
        }
    }

    fn input<'a>(
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

    fn document(&self, record: &crate::scene_project::ArtifactRecord) -> Result<String, String> {
        let bytes = read_verified(&self.root, &record.path, &record.content_hash)?;
        String::from_utf8(bytes).map_err(|e| e.to_string())
    }

    fn frames(
        &self,
        request: &OperationRequest,
        cancel: &CancellationToken,
    ) -> Result<ReconstructionRequest, String> {
        let keyframes = KeyframesArtifact::from_json(
            &self.document(self.input(request, ArtifactKind::Keyframes)?)?,
        )?;
        let mut frames = Vec::with_capacity(keyframes.frames.len());
        for (index, frame) in keyframes.frames.iter().enumerate() {
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
        Ok(ReconstructionRequest {
            frames,
            options: Default::default(),
        })
    }

    /// Resolve the declared sparse parent of dense through the reconciled
    /// manifest, verifying both its operation identity and content identity.
    /// Never scan unrelated files or substitute an independently solved camera.
    fn dense_cameras(
        &self,
        dense_record: &crate::scene_project::ArtifactRecord,
        dense: &DenseStageArtifact,
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
                    let camera = CameraStageArtifact::from_json(&self.document(record)?)
                        .map_err(|e| e.to_string())?;
                    if camera.content_hash() != dense.cameras_hash {
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
        let expected = match request.operation.kind {
            OperationKind::SparseReconstruction => 1,
            OperationKind::DenseReconstruction => 2,
            OperationKind::SurfaceMesh => 1,
            OperationKind::SceneAssembly => 2,
            _ => return None,
        };
        if request.inputs.len() != expected {
            return Some(AttemptOutcome::Unsupported(
                "native classic stage has unsupported extra or missing inputs".into(),
            ));
        }
        if cancel.is_canceled() {
            return Some(AttemptOutcome::Canceled);
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
        if let Some(path) = self.collision(&request.operation.id, &directory)? {
            return Err(format!(
                "native output would overwrite declared path {}",
                path.as_str()
            ));
        }
        let output =
            ProjectPath::new(format!("{directory}/artifact.json")).map_err(|e| e.to_string())?;
        let document = match request.operation.kind {
            OperationKind::SparseReconstruction => {
                crate::reconstruct_cameras(&self.frames(request, cancel)?)
                    .map_err(|e| e.to_string())?
                    .to_json()
            }
            OperationKind::DenseReconstruction => {
                let cameras = CameraStageArtifact::from_json(
                    &self.document(self.input(request, ArtifactKind::SparseReconstruction)?)?,
                )
                .map_err(|e| e.to_string())?;
                crate::reconstruct_dense(&self.frames(request, cancel)?, &cameras)
                    .map_err(|e| e.to_string())?
                    .to_json()
            }
            OperationKind::SurfaceMesh => {
                let record = self.input(request, ArtifactKind::DenseEvidence)?;
                let mut dense = DenseStageArtifact::from_json(&self.document(record)?)
                    .map_err(|e| e.to_string())?;
                let cameras = self.dense_cameras(record, &dense)?;
                let mesh = crate::reconstruct_mesh(&cameras, &dense).map_err(|e| e.to_string())?;
                // Dense is owned by this adapter. Apply consolidated positions
                // in place, then move its buffers into external interchange.
                for point in &mesh.mesh.consolidated_points {
                    let target = dense
                        .dense
                        .points
                        .get_mut(point.index)
                        .ok_or("consolidated point outside dense buffer")?;
                    *target = point.point;
                }
                let final_cameras: Vec<_> = cameras
                    .sparse
                    .final_cameras
                    .iter()
                    .map(|camera| camera.registered())
                    .collect();
                let evidence = ReconstructionEvidenceView::from_classic_parts(
                    &crate::reconstruction_evidence::ClassicEvidenceParts {
                        calibrated_pair: cameras.sparse.calibrated_pair.as_ref(),
                        registered_views: &cameras.sparse.registered_views,
                        final_cameras: &final_cameras,
                        reference_patches: &dense.dense.stats.reference_patches,
                        points: &dense.dense.points,
                        point_attributes: &dense.dense.point_attributes,
                        triangles: &mesh.mesh.triangles,
                        triangle_sources: &mesh.mesh.triangle_sources,
                    },
                )?;
                let provider = evidence.provider;
                let scale = evidence.scale;
                let evidence_cameras = evidence.cameras;
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
                    grid_sites: dense.dense.grid_sites,
                };
                artifact.to_json()
            }
            OperationKind::SceneAssembly => {
                to_document(&self.assemble(request, &directory, cancel)?)
            }
            _ => return Err("unsupported native stage".into()),
        };
        if cancel.is_canceled() {
            return Err("native stage canceled".into());
        }
        write_atomically(&output.resolve(&self.root), document.as_bytes())?;
        Ok(ProducedArtifact::new(
            output,
            ContentHash::of_bytes(document.as_bytes()),
            Reproducibility::Deterministic,
        ))
    }

    fn assemble(
        &self,
        request: &OperationRequest,
        directory: &str,
        cancel: &CancellationToken,
    ) -> Result<AssembledScene, String> {
        let sparse_record = self.input(request, ArtifactKind::SparseReconstruction)?;
        let sparse = CameraStageArtifact::from_json(&self.document(sparse_record)?)
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
        let dense = DenseStageArtifact::from_json(&self.document(dense_record)?)
            .map_err(|e| e.to_string())?;
        if self.dense_cameras(dense_record, &dense)?.content_hash() != sparse.content_hash() {
            return Err("scene camera does not match mesh ancestry".into());
        }
        let mesh = SurfaceMeshArtifact::from_json(&self.document(mesh_record)?)?;
        let evidence = mesh.evidence()?;
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
        let bytes = crate::textured_glb::encode_textured_glb(&evidence, &bake)?;
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
        scene.validate().map_err(|e| e.to_string())?;
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
