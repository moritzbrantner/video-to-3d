//! Validate externally registered feature tracks against the decoded frames before
//! their poses and points can seed the shared dense and surface stages.
use crate::multi_view::RegisteredCamera;
use crate::reconstruction_stages::{
    estimate_dense_evidence, finish_reconstruction, SparseReconstruction, StageCamera,
};
use crate::scene_project::ContentHash;
use crate::*;
use nalgebra::{Matrix3, Vector3};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
pub struct RegisteredSparseInput {
    schema_version: u32,
    provider: String,
    version: String,
    frames: Vec<Receipt>,
    width: u32,
    height: u32,
    focal_pixels: f64,
    cameras: Vec<Camera>,
    landmarks: Vec<Landmark>,
}
#[derive(Deserialize)]
struct Receipt {
    frame_index: usize,
    content_hash: ContentHash,
}
#[derive(Deserialize)]
struct Camera {
    frame_index: usize,
    rotation: [f64; 9],
    translation: [f64; 3],
}
#[derive(Deserialize)]
struct Landmark {
    xyz: [f64; 3],
    rgb: [u8; 3],
    observations: Vec<Observation>,
}
#[derive(Deserialize)]
struct Observation {
    frame_index: usize,
    xy: [f64; 2],
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

impl RegisteredSparseInput {
    pub(crate) fn camera_stage(
        self,
        request: &ReconstructionRequest,
        mut canceled: impl FnMut() -> bool,
    ) -> Result<CameraStageArtifact, String> {
        let provider = ReconstructionProviderDescriptor::new(
            self.provider.clone(),
            ReconstructionProviderClass::GeometricMultiView,
            Some(self.version.clone()),
        )?;
        let sparse = self.sparse(request)?;
        crate::reconstruction_stages::camera_stage_artifact(
            request,
            sparse,
            Some(provider),
            &mut ReconstructionControl::new(&mut canceled),
        )
        .map_err(|e| e.to_string())
    }

    fn validate_frames(&self, request: &ReconstructionRequest) -> Result<(), String> {
        crate::reconstruction::validate_request(request)?;
        if self.schema_version != 1 || self.provider != "colmap" || self.version != "4.2.1" {
            return Err("unsupported sparse camera provider or document version".into());
        }
        if self.frames.len() != request.frames.len()
            || self.cameras.len() < 3
            || self.cameras.len() > self.frames.len()
        {
            return Err(
                "sparse registration needs three or more cameras bound to the sampled frames"
                    .into(),
            );
        }
        let extent = f64::from(self.width.max(self.height));
        if !(0.1 * extent..=10. * extent).contains(&self.focal_pixels) {
            return Err("sparse registration returned an implausible focal length".into());
        }
        for (index, (r, f)) in self.frames.iter().zip(&request.frames).enumerate() {
            if r.frame_index != index
                || self.width != f.width
                || self.height != f.height
                || r.content_hash != ContentHash::of_bytes(&f.rgba)
            {
                return Err(format!(
                    "sparse camera receipt differs from sampled frame {index}"
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn sparse(
        self,
        request: &ReconstructionRequest,
    ) -> Result<SparseReconstruction, String> {
        self.validate_frames(request)?;
        let mut lookup = BTreeMap::new();
        let mut poses = Vec::new();
        for (i, c) in self.cameras.iter().enumerate() {
            let rotation = Matrix3::from_row_slice(&c.rotation);
            let translation = Vector3::from(c.translation);
            if !c
                .rotation
                .iter()
                .chain(&c.translation)
                .all(|v| v.is_finite())
                || (rotation * rotation.transpose() - Matrix3::identity()).norm() > 1e-5
                || (rotation.determinant() - 1.).abs() > 1e-5
                || c.frame_index >= self.frames.len()
                || lookup.insert(c.frame_index, i).is_some()
            {
                return Err("invalid or duplicate sparse camera pose".into());
            }
            poses.push(RegisteredCamera {
                frame_index: c.frame_index,
                rotation,
                translation,
            });
        }
        if (poses[0].rotation - Matrix3::identity()).norm() > 1e-5
            || poses[0].translation.norm() > 1e-5
            || (poses[1].translation.norm() - 1.).abs() > 1e-5
        {
            return Err("sparse camera seed does not use a unit-baseline reference frame".into());
        }
        let mut errors = vec![Vec::new(); poses.len()];
        let mut candidates = vec![0; poses.len()];
        let mut points = Vec::new();
        let mut regions = Vec::new();
        let mut angles = Vec::new();
        let mut epipolar = Vec::new();
        let mut seed_residuals = Vec::new();
        let mut seed_motion = Vec::new();
        let mut seed_parallax = Vec::new();
        let mut common = 0;
        let mut longest = 0;
        let mut three = 0;
        let mut observations = 0;
        for lm in self.landmarks {
            if !lm.xyz.iter().all(|v| v.is_finite()) {
                return Err("non-finite sparse landmark".into());
            }
            let point = Vector3::from(lm.xyz);
            let mut track = BTreeMap::new();
            let mut seen = BTreeSet::new();
            for ob in lm.observations {
                let i = *lookup
                    .get(&ob.frame_index)
                    .ok_or("track references unregistered camera")?;
                let [u, v] = ob.xy;
                if !seen.insert(i)
                    || !(0.0..f64::from(self.width)).contains(&u)
                    || !(0.0..f64::from(self.height)).contains(&v)
                {
                    return Err("duplicate or out-of-image sparse observation".into());
                }
                candidates[i] += 1;
                let q = poses[i].rotation * point + poses[i].translation;
                let ray = Vector3::new(
                    (u - f64::from(self.width) / 2.) / self.focal_pixels,
                    (v - f64::from(self.height) / 2.) / self.focal_pixels,
                    1.,
                );
                let residual = self.focal_pixels * (q.x / q.z - ray.x).hypot(q.y / q.z - ray.y);
                if q.z > 0. && residual <= 4. {
                    track.insert(i, (residual, ray));
                }
            }
            common += usize::from(seen.contains(&0) && seen.contains(&1));
            if track.len() < 2 {
                continue;
            }
            let mut directions = track
                .keys()
                .map(|&i| (point - poses[i].camera_center()).normalize());
            let first = directions.next().expect("two observations");
            if !directions.any(|v| v.dot(&first) < 0.5f64.to_radians().cos()) {
                continue;
            }
            if let (Some((e0, r0)), Some((e1, r1))) = (track.get(&0), track.get(&1)) {
                let r = poses[1].rotation;
                let t = poses[1].translation;
                let l1 = t.cross(&(r * r0));
                let l0 = r.transpose() * r1.cross(&t);
                let denom = (l1.x * l1.x + l1.y * l1.y + l0.x * l0.x + l0.y * l0.y).sqrt();
                if denom > 1e-10 {
                    let rotated = r * r0;
                    seed_motion.push(self.focal_pixels * (r1.x - r0.x).hypot(r1.y - r0.y));
                    seed_parallax.push(
                        self.focal_pixels
                            * (r1.x - rotated.x / rotated.z).hypot(r1.y - rotated.y / rotated.z),
                    );
                    epipolar.push(self.focal_pixels * r1.dot(&l1).abs() / denom);
                    angles.push(
                        (point - poses[0].camera_center())
                            .angle(&(point - poses[1].camera_center()))
                            .to_degrees(),
                    );
                    seed_residuals.push((e0 + e1) / 2.);
                }
            }
            for (&i, &(error, _)) in &track {
                errors[i].push(error);
            }
            longest = longest.max(track.len());
            three += usize::from(track.len() >= 3);
            observations += track.len();
            let mut supported = track.keys().map(|&i| poses[i].frame_index);
            regions.push(SurfaceEvidenceRegion {
                origin: EvidenceOrigin::GeometricMultiView,
                reference_frame: supported.next(),
                source_frames: supported.collect(),
                points: EvidenceRange::new(points.len(), 1),
            });
            let [r, g, b] = lm.rgb;
            let [x, y, z] = lm.xyz.map(|x| x as f32);
            points.push(Point3 {
                x,
                y,
                z,
                r,
                g,
                b,
                confidence: 1.,
            });
        }
        if angles.len() < 12 || angles.len() * 2 < common {
            return Err(format!(
                "sparse seed frames {} and {} have {} corroborated tracks of {} shared; support requires at least 12 and at least half of the shared tracks",
                poses[0].frame_index, poses[1].frame_index, angles.len(), common,
            ));
        }
        let seed_angle = median(&mut angles);
        if seed_angle < 1.5 {
            return Err(format!(
                "sparse seed median triangulation angle {seed_angle:.2}° is below 1.5°",
            ));
        }
        for (i, (accepted, &observed)) in errors.iter().zip(&candidates).enumerate() {
            if accepted.len() < 8 || accepted.len() * 2 < observed {
                return Err(format!(
                    "frame {} has {} corroborated observations of {}; support requires at least 8 and at least half of its candidate observations",
                    poses[i].frame_index, accepted.len(), observed,
                ));
            }
        }
        let mut cameras = Vec::new();
        let mut registered_views = Vec::new();
        let mut evidence_cameras = Vec::new();
        for (i, (c, pose)) in self.cameras.iter().zip(&poses).enumerate() {
            let error = median(&mut errors[i]) as f32;
            if error > 2. {
                return Err(format!(
                    "frame {} sparse reprojection error {error:.2}px",
                    c.frame_index
                ));
            }
            let rotation = c.rotation.map(|x| x as f32);
            let translation = c.translation.map(|x| x as f32);
            let inliers = errors[i].len();
            let confidence = inliers as f32 / candidates[i] as f32;
            let center = pose.camera_center();
            cameras.push(CameraPose {
                frame_index: c.frame_index,
                x: center.x as f32,
                y: center.y as f32,
                z: center.z as f32,
                matched_features: inliers,
            });
            evidence_cameras.push(EvidenceCamera {
                frame_index: c.frame_index,
                authority: EvidenceCameraAuthority::ProviderEstimated,
                rotation,
                translation,
                confidence: Some(confidence),
                median_reprojection_error_pixels: Some(error),
            });
            if i >= 2 {
                registered_views.push(RegisteredViewStats {
                    frame_index: c.frame_index,
                    correspondences: candidates[i],
                    inliers,
                    inlier_ratio: confidence,
                    median_reprojection_error_pixels: error,
                    rotation,
                    translation,
                    recovered_from_revisit: false,
                });
            }
        }
        let provider = ReconstructionProviderDescriptor::new(
            self.provider,
            ReconstructionProviderClass::GeometricMultiView,
            Some(self.version),
        )?;
        let evidence = ReconstructionEvidenceView::new(
            provider,
            EvidenceScale::ArbitraryMonocular,
            evidence_cameras,
            regions,
            &points,
            &[],
        )?;
        let diagnostic=format!("Local COLMAP {} registered cameras and {} triangulated landmarks passed Rust track reprojection checks.",evidence.cameras.len(),evidence.points.len());
        let seed = CalibratedPairStats {
            from_frame: poses[0].frame_index,
            to_frame: poses[1].frame_index,
            matches: common,
            inliers: angles.len(),
            inlier_ratio: angles.len() as f32 / common as f32,
            focal_pixels: self.focal_pixels as f32,
            median_sampson_error_pixels: median(&mut epipolar) as f32,
            median_reprojection_error_pixels: median(&mut seed_residuals) as f32,
            median_triangulation_angle_degrees: seed_angle as f32,
            relative_rotation: self.cameras[1].rotation.map(|x| x as f32),
            translation_direction: self.cameras[1].translation.map(|x| x as f32),
        };
        let seed_candidate = SeedCandidateStats {
            from_frame: seed.from_frame,
            to_frame: seed.to_frame,
            matches: common,
            median_motion: median(&mut seed_motion) as f32,
            median_parallax_residual: median(&mut seed_parallax) as f32,
            rejected_gate: None,
            selected: true,
            inliers: Some(seed.inliers),
            inlier_ratio: Some(seed.inlier_ratio),
            rotation_only_residual_pixels: None,
            required_points: Some(12),
            triangulated_points: Some(seed.inliers),
            behind_camera: None,
            failed_triangulations: None,
            high_reprojection: None,
            median_reprojection_error_pixels: Some(seed.median_reprojection_error_pixels),
            median_triangulation_angle_degrees: Some(seed.median_triangulation_angle_degrees),
        };
        let selection = KeyframeSelectionStats {
            segments: vec![ClipSegmentStats {
                first_frame: 0,
                last_frame: request.frames.len() - 1,
                keyframes: poses.len(),
                ends_with: None,
            }],
            frames: request
                .frames
                .iter()
                .enumerate()
                .map(|(frame_index, frame)| {
                    let evidence = crate::input_readiness::frame_evidence(
                        frame_index,
                        &crate::to_luma(frame),
                        frame.width,
                        frame.height,
                    );
                    FrameSelectionStats {
                        frame_index,
                        segment: 0,
                        decision: if lookup.contains_key(&frame_index) {
                            FrameDecision::Keyframe
                        } else {
                            FrameDecision::Unregistered
                        },
                        sharpness_ratio: evidence.sharpness_ratio,
                        mean_luma: evidence.mean_luma,
                    }
                })
                .collect(),
        };
        let multi_view = MultiViewStats {
            keyframes: lookup.keys().copied().collect(),
            keyframe_selection: selection,
            track_count: points.len(),
            tracks_three_plus: three,
            longest_track: longest,
            observations,
            linked_pairs: 0,
            registration_candidates: vec![],
            new_landmarks: Default::default(),
            bundle_adjustment: Default::default(),
            segment_solves: vec![],
        };
        let dense_sparse_points = points
            .iter()
            .map(|p| [p.x, p.y, p.z].map(f64::from))
            .collect();
        let dense_cameras: Vec<_> = poses.iter().map(StageCamera::from).collect();
        Ok(SparseReconstruction {
            options: request.options,
            image_width: self.width,
            image_height: self.height,
            focal_length_pixels: self.focal_pixels as f32,
            cameras,
            points,
            pairs: vec![],
            calibrated_pair: Some(seed),
            multi_view,
            revisits: Default::default(),
            registered_views,
            seed_candidates: vec![seed_candidate],
            bootstrap: BootstrapDiagnosis {
                decisive_gate: None,
                decisive_pair: None,
                escalation: None,
                rejections: vec![],
                summary: diagnostic.clone(),
            },
            final_cameras: dense_cameras.clone(),
            dense_cameras,
            dense_sparse_points,
            track_scope: Default::default(),
            expanded_views: 0,
            warnings_before_dense: vec![diagnostic],
            warnings_after_mesh: vec![],
        })
    }
}

pub fn reconstruct_browser_with_registered_geometry(
    request: &ReconstructionRequest,
    input: RegisteredSparseInput,
) -> Result<BrowserReconstructionResult, String> {
    let sparse = input.sparse(request)?;
    let dense = estimate_dense_evidence(&request.frames, &sparse);
    let reconstruction = finish_reconstruction(sparse, dense)?;
    crate::browser_contract::from_reconstruction(reconstruction, request.frames.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (ReconstructionRequest, RegisteredSparseInput) {
        let frames = (0..3)
            .map(|i| FrameInput {
                width: 120,
                height: 90,
                rgba: vec![i; 120 * 90 * 4],
            })
            .collect::<Vec<_>>();
        let receipts = frames
            .iter()
            .enumerate()
            .map(|(frame_index, f)| Receipt {
                frame_index,
                content_hash: ContentHash::of_bytes(&f.rgba),
            })
            .collect();
        let cameras = (0..3)
            .map(|frame_index| Camera {
                frame_index,
                rotation: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                translation: [-(frame_index as f64), 0., 0.],
            })
            .collect::<Vec<_>>();
        let landmarks = (0..24)
            .map(|i| {
                let xyz = [
                    (i % 6) as f64 * 0.15,
                    (i / 6) as f64 * 0.12,
                    5. + 0.1 * (i % 3) as f64,
                ];
                let observations = cameras
                    .iter()
                    .map(|c| Observation {
                        frame_index: c.frame_index,
                        xy: [
                            60. + 100. * (xyz[0] + c.translation[0]) / xyz[2],
                            45. + 100. * xyz[1] / xyz[2],
                        ],
                    })
                    .collect();
                Landmark {
                    xyz,
                    rgb: [5, 10, 20],
                    observations,
                }
            })
            .collect();
        (
            ReconstructionRequest {
                frames,
                options: Default::default(),
            },
            RegisteredSparseInput {
                schema_version: 1,
                provider: "colmap".into(),
                version: "4.2.1".into(),
                frames: receipts,
                width: 120,
                height: 90,
                focal_pixels: 100.,
                cameras,
                landmarks,
            },
        )
    }

    #[test]
    fn checked_tracks_drive_the_shared_camera_dense_and_mesh_contract() {
        let (request, input) = fixture();
        let result = reconstruct_browser_with_registered_geometry(&request, input).unwrap();
        assert_eq!(result.reconstruction.cameras.len(), 3);
        assert_eq!(result.reconstruction.points.len(), 24);
        assert_eq!(result.reconstruction.multi_view.observations, 72);
        assert_eq!(result.reconstruction.calibrated_pair.unwrap().inliers, 24);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn registered_stage_round_trips_and_binds_dense_to_the_same_frames() {
        let (request, input) = fixture();
        let cameras = input.camera_stage(&request, || false).unwrap();
        assert_eq!(cameras.registration_provider.as_ref().unwrap().id, "colmap");
        assert_eq!(cameras.sparse.final_cameras.len(), 3);
        let stored = CameraStageArtifact::from_json(&cameras.to_json()).unwrap();
        let dense = crate::reconstruct_dense(&request, &stored).unwrap();
        assert_eq!(dense.cameras_hash, cameras.content_hash());
        assert_eq!(dense.request_hash, crate::request_content_hash(&request));
    }

    #[test]
    fn gauge_support_counts_only_exported_landmarks_shared_by_both_cameras() {
        let (request, mut input) = fixture();
        for landmark in &mut input.landmarks[10..] {
            landmark.observations.remove(1);
        }
        let error = input.sparse(&request).unwrap_err();
        assert!(error.contains("10 corroborated tracks of 10 shared"));
    }

    #[test]
    fn an_added_camera_needs_its_own_corroborated_observations() {
        let (request, mut input) = fixture();
        for landmark in &mut input.landmarks[3..] {
            landmark.observations.retain(|ob| ob.frame_index != 2);
        }
        let error = input.sparse(&request).unwrap_err();
        assert!(error.contains("frame 2 has 3 corroborated observations"));
    }

    #[test]
    fn mismatched_pixels_or_inconsistent_provider_tracks_cannot_seed_geometry() {
        let (mut request, input) = fixture();
        request.frames[1].rgba[0] ^= 1;
        assert!(input.sparse(&request).unwrap_err().contains("receipt"));
        let (request, mut input) = fixture();
        for p in &mut input.landmarks {
            p.observations[1].xy[1] += 8.;
        }
        assert!(input.sparse(&request).unwrap_err().contains("support"));
        let (request, mut input) = fixture();
        input.cameras[2].rotation[0] = -1.;
        assert!(input.sparse(&request).unwrap_err().contains("pose"));
    }
}
