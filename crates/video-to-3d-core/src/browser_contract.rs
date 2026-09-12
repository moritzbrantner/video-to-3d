use crate::{CameraPose, DenseStats, ReconstructionRequest, ReconstructionResult};
use serde::Serialize;
use std::collections::HashSet;

const FEWER_THAN_TWO_REGISTERED_CAMERAS: &str =
    "fewer than two accepted registered cameras are available";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraKind {
    None,
    ApproximateMotion,
    Seed,
    Registered,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistrationStatus {
    Seed,
    Registered,
    InsufficientCorrespondences,
    PnpRejected,
    LowParallax,
    NotSelected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleAdjustmentStatus {
    NotRun,
    Accepted,
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseCameraRole {
    Reference,
    Source,
}

#[derive(Clone, Debug, Serialize)]
pub struct FrameCameraState {
    pub frame_index: usize,
    pub camera_kind: CameraKind,
    pub registration_status: RegistrationStatus,
    pub correspondences: usize,
    pub inliers: usize,
    pub median_reprojection_error_pixels: Option<f32>,
    pub recovered_from_revisit: bool,
    pub bundle_adjustment: BundleAdjustmentStatus,
    pub dense_role: Option<DenseCameraRole>,
    pub dense_ineligibility_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CameraPipelineState {
    pub approximate_motion_samples: Vec<CameraPose>,
    pub calibrated_seed_cameras: Vec<CameraPose>,
    pub registered_cameras: Vec<CameraPose>,
    pub dense_eligible_cameras: Vec<CameraPose>,
    pub frames: Vec<FrameCameraState>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BrowserReconstructionResult {
    #[serde(flatten)]
    pub reconstruction: ReconstructionResult,
    pub camera_state: CameraPipelineState,
}

pub fn reconstruct_browser(
    request: &ReconstructionRequest,
) -> Result<BrowserReconstructionResult, String> {
    let reconstruction = crate::reconstruct(request)?;
    let camera_state = CameraPipelineState::from_reconstruction(&reconstruction, request.frames.len());
    validate_browser_contract(&reconstruction, &camera_state, request.frames.len())?;
    Ok(BrowserReconstructionResult {
        reconstruction,
        camera_state,
    })
}

impl CameraPipelineState {
    fn from_reconstruction(reconstruction: &ReconstructionResult, frame_count: usize) -> Self {
        let seed_frames: HashSet<usize> = reconstruction
            .calibrated_pair
            .as_ref()
            .map(|pair| [pair.from_frame, pair.to_frame].into_iter().collect())
            .unwrap_or_default();

        let approximate_motion_samples = if reconstruction.calibrated_pair.is_none() {
            reconstruction.cameras.clone()
        } else {
            Vec::new()
        };
        let calibrated_seed_cameras = if reconstruction.calibrated_pair.is_some() {
            reconstruction
                .cameras
                .iter()
                .copied()
                .filter(|camera| seed_frames.contains(&camera.frame_index))
                .collect()
        } else {
            Vec::new()
        };
        let registered_cameras = if reconstruction.calibrated_pair.is_some() {
            reconstruction
                .cameras
                .iter()
                .copied()
                .filter(|camera| !seed_frames.contains(&camera.frame_index))
                .collect()
        } else {
            Vec::new()
        };

        let dense_frames: HashSet<usize> = reconstruction
            .dense
            .reference_frame
            .into_iter()
            .chain(reconstruction.dense.source_frames.iter().copied())
            .collect();
        let dense_eligible_cameras = calibrated_seed_cameras
            .iter()
            .chain(&registered_cameras)
            .copied()
            .filter(|camera| dense_frames.contains(&camera.frame_index))
            .collect();

        let bundle_adjustment = if reconstruction.multi_view.bundle_adjustment.accepted {
            BundleAdjustmentStatus::Accepted
        } else if reconstruction.multi_view.bundle_adjustment.attempted {
            BundleAdjustmentStatus::Rejected
        } else {
            BundleAdjustmentStatus::NotRun
        };

        let frames = (0..frame_count)
            .map(|frame_index| {
                let is_seed = seed_frames.contains(&frame_index);
                let registered_view = reconstruction
                    .registered_views
                    .iter()
                    .find(|view| view.frame_index == frame_index);
                let candidate = reconstruction
                    .multi_view
                    .registration_candidates
                    .iter()
                    .find(|candidate| candidate.frame_index == frame_index);
                let revisit_attempt = reconstruction
                    .revisits
                    .recoveries
                    .iter()
                    .find(|recovery| recovery.frame_index == frame_index);
                let has_approximate_motion = approximate_motion_samples
                    .iter()
                    .any(|camera| camera.frame_index == frame_index);
                let has_registered_camera = registered_cameras
                    .iter()
                    .any(|camera| camera.frame_index == frame_index);

                let camera_kind = if is_seed {
                    CameraKind::Seed
                } else if has_registered_camera {
                    CameraKind::Registered
                } else if has_approximate_motion {
                    CameraKind::ApproximateMotion
                } else {
                    CameraKind::None
                };

                let registration_status = if is_seed {
                    RegistrationStatus::Seed
                } else if registered_view.is_some() {
                    RegistrationStatus::Registered
                } else if candidate.is_some_and(|candidate| !candidate.pnp_ready) {
                    RegistrationStatus::InsufficientCorrespondences
                } else if candidate.is_some_and(|candidate| candidate.pnp_ready) {
                    RegistrationStatus::PnpRejected
                } else if frame_is_low_parallax(reconstruction, frame_index) {
                    RegistrationStatus::LowParallax
                } else {
                    RegistrationStatus::NotSelected
                };

                let correspondences = registered_view
                    .map(|view| view.correspondences)
                    .or_else(|| revisit_attempt.map(|attempt| attempt.correspondences))
                    .or_else(|| candidate.map(|candidate| candidate.seed_landmark_correspondences))
                    .or_else(|| {
                        is_seed.then(|| {
                            reconstruction
                                .calibrated_pair
                                .as_ref()
                                .map_or(0, |pair| pair.matches)
                        })
                    })
                    .unwrap_or(0);
                let inliers = registered_view
                    .map(|view| view.inliers)
                    .or_else(|| revisit_attempt.map(|attempt| attempt.inliers))
                    .or_else(|| {
                        is_seed.then(|| {
                            reconstruction
                                .calibrated_pair
                                .as_ref()
                                .map_or(0, |pair| pair.inliers)
                        })
                    })
                    .unwrap_or(0);
                let median_reprojection_error_pixels = registered_view
                    .map(|view| view.median_reprojection_error_pixels)
                    .or_else(|| {
                        revisit_attempt.and_then(|attempt| attempt.median_reprojection_error_pixels)
                    })
                    .or_else(|| {
                        is_seed.then(|| {
                            reconstruction
                                .calibrated_pair
                                .as_ref()
                                .map(|pair| pair.median_reprojection_error_pixels)
                        })
                        .flatten()
                    });
                let recovered_from_revisit = registered_view
                    .is_some_and(|view| view.recovered_from_revisit);

                let dense_role = if reconstruction.dense.reference_frame == Some(frame_index) {
                    Some(DenseCameraRole::Reference)
                } else if reconstruction.dense.source_frames.contains(&frame_index) {
                    Some(DenseCameraRole::Source)
                } else {
                    None
                };
                let accepted_camera = matches!(camera_kind, CameraKind::Seed | CameraKind::Registered);
                let dense_ineligibility_reason = if accepted_camera && dense_role.is_none() {
                    reconstruction.dense.skip_reason.clone().or_else(|| {
                        reconstruction.dense.attempted.then(|| {
                            "not selected after dense baseline/overlap/visible-landmark filtering or the bounded source-view cap"
                                .to_string()
                        })
                    })
                } else {
                    None
                };

                FrameCameraState {
                    frame_index,
                    camera_kind,
                    registration_status,
                    correspondences,
                    inliers,
                    median_reprojection_error_pixels,
                    recovered_from_revisit,
                    bundle_adjustment,
                    dense_role,
                    dense_ineligibility_reason,
                }
            })
            .collect();

        Self {
            approximate_motion_samples,
            calibrated_seed_cameras,
            registered_cameras,
            dense_eligible_cameras,
            frames,
        }
    }
}

fn frame_is_low_parallax(reconstruction: &ReconstructionResult, frame_index: usize) -> bool {
    let mut adjacent_pairs = reconstruction
        .pairs
        .iter()
        .filter(|pair| pair.from_frame == frame_index || pair.to_frame == frame_index)
        .peekable();
    adjacent_pairs.peek().is_some() && adjacent_pairs.all(|pair| pair.low_parallax)
}

fn validate_dense_registered_count(
    accepted_registered_cameras: usize,
    dense: &DenseStats,
) -> Result<(), String> {
    if dense
        .skip_reason
        .as_deref()
        .is_some_and(|reason| reason.contains(FEWER_THAN_TWO_REGISTERED_CAMERAS))
        && accepted_registered_cameras >= 2
    {
        return Err(format!(
            "camera-state invariant violated: dense reconstruction reports fewer than two accepted registered cameras, but {accepted_registered_cameras} accepted seed/registered cameras are exposed"
        ));
    }
    if dense.attempted && accepted_registered_cameras < 2 {
        return Err(format!(
            "camera-state invariant violated: dense reconstruction ran with only {accepted_registered_cameras} exposed accepted cameras"
        ));
    }
    Ok(())
}

fn validate_browser_contract(
    reconstruction: &ReconstructionResult,
    camera_state: &CameraPipelineState,
    frame_count: usize,
) -> Result<(), String> {
    if camera_state.frames.len() != frame_count {
        return Err("camera-state invariant violated: frame-state count does not match input frames".into());
    }

    let accepted_registered_cameras =
        camera_state.calibrated_seed_cameras.len() + camera_state.registered_cameras.len();
    validate_dense_registered_count(accepted_registered_cameras, &reconstruction.dense)?;

    if reconstruction.calibrated_pair.is_some() {
        if camera_state.calibrated_seed_cameras.len() != 2 {
            return Err(format!(
                "camera-state invariant violated: calibrated geometry must expose exactly two seed cameras, got {}",
                camera_state.calibrated_seed_cameras.len()
            ));
        }
        if !camera_state.approximate_motion_samples.is_empty() {
            return Err(
                "camera-state invariant violated: approximate motion samples cannot be exposed as accepted calibrated cameras"
                    .into(),
            );
        }
        if accepted_registered_cameras != reconstruction.cameras.len() {
            return Err(
                "camera-state invariant violated: accepted camera partition does not match reconstruction cameras"
                    .into(),
            );
        }
        if camera_state.registered_cameras.len() != reconstruction.registered_views.len() {
            return Err(
                "camera-state invariant violated: registered camera poses do not match registered-view evidence"
                    .into(),
            );
        }
    } else {
        if !camera_state.calibrated_seed_cameras.is_empty()
            || !camera_state.registered_cameras.is_empty()
            || !camera_state.dense_eligible_cameras.is_empty()
        {
            return Err(
                "camera-state invariant violated: uncalibrated reconstruction exposed accepted camera geometry"
                    .into(),
            );
        }
        if camera_state.approximate_motion_samples.len() != reconstruction.cameras.len() {
            return Err(
                "camera-state invariant violated: approximate motion samples do not match the fallback camera track"
                    .into(),
            );
        }
    }

    if reconstruction.dense.source_frames.len() != reconstruction.dense.source_views {
        return Err(format!(
            "camera-state invariant violated: dense source-frame evidence has {} frames for {} source views",
            reconstruction.dense.source_frames.len(),
            reconstruction.dense.source_views
        ));
    }

    if reconstruction.dense.attempted {
        if reconstruction.dense.reference_frame.is_none() {
            return Err(
                "camera-state invariant violated: attempted dense reconstruction has no reference frame"
                    .into(),
            );
        }
        let expected_dense_cameras = reconstruction.dense.source_views + 1;
        if camera_state.dense_eligible_cameras.len() != expected_dense_cameras {
            return Err(format!(
                "camera-state invariant violated: dense reconstruction used {expected_dense_cameras} reference/source cameras, but only {} are present in accepted camera state",
                camera_state.dense_eligible_cameras.len()
            ));
        }
    } else if !camera_state.dense_eligible_cameras.is_empty() {
        return Err(
            "camera-state invariant violated: skipped dense reconstruction exposed dense-eligible cameras"
                .into(),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameInput, ReconstructionOptions};

    fn flat_frame(width: u32, height: u32) -> FrameInput {
        let mut rgba = vec![120; width as usize * height as usize * 4];
        for pixel in rgba.as_chunks_mut::<4>().0 {
            pixel[3] = 255;
        }
        FrameInput {
            width,
            height,
            rgba,
        }
    }

    #[test]
    fn browser_contract_keeps_uncalibrated_motion_separate() {
        let frame = flat_frame(48, 32);
        let request = ReconstructionRequest {
            frames: vec![frame.clone(), frame],
            options: ReconstructionOptions::default(),
        };

        let result = reconstruct_browser(&request).expect("browser reconstruction");

        assert!(result.reconstruction.calibrated_pair.is_none());
        assert_eq!(result.camera_state.approximate_motion_samples.len(), 2);
        assert!(result.camera_state.calibrated_seed_cameras.is_empty());
        assert!(result.camera_state.registered_cameras.is_empty());
        assert!(result.camera_state.dense_eligible_cameras.is_empty());
        assert_eq!(result.camera_state.frames.len(), 2);
        assert!(result
            .camera_state
            .frames
            .iter()
            .all(|frame| frame.camera_kind == CameraKind::ApproximateMotion));
    }

    #[test]
    fn dense_registered_count_contradiction_fails_closed() {
        let dense = DenseStats {
            skip_reason: Some(FEWER_THAN_TWO_REGISTERED_CAMERAS.to_string()),
            ..DenseStats::default()
        };

        let error = validate_dense_registered_count(2, &dense)
            .expect_err("contradictory camera counts must be rejected");
        assert!(error.contains("invariant violated"));
        assert!(validate_dense_registered_count(1, &dense).is_ok());
    }
}
