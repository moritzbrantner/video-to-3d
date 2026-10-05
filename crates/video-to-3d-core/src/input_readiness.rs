//! Input normalization and reconstruction-readiness evidence.
//!
//! Before an expensive reconstruction, the sampled frames and their timing are
//! checked for the properties geometric reconstruction needs: translation
//! parallax rather than pure rotation, stable scene content between samples,
//! usable texture, sharp frames, steady exposure, overlap between neighbours,
//! and sound decoding. The result is diagnostic evidence: raw measurements,
//! the fixed thresholds they were compared against, and the issues that
//! follow. It never invents a confidence score, and a weak verdict never
//! blocks a declared world-model (generative) path; it only says whether
//! observed geometry is likely to succeed.
//!
//! Inputs are the browser-sampled analysis frames (already display-oriented
//! and converted to sRGB by the canvas) plus sampling metadata. Original media
//! identity is untouched: the evidence describes samples of the media, it does
//! not replace or re-encode it.

use crate::{detect_features, match_features, to_luma, FrameInput, ReconstructionOptions};
use nalgebra::{DMatrix, Matrix3, Vector3};
use serde::{Deserialize, Serialize};

pub const READINESS_SCHEMA_VERSION: u32 = 1;

/// Fixed thresholds. They are reported with every result so the evidence can
/// be audited, and changing them is an explicit, versioned decision.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ReadinessThresholds {
    /// Median feature motion (px) below which a pair counts as static.
    pub static_motion_pixels: f32,
    /// 75th-percentile residual (px) after the best homography above which a
    /// pair shows translation parallax; below it, motion is explained by
    /// rotation or a planar scene.
    pub parallax_residual_pixels: f32,
    /// Fraction of matches that must follow a second, distinct homography for a
    /// pair to show layered (piecewise-planar) parallax.
    pub min_secondary_motion_fraction: f32,
    /// Fraction of features matched between neighbours below which overlap is weak.
    pub min_overlap: f32,
    /// Homography inlier fraction below which scene content is unstable.
    pub min_stable_fraction: f32,
    /// Fraction of textured pixels below which a frame has weak texture.
    pub min_texture_density: f32,
    /// mean|Laplacian| / mean|gradient| below which a textured frame is blurred.
    pub min_sharpness_ratio: f32,
    /// Mean luma bounds for a usable exposure.
    pub min_mean_luma: f32,
    pub max_mean_luma: f32,
    /// Fraction of crushed or clipped pixels above which exposure is unusable.
    pub max_clipped_fraction: f32,
    /// Relative change of mean luma between neighbours above which exposure is unstable.
    pub max_exposure_change: f32,
    /// 8x8 block-boundary to interior gradient ratio above which decoding is blocky.
    pub max_blockiness: f32,
}

pub const THRESHOLDS: ReadinessThresholds = ReadinessThresholds {
    static_motion_pixels: 0.5,
    parallax_residual_pixels: 1.5,
    min_secondary_motion_fraction: 0.08,
    min_overlap: 0.15,
    min_stable_fraction: 0.4,
    min_texture_density: 0.03,
    min_sharpness_ratio: 0.55,
    min_mean_luma: 35.0,
    max_mean_luma: 220.0,
    max_clipped_fraction: 0.35,
    max_exposure_change: 0.2,
    max_blockiness: 1.8,
};

const TEXTURE_GRADIENT: i32 = 24;
const RANSAC_ITERATIONS: usize = 256;
const RANSAC_INLIER_PIXELS: f32 = 1.5;
const SECONDARY_MIN_MATCHES: usize = 8;

/// Sampling metadata supplied with the frames.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SamplingMetadata {
    /// Media duration in seconds.
    pub duration_seconds: f64,
    /// Display dimensions of the media (after orientation).
    pub display_width: u32,
    pub display_height: u32,
    /// Rotation still to apply to the coded frames, in degrees. Browsers apply
    /// container orientation before drawing, so this is usually 0.
    #[serde(default)]
    pub rotation_degrees: i32,
    /// Times that were requested for each sample, in seconds.
    pub requested_times: Vec<f64>,
    /// Times the decoder actually presented, in seconds.
    pub presented_times: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NormalizedSampling {
    pub duration_seconds: f64,
    pub display_width: u32,
    pub display_height: u32,
    /// Normalized to 0, 90, 180 or 270.
    pub rotation_degrees: u32,
    /// Presented times relative to the first sample.
    pub timestamps: Vec<f64>,
    pub median_interval_seconds: f64,
    pub max_interval_seconds: f64,
    /// Largest |presented - requested| in seconds.
    pub max_seek_error_seconds: f64,
    /// Pairs whose presented time did not advance.
    pub non_advancing_samples: Vec<usize>,
    /// Pixels are sRGB as produced by a 2D canvas.
    pub color_space: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FrameEvidence {
    pub frame_index: usize,
    pub mean_luma: f32,
    pub crushed_fraction: f32,
    pub clipped_fraction: f32,
    pub texture_density: f32,
    pub sharpness_ratio: f32,
    pub blockiness: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PairMotion {
    /// Identical decoded pixels.
    Duplicate,
    /// No measurable motion.
    Static,
    /// Motion explained by one homography: camera rotation or a planar scene.
    RotationOrPlanar,
    /// Residual motion that one homography cannot explain: translation parallax.
    Parallax,
    /// Too few matches to tell.
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PairEvidence {
    pub from_frame: usize,
    pub to_frame: usize,
    pub features: usize,
    pub matches: usize,
    pub overlap: f32,
    pub median_motion_pixels: f32,
    pub homography_inlier_fraction: f32,
    pub residual_p75_pixels: f32,
    /// Fraction of matches explained by a second, different homography among
    /// the first model's outliers (a separately moving depth layer).
    pub secondary_motion_fraction: f32,
    pub relative_exposure_change: f32,
    pub motion: PairMotion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueCode {
    NoParallax,
    MostlyRotation,
    WeakTexture,
    SevereBlur,
    PoorExposure,
    UnstableExposure,
    WeakOverlap,
    UnstableContent,
    DuplicateFrames,
    TimingIrregular,
    DecodeArtifacts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Warning,
    Blocking,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReadinessIssue {
    pub code: IssueCode,
    pub severity: Severity,
    pub frames: Vec<usize>,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessVerdict {
    Ready,
    Marginal,
    Unsuitable,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReadinessReport {
    pub schema_version: u32,
    pub sampling: NormalizedSampling,
    pub thresholds: ReadinessThresholds,
    pub frames: Vec<FrameEvidence>,
    pub pairs: Vec<PairEvidence>,
    pub issues: Vec<ReadinessIssue>,
    /// Readiness for observed (geometric) reconstruction only.
    pub geometric_verdict: ReadinessVerdict,
    /// Always true: weak readiness never blocks a declared generative path.
    pub generative_paths_allowed: bool,
}

impl ReadinessReport {
    /// One line per issue, for status displays.
    pub fn diagnostics(&self) -> Vec<String> {
        self.issues
            .iter()
            .map(|issue| {
                let severity = match issue.severity {
                    Severity::Warning => "warning",
                    Severity::Blocking => "blocking",
                };
                format!("input readiness {severity}: {}", issue.message)
            })
            .collect()
    }
}

/// Normalize sampling metadata. Fails closed on values that cannot describe
/// a real sampling run.
pub fn normalize_sampling(
    metadata: &SamplingMetadata,
    frame_count: usize,
) -> Result<NormalizedSampling, String> {
    if !(metadata.duration_seconds.is_finite() && metadata.duration_seconds > 0.0) {
        return Err("sampling metadata needs a positive finite duration".into());
    }
    if metadata.display_width == 0 || metadata.display_height == 0 {
        return Err("sampling metadata needs positive display dimensions".into());
    }
    if metadata.rotation_degrees % 90 != 0 {
        return Err(format!(
            "rotation of {} degrees is not a multiple of 90",
            metadata.rotation_degrees
        ));
    }
    if metadata.requested_times.len() != frame_count
        || metadata.presented_times.len() != frame_count
    {
        return Err(format!(
            "sampling metadata lists {} requested and {} presented times for {frame_count} frames",
            metadata.requested_times.len(),
            metadata.presented_times.len()
        ));
    }
    let times = metadata
        .requested_times
        .iter()
        .chain(&metadata.presented_times);
    if times.clone().any(|time| !time.is_finite() || *time < 0.0) {
        return Err("sampling times must be finite and non-negative".into());
    }
    let origin = metadata.presented_times.first().copied().unwrap_or(0.0);
    let timestamps: Vec<f64> = metadata
        .presented_times
        .iter()
        .map(|time| time - origin)
        .collect();
    let mut intervals: Vec<f64> = timestamps
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    let non_advancing_samples = intervals
        .iter()
        .enumerate()
        .filter(|(_, interval)| **interval <= 0.0)
        .map(|(index, _)| index + 1)
        .collect();
    let max_seek_error_seconds = metadata
        .requested_times
        .iter()
        .zip(&metadata.presented_times)
        .map(|(requested, presented)| (requested - presented).abs())
        .fold(0.0, f64::max);
    intervals.sort_by(f64::total_cmp);
    let median_interval_seconds = intervals.get(intervals.len() / 2).copied().unwrap_or(0.0);
    let max_interval_seconds = intervals.last().copied().unwrap_or(0.0);
    Ok(NormalizedSampling {
        duration_seconds: metadata.duration_seconds,
        display_width: metadata.display_width,
        display_height: metadata.display_height,
        rotation_degrees: metadata.rotation_degrees.rem_euclid(360) as u32,
        timestamps,
        median_interval_seconds,
        max_interval_seconds,
        max_seek_error_seconds,
        non_advancing_samples,
        color_space: "srgb",
    })
}

/// Measure readiness of sampled frames for geometric reconstruction.
pub fn assess_readiness(
    frames: &[FrameInput],
    metadata: &SamplingMetadata,
) -> Result<ReadinessReport, String> {
    if frames.len() < 2 {
        return Err("readiness needs at least two sampled frames".into());
    }
    for (index, frame) in frames.iter().enumerate() {
        if frame.width < 32 || frame.height < 24 {
            return Err(format!("frame {index} is smaller than 32x24 pixels"));
        }
        if frame.rgba.len() != frame.width as usize * frame.height as usize * 4 {
            return Err(format!("frame {index} has an invalid RGBA payload"));
        }
        if frame.width != frames[0].width || frame.height != frames[0].height {
            return Err("all sampled frames must share one size".into());
        }
    }
    let sampling = normalize_sampling(metadata, frames.len())?;
    let lumas: Vec<Vec<u8>> = frames.iter().map(to_luma).collect();
    let width = frames[0].width;
    let height = frames[0].height;
    let frame_evidence: Vec<FrameEvidence> = lumas
        .iter()
        .enumerate()
        .map(|(index, luma)| frame_evidence(index, luma, width, height))
        .collect();
    let options = ReconstructionOptions::default();
    let features: Vec<_> = lumas
        .iter()
        .map(|luma| detect_features(luma, width, height, options))
        .collect();
    let pairs: Vec<PairEvidence> = (1..frames.len())
        .map(|index| {
            pair_evidence(
                index - 1,
                index,
                frames[index - 1].rgba == frames[index].rgba,
                &features[index - 1],
                &features[index],
                options,
                &frame_evidence,
            )
        })
        .collect();
    let issues = issues(&sampling, &frame_evidence, &pairs);
    let geometric_verdict = if issues
        .iter()
        .any(|issue| issue.severity == Severity::Blocking)
    {
        ReadinessVerdict::Unsuitable
    } else if issues.is_empty() {
        ReadinessVerdict::Ready
    } else {
        ReadinessVerdict::Marginal
    };
    Ok(ReadinessReport {
        schema_version: READINESS_SCHEMA_VERSION,
        sampling,
        thresholds: THRESHOLDS,
        frames: frame_evidence,
        pairs,
        issues,
        geometric_verdict,
        generative_paths_allowed: true,
    })
}

fn frame_evidence(index: usize, luma: &[u8], width: u32, height: u32) -> FrameEvidence {
    let (w, h) = (width as usize, height as usize);
    let pixel_count = luma.len() as f32;
    let sum: u64 = luma.iter().map(|value| u64::from(*value)).sum();
    let crushed = luma.iter().filter(|value| **value <= 5).count() as f32;
    let clipped = luma.iter().filter(|value| **value >= 250).count() as f32;
    let at = |x: usize, y: usize| i32::from(luma[y * w + x]);
    let mut textured = 0usize;
    let mut interior = 0usize;
    let mut gradient_sum = 0f64;
    let mut laplacian_sum = 0f64;
    let mut boundary_diff = 0f64;
    let mut boundary_count = 0usize;
    let mut inner_diff = 0f64;
    let mut inner_count = 0usize;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = at(x + 1, y) - at(x - 1, y);
            let gy = at(x, y + 1) - at(x, y - 1);
            let gradient = gx.abs() + gy.abs();
            let laplacian =
                at(x + 1, y) + at(x - 1, y) + at(x, y + 1) + at(x, y - 1) - 4 * at(x, y);
            interior += 1;
            if gradient >= TEXTURE_GRADIENT {
                textured += 1;
                gradient_sum += f64::from(gradient);
                laplacian_sum += f64::from(laplacian.abs());
            }
        }
    }
    for y in 0..h {
        for x in 1..w {
            let difference = f64::from((at(x, y) - at(x - 1, y)).abs());
            if x % 8 == 0 {
                boundary_diff += difference;
                boundary_count += 1;
            } else {
                inner_diff += difference;
                inner_count += 1;
            }
        }
    }
    let mean = |sum: f64, count: usize| if count == 0 { 0.0 } else { sum / count as f64 };
    let inner = mean(inner_diff, inner_count);
    FrameEvidence {
        frame_index: index,
        mean_luma: sum as f32 / pixel_count,
        crushed_fraction: crushed / pixel_count,
        clipped_fraction: clipped / pixel_count,
        texture_density: textured as f32 / interior.max(1) as f32,
        sharpness_ratio: if gradient_sum > 0.0 {
            (laplacian_sum / gradient_sum) as f32
        } else {
            0.0
        },
        blockiness: if inner > 0.5 {
            (mean(boundary_diff, boundary_count) / inner) as f32
        } else {
            1.0
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn pair_evidence(
    from: usize,
    to: usize,
    identical: bool,
    source: &[crate::Feature],
    target: &[crate::Feature],
    options: ReconstructionOptions,
    frames: &[FrameEvidence],
) -> PairEvidence {
    let matches = match_features(source, target, options);
    let correspondences: Vec<([f32; 2], [f32; 2])> = matches
        .iter()
        .map(|feature_match| {
            let a = &source[feature_match.a];
            let b = &target[feature_match.b];
            ([a.x as f32, a.y as f32], [b.x as f32, b.y as f32])
        })
        .collect();
    let features = source.len().min(target.len());
    let overlap = if features == 0 {
        0.0
    } else {
        correspondences.len() as f32 / features as f32
    };
    let mut motions: Vec<f32> = correspondences
        .iter()
        .map(|(a, b)| ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt())
        .collect();
    motions.sort_by(f32::total_cmp);
    let median_motion_pixels = motions.get(motions.len() / 2).copied().unwrap_or(0.0);
    let (homography_inlier_fraction, residual_p75_pixels, secondary_motion_fraction) =
        match robust_homography(&correspondences) {
            Some(homography) => {
                let residuals: Vec<f32> = correspondences
                    .iter()
                    .map(|(a, b)| transfer_error(&homography, *a, *b))
                    .collect();
                let inliers = residuals
                    .iter()
                    .filter(|residual| **residual <= RANSAC_INLIER_PIXELS)
                    .count();
                // Random mismatches do not agree on a motion; a separately
                // moving depth layer does.
                let outliers: Vec<_> = correspondences
                    .iter()
                    .zip(&residuals)
                    .filter(|(_, residual)| **residual > THRESHOLDS.parallax_residual_pixels * 2.0)
                    .map(|(correspondence, _)| *correspondence)
                    .collect();
                let secondary = if outliers.len() >= SECONDARY_MIN_MATCHES {
                    robust_homography(&outliers).map_or(0, |second| {
                        outliers
                            .iter()
                            .filter(|(a, b)| {
                                transfer_error(&second, *a, *b) <= RANSAC_INLIER_PIXELS
                            })
                            .count()
                    })
                } else {
                    0
                };
                let mut sorted = residuals.clone();
                sorted.sort_by(f32::total_cmp);
                let count = residuals.len() as f32;
                (
                    inliers as f32 / count,
                    sorted[(sorted.len() * 3) / 4],
                    if secondary >= SECONDARY_MIN_MATCHES {
                        secondary as f32 / count
                    } else {
                        0.0
                    },
                )
            }
            None => (0.0, 0.0, 0.0),
        };
    let before = frames[from].mean_luma.max(1.0);
    let relative_exposure_change = (frames[to].mean_luma - frames[from].mean_luma).abs() / before;
    let motion = if identical {
        PairMotion::Duplicate
    } else if correspondences.len() < 8 {
        PairMotion::Unknown
    } else if median_motion_pixels < THRESHOLDS.static_motion_pixels {
        PairMotion::Static
    } else if residual_p75_pixels >= THRESHOLDS.parallax_residual_pixels
        || secondary_motion_fraction >= THRESHOLDS.min_secondary_motion_fraction
    {
        PairMotion::Parallax
    } else {
        PairMotion::RotationOrPlanar
    };
    PairEvidence {
        from_frame: from,
        to_frame: to,
        features,
        matches: correspondences.len(),
        overlap,
        median_motion_pixels,
        homography_inlier_fraction,
        residual_p75_pixels,
        secondary_motion_fraction,
        relative_exposure_change,
        motion,
    }
}

fn issues(
    sampling: &NormalizedSampling,
    frames: &[FrameEvidence],
    pairs: &[PairEvidence],
) -> Vec<ReadinessIssue> {
    let t = THRESHOLDS;
    let mut issues = Vec::new();
    let mut push = |code, severity, frames: Vec<usize>, message: String| {
        issues.push(ReadinessIssue {
            code,
            severity,
            frames,
            message,
        })
    };
    let frame_count = frames.len();
    let majority = |count: usize| count * 2 > frame_count;
    let select = |predicate: &dyn Fn(&FrameEvidence) -> bool| -> Vec<usize> {
        frames
            .iter()
            .filter(|frame| predicate(frame))
            .map(|frame| frame.frame_index)
            .collect()
    };

    let weak_texture = select(&|frame| frame.texture_density < t.min_texture_density);
    if !weak_texture.is_empty() {
        let blocking = majority(weak_texture.len());
        push(
            IssueCode::WeakTexture,
            if blocking {
                Severity::Blocking
            } else {
                Severity::Warning
            },
            weak_texture.clone(),
            format!(
                "{} of {frame_count} frames have weak texture (textured pixel fraction below {})",
                weak_texture.len(),
                t.min_texture_density
            ),
        );
    }
    let blurred = select(&|frame| {
        frame.texture_density >= t.min_texture_density
            && frame.sharpness_ratio < t.min_sharpness_ratio
    });
    if !blurred.is_empty() {
        push(
            IssueCode::SevereBlur,
            if majority(blurred.len()) {
                Severity::Blocking
            } else {
                Severity::Warning
            },
            blurred.clone(),
            format!(
                "{} of {frame_count} frames are severely blurred (sharpness ratio below {})",
                blurred.len(),
                t.min_sharpness_ratio
            ),
        );
    }
    let exposure = select(&|frame| {
        frame.mean_luma < t.min_mean_luma
            || frame.mean_luma > t.max_mean_luma
            || frame.crushed_fraction > t.max_clipped_fraction
            || frame.clipped_fraction > t.max_clipped_fraction
    });
    if !exposure.is_empty() {
        push(
            IssueCode::PoorExposure,
            if majority(exposure.len()) {
                Severity::Blocking
            } else {
                Severity::Warning
            },
            exposure.clone(),
            format!(
                "{} of {frame_count} frames are under- or over-exposed",
                exposure.len()
            ),
        );
    }
    let blocky = select(&|frame| frame.blockiness > t.max_blockiness);
    if !blocky.is_empty() {
        push(
            IssueCode::DecodeArtifacts,
            Severity::Warning,
            blocky.clone(),
            format!(
                "{} of {frame_count} frames show 8x8 block artifacts (blockiness above {})",
                blocky.len(),
                t.max_blockiness
            ),
        );
    }

    let pair_frames = |predicate: &dyn Fn(&PairEvidence) -> bool| -> Vec<usize> {
        pairs
            .iter()
            .filter(|pair| predicate(pair))
            .map(|pair| pair.to_frame)
            .collect()
    };
    let unstable_exposure =
        pair_frames(&|pair| pair.relative_exposure_change > t.max_exposure_change);
    if !unstable_exposure.is_empty() {
        push(
            IssueCode::UnstableExposure,
            Severity::Warning,
            unstable_exposure.clone(),
            format!(
                "exposure changes by more than {:.0}% between {} neighbouring samples",
                t.max_exposure_change * 100.0,
                unstable_exposure.len()
            ),
        );
    }
    let duplicates = pair_frames(&|pair| pair.motion == PairMotion::Duplicate);
    if !duplicates.is_empty() {
        push(
            IssueCode::DuplicateFrames,
            if duplicates.len() == pairs.len() { Severity::Blocking } else { Severity::Warning },
            duplicates.clone(),
            format!(
                "{} samples decoded to the same pixels as their predecessor (frozen or failed seeks)",
                duplicates.len()
            ),
        );
    }
    let weak_overlap =
        pair_frames(&|pair| pair.motion != PairMotion::Duplicate && pair.overlap < t.min_overlap);
    if !weak_overlap.is_empty() {
        push(
            IssueCode::WeakOverlap,
            if weak_overlap.len() * 2 > pairs.len() {
                Severity::Blocking
            } else {
                Severity::Warning
            },
            weak_overlap.clone(),
            format!(
                "{} neighbouring sample pairs share too few features (overlap below {})",
                weak_overlap.len(),
                t.min_overlap
            ),
        );
    }
    let unstable = pair_frames(&|pair| {
        matches!(
            pair.motion,
            PairMotion::Parallax | PairMotion::RotationOrPlanar
        ) && pair.homography_inlier_fraction < t.min_stable_fraction
    });
    if !unstable.is_empty() {
        push(
            IssueCode::UnstableContent,
            Severity::Warning,
            unstable.clone(),
            format!(
                "{} sample pairs have little consistent scene motion (moving content or mismatches)",
                unstable.len()
            ),
        );
    }

    let parallax = pairs
        .iter()
        .filter(|pair| pair.motion == PairMotion::Parallax)
        .count();
    let rotation = pair_frames(&|pair| pair.motion == PairMotion::RotationOrPlanar);
    if parallax == 0 {
        push(
            IssueCode::NoParallax,
            Severity::Blocking,
            Vec::new(),
            "no sample pair shows translation parallax; the camera only rotated, held still, or filmed a flat scene".into(),
        );
    } else if rotation.len() > parallax {
        push(
            IssueCode::MostlyRotation,
            Severity::Warning,
            rotation.clone(),
            format!(
                "{} sample pairs are pure or near-pure rotation versus {parallax} with parallax",
                rotation.len()
            ),
        );
    }

    if !sampling.non_advancing_samples.is_empty() {
        push(
            IssueCode::TimingIrregular,
            Severity::Warning,
            sampling.non_advancing_samples.clone(),
            format!(
                "{} samples did not advance in presentation time",
                sampling.non_advancing_samples.len()
            ),
        );
    }
    issues
}

/// Deterministic RANSAC over 4-point homographies, refit on the inliers.
fn robust_homography(correspondences: &[([f32; 2], [f32; 2])]) -> Option<Matrix3<f64>> {
    let n = correspondences.len();
    if n < 4 {
        return None;
    }
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15 ^ n as u64;
    let mut next = |bound: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as usize) % bound
    };
    let mut best: Option<(usize, Matrix3<f64>)> = None;
    for _ in 0..RANSAC_ITERATIONS {
        let mut sample = [0usize; 4];
        let mut filled = 0;
        while filled < 4 {
            let candidate = next(n);
            if !sample[..filled].contains(&candidate) {
                sample[filled] = candidate;
                filled += 1;
            }
        }
        let subset: Vec<_> = sample.iter().map(|index| correspondences[*index]).collect();
        let Some(homography) = fit_homography(&subset) else {
            continue;
        };
        let inliers = correspondences
            .iter()
            .filter(|(a, b)| transfer_error(&homography, *a, *b) <= RANSAC_INLIER_PIXELS)
            .count();
        if best.as_ref().is_none_or(|(count, _)| inliers > *count) {
            best = Some((inliers, homography));
        }
    }
    let (_, homography) = best?;
    let inliers: Vec<_> = correspondences
        .iter()
        .copied()
        .filter(|(a, b)| transfer_error(&homography, *a, *b) <= RANSAC_INLIER_PIXELS)
        .collect();
    if inliers.len() >= 4 {
        fit_homography(&inliers).or(Some(homography))
    } else {
        Some(homography)
    }
}

/// Normalized DLT homography mapping `a` to `b`.
fn fit_homography(correspondences: &[([f32; 2], [f32; 2])]) -> Option<Matrix3<f64>> {
    let normalize = |points: Vec<[f32; 2]>| -> Option<(Vec<[f64; 2]>, Matrix3<f64>)> {
        let n = points.len() as f64;
        let cx = points.iter().map(|p| f64::from(p[0])).sum::<f64>() / n;
        let cy = points.iter().map(|p| f64::from(p[1])).sum::<f64>() / n;
        let spread = points
            .iter()
            .map(|p| ((f64::from(p[0]) - cx).powi(2) + (f64::from(p[1]) - cy).powi(2)).sqrt())
            .sum::<f64>()
            / n;
        if spread < 1e-6 {
            return None;
        }
        let scale = std::f64::consts::SQRT_2 / spread;
        let transform = Matrix3::new(
            scale,
            0.0,
            -scale * cx,
            0.0,
            scale,
            -scale * cy,
            0.0,
            0.0,
            1.0,
        );
        Some((
            points
                .iter()
                .map(|p| {
                    [
                        (f64::from(p[0]) - cx) * scale,
                        (f64::from(p[1]) - cy) * scale,
                    ]
                })
                .collect(),
            transform,
        ))
    };
    let (source, source_transform) = normalize(correspondences.iter().map(|(a, _)| *a).collect())?;
    let (target, target_transform) = normalize(correspondences.iter().map(|(_, b)| *b).collect())?;
    let rows = correspondences.len() * 2;
    let mut system = DMatrix::<f64>::zeros(rows.max(9), 9);
    for (index, (a, b)) in source.iter().zip(&target).enumerate() {
        let (x, y, u, v) = (a[0], a[1], b[0], b[1]);
        let row = index * 2;
        let first = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u];
        let second = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v];
        for column in 0..9 {
            system[(row, column)] = first[column];
            system[(row + 1, column)] = second[column];
        }
    }
    let svd = system.svd(false, true);
    let v_t = svd.v_t?;
    let (smallest, _) = svd
        .singular_values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))?;
    let h = v_t.row(smallest);
    let normalized = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    let homography = target_transform.try_inverse()? * normalized * source_transform;
    let scale = homography[(2, 2)];
    if scale.abs() < 1e-12 || !homography.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(homography / scale)
}

fn transfer_error(homography: &Matrix3<f64>, a: [f32; 2], b: [f32; 2]) -> f32 {
    let projected = homography * Vector3::new(f64::from(a[0]), f64::from(a[1]), 1.0);
    if projected.z.abs() < 1e-12 {
        return f32::INFINITY;
    }
    let x = projected.x / projected.z;
    let y = projected.y / projected.z;
    ((x - f64::from(b[0])).powi(2) + (y - f64::from(b[1])).powi(2)).sqrt() as f32
}

#[cfg(test)]
#[path = "input_readiness_tests.rs"]
mod tests;
