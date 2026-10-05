use super::*;

const W: u32 = 192;
const H: u32 = 144;

fn hash(x: i32, y: i32, seed: u32) -> u8 {
    let mut value = (x as u32).wrapping_mul(0x9e37_79b1)
        ^ (y as u32).wrapping_mul(0x85eb_ca77)
        ^ seed.wrapping_mul(0xc2b2_ae3d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x2c1b_3c6d);
    value ^= value >> 12;
    (value & 0xff) as u8
}

/// Blocky random texture with plenty of corners.
fn texture(x: i32, y: i32, seed: u32) -> u8 {
    40 + (u16::from(hash(x.div_euclid(5), y.div_euclid(5), seed)) * 170 / 255) as u8
}

/// Background shifted by `background`, plus a nearer foreground block shifted
/// by `foreground` (different motion = translation parallax).
fn render(background: i32, foreground: Option<i32>, noise_seed: u32) -> Vec<u8> {
    let mut luma = vec![0u8; (W * H) as usize];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let mut value = texture(x + background, y, 1);
            if let Some(shift) = foreground {
                let left = 60 - shift;
                if (left..left + 72).contains(&x) && (36..108).contains(&y) {
                    value = texture(x - left, y, 2);
                }
            }
            let noise = i16::from(hash(x, y, noise_seed) % 3) - 1;
            luma[(y as u32 * W + x as u32) as usize] =
                (i16::from(value) + noise).clamp(0, 255) as u8;
        }
    }
    luma
}

fn frame(luma: &[u8]) -> FrameInput {
    FrameInput {
        width: W,
        height: H,
        rgba: luma
            .iter()
            .flat_map(|value| [*value, *value, *value, 255])
            .collect(),
    }
}

fn map(luma: &[u8], f: impl Fn(u8) -> u8) -> Vec<u8> {
    luma.iter().map(|value| f(*value)).collect()
}

fn box_blur(luma: &[u8], radius: i32) -> Vec<u8> {
    let mut out = vec![0u8; luma.len()];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let mut sum = 0u32;
            let mut count = 0u32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let (sx, sy) = (x + dx, y + dy);
                    if (0..W as i32).contains(&sx) && (0..H as i32).contains(&sy) {
                        sum += u32::from(luma[(sy as u32 * W + sx as u32) as usize]);
                        count += 1;
                    }
                }
            }
            out[(y as u32 * W + x as u32) as usize] = (sum / count) as u8;
        }
    }
    out
}

fn metadata(count: usize) -> SamplingMetadata {
    let times: Vec<f64> = (0..count).map(|index| 0.5 + index as f64 * 0.8).collect();
    SamplingMetadata {
        duration_seconds: 10.0,
        display_width: 1920,
        display_height: 1440,
        rotation_degrees: 0,
        requested_times: times.clone(),
        presented_times: times,
        presented_time_source: PresentedTimeSource::DecodedFrame,
    }
}

fn assess(lumas: &[Vec<u8>]) -> ReadinessReport {
    let frames: Vec<FrameInput> = lumas.iter().map(|luma| frame(luma)).collect();
    assess_readiness(&frames, &metadata(frames.len())).unwrap()
}

fn codes(report: &ReadinessReport) -> Vec<IssueCode> {
    report.issues.iter().map(|issue| issue.code).collect()
}

fn parallax_sequence() -> Vec<Vec<u8>> {
    (0..5)
        .map(|index| render(index * 2, Some(index * 6), index as u32 + 10))
        .collect()
}

#[test]
fn good_clip_with_parallax_is_ready() {
    let report = assess(&parallax_sequence());
    assert!(
        report
            .pairs
            .iter()
            .all(|pair| pair.motion == PairMotion::Parallax),
        "{:#?}",
        report.pairs
    );
    assert_eq!(
        report.geometric_verdict,
        ReadinessVerdict::Ready,
        "{:#?}",
        report.issues
    );
    assert!(report.generative_paths_allowed);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn pure_rotation_is_surfaced_as_no_parallax() {
    let lumas: Vec<Vec<u8>> = (0..5)
        .map(|index| render(index * 4, None, index as u32 + 20))
        .collect();
    let report = assess(&lumas);
    assert!(
        report
            .pairs
            .iter()
            .all(|pair| pair.motion == PairMotion::RotationOrPlanar),
        "{:#?}",
        report.pairs
    );
    assert!(codes(&report).contains(&IssueCode::NoParallax));
    assert_eq!(report.geometric_verdict, ReadinessVerdict::Unsuitable);
    assert!(
        report.generative_paths_allowed,
        "never blocks generative paths"
    );
}

#[test]
fn weak_texture_is_surfaced() {
    let lumas: Vec<Vec<u8>> = (0..4)
        .map(|index| map(&render(0, None, index + 30), |value| 120 + value / 64))
        .collect();
    let report = assess(&lumas);
    assert!(
        codes(&report).contains(&IssueCode::WeakTexture),
        "{:#?}",
        report.frames
    );
    assert_eq!(report.geometric_verdict, ReadinessVerdict::Unsuitable);
}

#[test]
fn severe_blur_is_surfaced() {
    let lumas: Vec<Vec<u8>> = parallax_sequence()
        .iter()
        .map(|luma| box_blur(luma, 3))
        .collect();
    let report = assess(&lumas);
    assert!(
        codes(&report).contains(&IssueCode::SevereBlur),
        "{:#?}",
        report.frames
    );
    let sharp = assess(&parallax_sequence());
    assert!(sharp.frames[0].sharpness_ratio > report.frames[0].sharpness_ratio);
}

#[test]
fn unstable_and_poor_exposure_are_surfaced() {
    let lumas: Vec<Vec<u8>> = parallax_sequence()
        .iter()
        .enumerate()
        .map(|(index, luma)| {
            if index % 2 == 0 {
                luma.clone()
            } else {
                map(luma, |value| (u16::from(value) * 6 / 10) as u8)
            }
        })
        .collect();
    let report = assess(&lumas);
    assert!(codes(&report).contains(&IssueCode::UnstableExposure));

    let dark: Vec<Vec<u8>> = parallax_sequence()
        .iter()
        .map(|luma| map(luma, |value| value / 8))
        .collect();
    let report = assess(&dark);
    assert!(
        codes(&report).contains(&IssueCode::PoorExposure),
        "{:#?}",
        report.frames
    );
}

#[test]
fn duplicate_frames_and_timing_are_surfaced() {
    let mut lumas = parallax_sequence();
    lumas[2] = lumas[1].clone();
    let frames: Vec<FrameInput> = lumas.iter().map(|luma| frame(luma)).collect();
    let mut sampling = metadata(frames.len());
    sampling.presented_times[2] = sampling.presented_times[1];
    let report = assess_readiness(&frames, &sampling).unwrap();
    assert_eq!(report.pairs[1].motion, PairMotion::Duplicate);
    assert!(codes(&report).contains(&IssueCode::DuplicateFrames));
    assert!(codes(&report).contains(&IssueCode::TimingIrregular));
    assert_eq!(report.sampling.non_advancing_samples, [2]);
    assert_eq!(report.geometric_verdict, ReadinessVerdict::Marginal);
}

#[test]
fn block_artifacts_are_surfaced() {
    let lumas: Vec<Vec<u8>> = parallax_sequence()
        .iter()
        .map(|luma| {
            let mut out = luma.clone();
            for by in 0..(H / 8) {
                for bx in 0..(W / 8) {
                    let mut sum = 0u32;
                    for y in 0..8 {
                        for x in 0..8 {
                            sum += u32::from(luma[((by * 8 + y) * W + bx * 8 + x) as usize]);
                        }
                    }
                    let mean = (sum / 64) as i32;
                    for y in 0..8 {
                        for x in 0..8 {
                            let index = ((by * 8 + y) * W + bx * 8 + x) as usize;
                            let detail = (i32::from(luma[index]) - mean) / 12;
                            out[index] = (mean + detail).clamp(0, 255) as u8;
                        }
                    }
                }
            }
            out
        })
        .collect();
    let report = assess(&lumas);
    assert!(
        codes(&report).contains(&IssueCode::DecodeArtifacts),
        "{:#?}",
        report.frames
    );
}

#[test]
fn readiness_is_deterministic() {
    let first = assess(&parallax_sequence());
    let second = assess(&parallax_sequence());
    assert_eq!(first, second);
}

#[test]
fn sampling_metadata_is_normalized_and_validated() {
    let mut sampling = metadata(3);
    sampling.rotation_degrees = -90;
    sampling.presented_times = vec![1.0, 1.75, 2.6];
    sampling.requested_times = vec![1.0, 1.8, 2.6];
    let normalized = normalize_sampling(&sampling, 3).unwrap();
    assert_eq!(normalized.rotation_degrees, 270);
    assert_eq!(normalized.timestamps[0], 0.0);
    assert!((normalized.max_seek_error_seconds - 0.05).abs() < 1e-9);
    assert!((normalized.median_interval_seconds - 0.85).abs() < 1e-9);

    sampling.rotation_degrees = 45;
    assert!(normalize_sampling(&sampling, 3).is_err());
    sampling.rotation_degrees = 0;
    assert!(normalize_sampling(&sampling, 4).is_err());
    sampling.presented_times[1] = f64::NAN;
    assert!(normalize_sampling(&sampling, 3).is_err());
}

#[test]
fn pair_matching_agrees_with_the_reconstruction_matcher() {
    // Includes displacements beyond the local match radius, where
    // reconstruction may switch to motion-guided matching.
    for step in [2, 70] {
        let lumas: Vec<Vec<u8>> = (0..4)
            .map(|index| render(index * step, Some(index * (step + 6)), index as u32 + 40))
            .collect();
        let frames: Vec<FrameInput> = lumas.iter().map(|luma| frame(luma)).collect();
        let report = assess_readiness(&frames, &metadata(frames.len())).unwrap();
        let reconstruction = crate::reconstruct_once(&crate::ReconstructionRequest {
            frames,
            options: ReconstructionOptions::default(),
        })
        .unwrap();
        let readiness: Vec<usize> = report.pairs.iter().map(|pair| pair.matches).collect();
        let reconstructed: Vec<usize> = reconstruction
            .pairs
            .iter()
            .map(|pair| pair.matches)
            .collect();
        assert_eq!(readiness, reconstructed, "step {step}");
    }
}
