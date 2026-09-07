use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FrameInput {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct ReconstructionOptions {
    pub max_features: usize,
    pub min_feature_distance: u32,
    pub descriptor_radius: u32,
    pub match_radius: u32,
    pub max_descriptor_distance: f32,
    pub ratio_threshold: f32,
}

impl Default for ReconstructionOptions {
    fn default() -> Self {
        Self {
            max_features: 320,
            min_feature_distance: 7,
            descriptor_radius: 3,
            match_radius: 42,
            max_descriptor_distance: 36.0,
            ratio_threshold: 0.82,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReconstructionRequest {
    pub frames: Vec<FrameInput>,
    #[serde(default)]
    pub options: ReconstructionOptions,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CameraPose {
    pub frame_index: usize,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub matched_features: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub confidence: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PairStats {
    pub from_frame: usize,
    pub to_frame: usize,
    pub features_from: usize,
    pub features_to: usize,
    pub matches: usize,
    pub median_dx: f32,
    pub median_dy: f32,
    pub median_motion: f32,
    pub low_parallax: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReconstructionResult {
    pub cameras: Vec<CameraPose>,
    pub points: Vec<Point3>,
    pub pairs: Vec<PairStats>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
struct Feature {
    x: u32,
    y: u32,
    score: f32,
    descriptor: Vec<i16>,
}

#[derive(Clone, Copy, Debug)]
struct FeatureMatch {
    a: usize,
    b: usize,
    distance: f32,
}

pub fn reconstruct(request: &ReconstructionRequest) -> Result<ReconstructionResult, String> {
    if request.frames.len() < 2 {
        return Err("at least two sampled frames are required".into());
    }

    let width = request.frames[0].width;
    let height = request.frames[0].height;
    if width < 32 || height < 24 {
        return Err("sampled frames are too small for reconstruction".into());
    }

    for frame in &request.frames {
        if frame.width != width || frame.height != height {
            return Err("all sampled frames must share dimensions".into());
        }
        let expected = width as usize * height as usize * 4;
        if frame.rgba.len() != expected {
            return Err(format!(
                "invalid RGBA payload: expected {expected} bytes, got {}",
                frame.rgba.len()
            ));
        }
    }

    let options = request.options;
    let luma_frames: Vec<Vec<u8>> = request.frames.iter().map(to_luma).collect();
    let features: Vec<Vec<Feature>> = luma_frames
        .iter()
        .map(|luma| detect_features(luma, width, height, options))
        .collect();

    let focal = 0.86 * width.max(height) as f32;
    let diagonal = ((width * width + height * height) as f32).sqrt();
    let mut cameras = Vec::with_capacity(request.frames.len());
    let mut points = Vec::new();
    let mut pairs = Vec::with_capacity(request.frames.len() - 1);
    let mut warnings = Vec::new();

    let mut camera = CameraPose {
        frame_index: 0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        matched_features: 0,
    };
    cameras.push(camera);

    for pair_index in 0..request.frames.len() - 1 {
        let matches = match_features(
            &features[pair_index],
            &features[pair_index + 1],
            options,
        );
        let mut dx_values = Vec::with_capacity(matches.len());
        let mut dy_values = Vec::with_capacity(matches.len());
        let mut motion_values = Vec::with_capacity(matches.len());

        for feature_match in &matches {
            let a = &features[pair_index][feature_match.a];
            let b = &features[pair_index + 1][feature_match.b];
            let dx = b.x as f32 - a.x as f32;
            let dy = b.y as f32 - a.y as f32;
            dx_values.push(dx);
            dy_values.push(dy);
            motion_values.push(dx.hypot(dy));
        }

        let median_dx = median(&mut dx_values);
        let median_dy = median(&mut dy_values);
        let median_motion = median(&mut motion_values);
        let low_parallax = matches.len() < 10 || median_motion < 1.4;

        let observed_baseline = (median_motion / diagonal * 10.0).clamp(0.05, 1.0);
        camera = CameraPose {
            frame_index: pair_index + 1,
            x: camera.x - median_dx / width as f32 * 1.5,
            y: camera.y + median_dy / height as f32 * 1.5,
            z: camera.z + observed_baseline,
            matched_features: matches.len(),
        };
        cameras.push(camera);

        for feature_match in &matches {
            let a = &features[pair_index][feature_match.a];
            let b = &features[pair_index + 1][feature_match.b];
            let dx = b.x as f32 - a.x as f32;
            let dy = b.y as f32 - a.y as f32;
            let residual = (dx - median_dx).hypot(dy - median_dy);
            let raw_motion = dx.hypot(dy);
            let disparity = residual * 0.7 + raw_motion * 0.3;
            let depth = (focal * observed_baseline / disparity.max(1.0)).clamp(0.35, 18.0);
            let normalized_x = (a.x as f32 - width as f32 * 0.5) / focal;
            let normalized_y = (a.y as f32 - height as f32 * 0.5) / focal;
            let source_camera = cameras[pair_index];
            let (r, g, b_color) = sample_rgb(&request.frames[pair_index], a.x, a.y);
            let descriptor_confidence =
                (1.0 - feature_match.distance / options.max_descriptor_distance).clamp(0.0, 1.0);
            let motion_confidence = (disparity / 6.0).clamp(0.15, 1.0);

            points.push(Point3 {
                x: source_camera.x + normalized_x * depth,
                y: source_camera.y - normalized_y * depth,
                z: source_camera.z + depth,
                confidence: descriptor_confidence * motion_confidence,
                r,
                g: g_color,
                b: b_color,
            });
        }

        pairs.push(PairStats {
            from_frame: pair_index,
            to_frame: pair_index + 1,
            features_from: features[pair_index].len(),
            features_to: features[pair_index + 1].len(),
            matches: matches.len(),
            median_dx,
            median_dy,
            median_motion,
            low_parallax,
        });
    }

    let low_pairs = pairs.iter().filter(|pair| pair.low_parallax).count();
    if low_pairs > pairs.len() / 2 {
        warnings.push(
            "Most frame pairs have weak parallax. Move the camera through the scene rather than only rotating it."
                .into(),
        );
    }
    if points.len() < 80 {
        warnings.push(
            "The sparse cloud is small. Try a more textured, well-lit scene with slower camera motion."
                .into(),
        );
    }
    warnings.push(
        "MVP geometry uses an uncalibrated parallax approximation; metric scale and camera rotation are not recovered yet."
            .into(),
    );

    Ok(ReconstructionResult {
        cameras,
        points,
        pairs,
        warnings,
    })
}

fn to_luma(frame: &FrameInput) -> Vec<u8> {
    frame
        .rgba
        .chunks_exact(4)
        .map(|pixel| {
            ((77 * pixel[0] as u16 + 150 * pixel[1] as u16 + 29 * pixel[2] as u16) >> 8) as u8
        })
        .collect()
}

fn detect_features(
    luma: &[u8],
    width: u32,
    height: u32,
    options: ReconstructionOptions,
) -> Vec<Feature> {
    let border = options.descriptor_radius.max(3) + 2;
    let mut candidates = Vec::new();

    for y in border..height.saturating_sub(border) {
        for x in border..width.saturating_sub(border) {
            let mut sxx = 0.0f32;
            let mut syy = 0.0f32;
            let mut sxy = 0.0f32;
            for wy in y - 1..=y + 1 {
                for wx in x - 1..=x + 1 {
                    let gx = sample_luma(luma, width, wx + 1, wy) as f32
                        - sample_luma(luma, width, wx - 1, wy) as f32;
                    let gy = sample_luma(luma, width, wx, wy + 1) as f32
                        - sample_luma(luma, width, wx, wy - 1) as f32;
                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                }
            }
            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            let score = det - 0.04 * trace * trace;
            if score > 1_000_000.0 {
                candidates.push((x, y, score));
            }
        }
    }

    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
    let min_distance_sq = (options.min_feature_distance * options.min_feature_distance) as i64;
    let mut selected: Vec<Feature> = Vec::with_capacity(options.max_features);

    for (x, y, score) in candidates {
        if selected.iter().any(|feature| {
            let dx = feature.x as i64 - x as i64;
            let dy = feature.y as i64 - y as i64;
            dx * dx + dy * dy < min_distance_sq
        }) {
            continue;
        }

        selected.push(Feature {
            x,
            y,
            score,
            descriptor: descriptor(luma, width, x, y, options.descriptor_radius),
        });
        if selected.len() >= options.max_features {
            break;
        }
    }

    selected
}

fn descriptor(luma: &[u8], width: u32, x: u32, y: u32, radius: u32) -> Vec<i16> {
    let side = radius * 2 + 1;
    let mut values = Vec::with_capacity((side * side) as usize);
    let mut sum = 0i32;
    for py in y - radius..=y + radius {
        for px in x - radius..=x + radius {
            let value = sample_luma(luma, width, px, py) as i16;
            values.push(value);
            sum += value as i32;
        }
    }
    let mean = sum / values.len() as i32;
    for value in &mut values {
        *value -= mean as i16;
    }
    values
}

fn match_features(
    a: &[Feature],
    b: &[Feature],
    options: ReconstructionOptions,
) -> Vec<FeatureMatch> {
    let radius_sq = (options.match_radius * options.match_radius) as i64;
    let mut proposals = Vec::new();

    for (a_index, feature_a) in a.iter().enumerate() {
        let mut best: Option<(usize, f32)> = None;
        let mut second = f32::INFINITY;

        for (b_index, feature_b) in b.iter().enumerate() {
            let dx = feature_b.x as i64 - feature_a.x as i64;
            let dy = feature_b.y as i64 - feature_a.y as i64;
            if dx * dx + dy * dy > radius_sq {
                continue;
            }
            let distance = descriptor_distance(&feature_a.descriptor, &feature_b.descriptor);
            match best {
                None => best = Some((b_index, distance)),
                Some((_, best_distance)) if distance < best_distance => {
                    second = best_distance;
                    best = Some((b_index, distance));
                }
                Some(_) if distance < second => second = distance,
                _ => {}
            }
        }

        if let Some((b_index, best_distance)) = best {
            let ratio_ok = second.is_infinite() || best_distance < second * options.ratio_threshold;
            if ratio_ok && best_distance <= options.max_descriptor_distance {
                proposals.push(FeatureMatch {
                    a: a_index,
                    b: b_index,
                    distance: best_distance,
                });
            }
        }
    }

    proposals.sort_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b[right.b]
                    .score
                    .partial_cmp(&b[left.b].score)
                    .unwrap_or(Ordering::Equal)
            })
    });
    let mut used_b = HashSet::new();
    proposals
        .into_iter()
        .filter(|proposal| used_b.insert(proposal.b))
        .collect()
}

fn descriptor_distance(a: &[i16], b: &[i16]) -> f32 {
    let total: i32 = a
        .iter()
        .zip(b)
        .map(|(left, right)| (*left as i32 - *right as i32).abs())
        .sum();
    total as f32 / a.len() as f32
}

fn sample_luma(luma: &[u8], width: u32, x: u32, y: u32) -> u8 {
    luma[y as usize * width as usize + x as usize]
}

fn sample_rgb(frame: &FrameInput, x: u32, y: u32) -> (u8, u8, u8) {
    let index = (y as usize * frame.width as usize + x as usize) * 4;
    (
        frame.rgba[index],
        frame.rgba[index + 1],
        frame.rgba[index + 2],
    )
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_frame(width: u32, height: u32, shift_x: i32) -> FrameInput {
        let mut rgba = vec![18u8; width as usize * height as usize * 4];
        for pixel in rgba.chunks_exact_mut(4) {
            pixel[3] = 255;
        }

        for (index, (base_x, base_y)) in [
            (18, 16),
            (37, 21),
            (58, 15),
            (24, 39),
            (48, 44),
            (70, 36),
            (15, 58),
            (41, 62),
            (66, 57),
        ]
        .into_iter()
        .enumerate()
        {
            let x = base_x + shift_x;
            if x < 3 || x >= width as i32 - 4 {
                continue;
            }
            let intensity = 90 + index as u8 * 16;
            for py in base_y - 2..=base_y + 2 {
                for px in x - 2..=x + 2 {
                    let edge =
                        px == x - 2 || px == x + 2 || py == base_y - 2 || py == base_y + 2;
                    let value = if edge { 245 } else { intensity };
                    let offset = (py as usize * width as usize + px as usize) * 4;
                    rgba[offset] = value;
                    rgba[offset + 1] = value.saturating_sub(index as u8 * 2);
                    rgba[offset + 2] = value / 2;
                }
            }
        }

        FrameInput {
            width,
            height,
            rgba,
        }
    }

    #[test]
    fn detects_repeatable_corners() {
        let frame = synthetic_frame(96, 80, 0);
        let luma = to_luma(&frame);
        let features = detect_features(
            &luma,
            frame.width,
            frame.height,
            ReconstructionOptions::default(),
        );
        assert!(features.len() >= 12, "found only {} features", features.len());
    }

    #[test]
    fn reconstructs_shifted_sequence_into_sparse_output() {
        let request = ReconstructionRequest {
            frames: vec![
                synthetic_frame(96, 80, 0),
                synthetic_frame(96, 80, 3),
                synthetic_frame(96, 80, 6),
            ],
            options: ReconstructionOptions {
                match_radius: 12,
                max_descriptor_distance: 55.0,
                ..ReconstructionOptions::default()
            },
        };

        let result = reconstruct(&request).expect("reconstruction should succeed");
        assert_eq!(result.cameras.len(), 3);
        assert_eq!(result.pairs.len(), 2);
        assert!(
            result.pairs[0].matches >= 6,
            "too few matches: {}",
            result.pairs[0].matches
        );
        assert!(result.pairs[0].median_dx > 1.0);
        assert!(!result.points.is_empty());
    }

    #[test]
    fn rejects_malformed_rgba_payloads() {
        let request = ReconstructionRequest {
            frames: vec![
                FrameInput {
                    width: 64,
                    height: 48,
                    rgba: vec![0; 4],
                },
                synthetic_frame(64, 48, 0),
            ],
            options: ReconstructionOptions::default(),
        };

        assert!(reconstruct(&request).is_err());
    }
}
