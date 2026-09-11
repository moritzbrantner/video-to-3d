use crate::{multi_view::RegisteredCamera, FrameInput, Point3};
use nalgebra::Vector3;
use serde::Serialize;
use std::cmp::Ordering;

const MAX_SOURCE_VIEWS: usize = 4;
const DEPTH_HYPOTHESES: usize = 24;
const MIN_VISIBLE_SPARSE_POINTS: usize = 8;
const PATCH_RADIUS: i32 = 1;
const MIN_REFERENCE_CONTRAST: f64 = 14.0;
const MAX_PHOTOMETRIC_ERROR: f64 = 18.0;
const MIN_AMBIGUITY_MARGIN: f64 = 1.0;
const MAX_RECIPROCAL_RELATIVE_DEPTH_ERROR: f64 = 0.08;

#[derive(Clone, Debug, Default, Serialize)]
pub struct DenseStats {
    pub attempted: bool,
    pub skip_reason: Option<String>,
    pub reference_frame: Option<usize>,
    pub source_views: usize,
    pub sampled_pixels: usize,
    pub depth_hypotheses: usize,
    pub accepted_points: usize,
    pub reciprocal_checked_points: usize,
    pub reciprocal_rejected_points: usize,
    pub median_supporting_views: Option<f32>,
    pub median_photometric_error: Option<f32>,
    pub search_min_depth: Option<f32>,
    pub search_max_depth: Option<f32>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct DenseAnalysis {
    pub stats: DenseStats,
    pub points: Vec<Point3>,
}

impl DenseAnalysis {
    fn skipped(reason: impl Into<String>) -> Self {
        Self {
            stats: DenseStats {
                skip_reason: Some(reason.into()),
                ..DenseStats::default()
            },
            points: Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    depth: f64,
    position: Vector3<f64>,
    support: usize,
    error: f64,
}

struct SourceView<'a> {
    camera: &'a RegisteredCamera,
    frame: &'a FrameInput,
    luma: Vec<u8>,
    search_min_depth: f64,
    search_max_depth: f64,
}

pub(super) fn estimate_depth_points(
    frames: &[FrameInput],
    cameras: &[RegisteredCamera],
    sparse_points: &[Vector3<f64>],
    focal: f64,
) -> DenseAnalysis {
    if frames.is_empty() {
        return DenseAnalysis::skipped("no sampled frames are available");
    }
    if cameras.len() < 2 {
        return DenseAnalysis::skipped("fewer than two accepted registered cameras are available");
    }
    if sparse_points.len() < MIN_VISIBLE_SPARSE_POINTS {
        return DenseAnalysis::skipped(format!(
            "fewer than {MIN_VISIBLE_SPARSE_POINTS} accepted sparse landmarks are available"
        ));
    }

    let width = frames[0].width;
    let height = frames[0].height;
    if width < 16 || height < 16 {
        return DenseAnalysis::skipped("sampled frames are too small for coarse depth estimation");
    }

    let Some((reference, mut visible_depths)) = cameras
        .iter()
        .filter_map(|camera| {
            let depths = visible_depths(camera, sparse_points, width, height, focal);
            (depths.len() >= MIN_VISIBLE_SPARSE_POINTS).then_some((camera, depths))
        })
        .max_by(|(left_camera, left_depths), (right_camera, right_depths)| {
            left_depths
                .len()
                .cmp(&right_depths.len())
                .then_with(|| right_camera.frame_index.cmp(&left_camera.frame_index))
        })
    else {
        return DenseAnalysis::skipped(format!(
            "no registered camera sees at least {MIN_VISIBLE_SPARSE_POINTS} accepted sparse landmarks inside its image"
        ));
    };

    visible_depths.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let lower = quantile(&visible_depths, 0.10);
    let upper = quantile(&visible_depths, 0.90);
    if !lower.is_finite() || !upper.is_finite() || lower <= 0.0 || upper <= 0.0 {
        return DenseAnalysis::skipped(
            "visible sparse landmarks do not provide a finite positive depth-search envelope",
        );
    }
    let search_min_depth = (lower * 0.8).max(1.0e-4);
    let search_max_depth = (upper * 1.25).max(search_min_depth * 1.2);
    let median_depth = quantile(&visible_depths, 0.50);

    let Some(reference_frame) = frames.get(reference.frame_index) else {
        return DenseAnalysis::skipped("the selected reference camera has no sampled frame");
    };

    let mut sources: Vec<(&RegisteredCamera, &FrameInput, f64, f64)> = cameras
        .iter()
        .filter(|camera| camera.frame_index != reference.frame_index)
        .filter_map(|camera| {
            let frame = frames.get(camera.frame_index)?;
            let baseline = (camera.camera_center() - reference.camera_center()).norm();
            if baseline < median_depth * 0.005
                || !projects_reference_center(reference, camera, median_depth, width, height, focal)
            {
                return None;
            }
            let (source_min_depth, source_max_depth) = depth_search_bounds(
                camera,
                sparse_points,
                frame.width,
                frame.height,
                focal,
            )?;
            Some((camera, frame, source_min_depth, source_max_depth))
        })
        .collect();
    sources.sort_by(|(left, _, _, _), (right, _, _, _)| {
        left.frame_index
            .abs_diff(reference.frame_index)
            .cmp(&right.frame_index.abs_diff(reference.frame_index))
            .then_with(|| left.frame_index.cmp(&right.frame_index))
    });
    sources.truncate(MAX_SOURCE_VIEWS);
    if sources.is_empty() {
        return DenseAnalysis::skipped(format!(
            "no second registered camera has sufficient baseline, overlap, and at least {MIN_VISIBLE_SPARSE_POINTS} visible sparse landmarks for reciprocal depth consistency"
        ));
    }

    let reference_luma = to_luma(reference_frame);
    let source_views: Vec<SourceView<'_>> = sources
        .into_iter()
        .map(
            |(camera, frame, source_min_depth, source_max_depth)| SourceView {
                camera,
                frame,
                luma: to_luma(frame),
                search_min_depth: source_min_depth,
                search_max_depth: source_max_depth,
            },
        )
        .collect();

    let stride = (width.min(height) / 48).clamp(4, 12) as usize;
    let border = (PATCH_RADIUS + 2) as u32;
    let mut points = Vec::new();
    let mut errors = Vec::new();
    let mut supports = Vec::new();
    let mut sampled_pixels = 0usize;
    let mut reciprocal_checked_points = 0usize;
    let mut reciprocal_rejected_points = 0usize;

    for y in (border..height - border).step_by(stride) {
        for x in (border..width - border).step_by(stride) {
            if patch_contrast(&reference_luma, width, height, x as f64, y as f64)
                < MIN_REFERENCE_CONTRAST
            {
                continue;
            }
            sampled_pixels += 1;

            let mut candidates = Vec::with_capacity(DEPTH_HYPOTHESES);
            for hypothesis in 0..DEPTH_HYPOTHESES {
                let t = if DEPTH_HYPOTHESES == 1 {
                    0.0
                } else {
                    hypothesis as f64 / (DEPTH_HYPOTHESES - 1) as f64
                };
                let inverse_depth =
                    (1.0 / search_min_depth) * (1.0 - t) + (1.0 / search_max_depth) * t;
                let depth = 1.0 / inverse_depth;
                let position =
                    unproject(reference, x as f64, y as f64, depth, width, height, focal);

                let mut source_errors = Vec::new();
                for source in &source_views {
                    let Some(error) = patch_error(
                        reference,
                        source.camera,
                        &reference_luma,
                        &source.luma,
                        width,
                        height,
                        source.frame.width,
                        source.frame.height,
                        x as f64,
                        y as f64,
                        depth,
                        focal,
                    ) else {
                        continue;
                    };
                    if error <= MAX_PHOTOMETRIC_ERROR {
                        source_errors.push(error);
                    }
                }
                if source_errors.is_empty() {
                    continue;
                }
                source_errors
                    .sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
                candidates.push(Candidate {
                    depth,
                    position,
                    support: source_errors.len(),
                    error: quantile(&source_errors, 0.50),
                });
            }

            candidates.sort_by(|left, right| {
                right
                    .support
                    .cmp(&left.support)
                    .then_with(|| {
                        left.error
                            .partial_cmp(&right.error)
                            .unwrap_or(Ordering::Equal)
                    })
                    .then_with(|| {
                        left.depth
                            .partial_cmp(&right.depth)
                            .unwrap_or(Ordering::Equal)
                    })
            });
            let Some(best) = candidates.first().copied() else {
                continue;
            };
            if best.error > MAX_PHOTOMETRIC_ERROR {
                continue;
            }

            let comparable_second = candidates
                .iter()
                .skip(1)
                .find(|candidate| candidate.support == best.support);
            let ambiguity_margin =
                comparable_second.map_or(f64::INFINITY, |second| second.error - best.error);
            if ambiguity_margin < MIN_AMBIGUITY_MARGIN {
                continue;
            }
            reciprocal_checked_points += 1;
            if !has_reciprocal_depth_agreement(
                best.position,
                reference,
                reference_frame,
                &reference_luma,
                &source_views,
                focal,
            ) {
                reciprocal_rejected_points += 1;
                continue;
            }

            let support_confidence = best.support as f64 / source_views.len() as f64;
            let error_confidence = 1.0 / (1.0 + best.error / 8.0);
            let margin_confidence = if ambiguity_margin.is_finite() {
                (ambiguity_margin / 6.0).clamp(0.2, 1.0)
            } else {
                1.0
            };
            let confidence =
                (support_confidence * error_confidence * margin_confidence).clamp(0.05, 1.0) as f32;
            let (r, g, b) = sample_rgb(reference_frame, x, y);
            points.push(Point3 {
                x: best.position.x as f32,
                y: best.position.y as f32,
                z: best.position.z as f32,
                confidence,
                r,
                g,
                b,
            });
            errors.push(best.error);
            supports.push(best.support as f64);
        }
    }

    DenseAnalysis {
        stats: DenseStats {
            attempted: true,
            skip_reason: None,
            reference_frame: Some(reference.frame_index),
            source_views: source_views.len(),
            sampled_pixels,
            depth_hypotheses: DEPTH_HYPOTHESES,
            accepted_points: points.len(),
            reciprocal_checked_points,
            reciprocal_rejected_points,
            median_supporting_views: median_option(&mut supports).map(|value| value as f32),
            median_photometric_error: median_option(&mut errors).map(|value| value as f32),
            search_min_depth: Some(search_min_depth as f32),
            search_max_depth: Some(search_max_depth as f32),
        },
        points,
    }
}

fn depth_search_bounds(
    camera: &RegisteredCamera,
    sparse_points: &[Vector3<f64>],
    width: u32,
    height: u32,
    focal: f64,
) -> Option<(f64, f64)> {
    let mut depths = visible_depths(camera, sparse_points, width, height, focal);
    if depths.len() < MIN_VISIBLE_SPARSE_POINTS {
        return None;
    }
    depths.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let lower = quantile(&depths, 0.10);
    let upper = quantile(&depths, 0.90);
    if !lower.is_finite() || !upper.is_finite() || lower <= 0.0 || upper <= 0.0 {
        return None;
    }
    let minimum = (lower * 0.8).max(1.0e-4);
    let maximum = (upper * 1.25).max(minimum * 1.2);
    Some((minimum, maximum))
}

fn has_reciprocal_depth_agreement(
    position: Vector3<f64>,
    reference_camera: &RegisteredCamera,
    reference_frame: &FrameInput,
    reference_luma: &[u8],
    sources: &[SourceView<'_>],
    focal: f64,
) -> bool {
    let Some((reference_x, reference_y, reference_depth)) = project(
        reference_camera,
        position,
        reference_frame.width,
        reference_frame.height,
        focal,
    ) else {
        return false;
    };

    sources.iter().any(|source| {
        let Some(direct_error) = patch_error(
            reference_camera,
            source.camera,
            reference_luma,
            &source.luma,
            reference_frame.width,
            reference_frame.height,
            source.frame.width,
            source.frame.height,
            reference_x,
            reference_y,
            reference_depth,
            focal,
        ) else {
            return false;
        };
        if direct_error > MAX_PHOTOMETRIC_ERROR {
            return false;
        }

        let Some((source_x, source_y, expected_source_depth)) = project(
            source.camera,
            position,
            source.frame.width,
            source.frame.height,
            focal,
        ) else {
            return false;
        };
        let Some(reciprocal_depth) = estimate_single_view_depth(
            source.camera,
            reference_camera,
            &source.luma,
            reference_luma,
            source.frame.width,
            source.frame.height,
            reference_frame.width,
            reference_frame.height,
            source_x,
            source_y,
            source.search_min_depth,
            source.search_max_depth,
            focal,
        ) else {
            return false;
        };

        reciprocal_depth_agrees(expected_source_depth, reciprocal_depth)
    })
}

#[allow(clippy::too_many_arguments)]
fn estimate_single_view_depth(
    reference_camera: &RegisteredCamera,
    source_camera: &RegisteredCamera,
    reference_luma: &[u8],
    source_luma: &[u8],
    reference_width: u32,
    reference_height: u32,
    source_width: u32,
    source_height: u32,
    x: f64,
    y: f64,
    search_min_depth: f64,
    search_max_depth: f64,
    focal: f64,
) -> Option<f64> {
    if patch_contrast(reference_luma, reference_width, reference_height, x, y)
        < MIN_REFERENCE_CONTRAST
    {
        return None;
    }

    let mut candidates = Vec::with_capacity(DEPTH_HYPOTHESES);
    for hypothesis in 0..DEPTH_HYPOTHESES {
        let t = if DEPTH_HYPOTHESES == 1 {
            0.0
        } else {
            hypothesis as f64 / (DEPTH_HYPOTHESES - 1) as f64
        };
        let inverse_depth =
            (1.0 / search_min_depth) * (1.0 - t) + (1.0 / search_max_depth) * t;
        let depth = 1.0 / inverse_depth;
        let Some(error) = patch_error(
            reference_camera,
            source_camera,
            reference_luma,
            source_luma,
            reference_width,
            reference_height,
            source_width,
            source_height,
            x,
            y,
            depth,
            focal,
        ) else {
            continue;
        };
        if error <= MAX_PHOTOMETRIC_ERROR {
            candidates.push((depth, error));
        }
    }

    candidates.sort_by(|(left_depth, left_error), (right_depth, right_error)| {
        left_error
            .partial_cmp(right_error)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left_depth
                    .partial_cmp(right_depth)
                    .unwrap_or(Ordering::Equal)
            })
    });
    candidates.first().map(|(depth, _)| *depth)
}

fn reciprocal_depth_agrees(expected: f64, estimated: f64) -> bool {
    if !expected.is_finite() || !estimated.is_finite() || expected <= 0.0 || estimated <= 0.0 {
        return false;
    }
    let scale = expected.max(estimated).max(1.0e-9);
    (expected - estimated).abs() / scale <= MAX_RECIPROCAL_RELATIVE_DEPTH_ERROR
}

fn visible_depths(
    camera: &RegisteredCamera,
    sparse_points: &[Vector3<f64>],
    width: u32,
    height: u32,
    focal: f64,
) -> Vec<f64> {
    sparse_points
        .iter()
        .filter_map(|point| {
            let (x, y, depth) = project(camera, *point, width, height, focal)?;
            in_image_bounds(x, y, width, height).then_some(depth)
        })
        .collect()
}

fn projects_reference_center(
    reference: &RegisteredCamera,
    source: &RegisteredCamera,
    depth: f64,
    width: u32,
    height: u32,
    focal: f64,
) -> bool {
    let point = unproject(
        reference,
        width as f64 * 0.5,
        height as f64 * 0.5,
        depth,
        width,
        height,
        focal,
    );
    project(source, point, width, height, focal)
        .is_some_and(|(x, y, _)| in_patch_bounds(x, y, width, height))
}

fn unproject(
    camera: &RegisteredCamera,
    x: f64,
    y: f64,
    depth: f64,
    width: u32,
    height: u32,
    focal: f64,
) -> Vector3<f64> {
    let camera_point = Vector3::new(
        (x - width as f64 * 0.5) / focal * depth,
        (y - height as f64 * 0.5) / focal * depth,
        depth,
    );
    camera.rotation.transpose() * (camera_point - camera.translation)
}

fn project(
    camera: &RegisteredCamera,
    point: Vector3<f64>,
    width: u32,
    height: u32,
    focal: f64,
) -> Option<(f64, f64, f64)> {
    let camera_point = camera.rotation * point + camera.translation;
    if !camera_point.iter().all(|value| value.is_finite()) || camera_point.z <= 1.0e-4 {
        return None;
    }
    Some((
        focal * camera_point.x / camera_point.z + width as f64 * 0.5,
        focal * camera_point.y / camera_point.z + height as f64 * 0.5,
        camera_point.z,
    ))
}

#[allow(clippy::too_many_arguments)]
fn patch_error(
    reference_camera: &RegisteredCamera,
    source_camera: &RegisteredCamera,
    reference_luma: &[u8],
    source_luma: &[u8],
    reference_width: u32,
    reference_height: u32,
    source_width: u32,
    source_height: u32,
    x: f64,
    y: f64,
    depth: f64,
    focal: f64,
) -> Option<f64> {
    let mut reference_values = Vec::with_capacity(9);
    let mut source_values = Vec::with_capacity(9);
    for dy in -PATCH_RADIUS..=PATCH_RADIUS {
        for dx in -PATCH_RADIUS..=PATCH_RADIUS {
            let reference_x = x + dx as f64;
            let reference_y = y + dy as f64;
            let world = unproject(
                reference_camera,
                reference_x,
                reference_y,
                depth,
                reference_width,
                reference_height,
                focal,
            );
            let (source_x, source_y, _) =
                project(source_camera, world, source_width, source_height, focal)?;
            if !in_bilinear_bounds(source_x, source_y, source_width, source_height) {
                return None;
            }
            reference_values.push(sample_bilinear(
                reference_luma,
                reference_width,
                reference_height,
                reference_x,
                reference_y,
            )?);
            source_values.push(sample_bilinear(
                source_luma,
                source_width,
                source_height,
                source_x,
                source_y,
            )?);
        }
    }

    let reference_mean = reference_values.iter().sum::<f64>() / reference_values.len() as f64;
    let source_mean = source_values.iter().sum::<f64>() / source_values.len() as f64;
    Some(
        reference_values
            .iter()
            .zip(&source_values)
            .map(|(reference, source)| {
                ((reference - reference_mean) - (source - source_mean)).abs()
            })
            .sum::<f64>()
            / reference_values.len() as f64,
    )
}

fn patch_contrast(luma: &[u8], width: u32, height: u32, x: f64, y: f64) -> f64 {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for dy in -PATCH_RADIUS..=PATCH_RADIUS {
        for dx in -PATCH_RADIUS..=PATCH_RADIUS {
            let Some(value) = sample_bilinear(luma, width, height, x + dx as f64, y + dy as f64)
            else {
                return 0.0;
            };
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
    }
    maximum - minimum
}

fn sample_bilinear(luma: &[u8], width: u32, height: u32, x: f64, y: f64) -> Option<f64> {
    if !in_bilinear_bounds(x, y, width, height) {
        return None;
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;
    let value = |px: u32, py: u32| luma[py as usize * width as usize + px as usize] as f64;
    let top = value(x0, y0) * (1.0 - tx) + value(x1, y0) * tx;
    let bottom = value(x0, y1) * (1.0 - tx) + value(x1, y1) * tx;
    Some(top * (1.0 - ty) + bottom * ty)
}

fn in_image_bounds(x: f64, y: f64, width: u32, height: u32) -> bool {
    x.is_finite() && y.is_finite() && x >= 0.0 && y >= 0.0 && x < width as f64 && y < height as f64
}

fn in_bilinear_bounds(x: f64, y: f64, width: u32, height: u32) -> bool {
    x.is_finite()
        && y.is_finite()
        && x >= 0.0
        && y >= 0.0
        && x < width.saturating_sub(1) as f64
        && y < height.saturating_sub(1) as f64
}

fn in_patch_bounds(x: f64, y: f64, width: u32, height: u32) -> bool {
    let margin = (PATCH_RADIUS + 1) as f64;
    x >= margin && y >= margin && x < width as f64 - margin && y < height as f64 - margin
}

fn to_luma(frame: &FrameInput) -> Vec<u8> {
    frame
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| {
            ((77 * pixel[0] as u16 + 150 * pixel[1] as u16 + 29 * pixel[2] as u16) >> 8) as u8
        })
        .collect()
}

fn sample_rgb(frame: &FrameInput, x: u32, y: u32) -> (u8, u8, u8) {
    let index = (y as usize * frame.width as usize + x as usize) * 4;
    (
        frame.rgba[index],
        frame.rgba[index + 1],
        frame.rgba[index + 2],
    )
}

fn quantile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let index = ((values.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    values[index]
}

fn median_option(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    Some(quantile(values, 0.50))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    fn camera(frame_index: usize, center_x: f64) -> RegisteredCamera {
        RegisteredCamera {
            frame_index,
            rotation: Matrix3::identity(),
            translation: Vector3::new(-center_x, 0.0, 0.0),
        }
    }

    fn texture(world_x: f64, world_y: f64) -> u8 {
        let value = 128.0
            + 55.0 * (world_x * 15.0 + world_y * 2.7).sin()
            + 40.0 * (world_y * 17.0 - world_x * 3.1).cos()
            + 25.0 * ((world_x + world_y) * 11.0).sin();
        value.round().clamp(0.0, 255.0) as u8
    }

    fn plane_frame(width: u32, height: u32, focal: f64, center_x: f64, depth: f64) -> FrameInput {
        let mut rgba = vec![0u8; width as usize * height as usize * 4];
        for y in 0..height {
            for x in 0..width {
                let world_x = center_x + (x as f64 - width as f64 * 0.5) / focal * depth;
                let world_y = (y as f64 - height as f64 * 0.5) / focal * depth;
                let value = texture(world_x, world_y);
                let index = (y as usize * width as usize + x as usize) * 4;
                rgba[index] = value;
                rgba[index + 1] = value;
                rgba[index + 2] = value;
                rgba[index + 3] = 255;
            }
        }
        FrameInput {
            width,
            height,
            rgba,
        }
    }

    fn plane_sparse_points(width: u32, height: u32, focal: f64, depth: f64) -> Vec<Vector3<f64>> {
        let mut points = Vec::new();
        for y in [10u32, 18, 26, 34] {
            for x in [12u32, 24, 36, 48, 56] {
                points.push(Vector3::new(
                    (x as f64 - width as f64 * 0.5) / focal * depth,
                    (y as f64 - height as f64 * 0.5) / focal * depth,
                    depth,
                ));
            }
        }
        points
    }

    #[test]
    fn estimates_coarse_depth_on_textured_plane() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let depth = 4.0;
        let frames = vec![
            plane_frame(width, height, focal, 0.0, depth),
            plane_frame(width, height, focal, 0.18, depth),
            plane_frame(width, height, focal, -0.16, depth),
        ];
        let cameras = vec![camera(0, 0.0), camera(1, 0.18), camera(2, -0.16)];
        let sparse = plane_sparse_points(width, height, focal, depth);

        let result = estimate_depth_points(&frames, &cameras, &sparse, focal);

        assert!(result.stats.attempted);
        assert!(result.stats.skip_reason.is_none());
        assert_eq!(result.stats.reference_frame, Some(0));
        assert_eq!(result.stats.source_views, 2);
        assert_eq!(
            result.stats.reciprocal_checked_points,
            result.stats.accepted_points + result.stats.reciprocal_rejected_points
        );
        assert!(
            result.points.len() >= 12,
            "accepted only {} reciprocal-consistent dense samples",
            result.points.len()
        );
        let mut depths: Vec<f64> = result.points.iter().map(|point| point.z as f64).collect();
        depths.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        let median_depth = quantile(&depths, 0.50);
        assert!(
            (median_depth - depth).abs() < 0.55,
            "median dense depth was {median_depth}"
        );
    }

    #[test]
    fn reciprocal_depth_search_recovers_source_view_depth() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let depth = 4.0;
        let frames = [
            plane_frame(width, height, focal, 0.0, depth),
            plane_frame(width, height, focal, 0.18, depth),
        ];
        let cameras = [camera(0, 0.0), camera(1, 0.18)];
        let sparse = plane_sparse_points(width, height, focal, depth);
        let (minimum, maximum) =
            depth_search_bounds(&cameras[1], &sparse, width, height, focal).expect("depth bounds");
        let world = Vector3::new(0.0, 0.0, depth);
        let (x, y, expected_depth) =
            project(&cameras[1], world, width, height, focal).expect("visible source point");
        let source_luma = to_luma(&frames[1]);
        let reference_luma = to_luma(&frames[0]);

        let estimated_depth = estimate_single_view_depth(
            &cameras[1],
            &cameras[0],
            &source_luma,
            &reference_luma,
            width,
            height,
            width,
            height,
            x,
            y,
            minimum,
            maximum,
            focal,
        )
        .expect("reciprocal source depth");

        assert!(
            reciprocal_depth_agrees(expected_depth, estimated_depth),
            "expected source depth {expected_depth}, got {estimated_depth}"
        );
    }

    #[test]
    fn reciprocal_consistency_rejects_wrong_depth() {
        assert!(reciprocal_depth_agrees(4.0, 4.2));
        assert!(!reciprocal_depth_agrees(4.0, 5.0));
    }

    #[test]
    fn rejects_textureless_depth_hypotheses() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let frame = FrameInput {
            width,
            height,
            rgba: vec![120; width as usize * height as usize * 4],
        };
        let cameras = vec![camera(0, 0.0), camera(1, 0.2)];
        let sparse = plane_sparse_points(width, height, focal, 4.0);

        let result = estimate_depth_points(&[frame.clone(), frame], &cameras, &sparse, focal);

        assert!(result.stats.attempted);
        assert!(result.stats.skip_reason.is_none());
        assert!(result.points.is_empty());
        assert_eq!(result.stats.accepted_points, 0);
    }

    #[test]
    fn does_not_attempt_dense_depth_without_registered_baseline() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let frame = plane_frame(width, height, focal, 0.0, 4.0);
        let sparse = plane_sparse_points(width, height, focal, 4.0);

        let result = estimate_depth_points(&[frame], &[camera(0, 0.0)], &sparse, focal);

        assert!(!result.stats.attempted);
        assert!(result
            .stats
            .skip_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("two accepted registered cameras")));
        assert!(result.points.is_empty());
    }

    #[test]
    fn visible_depths_ignore_offscreen_landmarks() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let camera = camera(0, 0.0);
        let sparse = vec![
            Vector3::new(0.0, 0.0, 4.0),
            Vector3::new(100.0, 0.0, 4.0),
            Vector3::new(0.0, -100.0, 4.0),
        ];

        let depths = visible_depths(&camera, &sparse, width, height, focal);

        assert_eq!(depths, vec![4.0]);
    }
}
