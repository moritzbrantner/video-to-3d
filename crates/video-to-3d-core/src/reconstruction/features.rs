//! Luminance conversion, corner detection and patch descriptors.

use super::control::ReconstructionControl;
use super::types::{Feature, FrameInput, ReconstructionOptions};
use std::cmp::Ordering;

pub(crate) fn to_luma(frame: &FrameInput) -> Vec<u8> {
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

pub(crate) fn detect_features(
    luma: &[u8],
    width: u32,
    height: u32,
    options: ReconstructionOptions,
) -> Vec<Feature> {
    // A callback that always returns false cannot interrupt feature extraction.
    detect_features_with_control(
        luma,
        width,
        height,
        options,
        &mut ReconstructionControl::new(&mut || false),
    )
    .unwrap_or_else(|_| unreachable!("uncancelable feature extraction"))
}

pub(super) fn detect_features_with_control(
    luma: &[u8],
    width: u32,
    height: u32,
    options: ReconstructionOptions,
    control: &mut ReconstructionControl<'_>,
) -> Result<Vec<Feature>, String> {
    let border = options.descriptor_radius.max(3) + 2;
    let mut candidates = Vec::new();

    for y in border..height.saturating_sub(border) {
        control.check()?;
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

    Ok(selected)
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

fn sample_luma(luma: &[u8], width: u32, x: u32, y: u32) -> u8 {
    luma[y as usize * width as usize + x as usize]
}

pub(super) fn sample_rgb(frame: &FrameInput, x: u32, y: u32) -> (u8, u8, u8) {
    let index = (y as usize * frame.width as usize + x as usize) * 4;
    (
        frame.rgba[index],
        frame.rgba[index + 1],
        frame.rgba[index + 2],
    )
}
