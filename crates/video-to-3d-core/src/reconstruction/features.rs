//! Luminance conversion, corner detection and patch descriptors.

use super::control::ReconstructionControl;
use super::types::{Feature, FrameInput, ReconstructionOptions};
use std::cmp::Ordering;

#[derive(Clone, Copy, PartialEq)]
pub(super) enum FeatureSpacing {
    Requested,
    RecoverScarceCorners,
}

pub(super) const SCARCE_FEATURE_TARGET: usize = 48;

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
        FeatureSpacing::Requested,
        &mut ReconstructionControl::new(&mut || false),
    )
    .unwrap_or_else(|_| unreachable!("uncancelable feature extraction"))
}

pub(super) fn detect_features_with_control(
    luma: &[u8],
    width: u32,
    height: u32,
    options: ReconstructionOptions,
    mode: FeatureSpacing,
    control: &mut ReconstructionControl<'_>,
) -> Result<Vec<Feature>, String> {
    let border = options.descriptor_radius.max(3) + 2;
    let mut candidates = Vec::new();
    let columns = width.saturating_sub(border).saturating_sub(border) as usize;
    // Gradient products of the rows `y - 1..=y + 1`, each starting at column `border - 1`.
    let mut window = [
        vec![[0i32; 3]; columns + 2],
        vec![[0i32; 3]; columns + 2],
        vec![[0i32; 3]; columns + 2],
    ];

    for y in border..height.saturating_sub(border) {
        control.check()?;
        if columns == 0 {
            continue;
        }
        if y == border {
            gradient_products(luma, width, border - 1, y - 1, &mut window[0]);
            gradient_products(luma, width, border - 1, y, &mut window[1]);
        } else {
            window.rotate_left(1);
        }
        gradient_products(luma, width, border - 1, y + 1, &mut window[2]);
        for column in 0..columns {
            let mut sum = [0i32; 3];
            for row in &window {
                for products in &row[column..column + 3] {
                    sum[0] += products[0];
                    sum[1] += products[1];
                    sum[2] += products[2];
                }
            }
            // Each sum is at most 9 * 255^2 in magnitude, so it converts to f32 exactly.
            let [sxx, syy, sxy] = sum.map(|value| value as f32);
            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            let score = det - 0.04 * trace * trace;
            if score > 1_000_000.0 {
                candidates.push((border + column as u32, y, score));
            }
        }
    }

    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
    let mut selected: Vec<Feature> = Vec::with_capacity(options.max_features);
    // Keep the well-separated set first. On feature-poor views, a second scan
    // can recover nearby corners on the same object. A descriptor radius plus
    // one keeps either feature's center outside the other's descriptor patch.
    let tighter_spacing = options.descriptor_radius.saturating_add(1);
    let sparse_target = match mode {
        FeatureSpacing::Requested => 0,
        FeatureSpacing::RecoverScarceCorners => options.max_features.min(SCARCE_FEATURE_TARGET),
    };
    for (spacing, limit) in [
        (options.min_feature_distance, options.max_features),
        (tighter_spacing, sparse_target),
    ] {
        if selected.len() >= limit {
            break;
        }
        if !selected.is_empty() && spacing >= options.min_feature_distance {
            break;
        }
        let min_distance_sq = u64::from(spacing).pow(2);
        for &(x, y, score) in &candidates {
            if selected.iter().any(|feature| {
                u64::from(feature.x.abs_diff(x)).pow(2) + u64::from(feature.y.abs_diff(y)).pow(2)
                    < min_distance_sq
            }) {
                continue;
            }
            selected.push(Feature {
                x,
                y,
                score,
                descriptor: descriptor(luma, width, x, y, options.descriptor_radius),
            });
            if selected.len() >= limit {
                break;
            }
        }
    }
    Ok(selected)
}

/// `[gx * gx, gy * gy, gx * gy]` of row `y`, for columns from `first_x`.
fn gradient_products(luma: &[u8], width: u32, first_x: u32, y: u32, row: &mut [[i32; 3]]) {
    for (x, products) in (first_x..).zip(row) {
        let gx =
            sample_luma(luma, width, x + 1, y) as i32 - sample_luma(luma, width, x - 1, y) as i32;
        let gy =
            sample_luma(luma, width, x, y + 1) as i32 - sample_luma(luma, width, x, y - 1) as i32;
        *products = [gx * gx, gy * gy, gx * gy];
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scarce_features_keep_separated_corners_and_recover_close_neighbors() {
        // Two small high-contrast buildings. Each roof/corner group is narrower
        // than the first selection distance but wider than a descriptor radius.
        let (w, h) = (64, 48);
        let mut image = vec![20; w * h];
        for y in 15..23 {
            for x in (12..18).chain(40..46) {
                image[y * w + x] = 210;
            }
        }
        let options = ReconstructionOptions {
            min_feature_distance: 14,
            ..ReconstructionOptions::default()
        };
        let select = |mode| {
            detect_features_with_control(
                &image,
                w as u32,
                h as u32,
                options,
                mode,
                &mut ReconstructionControl::new(&mut || false),
            )
            .unwrap()
            .into_iter()
            .map(|feature| (feature.x, feature.y))
            .collect::<Vec<_>>()
        };
        let separated = select(FeatureSpacing::Requested);
        let recovered = select(FeatureSpacing::RecoverScarceCorners);
        assert_eq!(separated.len(), 2);
        assert!(separated[0].0.abs_diff(separated[1].0) >= 14);
        assert!(recovered.starts_with(&separated));
        assert!(recovered.len() >= 6);
        assert_eq!(recovered, select(FeatureSpacing::RecoverScarceCorners));
        for (i, &(x, y)) in recovered.iter().enumerate() {
            assert!(recovered[..i]
                .iter()
                .all(|&(px, py)| x.abs_diff(px).pow(2) + y.abs_diff(py).pow(2) >= 16));
        }
    }
}
