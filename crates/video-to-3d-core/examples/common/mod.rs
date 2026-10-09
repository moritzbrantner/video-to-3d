//! Helpers shared by the golden benchmark examples. Not reconstruction semantics: these
//! only score pipeline output against generated truth.

use nalgebra::{Matrix3, Vector3};

/// Camera-center RMSE after the best similarity (Sim(3)) alignment of estimated centers
/// onto true centers, divided by the largest distance between any two true centers.
///
/// `matched` pairs each estimated center with its true center. Returns `None` for fewer
/// than three cameras, degenerate estimated centers, or a degenerate true trajectory.
pub fn normalized_center_rmse(
    matched: &[(Vector3<f64>, Vector3<f64>)],
    trajectory_span: f64,
) -> Option<f64> {
    if matched.len() < 3 || !(trajectory_span > 1e-12) {
        return None;
    }

    let count = matched.len() as f64;
    let estimated_centroid = matched
        .iter()
        .map(|(estimated, _)| *estimated)
        .sum::<Vector3<f64>>()
        / count;
    let truth_centroid = matched
        .iter()
        .map(|(_, truth)| *truth)
        .sum::<Vector3<f64>>()
        / count;

    let mut covariance = Matrix3::zeros();
    let mut estimated_variance = 0.0;
    for (estimated, truth) in matched {
        let estimated_centered = estimated - estimated_centroid;
        let truth_centered = truth - truth_centroid;
        covariance += estimated_centered * truth_centered.transpose();
        estimated_variance += estimated_centered.norm_squared();
    }
    if estimated_variance <= 1e-12 {
        return None;
    }

    let svd = covariance.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let v = v_t.transpose();
    let mut correction = Matrix3::identity();
    if (v * u.transpose()).determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    let rotation = v * correction * u.transpose();
    let signed_singular_sum = svd.singular_values[0]
        + svd.singular_values[1]
        + correction[(2, 2)] * svd.singular_values[2];
    let scale = signed_singular_sum / estimated_variance;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let translation = truth_centroid - scale * rotation * estimated_centroid;

    let squared_error = matched
        .iter()
        .map(|(estimated, truth)| {
            let aligned = scale * rotation * estimated + translation;
            (aligned - truth).norm_squared()
        })
        .sum::<f64>();
    Some((squared_error / count).sqrt() / trajectory_span)
}

/// Largest distance between any two of `centers`.
pub fn trajectory_span(centers: &[Vector3<f64>]) -> f64 {
    centers
        .iter()
        .enumerate()
        .flat_map(|(index, left)| {
            centers[index + 1..]
                .iter()
                .map(move |right| (left - right).norm())
        })
        .fold(0.0_f64, f64::max)
}
