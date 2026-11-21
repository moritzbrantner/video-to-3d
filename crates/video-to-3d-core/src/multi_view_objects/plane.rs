use super::{camera_center, ray_at, View};
use crate::object_identity::project;
use crate::object_placement::fit_plane;
use nalgebra::Vector3;

/// Intersect masked reference rays with a plane supported by scene landmarks.
/// A wall, pane, or floor exports as an open sheet with the visible boundary.
pub(super) fn plane_patch(
    reference: &View<'_>,
    views: &[&View<'_>],
    required: usize,
    support: &[[f64; 3]],
) -> Result<super::TriangleSurface, &'static str> {
    if support.len() < 12 {
        return Err("Too few multi-view landmarks support an architectural plane.");
    }
    let (normal, offset) = fit_plane(support).ok_or("Could not fit a structural plane.")?;
    let normal = Vector3::from(normal);
    let mut depth: Vec<_> = support
        .iter()
        .filter_map(|p| project(reference.cam, *p).map(|(_, d)| d))
        .collect();
    depth.sort_by(f64::total_cmp);
    let range = depth[depth.len() / 2];
    let inliers: Vec<_> = support
        .iter()
        .copied()
        .filter(|p| (normal.dot(&Vector3::from(*p)) - offset).abs() < range * 0.02)
        .collect();
    if inliers.len() < 12 || inliers.len() * 3 < support.len() * 2 {
        return Err("The structural landmarks do not agree on one plane.");
    }
    // Collinear points do not define a plane, even with a zero fit residual.
    let centre = inliers
        .iter()
        .fold(Vector3::zeros(), |sum, p| sum + Vector3::from(*p))
        / inliers.len() as f64;
    let covariance = inliers.iter().fold(nalgebra::Matrix3::zeros(), |sum, p| {
        let d = Vector3::from(*p) - centre;
        sum + d * d.transpose()
    });
    let mut variance: [f64; 3] = covariance.symmetric_eigen().eigenvalues.into();
    variance.sort_by(f64::total_cmp);
    if variance[1] <= f64::EPSILON || variance[1] < 0.0025 * variance[2] {
        return Err("The structural landmarks span only a line.");
    }
    let (normal, offset) =
        fit_plane(&inliers).ok_or("The structural inliers do not span a plane.")?;
    let normal = Vector3::from(normal);
    let eye = camera_center(reference.cam);
    let [x0, y0, x1, y1] = reference.obs.bbox;
    let step = ((x1 - x0).max(y1 - y0) / 72.).max(1.);
    let w = ((x1 - x0) / step).ceil() as usize + 1;
    let h = ((y1 - y0) / step).ceil() as usize + 1;
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut sites = vec![None; w * h];
    for y in 0..h {
        for x in 0..w {
            let ray = ray_at(reference.cam, [x0 + x as f64 * step, y0 + y as f64 * step]);
            let z = (offset - normal.dot(&eye)) / normal.dot(&ray);
            if !(range / 4. ..=range * 4.).contains(&z) {
                continue;
            }
            let p = (eye + z * ray).into();
            if views.iter().filter(|v| v.contains(p)).count() < required {
                continue;
            }
            sites[y * w + x] = Some(vertices.len() as u32);
            vertices.push(p);
        }
    }
    for y in 0..h.saturating_sub(1) {
        for x in 0..w.saturating_sub(1) {
            let i = y * w + x;
            for t in [[i, i + w, i + 1], [i + 1, i + w, i + w + 1]] {
                if let [Some(a), Some(b), Some(c)] = t.map(|j| sites[j]) {
                    faces.push([a, b, c]);
                }
            }
        }
    }
    Ok((vertices, faces))
}
