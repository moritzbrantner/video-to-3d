//! Estimate a foreground depth interval from textured patches inside the fitting masks.
use super::{camera_axes, camera_center, ray_at, FramePixels, View};
use crate::object_identity::project;
use nalgebra::Vector3;

pub(crate) struct DepthBand {
    pub near: f64,
    pub far: f64,
    pub matches: usize,
    pub samples: Vec<([usize; 2], f64)>,
}

/// Measured depths along the reference camera's optical axis. Callers provide
/// fitting frames only, so held-out RGB and masks cannot influence the pose.
pub(crate) fn registered_depths(
    reference_frame: usize,
    fitting: &[(
        &crate::semantic_inventory::SemanticObservation,
        &crate::object_identity::FrameCamera,
    )],
    frames: &[FramePixels<'_>],
    initial: f64,
    span: f64,
) -> Option<DepthBand> {
    if frames.is_empty() {
        return None;
    }
    let views = fitting
        .iter()
        .filter_map(|(obs, cam)| View::new(cam, obs))
        .collect::<Vec<_>>();
    let reference = views
        .iter()
        .find(|view| view.cam.frame_index == reference_frame)?;
    depth_band(
        reference,
        &views.iter().collect::<Vec<_>>(),
        frames,
        initial,
        span,
    )
}

fn sample(image: &FramePixels<'_>, uv: [f64; 2]) -> Option<f64> {
    let [x, y] = uv.map(|n| n - 0.5);
    if !(x >= 0. && y >= 0. && x + 1. < image.1 as f64 && y + 1. < image.2 as f64) {
        return None;
    }
    let index = (y as usize * image.1 + x as usize) * 4;
    let gray = |i: usize| {
        let pixel = &image.3[i..i + 3];
        (f64::from(pixel[0]) + 2. * f64::from(pixel[1]) + f64::from(pixel[2])) * 0.25
    };
    let blend = |a: f64, b: f64, t: f64| a + (b - a) * t;
    Some(blend(
        blend(gray(index), gray(index + 4), x.fract()),
        blend(
            gray(index + 4 * image.1),
            gray(index + 4 * image.1 + 4),
            x.fract(),
        ),
        y.fract(),
    ))
}

pub(super) fn depth_band(
    reference: &View<'_>,
    fitting: &[&View<'_>],
    frames: &[FramePixels<'_>],
    initial: f64,
    span: f64,
) -> Option<DepthBand> {
    let image_for = |v: &View<'_>| {
        frames.iter().find(|p| {
            p.0 == v.cam.frame_index
                && p.1 == v.width
                && p.2 == v.height
                && p.3.len() == 4 * p.1 * p.2
        })
    };
    let image = image_for(reference)?;
    let eye = camera_center(reference.cam);
    let sources: Vec<_> = fitting
        .iter()
        .filter(|s| s.cam.frame_index != reference.cam.frame_index)
        .filter_map(|s| Some((*s, image_for(s)?)))
        .collect();
    if sources.is_empty() {
        return None;
    }
    let bounds = reference.obs.bbox;
    let stride = (((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / 450.)
        .sqrt()
        .ceil() as usize)
        .max(2);
    let axes = camera_axes(reference.cam);
    let taps = [
        (0., 0.),
        (-2., 0.),
        (2., 0.),
        (0., -2.),
        (0., 2.),
        (-1., -1.),
        (1., -1.),
        (-1., 1.),
        (1., 1.),
    ];
    let mut matches = Vec::new();
    let steps = 80;
    for y in (bounds[1].max(0.) as usize..(bounds[3].ceil() as usize).min(reference.height))
        .step_by(stride)
    {
        'pixel: for x in (bounds[0].max(0.) as usize
            ..(bounds[2].ceil() as usize).min(reference.width))
            .step_by(stride)
        {
            if !reference.mask[y * reference.width + x] {
                continue;
            }
            let uv = [x as f64 + 0.5, y as f64 + 0.5];
            let mut template = [0.; 9];
            let mut rays = [Vector3::zeros(); 9];
            let central = ray_at(reference.cam, uv);
            for (i, (dx, dy)) in taps.iter().copied().enumerate() {
                let Some(value) = sample(image, [uv[0] + dx, uv[1] + dy]) else {
                    continue 'pixel;
                };
                template[i] = value;
                rays[i] = central + (axes[0] * dx + axes[1] * dy) / reference.cam.focal_pixels;
            }
            let mean = template.iter().sum::<f64>() / 9.;
            if template.iter().map(|v| (v - mean).abs()).sum::<f64>() < 45. {
                continue;
            }
            let mut costs = vec![f64::INFINITY; steps];
            let depth = |i: usize| initial / (2.0 - 1.5 * i as f64 / (steps - 1) as f64);
            let mut scores = Vec::with_capacity(sources.len());
            for (i, cost) in costs.iter_mut().enumerate() {
                let z = depth(i);
                scores.clear();
                for (src, img) in &sources {
                    if !src.contains((eye + central * z).into()) {
                        continue;
                    }
                    let mut total = 0.;
                    for (ray, patch) in rays.iter().zip(template) {
                        let val = project(src.cam, (eye + ray * z).into())
                            .and_then(|(uv, _)| sample(img, uv));
                        total += val.map_or(50., |v| (v - patch).abs().min(50.));
                    }
                    scores.push(total / 9.);
                }
                if scores.len() * 2 >= sources.len() && !scores.is_empty() {
                    scores.sort_by(f64::total_cmp);
                    *cost = scores[scores.len() / 2];
                }
            }
            let (best, &cost) = costs.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1))?;
            let runner = costs
                .iter()
                .enumerate()
                .filter(|(i, _)| i.abs_diff(best) > 4)
                .map(|(_, v)| *v)
                .fold(f64::INFINITY, f64::min);
            if best > 0 && best + 1 < steps && cost < 14. && runner > cost * 1.25 + 1.5 {
                matches.push(([x, y], depth(best)));
            }
        }
    }
    if matches.len() < 8 {
        return None;
    }
    matches.sort_by(|a, b| a.1.total_cmp(&b.1));
    let n = matches.len();
    let near = matches[n / 5].1;
    let far = matches[4 * n / 5].1;
    // The physical footprint of an image mask scales with its measured depth.
    let span = span * matches[n / 2].1 / initial;
    if far - near > span * 1.5 {
        return None;
    }
    Some(DepthBand {
        near: near - span * 0.12,
        far: far + span * 0.12,
        matches: n,
        samples: matches
            .into_iter()
            .filter(|(_, z)| *z >= near && *z <= far)
            .collect(),
    })
}
