//! Per-stage verification of a reference fixture against its exact truth (issue #129).
//!
//! Every stage of the pipeline is measured on its own, so a failing fixture names the
//! stage that broke instead of only an end result. The measurements land in
//! `stage-report.json`; `tools/check_reference_fixtures.py` applies the tolerances from
//! `benchmarks/reference-fixtures.json`. Sampling is checked separately against the
//! browser's own TypeScript sampling function (`tools/check_reference_sampling.ts`).
//!
//! Reconstructions live in an arbitrary monocular frame. Wherever a stage needs metric
//! world coordinates, the estimated cameras are aligned to the true ones by one
//! similarity: rotation from the camera orientations, scale and translation from the
//! camera centers. Orientation-based rotation keeps the alignment well defined for the
//! collinear centers of a lateral pan.

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Vector3};
use serde_json::{json, Value};
use video_to_3d_core::{
    feature_analysis::{analyze_rgba_pair, FeatureAlgorithm, FeatureOptions},
    surface_materials::{bake_surface_materials, ReferenceImage},
    BrowserReconstructionResult, EvidenceOrigin, ReconstructionEvidenceView,
};

use super::{browser_options, relief_parallax_pixels, round6, Fixture, MeshTopology, Pose, View};

pub const STAGE_REPORT_SCHEMA: &str = "video-to-3d/reference-stage-report/v1";

/// A match is correct when the matched target feature lies this close to the true
/// correspondence of the source feature. Features sit on whole pixels, so quantization
/// alone contributes up to √2/2 px.
const MATCH_TOLERANCE_PIXELS: f64 = 1.5;
/// A dense point is accurate when its depth is within this fraction of the true depth.
const DEPTH_TOLERANCE: f64 = 0.02;
/// A mesh triangle is supported when its centroid lies this close (metres, along the
/// heightfield axis) to the true surface.
const SURFACE_TOLERANCE_M: f64 = 0.05;
/// Barycentric sample points per textured triangle.
const TEXEL_SAMPLES: [[f64; 3]; 4] = [
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
];

/// Inputs every stage reads.
pub struct StageInputs<'a> {
    pub fixture: &'a Fixture,
    pub result: &'a BrowserReconstructionResult,
    pub registered_images: usize,
    pub normalized_pose_rmse: Option<f64>,
    pub decisive_gate: &'a str,
    pub topology: &'a MeshTopology,
}

pub fn stage_report(inputs: &StageInputs<'_>) -> Value {
    let reconstruction = &inputs.result.reconstruction;
    let evidence = ReconstructionEvidenceView::from_classic(reconstruction)
        .expect("reference fixture evidence view");
    let cameras = estimated_cameras(&evidence);
    let truth: Vec<Pose> = inputs.fixture.views.iter().map(|view| view.pose).collect();
    let alignment = align(&cameras, &truth);
    let blocked = |stage: &str| {
        json!({
            "available": false,
            "blocked_by": format!("{stage}: no registered cameras to align with the truth"),
        })
    };
    let registration = registration_stage(inputs, &cameras, &truth, alignment.as_ref());
    let aligned = alignment.map(|global| Aligned::new(inputs, &evidence, &cameras, global));
    json!({
        "schema": STAGE_REPORT_SCHEMA,
        "case": inputs.fixture.case.name(),
        "tolerances": {
            "match_pixels": MATCH_TOLERANCE_PIXELS,
            "relative_depth": DEPTH_TOLERANCE,
            "surface_m": SURFACE_TOLERANCE_M,
        },
        "stages": {
            "features": features_stage(inputs),
            "seed": seed_stage(inputs),
            "registration": registration,
            "dense": aligned.as_ref().map_or_else(|| blocked("dense"), |aligned| dense_stage(inputs, aligned)),
            "mesh": aligned.as_ref().map_or_else(|| blocked("mesh"), |aligned| mesh_stage(inputs, &evidence, aligned)),
            "texture": aligned.as_ref().map_or_else(|| blocked("texture"), |aligned| texture_stage(inputs, &evidence, aligned)),
        },
    })
}

// ---------------------------------------------------------------------------------------
// Shared measurement helpers

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    })
}

fn quantile(mut values: Vec<f64>, fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * fraction).round() as usize;
    Some(values[index])
}

fn rounded(value: Option<f64>) -> Value {
    value.map_or(Value::Null, |value| json!(round6(value)))
}

fn rotation_angle_degrees(rotation: &Matrix3<f64>) -> f64 {
    ((rotation.trace() - 1.0) * 0.5)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn matrix(values: &[f32; 9]) -> Matrix3<f64> {
    Matrix3::from_row_slice(&values.map(f64::from))
}

fn vector(values: &[f32; 3]) -> Vector3<f64> {
    Vector3::new(values[0] as f64, values[1] as f64, values[2] as f64)
}

/// True camera-space point through pixel `(x, y)` of `view`, using its exact depth.
/// Whole-pixel feature and grid positions address pixel `x`, whose center is `x + 0.5`
/// in the truth's half-integer convention.
fn true_world_point(fixture: &Fixture, view: &View, x: u32, y: u32) -> Vector3<f64> {
    let intrinsics = &fixture.intrinsics;
    let depth = view.depth[(y * intrinsics.width + x) as usize] as f64;
    let camera = intrinsics.ray(x as f64 + 0.5, y as f64 + 0.5) * depth;
    view.pose.rotation.transpose() * camera + view.pose.center
}

/// Projection of `world` into `view`, or `None` outside the image or behind an occluder.
fn visible_projection(fixture: &Fixture, view: &View, world: &Vector3<f64>) -> Option<(f64, f64)> {
    let intrinsics = &fixture.intrinsics;
    let camera = view.pose.world_to_camera(world);
    let (u, v) = intrinsics.project(&camera)?;
    if !(0.0..intrinsics.width as f64).contains(&u) || !(0.0..intrinsics.height as f64).contains(&v)
    {
        return None;
    }
    let surface = view.depth[(v as u32 * intrinsics.width + u as u32) as usize] as f64;
    (camera.z <= surface + 0.01 * surface).then_some((u, v))
}

struct EstimatedCamera {
    frame: usize,
    /// World-to-camera rotation of the monocular reconstruction frame.
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
}

impl EstimatedCamera {
    fn center(&self) -> Vector3<f64> {
        -(self.rotation.transpose() * self.translation)
    }
}

fn estimated_cameras(evidence: &ReconstructionEvidenceView<'_>) -> Vec<EstimatedCamera> {
    evidence
        .cameras
        .iter()
        .map(|camera| EstimatedCamera {
            frame: camera.frame_index,
            rotation: matrix(&camera.rotation),
            translation: vector(&camera.translation),
        })
        .collect()
}

/// `truth_world = scale · rotation · estimated_world + translation`.
struct Alignment {
    rotation: Matrix3<f64>,
    scale: f64,
    translation: Vector3<f64>,
}

impl Alignment {
    fn apply(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.scale * self.rotation * point + self.translation
    }
}

/// Similarity from the reconstruction frame onto the truth. The rotation is the chordal
/// mean of `R_trueᵀ · R_estimated` over all cameras; scale and translation are the least-
/// squares fit of the rotated centers. `None` with fewer than two cameras or no baseline.
fn align(cameras: &[EstimatedCamera], truth: &[Pose]) -> Option<Alignment> {
    if cameras.len() < 2 {
        return None;
    }
    let sum = cameras
        .iter()
        .map(|camera| truth[camera.frame].rotation.transpose() * camera.rotation)
        .sum::<Matrix3<f64>>();
    let svd = sum.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let mut correction = Matrix3::identity();
    if (u * v_t).determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    let rotation = u * correction * v_t;

    let count = cameras.len() as f64;
    let estimated: Vec<Vector3<f64>> = cameras
        .iter()
        .map(|camera| rotation * camera.center())
        .collect();
    let truth: Vec<Vector3<f64>> = cameras
        .iter()
        .map(|camera| truth[camera.frame].center)
        .collect();
    let estimated_mean = estimated.iter().sum::<Vector3<f64>>() / count;
    let truth_mean = truth.iter().sum::<Vector3<f64>>() / count;
    let (mut dot, mut variance) = (0.0, 0.0);
    for (estimated, truth) in estimated.iter().zip(&truth) {
        dot += (estimated - estimated_mean).dot(&(truth - truth_mean));
        variance += (estimated - estimated_mean).norm_squared();
    }
    let scale = dot / variance;
    (variance > 1e-12 && scale.is_finite() && scale > 0.0).then(|| Alignment {
        rotation,
        scale,
        translation: truth_mean - scale * estimated_mean,
    })
}

/// The global alignment plus one local similarity per dense reference. The local one
/// maps a reference's points through its own estimated camera onto its true camera,
/// with the scale that matches its depths to the truth, so dense, mesh and texture are
/// judged without the registration's drift (the registration stage owns that).
struct Aligned<'a> {
    global: Alignment,
    cameras: &'a [EstimatedCamera],
    /// `(globally scaled estimated depth, true depth)` of every observed dense point,
    /// per reference frame, in metres.
    depths: BTreeMap<usize, Vec<(f64, f64)>>,
    /// Median `estimated / true` depth per reference: its scale bias under `global`.
    ratios: BTreeMap<usize, f64>,
    local: BTreeMap<usize, Alignment>,
    /// Observed reference frame of every dense point.
    references: Vec<Option<usize>>,
}

impl<'a> Aligned<'a> {
    fn new(
        inputs: &StageInputs<'_>,
        evidence: &ReconstructionEvidenceView<'_>,
        cameras: &'a [EstimatedCamera],
        global: Alignment,
    ) -> Self {
        let fixture = inputs.fixture;
        let width = fixture.intrinsics.width;
        let sites = &inputs.result.reconstruction.dense_grid_sites;
        let references = observed_reference_of_point(evidence);
        let camera = |frame: usize| cameras.iter().find(|camera| camera.frame == frame);
        let mut depths: BTreeMap<usize, Vec<(f64, f64)>> = BTreeMap::new();
        if sites.len() == evidence.points.len() {
            for (index, reference) in references.iter().enumerate() {
                let Some((frame, camera)) =
                    reference.and_then(|frame| Some((frame, camera(frame)?)))
                else {
                    continue;
                };
                let depth = global.scale
                    * (camera.rotation * point(evidence, index) + camera.translation).z;
                let site = sites[index];
                let truth = fixture.views[frame].depth[(site.y * width + site.x) as usize] as f64;
                depths.entry(frame).or_default().push((depth, truth));
            }
        }
        let ratios: BTreeMap<usize, f64> = depths
            .iter()
            .filter_map(|(frame, depths)| {
                let ratio = median(depths.iter().map(|(depth, truth)| depth / truth).collect())?;
                (ratio.is_finite() && ratio > 0.0).then_some((*frame, ratio))
            })
            .collect();
        let local = ratios
            .iter()
            .filter_map(|(frame, ratio)| {
                let camera = camera(*frame)?;
                let pose = &fixture.views[*frame].pose;
                let scale = global.scale / ratio;
                Some((
                    *frame,
                    Alignment {
                        rotation: pose.rotation.transpose() * camera.rotation,
                        scale,
                        translation: pose.rotation.transpose() * (scale * camera.translation)
                            + pose.center,
                    },
                ))
            })
            .collect();
        Self {
            global,
            cameras,
            depths,
            ratios,
            local,
            references,
        }
    }

    /// Point `index` in truth world coordinates, through its reference's local alignment.
    fn point(&self, evidence: &ReconstructionEvidenceView<'_>, index: usize) -> Vector3<f64> {
        let alignment = self.references[index]
            .and_then(|frame| self.local.get(&frame))
            .unwrap_or(&self.global);
        alignment.apply(&point(evidence, index))
    }
}

fn point(evidence: &ReconstructionEvidenceView<'_>, index: usize) -> Vector3<f64> {
    let point = &evidence.points[index];
    Vector3::new(point.x as f64, point.y as f64, point.z as f64)
}

/// Reference frame of every point that observed multi-view evidence produced.
fn observed_reference_of_point(evidence: &ReconstructionEvidenceView<'_>) -> Vec<Option<usize>> {
    let mut reference = vec![None; evidence.points.len()];
    for region in &evidence.regions {
        let observed = matches!(
            region.origin,
            EvidenceOrigin::GeometricMultiView | EvidenceOrigin::RevalidatedCompletion
        );
        if let (true, Some(frame)) = (observed, region.reference_frame) {
            let end = (region.points.start + region.points.count).min(reference.len());
            for slot in &mut reference[region.points.start.min(end)..end] {
                *slot = Some(frame);
            }
        }
    }
    reference
}

// ---------------------------------------------------------------------------------------
// Stages

/// Feature detection and matching on every adjacent pair, with the authoritative
/// whole-pixel matcher and the browser's options, scored against true correspondences.
fn features_stage(inputs: &StageInputs<'_>) -> Value {
    let fixture = inputs.fixture;
    let reconstruction = &inputs.result.reconstruction;
    let options = browser_options();
    let feature_options = FeatureOptions {
        max_features: options.max_features,
        min_feature_distance: options.min_feature_distance,
        descriptor_radius: options.descriptor_radius,
        match_radius: options.match_radius,
        baseline_max_distance: options.max_descriptor_distance,
        ratio_threshold: options.ratio_threshold,
        ..FeatureOptions::default()
    };
    let (width, height) = (fixture.intrinsics.width, fixture.intrinsics.height);
    let mut pairs = Vec::new();
    let mut scores = Vec::new();
    for (index, pair) in fixture.views.windows(2).enumerate() {
        let (from, to) = (&pair[0], &pair[1]);
        let analysis = analyze_rgba_pair(
            &from.frame.rgba,
            &to.frame.rgba,
            width,
            height,
            FeatureAlgorithm::BaselineHarrisPatch,
            feature_options,
        )
        .expect("feature analysis of a reference pair");
        let mut errors = Vec::new();
        let mut correct = 0;
        for feature_match in &analysis.matches {
            let source = analysis.source_features[feature_match.source_index];
            let target = analysis.target_features[feature_match.target_index];
            let world = true_world_point(fixture, from, source.x as u32, source.y as u32);
            let Some((u, v)) = visible_projection(fixture, to, &world) else {
                continue;
            };
            let error =
                ((target.x as f64 + 0.5 - u).powi(2) + (target.y as f64 + 0.5 - v).powi(2)).sqrt();
            correct += usize::from(error <= MATCH_TOLERANCE_PIXELS);
            errors.push(error);
        }
        let score = PairScore {
            matches: analysis.matches.len(),
            correct,
            errors,
        };
        let pipeline = reconstruction
            .pairs
            .iter()
            .find(|stats| stats.from_frame == index && stats.to_frame == index + 1);
        let candidate = reconstruction
            .seed_candidates
            .iter()
            .find(|candidate| candidate.from_frame == index && candidate.to_frame == index + 1);
        pairs.push(json!({
            "from_frame": index,
            "to_frame": index + 1,
            "source_features": analysis.source_features.len(),
            "target_features": analysis.target_features.len(),
            "matches": score.matches,
            "correct_matches": score.correct,
            "precision": round6(score.precision()),
            "median_error_pixels": rounded(median(score.errors.clone())),
            "pipeline_matches": pipeline.map(|stats| stats.matches),
            "essential_inlier_ratio": candidate.and_then(|candidate| candidate.inlier_ratio),
        }));
        scores.push(score);
    }
    json!({
        "available": true,
        "matcher": "authoritative whole-pixel Harris-patch matcher with the browser's options",
        "metrics": pair_metrics(&scores),
        "pairs": pairs,
    })
}

/// One adjacent pair's matches scored against the true correspondences.
struct PairScore {
    matches: usize,
    /// Matches within `MATCH_TOLERANCE_PIXELS` of the true correspondence.
    correct: usize,
    /// Error of every match whose source point is visible in the target.
    errors: Vec<f64>,
}

impl PairScore {
    fn precision(&self) -> f64 {
        if self.matches == 0 {
            0.0
        } else {
            self.correct as f64 / self.matches as f64
        }
    }
}

/// Worst-pair aggregates, so one bad pair cannot hide behind accurate neighbours. A pair
/// without a scorable match has no median error and makes the worst-pair metric missing.
fn pair_metrics(scores: &[PairScore]) -> Value {
    let worst_median = scores
        .iter()
        .map(|score| median(score.errors.clone()).unwrap_or(f64::INFINITY))
        .reduce(f64::max);
    json!({
        "pairs": scores.len(),
        "min_pair_matches": scores.iter().map(|score| score.matches).min().unwrap_or(0),
        "min_match_precision": round6(scores.iter().map(PairScore::precision).reduce(f64::min).unwrap_or(0.0)),
        "max_pair_median_error_pixels": rounded(worst_median.filter(|value| value.is_finite())),
        "median_match_error_pixels": rounded(median(scores.iter().flat_map(|score| score.errors.iter().copied()).collect())),
    })
}

/// The chosen seed pair against the true relative pose, plus every candidate's evidence
/// next to its true baseline and relief parallax.
fn seed_stage(inputs: &StageInputs<'_>) -> Value {
    let fixture = inputs.fixture;
    let views = &fixture.views;
    let reconstruction = &inputs.result.reconstruction;
    let candidates: Vec<Value> = reconstruction
        .seed_candidates
        .iter()
        .map(|candidate| {
            let (from, to) = (&views[candidate.from_frame], &views[candidate.to_frame]);
            json!({
                "from_frame": candidate.from_frame,
                "to_frame": candidate.to_frame,
                "rejected_gate": candidate.rejected_gate,
                "selected": candidate.selected,
                "matches": candidate.matches,
                "inlier_ratio": candidate.inlier_ratio,
                "rotation_only_residual_pixels": candidate.rotation_only_residual_pixels,
                "median_parallax_residual_pixels": candidate.median_parallax_residual,
                "median_triangulation_angle_degrees": candidate.median_triangulation_angle_degrees,
                "true_baseline_m": round6((to.pose.center - from.pose.center).norm()),
                "true_relief_parallax_pixels": round6(relief_parallax_pixels(&fixture.intrinsics, from, &to.pose)),
            })
        })
        .collect();
    let mut metrics = json!({
        "seed_selected": reconstruction.calibrated_pair.is_some(),
        "decisive_gate": inputs.decisive_gate,
        "rotation_error_degrees": null,
        "translation_direction_error_degrees": null,
    });
    if let Some(pair) = &reconstruction.calibrated_pair {
        let (from, to) = (&views[pair.from_frame], &views[pair.to_frame]);
        let true_rotation = to.pose.rotation * from.pose.rotation.transpose();
        let true_translation = to.pose.rotation * (from.pose.center - to.pose.center);
        let rotation_error =
            rotation_angle_degrees(&(matrix(&pair.relative_rotation).transpose() * true_rotation));
        let direction = vector(&pair.translation_direction);
        let translation_error = (direction.normalize().dot(&true_translation.normalize()))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        let selected = reconstruction.seed_candidates.iter().find(|candidate| {
            candidate.from_frame == pair.from_frame && candidate.to_frame == pair.to_frame
        });
        metrics = json!({
            "seed_selected": true,
            "decisive_gate": inputs.decisive_gate,
            "from_frame": pair.from_frame,
            "to_frame": pair.to_frame,
            "rotation_error_degrees": round6(rotation_error),
            "translation_direction_error_degrees": round6(translation_error),
            "true_baseline_m": round6((to.pose.center - from.pose.center).norm()),
            "true_median_triangulation_angle_degrees": rounded(true_triangulation_angle(fixture, from, to)),
            "median_triangulation_angle_degrees": round6(pair.median_triangulation_angle_degrees as f64),
            "rotation_only_residual_pixels": selected.and_then(|candidate| candidate.rotation_only_residual_pixels),
            "inlier_ratio": pair.inlier_ratio,
        });
    }
    json!({
        "available": true,
        "metrics": metrics,
        "candidates": candidates,
    })
}

/// Median angle between the two cameras' rays to the surface points `from` sees.
fn true_triangulation_angle(fixture: &Fixture, from: &View, to: &View) -> Option<f64> {
    let intrinsics = &fixture.intrinsics;
    let mut angles = Vec::new();
    for y in (0..intrinsics.height).step_by(8) {
        for x in (0..intrinsics.width).step_by(8) {
            let world = true_world_point(fixture, from, x, y);
            if visible_projection(fixture, to, &world).is_none() {
                continue;
            }
            let (left, right) = (world - from.pose.center, world - to.pose.center);
            angles.push(
                (left.normalize().dot(&right.normalize()))
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees(),
            );
        }
    }
    median(angles)
}

fn registration_stage(
    inputs: &StageInputs<'_>,
    cameras: &[EstimatedCamera],
    truth: &[Pose],
    alignment: Option<&Alignment>,
) -> Value {
    let errors = alignment.map(|alignment| pose_errors(cameras, truth, alignment));
    let mut frames: Vec<usize> = cameras.iter().map(|camera| camera.frame).collect();
    frames.sort_unstable();
    json!({
        "available": true,
        "metrics": {
            "registered_images": inputs.registered_images,
            "normalized_pose_rmse": rounded(errors.map(|errors| errors.normalized_center_rmse)),
            "max_rotation_error_degrees": rounded(errors.map(|errors| errors.max_rotation_degrees)),
            "center_only_pose_rmse": rounded(inputs.normalized_pose_rmse),
            "alignment_scale": rounded(alignment.map(|alignment| alignment.scale)),
        },
        "registered_frames": frames,
    })
}

#[derive(Clone, Copy, Debug)]
struct PoseErrors {
    /// Center RMSE through the pose alignment, divided by the true trajectory span.
    normalized_center_rmse: f64,
    max_rotation_degrees: f64,
}

/// Center and orientation errors of every camera through one and the same similarity, so
/// centers and orientations that no single similarity reconciles cannot both pass.
fn pose_errors(cameras: &[EstimatedCamera], truth: &[Pose], alignment: &Alignment) -> PoseErrors {
    let centers: Vec<Vector3<f64>> = truth.iter().map(|pose| pose.center).collect();
    let span = super::common::trajectory_span(&centers);
    let squared = cameras
        .iter()
        .map(|camera| {
            (alignment.apply(&camera.center()) - truth[camera.frame].center).norm_squared()
        })
        .sum::<f64>();
    let max_rotation_degrees = cameras
        .iter()
        .map(|camera| {
            rotation_angle_degrees(
                &(truth[camera.frame].rotation.transpose()
                    * camera.rotation
                    * alignment.rotation.transpose()),
            )
        })
        .fold(0.0, f64::max);
    PoseErrors {
        normalized_center_rmse: (squared / cameras.len().max(1) as f64).sqrt() / span,
        max_rotation_degrees,
    }
}

/// Depth of every observed dense point in its reference camera against the true depth
/// through the same reference-grid pixel. The global alignment's scale carries the
/// registration's drift, so it is reported as each reference's scale bias, and the
/// gated error is measured once that bias is removed.
fn dense_stage(inputs: &StageInputs<'_>, aligned: &Aligned<'_>) -> Value {
    let mut rows = Vec::new();
    let (mut worst_error, mut worst_coverage) = (0.0_f64, f64::INFINITY);
    for patch in &inputs.result.reconstruction.dense.reference_patches {
        let frame = patch.reference_frame;
        let depths = aligned.depths.get(&frame).map_or(&[][..], Vec::as_slice);
        let ratio = aligned.ratios.get(&frame).copied();
        let errors: Vec<f64> = ratio.map_or_else(Vec::new, |ratio| {
            depths
                .iter()
                .map(|(depth, truth)| (depth / ratio - truth).abs() / truth)
                .collect()
        });
        let accurate = errors
            .iter()
            .filter(|&&error| error <= DEPTH_TOLERANCE)
            .count();
        let sampled = patch.sampled_pixels.max(1) as f64;
        let points = depths.len();
        let median_error = median(errors);
        worst_error = worst_error.max(median_error.unwrap_or(f64::INFINITY));
        let accurate_coverage = accurate as f64 / sampled;
        worst_coverage = worst_coverage.min(accurate_coverage);
        rows.push(json!({
            "reference_frame": frame,
            "sampled_pixels": patch.sampled_pixels,
            "points": points,
            "accurate_points": accurate,
            "scale_bias": rounded(ratio.map(|ratio| ratio - 1.0)),
            "median_relative_depth_error": rounded(median_error),
            "coverage": round6(points as f64 / sampled),
            "accurate_coverage": round6(accurate_coverage),
        }));
    }
    let any = !rows.is_empty();
    let max_bias = aligned
        .ratios
        .values()
        .map(|ratio| (ratio - 1.0).abs())
        .reduce(f64::max);
    json!({
        "available": true,
        "metrics": {
            "references": rows.len(),
            "max_median_relative_depth_error": rounded(any.then_some(worst_error)),
            "min_accurate_coverage": round6(if any { worst_coverage } else { 0.0 }),
            "max_reference_scale_bias": rounded(max_bias),
        },
        "references": rows,
    })
}

/// Aligned mesh against the true heightfield: surface error, unsupported triangles,
/// coverage of the surface the registered cameras see, duplicated overlap, connectivity.
fn mesh_stage(
    inputs: &StageInputs<'_>,
    evidence: &ReconstructionEvidenceView<'_>,
    aligned: &Aligned<'_>,
) -> Value {
    let fixture = inputs.fixture;
    let facade = &fixture.spec.facade;
    let references = &aligned.references;
    let vertex = |index: usize| aligned.point(evidence, index);
    let surface_error = |centroid: Vector3<f64>| {
        if facade.contains(centroid.x, centroid.y) {
            (centroid.z - facade.surface_z(centroid.x, centroid.y)).abs()
        } else {
            f64::INFINITY
        }
    };
    // The same surface through the global alignment: what the registration's drift adds.
    let global_errors: Vec<f64> = evidence
        .triangles
        .iter()
        .map(|triangle| {
            surface_error(
                [triangle.a, triangle.b, triangle.c]
                    .map(|index| aligned.global.apply(&point(evidence, index)))
                    .iter()
                    .sum::<Vector3<f64>>()
                    / 3.0,
            )
        })
        .filter(|error| error.is_finite())
        .collect();

    let [x0, x1, y0, y1] = facade.extent;
    let columns = ((x1 - x0) / super::MESH_SPACING).round() as usize;
    let rows = ((y1 - y0) / super::MESH_SPACING).round() as usize;
    let cell_center = |column: usize, row: usize| {
        (
            x0 + (column as f64 + 0.5) * super::MESH_SPACING,
            y0 + (row as f64 + 0.5) * super::MESH_SPACING,
        )
    };
    // Patches (reference frames, < 64) covering each cell with supported triangles.
    let mut covering = vec![0_u64; columns * rows];
    let mut errors = Vec::with_capacity(evidence.triangles.len());
    let mut unsupported = 0;
    for triangle in evidence.triangles {
        let corners = [triangle.a, triangle.b, triangle.c].map(vertex);
        let error = surface_error((corners[0] + corners[1] + corners[2]) / 3.0);
        errors.push(error);
        if error > SURFACE_TOLERANCE_M {
            unsupported += 1;
            continue;
        }
        let patch = references[triangle.a].map_or(63, |frame| frame.min(63));
        let column_range = |values: [f64; 3]| {
            let low = values.iter().copied().fold(f64::INFINITY, f64::min);
            let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (low, high)
        };
        let (min_x, max_x) = column_range(corners.map(|corner| corner.x));
        let (min_y, max_y) = column_range(corners.map(|corner| corner.y));
        let first_column = (((min_x - x0) / super::MESH_SPACING - 0.5).ceil().max(0.0)) as usize;
        let last_column = ((((max_x - x0) / super::MESH_SPACING - 0.5).floor()) as isize)
            .min(columns as isize - 1);
        let first_row = (((min_y - y0) / super::MESH_SPACING - 0.5).ceil().max(0.0)) as usize;
        let last_row =
            ((((max_y - y0) / super::MESH_SPACING - 0.5).floor()) as isize).min(rows as isize - 1);
        for row in first_row as isize..=last_row {
            for column in first_column as isize..=last_column {
                let (x, y) = cell_center(column as usize, row as usize);
                if inside_triangle_xy(&corners, x, y) {
                    covering[row as usize * columns + column as usize] |= 1 << patch;
                }
            }
        }
    }

    // Cells the registered cameras see, unoccluded, from at least two views.
    let registered: Vec<&View> = aligned
        .cameras
        .iter()
        .map(|camera| &fixture.views[camera.frame])
        .collect();
    let (mut observed, mut covered, mut duplicated) = (0_usize, 0_usize, 0_usize);
    for row in 0..rows {
        for column in 0..columns {
            let (x, y) = cell_center(column, row);
            let world = Vector3::new(x, y, facade.surface_z(x, y));
            let seen = registered
                .iter()
                .filter(|view| visible_projection(fixture, view, &world).is_some())
                .count();
            if seen < 2 {
                continue;
            }
            observed += 1;
            let patches = covering[row * columns + column];
            covered += usize::from(patches != 0);
            duplicated += usize::from(patches.count_ones() > 1);
        }
    }
    let topology = inputs.topology;
    let triangles = evidence.triangles.len();
    let finite: Vec<f64> = errors.iter().copied().filter(|e| e.is_finite()).collect();
    json!({
        "available": true,
        "metrics": {
            "triangles": triangles,
            "median_surface_error_m": rounded(median(finite)),
            "median_surface_error_global_alignment_m": rounded(median(global_errors)),
            "unsupported_triangle_share": round6(if triangles == 0 { 0.0 } else { unsupported as f64 / triangles as f64 }),
            "observed_surface_cells": observed,
            "surface_coverage": round6(if observed == 0 { 0.0 } else { covered as f64 / observed as f64 }),
            "duplicated_coverage_share": round6(if covered == 0 { 0.0 } else { duplicated as f64 / covered as f64 }),
            "mesh_components": topology.components,
            "true_components": 1,
            "per_patch_components": topology.per_patch_components,
            "largest_component_share": round6(topology.largest_component_share),
            "meshed_reference_patches": topology.meshed_reference_patches,
            "cross_reference_triangles": topology.cross_reference_triangles,
            "shared_vertices": topology.shared_vertices,
            "reference_patch_components": topology.reference_patch_components,
        },
        "cell_size_m": super::MESH_SPACING,
    })
}

fn inside_triangle_xy(corners: &[Vector3<f64>; 3], x: f64, y: f64) -> bool {
    let edge =
        |a: &Vector3<f64>, b: &Vector3<f64>| (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
    let signs = [
        edge(&corners[0], &corners[1]),
        edge(&corners[1], &corners[2]),
        edge(&corners[2], &corners[0]),
    ];
    signs.iter().all(|&sign| sign >= 0.0) || signs.iter().all(|&sign| sign <= 0.0)
}

/// Bake the reference-view materials and place sampled texels on the aligned surface:
/// where the true reference camera sees that point (pixels) and what color the true
/// surface has there.
fn texture_stage(
    inputs: &StageInputs<'_>,
    evidence: &ReconstructionEvidenceView<'_>,
    aligned: &Aligned<'_>,
) -> Value {
    let fixture = inputs.fixture;
    let facade = &fixture.spec.facade;
    let images: Vec<ReferenceImage<'_>> = fixture
        .views
        .iter()
        .enumerate()
        .map(|(frame_index, view)| ReferenceImage {
            frame_index,
            width: view.frame.width,
            height: view.frame.height,
            rgba: &view.frame.rgba,
        })
        .collect();
    let bake = match bake_surface_materials(
        evidence,
        &inputs.result.reconstruction.dense_grid_sites,
        &images,
    ) {
        Ok(bake) => bake,
        Err(error) => {
            return json!({ "available": false, "blocked_by": format!("texture bake failed: {error}") })
        }
    };
    let vertex = |index: usize| aligned.point(evidence, index);
    let (mut reprojection, mut color) = (Vec::new(), Vec::new());
    let mut textured = 0;
    for material in &bake.materials {
        let view = &fixture.views[material.reference_frame];
        let texture = &material.texture;
        let [crop_x, crop_y] = texture.crop_origin;
        for (triangle_index, uvs) in material.triangles.iter().zip(&material.corner_uvs) {
            textured += 1;
            let triangle = &evidence.triangles[*triangle_index];
            let corners = [triangle.a, triangle.b, triangle.c].map(vertex);
            for weights in TEXEL_SAMPLES {
                let point =
                    corners[0] * weights[0] + corners[1] * weights[1] + corners[2] * weights[2];
                let u = (0..3).map(|k| weights[k] * uvs[k][0] as f64).sum::<f64>();
                let v = (0..3).map(|k| weights[k] * uvs[k][1] as f64).sum::<f64>();
                let texel_x = u * texture.width as f64;
                let texel_y = v * texture.height as f64;
                if let Some((true_u, true_v)) = fixture
                    .intrinsics
                    .project(&view.pose.world_to_camera(&point))
                {
                    reprojection.push(
                        ((crop_x as f64 + texel_x - true_u).powi(2)
                            + (crop_y as f64 + texel_y - true_v).powi(2))
                        .sqrt(),
                    );
                }
                if facade.contains(point.x, point.y) {
                    let column = (texel_x as u32).min(texture.width - 1);
                    let row = (texel_y as u32).min(texture.height - 1);
                    let offset = ((row * texture.width + column) * 4) as usize;
                    let truth = facade.radiance(point.x, point.y);
                    let difference = (0..3)
                        .map(|channel| {
                            (texture.rgba[offset + channel] as f64 - truth[channel] * 255.0).abs()
                        })
                        .sum::<f64>()
                        / 3.0;
                    color.push(difference);
                }
            }
        }
    }
    let triangles = evidence.triangles.len();
    json!({
        "available": true,
        "metrics": {
            "materials": bake.materials.len(),
            "textured_triangle_share": round6(if triangles == 0 { 0.0 } else { textured as f64 / triangles as f64 }),
            "median_texel_reprojection_error_pixels": rounded(median(reprojection.clone())),
            "p90_texel_reprojection_error_pixels": rounded(quantile(reprojection, 0.9)),
            "median_texel_color_error": rounded(median(color)),
        },
        "color_truth": "shaded albedo (frame radiance) at the aligned texel's surface point, 8-bit levels",
        "fallback": bake.fallback.reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    fn pose(yaw: f64, center: Vector3<f64>) -> Pose {
        Pose {
            rotation: Rotation3::from_axis_angle(&Vector3::y_axis(), yaw)
                .matrix()
                .transpose(),
            center,
        }
    }

    #[test]
    fn median_averages_the_two_middle_values_of_an_even_count() {
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(Vec::new()), None);
        assert_eq!(quantile(vec![1.0, 2.0, 3.0, 4.0, 5.0], 0.9), Some(5.0));
    }

    #[test]
    fn alignment_recovers_a_similarity_from_collinear_lateral_cameras() {
        // A lateral pan: identity orientation, centers on one line. Center-only
        // alignment cannot fix the rotation about that line; orientations do.
        let truth: Vec<Pose> = (0..5)
            .map(|index| pose(0.0, Vector3::new(0.2 * index as f64, 0.0, 0.0)))
            .collect();
        let world_rotation = *Rotation3::from_euler_angles(0.3, -0.2, 0.7).matrix();
        let (scale, offset) = (0.25, Vector3::new(1.0, -2.0, 0.5));
        // estimated = world_rotationᵀ · (truth - offset) / scale
        let cameras: Vec<EstimatedCamera> = truth
            .iter()
            .enumerate()
            .map(|(frame, pose)| {
                let center = world_rotation.transpose() * (pose.center - offset) / scale;
                let rotation = pose.rotation * world_rotation;
                EstimatedCamera {
                    frame,
                    rotation,
                    translation: -(rotation * center),
                }
            })
            .collect();
        let alignment = align(&cameras, &truth).expect("alignment");
        assert!((alignment.scale - scale).abs() < 1e-9);
        assert!(rotation_angle_degrees(&(alignment.rotation.transpose() * world_rotation)) < 1e-6);
        for (camera, pose) in cameras.iter().zip(&truth) {
            assert!((alignment.apply(&camera.center()) - pose.center).norm() < 1e-9);
        }
    }

    #[test]
    fn alignment_needs_two_cameras_with_a_baseline() {
        let truth = vec![pose(0.0, Vector3::zeros()), pose(0.1, Vector3::zeros())];
        let cameras: Vec<EstimatedCamera> = truth
            .iter()
            .enumerate()
            .map(|(frame, pose)| EstimatedCamera {
                frame,
                rotation: pose.rotation,
                translation: Vector3::zeros(),
            })
            .collect();
        assert!(align(&cameras, &truth).is_none());
        assert!(align(&cameras[..1], &truth).is_none());
    }

    #[test]
    fn pose_errors_use_one_alignment_for_centers_and_orientations() {
        let truth: Vec<Pose> = (0..5)
            .map(|index| pose(0.0, Vector3::new(0.2 * index as f64, 0.0, 0.0)))
            .collect();
        let cameras = |center_frame: Matrix3<f64>, orientation_frame: Matrix3<f64>| {
            truth
                .iter()
                .enumerate()
                .map(|(frame, pose)| {
                    let center = center_frame.transpose() * pose.center * 3.0;
                    let rotation = pose.rotation * orientation_frame;
                    EstimatedCamera {
                        frame,
                        rotation,
                        translation: -(rotation * center),
                    }
                })
                .collect::<Vec<_>>()
        };
        let frame = *Rotation3::from_euler_angles(0.1, 0.4, -0.3).matrix();
        let consistent = cameras(frame, frame);
        let alignment = align(&consistent, &truth).expect("alignment");
        let errors = pose_errors(&consistent, &truth, &alignment);
        assert!(errors.normalized_center_rmse < 1e-9);
        assert!(errors.max_rotation_degrees < 1e-6);

        // Orientations agree with each other but not with the centers: the camera path
        // is turned 40° away from where the cameras look. A center-only fit would still
        // place every center exactly; the shared alignment must not.
        let turned = *Rotation3::from_euler_angles(0.1, 0.4 + 40f64.to_radians(), -0.3).matrix();
        let inconsistent = cameras(turned, frame);
        let alignment = align(&inconsistent, &truth).expect("alignment");
        let errors = pose_errors(&inconsistent, &truth, &alignment);
        assert!(errors.max_rotation_degrees < 1e-6);
        assert!(errors.normalized_center_rmse > 0.1, "{errors:?}");
    }

    #[test]
    fn visibility_respects_the_true_depth() {
        let (width, height) = (8, 6);
        let intrinsics = super::super::Intrinsics {
            width,
            height,
            focal: 10.0,
            cx: 4.0,
            cy: 3.0,
        };
        let view = View {
            pose: pose(0.0, Vector3::zeros()),
            frame: video_to_3d_core::FrameInput {
                width,
                height,
                rgba: vec![0; (width * height * 4) as usize],
            },
            depth: vec![2.0; (width * height) as usize],
        };
        let fixture = Fixture {
            case: super::super::Case::OverlappingReferences,
            spec: super::super::Case::OverlappingReferences.spec(),
            plan: super::super::sampling_plan(),
            intrinsics,
            views: Vec::new(),
        };
        let on_surface = Vector3::new(0.1, 0.0, 2.0);
        assert!(visible_projection(&fixture, &view, &on_surface).is_some());
        // Behind the surface the view sees there: occluded, not observed.
        assert!(visible_projection(&fixture, &view, &Vector3::new(0.15, 0.0, 3.0)).is_none());
        // Outside the image or behind the camera.
        assert!(visible_projection(&fixture, &view, &Vector3::new(5.0, 0.0, 2.0)).is_none());
        assert!(visible_projection(&fixture, &view, &Vector3::new(0.0, 0.0, -2.0)).is_none());
    }

    #[test]
    fn one_biased_pair_fails_the_worst_pair_error() {
        let accurate = || PairScore {
            matches: 100,
            correct: 100,
            errors: vec![0.2; 100],
        };
        // Every match within the 1.5 px precision tolerance, but biased by 1.2 px.
        let biased = PairScore {
            matches: 100,
            correct: 100,
            errors: vec![1.2; 100],
        };
        let metrics = pair_metrics(&[accurate(), accurate(), biased, accurate()]);
        assert_eq!(metrics["min_match_precision"], json!(1.0));
        assert_eq!(metrics["median_match_error_pixels"], json!(0.2));
        assert_eq!(metrics["max_pair_median_error_pixels"], json!(1.2));
        // A pair without a scorable match leaves the worst-pair error missing.
        let empty = PairScore {
            matches: 0,
            correct: 0,
            errors: Vec::new(),
        };
        let metrics = pair_metrics(&[accurate(), empty]);
        assert_eq!(metrics["max_pair_median_error_pixels"], Value::Null);
        assert_eq!(metrics["min_match_precision"], json!(0.0));
    }

    #[test]
    fn triangle_containment_ignores_winding() {
        let corners = [
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 1.0),
        ];
        let reversed = [corners[0], corners[2], corners[1]];
        for triangle in [corners, reversed] {
            assert!(inside_triangle_xy(&triangle, 0.25, 0.25));
            assert!(!inside_triangle_xy(&triangle, 0.75, 0.75));
        }
    }
}
