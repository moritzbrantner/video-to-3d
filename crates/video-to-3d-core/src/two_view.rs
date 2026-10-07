use super::{Feature, FeatureMatch};
use nalgebra::{DMatrix, Matrix3, Matrix4, Vector3};
use std::cmp::Ordering;

const RANSAC_SAMPLE_SIZE: usize = 8;
const RANSAC_TARGET_CONFIDENCE: f64 = 0.99;
const SAMPSON_THRESHOLD_PIXELS: f64 = 1.75;
const MIN_INLIER_RATIO: f64 = 0.5;
const MAX_ROTATION_ONLY_RESIDUAL_PIXELS: f64 = 1.25;
const MIN_TRIANGULATION_ANGLE_DEGREES: f64 = 0.5;
const MAX_REPROJECTION_ERROR_PIXELS: f64 = 4.0;

#[derive(Clone, Debug)]
pub(super) struct TriangulatedPoint {
    pub source_feature_index: usize,
    pub descriptor_distance: f32,
    pub position: Vector3<f64>,
    pub reprojection_error_pixels: f64,
    pub triangulation_angle_degrees: f64,
}

#[derive(Clone, Debug)]
pub(super) struct TwoViewEstimate {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub camera_center: Vector3<f64>,
    pub matches: usize,
    pub inliers: usize,
    pub median_sampson_error_pixels: f64,
    pub median_reprojection_error_pixels: f64,
    pub median_triangulation_angle_degrees: f64,
    pub points: Vec<TriangulatedPoint>,
}

impl TwoViewEstimate {
    pub(super) fn is_better_than(&self, other: &Self) -> bool {
        self.inliers > other.inliers
            || (self.inliers == other.inliers
                && self
                    .median_reprojection_error_pixels
                    .total_cmp(&other.median_reprojection_error_pixels)
                    == Ordering::Less)
    }
}

/// The calibrated two-view gate that rejected a seed-pair candidate, in pipeline order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TwoViewRejection {
    /// Fewer matches than the minimal eight-point sample.
    Matches,
    /// The essential-matrix RANSAC consensus is below the inlier count or ratio gate.
    Inliers,
    /// A pure rotation explains the inliers: no measurable translation baseline.
    Baseline,
    /// The best pose leaves too few inliers triangulated in front of both cameras.
    Cheirality,
    /// The best pose triangulates inliers in front of both cameras, but too few of them
    /// reproject within the pixel gate.
    Reprojection,
    /// Enough points survive, but their median triangulation angle is too small.
    TriangulationAngle,
}

/// Measured evidence of one calibrated two-view attempt, recorded whether or not it passed.
#[derive(Clone, Debug, Default)]
pub(super) struct TwoViewEvidence {
    pub matches: usize,
    pub inliers: usize,
    pub rotation_only_residual_pixels: Option<f64>,
    /// The pose hypothesis with the most surviving points, when one was evaluated.
    pub pose: Option<PoseEvidence>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PoseEvidence {
    /// Inliers that must triangulate in front of both cameras within the reprojection gate.
    pub required_points: usize,
    pub accepted_points: usize,
    pub behind_camera: usize,
    pub high_reprojection: usize,
    pub median_reprojection_error_pixels: Option<f64>,
    pub median_triangulation_angle_degrees: Option<f64>,
}

#[derive(Clone, Debug)]
pub(super) struct TwoViewOutcome {
    pub evidence: TwoViewEvidence,
    pub result: Result<TwoViewEstimate, TwoViewRejection>,
}

pub(super) const MIN_SEED_INLIERS: usize = RANSAC_SAMPLE_SIZE;
pub(super) const MIN_SEED_INLIER_RATIO: f64 = MIN_INLIER_RATIO;
pub(super) const MIN_SEED_ROTATION_RESIDUAL_PIXELS: f64 = MAX_ROTATION_ONLY_RESIDUAL_PIXELS;
pub(super) const MIN_SEED_TRIANGULATION_ANGLE_DEGREES: f64 = MIN_TRIANGULATION_ANGLE_DEGREES;
pub(super) const MAX_SEED_REPROJECTION_ERROR_PIXELS: f64 = MAX_REPROJECTION_ERROR_PIXELS;

#[derive(Clone, Debug)]
struct Correspondence {
    source_feature_index: usize,
    descriptor_distance: f32,
    x1: Vector3<f64>,
    x2: Vector3<f64>,
}

#[derive(Clone, Debug)]
struct PoseCandidate {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    camera_center: Vector3<f64>,
    points: Vec<TriangulatedPoint>,
    median_reprojection_error_pixels: f64,
    median_triangulation_angle_degrees: f64,
    behind_camera: usize,
    high_reprojection: usize,
}

impl PoseCandidate {
    fn evidence(&self, required_points: usize) -> PoseEvidence {
        let finite = |value: f64| value.is_finite().then_some(value);
        PoseEvidence {
            required_points,
            accepted_points: self.points.len(),
            behind_camera: self.behind_camera,
            high_reprojection: self.high_reprojection,
            median_reprojection_error_pixels: finite(self.median_reprojection_error_pixels),
            median_triangulation_angle_degrees: finite(self.median_triangulation_angle_degrees),
        }
    }

    fn outranks(&self, other: &Self) -> bool {
        self.points.len() > other.points.len()
            || (self.points.len() == other.points.len()
                && self.median_reprojection_error_pixels < other.median_reprojection_error_pixels)
    }

    /// The candidate's ranking and evidence, without its triangulated points.
    fn summary(&self, required_points: usize) -> PoseSummary {
        PoseSummary {
            accepted_points: self.points.len(),
            median_reprojection_error_pixels: self.median_reprojection_error_pixels,
            evidence: self.evidence(required_points),
            rejection: pose_rejection(self, required_points),
        }
    }
}

/// What a pose hypothesis contributes as evidence once it is not selected.
struct PoseSummary {
    accepted_points: usize,
    median_reprojection_error_pixels: f64,
    evidence: PoseEvidence,
    rejection: Option<TwoViewRejection>,
}

impl PoseSummary {
    fn outranks(&self, other: &Self) -> bool {
        self.accepted_points > other.accepted_points
            || (self.accepted_points == other.accepted_points
                && self.median_reprojection_error_pixels < other.median_reprojection_error_pixels)
    }
}

/// The pose gates of [`recover_pose`]: enough points in front of both cameras within the
/// reprojection gate, at a sufficient median triangulation angle.
fn pose_rejection(candidate: &PoseCandidate, required_points: usize) -> Option<TwoViewRejection> {
    pose_rejection_from_counts(
        candidate.points.len(),
        candidate.behind_camera,
        candidate.high_reprojection,
        candidate.median_triangulation_angle_degrees,
        required_points,
    )
}

/// Too few accepted points blame whichever loss was larger: points behind a camera
/// (cheirality, also on a tie) or points over the reprojection gate.
fn pose_rejection_from_counts(
    accepted_points: usize,
    behind_camera: usize,
    high_reprojection: usize,
    median_triangulation_angle_degrees: f64,
    required_points: usize,
) -> Option<TwoViewRejection> {
    if accepted_points < required_points {
        return Some(if behind_camera >= high_reprojection {
            TwoViewRejection::Cheirality
        } else {
            TwoViewRejection::Reprojection
        });
    }
    (median_triangulation_angle_degrees < MIN_TRIANGULATION_ANGLE_DEGREES)
        .then_some(TwoViewRejection::TriangulationAngle)
}

#[cfg(test)]
pub(super) fn estimate_two_view(
    source_features: &[Feature],
    target_features: &[Feature],
    matches: &[FeatureMatch],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> Option<TwoViewEstimate> {
    evaluate_two_view(
        source_features,
        target_features,
        matches,
        width,
        height,
        focal_pixels,
    )
    .result
    .ok()
}

/// Run every calibrated two-view gate and keep the measured evidence of the attempt, so
/// a rejected seed pair reports which gate failed and by how much.
pub(super) fn evaluate_two_view(
    source_features: &[Feature],
    target_features: &[Feature],
    matches: &[FeatureMatch],
    width: u32,
    height: u32,
    focal_pixels: f64,
) -> TwoViewOutcome {
    let mut evidence = TwoViewEvidence {
        matches: matches.len(),
        ..TwoViewEvidence::default()
    };
    let reject = |evidence: TwoViewEvidence, gate| TwoViewOutcome {
        evidence,
        result: Err(gate),
    };
    if matches.len() < RANSAC_SAMPLE_SIZE || focal_pixels <= 0.0 {
        return reject(evidence, TwoViewRejection::Matches);
    }

    let center_x = width as f64 * 0.5;
    let center_y = height as f64 * 0.5;
    let correspondences: Vec<Correspondence> = matches
        .iter()
        .map(|feature_match| {
            let source = &source_features[feature_match.a];
            let target = &target_features[feature_match.b];
            Correspondence {
                source_feature_index: feature_match.a,
                descriptor_distance: feature_match.distance,
                x1: Vector3::new(
                    (source.x as f64 - center_x) / focal_pixels,
                    (source.y as f64 - center_y) / focal_pixels,
                    1.0,
                ),
                x2: Vector3::new(
                    (target.x as f64 - center_x) / focal_pixels,
                    (target.y as f64 - center_y) / focal_pixels,
                    1.0,
                ),
            }
        })
        .collect();

    let threshold = (SAMPSON_THRESHOLD_PIXELS / focal_pixels).powi(2);
    let mut best_inliers = Vec::new();
    let mut best_median_error = f64::INFINITY;
    let mut best_essential = None;

    let mut iteration = 0;
    let mut iteration_limit = required_ransac_iterations(MIN_INLIER_RATIO);
    while iteration < iteration_limit {
        let sample = deterministic_sample(correspondences.len(), iteration);
        iteration += 1;
        let Some(essential) = estimate_essential(&correspondences, &sample) else {
            continue;
        };
        let mut inliers = Vec::new();
        let mut errors = Vec::new();
        for (index, correspondence) in correspondences.iter().enumerate() {
            let error = sampson_error(&essential, correspondence);
            if error.is_finite() && error <= threshold {
                inliers.push(index);
                errors.push(error);
            }
        }

        let median_error = median_f64(&mut errors);
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && median_error < best_median_error)
        {
            best_inliers = inliers;
            best_median_error = median_error;
            best_essential = Some(essential);

            let observed_inlier_ratio = best_inliers.len() as f64 / correspondences.len() as f64;
            if observed_inlier_ratio >= MIN_INLIER_RATIO {
                iteration_limit = iteration_limit
                    .min(required_ransac_iterations(observed_inlier_ratio).max(iteration));
            }
        }
    }

    evidence.inliers = best_inliers.len();
    let inlier_ratio = best_inliers.len() as f64 / correspondences.len() as f64;
    if best_inliers.len() < RANSAC_SAMPLE_SIZE || inlier_ratio < MIN_INLIER_RATIO {
        return reject(evidence, TwoViewRejection::Inliers);
    }

    let rotation_residual =
        rotation_only_residual_pixels(&correspondences, &best_inliers, focal_pixels)
            .filter(|residual| residual.is_finite());
    evidence.rotation_only_residual_pixels = rotation_residual;
    if rotation_residual.is_none_or(|residual| residual <= MAX_ROTATION_ONLY_RESIDUAL_PIXELS) {
        return reject(evidence, TwoViewRejection::Baseline);
    }

    // Keep the robust winning RANSAC hypothesis available. A consensus refit can
    // improve the geometry, but rounded pixel correspondences can also make the
    // constrained refit worse than the robust seed. Only select a refit that survives
    // the same cheirality, reprojection, and triangulation-angle gates.
    let Some(ransac_essential) = best_essential else {
        return reject(evidence, TwoViewRejection::Inliers);
    };
    let required_points = (best_inliers.len() * 3 / 5).max(8);
    // The strongest pose hypothesis of either model, kept as evidence when both fail.
    // Only its ranking and scalar evidence are kept, never its triangulated points.
    let mut strongest: Option<PoseSummary> = None;
    let mut gated = |candidate: Option<PoseCandidate>| {
        let candidate = candidate?;
        let summary = candidate.summary(required_points);
        let rejected = summary.rejection.is_some();
        if strongest
            .as_ref()
            .is_none_or(|current| summary.outranks(current))
        {
            strongest = Some(summary);
        }
        (!rejected).then_some(candidate)
    };
    let ransac_pose = gated(best_pose(
        &ransac_essential,
        &correspondences,
        &best_inliers,
        focal_pixels,
    ));
    let refined_model =
        estimate_essential(&correspondences, &best_inliers).and_then(|refined_essential| {
            gated(best_pose(
                &refined_essential,
                &correspondences,
                &best_inliers,
                focal_pixels,
            ))
            .map(|pose| (refined_essential, pose))
        });
    evidence.pose = strongest.as_ref().map(|summary| summary.evidence);

    let (essential, pose) = match (refined_model, ransac_pose) {
        (Some((refined_essential, refined_pose)), Some(ransac_pose)) => {
            let refined_is_better = refined_pose.points.len() > ransac_pose.points.len()
                || (refined_pose.points.len() == ransac_pose.points.len()
                    && refined_pose.median_reprojection_error_pixels
                        < ransac_pose.median_reprojection_error_pixels);
            if refined_is_better {
                (refined_essential, refined_pose)
            } else {
                (ransac_essential, ransac_pose)
            }
        }
        (Some(refined_model), None) => refined_model,
        (None, Some(ransac_pose)) => (ransac_essential, ransac_pose),
        (None, None) => {
            let gate = strongest
                .as_ref()
                .and_then(|summary| summary.rejection)
                .unwrap_or(TwoViewRejection::Cheirality);
            return reject(evidence, gate);
        }
    };
    evidence.pose = Some(pose.evidence(required_points));

    let mut sampson_errors_pixels: Vec<f64> = best_inliers
        .iter()
        .map(|&index| sampson_error(&essential, &correspondences[index]).sqrt() * focal_pixels)
        .filter(|error| error.is_finite())
        .collect();
    let median_sampson_error_pixels = median_f64(&mut sampson_errors_pixels);

    let estimate = TwoViewEstimate {
        rotation: pose.rotation,
        translation: pose.translation,
        camera_center: pose.camera_center,
        matches: matches.len(),
        inliers: best_inliers.len(),
        median_sampson_error_pixels,
        median_reprojection_error_pixels: pose.median_reprojection_error_pixels,
        median_triangulation_angle_degrees: pose.median_triangulation_angle_degrees,
        points: pose.points,
    };
    TwoViewOutcome {
        evidence,
        result: Ok(estimate),
    }
}

fn required_ransac_iterations(inlier_ratio: f64) -> usize {
    if !inlier_ratio.is_finite() || inlier_ratio <= 0.0 {
        return usize::MAX;
    }

    let inlier_ratio = inlier_ratio.min(1.0);
    let sample_success = inlier_ratio.powi(RANSAC_SAMPLE_SIZE as i32);
    if sample_success >= 1.0 {
        return 1;
    }

    let denominator = (1.0 - sample_success).ln();
    if denominator == 0.0 {
        return usize::MAX;
    }

    let iterations = ((1.0 - RANSAC_TARGET_CONFIDENCE).ln() / denominator).ceil();
    if iterations.is_finite() {
        iterations.max(1.0) as usize
    } else {
        usize::MAX
    }
}

fn deterministic_sample(len: usize, iteration: usize) -> Vec<usize> {
    if len == RANSAC_SAMPLE_SIZE {
        return (0..RANSAC_SAMPLE_SIZE).collect();
    }

    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (iteration as u64 + 1);
    let mut sample = Vec::with_capacity(RANSAC_SAMPLE_SIZE);
    while sample.len() < RANSAC_SAMPLE_SIZE {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let index = (state % len as u64) as usize;
        if !sample.contains(&index) {
            sample.push(index);
        }
    }
    sample
}

fn estimate_essential(
    correspondences: &[Correspondence],
    indices: &[usize],
) -> Option<Matrix3<f64>> {
    if indices.len() < RANSAC_SAMPLE_SIZE {
        return None;
    }

    // nalgebra computes a thin SVD for rectangular matrices. The minimal 8x9
    // design omits the ninth right-singular vector: exactly the one-dimensional
    // null space required by the eight-point algorithm. A zero row preserves the
    // homogeneous system while making the minimal fit square.
    let mut design = DMatrix::<f64>::zeros(indices.len().max(9), 9);
    for (row, &index) in indices.iter().enumerate() {
        let correspondence = &correspondences[index];
        let x1 = correspondence.x1.x;
        let y1 = correspondence.x1.y;
        let x2 = correspondence.x2.x;
        let y2 = correspondence.x2.y;
        design[(row, 0)] = x2 * x1;
        design[(row, 1)] = x2 * y1;
        design[(row, 2)] = x2;
        design[(row, 3)] = y2 * x1;
        design[(row, 4)] = y2 * y1;
        design[(row, 5)] = y2;
        design[(row, 6)] = x1;
        design[(row, 7)] = y1;
        design[(row, 8)] = 1.0;
    }

    let svd = design.svd(false, true);
    let v_t = svd.v_t?;
    let last_row = v_t.row(v_t.nrows() - 1);
    let raw = Matrix3::from_row_slice(&[
        last_row[0],
        last_row[1],
        last_row[2],
        last_row[3],
        last_row[4],
        last_row[5],
        last_row[6],
        last_row[7],
        last_row[8],
    ]);
    enforce_essential_constraints(raw)
}

fn enforce_essential_constraints(raw: Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = raw.svd(true, true);
    let mut u = svd.u?;
    let mut v_t = svd.v_t?;
    make_proper_basis(&mut u, &mut v_t);
    let sigma = (svd.singular_values[0] + svd.singular_values[1]) * 0.5;
    if !sigma.is_finite() || sigma <= f64::EPSILON {
        return None;
    }
    let diagonal = Matrix3::from_diagonal(&Vector3::new(sigma, sigma, 0.0));
    Some(u * diagonal * v_t)
}

fn make_proper_basis(u: &mut Matrix3<f64>, v_t: &mut Matrix3<f64>) {
    if u.determinant() < 0.0 {
        for row in 0..3 {
            u[(row, 2)] = -u[(row, 2)];
        }
    }
    if v_t.determinant() < 0.0 {
        for column in 0..3 {
            v_t[(2, column)] = -v_t[(2, column)];
        }
    }
}

fn sampson_error(essential: &Matrix3<f64>, correspondence: &Correspondence) -> f64 {
    let ex1 = essential * correspondence.x1;
    let etx2 = essential.transpose() * correspondence.x2;
    let numerator = correspondence.x2.dot(&ex1);
    let denominator = ex1.x * ex1.x + ex1.y * ex1.y + etx2.x * etx2.x + etx2.y * etx2.y;
    if denominator <= 1.0e-12 {
        return f64::INFINITY;
    }
    numerator * numerator / denominator
}

fn rotation_only_residual_pixels(
    correspondences: &[Correspondence],
    inliers: &[usize],
    focal_pixels: f64,
) -> Option<f64> {
    let mut covariance = Matrix3::<f64>::zeros();
    for &index in inliers {
        let source = correspondences[index].x1.normalize();
        let target = correspondences[index].x2.normalize();
        covariance += target * source.transpose();
    }

    let svd = covariance.svd(true, true);
    let mut u = svd.u?;
    let v_t = svd.v_t?;
    let mut rotation = u * v_t;
    if rotation.determinant() < 0.0 {
        for row in 0..3 {
            u[(row, 2)] = -u[(row, 2)];
        }
        rotation = u * v_t;
    }

    let mut residuals: Vec<f64> = inliers
        .iter()
        .map(|&index| {
            let source = correspondences[index].x1.normalize();
            let target = correspondences[index].x2.normalize();
            let predicted = rotation * source;
            predicted.dot(&target).clamp(-1.0, 1.0).acos() * focal_pixels
        })
        .filter(|residual| residual.is_finite())
        .collect();
    Some(median_f64(&mut residuals))
}

/// The pose hypothesis of `essential` with the most points in front of both cameras within
/// the reprojection gate; the caller applies the seed gates.
fn best_pose(
    essential: &Matrix3<f64>,
    correspondences: &[Correspondence],
    inliers: &[usize],
    focal_pixels: f64,
) -> Option<PoseCandidate> {
    let svd = essential.svd(true, true);
    let mut u = svd.u?;
    let mut v_t = svd.v_t?;
    make_proper_basis(&mut u, &mut v_t);

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let rotations = [
        proper_rotation(u * w * v_t),
        proper_rotation(u * w.transpose() * v_t),
    ];
    let translation = Vector3::new(u[(0, 2)], u[(1, 2)], u[(2, 2)]).normalize();

    let mut best: Option<PoseCandidate> = None;
    for rotation in rotations {
        for direction in [1.0, -1.0] {
            let candidate = evaluate_pose(
                rotation,
                translation * direction,
                correspondences,
                inliers,
                focal_pixels,
            );
            if best
                .as_ref()
                .is_none_or(|current| candidate.outranks(current))
            {
                best = Some(candidate);
            }
        }
    }
    best
}

fn proper_rotation(rotation: Matrix3<f64>) -> Matrix3<f64> {
    if rotation.determinant() < 0.0 {
        -rotation
    } else {
        rotation
    }
}

fn evaluate_pose(
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    correspondences: &[Correspondence],
    inliers: &[usize],
    focal_pixels: f64,
) -> PoseCandidate {
    let camera_center = -rotation.transpose() * translation;
    let mut points = Vec::new();
    let mut reprojection_errors = Vec::new();
    let mut triangulation_angles = Vec::new();
    let mut behind_camera = 0;
    let mut high_reprojection = 0;

    for &index in inliers {
        let correspondence = &correspondences[index];
        let Some(position) = triangulate(
            &correspondence.x1,
            &correspondence.x2,
            &rotation,
            &translation,
        ) else {
            behind_camera += 1;
            continue;
        };
        let camera_two_point = rotation * position + translation;
        if position.z <= 1.0e-6 || camera_two_point.z <= 1.0e-6 {
            behind_camera += 1;
            continue;
        }

        let reprojection_error_pixels =
            reprojection_error(&position, &camera_two_point, correspondence, focal_pixels);
        if !reprojection_error_pixels.is_finite()
            || reprojection_error_pixels > MAX_REPROJECTION_ERROR_PIXELS
        {
            high_reprojection += 1;
            continue;
        }

        let triangulation_angle_degrees = triangulation_angle(&position, &camera_center);
        if !triangulation_angle_degrees.is_finite() {
            continue;
        }

        reprojection_errors.push(reprojection_error_pixels);
        triangulation_angles.push(triangulation_angle_degrees);
        points.push(TriangulatedPoint {
            source_feature_index: correspondence.source_feature_index,
            descriptor_distance: correspondence.descriptor_distance,
            position,
            reprojection_error_pixels,
            triangulation_angle_degrees,
        });
    }

    PoseCandidate {
        rotation,
        translation,
        camera_center,
        points,
        median_reprojection_error_pixels: median_f64(&mut reprojection_errors),
        median_triangulation_angle_degrees: median_f64(&mut triangulation_angles),
        behind_camera,
        high_reprojection,
    }
}

fn triangulate(
    x1: &Vector3<f64>,
    x2: &Vector3<f64>,
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    let mut system = Matrix4::<f64>::zeros();
    let p1 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let p2 = [
        [
            rotation[(0, 0)],
            rotation[(0, 1)],
            rotation[(0, 2)],
            translation.x,
        ],
        [
            rotation[(1, 0)],
            rotation[(1, 1)],
            rotation[(1, 2)],
            translation.y,
        ],
        [
            rotation[(2, 0)],
            rotation[(2, 1)],
            rotation[(2, 2)],
            translation.z,
        ],
    ];

    for column in 0..4 {
        system[(0, column)] = x1.x * p1[2][column] - p1[0][column];
        system[(1, column)] = x1.y * p1[2][column] - p1[1][column];
        system[(2, column)] = x2.x * p2[2][column] - p2[0][column];
        system[(3, column)] = x2.y * p2[2][column] - p2[1][column];
    }

    let svd = system.svd(false, true);
    let v_t = svd.v_t?;
    let homogeneous = v_t.row(3);
    let w = homogeneous[3];
    if !w.is_finite() || w.abs() <= 1.0e-12 {
        return None;
    }
    let point = Vector3::new(homogeneous[0] / w, homogeneous[1] / w, homogeneous[2] / w);
    point.iter().all(|value| value.is_finite()).then_some(point)
}

fn reprojection_error(
    point_one: &Vector3<f64>,
    point_two: &Vector3<f64>,
    correspondence: &Correspondence,
    focal_pixels: f64,
) -> f64 {
    let projected_one = Vector3::new(point_one.x / point_one.z, point_one.y / point_one.z, 1.0);
    let projected_two = Vector3::new(point_two.x / point_two.z, point_two.y / point_two.z, 1.0);
    let error_one =
        (projected_one.x - correspondence.x1.x).hypot(projected_one.y - correspondence.x1.y);
    let error_two =
        (projected_two.x - correspondence.x2.x).hypot(projected_two.y - correspondence.x2.y);
    (error_one + error_two) * 0.5 * focal_pixels
}

fn triangulation_angle(point: &Vector3<f64>, camera_two_center: &Vector3<f64>) -> f64 {
    let ray_one = point.normalize();
    let ray_two = (*point - camera_two_center).normalize();
    ray_one.dot(&ray_two).clamp(-1.0, 1.0).acos().to_degrees()
}

fn median_f64(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_pose_points_blame_the_larger_loss() {
        let angle = MIN_TRIANGULATION_ANGLE_DEGREES * 4.0;
        // 20 inliers need 12; 6 survive. More points behind a camera: cheirality.
        assert_eq!(
            pose_rejection_from_counts(6, 10, 4, angle, 12),
            Some(TwoViewRejection::Cheirality)
        );
        // More points over the reprojection gate: reprojection.
        assert_eq!(
            pose_rejection_from_counts(6, 4, 10, angle, 12),
            Some(TwoViewRejection::Reprojection)
        );
        // A tie blames cheirality.
        assert_eq!(
            pose_rejection_from_counts(6, 7, 7, angle, 12),
            Some(TwoViewRejection::Cheirality)
        );
        // Enough points: only the triangulation angle can still reject.
        assert_eq!(pose_rejection_from_counts(12, 7, 1, angle, 12), None);
        assert_eq!(
            pose_rejection_from_counts(12, 0, 0, MIN_TRIANGULATION_ANGLE_DEGREES * 0.5, 12),
            Some(TwoViewRejection::TriangulationAngle)
        );
    }

    fn project(
        point: Vector3<f64>,
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
    ) -> Vector3<f64> {
        let camera_point = rotation * point + translation;
        Vector3::new(
            camera_point.x / camera_point.z,
            camera_point.y / camera_point.z,
            1.0,
        )
    }

    fn feature_from_bearing(
        bearing: Vector3<f64>,
        width: u32,
        height: u32,
        focal: f64,
        descriptor: i16,
    ) -> Feature {
        let x = (bearing.x * focal + width as f64 * 0.5).round();
        let y = (bearing.y * focal + height as f64 * 0.5).round();
        Feature {
            x: x.clamp(0.0, width.saturating_sub(1) as f64) as u32,
            y: y.clamp(0.0, height.saturating_sub(1) as f64) as u32,
            score: 1.0,
            descriptor: vec![descriptor],
        }
    }

    fn synthetic_correspondences(
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
    ) -> (Vec<Feature>, Vec<Feature>, Vec<FeatureMatch>) {
        let width = 640;
        let height = 480;
        let focal = 500.0;
        let mut source = Vec::new();
        let mut target = Vec::new();
        let mut matches = Vec::new();

        for index in 0..30 {
            let point = Vector3::new(
                (index % 6) as f64 * 0.36 - 0.9,
                (index / 6) as f64 * 0.28 - 0.56,
                3.2 + (index % 5) as f64 * 0.42,
            );
            let x1 = project(point, Matrix3::identity(), Vector3::zeros());
            let x2 = project(point, rotation, translation);
            source.push(feature_from_bearing(x1, width, height, focal, index as i16));
            target.push(feature_from_bearing(x2, width, height, focal, index as i16));
            matches.push(FeatureMatch {
                a: index,
                b: index,
                distance: 0.2,
            });
        }

        (source, target, matches)
    }

    fn exact_correspondences(
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
    ) -> Vec<Correspondence> {
        let points = [
            Vector3::new(-0.9, -0.6, 3.2),
            Vector3::new(-0.3, -0.5, 4.1),
            Vector3::new(0.4, -0.4, 5.0),
            Vector3::new(0.9, -0.2, 3.7),
            Vector3::new(-0.8, 0.3, 4.5),
            Vector3::new(-0.1, 0.6, 3.5),
            Vector3::new(0.5, 0.5, 5.3),
            Vector3::new(1.0, 0.2, 4.0),
        ];

        points
            .into_iter()
            .enumerate()
            .map(|(index, point)| Correspondence {
                source_feature_index: index,
                descriptor_distance: 0.0,
                x1: project(point, Matrix3::identity(), Vector3::zeros()),
                x2: project(point, rotation, translation),
            })
            .collect()
    }

    #[test]
    fn eight_point_fit_keeps_the_full_right_null_space() {
        let yaw = 0.06f64;
        let rotation = Matrix3::new(
            yaw.cos(),
            0.0,
            yaw.sin(),
            0.0,
            1.0,
            0.0,
            -yaw.sin(),
            0.0,
            yaw.cos(),
        );
        let correspondences = exact_correspondences(rotation, Vector3::new(-0.7, 0.04, 0.03));
        let indices: Vec<usize> = (0..8).collect();
        let essential = estimate_essential(&correspondences, &indices).expect("essential matrix");

        let max_error = correspondences
            .iter()
            .map(|correspondence| sampson_error(&essential, correspondence))
            .fold(0.0, f64::max);
        assert!(max_error < 1.0e-8, "maximum Sampson error: {max_error}");
    }

    #[test]
    fn ransac_budget_meets_the_minimum_inlier_confidence_contract() {
        let iterations = required_ransac_iterations(MIN_INLIER_RATIO);
        let sample_success = MIN_INLIER_RATIO.powi(RANSAC_SAMPLE_SIZE as i32);
        let achieved_confidence = 1.0 - (1.0 - sample_success).powi(iterations as i32);
        let legacy_confidence = 1.0 - (1.0 - sample_success).powi(96);

        assert!(achieved_confidence >= RANSAC_TARGET_CONFIDENCE);
        assert!(legacy_confidence < RANSAC_TARGET_CONFIDENCE);
        assert!(required_ransac_iterations(0.7) < iterations);
    }

    #[test]
    fn triangulates_known_two_view_point() {
        let rotation = Matrix3::identity();
        let translation = Vector3::new(-1.0, 0.0, 0.0);
        let point = Vector3::new(0.4, -0.2, 4.0);
        let x1 = project(point, Matrix3::identity(), Vector3::zeros());
        let x2 = project(point, rotation, translation);
        let recovered = triangulate(&x1, &x2, &rotation, &translation).expect("triangulation");
        assert!((recovered - point).norm() < 1.0e-8);
    }

    #[test]
    fn recovers_translating_two_view_fixture_with_outliers() {
        let yaw = 0.07f64;
        let rotation = Matrix3::new(
            yaw.cos(),
            0.0,
            yaw.sin(),
            0.0,
            1.0,
            0.0,
            -yaw.sin(),
            0.0,
            yaw.cos(),
        );
        let translation = Vector3::new(-0.72, 0.03, 0.04);
        let (source, mut target, matches) = synthetic_correspondences(rotation, translation);
        for feature in target.iter_mut().skip(26) {
            feature.x = feature.x.saturating_add(70).min(635);
            feature.y = feature.y.saturating_sub(45);
        }

        let estimate = estimate_two_view(&source, &target, &matches, 640, 480, 500.0)
            .expect("calibrated two-view estimate");
        assert!(estimate.inliers >= 24, "inliers: {}", estimate.inliers);
        assert!(
            estimate.points.len() >= 20,
            "points: {}",
            estimate.points.len()
        );
        assert!(estimate.median_reprojection_error_pixels < 1.5);
        assert!(estimate.median_triangulation_angle_degrees > 0.5);

        let rotation_delta = estimate.rotation * rotation.transpose();
        let cosine = ((rotation_delta.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        assert!(cosine.acos() < 0.08);
        assert!(estimate.translation.dot(&translation.normalize()) > 0.85);
    }

    #[test]
    fn pure_rotation_is_rejected_as_two_view_baseline() {
        let yaw = 0.09f64;
        let rotation = Matrix3::new(
            yaw.cos(),
            0.0,
            yaw.sin(),
            0.0,
            1.0,
            0.0,
            -yaw.sin(),
            0.0,
            yaw.cos(),
        );
        let (source, target, matches) = synthetic_correspondences(rotation, Vector3::zeros());
        assert!(estimate_two_view(&source, &target, &matches, 640, 480, 500.0).is_none());

        // The rejection names the baseline gate and keeps the measured residual.
        let outcome = evaluate_two_view(&source, &target, &matches, 640, 480, 500.0);
        assert_eq!(outcome.result.err(), Some(TwoViewRejection::Baseline));
        assert!(outcome.evidence.inliers >= 24);
        let residual = outcome.evidence.rotation_only_residual_pixels.unwrap();
        assert!(
            residual <= MAX_ROTATION_ONLY_RESIDUAL_PIXELS,
            "residual {residual}"
        );
        assert!(outcome.evidence.pose.is_none());
    }

    #[test]
    fn accepted_pair_records_the_evidence_of_every_gate() {
        let (source, target, matches) =
            synthetic_correspondences(Matrix3::identity(), Vector3::new(-0.72, 0.03, 0.04));
        let outcome = evaluate_two_view(&source, &target, &matches, 640, 480, 500.0);
        let estimate = outcome.result.expect("translating pair is accepted");
        let evidence = outcome.evidence;
        assert_eq!(evidence.matches, 30);
        assert_eq!(evidence.inliers, estimate.inliers);
        assert!(
            evidence.rotation_only_residual_pixels.unwrap() > MAX_ROTATION_ONLY_RESIDUAL_PIXELS
        );
        let pose = evidence.pose.expect("pose evidence");
        assert_eq!(pose.accepted_points, estimate.points.len());
        assert!(pose.accepted_points >= pose.required_points);
        assert_eq!(
            pose.median_triangulation_angle_degrees,
            Some(estimate.median_triangulation_angle_degrees)
        );
    }

    #[test]
    fn too_few_matches_are_rejected_before_ransac() {
        let (source, target, matches) =
            synthetic_correspondences(Matrix3::identity(), Vector3::new(-0.72, 0.03, 0.04));
        let outcome = evaluate_two_view(&source, &target, &matches[..7], 640, 480, 500.0);
        assert_eq!(outcome.result.err(), Some(TwoViewRejection::Matches));
        assert_eq!(outcome.evidence.matches, 7);
        assert_eq!(outcome.evidence.inliers, 0);
    }

    #[test]
    fn inconsistent_matches_are_rejected_at_the_inlier_gate() {
        let (source, target, mut matches) =
            synthetic_correspondences(Matrix3::identity(), Vector3::new(-0.72, 0.03, 0.04));
        // Shuffle every target: no essential matrix explains half of the matches.
        for (index, feature_match) in matches.iter_mut().enumerate() {
            feature_match.b = (index * 7 + 3) % 30;
        }
        let outcome = evaluate_two_view(&source, &target, &matches, 640, 480, 500.0);
        assert_eq!(outcome.result.err(), Some(TwoViewRejection::Inliers));
        assert!((outcome.evidence.inliers as f64) < MIN_INLIER_RATIO * 30.0);
    }

    #[test]
    fn short_baseline_to_deep_points_is_rejected_at_the_angle_gate() {
        // Depths from 1 to 13 units with a 0.03-unit baseline: the depth spread leaves
        // parallax no rotation explains, but the median ray angle stays below the gate.
        let (width, height, focal) = (640, 480, 500.0);
        let translation = Vector3::new(-0.03, 0.0, 0.0);
        let mut source = Vec::new();
        let mut target = Vec::new();
        let mut matches = Vec::new();
        for index in 0..40 {
            let depth = 1.0 + (index % 8) as f64 * 1.7;
            let point = Vector3::new(
                ((index % 5) as f64 * 0.3 - 0.6) * depth,
                ((index / 5) as f64 * 0.1 - 0.35) * depth,
                depth,
            );
            let x1 = project(point, Matrix3::identity(), Vector3::zeros());
            let x2 = project(point, Matrix3::identity(), translation);
            source.push(feature_from_bearing(x1, width, height, focal, index as i16));
            target.push(feature_from_bearing(x2, width, height, focal, index as i16));
            matches.push(FeatureMatch {
                a: index,
                b: index,
                distance: 0.2,
            });
        }
        let outcome = evaluate_two_view(&source, &target, &matches, width, height, focal);
        assert_eq!(
            outcome.result.as_ref().err(),
            Some(&TwoViewRejection::TriangulationAngle),
            "evidence: {:?}",
            outcome.evidence
        );
        let pose = outcome.evidence.pose.expect("pose evidence");
        assert!(pose.accepted_points >= pose.required_points);
        assert!(pose.median_triangulation_angle_degrees.unwrap() < MIN_TRIANGULATION_ANGLE_DEGREES);
    }
}
