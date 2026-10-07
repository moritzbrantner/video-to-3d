//! Per-region geometry confidence over validated reconstruction evidence.
//!
//! The field scores every provenance region of a [`ReconstructionEvidenceView`]
//! from five explicit factors, each in `[0, 1]`:
//! - **camera support**: distinct accepted supporting source cameras,
//!   saturating at [`SUPPORT_SATURATION_VIEWS`];
//! - **reprojection**: the worst median reprojection error among the
//!   reference and supporting cameras, `1 / (1 + error / 1 px)`; a camera
//!   without a reprojection estimate scores [`UNKNOWN_FACTOR`];
//! - **reciprocal agreement**: the lower median of the provider-recorded point
//!   confidence (for the built-in provider the product of source support,
//!   reciprocal depth consistency, photometric error and depth-margin
//!   evidence);
//! - **triangulation strength**: the lower median, over the region's points, of
//!   the widest reference/source ray angle, saturating at
//!   [`TRIANGULATION_SATURATION_DEGREES`];
//! - **reference exclusivity**: one minus the fraction of the region's
//!   triangles that also touch a region of another (or no) reference camera,
//!   i.e. the reference-ownership ambiguity the material bake also refuses.
//!
//! The region confidence is the geometric mean of the factors multiplied by a
//! provenance weight, so a single missing factor zeroes it and learned or
//! generative evidence can never outrank equally supported observed geometry.
//! Generative completion has no geometric evidence and is always
//! [`ConfidenceBand::Unsupported`].
//!
//! The field is a pure function of the accepted evidence. It never reads or
//! changes geometry outside the borrowed view, and it records no confidence for
//! space without accepted points: holes stay unsupported.

use crate::{
    EvidenceCamera, EvidenceOrigin, EvidenceRange, MeshTriangle, ReconstructionEvidenceView,
};
use serde::Serialize;
use std::collections::HashMap;

pub const GEOMETRY_CONFIDENCE_SCHEMA_VERSION: u32 = 1;
/// Supporting source cameras at which camera support saturates.
pub const SUPPORT_SATURATION_VIEWS: usize = 3;
/// Median reprojection error (pixels) that halves the reprojection factor.
pub const REPROJECTION_HALF_SCORE_PIXELS: f32 = 1.0;
/// Reference/source ray angle (degrees) at which triangulation saturates.
pub const TRIANGULATION_SATURATION_DEGREES: f32 = 8.0;
/// Factor used when a provider supplies no estimate for it.
pub const UNKNOWN_FACTOR: f32 = 0.5;
/// Lower bounds of the high and medium bands.
pub const HIGH_CONFIDENCE: f32 = 0.6;
pub const MEDIUM_CONFIDENCE: f32 = 0.35;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceBand {
    Unsupported,
    Low,
    Medium,
    High,
}

impl ConfidenceBand {
    pub fn of(confidence: f32) -> Self {
        if confidence >= HIGH_CONFIDENCE {
            Self::High
        } else if confidence >= MEDIUM_CONFIDENCE {
            Self::Medium
        } else if confidence > 0.0 {
            Self::Low
        } else {
            Self::Unsupported
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ConfidenceFactors {
    pub camera_support: f32,
    pub reprojection: f32,
    pub reciprocal_agreement: f32,
    pub triangulation: f32,
    pub reference_exclusivity: f32,
}

impl ConfidenceFactors {
    const ZERO: Self = Self {
        camera_support: 0.0,
        reprojection: 0.0,
        reciprocal_agreement: 0.0,
        triangulation: 0.0,
        reference_exclusivity: 0.0,
    };

    fn geometric_mean(&self) -> f32 {
        let factors = [
            self.camera_support,
            self.reprojection,
            self.reciprocal_agreement,
            self.triangulation,
            self.reference_exclusivity,
        ];
        if factors.iter().any(|factor| *factor <= 0.0) {
            return 0.0;
        }
        let log_sum: f64 = factors.iter().map(|factor| f64::from(*factor).ln()).sum();
        (log_sum / factors.len() as f64).exp().clamp(0.0, 1.0) as f32
    }
}

/// Confidence of one evidence region; `region` indexes `evidence.regions`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RegionConfidence {
    pub region: usize,
    pub origin: EvidenceOrigin,
    pub reference_frame: Option<usize>,
    pub source_frames: Vec<usize>,
    pub points: EvidenceRange,
    pub factors: ConfidenceFactors,
    pub provenance_weight: f32,
    pub confidence: f32,
    pub band: ConfidenceBand,
    pub worst_reprojection_error_pixels: Option<f32>,
    pub median_triangulation_angle_degrees: Option<f32>,
    /// Triangles with at least one vertex in this region.
    pub triangles: usize,
    /// Of those, triangles that also touch another (or no) reference camera.
    pub mixed_reference_triangles: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct BandCounts {
    pub high: usize,
    pub medium: usize,
    pub low: usize,
    pub unsupported: usize,
}

impl BandCounts {
    fn add(&mut self, band: ConfidenceBand, count: usize) {
        match band {
            ConfidenceBand::High => self.high += count,
            ConfidenceBand::Medium => self.medium += count,
            ConfidenceBand::Low => self.low += count,
            ConfidenceBand::Unsupported => self.unsupported += count,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GeometryConfidenceSummary {
    pub schema_version: u32,
    pub regions: BandCounts,
    pub points: BandCounts,
    pub triangles: BandCounts,
    pub min_region_confidence: Option<f32>,
    pub max_region_confidence: Option<f32>,
}

/// Region confidences plus a point-to-region index over the borrowed evidence.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GeometryConfidenceField {
    pub schema_version: u32,
    pub point_count: usize,
    pub regions: Vec<RegionConfidence>,
    /// `(start, index into regions)`, sorted by start; regions partition points.
    #[serde(skip)]
    starts: Vec<(usize, usize)>,
}

impl GeometryConfidenceField {
    pub fn from_evidence(evidence: &ReconstructionEvidenceView<'_>) -> Self {
        let cameras: HashMap<usize, &EvidenceCamera> = evidence
            .cameras
            .iter()
            .map(|camera| (camera.frame_index, camera))
            .collect();
        let mut starts: Vec<(usize, usize)> = evidence
            .regions
            .iter()
            .enumerate()
            .map(|(index, region)| (region.points.start, index))
            .collect();
        starts.sort_unstable();
        let region_of =
            |point: usize| locate(&starts, point, |region| evidence.regions[region].points);

        let mut touching = vec![0usize; evidence.regions.len()];
        let mut mixed = vec![0usize; evidence.regions.len()];
        for triangle in evidence.triangles {
            let corners = [triangle.a, triangle.b, triangle.c].map(region_of);
            let references = corners
                .map(|region| region.and_then(|region| evidence.regions[region].reference_frame));
            let ambiguous = references.iter().any(|reference| reference.is_none())
                || references[0] != references[1]
                || references[1] != references[2];
            let mut seen: Vec<usize> = corners.iter().flatten().copied().collect();
            seen.sort_unstable();
            seen.dedup();
            for region in seen {
                touching[region] += 1;
                if ambiguous {
                    mixed[region] += 1;
                }
            }
        }

        let regions = evidence
            .regions
            .iter()
            .enumerate()
            .map(|(index, region)| {
                let points = &evidence.points
                    [region.points.start..region.points.start + region.points.count];
                let reference = region.reference_frame.and_then(|frame| cameras.get(&frame));
                let sources: Vec<&EvidenceCamera> = region
                    .source_frames
                    .iter()
                    .filter_map(|frame| cameras.get(frame).copied())
                    .collect();
                let provenance_weight = provenance_weight(region.origin);

                let (factors, worst_error, median_angle) = match reference {
                    Some(reference) if provenance_weight > 0.0 && !sources.is_empty() => {
                        let support =
                            (sources.len() as f32 / SUPPORT_SATURATION_VIEWS as f32).min(1.0);
                        let mut reprojection = 1.0f32;
                        let mut worst_error: Option<f32> = None;
                        for camera in std::iter::once(*reference).chain(sources.iter().copied()) {
                            let score = match camera.median_reprojection_error_pixels {
                                Some(error) => {
                                    worst_error = Some(worst_error.map_or(error, |w| w.max(error)));
                                    1.0 / (1.0 + error / REPROJECTION_HALF_SCORE_PIXELS)
                                }
                                None => UNKNOWN_FACTOR,
                            };
                            reprojection = reprojection.min(score);
                        }
                        let agreement = lower_median(points.iter().map(|point| point.confidence))
                            .unwrap_or(0.0);
                        let reference_center = camera_center(reference);
                        let source_centers: Vec<[f64; 3]> =
                            sources.iter().map(|camera| camera_center(camera)).collect();
                        let median_angle = lower_median(points.iter().map(|point| {
                            let point = [point.x as f64, point.y as f64, point.z as f64];
                            source_centers
                                .iter()
                                .map(|source| ray_angle_degrees(point, reference_center, *source))
                                .fold(0.0f32, f32::max)
                        }));
                        let triangulation = median_angle.map_or(0.0, |angle| {
                            (angle / TRIANGULATION_SATURATION_DEGREES).clamp(0.0, 1.0)
                        });
                        let exclusivity = if touching[index] == 0 {
                            1.0
                        } else {
                            1.0 - mixed[index] as f32 / touching[index] as f32
                        };
                        (
                            ConfidenceFactors {
                                camera_support: support,
                                reprojection,
                                reciprocal_agreement: agreement.clamp(0.0, 1.0),
                                triangulation,
                                reference_exclusivity: exclusivity,
                            },
                            worst_error,
                            median_angle,
                        )
                    }
                    _ => (ConfidenceFactors::ZERO, None, None),
                };
                let confidence = factors.geometric_mean() * provenance_weight;
                RegionConfidence {
                    region: index,
                    origin: region.origin,
                    reference_frame: region.reference_frame,
                    source_frames: region.source_frames.clone(),
                    points: region.points,
                    factors,
                    provenance_weight,
                    confidence,
                    band: ConfidenceBand::of(confidence),
                    worst_reprojection_error_pixels: worst_error,
                    median_triangulation_angle_degrees: median_angle,
                    triangles: touching[index],
                    mixed_reference_triangles: mixed[index],
                }
            })
            .collect();

        Self {
            schema_version: GEOMETRY_CONFIDENCE_SCHEMA_VERSION,
            point_count: evidence.points.len(),
            regions,
            starts,
        }
    }

    /// Confidence of one evidence region, by its index in `evidence.regions`.
    pub fn region(&self, region: usize) -> Option<&RegionConfidence> {
        self.regions.get(region)
    }

    /// Whether this field is the one `evidence` yields. Confidence depends on cameras,
    /// point positions and confidences, topology and regions, so the field is recomputed
    /// and compared whole rather than trusted by matching metadata.
    pub fn describes(&self, evidence: &ReconstructionEvidenceView<'_>) -> bool {
        *self == Self::from_evidence(evidence)
    }

    /// The region that owns an accepted point.
    pub fn region_of_point(&self, point: usize) -> Option<&RegionConfidence> {
        let region = locate(&self.starts, point, |region| self.regions[region].points)?;
        Some(&self.regions[region])
    }

    /// Confidence of an accepted point; `0` for anything outside the evidence.
    pub fn point_confidence(&self, point: usize) -> f32 {
        self.region_of_point(point)
            .map_or(0.0, |region| region.confidence)
    }

    /// Conservative triangle confidence: its weakest vertex region.
    pub fn triangle_confidence(&self, triangle: &MeshTriangle) -> f32 {
        [triangle.a, triangle.b, triangle.c]
            .into_iter()
            .map(|point| self.point_confidence(point))
            .fold(1.0, f32::min)
    }

    /// Regions a completion policy with this confidence ceiling may consider
    /// filling or replacing. Space outside every region is not listed: it has
    /// no accepted evidence and stays an unsupported hole.
    pub fn regions_below(&self, ceiling: f32) -> impl Iterator<Item = &RegionConfidence> {
        self.regions
            .iter()
            .filter(move |region| region.confidence < ceiling)
    }

    pub fn summary(&self, triangles: &[MeshTriangle]) -> GeometryConfidenceSummary {
        let mut summary = GeometryConfidenceSummary {
            schema_version: self.schema_version,
            ..GeometryConfidenceSummary::default()
        };
        for region in &self.regions {
            summary.regions.add(region.band, 1);
            summary.points.add(region.band, region.points.count);
            summary.min_region_confidence = Some(
                summary
                    .min_region_confidence
                    .map_or(region.confidence, |value| value.min(region.confidence)),
            );
            summary.max_region_confidence = Some(
                summary
                    .max_region_confidence
                    .map_or(region.confidence, |value| value.max(region.confidence)),
            );
        }
        for triangle in triangles {
            summary
                .triangles
                .add(ConfidenceBand::of(self.triangle_confidence(triangle)), 1);
        }
        summary
    }

    pub fn diagnostic(&self, triangles: &[MeshTriangle]) -> String {
        let summary = self.summary(triangles);
        if self.regions.is_empty() {
            return "Geometry confidence: no accepted surface regions; every area is unsupported."
                .into();
        }
        format!(
            "Geometry confidence v{}: {} regions ({} high, {} medium, {} low, {} unsupported) from camera support, reprojection, reciprocal agreement, triangulation and reference exclusivity; confidence {:.2}-{:.2}; triangles {} high, {} medium, {} low, {} unsupported. Areas without accepted points carry no confidence.",
            summary.schema_version,
            self.regions.len(),
            summary.regions.high,
            summary.regions.medium,
            summary.regions.low,
            summary.regions.unsupported,
            summary.min_region_confidence.unwrap_or(0.0),
            summary.max_region_confidence.unwrap_or(0.0),
            summary.triangles.high,
            summary.triangles.medium,
            summary.triangles.low,
            summary.triangles.unsupported,
        )
    }
}

/// Weight that keeps provenance classes ordered at equal evidence.
pub fn provenance_weight(origin: EvidenceOrigin) -> f32 {
    match origin {
        EvidenceOrigin::GeometricMultiView => 1.0,
        EvidenceOrigin::RevalidatedCompletion => 0.85,
        EvidenceOrigin::LearnedMultiView => 0.5,
        EvidenceOrigin::GenerativeCompletion => 0.0,
    }
}

/// The region whose point range holds `point`, given region starts sorted
/// ascending.
fn locate(
    starts: &[(usize, usize)],
    point: usize,
    range_of: impl Fn(usize) -> EvidenceRange,
) -> Option<usize> {
    let slot = starts.partition_point(|(start, _)| *start <= point);
    let (_, region) = *starts.get(slot.checked_sub(1)?)?;
    let range = range_of(region);
    (point < range.start + range.count).then_some(region)
}

fn lower_median(values: impl Iterator<Item = f32>) -> Option<f32> {
    let mut values: Vec<f32> = values.filter(|value| value.is_finite()).collect();
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f32::total_cmp);
    Some(values[(values.len() - 1) / 2])
}

/// World-space center `-Rᵀ t` of a world-to-camera pose.
fn camera_center(camera: &EvidenceCamera) -> [f64; 3] {
    let r = camera.rotation.map(f64::from);
    let t = camera.translation.map(f64::from);
    [
        -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
        -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
        -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
    ]
}

fn ray_angle_degrees(point: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f32 {
    let u = [a[0] - point[0], a[1] - point[1], a[2] - point[2]];
    let v = [b[0] - point[0], b[1] - point[1], b[2] - point[2]];
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let norms = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt()
        * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norms <= f64::EPSILON {
        return 0.0;
    }
    (dot / norms).clamp(-1.0, 1.0).acos().to_degrees() as f32
}
