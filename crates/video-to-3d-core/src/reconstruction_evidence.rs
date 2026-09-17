use crate::{MeshTriangle, Point3, ReconstructionResult};
use serde::Serialize;
use std::collections::HashSet;

pub const RECONSTRUCTION_EVIDENCE_SCHEMA_VERSION: u32 = 1;
const CLASSIC_PROVIDER_ID: &str = "video-to-3d-core/classic";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionProviderClass {
    GeometricMultiView,
    LearnedMultiView,
    GenerativeCompletion,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ReconstructionProviderDescriptor {
    pub id: String,
    pub class: ReconstructionProviderClass,
    pub version: Option<String>,
}

impl ReconstructionProviderDescriptor {
    pub fn new(
        id: impl Into<String>,
        class: ReconstructionProviderClass,
        version: Option<String>,
    ) -> Result<Self, String> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err("reconstruction evidence provider id must not be empty".into());
        }
        Ok(Self { id, class, version })
    }

    pub fn classic() -> Self {
        Self {
            id: CLASSIC_PROVIDER_ID.into(),
            class: ReconstructionProviderClass::GeometricMultiView,
            version: Some(env!("CARGO_PKG_VERSION").into()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceScale {
    ArbitraryMonocular,
    Metric,
    ProviderLocal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceCameraAuthority {
    CalibratedSeed,
    RegisteredGeometry,
    ProviderEstimated,
}

#[derive(Clone, Debug, Serialize)]
pub struct EvidenceCamera {
    pub frame_index: usize,
    pub authority: EvidenceCameraAuthority,
    pub rotation: [f32; 9],
    pub translation: [f32; 3],
    pub confidence: Option<f32>,
    pub median_reprojection_error_pixels: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceOrigin {
    GeometricMultiView,
    RevalidatedCompletion,
    LearnedMultiView,
    GenerativeCompletion,
}

impl EvidenceOrigin {
    fn requires_camera_evidence(self) -> bool {
        !matches!(self, Self::GenerativeCompletion)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct EvidenceRange {
    pub start: usize,
    pub count: usize,
}

impl EvidenceRange {
    pub fn new(start: usize, count: usize) -> Self {
        Self { start, count }
    }

    fn end(self) -> Option<usize> {
        self.start.checked_add(self.count)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SurfaceEvidenceRegion {
    pub origin: EvidenceOrigin,
    pub reference_frame: Option<usize>,
    pub source_frames: Vec<usize>,
    pub points: EvidenceRange,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ReconstructionEvidenceSummary {
    pub cameras: usize,
    pub regions: usize,
    pub geometric_points: usize,
    pub revalidated_completion_points: usize,
    pub learned_points: usize,
    pub generative_completion_points: usize,
    pub total_points: usize,
    pub triangles: usize,
}

#[derive(Debug, Serialize)]
pub struct ReconstructionEvidenceView<'a> {
    pub schema_version: u32,
    pub provider: ReconstructionProviderDescriptor,
    pub scale: EvidenceScale,
    pub cameras: Vec<EvidenceCamera>,
    pub regions: Vec<SurfaceEvidenceRegion>,
    pub points: &'a [Point3],
    pub triangles: &'a [MeshTriangle],
}

impl<'a> ReconstructionEvidenceView<'a> {
    pub fn new(
        provider: ReconstructionProviderDescriptor,
        scale: EvidenceScale,
        cameras: Vec<EvidenceCamera>,
        regions: Vec<SurfaceEvidenceRegion>,
        points: &'a [Point3],
        triangles: &'a [MeshTriangle],
    ) -> Result<Self, String> {
        let evidence = Self {
            schema_version: RECONSTRUCTION_EVIDENCE_SCHEMA_VERSION,
            provider,
            scale,
            cameras,
            regions,
            points,
            triangles,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn from_classic(reconstruction: &'a ReconstructionResult) -> Result<Self, String> {
        let cameras = classic_camera_evidence(reconstruction)?;
        let regions = classic_surface_regions(reconstruction);
        Self::new(
            ReconstructionProviderDescriptor::classic(),
            EvidenceScale::ArbitraryMonocular,
            cameras,
            regions,
            &reconstruction.dense_points,
            &reconstruction.mesh_triangles,
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != RECONSTRUCTION_EVIDENCE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported reconstruction evidence schema version {}",
                self.schema_version
            ));
        }
        if self.provider.id.trim().is_empty() {
            return Err("reconstruction evidence provider id must not be empty".into());
        }

        validate_cameras(&self.cameras)?;
        validate_points(self.points)?;
        validate_regions(&self.regions, self.points.len(), &self.cameras)?;
        validate_triangles(self.triangles, self.points.len())?;
        Ok(())
    }

    pub fn summary(&self) -> ReconstructionEvidenceSummary {
        let mut summary = ReconstructionEvidenceSummary {
            cameras: self.cameras.len(),
            regions: self.regions.len(),
            total_points: self.points.len(),
            triangles: self.triangles.len(),
            ..ReconstructionEvidenceSummary::default()
        };
        for region in &self.regions {
            match region.origin {
                EvidenceOrigin::GeometricMultiView => {
                    summary.geometric_points += region.points.count;
                }
                EvidenceOrigin::RevalidatedCompletion => {
                    summary.revalidated_completion_points += region.points.count;
                }
                EvidenceOrigin::LearnedMultiView => {
                    summary.learned_points += region.points.count;
                }
                EvidenceOrigin::GenerativeCompletion => {
                    summary.generative_completion_points += region.points.count;
                }
            }
        }
        summary
    }

    pub fn diagnostic(&self) -> String {
        let summary = self.summary();
        format!(
            "Provider-neutral reconstruction evidence v{} validated for {}: {} accepted cameras, {} provenance regions, {} shared dense points ({} geometric, {} revalidated completion, {} learned, {} generative completion), and {} accepted triangles. The evidence view borrows the existing geometry buffers instead of materializing a second point cloud or mesh.",
            self.schema_version,
            self.provider.id,
            summary.cameras,
            summary.regions,
            summary.total_points,
            summary.geometric_points,
            summary.revalidated_completion_points,
            summary.learned_points,
            summary.generative_completion_points,
            summary.triangles,
        )
    }
}

fn classic_camera_evidence(
    reconstruction: &ReconstructionResult,
) -> Result<Vec<EvidenceCamera>, String> {
    let Some(seed) = reconstruction.calibrated_pair.as_ref() else {
        if !reconstruction.registered_views.is_empty() || !reconstruction.dense_points.is_empty() {
            return Err(
                "classic reconstruction evidence contains registered/dense geometry without a calibrated seed pair"
                    .into(),
            );
        }
        return Ok(Vec::new());
    };

    let mut cameras = vec![
        EvidenceCamera {
            frame_index: seed.from_frame,
            authority: EvidenceCameraAuthority::CalibratedSeed,
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [0.0, 0.0, 0.0],
            confidence: Some(seed.inlier_ratio),
            median_reprojection_error_pixels: Some(seed.median_reprojection_error_pixels),
        },
        EvidenceCamera {
            frame_index: seed.to_frame,
            authority: EvidenceCameraAuthority::CalibratedSeed,
            rotation: seed.relative_rotation,
            translation: seed.translation_direction,
            confidence: Some(seed.inlier_ratio),
            median_reprojection_error_pixels: Some(seed.median_reprojection_error_pixels),
        },
    ];

    for view in &reconstruction.registered_views {
        if view.frame_index == seed.from_frame || view.frame_index == seed.to_frame {
            return Err(format!(
                "classic reconstruction evidence duplicates calibrated seed frame {} in registered-view evidence",
                view.frame_index
            ));
        }
        cameras.push(EvidenceCamera {
            frame_index: view.frame_index,
            authority: EvidenceCameraAuthority::RegisteredGeometry,
            rotation: view.rotation,
            translation: view.translation,
            confidence: Some(view.inlier_ratio),
            median_reprojection_error_pixels: Some(view.median_reprojection_error_pixels),
        });
    }

    Ok(cameras)
}

fn classic_surface_regions(reconstruction: &ReconstructionResult) -> Vec<SurfaceEvidenceRegion> {
    let mut regions = Vec::with_capacity(reconstruction.dense.reference_patches.len() * 2);
    for patch in &reconstruction.dense.reference_patches {
        if patch.primary_points > 0 {
            regions.push(SurfaceEvidenceRegion {
                origin: EvidenceOrigin::GeometricMultiView,
                reference_frame: Some(patch.reference_frame),
                source_frames: patch.source_frames.clone(),
                points: EvidenceRange::new(patch.primary_start, patch.primary_points),
            });
        }
        if patch.completed_points > 0 {
            regions.push(SurfaceEvidenceRegion {
                origin: EvidenceOrigin::RevalidatedCompletion,
                reference_frame: Some(patch.reference_frame),
                source_frames: patch.source_frames.clone(),
                points: EvidenceRange::new(patch.completion_start, patch.completed_points),
            });
        }
    }
    regions
}

fn validate_cameras(cameras: &[EvidenceCamera]) -> Result<(), String> {
    let mut frames = HashSet::with_capacity(cameras.len());
    for camera in cameras {
        if !frames.insert(camera.frame_index) {
            return Err(format!(
                "reconstruction evidence contains duplicate camera frame {}",
                camera.frame_index
            ));
        }
        if camera
            .rotation
            .iter()
            .chain(camera.translation.iter())
            .any(|value| !value.is_finite())
        {
            return Err(format!(
                "reconstruction evidence camera {} contains a non-finite pose",
                camera.frame_index
            ));
        }
        if camera
            .confidence
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(format!(
                "reconstruction evidence camera {} has invalid confidence",
                camera.frame_index
            ));
        }
        if camera
            .median_reprojection_error_pixels
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(format!(
                "reconstruction evidence camera {} has invalid reprojection error",
                camera.frame_index
            ));
        }
    }
    Ok(())
}

fn validate_points(points: &[Point3]) -> Result<(), String> {
    for (index, point) in points.iter().enumerate() {
        if !point.x.is_finite()
            || !point.y.is_finite()
            || !point.z.is_finite()
            || !point.confidence.is_finite()
            || !(0.0..=1.0).contains(&point.confidence)
        {
            return Err(format!(
                "reconstruction evidence point {index} contains invalid geometry/confidence"
            ));
        }
    }
    Ok(())
}

fn validate_regions(
    regions: &[SurfaceEvidenceRegion],
    point_count: usize,
    cameras: &[EvidenceCamera],
) -> Result<(), String> {
    if point_count == 0 {
        if regions.is_empty() {
            return Ok(());
        }
        return Err("reconstruction evidence has surface regions but no dense points".into());
    }
    if regions.is_empty() {
        return Err("reconstruction evidence dense points have no provenance regions".into());
    }

    let camera_frames: HashSet<usize> = cameras.iter().map(|camera| camera.frame_index).collect();
    let mut intervals = Vec::with_capacity(regions.len());
    for (region_index, region) in regions.iter().enumerate() {
        if region.points.count == 0 {
            return Err(format!(
                "reconstruction evidence region {region_index} has an empty point range"
            ));
        }
        let Some(end) = region.points.end() else {
            return Err(format!(
                "reconstruction evidence region {region_index} point range overflows"
            ));
        };
        if end > point_count {
            return Err(format!(
                "reconstruction evidence region {region_index} point range {}..{} exceeds {point_count} dense points",
                region.points.start, end
            ));
        }

        if region.origin.requires_camera_evidence() {
            let Some(reference_frame) = region.reference_frame else {
                return Err(format!(
                    "reconstruction evidence region {region_index} requires a reference camera"
                ));
            };
            if !camera_frames.contains(&reference_frame) {
                return Err(format!(
                    "reconstruction evidence region {region_index} references unaccepted camera frame {reference_frame}"
                ));
            }
            if region.source_frames.is_empty() {
                return Err(format!(
                    "reconstruction evidence region {region_index} has no supporting source cameras"
                ));
            }
            for source_frame in &region.source_frames {
                if !camera_frames.contains(source_frame) {
                    return Err(format!(
                        "reconstruction evidence region {region_index} references unaccepted source camera frame {source_frame}"
                    ));
                }
            }
        }

        intervals.push((region.points.start, end, region_index));
    }

    intervals.sort_by_key(|(start, _, _)| *start);
    let mut cursor = 0usize;
    for (start, end, region_index) in intervals {
        if start != cursor {
            return Err(format!(
                "reconstruction evidence provenance is not an exact non-overlapping partition at region {region_index}: expected point {cursor}, found range {start}..{end}"
            ));
        }
        cursor = end;
    }
    if cursor != point_count {
        return Err(format!(
            "reconstruction evidence provenance covers {cursor} of {point_count} dense points"
        ));
    }

    Ok(())
}

fn validate_triangles(triangles: &[MeshTriangle], point_count: usize) -> Result<(), String> {
    for (index, triangle) in triangles.iter().enumerate() {
        if triangle.a >= point_count || triangle.b >= point_count || triangle.c >= point_count {
            return Err(format!(
                "reconstruction evidence triangle {index} references a point outside the shared evidence buffer"
            ));
        }
        if triangle.a == triangle.b || triangle.b == triangle.c || triangle.c == triangle.a {
            return Err(format!(
                "reconstruction evidence triangle {index} is topologically degenerate"
            ));
        }
        if !triangle.confidence.is_finite() || !(0.0..=1.0).contains(&triangle.confidence) {
            return Err(format!(
                "reconstruction evidence triangle {index} has invalid confidence"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameInput, ReconstructionOptions, ReconstructionRequest};

    fn point(x: f32, y: f32, z: f32) -> Point3 {
        Point3 {
            x,
            y,
            z,
            confidence: 0.9,
            r: 128,
            g: 128,
            b: 128,
        }
    }

    fn camera(frame_index: usize) -> EvidenceCamera {
        EvidenceCamera {
            frame_index,
            authority: EvidenceCameraAuthority::ProviderEstimated,
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [0.0, 0.0, 0.0],
            confidence: Some(0.8),
            median_reprojection_error_pixels: None,
        }
    }

    #[test]
    fn provider_boundary_borrows_geometry_buffers() {
        let points = vec![
            point(0.0, 0.0, 1.0),
            point(1.0, 0.0, 1.0),
            point(0.0, 1.0, 1.0),
        ];
        let triangles = vec![MeshTriangle {
            a: 0,
            b: 1,
            c: 2,
            confidence: 0.9,
        }];
        let regions = vec![SurfaceEvidenceRegion {
            origin: EvidenceOrigin::LearnedMultiView,
            reference_frame: Some(0),
            source_frames: vec![1],
            points: EvidenceRange::new(0, 3),
        }];
        let evidence = ReconstructionEvidenceView::new(
            ReconstructionProviderDescriptor::new(
                "test/learned",
                ReconstructionProviderClass::LearnedMultiView,
                Some("1".into()),
            )
            .expect("provider"),
            EvidenceScale::ProviderLocal,
            vec![camera(0), camera(1)],
            regions,
            &points,
            &triangles,
        )
        .expect("valid evidence");

        assert!(std::ptr::eq(evidence.points.as_ptr(), points.as_ptr()));
        assert!(std::ptr::eq(
            evidence.triangles.as_ptr(),
            triangles.as_ptr()
        ));
        assert_eq!(evidence.summary().learned_points, 3);
    }

    #[test]
    fn provenance_must_partition_the_shared_point_buffer() {
        let points = vec![point(0.0, 0.0, 1.0), point(1.0, 0.0, 1.0)];
        let regions = vec![SurfaceEvidenceRegion {
            origin: EvidenceOrigin::LearnedMultiView,
            reference_frame: Some(0),
            source_frames: vec![1],
            points: EvidenceRange::new(1, 1),
        }];
        let error = ReconstructionEvidenceView::new(
            ReconstructionProviderDescriptor::new(
                "test/learned",
                ReconstructionProviderClass::LearnedMultiView,
                None,
            )
            .expect("provider"),
            EvidenceScale::ProviderLocal,
            vec![camera(0), camera(1)],
            regions,
            &points,
            &[],
        )
        .expect_err("provenance gap must fail closed");
        assert!(error.contains("exact non-overlapping partition"));
    }

    #[test]
    fn generative_completion_can_remain_explicitly_camera_free() {
        let points = vec![point(0.0, 0.0, 1.0)];
        let regions = vec![SurfaceEvidenceRegion {
            origin: EvidenceOrigin::GenerativeCompletion,
            reference_frame: None,
            source_frames: Vec::new(),
            points: EvidenceRange::new(0, 1),
        }];
        let evidence = ReconstructionEvidenceView::new(
            ReconstructionProviderDescriptor::new(
                "test/completion",
                ReconstructionProviderClass::GenerativeCompletion,
                None,
            )
            .expect("provider"),
            EvidenceScale::ProviderLocal,
            Vec::new(),
            regions,
            &points,
            &[],
        )
        .expect("explicit generative completion provenance is valid");
        assert_eq!(evidence.summary().generative_completion_points, 1);
    }

    #[test]
    fn classic_uncalibrated_result_stays_empty_and_borrowed() {
        let mut rgba = vec![120; 48 * 32 * 4];
        for pixel in rgba.as_chunks_mut::<4>().0 {
            pixel[3] = 255;
        }
        let frame = FrameInput {
            width: 48,
            height: 32,
            rgba,
        };
        let request = ReconstructionRequest {
            frames: vec![frame.clone(), frame],
            options: ReconstructionOptions::default(),
        };
        let reconstruction = crate::reconstruct(&request).expect("reconstruction");
        let evidence = ReconstructionEvidenceView::from_classic(&reconstruction)
            .expect("classic evidence");

        assert!(evidence.cameras.is_empty());
        assert!(evidence.regions.is_empty());
        assert!(std::ptr::eq(
            evidence.points.as_ptr(),
            reconstruction.dense_points.as_ptr()
        ));
        assert_eq!(evidence.provider.id, CLASSIC_PROVIDER_ID);
    }
}
