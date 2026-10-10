//! Portable reference-view surface materials.
//!
//! The bake packages accepted reference-view appearance for already accepted
//! topology. It reads the validated [`ReconstructionEvidenceView`], the
//! Rust-owned reference-grid site of every dense point, and the sampled
//! reference images, and partitions the accepted triangles into one textured
//! material per reference view plus one untextured vertex-color fallback.
//!
//! Invariants:
//! - Geometry is read-only. Every accepted triangle appears exactly once in the
//!   output, either textured or in the fallback; no vertex or triangle is
//!   created, moved, repaired, or dropped.
//! - A triangle is textured by the same-reference rule only when all three
//!   vertices belong to observed geometry (geometric multi-view or revalidated
//!   completion) of the same reference view, the reference image is
//!   available, and the triangle has a non-degenerate footprint inside that
//!   image.
//! - A mixed-reference (seam) triangle with three observed vertices is
//!   textured only by the fail-closed single-camera projection rule of
//!   [`seams`]: one accepted camera must see all three vertices through its
//!   accepted pose, pass a depth test against the accepted geometry, and
//!   reproduce its own accepted grid sites. Cameras are never blended.
//! - Everything else keeps the fallback and is counted by reason; rejected
//!   seam triangles are additionally counted by the furthest rule check.
//! - Every material names its camera, the rule that admitted it, camera
//!   authority and pose, supporting source frames, and provenance classes.
//! - Every material carries an appearance key derived only from inputs that
//!   affect its own appearance (reference camera, cropped reference pixels,
//!   and its own triangles' reference-image footprints). Reprojection or
//!   framing changes of one reference therefore invalidate only that
//!   reference's material.

#[path = "surface_material_seams.rs"]
mod seams;

pub use seams::{SeamProjection, POSE_CONSISTENCY_PIXELS};

use crate::scene_project::ContentHash;
use crate::{
    DenseGridSite, EvidenceCamera, EvidenceCameraAuthority, EvidenceOrigin,
    ReconstructionEvidenceView,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

pub const SURFACE_MATERIAL_BAKE_SCHEMA_VERSION: u32 = 2;
/// Bump whenever the same-reference bake semantics change so recorded
/// appearance keys go stale. Seam materials use their own rule revision.
const BAKE_REVISION: u32 = 1;
/// Texels kept around the used footprint so bilinear sampling at a triangle
/// edge never reads outside the cropped reference image.
const CROP_MARGIN_PIXELS: u32 = 1;

/// One sampled reference image, borrowed from the caller.
#[derive(Clone, Copy, Debug)]
pub struct ReferenceImage<'a> {
    pub frame_index: usize,
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA, `width * height * 4` bytes.
    pub rgba: &'a [u8],
}

/// Frame and dimensions of a reference image whose pixels are loaded only if
/// the bake textures something from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReferenceShape {
    pub frame_index: usize,
    pub width: u32,
    pub height: u32,
    /// Byte length of pixels already in memory, checked up front; `None` when
    /// the pixels are loaded later (and checked then).
    pub available_bytes: Option<usize>,
}

impl ReferenceShape {
    fn rgba_bytes(&self) -> Option<usize> {
        (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
    }

    fn malformed(&self) -> String {
        format!(
            "reference image for frame {} must be non-empty RGBA of {}x{} pixels",
            self.frame_index, self.width, self.height
        )
    }
}

/// The rule that admitted a material's triangles into its texture.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum MaterialRule {
    /// All three vertices are observed geometry of the material's reference
    /// view; corners are addressed by their accepted reference-grid sites.
    #[default]
    SameReferenceGrid,
    /// Mixed-reference seam triangle admitted by one accepted camera that sees
    /// all three vertices; corners are projected through its accepted pose.
    SeamSingleCameraProjection,
}

impl MaterialRule {
    fn label(self) -> &'static str {
        match self {
            Self::SameReferenceGrid => "",
            Self::SeamSingleCameraProjection => "seam ",
        }
    }
}

/// Identifies one appearance artifact: the camera frame and the admitting rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct AppearanceArtifact {
    pub reference_frame: usize,
    pub rule: MaterialRule,
}

/// Appearance key recorded by an earlier bake.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordedAppearance {
    pub reference_frame: usize,
    /// Absent in bakes from before seam texturing: those were same-reference.
    #[serde(default)]
    pub rule: MaterialRule,
    pub appearance_key: ContentHash,
}

/// Cropped reference pixels of one material.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BakedTexture {
    /// Crop origin in the reference image, in pixels.
    pub crop_origin: [u32; 2],
    pub width: u32,
    pub height: u32,
    pub source_image_width: u32,
    pub source_image_height: u32,
    /// Hash of the cropped RGBA pixels (with their dimensions).
    pub content_hash: ContentHash,
    #[serde(skip)]
    pub rgba: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BakedReferenceMaterial {
    /// Frame of the camera whose reference image textures this material.
    pub reference_frame: usize,
    /// The rule that admitted every triangle of this material.
    pub rule: MaterialRule,
    /// Seam materials: the reference views owning the textured seam vertices.
    /// Empty for same-reference materials.
    pub seam_reference_frames: Vec<usize>,
    /// Seam materials: focal length (pixels, principal point at the image
    /// center) used to project the seam vertices.
    pub focal_pixels: Option<f32>,
    pub camera_authority: EvidenceCameraAuthority,
    /// World-to-camera rotation (row-major) of the reference camera.
    pub camera_rotation: [f32; 9],
    pub camera_translation: [f32; 3],
    /// Reference frame plus every supporting source frame of its regions.
    pub source_frames: Vec<usize>,
    /// Provenance classes of the vertices this material textures.
    pub provenance: Vec<EvidenceOrigin>,
    /// Indices into the evidence triangle list, in mesh order.
    pub triangles: Vec<usize>,
    /// Texture coordinates per triangle corner `[a, b, c]`, normalized to the
    /// cropped texture with `(0, 0)` at its top-left corner.
    #[serde(skip)]
    pub corner_uvs: Vec<[[f32; 2]; 3]>,
    pub texture: BakedTexture,
    pub appearance_key: ContentHash,
}

/// Why no accepted camera admitted a mixed-reference seam triangle, counted
/// by the furthest check of the single-camera projection rule any camera
/// reached. The counts sum to [`FallbackReasons::mixed_reference`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeamRejections {
    /// No intrinsics were supplied, or no accepted camera with a reference
    /// image owns observed points to check its projection against.
    pub no_candidate_camera: usize,
    /// Candidate cameras do not reproduce their own accepted grid sites.
    pub pose_inconsistent: usize,
    /// A vertex lies behind every otherwise eligible camera.
    pub behind_camera: usize,
    /// A vertex projects outside every otherwise eligible reference image.
    pub out_of_frame: usize,
    /// Degenerate projected footprint or grazing viewing angle.
    pub grazing_view: usize,
    /// Accepted geometry lies clearly in front of the triangle.
    pub occluded: usize,
    /// Visibility cannot be decided at pixel resolution (depth inside the
    /// tolerance band, or next to an occluding depth edge).
    pub ambiguous: usize,
}

impl SeamRejections {
    fn count(&mut self, reason: seams::SeamRejection) {
        use seams::SeamRejection as R;
        *match reason {
            R::NoCandidateCamera => &mut self.no_candidate_camera,
            R::PoseInconsistent => &mut self.pose_inconsistent,
            R::BehindCamera => &mut self.behind_camera,
            R::OutOfFrame => &mut self.out_of_frame,
            R::GrazingView => &mut self.grazing_view,
            R::Occluded => &mut self.occluded,
            R::Ambiguous => &mut self.ambiguous,
        } += 1;
    }
}

/// Why an accepted triangle keeps the untextured fallback.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FallbackReasons {
    /// Vertices belong to different reference views and no single accepted
    /// camera passed the seam rule (broken down in `seam`).
    pub mixed_reference: usize,
    /// Absent in bakes from before seam texturing.
    #[serde(default)]
    pub seam: SeamRejections,
    /// A vertex has no observed reference view (learned or generative evidence).
    pub unobserved_reference: usize,
    /// The reference image was not supplied.
    pub missing_reference_image: usize,
    /// The reference-grid mapping is missing or outside the reference image.
    pub missing_grid_site: usize,
    /// The triangle covers no area in the reference image.
    pub degenerate_footprint: usize,
}

impl FallbackReasons {
    pub fn total(&self) -> usize {
        self.mixed_reference
            + self.unobserved_reference
            + self.missing_reference_image
            + self.missing_grid_site
            + self.degenerate_footprint
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FallbackSurface {
    /// Indices into the evidence triangle list, in mesh order.
    pub triangles: Vec<usize>,
    pub reasons: FallbackReasons,
}

/// Which recorded appearance artifacts a bake keeps, replaces, adds or drops.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct AppearanceInvalidation {
    pub reused: Vec<AppearanceArtifact>,
    pub invalidated: Vec<AppearanceArtifact>,
    pub added: Vec<AppearanceArtifact>,
    pub removed: Vec<AppearanceArtifact>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SurfaceMaterialBake {
    pub schema_version: u32,
    pub point_count: usize,
    pub triangle_count: usize,
    pub materials: Vec<BakedReferenceMaterial>,
    pub fallback: FallbackSurface,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ownership {
    Unobserved,
    Observed { reference: usize, region: usize },
}

impl SurfaceMaterialBake {
    pub fn textured_triangles(&self) -> usize {
        self.materials
            .iter()
            .map(|material| material.triangles.len())
            .sum()
    }

    /// Triangles textured by the seam rule.
    pub fn seam_triangles(&self) -> usize {
        self.materials
            .iter()
            .filter(|material| material.rule == MaterialRule::SeamSingleCameraProjection)
            .map(|material| material.triangles.len())
            .sum()
    }

    pub fn recorded_appearance(&self) -> Vec<RecordedAppearance> {
        self.materials
            .iter()
            .map(|material| RecordedAppearance {
                reference_frame: material.reference_frame,
                rule: material.rule,
                appearance_key: material.appearance_key.clone(),
            })
            .collect()
    }

    /// Compare against appearance keys recorded by an earlier bake.
    pub fn invalidation_against(&self, previous: &[RecordedAppearance]) -> AppearanceInvalidation {
        let previous: BTreeMap<AppearanceArtifact, &ContentHash> = previous
            .iter()
            .map(|record| {
                (
                    AppearanceArtifact {
                        reference_frame: record.reference_frame,
                        rule: record.rule,
                    },
                    &record.appearance_key,
                )
            })
            .collect();
        let mut result = AppearanceInvalidation::default();
        let mut current = BTreeSet::new();
        for material in &self.materials {
            let artifact = material.artifact();
            current.insert(artifact);
            match previous.get(&artifact) {
                Some(key) if **key == material.appearance_key => result.reused.push(artifact),
                Some(_) => result.invalidated.push(artifact),
                None => result.added.push(artifact),
            }
        }
        result.removed = previous
            .keys()
            .copied()
            .filter(|artifact| !current.contains(artifact))
            .collect();
        result
    }

    pub fn diagnostic(&self) -> String {
        let frames = |rule: MaterialRule| {
            self.materials
                .iter()
                .filter(|material| material.rule == rule)
                .map(|material| material.reference_frame.to_string())
                .collect::<Vec<_>>()
        };
        let references = frames(MaterialRule::SameReferenceGrid);
        let seam_cameras = frames(MaterialRule::SeamSingleCameraProjection);
        let reasons = &self.fallback.reasons;
        let seam = &reasons.seam;
        let textured = if self.materials.is_empty() {
            "no reference view textured any accepted triangle".to_owned()
        } else {
            let mut textured = format!(
                "{} reference material(s) from frame(s) {}",
                references.len(),
                if references.is_empty() {
                    "none".to_owned()
                } else {
                    references.join(", ")
                }
            );
            if !seam_cameras.is_empty() {
                textured.push_str(&format!(
                    ", {} seam triangle(s) in {} seam material(s) from camera frame(s) {} by the single-camera projection rule",
                    self.seam_triangles(),
                    seam_cameras.len(),
                    seam_cameras.join(", ")
                ));
            }
            textured
        };
        format!(
            "Surface material bake v{}: {} of {} accepted triangles textured ({textured}); {} keep the vertex-color fallback ({} mixed reference [seam rule: {} no candidate camera, {} pose inconsistent, {} behind camera, {} out of frame, {} grazing view, {} occluded, {} ambiguous], {} unobserved reference, {} missing reference image, {} missing grid site, {} degenerate footprint). Geometry is unchanged: {} points and {} triangles exported as accepted.",
            self.schema_version,
            self.textured_triangles(),
            self.triangle_count,
            self.fallback.triangles.len(),
            reasons.mixed_reference,
            seam.no_candidate_camera,
            seam.pose_inconsistent,
            seam.behind_camera,
            seam.out_of_frame,
            seam.grazing_view,
            seam.occluded,
            seam.ambiguous,
            reasons.unobserved_reference,
            reasons.missing_reference_image,
            reasons.missing_grid_site,
            reasons.degenerate_footprint,
            self.point_count,
            self.triangle_count,
        )
    }
}

impl BakedReferenceMaterial {
    pub fn artifact(&self) -> AppearanceArtifact {
        AppearanceArtifact {
            reference_frame: self.reference_frame,
            rule: self.rule,
        }
    }
}

impl AppearanceInvalidation {
    pub fn diagnostic(&self) -> String {
        let list = |artifacts: &[AppearanceArtifact]| {
            if artifacts.is_empty() {
                "none".to_owned()
            } else {
                artifacts
                    .iter()
                    .map(|artifact| {
                        format!("{}{}", artifact.rule.label(), artifact.reference_frame)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };
        format!(
            "Appearance artifacts versus the previous bake: reused {}, invalidated {}, added {}, removed {}.",
            list(&self.reused),
            list(&self.invalidated),
            list(&self.added),
            list(&self.removed),
        )
    }
}

/// Bake reference-view materials for the accepted topology of `evidence`
/// without camera intrinsics: every mixed-reference seam triangle keeps the
/// fallback (counted as `no_candidate_camera`).
pub fn bake_surface_materials(
    evidence: &ReconstructionEvidenceView<'_>,
    grid_sites: &[DenseGridSite],
    reference_images: &[ReferenceImage<'_>],
) -> Result<SurfaceMaterialBake, String> {
    bake_surface_materials_with_seams(evidence, grid_sites, reference_images, None)
}

/// Bake reference-view materials for the accepted topology of `evidence`.
///
/// `grid_sites[i]` is the pixel of dense point `i` in its own reference image.
/// `seam_projection` carries the pinhole intrinsics of the accepted cameras;
/// without it no seam triangle is textured. Structural contract violations
/// (invalid evidence, malformed or duplicate reference images, invalid
/// intrinsics) are errors; every per-triangle problem falls back.
pub fn bake_surface_materials_with_seams(
    evidence: &ReconstructionEvidenceView<'_>,
    grid_sites: &[DenseGridSite],
    reference_images: &[ReferenceImage<'_>],
    seam_projection: Option<SeamProjection>,
) -> Result<SurfaceMaterialBake, String> {
    bake_with_cancel(
        evidence,
        grid_sites,
        reference_images,
        seam_projection,
        || false,
    )
}

/// The deterministic bake without seam intrinsics, with cooperative
/// cancellation. The callback is polled at bounded intervals during region,
/// triangle, seam, and pixel processing; a canceled operation never returns a
/// partial bake as successful.
#[cfg(test)]
pub(crate) fn bake_surface_materials_with_cancel(
    evidence: &ReconstructionEvidenceView<'_>,
    grid_sites: &[DenseGridSite],
    reference_images: &[ReferenceImage<'_>],
    is_canceled: impl FnMut() -> bool,
) -> Result<SurfaceMaterialBake, String> {
    bake_with_cancel(evidence, grid_sites, reference_images, None, is_canceled)
}

fn bake_with_cancel(
    evidence: &ReconstructionEvidenceView<'_>,
    grid_sites: &[DenseGridSite],
    reference_images: &[ReferenceImage<'_>],
    seam_projection: Option<SeamProjection>,
    is_canceled: impl FnMut() -> bool,
) -> Result<SurfaceMaterialBake, String> {
    let shapes: Vec<ReferenceShape> = reference_images
        .iter()
        .map(|image| ReferenceShape {
            frame_index: image.frame_index,
            width: image.width,
            height: image.height,
            available_bytes: Some(image.rgba.len()),
        })
        .collect();
    bake_incremental(
        evidence,
        grid_sites,
        &shapes,
        seam_projection,
        |frame| {
            reference_images
                .iter()
                .find(|image| image.frame_index == frame)
                .map(|image| Cow::Borrowed(image.rgba))
                .ok_or_else(|| format!("reference image for frame {frame} was not supplied"))
        },
        is_canceled,
    )
}

/// The bake with reference pixels loaded on demand.
///
/// Triangles are first assigned from `shapes` (frame and dimensions) alone;
/// `load` is then called once per participating reference (a frame that
/// textures at least one same-reference or seam triangle), in ascending frame
/// order, and its pixels are dropped once that frame's materials are cropped.
/// Peak reference memory is therefore one full image plus the crops, and a
/// reference that textures nothing is never loaded. The output is identical
/// to a bake over fully loaded images.
pub(crate) fn bake_incremental<'p>(
    evidence: &ReconstructionEvidenceView<'_>,
    grid_sites: &[DenseGridSite],
    shapes: &[ReferenceShape],
    seam_projection: Option<SeamProjection>,
    mut load: impl FnMut(usize) -> Result<Cow<'p, [u8]>, String>,
    mut is_canceled: impl FnMut() -> bool,
) -> Result<SurfaceMaterialBake, String> {
    let mut check_canceled = || {
        if is_canceled() {
            Err("texture bake was canceled".to_owned())
        } else {
            Ok(())
        }
    };
    check_canceled()?;
    evidence.validate_with_cancel(&mut check_canceled)?;
    check_canceled()?;
    if let Some(projection) = seam_projection {
        if !projection.focal_pixels.is_finite() || projection.focal_pixels <= 0.0 {
            return Err("seam projection focal length must be finite and positive".into());
        }
    }
    let mut images = BTreeMap::new();
    for image in shapes {
        check_canceled()?;
        let expected = image.rgba_bytes();
        if image.width == 0
            || image.height == 0
            || expected.is_none()
            || image
                .available_bytes
                .is_some_and(|bytes| expected != Some(bytes))
        {
            return Err(image.malformed());
        }
        if images.insert(image.frame_index, *image).is_some() {
            return Err(format!(
                "reference image for frame {} was supplied twice",
                image.frame_index
            ));
        }
    }

    let points = evidence.points;
    let mut ownership = vec![Ownership::Unobserved; points.len()];
    for (region_index, region) in evidence.regions.iter().enumerate() {
        if region_index % 64 == 0 {
            check_canceled()?;
        }
        let observed = matches!(
            region.origin,
            EvidenceOrigin::GeometricMultiView | EvidenceOrigin::RevalidatedCompletion
        );
        let (true, Some(reference)) = (observed, region.reference_frame) else {
            continue;
        };
        let end = region.points.start + region.points.count;
        for (index, owner) in ownership[region.points.start..end].iter_mut().enumerate() {
            if index % 1024 == 0 {
                check_canceled()?;
            }
            *owner = Ownership::Observed {
                reference,
                region: region_index,
            };
        }
    }
    let grid_mapped = grid_sites.len() == points.len();

    let mut fallback = FallbackSurface::default();
    // reference frame -> (triangle indices, contributing regions)
    let mut assigned: BTreeMap<usize, (Vec<usize>, BTreeSet<usize>)> = BTreeMap::new();
    let mut seam_candidates = Vec::new();
    for (triangle_index, triangle) in evidence.triangles.iter().enumerate() {
        if triangle_index % 1024 == 0 {
            check_canceled()?;
        }
        let corners = [triangle.a, triangle.b, triangle.c].map(|index| ownership[index]);
        let mut references = Vec::with_capacity(3);
        let mut regions = Vec::with_capacity(3);
        for corner in corners {
            match corner {
                Ownership::Observed { reference, region } => {
                    references.push(reference);
                    regions.push(region);
                }
                Ownership::Unobserved => {}
            }
        }
        let reasons = &mut fallback.reasons;
        if references.len() < 3 {
            reasons.unobserved_reference += 1;
        } else if references
            .iter()
            .any(|reference| *reference != references[0])
        {
            // Decided by the seam rule below.
            seam_candidates.push(triangle_index);
            continue;
        } else if !images.contains_key(&references[0]) {
            reasons.missing_reference_image += 1;
        } else if !grid_mapped
            || [triangle.a, triangle.b, triangle.c].iter().any(|index| {
                let site = grid_sites[*index];
                let image = &images[&references[0]];
                site.x >= image.width || site.y >= image.height
            })
        {
            reasons.missing_grid_site += 1;
        } else if footprint_double_area(
            grid_sites[triangle.a],
            grid_sites[triangle.b],
            grid_sites[triangle.c],
        ) == 0
        {
            reasons.degenerate_footprint += 1;
        } else {
            let entry = assigned.entry(references[0]).or_default();
            entry.0.push(triangle_index);
            entry.1.extend(regions);
            continue;
        }
        fallback.triangles.push(triangle_index);
    }

    // camera frame -> admitted seam triangles with their corner pixels
    let mut seam_assigned: BTreeMap<usize, Vec<(usize, CornerPixels)>> = BTreeMap::new();
    let seam_results = seams::evaluate_seams(
        evidence,
        &ownership,
        grid_mapped.then_some(grid_sites),
        &images,
        seam_projection,
        &seam_candidates,
        &mut check_canceled,
    )?;
    for (triangle_index, result) in seam_candidates.iter().zip(seam_results) {
        match result {
            Ok(admission) => seam_assigned
                .entry(admission.camera_frame)
                .or_default()
                .push((*triangle_index, admission.corner_pixels)),
            Err(reason) => {
                fallback.reasons.mixed_reference += 1;
                fallback.reasons.seam.count(reason);
                fallback.triangles.push(*triangle_index);
            }
        }
    }
    fallback.triangles.sort_unstable();

    let camera_for = |frame: usize| {
        evidence
            .cameras
            .iter()
            .find(|camera| camera.frame_index == frame)
            .ok_or_else(|| format!("reference frame {frame} has no accepted evidence camera"))
    };
    // Missing cameras fail before any reference image is loaded, in the
    // order the materials are emitted.
    for frame in assigned.keys().chain(seam_assigned.keys()) {
        camera_for(*frame)?;
    }
    // Only references that texture something are loaded, one at a time; a
    // camera used by both rules is loaded once. Same-reference materials still
    // precede seam materials in the output.
    let participating: BTreeSet<usize> = assigned
        .keys()
        .chain(seam_assigned.keys())
        .copied()
        .collect();
    let (mut assigned, mut seam_assigned) = (assigned, seam_assigned);
    let mut same_reference = Vec::with_capacity(assigned.len());
    let mut seam_materials = Vec::with_capacity(seam_assigned.len());
    for frame in participating {
        check_canceled()?;
        let shape = images[&frame];
        let pixels = load(frame)?;
        if Some(pixels.len()) != shape.rgba_bytes() {
            return Err(shape.malformed());
        }
        let image = ReferenceImage {
            frame_index: frame,
            width: shape.width,
            height: shape.height,
            rgba: &pixels,
        };
        if let Some((triangles, regions)) = assigned.remove(&frame) {
            let reference = frame;
            let camera = camera_for(reference)?;
            let (source_frames, provenance) = region_provenance(evidence, reference, &regions);

            let mut corner_pixels = Vec::with_capacity(triangles.len());
            for (ordinal, index) in triangles.iter().enumerate() {
                if ordinal % 1024 == 0 {
                    check_canceled()?;
                }
                let triangle = &evidence.triangles[*index];
                corner_pixels.push([triangle.a, triangle.b, triangle.c].map(|point| {
                    let site = grid_sites[point];
                    [site.x as f64, site.y as f64]
                }));
            }
            let (texture, corner_uvs) = crop_texture(&image, &corner_pixels, &mut check_canceled)?;

            // The key covers exactly what this material's appearance depends on:
            // its camera, the reference framing and crop, the cropped pixels, and
            // its own triangles' footprints. Dense point indices and positions are
            // excluded, so changes in other references never alter it.
            let mut key = Sha256::new();
            key.update(b"video-to-3d/surface-material");
            key.update(BAKE_REVISION.to_le_bytes());
            hash_camera_and_texture(
                &mut key,
                reference,
                camera,
                &texture,
                &source_frames,
                &provenance,
            );
            for (ordinal, triangle) in triangles
                .iter()
                .map(|index| &evidence.triangles[*index])
                .enumerate()
            {
                if ordinal % 1024 == 0 {
                    check_canceled()?;
                }
                for site in [triangle.a, triangle.b, triangle.c].map(|index| grid_sites[index]) {
                    key.update(site.x.to_le_bytes());
                    key.update(site.y.to_le_bytes());
                }
            }

            same_reference.push(BakedReferenceMaterial {
                reference_frame: reference,
                rule: MaterialRule::SameReferenceGrid,
                seam_reference_frames: Vec::new(),
                focal_pixels: None,
                camera_authority: camera.authority,
                camera_rotation: camera.rotation,
                camera_translation: camera.translation,
                source_frames,
                provenance,
                triangles,
                corner_uvs,
                texture,
                appearance_key: content_hash(key),
            });
        }
        if let Some(admitted) = seam_assigned.remove(&frame) {
            let camera_frame = frame;
            let camera = camera_for(camera_frame)?;
            let focal = seam_projection
                .expect("seam triangles are admitted only with intrinsics")
                .focal_pixels;
            let mut regions = BTreeSet::new();
            let mut seam_reference_frames = BTreeSet::new();
            let mut triangles = Vec::with_capacity(admitted.len());
            let mut corner_pixels = Vec::with_capacity(admitted.len());
            for (ordinal, (index, pixels)) in admitted.iter().enumerate() {
                if ordinal % 1024 == 0 {
                    check_canceled()?;
                }
                triangles.push(*index);
                corner_pixels.push(*pixels);
                let triangle = &evidence.triangles[*index];
                for point in [triangle.a, triangle.b, triangle.c] {
                    if let Ownership::Observed { reference, region } = ownership[point] {
                        regions.insert(region);
                        seam_reference_frames.insert(reference);
                    }
                }
            }
            let (source_frames, provenance) = region_provenance(evidence, camera_frame, &regions);
            let (texture, corner_uvs) = crop_texture(&image, &corner_pixels, &mut check_canceled)?;

            // Seam appearance depends on the camera and intrinsics, the cropped
            // pixels, the seam vertices' accepted positions (they are projected),
            // and the references owning them. Occlusion decisions enter through
            // the admitted triangle set.
            let mut key = Sha256::new();
            key.update(b"video-to-3d/surface-material/seam-single-camera-projection");
            key.update(seams::SEAM_RULE_REVISION.to_le_bytes());
            key.update(focal.to_bits().to_le_bytes());
            hash_camera_and_texture(
                &mut key,
                camera_frame,
                camera,
                &texture,
                &source_frames,
                &provenance,
            );
            for frame in &seam_reference_frames {
                key.update((*frame as u64).to_le_bytes());
            }
            for (ordinal, triangle) in triangles
                .iter()
                .map(|index| &evidence.triangles[*index])
                .enumerate()
            {
                if ordinal % 1024 == 0 {
                    check_canceled()?;
                }
                for point in [triangle.a, triangle.b, triangle.c].map(|index| &points[index]) {
                    for value in [point.x, point.y, point.z] {
                        key.update(value.to_bits().to_le_bytes());
                    }
                }
            }

            seam_materials.push(BakedReferenceMaterial {
                reference_frame: camera_frame,
                rule: MaterialRule::SeamSingleCameraProjection,
                seam_reference_frames: seam_reference_frames.into_iter().collect(),
                focal_pixels: Some(focal),
                camera_authority: camera.authority,
                camera_rotation: camera.rotation,
                camera_translation: camera.translation,
                source_frames,
                provenance,
                triangles,
                corner_uvs,
                texture,
                appearance_key: content_hash(key),
            });
        }
    }
    let mut materials = same_reference;
    materials.extend(seam_materials);

    // In particular, an all-fallback bake must not report success if canceled.
    check_canceled()?;
    Ok(SurfaceMaterialBake {
        schema_version: SURFACE_MATERIAL_BAKE_SCHEMA_VERSION,
        point_count: points.len(),
        triangle_count: evidence.triangles.len(),
        materials,
        fallback,
    })
}

/// Supporting source frames (plus `camera_frame`) and sorted provenance
/// classes of the contributing regions.
fn region_provenance(
    evidence: &ReconstructionEvidenceView<'_>,
    camera_frame: usize,
    regions: &BTreeSet<usize>,
) -> (Vec<usize>, Vec<EvidenceOrigin>) {
    let mut source_frames = BTreeSet::from([camera_frame]);
    let mut provenance = Vec::new();
    for region in regions.iter().map(|index| &evidence.regions[*index]) {
        source_frames.extend(region.source_frames.iter().copied());
        if let Some(reference) = region.reference_frame {
            source_frames.insert(reference);
        }
        if !provenance.contains(&region.origin) {
            provenance.push(region.origin);
        }
    }
    provenance.sort_by_key(|origin| *origin as u8);
    (source_frames.into_iter().collect(), provenance)
}

/// Normalized texture coordinates of triangle corners `[a, b, c]`.
type CornerUvs = [[f32; 2]; 3];

/// Crop the used footprint (plus margin) out of `image` and map the corner
/// pixels (pixel `x` centered at coordinate `x`) to normalized crop UVs.
fn crop_texture(
    image: &ReferenceImage<'_>,
    corner_pixels: &[CornerPixels],
    check_canceled: &mut impl FnMut() -> Result<(), String>,
) -> Result<(BakedTexture, Vec<CornerUvs>), String> {
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (u32::MAX, u32::MAX, 0u32, 0u32);
    // Projected corners carry float noise; the crop margin already covers the
    // neighbouring texel a bilinear sample reads, so snap within a tolerance.
    const SNAP: f64 = 1.0e-3;
    for (ordinal, [x, y]) in corner_pixels.iter().flatten().enumerate() {
        if ordinal % (3 * 1024) == 0 {
            check_canceled()?;
        }
        min_x = min_x.min((x + SNAP).floor().max(0.0) as u32);
        min_y = min_y.min((y + SNAP).floor().max(0.0) as u32);
        max_x = max_x.max((x - SNAP).ceil().max(0.0) as u32);
        max_y = max_y.max((y - SNAP).ceil().max(0.0) as u32);
    }
    let x0 = min_x.saturating_sub(CROP_MARGIN_PIXELS);
    let y0 = min_y.saturating_sub(CROP_MARGIN_PIXELS);
    let x1 = (max_x + CROP_MARGIN_PIXELS).min(image.width - 1);
    let y1 = (max_y + CROP_MARGIN_PIXELS).min(image.height - 1);
    let (crop_width, crop_height) = (x1 - x0 + 1, y1 - y0 + 1);
    let mut rgba = Vec::with_capacity(crop_width as usize * crop_height as usize * 4);
    for y in y0..=y1 {
        if (y - y0) % 64 == 0 {
            check_canceled()?;
        }
        let row = (y as usize * image.width as usize + x0 as usize) * 4;
        rgba.extend_from_slice(&image.rgba[row..row + crop_width as usize * 4]);
    }
    let mut texture_hasher = Sha256::new();
    texture_hasher.update(crop_width.to_le_bytes());
    texture_hasher.update(crop_height.to_le_bytes());
    for chunk in rgba.chunks(1024 * 1024) {
        check_canceled()?;
        texture_hasher.update(chunk);
    }
    let mut corner_uvs = Vec::with_capacity(corner_pixels.len());
    for (ordinal, corners) in corner_pixels.iter().enumerate() {
        if ordinal % 1024 == 0 {
            check_canceled()?;
        }
        corner_uvs.push(corners.map(|[x, y]| {
            [
                ((x - x0 as f64 + 0.5) / crop_width as f64) as f32,
                ((y - y0 as f64 + 0.5) / crop_height as f64) as f32,
            ]
        }));
    }
    let texture = BakedTexture {
        crop_origin: [x0, y0],
        width: crop_width,
        height: crop_height,
        source_image_width: image.width,
        source_image_height: image.height,
        content_hash: content_hash(texture_hasher),
        rgba,
    };
    Ok((texture, corner_uvs))
}

fn hash_camera_and_texture(
    key: &mut Sha256,
    frame: usize,
    camera: &EvidenceCamera,
    texture: &BakedTexture,
    source_frames: &[usize],
    provenance: &[EvidenceOrigin],
) {
    key.update((frame as u64).to_le_bytes());
    key.update([camera.authority as u8]);
    for value in camera.rotation.iter().chain(&camera.translation) {
        key.update(value.to_bits().to_le_bytes());
    }
    for value in [
        texture.source_image_width,
        texture.source_image_height,
        texture.crop_origin[0],
        texture.crop_origin[1],
    ] {
        key.update(value.to_le_bytes());
    }
    key.update(texture.content_hash.as_str().as_bytes());
    for frame in source_frames {
        key.update((*frame as u64).to_le_bytes());
    }
    for origin in provenance {
        key.update([*origin as u8]);
    }
}

/// Continuous pixel coordinates of triangle corners `[a, b, c]`.
type CornerPixels = [[f64; 2]; 3];

fn footprint_double_area(a: DenseGridSite, b: DenseGridSite, c: DenseGridSite) -> i64 {
    let (ax, ay) = (a.x as i64, a.y as i64);
    (b.x as i64 - ax) * (c.y as i64 - ay) - (b.y as i64 - ay) * (c.x as i64 - ax)
}

fn content_hash(hasher: Sha256) -> ContentHash {
    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    ContentHash::new(format!("sha256:{hex}")).expect("sha256 digests are valid content hashes")
}

#[cfg(test)]
#[path = "surface_materials_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "surface_material_seams_tests.rs"]
mod seam_tests;
