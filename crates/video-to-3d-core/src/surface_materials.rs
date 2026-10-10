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
//! - A triangle is textured only when all three vertices belong to observed
//!   geometry (geometric multi-view or revalidated completion) of the same
//!   reference view, the reference image is available, and the triangle has a
//!   non-degenerate footprint inside that image. Everything else keeps the
//!   fallback and is counted by reason.
//! - Every material names its reference camera, camera authority and pose,
//!   supporting source frames, and provenance classes.
//! - Every material carries an appearance key derived only from inputs that
//!   affect its own appearance (reference camera, cropped reference pixels,
//!   and its own triangles' reference-image footprints). Reprojection or
//!   framing changes of one reference therefore invalidate only that
//!   reference's material.

use crate::scene_project::ContentHash;
use crate::{DenseGridSite, EvidenceCameraAuthority, EvidenceOrigin, ReconstructionEvidenceView};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const SURFACE_MATERIAL_BAKE_SCHEMA_VERSION: u32 = 1;
/// Bump whenever the bake semantics change so recorded appearance keys go stale.
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

/// Appearance key recorded by an earlier bake.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordedAppearance {
    pub reference_frame: usize,
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
    pub reference_frame: usize,
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

/// Why an accepted triangle keeps the untextured fallback.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FallbackReasons {
    /// Vertices belong to different reference views.
    pub mixed_reference: usize,
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
    pub reused: Vec<usize>,
    pub invalidated: Vec<usize>,
    pub added: Vec<usize>,
    pub removed: Vec<usize>,
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

    pub fn recorded_appearance(&self) -> Vec<RecordedAppearance> {
        self.materials
            .iter()
            .map(|material| RecordedAppearance {
                reference_frame: material.reference_frame,
                appearance_key: material.appearance_key.clone(),
            })
            .collect()
    }

    /// Compare against appearance keys recorded by an earlier bake.
    pub fn invalidation_against(&self, previous: &[RecordedAppearance]) -> AppearanceInvalidation {
        let previous: BTreeMap<usize, &ContentHash> = previous
            .iter()
            .map(|record| (record.reference_frame, &record.appearance_key))
            .collect();
        let mut result = AppearanceInvalidation::default();
        for material in &self.materials {
            match previous.get(&material.reference_frame) {
                Some(key) if **key == material.appearance_key => {
                    result.reused.push(material.reference_frame)
                }
                Some(_) => result.invalidated.push(material.reference_frame),
                None => result.added.push(material.reference_frame),
            }
        }
        let current: BTreeSet<usize> = self
            .materials
            .iter()
            .map(|material| material.reference_frame)
            .collect();
        result.removed = previous
            .keys()
            .copied()
            .filter(|frame| !current.contains(frame))
            .collect();
        result
    }

    pub fn diagnostic(&self) -> String {
        let frames = self
            .materials
            .iter()
            .map(|material| material.reference_frame.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let reasons = &self.fallback.reasons;
        let textured = if self.materials.is_empty() {
            "no reference view textured any accepted triangle".to_owned()
        } else {
            format!(
                "{} reference material(s) from frame(s) {frames}",
                self.materials.len()
            )
        };
        format!(
            "Surface material bake v{}: {} of {} accepted triangles textured ({textured}); {} keep the vertex-color fallback ({} mixed reference, {} unobserved reference, {} missing reference image, {} missing grid site, {} degenerate footprint). Geometry is unchanged: {} points and {} triangles exported as accepted.",
            self.schema_version,
            self.textured_triangles(),
            self.triangle_count,
            self.fallback.triangles.len(),
            reasons.mixed_reference,
            reasons.unobserved_reference,
            reasons.missing_reference_image,
            reasons.missing_grid_site,
            reasons.degenerate_footprint,
            self.point_count,
            self.triangle_count,
        )
    }
}

impl AppearanceInvalidation {
    pub fn diagnostic(&self) -> String {
        let list = |frames: &[usize]| {
            if frames.is_empty() {
                "none".to_owned()
            } else {
                frames
                    .iter()
                    .map(usize::to_string)
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

/// Bake reference-view materials for the accepted topology of `evidence`.
///
/// `grid_sites[i]` is the pixel of dense point `i` in its own reference image.
/// Structural contract violations (invalid evidence, malformed or duplicate
/// reference images) are errors; every per-triangle problem falls back.
pub fn bake_surface_materials(
    evidence: &ReconstructionEvidenceView<'_>,
    grid_sites: &[DenseGridSite],
    reference_images: &[ReferenceImage<'_>],
) -> Result<SurfaceMaterialBake, String> {
    bake_surface_materials_with_cancel(evidence, grid_sites, reference_images, || false)
}

/// The same deterministic bake with cooperative cancellation. The callback is
/// polled at bounded intervals during region, triangle, and pixel processing;
/// a canceled operation never returns a partial bake as successful.
pub(crate) fn bake_surface_materials_with_cancel(
    evidence: &ReconstructionEvidenceView<'_>,
    grid_sites: &[DenseGridSite],
    reference_images: &[ReferenceImage<'_>],
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
    let mut images = BTreeMap::new();
    for image in reference_images {
        check_canceled()?;
        let expected = (image.width as usize)
            .checked_mul(image.height as usize)
            .and_then(|pixels| pixels.checked_mul(4));
        if image.width == 0 || image.height == 0 || expected != Some(image.rgba.len()) {
            return Err(format!(
                "reference image for frame {} must be non-empty RGBA of {}x{} pixels",
                image.frame_index, image.width, image.height
            ));
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
            reasons.mixed_reference += 1;
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

    let mut materials = Vec::with_capacity(assigned.len());
    for (reference, (triangles, regions)) in assigned {
        check_canceled()?;
        let camera = evidence
            .cameras
            .iter()
            .find(|camera| camera.frame_index == reference)
            .ok_or_else(|| {
                format!("reference frame {reference} has no accepted evidence camera")
            })?;
        let image = images[&reference];
        let mut source_frames = BTreeSet::from([reference]);
        let mut provenance = Vec::new();
        for region in regions.iter().map(|index| &evidence.regions[*index]) {
            source_frames.extend(region.source_frames.iter().copied());
            if !provenance.contains(&region.origin) {
                provenance.push(region.origin);
            }
        }
        provenance.sort_by_key(|origin| *origin as u8);

        let (mut min_x, mut min_y, mut max_x, mut max_y) = (u32::MAX, u32::MAX, 0u32, 0u32);
        for (ordinal, triangle) in triangles
            .iter()
            .map(|index| &evidence.triangles[*index])
            .enumerate()
        {
            if ordinal % 1024 == 0 {
                check_canceled()?;
            }
            for site in [triangle.a, triangle.b, triangle.c].map(|index| grid_sites[index]) {
                min_x = min_x.min(site.x);
                min_y = min_y.min(site.y);
                max_x = max_x.max(site.x);
                max_y = max_y.max(site.y);
            }
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
        let texture_hash = content_hash(texture_hasher);

        // Grid sites address pixel centers of the reference image.
        let mut corner_uvs = Vec::with_capacity(triangles.len());
        for (ordinal, index) in triangles.iter().enumerate() {
            if ordinal % 1024 == 0 {
                check_canceled()?;
            }
            let triangle = &evidence.triangles[*index];
            corner_uvs.push([triangle.a, triangle.b, triangle.c].map(|point| {
                let site = grid_sites[point];
                [
                    ((site.x - x0) as f32 + 0.5) / crop_width as f32,
                    ((site.y - y0) as f32 + 0.5) / crop_height as f32,
                ]
            }));
        }

        // The key covers exactly what this material's appearance depends on:
        // its camera, the reference framing and crop, the cropped pixels, and
        // its own triangles' footprints. Dense point indices and positions are
        // excluded, so changes in other references never alter it.
        let mut key = Sha256::new();
        key.update(b"video-to-3d/surface-material");
        key.update(BAKE_REVISION.to_le_bytes());
        key.update((reference as u64).to_le_bytes());
        key.update([camera.authority as u8]);
        for value in camera.rotation.iter().chain(&camera.translation) {
            key.update(value.to_bits().to_le_bytes());
        }
        for value in [image.width, image.height, x0, y0] {
            key.update(value.to_le_bytes());
        }
        key.update(texture_hash.as_str().as_bytes());
        for frame in &source_frames {
            key.update((*frame as u64).to_le_bytes());
        }
        for origin in &provenance {
            key.update([*origin as u8]);
        }
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

        materials.push(BakedReferenceMaterial {
            reference_frame: reference,
            camera_authority: camera.authority,
            camera_rotation: camera.rotation,
            camera_translation: camera.translation,
            source_frames: source_frames.into_iter().collect(),
            provenance,
            triangles,
            corner_uvs,
            texture: BakedTexture {
                crop_origin: [x0, y0],
                width: crop_width,
                height: crop_height,
                source_image_width: image.width,
                source_image_height: image.height,
                content_hash: texture_hash,
                rgba,
            },
            appearance_key: content_hash(key),
        });
    }

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
