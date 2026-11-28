//! Verify external geometric depth against another registered view, then pass the retained
//! reference grids to the shared topology, provenance, material and GLB exporters.
use crate::multi_view::RegisteredCamera;
use crate::registered_sparse::RegisteredSparseInput;
use crate::surface_materials::{bake_surface_materials_with_seams, ReferenceImage, SeamProjection};
use crate::textured_glb::{encode_textured_glb_with_camera_provider, CameraProviderMetadata};
use crate::*;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DepthGrids {
    version: u32,
    stride: usize,
    /// A multiple of the source grid spacing. Omit to mesh at the supplied spacing.
    reference_stride: Option<usize>,
    maps: Vec<Grid>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Grid {
    frame: usize,
    sources: Vec<usize>,
    depth: Vec<f32>,
}

#[derive(Serialize)]
pub struct SurfaceReport {
    pub registered_views: usize,
    pub reference_views: usize,
    pub verified_points: usize,
    pub triangles: usize,
    pub textured_triangles: usize,
    pub object_surfaces: usize,
    pub object_triangles: usize,
    /// Where accepted geometry was lost on its way from depth samples to the exported surface.
    pub diagnostics: SurfaceDiagnostics,
}

/// Stage funnel for one export. Counts are deterministic for identical inputs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SurfaceDiagnostics {
    /// Reference-grid depth samples checked against each supporting source view.
    pub verification: VerificationFunnel,
    pub meshing: MeshingFunnel,
    pub fusion: ViewFusionFunnel,
    /// Why exported triangles keep the untextured fallback material.
    pub untextured: crate::surface_materials::FallbackReasons,
    pub topology: SurfaceTopology,
    /// The verification check that rejected the most reference samples.
    pub dominant_rejection: Option<&'static str>,
}

/// One reference sample is counted once per source view that attempted to verify it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct VerificationFunnel {
    pub reference_samples: usize,
    pub missing_reference_depth: usize,
    pub insufficient_parallax: usize,
    pub behind_source: usize,
    pub outside_source: usize,
    pub missing_source_depth: usize,
    pub depth_disagreement: usize,
    pub color_disagreement: usize,
    pub accepted: usize,
}

impl VerificationFunnel {
    fn add(&mut self, other: &Self) {
        self.reference_samples += other.reference_samples;
        self.missing_reference_depth += other.missing_reference_depth;
        self.insufficient_parallax += other.insufficient_parallax;
        self.behind_source += other.behind_source;
        self.outside_source += other.outside_source;
        self.missing_source_depth += other.missing_source_depth;
        self.depth_disagreement += other.depth_disagreement;
        self.color_disagreement += other.color_disagreement;
        self.accepted += other.accepted;
    }

    fn dominant_rejection(&self) -> Option<&'static str> {
        [
            ("missing_reference_depth", self.missing_reference_depth),
            ("insufficient_parallax", self.insufficient_parallax),
            ("behind_source", self.behind_source),
            ("outside_source", self.outside_source),
            ("missing_source_depth", self.missing_source_depth),
            ("depth_disagreement", self.depth_disagreement),
            ("color_disagreement", self.color_disagreement),
        ]
        .into_iter()
        .filter(|(_, count)| *count > 0)
        .max_by_key(|(name, count)| (*count, std::cmp::Reverse(*name)))
        .map(|(name, _)| name)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct MeshingFunnel {
    /// Depth maps available for validation, and how many were meshed as references.
    pub depth_views: usize,
    pub meshed_reference_views: usize,
    pub candidate_cells: usize,
    pub rejected_discontinuities: usize,
    pub rejected_degenerate: usize,
    /// Reference samples at least one source verified, and those more than one verified
    /// (each meshed once).
    pub verified_samples: usize,
    pub multiply_verified_samples: usize,
    /// Samples whose multi-view consensus depth differs from the reference depth alone by
    /// more than 1%: a high share means noisy per-view depth.
    pub consensus_moved_over_one_percent: usize,
    pub exported_triangles: usize,
}

/// How independently meshed reference views were joined (see `surface_fusion.rs` and
/// `surface_merge.rs`): only mutually nearest, normal-aligned seam vertices are fused, and
/// only views linked that way may own each other's duplicated overlap.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ViewFusionFunnel {
    pub reference_views: usize,
    pub fused_seam_pairs: usize,
    pub cross_reference_triangles: usize,
    pub removed_duplicate_triangles: usize,
    pub stitched_seam_triangles: usize,
    /// Clip vertices on owner edges, with source-triangle lineage (derived, not observed).
    pub derived_vertices: usize,
    /// Connected components of the reference-view graph after the merge.
    pub patch_components: usize,
    /// Triangles dropped because no clip split was possible without a T-junction.
    pub dropped_unconformable_triangles: usize,
    /// Partially overlapping triangles kept to preserve supported geometry.
    pub remaining_overlap_triangles: usize,
    /// Removed seam triangles that another view owned only in part.
    pub trimmed_seam_triangles: usize,
}

/// Connectivity of the exported surface. Separate reference views do not share vertices, so
/// each meshed reference view is at least one island even where two views observed the same wall.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SurfaceTopology {
    pub islands: usize,
    /// Islands with fewer than [`SMALL_ISLAND_TRIANGLES`] triangles.
    pub small_islands: usize,
    pub small_island_triangles: usize,
    pub largest_island_triangles: usize,
    pub boundary_edges: usize,
}

pub const SMALL_ISLAND_TRIANGLES: usize = 32;

fn surface_topology(vertex_count: usize, triangles: &[MeshTriangle]) -> SurfaceTopology {
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut parent: Vec<usize> = (0..vertex_count).collect();
    let mut edges = BTreeMap::<(usize, usize), usize>::new();
    for t in triangles {
        for (a, b) in [(t.a, t.b), (t.b, t.c), (t.c, t.a)] {
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    let mut sizes = BTreeMap::<usize, usize>::new();
    for t in triangles {
        *sizes.entry(root(&mut parent, t.a)).or_default() += 1;
    }
    let small = || sizes.values().filter(|&&n| n < SMALL_ISLAND_TRIANGLES);
    SurfaceTopology {
        islands: sizes.len(),
        small_islands: small().count(),
        small_island_triangles: small().sum(),
        largest_island_triangles: sizes.values().copied().max().unwrap_or(0),
        boundary_edges: edges.values().filter(|&&n| n == 1).count(),
    }
}

struct View<'a> {
    grid: &'a Grid,
    camera: &'a RegisteredCamera,
    image: &'a FrameInput,
}

impl View<'_> {
    fn rgb(&self, x: usize, y: usize) -> [u8; 3] {
        let offset = (y * self.image.width as usize + x) * 4;
        [
            self.image.rgba[offset],
            self.image.rgba[offset + 1],
            self.image.rgba[offset + 2],
        ]
    }
}

fn verify_patch(
    reference: &View<'_>,
    source: &View<'_>,
    stride: usize,
    reference_stride: usize,
    focal: f64,
    funnel: &mut VerificationFunnel,
) -> Vec<Verified> {
    let width = reference.image.width as usize;
    let height = reference.image.height as usize;
    let cols = width.div_ceil(stride);
    let rows = height.div_ceil(stride);
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let mut points = Vec::new();
    let source_center = source.camera.camera_center();
    let ref_center = reference.camera.camera_center();
    let min_parallax_cos = 0.5_f64.to_radians().cos();
    let reference_center_depth =
        (source.camera.rotation * ref_center + source.camera.translation).z;
    for (i, &z) in reference.grid.depth.iter().enumerate() {
        let x = (i % cols) * stride;
        let y = (i / cols) * stride;
        if !x.is_multiple_of(reference_stride) || !y.is_multiple_of(reference_stride) {
            continue;
        }
        funnel.reference_samples += 1;
        if z <= 0. {
            funnel.missing_reference_depth += 1;
            continue;
        }
        let ray = Vector3::new(
            (x as f64 + 0.5 - cx) / focal,
            (y as f64 + 0.5 - cy) / focal,
            1.,
        );
        let world = reference.camera.rotation.transpose()
            * (ray * f64::from(z) - reference.camera.translation);
        let direction = world - ref_center;
        let from_source = world - source_center;
        if direction.dot(&from_source) >= min_parallax_cos * direction.norm() * from_source.norm() {
            funnel.insufficient_parallax += 1;
            continue;
        }
        let q = source.camera.rotation * world + source.camera.translation;
        if q.z <= 0. {
            funnel.behind_source += 1;
            continue;
        }
        let u = focal * q.x / q.z + cx - 0.5;
        let v = focal * q.y / q.z + cy - 0.5;
        let col = (u / stride as f64).round() as usize;
        let row = (v / stride as f64).round() as usize;
        if !(0.0..width as f64).contains(&u)
            || !(0.0..height as f64).contains(&v)
            || col >= cols
            || row >= rows
        {
            funnel.outside_source += 1;
            continue;
        }
        let other = f64::from(source.grid.depth[row * cols + col]);
        // Require an independent measurement in the source map. A foreground/background
        // transition, an unobserved pixel, or an inconsistent depth cannot create topology.
        if other <= 0. {
            funnel.missing_source_depth += 1;
            continue;
        }
        if (other - q.z).abs() > 0.02 * q.z {
            funnel.depth_disagreement += 1;
            continue;
        }
        let [r, g, b] = reference.rgb(x, y);
        let rgb = source.rgb(
            (u.round() as usize).min(width - 1),
            (v.round() as usize).min(height - 1),
        );
        let color_error = u32::from(r.abs_diff(rgb[0]))
            + u32::from(g.abs_diff(rgb[1]))
            + u32::from(b.abs_diff(rgb[2]));
        if color_error > 105 {
            funnel.color_disagreement += 1;
            continue;
        }
        funnel.accepted += 1;
        points.push(Verified {
            point: Point3 {
                x: world.x as f32,
                y: world.y as f32,
                z: world.z as f32,
                r,
                g,
                b,
                confidence: (1. - (other - q.z).abs() / q.z) as f32,
            },
            site: DenseGridSite {
                x: x as u32,
                y: y as u32,
            },
            // The reference-ray point whose depth from the source equals the source's own
            // measurement: depth seen from the source is affine along the reference ray.
            source_depth_ratio: (other - reference_center_depth) / (q.z - reference_center_depth),
        });
    }
    points
}

/// One reference sample a source camera verified.
struct Verified {
    /// Position from the reference depth alone.
    point: Point3,
    site: DenseGridSite,
    /// The source's own depth measurement over the reference sample's depth seen from the
    /// source (within the 2% agreement band). Scaling the reference ray by it places the
    /// sample where the source measured the surface, to first order.
    source_depth_ratio: f64,
}

struct VerifiedSample {
    point: Point3,
    site: DenseGridSite,
    /// Every source camera that independently verified this reference sample.
    sources: Vec<usize>,
}

/// Merge the samples each source verified into one reference grid, meshed once, so one
/// reference view stays a connected surface instead of source-wise islands (the TUM diagnosis
/// of 2026-11-21 measured 3,586 islands before, 466 after). Confidence keeps the strongest
/// agreement. Output is ordered by grid row, then column.
///
/// The sample is placed on its reference ray at the mean of the reference depth and every
/// verifying source's own measurement (each already within the 2% agreement band). Each
/// view's depth alone leaves overlapping views as separate layers a few centimetres apart,
/// which fusion can only join with steep, untexturable steps; the shared measurements pull
/// those layers onto one surface. Only agreeing observed depths are averaged, so the sample
/// stays geometric multi-view evidence of exactly the cameras listed in `sources`.
fn verified_union(
    reference_center: Vector3<f64>,
    candidates: Vec<(usize, Vec<Verified>)>,
    funnel: &mut MeshingFunnel,
) -> Vec<VerifiedSample> {
    let mut union = BTreeMap::<(u32, u32), (VerifiedSample, f64)>::new();
    for (source, patch) in candidates {
        for verified in patch {
            let Verified {
                point,
                site,
                source_depth_ratio,
            } = verified;
            union
                .entry((site.y, site.x))
                .and_modify(|(sample, ratios)| {
                    sample.point.confidence = sample.point.confidence.max(point.confidence);
                    sample.sources.push(source);
                    *ratios += source_depth_ratio;
                })
                .or_insert((
                    VerifiedSample {
                        point,
                        site,
                        sources: vec![source],
                    },
                    1. + source_depth_ratio,
                ));
        }
    }
    union
        .into_values()
        .map(|(mut sample, ratios)| {
            // The reference measurement counts once, each verifying source once.
            let scale = ratios / (sample.sources.len() + 1) as f64;
            sample.sources.sort_unstable();
            sample.sources.dedup();
            if (scale - 1.).abs() > 0.01 {
                funnel.consensus_moved_over_one_percent += 1;
            }
            let p = &mut sample.point;
            let at =
                |value: f32, center: f64| (center + (f64::from(value) - center) * scale) as f32;
            [p.x, p.y, p.z] = [
                at(p.x, reference_center.x),
                at(p.y, reference_center.y),
                at(p.z, reference_center.z),
            ];
            sample
        })
        .collect()
}

pub fn export(
    request: &ReconstructionRequest,
    registration: RegisteredSparseInput,
    grids: DepthGrids,
    inventory: Option<(
        &crate::semantic_inventory::SemanticInventoryArtifact,
        crate::scene_project::ContentHash,
    )>,
) -> Result<(Vec<u8>, SurfaceReport), String> {
    let sparse = registration.sparse(request)?;
    let width = sparse.image_width;
    let height = sparse.image_height;
    let focal = f64::from(sparse.focal_length_pixels);
    let cameras: Vec<_> = sparse
        .dense_cameras
        .iter()
        .map(|c| c.registered())
        .collect();
    let reference_stride = grids.reference_stride.unwrap_or(grids.stride);
    if grids.version != 1
        || !(2..=64).contains(&grids.stride)
        || !(grids.stride..=64).contains(&reference_stride)
        || !reference_stride.is_multiple_of(grids.stride)
        || grids.maps.len() < 3
        || grids.maps.len() > cameras.len()
    {
        return Err(
            "incompatible geometric depth grid version, spacing or registered camera count".into(),
        );
    }
    let length = (width as usize).div_ceil(grids.stride) * (height as usize).div_ceil(grids.stride);
    let mut views = BTreeMap::new();
    for grid in &grids.maps {
        let camera = cameras
            .iter()
            .find(|c| c.frame_index == grid.frame)
            .ok_or("unregistered geometric depth map")?;
        if grid.depth.len() != length
            || grid.depth.iter().any(|z| !z.is_finite() || *z < 0.)
            || grid.sources.len() > 4
            || grid.sources.contains(&grid.frame)
            || views
                .insert(
                    grid.frame,
                    View {
                        grid,
                        camera,
                        image: &request.frames[grid.frame],
                    },
                )
                .is_some()
        {
            return Err("invalid geometric depth map or reference view".into());
        }
    }
    let mut points = Vec::new();
    let mut sites = Vec::new();
    let mut triangles = Vec::new();
    let mut regions = Vec::new();
    let mut references = Vec::new();
    let mut cancel = || false;
    let mut control = ReconstructionControl::new(&mut cancel);
    let mut diagnostics = SurfaceDiagnostics::default();
    diagnostics.meshing.depth_views = views.len();
    // Keep the export bounded while covering the camera path. The adjacent views still
    // supply depth validation even when their own reference grid is not meshed.
    let step = views.len().div_ceil(36).max(1);
    for reference in views.values().step_by(step) {
        let mut candidates = Vec::new();
        for &source in &reference.grid.sources {
            let other = views
                .get(&source)
                .ok_or("geometric depth source is unregistered")?;
            let mut funnel = VerificationFunnel::default();
            let patch = verify_patch(
                reference,
                other,
                grids.stride,
                reference_stride,
                focal,
                &mut funnel,
            );
            diagnostics.verification.add(&funnel);
            candidates.push((source, patch));
        }
        let previous_len = triangles.len();
        let union = verified_union(
            reference.camera.camera_center(),
            candidates,
            &mut diagnostics.meshing,
        );
        diagnostics.meshing.verified_samples += union.len();
        diagnostics.meshing.multiply_verified_samples += union
            .iter()
            .filter(|sample| sample.sources.len() > 1)
            .count();
        if !union.is_empty() {
            let p: Vec<_> = union.iter().map(|sample| sample.point).collect();
            let g: Vec<_> = union.iter().map(|sample| sample.site).collect();
            let stats = DenseStats {
                attempted: true,
                reference_frame: Some(reference.grid.frame),
                grid_stride: reference_stride,
                ..Default::default()
            };
            let mesh = crate::mesh::reconstruct_dense_mesh_with_control(
                &p,
                &g,
                &stats,
                &cameras,
                width,
                height,
                focal,
                &mut control,
            );
            let meshing = &mut diagnostics.meshing;
            meshing.candidate_cells += mesh.stats.candidate_cells;
            meshing.rejected_discontinuities += mesh.stats.rejected_discontinuities;
            meshing.rejected_degenerate += mesh.stats.rejected_degenerate;
            let start = points.len();
            let mut indices = vec![None; p.len()];
            // Provenance names exactly the cameras that verified a retained vertex.
            let mut sources = BTreeSet::new();
            for mut face in mesh.triangles {
                for corner in [&mut face.a, &mut face.b, &mut face.c] {
                    let i = *corner;
                    *corner = *indices[i].get_or_insert_with(|| {
                        let index = points.len();
                        points.push(p[i]);
                        sites.push(g[i]);
                        sources.extend(&union[i].sources);
                        index
                    });
                }
                triangles.push(face);
            }
            if points.len() > start {
                regions.push(SurfaceEvidenceRegion {
                    origin: EvidenceOrigin::GeometricMultiView,
                    reference_frame: Some(reference.grid.frame),
                    source_frames: sources.into_iter().collect(),
                    points: EvidenceRange::new(start, points.len() - start),
                });
            }
        }
        if triangles.len() == previous_len {
            continue;
        }
        references.push(ReferenceImage {
            frame_index: reference.grid.frame,
            width,
            height,
            rgba: &reference.image.rgba,
        });
    }
    if triangles.is_empty() {
        return Err("COLMAP stereo produced no cross-view verified surface triangles".into());
    }
    diagnostics.meshing.meshed_reference_views = references.len();
    diagnostics.dominant_rejection = diagnostics.verification.dominant_rejection();
    // Reference views were meshed independently; join the views that observed the same
    // surface with the classic pipeline's fusion and ownership merge (Rust owns fusion).
    let consolidated = crate::surface_fusion::consolidate_reference_views(
        points,
        triangles,
        &regions,
        &cameras,
        &mut control,
    )?;
    let points = consolidated.points;
    let triangles = consolidated.triangles;
    let derived_vertices = consolidated.derived_vertices;
    let triangle_sources = consolidated.triangle_sources;
    let merge = &consolidated.merge;
    diagnostics.fusion = ViewFusionFunnel {
        reference_views: consolidated.support_groups,
        fused_seam_pairs: consolidated.fused_pairs,
        cross_reference_triangles: merge.cross_reference_triangles,
        removed_duplicate_triangles: merge.removed_duplicate_triangles,
        stitched_seam_triangles: merge.stitched_seam_triangles,
        derived_vertices: merge.derived_vertices,
        patch_components: merge.patch_components,
        dropped_unconformable_triangles: merge.dropped_unconformable_triangles,
        remaining_overlap_triangles: merge.remaining_overlap_triangles,
        trimmed_seam_triangles: merge.trimmed_seam_triangles,
    };
    diagnostics.meshing.exported_triangles = triangles.len();
    diagnostics.topology = surface_topology(points.len(), &triangles);
    let evidence_cameras = cameras
        .iter()
        .map(|c| EvidenceCamera {
            frame_index: c.frame_index,
            authority: EvidenceCameraAuthority::RegisteredGeometry,
            rotation: std::array::from_fn(|i| c.rotation[(i / 3, i % 3)] as f32),
            translation: [
                c.translation.x as f32,
                c.translation.y as f32,
                c.translation.z as f32,
            ],
            confidence: None,
            median_reprojection_error_pixels: None,
        })
        .collect();
    let evidence = ReconstructionEvidenceView::new_with_derived_vertices(
        ReconstructionProviderDescriptor::new(
            "colmap-patch-match",
            ReconstructionProviderClass::GeometricMultiView,
            Some("4.2.1".into()),
        )?,
        EvidenceScale::ArbitraryMonocular,
        evidence_cameras,
        regions,
        &points,
        &triangles,
        &derived_vertices,
    )?
    .with_triangle_sources(&triangle_sources)?;
    // Cross-reference triangles are textured from a camera that sees all their corners.
    let bake = bake_surface_materials_with_seams(
        &evidence,
        &sites,
        &references,
        Some(SeamProjection {
            focal_pixels: focal as f32,
        }),
    )?;
    diagnostics.untextured = bake.fallback.reasons.clone();
    let partition = inventory
        .map(|(inventory, hash)| {
            crate::object_surface::partition_object_surfaces(
                &evidence,
                inventory,
                hash,
                [width, height],
                Some(focal as f32),
            )
        })
        .transpose()?;
    let bytes = encode_textured_glb_with_camera_provider(
        &evidence,
        &bake,
        None,
        partition.as_ref(),
        Some(&CameraProviderMetadata {
            id: "colmap".into(),
            version: "4.2.1".into(),
        }),
    )?;
    let report = SurfaceReport {
        registered_views: cameras.len(),
        reference_views: references.len(),
        verified_points: points.len() - derived_vertices.len(),
        triangles: triangles.len(),
        textured_triangles: bake.textured_triangles(),
        object_surfaces: partition.as_ref().map_or(0, |part| {
            part.groups
                .iter()
                .flatten()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        }),
        object_triangles: partition
            .as_ref()
            .map_or(0, |part| part.groups.iter().flatten().count()),
        diagnostics,
    };
    Ok((bytes, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    #[test]
    fn sources_verifying_one_sample_share_one_vertex() {
        let sample = |x: u32, y: u32, confidence: f32| Verified {
            point: Point3 {
                x: x as f32,
                y: y as f32,
                z: 5.,
                r: 1,
                g: 2,
                b: 3,
                confidence,
            },
            site: DenseGridSite { x, y },
            source_depth_ratio: 1.,
        };
        // Source 7 verified the left half of a 3x2 grid, source 3 the right half; they overlap
        // in the middle column.
        let mut funnel = MeshingFunnel::default();
        let union = verified_union(
            Vector3::zeros(),
            vec![
                (
                    7,
                    vec![
                        sample(0, 0, 0.9),
                        sample(4, 0, 0.9),
                        sample(0, 4, 0.9),
                        sample(4, 4, 0.9),
                    ],
                ),
                (
                    3,
                    vec![
                        sample(4, 0, 0.95),
                        sample(8, 0, 0.95),
                        sample(4, 4, 0.8),
                        sample(8, 4, 0.95),
                    ],
                ),
            ],
            &mut funnel,
        );
        let sites: Vec<_> = union.iter().map(|s| (s.site.x, s.site.y)).collect();
        assert_eq!(sites, [(0, 0), (4, 0), (8, 0), (0, 4), (4, 4), (8, 4)]);
        let sources: Vec<_> = union.iter().map(|s| s.sources.clone()).collect();
        assert_eq!(
            sources,
            [vec![7], vec![3, 7], vec![3], vec![7], vec![3, 7], vec![3]]
        );
        // The strongest verification survives; agreeing measurements leave the position alone.
        assert_eq!(union[1].point.confidence, 0.95);
        assert_eq!(union[4].point.confidence, 0.9);
        assert_eq!([union[4].point.x, union[4].point.z], [4., 5.]);
        assert_eq!(funnel.consensus_moved_over_one_percent, 0);

        // Meshed once, both halves form one island; meshed per source they would be two.
        let cameras = [0, 3, 7].map(|frame_index| RegisteredCamera {
            frame_index,
            rotation: nalgebra::Matrix3::identity(),
            translation: Vector3::zeros(),
        });
        let points: Vec<_> = union
            .iter()
            .map(|s| Point3 {
                x: (s.site.x as f32 + 0.5 - 6.) / 20. * 5.,
                y: (s.site.y as f32 + 0.5 - 4.) / 20. * 5.,
                ..s.point
            })
            .collect();
        let sites: Vec<_> = union.iter().map(|s| s.site).collect();
        let stats = DenseStats {
            attempted: true,
            reference_frame: Some(0),
            grid_stride: 4,
            ..Default::default()
        };
        let mut cancel = || false;
        let mesh = crate::mesh::reconstruct_dense_mesh_with_control(
            &points,
            &sites,
            &stats,
            &cameras,
            12,
            8,
            20.,
            &mut ReconstructionControl::new(&mut cancel),
        );
        assert_eq!(mesh.triangles.len(), 4);
        assert_eq!(surface_topology(points.len(), &mesh.triangles).islands, 1);
    }

    #[test]
    fn agreeing_source_depths_move_a_sample_along_its_reference_ray() {
        let at = |source_depth_ratio: f64| Verified {
            point: Point3 {
                x: 2.,
                y: 0.,
                z: 4.,
                r: 0,
                g: 0,
                b: 0,
                confidence: 1.,
            },
            site: DenseGridSite { x: 0, y: 0 },
            source_depth_ratio,
        };
        let center = Vector3::new(1., 0., 0.);
        let mut funnel = MeshingFunnel::default();
        // Reference 1, sources 1.02 and 1.03: the mean 1.0167 scales the ray from the center.
        let union = verified_union(
            center,
            vec![(4, vec![at(1.02)]), (9, vec![at(1.03)])],
            &mut funnel,
        );
        let p = union[0].point;
        let scale = (1. + 1.02 + 1.03) / 3.;
        assert!((f64::from(p.x) - (1. + scale)).abs() < 1e-6);
        assert!((f64::from(p.z) - 4. * scale).abs() < 1e-6);
        assert_eq!(p.y, 0.);
        assert_eq!(union[0].sources, [4, 9]);
        assert_eq!(funnel.consensus_moved_over_one_percent, 1);

        // A single agreeing source moves the sample halfway, here by less than 1%.
        let mut funnel = MeshingFunnel::default();
        let union = verified_union(center, vec![(4, vec![at(0.99)])], &mut funnel);
        assert!((f64::from(union[0].point.z) - 4. * 0.995).abs() < 1e-6);
        assert_eq!(funnel.consensus_moved_over_one_percent, 0);
    }

    #[test]
    fn source_depth_ratio_lands_on_the_source_measurement() {
        // Reference at the origin looking down +z; source one metre to the right, same axes.
        let frame = FrameInput {
            width: 24,
            height: 16,
            rgba: vec![100; 24 * 16 * 4],
        };
        let near = Grid {
            frame: 0,
            sources: vec![1],
            depth: vec![8.; 24],
        };
        let far = Grid {
            frame: 1,
            sources: vec![0],
            depth: vec![8.08; 24],
        };
        let first = RegisteredCamera {
            frame_index: 0,
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        };
        let second = RegisteredCamera {
            frame_index: 1,
            rotation: Matrix3::identity(),
            translation: Vector3::new(-1., 0., 0.),
        };
        let view = |camera, grid| View {
            camera,
            grid,
            image: &frame,
        };
        let verified = verify_patch(
            &view(&first, &near),
            &view(&second, &far),
            4,
            4,
            20.,
            &mut VerificationFunnel::default(),
        );
        assert!(!verified.is_empty());
        // Fronto-parallel planes: the source measured 1% deeper everywhere it sees the sample.
        for v in &verified {
            assert!(
                (v.source_depth_ratio - 1.01).abs() < 1e-6,
                "{}",
                v.source_depth_ratio
            );
        }
    }

    #[test]
    fn plane_requires_agreeing_depth_color_and_camera_baseline() {
        let frame = FrameInput {
            width: 24,
            height: 16,
            rgba: vec![100; 24 * 16 * 4],
        };
        let other_color = FrameInput {
            width: 24,
            height: 16,
            rgba: vec![200; 24 * 16 * 4],
        };
        let first = RegisteredCamera {
            frame_index: 0,
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        };
        let second = RegisteredCamera {
            frame_index: 1,
            rotation: Matrix3::identity(),
            translation: Vector3::new(-1., 0., 0.),
        };
        let map = Grid {
            frame: 0,
            sources: vec![1],
            depth: vec![5.; 24],
        };
        let wrong = Grid {
            frame: 1,
            sources: vec![0],
            depth: vec![8.; 24],
        };
        let reference = View {
            camera: &first,
            grid: &map,
            image: &frame,
        };
        let source = View {
            camera: &second,
            grid: &map,
            image: &frame,
        };
        let mut funnel = VerificationFunnel::default();
        let verified = verify_patch(&reference, &source, 4, 4, 20., &mut funnel);
        assert_eq!(verified.len(), 20);
        // Both maps measure 8 m: the source puts every sample exactly where the reference does.
        assert!(verified
            .iter()
            .all(|v| (v.source_depth_ratio - 1.).abs() < 1e-12));
        // The one-pixel baseline shifts a column of samples out of the source image.
        assert_eq!(
            funnel,
            VerificationFunnel {
                reference_samples: 24,
                outside_source: 4,
                accepted: 20,
                ..Default::default()
            }
        );
        let coarse = verify_patch(
            &reference,
            &source,
            4,
            8,
            20.,
            &mut VerificationFunnel::default(),
        );
        assert_eq!(coarse.len(), 4);
        assert!(coarse
            .iter()
            .all(|v| v.site.x % 8 == 0 && v.site.y % 8 == 0));
        for (rejected, expected) in [
            (
                View {
                    grid: &wrong,
                    ..source
                },
                "depth_disagreement",
            ),
            (
                View {
                    image: &other_color,
                    ..source
                },
                "color_disagreement",
            ),
            (
                View {
                    camera: &first,
                    ..source
                },
                "insufficient_parallax",
            ),
        ] {
            let mut funnel = VerificationFunnel::default();
            assert!(verify_patch(&reference, &rejected, 4, 4, 20., &mut funnel).is_empty());
            assert_eq!(funnel.accepted, 0);
            assert_eq!(funnel.dominant_rejection(), Some(expected));
        }
    }

    #[test]
    fn topology_counts_islands_and_open_boundaries() {
        let face = |a, b, c| MeshTriangle {
            a,
            b,
            c,
            confidence: 1.,
        };
        // A two-triangle quad, plus the same quad meshed again from another reference view
        // with its own vertices: the duplicate is a separate island, not a joined surface.
        let triangles = [face(0, 1, 2), face(1, 3, 2), face(4, 5, 6), face(5, 7, 6)];
        let topology = surface_topology(8, &triangles);
        assert_eq!(
            topology,
            SurfaceTopology {
                islands: 2,
                small_islands: 2,
                small_island_triangles: 4,
                largest_island_triangles: 2,
                boundary_edges: 8,
            }
        );
        let large: Vec<_> = (0..SMALL_ISLAND_TRIANGLES)
            .map(|i| face(i, i + 1, i + 2))
            .collect();
        let topology = surface_topology(SMALL_ISLAND_TRIANGLES + 2, &large);
        assert_eq!((topology.islands, topology.small_islands), (1, 0));
        assert_eq!(topology.largest_island_triangles, SMALL_ISLAND_TRIANGLES);
    }
}
