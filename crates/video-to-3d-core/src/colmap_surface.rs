//! Verify external geometric depth against another registered view, then pass the retained
//! reference grids to the shared topology, provenance, material and GLB exporters.
use crate::multi_view::RegisteredCamera;
use crate::registered_sparse::RegisteredSparseInput;
use crate::surface_materials::{bake_surface_materials, ReferenceImage};
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
) -> Vec<(Point3, DenseGridSite)> {
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
    for (i, &z) in reference.grid.depth.iter().enumerate() {
        if z <= 0. {
            continue;
        }
        let x = (i % cols) * stride;
        let y = (i / cols) * stride;
        if !x.is_multiple_of(reference_stride) || !y.is_multiple_of(reference_stride) {
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
            continue;
        }
        let q = source.camera.rotation * world + source.camera.translation;
        if q.z <= 0. {
            continue;
        }
        let u = focal * q.x / q.z + cx - 0.5;
        let v = focal * q.y / q.z + cy - 0.5;
        if !(0.0..width as f64).contains(&u) || !(0.0..height as f64).contains(&v) {
            continue;
        }
        let col = (u / stride as f64).round() as usize;
        let row = (v / stride as f64).round() as usize;
        if col >= cols || row >= rows {
            continue;
        }
        let other = f64::from(source.grid.depth[row * cols + col]);
        // Require an independent measurement in the source map. A foreground/background
        // transition, an unobserved pixel, or an inconsistent depth cannot create topology.
        if other <= 0. || (other - q.z).abs() > 0.02 * q.z {
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
            continue;
        }
        points.push((
            Point3 {
                x: world.x as f32,
                y: world.y as f32,
                z: world.z as f32,
                r,
                g,
                b,
                confidence: (1. - (other - q.z).abs() / q.z) as f32,
            },
            DenseGridSite {
                x: x as u32,
                y: y as u32,
            },
        ));
    }
    points
}

/// A reference cell belongs to the first supporting source that can mesh it. A second
/// camera may establish a different cell where the first lacked visibility.
fn uncovered_triangles(
    mut triangles: Vec<MeshTriangle>,
    sites: &[DenseGridSite],
    covered: &mut BTreeSet<(u32, u32)>,
) -> Vec<MeshTriangle> {
    let mut newly_covered = BTreeSet::new();
    triangles.retain(|triangle| {
        let [a, b, c] = [triangle.a, triangle.b, triangle.c].map(|i| sites[i]);
        let cell = (a.x.min(b.x).min(c.x), a.y.min(b.y).min(c.y));
        if covered.contains(&cell) {
            return false;
        }
        newly_covered.insert(cell);
        true
    });
    covered.append(&mut newly_covered);
    triangles
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
    // Keep the export bounded while covering the camera path. The adjacent views still
    // supply depth validation even when their own reference grid is not meshed.
    let step = views.len().div_ceil(36).max(1);
    for reference in views.values().step_by(step) {
        let mut candidates = Vec::new();
        for &source in &reference.grid.sources {
            let other = views
                .get(&source)
                .ok_or("geometric depth source is unregistered")?;
            let patch = verify_patch(reference, other, grids.stride, reference_stride, focal);
            candidates.push((source, patch));
        }
        candidates.sort_by_key(|(_, patch)| std::cmp::Reverse(patch.len()));
        let mut covered_cells = BTreeSet::new();
        let previous_len = triangles.len();
        for (source, patch) in candidates {
            let (p, g): (Vec<_>, Vec<_>) = patch.into_iter().unzip();
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
            let retained = uncovered_triangles(mesh.triangles, &g, &mut covered_cells);
            if retained.is_empty() {
                continue;
            }
            let start = points.len();
            let mut indices = vec![None; p.len()];
            for mut face in retained {
                for corner in [&mut face.a, &mut face.b, &mut face.c] {
                    let i = *corner;
                    *corner = *indices[i].get_or_insert_with(|| {
                        let index = points.len();
                        points.push(p[i]);
                        sites.push(g[i]);
                        index
                    });
                }
                triangles.push(face);
            }
            regions.push(SurfaceEvidenceRegion {
                origin: EvidenceOrigin::GeometricMultiView,
                reference_frame: Some(reference.grid.frame),
                source_frames: vec![source],
                points: EvidenceRange::new(start, points.len() - start),
            });
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
    let evidence = ReconstructionEvidenceView::new(
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
    )?;
    let bake = bake_surface_materials(&evidence, &sites, &references)?;
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
        verified_points: points.len(),
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
    };
    Ok((bytes, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    #[test]
    fn alternate_source_adds_only_previously_unmeshed_cells() {
        let sites =
            [(0, 0), (4, 0), (8, 0), (0, 4), (4, 4), (8, 4)].map(|(x, y)| DenseGridSite { x, y });
        let face = |a, b, c| MeshTriangle {
            a,
            b,
            c,
            confidence: 1.,
        };
        let first = vec![face(0, 1, 4), face(0, 4, 3)];
        let alternate = vec![face(0, 1, 3), face(1, 4, 3), face(1, 2, 5), face(1, 5, 4)];
        let mut covered = BTreeSet::new();
        assert_eq!(
            uncovered_triangles(first.clone(), &sites, &mut covered),
            first
        );
        assert_eq!(
            uncovered_triangles(alternate, &sites, &mut covered),
            vec![face(1, 2, 5), face(1, 5, 4)]
        );
        assert_eq!(covered, BTreeSet::from([(0, 0), (4, 0)]));
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
        assert_eq!(verify_patch(&reference, &source, 4, 4, 20.).len(), 20);
        let coarse = verify_patch(&reference, &source, 4, 8, 20.);
        assert_eq!(coarse.len(), 4);
        assert!(coarse
            .iter()
            .all(|(_, site)| site.x % 8 == 0 && site.y % 8 == 0));
        for rejected in [
            View {
                grid: &wrong,
                ..source
            },
            View {
                image: &other_color,
                ..source
            },
            View {
                camera: &first,
                ..source
            },
        ] {
            assert!(verify_patch(&reference, &rejected, 4, 4, 20.).is_empty());
        }
    }
}
