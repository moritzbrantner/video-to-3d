//! Connected surface across overlapping reference patches.
//!
//! Every reference patch is triangulated on its own reference grid, so a slow pan
//! over one surface yields overlapping per-reference sheets. Surface fusion
//! (`surface_fusion.rs`) aligns mutually nearest seam vertices of different
//! camera-support groups to one position but never changes topology. This pass
//! turns that accepted alignment into topology, fail-closed:
//!
//! - **Shared vertices.** Each fused pair becomes one vertex: triangles of the
//!   lower-priority patch are re-indexed onto the higher-priority patch's
//!   vertex. Only pairs fusion accepted are merged; both points stay in the
//!   point buffer with their own provenance, and no vertex is created or moved.
//! - **No duplicated overlap.** Ownership is decided in the image of the
//!   higher-priority patch's reference camera, where that patch's grid
//!   triangles tile its footprint without overlap: a lower-priority triangle is
//!   projected into that image and the share of its area lying on kept
//!   triangles at the same depth is measured: within a relative bound, after
//!   removing the two patches' relative depth scale (per-reference depth scale
//!   differs by about a percent). Only patches with a fusion-accepted seam pair
//!   between them may own each other's overlap. A triangle at least
//!   [`MIN_IMAGE_OWNERSHIP_SHARE`] owned by one higher-priority reference is
//!   removed. The part of a removed seam triangle that was not owned is
//!   reported as trimmed seam area, so the bounded loss stays observable.
//!   Triangles at another depth (separate surfaces, occluders) are never owned,
//!   and a gap no patch covered stays a gap.
//! - **3D fallback.** Without a reference camera, a lower-priority triangle is
//!   removed only if its entire area is covered by higher-priority, aligned,
//!   depth-consistent triangles from a patch with a fusion-accepted link.
//! - **Cross-reference topology.** Triangles referencing accepted shared
//!   vertices retain cross-patch connections wherever they extend coverage.
//!
//! Patch priority is deterministic: more accepted triangles first, then the
//! lower support-group index.

use crate::{MeshTriangle, Point3};
use nalgebra::{Matrix3, Vector2, Vector3};
use std::collections::{BTreeSet, HashMap};

const GEOMETRIC_EPSILON: f64 = 1.0e-9;
/// Largest distance along the kept triangle's normal, as a fraction of the
/// median accepted edge length, at which a triangle counts as the same surface.
const MAX_OVERLAP_DISTANCE_FRACTION: f64 = 0.9;
/// Maximum grid cells occupied by one triangle before falling back to the exact list.
const MAX_GRID_CELLS: usize = 512;
/// Fail closed rather than processing a pathological number of clipping fragments.
const MAX_UNCOVERED_PIECES: usize = 128;
const MIN_NORMAL_ALIGNMENT: f64 = 0.94;
/// Share of a lower-priority triangle's projected area that one higher-priority
/// reference must own (kept triangles at the same depth) for it to be removed.
/// Below full ownership the triangle sits on the seam; at most the remaining
/// share of one triangle is trimmed there, and it is reported.
pub(crate) const MIN_IMAGE_OWNERSHIP_SHARE: f64 = 0.8;
/// Largest depth disagreement, as a share of the owner's depth, at which two
/// references see the same surface, checked at every corner of the overlap
/// after removing the two patches' relative depth scale. Single grid triangles
/// tilt with per-point depth noise of about a percent, so their corners spread
/// by up to about 3 %; a layer nearer to the surface than this is below the
/// dense depth's resolution. A separate layer in front of the surface (a
/// railing 0.2 m before a wall 5 m away is 4 %) differs by more.
const MAX_RELATIVE_DEPTH_DISAGREEMENT: f64 = 0.035;
/// Overlaps used to estimate two patches' relative depth scale must agree to
/// this share before the scale is removed; per-reference scale bias is a few
/// percent at most.
const MAX_RELATIVE_SCALE_SAMPLE: f64 = 0.1;

/// World-to-camera pose of a reference view: `x_cam = rotation * x + translation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReferenceCamera {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

impl ReferenceCamera {
    /// Normalized image coordinates and depth; `None` behind the camera.
    fn project(&self, point: Vector3<f64>) -> Option<(Vector2<f64>, f64)> {
        let camera = self.rotation * point + self.translation;
        (camera.z > GEOMETRIC_EPSILON && camera.iter().all(|value| value.is_finite())).then(|| {
            (
                Vector2::new(camera.x / camera.z, camera.y / camera.z),
                camera.z,
            )
        })
    }

    /// The plane through `corners` in camera coordinates, as `(n, d)` with
    /// `n · x = d`.
    fn plane(&self, corners: &[Vector3<f64>; 3]) -> Option<(Vector3<f64>, f64)> {
        let [a, b, c] = corners.map(|corner| self.rotation * corner + self.translation);
        let normal = (b - a).cross(&(c - a));
        (normal.norm() > GEOMETRIC_EPSILON).then(|| (normal, normal.dot(&a)))
    }

    fn project_triangle(&self, corners: &[Vector3<f64>; 3]) -> Option<[(Vector2<f64>, f64); 3]> {
        let [a, b, c] = corners.map(|corner| self.project(corner));
        Some([a?, b?, c?])
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct MergeStats {
    pub fused_pairs: usize,
    pub shared_vertices: usize,
    pub cross_reference_triangles: usize,
    /// Lower-priority triangles dropped because a kept patch already meshes
    /// that surface.
    pub removed_duplicate_triangles: usize,
    /// Partially covered lower-priority triangles kept to preserve accepted surface.
    pub remaining_overlap_triangles: usize,
    /// Removed seam triangles that a higher-priority reference owned only in
    /// part (at least `MIN_IMAGE_OWNERSHIP_SHARE`).
    pub trimmed_seam_triangles: usize,
    /// Unowned surface area of the trimmed seam triangles (measured on each
    /// triangle's plane, not in the image), as a share of the accepted mesh
    /// area before the merge.
    pub trimmed_seam_area_share: f64,
    /// Retained triangles per support group (the patch each triangle was
    /// triangulated in), indexed like `membership`'s groups.
    pub retained_by_group: Vec<usize>,
    /// Connected components of the patch graph (patches joined by
    /// cross-reference triangles); `0` without meshed patches.
    pub patch_components: usize,
    pub meshed_patches: usize,
}

impl MergeStats {
    pub fn diagnostic(&self, support_groups: usize) -> String {
        format!(
            "Overlapping reference patches: {} fused seam pair(s) became shared vertices ({} shared vertices, {} cross-reference triangles); {} duplicated overlap triangle(s) were dropped in favour of the patch already meshing that surface, {} of them seam triangles owned only in part, trimming {:.2}% of the accepted mesh area, and {} partially overlapping triangle(s) were retained to preserve supported geometry. {} of {} camera-support group(s) carry triangles, forming {} connected patch component(s). Only fusion-accepted pairs were merged; no vertex was created or moved and gaps no patch covered stay open.",
            self.fused_pairs,
            self.shared_vertices,
            self.cross_reference_triangles,
            self.removed_duplicate_triangles,
            self.trimmed_seam_triangles,
            self.trimmed_seam_area_share * 100.0,
            self.remaining_overlap_triangles,
            self.meshed_patches,
            support_groups,
            self.patch_components,
        )
    }
}

pub(super) fn merge_fused_patches(
    points: &[Point3],
    triangles: &mut Vec<MeshTriangle>,
    membership: &[Option<usize>],
    pairs: &[(usize, usize)],
    reference_cameras: &[Option<ReferenceCamera>],
) -> MergeStats {
    let group_of = |vertex: usize| membership.get(vertex).copied().flatten();
    let valid = |triangle: &MeshTriangle| {
        [triangle.a, triangle.b, triangle.c]
            .iter()
            .all(|vertex| *vertex < points.len())
    };
    if points.len() != membership.len() || !triangles.iter().all(valid) {
        return MergeStats::default();
    }

    // Components are built from ORIGINAL patch topology, before remapping any
    // accepted seam vertices. One accepted correspondence does not authorize
    // deleting an unrelated disconnected island from the same two cameras.
    let mut component_parent: Vec<usize> = (0..points.len()).collect();
    for triangle in triangles.iter() {
        if group_of(triangle.a).is_none()
            || group_of(triangle.a) != group_of(triangle.b)
            || group_of(triangle.a) != group_of(triangle.c)
        {
            continue;
        }
        let representative = find(&mut component_parent, triangle.a);
        for vertex in [triangle.b, triangle.c] {
            let root = find(&mut component_parent, vertex);
            component_parent[root] = representative;
        }
    }
    let component_by_vertex: Vec<usize> = (0..points.len())
        .map(|vertex| find(&mut component_parent, vertex))
        .collect();
    let component_by_triangle: Vec<usize> = triangles
        .iter()
        .map(|triangle| component_by_vertex[triangle.a])
        .collect();

    // Each triangle belongs to the patch of its own vertices (patches are
    // triangulated independently, so all three agree).
    let patch_of: Vec<Option<usize>> = triangles.iter().map(|t| group_of(t.a)).collect();
    let mut triangle_counts: HashMap<usize, usize> = HashMap::new();
    for patch in patch_of.iter().flatten() {
        *triangle_counts.entry(*patch).or_default() += 1;
    }
    let mut order: Vec<usize> = triangle_counts.keys().copied().collect();
    order.sort_by(|a, b| triangle_counts[b].cmp(&triangle_counts[a]).then(a.cmp(b)));
    let rank: HashMap<usize, usize> = order
        .iter()
        .enumerate()
        .map(|(rank, patch)| (*patch, rank))
        .collect();
    let rank_of = |vertex: usize| {
        group_of(vertex)
            .and_then(|group| rank.get(&group).copied())
            .unwrap_or(usize::MAX)
    };

    // Shared vertices: the lower-priority side of each fused pair maps onto the
    // higher-priority side.
    let mut canonical: Vec<usize> = (0..points.len()).collect();
    let mut fused_pairs = 0;
    for &(first, second) in pairs {
        if first >= points.len()
            || second >= points.len()
            || group_of(first).is_none()
            || group_of(first) == group_of(second)
            || canonical[first] != first
            || canonical[second] != second
        {
            continue;
        }
        let (keep, drop) = if (rank_of(first), first) <= (rank_of(second), second) {
            (first, second)
        } else {
            (second, first)
        };
        canonical[drop] = keep;
        fused_pairs += 1;
    }
    // Only ORIGINAL mesh components containing a fusion-accepted pair may
    // deduplicate each other. The same cameras can contain unrelated islands.
    let linked_components: BTreeSet<(usize, usize)> = pairs
        .iter()
        .filter_map(|&(a, b)| {
            if a >= points.len() || b >= points.len() || canonical[a] != canonical[b] {
                return None;
            }
            let (Some(a_group), Some(b_group)) = (group_of(a), group_of(b)) else {
                return None;
            };
            if a_group == b_group {
                return None;
            }
            let (a, b) = (component_by_vertex[a], component_by_vertex[b]);
            Some((a.min(b), a.max(b)))
        })
        .collect();
    // Patches with at least one fusion-accepted pair: only these may own each
    // other's overlap in the image test.
    let linked_patches: BTreeSet<(usize, usize)> = pairs
        .iter()
        .filter(|&&(a, b)| a < points.len() && b < points.len() && canonical[a] == canonical[b])
        .filter_map(|&(a, b)| match (group_of(a), group_of(b)) {
            (Some(left), Some(right)) if left != right => Some((left.min(right), left.max(right))),
            _ => None,
        })
        .collect();
    for triangle in triangles.iter_mut() {
        let mapped = [triangle.a, triangle.b, triangle.c].map(|vertex| canonical[vertex]);
        // Never replace valid topology with a degenerate triangle.
        if mapped[0] != mapped[1] && mapped[1] != mapped[2] && mapped[0] != mapped[2] {
            [triangle.a, triangle.b, triangle.c] = mapped;
        }
    }

    // Reject only faces whose ENTIRE projected area is supported by the union
    // of higher-priority patch triangles. A centroid test silently discards
    // valid corners and holes. Use a bounded triangle-AABB index (including
    // interiors of large triangles), and an exact planar polygon subtraction.
    let position = |vertex: usize| {
        let point = points[vertex];
        Vector3::new(f64::from(point.x), f64::from(point.y), f64::from(point.z))
    };
    let geometry: Vec<Option<TriangleGeometry>> = triangles
        .iter()
        .map(|triangle| TriangleGeometry::new([triangle.a, triangle.b, triangle.c].map(position)))
        .collect();
    let mut edges: Vec<f64> = geometry
        .iter()
        .flatten()
        .flat_map(|geometry| geometry.edges)
        .collect();
    edges.sort_by(f64::total_cmp);
    let Some(&median_edge) = edges.get(edges.len() / 2) else {
        return MergeStats::default();
    };
    // Reference-scale drift can exceed half an edge, but only a confirmed
    // fused patch pair can use this bounded normal-distance allowance.
    let tolerance = median_edge * MAX_OVERLAP_DISTANCE_FRACTION;
    let cell = (median_edge * 2.0).max(GEOMETRIC_EPSILON);
    let mut keep = vec![true; triangles.len()];
    let mut kept_cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut large_surfaces = Vec::new();
    let mut all_kept = Vec::new();
    let mut removed_duplicate_triangles = 0;
    let mut remaining_overlap_triangles = 0;
    let mut trimmed_seam_triangles = 0;
    let mut trimmed_seam_area = 0.0;
    let total_area: f64 = geometry.iter().flatten().map(TriangleGeometry::area).sum();
    // Kept triangles of each processed patch with a reference camera,
    // projected into that camera and indexed on a grid over its image.
    let mut footprints: Vec<Footprint> = Vec::new();

    for (patch_rank, patch) in order.iter().enumerate() {
        let members: Vec<usize> = patch_of
            .iter()
            .enumerate()
            .filter_map(|(index, owner)| (*owner == Some(*patch)).then_some(index))
            .collect();
        if patch_rank > 0 {
            // Fusion-linked owners, each with this patch's depth scale relative
            // to it (the median over loosely agreeing overlaps).
            let owners: Vec<(&Footprint, f64)> = footprints
                .iter()
                .filter(|footprint| {
                    linked_patches
                        .contains(&(footprint.patch.min(*patch), footprint.patch.max(*patch)))
                })
                .map(|footprint| {
                    let mut ratios: Vec<f64> = members
                        .iter()
                        .filter_map(|&index| geometry[index].as_ref())
                        .flat_map(|candidate| footprint.depth_ratios(&candidate.corners))
                        .collect();
                    ratios.sort_by(f64::total_cmp);
                    (
                        footprint,
                        ratios.get(ratios.len() / 2).copied().unwrap_or(1.0),
                    )
                })
                .collect();
            for &index in &members {
                let Some(candidate) = &geometry[index] else {
                    continue;
                };
                let owned = owners
                    .iter()
                    .map(|(footprint, scale)| footprint.ownership(&candidate.corners, *scale))
                    .fold(Ownership::default(), |best, next| {
                        if next.image_share > best.image_share {
                            next
                        } else {
                            best
                        }
                    });
                if owned.image_share >= MIN_IMAGE_OWNERSHIP_SHARE {
                    keep[index] = false;
                    removed_duplicate_triangles += 1;
                    if owned.image_share < 1.0 - OWNERSHIP_EPSILON {
                        trimmed_seam_triangles += 1;
                        trimmed_seam_area += (candidate.area() - owned.surface_area).max(0.0);
                    }
                    continue;
                }
                let mut nearby = BTreeSet::<usize>::new();
                let bounds = candidate.grid_bounds(cell, 0.0);
                if let Some(keys) = grid_cells(&bounds) {
                    for key in keys {
                        if let Some(indices) = kept_cells.get(&key) {
                            nearby.extend(indices);
                        }
                    }
                    nearby.extend(&large_surfaces);
                } else {
                    nearby.extend(&all_kept);
                }
                let candidates = nearby
                    .into_iter()
                    .filter(|&other| {
                        let (left, right) =
                            (component_by_triangle[index], component_by_triangle[other]);
                        linked_components.contains(&(left.min(right), left.max(right)))
                    })
                    .filter_map(|other| geometry[other].as_ref());
                let (covered, partially_covered) = candidate.covered_by(candidates, tolerance);
                if covered {
                    keep[index] = false;
                    removed_duplicate_triangles += 1;
                } else if partially_covered || owned.image_share > OWNERSHIP_EPSILON {
                    // In the absence of proven full coverage keep the original
                    // face: it may be the only evidence at a hole or boundary.
                    remaining_overlap_triangles += 1;
                }
            }
        }
        // Do not index this patch until it has been processed: triangles of a
        // single camera patch cannot erase each other.
        if let Some(camera) = reference_cameras.get(*patch).copied().flatten() {
            let kept: Vec<[Vector3<f64>; 3]> = members
                .iter()
                .filter(|&&index| keep[index])
                .filter_map(|&index| geometry[index].as_ref().map(|surface| surface.corners))
                .collect();
            if let Some(footprint) = Footprint::new(*patch, camera, &kept) {
                footprints.push(footprint);
            }
        }
        for &index in &members {
            if !keep[index] {
                continue;
            }
            let Some(surface) = &geometry[index] else {
                continue;
            };
            all_kept.push(index);
            if let Some(keys) = grid_cells(&surface.grid_bounds(cell, tolerance)) {
                for key in keys {
                    kept_cells.entry(key).or_default().push(index);
                }
            } else {
                large_surfaces.push(index);
            }
        }
    }
    let group_count = membership
        .iter()
        .flatten()
        .max()
        .map_or(0, |group| group + 1);
    let mut retained_by_group = vec![0; group_count];
    for (index, owner) in patch_of.iter().enumerate() {
        if let (true, Some(group)) = (keep[index], owner) {
            retained_by_group[*group] += 1;
        }
    }
    let mut index = 0;
    triangles.retain(|_| {
        index += 1;
        keep[index - 1]
    });

    // Topology after the merge.
    let mut patches: BTreeSet<usize> = BTreeSet::new();
    let mut shared = BTreeSet::new();
    let mut cross_reference_triangles = 0;
    let mut links = Vec::new();
    for triangle in triangles.iter() {
        let groups: Vec<Option<usize>> = [triangle.a, triangle.b, triangle.c]
            .iter()
            .map(|vertex| group_of(*vertex))
            .collect();
        patches.extend(groups.iter().flatten());
        if groups.iter().any(|group| *group != groups[0]) {
            cross_reference_triangles += 1;
            shared.extend([triangle.a, triangle.b, triangle.c]);
            let present: Vec<usize> = groups.iter().flatten().copied().collect();
            for pair in present.windows(2) {
                links.push((pair[0], pair[1]));
            }
        }
    }
    let patch_list: Vec<usize> = patches.iter().copied().collect();
    let mut parent: Vec<usize> = (0..patch_list.len()).collect();
    let slot = |group: usize| patch_list.binary_search(&group).ok();
    for (left, right) in links {
        if let (Some(left), Some(right)) = (slot(left), slot(right)) {
            let (left, right) = (find(&mut parent, left), find(&mut parent, right));
            parent[left] = right;
        }
    }
    let patch_components = (0..patch_list.len())
        .filter(|index| find(&mut parent, *index) == *index)
        .count();

    MergeStats {
        fused_pairs,
        shared_vertices: shared.len(),
        cross_reference_triangles,
        removed_duplicate_triangles,
        remaining_overlap_triangles,
        trimmed_seam_triangles,
        trimmed_seam_area_share: if total_area > 0.0 {
            trimmed_seam_area / total_area
        } else {
            0.0
        },
        retained_by_group,
        patch_components,
        meshed_patches: patch_list.len(),
    }
}

fn find(parent: &mut [usize], index: usize) -> usize {
    let mut root = index;
    while parent[root] != root {
        root = parent[root];
    }
    let mut node = index;
    while parent[node] != root {
        let next = parent[node];
        parent[node] = root;
        node = next;
    }
    root
}

struct TriangleGeometry {
    corners: [Vector3<f64>; 3],
    normal: Vector3<f64>,
    edges: [f64; 3],
}

impl TriangleGeometry {
    fn new(corners: [Vector3<f64>; 3]) -> Option<Self> {
        if corners
            .iter()
            .any(|corner| !corner.iter().all(|v| v.is_finite()))
        {
            return None;
        }
        let normal = (corners[1] - corners[0]).cross(&(corners[2] - corners[0]));
        let length = normal.norm();
        if !length.is_finite() || length <= GEOMETRIC_EPSILON {
            return None;
        }
        Some(Self {
            corners,
            normal: normal / length,
            edges: [
                (corners[1] - corners[0]).norm(),
                (corners[2] - corners[1]).norm(),
                (corners[0] - corners[2]).norm(),
            ],
        })
    }

    fn area(&self) -> f64 {
        (self.corners[1] - self.corners[0])
            .cross(&(self.corners[2] - self.corners[0]))
            .norm()
            * 0.5
    }

    fn grid_bounds(&self, cell: f64, padding: f64) -> [(i64, i64); 3] {
        std::array::from_fn(|axis| {
            let low = self
                .corners
                .iter()
                .map(|point| point[axis])
                .fold(f64::INFINITY, f64::min);
            let high = self
                .corners
                .iter()
                .map(|point| point[axis])
                .fold(f64::NEG_INFINITY, f64::max);
            (
                ((low - padding) / cell).floor() as i64,
                ((high + padding) / cell).floor() as i64,
            )
        })
    }

    /// Subtract supported higher-priority triangles from the candidate's area.
    /// Each subtraction is a planar polygon clipping operation, but the removed
    /// region must also agree in normal and depth at EVERY polygon corner.
    /// No output geometry is generated: partial coverage always keeps the face.
    fn covered_by<'a>(
        &self,
        surfaces: impl Iterator<Item = &'a TriangleGeometry>,
        tolerance: f64,
    ) -> (bool, bool) {
        let origin = self.corners[0];
        let u = (self.corners[1] - origin).normalize();
        let v = self.normal.cross(&u);
        let project = |point: Vector3<f64>| {
            let relative = point - origin;
            Vector2::new(relative.dot(&u), relative.dot(&v))
        };
        let area_epsilon = self.edges.iter().copied().fold(1.0e-12, f64::max)
            * self.edges.iter().copied().fold(1.0e-12, f64::max)
            * 1.0e-10;
        let mut remaining = vec![self.corners.map(&project).to_vec()];
        let mut partially_covered = false;
        for surface in surfaces {
            if self.normal.dot(&surface.normal).abs() < MIN_NORMAL_ALIGNMENT {
                continue;
            }
            let projected = surface.corners.map(&project);
            let winding = cross2(projected[1] - projected[0], projected[2] - projected[0]);
            if winding.abs() <= area_epsilon {
                continue;
            }
            let orientation = winding.signum();
            let mut next = Vec::new();
            for polygon in remaining {
                let mut inside = polygon;
                let mut outside = Vec::new();
                for edge in 0..3 {
                    let (included, excluded) = split_polygon(
                        &inside,
                        projected[edge],
                        projected[(edge + 1) % 3],
                        orientation,
                    );
                    if polygon_area(&excluded) > area_epsilon {
                        outside.push(excluded);
                    }
                    inside = included;
                    if inside.is_empty() {
                        break;
                    }
                }
                if polygon_area(&inside) > area_epsilon {
                    // Project back onto the candidate's plane and compare
                    // signed distances to the kept surface. Endpoint checks
                    // bound the entire linear distance over this convex piece.
                    let supported = inside.iter().all(|p| {
                        let point = origin + u * p.x + v * p.y;
                        (point - surface.corners[0]).dot(&surface.normal).abs() <= tolerance
                    });
                    if supported {
                        partially_covered = true;
                    } else {
                        outside.push(inside);
                    }
                }
                next.extend(outside);
                if next.len() > MAX_UNCOVERED_PIECES {
                    return (false, partially_covered);
                }
            }
            remaining = next;
            if remaining.is_empty() {
                return (true, true);
            }
        }
        (false, partially_covered)
    }
}

const OWNERSHIP_EPSILON: f64 = 1.0e-9;
/// Most image grid cells one projected triangle may span before it is skipped
/// (it cannot be owned; the 3D path still judges it).
const MAX_IMAGE_GRID_CELLS: usize = 4096;

/// The kept triangles of one reference patch in its own reference image.
/// Grid triangles of one patch tile its footprint without overlap, so owned
/// areas of different kept triangles add up.
struct Footprint {
    /// Support group of the patch.
    patch: usize,
    camera: ReferenceCamera,
    triangles: Vec<[(Vector2<f64>, f64); 3]>,
    cell: f64,
    cells: HashMap<(i64, i64), Vec<usize>>,
}

impl Footprint {
    fn new(patch: usize, camera: ReferenceCamera, kept: &[[Vector3<f64>; 3]]) -> Option<Self> {
        let triangles: Vec<[(Vector2<f64>, f64); 3]> = kept
            .iter()
            .filter_map(|corners| camera.project_triangle(corners))
            .filter(|projected| image_area(projected) > GEOMETRIC_EPSILON * GEOMETRIC_EPSILON)
            .collect();
        let mut edges: Vec<f64> = triangles
            .iter()
            .flat_map(|t| (0..3).map(move |i| (t[(i + 1) % 3].0 - t[i].0).norm()))
            .collect();
        edges.sort_by(f64::total_cmp);
        let cell = (edges.get(edges.len() / 2)? * 2.0).max(GEOMETRIC_EPSILON);
        let mut cells: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (index, triangle) in triangles.iter().enumerate() {
            if let Some(keys) = image_cells(triangle, cell) {
                for key in keys {
                    cells.entry(key).or_default().push(index);
                }
            }
        }
        Some(Self {
            patch,
            camera,
            triangles,
            cell,
            cells,
        })
    }

    /// Every overlap of the candidate's projection with a kept triangle: the
    /// overlap polygon (normalized image coordinates) and that kept triangle.
    fn overlaps(
        &self,
        corners: &[Vector3<f64>; 3],
    ) -> Option<(ProjectedTriangle, Vec<Overlap<'_>>)> {
        let candidate = self.camera.project_triangle(corners)?;
        if image_area(&candidate) <= GEOMETRIC_EPSILON * GEOMETRIC_EPSILON {
            return None;
        }
        let keys = image_cells(&candidate, self.cell)?;
        let nearby: BTreeSet<usize> = keys
            .iter()
            .filter_map(|key| self.cells.get(key))
            .flatten()
            .copied()
            .collect();
        let polygon: Vec<Vector2<f64>> = candidate.iter().map(|corner| corner.0).collect();
        let mut overlaps = Vec::new();
        for index in nearby {
            let kept = &self.triangles[index];
            let winding = cross2(kept[1].0 - kept[0].0, kept[2].0 - kept[0].0);
            let mut inside = polygon.clone();
            for edge in 0..3 {
                inside = split_polygon(
                    &inside,
                    kept[edge].0,
                    kept[(edge + 1) % 3].0,
                    winding.signum(),
                )
                .0;
                if inside.is_empty() {
                    break;
                }
            }
            if polygon_area(&inside) > 0.0 {
                overlaps.push((inside, kept));
            }
        }
        Some((candidate, overlaps))
    }

    /// Candidate-to-owner depth ratios at the centroids of overlaps that agree
    /// within `MAX_RELATIVE_SCALE_SAMPLE`.
    fn depth_ratios(&self, corners: &[Vector3<f64>; 3]) -> Vec<f64> {
        let Some((candidate, overlaps)) = self.overlaps(corners) else {
            return Vec::new();
        };
        overlaps
            .iter()
            .filter_map(|(polygon, kept)| {
                let centroid = polygon.iter().sum::<Vector2<f64>>() / polygon.len() as f64;
                let ratio = inverse_depth(kept, centroid)? / inverse_depth(&candidate, centroid)?;
                ((ratio - 1.0).abs() <= MAX_RELATIVE_SCALE_SAMPLE).then_some(ratio)
            })
            .collect()
    }

    /// The share of the candidate's projected area lying on kept triangles at
    /// the same depth once the candidate's relative depth `scale` is removed,
    /// and the surface area of those owned parts on the candidate's plane.
    fn ownership(&self, corners: &[Vector3<f64>; 3], scale: f64) -> Ownership {
        let Some((candidate, overlaps)) = self.overlaps(corners) else {
            return Ownership::default();
        };
        let total = image_area(&candidate);
        let plane = self.camera.plane(corners);
        let (mut image, mut surface) = (0.0, 0.0);
        for (polygon, kept) in &overlaps {
            // Inverse depth is affine in normalized image coordinates, so
            // checking every corner of the convex overlap bounds the
            // disagreement inside it.
            let agrees = polygon.iter().all(|point| {
                match (
                    inverse_depth(&candidate, *point),
                    inverse_depth(kept, *point),
                ) {
                    (Some(candidate), Some(kept)) => {
                        let (candidate, kept) = (1.0 / candidate, scale / kept);
                        (candidate - kept).abs() <= MAX_RELATIVE_DEPTH_DISAGREEMENT * kept
                    }
                    _ => false,
                }
            });
            if agrees {
                image += polygon_area(polygon);
                surface += plane.map_or(0.0, |plane| back_projected_area(polygon, plane));
            }
        }
        Ownership {
            image_share: (image / total).min(1.0),
            surface_area: surface,
        }
    }
}

/// A triangle in normalized image coordinates, with each corner's depth.
type ProjectedTriangle = [(Vector2<f64>, f64); 3];
/// An overlap polygon (normalized image coordinates) and the kept triangle it lies on.
type Overlap<'a> = (Vec<Vector2<f64>>, &'a ProjectedTriangle);

/// How much of a candidate triangle one reference owns.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Ownership {
    /// Share of the candidate's projected area.
    image_share: f64,
    /// Owned area on the candidate's plane, in world units.
    surface_area: f64,
}

/// Area of an image polygon lifted onto a camera-space plane `n · x = d`.
fn back_projected_area(polygon: &[Vector2<f64>], (normal, offset): (Vector3<f64>, f64)) -> f64 {
    let lifted: Option<Vec<Vector3<f64>>> = polygon
        .iter()
        .map(|point| {
            let ray = Vector3::new(point.x, point.y, 1.0);
            let along = normal.dot(&ray);
            (along.abs() > GEOMETRIC_EPSILON).then(|| ray * (offset / along))
        })
        .collect();
    let Some(lifted) = lifted else {
        return 0.0;
    };
    (0..lifted.len())
        .map(|index| lifted[index].cross(&lifted[(index + 1) % lifted.len()]))
        .sum::<Vector3<f64>>()
        .norm()
        * 0.5
}

fn image_area(triangle: &[(Vector2<f64>, f64); 3]) -> f64 {
    cross2(triangle[1].0 - triangle[0].0, triangle[2].0 - triangle[0].0).abs() * 0.5
}

/// Inverse depth of the triangle's plane through `point` (barycentric
/// interpolation, valid also slightly outside the triangle).
fn inverse_depth(triangle: &[(Vector2<f64>, f64); 3], point: Vector2<f64>) -> Option<f64> {
    let [(a, za), (b, zb), (c, zc)] = *triangle;
    let determinant = cross2(b - a, c - a);
    if determinant.abs() <= GEOMETRIC_EPSILON * GEOMETRIC_EPSILON {
        return None;
    }
    let wb = cross2(point - a, c - a) / determinant;
    let wc = cross2(b - a, point - a) / determinant;
    let inverse = (1.0 - wb - wc) / za + wb / zb + wc / zc;
    (inverse.is_finite() && inverse > 0.0).then_some(inverse)
}

fn image_cells(triangle: &[(Vector2<f64>, f64); 3], cell: f64) -> Option<Vec<(i64, i64)>> {
    let bounds: [(i64, i64); 2] = std::array::from_fn(|axis| {
        let values = triangle.map(|corner| corner.0[axis]);
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        ((low / cell).floor() as i64, (high / cell).floor() as i64)
    });
    let columns: usize = bounds[0]
        .1
        .checked_sub(bounds[0].0)?
        .checked_add(1)?
        .try_into()
        .ok()?;
    let rows: usize = bounds[1]
        .1
        .checked_sub(bounds[1].0)?
        .checked_add(1)?
        .try_into()
        .ok()?;
    if columns.checked_mul(rows)? > MAX_IMAGE_GRID_CELLS {
        return None;
    }
    Some(
        (bounds[1].0..=bounds[1].1)
            .flat_map(|y| (bounds[0].0..=bounds[0].1).map(move |x| (x, y)))
            .collect(),
    )
}

/// Enumerate all occupied AABB cells, including the triangle interior. If the
/// finite grid would be too large, let the caller use the exact unindexed list.
fn grid_cells(bounds: &[(i64, i64); 3]) -> Option<Vec<(i64, i64, i64)>> {
    let mut count = 1_usize;
    for &(first, last) in bounds {
        let span: usize = last.checked_sub(first)?.checked_add(1)?.try_into().ok()?;
        count = count.checked_mul(span)?;
        if count > MAX_GRID_CELLS {
            return None;
        }
    }
    let mut cells = Vec::with_capacity(count);
    for z in bounds[2].0..=bounds[2].1 {
        for y in bounds[1].0..=bounds[1].1 {
            for x in bounds[0].0..=bounds[0].1 {
                cells.push((x, y, z));
            }
        }
    }
    Some(cells)
}

fn cross2(a: Vector2<f64>, b: Vector2<f64>) -> f64 {
    a.x * b.y - a.y * b.x
}

fn polygon_area(points: &[Vector2<f64>]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    (0..points.len())
        .map(|index| cross2(points[index], points[(index + 1) % points.len()]))
        .sum::<f64>()
        .abs()
        * 0.5
}

fn split_polygon(
    polygon: &[Vector2<f64>],
    a: Vector2<f64>,
    b: Vector2<f64>,
    orientation: f64,
) -> (Vec<Vector2<f64>>, Vec<Vector2<f64>>) {
    let signed = |point: Vector2<f64>| orientation * cross2(b - a, point - a);
    let mut included = Vec::new();
    let mut excluded = Vec::new();
    if polygon.is_empty() {
        return (included, excluded);
    }
    for index in 0..polygon.len() {
        let previous = polygon[(index + polygon.len() - 1) % polygon.len()];
        let current = polygon[index];
        let before = signed(previous);
        let after = signed(current);
        if (before < 0.0 && after > 0.0) || (before > 0.0 && after < 0.0) {
            let intersection = previous + (current - previous) * (before / (before - after));
            included.push(intersection);
            excluded.push(intersection);
        }
        if after >= 0.0 {
            included.push(current);
        }
        if after <= 0.0 {
            excluded.push(current);
        }
    }
    (included, excluded)
}

#[cfg(test)]
#[path = "surface_merge_tests.rs"]
mod tests;
