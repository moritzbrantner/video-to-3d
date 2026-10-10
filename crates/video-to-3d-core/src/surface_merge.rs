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
//! - **No duplicated overlap.** A lower-priority triangle is removed only if
//!   its entire area is covered by higher-priority, aligned, depth-consistent
//!   triangles from a patch with a fusion-accepted link. Triangles spanning
//!   holes or boundaries stay to preserve uniquely observed geometry.
//!   so the union keeps every observed area once; a gap no patch covered stays
//!   a gap, and a surface farther away than the tolerance is never merged.
//! - **Cross-reference topology.** Triangles referencing accepted shared
//!   vertices retain cross-patch connections wherever they extend coverage.
//!
//! Patch priority is deterministic: more accepted triangles first, then the
//! lower support-group index.

use crate::{MeshTriangle, Point3};
use nalgebra::{Vector2, Vector3};
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MergeStats {
    pub fused_pairs: usize,
    pub shared_vertices: usize,
    pub cross_reference_triangles: usize,
    /// Lower-priority triangles dropped because a kept patch already meshes
    /// that surface.
    pub removed_duplicate_triangles: usize,
    /// Partially covered lower-priority triangles kept to preserve accepted surface.
    pub remaining_overlap_triangles: usize,
    /// Connected components of the patch graph (patches joined by
    /// cross-reference triangles); `0` without meshed patches.
    pub patch_components: usize,
    pub meshed_patches: usize,
}

impl MergeStats {
    pub fn diagnostic(&self, support_groups: usize) -> String {
        format!(
            "Overlapping reference patches: {} fused seam pair(s) became shared vertices ({} shared vertices, {} cross-reference triangles); {} duplicated overlap triangle(s) were dropped in favour of the patch already meshing that surface and {} partially overlapping triangle(s) were retained to preserve supported geometry. {} of {} camera-support group(s) carry triangles, forming {} connected patch component(s). Only fusion-accepted pairs were merged; no vertex was created or moved and gaps no patch covered stay open.",
            self.fused_pairs,
            self.shared_vertices,
            self.cross_reference_triangles,
            self.removed_duplicate_triangles,
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
    // Only patch pairs with an explicitly fusion-accepted correspondence may
    // deduplicate faces. Proximity alone is not evidence of a shared surface.
    let linked_patches: BTreeSet<(usize, usize)> = pairs
        .iter()
        .filter_map(|&(a, b)| {
            if a >= points.len() || b >= points.len() || canonical[a] != canonical[b] {
                return None;
            }
            let (Some(a), Some(b)) = (group_of(a), group_of(b)) else {
                return None;
            };
            (a != b).then_some((a.min(b), a.max(b)))
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

    for (patch_rank, patch) in order.iter().enumerate() {
        let members: Vec<usize> = patch_of
            .iter()
            .enumerate()
            .filter_map(|(index, owner)| (*owner == Some(*patch)).then_some(index))
            .collect();
        if patch_rank > 0 {
            for &index in &members {
                let Some(candidate) = &geometry[index] else {
                    continue;
                };
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
                        let Some(other_patch) = patch_of[other] else {
                            return false;
                        };
                        linked_patches.contains(&(
                            usize::min(*patch, other_patch),
                            usize::max(*patch, other_patch),
                        ))
                    })
                    .filter_map(|other| geometry[other].as_ref());
                let (covered, partially_covered) = candidate.covered_by(candidates, tolerance);
                if covered {
                    keep[index] = false;
                    removed_duplicate_triangles += 1;
                } else if partially_covered {
                    // In the absence of proven full coverage keep the original
                    // face: it may be the only evidence at a hole or boundary.
                    remaining_overlap_triangles += 1;
                }
            }
        }
        // Do not index this patch until it has been processed: triangles of a
        // single camera patch cannot erase each other.
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
        patch_components,
        meshed_patches: patch_list.len(),
    }
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
        let mut remaining = vec![self.corners.map(|corner| project(corner)).to_vec()];
        let mut partially_covered = false;
        for surface in surfaces {
            if self.normal.dot(&surface.normal).abs() < MIN_NORMAL_ALIGNMENT {
                continue;
            }
            let projected = surface.corners.map(|corner| project(corner));
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
