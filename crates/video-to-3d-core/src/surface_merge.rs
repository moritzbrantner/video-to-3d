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
//! - **No duplicated overlap.** A lower-priority triangle whose centroid lies on
//!   a kept higher-priority triangle (inside it, within a fraction of the median
//!   edge length along its normal, with aligned normals) is the same surface
//!   meshed twice and is dropped. Triangles over the other patch's holes stay,
//!   so the union keeps every observed area once; a gap no patch covered stays
//!   a gap, and a surface farther away than the tolerance is never merged.
//! - **Cross-reference topology.** Lower-priority triangles that use a shared
//!   vertex are kept even when they overlap, so the seam stays connected.
//!
//! Patch priority is deterministic: more accepted triangles first, then the
//! lower support-group index.

use crate::{MeshTriangle, Point3};
use nalgebra::Vector3;
use std::collections::{BTreeSet, HashMap};

const GEOMETRIC_EPSILON: f64 = 1.0e-9;
/// Largest distance along the kept triangle's normal, as a fraction of the
/// median accepted edge length, at which a triangle counts as the same surface.
const MAX_OVERLAP_DISTANCE_FRACTION: f64 = 0.5;
/// Barycentric slack so a centroid on a shared edge of two kept triangles is
/// still inside one of them.
const BARYCENTRIC_SLACK: f64 = 1.0e-6;
const MIN_NORMAL_ALIGNMENT: f64 = 0.94;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MergeStats {
    pub fused_pairs: usize,
    pub shared_vertices: usize,
    pub cross_reference_triangles: usize,
    /// Lower-priority triangles dropped because a kept patch already meshes
    /// that surface.
    pub removed_duplicate_triangles: usize,
    /// Lower-priority triangles that still overlap a kept patch because they
    /// carry a shared seam vertex.
    pub remaining_overlap_triangles: usize,
    /// Connected components of the patch graph (patches joined by
    /// cross-reference triangles); `0` without meshed patches.
    pub patch_components: usize,
    pub meshed_patches: usize,
}

impl MergeStats {
    pub fn diagnostic(&self, support_groups: usize) -> String {
        format!(
            "Overlapping reference patches: {} fused seam pair(s) became shared vertices ({} shared vertices, {} cross-reference triangles); {} duplicated overlap triangle(s) were dropped in favour of the patch already meshing that surface and {} seam triangle(s) still overlap it. {} of {} camera-support group(s) carry triangles, forming {} connected patch component(s). Only fusion-accepted pairs were merged; no vertex was created or moved and gaps no patch covered stay open.",
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
    let mut uses_shared = vec![false; triangles.len()];
    for (index, triangle) in triangles.iter_mut().enumerate() {
        let mapped = [triangle.a, triangle.b, triangle.c].map(|vertex| canonical[vertex]);
        // A collapsed triangle keeps its own vertices instead.
        if mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[0] == mapped[2] {
            continue;
        }
        uses_shared[index] = mapped != [triangle.a, triangle.b, triangle.c];
        [triangle.a, triangle.b, triangle.c] = mapped;
    }

    // Drop lower-priority triangles that mesh a kept surface a second time.
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
    let tolerance = median_edge * MAX_OVERLAP_DISTANCE_FRACTION;
    let cell = median_edge.max(GEOMETRIC_EPSILON);
    let cell_of = |point: Vector3<f64>| {
        (
            (point.x / cell).floor() as i64,
            (point.y / cell).floor() as i64,
            (point.z / cell).floor() as i64,
        )
    };

    let mut keep = vec![true; triangles.len()];
    let mut kept_cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut removed_duplicate_triangles = 0;
    let mut remaining_overlap_triangles = 0;
    for (patch_rank, patch) in order.iter().enumerate() {
        let members: Vec<usize> = (0..triangles.len())
            .filter(|index| patch_of[*index] == Some(*patch))
            .collect();
        if patch_rank > 0 {
            for &index in &members {
                let Some(candidate) = &geometry[index] else {
                    continue;
                };
                let (cx, cy, cz) = cell_of(candidate.centroid);
                let reach = (tolerance / cell).ceil() as i64 + 1;
                let mut covered = false;
                'search: for dz in -reach..=reach {
                    for dy in -reach..=reach {
                        for dx in -reach..=reach {
                            let Some(kept) = kept_cells.get(&(cx + dx, cy + dy, cz + dz)) else {
                                continue;
                            };
                            for &other in kept {
                                let Some(surface) = &geometry[other] else {
                                    continue;
                                };
                                if surface.covers(candidate, tolerance) {
                                    covered = true;
                                    break 'search;
                                }
                            }
                        }
                    }
                }
                if !covered {
                    continue;
                }
                if uses_shared[index] {
                    remaining_overlap_triangles += 1;
                } else {
                    keep[index] = false;
                    removed_duplicate_triangles += 1;
                }
            }
        }
        // Index every vertex cell of the kept triangles so a centroid finds
        // large triangles too.
        for &index in &members {
            if !keep[index] {
                continue;
            }
            let Some(surface) = &geometry[index] else {
                continue;
            };
            let mut cells = BTreeSet::new();
            for corner in surface.corners.iter().chain([&surface.centroid]) {
                cells.insert(cell_of(*corner));
            }
            for key in cells {
                kept_cells.entry(key).or_default().push(index);
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
    centroid: Vector3<f64>,
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
            centroid: (corners[0] + corners[1] + corners[2]) / 3.0,
            normal: normal / length,
            edges: [
                (corners[1] - corners[0]).norm(),
                (corners[2] - corners[1]).norm(),
                (corners[0] - corners[2]).norm(),
            ],
        })
    }

    /// Whether `other`'s centroid lies on this triangle: within `tolerance`
    /// along this normal, inside it, and with aligned normals.
    fn covers(&self, other: &TriangleGeometry, tolerance: f64) -> bool {
        if self.normal.dot(&other.normal).abs() < MIN_NORMAL_ALIGNMENT {
            return false;
        }
        let offset = other.centroid - self.corners[0];
        if self.normal.dot(&offset).abs() > tolerance {
            return false;
        }
        let projected = other.centroid - self.normal * self.normal.dot(&offset);
        let [a, b, c] = self.corners;
        let area = |p: Vector3<f64>, q: Vector3<f64>, r: Vector3<f64>| {
            (q - p).cross(&(r - p)).dot(&self.normal)
        };
        let total = area(a, b, c);
        if total.abs() <= GEOMETRIC_EPSILON {
            return false;
        }
        [
            area(projected, b, c),
            area(a, projected, c),
            area(a, b, projected),
        ]
        .iter()
        .all(|part| part / total >= -BARYCENTRIC_SLACK)
    }
}

#[cfg(test)]
#[path = "surface_merge_tests.rs"]
mod tests;
