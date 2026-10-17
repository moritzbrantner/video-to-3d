//! Re-stitching trimmed overlap seams (#137).
//!
//! Image ownership removes a lower-priority seam triangle once a linked
//! higher-priority reference owns at least [`MIN_IMAGE_OWNERSHIP_SHARE`] of
//! it, which leaves a thin gap between the owner patch's boundary and the
//! remaining lower-priority boundary. This pass closes that gap with
//! cross-reference triangles over existing vertices, fail-closed: a stitch
//! triangle joins a retained boundary edge of one side to a retained boundary
//! vertex of the other, and is accepted only if, in the owner reference
//! camera's image,
//!
//! - it lies (up to [`MAX_UNOBSERVED_STITCH_SHARE`]) on the removed seam
//!   triangles of that owner and lower patch, at their depth within
//!   [`MAX_RELATIVE_DEPTH_DISAGREEMENT`] after removing the pair's relative
//!   depth scale: only area the removed triangles observed is bridged;
//! - it overlaps no retained triangle near it in 3D and no earlier stitch
//!   (up to [`MAX_STITCH_OVERLAP_SHARE`]);
//! - it keeps the orientation of the retained triangle across its boundary
//!   edge, and no edge ends up shared by more than two triangles.
//!
//! Vertices referenced only by removed triangles are never used, and only
//! owner vertices of components linked to the removed triangle's component
//! (the linking ownership uses) take part. Boundary edges are visited in
//! vertex order and each takes its shortest-perimeter valid candidate, so the
//! result is deterministic. A seam whose lower side still overlaps the owner
//! (an overlap seam rather than a gap) has no room for a stitch and stays
//! disconnected.
//!
//! Work and memory are bounded: every lookup takes postings from a
//! [`PostingBudget`], each removed triangle offers at most
//! [`MAX_SEAM_VERTICES`] vertices, each vertex lies on at most
//! [`MAX_SEAMS_PER_VERTEX`] seams, and there are at most as many stitches as
//! removed triangles. Mesh indices are stored as `u32`; a mesh beyond that
//! range is not stitched.

#[cfg(doc)]
use super::MIN_IMAGE_OWNERSHIP_SHARE;
use super::{
    cross2, image_aabb, inverse_depth, polygon_area, split_polygon, Footprint, LevelGrid,
    PostingBudget, PostingTally, ProjectedTriangle, QueryLimit, TriangleGeometry, MAX_GRID_CELLS,
    MAX_IMAGE_GRID_CELLS, MAX_RELATIVE_DEPTH_DISAGREEMENT, MAX_UNCOVERED_PIECES,
};
use crate::MeshTriangle;
use nalgebra::{Vector2, Vector3};
use std::collections::{BTreeSet, HashMap};
use std::mem::size_of;

/// Most stitch vertices (its own retained boundary corners and the owner's
/// boundary vertices on it) one removed seam triangle may offer; a seam
/// triangle with more is not stitched.
pub(super) const MAX_SEAM_VERTICES: usize = 6;
/// Most removed seam triangles (lowest removal order first) one vertex is
/// associated with.
pub(super) const MAX_SEAMS_PER_VERTEX: usize = 16;
/// Largest share of a stitch triangle's image area that may lie off the
/// removed seam triangles (or at another depth): numerical slack only, so a
/// gap no removed triangle observed is never bridged.
pub(super) const MAX_UNOBSERVED_STITCH_SHARE: f64 = 1.0e-6;
/// Largest share of a stitch triangle's image area that may overlap retained
/// triangles or earlier stitches: numerical slack only.
pub(super) const MAX_STITCH_OVERLAP_SHARE: f64 = 1.0e-6;
/// Barycentric slack for an owner vertex lying on a seam triangle's edge.
const INSIDE_TOLERANCE: f64 = 1.0e-6;

/// A lower-priority triangle removed because one owner's image owned it.
#[derive(Clone, Copy, Debug)]
pub(super) struct ImageRemoval {
    /// Index of the removed triangle in the merge's input buffer.
    pub triangle: usize,
    /// The owning [`Footprint`], by position in the merge's footprint list.
    pub footprint: usize,
    /// The removed triangle's patch depth scale relative to the owner.
    pub scale: f64,
}

/// One retained triangle's directed edge, keyed by its undirected endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HalfEdge {
    low: u32,
    high: u32,
    from: u32,
    triangle: u32,
}

/// Use of an undirected edge once stitches touched it.
#[derive(Clone, Copy, Debug)]
struct EdgeUse {
    count: u32,
    /// Start of the edge as the first triangle using it traverses it.
    from: u32,
}

/// The stitch vertices one removed seam triangle offers.
#[derive(Clone, Copy, Debug)]
struct Seam {
    vertices: [u32; MAX_SEAM_VERTICES],
    len: u8,
}

impl Seam {
    fn vertices(&self) -> impl Iterator<Item = usize> + '_ {
        self.vertices[..usize::from(self.len)]
            .iter()
            .map(|&vertex| vertex as usize)
    }
}

pub(super) struct SeamInputs<'a> {
    pub vertices: usize,
    pub triangles: &'a [MeshTriangle],
    pub keep: &'a [bool],
    pub geometry: &'a [Option<TriangleGeometry>],
    pub membership: &'a [Option<usize>],
    pub patch_of: &'a [Option<usize>],
    pub footprints: &'a [Footprint],
    pub removals: &'a [ImageRemoval],
    pub kept_grid: &'a LevelGrid<3>,
    pub all_kept: &'a [usize],
    pub tolerance: f64,
    pub cell: f64,
}

/// Conservative peak bytes [`stitch_trimmed_seams`] allocates for a mesh of
/// `vertices` points and `triangles` triangles, charged like
/// [`super::working_set_bytes`]: three half-edges per triangle, at most one
/// removal (and so one stitch) per triangle, and per-edge candidate scratch
/// bounded by the per-vertex seam limit.
pub(super) fn working_set_bytes(vertices: usize, triangles: usize) -> usize {
    const fn hash_entry(payload: usize) -> usize {
        payload + 4 * size_of::<usize>() + 32
    }
    let per_vertex = size_of::<bool>(); // boundary flag
    let per_triangle = 3 * size_of::<HalfEdge>()
        + size_of::<Option<Seam>>()
        + MAX_SEAM_VERTICES * size_of::<(u32, u32)>() // seams by vertex
        + 3 * size_of::<u32>() // boundary edge list
        // One stitch per removal: the triangle, its edges' uses, one grid
        // posting and bucket.
        + size_of::<MeshTriangle>()
        + 3 * hash_entry(size_of::<(u32, u32)>() + size_of::<EdgeUse>())
        + size_of::<usize>()
        + hash_entry(size_of::<(i32, [i64; 3])>() + size_of::<Vec<usize>>());
    // Per-edge scratch: the seams at both ends and their candidates.
    let scratch = 2
        * MAX_SEAMS_PER_VERTEX
        * (btree_entry(size_of::<usize>())
            + MAX_SEAM_VERTICES * (btree_entry(size_of::<usize>()) + 2 * size_of::<[usize; 4]>()));
    vertices
        .saturating_mul(per_vertex)
        .saturating_add(triangles.saturating_mul(per_triangle))
        .saturating_add(scratch)
}

const fn btree_entry(payload: usize) -> usize {
    payload + 4 * size_of::<usize>() + 32
}

/// Signed area of a projected triangle (counter-clockwise positive).
fn signed_image_area(triangle: &ProjectedTriangle) -> f64 {
    let [(a, _), (b, _), (c, _)] = *triangle;
    cross2(b - a, c - a) * 0.5
}

/// Whether `point` lies in the projected triangle, edges included.
fn contains(triangle: &ProjectedTriangle, point: Vector2<f64>) -> bool {
    let [(a, _), (b, _), (c, _)] = *triangle;
    let determinant = cross2(b - a, c - a);
    if determinant.abs() <= f64::EPSILON {
        return false;
    }
    let wb = cross2(point - a, c - a) / determinant;
    let wc = cross2(b - a, point - a) / determinant;
    [1.0 - wb - wc, wb, wc]
        .iter()
        .all(|weight| *weight >= -INSIDE_TOLERANCE)
}

fn polygon_of(triangle: &ProjectedTriangle) -> Vec<Vector2<f64>> {
    triangle.iter().map(|corner| corner.0).collect()
}

/// Area of the polygon's intersection with a projected triangle.
fn intersection_area(polygon: &[Vector2<f64>], triangle: &ProjectedTriangle) -> f64 {
    let orientation = signed_image_area(triangle).signum();
    let mut inside = polygon.to_vec();
    for edge in 0..3 {
        inside = split_polygon(
            &inside,
            triangle[edge].0,
            triangle[(edge + 1) % 3].0,
            orientation,
        )
        .0;
        if inside.is_empty() {
            return 0.0;
        }
    }
    polygon_area(&inside)
}

/// Image area of `stitch` that does not lie on `seams` at their depth; `None`
/// when the subtraction exceeds [`MAX_UNCOVERED_PIECES`] fragments.
fn unobserved_area(stitch: &ProjectedTriangle, seams: &[ProjectedTriangle]) -> Option<f64> {
    let area_epsilon = signed_image_area(stitch).abs() * 1.0e-12;
    let mut remaining = vec![polygon_of(stitch)];
    for seam in seams {
        let orientation = signed_image_area(seam).signum();
        let mut next = Vec::new();
        for polygon in remaining {
            let mut inside = polygon;
            let mut outside = Vec::new();
            for edge in 0..3 {
                let (included, excluded) =
                    split_polygon(&inside, seam[edge].0, seam[(edge + 1) % 3].0, orientation);
                if polygon_area(&excluded) > area_epsilon {
                    outside.push(excluded);
                }
                inside = included;
                if inside.is_empty() {
                    break;
                }
            }
            if polygon_area(&inside) > area_epsilon {
                // Inverse depth is affine on the convex piece of both
                // triangles, so the corners bound the disagreement.
                let agrees = inside.iter().all(|point| {
                    match (inverse_depth(stitch, *point), inverse_depth(seam, *point)) {
                        (Some(stitch), Some(seam)) => {
                            let (stitch, seam) = (1.0 / stitch, 1.0 / seam);
                            (stitch - seam).abs() <= MAX_RELATIVE_DEPTH_DISAGREEMENT * seam
                        }
                        _ => false,
                    }
                });
                if !agrees {
                    outside.push(inside);
                }
            }
            next.extend(outside);
            if next.len() > MAX_UNCOVERED_PIECES {
                return None;
            }
        }
        remaining = next;
        if remaining.is_empty() {
            break;
        }
    }
    Some(remaining.iter().map(|polygon| polygon_area(polygon)).sum())
}

/// Retained edges, sorted, and the uses stitches added.
struct Edges {
    retained: Vec<HalfEdge>,
    stitched: HashMap<(u32, u32), EdgeUse>,
}

impl Edges {
    fn key(a: u32, b: u32) -> (u32, u32) {
        (a.min(b), a.max(b))
    }

    /// The retained half-edges of an undirected edge.
    fn retained(&self, a: u32, b: u32) -> &[HalfEdge] {
        let (low, high) = Self::key(a, b);
        let start = self
            .retained
            .partition_point(|edge| (edge.low, edge.high) < (low, high));
        let end = self
            .retained
            .partition_point(|edge| (edge.low, edge.high) <= (low, high));
        &self.retained[start..end]
    }

    /// The edge's use by retained triangles and stitches.
    fn usage(&self, a: u32, b: u32) -> Option<EdgeUse> {
        self.stitched.get(&Self::key(a, b)).copied().or_else(|| {
            let retained = self.retained(a, b);
            retained.first().map(|first| EdgeUse {
                count: retained.len() as u32,
                from: first.from,
            })
        })
    }

    fn add(&mut self, from: u32, to: u32) {
        let usage = self
            .usage(from, to)
            .map_or(EdgeUse { count: 1, from }, |usage| EdgeUse {
                count: usage.count + 1,
                ..usage
            });
        self.stitched.insert(Self::key(from, to), usage);
    }
}

/// Stitch triangles closing the gaps trimmed seams left, to append to the
/// merged triangle buffer.
pub(super) fn stitch_trimmed_seams(
    inputs: &SeamInputs<'_>,
    position: &dyn Fn(usize) -> Vector3<f64>,
    linked: &dyn Fn(usize, usize) -> bool,
    tally: &mut PostingTally,
) -> Vec<MeshTriangle> {
    let SeamInputs {
        vertices,
        triangles,
        keep,
        geometry,
        membership,
        patch_of,
        footprints,
        removals,
        cell,
        ..
    } = *inputs;
    if removals.is_empty()
        || u32::try_from(vertices).is_err()
        || u32::try_from(triangles.len()).is_err()
        || u32::try_from(removals.len()).is_err()
    {
        return Vec::new();
    }
    let group_of = |vertex: usize| membership.get(vertex).copied().flatten();

    // Retained half-edges, sorted by undirected edge and triangle.
    let mut retained = Vec::with_capacity(3 * triangles.len());
    for (index, triangle) in triangles.iter().enumerate() {
        if !keep[index] {
            continue;
        }
        let [a, b, c] = [triangle.a, triangle.b, triangle.c].map(|vertex| vertex as u32);
        for (from, to) in [(a, b), (b, c), (c, a)] {
            if from != to {
                let (low, high) = Edges::key(from, to);
                retained.push(HalfEdge {
                    low,
                    high,
                    from,
                    triangle: index as u32,
                });
            }
        }
    }
    retained.sort_unstable();
    let mut edges = Edges {
        retained,
        stitched: HashMap::new(),
    };
    let mut on_boundary = vec![false; vertices];
    let mut boundary = Vec::new();
    let mut start = 0;
    while start < edges.retained.len() {
        let edge = edges.retained[start];
        let end = start
            + edges.retained[start..]
                .iter()
                .take_while(|other| (other.low, other.high) == (edge.low, edge.high))
                .count();
        if end - start == 1 {
            on_boundary[edge.low as usize] = true;
            on_boundary[edge.high as usize] = true;
            boundary.push(start as u32);
        }
        start = end;
    }

    // Each removed seam triangle with the retained boundary vertices it may
    // stitch: its own corners and the owner's (linked) boundary vertices on it
    // in the owner's image.
    let mut seams: Vec<Option<Seam>> = Vec::with_capacity(removals.len());
    let mut seams_by_vertex: Vec<(u32, u32)> = Vec::new();
    for (slot, removal) in removals.iter().enumerate() {
        let seam = (|| {
            let footprint = footprints.get(removal.footprint)?;
            let projected =
                footprint.project_candidate(&geometry[removal.triangle].as_ref()?.corners)?;
            let lower = patch_of[removal.triangle];
            let removed = &triangles[removal.triangle];
            let mut stitchable: BTreeSet<usize> = [removed.a, removed.b, removed.c]
                .into_iter()
                .filter(|&vertex| on_boundary[vertex] && group_of(vertex).is_some())
                .collect();
            let mut nearby = BTreeSet::new();
            let (low, high) = image_aabb(&projected);
            let mut budget = PostingBudget::per_query();
            let found = footprint.grid.query_bounded(
                low,
                high,
                MAX_IMAGE_GRID_CELLS,
                &mut budget,
                &mut nearby,
            );
            tally.record(&budget, &found);
            found.ok()?;
            for kept in nearby {
                if !linked(removal.triangle, footprint.components[kept]) {
                    continue;
                }
                for vertex in footprint.vertices[kept] {
                    if !on_boundary[vertex]
                        || group_of(vertex).is_none()
                        || group_of(vertex) == lower
                        || stitchable.contains(&vertex)
                    {
                        continue;
                    }
                    let inside = footprint
                        .camera
                        .project(position(vertex))
                        .is_some_and(|(image, _)| contains(&projected, image));
                    if inside {
                        stitchable.insert(vertex);
                    }
                }
            }
            // Fail closed on a pathological seam rather than truncating it.
            if stitchable.len() > MAX_SEAM_VERTICES {
                return None;
            }
            let mut seam = Seam {
                vertices: [0; MAX_SEAM_VERTICES],
                len: stitchable.len() as u8,
            };
            for (target, vertex) in seam.vertices.iter_mut().zip(stitchable) {
                *target = vertex as u32;
            }
            Some(seam)
        })();
        if let Some(seam) = &seam {
            seams_by_vertex.extend(seam.vertices().map(|vertex| (vertex as u32, slot as u32)));
        }
        seams.push(seam);
    }
    seams_by_vertex.sort_unstable();

    let context = StitchContext {
        inputs,
        position,
        seams: &seams,
        seams_by_vertex: &seams_by_vertex,
    };
    let length = |a: usize, b: usize| (position(a) - position(b)).norm();
    let mut stitch_grid = LevelGrid::<3>::new(cell);
    let mut output: Vec<MeshTriangle> = Vec::new();
    for edge in boundary {
        if output.len() >= removals.len() {
            break;
        }
        let edge = edges.retained[edge as usize];
        let adjacent = edge.triangle as usize;
        let from = edge.from as usize;
        let to = if edge.from == edge.low {
            edge.high
        } else {
            edge.low
        } as usize;
        if group_of(from) != group_of(to) {
            continue;
        }
        // Candidates: the boundary edge reversed (to keep the orientation of
        // the triangle across it), then a vertex of the other side offered by
        // a seam at either end of the edge, shortest perimeter first.
        let slots: BTreeSet<usize> = seams_of(&seams_by_vertex, from)
            .chain(seams_of(&seams_by_vertex, to))
            .collect();
        let mut seen = BTreeSet::new();
        let mut candidates: Vec<(f64, [usize; 4])> = Vec::new();
        for slot in slots {
            let Some(seam) = &seams[slot] else {
                continue;
            };
            let lower = patch_of[removals[slot].triangle];
            let edge_is_lower = group_of(from) == lower;
            for vertex in seam.vertices() {
                if vertex == from
                    || vertex == to
                    || (group_of(vertex) == lower) == edge_is_lower
                    || !seen.insert(vertex)
                {
                    continue;
                }
                let perimeter = length(from, to) + length(to, vertex) + length(vertex, from);
                candidates.push((perimeter, [to, from, vertex, slot]));
            }
        }
        candidates.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        for (_, [a, b, c, slot]) in candidates {
            let stitch = Stitch {
                corners: [a, b, c],
                removal: slot,
                adjacent,
            };
            let Some((low, high)) = context.check(&stitch, &edges, &stitch_grid, &output, tally)
            else {
                continue;
            };
            for (from, to) in [(a, b), (b, c), (c, a)] {
                edges.add(from as u32, to as u32);
            }
            stitch_grid.insert(output.len(), low, high);
            output.push(MeshTriangle {
                a,
                b,
                c,
                confidence: triangles[adjacent]
                    .confidence
                    .min(triangles[removals[slot].triangle].confidence),
            });
            break;
        }
    }
    output
}

/// The removals (seam slots) a vertex lies on, lowest first, at most
/// [`MAX_SEAMS_PER_VERTEX`].
fn seams_of(by_vertex: &[(u32, u32)], vertex: usize) -> impl Iterator<Item = usize> + '_ {
    let vertex = vertex as u32;
    let start = by_vertex.partition_point(|&(other, _)| other < vertex);
    by_vertex[start..]
        .iter()
        .take_while(move |&&(other, _)| other == vertex)
        .take(MAX_SEAMS_PER_VERTEX)
        .map(|&(_, slot)| slot as usize)
}

/// A proposed stitch: its corners (the reversed boundary edge, then the
/// other side's vertex), the removal whose seam proposed it, and the retained
/// triangle across its boundary edge.
struct Stitch {
    corners: [usize; 3],
    removal: usize,
    adjacent: usize,
}

struct StitchContext<'a> {
    inputs: &'a SeamInputs<'a>,
    position: &'a dyn Fn(usize) -> Vector3<f64>,
    seams: &'a [Option<Seam>],
    seams_by_vertex: &'a [(u32, u32)],
}

impl StitchContext<'_> {
    fn project_seam(&self, removal: &ImageRemoval) -> Option<ProjectedTriangle> {
        let footprint = self.inputs.footprints.get(removal.footprint)?;
        footprint.project_candidate(&self.inputs.geometry[removal.triangle].as_ref()?.corners)
    }

    /// The stitch's padded 3D bounds when it passes every check.
    fn check(
        &self,
        stitch: &Stitch,
        edges: &Edges,
        stitch_grid: &LevelGrid<3>,
        stitches: &[MeshTriangle],
        tally: &mut PostingTally,
    ) -> Option<([f64; 3], [f64; 3])> {
        let SeamInputs {
            geometry,
            membership,
            patch_of,
            footprints,
            removals,
            kept_grid,
            all_kept,
            tolerance,
            ..
        } = *self.inputs;
        let group_of = |vertex: usize| membership.get(vertex).copied().flatten();
        let [a, b, c] = stitch.corners;
        // Topology: every edge stays shared by at most two triangles, which
        // traverse it in opposite directions.
        let manifold = [(a, b), (b, c), (c, a)].iter().all(|&(from, to)| {
            edges
                .usage(from as u32, to as u32)
                .is_none_or(|usage| usage.count < 2 && usage.from != from as u32)
        });
        if !manifold {
            return None;
        }
        let surface = TriangleGeometry::new(stitch.corners.map(self.position))?;
        let removal = &removals[stitch.removal];
        let footprint = &footprints[removal.footprint];
        let lower = patch_of[removal.triangle];
        let mut projected = footprint.camera.project_triangle(&surface.corners)?;
        let area = signed_image_area(&projected);
        let adjacent = geometry[stitch.adjacent]
            .as_ref()
            .and_then(|adjacent| footprint.camera.project_triangle(&adjacent.corners))
            .map(|adjacent| signed_image_area(&adjacent))?;
        // The stitch lies across its boundary edge, with the winding of the
        // retained triangle on the other side.
        if area.abs() <= f64::EPSILON * f64::EPSILON || adjacent.signum() != area.signum() {
            return None;
        }
        let area = area.abs();

        // Observed: on the removed seam triangles of this owner and lower
        // patch, at their depth (owner-side corners carry the owner's depth
        // scale, which the pair's relative scale removes).
        for (corner, vertex) in projected.iter_mut().zip(stitch.corners) {
            if group_of(vertex) != lower {
                corner.1 *= removal.scale;
            }
        }
        let cover: BTreeSet<usize> = stitch
            .corners
            .iter()
            .flat_map(|&vertex| seams_of(self.seams_by_vertex, vertex))
            .filter(|&slot| {
                self.seams[slot].is_some()
                    && removals[slot].footprint == removal.footprint
                    && patch_of[removals[slot].triangle] == lower
            })
            .collect();
        let cover: Vec<ProjectedTriangle> = cover
            .into_iter()
            .filter_map(|slot| self.project_seam(&removals[slot]))
            .collect();
        if unobserved_area(&projected, &cover)? > MAX_UNOBSERVED_STITCH_SHARE * area {
            return None;
        }

        // No overlap, in the owner's image, with retained triangles near it in
        // 3D or with earlier stitches. A neighbour the image cannot project
        // fails the stitch closed.
        let polygon = polygon_of(&projected);
        let (low, high) = surface.aabb(tolerance);
        let mut nearby = BTreeSet::new();
        let mut budget = PostingBudget::per_query();
        let found =
            match kept_grid.query_bounded(low, high, MAX_GRID_CELLS, &mut budget, &mut nearby) {
                Err(QueryLimit::Cells) => budget.take_into(all_kept, &mut nearby),
                found => found,
            };
        tally.record(&budget, &found);
        found.ok()?;
        let mut overlap = 0.0;
        for other in nearby {
            let Some(other) = &geometry[other] else {
                continue;
            };
            overlap += intersection_area(
                &polygon,
                &footprint.camera.project_triangle(&other.corners)?,
            );
        }
        let mut nearby = BTreeSet::new();
        let mut budget = PostingBudget::per_query();
        let found = stitch_grid.query_bounded(low, high, MAX_GRID_CELLS, &mut budget, &mut nearby);
        tally.record(&budget, &found);
        found.ok()?;
        for other in nearby {
            let other = &stitches[other];
            let corners = [other.a, other.b, other.c].map(self.position);
            overlap += intersection_area(&polygon, &footprint.camera.project_triangle(&corners)?);
        }
        (overlap <= MAX_STITCH_OVERLAP_SHARE * area).then_some((low, high))
    }
}
