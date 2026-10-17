//! Clipping retained overlap seams (#137).
//!
//! Image ownership removes a lower-priority triangle that a linked owner
//! reference owns at least [`super::MIN_IMAGE_OWNERSHIP_SHARE`]; seam stitching
//! closes the gaps such removals leave over existing vertices. A triangle
//! owned only in part stays, so it still covers part of the owner's surface
//! twice and is not connected to it: an *overlap seam*. This pass clips such
//! a triangle against the owner and zippers what remains onto the owner's
//! boundary, fail-closed:
//!
//! - **Only observed, same-surface ownership is subtracted.** In the owner
//!   camera's image the triangle loses exactly the owner triangles that
//!   ownership counted: linked components at the same depth after removing the
//!   pair's relative depth scale. Owner triangles at another depth (separate
//!   surfaces, occluders) are never subtracted, and nothing is added outside
//!   the triangle's own projection, so a gap no triangle observed stays open.
//! - **Derived vertices carry lineage.** Where the triangle's boundary crosses
//!   an owner edge, a vertex is derived on that owner edge (perspective-correct,
//!   so it lies on the owner's surface). It records its immutable source
//!   triangle (the owner triangle's index in the merge input and its
//!   pre-canonicalization corners) and its world-space barycentric weights.
//!   It is not a new observation: it keeps its parents' evidence origin and is
//!   appended after every observed point. A crossing whose owner edge joins
//!   corners of different evidence origins is not derived, and its triangle is
//!   left unclipped.
//! - **Edge-conforming zippers.** Every vertex a remainder places inside an
//!   existing edge (derived crossings, owner vertices on the triangle's edges,
//!   the triangle's corners on owner edges) is recorded on that edge, and
//!   every triangle using such an edge is split there, so no vertex lies inside
//!   another triangle's edge (no T-junction).
//! - **Verified or untouched.** A clip is accepted only if its planar
//!   arrangement is one connected graph, every face triangulates, and the
//!   remainder's image area plus the subtracted overlap equals the
//!   triangle's image area; otherwise the triangle is kept whole and counted.
//!
//! Work is bounded: a clip involves at most [`MAX_CLIP_OWNER_TRIANGLES`]
//! owner triangles (the ownership query's posting bound applies first) and at
//! most [`MAX_CLIP_NODES`] arrangement nodes.

use super::{
    cross2, inverse_depth, polygon_area, Footprint, PostingTally,
    MAX_RELATIVE_DEPTH_DISAGREEMENT,
};
use crate::{DerivedVertexSource, EvidenceOrigin, MeshTriangle, Point3};
use nalgebra::{Vector2, Vector3};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Most agreeing owner triangles one clip may subtract.
pub(super) const MAX_CLIP_OWNER_TRIANGLES: usize = 48;
/// Most arrangement nodes one clip may build.
pub(super) const MAX_CLIP_NODES: usize = 160;
/// Relative image-space tolerance (of the clipped triangle's longest edge)
/// for coincidence and point-on-segment decisions: above the single-precision
/// noise of the stored points, far below any accepted triangle.
const RELATIVE_TOLERANCE: f64 = 1.0e-5;
/// Relative tolerance on the clip's area balance.
const AREA_TOLERANCE: f64 = 1.0e-4;

/// A retained, partially owned triangle to clip against one owner footprint.
#[derive(Clone, Copy, Debug)]
pub(super) struct ClipJob {
    /// Index of the triangle in the merge's input buffer.
    pub triangle: usize,
    /// The owning [`Footprint`], by position in the merge's footprint list.
    pub footprint: usize,
    /// The triangle's patch depth scale relative to the owner.
    pub scale: f64,
}

/// A derived vertex before it is appended to the point buffer.
#[derive(Clone, Debug)]
pub(super) struct DerivedPoint {
    pub point: Point3,
    pub group: Option<usize>,
    pub source: DerivedVertexSource,
}

/// Sorted endpoints of an existing (or pre-canonicalization) mesh edge.
type EdgeKey = (usize, usize);

fn key(a: usize, b: usize) -> EdgeKey {
    (a.min(b), a.max(b))
}

/// Everything the clips of one merge accumulate.
pub(super) struct ClipState {
    /// Index of the first derived vertex in the point buffer.
    base: usize,
    pub derived: Vec<DerivedPoint>,
    /// Derived crossings by (subject edge, owner edge), so neighbouring clips
    /// that cross the same pair of edges share one vertex.
    crossings: HashMap<(EdgeKey, EdgeKey), usize>,
    /// Vertices recorded inside an edge, with their parameter from `key.0`
    /// (0) to `key.1` (1).
    carriers: BTreeMap<EdgeKey, Vec<(f64, usize)>>,
    /// The edges each recorded vertex lies inside.
    vertex_carriers: HashMap<usize, Vec<EdgeKey>>,
}

impl ClipState {
    pub(super) fn new(base: usize) -> Self {
        Self {
            base,
            derived: Vec::new(),
            crossings: HashMap::new(),
            carriers: BTreeMap::new(),
            vertex_carriers: HashMap::new(),
        }
    }

    fn record(&mut self, edge: EdgeKey, parameter: f64, vertex: usize) {
        if edge.0 == vertex || edge.1 == vertex {
            return;
        }
        let entries = self.carriers.entry(edge).or_default();
        if entries.iter().any(|&(_, existing)| existing == vertex) {
            return;
        }
        entries.push((parameter, vertex));
        let carriers = self.vertex_carriers.entry(vertex).or_default();
        if !carriers.contains(&edge) {
            carriers.push(edge);
        }
    }

    pub(super) fn has_records(&self) -> bool {
        !self.carriers.is_empty()
    }
}

/// What a clip reads from the merge.
pub(super) struct ClipInputs<'a> {
    pub points: &'a [Point3],
    pub triangles: &'a [MeshTriangle],
    pub corners: &'a dyn Fn(usize) -> Option<[Vector3<f64>; 3]>,
    pub footprints: &'a [Footprint],
    pub point_origins: &'a [Option<EvidenceOrigin>],
    pub group_of: &'a dyn Fn(usize) -> Option<usize>,
    /// Pre-canonicalization corners of a triangle of the merge input.
    pub source_corners: &'a dyn Fn(usize) -> [usize; 3],
    /// Whether a footprint triangle's original component may own the job's
    /// triangle.
    pub linked: &'a dyn Fn(usize, usize) -> bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum NodeKind {
    Vertex(usize),
    Crossing {
        subject: (usize, usize),
        subject_t: f64,
        owner: (usize, usize),
        owner_t: f64,
        /// Footprint triangle (agreeing) that carries the owner edge.
        owner_triangle: usize,
    },
}

#[derive(Clone, Copy, Debug)]
struct Node {
    position: Vector2<f64>,
    kind: NodeKind,
}

/// A segment of the arrangement: endpoint nodes, the vertex ids and image
/// depths of its endpoints, and the nodes inside it by parameter.
struct Segment {
    nodes: [usize; 2],
    ids: [usize; 2],
    depths: [f64; 2],
    inside: Vec<(f64, usize)>,
}

fn distance_to_line(a: Vector2<f64>, b: Vector2<f64>, p: Vector2<f64>) -> f64 {
    let direction = b - a;
    let length = direction.norm();
    if length <= 0.0 {
        return (p - a).norm();
    }
    cross2(direction, p - a) / length
}

fn parameter(a: Vector2<f64>, b: Vector2<f64>, p: Vector2<f64>) -> f64 {
    let direction = b - a;
    (p - a).dot(&direction) / direction.norm_squared()
}

/// World-space parameter along a segment whose endpoints have camera depths
/// `depths`, of the point at image parameter `t`.
fn world_parameter(t: f64, depths: [f64; 2]) -> f64 {
    let numerator = t * depths[0];
    let denominator = numerator + (1.0 - t) * depths[1];
    if denominator.abs() <= f64::MIN_POSITIVE {
        t
    } else {
        (numerator / denominator).clamp(0.0, 1.0)
    }
}

/// Strict barycentric containment with a tolerance in image units.
fn strictly_inside(triangle: &[Vector2<f64>; 3], point: Vector2<f64>, tolerance: f64) -> bool {
    let orientation = cross2(triangle[1] - triangle[0], triangle[2] - triangle[0]).signum();
    (0..3).all(|edge| {
        orientation * distance_to_line(triangle[edge], triangle[(edge + 1) % 3], point) > tolerance
    })
}

/// Ear clipping of a simple counter-clockwise polygon (collinear vertices
/// allowed); `None` when no valid ear remains.
pub(super) fn ear_clip(polygon: &[Vector2<f64>], tolerance: f64) -> Option<Vec<[usize; 3]>> {
    let mut remaining: Vec<usize> = (0..polygon.len()).collect();
    let mut triangles = Vec::with_capacity(polygon.len().saturating_sub(2));
    let area_floor = tolerance * tolerance;
    while remaining.len() > 3 {
        let count = remaining.len();
        let mut found = None;
        for slot in 0..count {
            let (previous, current, next) = (
                remaining[(slot + count - 1) % count],
                remaining[slot],
                remaining[(slot + 1) % count],
            );
            let (a, b, c) = (polygon[previous], polygon[current], polygon[next]);
            if cross2(b - a, c - a) <= area_floor {
                continue;
            }
            let ear = [a, b, c];
            let blocked = remaining.iter().any(|&other| {
                if other == previous || other == current || other == next {
                    return false;
                }
                let p = polygon[other];
                if (p - a).norm() <= tolerance
                    || (p - b).norm() <= tolerance
                    || (p - c).norm() <= tolerance
                {
                    return false;
                }
                // Inside or on the ear's boundary blocks it.
                (0..3).all(|edge| distance_to_line(ear[edge], ear[(edge + 1) % 3], p) >= -tolerance)
            });
            if !blocked {
                found = Some(slot);
                break;
            }
        }
        let slot = found?;
        let count = remaining.len();
        triangles.push([
            remaining[(slot + count - 1) % count],
            remaining[slot],
            remaining[(slot + 1) % count],
        ]);
        remaining.remove(slot);
    }
    let [a, b, c] = [remaining[0], remaining[1], remaining[2]];
    if cross2(polygon[b] - polygon[a], polygon[c] - polygon[a]) <= area_floor {
        return None;
    }
    triangles.push([a, b, c]);
    Some(triangles)
}

/// Why a job could not be clipped (it is then kept whole).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ClipFailure {
    /// No agreeing owner triangle (nothing to subtract).
    NothingOwned,
    /// The bounded lookup or the clip's own bounds stopped it.
    Bounded,
    /// The arrangement was not a verified planar subdivision.
    Unverified,
    /// A crossing would join corners of different evidence origins.
    MixedOrigins,
}

/// One clip's result: pieces over vertex ids (existing or reserved derived).
pub(super) type Pieces = Vec<[usize; 3]>;

/// Clip one job's triangle. On success the derived vertices and edge records
/// are committed to `state` and the remainder's triangles returned in the
/// triangle's own orientation; on failure `state` is unchanged.
pub(super) fn clip_overlap(
    inputs: &ClipInputs<'_>,
    state: &mut ClipState,
    job: &ClipJob,
    tally: &mut PostingTally,
) -> Result<Pieces, ClipFailure> {
    let footprint = &inputs.footprints[job.footprint];
    let triangle = &inputs.triangles[job.triangle];
    let world = (inputs.corners)(job.triangle).ok_or(ClipFailure::Unverified)?;
    let candidate = footprint
        .project_candidate(&world)
        .ok_or(ClipFailure::Unverified)?;
    // The owner triangles ownership counted: linked, same depth after scale.
    let mut owned: Vec<(usize, f64)> = Vec::new();
    let mut overflow = false;
    footprint
        .visit_overlaps(
            &candidate,
            &|index| (inputs.linked)(job.triangle, footprint.components[index]),
            &mut |polygon, index| {
                let kept = &footprint.triangles[index];
                let agrees = polygon.iter().all(|point| {
                    match (
                        inverse_depth(&candidate, *point),
                        inverse_depth(kept, *point),
                    ) {
                        (Some(own), Some(owner)) => {
                            let (own, owner) = (1.0 / own, job.scale / owner);
                            (own - owner).abs() <= MAX_RELATIVE_DEPTH_DISAGREEMENT * owner
                        }
                        _ => false,
                    }
                });
                if agrees {
                    owned.push((index, polygon_area(&polygon)));
                    if owned.len() > MAX_CLIP_OWNER_TRIANGLES {
                        overflow = true;
                        return false;
                    }
                }
                true
            },
            tally,
        )
        .map_err(|_| ClipFailure::Bounded)?;
    if overflow {
        return Err(ClipFailure::Bounded);
    }
    if owned.is_empty() {
        return Err(ClipFailure::NothingOwned);
    }
    let owned_area: f64 = owned.iter().map(|(_, area)| area).sum();

    // Counter-clockwise subject cycle in the owner image.
    let subject_ids = [triangle.a, triangle.b, triangle.c];
    let signed = cross2(
        candidate[1].0 - candidate[0].0,
        candidate[2].0 - candidate[0].0,
    );
    let flipped = signed < 0.0;
    let order: [usize; 3] = if flipped { [0, 2, 1] } else { [0, 1, 2] };
    let subject: [(Vector2<f64>, f64, usize); 3] =
        order.map(|corner| (candidate[corner].0, candidate[corner].1, subject_ids[corner]));
    let extent = (0..3)
        .map(|edge| (subject[(edge + 1) % 3].0 - subject[edge].0).norm())
        .fold(0.0, f64::max);
    let tolerance = extent * RELATIVE_TOLERANCE;
    let subject_area = signed.abs() * 0.5;

    let mut nodes: Vec<Node> = Vec::new();
    let mut by_id: HashMap<usize, usize> = HashMap::new();
    for &(position, _, id) in &subject {
        by_id.insert(id, nodes.len());
        nodes.push(Node {
            position,
            kind: NodeKind::Vertex(id),
        });
    }
    // Owner vertices; coincident ones snap onto the subject corner.
    let mut owner_depth: HashMap<usize, f64> = HashMap::new();
    for &(index, _) in &owned {
        for corner in 0..3 {
            let id = footprint.vertices[index][corner];
            let (position, depth) = footprint.triangles[index][corner];
            owner_depth.insert(id, depth);
            if by_id.contains_key(&id) {
                continue;
            }
            let node = subject
                .iter()
                .position(|(corner_position, _, _)| (*corner_position - position).norm() <= tolerance)
                .unwrap_or_else(|| {
                    nodes.push(Node {
                        position,
                        kind: NodeKind::Vertex(id),
                    });
                    nodes.len() - 1
                });
            by_id.insert(id, node);
        }
    }
    let subject_nodes = [0usize, 1, 2];
    let mut segments: Vec<Segment> = (0..3)
        .map(|edge| {
            let next = (edge + 1) % 3;
            Segment {
                nodes: [edge, next],
                ids: [subject[edge].2, subject[next].2],
                depths: [subject[edge].1, subject[next].1],
                inside: Vec::new(),
            }
        })
        .collect();
    // Owner edges, deduplicated, each with one agreeing triangle carrying it.
    let mut owner_edges: BTreeMap<EdgeKey, (usize, usize, usize)> = BTreeMap::new();
    for &(index, _) in &owned {
        let ids = footprint.vertices[index];
        for corner in 0..3 {
            let (a, b) = (ids[corner], ids[(corner + 1) % 3]);
            owner_edges.entry(key(a, b)).or_insert((a, b, index));
        }
    }
    let owner_first = segments.len();
    let mut owner_triangle_of_segment = Vec::new();
    for (&_, &(a, b, index)) in &owner_edges {
        let (node_a, node_b) = (by_id[&a], by_id[&b]);
        if node_a == node_b {
            continue;
        }
        segments.push(Segment {
            nodes: [node_a, node_b],
            ids: [a, b],
            depths: [owner_depth[&a], owner_depth[&b]],
            inside: Vec::new(),
        });
        owner_triangle_of_segment.push(index);
    }

    // Owner vertices inside subject edges, subject corners inside owner edges.
    let owner_nodes: BTreeSet<usize> = by_id
        .values()
        .copied()
        .filter(|node| !subject_nodes.contains(node))
        .collect();
    for edge in 0..3 {
        let [from, to] = segments[edge].nodes;
        let (a, b) = (nodes[from].position, nodes[to].position);
        let length = (b - a).norm();
        for &node in &owner_nodes {
            let p = nodes[node].position;
            let t = parameter(a, b, p);
            if distance_to_line(a, b, p).abs() <= tolerance
                && t * length > tolerance
                && (1.0 - t) * length > tolerance
            {
                segments[edge].inside.push((t, node));
            }
        }
    }
    for segment in owner_first..segments.len() {
        let [from, to] = segments[segment].nodes;
        let (a, b) = (nodes[from].position, nodes[to].position);
        let length = (b - a).norm();
        for &node in &subject_nodes {
            if node == from || node == to {
                continue;
            }
            let p = nodes[node].position;
            let t = parameter(a, b, p);
            if distance_to_line(a, b, p).abs() <= tolerance
                && t * length > tolerance
                && (1.0 - t) * length > tolerance
            {
                segments[segment].inside.push((t, node));
            }
        }
    }
    // Proper crossings between subject and owner edges.
    for edge in 0..3 {
        for segment in owner_first..segments.len() {
            let [p, q] = segments[edge].nodes;
            let [a, b] = segments[segment].nodes;
            if p == a || p == b || q == a || q == b {
                continue;
            }
            let (pp, qp, ap, bp) = (
                nodes[p].position,
                nodes[q].position,
                nodes[a].position,
                nodes[b].position,
            );
            let (da, db) = (distance_to_line(pp, qp, ap), distance_to_line(pp, qp, bp));
            let (dp, dq) = (distance_to_line(ap, bp, pp), distance_to_line(ap, bp, qp));
            if [da, db, dp, dq].iter().any(|d| d.abs() <= tolerance) {
                continue;
            }
            if da.signum() == db.signum() || dp.signum() == dq.signum() {
                continue;
            }
            let subject_t = dp / (dp - dq);
            let owner_t = da / (da - db);
            if nodes.len() >= MAX_CLIP_NODES {
                return Err(ClipFailure::Bounded);
            }
            let node = nodes.len();
            nodes.push(Node {
                position: pp + (qp - pp) * subject_t,
                kind: NodeKind::Crossing {
                    subject: (segments[edge].ids[0], segments[edge].ids[1]),
                    subject_t,
                    owner: (segments[segment].ids[0], segments[segment].ids[1]),
                    owner_t,
                    owner_triangle: owner_triangle_of_segment[segment - owner_first],
                },
            });
            segments[edge].inside.push((subject_t, node));
            segments[segment].inside.push((owner_t, node));
        }
    }
    if nodes.len() > MAX_CLIP_NODES {
        return Err(ClipFailure::Bounded);
    }

    // Planar graph: subject sub-segments, and owner sub-segments strictly
    // inside the subject.
    let subject_polygon = [subject[0].0, subject[1].0, subject[2].0];
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (index, segment) in segments.iter_mut().enumerate() {
        segment.inside.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut chain = vec![segment.nodes[0]];
        chain.extend(segment.inside.iter().map(|&(_, node)| node));
        chain.push(segment.nodes[1]);
        chain.dedup();
        for pair in chain.windows(2) {
            let (from, to) = (pair[0], pair[1]);
            if from == to {
                continue;
            }
            if index >= owner_first {
                let middle = (nodes[from].position + nodes[to].position) * 0.5;
                if !strictly_inside(&subject_polygon, middle, tolerance) {
                    continue;
                }
            }
            edges.insert((from.min(to), from.max(to)));
        }
    }
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for &(a, b) in &edges {
        neighbours[a].push(b);
        neighbours[b].push(a);
    }
    // One connected graph without dangling edges.
    let mut parent: Vec<usize> = (0..nodes.len()).collect();
    for &(a, b) in &edges {
        let (a, b) = (super::find(&mut parent, a), super::find(&mut parent, b));
        parent[a] = b;
    }
    let used: Vec<usize> = (0..nodes.len())
        .filter(|&node| !neighbours[node].is_empty())
        .collect();
    let root = super::find(&mut parent, used[0]);
    if used
        .iter()
        .any(|&node| super::find(&mut parent, node) != root || neighbours[node].len() < 2)
    {
        return Err(ClipFailure::Unverified);
    }
    for (node, list) in neighbours.iter_mut().enumerate() {
        let origin = nodes[node].position;
        list.sort_by(|a, b| {
            let (da, db) = (nodes[*a].position - origin, nodes[*b].position - origin);
            da.y.atan2(da.x).total_cmp(&db.y.atan2(db.x))
        });
    }
    // Faces with the face on the left of every half-edge.
    let mut visited: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut faces: Vec<Vec<usize>> = Vec::new();
    let mut outer = 0;
    let limit = 2 * edges.len() + 2;
    for &(a, b) in &edges {
        for (start, next) in [(a, b), (b, a)] {
            if visited.contains(&(start, next)) {
                continue;
            }
            let mut face = Vec::new();
            let (mut from, mut to) = (start, next);
            loop {
                if !visited.insert((from, to)) || face.len() > limit {
                    return Err(ClipFailure::Unverified);
                }
                face.push(from);
                let around = &neighbours[to];
                let back = around
                    .iter()
                    .position(|&node| node == from)
                    .ok_or(ClipFailure::Unverified)?;
                let following = around[(back + around.len() - 1) % around.len()];
                (from, to) = (to, following);
                if (from, to) == (start, next) {
                    break;
                }
            }
            let polygon: Vec<Vector2<f64>> = face.iter().map(|&node| nodes[node].position).collect();
            let area = signed_area(&polygon);
            if area > 0.0 {
                faces.push(face);
            } else {
                outer += 1;
            }
        }
    }
    if outer != 1 {
        return Err(ClipFailure::Unverified);
    }
    let owner_polygons: Vec<[Vector2<f64>; 3]> = owned
        .iter()
        .map(|&(index, _)| footprint.triangles[index].map(|corner| corner.0))
        .collect();
    let mut kept: Vec<[usize; 3]> = Vec::new();
    let mut kept_area = 0.0;
    let mut removed_area = 0.0;
    for face in &faces {
        let polygon: Vec<Vector2<f64>> = face.iter().map(|&node| nodes[node].position).collect();
        let ears = ear_clip(&polygon, tolerance).ok_or(ClipFailure::Unverified)?;
        let (largest, _) = ears
            .iter()
            .map(|ear| {
                let [a, b, c] = ear.map(|corner| polygon[corner]);
                ((a + b + c) / 3.0, cross2(b - a, c - a))
            })
            .fold((Vector2::zeros(), f64::NEG_INFINITY), |best, next| {
                if next.1 > best.1 {
                    next
                } else {
                    best
                }
            });
        let area = signed_area(&polygon);
        if owner_polygons
            .iter()
            .any(|owner| strictly_inside(owner, largest, 0.0))
        {
            removed_area += area;
            continue;
        }
        kept_area += area;
        kept.extend(ears.iter().map(|ear| ear.map(|corner| face[corner])));
    }
    let balance = AREA_TOLERANCE * subject_area;
    if (kept_area + removed_area - subject_area).abs() > balance
        || (removed_area - owned_area).abs() > balance
    {
        return Err(ClipFailure::Unverified);
    }

    // Resolve nodes to vertex ids, deriving the used crossings.
    let used_nodes: BTreeSet<usize> = kept.iter().flatten().copied().collect();
    let mut ids: HashMap<usize, usize> = HashMap::new();
    let mut fresh: Vec<(usize, DerivedPoint, (EdgeKey, EdgeKey))> = Vec::new();
    let mut next_index = state.base + state.derived.len();
    for &node in &used_nodes {
        let id = match nodes[node].kind {
            NodeKind::Vertex(id) => id,
            NodeKind::Crossing {
                subject: (p, q),
                owner: (a, b),
                owner_t,
                owner_triangle,
                ..
            } => {
                let crossing = (key(p, q), key(a, b));
                if let Some(&existing) = state.crossings.get(&crossing) {
                    existing
                } else {
                    let origin = inputs.point_origins.get(a).copied().flatten();
                    if origin != inputs.point_origins.get(b).copied().flatten() {
                        return Err(ClipFailure::MixedOrigins);
                    }
                    let s = world_parameter(owner_t, [owner_depth[&a], owner_depth[&b]]);
                    let mesh_index = footprint.mesh_indices[owner_triangle];
                    let current = footprint.vertices[owner_triangle];
                    let barycentric = current.map(|corner| {
                        if corner == a {
                            1.0 - s
                        } else if corner == b {
                            s
                        } else {
                            0.0
                        }
                    });
                    let (pa, pb) = (inputs.points[a], inputs.points[b]);
                    let lerp = |x: f32, y: f32| (f64::from(x) * (1.0 - s) + f64::from(y) * s) as f32;
                    let lerp8 = |x: u8, y: u8| {
                        (f64::from(x) * (1.0 - s) + f64::from(y) * s).round().clamp(0.0, 255.0) as u8
                    };
                    let point = Point3 {
                        x: lerp(pa.x, pb.x),
                        y: lerp(pa.y, pb.y),
                        z: lerp(pa.z, pb.z),
                        confidence: lerp(pa.confidence, pb.confidence),
                        r: lerp8(pa.r, pb.r),
                        g: lerp8(pa.g, pb.g),
                        b: lerp8(pa.b, pb.b),
                    };
                    let id = next_index;
                    next_index += 1;
                    fresh.push((
                        id,
                        DerivedPoint {
                            point,
                            group: (inputs.group_of)(a),
                            source: DerivedVertexSource {
                                point_index: id,
                                source_triangle: mesh_index,
                                source_corners: (inputs.source_corners)(mesh_index),
                                barycentric,
                            },
                        },
                        crossing,
                    ));
                    id
                }
            }
        };
        ids.insert(node, id);
    }
    // Commit: derived points, then edge records of every used node.
    for (id, derived, crossing) in fresh {
        state.crossings.insert(crossing, id);
        state.derived.push(derived);
    }
    for segment in &segments {
        let edge = key(segment.ids[0], segment.ids[1]);
        let forward = edge.0 == segment.ids[0];
        for &(t, node) in &segment.inside {
            let Some(&id) = ids.get(&node) else {
                continue;
            };
            let s = world_parameter(t, segment.depths);
            state.record(edge, if forward { s } else { 1.0 - s }, id);
        }
    }
    Ok(kept
        .into_iter()
        .map(|[a, b, c]| {
            let corners = [ids[&a], ids[&b], ids[&c]];
            if flipped {
                [corners[0], corners[2], corners[1]]
            } else {
                corners
            }
        })
        .collect())
}

fn signed_area(polygon: &[Vector2<f64>]) -> f64 {
    (0..polygon.len())
        .map(|index| cross2(polygon[index], polygon[(index + 1) % polygon.len()]))
        .sum::<f64>()
        * 0.5
}

/// Split every triangle at the vertices recorded inside its edges, so no
/// vertex lies inside another triangle's edge. A split triangle keeps its slot
/// for its first piece and appends the rest; returns the split slots.
pub(super) fn conform_edges(
    state: &ClipState,
    triangles: &mut Vec<MeshTriangle>,
    position: &dyn Fn(usize) -> Vector3<f64>,
    control: &mut crate::ReconstructionControl<'_>,
) -> Option<Vec<usize>> {
    let parameter_on = |edge: EdgeKey, vertex: usize| -> Option<f64> {
        if vertex == edge.0 {
            Some(0.0)
        } else if vertex == edge.1 {
            Some(1.0)
        } else {
            state
                .carriers
                .get(&edge)?
                .iter()
                .find(|(_, other)| *other == vertex)
                .map(|(t, _)| *t)
        }
    };
    let mut split = Vec::new();
    let mut appended = Vec::new();
    for slot in 0..triangles.len() {
        if slot % 256 == 0 && control.is_canceled() {
            return None;
        }
        let triangle = triangles[slot];
        let corners = [triangle.a, triangle.b, triangle.c];
        let mut boundary: Vec<(usize, Vector2<f64>)> = Vec::new();
        let frame = [Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0), Vector2::new(0.0, 1.0)];
        let mut inserted = false;
        for edge in 0..3 {
            let (u, v) = (corners[edge], corners[(edge + 1) % 3]);
            boundary.push((u, frame[edge]));
            let mut carriers: Vec<EdgeKey> = vec![key(u, v)];
            for endpoint in [u, v] {
                if let Some(list) = state.vertex_carriers.get(&endpoint) {
                    carriers.extend(list.iter().copied());
                }
            }
            let carrier = carriers.into_iter().find(|&carrier| {
                state.carriers.contains_key(&carrier)
                    && parameter_on(carrier, u).is_some()
                    && parameter_on(carrier, v).is_some()
            });
            let Some(carrier) = carrier else {
                continue;
            };
            let (tu, tv) = (parameter_on(carrier, u)?, parameter_on(carrier, v)?);
            if (tv - tu).abs() <= f64::EPSILON {
                continue;
            }
            let mut between: Vec<(f64, usize)> = state.carriers[&carrier]
                .iter()
                .filter(|&&(t, vertex)| {
                    vertex != u && vertex != v && {
                        let relative = (t - tu) / (tv - tu);
                        relative > 1.0e-9 && relative < 1.0 - 1.0e-9
                    }
                })
                .map(|&(t, vertex)| ((t - tu) / (tv - tu), vertex))
                .collect();
            between.sort_by(|a, b| a.0.total_cmp(&b.0));
            between.dedup_by_key(|entry| entry.1);
            let (start, end) = (frame[edge], frame[(edge + 1) % 3]);
            for (relative, vertex) in between {
                boundary.push((vertex, start + (end - start) * relative));
                inserted = true;
            }
        }
        if !inserted {
            continue;
        }
        let polygon: Vec<Vector2<f64>> = boundary.iter().map(|(_, at)| *at).collect();
        let Some(ears) = ear_clip(&polygon, 1.0e-12) else {
            continue;
        };
        // Reject a split whose pieces would be degenerate in space.
        let degenerate = ears.iter().any(|ear| {
            let [a, b, c] = ear.map(|corner| position(boundary[corner].0));
            (b - a).cross(&(c - a)).norm() <= f64::EPSILON
        });
        if degenerate {
            continue;
        }
        let mut pieces = ears.into_iter().map(|ear| MeshTriangle {
            a: boundary[ear[0]].0,
            b: boundary[ear[1]].0,
            c: boundary[ear[2]].0,
            confidence: triangle.confidence,
        });
        triangles[slot] = pieces.next()?;
        appended.extend(pieces);
        split.push(slot);
    }
    triangles.extend(appended);
    Some(split)
}
