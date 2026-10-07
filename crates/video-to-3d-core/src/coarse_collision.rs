//! Conservative coarse collision geometry derived from accepted surfaces.
//!
//! The collider is a separate, lower-detail representation; the visual
//! surface is only read. It is built in three steps:
//! 1. Eligibility: a triangle contributes only when all three vertices are
//!    observed geometry (geometric multi-view or revalidated completion) and
//!    its [`GeometryConfidenceField::triangle_confidence`] reaches the minimum.
//!    Learned and generative evidence never become collision geometry here.
//! 2. Occupancy: a uniform grid whose cell is a multiple of the median accepted
//!    edge length (bounded per axis) marks every cell that an eligible triangle
//!    actually intersects (exact separating-axis test with a small outward
//!    tolerance), so the union of cells contains every eligible triangle.
//! 3. Simplification: occupied cells are merged into axis-aligned boxes, first
//!    into runs along X, then identical runs along Y.
//!
//! Nothing is inferred between accepted triangles: an empty cell stays empty,
//! so an unsupported hole wider than about two cells stays open. A collider
//! surface lies at most one cell diagonal (`max_surface_offset`) away from the
//! accepted surface it covers.

use crate::geometry_confidence::GeometryConfidenceField;
use crate::{EvidenceOrigin, ReconstructionEvidenceView};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

pub const COARSE_COLLIDER_SCHEMA_VERSION: u32 = 1;
pub const COLLIDER_ROLE: &str = "collision";

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CoarseColliderOptions {
    /// Minimum triangle confidence for collision geometry.
    pub min_confidence: f32,
    /// Grid cell edge as a multiple of the median accepted edge length.
    pub cell_edge_multiple: f32,
    /// Upper bound of grid cells along the longest axis.
    pub max_cells_per_axis: u32,
}

impl Default for CoarseColliderOptions {
    fn default() -> Self {
        Self {
            min_confidence: 0.15,
            cell_edge_multiple: 2.0,
            max_cells_per_axis: 64,
        }
    }
}

/// Axis-aligned box in the reconstruction frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ColliderBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl ColliderBox {
    pub fn contains(&self, point: [f32; 3]) -> bool {
        (0..3).all(|axis| self.min[axis] <= point[axis] && point[axis] <= self.max[axis])
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ColliderExclusions {
    /// A vertex is learned or generative evidence.
    pub unobserved_provenance: usize,
    /// Observed, but below `min_confidence`.
    pub low_confidence: usize,
    /// The confidence field does not describe the supplied evidence, so no confidence
    /// could be trusted: every triangle is excluded (fail closed).
    pub mismatched_confidence: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CoarseCollider {
    pub schema_version: u32,
    /// Always [`COLLIDER_ROLE`]; never part of the visual surface.
    pub role: &'static str,
    pub options: CoarseColliderOptions,
    pub accepted_triangles: usize,
    pub source_triangles: usize,
    pub excluded: ColliderExclusions,
    /// `None` when no triangle was eligible.
    pub cell_size: Option<f32>,
    pub occupied_cells: usize,
    pub box_count: usize,
    /// Bound on how far the collider extends beyond the accepted surface.
    pub max_surface_offset: Option<f32>,
    #[serde(skip)]
    pub boxes: Vec<ColliderBox>,
}

impl CoarseCollider {
    pub fn from_evidence(
        evidence: &ReconstructionEvidenceView<'_>,
        confidence: &GeometryConfidenceField,
        options: CoarseColliderOptions,
    ) -> Self {
        let points = evidence.points;
        let mut excluded = ColliderExclusions::default();
        let mut eligible = Vec::new();
        // The field must have been built from this evidence; provenance is read from the
        // evidence itself either way, so a stale field can never admit unobserved geometry.
        let field_matches = confidence.describes(evidence);
        for triangle in evidence.triangles {
            let corners = [triangle.a, triangle.b, triangle.c];
            let observed = corners.iter().all(|&point| {
                evidence.regions.iter().any(|region| {
                    region.points.start <= point
                        && point < region.points.start + region.points.count
                        && matches!(
                            region.origin,
                            EvidenceOrigin::GeometricMultiView
                                | EvidenceOrigin::RevalidatedCompletion
                        )
                })
            });
            if !field_matches {
                excluded.mismatched_confidence += 1;
            } else if !observed {
                excluded.unobserved_provenance += 1;
            } else if confidence
                .triangle_confidence(triangle)
                // The triangle's own vertex minimum: a few weak points do not move the
                // region's lower median, but they still keep their triangles out.
                .min(triangle.confidence)
                < options.min_confidence
            {
                excluded.low_confidence += 1;
            } else {
                eligible.push(corners.map(|index| {
                    let point = &points[index];
                    [point.x as f64, point.y as f64, point.z as f64]
                }));
            }
        }

        let mut collider = Self {
            schema_version: COARSE_COLLIDER_SCHEMA_VERSION,
            role: COLLIDER_ROLE,
            options,
            accepted_triangles: evidence.triangles.len(),
            source_triangles: eligible.len(),
            excluded,
            cell_size: None,
            occupied_cells: 0,
            box_count: 0,
            max_surface_offset: None,
            boxes: Vec::new(),
        };
        let Some(grid) = Grid::fit(&eligible, &options) else {
            return collider;
        };

        let mut occupied = BTreeSet::new();
        for triangle in &eligible {
            grid.mark(triangle, &mut occupied);
        }
        collider.boxes = grid.merge(&occupied);
        collider.cell_size = Some(grid.cell as f32);
        collider.occupied_cells = occupied.len();
        collider.box_count = collider.boxes.len();
        collider.max_surface_offset = Some((grid.cell * 3f64.sqrt()) as f32);
        collider
    }

    pub fn diagnostic(&self) -> String {
        if self.excluded.mismatched_confidence > 0 {
            return format!(
                "Coarse collider: none. The geometry-confidence field does not describe this evidence, so all {} accepted triangles were excluded (fail closed).",
                self.accepted_triangles
            );
        }
        let Some(cell) = self.cell_size else {
            return format!(
                "Coarse collider: none. {} accepted triangles, {} learned/generative and {} below confidence {:.2}; no collision geometry is invented.",
                self.accepted_triangles,
                self.excluded.unobserved_provenance,
                self.excluded.low_confidence,
                self.options.min_confidence,
            );
        };
        format!(
            "Coarse collider v{}: {} boxes ({} grid cells of {:.3}) from {} of {} accepted triangles; excluded {} learned/generative and {} below confidence {:.2}. Kept separate from the visual mesh, at most {:.3} beyond the accepted surface; unsupported holes stay open.",
            self.schema_version,
            self.box_count,
            self.occupied_cells,
            cell,
            self.source_triangles,
            self.accepted_triangles,
            self.excluded.unobserved_provenance,
            self.excluded.low_confidence,
            self.options.min_confidence,
            self.max_surface_offset.unwrap_or(0.0),
        )
    }
}

struct Grid {
    origin: [f64; 3],
    cell: f64,
    dims: [i64; 3],
}

impl Grid {
    fn fit(triangles: &[[[f64; 3]; 3]], options: &CoarseColliderOptions) -> Option<Self> {
        if triangles.is_empty() {
            return None;
        }
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        let mut edges = Vec::with_capacity(triangles.len() * 3);
        for triangle in triangles {
            for (index, vertex) in triangle.iter().enumerate() {
                for axis in 0..3 {
                    min[axis] = min[axis].min(vertex[axis]);
                    max[axis] = max[axis].max(vertex[axis]);
                }
                edges.push(distance(*vertex, triangle[(index + 1) % 3]));
            }
        }
        edges.sort_unstable_by(f64::total_cmp);
        let median_edge = edges[(edges.len() - 1) / 2];
        let extent = (0..3).map(|axis| max[axis] - min[axis]).fold(0.0, f64::max);
        let cell = (median_edge * f64::from(options.cell_edge_multiple))
            .max(extent / f64::from(options.max_cells_per_axis.max(1)));
        if !cell.is_finite() || cell <= 0.0 {
            return None;
        }
        let dims = [0, 1, 2].map(|axis| ((max[axis] - min[axis]) / cell).floor() as i64 + 1);
        Some(Self {
            origin: min,
            cell,
            dims,
        })
    }

    fn cell_of(&self, value: f64, axis: usize) -> i64 {
        (((value - self.origin[axis]) / self.cell).floor() as i64).clamp(0, self.dims[axis] - 1)
    }

    fn mark(&self, triangle: &[[f64; 3]; 3], occupied: &mut BTreeSet<(i64, i64, i64)>) {
        let range = |axis: usize| {
            let low = triangle
                .iter()
                .map(|v| v[axis])
                .fold(f64::INFINITY, f64::min);
            let high = triangle
                .iter()
                .map(|v| v[axis])
                .fold(f64::NEG_INFINITY, f64::max);
            // One extra cell on each side lets the tolerant overlap test pick
            // up cells whose boundary the triangle merely touches.
            (
                (self.cell_of(low, axis) - 1).max(0),
                (self.cell_of(high, axis) + 1).min(self.dims[axis] - 1),
            )
        };
        let (x0, x1) = range(0);
        let (y0, y1) = range(1);
        let (z0, z1) = range(2);
        let half = self.cell * 0.5 * (1.0 + 1e-6) + 1e-12;
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let center = [x, y, z].map(|i| i as f64);
                    let center =
                        [0, 1, 2].map(|axis| self.origin[axis] + (center[axis] + 0.5) * self.cell);
                    if triangle_box_overlap(center, half, triangle) {
                        occupied.insert((z, y, x));
                    }
                }
            }
        }
    }

    fn merge(&self, occupied: &BTreeSet<(i64, i64, i64)>) -> Vec<ColliderBox> {
        // Runs along X keyed by (z, x0, x1), listing their rows.
        let mut runs: BTreeMap<(i64, i64, i64), Vec<i64>> = BTreeMap::new();
        let mut cells = occupied.iter().peekable();
        while let Some(&(z, y, x0)) = cells.next() {
            let mut x1 = x0;
            while cells.peek() == Some(&&(z, y, x1 + 1)) {
                x1 += 1;
                cells.next();
            }
            runs.entry((z, x0, x1)).or_default().push(y);
        }
        let mut boxes = Vec::new();
        for ((z, x0, x1), rows) in runs {
            let mut index = 0;
            while index < rows.len() {
                let y0 = rows[index];
                let mut y1 = y0;
                while index + 1 < rows.len() && rows[index + 1] == y1 + 1 {
                    index += 1;
                    y1 += 1;
                }
                index += 1;
                let low = [x0, y0, z];
                let high = [x1 + 1, y1 + 1, z + 1];
                boxes.push(ColliderBox {
                    min: [0, 1, 2]
                        .map(|axis| round_down(self.origin[axis] + low[axis] as f64 * self.cell)),
                    max: [0, 1, 2]
                        .map(|axis| round_up(self.origin[axis] + high[axis] as f64 * self.cell)),
                });
            }
        }
        boxes.sort_by(|left, right| {
            (left.min[2], left.min[1], left.min[0])
                .partial_cmp(&(right.min[2], right.min[1], right.min[0]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        boxes
    }
}

fn round_down(value: f64) -> f32 {
    let rounded = value as f32;
    if f64::from(rounded) > value {
        rounded - rounded.abs().max(f32::MIN_POSITIVE) * f32::EPSILON
    } else {
        rounded
    }
}

fn round_up(value: f64) -> f32 {
    let rounded = value as f32;
    if f64::from(rounded) < value {
        rounded + rounded.abs().max(f32::MIN_POSITIVE) * f32::EPSILON
    } else {
        rounded
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Separating-axis triangle/cube overlap (Akenine-Möller).
fn triangle_box_overlap(center: [f64; 3], half: f64, triangle: &[[f64; 3]; 3]) -> bool {
    let v = triangle.map(|vertex| sub(vertex, center));
    let separated = |axis: [f64; 3]| {
        let projections = v.map(|vertex| dot(vertex, axis));
        let radius = half * (axis[0].abs() + axis[1].abs() + axis[2].abs());
        let low = projections.iter().copied().fold(f64::INFINITY, f64::min);
        let high = projections
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        low > radius || high < -radius
    };
    let unit = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if unit.iter().any(|axis| separated(*axis)) {
        return false;
    }
    let edges = [sub(v[1], v[0]), sub(v[2], v[1]), sub(v[0], v[2])];
    if edges
        .iter()
        .flat_map(|edge| unit.iter().map(move |axis| cross(*axis, *edge)))
        .any(separated)
    {
        return false;
    }
    !separated(cross(edges[0], edges[1]))
}
