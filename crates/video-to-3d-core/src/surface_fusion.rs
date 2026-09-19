use crate::{
    MeshTriangle, Point3, ReconstructionEvidenceView, ReconstructionResult, SurfaceEvidenceRegion,
};
use nalgebra::Vector3;
use std::collections::HashMap;

const GEOMETRIC_EPSILON: f64 = 1.0e-9;
const MAX_LOCAL_EDGE_FRACTION: f64 = 0.28;
const MAX_GLOBAL_EDGE_FRACTION: f64 = 0.35;
const MIN_NORMAL_ALIGNMENT: f64 = 0.94;
const MIN_INCIDENT_NORMAL_ALIGNMENT: f64 = 0.985;
const MIN_INCIDENT_AREA_RATIO: f64 = 0.75;
const MIN_CONFIDENCE_WEIGHT: f64 = 0.05;
const PRE_FUSION_WARNING_CLAIM: &str =
    "texture projection, arbitrary multi-reference surface fusion, and metric scale are not claimed yet.";
const POST_FUSION_WARNING_CLAIM: &str = "texture projection and metric scale are not claimed yet.";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FusionStats {
    eligible_vertices: usize,
    candidate_pairs: usize,
    rejected_normal_pairs: usize,
    rejected_topology_pairs: usize,
    fused_pairs: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct EvidenceSupportKey {
    reference_frame: usize,
    source_frames: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct FusionMove {
    first: usize,
    second: usize,
    position: Vector3<f64>,
}

struct VertexSurfaceEvidence {
    normals: Vec<Option<Vector3<f64>>>,
    local_scales: Vec<Option<f64>>,
    incident_triangles: Vec<Vec<usize>>,
    global_edge_scale: Option<f64>,
}

pub(super) fn consolidate_surface_evidence(
    reconstruction: &mut ReconstructionResult,
) -> Result<(), String> {
    if reconstruction.mesh_triangles.is_empty() || reconstruction.dense_points.is_empty() {
        return Ok(());
    }

    let (membership, support_group_count) = {
        let evidence = ReconstructionEvidenceView::from_classic(reconstruction)?;
        evidence_support_membership(&evidence.regions, evidence.points.len())
    };
    if support_group_count < 2 {
        return Ok(());
    }

    let stats = fuse_mutually_supported_vertices(
        &mut reconstruction.dense_points,
        &reconstruction.mesh_triangles,
        &membership,
    );

    if stats.fused_pairs > 0 {
        remove_stale_surface_fusion_claim(&mut reconstruction.warnings);
    }
    let diagnostic = if stats.fused_pairs > 0 {
        format!(
            "Cross-reference surface fusion aligned {} mutually nearest seam vertex pairs across {} validated camera-support groups ({} bounded cross-group candidates considered; {} rejected for surface-normal disagreement and {} because the complete proposed seam move set would weaken accepted triangle topology). No vertices or triangles were fabricated.",
            stats.fused_pairs,
            support_group_count,
            stats.candidate_pairs,
            stats.rejected_normal_pairs,
            stats.rejected_topology_pairs,
        )
    } else {
        format!(
            "Cross-reference surface fusion kept {} validated camera-support groups separate: {} mesh vertices were eligible and {} bounded cross-group candidates were considered, but none passed the mutual-nearest, aligned-normal, and topology-preservation gates. No unsupported holes were closed.",
            support_group_count,
            stats.eligible_vertices,
            stats.candidate_pairs,
        )
    };
    reconstruction.warnings.push(diagnostic);
    Ok(())
}

fn evidence_support_membership(
    regions: &[SurfaceEvidenceRegion],
    point_count: usize,
) -> (Vec<Option<usize>>, usize) {
    let mut membership = vec![None; point_count];
    let mut support_groups = HashMap::<EvidenceSupportKey, usize>::new();

    for region in regions {
        let Some(reference_frame) = region.reference_frame else {
            continue;
        };
        let mut source_frames = region.source_frames.clone();
        source_frames.sort_unstable();
        let key = EvidenceSupportKey {
            reference_frame,
            source_frames,
        };
        let next_group = support_groups.len();
        let group = *support_groups.entry(key).or_insert(next_group);
        let Some(end) = region.points.start.checked_add(region.points.count) else {
            continue;
        };
        let Some(range) = membership.get_mut(region.points.start..end) else {
            continue;
        };
        range.fill(Some(group));
    }

    (membership, support_groups.len())
}

fn remove_stale_surface_fusion_claim(warnings: &mut [String]) {
    for warning in warnings {
        if warning.contains(PRE_FUSION_WARNING_CLAIM) {
            *warning = warning.replace(PRE_FUSION_WARNING_CLAIM, POST_FUSION_WARNING_CLAIM);
        }
    }
}

fn fuse_mutually_supported_vertices(
    points: &mut [Point3],
    triangles: &[MeshTriangle],
    membership: &[Option<usize>],
) -> FusionStats {
    if points.len() != membership.len() {
        return FusionStats::default();
    }

    let VertexSurfaceEvidence {
        normals,
        local_scales,
        incident_triangles,
        global_edge_scale,
    } = vertex_surface_evidence(points, triangles);
    let Some(global_edge_scale) = global_edge_scale else {
        return FusionStats::default();
    };
    let cell_size = (global_edge_scale * MAX_GLOBAL_EDGE_FRACTION).max(GEOMETRIC_EPSILON);

    let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut eligible_vertices = 0usize;
    for index in 0..points.len() {
        if membership[index].is_none() || normals[index].is_none() || local_scales[index].is_none()
        {
            continue;
        }
        let Some(position) = point_position(points[index]) else {
            continue;
        };
        buckets
            .entry(spatial_cell(position, cell_size))
            .or_default()
            .push(index);
        eligible_vertices += 1;
    }

    let mut nearest: Vec<Option<(usize, f64)>> = vec![None; points.len()];
    let mut candidate_pairs = 0usize;
    let mut rejected_normal_pairs = 0usize;

    for index in 0..points.len() {
        let (Some(support_group), Some(normal), Some(local_scale), Some(position)) = (
            membership[index],
            normals[index],
            local_scales[index],
            point_position(points[index]),
        ) else {
            continue;
        };
        let cell = spatial_cell(position, cell_size);

        for dz in -1i64..=1 {
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let neighbor_cell = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                    let Some(indices) = buckets.get(&neighbor_cell) else {
                        continue;
                    };
                    for &other_index in indices {
                        if other_index == index || membership[other_index] == Some(support_group) {
                            continue;
                        }
                        let (Some(other_normal), Some(other_scale), Some(other_position)) = (
                            normals[other_index],
                            local_scales[other_index],
                            point_position(points[other_index]),
                        ) else {
                            continue;
                        };
                        let maximum_distance = (local_scale.min(other_scale)
                            * MAX_LOCAL_EDGE_FRACTION)
                            .min(global_edge_scale * MAX_GLOBAL_EDGE_FRACTION);
                        let distance = (other_position - position).norm();
                        if !distance.is_finite() || distance > maximum_distance {
                            continue;
                        }
                        if index < other_index {
                            candidate_pairs += 1;
                        }

                        let normal_alignment = normal.dot(&other_normal).abs();
                        if normal_alignment < MIN_NORMAL_ALIGNMENT {
                            if index < other_index {
                                rejected_normal_pairs += 1;
                            }
                            continue;
                        }

                        let replace =
                            nearest[index].is_none_or(|(current_index, current_distance)| {
                                distance < current_distance - GEOMETRIC_EPSILON
                                    || ((distance - current_distance).abs() <= GEOMETRIC_EPSILON
                                        && other_index < current_index)
                            });
                        if replace {
                            nearest[index] = Some((other_index, distance));
                        }
                    }
                }
            }
        }
    }

    let mut proposed_moves = Vec::new();
    for index in 0..points.len() {
        let Some((other_index, _)) = nearest[index] else {
            continue;
        };
        if index >= other_index
            || nearest[other_index].is_none_or(|(candidate, _)| candidate != index)
        {
            continue;
        }
        let (Some(position), Some(other_position)) = (
            point_position(points[index]),
            point_position(points[other_index]),
        ) else {
            continue;
        };
        let weight = f64::from(points[index].confidence).max(MIN_CONFIDENCE_WEIGHT);
        let other_weight = f64::from(points[other_index].confidence).max(MIN_CONFIDENCE_WEIGHT);
        let fused = (position * weight + other_position * other_weight) / (weight + other_weight);
        proposed_moves.push(FusionMove {
            first: index,
            second: other_index,
            position: fused,
        });
    }

    let (accepted_moves, rejected_topology_pairs) =
        topology_safe_moves(points, triangles, &incident_triangles, proposed_moves);
    for movement in &accepted_moves {
        set_point_position(&mut points[movement.first], movement.position);
        set_point_position(&mut points[movement.second], movement.position);
    }

    FusionStats {
        eligible_vertices,
        candidate_pairs,
        rejected_normal_pairs,
        rejected_topology_pairs,
        fused_pairs: accepted_moves.len(),
    }
}

fn topology_safe_moves(
    points: &[Point3],
    triangles: &[MeshTriangle],
    incident_triangles: &[Vec<usize>],
    moves: Vec<FusionMove>,
) -> (Vec<FusionMove>, usize) {
    if moves.is_empty() {
        return (moves, 0);
    }

    let mut active = vec![true; moves.len()];
    let mut move_by_vertex = vec![None; points.len()];
    for (move_index, movement) in moves.iter().enumerate() {
        if movement.first >= points.len() || movement.second >= points.len() {
            active[move_index] = false;
            continue;
        }
        move_by_vertex[movement.first] = Some(move_index);
        move_by_vertex[movement.second] = Some(move_index);
    }

    loop {
        let mut touched = vec![false; triangles.len()];
        let mut touched_triangles = Vec::new();
        for (move_index, movement) in moves.iter().enumerate() {
            if !active[move_index] {
                continue;
            }
            for vertex in [movement.first, movement.second] {
                let Some(vertex_incident) = incident_triangles.get(vertex) else {
                    continue;
                };
                for &triangle_index in vertex_incident {
                    if triangle_index < triangles.len() && !touched[triangle_index] {
                        touched[triangle_index] = true;
                        touched_triangles.push(triangle_index);
                    }
                }
            }
        }

        let mut reject = vec![false; moves.len()];
        for triangle_index in touched_triangles {
            let triangle = &triangles[triangle_index];
            if triangle_preserved_with_moves(points, triangle, &moves, &active, &move_by_vertex) {
                continue;
            }
            for vertex in [triangle.a, triangle.b, triangle.c] {
                let Some(move_index) = move_by_vertex.get(vertex).copied().flatten() else {
                    continue;
                };
                if active[move_index] {
                    reject[move_index] = true;
                }
            }
        }

        if !reject.iter().any(|rejected| *rejected) {
            break;
        }
        for (move_index, rejected) in reject.into_iter().enumerate() {
            if rejected {
                active[move_index] = false;
            }
        }
    }

    let rejected = active.iter().filter(|accepted| !**accepted).count();
    let accepted = moves
        .into_iter()
        .zip(active)
        .filter_map(|(movement, accepted)| accepted.then_some(movement))
        .collect();
    (accepted, rejected)
}

fn triangle_preserved_with_moves(
    points: &[Point3],
    triangle: &MeshTriangle,
    moves: &[FusionMove],
    active: &[bool],
    move_by_vertex: &[Option<usize>],
) -> bool {
    if triangle.a >= points.len() || triangle.b >= points.len() || triangle.c >= points.len() {
        return false;
    }
    let (Some(a), Some(b), Some(c)) = (
        point_position(points[triangle.a]),
        point_position(points[triangle.b]),
        point_position(points[triangle.c]),
    ) else {
        return false;
    };
    let original_normal = (b - a).cross(&(c - a));
    let original_area_scale = original_normal.norm();
    if !original_area_scale.is_finite() || original_area_scale <= GEOMETRIC_EPSILON {
        return false;
    }

    let Some(proposed_a) = proposed_position(triangle.a, points, moves, active, move_by_vertex)
    else {
        return false;
    };
    let Some(proposed_b) = proposed_position(triangle.b, points, moves, active, move_by_vertex)
    else {
        return false;
    };
    let Some(proposed_c) = proposed_position(triangle.c, points, moves, active, move_by_vertex)
    else {
        return false;
    };
    let proposed_normal = (proposed_b - proposed_a).cross(&(proposed_c - proposed_a));
    let proposed_area_scale = proposed_normal.norm();
    if !proposed_area_scale.is_finite()
        || proposed_area_scale < original_area_scale * MIN_INCIDENT_AREA_RATIO
    {
        return false;
    }
    original_normal.dot(&proposed_normal) / (original_area_scale * proposed_area_scale)
        >= MIN_INCIDENT_NORMAL_ALIGNMENT
}

fn proposed_position(
    vertex: usize,
    points: &[Point3],
    moves: &[FusionMove],
    active: &[bool],
    move_by_vertex: &[Option<usize>],
) -> Option<Vector3<f64>> {
    if let Some(move_index) = move_by_vertex.get(vertex).copied().flatten() {
        if active.get(move_index).copied().unwrap_or(false) {
            return moves.get(move_index).map(|movement| movement.position);
        }
    }
    points.get(vertex).copied().and_then(point_position)
}

fn vertex_surface_evidence(points: &[Point3], triangles: &[MeshTriangle]) -> VertexSurfaceEvidence {
    let mut normal_sums = vec![Vector3::zeros(); points.len()];
    let mut edge_sums = vec![0.0f64; points.len()];
    let mut edge_counts = vec![0usize; points.len()];
    let mut incident_triangles = vec![Vec::new(); points.len()];
    let mut edge_lengths = Vec::with_capacity(triangles.len() * 3);

    for (triangle_index, triangle) in triangles.iter().enumerate() {
        if triangle.a == triangle.b
            || triangle.b == triangle.c
            || triangle.c == triangle.a
            || triangle.a >= points.len()
            || triangle.b >= points.len()
            || triangle.c >= points.len()
        {
            continue;
        }
        let (Some(a), Some(b), Some(c)) = (
            point_position(points[triangle.a]),
            point_position(points[triangle.b]),
            point_position(points[triangle.c]),
        ) else {
            continue;
        };
        let ab = b - a;
        let ac = c - a;
        let bc = c - b;
        let ab_length = ab.norm();
        let ac_length = ac.norm();
        let bc_length = bc.norm();
        if [ab_length, ac_length, bc_length]
            .iter()
            .any(|length| !length.is_finite() || *length <= GEOMETRIC_EPSILON)
        {
            continue;
        }

        let normal = ab.cross(&ac);
        let normal_length = normal.norm();
        if !normal_length.is_finite() || normal_length <= GEOMETRIC_EPSILON {
            continue;
        }
        let confidence = f64::from(triangle.confidence).clamp(MIN_CONFIDENCE_WEIGHT, 1.0);
        let weighted_normal = normal / normal_length * confidence;
        for index in [triangle.a, triangle.b, triangle.c] {
            normal_sums[index] += weighted_normal;
            incident_triangles[index].push(triangle_index);
        }

        edge_sums[triangle.a] += ab_length + ac_length;
        edge_counts[triangle.a] += 2;
        edge_sums[triangle.b] += ab_length + bc_length;
        edge_counts[triangle.b] += 2;
        edge_sums[triangle.c] += ac_length + bc_length;
        edge_counts[triangle.c] += 2;
        edge_lengths.extend_from_slice(&[ab_length, ac_length, bc_length]);
    }

    edge_lengths.sort_by(f64::total_cmp);
    let global_edge_scale = median(&edge_lengths);
    let normals = normal_sums
        .into_iter()
        .map(|normal| {
            let length = normal.norm();
            (length.is_finite() && length > GEOMETRIC_EPSILON).then_some(normal / length)
        })
        .collect();
    let local_scales = edge_sums
        .into_iter()
        .zip(edge_counts)
        .map(|(sum, count)| (count > 0 && sum.is_finite()).then_some(sum / count as f64))
        .collect();

    VertexSurfaceEvidence {
        normals,
        local_scales,
        incident_triangles,
        global_edge_scale,
    }
}

fn spatial_cell(position: Vector3<f64>, cell_size: f64) -> (i64, i64, i64) {
    (
        (position.x / cell_size).floor() as i64,
        (position.y / cell_size).floor() as i64,
        (position.z / cell_size).floor() as i64,
    )
}

fn point_position(point: Point3) -> Option<Vector3<f64>> {
    let position = Vector3::new(f64::from(point.x), f64::from(point.y), f64::from(point.z));
    position
        .iter()
        .all(|value| value.is_finite())
        .then_some(position)
}

fn set_point_position(point: &mut Point3, position: Vector3<f64>) {
    point.x = position.x as f32;
    point.y = position.y as f32;
    point.z = position.z as f32;
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[middle - 1] + values[middle]) * 0.5)
    } else {
        Some(values[middle])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EvidenceOrigin, EvidenceRange};

    fn point(x: f32, y: f32, z: f32, confidence: f32) -> Point3 {
        Point3 {
            x,
            y,
            z,
            confidence,
            r: 128,
            g: 128,
            b: 128,
        }
    }

    fn triangle(a: usize, b: usize, c: usize) -> MeshTriangle {
        MeshTriangle {
            a,
            b,
            c,
            confidence: 0.9,
        }
    }

    fn region(
        origin: EvidenceOrigin,
        reference_frame: Option<usize>,
        source_frames: Vec<usize>,
        start: usize,
        count: usize,
    ) -> SurfaceEvidenceRegion {
        SurfaceEvidenceRegion {
            origin,
            reference_frame,
            source_frames,
            points: EvidenceRange::new(start, count),
        }
    }

    #[test]
    fn groups_evidence_regions_by_camera_support_instead_of_dense_patch_identity() {
        let regions = vec![
            region(
                EvidenceOrigin::GeometricMultiView,
                Some(0),
                vec![2, 1],
                0,
                2,
            ),
            region(
                EvidenceOrigin::RevalidatedCompletion,
                Some(0),
                vec![1, 2],
                2,
                1,
            ),
            region(EvidenceOrigin::LearnedMultiView, Some(3), vec![4], 3, 1),
            region(EvidenceOrigin::GenerativeCompletion, None, Vec::new(), 4, 1),
        ];

        let (membership, groups) = evidence_support_membership(&regions, 5);

        assert_eq!(groups, 2);
        assert_eq!(membership, vec![Some(0), Some(0), Some(0), Some(1), None]);
    }

    #[test]
    fn aligns_mutually_nearest_vertices_from_consistent_evidence_groups() {
        let mut points = vec![
            point(0.0, 0.0, 0.0, 1.0),
            point(1.0, 0.0, 0.0, 1.0),
            point(0.0, 1.0, 0.0, 1.0),
            point(0.0, 0.0, 0.06, 1.0),
            point(1.0, 0.0, 0.06, 1.0),
            point(0.0, 1.0, 0.06, 1.0),
        ];
        let triangles = [triangle(0, 1, 2), triangle(3, 4, 5)];
        let membership = vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)];

        let stats = fuse_mutually_supported_vertices(&mut points, &triangles, &membership);

        assert_eq!(stats.fused_pairs, 3);
        assert_eq!(stats.rejected_normal_pairs, 0);
        assert_eq!(stats.rejected_topology_pairs, 0);
        for (left, right) in [(0, 3), (1, 4), (2, 5)] {
            assert!((points[left].z - 0.03).abs() < 1.0e-6);
            assert!((points[left].z - points[right].z).abs() < 1.0e-6);
        }
    }

    #[test]
    fn rejects_nearby_vertices_when_surface_normals_disagree() {
        let mut points = vec![
            point(0.0, 0.0, 0.0, 1.0),
            point(1.0, 0.0, 0.0, 1.0),
            point(0.0, 1.0, 0.0, 1.0),
            point(0.0, 0.0, 0.05, 1.0),
            point(1.0, 0.0, 0.05, 1.0),
            point(0.0, 0.05, 1.0, 1.0),
        ];
        let triangles = [triangle(0, 1, 2), triangle(3, 4, 5)];
        let membership = vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)];

        let stats = fuse_mutually_supported_vertices(&mut points, &triangles, &membership);

        assert_eq!(stats.fused_pairs, 0);
        assert!(stats.rejected_normal_pairs >= 2);
        assert_eq!(points[0].z, 0.0);
        assert_eq!(points[3].z, 0.05);
    }

    #[test]
    fn rejects_a_seam_move_that_would_weaken_an_accepted_triangle() {
        let mut points = vec![
            point(0.0, 0.0, 0.0, 1.0),
            point(1.0, 0.0, 0.0, 1.0),
            point(0.9, 0.01, 0.0, 1.0),
            point(0.0, 0.12, 0.0, 1.0),
            point(1.0, 0.12, 0.0, 1.0),
            point(0.0, 1.12, 0.0, 1.0),
        ];
        let triangles = [triangle(0, 1, 2), triangle(3, 4, 5)];
        let membership = vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)];

        let stats = fuse_mutually_supported_vertices(&mut points, &triangles, &membership);

        assert_eq!(stats.fused_pairs, 0);
        assert!(stats.rejected_topology_pairs > 0);
        assert_eq!(points[0].y, 0.0);
        assert_eq!(points[3].y, 0.12);
    }

    #[test]
    fn validates_the_complete_move_set_against_original_topology() {
        let points = vec![
            point(0.0, 0.0, 0.0, 1.0),
            point(1.0, 0.0, 0.0, 1.0),
            point(0.0, 1.0, 0.0, 1.0),
            point(0.2, 0.0, 0.0, 1.0),
            point(0.8, 0.0, 0.0, 1.0),
        ];
        let triangles = [triangle(0, 1, 2)];
        let incident = vec![vec![0], vec![0], vec![0], Vec::new(), Vec::new()];
        let moves = vec![
            FusionMove {
                first: 0,
                second: 3,
                position: Vector3::new(0.2, 0.0, 0.0),
            },
            FusionMove {
                first: 1,
                second: 4,
                position: Vector3::new(0.8, 0.0, 0.0),
            },
        ];

        let (accepted, rejected) = topology_safe_moves(&points, &triangles, &incident, moves);

        assert!(accepted.is_empty());
        assert_eq!(rejected, 2);
    }

    #[test]
    fn never_fuses_vertices_from_the_same_evidence_group() {
        let mut points = vec![
            point(0.0, 0.0, 0.0, 1.0),
            point(1.0, 0.0, 0.0, 1.0),
            point(0.0, 1.0, 0.0, 1.0),
        ];
        let triangles = [triangle(0, 1, 2)];
        let membership = vec![Some(0), Some(0), Some(0)];

        let stats = fuse_mutually_supported_vertices(&mut points, &triangles, &membership);

        assert_eq!(stats.fused_pairs, 0);
        assert_eq!(stats.candidate_pairs, 0);
    }

    #[test]
    fn removes_the_stale_not_claimed_warning_after_successful_fusion() {
        let mut warnings = vec![format!("Surface preview; {PRE_FUSION_WARNING_CLAIM}")];

        remove_stale_surface_fusion_claim(&mut warnings);

        assert!(!warnings[0].contains("arbitrary multi-reference surface fusion"));
        assert!(warnings[0].contains(POST_FUSION_WARNING_CLAIM));
    }
}
