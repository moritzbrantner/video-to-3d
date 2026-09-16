use crate::{MeshTriangle, Point3, ReconstructionResult};
use nalgebra::Vector3;
use std::collections::HashMap;

const GEOMETRIC_EPSILON: f64 = 1.0e-9;
const MAX_LOCAL_EDGE_FRACTION: f64 = 0.28;
const MAX_GLOBAL_EDGE_FRACTION: f64 = 0.35;
const MIN_NORMAL_ALIGNMENT: f64 = 0.94;
const MIN_CONFIDENCE_WEIGHT: f64 = 0.05;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FusionStats {
    eligible_vertices: usize,
    candidate_pairs: usize,
    rejected_normal_pairs: usize,
    fused_pairs: usize,
}

pub(super) fn consolidate_surface_patches(reconstruction: &mut ReconstructionResult) {
    let patch_count = reconstruction.dense.reference_patches.len();
    if patch_count < 2 || reconstruction.mesh_triangles.is_empty() {
        return;
    }

    let membership = patch_membership(reconstruction);
    let stats = fuse_mutually_supported_vertices(
        &mut reconstruction.dense_points,
        &reconstruction.mesh_triangles,
        &membership,
    );

    let diagnostic = if stats.fused_pairs > 0 {
        format!(
            "Cross-reference surface fusion aligned {} mutually nearest seam vertex pairs across {} accepted reference patches ({} bounded cross-patch candidates considered; {} rejected for surface-normal disagreement). No vertices or triangles were fabricated.",
            stats.fused_pairs,
            patch_count,
            stats.candidate_pairs,
            stats.rejected_normal_pairs,
        )
    } else {
        format!(
            "Cross-reference surface fusion kept {} accepted reference patches separate: {} mesh vertices were eligible and {} bounded cross-patch candidates were considered, but none passed the mutual-nearest and aligned-normal gates. No unsupported holes were closed.",
            patch_count,
            stats.eligible_vertices,
            stats.candidate_pairs,
        )
    };
    reconstruction.warnings.push(diagnostic);
}

fn patch_membership(reconstruction: &ReconstructionResult) -> Vec<Option<usize>> {
    let mut encoded = vec![0usize; reconstruction.dense_points.len()];
    for (patch_index, patch) in reconstruction.dense.reference_patches.iter().enumerate() {
        assign_patch_range(
            &mut encoded,
            patch.primary_start,
            patch.primary_points,
            patch_index,
        );
        assign_patch_range(
            &mut encoded,
            patch.completion_start,
            patch.completed_points,
            patch_index,
        );
    }

    encoded
        .into_iter()
        .map(|value| match value {
            0 | usize::MAX => None,
            value => Some(value - 1),
        })
        .collect()
}

fn assign_patch_range(
    membership: &mut [usize],
    start: usize,
    count: usize,
    patch_index: usize,
) {
    let Some(end) = start.checked_add(count) else {
        return;
    };
    let Some(range) = membership.get_mut(start..end) else {
        return;
    };
    let encoded_patch = patch_index + 1;
    for slot in range {
        if *slot == 0 {
            *slot = encoded_patch;
        } else if *slot != encoded_patch {
            *slot = usize::MAX;
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

    let (normals, local_scales, global_edge_scale) = vertex_surface_evidence(points, triangles);
    let Some(global_edge_scale) = global_edge_scale else {
        return FusionStats::default();
    };
    let cell_size = (global_edge_scale * MAX_GLOBAL_EDGE_FRACTION).max(GEOMETRIC_EPSILON);

    let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut eligible_vertices = 0usize;
    for index in 0..points.len() {
        if membership[index].is_none()
            || normals[index].is_none()
            || local_scales[index].is_none()
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
        let (Some(patch_index), Some(normal), Some(local_scale), Some(position)) = (
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
                        if other_index == index || membership[other_index] == Some(patch_index) {
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

                        let replace = nearest[index].is_none_or(|(current_index, current_distance)| {
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

    let mut fused_pairs = 0usize;
    for index in 0..points.len() {
        let Some((other_index, _)) = nearest[index] else {
            continue;
        };
        if index >= other_index
            || nearest[other_index].is_none_or(|(candidate, _)| candidate != index)
        {
            continue;
        }
        let (Some(position), Some(other_position)) =
            (point_position(points[index]), point_position(points[other_index]))
        else {
            continue;
        };
        let weight = f64::from(points[index].confidence).max(MIN_CONFIDENCE_WEIGHT);
        let other_weight =
            f64::from(points[other_index].confidence).max(MIN_CONFIDENCE_WEIGHT);
        let fused = (position * weight + other_position * other_weight) / (weight + other_weight);
        set_point_position(&mut points[index], fused);
        set_point_position(&mut points[other_index], fused);
        fused_pairs += 1;
    }

    FusionStats {
        eligible_vertices,
        candidate_pairs,
        rejected_normal_pairs,
        fused_pairs,
    }
}

fn vertex_surface_evidence(
    points: &[Point3],
    triangles: &[MeshTriangle],
) -> (Vec<Option<Vector3<f64>>>, Vec<Option<f64>>, Option<f64>) {
    let mut normal_sums = vec![Vector3::zeros(); points.len()];
    let mut edge_sums = vec![0.0f64; points.len()];
    let mut edge_counts = vec![0usize; points.len()];
    let mut edge_lengths = Vec::with_capacity(triangles.len() * 3);

    for triangle in triangles {
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
        .map(|(sum, count)| {
            (count > 0 && sum.is_finite()).then_some(sum / count as f64)
        })
        .collect();

    (normals, local_scales, global_edge_scale)
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
    position.iter().all(|value| value.is_finite()).then_some(position)
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
    if values.len() % 2 == 0 {
        Some((values[middle - 1] + values[middle]) * 0.5)
    } else {
        Some(values[middle])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn aligns_mutually_nearest_vertices_from_consistent_reference_patches() {
        let mut points = vec![
            point(0.0, 0.0, 0.0, 1.0),
            point(1.0, 0.0, 0.0, 1.0),
            point(0.0, 1.0, 0.0, 1.0),
            point(0.0, 0.0, 0.06, 1.0),
            point(1.0, 0.0, 0.06, 1.0),
            point(0.0, 1.0, 0.06, 1.0),
        ];
        let triangles = [triangle(0, 1, 2), triangle(3, 4, 5)];
        let membership = vec![
            Some(0),
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(1),
        ];

        let stats = fuse_mutually_supported_vertices(&mut points, &triangles, &membership);

        assert_eq!(stats.fused_pairs, 3);
        assert_eq!(stats.rejected_normal_pairs, 0);
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
        let membership = vec![
            Some(0),
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(1),
        ];

        let stats = fuse_mutually_supported_vertices(&mut points, &triangles, &membership);

        assert_eq!(stats.fused_pairs, 0);
        assert!(stats.rejected_normal_pairs >= 2);
        assert_eq!(points[0].z, 0.0);
        assert_eq!(points[3].z, 0.05);
    }

    #[test]
    fn never_fuses_vertices_from_the_same_reference_patch() {
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
}
