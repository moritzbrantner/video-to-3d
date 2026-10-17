use crate::{dense::estimate_depth_points, multi_view::RegisteredCamera, FrameInput};
use nalgebra::{Matrix3, Vector3};

fn camera(frame_index: usize, center_x: f64) -> RegisteredCamera {
    RegisteredCamera {
        frame_index,
        rotation: Matrix3::identity(),
        translation: Vector3::new(-center_x, 0.0, 0.0),
    }
}

fn texture(world_x: f64, world_y: f64) -> u8 {
    let value = 128.0
        + 55.0 * (world_x * 15.0 + world_y * 2.7).sin()
        + 40.0 * (world_y * 17.0 - world_x * 3.1).cos()
        + 25.0 * ((world_x + world_y) * 11.0).sin();
    value.round().clamp(0.0, 255.0) as u8
}

fn plane_frame(width: u32, height: u32, focal: f64, center_x: f64, depth: f64) -> FrameInput {
    let mut rgba = vec![0u8; width as usize * height as usize * 4];
    for y in 0..height {
        for x in 0..width {
            let world_x = center_x + (x as f64 - width as f64 * 0.5) / focal * depth;
            let world_y = (y as f64 - height as f64 * 0.5) / focal * depth;
            let value = texture(world_x, world_y);
            let index = (y as usize * width as usize + x as usize) * 4;
            rgba[index] = value;
            rgba[index + 1] = value;
            rgba[index + 2] = value;
            rgba[index + 3] = 255;
        }
    }
    FrameInput {
        width,
        height,
        rgba,
    }
}

fn plane_sparse_points(width: u32, height: u32, focal: f64, depth: f64) -> Vec<Vector3<f64>> {
    let mut points = Vec::new();
    for y in [10u32, 18, 26, 34] {
        for x in [12u32, 24, 36, 48, 56] {
            points.push(Vector3::new(
                (x as f64 - width as f64 * 0.5) / focal * depth,
                (y as f64 - height as f64 * 0.5) / focal * depth,
                depth,
            ));
        }
    }
    points
}

#[test]
fn verified_surface_completion_adds_expected_plane_geometry() {
    let width = 68;
    let height = 48;
    let focal = 60.0;
    let expected_depth = 4.0;
    let frames = vec![
        plane_frame(width, height, focal, 0.0, expected_depth),
        plane_frame(width, height, focal, 0.25, expected_depth),
        plane_frame(width, height, focal, -0.22, expected_depth),
    ];
    let cameras = vec![camera(0, 0.0), camera(1, 0.25), camera(2, -0.22)];
    let sparse = plane_sparse_points(width, height, focal, expected_depth);

    let result = estimate_depth_points(&frames, &cameras, &sparse, focal);

    assert!(result.stats.attempted);
    assert!(
        result.stats.surface_completion_proposals > 0,
        "fixture must create at least one coherent surface proposal",
    );
    assert!(
        result.stats.surface_completed_points > 0,
        "fixture must exercise successful verified surface completion",
    );
    assert_eq!(
        result.stats.surface_completion_proposals,
        result.stats.surface_completed_points
            + result.stats.surface_completion_rejected_texture
            + result.stats.surface_completion_rejected_cross_view
            + result.stats.surface_completion_rejected_reciprocal
            + result.stats.surface_completion_rejected_fusion
            + result.stats.surface_completion_rejected_footprint,
    );

    let primary_points = result
        .points
        .len()
        .checked_sub(result.stats.surface_completed_points)
        .expect("completion count cannot exceed accepted points");
    let completed_points = &result.points[primary_points..];
    let completed_sites = &result.grid_sites[primary_points..];
    assert_eq!(completed_points.len(), completed_sites.len());
    assert_eq!(
        completed_points.len(),
        result.stats.surface_completed_points
    );

    for (point, site) in completed_points.iter().zip(completed_sites) {
        assert!(
            (point.z as f64 - expected_depth).abs() < 0.55,
            "completed site ({}, {}) has unexpected depth {}",
            site.x,
            site.y,
            point.z,
        );
        assert!(point.confidence >= 0.05 && point.confidence <= 0.9);
    }
}

#[test]
fn accepted_plane_mesh_bakes_into_a_textured_glb_without_changing_geometry() {
    use crate::surface_materials::{bake_surface_materials, ReferenceImage};
    use crate::textured_glb::encode_textured_glb;
    use crate::{
        classic_reference_patch_regions, EvidenceCamera, EvidenceCameraAuthority, EvidenceScale,
        ReconstructionEvidenceView, ReconstructionProviderDescriptor,
    };

    let width = 68;
    let height = 48;
    let focal = 60.0;
    let depth = 4.0;
    let frames = vec![
        plane_frame(width, height, focal, 0.0, depth),
        plane_frame(width, height, focal, 0.25, depth),
        plane_frame(width, height, focal, -0.22, depth),
    ];
    let cameras = vec![camera(0, 0.0), camera(1, 0.25), camera(2, -0.22)];
    let sparse = plane_sparse_points(width, height, focal, depth);
    let dense = estimate_depth_points(&frames, &cameras, &sparse, focal);
    let mesh = crate::mesh::reconstruct_dense_mesh(
        &dense.points,
        &dense.grid_sites,
        &dense.stats,
        &cameras,
        width,
        height,
        focal,
    );
    assert!(
        !mesh.triangles.is_empty(),
        "fixture must produce accepted topology"
    );

    let evidence_cameras = cameras
        .iter()
        .map(|camera| EvidenceCamera {
            frame_index: camera.frame_index,
            authority: EvidenceCameraAuthority::RegisteredGeometry,
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [camera.translation.x as f32, 0.0, 0.0],
            confidence: Some(0.9),
            median_reprojection_error_pixels: Some(0.5),
        })
        .collect();
    let evidence = ReconstructionEvidenceView::new(
        ReconstructionProviderDescriptor::classic(),
        EvidenceScale::ArbitraryMonocular,
        evidence_cameras,
        classic_reference_patch_regions(&dense.stats.reference_patches),
        &dense.points,
        &mesh.triangles,
    )
    .expect("accepted reconstruction is valid evidence");
    let images = frames
        .iter()
        .enumerate()
        .map(|(frame_index, frame)| ReferenceImage {
            frame_index,
            width: frame.width,
            height: frame.height,
            rgba: &frame.rgba,
        })
        .collect::<Vec<_>>();
    let points_before = dense
        .points
        .iter()
        .map(|point| [point.x, point.y, point.z])
        .collect::<Vec<_>>();

    let bake = bake_surface_materials(&evidence, &dense.grid_sites, &images).unwrap();
    assert!(bake.textured_triangles() > 0, "{}", bake.diagnostic());
    assert_eq!(
        bake.textured_triangles() + bake.fallback.triangles.len(),
        mesh.triangles.len()
    );
    // Each triangle of a single reference patch is textured from that patch.
    assert_eq!(bake.fallback.reasons.mixed_reference, 0);
    let glb = encode_textured_glb(&evidence, &bake).unwrap();
    assert_eq!(&glb[..4], b"glTF");
    assert_eq!(
        points_before,
        dense
            .points
            .iter()
            .map(|point| [point.x, point.y, point.z])
            .collect::<Vec<_>>()
    );
}

#[test]
fn accepted_plane_reconstruction_has_confidence_and_a_containing_collider() {
    use crate::coarse_collision::{CoarseCollider, CoarseColliderOptions};
    use crate::geometry_confidence::{AgreementSource, ConfidenceBand, GeometryConfidenceField};
    use crate::{
        classic_reference_patch_regions, EvidenceCamera, EvidenceCameraAuthority, EvidenceOrigin,
        EvidenceScale, ReconstructionEvidenceView, ReconstructionProviderDescriptor,
    };

    let width = 68;
    let height = 48;
    let focal = 60.0;
    let depth = 4.0;
    let frames = vec![
        plane_frame(width, height, focal, 0.0, depth),
        plane_frame(width, height, focal, 0.25, depth),
        plane_frame(width, height, focal, -0.22, depth),
    ];
    let cameras = vec![camera(0, 0.0), camera(1, 0.25), camera(2, -0.22)];
    let sparse = plane_sparse_points(width, height, focal, depth);
    let dense = estimate_depth_points(&frames, &cameras, &sparse, focal);
    let mesh = crate::mesh::reconstruct_dense_mesh(
        &dense.points,
        &dense.grid_sites,
        &dense.stats,
        &cameras,
        width,
        height,
        focal,
    );
    assert!(!mesh.triangles.is_empty());
    let evidence_cameras = cameras
        .iter()
        .map(|camera| EvidenceCamera {
            frame_index: camera.frame_index,
            authority: EvidenceCameraAuthority::RegisteredGeometry,
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [camera.translation.x as f32, 0.0, 0.0],
            confidence: Some(0.9),
            median_reprojection_error_pixels: Some(0.5),
        })
        .collect();
    let evidence = ReconstructionEvidenceView::new(
        ReconstructionProviderDescriptor::classic(),
        EvidenceScale::ArbitraryMonocular,
        evidence_cameras,
        classic_reference_patch_regions(&dense.stats.reference_patches),
        &dense.points,
        &mesh.triangles,
    )
    .expect("accepted reconstruction is valid evidence")
    .with_point_attributes(&dense.point_attributes)
    .expect("dense reciprocal/depth-margin attributes are valid evidence");

    let field = GeometryConfidenceField::from_evidence(&evidence);
    assert_eq!(field.regions.len(), evidence.regions.len());
    for region in &field.regions {
        assert_ne!(region.band, ConfidenceBand::Unsupported, "{region:?}");
        assert_eq!(
            region.factors.agreement_source,
            AgreementSource::PointAttributes
        );
        assert!(region.factors.depth_ambiguity.is_some_and(|f| f > 0.0));
        assert!(region.factors.reciprocal_agreement > 0.0);
        // Baselines of 0.22-0.47 at depth 4 span about 3 to 7 degrees.
        let angle = region.median_triangulation_angle_degrees.unwrap();
        assert!((2.5..7.5).contains(&angle), "triangulation angle {angle}");
    }
    let geometric = field
        .regions
        .iter()
        .find(|region| region.origin == EvidenceOrigin::GeometricMultiView)
        .unwrap();
    if let Some(completion) = field
        .regions
        .iter()
        .find(|region| region.origin == EvidenceOrigin::RevalidatedCompletion)
    {
        assert!(completion.confidence < geometric.confidence);
    }

    let collider =
        CoarseCollider::from_evidence(&evidence, &field, CoarseColliderOptions::default());
    assert!(!collider.boxes.is_empty(), "{}", collider.diagnostic());
    assert!(collider.boxes.len() < mesh.triangles.len());
    for triangle in &mesh.triangles {
        if field.triangle_confidence(triangle).min(triangle.confidence)
            < collider.options.min_confidence
        {
            continue;
        }
        for index in [triangle.a, triangle.b, triangle.c] {
            let point = &dense.points[index];
            assert!(collider
                .boxes
                .iter()
                .any(|collider_box| collider_box.contains([point.x, point.y, point.z])));
        }
    }
}

/// The built-in reconstruction result for `frames` with the given registered
/// cameras, dense patches and mesh, as the classic pipeline assembles it before
/// `finalize_reconstruction`: camera 0 and 1 are the calibrated seed pair,
/// every further camera a registered view with its retained pose.
fn classic_result(
    cameras: &[RegisteredCamera],
    dense: crate::dense::DenseAnalysis,
    mesh: crate::mesh::MeshAnalysis,
) -> crate::ReconstructionResult {
    // An uncalibrated result carries every non-geometry field.
    let flat = FrameInput {
        width: 48,
        height: 32,
        rgba: [120, 120, 120, 255].repeat(48 * 32),
    };
    let mut result = crate::reconstruct(&crate::ReconstructionRequest {
        frames: vec![flat.clone(), flat],
        options: crate::ReconstructionOptions::default(),
    })
    .expect("uncalibrated shell");
    let flatten = |camera: &RegisteredCamera| {
        let r = camera.rotation;
        (
            [0, 1, 2].map(|row| [0, 1, 2].map(|col| r[(row, col)] as f32)),
            [0, 1, 2].map(|axis| camera.translation[axis] as f32),
        )
    };
    let (_, seed_translation) = flatten(&cameras[1]);
    result.calibrated_pair = Some(crate::CalibratedPairStats {
        from_frame: cameras[0].frame_index,
        to_frame: cameras[1].frame_index,
        matches: 100,
        inliers: 90,
        inlier_ratio: 0.9,
        focal_pixels: 60.0,
        median_sampson_error_pixels: 0.3,
        median_reprojection_error_pixels: 0.5,
        median_triangulation_angle_degrees: 4.0,
        relative_rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation_direction: seed_translation,
    });
    result.registered_views = cameras[2..]
        .iter()
        .map(|camera| {
            let (rows, translation) = flatten(camera);
            crate::RegisteredViewStats {
                frame_index: camera.frame_index,
                correspondences: 50,
                inliers: 45,
                inlier_ratio: 0.9,
                median_reprojection_error_pixels: 0.5,
                rotation: rows.concat().try_into().unwrap(),
                translation,
                recovered_from_revisit: false,
            }
        })
        .collect();
    result.multi_view.bundle_adjustment.final_cameras = cameras.to_vec();
    result.dense_points = dense.points;
    result.dense_grid_sites = dense.grid_sites;
    result.dense_point_attributes = dense.point_attributes;
    result.dense = dense.stats;
    result.mesh_triangles = mesh.triangles;
    result.mesh = mesh.stats;
    result
}

/// A slow pan over one textured plane, before and after surface
/// consolidation: overlapping reference patches, triangulated per reference by
/// the mesh stage and connected by surface fusion and the overlap merge into
/// shared seam vertices.
struct PanSurface {
    frames: Vec<FrameInput>,
    focal: f64,
    /// The classic result as the mesh stage produced it (no consolidation).
    before: crate::ReconstructionResult,
    /// The same result after `consolidate_surface_evidence`.
    after: crate::ReconstructionResult,
    /// Index into `before.mesh_triangles` of a stitched triangle with
    /// genuinely mixed-reference support, when one was requested.
    stitched: Option<usize>,
    /// `(after index, before index)` of every triangle consolidation retained
    /// from the mesh output (possibly re-indexed onto a fused partner).
    retained: Vec<(usize, usize)>,
    /// Indices (in `after`) of the triangles consolidation created: seam
    /// stitches, clip pieces and edge-conforming split pieces, each matched
    /// by source (see [`match_consolidated_triangles`]).
    created: Vec<usize>,
}

/// Reference frame owning each point of `reconstruction` as observed
/// (geometric or revalidated) evidence, derived from the evidence regions.
fn observed_reference_by_point(reconstruction: &crate::ReconstructionResult) -> Vec<Option<usize>> {
    use crate::{EvidenceOrigin, ReconstructionEvidenceView};
    let evidence =
        ReconstructionEvidenceView::from_classic(reconstruction).expect("valid classic evidence");
    let mut owners = vec![None; evidence.points.len()];
    for region in &evidence.regions {
        if !matches!(
            region.origin,
            EvidenceOrigin::GeometricMultiView | EvidenceOrigin::RevalidatedCompletion
        ) {
            continue;
        }
        let end = region.points.start + region.points.count;
        owners[region.points.start..end].fill(region.reference_frame);
    }
    owners
}

fn corner_references(
    owners: &[Option<usize>],
    triangle: &crate::MeshTriangle,
) -> [Option<usize>; 3] {
    [triangle.a, triangle.b, triangle.c].map(|index| owners[index])
}

fn is_single_reference(references: [Option<usize>; 3]) -> bool {
    references[0].is_some()
        && references
            .iter()
            .all(|reference| *reference == references[0])
}

fn is_mixed_reference(references: [Option<usize>; 3]) -> bool {
    references.iter().all(Option::is_some) && !is_single_reference(references)
}

/// Builds the pan surface. With `stitch`, one extra triangle spanning a vertex
/// of the highest-priority (largest) patch and two vertices of another patch
/// is added to the mesh output before consolidation: genuinely mixed support
/// that no canonicalization produced (as a future stitched seam would be).
fn pan_surface(stitch: bool) -> PanSurface {
    let width = 68;
    let height = 48;
    let focal = 60.0;
    let depth = 4.0;
    let frames = vec![
        plane_frame(width, height, focal, 0.0, depth),
        plane_frame(width, height, focal, 0.25, depth),
        plane_frame(width, height, focal, -0.22, depth),
    ];
    let cameras = vec![camera(0, 0.0), camera(1, 0.25), camera(2, -0.22)];
    let sparse = plane_sparse_points(width, height, focal, depth);
    let dense = estimate_depth_points(&frames, &cameras, &sparse, focal);
    assert!(
        dense.stats.reference_patches.len() >= 2,
        "fixture must produce overlapping reference patches"
    );
    let mesh = crate::mesh::reconstruct_dense_mesh(
        &dense.points,
        &dense.grid_sites,
        &dense.stats,
        &cameras,
        width,
        height,
        focal,
    );
    let mut before = classic_result(&cameras, dense, mesh);
    let owners = observed_reference_by_point(&before);
    for triangle in &before.mesh_triangles {
        assert!(
            is_single_reference(corner_references(&owners, triangle)),
            "the mesh stage triangulates every reference patch on its own"
        );
    }

    let stitched = stitch.then(|| {
        let mut counts = std::collections::BTreeMap::<usize, usize>::new();
        for triangle in &before.mesh_triangles {
            *counts.entry(owners[triangle.a].unwrap()).or_default() += 1;
        }
        let (&largest, _) = counts
            .iter()
            .max_by(|(fa, ca), (fb, cb)| ca.cmp(cb).then(fb.cmp(fa)))
            .expect("meshed patches");
        let position = |index: usize| {
            let point = &before.dense_points[index];
            Vector3::new(point.x as f64, point.y as f64, point.z as f64)
        };
        // The other-patch triangle whose first corner lies closest to a
        // vertex of the largest patch; that vertex replaces the corner.
        let mut best: Option<(f64, usize, usize)> = None;
        for (triangle_index, triangle) in before.mesh_triangles.iter().enumerate() {
            if owners[triangle.a] == Some(largest) {
                continue;
            }
            for (vertex, owner) in owners.iter().enumerate() {
                if *owner != Some(largest) {
                    continue;
                }
                let distance = (position(vertex) - position(triangle.a)).norm();
                if distance > 1.0e-6 && best.is_none_or(|(d, _, _)| distance < d) {
                    best = Some((distance, triangle_index, vertex));
                }
            }
        }
        let (_, triangle_index, vertex) = best.expect("patches overlap");
        let source = before.mesh_triangles[triangle_index];
        let stitched = crate::MeshTriangle {
            a: vertex,
            ..source
        };
        let corners = [stitched.a, stitched.b, stitched.c].map(position);
        assert!(
            (corners[1] - corners[0])
                .cross(&(corners[2] - corners[0]))
                .norm()
                > 1.0e-6,
            "stitched triangle must not be degenerate"
        );
        assert!(is_mixed_reference(corner_references(&owners, &stitched)));
        before.mesh_triangles.push(stitched);
        before.mesh_triangles.len() - 1
    });

    let mut after = before.clone();
    crate::surface_fusion::consolidate_surface_evidence(&mut after)
        .expect("surface consolidation succeeds");
    // Owner amendment 2026-10-17 (#137): consolidation may append derived
    // vertices where it clips overlap seams, but never removes or reorders
    // observed points, and every appended point carries lineage.
    assert_observed_points_kept_and_appended_points_carry_lineage(&before, &after);
    let (retained, created) = match_consolidated_triangles(&before, &after);
    PanSurface {
        frames,
        focal,
        before,
        after,
        stitched,
        retained,
        created,
    }
}

/// Consolidation never removes or reorders observed points: the observed
/// prefix of `after` is `before`'s point buffer (only positions surface fusion
/// moved may differ; colours and grid sites stay with their point), and
/// every point after it is a derived vertex with one lineage record, in
/// point order, resolving an immutable source triangle of `before` over
/// observed corners whose world-space barycentrics reconstruct it.
fn assert_observed_points_kept_and_appended_points_carry_lineage(
    before: &crate::ReconstructionResult,
    after: &crate::ReconstructionResult,
) {
    let observed = before.dense_points.len();
    assert!(before.mesh_derived_vertices.is_empty());
    assert_eq!(
        after.dense_points.len() - after.mesh_derived_vertices.len(),
        observed,
        "consolidation never removes observed points; every appended point carries lineage"
    );
    for (index, (b, a)) in before
        .dense_points
        .iter()
        .zip(&after.dense_points)
        .enumerate()
    {
        assert_eq!(
            (b.r, b.g, b.b),
            (a.r, a.g, a.b),
            "observed point {index} keeps its order and colour"
        );
    }
    assert_eq!(
        after.dense_grid_sites, before.dense_grid_sites,
        "grid sites cover only the observed points and none is invented"
    );
    let position = |index: usize| {
        let point = &after.dense_points[index];
        Vector3::new(point.x as f64, point.y as f64, point.z as f64)
    };
    for (ordinal, record) in after.mesh_derived_vertices.iter().enumerate() {
        assert_eq!(
            record.point_index,
            observed + ordinal,
            "lineage in point order"
        );
        let parent = before
            .mesh_triangles
            .get(record.source_triangle)
            .expect("lineage resolves an immutable pre-clip source triangle");
        assert_eq!(record.source_corners, [parent.a, parent.b, parent.c]);
        assert!(record
            .source_corners
            .iter()
            .all(|&corner| corner < observed));
        assert!(record
            .barycentric
            .iter()
            .all(|w| w.is_finite() && *w >= -1.0e-7 && *w <= 1.0 + 1.0e-7));
        assert!((record.barycentric.iter().sum::<f64>() - 1.0).abs() < 1.0e-7);
        let expected: Vector3<f64> = record
            .source_corners
            .iter()
            .zip(record.barycentric)
            .map(|(&corner, weight)| position(corner) * weight)
            .sum();
        assert!(
            (expected - position(record.point_index)).norm() < 1.0e-4,
            "derived point {} reconstructs from its source corners",
            record.point_index
        );
    }
}

/// Splits `after`'s triangles into those retained from `before` (in order;
/// consolidation keeps the triangle order, removes owned duplicates, and
/// re-indexes a corner only onto a fused partner, which fusion moved to the
/// same position) and those it created. A created triangle (a seam stitch, a
/// clip piece or an edge-conforming split piece) must be matched by source:
/// every corner is an observed input vertex or a derived vertex whose lineage
/// resolves to observed input vertices (checked by
/// [`assert_observed_points_kept_and_appended_points_carry_lineage`]).
fn match_consolidated_triangles(
    before: &crate::ReconstructionResult,
    after: &crate::ReconstructionResult,
) -> (Vec<(usize, usize)>, Vec<usize>) {
    let points = &after.dense_points;
    let observed = before.dense_points.len();
    let same_position = |left: usize, right: usize| {
        let (l, r) = (&points[left], &points[right]);
        l.x.to_bits() == r.x.to_bits()
            && l.y.to_bits() == r.y.to_bits()
            && l.z.to_bits() == r.z.to_bits()
    };
    let derived: std::collections::BTreeMap<usize, &crate::DerivedVertexSource> = after
        .mesh_derived_vertices
        .iter()
        .map(|record| (record.point_index, record))
        .collect();
    let mut retained = Vec::new();
    let mut created = Vec::new();
    let mut next = 0;
    for (index, triangle) in after.mesh_triangles.iter().enumerate() {
        let corners = [triangle.a, triangle.b, triangle.c];
        let found = corners
            .iter()
            .all(|&vertex| vertex < observed)
            .then(|| {
                (next..before.mesh_triangles.len()).find(|&source| {
                    let source = &before.mesh_triangles[source];
                    let original = [source.a, source.b, source.c];
                    original.iter().zip(&corners).any(|(b, a)| b == a)
                        && original
                            .iter()
                            .zip(&corners)
                            .all(|(b, a)| b == a || same_position(*b, *a))
                })
            })
            .flatten();
        if let Some(source) = found {
            retained.push((index, source));
            next = source + 1;
            continue;
        }
        for vertex in corners {
            assert!(
                vertex < observed
                    || derived.get(&vertex).is_some_and(|record| record
                        .source_corners
                        .iter()
                        .all(|&corner| corner < observed)),
                "created triangle {index} {corners:?} is not matched by source: vertex {vertex} is neither observed nor derived from observed corners"
            );
        }
        created.push(index);
    }
    (retained, created)
}

impl PanSurface {
    fn images(&self) -> Vec<crate::surface_materials::ReferenceImage<'_>> {
        self.frames
            .iter()
            .enumerate()
            .map(
                |(frame_index, frame)| crate::surface_materials::ReferenceImage {
                    frame_index,
                    width: frame.width,
                    height: frame.height,
                    rgba: &frame.rgba,
                },
            )
            .collect()
    }

    fn bake(
        &self,
        reconstruction: &crate::ReconstructionResult,
        projection: bool,
    ) -> crate::surface_materials::SurfaceMaterialBake {
        use crate::surface_materials::{bake_surface_materials_with_seams, SeamProjection};
        let evidence = crate::ReconstructionEvidenceView::from_classic(reconstruction)
            .expect("valid classic evidence");
        let points = geometry_snapshot(reconstruction);
        let bake = bake_surface_materials_with_seams(
            &evidence,
            &reconstruction.dense_grid_sites,
            &self.images(),
            projection.then_some(SeamProjection {
                focal_pixels: self.focal as f32,
            }),
        )
        .expect("bake succeeds");
        // Texture processing never creates or repairs geometry.
        assert_eq!(points, geometry_snapshot(reconstruction));
        assert_eq!(bake.point_count, reconstruction.dense_points.len());
        assert_eq!(bake.triangle_count, reconstruction.mesh_triangles.len());
        let mut seen = vec![0usize; reconstruction.mesh_triangles.len()];
        for index in bake
            .materials
            .iter()
            .flat_map(|material| &material.triangles)
            .chain(&bake.fallback.triangles)
        {
            seen[*index] += 1;
        }
        assert!(
            seen.iter().all(|count| *count == 1),
            "every accepted triangle is textured or kept as fallback exactly once"
        );
        bake
    }

    /// Indices (in `after`) of the triangles consolidation created with
    /// genuinely mixed-reference support: the seam stitches over existing
    /// vertices of two reference patches, and clip/split pieces whose
    /// corners span references, a derived corner counting as its source
    /// triangle's reference. (Before #137 clipping these were exactly the
    /// stitches appended after the retained triangles.)
    fn created_mixed(&self) -> Vec<usize> {
        let owners = observed_reference_by_point(&self.after);
        let observed = self.before.dense_points.len();
        let owner = |vertex: usize| {
            if vertex < observed {
                return owners[vertex];
            }
            let record = self
                .after
                .mesh_derived_vertices
                .iter()
                .find(|record| record.point_index == vertex)
                .expect("derived lineage");
            let sources = record.source_corners.map(|corner| owners[corner]);
            is_single_reference(sources).then_some(sources[0]).flatten()
        };
        let mixed: Vec<usize> = self
            .created
            .iter()
            .copied()
            .filter(|&index| {
                let t = &self.after.mesh_triangles[index];
                is_mixed_reference([t.a, t.b, t.c].map(owner))
            })
            .collect();
        assert!(
            mixed.len() >= self.stitched_seam_triangles(),
            "every reported seam stitch is a created mixed-reference triangle"
        );
        mixed
    }

    /// Seam stitch triangles consolidation appended after the retained
    /// triangles, as its merge diagnostic reports them.
    fn stitched_seam_triangles(&self) -> usize {
        let phrase = " seam stitch triangle(s)";
        let warning = self
            .after
            .warnings
            .iter()
            .find(|warning| warning.starts_with("Overlapping reference patches:"))
            .expect("merge diagnostic");
        let end = warning
            .find(phrase)
            .expect("stitch count in the diagnostic");
        let start = warning[..end]
            .rfind(|c: char| !c.is_ascii_digit())
            .map_or(0, |index| index + 1);
        warning[start..end].parse().expect("stitch count")
    }

    /// For every retained triangle of `after`, `(after index, before index)`
    /// of the mesh triangle it was retained from; created triangles are
    /// matched by source instead (see [`match_consolidated_triangles`]).
    fn retained_from(&self) -> Vec<(usize, usize)> {
        self.retained.clone()
    }
}

fn geometry_snapshot(reconstruction: &crate::ReconstructionResult) -> Vec<[u32; 4]> {
    reconstruction
        .dense_points
        .iter()
        .map(|point| {
            [
                point.x.to_bits(),
                point.y.to_bits(),
                point.z.to_bits(),
                point.confidence.to_bits(),
            ]
        })
        .chain(reconstruction.mesh_triangles.iter().map(|triangle| {
            [
                triangle.a as u32,
                triangle.b as u32,
                triangle.c as u32,
                triangle.confidence.to_bits(),
            ]
        }))
        .collect()
}

/// Rule, reference frame, and absolute image-pixel corners of every textured
/// triangle of `bake`, by triangle index.
fn textured_corners(
    bake: &crate::surface_materials::SurfaceMaterialBake,
) -> std::collections::BTreeMap<usize, (crate::surface_materials::MaterialRule, usize, [[f64; 2]; 3])>
{
    let mut corners = std::collections::BTreeMap::new();
    for material in &bake.materials {
        let origin = material.texture.crop_origin.map(f64::from);
        let size = [material.texture.width, material.texture.height].map(f64::from);
        for (triangle, uvs) in material.triangles.iter().zip(&material.corner_uvs) {
            let pixels = uvs.map(|uv| {
                [
                    origin[0] + uv[0] as f64 * size[0],
                    origin[1] + uv[1] as f64 * size[1],
                ]
            });
            corners.insert(*triangle, (material.rule, material.reference_frame, pixels));
        }
    }
    corners
}

/// Asserts that every retained triangle the unconsolidated bake textured from
/// its own reference grid is, after consolidation, still textured by the
/// same-reference rule from that reference frame at the same grid pixels,
/// and returns the canonicalization-born triangles (single reference before
/// the merge, mixed reference by their consolidated indices).
fn assert_retained_triangles_keep_their_reference(
    surface: &PanSurface,
    before: &crate::surface_materials::SurfaceMaterialBake,
    after: &crate::surface_materials::SurfaceMaterialBake,
) -> Vec<usize> {
    use crate::surface_materials::MaterialRule;
    let owners = observed_reference_by_point(&surface.after);
    let before_corners = textured_corners(before);
    let after_corners = textured_corners(after);
    let mut born = Vec::new();
    for (index, source) in surface.retained_from() {
        if Some(source) == surface.stitched {
            continue;
        }
        let canonicalized = is_mixed_reference(corner_references(
            &owners,
            &surface.after.mesh_triangles[index],
        ));
        if canonicalized {
            born.push(index);
        }
        let Some(&(rule, frame, pixels)) = before_corners.get(&source) else {
            continue;
        };
        assert_eq!(rule, MaterialRule::SameReferenceGrid);
        let kept = after_corners.get(&index);
        let Some(&(after_rule, after_frame, after_pixels)) = kept else {
            panic!(
                "retained triangle {index} (mesh triangle {source}, canonicalized: {canonicalized}) lost its frame-{frame} reference texture: {}",
                after.diagnostic()
            );
        };
        assert_eq!(
            (after_rule, after_frame),
            (MaterialRule::SameReferenceGrid, frame),
            "retained triangle {index} (canonicalized: {canonicalized}) must stay textured from its original reference grid: {}",
            after.diagnostic()
        );
        for (corner, (expected, actual)) in pixels.iter().zip(&after_pixels).enumerate() {
            assert!(
                (expected[0] - actual[0]).abs() < 1.0e-3
                    && (expected[1] - actual[1]).abs() < 1.0e-3,
                "retained triangle {index} corner {corner} (canonicalized: {canonicalized}) must sample its original grid site {expected:?}, not {actual:?}"
            );
        }
    }
    born
}

#[test]
fn canonicalized_seam_triangles_keep_their_original_reference_texture() {
    use crate::surface_materials::MaterialRule;

    let surface = pan_surface(false);
    for projection in [true, false] {
        let before = surface.bake(&surface.before, projection);
        assert_eq!(before.fallback.reasons.mixed_reference, 0);
        assert_eq!(before.seam_triangles(), 0);
        let after = surface.bake(&surface.after, projection);

        let born = assert_retained_triangles_keep_their_reference(&surface, &before, &after);
        assert!(
            !born.is_empty(),
            "overlap canonicalization must re-index retained lower-priority seam triangles onto higher-priority vertices"
        );
        // Only triangles consolidation created (#137 seam stitches and
        // clip/split pieces) have genuinely mixed support: they go to the
        // seam rule (or fail closed to the fallback without intrinsics); no
        // canonicalization-born (retained) triangle does.
        let stitches = surface.created_mixed();
        assert_eq!(
            after.fallback.reasons.mixed_reference + after.seam_triangles(),
            stitches.len(),
            "canonicalization-born seam triangles must not count as mixed_reference (projection: {projection}): {}",
            after.diagnostic()
        );
        for material in after
            .materials
            .iter()
            .filter(|material| material.rule == MaterialRule::SeamSingleCameraProjection)
        {
            assert!(
                material
                    .triangles
                    .iter()
                    .all(|index| stitches.contains(index)),
                "canonicalization-born seam triangles must not go to the seam rule: {}",
                after.diagnostic()
            );
        }
        if !projection {
            assert_eq!(after.seam_triangles(), 0, "{}", after.diagnostic());
        }
    }
}

#[test]
fn connected_plane_surface_produces_seams_the_single_camera_rule_textures() {
    use crate::surface_materials::MaterialRule;

    let surface = pan_surface(true);
    let stitched_before = surface.stitched.expect("stitched seam");
    let retained = surface.retained_from();
    let stitched = retained
        .iter()
        .find(|(_, source)| *source == stitched_before)
        .map(|(index, _)| *index)
        .expect("consolidation keeps the stitched seam triangle");
    let owners = observed_reference_by_point(&surface.after);
    assert!(
        is_mixed_reference(corner_references(
            &owners,
            &surface.after.mesh_triangles[stitched]
        )),
        "the stitched seam keeps mixed-reference support after consolidation"
    );

    let before = surface.bake(&surface.before, true);
    let bake = surface.bake(&surface.after, true);
    // Canonicalization-born triangles keep their original reference texture...
    let born = assert_retained_triangles_keep_their_reference(&surface, &before, &bake);
    assert!(
        !born.is_empty(),
        "the connected surface must contain canonicalization-born cross-reference triangles"
    );
    // ...so the only mixed-reference triangles are the genuinely mixed seam
    // and the created mixed-support triangles (#137 seam stitches and
    // clip/split pieces), and the single-camera rule textures them.
    let mut mixed = surface.created_mixed();
    mixed.push(stitched);
    assert_eq!(
        bake.seam_triangles() + bake.fallback.reasons.mixed_reference,
        mixed.len(),
        "only the genuinely mixed seam is decided by the seam rule: {}",
        bake.diagnostic()
    );
    let seam_material = bake
        .materials
        .iter()
        .find(|material| material.triangles.contains(&stitched))
        .unwrap_or_else(|| {
            panic!(
                "the single-camera rule must texture the genuinely mixed seam of a fronto-parallel plane seen by all cameras: {}",
                bake.diagnostic()
            )
        });
    assert_eq!(seam_material.rule, MaterialRule::SeamSingleCameraProjection);
    for material in bake
        .materials
        .iter()
        .filter(|material| material.rule == MaterialRule::SeamSingleCameraProjection)
    {
        assert!(material.triangles.iter().all(|index| mixed.contains(index)));
        assert!(material.seam_reference_frames.len() >= 2);
        assert_eq!(material.focal_pixels, Some(surface.focal as f32));
    }

    // Without intrinsics the seam rule fails closed for the genuinely mixed
    // triangles only; canonicalization-born triangles stay textured.
    let before = surface.bake(&surface.before, false);
    let closed = surface.bake(&surface.after, false);
    assert_retained_triangles_keep_their_reference(&surface, &before, &closed);
    assert_eq!(closed.seam_triangles(), 0);
    assert_eq!(
        closed.fallback.reasons.mixed_reference,
        mixed.len(),
        "{}",
        closed.diagnostic()
    );
    assert!(closed.fallback.triangles.contains(&stitched));
}

/// The staged entry points on a clip whose surface fusion moves dense vertices (the
/// acceptance clips of `tests/reconstruction_stages.rs` keep their support groups
/// separate): the mesh artifact records exactly the moved points, round-trips, and
/// the composed chain equals the single-call tail on the same stage outputs.
#[test]
fn staged_mesh_stage_records_the_dense_points_surface_fusion_moves() {
    use crate::reconstruction_stages::{
        estimate_dense_evidence, finish_reconstruction, StageCamera,
    };
    use crate::{
        compose_reconstruction, reconstruct_dense, reconstruct_mesh, request_content_hash,
        CameraStageArtifact, MeshStageArtifact, ReconstructionOptions, ReconstructionRequest,
        CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION,
    };

    let (width, height, focal, depth) = (68, 48, 60.0, 4.0);
    let request = ReconstructionRequest {
        frames: vec![
            plane_frame(width, height, focal, 0.0, depth),
            plane_frame(width, height, focal, 0.25, depth),
            plane_frame(width, height, focal, -0.22, depth),
        ],
        options: ReconstructionOptions::default(),
    };
    let cameras = vec![camera(0, 0.0), camera(1, 0.25), camera(2, -0.22)];
    let sparse_points = plane_sparse_points(width, height, focal, depth);

    // The camera stage output of these registered views, built the way `classic_result`
    // builds the single-call result around them.
    let shell = classic_result(
        &cameras,
        crate::dense::DenseAnalysis::default(),
        crate::mesh::MeshAnalysis::default(),
    );
    let flat = FrameInput {
        width: 48,
        height: 32,
        rgba: [120, 120, 120, 255].repeat(48 * 32),
    };
    let mut sparse = crate::reconstruct_sparse(&ReconstructionRequest {
        frames: vec![flat.clone(), flat],
        options: ReconstructionOptions::default(),
    })
    .expect("uncalibrated shell");
    sparse.image_width = width;
    sparse.image_height = height;
    sparse.focal_length_pixels = focal as f32;
    sparse.calibrated_pair = shell.calibrated_pair;
    sparse.registered_views = shell.registered_views;
    sparse.final_cameras = cameras.iter().map(StageCamera::from).collect();
    sparse.dense_cameras = sparse.final_cameras.clone();
    sparse.dense_sparse_points = sparse_points
        .iter()
        .map(|point| [point.x, point.y, point.z])
        .collect();

    let camera_artifact = CameraStageArtifact {
        schema_version: CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION,
        request_hash: request_content_hash(&request),
        sparse: sparse.clone(),
    };
    let dense_artifact = reconstruct_dense(&request, &camera_artifact).expect("dense stage");
    assert!(
        dense_artifact.dense.stats.reference_patches.len() >= 2,
        "fixture must produce overlapping reference patches"
    );
    let mesh_artifact = reconstruct_mesh(&camera_artifact, &dense_artifact).expect("mesh stage");
    let surface = &mesh_artifact.mesh;
    assert!(
        surface
            .consolidation_warnings
            .iter()
            .any(|warning| warning.starts_with("Cross-reference surface fusion aligned")),
        "{:?}",
        surface.consolidation_warnings
    );
    assert!(
        !surface.consolidated_points.is_empty(),
        "fusion moved seam vertices"
    );
    let dense_points = &dense_artifact.dense.points;
    assert!(
        surface.consolidated_points.len() < dense_points.len(),
        "only moved points are recorded"
    );
    for consolidated in &surface.consolidated_points {
        assert_ne!(dense_points[consolidated.index], consolidated.point);
    }

    let stored = MeshStageArtifact::from_json(&mesh_artifact.to_json()).expect("stored mesh");
    assert_eq!(stored, mesh_artifact);
    let composed = compose_reconstruction(&camera_artifact, &dense_artifact, &stored)
        .expect("compose one chain");
    let single = finish_reconstruction(
        sparse.clone(),
        estimate_dense_evidence(&request.frames, &sparse),
    )
    .expect("single-call tail");
    assert_ne!(
        composed.dense_points, *dense_points,
        "consolidated points applied"
    );
    assert_eq!(
        serde_json::to_value(&composed).unwrap(),
        serde_json::to_value(&single).unwrap()
    );
}
