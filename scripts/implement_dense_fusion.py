from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def update_dense() -> None:
    path = Path("crates/video-to-3d-core/src/dense.rs")
    text = path.read_text()

    text = replace_once(
        text,
        "const MAX_RECIPROCAL_RELATIVE_DEPTH_ERROR: f64 = 0.08;\n",
        "const MAX_RECIPROCAL_RELATIVE_DEPTH_ERROR: f64 = 0.08;\n"
        "const MAX_FUSION_RELATIVE_POSITION_ERROR: f64 = 0.08;\n"
        "const MIN_FUSION_OBSERVATIONS: usize = 2;\n",
        "fusion constants",
    )

    text = replace_once(
        text,
        "    pub reciprocal_rejected_points: usize,\n",
        "    pub reciprocal_rejected_points: usize,\n"
        "    pub reciprocal_consistent_points: usize,\n"
        "    pub fusion_input_observations: usize,\n"
        "    pub fusion_rejected_observations: usize,\n"
        "    pub fusion_rejected_points: usize,\n"
        "    pub median_fusion_observations: Option<f32>,\n",
        "dense stats fields",
    )

    source_view_marker = "struct SourceView<'a> {"
    fused_struct = """#[derive(Clone, Copy, Debug)]
struct FusedDepthObservation {
    position: Vector3<f64>,
    observations: usize,
    rejected_observations: usize,
}

"""
    text = replace_once(
        text,
        source_view_marker,
        fused_struct + source_view_marker,
        "fused observation struct",
    )

    counter_marker = "    let mut reciprocal_rejected_points = 0usize;\n"
    text = replace_once(
        text,
        counter_marker,
        counter_marker
        + "    let mut reciprocal_consistent_points = 0usize;\n"
        + "    let mut fusion_input_observations = 0usize;\n"
        + "    let mut fusion_rejected_observations = 0usize;\n"
        + "    let mut fusion_rejected_points = 0usize;\n"
        + "    let mut fusion_observation_counts = Vec::new();\n",
        "fusion counters",
    )

    block_start = text.index("            reciprocal_checked_points += 1;")
    block_end_marker = "            supports.push(best.support as f64);"
    block_end = text.index(block_end_marker, block_start) + len(block_end_marker)
    fusion_block = """            reciprocal_checked_points += 1;
            let reciprocal_positions = reciprocal_depth_observations(
                best.position,
                reference,
                reference_frame,
                &reference_luma,
                &source_views,
                focal,
            );
            if reciprocal_positions.is_empty() {
                reciprocal_rejected_points += 1;
                continue;
            }
            reciprocal_consistent_points += 1;
            fusion_input_observations += 1 + reciprocal_positions.len();

            let fused = fuse_depth_observations(
                best.position,
                &reciprocal_positions,
                best.depth,
            );
            fusion_rejected_observations += fused.rejected_observations;
            if fused.observations < MIN_FUSION_OBSERVATIONS {
                fusion_rejected_points += 1;
                continue;
            }
            fusion_observation_counts.push(fused.observations as f64);

            let support_confidence = best.support as f64 / source_views.len() as f64;
            let reciprocal_confidence =
                ((fused.observations - 1) as f64 / source_views.len() as f64).clamp(0.0, 1.0);
            let error_confidence = 1.0 / (1.0 + best.error / 8.0);
            let margin_confidence = if ambiguity_margin.is_finite() {
                (ambiguity_margin / 6.0).clamp(0.2, 1.0)
            } else {
                1.0
            };
            let confidence = (support_confidence
                * reciprocal_confidence
                * error_confidence
                * margin_confidence)
                .clamp(0.05, 1.0) as f32;
            let (r, g, b) = sample_rgb(reference_frame, x, y);
            points.push(Point3 {
                x: fused.position.x as f32,
                y: fused.position.y as f32,
                z: fused.position.z as f32,
                confidence,
                r,
                g,
                b,
            });
            errors.push(best.error);
            supports.push(best.support as f64);"""
    text = text[:block_start] + fusion_block + text[block_end:]

    stats_marker = "            reciprocal_rejected_points,\n"
    text = replace_once(
        text,
        stats_marker,
        stats_marker
        + "            reciprocal_consistent_points,\n"
        + "            fusion_input_observations,\n"
        + "            fusion_rejected_observations,\n"
        + "            fusion_rejected_points,\n"
        + "            median_fusion_observations: median_option(&mut fusion_observation_counts)\n"
        + "                .map(|value| value as f32),\n",
        "stats output",
    )

    function_start = text.index("fn has_reciprocal_depth_agreement(")
    function_end = text.index(
        "\n#[allow(clippy::too_many_arguments)]\nfn estimate_single_view_depth(",
        function_start,
    )
    functions = """fn reciprocal_depth_observations(
    position: Vector3<f64>,
    reference_camera: &RegisteredCamera,
    reference_frame: &FrameInput,
    reference_luma: &[u8],
    sources: &[SourceView<'_>],
    focal: f64,
) -> Vec<Vector3<f64>> {
    let Some((reference_x, reference_y, reference_depth)) = project(
        reference_camera,
        position,
        reference_frame.width,
        reference_frame.height,
        focal,
    ) else {
        return Vec::new();
    };

    sources
        .iter()
        .filter_map(|source| {
            let direct_error = patch_error(
                reference_camera,
                source.camera,
                reference_luma,
                &source.luma,
                reference_frame.width,
                reference_frame.height,
                source.frame.width,
                source.frame.height,
                reference_x,
                reference_y,
                reference_depth,
                focal,
            )?;
            if direct_error > MAX_PHOTOMETRIC_ERROR {
                return None;
            }

            let (source_x, source_y, expected_source_depth) = project(
                source.camera,
                position,
                source.frame.width,
                source.frame.height,
                focal,
            )?;
            let reciprocal_depth = estimate_single_view_depth(
                source.camera,
                reference_camera,
                &source.luma,
                reference_luma,
                source.frame.width,
                source.frame.height,
                reference_frame.width,
                reference_frame.height,
                source_x,
                source_y,
                source.search_min_depth,
                source.search_max_depth,
                focal,
            )?;
            if !reciprocal_depth_agrees(expected_source_depth, reciprocal_depth) {
                return None;
            }

            let reciprocal_position = unproject(
                source.camera,
                source_x,
                source_y,
                reciprocal_depth,
                source.frame.width,
                source.frame.height,
                focal,
            );
            reciprocal_position
                .iter()
                .all(|value| value.is_finite())
                .then_some(reciprocal_position)
        })
        .collect()
}

fn fuse_depth_observations(
    primary: Vector3<f64>,
    reciprocal: &[Vector3<f64>],
    reference_depth: f64,
) -> FusedDepthObservation {
    let radius = (reference_depth.abs() * MAX_FUSION_RELATIVE_POSITION_ERROR).max(1.0e-4);
    let mut sum = primary;
    let mut observations = 1usize;
    let mut rejected_observations = 0usize;

    for position in reciprocal {
        if !position.iter().all(|value| value.is_finite()) || (*position - primary).norm() > radius
        {
            rejected_observations += 1;
            continue;
        }
        sum += position;
        observations += 1;
    }

    FusedDepthObservation {
        position: sum / observations as f64,
        observations,
        rejected_observations,
    }
}
"""
    text = text[:function_start] + functions + text[function_end:]

    old_assertion = """        assert_eq!(
            result.stats.reciprocal_checked_points,
            result.stats.accepted_points + result.stats.reciprocal_rejected_points
        );"""
    new_assertion = """        assert_eq!(
            result.stats.reciprocal_checked_points,
            result.stats.reciprocal_consistent_points + result.stats.reciprocal_rejected_points
        );
        assert_eq!(
            result.stats.reciprocal_consistent_points,
            result.stats.accepted_points + result.stats.fusion_rejected_points
        );
        assert!(
            result.stats.fusion_input_observations
                >= result.stats.reciprocal_consistent_points * MIN_FUSION_OBSERVATIONS
        );
        assert!(
            result
                .stats
                .median_fusion_observations
                .is_some_and(|observations| observations >= MIN_FUSION_OBSERVATIONS as f32)
        );"""
    text = replace_once(text, old_assertion, new_assertion, "plane fusion assertions")

    test_marker = "    #[test]\n    fn reciprocal_consistency_rejects_wrong_depth() {"
    fusion_test = """    #[test]
    fn fusion_rejects_spatially_inconsistent_reciprocal_observations() {
        let primary = Vector3::new(0.0, 0.0, 4.0);
        let reciprocal = [
            Vector3::new(0.02, -0.01, 4.04),
            Vector3::new(0.9, 0.0, 4.0),
        ];

        let fused = fuse_depth_observations(primary, &reciprocal, 4.0);

        assert_eq!(fused.observations, 2);
        assert_eq!(fused.rejected_observations, 1);
        assert!((fused.position.z - 4.02).abs() < 1.0e-9);
        assert!(fused.position.x.abs() < 0.02);
    }

"""
    text = replace_once(text, test_marker, fusion_test + test_marker, "fusion regression test")

    path.write_text(text)


def update_web() -> None:
    path = Path("apps/web/src/reconstruction.ts")
    text = path.read_text()
    marker = "  reciprocal_rejected_points: number;\n"
    text = replace_once(
        text,
        marker,
        marker
        + "  reciprocal_consistent_points: number;\n"
        + "  fusion_input_observations: number;\n"
        + "  fusion_rejected_observations: number;\n"
        + "  fusion_rejected_points: number;\n"
        + "  median_fusion_observations: number | null;\n",
        "TypeScript dense stats",
    )
    path.write_text(text)

    path = Path("apps/web/src/SceneCanvas.tsx")
    text = path.read_text()
    start = text.index("  const denseDiagnostic = reconstruction.dense.skip_reason")
    end = text.index(";\n\n  return (", start) + 1
    diagnostic = """  const denseDiagnostic = reconstruction.dense.skip_reason
    ? `Dense depth skipped: ${reconstruction.dense.skip_reason}`
    : reconstruction.dense.accepted_points > 0
      ? `Fused dense depth: ${reconstruction.dense.accepted_points} scene points from ${reconstruction.dense.fusion_input_observations} reciprocal-consistent multi-view observations; reciprocal depth rejected ${reconstruction.dense.reciprocal_rejected_points} of ${reconstruction.dense.reciprocal_checked_points} primary candidates and spatial fusion rejected ${reconstruction.dense.fusion_rejected_observations} reverse observations`
      : reconstruction.dense.reciprocal_consistent_points > 0
        ? `Dense depth found ${reconstruction.dense.reciprocal_consistent_points} reciprocal-consistent primary candidates, but fusion rejected all remaining geometry (${reconstruction.dense.fusion_rejected_observations} inconsistent reverse observations)`
        : reconstruction.dense.reciprocal_checked_points > 0
          ? `Coarse dense depth ran, but reciprocal depth rejected ${reconstruction.dense.reciprocal_rejected_points} of ${reconstruction.dense.reciprocal_checked_points} primary candidates after the texture and ambiguity gates`
          : "Coarse dense depth ran, but no depth hypothesis passed the texture and ambiguity gates";"""
    text = text[:start] + diagnostic + text[end:]
    path.write_text(text)


def update_docs() -> None:
    path = Path("ROADMAP.md")
    text = path.read_text()
    text = replace_once(
        text,
        "- **Reciprocal depth consistency — current slice.**",
        "- **Reciprocal depth consistency — integrated.**",
        "reciprocal roadmap status",
    )
    text = replace_once(
        text,
        "- Dense point fusion. Merge consistent depth observations while preserving support/confidence and rejecting duplicates/outliers.",
        "- **Dense point fusion — current slice.** Convert each accepted reciprocal source depth back into Rust-owned world geometry, reject reverse observations that are spatially inconsistent with the primary hypothesis, and fuse the remaining multi-view observations into one confidence-aware scene sample without duplicating geometry.",
        "fusion roadmap item",
    )
    old_boundary = "Current boundary: Rust now applies a bounded reciprocal source-view consistency veto to coarse depth hypotheses after the original reference-view acceptance gates. The surviving points are still separate coarse samples from one selected reference, not fused scene evidence. Dense point fusion, fused surfaces, meshing, metric scale, full-resolution depth maps, and general multi-reference aggregation remain outside the implemented boundary.\n\nNext implementation slice: fuse reciprocal-consistent depth observations without creating duplicate or weakly supported geometry, while preserving the current confidence and fail-closed semantics."
    new_boundary = "Current boundary: Rust now turns reciprocal-consistent source depth estimates into explicit world-space observations, applies a bounded spatial-consistency gate, and fuses accepted observations with the primary hypothesis into one dense scene point. Confidence is reduced when only a subset of eligible source views survives the reciprocal and fusion gates. The primary search still starts from one selected reference; fused surfaces, meshing, metric scale, full-resolution depth maps, and general multi-reference aggregation remain outside the implemented boundary.\n\nNext implementation slice: add optional mesh reconstruction and texture projection over accepted fused geometry, keeping point fusion authoritative and fail-closed."
    text = replace_once(text, old_boundary, new_boundary, "roadmap boundary")
    path.write_text(text)

    path = Path("README.md")
    text = path.read_text()
    text = replace_once(
        text,
        "Accepted dense samples remain separate from the sparse map.",
        "Reciprocal source depths are converted back to world-space observations, spatial outliers are rejected, and the surviving multi-view observations are fused into one dense scene point. Fused dense points remain separate from the sparse map.",
        "README overview",
    )
    text = replace_once(
        text,
        "27. Surviving dense samples are returned and rendered separately from sparse landmarks. This remains a coarse depth result rather than fused dense scene geometry.",
        "27. Each reciprocal source depth is unprojected into the shared sparse-SfM coordinate frame. Reverse observations farther than 8% of the primary reference depth from the primary hypothesis are rejected as spatially inconsistent.\n28. A primary hypothesis plus at least one spatially consistent reciprocal observation is averaged into one dense scene point. Confidence includes the fraction of eligible source views that survived reciprocal fusion; the browser reports fusion input/rejection evidence.\n29. Fused dense points are returned and rendered separately from sparse landmarks. This remains a coarse fused point cloud, not a mesh or full multi-reference depth reconstruction.",
        "README steps",
    )
    text = replace_once(
        text,
        "reciprocal source-view agreement is now enforced, but dense point fusion, general multi-reference aggregation, mesh reconstruction, and metric-scale recovery are not implemented yet.",
        "reciprocal source-view observations are now fused into bounded scene points, but general multi-reference aggregation, mesh reconstruction, and metric-scale recovery are not implemented yet.",
        "README boundary",
    )
    text = replace_once(
        text,
        "conservative coarse registered-view depth estimation, and reciprocal source-view depth consistency.",
        "conservative coarse registered-view depth estimation, reciprocal source-view depth consistency, and bounded dense point fusion.",
        "README architecture",
    )
    path.write_text(text)


update_dense()
update_web()
update_docs()
