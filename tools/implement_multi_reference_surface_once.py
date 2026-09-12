from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing expected {label} pattern")
    return text.replace(old, new, 1)


def write(path: str, text: str) -> None:
    Path(path).write_text(text)


# Dense reconstruction: preserve the existing per-reference estimator and orchestrate
# a bounded set of well-separated registered reference cameras around it.
dense_path = "crates/video-to-3d-core/src/dense.rs"
dense = Path(dense_path).read_text()
dense = replace_once(
    dense,
    "use std::{cmp::Ordering, collections::BTreeMap};",
    "use std::{cmp::Ordering, collections::{BTreeMap, BTreeSet}};",
    "dense imports",
)
dense = replace_once(
    dense,
    "const MAX_SOURCE_VIEWS: usize = 4;",
    "const MAX_REFERENCE_VIEWS: usize = 3;\nconst MAX_SOURCE_VIEWS: usize = 4;",
    "dense reference bound",
)
dense = replace_once(
    dense,
    "#[derive(Clone, Debug, Default, Serialize)]\npub struct DenseStats {",
    '''#[derive(Clone, Debug, Default, Serialize)]
pub struct DensePatchStats {
    pub reference_frame: usize,
    pub attempted: bool,
    pub skip_reason: Option<String>,
    pub source_frames: Vec<usize>,
    pub accepted_points: usize,
    pub surface_completed_points: usize,
    pub grid_stride: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DenseStats {''',
    "dense patch stats",
)
dense = replace_once(
    dense,
    "    pub reference_frame: Option<usize>,\n    pub source_views: usize,",
    "    pub reference_frame: Option<usize>,\n    pub reference_frames: Vec<usize>,\n    pub patches: Vec<DensePatchStats>,\n    pub source_views: usize,",
    "dense aggregate provenance fields",
)
dense = replace_once(
    dense,
    "pub(super) struct DenseGridSite {\n    pub x: u32,\n    pub y: u32,\n}",
    "pub(super) struct DenseGridSite {\n    pub reference_frame: usize,\n    pub x: u32,\n    pub y: u32,\n}",
    "dense grid provenance",
)

old_dense_signature = '''pub(super) fn estimate_depth_points(
    frames: &[FrameInput],
    cameras: &[RegisteredCamera],
    sparse_points: &[Vector3<f64>],
    focal: f64,
) -> DenseAnalysis {'''
new_dense_wrapper = '''pub(super) fn estimate_depth_points(
    frames: &[FrameInput],
    cameras: &[RegisteredCamera],
    sparse_points: &[Vector3<f64>],
    focal: f64,
) -> DenseAnalysis {
    let mut primary = estimate_depth_points_for_reference(
        frames,
        cameras,
        sparse_points,
        focal,
        None,
    );
    if !primary.stats.attempted {
        return primary;
    }
    let Some(primary_reference) = primary.stats.reference_frame else {
        return primary;
    };
    let Some(primary_camera) = cameras
        .iter()
        .find(|camera| camera.frame_index == primary_reference)
    else {
        return primary;
    };
    let Some(first_frame) = frames.first() else {
        return primary;
    };

    let mut additional_references: Vec<(usize, usize, f64)> = cameras
        .iter()
        .filter(|camera| camera.frame_index != primary_reference)
        .filter_map(|camera| {
            let visible = visible_depths(
                camera,
                sparse_points,
                first_frame.width,
                first_frame.height,
                focal,
            )
            .len();
            if visible < MIN_VISIBLE_SPARSE_POINTS {
                return None;
            }
            let separation = (camera.camera_center() - primary_camera.camera_center()).norm();
            separation
                .is_finite()
                .then_some((camera.frame_index, visible, separation))
        })
        .collect();
    additional_references.sort_by(
        |(left_frame, left_visible, left_separation),
         (right_frame, right_visible, right_separation)| {
            right_separation
                .partial_cmp(left_separation)
                .unwrap_or(Ordering::Equal)
                .then_with(|| right_visible.cmp(left_visible))
                .then_with(|| left_frame.cmp(right_frame))
        },
    );
    additional_references.truncate(MAX_REFERENCE_VIEWS.saturating_sub(1));

    let mut patches = vec![dense_patch_stats(primary_reference, &primary.stats)];
    let mut source_frames: BTreeSet<usize> =
        primary.stats.source_frames.iter().copied().collect();
    primary.stats.reference_frames = vec![primary_reference];
    primary.stats.patches.clear();

    for (reference_frame, _, _) in additional_references {
        let analysis = estimate_depth_points_for_reference(
            frames,
            cameras,
            sparse_points,
            focal,
            Some(reference_frame),
        );
        patches.push(dense_patch_stats(reference_frame, &analysis.stats));
        if !analysis.stats.attempted || analysis.points.is_empty() {
            continue;
        }

        primary.stats.reference_frames.push(reference_frame);
        source_frames.extend(analysis.stats.source_frames.iter().copied());
        primary.stats.sampled_pixels += analysis.stats.sampled_pixels;
        primary.stats.surface_completion_proposals +=
            analysis.stats.surface_completion_proposals;
        primary.stats.surface_completed_points += analysis.stats.surface_completed_points;
        primary.stats.surface_completion_rejected_texture +=
            analysis.stats.surface_completion_rejected_texture;
        primary.stats.surface_completion_rejected_cross_view +=
            analysis.stats.surface_completion_rejected_cross_view;
        primary.stats.surface_completion_rejected_reciprocal +=
            analysis.stats.surface_completion_rejected_reciprocal;
        primary.stats.surface_completion_rejected_fusion +=
            analysis.stats.surface_completion_rejected_fusion;
        primary.stats.surface_completion_rejected_footprint +=
            analysis.stats.surface_completion_rejected_footprint;
        primary.stats.reciprocal_checked_points += analysis.stats.reciprocal_checked_points;
        primary.stats.reciprocal_rejected_points += analysis.stats.reciprocal_rejected_points;
        primary.stats.reciprocal_consistent_points +=
            analysis.stats.reciprocal_consistent_points;
        primary.stats.fusion_input_observations += analysis.stats.fusion_input_observations;
        primary.stats.fusion_rejected_observations +=
            analysis.stats.fusion_rejected_observations;
        primary.stats.fusion_rejected_points += analysis.stats.fusion_rejected_points;
        primary.points.extend(analysis.points);
        primary.grid_sites.extend(analysis.grid_sites);
    }

    primary.stats.reference_frames.sort_unstable();
    primary.stats.reference_frames.dedup();
    if let Some(position) = primary
        .stats
        .reference_frames
        .iter()
        .position(|frame| *frame == primary_reference)
    {
        primary.stats.reference_frames.swap(0, position);
    }
    primary.stats.source_frames = source_frames.into_iter().collect();
    primary.stats.source_views = primary.stats.source_frames.len();
    primary.stats.accepted_points = primary.points.len();
    primary.stats.patches = patches;
    primary
}

fn dense_patch_stats(reference_frame: usize, stats: &DenseStats) -> DensePatchStats {
    DensePatchStats {
        reference_frame,
        attempted: stats.attempted,
        skip_reason: stats.skip_reason.clone(),
        source_frames: stats.source_frames.clone(),
        accepted_points: stats.accepted_points,
        surface_completed_points: stats.surface_completed_points,
        grid_stride: stats.grid_stride,
    }
}

fn estimate_depth_points_for_reference(
    frames: &[FrameInput],
    cameras: &[RegisteredCamera],
    sparse_points: &[Vector3<f64>],
    focal: f64,
    forced_reference_frame: Option<usize>,
) -> DenseAnalysis {'''
dense = replace_once(dense, old_dense_signature, new_dense_wrapper, "dense estimator signature")
dense = replace_once(
    dense,
    "    let Some((reference, mut visible_depths)) = cameras\n        .iter()\n        .filter_map(|camera| {",
    "    let Some((reference, mut visible_depths)) = cameras\n        .iter()\n        .filter(|camera| forced_reference_frame.is_none_or(|frame| camera.frame_index == frame))\n        .filter_map(|camera| {",
    "forced dense reference filter",
)
dense = dense.replace(
    "DenseGridSite { x, y }",
    "DenseGridSite { reference_frame: reference.frame_index, x, y }",
)
write(dense_path, dense)


# Mesh reconstruction: build topology independently per dense-reference grid and only
# co-render patches after every triangle has passed the existing per-reference gates.
mesh_path = "crates/video-to-3d-core/src/mesh.rs"
mesh = Path(mesh_path).read_text()
mesh = replace_once(
    mesh,
    "pub struct MeshTriangle {\n    pub a: usize,",
    "pub struct MeshTriangle {\n    pub reference_frame: usize,\n    pub a: usize,",
    "mesh triangle provenance",
)
mesh = replace_once(
    mesh,
    "#[derive(Clone, Debug, Default, Serialize)]\npub struct MeshStats {",
    '''#[derive(Clone, Debug, Default, Serialize)]
pub struct MeshPatchStats {
    pub reference_frame: usize,
    pub attempted: bool,
    pub skip_reason: Option<String>,
    pub grid_vertices: usize,
    pub rejected_grid_vertices: usize,
    pub candidate_triangles: usize,
    pub accepted_triangles: usize,
    pub rejected_discontinuities: usize,
    pub rejected_degenerate: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MeshStats {''',
    "mesh patch stats",
)
mesh = replace_once(
    mesh,
    "    pub reference_frame: Option<usize>,\n    pub grid_vertices: usize,",
    "    pub reference_frame: Option<usize>,\n    pub reference_frames: Vec<usize>,\n    pub surface_patches: usize,\n    pub patches: Vec<MeshPatchStats>,\n    pub grid_vertices: usize,",
    "mesh aggregate provenance fields",
)
mesh = replace_once(
    mesh,
    "struct GridVertex {\n    point_index: usize,",
    "struct GridVertex {\n    reference_frame: usize,\n    point_index: usize,",
    "grid vertex provenance",
)

old_mesh_signature = '''pub(super) fn reconstruct_dense_mesh(
    dense_points: &[Point3],
    grid_sites: &[DenseGridSite],
    dense: &DenseStats,
    cameras: &[RegisteredCamera],
    width: u32,
    height: u32,
    focal: f64,
) -> MeshAnalysis {'''
new_mesh_wrapper = '''pub(super) fn reconstruct_dense_mesh(
    dense_points: &[Point3],
    grid_sites: &[DenseGridSite],
    dense: &DenseStats,
    cameras: &[RegisteredCamera],
    width: u32,
    height: u32,
    focal: f64,
) -> MeshAnalysis {
    if !dense.attempted {
        return MeshAnalysis::skipped(
            dense.reference_frame,
            "dense reconstruction was not attempted",
        );
    }
    if dense_points.len() < 3 {
        return MeshAnalysis::skipped(
            dense.reference_frame,
            "fewer than three accepted fused dense points are available",
        );
    }
    if dense_points.len() != grid_sites.len() {
        return MeshAnalysis::skipped(
            dense.reference_frame,
            "dense point and reference-grid evidence counts disagree",
        );
    }
    if dense.grid_stride == 0 {
        return MeshAnalysis::skipped(
            dense.reference_frame,
            "dense reconstruction did not expose a valid reference-grid stride",
        );
    }
    if width == 0 || height == 0 || !focal.is_finite() || focal <= 0.0 {
        return MeshAnalysis::skipped(
            dense.reference_frame,
            "the reference camera does not provide finite positive projection parameters",
        );
    }

    let reference_frames = if dense.reference_frames.is_empty() {
        dense.reference_frame.into_iter().collect::<Vec<_>>()
    } else {
        dense.reference_frames.clone()
    };
    if reference_frames.is_empty() {
        return MeshAnalysis::skipped(None, "dense reconstruction did not select a reference frame");
    }

    let mut stats = MeshStats {
        attempted: true,
        reference_frame: reference_frames.first().copied(),
        reference_frames: reference_frames.clone(),
        ..MeshStats::default()
    };
    let mut triangles = Vec::new();
    let mut patches = Vec::new();

    for reference_frame in reference_frames {
        let patch = reconstruct_dense_mesh_for_reference(
            dense_points,
            grid_sites,
            dense,
            reference_frame,
            cameras,
            width,
            height,
            focal,
        );
        patches.push(MeshPatchStats {
            reference_frame,
            attempted: patch.stats.attempted,
            skip_reason: patch.stats.skip_reason.clone(),
            grid_vertices: patch.stats.grid_vertices,
            rejected_grid_vertices: patch.stats.rejected_grid_vertices,
            candidate_triangles: patch.stats.candidate_triangles,
            accepted_triangles: patch.stats.accepted_triangles,
            rejected_discontinuities: patch.stats.rejected_discontinuities,
            rejected_degenerate: patch.stats.rejected_degenerate,
        });
        stats.grid_vertices += patch.stats.grid_vertices;
        stats.rejected_grid_vertices += patch.stats.rejected_grid_vertices;
        stats.candidate_cells += patch.stats.candidate_cells;
        stats.candidate_triangles += patch.stats.candidate_triangles;
        stats.rejected_discontinuities += patch.stats.rejected_discontinuities;
        stats.rejected_degenerate += patch.stats.rejected_degenerate;
        triangles.extend(patch.triangles);
    }

    stats.accepted_triangles = triangles.len();
    stats.surface_patches = patches
        .iter()
        .filter(|patch| patch.accepted_triangles > 0)
        .count();
    stats.patches = patches;
    if triangles.is_empty() {
        stats.skip_reason = Some("no reference surface patch passed the mesh gates".into());
    }
    MeshAnalysis { stats, triangles }
}

fn reconstruct_dense_mesh_for_reference(
    dense_points: &[Point3],
    grid_sites: &[DenseGridSite],
    dense: &DenseStats,
    reference_frame: usize,
    cameras: &[RegisteredCamera],
    width: u32,
    height: u32,
    focal: f64,
) -> MeshAnalysis {'''
mesh = replace_once(mesh, old_mesh_signature, new_mesh_wrapper, "mesh reconstruction signature")
old_reference_extract = '''    let Some(reference_frame) = dense.reference_frame else {
        return MeshAnalysis::skipped(None, "dense reconstruction did not select a reference frame");
    };
'''
mesh = replace_once(mesh, old_reference_extract, "", "single mesh reference extraction")
mesh = replace_once(
    mesh,
    "    let stride = dense.grid_stride as f64;\n    let mut grid = BTreeMap::<(i32, i32), GridVertex>::new();\n\n    for (point_index, (point, grid_site)) in dense_points.iter().zip(grid_sites).enumerate() {",
    '''    let stride = dense.grid_stride as f64;
    let patch_point_count = grid_sites
        .iter()
        .filter(|site| site.reference_frame == reference_frame)
        .count();
    let mut grid = BTreeMap::<(i32, i32), GridVertex>::new();

    for (point_index, (point, grid_site)) in dense_points.iter().zip(grid_sites).enumerate() {
        if grid_site.reference_frame != reference_frame {
            continue;
        }''',
    "mesh patch grid filter",
)
mesh = replace_once(
    mesh,
    "        let vertex = GridVertex {\n            point_index,",
    "        let vertex = GridVertex {\n            reference_frame,\n            point_index,",
    "mesh grid vertex reference",
)
mesh = mesh.replace(
    "rejected_grid_vertices: dense_points.len().saturating_sub(grid.len()),",
    "rejected_grid_vertices: patch_point_count.saturating_sub(grid.len()),",
)
mesh = replace_once(
    mesh,
    "            reference_frame: Some(reference_frame),\n            grid_vertices: grid.len(),",
    "            reference_frame: Some(reference_frame),\n            reference_frames: vec![reference_frame],\n            surface_patches: usize::from(!build.triangles.is_empty()),\n            patches: Vec::new(),\n            grid_vertices: grid.len(),",
    "mesh final reference stats",
)
mesh = replace_once(
    mesh,
    "    Ok(MeshTriangle {\n        a: a.point_index,",
    "    Ok(MeshTriangle {\n        reference_frame: a.reference_frame,\n        a: a.point_index,",
    "mesh triangle reference output",
)
mesh = mesh.replace(
    "DenseGridSite { x:",
    "DenseGridSite { reference_frame: 0, x:",
)
write(mesh_path, mesh)


# Rust browser contract: dense eligibility is the union of every reference and source,
# with an explicit role for cameras that serve as both.
contract_path = "crates/video-to-3d-core/src/browser_contract.rs"
contract = Path(contract_path).read_text()
contract = replace_once(
    contract,
    "pub enum DenseCameraRole {\n    Reference,\n    Source,\n}",
    "pub enum DenseCameraRole {\n    Reference,\n    Source,\n    ReferenceAndSource,\n}",
    "dense camera role",
)
contract = replace_once(
    contract,
    '''        let dense_frames: HashSet<usize> = reconstruction
            .dense
            .reference_frame
            .into_iter()
            .chain(reconstruction.dense.source_frames.iter().copied())
            .collect();''',
    '''        let dense_reference_frames: HashSet<usize> = if reconstruction.dense.reference_frames.is_empty() {
            reconstruction.dense.reference_frame.into_iter().collect()
        } else {
            reconstruction.dense.reference_frames.iter().copied().collect()
        };
        let dense_frames: HashSet<usize> = dense_reference_frames
            .iter()
            .copied()
            .chain(reconstruction.dense.source_frames.iter().copied())
            .collect();''',
    "dense camera union",
)
contract = replace_once(
    contract,
    '''                let dense_role = if reconstruction.dense.reference_frame == Some(frame_index) {
                    Some(DenseCameraRole::Reference)
                } else if reconstruction.dense.source_frames.contains(&frame_index) {
                    Some(DenseCameraRole::Source)
                } else {
                    None
                };''',
    '''                let dense_role = match (
                    dense_reference_frames.contains(&frame_index),
                    reconstruction.dense.source_frames.contains(&frame_index),
                ) {
                    (true, true) => Some(DenseCameraRole::ReferenceAndSource),
                    (true, false) => Some(DenseCameraRole::Reference),
                    (false, true) => Some(DenseCameraRole::Source),
                    (false, false) => None,
                };''',
    "per-frame dense role",
)
old_contract_attempted = '''    if reconstruction.dense.attempted {
        if reconstruction.dense.reference_frame.is_none() {
            return Err(
                "camera-state invariant violated: attempted dense reconstruction has no reference frame"
                    .into(),
            );
        }
        let expected_dense_cameras = reconstruction.dense.source_views + 1;
        if camera_state.dense_eligible_cameras.len() != expected_dense_cameras {
            return Err(format!(
                "camera-state invariant violated: dense reconstruction used {expected_dense_cameras} reference/source cameras, but only {} are present in accepted camera state",
                camera_state.dense_eligible_cameras.len()
            ));
        }
    } else if !camera_state.dense_eligible_cameras.is_empty() {'''
new_contract_attempted = '''    if reconstruction.dense.attempted {
        if reconstruction.dense.reference_frame.is_none()
            || reconstruction.dense.reference_frames.is_empty()
        {
            return Err(
                "camera-state invariant violated: attempted dense reconstruction has no reference frame"
                    .into(),
            );
        }
        if reconstruction.dense.reference_frame != reconstruction.dense.reference_frames.first().copied() {
            return Err(
                "camera-state invariant violated: primary dense reference is not the first explicit reference frame"
                    .into(),
            );
        }
        let reference_set: HashSet<usize> =
            reconstruction.dense.reference_frames.iter().copied().collect();
        if reference_set.len() != reconstruction.dense.reference_frames.len() {
            return Err(
                "camera-state invariant violated: dense reference frame ids are not unique".into(),
            );
        }
        let expected_dense_frames: HashSet<usize> = reference_set
            .iter()
            .copied()
            .chain(reconstruction.dense.source_frames.iter().copied())
            .collect();
        let actual_dense_frames: HashSet<usize> = camera_state
            .dense_eligible_cameras
            .iter()
            .map(|camera| camera.frame_index)
            .collect();
        if expected_dense_frames != actual_dense_frames {
            return Err(
                "camera-state invariant violated: dense reference/source cameras differ from accepted camera state"
                    .into(),
            );
        }
        if reconstruction
            .mesh
            .reference_frames
            .iter()
            .any(|frame| !reference_set.contains(frame))
        {
            return Err(
                "camera-state invariant violated: mesh patch references are not dense reference frames"
                    .into(),
            );
        }
    } else if !camera_state.dense_eligible_cameras.is_empty() {'''
contract = replace_once(
    contract,
    old_contract_attempted,
    new_contract_attempted,
    "browser contract attempted dense block",
)
write(contract_path, contract)


# TypeScript contract mirrors the Rust-owned provenance and union semantics.
ts_path = "apps/web/src/reconstruction.ts"
ts = Path(ts_path).read_text()
ts = replace_once(
    ts,
    "export type DenseStats = {",
    '''export type DensePatchStats = {
  reference_frame: number;
  attempted: boolean;
  skip_reason: string | null;
  source_frames: number[];
  accepted_points: number;
  surface_completed_points: number;
  grid_stride: number;
};

export type DenseStats = {''',
    "TypeScript dense patch type",
)
ts = replace_once(
    ts,
    "  reference_frame: number | null;\n  source_views: number;",
    "  reference_frame: number | null;\n  reference_frames: number[];\n  patches: DensePatchStats[];\n  source_views: number;",
    "TypeScript dense provenance",
)
ts = replace_once(
    ts,
    "export type MeshTriangle = {\n  a: number;",
    "export type MeshTriangle = {\n  reference_frame: number;\n  a: number;",
    "TypeScript mesh triangle provenance",
)
ts = replace_once(
    ts,
    "export type MeshStats = {",
    '''export type MeshPatchStats = {
  reference_frame: number;
  attempted: boolean;
  skip_reason: string | null;
  grid_vertices: number;
  rejected_grid_vertices: number;
  candidate_triangles: number;
  accepted_triangles: number;
  rejected_discontinuities: number;
  rejected_degenerate: number;
};

export type MeshStats = {''',
    "TypeScript mesh patch type",
)
ts = replace_once(
    ts,
    "  reference_frame: number | null;\n  grid_vertices: number;",
    "  reference_frame: number | null;\n  reference_frames: number[];\n  surface_patches: number;\n  patches: MeshPatchStats[];\n  grid_vertices: number;",
    "TypeScript mesh provenance",
)
ts = replace_once(
    ts,
    'export type DenseCameraRole = "reference" | "source";',
    'export type DenseCameraRole = "reference" | "source" | "reference_and_source";',
    "TypeScript dense camera role",
)
old_ts_attempted = '''  if (result.dense.attempted) {
    if (result.dense.reference_frame === null) {
      throw new Error(
        "camera-state contract mismatch: attempted dense reconstruction has no reference frame",
      );
    }
    if (acceptedCameras.length < 2) {
      throw new Error(
        `camera-state contract mismatch: dense reconstruction ran with only ${acceptedCameras.length} accepted cameras`,
      );
    }
    const expectedDenseFrames = new Set([
      result.dense.reference_frame,
      ...result.dense.source_frames,
    ]);
    const actualDenseFrames = frameIndexSet(state.dense_eligible_cameras);
    if (!sameFrameSet(expectedDenseFrames, actualDenseFrames)) {
      throw new Error(
        "camera-state contract mismatch: dense reference/source frames differ from dense-eligible camera state",
      );
    }
    if ([...actualDenseFrames].some((frameIndex) => !acceptedFrames.has(frameIndex))) {
      throw new Error(
        "camera-state contract mismatch: dense-eligible camera is not part of accepted camera geometry",
      );
    }
  } else if (state.dense_eligible_cameras.length !== 0) {'''
new_ts_attempted = '''  if (result.dense.attempted) {
    if (result.dense.reference_frame === null || result.dense.reference_frames.length === 0) {
      throw new Error(
        "camera-state contract mismatch: attempted dense reconstruction has no reference frame",
      );
    }
    if (result.dense.reference_frames[0] !== result.dense.reference_frame) {
      throw new Error(
        "camera-state contract mismatch: primary dense reference is not first in reference_frames",
      );
    }
    if (new Set(result.dense.reference_frames).size !== result.dense.reference_frames.length) {
      throw new Error("camera-state contract mismatch: dense reference frame ids are not unique");
    }
    if (acceptedCameras.length < 2) {
      throw new Error(
        `camera-state contract mismatch: dense reconstruction ran with only ${acceptedCameras.length} accepted cameras`,
      );
    }
    const expectedDenseFrames = new Set([
      ...result.dense.reference_frames,
      ...result.dense.source_frames,
    ]);
    const actualDenseFrames = frameIndexSet(state.dense_eligible_cameras);
    if (!sameFrameSet(expectedDenseFrames, actualDenseFrames)) {
      throw new Error(
        "camera-state contract mismatch: dense reference/source frames differ from dense-eligible camera state",
      );
    }
    if ([...actualDenseFrames].some((frameIndex) => !acceptedFrames.has(frameIndex))) {
      throw new Error(
        "camera-state contract mismatch: dense-eligible camera is not part of accepted camera geometry",
      );
    }
    if (result.mesh.reference_frames.some((frameIndex) => !result.dense.reference_frames.includes(frameIndex))) {
      throw new Error(
        "camera-state contract mismatch: mesh patch reference is not a dense reference frame",
      );
    }
  } else if (state.dense_eligible_cameras.length !== 0) {'''
ts = replace_once(ts, old_ts_attempted, new_ts_attempted, "TypeScript attempted dense contract")
write(ts_path, ts)


browser_test_path = "scripts/test-browser-contract.ts"
browser_test = Path(browser_test_path).read_text()
browser_test = replace_once(
    browser_test,
    "  reference_frame: null,\n  source_views: 0,",
    "  reference_frame: null,\n  reference_frames: [],\n  patches: [],\n  source_views: 0,",
    "browser contradiction multi-reference reset",
)
write(browser_test_path, browser_test)


# Evidence viewer: model mode remains surface-first; evidence mode tints each accepted
# patch by its Rust-owned reference-frame provenance and exposes per-patch outcomes.
scene_path = "apps/web/src/SceneCanvas.tsx"
scene = Path(scene_path).read_text()
scene = replace_once(
    scene,
    '''function clampedChannel(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}
''',
    '''function clampedChannel(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}

const SURFACE_PATCH_COLORS = [
  [111, 220, 255],
  [255, 190, 112],
  [173, 255, 151],
  [223, 151, 255],
] as const;

function surfacePatchColor(referenceFrame: number): readonly [number, number, number] {
  return SURFACE_PATCH_COLORS[
    Math.abs(referenceFrame) % SURFACE_PATCH_COLORS.length
  ];
}
''',
    "surface patch palette",
)
scene = replace_once(
    scene,
    '''        const light = 0.48 + facing * 0.52;
        return [
          {
            triangle,
            projected: [projectedA, projectedB, projectedC] as const,
            depth: (projectedA.z + projectedB.z + projectedC.z) / 3,
            r: clampedChannel(((a.r + b.r + c.r) / 3) * light),
            g: clampedChannel(((a.g + b.g + c.g) / 3) * light),
            b: clampedChannel(((a.b + b.b + c.b) / 3) * light),
          },
        ];''',
    '''        const light = 0.48 + facing * 0.52;
        const patchColor = surfacePatchColor(triangle.reference_frame);
        return [
          {
            triangle,
            projected: [projectedA, projectedB, projectedC] as const,
            depth: (projectedA.z + projectedB.z + projectedC.z) / 3,
            r:
              renderMode === "evidence"
                ? clampedChannel(patchColor[0] * light)
                : clampedChannel(((a.r + b.r + c.r) / 3) * light),
            g:
              renderMode === "evidence"
                ? clampedChannel(patchColor[1] * light)
                : clampedChannel(((a.g + b.g + c.g) / 3) * light),
            b:
              renderMode === "evidence"
                ? clampedChannel(patchColor[2] * light)
                : clampedChannel(((a.b + b.b + c.b) / 3) * light),
          },
        ];''',
    "surface patch evidence tint",
)
scene = replace_once(
    scene,
    '''  const meshGridRejected = reconstruction.mesh.rejected_grid_vertices;
''',
    '''  const surfaceCoverageDiagnostic =
    reconstruction.mesh.patches.length > 0
      ? `Surface coverage: ${reconstruction.mesh.surface_patches} accepted reference patches from ${reconstruction.mesh.reference_frames.length} dense references (${reconstruction.mesh.patches
          .map((patch) =>
            patch.accepted_triangles > 0
              ? `frame ${patch.reference_frame}: ${patch.accepted_triangles} triangles`
              : `frame ${patch.reference_frame}: rejected${patch.skip_reason ? ` (${patch.skip_reason})` : ""}`,
          )
          .join("; ")})`
      : "Surface coverage: no reference patch evidence is available";

  const meshGridRejected = reconstruction.mesh.rejected_grid_vertices;
''',
    "surface coverage diagnostic",
)
scene = replace_once(
    scene,
    '''      ? `Surface model: ${reconstruction.mesh.accepted_triangles} accepted triangles; ${meshAdmissionDiagnostic}rejected ${reconstruction.mesh.rejected_discontinuities} discontinuity bridges and ${reconstruction.mesh.rejected_degenerate} degenerate/orientation-flipped candidates`''',
    '''      ? `Surface model: ${reconstruction.mesh.accepted_triangles} accepted triangles; ${surfaceCoverageDiagnostic}; ${meshAdmissionDiagnostic}rejected ${reconstruction.mesh.rejected_discontinuities} discontinuity bridges and ${reconstruction.mesh.rejected_degenerate} degenerate/orientation-flipped candidates`''',
    "surface model coverage text",
)
legend_marker = '''      <div className="viewer-diagnostic" aria-live="polite">
        {primaryDiagnostic}
      </div>'''
legend = '''      {renderMode === "evidence" && reconstruction.mesh.patches.length > 1 ? (
        <div
          style={{
            position: "absolute",
            right: 14,
            top: 14,
            zIndex: 2,
            display: "grid",
            gap: 4,
            padding: "7px 9px",
            borderRadius: 8,
            background: "rgba(4, 10, 13, 0.76)",
            fontSize: "0.68rem",
          }}
          aria-label="Surface patch evidence"
        >
          {reconstruction.mesh.patches.map((patch) => {
            const color = surfacePatchColor(patch.reference_frame);
            return (
              <div key={patch.reference_frame} style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <span
                  aria-hidden="true"
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: 2,
                    background: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
                  }}
                />
                <span>
                  Frame {patch.reference_frame}: {patch.accepted_triangles} triangles
                  {patch.rejected_discontinuities + patch.rejected_degenerate > 0
                    ? ` · ${patch.rejected_discontinuities + patch.rejected_degenerate} rejected`
                    : ""}
                </span>
              </div>
            );
          })}
        </div>
      ) : null}
      <div className="viewer-diagnostic" aria-live="polite">
        {primaryDiagnostic}
      </div>'''
scene = replace_once(scene, legend_marker, legend, "surface patch legend")
write(scene_path, scene)


# Trevi output becomes the durable field-acceptance evidence for multi-reference work.
trevi_path = "crates/video-to-3d-core/examples/trevi_fixture.rs"
trevi = Path(trevi_path).read_text()
trevi = replace_once(
    trevi,
    '    println!("trevi.dense_points={}", reconstruction.dense_points.len());\n    println!("trevi.mesh_triangles={}", reconstruction.mesh_triangles.len());',
    '    println!("trevi.dense_points={}", reconstruction.dense_points.len());\n    println!("trevi.dense_reference_frames={}", reconstruction.dense.reference_frames.len());\n    println!("trevi.mesh_triangles={}", reconstruction.mesh_triangles.len());\n    println!("trevi.mesh_surface_patches={}", reconstruction.mesh.surface_patches);',
    "Trevi multi-reference counters",
)
write(trevi_path, trevi)


# Permanent acceptance gate now requires actual multi-reference surface coverage.
gate_path = ".github/workflows/trevi-real-footage.yml"
gate = Path(gate_path).read_text()
gate = replace_once(
    gate,
    '''          dense="$(value trevi.dense_points)"
          mesh="$(value trevi.mesh_triangles)"
          test "$seed" -eq 2''',
    '''          dense="$(value trevi.dense_points)"
          dense_refs="$(value trevi.dense_reference_frames)"
          mesh="$(value trevi.mesh_triangles)"
          mesh_patches="$(value trevi.mesh_surface_patches)"
          test "$seed" -eq 2''',
    "Trevi gate multi-reference values",
)
gate = replace_once(
    gate,
    '''          test "$dense" -gt 0
          test "$mesh" -gt 0
          echo "Trevi accepted: seed=$seed cameras=$accepted dense_points=$dense mesh_triangles=$mesh"''',
    '''          test "$dense" -gt 0
          if [ "$dense_refs" -lt 2 ]; then
            echo "Trevi did not retain at least two accepted dense reference patches (got $dense_refs)." >&2
            exit 3
          fi
          test "$mesh" -gt 0
          if [ "$mesh_patches" -lt 2 ]; then
            echo "Trevi did not produce at least two accepted mesh surface patches (got $mesh_patches)." >&2
            exit 4
          fi
          echo "Trevi accepted: seed=$seed cameras=$accepted dense_points=$dense dense_refs=$dense_refs mesh_triangles=$mesh mesh_patches=$mesh_patches"''',
    "Trevi gate multi-reference assertions",
)
write(gate_path, gate)


# Mark the implemented roadmap boundary only after this script's workflow validates all layers.
roadmap_path = "ROADMAP.md"
roadmap = Path(roadmap_path).read_text()
roadmap = replace_once(
    roadmap,
    '''- **Multi-reference surface coverage — next.** Run bounded dense/mesh reconstruction from more than one well-supported registered reference view and fuse or co-render mutually consistent surface patches in the shared reconstruction frame. Preserve per-reference visibility and continuity evidence; do not close unsupported holes merely to make the model watertight.
- Texture projection over accepted bounded mesh topology after surface coverage is strong enough that texturing improves a model rather than just decorating sparse fragments.''',
    '''- **Multi-reference surface coverage — integrated.** Select up to three well-supported registered reference views, keeping the strongest reference first and preferring additional cameras separated from it in the accepted reconstruction frame. Run the existing fail-closed depth, reciprocal-consistency, fusion, completion, and mesh gates independently for each reference. Keep reference-grid provenance on every accepted dense sample and triangle, never connect topology across references, and co-render accepted patches in the shared reconstruction frame. Per-reference diagnostics expose accepted and rejected patch outcomes without closing unsupported holes.
- **Texture projection — next.** Project source imagery only over accepted bounded mesh topology now that multiple reference patches can cover a larger scene; preserve visibility and provenance instead of using texture to hide unsupported geometry.''',
    "roadmap multi-reference status",
)
roadmap = replace_once(
    roadmap,
    '''The dense search and mesh topology are still anchored to one selected reference, so larger scenes can remain fragmentary from viewpoints not well covered by that reference. Watertight completion, texture projection, metric scale, full-resolution depth maps, and general multi-reference surface aggregation remain outside the implemented boundary.

Next implementation slice: increase actual model coverage with additional registered reference views and conservative surface-patch fusion before spending more work on texture projection.''',
    '''Dense search remains bounded and per-reference, while accepted patches now share the same Rust-owned reconstruction frame and retain their reference provenance through the browser. Mesh topology never crosses reference grids, so overlap can be inspected without inventing watertight geometry. Cross-reference geometric fusion, texture projection, metric scale, and full-resolution depth maps remain outside the implemented boundary.

Next implementation slice: project textures over accepted multi-reference surface patches, then make dense execution progressive/chunked for larger browser workloads.''',
    "roadmap dense boundary",
)
write(roadmap_path, roadmap)

print("multi-reference surface transformation applied")
