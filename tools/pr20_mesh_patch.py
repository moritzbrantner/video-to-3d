from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1))


mesh = Path("crates/video-to-3d-core/src/mesh.rs")
replace_once(
    mesh,
    "    pub grid_vertices: usize;\n    pub candidate_cells: usize;\n",
    "    pub grid_vertices: usize;\n    pub rejected_grid_vertices: usize;\n    pub candidate_cells: usize;\n",
    "mesh stats field",
)
replace_once(
    mesh,
    '''    if grid.len() < 3 {
        return MeshAnalysis::skipped(
            Some(reference_frame),
            "fewer than three fused dense points align with the reference sampling grid",
        );
    }
''',
    '''    if grid.len() < 3 {
        return MeshAnalysis {
            stats: MeshStats {
                attempted: true,
                skip_reason: Some(
                    "fewer than three fused dense points align with the reference sampling grid"
                        .into(),
                ),
                reference_frame: Some(reference_frame),
                grid_vertices: grid.len(),
                rejected_grid_vertices: dense_points.len().saturating_sub(grid.len()),
                ..MeshStats::default()
            },
            triangles: Vec::new(),
        };
    }
''',
    "mesh low-survivor evidence",
)
replace_once(
    mesh,
    "            grid_vertices: grid.len(),\n            candidate_cells,\n",
    "            grid_vertices: grid.len(),\n            rejected_grid_vertices: dense_points.len().saturating_sub(grid.len()),\n            candidate_cells,\n",
    "mesh accepted-path evidence",
)
replace_once(
    mesh,
    "    #[test]\n    fn triangulates_a_smooth_reference_grid_quad() {\n",
    '''    #[test]
    fn preserves_admission_stats_when_too_few_grid_vertices_survive() {
        let width = 68;
        let height = 48;
        let focal = 60.0;
        let points = vec![
            point_at_pixel(3.0, 3.0, 4.0, width, height, focal, 0.8),
            point_at_pixel(7.0, 3.0, 4.0, width, height, focal, 0.7),
            point_at_pixel(6.0, 7.0, 4.0, width, height, focal, 0.9),
        ];
        let sites = vec![
            DenseGridSite { x: 3, y: 3 },
            DenseGridSite { x: 7, y: 3 },
            DenseGridSite { x: 3, y: 7 },
        ];

        let result = reconstruct_dense_mesh(
            &points,
            &sites,
            &dense_stats(),
            &[camera()],
            width,
            height,
            focal,
        );

        assert!(result.stats.attempted);
        assert!(result.stats.skip_reason.is_some());
        assert_eq!(result.stats.grid_vertices, 2);
        assert_eq!(result.stats.rejected_grid_vertices, 1);
        assert!(result.triangles.is_empty());
    }

    #[test]
    fn triangulates_a_smooth_reference_grid_quad() {
''',
    "mesh low-survivor regression",
)

web_types = Path("apps/web/src/reconstruction.ts")
replace_once(
    web_types,
    "  grid_vertices: number;\n  candidate_cells: number;\n",
    "  grid_vertices: number;\n  rejected_grid_vertices: number;\n  candidate_cells: number;\n",
    "web mesh stats type",
)

canvas = Path("apps/web/src/SceneCanvas.tsx")
replace_once(
    canvas,
    '''  const meshGridRejected = Math.max(
    0,
    reconstruction.dense_points.length - reconstruction.mesh.grid_vertices,
  );
''',
    "  const meshGridRejected = reconstruction.mesh.rejected_grid_vertices;\n",
    "mesh admission UI source",
)
replace_once(
    canvas,
    '''    : reconstruction.mesh.skip_reason
      ? `Mesh skipped: ${reconstruction.mesh.skip_reason}`
      : null;
''',
    '''    : reconstruction.mesh.skip_reason
      ? `Mesh skipped: ${reconstruction.mesh.skip_reason}${
          meshGridRejected > 0
            ? `; ${meshGridRejected} fused points were rejected at mesh-grid admission`
            : ""
        }`
      : null;
''',
    "mesh skipped UI diagnostic",
)

roadmap = Path("ROADMAP.md")
replace_once(
    roadmap,
    "- **Browser registration recovery — integrated.** The WASM adapter keeps the ordinary reconstruction as the fast path, but when the result lacks a calibrated seed or sufficient multi-view evidence it can retry with a bounded search radius informed by image width and observed adjacent displacement, followed by a denser-feature retry with stricter descriptor ambiguity filtering. The adapter deterministically retains only the stronger accepted reconstruction; seed-pair, PnP, bundle-adjustment, dense-depth, and mesh acceptance remain owned by the Rust core.\n",
    "- **Core-owned motion-guided registration recovery — integrated.** `video-to-3d-core::reconstruct` keeps the ordinary local matcher as the fast path. When an adjacent pair is starved, a strict global descriptor consensus estimates only the dominant image displacement and recenters the same bounded local descriptor search around that prediction. If accepted multi-view evidence is still insufficient, the core can retry with a bounded width/displacement-informed radius and then a denser-feature variant with stricter descriptor ambiguity filtering, deterministically retaining only the stronger accepted reconstruction. The WASM crate remains serialization/adaptation only; seed-pair, PnP, bundle-adjustment, dense-depth, and mesh acceptance remain unchanged.\n",
    "roadmap recovery ownership",
)
