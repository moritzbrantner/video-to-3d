from pathlib import Path

path = Path("crates/video-to-3d-core/src/mesh.rs")
text = path.read_text()

wrong_early = '''            reference_frame: Some(reference_frame),
            reference_frames: vec![reference_frame],
            surface_patches: usize::from(!build.triangles.is_empty()),
            patches: Vec::new(),
            grid_vertices: grid.len(),
            rejected_grid_vertices: patch_point_count.saturating_sub(grid.len()),
            ..MeshStats::default()
'''
right_early = '''            reference_frame: Some(reference_frame),
            reference_frames: vec![reference_frame],
            surface_patches: 0,
            patches: Vec::new(),
            grid_vertices: grid.len(),
            rejected_grid_vertices: patch_point_count.saturating_sub(grid.len()),
            ..MeshStats::default()
'''
if wrong_early not in text:
    raise SystemExit("expected early mesh stats patch was not found")
text = text.replace(wrong_early, right_early, 1)

final_old = '''            reference_frame: Some(reference_frame),
            grid_vertices: grid.len(),
            rejected_grid_vertices: patch_point_count.saturating_sub(grid.len()),
            candidate_cells,
'''
final_new = '''            reference_frame: Some(reference_frame),
            reference_frames: vec![reference_frame],
            surface_patches: usize::from(!build.triangles.is_empty()),
            patches: Vec::new(),
            grid_vertices: grid.len(),
            rejected_grid_vertices: patch_point_count.saturating_sub(grid.len()),
            candidate_cells,
'''
if final_old not in text:
    raise SystemExit("expected final mesh stats block was not found")
text = text.replace(final_old, final_new, 1)

path.write_text(text)
print("corrected mesh aggregate field placement")
