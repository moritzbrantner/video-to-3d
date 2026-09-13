from pathlib import Path


dense_path = Path("crates/video-to-3d-core/src/dense.rs")
dense = dense_path.read_text()
old_dense_stats = """            reference_frame: Some(reference.frame_index),
            source_views: source_views.len(),"""
new_dense_stats = """            reference_frame: Some(reference.frame_index),
            reference_frames: vec![reference.frame_index],
            patches: Vec::new(),
            source_views: source_views.len(),"""
if old_dense_stats not in dense:
    raise SystemExit("expected dense per-reference stats initializer was not found")
dense = dense.replace(old_dense_stats, new_dense_stats, 1)
dense_path.write_text(dense)


path = Path("crates/video-to-3d-core/src/reconstruction.rs")
text = path.read_text()

old_dense = "pub use dense::DenseStats;"
new_dense = "pub use dense::{DensePatchStats, DenseStats};"
if old_dense not in text:
    raise SystemExit("expected dense public re-export was not found")
text = text.replace(old_dense, new_dense, 1)

old_mesh = "pub use mesh::{MeshStats, MeshTriangle};"
new_mesh = "pub use mesh::{MeshPatchStats, MeshStats, MeshTriangle};"
if old_mesh not in text:
    raise SystemExit("expected mesh public re-export was not found")
text = text.replace(old_mesh, new_mesh, 1)

path.write_text(text)
print("repaired multi-reference stats initialization and exported patch evidence types")
