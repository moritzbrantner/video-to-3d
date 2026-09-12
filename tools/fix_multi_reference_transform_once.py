from pathlib import Path

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
print("exported multi-reference patch evidence types")
