# Reconstruction benchmark strategy

The benchmark stack separates fast pull-request gates from larger real-world canaries.

## Pull-request golden matrix

Every reconstruction pull request is evaluated on deterministic rendered scenes with known camera trajectories. The same images are passed to `video-to-3d` and COLMAP.

Current cases:

- `lateral`: the original sideways-translation baseline; strong parallax and stable overlap.
- `revisit`: the camera moves across the scene and then returns, exercising non-adjacent evidence, recovery, and loop/drift behavior.
- `forward`: forward-dominant motion with a smaller lateral component, exercising weaker triangulation geometry without becoming a deliberately degenerate case.

The gate checks registration coverage, sparse-map size, reprojection error, and camera-center RMSE against the known trajectory after similarity alignment. Runtime-profiler captures Rust and COLMAP independently for every case. Moonlight continues to compare repository revisions.

Synthetic trajectory truth is authoritative only for these generated scenes. COLMAP remains a reference implementation, not the owner of product geometry.

## Real-image benchmark candidates

### COLMAP South Building

Use as the first real-world sparse-SfM canary. COLMAP documents South Building as a 128-image dataset captured with one camera, and publishes the archive as a release asset.

- Source: https://colmap.github.io/datasets.html
- Pinned asset: https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip
- Published archive size: 419,421,847 bytes

This dataset is useful for end-to-end robustness and performance comparison, but it does not provide independent geometric ground truth. It should therefore remain a canary/reference comparison rather than replace the known-trajectory gate.

### Strecha Fountain-P11 and Herz-Jesu-P8

These are strong next datasets for independent sparse-pose validation because they are classic calibrated multi-view benchmarks with camera parameters and LIDAR-backed scene ground truth. Fountain-P11 has 11 high-resolution images and Herz-Jesu-P8 has 8.

They are a better fit than South Building for absolute pose/geometry scoring once a stable, checksum-pinned source is incorporated into the pipeline.

### ETH3D

ETH3D provides laser-scanner ground truth across varied indoor/outdoor multi-view scenes and is a strong later-stage benchmark. Its high-resolution datasets are substantially larger and its primary evaluation is dense multi-view accuracy/completeness, so it is better suited once `video-to-3d` advances beyond sparse SfM.

- Source: https://www.eth3d.net/datasets

## Gate policy

- Pull requests must keep the deterministic golden matrix green.
- Runtime ratios remain evidence until repeated measurements establish stable envelopes per scene.
- A reference implementation may veto a clear quality regression, but representation-specific counts do not need to match exactly.
- Real-image datasets should be pinned to an immutable or checksum-verified source before becoming required gates.
- Larger public datasets belong in slower scheduled/manual lanes unless their cost is small enough to justify every pull request.
