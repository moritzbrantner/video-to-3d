# Roadmap

## Slice 1 — Sparse reconstruction MVP

- One browser-local video input.
- Deterministic keyframe sampling.
- Rust/WASM corner detection and patch descriptors.
- Adjacent-frame feature matching with diagnostics.
- Approximate camera trajectory and sparse colored point cloud.
- Interactive 3D preview shared by web and Tauri.
- Static Next.js export ready for GitHub Pages.

Exit criterion: a suitable short handheld clip produces visible camera motion, a recognizable sparse spatial structure, and enough diagnostics to explain low-quality reconstructions.

## Slice 2 — Calibrated two-view geometry

- Camera intrinsics input/estimation.
- Normalized coordinates.
- Essential/fundamental matrix estimation under RANSAC.
- Relative rotation/translation recovery with cheirality checks.
- True linear triangulation.
- Reprojection-error diagnostics.

Exit criterion: synthetic and recorded two-view fixtures recover known camera motion and 3D points within explicit error bounds.

## Slice 3 — Multi-view sparse SfM

- Keyframe selection based on overlap and parallax.
- Multi-frame feature tracks rather than isolated pair matches.
- Incremental camera registration.
- PnP for newly registered views.
- Bundle adjustment over cameras and sparse landmarks.
- Loop/revisit handling and failed-registration recovery.

Exit criterion: longer videos produce a stable sparse model without unbounded trajectory drift.

## Slice 4 — Dense reconstruction

- Depth-map estimation from registered views.
- Multi-view consistency filtering.
- Dense point fusion.
- Optional mesh reconstruction and texture projection.
- Memory-aware native and WASM execution strategies.

## Slice 5 — 3D Gaussian splatting

- Initialize splats from registered cameras and sparse/dense points.
- Optimize position, covariance, opacity, and spherical-harmonic appearance.
- Native GPU training path first; evaluate WebGPU training separately.
- Browser splat viewer and portable export format.
- Quality/performance comparisons against dense point and mesh representations.

## Slice 6 — Multiple videos and production hardening

- Multiple clips for the same scene.
- Session persistence and resumable processing.
- Camera/lens profiles and metadata import.
- Background native processing in Tauri.
- Deterministic project manifests and reproducible exports.
- Larger-scene chunking and progressive visualization.
