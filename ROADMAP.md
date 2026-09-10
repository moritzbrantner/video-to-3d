# Roadmap

## Slice 1 — Sparse reconstruction MVP — integrated

- One browser-local video input.
- Deterministic keyframe sampling.
- Rust/WASM corner detection and patch descriptors.
- Adjacent-frame feature matching with diagnostics.
- Conservative approximate camera trajectory and sparse colored point cloud.
- Interactive 3D preview shared by web and Tauri.
- Static Next.js export ready for GitHub Pages.

Exit criterion met: the full browser-local video → Rust/WASM → sparse preview loop is integrated, with low-parallax and pure-motion safeguards instead of fabricated camera movement.

## Slice 2 — Calibrated two-view geometry — implementation integrated

- Normalize matched image coordinates with a supplied focal length or image-size focal estimate.
- Estimate an essential matrix with deterministic eight-point RANSAC and a confidence-derived adaptive trial budget.
- Retain the robust winning hypothesis and accept a consensus refit only when it survives the same geometry-quality gates; expose Sampson-error evidence for the selected model.
- Reject rotation-only degeneracy before accepting a translation baseline.
- Recover relative rotation/translation from the four essential-matrix pose hypotheses.
- Select pose by cheirality and minimum triangulation angle.
- Triangulate true two-view landmarks with linear DLT and filter by reprojection error.
- Select the strongest valid adjacent pair while retaining an explicitly labeled conservative fallback.

Implementation status: deterministic synthetic fixtures and hosted checks cover the calibrated geometry path. Recorded-footage acceptance should still be captured before treating the slice as fully field-accepted.

## Slice 3 — Multi-view sparse SfM — current

- Initial keyframe screening based on adjacent feature overlap and accumulated residual parallax.
- Deterministic multi-frame feature tracks chained from one-to-one adjacent matches.
- Associate triangulated seed landmarks with their multi-frame tracks.
- Screen other selected keyframes for real seed-landmark 2D↔3D correspondence readiness.
- Register eligible additional cameras with bounded deterministic robust PnP and explicit inlier/reprojection acceptance gates.
- Bounded non-adjacent selected-keyframe revisit screening using mutual descriptor evidence.
- Failed-registration recovery through direct calibrated-seed-frame associations, reusing the existing robust PnP acceptance gate and leaving failed retries unregistered.
- Triangulate new landmarks from newly registered or recovered views with positive-depth, multi-view support, reprojection, and triangulation-angle acceptance gates.
- Bundle adjustment over cameras and sparse landmarks, with a fixed calibrated seed gauge and explicit no-regression acceptance.
- Bounded global drift correction from geometrically validated non-adjacent revisit constraints without introducing metric-scale claims.

Current boundary: Rust can register additional selected camera poses, recover an otherwise unregistered selected keyframe when strong direct non-adjacent evidence back to the calibrated seed frame yields enough accepted seed-landmark correspondences for the existing robust PnP gate, grow the sparse map from genuinely supported non-seed tracks, and jointly refine the accepted registered cameras and sparse landmarks with bounded Huber-weighted block-coordinate bundle adjustment. Revisit evidence and recovery attempts are explicit outputs; descriptor recurrence alone never changes geometry. The calibrated seed-pair cameras remain fixed, preserving the arbitrary monocular coordinate frame. The next implementation slice is bounded global drift correction using validated non-adjacent constraints rather than treating a recovery pose as full loop closure.

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