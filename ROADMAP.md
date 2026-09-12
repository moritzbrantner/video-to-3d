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

## Slice 3 — Multi-view sparse SfM — integrated

- Initial keyframe screening based on adjacent feature overlap and accumulated residual parallax.
- Deterministic multi-frame feature tracks chained from one-to-one adjacent matches.
- Associate triangulated seed landmarks with their multi-frame tracks.
- Screen other selected keyframes for real seed-landmark 2D↔3D correspondence readiness.
- Register eligible additional cameras with bounded deterministic robust PnP and explicit inlier/reprojection acceptance gates.
- Bounded non-adjacent selected-keyframe revisit screening using mutual descriptor evidence.
- Failed-registration recovery through direct calibrated-seed-frame associations, reusing the existing robust PnP acceptance gate and leaving failed retries unregistered.
- Triangulate new landmarks from newly registered or recovered views with positive-depth, multi-view support, reprojection, and triangulation-angle acceptance gates.
- Bundle adjustment over cameras and sparse landmarks, with a fixed calibrated seed gauge and explicit no-regression acceptance.
- Seed-anchored bounded drift correction for already registered selected keyframes when validated direct non-adjacent evidence back to the calibrated seed produces an independent robust PnP pose. Closure candidates are limited by seed-normalized center disagreement and rotation disagreement, then must survive the existing bundle-adjustment no-regression gate; otherwise the entire closure attempt rolls back.
- Deterministic external acceptance matrix against COLMAP on rendered scenes with known camera trajectories, plus runtime-profiler and Moonlight evidence. COLMAP remains a CI reference rather than a product dependency or geometry authority.

Current boundary: Rust can register additional selected camera poses, recover an otherwise unregistered selected keyframe from strong direct non-adjacent seed evidence, grow the sparse map from genuinely supported non-seed tracks, jointly refine accepted cameras and landmarks, and close bounded accumulated drift when a registered selected keyframe directly revisits the calibrated seed. Descriptor recurrence alone never changes geometry; every recovery or closure still passes explicit 3D↔2D and reprojection gates. The calibrated seed-pair cameras remain fixed, preserving the arbitrary monocular coordinate frame. This is not a general pose graph: arbitrary non-seed-to-non-seed loop constraints, metric-scale recovery, and cross-video tracks remain outside this slice.

Implementation exit criterion met: the in-product sparse pipeline has fail-closed registration, map growth, refinement, bounded seed-return drift correction, and an external deterministic quality/runtime reference without transferring product ownership to COLMAP.

## Slice 4 — Dense reconstruction — in progress

- **Coarse registered-view depth estimation — integrated.** Choose a reference from the final accepted sparse camera geometry, derive a bounded depth-search envelope from visible accepted sparse landmarks, evaluate a memory-bounded inverse-depth plane sweep over textured sample pixels, and retain only photometrically supported, non-ambiguous depth hypotheses. Emit accepted dense samples separately from the sparse map with explicit diagnostics; do not fabricate dense geometry when texture, baseline, or matching evidence is insufficient.
- **Reciprocal depth consistency — integrated.** Give each eligible source view its own sparse-landmark-derived depth-search envelope. A primary reference-view hypothesis that already passed texture, photometric-support, and ambiguity gates is retained only when at least one directly supporting source view independently selects a reverse-search depth within an 8% relative-depth envelope of the projected candidate. Surface how many primary candidates reached this veto and how many it rejected.
- **Dense point fusion — integrated.** Convert each accepted reciprocal source depth back into Rust-owned world geometry, reject reverse observations that are spatially inconsistent with the primary hypothesis, and fuse the remaining multi-view observations into one confidence-aware scene sample without duplicating geometry. Rust-owned diagnostics distinguish failures at the earlier texture/photometric gates, reciprocal consistency, and the later spatial/minimum-observation fusion gate.
- **Bounded reference-grid mesh reconstruction — current slice.** Carry each accepted dense sample's original reference-grid site alongside its fused 3D position so fusion cannot change topology identity. A fused vertex remains mesh-eligible only while its reference reprojection stays strictly inside the half-stride footprint of that original sample site. Connect only neighboring original sample sites, require each candidate triangle to preserve positive signed winding and a minimum projected area in the reference image, and retain it only when relative depth, 3D edge length, and non-degenerate 3D area also remain inside explicit fail-closed gates. The output is a local surface preview, not a watertight model.
- Texture projection over accepted bounded mesh topology.
- Memory-aware native and WASM execution strategies, including progressive/chunked processing where full-resolution depth would exceed practical browser memory budgets.

Current boundary: Rust now turns reciprocal-consistent source depth estimates into explicit world-space observations, applies a bounded spatial-consistency gate, fuses accepted observations with the primary hypothesis, preserves each sample's original grid identity, rejects fused vertices that leave their original half-stride footprint, and derives a bounded reference-grid triangle surface only across neighboring samples that preserve positive reference-image winding and pass projected-area, depth-continuity, 3D edge, and non-degeneracy gates. The primary search and mesh topology are still anchored to one selected reference. Watertight surface completion, texture projection, metric scale, full-resolution depth maps, and general multi-reference surface aggregation remain outside the implemented boundary.

Next implementation slice: project image appearance onto accepted bounded mesh topology while keeping Rust-owned geometry authoritative, preserving visibility/continuity gates, and making missing texture evidence explicit rather than filling it heuristically.

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