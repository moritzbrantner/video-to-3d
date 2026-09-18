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
- **Core-owned motion-guided registration recovery — integrated.** `video-to-3d-core::reconstruct` keeps the ordinary local matcher as the fast path. When an adjacent pair is starved, a strict global descriptor consensus estimates only the dominant image displacement and recenters the same bounded local descriptor search around that prediction. If accepted multi-view evidence is still insufficient, the core can retry with a bounded width/displacement-informed radius and then a denser-feature variant with stricter descriptor ambiguity filtering, deterministically retaining only the stronger accepted reconstruction. The WASM crate remains serialization/adaptation only; seed-pair, PnP, bundle-adjustment, dense-depth, and mesh acceptance remain unchanged.
- **Camera evidence labeling — integrated.** The browser renders pre-calibration adjacent-motion samples as a dim approximate path and labels them separately from accepted registered cameras. Registered-camera counts are never inferred from the fallback motion preview.

Current boundary: Rust can register additional selected camera poses, recover an otherwise unregistered selected keyframe from strong direct non-adjacent seed evidence, grow the sparse map from genuinely supported non-seed tracks, jointly refine accepted cameras and landmarks, and close bounded accumulated drift when a registered selected keyframe directly revisits the calibrated seed. Descriptor recurrence alone never changes geometry; every recovery or closure still passes explicit 3D↔2D and reprojection gates. The calibrated seed-pair cameras remain fixed, preserving the arbitrary monocular coordinate frame. This is not a general pose graph: arbitrary non-seed-to-non-seed loop constraints, metric-scale recovery, and cross-video tracks remain outside this slice.

Implementation exit criterion met: the in-product sparse pipeline has fail-closed registration, map growth, refinement, bounded seed-return drift correction, and an external deterministic quality/runtime reference without transferring product ownership to COLMAP.

## Slice 4 — Dense reconstruction — in progress

- **Coarse registered-view depth estimation — integrated.** Choose a reference from the final accepted sparse camera geometry, derive a bounded depth-search envelope from visible accepted sparse landmarks, evaluate a memory-bounded inverse-depth plane sweep over textured sample pixels, and retain only photometrically supported, non-ambiguous depth hypotheses. Emit accepted dense samples separately from the sparse map with explicit diagnostics; do not fabricate dense geometry when texture, baseline, or matching evidence is insufficient.
- **Reciprocal depth consistency — integrated.** Give each eligible source view its own sparse-landmark-derived depth-search envelope. A primary reference-view hypothesis that already passed texture, photometric-support, and ambiguity gates is retained only when at least one directly supporting source view independently selects a reverse-search depth within an 8% relative-depth envelope of the projected candidate. Surface how many primary candidates reached this veto and how many it rejected.
- **Dense point fusion — integrated.** Convert each accepted reciprocal source depth back into Rust-owned world geometry, reject reverse observations that are spatially inconsistent with the primary hypothesis, and fuse the remaining multi-view observations into one confidence-aware scene sample without duplicating geometry. Rust-owned diagnostics distinguish failures at the earlier texture/photometric gates, reciprocal consistency, and the later spatial/minimum-observation fusion gate.
- **Bounded reference-grid mesh reconstruction — integrated.** Carry each accepted dense sample's original reference-grid site alongside its fused 3D position so fusion cannot change topology identity. A fused vertex remains mesh-eligible only while its reference reprojection stays strictly inside the half-stride footprint of that original sample site. Connect neighboring original sample sites, preserve positive signed winding and a minimum projected area in the reference image, and retain triangles only when relative depth, 3D edge length, and non-degenerate 3D area remain inside explicit fail-closed gates.
- **Surface-first model output — current slice.** Treat the mesh as the product output rather than letting diagnostic point clouds visually dominate. Geometry controls viewer framing instead of the camera path. Accepted surface triangles render as an opaque shaded model by default; sparse points, dense samples, and camera markers move to an explicit evidence view. Rust runs up to two bounded local completion passes over empty dense-grid sites: at least three neighboring accepted samples must agree within a 6% depth spread before they may propose a depth, and the proposal becomes geometry only after it independently re-passes reference texture, direct cross-view photometric support, reciprocal source-depth consistency, spatial fusion, and original-grid-footprint checks. The mesh can additionally bridge exactly one missing grid sample using only existing accepted vertices and a stricter 8% triangle depth-continuity gate.
- **Provider-neutral reconstruction evidence — integrated.** Schema v1 normalizes accepted cameras, surface provenance regions, confidence-bearing points, and accepted topology behind `ReconstructionEvidenceView`. The built-in Rust reconstruction is the first provider adapter. The in-process view borrows existing point/triangle buffers instead of rematerializing them, validates exact provenance coverage and provider/camera/topology invariants, and emits a browser-visible diagnostic. Learned multi-view and generative providers have explicit, distinct provenance classes and cannot silently become geometric evidence.
- **Multi-reference surface coverage — next.** Run bounded dense/mesh reconstruction from more than one well-supported registered reference view and fuse or co-render mutually consistent surface patches in the shared reconstruction frame. Preserve per-reference visibility and continuity evidence; do not close unsupported holes merely to make the model watertight. Move cross-reference fusion toward the common evidence regions so provider-specific patch structures do not become the long-term fusion API.
- Texture projection over accepted bounded mesh topology after surface coverage is strong enough that texturing improves a model rather than just decorating sparse fragments.
- Memory-aware native and WASM execution strategies, including progressive/chunked processing where full-resolution depth would exceed practical browser memory budgets.
- **Native provider adapters — planned.** Add an owned/interchange form of the evidence schema for an opt-in local native process, then evaluate one learned multi-view backend against the built-in reconstruction and existing COLMAP fixtures. Model inference may be native-only, but evidence validation/fusion remains Rust-owned and shared with the browser baseline.
- **Hybrid reconstruction — planned.** Keep accepted geometric camera/consistency constraints authoritative while allowing learned depth/confidence to contribute evidence through the same validation path. Generative completion remains a later, separately labeled post-reconstruction stage.

Current boundary: Rust turns reciprocal-consistent source depth estimates into explicit world-space observations, applies a bounded spatial-consistency gate, fuses accepted observations with the primary hypothesis, and can use already accepted neighboring surface evidence to propose additional interior samples. Those proposals are not accepted by interpolation alone: they must re-pass the same image and multi-view evidence before joining the dense geometry. Grid identity and half-stride admission remain authoritative, and mesh construction still rejects winding flips, excessive depth jumps, long edges, and degenerate triangles. The provider-neutral evidence contract now gives these accepted outputs a common provenance-aware boundary without copying the geometry buffers or transferring validation authority to an external model. Watertight completion, metric scale, full-resolution depth maps, provider-native execution, and general cross-provider surface aggregation remain outside the implemented boundary.

Next implementation slice: reconcile the active multi-reference seam-fusion work with `ReconstructionEvidenceView`, then add the owned native-provider interchange form before integrating a learned backend.

## Slice 4A — Learned evidence and escalation — planned

The learned path strengthens observations and proposes reconstruction evidence; it does not replace Rust-owned geometry validation, fusion, or provenance.

- **Learned local features — planned.** Evaluate XFeat as the browser-capable learned feature baseline, first as an optional provider beside the current detector/descriptor path. Preserve a cheap deterministic path and compare accepted geometry, not just match counts.
- **Adaptive learned matching — planned.** Evaluate LightGlue as an escalation when the ordinary matcher lacks enough reliable correspondence evidence. Keep pair selection, essential-matrix/PnP acceptance, and downstream geometry gates Rust-owned.
- **Dense correspondence recovery — planned.** Evaluate RoMa for difficult wide-baseline, revisit, or texture-poor pairs where sparse matching fails. Dense confidence remains evidence and must be geometrically revalidated before it can affect cameras or surfaces.
- **Learned calibration priors — planned.** Evaluate AnyCalib and GeoCalib for focal length, principal point, lens/distortion, and gravity priors. These are proposals only; accepted intrinsics must survive the same reconstruction consistency checks as manually supplied or image-size-derived values.
- **Long-range point tracks — planned.** Evaluate TAPNext++ and CoTracker3 as optional multi-frame track providers. Learned trajectories must cross Rust epipolar, visibility, positive-depth, triangulation, reprojection, and bundle-adjustment gates before becoming geometry.
- **Interchangeable native multi-view providers — planned.** Add one native research adapter through the owned form of `ReconstructionEvidenceView`. Use MapAnything as a research integration surface where practical so Pi3X, VGGT-Ω, Depth Anything 3, MASt3R/MUSt3R/Pow3R-class providers can be compared without creating model-specific geometry authority.
- **Conditioned dense reconstruction — planned.** Prefer providers that can consume already accepted cameras, intrinsics, or depth so the classical Rust solution can condition learned dense geometry instead of being discarded.
- **Streaming reconstruction research — planned.** Compare LingBot-Map and STream3R for long-video bounded-state processing, keyframe/cache policies, and drift recovery. Borrow the state-management ideas even when the full models remain native-only.
- **Dynamic-scene geometry — later.** Evaluate 4RC/MonST3R-class approaches for explicit scene motion so moving people, vehicles, water, and other dynamic content can be modeled or rejected geometrically instead of relying only on semantic masks.
- **Browser deployment rule.** Prefer compact ONNX/WebGPU or equivalent browser-local models only when their download, memory, and latency budgets remain practical. Large multi-view transformers stay opt-in native/Tauri research providers.
- **Trevi Pages field canary — integrated.** Keep an openly licensed Trevi Fountain clip available directly in GitHub Pages as a one-click real-video smoke test. Treat it as a difficult field canary with camera motion, water, and changing local appearance, not as ground-truth quality evidence; synthetic fixtures and reference datasets remain authoritative for regression metrics.

Exit criterion: at least one learned observation provider improves accepted reconstruction evidence on deterministic fixtures and the Trevi field canary without bypassing Rust-owned acceptance, and at least one interchangeable native multi-view provider is benchmarked through the common evidence boundary.

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
