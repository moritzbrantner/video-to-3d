# Video to 3D

A privacy-first Rust/Tauri + WebAssembly experiment for reconstructing a 3D scene from ordinary video.

The current implementation has an integrated **Slice 3 sparse-SfM pipeline** and an advancing **Slice 4 dense-reconstruction foundation**. One or more videos can be selected and are decoded locally in the browser, then processed independently and sequentially. For each clip, Rust/WASM detects and matches features, chains adjacent matches into deterministic multi-frame tracks, screens keyframe candidates from overlap and residual parallax, links the actually triangulated landmarks from the strongest calibrated seed pair into other selected keyframes, and can register additional cameras with bounded deterministic robust PnP when their 3D↔2D evidence passes inlier and reprojection gates. A bounded set of selected non-adjacent keyframe pairs is also screened with mutual descriptor matches. Strong direct evidence back to the calibrated seed can recover an otherwise unregistered selected keyframe or propose a bounded correction for an already registered camera whose pose has drifted. Revisit corrections are retained only when the existing rollback-safe bundle-adjustment acceptance also holds. Once an additional view is accepted, non-seed tracks observed by registered cameras can be triangulated into new sparse landmarks when they pass positive-depth, multi-view support, reprojection, and triangulation-angle gates. After the final accepted sparse geometry is established, Rust can run a conservative coarse plane-sweep depth pass over textured reference-frame samples. A primary depth hypothesis is retained only after the existing photometric and ambiguity gates and a reciprocal source-view depth check derived from that source camera's own visible sparse-landmark envelope. Reciprocal source depths are converted back to world-space observations, spatial outliers are rejected, and the surviving multi-view observations are fused into one dense scene point. Accepted fused points can then be connected into a bounded reference-grid triangle surface only when neighboring samples pass explicit depth-continuity, 3D edge-length, and non-degenerate-area gates. Dense points and mesh topology remain separate from the sparse map. If evidence is weak, neither dense points nor surface triangles are invented. If no pair passes the calibrated geometry gates, the app explicitly falls back to the conservative slice-1 preview instead of fabricating a calibrated result.

## Current slice

1. Pick one or more local video files.
2. The browser processes selected videos sequentially and samples up to 18 reduced-resolution frames per video.
3. Rust/WASM detects Harris-style corners and normalized local patch descriptors.
4. Rust matches adjacent frames with a local search, ratio test, and unique-match selection.
5. Adjacent one-to-one matches are chained into deterministic feature tracks spanning multiple sampled frames.
6. Rust selects keyframe candidates using adjacent overlap and accumulated residual parallax, while keeping weak or stationary sequences from inventing useful baselines.
7. Candidate calibrated pairs are normalized using a caller-supplied focal length or an image-size focal estimate.
8. Deterministic eight-point RANSAC estimates an essential matrix and rejects epipolar outliers; a consensus refit is used only when it survives the same downstream geometry-quality gates as the robust winning hypothesis.
9. A rotation-only fit rejects pure camera rotation before it can masquerade as translation.
10. Four relative-pose hypotheses are tested by cheirality; valid inliers are triangulated with linear DLT.
11. Triangulated seed landmarks are associated with their deterministic feature tracks and counted as real 2D↔3D correspondences in other selected keyframes.
12. Another selected keyframe becomes PnP-eligible only when at least eight seed landmarks survive into it.
13. A bounded deterministic robust pose solve uses spatial DLT for well-conditioned 3D landmark sets and a plane-homography decomposition for planar or nearly planar landmark sets. Both paths reject degenerate samples, points behind the camera, and large reprojection residuals, refit on inliers when that improves the accepted model, and require minimum inlier-count, inlier-ratio, and median-reprojection-error gates.
14. Accepted PnP camera centers are added in the same arbitrary monocular coordinate frame as the seed pair.
15. Before any expensive non-adjacent descriptor work, Rust deterministically preselects at most 12 non-adjacent selected-keyframe pairs, prioritizing pairs involving the calibrated seed source and larger temporal separation. Revisit matching uses at most the strongest 192 detected features per frame, performs mutual descriptor matching, and caches the accepted pair evidence so recovery and closure do not repeat the search.
16. Strong screened pairs are exposed as revisit evidence but do not change geometry by themselves. Adjacent pairs are never eligible for this path.
17. For an unregistered selected keyframe, only cached strong non-adjacent evidence back to the calibrated seed frame can map accepted seed features directly to seed 3D landmarks. At least eight such 3D↔2D correspondences are required before the existing robust PnP solver is retried; a failed retry leaves the frame unregistered.
18. Recovered cameras enter exactly the same registered-camera set as ordinary PnP results and therefore participate in subsequent triangulation and bundle adjustment without a second geometry authority.
19. For an already registered selected keyframe with strong direct seed revisit evidence, the same robust seed-landmark PnP produces an independent absolute-pose check. A correction is eligible only when its camera-center disagreement is at most 0.35 seed baselines, its rotation disagreement is at most 8°, and the disagreement is large enough to be meaningful rather than numerical churn.
20. An eligible closure pose is used only as a bounded endpoint correction before another pass through the existing bundle-adjustment authority. The correction is committed only if bundle adjustment passes its ordinary no-regression gate and the final endpoint does not move away from the closure evidence; otherwise cameras, landmarks, bundle-adjustment diagnostics, and the closure acceptance flag all roll back together.
21. Non-seed feature tracks observed by at least one accepted additional camera are triangulated across registered views. New landmarks require positive depth, sufficient observation support, bounded reprojection error, and a minimum triangulation angle before they are added to the sparse cloud.
22. Bundle adjustment gathers only registered observations that already satisfy the positive-depth and 4 px support boundary, then alternates bounded Huber-weighted Gauss–Newton landmark and camera-pose updates. The two calibrated seed cameras remain fixed to preserve the monocular gauge.
23. Adjusted geometry replaces the pre-adjustment reconstruction only when robust cost improves, reprojection RMSE does not regress, and median reprojection error stays within the explicit no-regression boundary. Otherwise the original accepted PnP/triangulated geometry is retained.
24. The first dense pass chooses a reference from the final accepted registered-camera geometry, derives an inverse-depth search envelope from visible accepted sparse landmarks, and evaluates a bounded set of hypotheses on a coarse grid of textured pixels using registered source views.
25. A primary dense hypothesis must have bounded photometric error and enough separation from competing hypotheses before it can proceed. Each eligible source view also derives its own bounded search envelope from the sparse landmarks visible from that camera.
26. Before a primary sample is emitted, Rust projects it into directly supporting source views and independently runs the reverse depth search from the source back toward the reference. At least one source estimate must agree with the projected depth within an 8% relative-depth envelope. The browser exposes how many primary candidates reached this reciprocal veto and how many it rejected.
27. Each reciprocal source depth is unprojected into the shared sparse-SfM coordinate frame. Reverse observations farther than 8% of the primary reference depth from the primary hypothesis are rejected as spatially inconsistent.
28. A primary hypothesis plus at least one spatially consistent reciprocal observation is averaged into one dense scene point. Confidence includes the fraction of eligible source views that survived reciprocal fusion; the browser reports fusion input/rejection evidence.
29. Fused dense points are returned and rendered separately from sparse landmarks.
30. Rust reprojects accepted fused points into the selected dense reference grid and considers only neighboring grid cells with at least three accepted vertices.
31. Candidate surface triangles are retained only when their reference-depth spread, 3D edge footprint, and area pass explicit continuity and non-degeneracy gates; gaps remain gaps instead of being bridged.
32. The browser renders Rust-owned triangle indices as a lightweight surface preview behind the point clouds. This mesh is reference-grid-local and non-watertight; texture projection and general multi-reference surface fusion are not claimed yet.

Scale remains arbitrary because monocular video has no metric baseline. Additional registered cameras, sparse landmarks, and coarse dense samples share the seed pair's coordinate frame. The implemented loop closure is deliberately seed-anchored rather than a general pose graph: it can close bounded drift for a registered selected keyframe that directly revisits the calibrated seed, but it does not add arbitrary non-seed-to-non-seed constraints. The dense path is likewise deliberately incomplete: reciprocal source-view observations are fused into bounded scene points and can form a conservative reference-grid-local triangle surface, but watertight completion, texture projection, general multi-reference surface aggregation, and metric-scale recovery are not implemented yet. Each selected video also remains independent; cross-video reconstruction belongs to slice 6.

## Good footage

Use 5–20 second clips with a slowly translating camera, a static scene, visible texture, stable exposure, and no cuts. Walking sideways or around a small object is better than standing still and rotating in place. Pure camera rotation has no triangulation baseline and is rejected by the calibrated geometry gate. Dense depth additionally benefits from textured surfaces that remain visible from several registered viewpoints.

## Architecture

- `apps/web`: Next.js static export used both by GitHub Pages and Tauri. It owns local decode/sampling and visualization, including rendering Rust-owned dense samples, mesh topology, and reconstruction diagnostics without recomputing geometry.
- `apps/desktop`: thin Tauri 2 shell around the exported web app.
- `crates/video-to-3d-core`: platform-neutral Rust reconstruction kernel, including deterministic feature tracking, keyframe screening, calibrated two-view geometry, seed-landmark track association, spatial and planar robust PnP camera registration, bounded non-adjacent revisit screening, failed-registration recovery, seed-anchored bounded loop closure, multi-view new-landmark triangulation, bounded block-coordinate bundle adjustment, geometry acceptance, conservative coarse registered-view depth estimation, reciprocal source-view depth consistency, bounded dense point fusion, and conservative reference-grid mesh topology.
- `crates/video-to-3d-wasm`: `wasm-bindgen` adapter for the browser.
- `ROADMAP.md`: progression from sparse SfM through dense reconstruction to 3D Gaussian splatting.

Uploaded video stays browser-local. The browser owns decoding, sampling, presentation, and transient per-video batch state; Rust owns reconstruction and dense-depth semantics and output. The static site has no upload API.

## Development

Requirements: Bun 1.4+, stable Rust with `wasm32-unknown-unknown`, and `wasm-bindgen-cli` 0.2.104.

```bash
bun install
cargo install wasm-bindgen-cli --version 0.2.104 --locked
bun run dev
```

For desktop:

```bash
bun run tauri:dev
```

Validation:

```bash
bun run check
```

## GitHub Pages demo

The web app is a Next.js static export. With `GITHUB_PAGES=true`, it builds with the `/video-to-3d` base path, compiles the Rust adapter to browser WebAssembly, and emits the complete site into `apps/web/out`.

The Pages workflow builds the same export on pull requests and deploys `main` through GitHub Pages Actions. The resulting repository Pages URL is expected to be:

`https://moritzbrantner.github.io/video-to-3d/`

GitHub requires the repository's Pages source to be set to **GitHub Actions** once. After that one-time repository setting, pushes to `main` are deployed by the workflow without a separate server or upload backend.