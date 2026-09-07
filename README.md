# Video to 3D

A privacy-first Rust/Tauri + WebAssembly experiment for reconstructing a 3D scene from ordinary video.

The current implementation is **calibrated two-view sparse reconstruction**, not a COLMAP replacement yet. One or more videos can be selected and are decoded locally in the browser, then processed independently and sequentially. For each clip, Rust/WASM detects and matches features, screens adjacent pairs for useful parallax, fits essential matrices with deterministic RANSAC, recovers relative camera pose by cheirality, and linearly triangulates the strongest valid pair. If no pair passes the calibrated geometry gates, the app explicitly falls back to the conservative slice-1 preview instead of fabricating a calibrated result.

## Current slice

1. Pick one or more local video files.
2. The browser processes selected videos sequentially and samples up to 18 reduced-resolution frames per video.
3. Rust/WASM detects Harris-style corners and normalized local patch descriptors.
4. Rust matches adjacent frames with a local search, ratio test, and unique-match selection.
5. Candidate pairs are normalized using a caller-supplied focal length or an image-size focal estimate.
6. Deterministic eight-point RANSAC estimates an essential matrix and rejects epipolar outliers; a consensus refit is used only when it survives the same downstream geometry-quality gates as the robust winning hypothesis.
7. A rotation-only fit rejects pure camera rotation before it can masquerade as translation.
8. Four relative-pose hypotheses are tested by cheirality; valid inliers are triangulated with linear DLT.
9. The strongest adjacent pair is displayed with its registered camera centers, colored sparse points, Sampson error, reprojection error, and triangulation-angle evidence.

Scale remains arbitrary because a monocular two-view reconstruction has no metric baseline. Each selected video is reconstructed independently in this slice. Cross-video feature tracks, shared camera registration, PnP, and bundle adjustment belong to slice 3.

## Good footage

Use 5–20 second clips with a slowly translating camera, a static scene, visible texture, stable exposure, and no cuts. Walking sideways or around a small object is better than standing still and rotating in place. Pure camera rotation has no triangulation baseline and is rejected by the calibrated geometry gate.

## Architecture

- `apps/web`: Next.js static export used both by GitHub Pages and Tauri.
- `apps/desktop`: thin Tauri 2 shell around the exported web app.
- `crates/video-to-3d-core`: platform-neutral Rust reconstruction kernel, including deterministic two-view geometry and geometry acceptance.
- `crates/video-to-3d-wasm`: `wasm-bindgen` adapter for the browser.
- `ROADMAP.md`: progression from sparse SfM to dense reconstruction and 3D Gaussian splatting.

Uploaded video stays browser-local. The browser owns decoding, sampling, presentation, and transient per-video batch state; Rust owns reconstruction semantics and output. The static site has no upload API.

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
