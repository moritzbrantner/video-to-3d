# Video to 3D

A privacy-first Rust/Tauri + WebAssembly experiment for reconstructing a 3D scene from ordinary video.

The current MVP is a **sparse reconstruction preview**, not a COLMAP replacement yet. Video is decoded locally in the browser, sampled into frames, passed to Rust/WASM for corner detection and adjacent-frame matching, then converted into an approximate camera trajectory and colored sparse point cloud. The browser only visualizes Rust-owned reconstruction output.

## MVP

1. Pick one or more local video files.
2. The browser processes selected videos sequentially and samples up to 18 reduced-resolution frames per video.
3. Rust/WASM detects Harris-style corners and builds normalized local patch descriptors.
4. Rust matches adjacent frames with a local search, ratio test, and unique-match selection.
5. The MVP estimates camera translation and approximate depth from observed parallax.
6. A canvas viewer displays each video's camera path and sparse colored points, with pair-by-pair diagnostics.

Multiple files are intentionally reconstructed independently in this slice. Cross-video feature tracks, shared calibration, and one combined multi-video scene belong to later multi-view work.

This intentionally stops short of claiming calibrated multi-view geometry. The approximation is useful for proving the full product loop and for collecting evidence about feature quality and footage suitability before introducing heavier SfM machinery.

## Good MVP footage

Use 5–20 second clips with a slowly translating camera, a static scene, visible texture, stable exposure, and no cuts. Walking sideways or around a small object is better than standing still and rotating in place. Pure camera rotation produces little triangulation parallax and should be reported as low-parallax input.

## Architecture

- `apps/web`: Next.js static export used both by GitHub Pages and Tauri.
- `apps/desktop`: thin Tauri 2 shell around the exported web app.
- `crates/video-to-3d-core`: platform-neutral Rust reconstruction kernel.
- `crates/video-to-3d-wasm`: `wasm-bindgen` adapter for the browser.
- `ROADMAP.md`: progression from this sparse preview to calibrated SfM, dense reconstruction, and 3D Gaussian splatting.

Uploaded video stays browser-local. The static site has no upload API.

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
