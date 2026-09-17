# Agent guide

- Rust owns reusable reconstruction semantics. Keep feature detection, matching, camera-path estimation, point generation, evidence validation/fusion, and scene data contracts in `crates/video-to-3d-core`.
- Reconstruction backends are evidence providers, not geometry authorities. Classical, learned multi-view, and generative providers must cross the versioned `ReconstructionEvidenceView` boundary before their output can participate in shared validation/fusion.
- Preserve provenance. Geometric multi-view evidence, image-revalidated completion, learned multi-view evidence, and generative completion must remain distinguishable; generative completion must never be silently relabeled as observed geometry.
- Do not duplicate large geometry buffers merely to normalize provider output. Prefer borrowed/ranged evidence views and explicit provenance metadata; materialize snapshots only when an export or external interchange actually requires ownership.
- The browser owns local video decoding, frame sampling, visualization, and transient UI state. Uploaded video must not leave the device.
- `crates/video-to-3d-wasm` is an adapter only; do not duplicate reconstruction logic there.
- Tauri is a shell around the same statically exported Next.js application. Native-only provider execution may be added behind the shared evidence contract, but it must not fork validation/fusion semantics.
- Every algorithmic change needs deterministic Rust tests and a diagnostic signal that makes failure observable in the UI.
