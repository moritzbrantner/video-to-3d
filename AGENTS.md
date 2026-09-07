# Agent guide

- Rust owns reusable reconstruction semantics. Keep feature detection, matching, camera-path estimation, point generation, and scene data contracts in `crates/video-to-3d-core`.
- The browser owns local video decoding, frame sampling, visualization, and transient UI state. Uploaded video must not leave the device.
- `crates/video-to-3d-wasm` is an adapter only; do not duplicate reconstruction logic there.
- Tauri is a shell around the same statically exported Next.js application. Do not fork desktop reconstruction behavior.
- Every algorithmic change needs deterministic Rust tests and a diagnostic signal that makes failure observable in the UI.
