# Reconstruction providers

`video-to-3d` treats reconstruction backends as evidence providers rather than as geometry authorities.

The stable architectural direction is:

```text
video / keyframes
      |
      +--> built-in Rust SfM + dense depth
      +--> native learned multi-view provider (future)
      +--> external classical provider (future)
      +--> generative completion provider (future, separate provenance)
                 |
                 v
        ReconstructionEvidenceView
                 |
          core validation/fusion
                 |
          canonical surface output
```

## Current integration

`crates/video-to-3d-core/src/reconstruction_evidence.rs` defines schema version 1 of the provider-neutral evidence contract.

The built-in Rust reconstruction is the first adapter. Its accepted camera poses, dense-reference patch ranges, confidence-bearing dense point buffer, and accepted triangle buffer are exposed through a borrowed evidence view. Large point and triangle buffers are not cloned merely to normalize the provider boundary.

The browser reconstruction path validates this evidence contract and emits a diagnostic through the existing reconstruction warnings. A provider that produces invalid ranges, non-finite geometry, invalid confidence, duplicate cameras, unsupported camera references, or invalid triangle indices fails closed.

## Provenance

Surface evidence keeps an explicit origin:

- `geometric_multi_view`: directly accepted multi-view dense evidence.
- `revalidated_completion`: locally proposed completion that independently re-passed image and multi-view checks.
- `learned_multi_view`: evidence supplied by a learned multi-view reconstruction backend.
- `generative_completion`: geometry inferred without direct multi-view observation support.

Generative completion is intentionally allowed to be camera-free, but it remains explicitly labeled and must not silently become observed geometry.

## Provider classes

The contract distinguishes three broad provider classes without coupling the core to a specific model:

- `geometric_multi_view`
- `learned_multi_view`
- `generative_completion`

A future VGGT-family, MUSt3R-style, COLMAP, TripoSR, TRELLIS, or other adapter should identify itself through a provider descriptor and map its output into the same evidence boundary. No provider should own surface fusion or bypass the Rust validation policy.

## Execution boundary

The browser/WASM path remains the portable baseline and keeps all uploaded video local.

Heavy learned providers should initially run only through an opt-in native/Tauri execution path or another explicitly local process. The native adapter may own model loading and inference, but the returned evidence must enter the same Rust validation/fusion path used by the built-in provider. This avoids a second geometry authority and keeps provider-specific dependencies out of the browser bundle.

## Next slices

1. Make conservative multi-reference surface fusion consume the common evidence regions rather than provider-specific dense patch metadata.
2. Add an owned/interchange form of schema v1 for native provider process boundaries while keeping the in-process view zero-copy.
3. Implement one learned multi-view native adapter behind the contract and benchmark it against the built-in path and the existing COLMAP reference fixtures.
4. Add hybrid mode: geometric camera/consistency constraints remain authoritative while learned depth contributes confidence-bearing evidence.
5. Add optional generative completion only after observed/inferred geometry is finalized, retaining provenance in export and visualization.

Provider quality should be compared using the same deterministic fixtures and reconstruction metrics rather than by changing validation thresholds per backend.
