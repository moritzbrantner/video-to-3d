# Reconstruction providers

The authoritative product and architecture boundaries are [Decision Atlas](DECISION-ATLAS.md) IDs `2.1`, `2.2`, `2.3`, `2.4`, and `2.6`. This document describes their technical evidence interface, current integration and implementation steps; it does not establish independent product decisions.

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

## Adoption and integration evidence

This is a **source-backed status snapshot**, not another product Decision Atlas or a mandate
to replace the working renderer/reconstruction path. Status means **where code executes**;
it does not imply the three-real-video scene-to-Blender acceptance target is met.

| Capability/provider | Current role | Source of usage and proof |
| --- | --- | --- |
| Built-in Rust SfM/dense reconstruction | **Active product implementation; not yet product-accepted** | [Core reconstruction](../crates/video-to-3d-core/src/lib.rs), [WASM adapter](../crates/video-to-3d-wasm/src/lib.rs), [evidence contract](../crates/video-to-3d-core/src/reconstruction_evidence.rs), and [real-video/benchmark evidence](../benchmarks/README.md). It runs the ordinary browser path, but passing its current technical gates does not prove editable real-scene acceptance. |
| External COLMAP via `pycolmap==4.2.1` | **Reference/benchmark only; not an in-product reconstruction provider** | [Pinned version](../tools/requirements-colmap.txt), [reference runner](../tools/colmap_golden.py), [golden comparison](../tools/check_colmap_golden.py), and [runtime-profiler scenario](../profiles/runtime-profiler/golden-colmap.json). The same fixture images are compared against the internal Rust path; COLMAP output is **not** ground truth. |
| COLMAP sparse text interchange | **Implemented import/interop capability, not COLMAP execution** | [Rust parser](../crates/video-to-3d-core/src/colmap.rs) reads cameras/images/points. [Spatial Media Explorer](https://github.com/moritzbrantner/spatial-media-explorer/blob/main/Cargo.toml) directly consumes `video-to-3d-core` plus the [3D Lab spatial contract](https://github.com/moritzbrantner/3d-lab/blob/main/crates/three-d-spatial/src/lib.rs) to display existing COLMAP reconstructions and annotations. The Explorer does not run COLMAP itself. |
| COLMAP as an in-product native reconstruction provider | **Proposed; not integrated** | The provider-neutral contract above can accept an adapted provider, but there is no product invocation/admission path for COLMAP reconstruction. Evaluate against the existing golden runner and real-video acceptance before promoting it. |
| Learned multi-view / generative completion | **Provider contract exists; production providers pending** | This document's provider classes describe the interface, not an implemented model-backed path; compare future providers using the same evidence/validation authority. |

**To track adoption:** name the capability, distinguish direct runtime use from
reference/test-only use and format-only interoperability, link the actual import/call
sites and validation evidence, and identify the in-house implementation's missing
consumer requirements. When code changes, update the affected row in that
integration PR. Derive reverse consumers from source dependencies and repository
metadata; do not keep a separate manually mirrored fleet dependency list.
The [benchmark strategy](../benchmarks/README.md) owns measurement details.

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
