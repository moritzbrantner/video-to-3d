# Independent semantic-inventory acceptance (#60, first slice)

Authority: Decision Atlas 4.3, 4.5, 4.6, 8.7 (still in force where 8.8 does
not replace it) and **8.8** (owner answers on #60, 2026-10-17), issue #60 and
its comments, `AGENTS.md`. Source baseline: `origin/main` at `e73d999`. These
tests were written by an independent acceptance context **before** any
production code. The implementer must not weaken them; see "Protected and
translatable" below.

## Scope of this slice

Native only. A provider operation `semantic_inventory` in the CLI/ProjectStore
consumes browser-owned keyframes (`ingest_video` output; optionally
`sparse_reconstruction` for registered-frame identity) and records a versioned,
provider-neutral semantic evidence artifact. The model runtime is a pinned
Python/PyTorch sidecar (Grounding DINO + SAM 2.1, weights fetched on first need
into a local cache, idempotently). The operation boundary and artifact are
runtime-independent so in-process ONNX (#216) and browser inference (#214) can
replace the sidecar without changing consumers. UI is #215; identity (#61),
mask consistency (#62) and object packages (#63) consume this artifact later.

Semantics are evidence only: no geometry or camera authority, no 3D fields,
and no geometry producer may consume them.

## Proposed public API (names translatable)

- `video_to_3d_core::semantic_inventory` module with
  `SemanticInventoryArtifact` (`from_json` parses + validates fail-closed,
  `to_json` canonical, `validate`, `validate_against_keyframes(&KeyframesArtifact)`),
  `SEMANTIC_INVENTORY_SCHEMA_VERSION = 1`, `default_room_vocabulary() -> Vec<String>`.
- `OperationKind::SemanticInventory` / `ArtifactKind::SemanticInventory`,
  serialized `semantic_inventory`; execution `Provider`; allowed upstream ⊆
  {`ingest_video`, `sparse_reconstruction`} and must consume exactly one
  `ingest_video`.
- `OperationDeclaration.vocabulary: Option<Vec<String>>` — per-project
  override, only on semantic operations, non-empty, no blank or duplicate terms,
  order significant, part of the semantic operation identity only.
- `BuiltInExecutor::with_semantic_sidecar(program)`; otherwise the
  `VIDEO_TO_3D_SEMANTIC_SIDECAR` env var (an executable path); otherwise the
  implementation's pinned default sidecar.
- Receipt observations `segmentation` = `available` | `unavailable` and
  `segmentation_reason` (shown by `video-to-3d build` under
  `operations.<id>.observations`).

## Artifact (schema version 1)

```json
{
  "schema_version": 1,
  "status": "available | unavailable",
  "unavailable_reason": "required and non-blank iff unavailable",
  "provider": {
    "id": "manifest provider id", "revision": "manifest provider revision",
    "models": [{"role": "detector|segmenter", "id": "...", "version": "...", "url": "..."}],
    "runtime": {"name": "pytorch", "version": "...", "device": "cuda|cpu", "device_name": "NVIDIA GeForce RTX 3060 Ti"}
  },
  "vocabulary": {"source": "default | project", "terms": ["verbatim", "..."]},
  "frames": [{"frame_index": 0, "width": 360, "height": 270, "content_hash": "sha256:..."}],
  "observations": [{
    "id": "obs-0", "frame_index": 0, "label": "<vocabulary term>",
    "bbox": [x_min, y_min, x_max, y_max],
    "detection_confidence": 0.7,
    "mask": {"encoding": "rle_row_major", "width": 360, "height": 270, "counts": [bg, fg, bg, ...]},
    "mask_confidence": 0.9
  }],
  "tracks": [{"id": "track-0", "label": "...", "observations": ["obs-0", "obs-1"], "association_confidence": 0.8}],
  "uncertainties": [{"kind": "missing|weak|ambiguous", "frame_index": 2, "label": "...", "observation": null, "reason": "..."}],
  "measurements": {"elapsed_seconds": 3.5, "peak_gpu_memory_bytes": 1500000000}
}
```

Conventions: `bbox` is continuous pixel coordinates of the keyframe, origin
top-left, `0 <= x_min < x_max <= width`, `0 <= y_min < y_max <= height`.
`rle_row_major` counts alternate background/foreground starting with
background (the first run may be 0), sum exactly `width * height`, mask size
equals the frame size, at least one foreground pixel, and every foreground
pixel `(col, row)` must intersect the box (`col + 1 > x_min && col < x_max`,
same for rows). `mask` and `mask_confidence` are both present or both null.

Shared validation fails closed on (core tests): unknown schema version;
frame/observation/uncertainty frame index not among the recorded frames;
duplicated frames; frames not matching the source keyframes (index, size,
content hash); boxes outside the frame, degenerate, inverted or malformed;
non-finite numbers; masks inconsistent with frame size, run lengths, outside
their box, empty or of unknown encoding; confidences outside [0, 1];
observation labels or uncertainty labels outside the recorded vocabulary;
tracks that are empty, repeat or reference missing observations, or mix
labels; duplicate observation/track ids; unknown uncertainty kinds or blank
reasons; `unavailable` with observations/tracks or without a reason;
`available` with a reason; missing provider id/revision, model identity
(id/version/url) or runtime for available evidence; missing or invalid
vocabulary; negative measurements; **any unknown field**, explicitly including
cameras, poses, points, meshes, depth, world positions or provenance claims.

Default room vocabulary contains at least wall, door, window, floor, ceiling,
chair and table or desk (owner granularity decision on #60: individual
architecture elements and meaningful furniture/objects are separate
identities; individuality is carried by observations and tracks, not by
vocabulary terms).

## Sidecar protocol `video-to-3d-semantic-sidecar/1`

Invocation: `<program> <request.json> <response.json>` with absolute paths.
The program writes the response and exits 0. Local only: no remote endpoints
in the request, frames never leave the device; weight download on first need
is the sidecar's concern.

Request (written by Rust):

```json
{
  "protocol": "video-to-3d-semantic-sidecar/1",
  "provider": {"id": "...", "revision": "..."},
  "vocabulary": ["terms in recorded order"],
  "frames": [{"frame_index": 0, "width": 640, "height": 480,
              "content_hash": "sha256:...", "rgba_path": "/abs/path/raw-rgba8"}]
}
```

Every keyframe is offered; `rgba_path` is readable raw row-major RGBA8 whose
hash equals `content_hash`.

Response (validated by Rust with the shared artifact validation; Rust supplies
schema version, provider id/revision, vocabulary and frames):

```json
{"protocol": "video-to-3d-semantic-sidecar/1", "status": "available",
 "models": [...], "runtime": {...}, "observations": [...], "tracks": [...],
 "uncertainties": [...], "measurements": {...}}
```

or `{"protocol": "...", "status": "unavailable", "reason": "..."}` (for
example no CUDA device). Outcomes the native tests pin:

| Sidecar behaviour | Required result |
| --- | --- |
| valid available response | operation `succeeded`, artifact + receipt, reuse on rebuild without re-running the sidecar |
| `status: unavailable` | recorded unavailable artifact keeping the sidecar's reason verbatim, no observations, vocabulary recorded |
| program cannot be started | recorded unavailable artifact with a reason; build otherwise completes; exit code not 1/2/64 |
| response fails shared validation (box outside frame, label outside vocabulary) | operation `failed` (exit 4), no artifact, no receipt; geometry completes |
| non-zero exit / crash (e.g. OOM) | never available evidence: `failed` or an unavailable record |
| cancellation while running | sidecar process killed and reaped, `canceled`, no artifact/receipt; resume succeeds |
| unavailable earlier, provider runs now | next build re-executes and records available evidence (no stale unavailable reuse) |

Vocabulary override or provider revision changes re-execute only the semantic
operation; cameras, dense, mesh and assembly bytes are identical with and
without the semantic operation.

## Real-model acceptance

`crates/video-to-3d-cli/tests/semantic_inventory_real.rs` (`--ignored`) uses
the pinned #179 browser capture (`VIDEO_TO_3D_REAL_ROOM_INPUT`,
`VIDEO_TO_3D_REAL_ROOM_WITNESS_HASH`; local input recorded in
`real-room-native-acceptance.md`), the default or `VIDEO_TO_3D_SEMANTIC_SIDECAR`
sidecar and a project vocabulary override starting with `yellow die`. On the
GPU it requires: available evidence with detector and segmenter identity,
`runtime.device = cuda`, a `device_name` containing `3060`, measured
`elapsed_seconds > 0` and `0 < peak_gpu_memory_bytes <= 8 GiB`; die-like
(`die`/`dice`/`cube`) masked observations on at least two of the
independently inventoried visible samples (0–15 and 72–86 of the dense
87-sample capture, #230); one die-like track
linking at least two of them; and an uncertainty record for every visible
sample without a die mask. No pixel positions are asserted (the inventory is
not a pixel annotation). The CPU-only smoke (`CUDA_VISIBLE_DEVICES=""`)
requires an unavailable record with a reason and completed geometry. The
media input of that project is the pinned capture document, not a native
decode. `VIDEO_TO_3D_SEMANTIC_REPORT` collects the measured report.

```sh
VIDEO_TO_3D_REAL_ROOM_INPUT=/home/moenarch/.local/share/coding-agent/video179-independent-browser-input \
VIDEO_TO_3D_REAL_ROOM_WITNESS_HASH=sha256:f45e0cf9ade9a203dc52da6be87c6154ef5c501b454eedbc3d6ec63f46bea92d \
VIDEO_TO_3D_SEMANTIC_REPORT=/tmp/semantic-real.json \
cargo test --release -j 6 -p video-to-3d-cli --test semantic_inventory_real -- --ignored --test-threads 1
```

## Protected and translatable

Protected (changing them needs independent re-verification against the Atlas):
every fail-closed case above, unavailable honesty, no invented masks, label ⊆
vocabulary, verbatim override recording, identity/invalidation scope, reuse,
cancellation semantics, geometry byte-identity, no geometry producer consuming
semantics, the real-model thresholds (≥2 visible-sample die masks, ≥2-sample
track, measured card/memory/time, honest gaps) and the 8 GiB bound.

Translatable by the implementer as a mechanical rename (record it in the PR):
Rust module/type/function names, JSON field names and enum spellings, the env
var name, the protocol string and receipt observation keys — provided each
test keeps the same inputs, cases and assertions. Mask encoding may change
only to an equally compact, exactly specified encoding in keyframe pixel space
with the same validation cases. Test-double scripts and fixtures are
acceptance material: changes need independent re-verification.

Not automated: no-network inference after the weight cache is populated, and
weight licence review (8.8 removed hash ceremony; a pinned version and URL is
the identity).

## Baseline (source `e73d999`, before production code)

- `cargo test -j 6 -p video-to-3d-core --test semantic_inventory_contract`:
  **compile-RED** — unresolved `video_to_3d_core::semantic_inventory`, missing
  `OperationKind::SemanticInventory`, `ArtifactKind::SemanticInventory`,
  `OperationDeclaration::vocabulary`.
- `cargo test -j 6 -p video-to-3d-cli --test semantic_inventory_native`:
  **compile-RED** — the same missing module/variants plus
  `BuiltInExecutor::with_semantic_sidecar`.
- `semantic_inventory_real` (ignored tier): **compile-RED** for the missing
  module; runtime additionally needs the real sidecar (PyTorch is not
  installed on the measurement machine yet).

Every one of these must pass on the implementation PR head; the ignored tier
must be run on the RTX 3060 Ti and its report attached.
