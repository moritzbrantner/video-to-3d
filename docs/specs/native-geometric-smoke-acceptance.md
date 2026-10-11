# Independent acceptance: native geometric scene smoke (#150)

Authority: issue #150 and its owner decision B (2026-10-11), the public staged
contract in #172, `AGENTS.md`, and
[the whole-scene product decisions](whole-scene-product-decisions.md).
Baseline: `3553210` (PR #173 merged). This acceptance was established in a separate
context before native executor implementation.

## Outcome and boundary

One `video-to-3d build project.json` invocation, after a content-verified sampled
keyframe handoff, computes sparse cameras, dense evidence, a consolidated observed
surface, and a canonical assembled scene with a consumable environment mesh.
The actual CLI, `ProjectStore`, and `scene_runner` own execution, receipt verification,
attempt bounds, scheduling, invalidation and resume. Do not add another executor
or geometry authority. Browser video decoding and frame sampling remain external
and local; native operations must not decode a video or upload input/derived pixels.

The deterministic smoke reuses the **actual golden lateral renderer** and the
existing browser/default reconstruction options. Its source is explicitly labeled
as generated sampled frames, not an encoded video or a real acceptance clip. The
fixture importer writes only the upstream keyframes and their receipt; it supplies
no cameras, dense evidence, mesh, assembled scene or fabricated object inventory.
The reconstructed environment is an observed preview in arbitrary monocular units.

This is the bounded first slice promised by #150, not completion of #77's three
real clips, object decomposition, meaningful editable furniture/architectural
inventory, learned/generative completion, #84/#151 Blender export/import, or #76's
aggregate product acceptance. Those existing issues remain the owners of their
requirements. The synthetic-room fixture and its independent inventory remain
unchanged; this smoke does not reinterpret its object oracle.

## Executable contract

`crates/video-to-3d-cli/tests/native_build.rs` exercises the actual CLI binary from
an unrelated working directory and the public runner/store seam:

- The unmodified golden renderer with default options must yield calibrated
  camera evidence, nonempty dense points and accepted triangles (fixture sanity).
- Cold native build reuses imported keyframes and independently computes the
  three reconstruction operations and scene assembly. Every output has a versioned,
  deterministic receipt whose identity, path, byte length and content hash agree
  with the actual file and manifest record.
- Sparse/dense artifacts use the public versioned core stage contracts and retain
  the camera/request content identities. Native mesh output uses the existing
  `SurfaceMeshArtifact`, validates through `ReconstructionEvidenceView`, and equals
  the single-call classic evidence on all serialized fields.
- The canonical `AssembledScene` has real cameras and an environment visual whose
  mesh resource is hash-verified GLB data. Its provenance is geometric multi-view;
  the smoke does not claim editable object decomposition.
- Rebuild executes no operations. Dense corruption rebuilds only dense, mesh and
  assembly while preserving the upstream sparse bytes and verified ingest receipt.
  Corrupting the assembled scene's mesh resource invalidates its scene receipt and
  rebuilds assembly rather than reporting a current unusable scene.
- Cancellation after actual sparse computation persists its receipt, dispatches no
  descendants, and resumes to byte-identical stage/scene outputs of a cold run.
- Missing imported keyframes remain explicitly unsupported at ingest with blocked
  descendants and no fabricated outputs. An unavailable declared local provider
  remains nonpassing and blocks downstream scene assembly without a cloud fallback.

Additional existing acceptance is reused without weakening its expectations:
`scene_runner_tests.rs` covers scheduling, cancellation, provider fallback and
bounded retries; `scene_store_tests.rs` covers sidecar validation, content identity,
cache transparency and crash receipts; `tests/reconstruction_stages.rs` covers
schema/type/edited-input rejection, stored stage resumption and recovery parity;
`crates/video-to-3d-cli/tests/cli.rs` covers command JSON/status/inspection contracts.
Existing browser, evidence/provenance, golden, reference and room gates remain
required at implementation completion.

## Architecture review and completion evidence

Each native operation computes only its own stage. Sparse serialization must not
contain pixels or dense/mesh outputs; dense must not embed another camera/sparse
buffer; mesh/assembly materialize owned external interchange only when needed.
Borrowed stage inputs and content references preserve the #172 ownership boundary.
No extra copies of large geometry buffers merely to fill adapter structures. These
ownership/work guarantees require direct implementation review because the current
repository has no allocation/solver-count instrumentation proving them mechanically.

Use `cargo test --release -p video-to-3d-cli --test native_build` for the smoke
contract; release mode avoids making the generated geometric fixture needlessly
expensive in debug. Normal workspace tests still discover this contract. Require
all recorded acceptance plus the repository's `bun run check` completion gate. The
inspection tool reports undeclared dependency relationships at these paths, so no
proven affected-test selection exists: run the full applicable suite before merge.
Record unavailable desktop/tooling checks honestly and use actual hosted CI evidence
for any local environment gap. A native headless technical enabler does not require
an artificial Pages route; the real CLI and content-verified scene files are its
observable surface.
