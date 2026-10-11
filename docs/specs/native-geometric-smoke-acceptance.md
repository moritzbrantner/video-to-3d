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
  mesh resource is hash-verified GLB data. Its provenance exactly preserves the
  standalone classic evidence, including separately labeled image-revalidated
  completion when present; learned/generative geometry cannot be fabricated;
  the smoke does not claim editable object decomposition.
- Rebuild executes no operations. Dense corruption rebuilds dense and restores
  identical deterministic bytes while preserving upstream and valid descendants.
  Descendants rebuild only when their verified input content actually changes, as
  independently covered by the protected existing store contract.
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

## Approved room smoke reproduction

Issue #150's owner explicitly permits #75's deterministic synthetic room until
#74 pins the controlled real clip. Generate it through the unchanged canonical
renderer, then run the same native acceptance contract against its exported frames:

```bash
cargo run --release -p video-to-3d-core --example synthetic_room_fixture -- "$HOME/.local/share/coding-agent/video-150-controlled-room" controlled-room
VIDEO_TO_3D_ROOM_FIXTURE="$HOME/.local/share/coding-agent/video-150-controlled-room" cargo test --release -p video-to-3d-cli --test native_build
```

The reader consumes only original exported PPM frames; it adds no renderer or
geometry authority. `controlled-room-inputs.sha256` pins all 18 original PPM bytes
(manifest SHA-256 `284fb813c1e79a951f41da02d87e888f5689840fd4d7161b91fac9ca7d4e83fa`).
Provenance: canonical #75 seed `0x00755ce9e0ac1e00`, notional 1920×1080/16s,
18 sampled 360×203 frames, 3×3 ray averaging, timestamps 0.8..15.2s inclusive.
This is generated room evidence, never a genuine filmed room. Baseline renderer
reports 766 dense points and 120 triangles; the independent nonempty geometry
check passes. Native acceptance is expected red: five pass, five fail because
native sparse reconstruction is unsupported. Both golden and this room invocation
must pass after implementation. Keep #150 open until that demonstration succeeds;
#74/#77 still require the owner-approved real controlled recording and handhelds.

## Independent review regression contract (4606303)

The public in-memory `BuiltInExecutor::new` must execute a cold complete pipeline
through the existing runner/store. A constructor snapshot must not strand later
waves: mesh and assembly must resolve the verified sparse/dense ancestry actually
produced in that run. File-backed CLI success does not prove this public path.

Sparse, dense, mesh and assembly implementation identities must differ from their
pre-native revision-1 identities. Opaque revision-1 receipts must become stale and
rebuild through current semantics, rather than being reused or consuming a dense
attempt on an incompatible sparse payload. Revision identity coverage is necessary
but the migration gate must also exercise an actual legacy receipt and descendants.

Unsupported artifact-kind combinations are rejected before payload parsing even
when the arity is correct. Reserved native output declarations are unsupported,
write no result and charge zero attempts. Removing that declaration allows the same
operation identity to run with the default one-attempt budget. Malformed supported
payloads remain failures; do not hide genuine corruption as unsupported.

The adapter already owns dense evidence. Consolidation must consume that ownership
without allocating a second full dense point buffer. An independently observable
core test must record its existing point allocation pointer/capacity, pass ownership
through mesh consolidation, and assert preserved allocation and exact borrowed-path
mesh/output parity. Pointer preservation alone cannot detect a temporary clone;
source/call-graph review must confirm no full working copy. The existing borrowed
API remains immutable and byte-identical. Function names and tuple/struct result
shape are implementation translation details, not an additional geometry authority.

Cancellation-aware variants of the same staged solvers must accept a cheap
`FnMut() -> bool` callback. Each real golden fixture stage must abort without an
artifact when cancellation is requested after its initial poll, with deterministic
observable intermediate work polling. Tests compare uncanceled variants byte-for-byte
with existing public staged APIs, and exercise cancellation at more than one
intermediate work boundary. Exact implementation poll counts and wall-clock limits
are not acceptance requirements. Native dispatch must wire its existing token into
these variants and publish neither a success receipt nor partial accepted output.
Existing between-wave cancellation remains a separate regression.

## Independent cancellation adjacency repair (frozen9f7e485)

Review4243558325 identifies native assembly reading with a token, then calling
SurfaceMeshArtifact::from_json with no-op validation and evidence() with a second
whole-mesh noncancelable validation. Existing
scene_artifacts_tests::surface_mesh_parsing_polls_for_cancellation_during_validation
already independently proves cancellation after entry aborts validation of the
real accepted mesh fixture. Reuse it; do not manufacture a new parser authority.
The added parsed_trusted_mesh_reuses_borrowed_evidence_without_geometry_copy test
proves validated parsing can yield the existing borrowed evidence_view sharing
owned point/triangle allocations, exact geometry/provenance and regions.

Native assembly must parse/validate once with the actual token and reuse that
trusted borrow, avoiding another raw/evidence validation traversal. This wiring
is a mandatory source/call-graph audit, not mechanically proven by the parser unit
test. GLB consumers must not reintroduce token-free whole-geometry validation at
this same trusted boundary; record any unavoidable independent export trust check
explicitly and carry cooperative cancellation through expensive work. No new
schema/provenance label or geometry authority is introduced.

Review4243558328 identifies vertex_surface_evidence's initial whole-triangle
prepass outside controlled fusion. fusion_prepass_acceptance_tests exercises that
actual private boundary with real Point3/triangle data. The proposed routine
vertex_surface_evidence_with_control returns no accepted evidence on cancellation
requested at callbacks2/4, after initial entry; noop control preserves normals,
local edge scales, incident triangle membership and median edge scale versus the
original helper and a simple exact geometric oracle. API/result spelling is
independently translatable; no arbitrary wall-time/thread-race test is allowed.

Callback counts alone cannot distinguish work-loop polling from dummy entry polls.
Review actual triangle processing, point-derived normalization/scales and edge
sorting/cancellation boundaries after repair: no whole prepass may remain
uncontrolled before later fusion polls. The existing controlled fusion caller must
pass the same sticky ReconstructionControl, not a fresh no-op/second solver.
Repeated triangles here are deterministic private prepass inputs, not product
scene or object-recognition evidence. Baseline helper lacks the controlled seam:
new prepass tests are compile-red until distinct production repair; assembly parse/
borrow tests can be green while native wiring remains defective. Record chronology
as independent repair acceptance before this fix, not original feature tests-first.

### Supplemental native GLB cancellation audit after 88ce75d

This coverage is authored after the repair, not a pre-production RED contract.
`native_glb_cancel_acceptance_tests.rs` reuses the existing independently authored
`surface_materials_tests::Fixture`, validates its real evidence, and constructs the
same all-triangle vertex-color fallback as native assembly. The controlled no-op
export must equal the public export byte for byte. Cancellation after initial
polling, actual assignment/fallback triangle work, position components and accessor
writes returns an error with no output. The source audit, not callback counts alone,
establishes that polls occur inside real loops; no timing bound is imposed.

On 88ce75d, native assembly calls `SurfaceMeshArtifact::from_json_with_cancel` with
the active token exactly once and subsequently borrows `evidence_view`. It does not
call the revalidating `evidence()` accessor. The GLB encoder does not revalidate the
mesh; it checks bake coverage and polls triangle/position/color/index/accessor work,
chunked buffer copying and final output writes. Fusion's triangle normal/edge
prepass polls each triangle, then normal/local-scale calculations each vertex;
edge-length sorting is atomic with checks on either side. Classic-part evidence
construction uses one cancel-aware validation traversal without duplicate points.

Limits: serde JSON parsing/serialization, individual numerical SVD calls and sorts
remain indivisible operations. `classic_camera_evidence`, classic region construction,
`evidence_support_membership` and `support_group_references` prepare metadata without
internal polling; membership range fills and provenance range fills are indivisible.
These remaining windows are disclosed, not claimed canceled during their execution.
The fixed findings concerning raw assembly validation and whole triangle prepass
are resolved by actual token wiring and work-loop polls. This does not prove hard
real-time cancellation or full product acceptance (#77), Blender/portable export
(#84/#151), desktop availability or owner-room approval.


### Independent final stage serialization cancellation (#174 review4243648144)

Frozen baseline `5aadd49` still calls noncancelable dense/mesh `to_json()` at
`scene_artifacts/native.rs` final stage output. Solver completion does not make
walking every geometry array an acceptable atomic cancellation window. Independent
`stage_serialization_cancel_acceptance_tests.rs` was authored before this repair.
It uses actual outputs of the existing deterministic golden multiview fixture,
including dense points/grid sites/attributes and mesh triangles. Its optional
source-corner array is empty on the independently verified frozen baseline.

Proposed routine `to_json_with_cancel(FnMut()->bool)` names may be independently
translated to a shared serializer/write seam. False cancellation must produce
byte-exact existing `to_json()` and a round-trippable artifact. Cancellation at
callbacks2/4, early emission32, halfway through actual uncanceled polls and just
before completion must return an error containing `canceled`, never successful or
partial serialized output, stop before the full uncanceled poll count, and leave
source arrays unchanged. Nonfinite dense point/mesh confidence must fail validation
instead of serde converting it to null. Initial baseline is compile-red solely
for eight missing-method diagnostics; no new runtime test is claimed executed.

**Mandatory source audit before closure:** active native token reaches the actual
Serde serialization sink while it emits array/scalar bytes, and any required
finite/schema validation traversal. Use existing serde_json serializer with a
cancel-aware `std::io::Write` Vec sink (or equivalent existing Serde boundary), not
a second JSON parser/serializer/schema. Entry/exit checks or an artificial polling
prewalk followed by atomic `to_string` cannot satisfy this contract even if callback
counts appear green. On cancellation discard the private sink and propagate the
canceled error; do not publish bytes/artifact/build outputs. Native dense/mesh
callers must invoke this path instead of default `to_json` with the same build
cancellation token. Existing native operation stop/no downstream artifact contracts
remain applicable. Canceling a large one-shot string write should also be bounded
rather than copying the whole string before polling.

After this repair is validated, update the earlier atomic-JSON limitation to
separate cancelable stage serialization from still-atomic JSON parsing, numerical
SVD, sorts and disclosed nonserialization metadata windows. Until then, the old
limitation describes baseline only and is not an acceptance exemption. This is
not hard real-time cancellation, reconstruction recognition or product #77 closure.

### Actual native mesh JSON read/write boundaries before repair

Supplemental protected coverage authored against frozen parser baseline `9f7e485`
and current acceptance head `0ef55b8` before the JSON repair. Existing mesh fixture
vertices/provenance are retained; 8192 repetitions of its valid triangles produce
about1.5MB of actual geometry JSON, not whitespace or dummy pre-work polls. Native
mesh validation accepts that fixture. No-op controlled parsing equals existing
parsed artifact serialization and geometry. Adding trailing invalid JSON makes
existing parsing fail; a callback requesting cancellation after reads2/4 must instead
abort before reaching the trailing syntax error. Frozen9f is genuinely one GREEN/
one RED: cancellation test receives trailing characters with polls=0. The after4
iteration remains unexecuted in that fail-fast RED run.

`native_mesh_json_write_acceptance_tests.rs` separately protects the **actual**
SurfaceMeshArtifact used by native SurfaceMesh output: false callback equals existing
pretty JSON plus newline, mid-write cancellation returns no output, and nonfinite
geometry rejects. Current0ef compiles RED because actual SurfaceMeshArtifact lacks
to_json_with_cancel (earlier MeshStageArtifact tests are complementary, not proof
of this native branch). API spelling may be translated independently; preserve
legacy byte format and active-token wiring, not an alternate artifact format.

Source audit required after repair: use existing serde deserialization through an
actual cancel-aware Read, and serde serialization through actual cancel-aware Write;
no custom parser/schema, input pre-scan, whitespace padding or dummy callback loop.
Both version-header skip traversal and typed full traversal must use that control.
Native CameraStageArtifact/DenseStageArtifact readers currently use the analogous
two uncancelable serde traversals; their native read call sites must receive the
active token as well. Those analogous paths are source-audit requirements, not new
runtime coverage claimed by the two mesh parser tests. Existing constructor/evidence
validation must preserve its token, schema/hash/provenance and no large duplicate
geometry guarantees.

All four native output branches require source-audited active-token writers:
SparseCamera/DenseEvidence preserve compact stage JSON; SurfaceMesh preserves actual
SurfaceMeshArtifact pretty JSON/newline; SceneAssembly preserves assembled-scene
pretty JSON/newline. No-op callback parity and nonfinite safeguards must remain.
These linear JSON walks are now explicit required cooperative boundaries; earlier
notes disclosing atomic JSON were honest limitations, not proof of interruption.
Individual SVD/sorts remain their separately disclosed indivisible operations.


Independent fixture correction after serialization implementation began: frozen
`9f7e485` golden Lateral run through actual cameras/dense/mesh reports629 dense
points,629 grid sites,629 attributes,9 mesh triangles and **zero** triangle sources.
The initial protected serialization fixture incorrectly asserted nonempty sources,
so its six tests stopped before reaching their new serializer assertions. Correcting
that source claim to the actual empty optional array preserves every geometry,
byte-parity, cancellation and nonfinite assertion. No geometry is manufactured to
force source-corner coverage. Reuse the same existing golden module across both
acceptance files instead of including example/common twice (strictClippy duplicate_mod).
This fixture correction follows production work and is not claimed test-first; the
original missing-serializer compile RED remains honest baseline evidence.

### Independent verification of JSON repair 96cc102

Read-only source audit confirms the active token reaches Sparse/Dense compact
writers, actual SurfaceMeshArtifact pretty/newline writer and assembled-scene
pretty/newline writer. Both version-header skip and typed Serde read traversals use
the bounded Read (8KiB refills). Dense ancestry header skips geometry through that
same reader; camera ancestry identity hashes stream the existing compact serialization
through a cancel-aware hashing writer. SurfaceMesh parses/validates once with token,
then assembly borrows trusted evidence. Writes poll actual bounded Serde fragments,
not a pre-work callback loop, and preserve UTF8 and existing byte format.

Independent protected runs on96cc102:22 cancellation-filter cases,2 fusion-prepass
cases,1owned-allocation/byte-parity case and1trusted-borrow case pass. Supplemental
actual native mesh witness (authored after96cc102) counts its exact real validation
polls and total no-op polls, then cancels halfway through remaining emission and
near completion **after** validation. Both abort without output or source mutation;
this strengthens the earlier2/4cases that could stop during validation alone.

Using the producer-built current96cc102 release CLI, a private copy of the retained
canonical#75 synthetic room project first reuses all5verified receipts. Corrupting
only its own dense output causes one real dense execution, restores every original
native artifact hash and preserves valid descendants. A following build executes
nothing. Removing only that copy's native outputs executes all4actualnative stages
(sparse/dense/mesh/assemble), preserves imported sampled-frame evidence, and restores
all original native file hashes. EnvironmentGLB remains8216bytes, SHA256
`d030900c4f98b2f14ce1e406648abe49d7ea178da2a8fcaaa11eaaac8dd9c5df`.
Local evidence is under coding-agent/video174-independent-room-*.json; copied project
video174-json-independent-room. This is real deterministic synthetic geometry,
not owner-filmed-room or#77 product acceptance. No desktop/Blender result is claimed.
Individual SVD, std sorts and disclosed metadata/range-fill windows remain atomic;
this is cooperative cancellation, not a fixed response-time guarantee.
