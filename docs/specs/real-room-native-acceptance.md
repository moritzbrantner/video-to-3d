# Independent approved real-room native acceptance (#179)

Current authority is Atlas 3.1, 3.8, 4.3 and 6.1–6.3 and issue #179, the remaining
real-input successor of #150. Issue #187 / PR #188 merged at
`6cf3e4eb512c918377e90cdaf615fe007d3cea06` establishes the approved original,
fixed sample plan and browser-decodable carrier. The owner priority recorded on
#137 (comment 6114048524) parks the older #171 polish work behind this native
scene execution. This is a technical execution enabler, not the three-video
object-decomposition/Blender product exit.

## Independent input and boundary

`scripts/capture-real-room-native-acceptance.ts` is an acceptance input author.
It validates the exact original SHA and length against the unchanged independent
controlled-room expectations, validates the prepared carrier and receipt against the independent
`approved-real-room-capture-identity.json` pins, full
interval, plan and source binding, then bundles the existing production
`sampleVideoWithInfo`. Actual Chromium decodes the carrier and retains all 18
360×270 RGBA buffers. Browser-side SHA-256 and host-side SHA-256 must agree for
each retained buffer. Requested times and actual decoded presentation times are
both retained; native keyframes use the decoded times. The approved whole-source
plan remains exact metadata; the browser's duration rounding may move requested
times by at most one original frame interval (1/30 second).

No offline PPM, TUM depth/trajectory, screenshot, generated scene or second
decoder stands in for these pixels. Browser pixel identity is specific to the
recorded browser/version/run; no cross-browser or offline pixel equivalence is
claimed. The capture JSON's externally recorded content hash binds metadata,
pixel identities, source/carrier identity and browser version. Unchanged bytes,
not a future decoder rerun, establish deterministic imported receipt reuse.

Initial actual capture at unchanged production `6cf3e4e`: Chromium
`156.0.8078.4`, 18 retained nonempty frames; transfer witness
`sha256:f45e0cf9ade9a203dc52da6be87c6154ef5c501b454eedbc3d6ec63f46bea92d`.
Local input is `/home/moenarch/.local/share/coding-agent/video179-independent-browser-input`.
The original remains the exact 26,026,292-byte file with SHA-256
`3200d7a76f5b3307e33863d4b38dbad414527ce14084f4c1f6303a0f69eb794c`.

## Native execution contract

`crates/video-to-3d-cli/tests/real_room_native.rs` independently validates that
transfer and its externally pinned hash. A narrow test-only importer feeds actual
`SampledFrame` values through `write_project_keyframes_artifact` and ProjectStore.
It writes only ingest output; Store owns actual operation identities, receipts,
hashes, accounting and persistence. Original movie bytes are a verified local
project input. Reconstruction, assembly and diagnostic authority remain the
existing native stages and the real `video-to-3d build`, run from unrelated cwd.
No new native decoder, generic ingest command or second orchestration is needed.

Protected outcomes:

- A changed pixel fails independent transfer validation. Updating its manifest
  hash too still fails the externally pinned witness identity.
- The real CLI must execute sparse reconstruction, reuse actual ingest, emit
  one valid machine-readable result and expose any earliest unsupported/failed
  stage usefully. Internal, malformed-project and usage errors do not satisfy
  native execution acceptance. A nonpassing geometry result remains nonpassing.
- Every persisted artifact must have a matching real receipt/hash; full Store
  reconciliation must accept it. An unchanged replay performs no new work and
  keeps every completed artifact byte-identical, even if later stages cannot run.
- Corrupting actual imported RGBA after receipt creation must invalidate ingest
  through real CLI reconciliation and block all reconstruction stages, without
  inventing a replacement decoder or scene.
- A second real project cancels only after a successful actual sparse attempt.
  Both cold and interrupted execution must have sparse success plus a verified
  receipt and retained artifact before this cancellation/resume proof can pass.
  Later stages must have no receipts. Resumption must agree with the cold result
  and preserve exact completed artifact bytes. If sparse itself fails, later
  geometry and post-success cancellation assertions cannot be claimed passed.
- Actual stage documents, CLI cold/replay/resume results and inspect diagnostics
  are retained in the optional external JSON report. A completed canonical scene
  must validate; this does not assert recognizable/editable object quality.

## Actual independent baseline

Against unchanged production `6cf3e4e`, all **three explicit native contracts
pass**, zero ignored, in 1.13 seconds (after disposable release compilation).
Log `/tmp/video179-independent-native-baseline-final.log`; actual report
`/home/moenarch/.local/share/coding-agent/video179-independent-native-baseline.json`.
Sparse, dense, mesh and assembly execute successfully, actual ingest is reused,
unchanged replay executes nothing, and cancellation after successful sparse
followed by resumption preserves cold artifact bytes. Actual imported-pixel
corruption is rejected at ingest before reconstruction. No application repair
is justified merely to create a red baseline.

The output has **two registered cameras, 599 mesh points, 232 triangles and
zero independently editable native assets**, below the approved minimum of
three recognizable editable objects. Its quality status is explicitly
**nonpassing** despite native build completion. Only an arbitrary-scale
environment asset exists. Independent recognition, Blender editability and
the three-real-video exit remain unrun. These diagnostic observations are
not new expected thresholds or a tuning target. A separate implementation
context must verify and document this result before PR completion.

## Reproduction and gates

Run the capture from repository root with explicit local paths:

```sh
bun scripts/capture-real-room-native-acceptance.ts --original /path/to/long-office-rgb.avi --prepared-root /path/to/prepared --output-root /fresh/browser-input --automation-module /absolute/path/to/playwright/index.mjs
```

Record its printed `witness_hash` independently, then run the explicit native
acceptance tier (the ordinary suite deliberately ignores external real assets):

```sh
VIDEO_TO_3D_REAL_ROOM_INPUT=/fresh/browser-input VIDEO_TO_3D_REAL_ROOM_WITNESS_HASH=sha256:CAPTURE_HASH VIDEO_TO_3D_REAL_ROOM_SOURCE=/path/to/long-office-rgb.avi VIDEO_TO_3D_REAL_ROOM_REPORT=/path/to/result.json cargo test --release --locked -p video-to-3d-cli --test real_room_native -- --ignored
```

The command above requires only repository toolchains and caller-provided input
paths. A private disposable-build wrapper may optionally surround `cargo test`;
it is not a repository dependency or a required reproduction entrypoint.

The carrier/receipt pins were independently checked through the existing #188
`verify_prepared_case` using the exact original and unchanged decoder recipe.
The carrier has SHA-256
`3fd48dee1f59cf3b472b2f5d6a6a25bc846441a964cd682be975618d4e2ef50c`
and 212,474,024 bytes; the 7,136-byte receipt has SHA-256
`f680f6d426e03e879de79b222d3cc0dd5af79b8dcd10e005f04b448ca8de21c0`.
A different preparation/decoder version needs independent verification and an
explicit pin update; rewriting a colocated receipt cannot authorize new pixels.
Original approved source and sampling expectations remain unchanged.

Run the pre-browser substitution regressions with the same local files:

```sh
VIDEO_TO_3D_REAL_ROOM_SOURCE=/path/to/long-office-rgb.avi VIDEO_TO_3D_REAL_ROOM_PREPARED=/path/to/prepared bun test scripts/real-room-capture-input.test.ts
```

They check the valid canonical preparation and fresh negative copies with a
changed carrier, including a self-consistently rewritten colocated receipt.
Rejected copies must fail before a browser module is imported or any output is
created. Missing external inputs skip this separate tier and prove no pass.

The native contract also injects clearly labeled sparse/dense/assembly failures
only as acceptance-classification regressions. Failed sparse cannot trigger cancellation
or qualify for resume proof. Later failure reports actual retained camera
evidence from assembly, mesh or sparse (with the source stage), and JSON null
for unavailable editable-asset/camera counts, never an invented zero. Cold,
replay and resumed exit codes, completion flags and per-operation outcome
classes must agree; downstream nonpassing outcomes never become completion.
Actual reconstructed quality still remains subject to independent recognition
and Blender acceptance.

Independent authoring and baseline execution precede the worker handoff. A green
baseline is allowed: this issue repairs missing executable real-input evidence
and must not manufacture an application failure. A distinct worker must verify
the actual result, diagnose genuine failures without relaxing expectations,
record limitations and run applicable repository checks before review/merge.
Full root `bun run check` remains required. Its Linux desktop checks need the
D-Bus, GTK 3 and WebKitGTK 4.1 development metadata and matching runtime libraries;
missing metadata or dangling development-library links are environment failures.
The review-closure run passes after repairing those host prerequisites, without
application changes. The canonical Python tier runs 60 tests with 10 optional
external-input skips; this does not claim a fresh all-corpus run. The three
explicit real-input native contracts and three capture-identity checks run
separately without skips. Fresh Chromium capture retains the original 18-frame
witness, and actual native quality remains nonpassing as recorded above.

The existing WASM build assumes the repository's `target` directory; #192 tracks
that limitation for callers using a different `CARGO_TARGET_DIR`. This run
pointed the repository target at fresh disposable Cargo output, then restored
the original target and public assets. Browser contracts therefore consumed
this run's generated WASM, rather than previous build outputs.

Shared convention sourceRevision:
`5ad0db36974c8b0e5f9812c90ee34a4f227852c3`.
