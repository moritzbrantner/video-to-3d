# Approved handheld fixture preparation acceptance (#176)

Independent author/context: video_172_acceptance. Protected baseline2f47f3c;
shared conventions sourceRevision46d8793bb3034326561f876dcc67dbaa5aa1e432.
Authority: Decision Atlas3.1.2,3.2–3.6,4.3,6.1–6.2; owner approval in
[issue74](https://github.com/moritzbrantner/video-to-3d/issues/74#issuecomment-6113239395).
This contract prepares real input fixtures; it does not claim implementation,
recognizable reconstructed output, Blender acceptance or the #77 three-video gate.

## Immutable source-owned expectations

`tools/tests/fixtures/approved-handheld-expectations.json` is independently authored
acceptance, not a production corpus planner. It records the two approved original
SHA256/byte lengths, official original/description URLs, CC BY3.0 title/creator/
license attribution, authorized trim and per-item visibility/uncertainty. Production
uses the existing `benchmarks/real-video-corpus.json` and existing tools. Routine
field/API translations require independent re-verification; semantic expectations
must not change to fit reconstructed output. Existing unapproved provider/manual
canary redistribution stays rejected; these approved licensed originals are the
explicitly attributed exception, not a global relaxation.

H1: Andrew Mason, hotel room tour; originalSHA256
`9d08559f3c18725848d0b53849dbc528c6ec803d4bc617cdafb2c16bdaf3304e`,
11770437bytes,128.821s,VP9,640x480,30fps. Authorized indoor source interval[60,86),
26s, selected before reconstruction. H3: Obakeneko, Noi Bai airport waiting room;
SHA256`7885f38454d5ecd9bfc7f1e0a74d5522d8e2d1deea1d65338fbcd07871267fe4`,
14334086bytes,25.3s,Theora,640x480,30fps; whole clip. Both are CC BY3.0 Unported,
not Commons structured-data CC0. Retain original bytes/hash and attribution;
identify trim, scale and contact-sheet changes. No controlled-room input is
approved by this contract. Depth/pose/metric truth is not provided.

## Preparation boundary

Reuse `buildVideoSamplingPlan`:1.25fps,max18,360x270 analysis,5%..95% inclusive
samples. H1 segment-relative1.3..24.7s maps source61.3..84.7s; H3
1.265..24.035s. Requested times, actual selected decoded ordinals/PTS, decoder/
version/scaling/colorspace boundary and per-frame hashes are retained. Request
arithmetic is checked by `scripts/test-handheld-sampling-acceptance.ts` against the
existing public helper. This proves the plan, not pixel equivalence.

The independent source review used nearest original decoded best-effort PTS,
earlier tie wins, ffmpeg8.0.1 bilinear360x270P6PPM; requested/selected time errors
were≤0.016571s. Browser `video.ts` uses HTMLVideoElement seek/presented-time and
canvas scaling. FFmpeg and browser selection, colorspace and scaled pixels are
not automatically equivalent. The production preparation must declare its actual
boundary; it may reuse existing runner's offline decoder. Do not label offline
frames as browser-verified. Actual browser source seek/canvas verification is
preferred and remains an explicit gap until run, including H1 trim duration/PTS.
A changed decoder/frame selection that materially changes inventory coverage must
be independently inspected, not silently substituted.

H1 requires actual deterministic derived clip bytes/hash plus original/trim
provenance, so a browser can consume the indoor input rather than unknowingly
sampling the whole tour. If an implementation instead uses a virtual trim, its
explicit boundary must enforce the interval, produce the matching source frames,
and demonstrate truthful browser segment handling; coordinate independently before
translating the encoded-clip expectation. Never fabricate a derived hash. H3 whole
clip may use the unchanged original as its derived input. Preparation-only mode
reuses `tools/run_real_video_case.ts` and emits no reconstruction output; it must
not invoke Rust reconstruction, learned providers, geometry generation or Blender.

Extend existing `sampling.json` receipt with verified original/derived file path,
SHA256/length/source hash/trim transform, decoder boundary/version and18 frame
path/hash/ordinal/PTS/requested timestamp entries. Paths resolve safely within
asset/prepared roots; escaping paths, symlinks outside roots and mismatching bytes
fail closed. Case ID/plan/inventory must agree with immutable source expectations.
The proposed existing checker seam `verify_prepared_case(case,asset_root,
prepared_root)->errors` is a routine signature, not a new framework; it may be
translated independently while retaining successful real-source verification and
original/derived/frame tamper rejection. Normal validation must not trust receipts
without checking their actual files. Original media remains local product input.

## Observable inventory and limits

Zero-based fixed sample indices were independently inspected before reconstruction:
H1 bed3–7; desk1–2/13; deskchair1 (coat-occluded); armchair2–3/7–8/13–15;
sofa16–17 (partial16); coffee-table17 (single sampled view). Floor/interior wall/
balcony opening are architecture. Deskchair, sofa and table have limited coverage;
retain uncertainty, not invented hidden backsides or observed labels for completion.
Additional foreground chair is not added to inflate counts. People, moving
occluders, mirror reflections and exterior sea/buildings are not static furnishings.

H3 waiting-seat bank0–2/16–17; queue-barrier assembly10–12/16; freestanding shop
merchandise display10–11. Hall floor/shop front/column/gate-window opening are
architecture. Travelers/luggage/strollers/reflection copies are excluded static
inventory. These three semantic assemblies derive from footage, not the controlled
room's numeric minimum. Each named expected furnishing must eventually be recognized
and independently editable, not pass through a count of arbitrary mesh chunks.

Existing Blender expectation schema can consume independently owned architecture/
object IDs, materials/mesh/edit independence; physical bounds are omitted without
true scale. Source visibility references do not supply fixture-authored geometry.
A generic named node is not proof of recognition. Actual pipeline-produced meshes,
source-view recognition, observed/completed provenance and Blender independent edit
remain separate unrun #77 gates. H1 limited views must remain visible in that gate.

## Commands and fail-closed handoff

- `python3 -m unittest discover -s tools/tests -p 'test_approved_handheld_acceptance.py'`
- `bun scripts/test-handheld-sampling-acceptance.ts`
- Reused `bun scripts/test-video-sampling-contract.ts` and existing corpus tests.
- Real-source tier: set `VIDEO_HANDHELD_ASSET_ROOT` to the licensed original root,
  `VIDEO_HANDHELD_PREPARED_ROOT` to actual production preparation outputs, then run
  the same Python tests. Absence is reported skipped, never counted as a pass.

Initial baseline lacks both corpus entries, approved attribution/hash validation,
derived preparation integrity verification and protected inventory validation.
Independent tests are expected red until a distinct implementation context adds
preparation. Initial source review exists locally, but production preparation and
browser decode acceptance remain unrun. No decoder, corpus entry or production
geometry is added by this acceptance commit. Full applicable verification is needed
when impact mapping cannot select affected checks reliably.

## Independent source-correspondence strengthening (before #176 production)

Supplemental authoring baseline3318546 (default3861cd1 integrated); original
expectationsbb3a8f0 remain unchanged. `approved-handheld-source-review.json` retains
all36 previously reviewed source ordinals/PTS/PPM hashes under FFmpeg8.0.1. It is
acceptance evidence, not a production corpus or fixture generator. The acceptance
oracle independently runs ffprobe on the hash-verified original, enumerates decoded
video frames in their original order, restricts candidates to the authorized
half-open trim interval, and minimizes exact decimal distance to each fixed source
request. Equal distances select earlier decoded PTS, then earlier ordinal. The
small tie/trim-end case protects that reference rule without asserting FFmpeg and
browser seek equivalence.

The source-only tier requires `VIDEO_HANDHELD_ASSET_ROOT` but no prepared outputs.
It decodes those selected original ordinals in one FFmpeg pass, using bilinear
360x270 scaling and passthrough timing, and reproduces every source-review PPM
hash. Prepared receipts must match the independently recomputed original ordinal
and best-effort PTS; their P6 dimensions and RGB pixel payload must equal the
independent original decode, not merely their own receipt hash. P6 header formatting
alone is not pixel identity. This is the explicit offline decoder boundary; it
still supplies no HTMLVideoElement/canvas acceptance or reconstructed geometry.

The actual-prepared negative tier substitutes another real sampled frame and
updates its receipt hash, separately substitutes another real in-range ordinal/PTS,
and combines both substitutions. Each must fail the production checker, despite
self-consistent bytes/hash and valid in-range timestamps. The frame requests,
source approval, trim,18-sample plan and visibility inventory stay fixed. Tests
copy existing actual preparation outputs for these mutations; no preparation or
geometry stand-in is authored here.

On unchanged3318546, both source-reference tests pass against the actual approved
originals; all36 frame hashes match the pre-reconstruction review. The preexisting
metadata tier remains1 passing method/4 failing methods (14 failed subcases), and
the actual-prepared class is skipped because production outputs do not exist. The
new wrong-frame receipt negatives are therefore authored but unrun, not claimed
as reproduced production failures. The fixed browser sampling-plan test passes.

## Actual browser-carrier witness before compatibility repair

Supplemental protected witness baseline1dc364d, authored before the carrier repair.
`scripts/test-handheld-browser-carrier-acceptance.ts` bundles and executes the
existing `apps/web/src/video.ts` in actual Chromium. It uses the receipt's real
hash-checked derived bytes as a browser File, then calls `sampleVideoWithInfo`
through HTMLVideoElement seek and canvas readback. It checks both approved interval
durations, usable640x480 source dimensions,18 actual nonempty360x270 frames and
finite in-interval presented times. Receipt source hash and approved trim remain
fixed. It does not decode media through a test substitute or equate browser pixels
to FFmpeg PPMs. Browser version and per-case pass/failure observations are emitted.

Baseline actual run: H1 `derived.webm` passes; copied H3 `derived.ogv` fails with
`the selected video does not expose usable dimensions`. The offline real-source /
prepared tier remains independent of this actual browser compatibility failure.
H3's unchanged original as carrier was optional, not required: a deterministic
whole-content lossless VP9 derivative is an allowed carrier while preserving the
original approved OGV bytes/hash/attribution, whole-clip interval, original source
ordinal/PTS/PPM expectations and truthful transform/decoder receipt. This permission
does not approve a different source, content interval or browser pixel equivalence.

Mandatory actual witness after preparation:

```sh
bun scripts/test-handheld-browser-carrier-acceptance.ts \
  --prepared-root "$VIDEO_HANDHELD_PREPARED_ROOT" \
  --automation-module "$VIDEO_BROWSER_AUTOMATION_MODULE" \
  --report /path/to/browser-carrier-evidence.json
```

The automation module is the existing environment's installed Playwright module
with real Chromium available, not an application media dependency. The Python
actual-prepared tier invokes the same witness when that environment variable is
provided. Missing automation is explicitly skipped and is not browser acceptance.
This headless carrier proof does not claim physical desktop/phone inspection,
owner visual approval, source inventory recognition or reconstructed #77 output.

## Fixed approval boundary negatives before authorization repair

Independent source audit at frozen143a1eb found that validated original SHA/approval
URL did not enforce the authorized interval or fixed plan at the public custom
manifest boundary. Supplementary tests preserve canonical positives and challenge:
H1[59,85), keeping exactly the same26s/browser plan; H3[0,24.3), using the actual
public browser helper's corresponding plan; H1's first requested time shifted by
0.001s, still ordered/in-range with18 samples; and altered declared original duration.
Approval rejection must identify trim/interval, plan or approved-source metadata,
rather than accidentally rejecting an unrelated schema or planner mismatch.

Frozen baseline metadata validation accepts all four mutations (genuine RED).
Actual existing Bun prepare-only H1 shifted-interval command returns zero (RED),
and actual direct Python preparation with the alternate H1 time also returns zero
(RED). These execute real licensed sources and the existing preparation boundary;
no geometry or substitute decoder is supplied. The H3 negative preparation was
started before repair; its completion is reported separately, not assumed here.
Early-rejection/no-output assertions remain required after repair; those later
assertions are unexecuted when the initial expected rejection already fails.

The approved IDs bind the precise authorized trim, original duration and existing
18-sample public-helper plan; other generic corpus cases do not inherit this policy.
A changed manifest may not retain the fixed owner approval while changing those
inputs. Both public preparation entry points must reject before creating files or
altering an existing sentinel. This adds no new source approval, inventory, sampling
policy or browser/offline pixel equivalence claim.


## Supplemental canonical attribution boundary (review 4244038768)

Authored independently against frozen `1c1dbdd8f881e965ad3d21d32380a40d2b9618b1`, before repair. An approved case ID must not permit custom manifests to redefine its creator, source title, official source/download URLs, original byte length/filename, license/mode, source hash, redistribution value or owner-approval URL. Canonical metadata is the existing approved repository corpus, not a second hardcoded production catalog. Existing source approvals, trim, eighteen-sample plans, inventories and decoding boundaries remain unchanged. This scoped policy concerns the two approved fixed IDs, not a new global ban on custom generic corpus assets.

The supplemental metadata method tests eleven immutable asset leaves for both cases while preserving nonempty valid strings/HTTPS where applicable. At the frozen head, fourteen mutations are genuinely admitted by `validate_manifest`: creator, title, source URL, download URL, byte length, expected filename and redistribution value for each case. License URL, mode, source hash and owner approval mutations already reject and remain complementary GREEN cases. The exact canonical pair remains GREEN.

Actual public Bun prepare-only and direct Python prepare calls use the verified approved original bytes plus a controlled pre-existing case folder with a retained sentinel and no sampling receipt. At baseline, metadata admits these requests but the calls reject later for the unrelated missing receipt. This is **not successful unauthorized preparation and is not another independent defect reproduction**; the canonical prewrite guard is not witnessed at baseline. The supplemental post-repair boundary expectations require a useful offending attribution field/canonical rejection before output use, no new files and unchanged sentinel bytes/mtime. There is no substitute decoder, forged source, copied reconstruction output or injected provider result.

Actual verification uses the genuine completed prepared root and must reject a different source URL before treating its otherwise valid receipts/bytes as approved. That actual prepared-root verifier test is UNRUN/SKIP in the cheap baseline run; it must run with the real prepared root after repair. Original eighteen methods are unchanged. No offline/browser pixel-equivalence or product/Blender/physical-device result is added by this provenance supplement.


## Supplemental known-ID preparation omission boundary

Authored against frozen `b24b121c88ab3320112d028d6a898400e3c581af` before the adjacent repair. Removing preparation cannot turn either approved fixed ID into an unrelated generic source: the existing canonical asset and preparation authority still apply. Exact approved positives and all four existing nonapproved legacy corpus cases remain accepted. Metadata validator and its actual public Python CLI must reject with the known case ID and an approved/canonical preparation reason. Direct prepare/verify must reject before source or receipt access and writes.

The initial actual-source baseline records two genuine metadata admissions and two successful public metadata CLI admissions. Actual Bun H3 prepare-only additionally deletes a throwaway retained sentinel before failing later; this is a genuine prewrite violation, not a claim that unauthorized reconstruction completed. Python prepare-only already rejects generically without touching the sentinel; its missing case-specific diagnostic and the raw direct-helper KeyError are complementary diagnostic gaps, not successful unauthorized preparation. No actual prepared root is mutated. All previous twenty-one methods remain unchanged; this supplement does not alter approvals, source bytes, inventory, eighteen-sample plan, or legacy sampling.
