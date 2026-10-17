# Independent controlled-room preparation acceptance (#187)

Authority is Atlas 3.1/3.1.1/3.8 and issue #74's approved original TUM RGB movie.
This contract is independently authored at baseline
`ff9a7a37b5513b609345f154cf3a9e607015bf96` before preparation implementation.
It closes neither real-input native execution #179 nor the three-video product
exit #77. No depth, trajectory or synthetic scene is a production input here.

## Source and fixed observations

The original `long-office-rgb.avi` has SHA-256
`3200d7a76f5b3307e33863d4b38dbad414527ce14084f4c1f6303a0f69eb794c`,
26,026,292 bytes, 640x480 at 30 fps and duration 86.166667 seconds. Preserve the
whole movie. The fixture's interval 0–86.166667 covers the original publication;
it is not a favorable trim selected from reconstruction output.

The [official dataset license](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
and [sequence/source](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg3_long_office_household)
were rechecked on 2026-10-14. They support the recorded CC BY 4.0 attribution,
source movie and requested Sturm et al., IROS 2012 citation. The sequence's
separate trajectory duration/statistics do not replace RGB movie timing or
establish recovered metric geometry.

The unchanged public `buildVideoSamplingPlan` selects 18 requests across the
whole loop at 360x270. Root acceptance inspected an independently decoded
18-frame contact sheet before any reconstruction tuning. The fixture records
conservative visible sample sets for the blue globe/support, yellow die, purple
office chair and yellow-shirt plush bear. These are four distinct source object
candidates; the controlled product gate still requires at least three
recognizable, independently editable objects. Source observations do not prove
that a pipeline produced any of them. Hidden surfaces, contact geometry and
exact dimensions remain unknown. The bear is separate from its support chair;
desk divider panels must not become permanent room walls by relabeling.

`approved-controlled-room-expectations.json` pins source, license, attribution,
full interval, sampling and nonmetric inventory. The separate source-review
fixture pins nearest-original PTS/ordinal and PPM hashes. Its oracle reuses the
existing independently authored linear nearest-PTS reference and original RGB
decoder, not producer selection or reconstructed output. Earlier-frame tie
semantics and pixel parsing remain covered by the existing handheld oracle.

## Executable acceptance and handoff

With the licensed original in an asset root:

```sh
VIDEO_CONTROLLED_ASSET_ROOT=/path/to/originals python3 -m unittest discover \
  -s tools/tests -p test_approved_controlled_room_acceptance.py
```

The six cases require approved metadata/corpus pinning, unchanged full-loop
expectations, exact original-derived samples, successful actual public
`--prepare-only`, no geometry production or source changes, repeat preparation
without output rewrites, rejection of self-consistent wrong-frame/PTS receipts,
and unauthorized-interval rejection before writes. Missing originals skip only
explicit real-source cases and cannot establish completion.

Baseline: **1 PASS / 5 genuine failures**, no skips. The pass independently
verifies all 18 fixed original-source frame witnesses. Failures expose missing
controlled corpus entry and the old H1/H3-only approval boundary. Mutation and
unauthorized-interval checks stop at the valid-source precondition; late
preparation/portability/idempotence/tamper assertions remain unrun. Log:
`/tmp/video187-independent-baseline.log`.

Before merge, independently exercise the actual derived carrier with the
unchanged browser `sampleVideoWithInfo` through the existing witness:

```sh
bun scripts/test-handheld-browser-carrier-acceptance.ts \
  --prepared-root /path/to/prepared \
  --expectations tools/tests/fixtures/approved-controlled-room-expectations.json \
  --automation-module /absolute/path/to/playwright/index.mjs \
  --report /path/to/browser-report.json
```

The only witness change accepts an explicit expectation path; omitting it keeps
the original H1/H3 input and behavior. Browser decode evidence must remain
separate from offline reference pixels; no cross-decoder pixel-equivalence
claim follows. Actual controlled carrier decoding is **UNRUN** at baseline.

After the distinct worker implements the approval adapter/corpus data, all six
cases with originals and the browser witness are mandatory, alongside existing
H1/H3 acceptance and the repository completion gate. Impact relationships are
partial, so focused tests cannot replace `bun run check`. Shared conventions
sourceRevision: `5ad0db36974c8b0e5f9812c90ee34a4f227852c3`.

The independent author stops after this contract. Production remains in the
existing generic checker/runner; do not add another preparation engine or tune
reconstruction to these observations. Actual controlled scene-to-Blender
acceptance and H1/H3 native outcomes remain separate and unverified here.
