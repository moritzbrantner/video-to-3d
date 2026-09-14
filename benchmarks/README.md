# Reconstruction benchmark strategy

The benchmark stack separates fast pull-request gates from larger real-world canaries.

## Pull-request golden matrix

Every reconstruction pull request is evaluated on deterministic rendered scenes with known camera trajectories. The same images are passed to `video-to-3d` and COLMAP.

Current cases:

- `lateral`: the original sideways-translation baseline; strong parallax and stable overlap.
- `revisit`: the camera moves across the scene and then returns, exercising non-adjacent evidence, recovery, and loop/drift behavior.
- `forward`: forward-dominant motion with a smaller lateral component, exercising weaker triangulation geometry without becoming a deliberately degenerate case.

The gate checks registration coverage, sparse-map size, reprojection error, and camera-center RMSE against the known trajectory after similarity alignment. Runtime-profiler captures Rust and COLMAP independently for every case. Moonlight continues to compare repository revisions.

Synthetic trajectory truth is authoritative only for these generated scenes. COLMAP remains a reference implementation, not the owner of product geometry.

## Real-image benchmark candidates

### COLMAP South Building

Use as the first real-world sparse-SfM canary. COLMAP documents South Building as a 128-image dataset captured with one camera, and publishes the archive as a release asset.

- Source: https://colmap.github.io/datasets.html
- Pinned asset: https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip
- Published archive size: 419,421,847 bytes

This dataset is useful for end-to-end robustness and performance comparison, but it does not provide independent geometric ground truth. It should therefore remain a canary/reference comparison rather than replace the known-trajectory gate.

### Strecha Fountain-P11 and Herz-Jesu-P8

These are strong next datasets for independent sparse-pose validation because they are classic calibrated multi-view benchmarks with camera parameters and LIDAR-backed scene ground truth. Fountain-P11 has 11 high-resolution images and Herz-Jesu-P8 has 8.

They are a better fit than South Building for absolute pose/geometry scoring once a stable, checksum-pinned source is incorporated into the pipeline.

### ETH3D

ETH3D provides laser-scanner ground truth across varied indoor/outdoor multi-view scenes and is a strong later-stage benchmark. Its high-resolution datasets are substantially larger and its primary evaluation is dense multi-view accuracy/completeness, so it is better suited once `video-to-3d` advances beyond sparse SfM.

- Source: https://www.eth3d.net/datasets

## Real-video corpus

`real-video-corpus.json` is the machine-readable authority for recorded-footage acceptance. Large source videos and evaluation datasets are never committed to this repository. The manifest records provenance, acquisition identifiers, expected local paths, evaluation role, scoring policy, and whether a case may run unattended.

The first quantitative cases are Tanks and Temples `Ignatius` and `Church`. They use the original high-resolution video plus the provider evaluation bundle containing laser-scanner ground truth, the COLMAP camera reference, scene alignment, frame mapping, and crop metadata. The repository acquisition tool reads provider identifiers from the manifest, verifies the official video checksum, and extracts only the files required for the selected scene.

The first field canaries are the Statue of Liberty and Roman Colosseum orbit videos. They are deliberately manual: the manifest records the original video and a published reconstruction made from that footage, but CI does not download YouTube content and does not redistribute any source video.

### Sampling contract

The benchmark runner imports the same pure sampling-plan function used by the browser. The shared contract owns timestamp selection, frame count, and the 360-pixel analysis resolution. Browser DOM APIs still own normal in-product decoding; the benchmark uses FFmpeg only as an external test decoder so real footage can be exercised headlessly without creating a second reconstruction implementation.

Default field settings remain 1.25 sampled frames per second with an 18-frame cap, distributed over the 5%–95% interior of the clip. The runner also records the source frame rate so each accepted sampled camera can be associated with the nearest provider reference frame. The benchmark manifest is validated in ordinary CI so provenance, scoring, or scheduling policy cannot drift silently.

### Quantitative ground-truth evidence

For Tanks and Temples cases, the scorer first matches accepted sampled cameras to the provider frame mapping, transforms the published COLMAP camera centers into the laser-scanner coordinate system, and estimates one similarity transform from `video-to-3d` camera centers to those references. It reports camera-center RMSE/median error, then applies that same transform to accepted sparse and dense geometry and reports bounded nearest-neighbor precision, recall, F-score, median accuracy, and median completeness at the provider scene distance threshold.

This is intentionally **not** labeled as the official Tanks and Temples leaderboard score. The official evaluation performs its own crop-aware registration and ICP refinement. Our scheduled evidence uses camera-reference alignment followed by deterministic, memory-bounded, uncropped nearest-neighbor scoring so it remains reproducible and useful as a repository-owned regression baseline without letting the external evaluator become geometry authority.

### Running a quantitative case

Review the current Tanks and Temples dataset terms before downloading data. Then:

```bash
python3 -m venv .benchmark-venv
.benchmark-venv/bin/python -m pip install gdown==5.2.0 numpy==2.5.3 scipy==1.18.1 plyfile==1.1.5
.benchmark-venv/bin/python tools/fetch_tanks_and_temples.py \
  --case tanks-temples-ignatius \
  --with-evaluation
bun tools/run_real_video_case.ts --case tanks-temples-ignatius
.benchmark-venv/bin/python tools/score_tanks_and_temples.py --case tanks-temples-ignatius
```

Outputs are written under `target/real-video/<case-id>/`:

- `sampling.json`: exact source metadata, source frame rate, and selected timestamps.
- `metrics.json`: reconstruction counts and final accepted reprojection evidence from the Rust core.
- `sparse.ply` and `dense.ply`: accepted reconstruction geometry.
- `cameras.csv`: accepted camera centers when calibrated geometry exists.
- `ground-truth-score.json`: camera-reference alignment and laser-ground-truth geometry evidence, explicitly marked as non-official Tanks and Temples scoring.

The scheduled `Real Video Evidence` workflow is opt-in because dataset terms must be reviewed outside the code. Set repository variable `TANKS_AND_TEMPLES_TERMS_ACCEPTED=true` only after accepting the current terms. Scheduled case IDs are derived from `automation.scheduled` in the validated manifest rather than duplicated in workflow code. Manual workflow runs require the same acknowledgement per run and accept a quantitative manifest case ID. Raw videos, extracted reference data, and sampled frames are never uploaded as workflow artifacts.

### Running a field canary

Acquire the source video yourself under terms that allow your use, place it at the manifest's `expected_file` under `.benchmark-data/`, then run the same reconstruction runner. For example:

```bash
bun tools/run_real_video_case.ts --case youtube-statue-of-liberty
```

Field canaries are observational evidence, not merge gates. Their published third-party reconstruction demonstrates that the source footage is reconstructable, but it is not geometry authority for `video-to-3d`.

## Gate policy

- Pull requests must keep the deterministic golden matrix green.
- The real-video corpus manifest and shared sampling contract are ordinary pull-request gates; downloading large external videos or evaluation data is not.
- Runtime ratios remain evidence until repeated measurements establish stable envelopes per scene.
- A reference implementation may veto a clear quality regression, but representation-specific counts do not need to match exactly.
- Quantitative real-video cases remain evidence-only until repeated runs establish repository-owned envelopes for registration, camera error, geometry accuracy/completeness, and dense coverage.
- Real-image and real-video datasets should be pinned to an immutable or checksum-verified source before becoming required gates.
- Larger public datasets belong in slower scheduled/manual lanes unless their cost is small enough to justify every pull request.
