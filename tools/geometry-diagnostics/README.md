# Geometry diagnostics

Controlled pose/depth ablation of the observed-surface route on the pinned TUM RGB movie
(`freiburg3_long_office_household`). Findings: [`docs/geometry-fragmentation-diagnosis-2026-11-21.md`](../../docs/geometry-fragmentation-diagnosis-2026-11-21.md).

TUM motion capture and Kinect depth are **evaluation oracles only**. They create the oracle
combinations and score held-out views. They never reach the production route, and the scripts
here never write into a production project.

| Combination | Poses | Depth |
| --- | --- | --- |
| `reference_pose__measured_depth` | motion capture | Kinect |
| `estimated_pose__measured_depth` | accepted COLMAP registration | Kinect |
| `reference_pose__estimated_depth` | motion capture | COLMAP geometric PatchMatch |
| `estimated_pose__estimated_depth` | accepted COLMAP registration | COLMAP geometric PatchMatch (production) |
| `estimated_pose__learned_depth` | accepted COLMAP registration | MapAnything only |
| `estimated_pose__filled_depth` | accepted COLMAP registration | PatchMatch, MapAnything only where PatchMatch is empty |

Rules the harness enforces:

- Estimated reconstructions are aligned to motion capture **once**, by one similarity transform fitted to camera centres. Depth maps get that one global scale. There is no per-frame or per-view scale fitting. MapAnything gets one global scale from COLMAP's own sparse landmarks, without an oracle. Its per-group scale ratios are reported, not applied.
- Kinect zeros stay invalid. Coverage is always a fraction of valid Kinect rays.
- Reference-pose combinations re-render frames and Kinect depth into an ideal centred camera. The Rust stage models a centred principal point, so this keeps calibration error out of the oracle. Oracle sparse tracks are synthesised from Kinect depth only so that the unchanged Rust registration checks accept motion-capture cameras.
- Every combination is fused by the unchanged Rust `colmap-surface` stage and by an Open3D TSDF reference over the same 32 reference views. COLMAP's own stereo fusion plus a Poisson mesh is also scored.

Each held-out ray with valid Kinect depth falls into exactly one class:

- **correct**: within 10 cm.
- **false positive in front**: reconstructed geometry where the sensor saw free space.
- **behind truth**: the ray passes through a hole in the true surface and hits something farther away.
- **missing**: no surface along the ray.

Surface precision samples 200k points on each mesh. It compares them with a dense Kinect TSDF of the whole clip, plus free-space witnesses from motion-capture Kinect views.

## Run

The inputs are the artifacts of the production dense route (`tools/colmap-sparse/run.sh`, `tools/colmap-dense/reconstruct.py`, `tools/colmap-dense/export_grids.py`) and the extracted TUM archive.

```sh
abl() { cmd=$1; shift; uv run --script tools/geometry-diagnostics/tum_ablation.py "$cmd" --out target/geometry-diagnostics/ablation \
  --dataset target/tum-rgbd --keyframes target/tum-quality/native-frames/keyframes.json \
  --registered target/tum-colmap-camera/final108.json --sparse target/tum-colmap-camera/final108/sparse/0 \
  --stereo target/tum-dense-trials/cluster32 --grids target/tum-dense-trials/cluster32/grids-twoscale.json \
  --evaluation target/tum-colmap-camera/input/manifest.json --cli target/release/video-to-3d "$@"; }
cargo build --release -p video-to-3d-cli
abl prepare && abl reference && abl colmap
# Optional learned depth (8 GB GPU, interpreter from scripts/setup_mapanything.sh):
~/.cache/video-to-3d/mapanything-venv/bin/python tools/geometry-diagnostics/mapanything_depth.py \
  target/tum-quality/native-frames/keyframes.json target/tum-colmap-camera/final108.json \
  target/geometry-diagnostics/ablation/inputs/summary.json <map-anything-apache snapshot> \
  target/geometry-diagnostics/ablation/mapanything && abl learned
abl rust && abl tsdf && abl depth-maps && abl evaluate && abl report
abl stage-visuals
abl visuals --labels tsdf__reference_pose__measured_depth rust__estimated_pose__estimated_depth tsdf__estimated_pose__estimated_depth
```

`report` writes `report.md`, with the held-out ray table, the surface shape and precision table, per-view depth before fusion, reference planes, and the stage attribution. `evaluate` reproduces `tools/colmap-dense/score_surface.py` exactly for the production surface: 70.44% hit rate and 54.25% of valid rays within 10 cm on the 85 shared evaluation views.
