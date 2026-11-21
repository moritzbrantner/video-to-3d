# Native observed surfaces

The local studio can run CUDA PatchMatch on the selected sparse COLMAP model. It bounds each image to 360 pixels on its longer side and selects at most 32 stereo views across the camera path. It groups nearby camera views by shared sparse landmarks. Only views that see the shared landmarks at a median triangulation angle of at least 3° become PatchMatch sources, matching PatchMatch's own filter. Adjacent frames of a handheld walk share the most landmarks but have almost no baseline: on H3 they gave a median of 1.24°, and geometric consistency kept 0.3% of pixels. Geometric PatchMatch supplies depth candidates. Depth is sampled twice as finely in the supporting views as in the reference mesh. Each reference sample needs at least one source camera to verify it. Samples verified by different sources have the same position, taken from the reference depth. Each reference view's verified samples are therefore meshed once, as one connected grid, and the evidence region lists every source that verified a retained vertex. Rust checks the camera registration against the original keyframes, requires agreement in color and depth from another camera with measurable parallax, and applies its existing mesh and material exporters to accepted grid points. The GLB records the geometric provider, camera poses and arbitrary monocular scale.

Use the workspace produced by the optional third argument of the sparse registration sidecar. The target GLB path must be new. An optional fifth argument to the dense wrapper supplies a semantic inventory. Rust verifies its frame receipts and reuses the same multiview mask checks as the browser to partition already observed triangles into selectable objects. The report gives the number of objects and attributed triangles.

```sh
tools/colmap-sparse/run.sh keyframes.json registered.json sparse-work
tools/colmap-dense/run.sh keyframes.json registered.json sparse-work scene.glb
```

The pinned CUDA wheel requires a compatible NVIDIA device and driver. The scripts use uv to acquire the local runtime and Cargo for the Rust export. The studio retains its existing observed-surface route when geometric stereo is unavailable.

## Inspect stereo and score an export

To retain the intermediate depth maps and fused PLY for inspection, run the stereo command directly. The output directory must be empty.

```sh
uv run --script tools/colmap-dense/reconstruct.py sparse-work/images sparse-work/sparse/0 stereo-output --max-image-size 360
uv run --script tools/colmap-dense/score_tum.py stereo-output sparse-work/sparse/0 extracted-tum-rgbd keyframes.json
uv run --script tools/colmap-dense/score_surface.py scene.glb extracted-tum-rgbd keyframes.json
```

Use the model named by `sparse-work/selected-model.txt`. The scoring commands use the official Freiburg 3 long-office RGB-D dataset. Each movie frame corresponds to one ordered entry in `rgb.txt` at 30 fps. The scorers fit global scale from registered camera centers against the motion-capture trajectory. The depth maps and trajectory are used exclusively during evaluation. The GLB scorer raycasts the observed triangles at published RGB camera poses and compares axial depth at registered Kinect pixels. It reports how many valid rays hit the mesh, the median and 90th-percentile depth error among hits, and the fraction of all valid rays within 10 cm. It also evaluates camera poses between the input frames. Use `--evaluation-frames` to compare exports on a common set of movie times. [TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg3_long_office_household)

The following comparisons share 85 source times and 339,023 valid Kinect rays drawn from the 87-frame evaluation profile. Both 108-frame rows use the same final calibrated cameras.

| RGB reconstruction | Mesh hit rate | Valid rays within 10 cm | Median error on hits | 90th-percentile error on hits |
| --- | ---: | ---: | ---: | ---: |
| Rust camera and dense stages, 87 input frames | 3.35% | 0.64% | 18.84 cm | 95.41 cm |
| COLMAP cameras and Rust dense stage, 87 input frames | 11.67% | 6.00% | 9.30 cm | 110.03 cm |
| 108 input frames, previous 24-view stereo and single-source grid | 51.19% | 38.52% | 2.36 cm | 52.49 cm |
| 108 input frames, 32-view stereo and supplemental camera support | 70.44% | 54.25% | 2.63 cm | 32.55 cm |
| Same stereo, verified samples meshed once per reference view | 71.71% | 55.14% | 2.69 cm | 32.00 cm |

The 32-view reconstruction exported 108,445 textured triangles. It covered 54.41% of valid Kinect rays within 10 cm at 80 intervening viewpoints withheld from both frame selections. In a run sharing the RTX 3060 Ti with other GPU workloads, stereo took 404 seconds. The subsequent Rust export with the saved inventory took 10.83 seconds.

The inventory assigned 2,606 observed triangles to five groups labeled as yellow dice. One independently selectable group contained 897 triangles. Against reference silhouettes in two original frames, that group's intersection-over-union was 0.894 and 0.716. The median absolute Kinect depth errors on overlapping pixels were 1.47 and 1.30 cm. The four other labeled groups did not overlap the die in those two views.

Meshing each reference view once cut disconnected surface islands from 3,586 to 466 on this run, and from 780 to 153 on H1. It exported 115,647 triangles. The same inventory assigned 2,735 triangles to five yellow-die groups. The selectable die grew to 930 triangles, with intersection-over-union of 0.894 and 0.717 and unchanged depth errors. It still moved and edited independently in headless Blender. The export now reports its stage funnel; see [the fragmentation diagnosis](../../docs/geometry-fragmentation-diagnosis-2026-11-21.md).
