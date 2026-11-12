# Product acceptance: first local hybrid scenes-to-Blender release (v2)

Status: **owner-defined target; not yet implemented or passed**. The [Decision Atlas](../DECISION-ATLAS.md) owns product intent (notably `1.1`–`1.6`, `2.1`–`2.3`, `3.1`–`3.6`, `4.1`–`4.5`); the [2026-10-10 session index](whole-scene-product-decisions.md) retains the historical option selections only. This specification owns detailed acceptance requirements, including independently authored fixture expectations and tolerances.

## User-visible goal
Given **three fixed, licensed videos**—one controlled short room video with camera translation and at least three recognizable objects, plus **two ordinary handheld recordings** with imperfect motion—the same documented `video-to-3d build <project>` workflow must produce recognizable environments and independently editable Blender objects for **all three**. No human or agent selects intermediate operations. A successful controlled clip is a milestone, not the release gate.

## Fixture ownership and independent oracle
- Pin **three distinct, legally usable real clips** with individual SHA-256 hashes, reproducible frame sampling, independent object/architectural inventories, and expected editable hierarchy/relationships. Specify difficulty characteristics for the two ordinary handheld clips in advance; never claim unseen geometry as measured ground truth.
- Author acceptance fixtures/expected results separately from the implementation PR/agent. Version fixture, model/provider, and quality policy.
- Include a deterministic synthetic control scene with known cameras, object identifiers, silhouettes, transforms and coordinate conventions **in addition** to the three real-video canaries. Distinguish measured ground truth from qualitative inspection; independently set per-fixture object counts and tolerances.
- **Initial release is local-only:** no cloud inference, external provider call or data upload (including derived frames). Cloud workflows in the long-term roadmap are deferred. Record/version local model providers and bound memory, runtime and fallbacks. Browser video decoding/frame sampling remains authoritative even in the desktop shell.

## Required artifacts
1. One resumable scene-project manifest with content-addressed operation receipts and provider/model versions.
2. One shared coordinate frame containing accepted cameras and static environment; visually usable preview and an explicit report of unobserved/invented regions.
3. A hierarchy separating **major architectural elements** (individual walls, windows, doors) and meaningful furniture/props as individually editable nodes. For the controlled room clip require at least three **distinct scene object nodes**, independently selectable and transformable in Blender, with at least one independently editable mesh asset. The two handheld clips must satisfy their own independently written meaningful-object inventories; do not assume identical numeric counts. Proxy nodes may aid identification but do not count as finished reconstructed/generated meshes. Three disconnected chunks of one mesh do not pass. A future user-selectable granularity setting must not delay this default.
4. Each object has stable identity across selected keyframes, semantic label (including uncertain), source-frame/mask references, transform, origin/pivot, material, and provenance (observed / learned / revalidated / generative).
5. A Blender-compatible handoff (prefer standard GLB with meshes, materials, transforms and names, plus sidecar for provenance not expressible in GLB). A **headless Blender import test** asserts node count, distinct mesh datablocks or declared reusable instances, non-degenerate bounds, transform validity, material references and scene graph hierarchy. Test independently moving an object while the environment remains unchanged.
6. At least a coarse environment collision representation and a scene quality report with explicit failures, unsupported capabilities and incomplete sections.
7. Both **observation-only** and **AI-completed** scene modes, maintaining evidence classes, confidence and ability to inspect/correct uncertain regions. Editable **mesh is the default Blender representation**; where splat and mesh data coexist, allow switching/inspection without claiming splats are editable meshes.
8. A **shared browser/desktop UI**: GitHub Pages or exported web preview shows inspectable scenes and object hierarchy; desktop/native processing supplies heavyweight local AI and Blender export. Full AI inference need not run in GitHub Pages.

## Acceptance tests (non-negotiable)
- **E2E**: On **each of the three pinned real clips**, one command produces required artifacts; import each in a pinned Blender version, inspect named environment/object hierarchy and render a canonical view. The aggregate release result is green **only if all three pass**. Fail closed on missing stages/artifacts or unsupported required capabilities; do not pass because only the guided control case succeeds.
- **Geometry**: On synthetic ground-truth fixture, assert cameras/transforms and object silhouette/depth-order alignment against separately authored tolerances. Report occlusions and missing evidence instead of inventing exact geometry.
- **Scene identity**: Controlled fixture has at least three identifiable, separately editable objects; each ordinary handheld fixture has independently authored expected movable objects and architectural elements. Track identity remains stable across chosen keyframes; every mesh/object uses the same coordinate convention as cameras.
- **Provenance and completion**: Observation-only view excludes invented geometry; AI-completed view visibly distinguishes generated/revalidated material; uncertain object predictions are measured against available video evidence and can be corrected. Confidence and source references are retained; changing only one object invalidates only dependent artifact receipts.
- **Export**: Blender headless import and transform-edit smoke test; exported GLB and sidecar pass schema/path/hash checks; round-trip retains independent objects.
- **Robustness**: Deterministic resume after interruption; cache reuse; rejected provider evidence never bypasses Rust validation; missing providers yield explicit non-passing status, not a green acceptance result.
- **Visibility**: For **each clip**, CI attaches input reference frames, rendered scene screenshot, Blender hierarchy/object inventory and a machine-readable quality report. The corresponding scene is inspectable through the browser/web scene preview. No success declaration based solely on unit tests, one curated clip or mesh-quality submetrics.
- **Automation/checkpoints**: Fully automatic by default; optional checkpoints allow a user to review detected objects and decisions before expensive local processing. Normal runs do not require human/agent intervention.
- **Adaptive hardware**: Detect CPU/GPU availability, choose a supported local variant, provide a CPU-only smoke path and report supported quality/resource limits. A CPU-only fallback is not claimed to reach GPU-quality benchmarks without evidence. No hidden cloud fallback.

## Implementation sequencing

Execution priorities and work-loop issue selection belong to the [whole-scene roadmap](../video-to-world-roadmap.md) (Atlas `6.1`, `6.2`, `6.3`). They do not modify any of the required artifacts or acceptance tests above. The independent fixture and three-real-video release contract remains authoritative even when a roadmap milestone or unit/synthetic smoke test passes.

## First model-backed object from the approved TUM capture (#255)

The native assembly operation can consume both `object_generation` and
`object_placement` alongside its sparse reconstruction and surface mesh.
Placement must share those exact inputs, and only placements accepted by the
existing camera, mask and depth checks enter the completed scene. The generated
asset uses a separate root mesh node and retains generative geometry provenance.
The observed GLB excludes the generated node. The source frame ordinal, RGBA
hash, mask hash and original browser-presented timestamp are recorded with the
generation request. That timestamp retains the capture's f64 value without a
conversion to ticks or a claim about the original container PTS.

The explicit yellow-die test uses the Asset Tooling
`mesh.image-to-3d.generate@1` operation with TripoSR on CUDA. It prepares one
masked RGB crop, selects the die-like package with the most observed points
and requires that it belong to the track checked against the approved visible
samples. A tied selection fails. Other objects and clean plates are outside
this run. Generated mesh provenance and the provider's approximate
reproducibility label remain distinct from the acceptance verdict for its
placement.

On Ubuntu, start with the approved 87-frame browser capture described in
[real-room-native-acceptance.md](real-room-native-acceptance.md) and
[semantic-inventory-acceptance.md](semantic-inventory-acceptance.md).
The command needs Rust, Bun, uv, Blender, NVIDIA CUDA, and the local
[Asset Tooling TripoSR runtime and model ZIP](https://github.com/moritzbrantner/asset-tooling/blob/3dd0746918dbf07a440772049791500e201de979/docs/triposr.md).
Put the TripoSR runtime's `python3` (PyTorch with CUDA, Pillow, NumPy,
Transformers, Hugging Face Hub, OmegaConf, einops, trimesh and torchmcubes)
on PATH. The Grounding DINO and SAM 2.1 sidecar uses its separate uv environment.
Install Asset Tooling's dependencies with `bun install --frozen-lockfile` in
a checkout at `3dd0746918dbf07a440772049791500e201de979`.

From the Video-to-3D checkout with the #255 changes, set the local paths and
the independently recorded model ZIP digest, then run:

```bash
export VIDEO_TO_3D_REAL_ROOM_INPUT="$HOME/.local/share/coding-agent/video179-independent-browser-input"
export VIDEO_TO_3D_REAL_ROOM_WITNESS_HASH=sha256:f45e0cf9ade9a203dc52da6be87c6154ef5c501b454eedbc3d6ec63f46bea92d
export VIDEO_TO_3D_ASSET_TOOLING=/absolute/path/to/asset-tooling
export VIDEO_TO_3D_TRIPOSR_BUNDLE=/absolute/path/to/triposr.zip
export VIDEO_TO_3D_TRIPOSR_BUNDLE_SHA256='<64-character bundle SHA-256>'
export VIDEO_TO_3D_SEMANTIC_SIDECAR="$PWD/tools/semantic-sidecar/video-to-3d-semantic-sidecar"
export VIDEO_TO_3D_OBJECT_GENERATION_SIDECAR="$PWD/tools/semantic-sidecar/asset-tooling-yellow-die.ts"
export VIDEO_TO_3D_OBJECT_EVIDENCE="$PWD/target/yellow-die-255"
export VIDEO_TO_3D_OBJECT_REPORT="$VIDEO_TO_3D_OBJECT_EVIDENCE/run.json"
mkdir -p "$PWD/target"
timeout --kill-after=30s 60m cargo test --release --locked -p video-to-3d-cli \
  --test object_identity_real real_yellow_die_exports_an_independent_blender_mesh \
  -- --ignored --exact --nocapture
```

Choose a fresh evidence directory on each attempt. The report records the
stage states, selected track and references, model and bundle evidence,
generation time, sampled whole-device VRAM, vertex and triangle counts, and
per-view placement residuals. The GPU memory sample includes driver and other
process usage. The unchanged build is replayed to check receipt reuse.
When the gate passes, the directory also contains the Blender report, preview
and editable `yellow-die.blend`, plus the native GLB under
`project/exports/scene.glb`. The Blender gate checks that translating the die
leaves the observed room mesh and other objects unchanged. This is a
single-object gate within the controlled-room acceptance plan.
