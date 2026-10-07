# Video clip → whole scene product roadmap

## Product target

The project succeeds when a user can provide a short video clip and receive a usable 3D scene without an agent deciding which step to run next.

The default path must be a deterministic, resumable program:

~~~text
video clip
   ↓
ingest + keyframes
   ↓
observed reconstruction
   ↓
learned reconstruction escalation
   ↓
scene understanding + object decomposition
   ↓
static-world completion + editable object generation
   ↓
registration + validation + scene assembly
   ↓
physics/navigation/material/light preparation
   ↓
interactive preview + export
~~~

Claude, Codex, ChatGPT, or a human may author a project spec, choose an optional provider, or inspect failures. They are not the workflow engine.

This roadmap intentionally goes beyond the current sparse/dense reconstruction roadmap. The existing Rust geometry pipeline remains the reconstruction authority. External models are providers of evidence or explicitly labeled generative completion.

## Definition of "whole scene"

A successful run produces a versioned project manifest containing:

- accepted cameras and reconstruction diagnostics;
- observed sparse/dense geometry;
- a navigable visual environment, preferably both a high-fidelity representation and a coarse mesh representation;
- individually editable foreground/dynamic objects where they can be isolated;
- explicit provenance for observed, learned, revalidated, and generative geometry;
- object transforms in the shared scene coordinate system;
- materials/textures sufficient for useful realtime inspection;
- collider/navigation proxies where derivable;
- optional estimated lights/environment lighting;
- optional ambient and object-specific audio;
- content-addressed source/intermediate/output identities;
- exportable scene bundles suitable for Three.js first, with GLB/USD-style downstream interoperability where practical.

The product does **not** promise that every input contains enough evidence for faithful geometric reconstruction. It does promise deterministic behavior: use the strongest accepted evidence available, escalate through configured providers, preserve provenance, and either produce the best supported scene or explain exactly which acceptance gate prevented it.

## Operating principles

1. **No agent orchestration.** A program owns the state machine, retries, cache reuse, provider selection policy, and resumability.
2. **Observed geometry remains authoritative.** Generative completion can fill unseen regions but cannot silently rewrite observed evidence.
3. **Provider-neutral boundaries.** World models, learned reconstruction models, image editors, segmentation models, image-to-3D systems, and audio systems are replaceable adapters.
4. **Idempotent convergence.** Re-running the same declared project reconciles existing verified outputs instead of repeating expensive work.
5. **Explicit escalation.** Cheap/local/deterministic paths run first; learned or paid providers run only when declared quality gates justify them.
6. **One coordinate system.** Every camera, mesh, splat, generated object, collider, and annotation is registered into one scene frame before assembly.
7. **Quality is tested on scenes, not screenshots.** Each milestone adds objective acceptance evidence and a human-inspectable preview.
8. **Source video stays on-device.** Browser decoding and sampling remain local. Cloud operations may consume only explicitly permitted locally derived artifacts; provider policy cannot authorize source-video uploads.
9. **Shared evidence validation gates provider geometry.** Learned and generated surfaces cross the versioned `ReconstructionEvidenceView` boundary before participating in shared validation/fusion, with their original provenance retained.

---

# Milestone 0 — Make the capability a product contract

## 0.1 Versioned scene-project manifest

Define a stable project document that records input media, requested quality mode, provider policy, scene outputs, provenance classes, operation identities, and export targets.

Acceptance:
- one manifest completely describes a run without ambient shell state;
- paths are portable and project-relative;
- secret/API credentials are referenced by capability name, never persisted in the manifest;
- the same manifest can be resumed after process restart.

Status: schema v1 is implemented in `crates/video-to-3d-core/src/scene_project.rs` (`SceneProjectManifest`). It records media identities, quality mode, provider policy (local/cloud, cost bounds, cloud-upload allowlist, fallback order, credential capability names), the declared operation graph with content-derived operation identities, recorded artifacts, and export targets. Quality modes gate which operation kinds may be declared; loading fails closed on unknown versions/fields and invalid combinations.

## 0.2 Canonical scene artifact model

Define the assembled scene contract without inventing another renderer-specific scene graph.

At minimum distinguish:
- cameras;
- environment visual representation;
- environment collision representation;
- accepted surface meshes;
- Gaussian splat fields;
- editable object assets;
- object transforms;
- materials/textures;
- lights/environment lighting;
- semantic labels;
- audio;
- provenance and confidence.

Acceptance:
- an asset can be visual-only, collision-only, or both;
- splat and mesh representations can coexist;
- generative objects cannot be mistaken for observed surfaces.

Status: schema v1 is implemented in `crates/video-to-3d-core/src/scene_model.rs` (`AssembledScene`). Every camera, asset, light, and audio anchor shares one right-handed, Y-up frame with an explicit unit policy (`arbitrary_monocular` or `meters`). Provenance lives on content resources (and on primitive collision proxies), so assets cannot relabel generative completion as observed geometry; `asset_provenance` derives an asset's provenance from what it references.

## 0.3 Product quality modes

Add explicit modes rather than hidden heuristic bundles:

- **preview** — fast local reconstruction and coarse outputs;
- **standard** — learned escalation + scene decomposition + usable completion;
- **production** — expensive provider escalation, higher-quality meshes/textures, stronger validation/export.

Acceptance:
- changing quality mode changes declared operations, not hidden behavior;
- diagnostics explain each escalation decision.

---

# Milestone 1 — Deterministic pipeline runner

## 1.1 Operation DAG executor

Implement a small runner for project operations. Reuse the existing asset-tooling operation model where possible instead of introducing a general orchestration framework.

Required semantics:
- typed inputs/outputs;
- dependency evaluation;
- content/build identity;
- success/failure/unsupported states;
- resumable checkpoints;
- cancellation;
- bounded retries;
- concurrency only where operations are independent.

Status: implemented in `crates/video-to-3d-core/src/scene_runner.rs` (`scene_runner::run`). It derives work from the manifest, reuses recorded artifacts whose operation identity is current, executes ready operations in deterministic waves (threads only when `max_concurrency > 1`), retries transient failures within `max_attempts`, walks reachable providers in fallback order on unsupported/permanent failures, honours a cancellation token, and records produced artifacts back into the manifest. Byte verification of recorded artifacts is left to receipt-backed reconciliation (1.2).

## 1.2 Receipt-backed cache and reconciliation

Every expensive operation records:
- normalized request identity;
- provider/model/revision;
- input hashes;
- output hashes;
- deterministic observations;
- reproducibility classification.

Acceptance:
- repeated builds reuse verified outputs;
- changed inputs invalidate only downstream work;
- interrupted builds resume from the last verified boundary.

Status: implemented in `crates/video-to-3d-core/src/scene_store.rs` (`ProjectStore`). Each recorded artifact has a receipt under `.video-to-3d/receipts/` (identity, provider/revision, input hashes, output hash and size, observations, reproducibility class). `reconcile` verifies media, outputs and receipts before every build and drops exactly the records that fail; `build` persists receipts, then the manifest, after each wave. The size/mtime hash cache is used only for read-only `VerifyMode::Cached` checks.

## 1.3 Provider policy and escalation

Represent provider preference as data:
- local-only;
- local-first;
- allow-paid-cloud;
- maximum cost/operation;
- maximum total project cost;
- quality fallback ordering.

Provider selection must be mechanical after the policy is declared.

## 1.4 CLI entry point

Target UX:

~~~text
video-to-3d build project.json
video-to-3d status project.json
video-to-3d inspect project.json
video-to-3d export project.json --target web
~~~

Tauri/browser UI may invoke the same project semantics.

Status: `crates/video-to-3d-cli` provides the `video-to-3d` binary with `build`, `status` (read-only, cached verification), `inspect` (operations, identities, providers, receipts, verification results, assembled-scene provenance), and `export --target web`. Output is one JSON document on stdout with diagnostics on stderr. Exit codes: 0 success, 1 internal, 2 invalid project, 3 unsupported capability, 4 operation/provider failure, 5 incomplete, 64 usage. No native operation executors or web exporter exist yet, so `build` and `export` currently report `unsupported` explicitly.

---

# Milestone 2 — Robust video intake

## 2.1 Media normalization

Normalize orientation, timestamps, color metadata, frame timing, and decode diagnostics while preserving original media identity.

## 2.2 Motion-aware keyframe selection

Extend current deterministic sampling with:
- baseline diversity;
- blur rejection;
- exposure-change diagnostics;
- scene-cut rejection;
- redundant-frame suppression;
- loop/revisit candidates.

## 2.3 Clip segmentation

Split obviously discontinuous input into reconstruction segments rather than forcing incompatible frames into one solve.

Examples:
- hard cuts;
- teleport-like camera changes;
- long occlusion transitions;
- extreme motion blur.

## 2.4 Reconstruction readiness score

Before expensive work, classify whether the clip has:
- sufficient translation/parallax;
- enough texture;
- enough stable scene content;
- usable exposure;
- enough temporal overlap.

Do not fabricate confidence. A weak clip can still proceed to world-model generation, but its geometric evidence remains weak.

Status (2.1 and 2.4): implemented in `crates/video-to-3d-core/src/input_readiness.rs` and shown in the browser's "Input readiness" panel. Sampling metadata (duration, display size, rotation, requested and presented sample times) is normalized and validated. Each frame gets texture density, a blur ratio, exposure and clipping, and 8x8 blockiness. Each neighbouring pair gets overlap, median motion, the 75th-percentile residual after a robust homography, and a second-homography fraction that classifies motion as duplicate, static, rotation/planar or parallax. Issues are reported against fixed, published thresholds with a geometric verdict (`ready`, `marginal` or `unsuitable`). `generative_paths_allowed` is always true and readiness never blocks reconstruction. Browser canvas output is sRGB; container colour metadata is not exposed to the page and is not claimed.

---

# Milestone 3 — Finish the observed reconstruction baseline

This continues the existing Slice 4 roadmap rather than replacing it.

## 3.1 Progressive dense execution

Complete progressive native/WASM dense reconstruction inside the existing working-set budget.

## 3.2 Better multi-reference surface coverage

Improve reference selection and seam fusion until ordinary indoor/outdoor clips produce useful surface coverage rather than isolated local patches.

## 3.3 Texture baking/export

Move from browser-only texture projection to an exportable texture/material representation while preserving exact source-camera provenance.

Status: per-reference baked textures and a self-contained textured GLB export are integrated (`surface_materials`, `textured_glb`). Seam-aware multi-view blending and scene-project `texture_bake` execution remain open.

## 3.4 Geometry confidence field

Expose per-region confidence derived from:
- camera support;
- reprojection;
- reciprocal depth agreement;
- triangulation strength;
- reference ambiguity.

This later controls whether learned/generative systems may fill or replace missing regions.

Status: integrated in `geometry_confidence` (`GeometryConfidenceField`). Every evidence region gets the five factors, a provenance-weighted confidence and a band; completion policies query by region, point, or triangle (`regions_below`). Learned/generative fill policies that consume it remain open.

## 3.5 Coarse collision mesh from accepted geometry

Produce a conservative low-complexity collision representation without changing visual geometry.

Status: integrated in `coarse_collision` (`CoarseCollider`): merged axis-aligned boxes over grid cells intersected by confident observed triangles, exported as a separate glTF `collision` scene. Per-object colliders and scene-model collision proxies remain Milestone 10 work.

---

# Milestone 4 — Learned reconstruction escalation

## 4.1 Owned native provider interchange

Complete the owned form of `ReconstructionEvidenceView` for native/Tauri worker processes.

## 4.2 First learned multi-view provider

Integrate one strong feed-forward multi-view provider through the common evidence contract.

Preferred research surface: MapAnything so multiple model families can later be compared behind one adapter.

Acceptance:
- model output never bypasses Rust validation;
- provider-specific coordinates are normalized explicitly;
- camera/intrinsic confidence is retained.

## 4.3 Learned feature/matcher escalation

Add optional learned features/matching for frames where the built-in path fails:
- XFeat-class local features;
- LightGlue-class matching;
- dense correspondence escalation for difficult pairs.

## 4.4 Learned calibration priors

Allow model-estimated focal/principal point/distortion/gravity as proposals, then validate them against reconstruction consistency.

## 4.5 Hybrid geometric + learned reconstruction

Fuse accepted learned depth/correspondence with geometric cameras rather than replacing the classical solution wholesale.

## 4.6 Provider benchmark harness

Benchmark built-in, learned, and hybrid modes on identical deterministic fixtures and real-video canaries.

Metrics include:
- registered camera fraction;
- camera error where ground truth exists;
- accepted surface coverage;
- reprojection/consistency error;
- topology validity;
- memory;
- runtime;
- provider cost.

---

# Milestone 5 — Scene understanding and decomposition

This is the major capability missing from the current reconstruction-focused architecture.

## 5.1 Keyframe semantic inventory

Generate a structured scene inventory:
- static environment regions;
- likely movable objects;
- people/animals;
- vegetation;
- reflective/transparent regions;
- sky/water/fire/smoke;
- floor/ground/walls where identifiable.

This is evidence/metadata, not geometry authority.

## 5.2 Cross-frame object tracks

Associate the same object across keyframes so object extraction uses multiple views rather than treating every frame independently.

## 5.3 Static-versus-dynamic classification

Estimate whether each tracked region is:
- static scene structure;
- movable-but-stationary object;
- actually moving;
- uncertain.

## 5.4 Object masks with temporal consistency

Create temporally stable masks for selected editable objects.

Acceptance:
- masks can be reprojected/compared across registered cameras;
- inconsistent masks are surfaced instead of silently accepted.

## 5.5 Scene hierarchy proposal

Produce a semantic hierarchy useful for editing:
- environment;
- furniture/props;
- vehicles;
- characters;
- vegetation;
- effects/ephemera.

Keep it easy to override.

---

# Milestone 6 — Static environment completion

The system should have two independent routes to a usable background.

## 6.1 Reconstruction-first environment

Build the static environment from accepted observed/learned geometry whenever coverage is sufficient.

## 6.2 Clean-plate generation

For objects selected as editable foreground assets, remove them from representative source views and generate clean plates.

Requirements:
- preserve camera/frame identity;
- retain edit masks;
- record provider/model;
- validate edited regions against unedited surroundings;
- never relabel edited pixels as observations.

## 6.3 World-model provider adapter

Add a provider-neutral world-generation operation with explicit conditioning capabilities. Video conditioning is available only to providers running locally; the uploaded source video must never leave the device. Cloud adapters may receive only explicitly permitted locally derived images or other artifacts after the user authorizes that operation.

A first optional, replaceable cloud adapter may target Marble if its supported conditioning can satisfy this boundary. A cloud provider that requires source-video upload is unsupported, regardless of quality mode or paid-cloud policy.

## 6.4 World-model import

Normalize returned:
- Gaussian splat/SPZ or equivalent;
- visual mesh;
- collider mesh;
- provider cameras/transforms;
- metadata.

Do not assume provider coordinates match the reconstruction frame. Compatible generated meshes/surfaces must enter the owned, versioned `ReconstructionEvidenceView` contract with generative-completion provenance and pass Rust-owned validation before shared fusion or per-region selection. Import and registration alone do not accept geometry. Splats without a compatible surface evidence view remain explicitly generative visual artifacts and cannot participate in shared surface fusion.

## 6.5 World-to-observed registration

Register generated world output against accepted reconstructed cameras/surfaces.

Use multiple anchors:
- camera poses where available;
- robust point/surface alignment;
- semantic/object anchors only as secondary evidence.

Reject implausible alignment rather than visually fudging it.

## 6.6 Selective environment completion

Use the world model primarily for unsupported/unseen regions when observed reconstruction is already strong.

Goal: preserve real evidence while filling the back sides, corners, ceilings, distant structure, and other missing regions.

## 6.7 Environment representation policy

Choose among:
- observed mesh;
- generated mesh;
- splat;
- hybrid mesh+splat;

per region based on accepted evidence and target runtime. Only surfaces admitted through the shared evidence gate may participate in shared validation/fusion; visual-only generative artifacts retain their separate provenance and do not become observed geometry.

---

# Milestone 7 — Editable object generation

## 7.1 Multi-view object reference extraction

For each selected object, prepare:
- best front/reference view;
- additional side/back views when visible;
- masks;
- estimated scale;
- observed point/surface snippets;
- semantic description.

## 7.2 Provider-neutral image/multiview-to-3D operation

Expose one operation contract in asset-tooling for image/multi-view → 3D.

Initial adapters should reuse existing local providers and allow cloud providers.

## 7.3 Local object generation path

Dogfood existing asset-tooling support:
- TRELLIS.2;
- Stable Fast 3D;
- TripoSR fallback.

## 7.4 Cloud object generation path

Add a high-quality provider adapter, e.g. Hunyuan3D-class multi-view/image-to-3D.

Provider choice is policy-driven rather than agent-selected.

## 7.5 Object geometry validation

Validate generated objects for:
- finite geometry;
- manifold/degenerate diagnostics;
- texture/material completeness;
- sane bounds;
- pivot/orientation;
- declared units;
- visual agreement with reference views.

## 7.6 Object reconstruction versus generation arbitration

When enough multi-view evidence exists, prefer reconstructed geometry.
When evidence is incomplete, allow generated completion.
Preserve which regions came from which source.

## 7.7 Rigging/animation hook

For recognized characters/animals, allow optional downstream rigging/animation preparation through existing asset-tooling/3d-lab contracts. This is not on the minimum critical path for whole-scene generation.

---

# Milestone 8 — Register objects into the scene

## 8.1 Scale estimation

Estimate object scale from reconstructed scene context instead of accepting arbitrary generator scale.

Evidence may include:
- observed object point cloud;
- camera geometry;
- floor contact;
- known semantic size priors as weak fallback.

## 8.2 Pose estimation

Fit generated/reconstructed object geometry into its observed multi-view masks/silhouettes.

## 8.3 Contact and support constraints

Detect likely support surfaces:
- floor;
- table;
- shelf;
- wall mount.

Prevent floating/sunken placement where evidence is strong.

## 8.4 Occlusion validation

Render placed objects back into accepted cameras and compare:
- silhouette;
- depth ordering;
- coverage;
- visible landmarks.

## 8.5 Scene assembly acceptance

An object enters the canonical assembled scene only after placement passes declared checks or is marked as manual-review/approximate.

---

# Milestone 9 — Materials, lighting, and visual coherence

## 9.1 Material normalization

Normalize generated and reconstructed materials into the existing production GLB/PBR processing path.

## 9.2 Environment lighting estimate

Estimate a practical environment light/HDR-like approximation from video/world evidence.

## 9.3 Shadow/contact coherence

Generate or approximate contact shadows for inserted editable objects so they do not visually float against a photoreal environment.

## 9.4 Appearance matching

Support bounded color/exposure/material adjustment to make generated objects visually consistent with the source scene while preserving source/provenance.

## 9.5 Level-of-detail derivatives

Create preview/runtime/production variants without regenerating semantic identity.

---

# Milestone 10 — Physics and navigability

## 10.1 Static collider generation

Generate coarse colliders from observed or provider collider meshes.

## 10.2 Object colliders

Use existing 3d-lab/asset-tooling processing to derive safe simplified colliders for editable objects.

## 10.3 Walkable surface extraction

Identify candidate walkable surfaces and produce a navigation preview.

## 10.4 Physical plausibility diagnostics

Surface:
- unsupported floating objects;
- severe collider/visual mismatch;
- blocked walkable areas;
- impossible object intersections.

## 10.5 Runtime-neutral physics metadata

Keep collision semantics suitable for downstream engines without coupling the project to one engine.

---

# Milestone 11 — Hybrid realtime viewer

## 11.1 Splat rendering

Integrate an efficient splat viewer so world-model and future 3DGS outputs can be inspected directly.

## 11.2 Mesh + splat composition

Render:
- splat/static environment;
- editable mesh objects;
- reconstructed mesh patches;
- collision/debug overlays;

in one coordinate frame.

## 11.3 Provenance inspection

Allow selecting a point/triangle/object and seeing whether it is:
- geometric observation;
- learned multi-view evidence;
- revalidated completion;
- generative environment;
- generated object.

## 11.4 Scene editing basics

Minimum useful editor:
- hide/show;
- translate/rotate/scale editable objects;
- choose representation per environment region;
- replace/regenerate one object;
- rerun one failed stage without rebuilding the project.

---

# Milestone 12 — Audio and ambience

Audio is valuable but must remain off the critical path for visual scene completion.

## 12.1 Ambient scene description

Derive a bounded semantic audio description from scene inventory.

## 12.2 Ambient audio generation

Reuse asset-tooling's provider-neutral audio operation.

## 12.3 Object interaction SFX

Optionally generate sound sets for selected editable/physical objects.

## 12.4 Spatial audio anchors

Associate ambient zones and object SFX with scene transforms without inventing a new runtime audio engine.

---

# Milestone 13 — Export and downstream use

## 13.1 Self-contained web scene bundle

Export a portable bundle containing:
- scene manifest;
- meshes/materials/textures;
- splats;
- colliders;
- audio;
- provenance;
- viewer configuration.

## 13.2 GLB-oriented export

Export mesh-compatible scene content through existing production GLB normalization.

## 13.3 DCC handoff

Provide a Blender-friendly import package retaining scene coordinates and provenance sidecars.

## 13.4 Game-engine handoff

Document/test a minimal Unity/Unreal/Godot-friendly path without moving engine-specific semantics into core.

## 13.5 Re-editability

An exported project can be reopened and one object/provider result can be replaced without reconstructing unrelated assets.

---

# Milestone 14 — Quality assurance that makes the capability dependable

## 14.1 Scene acceptance corpus

Maintain representative clips, not just one demo:

1. textured tabletop object orbit;
2. ordinary indoor room;
3. cluttered room with furniture;
4. outdoor courtyard/building;
5. vegetation-heavy outdoor clip;
6. reflective/glossy surfaces;
7. weak-texture walls;
8. clip with a moving person/object;
9. loop/revisit camera path;
10. difficult public canary such as Trevi.

Licensing and exact source identity must be recorded.

## 14.2 Ground-truth synthetic scenes

Render known scenes with exact cameras, depth, segmentation, materials, object transforms, and geometry.

These provide objective end-to-end regression evidence.

## 14.3 Whole-scene metrics

Track:
- registered-camera ratio/error;
- surface coverage;
- depth/reprojection consistency;
- mesh topology validity;
- object detection/track consistency;
- object silhouette reprojection score;
- object transform error on synthetic scenes;
- collision mismatch;
- unobserved/generated area fraction;
- export validity;
- runtime/memory;
- provider calls and monetary cost.

## 14.4 Visual regression captures

Keep stable camera viewpoints for human review without using screenshot similarity as the geometry authority.

## 14.5 Fault injection

Test:
- provider timeout;
- malformed response;
- quota exhaustion;
- interrupted download;
- partial cache;
- corrupted model output;
- unavailable GPU;
- browser memory limit;
- user cancellation.

The project must remain resumable.

## 14.6 End-to-end acceptance command

Target:

~~~text
video-to-3d accept fixtures/room/project.json
~~~

It builds the scene from clean state, validates the artifact graph, runs geometry/scene/export checks, and emits a concise quality report.

---

# Milestone 15 — Cost, privacy, and execution profiles

## 15.1 Fully local profile

Provide the best no-cloud pipeline available:
- built-in reconstruction;
- native learned reconstruction;
- local image-to-3D;
- local image edit/segmentation where practical;
- local asset processing.

## 15.2 Cloud-augmented profile

Allow optional high-value calls:
- world model;
- high-quality object generation;
- image cleanup;
- audio.

## 15.3 Cost estimator

Before execution, estimate which paid operations may run and their configured maximum spend.

## 15.4 Privacy classification

Source video always stays local. Reject any cloud request containing source-video bytes, including a repackaged or transcoded source clip. Clearly identify which optional operations upload explicitly permitted locally derived imagery or other artifacts, require authorization before transfer, and record that choice in the operation receipt. Local-only policy forbids all such transfers.

## 15.5 Provider substitution tests

At least two providers should satisfy each strategically important generative boundary over time so the product is not architecturally coupled to one vendor.

---

# Milestone 16 — Advanced scene fidelity

These are after the first dependable whole-scene pipeline exists.

## 16.1 Multi-video same-scene fusion

Merge multiple clips into one project and coordinate system.

## 16.2 Dynamic-scene reconstruction

Add explicit support for moving people/vehicles/objects instead of only masking them from static reconstruction.

## 16.3 3D Gaussian splat optimization

Continue the existing Slice 5 work using accepted cameras/geometry as initialization.

## 16.4 World expansion

Allow expanding beyond the observed clip while clearly marking generated territory.

## 16.5 Semantic scene editing

Examples:
- remove object;
- replace chair;
- clear table;
- extend room;
- regenerate one poor region.

Edits operate on the project graph and invalidate only affected outputs.

## 16.6 Scene composition

Combine generated/reconstructed sub-scenes under explicit transforms and provenance.

---

# Critical path to the first "magic" demo

Do not wait for every milestone above. The first compelling end-to-end result should follow this path:

~~~text
M0 project contract
  ↓
M1 deterministic runner
  ↓
M2 video/keyframes
  ↓
M3 current reconstruction
  ↓
M5 semantic object decomposition
  ├───────────────┐
  ↓               ↓
M6 static world   M7 object generation
  └───────┬───────┘
          ↓
M8 registration
          ↓
M11 hybrid viewer
          ↓
M13 export
          ↓
M14 whole-scene acceptance
~~~

For the first demo, use a short clip of a scene with:
- a mostly static room/environment;
- 2–5 clearly separable foreground objects;
- visible camera translation;
- modest reflections;
- no cuts.

The output should visibly contain:
- a navigable static world;
- at least one independently editable generated/reconstructed object correctly placed;
- a coarse collision representation;
- one-click provenance/debug view;
- a portable scene bundle.

Once this path works, increase scene difficulty systematically rather than adding more orchestration.

---

# Suggested implementation order

## P0 — prove the product loop

1. 0.1 project manifest
2. 0.2 scene artifact model
3. 1.1 operation DAG executor
4. 1.2 receipts/cache/resume
5. 1.4 CLI
6. 2.1–2.4 robust intake/readiness
7. 3.1 progressive dense execution
8. 5.1 semantic inventory
9. 5.2–5.4 object tracking/masks
10. 6.2 clean plates
11. 6.3 world-model provider
12. 6.4–6.5 world import/registration
13. 7.1 object references
14. 7.2 provider-neutral object generation
15. 7.3 local provider path
16. 7.4 cloud provider path
17. 8.1–8.4 object placement/validation
18. 11.1–11.3 hybrid viewer/provenance
19. 13.1 web scene bundle
20. 14.1–14.3 acceptance corpus/metrics

Exit criterion: a representative short video reliably produces a complete inspectable scene through one command with no agent deciding the next operation.

## P1 — make results good and useful

- learned reconstruction escalation;
- improved multi-reference surfaces;
- texture baking;
- selective world completion;
- material/light coherence;
- colliders/walkability;
- editing/regeneration;
- production GLB/DCC export;
- fault injection and cost/privacy profiles.

## P2 — broaden the domain

- multiple videos;
- dynamic scenes;
- splat optimization;
- larger-world expansion/composition;
- semantic scene editing;
- character rigging/animation;
- advanced audio.

---

# Architectural ownership

## video-to-3d owns

- media-to-scene project semantics;
- camera/reconstruction authority;
- evidence validation/fusion;
- scene coordinate system;
- scene assembly policy;
- registration;
- quality gates;
- whole-scene acceptance;
- viewer-level provenance.

## asset-tooling owns

- provider-neutral generation operations;
- local/cloud model adapters where they produce reusable assets;
- content-addressed object/material/audio assets;
- generation receipts/provenance;
- asset normalization and processing;
- GLB/material/collider derivative workflows.

## 3d-lab owns

- authoritative reusable geometry/scene processors already delegated there;
- mesh normalization/simplification/export semantics where established.

Do not create a new generic workflow/orchestration repository for this roadmap. The project runner should be only as general as needed to execute the declared media-to-scene operation graph.

---

# Capability acceptance statement

The roadmap is complete when **"video clip → whole scene" is an automated tested capability rather than a lucky demo**:

- one command starts or resumes the build;
- no language model is required to sequence tasks;
- a useful scene can combine observed reconstruction, learned evidence, and generative completion without losing provenance;
- foreground objects can become independently editable assets;
- the static world remains navigable even when direct reconstruction is incomplete;
- every expensive provider call is recorded and resumable;
- failures are localized and retryable;
- the result exports into a normal realtime/DCC workflow;
- a maintained scene corpus continuously proves the capability across progressively harder inputs.
