# Video-to-3D Decision Atlas

This file is the append-only product-intent map for `video-to-3d`: the durable place for **what the product should be**.

## Rules

- The Atlas has exactly **two semantic leaf-entry kinds: Decision and Question**. Section headings only organize entries and are not a third kind. Existing leaves not explicitly marked `[question]` are Decisions.
- A **Decision** is owner-selected product intent. Agents follow it until a later Decision explicitly supersedes it. A **Question** is a consequential unresolved choice; it is not automatically a blocker.
- Coding agents retrieve relevant Atlas entries before making consequential product choices. Follow applicable Decisions. If an unanswered Question materially affects the current task, ask the owner **one question at a time**. If work exposes a new consequential fork, append a Question instead of choosing silently.
- When a Question is answered, preserve it and append a Decision with `Answers X.Y`. When intent changes, preserve the old Decision and append a Decision with `Supersedes X.Y`. Do not add parallel semantic categories such as constraint, deferred decision, assumption, or principle; express those meanings in the Decision/Question text and relationships.
- Entry IDs are stable. Never renumber or reuse an ID. New entries are appended under the narrowest existing parent; if no suitable parent exists, append a new top-level section.
- Cross-references use Atlas IDs such as `1.3` or `3.2.1`.
- The Atlas records product intent, not implementation status. A Decision being present does not claim that the capability exists. Detailed specifications and acceptance tests may elaborate Decisions, but must not silently contradict them.

Sources seeded into this atlas:
- `AGENTS.md`
- `docs/specs/whole-scene-product-decisions.md`
- `docs/specs/hybrid-room-to-blender-acceptance.md`

## 1. Product shape

- **1.1 Major scene decomposition is the default.**  
  Status: adopted 2026-10-10.  
  Individual walls, doors, windows, furniture, and other meaningful objects are independently editable scene identities by default. Disconnected triangles of one mesh do not count as separate objects. Finer/coarser granularity may become configurable later. See 3.2 and 7.2.

- **1.2 Preserve observation-only and AI-completed scene modes.**  
  Status: adopted 2026-10-10.  
  The product exposes both geometry supported by observations and an AI-completed scene with plausible missing geometry. Completion must not be presented as filmed evidence. See 2.3, 3.4, and 5.2.

- **1.3 Editable meshes are the default DCC representation.**  
  Status: adopted 2026-10-10.  
  Blender receives editable meshes by default. Higher-fidelity representations such as Gaussian splats may accompany them. A representation switch is required whenever both mesh and splat representations actually exist; splats do not substitute for mesh editability. See 3.3 and 4.2.

- **1.4 Ordinary handheld video is the product target.**  
  Status: adopted 2026-10-10.  
  Imperfect handheld recordings are supported product input. A guided capture may be used as an engineering milestone, but cannot become the final product limitation. See 3.1.

- **1.5 Generated content requires video-grounded validation.**  
  Status: adopted 2026-10-10.  
  Plausible objects or missing geometry may be generated when needed, but placement and appearance must be checked against retained video evidence. Unsupported predictions remain uncertain rather than becoming verified geometry. See 2.3, 3.4, and 5.2.

- **1.6 Normal processing is automatic.**  
  Status: adopted 2026-10-10.  
  Normal runs require no human or agent orchestration. Optional review checkpoints may appear before expensive local stages so a user can correct proposed object identities or scene decisions. See 4.4.

- **1.7 Hybrid per-region mesh/splat viewing is the default when both exist.**  
  Status: adopted 2026-10-12. Answers 8.4 (owner selected B).  
  Show high-fidelity splat surroundings and independently editable mesh objects together in one aligned scene when those representations exist. Let users select individual objects or scene regions and switch between mesh and splat display wherever both representations are actually available, without replacing the rest of the scene, jumping the camera, or modifying the underlying geometry, transforms, object identity, or evidence provenance. Meshes remain the editable representation and the default Blender/DCC export; splats are visual companions, not a substitute for mesh editability. When a region has only one representation, show it honestly and indicate that the other is unavailable rather than fabricating it. Preserve observation-only versus AI-completed provenance independently of display choice. Do not make optional splat generation or optimization a prerequisite for the initial three-real-video editable-mesh acceptance gate. See 1.2–1.3, 2.3, 3.2–3.4, and 4.2.  

## 2. Authority and provenance

- **2.1 Rust owns reusable reconstruction semantics.**  
  Status: repository architecture.  
  Feature detection, matching, camera-path estimation, point generation, evidence validation/fusion, and shared scene data contracts belong in `crates/video-to-3d-core`.

- **2.2 Reconstruction backends are evidence providers, not geometry authorities.**  
  Status: repository architecture.  
  Classical, learned multi-view, and generative backends must cross the versioned `ReconstructionEvidenceView` boundary before their output can participate in shared validation or fusion. See 2.1 and 2.3.

- **2.3 Evidence provenance remains explicit.**  
  Status: repository architecture.  
  Geometric multi-view evidence, learned evidence, image-revalidated completion, and generative completion remain distinguishable. Generative or completed geometry must never be silently relabeled as observed geometry. See 1.2 and 1.5.

- **2.4 Avoid unnecessary geometry copies.**  
  Status: repository architecture.  
  Prefer borrowed/ranged evidence and explicit provenance metadata. Materialize owned snapshots only when export or external interchange requires ownership.

- **2.5 WASM is an adapter boundary.**  
  Status: repository architecture.  
  `crates/video-to-3d-wasm` adapts shared logic; it does not duplicate reconstruction semantics from 2.1.

- **2.6 Native providers must reuse shared validation/fusion semantics.**  
  Status: repository architecture.  
  Tauri/native workers may execute heavyweight local providers, but their evidence enters through the same authority boundary from 2.2 rather than creating a second native reconstruction authority.

## 3. Product acceptance

- **3.1 The initial release gate uses three real videos.**  
  Status: adopted 2026-10-10.  
  The first product exit requires three distinct, licensed, pinned real videos: one controlled room and two ordinary handheld recordings. All three must pass. A synthetic fixture supplements this gate but cannot replace it. See 1.4 and 3.5.

  - **3.1.1 Controlled-room fixture.**  
    The controlled room retains the requirement for at least three recognizable, independently selectable scene objects.

  - **3.1.2 Handheld fixture expectations are independently authored.**  
    The two handheld clips receive their own observable object/architecture inventories before implementation is tuned. Do not copy the controlled-room numeric object threshold without an independent fixture decision.

- **3.2 Scene identity must survive into exported/editable output.**  
  Status: adopted 2026-10-10.  
  Meaningful architectural elements and objects remain separately identifiable, selectable, and transformable through the scene pipeline. Proxy nodes may aid identification but do not count as finished independently editable geometry. See 1.1.

- **3.3 Blender import/edit is part of product acceptance.**  
  Status: adopted 2026-10-10.  
  Acceptance is not satisfied by GLB generation alone. A pinned headless Blender test imports production output, verifies hierarchy/materials/transforms, and proves that an object can be moved independently without altering the environment. See 1.3.

- **3.4 Acceptance must expose uncertainty and provenance.**  
  Status: adopted 2026-10-10.  
  Both observation-only and AI-completed modes retain provenance, confidence, source references, and explicit incomplete/unsupported regions. See 1.2, 1.5, and 2.3.

- **3.5 Synthetic exact-truth fixtures are verification oracles, not release substitutes.**  
  Status: adopted 2026-10-10.  
  Deterministic synthetic fixtures with exact cameras, geometry, segmentation, and transforms validate algorithmic stages and tolerances. They complement, but never replace, the three-real-video release gate in 3.1.

- **3.6 Product acceptance is end-to-end.**  
  Status: adopted 2026-10-10.  
  The aggregate gate is green only when the complete local video → scene → Blender workflow passes for all three pinned real clips. Unit tests, one curated clip, successful camera estimation, or a visually pleasing flat mesh do not close the product gate.


- **3.7** [question] **Controlled-room source for the initial three-video gate.**  
  Resolved 2026-10-12: should the original RGB movie from TUM RGB-D `freiburg3_long_office_household` become the controlled-room fixture required by 3.1.1, or should the repository keep searching/recording another source?

- **3.8 Controlled-room fixture uses the TUM RGB-D office/household RGB movie.**  
  Status: adopted 2026-10-12. Answers 3.7.  
  Use the original RGB movie from TUM RGB-D `freiburg3_long_office_household` as the controlled-room member of the three-real-video initial release gate. Preserve its CC BY 4.0 attribution and pinned content identity. The fixture must satisfy 3.1.1 through independently authored observable acceptance expectations; TUM depth and trajectory data are not production reconstruction inputs. Selecting the source does not itself satisfy the end-to-end product gate in 3.6.


## 4. Application and execution

- **4.1 Initial release processing is fully local.**  
  Status: adopted 2026-10-10.  
  No cloud inference, provider calls, source uploads, or derived-frame uploads are permitted in the initial release. Cloud execution requires a later explicit owner decision, subject to the persistent source-video privacy invariant in 4.3.

- **4.2 Browser and desktop share one application interface.**  
  Status: adopted 2026-10-10.  
  GitHub Pages/browser provides lightweight reconstruction and inspectable previews. Desktop/local-native execution provides heavyweight AI and Blender export. The user-facing product remains one interface rather than two separate applications.

- **4.3 Browser-side code owns video decoding and frame sampling.**  
  Status: repository architecture.  
  This remains true inside the Tauri desktop shell. Native workers must not create a competing authoritative decode/sampling path. Uploaded source video must not leave the device; this repository privacy invariant also constrains any later cloud policy. See 4.1 and 4.2.

- **4.4 Optional review checkpoints occur before expensive local stages.**  
  Status: adopted 2026-10-10.  
  The automatic default from 1.6 remains intact. Review checkpoints may allow corrections to object identity, uncertainty, or scene placement while preserving original evidence and deterministic invalidation/resume behavior.

- **4.5 Hardware selection adapts locally and reports limits honestly.**  
  Status: adopted 2026-10-10.  
  Provider/model choice adapts to available CPU/GPU resources. CPU-only operation must remain usable, potentially at reduced quality or speed, and requires a CPU-only smoke check. The exact minimum quality tier remains unanswered in 8.2. Requested versus attainable quality and unavailable capabilities must be explicit; there is no hidden cloud fallback. See 4.1.

- **4.6 CPU-only operation preserves a useful observation-only workflow.**  
  Status: adopted 2026-10-12. Answers 8.2 (owner selected B).  
  On machines without a suitable GPU, users can locally reconstruct video-supported geometry, inspect the accepted scene, and export supported independently editable geometry via the existing scene/Blender handoff. CPU-only processing may be slower or lower quality; GPU-dependent AI-completion providers and their representations may be unavailable. Report unavailable or degraded capabilities and reasons explicitly rather than silently skipping stages, fabricating observed geometry, calling a cloud service, or marking a missing AI-completed representation as present. A CPU-only smoke test must exercise real local observed reconstruction, inspection and supported export when accepted geometry is available, and honest unavailable-completion reporting when it is not. This does not waive the full three-real-video product gate on declared capable local hardware in 3.6. See 1.2–1.5, 3.3–3.6, and 4.1–4.5.  

- **4.7 First learned multi-view provider prioritizes a reproducible 8 GB local GPU fit.**  
  Status: adopted 2026-10-12. Answers 8.1 (owner delegated provider choice based on GeForce RTX 3060 with 8 GB VRAM).  
  Evaluate `facebook/map-anything-apache` (Apache 2.0 checkpoint) first as the candidate local learned multi-view evidence provider, using its documented memory-efficient inference, mixed precision and minimal inference minibatch, and existing camera/calibration inputs when available. This is a prioritized **candidate, not a verified claim that it fits 8 GB or satisfies the product**. Before enabling it as a supported production profile, run a reproducible native on-device benchmark on an RTX 3060-class **8 GB** GPU, recording model/checkpoint and license, software versions, image resolution, view counts, actual peak device VRAM, runtime, success/OOM and accepted geometric evidence on relevant pinned clips. Select the largest demonstrably supported batch/view policy rather than guessing; if it cannot fit or produce useful accepted evidence, report unavailable capability, retain the existing local observed-geometry fallback, and revisit provider selection using measured evidence instead of silently substituting a model. Learned output remains subject to the common Rust `ReconstructionEvidenceView` validation/fusion boundary, explicit provenance and the unchanged three-real-video scene-to-Blender product acceptance. No cloud inference, media upload or implicit CPU/GPU equivalence. Source: [MapAnything repository](https://github.com/facebookresearch/map-anything) and [Apache checkpoint](https://huggingface.co/facebook/map-anything-apache). See 2.2–2.3, 3.1–3.6, 4.1–4.6.  

## 5. Quality and correction

- **5.1 Every algorithmic change requires deterministic Rust coverage and observable diagnostics.**  
  Status: repository architecture.  
  Algorithmic changes must have deterministic Rust tests and a diagnostic signal that makes failure observable in the UI.

- **5.2 Object placement is accepted only against retained source evidence.**  
  Status: adopted product direction.  
  Scale, pose, support relationships, occlusion, depth ordering, and appearance should be validated by reprojecting candidates into registered source views. Contradictory views prevent silent acceptance. See 1.5 and 3.4.

- **5.3 Corrections must not rewrite source evidence.**  
  Status: adopted product direction.  
  User corrections alter proposed interpretation or generated/completed scene state, not the original captured evidence. Derived artifacts should invalidate only along their actual dependency graph.

- **5.4 Provide meaningful corrections throughout the processing pipeline, starting in source frames.**  
  Status: adopted 2026-10-12. Answers 8.3 (owner chose B as the first slice).  
  Every meaningful processing stage should ultimately allow the user to review and correct its derived interpretations or outputs when appropriate, rather than forcing an all-or-nothing rerun. Begin with optional video-based correction of object identities, masks and cross-frame tracks in sampled source frames before expensive local reconstruction or completion stages. Preserve automatic processing as the default; no user checkpoint is mandatory for normal operation. Corrections must not modify original video or reclassify generated guesses as observed evidence, must retain provenance, and should invalidate/recompute only dependent downstream work through the existing deterministic project runner. Other stage-specific correction affordances may be delivered incrementally as their outputs become inspectable; this direction does not require inventing an editor for every internal algorithmic substep in the first bounded slice. See 1.6, 4.4, 5.2 and 5.3.  

## 6. Delivery priority

- **6.1 Product-exit work outranks unrelated reconstruction polish while the product gate is red.**  
  Status: adopted from current acceptance policy.  
  When an actionable dependency on the three-video product gate exists, prefer it over unrelated P1/P2 mesh/material refinements. Correctness, safety, provenance, and bounded-resource defects in work already being merged remain valid blockers.

- **6.2 The executable critical path is the three-video scene-to-Blender slice.**  
  Status: adopted from current acceptance policy.  
  Work should converge on the earliest failing dependency of the complete pipeline: fixture corpus → runnable local build → provider evidence → semantic/object identity → placement validation → scene hierarchy → Blender handoff → end-to-end acceptance.

- **6.3 Progress is reported in three layers.**  
  Status: adopted 2026-10-10.  
  Report separately:
  1. code/infrastructure merged,
  2. functional capability demonstrated,
  3. three-video product acceptance passed.

  Merged PR counts alone are not product progress.

## 7. Further decisions

- **7.1 Cloud execution is deferred.**  
  Cloud providers, upload policy, paid-provider exposure, and cloud/local profile selection are not part of the initial release. Their eventual shape remains unanswered in Question 8.5, subject to the source-video privacy invariant in 4.3. Nothing in the current schema implies permission to use cloud execution. See 4.1.

- **7.2 User-configurable scene granularity is deferred.**  
  The initial default remains major architectural elements plus meaningful objects from 1.1. The exact UI and supported granularity levels remain unanswered rather than forming a separate "deferred decision" category.

- **7.3 Exact numerical quality tolerances remain fixture-owned where not already defined.**  
  Agents must not invent arbitrary universal thresholds merely to make implementation decisions. Fixture-specific tolerances belong to independently authored acceptance evidence.

## 8. Questions

These are unresolved consequential choices. They remain Questions until the owner answers them; answering preserves the Question and appends a linked Decision.

- **8.1** [question] **First local learned provider selection.**  
  Unresolved: which concrete local model/provider family becomes the first production learned multi-view evidence source under 2.2 and 4.5.

- **8.2** [question] **Minimum acceptable CPU-only product experience.**  
  Unresolved: which stages/quality tier are required versus explicitly unavailable on CPU-only hardware under 4.5.

- **8.3** [question] **User correction semantics at review checkpoints.**  
  Unresolved: which edits are permitted in the first release checkpoint UI and which changes require recomputation, subject to 4.4 and 5.3.

- **8.4** [question] **Representation switching UX.**  
  Unresolved: exact behavior for mesh/splat switching when both exist, subject to 1.3 and 4.2.

- **8.5** [question] **Post-v1 cloud policy.**  
  Unresolved and intentionally deferred by 7.1. Any later cloud/derived-frame upload policy remains subject to 4.3: source video must not leave the device.
