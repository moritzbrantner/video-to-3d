# Owner-approved whole-scene product decisions

Decided: **2026-10-10**, in a ten-question product-design session. Status: **product intent**, not a claim that these capabilities work yet. This document governs the initial product target alongside [the whole-scene roadmap](../video-to-world-roadmap.md) and [the independent acceptance target](hybrid-room-to-blender-acceptance.md).

## Decisions

1. **Scene decomposition (B):** Major scene elements are independently editable: individual walls, doors, windows, furniture and other meaningful objects. This granularity is the initial default; make granularity configurable later. Do not mistake disconnected triangles of one mesh for distinct objects.
2. **Unobserved geometry (C):** Offer both an **observation-only** view of supported geometry and an **AI-completed** scene with plausible missing geometry. Generated/revalidated/observed evidence stays identifiable; a completed surface is not evidence that it was filmed.
3. **Representations (C):** Support editable meshes and high-fidelity representations such as Gaussian splats where available. The user can switch representations. **Meshes are the default for Blender**, while splats may be companion visual artifacts.
4. **Input videos (B):** Ordinary handheld recordings are the product target, including imperfect camera motion. A guided, well-captured video is a valid first engineering milestone but not an acceptable final limitation.
5. **Accuracy policy (C):** Generate plausible objects when needed, **validate** their placement and appearance against video evidence, expose confidence/uncertainty, and let users correct results. Do not accept unsupported predictions as verified geometry.
6. **Automation (C):** Run automatically by default, but offer **optional** review checkpoints before expensive stages so a user can correct object detection or scene decisions. No human or agent is required to orchestrate normal processing.
7. **Execution locality (A):** **The initial product is fully local. No cloud execution, provider calls or data uploads**, including derived images. Cloud capabilities are explicitly **deferred** and require a later owner decision. Do not interpret existing cloud-provider policy fields as permission to use them in this release.
8. **Hardware (C):** Adapt model choice to available hardware; CPU-only computers should have a usable, though potentially slower or reduced-quality, path. State requested versus attainable quality and resource limitations honestly. Do not silently run an unavailable expensive model.
9. **Application surfaces (C):** Browser and desktop share the same application interface. Browser/GitHub Pages provides lightweight reconstruction and inspectable previews; desktop/local-native workers run heavyweight AI and Blender export. Preserve the repository rule that browser-side code owns video decoding and frame sampling, including within the desktop shell.
10. **Initial product exit (B):** **Three distinct licensed, pinned real videos**: one controlled room and **two ordinary handheld recordings**. All must yield recognizable environments and separately editable meaningful objects in Blender, not just successful camera estimates or a visually pleasing flat mesh. Fixture-specific expected inventories are defined **independently** before implementation is tuned. The controlled room benchmark retains the existing requirement of at least three separate object nodes; do not invent an identical numeric target for the other clips without an independent fixture oracle.

## First delivery versus follow-on flexibility

The **first acceptance** is a completely local hybrid video-to-scene run demonstrated on those three real videos, with the separately-authored synthetic scene oracle for ground-truth geometric checks. It yields a coherent scene hierarchy, observation-only and completed representations, provenance, confidence diagnostics, Blender-compatible object editing, and a user-inspectable browser preview or exported web bundle. A representation switch is required whenever both representations actually exist; lack of splats must be reported, not disguised.

The first build can use a guided control scene while development proceeds, but that alone cannot close the product acceptance issue. The native and CPU-smoke hardware paths must report limitations; passing the complete three-video quality gate is evaluated on explicitly declared available local hardware. Do not claim equivalent quality on all devices without evidence.

The eventual granularity setting, richer cloud profiles, advanced dynamic-video support and exact numerical tolerances remain future work or independently authored fixture decisions. Avoid selecting them by agent preference.

## Authority and execution

- Repository-owned product decisions and the acceptance specification define product intent; independently authored acceptance fixtures and tests verify behavior; the implementation work loop does not rewrite those expectations to make a test green.
- Observe the existing Rust/WASM/Tauri ownership split and provenance-aware evidence validation contract in `AGENTS.md`.
- Critical-path issues: #77 (product exit), #74/#75 (real/synthetic fixtures), #76 (acceptance command), #150 (native build path), #60–#63 (object decomposition), #68–#71 (objects/viewer), #78/#79 (local learned escalation), #84 (Blender export), #151 (headless Blender verification). Cloud issues remain post-v1.
- Report **implementation progress**, **functional capability**, and **three-video product acceptance** separately. A passing controlled-room fixture or merged reconstruction PR is not the final product gate.
