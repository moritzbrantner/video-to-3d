# Product acceptance: first hybrid room-to-Blender scene (v1)

Status: **proposed contract; not yet implemented or passed**. Owner decision: **hybrid scene reconstruction (B)**, 2026-10-10.

## User-visible goal
Given one fixed, licensed, mostly static short room video with camera translation and at least three visually distinguishable objects, a single documented `video-to-3d build <project>` invocation must produce a coherent environment and separately editable Blender objects. No human or agent selects intermediate operations.

## Fixture ownership and independent oracle
- Check in or record a reproducibly downloadable, legally usable input clip with pinned SHA-256, fixed frame sampling, and an independently authored object inventory and expected relationships; don't assert hidden physical geometry unavailable in video.
- Author acceptance fixtures/expected results separately from the implementation PR/agent. Version fixture, model/provider, and quality policy.
- Include a deterministic synthetic control scene with known cameras, object identifiers, silhouettes, transforms and coordinate conventions, plus a real room canary. Distinguish measured ground truth from qualitative inspection.
- Start with local/native execution; external provider calls must be policy gated, logged, versioned and bounded. Browser source-video privacy remains unchanged.

## Required artifacts
1. One resumable scene-project manifest with content-addressed operation receipts and provider/model versions.
2. One shared coordinate frame containing accepted cameras and static environment; visually usable preview and an explicit report of unobserved/invented regions.
3. At least three **distinct scene object nodes**, independently selectable and transformable in Blender. At least one must contain its own editable mesh asset; placeholders/proxy geometry may count only for object identification, not for final reconstructed/generated mesh fidelity. Objects must not merely be three chunks of one environment mesh.
4. Each object has stable identity across selected keyframes, semantic label (including uncertain), source-frame/mask references, transform, origin/pivot, material, and provenance (observed / learned / revalidated / generative).
5. A Blender-compatible handoff (prefer standard GLB with meshes, materials, transforms and names, plus sidecar for provenance not expressible in GLB). A **headless Blender import test** asserts node count, distinct mesh datablocks or declared reusable instances, non-degenerate bounds, transform validity, material references and scene graph hierarchy. Test independently moving an object while the environment remains unchanged.
6. At least a coarse environment collision representation and a scene quality report with explicit failures, unsupported capabilities and incomplete sections.

## Acceptance tests (non-negotiable)
- **E2E**: On the pinned real fixture, one command produces the required artifacts; import in pinned Blender version; inspect named environment and object hierarchy and render a canonical view; fail if an operation reports unsupported or expected scene artifacts are absent.
- **Geometry**: On synthetic ground-truth fixture, assert cameras/transforms and object silhouette/depth-order alignment against separately authored tolerances. Report occlusions and missing evidence instead of inventing exact geometry.
- **Scene identity**: At least three identifiable, separately editable objects; track identity stable across chosen keyframes; every mesh/object placed in the same coordinate convention as cameras.
- **Provenance**: Generated assets are not reclassified as observed; confidence and source references are retained; changing only one object invalidates only dependent artifact receipts.
- **Export**: Blender headless import and transform-edit smoke test; exported GLB and sidecar pass schema/path/hash checks; round-trip retains independent objects.
- **Robustness**: Deterministic resume after interruption; cache reuse; rejected provider evidence never bypasses Rust validation; missing providers yield explicit non-passing status, not a green acceptance result.
- **Visibility**: CI attaches before/after reference frames, rendered scene screenshot, Blender hierarchy/object inventory and a machine-readable quality report. No success declaration based solely on unit tests or mesh-quality submetrics.

## Priority and dependency policy
**P0 / critical path:** fixture & independently authored oracle; running build pipeline; provider interchange and one learned provider (if ordinary geometry fails); semantic inventory → cross-frame tracks → masks → object reference packages → object reconstruction/generation → placement/reprojection validation → assembled scene/viewer → Blender handoff → E2E acceptance command. Prefer the smallest **vertical slice** that produces the whole artifact, with honest quality limitations.

**P1 / quality upgrades:** learned matching escalation, hybrid depth refinement, stronger static world completion, improved seam/coverage on fixture failures, material/lighting polish, production-grade export.

**P2 / optional until P0 is green:** audio, scene expansion, cross-video fusion, dynamic effects, general splat optimization, broad mesh micro-optimization unrelated to a failing P0 gate.

Do not postpone correctness, safety, provenance or bounded-resource defects in existing PRs. Resolve those on their own merits, but do not treat closing such tasks as crossing the product acceptance gate.

## Work-loop selection
Before selecting a new issue, run the first whole-scene acceptance command (or use the latest pinned result); identify the **earliest failing dependency**. Prioritize an open issue unblocking that failure. Require the PR to show a changed end-to-end artifact or describe why a prerequisite is necessary. Track separate statuses for code merged, fixture passing and real video passing. The web GitHub Pages demo is a local preview, **not** proof that the native hybrid/Blender pipeline passes.
