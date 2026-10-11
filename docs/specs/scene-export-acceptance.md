# Independent canonical scene export acceptance (#84)

Authority: issue #84, `AGENTS.md`, whole-scene product decisions and
hybrid-room-to-Blender acceptance. Baseline `301d393`; this contract is authored
before export production changes in a context separate from implementation.

## Bounded outcome

`video-to-3d export project.json --target glb` consumes a verified canonical
`AssembledScene` through existing manifest/receipt boundaries and writes the
manifest-declared GLB path with a portable provenance/collider bundle. This is
an export adapter, not a second reconstruction provider or executor. The
existing `ExportFormat::Glb` declaration owns the format; no speculative Rust
serializer API is selected by these tests.

`--scene-mode observed|completed` selects the owner-required scene views. Completed
preserves explicitly labeled generative resources; observed excludes them and
retains genuinely observed geometric evidence under the existing canonical
classification. Authored content is not reclassified as observed. No completion is relabeled as filmed evidence.
Mesh remains the editable default; unsupported splat companionship must remain
explicit rather than silently being lost. No external execution or uploads.

GLB node extras retain each canonical asset ID as `video_to_3d_id`; node/material
names remain identifiable. Parent hierarchy and local transforms preserve the
canonical right-handed Y-up frame exactly once. Do not apply the flat legacy
writer's camera-frame flip again to already canonical data. A valid glTF matrix
or TRS representation is accepted. Blender world bounds use the independent
synthetic truth's authored 1 mm tolerance, not an implementation-derived oracle.

A versioned sidecar retains canonical source scene fields, resource hashes/paths,
confidence, source-frame references, semantic labels, provenance and collision
references unchanged, plus a versioned export hash and resource-to-export-node
mapping. The JSON filename/envelope may vary: tests discover canonical metadata
at the root or in a `scene` envelope. Geometry belongs in the GLB/resources, not
large JSON buffers. Export resource mappings must make the bundle usable after
the original project/provider workspace is absent; source paths are provenance,
not permission to read outside the verified project root. A copy of referenced
resources is permissible when required for interchange. Required payloads must
be embedded or portable; network URLs, missing/tampered resources and symlink
escapes fail closed. Asset-tooling/3d-lab's established production normalization
and validation paths own mesh/material validation; do not replace them with a
second private validator.

## Independent inputs and executable gates

`prepare_export_acceptance.py` reads unchanged #75 PLY truth through the existing
pure `build_truth_scene.read_ply` helper. Blender encodes each actual local mesh as
an input GLB; it does **not** export the final canonical scene. Known truth TRS is
converted to the canonical frame. A rotated/translated collision-only room group
parents the original 12 elements without changing their world placement. Real UVs,
a small checker PNG and material factors exercise material/texture export. A
clearly labeled completion sentinel uses a separately stored copy of real crate
geometry; it is not an inferred object or an addition to the product inventory.
The main hierarchy oracle is authored synthetic input, never filmed output. The
observation-mode case additionally reconstructs the unchanged original #75 sampled
frames through the existing canonical core and flat surface encoder, producing a
real geometric-multi-view environment resource with actual accepted cameras/source
frames. That case proves selection retains observed evidence and excludes generated
content; it makes no claim of semantic decomposition or real-video product success.
Revalidated completion remains an observed class under existing model semantics;
this fixture does not fabricate a revalidation pipeline.

`scene_export.rs` imports this prepared canonical scene via the existing public
`ProjectStore` test executor seam and records legitimate receipts. It does not
claim native ingest/reconstruction/decomposition passes. Seven explicit acceptance
tests cover valid real resources, exact asset IDs/hierarchy/local transforms,
materials and embedded images, deterministic export and unchanged source resources,
canonical provenance/collision preservation, missing/tampered payloads, symlink
escape, observation/completed selection and workspace-independent portability.
These tests are explicitly ignored by ordinary unit runs because they require
prepared Blender inputs; **the explicit acceptance command is a mandatory gate**,
not optional evidence or an ordinary-suite pass.

```bash
tools/blender/export_acceptance.sh /path/to/fresh-evidence /path/to/blender
```

Use Blender 5.0.1 with the existing workflow's published archive SHA-256
`8019580ee1b7262e505f4196a00237ccf743c88d205b38d34201510676e60b09`.
The command generates original truth, prepares resource inputs, runs the actual
CLI integration contract, removes the exported project's original workspace,
then invokes the #151 harness on **production export output** with independently
derived #75 bounds/inventory. It leaves the report and canonical render for review. A separate Blender check compares imported checker pixels and material/texture links with the original input oracle; actual product-output evidence remains pending until implementation.

The harness's optional fixture-owned `expected_parent_ids` verifies exact canonical
grouping and world-space independent edits. Its absent-field strict behavior stays
unchanged. Existing negative self-tests still reject joined islands, missing nodes,
object/architecture coupling and shared mesh data. Grouped reference inputs pass
only with the correct parent map; wrong expected parent fails. They validate the
harness and do not count as product-export acceptance.

## Remaining review and completion gates

Before merge, run this explicit acceptance, `tools/blender/self_test.sh`, existing
scene-model/surface-GLB/collider/CLI contracts and the full applicable repository
suite. Tool inspection reports undeclared relationships, so a focused selection
cannot establish completion. Required gate remains `bun run check`; document any
unavailable environment separately. Resolved shared sourceRevision:
`46d8793bb3034326561f876dcc67dbaa5aa1e432`.

Direct review must verify sidecar schema/export hashes/resource-node mappings,
portable dependency closure, provider/geometry ownership, standard production
normalization and large-buffer reuse. The initial tests preserve canonical metadata
but intentionally do not prescribe a new mapping serializer; review must not claim
that schema integrity or normalization integration is already automatically proven.

This draft references #84. Full #84 closure requires all approved scope to be
verified, including portable Blender import and trustworthy metadata. The three
pinned real clips and production semantic inventories remain #74/#151/#77; this
synthetic export fixture cannot close those product gates. Native assembly #150
is an independent prerequisite, and its flat environment GLB does not substitute
for this canonical scene adapter.
