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


## Supplemental mixed appearance safety before export implementation

Atlas1.2/2.3/3.4 and hybrid acceptance distinguish observed geometry from invented
geometry and preserve provenance in both modes. They do not settle exact product
presentation of generated appearance on observed geometry. The supplemental eighth
explicit test is a technical fail-closed boundary, not a new appearance decision:
real unchanged75 reconstructed observed-smoke geometry receives explicitly labeled
generative material factors through the canonical input fixture. Its real fallback
mesh has no UVs; this input removes that material's texture binding rather than
inventing a UV layout, inferred geometry or a textured reconstruction. Other seven
fixtures/expectations are unchanged.

Observed export may succeed only by retaining actual observed geometry and the
explicit material factors/identity, excluding invented completion geometry, and
preserving full original canonical material/resource/asset/camera provenance
unchanged. It may instead reject mixed appearance with a useful unsupported
capability explanation and no apparently valid GLB. It cannot silently omit the
observed asset, substitute appearance or relabel it as observed. This does not
mandate all visual provenance satisfy is_observed_geometry and does not require
generated textures. Exact product presentation/full feature acceptance remains
unresolved by this technical test.

Honest baseline at d5530a5: the supplemental fixture genuinely reconstructs75
frames, validates its canonical scene and both provenance classes, then the actual
CLI rejects targetglb before adaptation. The test is RED at missing CLI capability;
all successful output/late metadata assertions and useful mixed-appearance rejection
are UNRUN. It is not a reproduced material-selection bug in implemented export.
Log: /tmp/video84-mixed-appearance-baseline-final.log.

Sidecar schema, final GLB hash and resource/node mapping integrity still require
independent direct audit. Current flexible sidecar discovery strips its envelope;
no existing production export receipt shape is available to validate or positively
prove tampered final output rejection. Do not bless arbitrary hash/version fields
or invent proof from a manufactured final bundle. The minimal independently authored consumer representation below now establishes
that verifier before production. The original mandatory ownership/large-buffer
audit and actual Blender gate remain controlling.


## Portable export receipt consumer v1 (independent supplemental interface)

A sidecar retains the complete unchanged canonical scene at its root or under
`scene`. A sibling `export_receipt` is a narrowly scoped interchange receipt;
it is not another ProjectStore artifact ledger or reconstruction authority.
Names may be translated independently before producer implementation, while
consumer semantic assertions remain fixed. Proposed interoperable shape:

```json
{
  "scene": { "schema_version": 1, "frame": {}, "resources": [], "assets": [] },
  "export_receipt": {
    "schema_version": 1,
    "target": "glb",
    "scene_mode": "completed",
    "output": { "path": "scene.glb", "content_hash": "...", "byte_length": 123 },
    "assets": {
      "canonical-id": { "node_index": 0, "selection": "visual" }
    },
    "resources": {
      "canonical-resource-id": {
        "path": "resources/copied-original.glb",
        "content_hash": "...",
        "byte_length": 123,
        "mesh_indices": [0],
        "material_indices": []
      }
    }
  }
}
```

The example omits real required scene fields; it describes receipt shape only.
Output/resource paths resolve against the portable export directory, not the
original project. Receipt output hash/length cover FINAL GLB bytes after extras
annotation. Asset mappings identify actual stable-ID node extras exactly, include
required ancestors, retain actual parent edges and declare selection as visual,
ancestor or collision. Completed mode includes every canonical mesh visual;
observed selection remains covered independently. Mesh resource mappings name
actual selected GLB mesh indices and their material indices; texture resources
map exported canonical materials that actually reference them. Index sets are
exact, not arbitrary in-range suggestions. All canonical resources (including
collision-only payloads) are copied as original bytes with unchanged source
hashes and measured lengths. These mappings supplement, never overwrite,
canonical source paths/provenance. Copy-all is a permissible simple portable
closure; this does not invent source geometry or material normalization.

The test-owned consumer checks version/target/mode, final GLB length/hash,
canonical node identity/parent/selection/material bindings and portable original
resource hash/length/index closure. A new actual production-output test first
requires unchanged entire canonical metadata, then verifies the receipt. Its
baseline fails at unsupported CLI GLB target; all final-output checks remain
UNRUN. A separate self-test uses an actual prepared crate GLB payload with
independently authored two-node metadata around its unchanged BIN, plus valid
canonical source input. It passes and rejects wrong version, node map, corrupted
resource bytes and changed node identity EVEN WHEN final hash/length are updated
self-consistently. This is verifier harness proof, not a production export,
coordinate/material normalization claim, or reconstructed semantic inventory.

Baseline receipts: /tmp/video84-receipt-baseline.log. Existing seven contracts
remain intact; explicit tier now has original7 + mixed appearance + actual receipt
+ verifier self-test (10 tests). All ten run through the existing mandatory
export_acceptance.sh --ignored invocation; the Blender gate still consumes only
actual CLI portable output, never the verifier-only authored container.


Final supplemental pre-production gate: explicit tier2PASS/8RED (original real
input test and new verifier harness pass; eight actual CLI-dependent cases stop
at missingGLB export capability), strict focused release Clippy/workspacefmt and
owned whitespace PASS. Log /tmp/video84-supplement-all-baseline-final.log and
/tmp/video84-supplement-clippy.log. No production-export receipt/Blender/presentation
success is claimed. Original seven test bodies remain unchanged; supplemental
fixture factory is gated off for every original call.
