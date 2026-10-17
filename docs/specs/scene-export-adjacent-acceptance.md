# Independent adjacent scene-export acceptance

These protected cases were authored retrospectively against the uncommitted first #84 production adapter, before its next repair. They supplement, and do not replace or weaken, the original ten cases.

`crates/video-to-3d-cli/tests/acceptance/scene_export_adjacent.rs` reuses the actual public ProjectStore and CLI harness. Source GLB edits preserve the existing BIN and repack only standard JSON/container lengths. Canonical resource hashes are recomputed, the canonical model must validate, and removal of the owned assembled artifact forces a real ProjectStore rebuild rather than reusing the old scene. No successful export, receipt or geometry outcome is fabricated.

1. Materialless canonical visuals and actual materialless GLB resources are valid. Export must succeed without inventing materials. Baseline: genuine RED, public CLI rejects `shared exporter omitted material table`.
2. A valid source GLB with an explicitly empty default scene and its mesh only in an alternate scene cannot silently become raw mesh geometry. This source layout must fail usefully before writing output unless the production adapter explicitly preserves that selection semantics. Baseline: genuine RED, CLI succeeds and exports alternate geometry.
3. Legitimate resource IDs `a` and `a-` replace two real mesh-resource IDs and all canonical visual references, leaving payloads and asset identities unchanged. Normalization must not invalidate portable resource-to-actual-mesh mappings. The unchanged independent receipt consumer verifies the actual generated bundle. Baseline: genuine RED, `resource mesh mapping` rejection.

Final baseline log: `/tmp/video84-adjacent-final-baseline.log`, **all 10 original cases GREEN and all 3 new cases genuinely RED** (0 ignored), using the actual prepared fixture and fresh `/tmp/video84-adjacent-fresh-portable` output directory. The separately recorded Blender98/material13 run remains distinct; no duplicate Blender run or full quality-gate pass is claimed. Earlier diagnostic `/tmp/video84-adjacent-all.log` lacked the old portable case's output environment setting; the final invocation resolves that setup omission.

Live conventions resolved and all 25 files read, sourceRevision `46d8793bb3034326561f876dcc67dbaa5aa1e432`. Repository AGENTS/CLAUDE and the relevant Decision Atlas export/ownership constraints apply.

The render harness currently frames every imported mesh, including the deliberately outlying completion sentinel. Actual existing import-report bounds give all-mesh X center2.825meters versus matched declared-inventory X center0; the outlying sentinel extends X maximum8.25 versus2.60. That is a visualization concern, not proof of incorrect exported geometry; the render harness remains unchanged. Original finishing/character experiments are unrelated to this export acceptance.

Reviewer visualization translation: the render call now frames only the already matched declared-inventory meshes. All imported objects remain present, all geometry/material/state and 98 structural/edit checks remain unchanged, and all-mesh sharing checks still include extra meshes. The completion sentinel remains exported and visible if inside the view; it no longer chooses the review camera/light bounds. This is not a production selection or geometry repair.

## Independent review-regression handoff, 2026-10-14

A separate acceptance context extends the public CLI contract against baseline
`d6cd7bd004e77281f0caf2daef152d5cc78f105c`, before the next production repair.
Atlas 1.3, 2.1–2.4 and 3.2–3.4, the public `AssembledScene` default-field contract,
and the existing protected-project ownership boundary govern these cases.

- Omitting the serde-default `materials` field from a valid materialless scene
  is equivalent to declaring an empty table. The existing explicit-empty case
  remains; a new omitted-field case shares its unchanged real resource setup.
  Baseline: RED, actual CLI reports `canonical material table is absent`.
- Export directories must not alias protected resource/artifact subtrees or
  contain the project manifest. The new test supplies valid portable manifest
  paths through the actual CLI and requires useful rejection with every project
  byte unchanged. Baseline: RED on Linux at `resources/export/scene.glb`, which
  the CLI accepts. Later artifact and manifest cases are unrun at this baseline
  because the first assertion stops the case. The same forward-slash manifest
  paths resolve to native Windows components; Windows execution is **unrun**.

Command: `VIDEO_TO_3D_EXPORT_FIXTURE=/home/moenarch/.local/share/coding-agent/video-84-fixture
python3 /home/moenarch/.local/share/coding-agent/disk-maintenance.py run --
cargo test --release --locked -p video-to-3d-cli --test scene_export -- --ignored
omitted_default_material_table_exports_without_inventing_materials
export_directory_cannot_alias_protected_project_subtrees`.
Result: 0 passed, 2 genuinely failed, 0 ignored, 13 filtered out. The disposable
target was cleaned. `cargo fmt --all` and `git diff --check` passed. Current shared
conventions sourceRevision: `5ad0db36974c8b0e5f9812c90ee34a4f227852c3`.
All fifteen explicit contracts plus portable Blender and the repository completion
gate remain required at the repaired head; these regression results do not prove
the three-real-video user journey.

## Independent sibling-export regression handoff

Before the next repair, a separate acceptance context tests baseline
`1d2ef15217b256b8d6912737c4c9d748ce5e832c` through the same actual public CLI.
Valid GLB `exports/scene.glb` and web `exports/web` targets may share their parent
while concrete outputs remain isolated. The success case requires a real GLB,
stable asset identity, valid portable receipt, unchanged sources and unchanged
pre-existing sibling web bytes. Baseline: genuine RED, CLI rejects the containing
`exports` directory. Late success assertions remain unrun at baseline.

A separate negative case retains protection when GLB output is nested under
the declared web subtree, or web output overlaps the portable resource subtree.
Both valid manifests are accepted by ProjectStore; actual CLI rejects them and
preserves every project byte. Baseline: GREEN. This supplements rather than
weakens resource/artifact/manifest directory protection. Windows execution
remains unrun. These two additions bring the explicit mandatory tier to 17.

Disposable release invocation with the existing prepared fixture and the two
test-name filters returned 1 pass, 1 genuine failure, 0 ignored, 15 filtered out.
Root acceptance author stops after committing these tests; a separate worker
repairs production. All 17 cases, portable Blender and applicable full gates
remain required after repair. No three-real-video success is inferred.

## Independent physical sibling-alias regression handoff

Review finding 4276217101 exposes a remaining ownership violation at baseline
`5204c9065ca13f4c7e01eb9b21c7224e3544b584`. A separate acceptance context
adds four actual public-CLI cases before the production repair: a sibling web
declaration physically aliases the GLB file, sidecar file, resource-copy
directory, or a missing child beneath an existing symlink into that directory.
All manifests validate through ProjectStore. Existing targets carry independent
web-owned bytes. Export must reject usefully before changing any project byte;
ordinary isolated siblings must continue passing the earlier positive contract.

These four cases use native Unix symlinks and are explicitly `cfg(unix)`.
Windows symlink execution remains unrun, not inferred from Linux. The complete
explicit Unix acceptance tier now has 21 cases. The fixtures remain synthetic
canonical scenes with real GLB resources; no reconstruction journey is claimed.

Baseline command uses the unchanged prepared fixture and disposable release
target: `cargo test --release --locked -p video-to-3d-cli --test scene_export --
--ignored sibling_resource_symlink_alias sibling_glb_symlink_alias
sibling_sidecar_symlink_alias sibling_missing_child_beneath_symlink`.
Result: **0 passed, 4 genuinely failed, 0 ignored, 17 filtered out**. Each actual
CLI wrongly succeeds; useful rejection and byte-preservation assertions remain
unrun at baseline. Log: `/tmp/video84-physical-alias-baseline.log`.
Shared convention sourceRevision remains
`5ad0db36974c8b0e5f9812c90ee34a4f227852c3`. `cargo fmt --all` and
`git diff --check` pass. The acceptance author stops after the test-only commit;
the distinct production worker must run all 21 cases and applicable full gates.
