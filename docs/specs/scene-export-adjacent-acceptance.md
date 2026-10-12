# Independent adjacent scene-export acceptance

These protected cases were authored retrospectively against the uncommitted first #84 production adapter, before its next repair. They supplement, and do not replace or weaken, the original ten cases.

`crates/video-to-3d-cli/tests/acceptance/scene_export_adjacent.rs` reuses the actual public ProjectStore and CLI harness. Source GLB edits preserve the existing BIN and repack only standard JSON/container lengths. Canonical resource hashes are recomputed, the canonical model must validate, and removal of the owned assembled artifact forces a real ProjectStore rebuild rather than reusing the old scene. No successful export, receipt or geometry outcome is fabricated.

1. Materialless canonical visuals and actual materialless GLB resources are valid. Export must succeed without inventing materials. Baseline: genuine RED, public CLI rejects `shared exporter omitted material table`.
2. A valid source GLB with an explicitly empty default scene and its mesh only in an alternate scene cannot silently become raw mesh geometry. This source layout must fail usefully before writing output unless the production adapter explicitly preserves that selection semantics. Baseline: genuine RED, CLI succeeds and exports alternate geometry.
3. Legitimate resource IDs `a` and `a-` replace two real mesh-resource IDs and all canonical visual references, leaving payloads and asset identities unchanged. Normalization must not invalidate portable resource-to-actual-mesh mappings. The unchanged independent receipt consumer verifies the actual generated bundle. Baseline: genuine RED, `resource mesh mapping` rejection.

Baseline log: `/tmp/video84-adjacent-all.log`, 9 existing cases GREEN, all 3 new cases RED, and the old portable Blender case unrun at its explicit environment setup boundary because `VIDEO_TO_3D_EXPORT_PORTABLE` was not set for this diagnostic invocation. The separately recorded original ten-case/Blender run remains distinct. No full quality-gate pass is claimed here.

Live conventions resolved and all 25 files read, sourceRevision `46d8793bb3034326561f876dcc67dbaa5aa1e432`. Repository AGENTS/CLAUDE and the relevant Decision Atlas export/ownership constraints apply.

The render harness currently frames every imported mesh, including the deliberately outlying completion sentinel. That is a visualization concern, not proof of incorrect exported geometry; the render harness remains unchanged. Original finishing/character experiments are unrelated to this export acceptance.
