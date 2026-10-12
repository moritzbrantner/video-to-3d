# Overlap clipping and derived-vertex acceptance (#137)

This supplemental contract is authored against existing PR #171 after its existing-vertex implementation, before the next repair. It preserves the previous seam contract and all reference-stage gates. The owner selected source-triangle and barycentric provenance for clipped overlap seams in [the #137 decision](https://github.com/moritzbrantner/video-to-3d/issues/137#issuecomment-6113239055). This is a reconstruction contract, not approval of a new geometry provider or a generic clipping library.

## Observable geometry

An independently authored pair of linked, observed planar patches has an overlap boundary crossing the interior of source triangles. Their union is known from their input geometry. Both constant-depth and sloped-depth cases must yield one edge-conforming connected surface with consistent winding, no positive-area double coverage, no unsupported extension, and coverage of the observed union. A vertex lying inside another retained edge without splitting that edge is a T-junction and does not establish an edge-conforming zipper. Connectivity alone, or filling a bounding rectangle outside the source union, is insufficient.

The existing gap, depth disagreement, unlinked-patch, evidence-origin separation, deterministic ordering and bounded-resource tests remain applicable. Existing seams satisfiable using retained vertices retain their original no-new-points expectation. Clipping cannot invent a bridge where no source triangle observed the gap.

## Geometric lineage

Each newly derived vertex has a resolvable immutable pre-clip source triangle and finite barycentric weights inside that triangle, summing to one within numerical tolerance. Its world position must reconstruct from those source corners. Sloped-depth tests independently distinguish world-space barycentric reconstruction from an incorrect linear interpolation of projected image coordinates/depth. Input buffers are not rewritten merely to make the lineage assertion pass.

Derived geometry retains the source evidence's provider, coordinate/scale, camera support and provenance classification. A derived vertex is not a new independent multi-view observation; deriving it does not increase observation/support counts or turn completion into observed evidence. Representation and routine API names are flexible. References must resolve to immutable parents; invalid indices, nonfinite/outside weights, inconsistent positions and cycles must fail closed. Sparse lineage must not require a duplicate full original point/triangle buffer.

## Existing consumer boundaries

The validated evidence view, surface-mesh artifact JSON round trip and stored mesh resumption must preserve geometric lineage and reject malformed lineage. Texture/material reprojection must resolve the same original source coordinates, including clipped vertices; fabricated integer grid sites cannot disguise missing source correspondence. The GLB export must preserve the accepted geometry and classification. The existing owned-dense no-duplication contract remains applicable when no append is needed. The new owned append boundary must not introduce a second reconstruction executor or provider geometry authority.

## Full reference gate

The existing `overlapping-references` fixture retains `mesh_components <= 40` and all sampling, feature, seed, registration, dense, coverage, unsupported/duplicate-area and texture checks. The retained COLMAP report currently observes 115 components and therefore fails; passing a small synthetic seam oracle does not close this gate. The separate #138 expected failures remain unchanged. No filmed-scene, Blender, object-inventory or full #77 product result follows from this synthetic seam contract.

## Source and chronology

The baseline PR head is `ac262c86b64f958d9423dc5c6198ba3b88e5b39c`; mechanical integration of current main `3861cd194ecc21946d3dc36dcb137e3b28805cae` produced `15718de65e29ab68b92eb4281eee0283c825aab4` without changing the protected existing `surface_merge_tests.rs`. Shared policy resolves to `46d8793bb3034326561f876dcc67dbaa5aa1e432`. Existing video reconstruction image ownership already contains bounded polygon splitting; inspected 3d-lab core/spatial APIs do not expose a clipping/lineage primitive. This contract specifies outputs and integrity rather than selecting a new speculative algorithm. Executable supplemental test discovery and routine seam translation are recorded separately before production repair.

The integrated baseline passes 393 core tests and 18 reference-fixture example tests. The first additive partial-overlap witness compiles and fails on a real positive-area overlap of `0.00006956521301954607` between an owner triangle and a retained lower triangle. Its accepted fusion-link precondition passes; later coverage/connectivity assertions are unexecuted after that overlap failure. This witness initially exercises the existing borrowed merge seam; its output projection must be translated independently to an owned derived-geometry result when that routine seam is established. It is not yet a complete executable lineage/artifact/texture contract and is not a production-ready handoff.

The owned-result flat and sloped oracle now requires input-capacity reuse, original-coordinate preservation, one sparse lineage record per appended point, immutable source-triangle/world-barycentric reconstruction, independent source-union area, no overlapping output, one component and no T-junction. Against the baseline it is compile-RED solely because the owned merge capability does not exist (`E0425`, `merge_fused_patches_owned`); no geometric runtime assertion has run through that new seam yet. The proposed names/packing remain independently translatable. Source-triangle identities must remain resolvable after source triangles are clipped/removed, through sparse immutable parent records rather than serializing a second full original mesh. Artifact/material/store consumer tests remain pending.
