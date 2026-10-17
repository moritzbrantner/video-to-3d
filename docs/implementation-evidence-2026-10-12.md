# Retrospective implementation evidence — 2026-10-12

This dated audit records existing narrow implementation claims and missing verification. Product intent remains in the [Decision Atlas](DECISION-ATLAS.md); these checkboxes describe implementation evidence, while Atlas approval checkboxes describe settled intent. This audit adds no Decisions or Questions. Referenced historical checks were not rerun by this documentation repair. Synthetic checks and merged PRs do not establish the three-real-video scene-to-Blender release gate.

## Atlas 1.6 Normal processing is automatic.

- [x] The **bounded native geometric fixture pipeline** runs camera, dense, mesh and assembled-environment stages through the existing receipt-backed CLI/store without manual intermediate calls ([merged #174](https://github.com/moritzbrantner/video-to-3d/pull/174), `crates/video-to-3d-core/src/scene_artifacts/native.rs::BuiltInExecutor`, `crates/video-to-3d-cli/tests/native_build.rs`).

- [ ] Full three-real-video reconstruction into editable scene objects without agent orchestration remains unverified ([#150](https://github.com/moritzbrantner/video-to-3d/issues/150), [#77](https://github.com/moritzbrantner/video-to-3d/issues/77)).

## Atlas 2.1 Rust owns reusable reconstruction semantics.

- [x] Public `reconstruct_cameras`, `reconstruct_dense`, `reconstruct_mesh` and `compose_reconstruction` exist at `crates/video-to-3d-core/src/reconstruction_stages.rs::{reconstruct_cameras,reconstruct_dense,reconstruct_mesh,compose_reconstruction}`; versioned replay/parity tests are in `crates/video-to-3d-core/tests/reconstruction_stages.rs` ([merged #173](https://github.com/moritzbrantner/video-to-3d/pull/173)).

## Atlas 3.1 The initial release gate uses three real videos.

- [x] Two approved handheld inputs (H1/H3) have source attribution, hashes and fixture-preparation code; exact manifest/candidate review is protected by `tools/check_real_video_manifest.py` and `tools/tests/test_approved_inventory_immutability.py` ([merged #177](https://github.com/moritzbrantner/video-to-3d/pull/177)).
- [ ] Independent browser decoding acceptance for both prepared H1/H3 carriers remains unverified. The recorded pre-repair run passed H1 and failed H3; the required post-repair two-carrier browser tier has not been independently demonstrated. See `specs/approved-handheld-fixture-acceptance.md` and `tools/tests/test_approved_handheld_acceptance.py`.

- [ ] The approved controlled-room TUM RGB video is not yet established as a reproducibly prepared real-video input fixture; source selection in Atlas 3.8 is a **decision**, not implemented footage-processing or independently passing reconstruction ([#74](https://github.com/moritzbrantner/video-to-3d/issues/74)).

- [ ] An independent whole-scene acceptance run across all three real video sources and actual Blender outputs is still required ([#74](https://github.com/moritzbrantner/video-to-3d/issues/74), [#76](https://github.com/moritzbrantner/video-to-3d/issues/76), [#77](https://github.com/moritzbrantner/video-to-3d/issues/77)).

## Atlas 3.3 Blender import/edit is part of product acceptance.

- [x] A reusable strict headless Blender **acceptance harness** is implemented in `tools/blender/scene_import_acceptance.py` and `tools/blender/self_test.sh`, with synthetic truth/negative controls ([merged #168](https://github.com/moritzbrantner/video-to-3d/pull/168)).

- [ ] Portable canonical-scene GLB exporter remains in unfinished draft [PR #175](https://github.com/moritzbrantner/video-to-3d/pull/175); its synthetic acceptance does not establish independent editable objects from all three real videos ([#84](https://github.com/moritzbrantner/video-to-3d/issues/84), [#151](https://github.com/moritzbrantner/video-to-3d/issues/151)).

## Atlas 3.5 Synthetic exact-truth fixtures are verification oracles, not release substitutes.

- [x] Deterministic synthetic room exact-truth fixture generator exists at `crates/video-to-3d-core/examples/synthetic_room_fixture.rs`, with Blender truth generation `tools/blender/build_truth_scene.py` ([merged #166](https://github.com/moritzbrantner/video-to-3d/pull/166)).

- [ ] Passing a synthetic scene alone does not close the three-video, separately editable object release gate ([#77](https://github.com/moritzbrantner/video-to-3d/issues/77)).

## Atlas 3.6 Product acceptance is end-to-end.

- [x] Bounded reconstruction stages, native synthetic smoke, fixture provenance and Blender self-test are independently available ([PR #173](https://github.com/moritzbrantner/video-to-3d/pull/173), [#174](https://github.com/moritzbrantner/video-to-3d/pull/174), [#168](https://github.com/moritzbrantner/video-to-3d/pull/168)).

- [ ] **Actual product exit is still unverified:** real three-clip processing, object separation, canonical portable Blender export and independent edit validation ([#76](https://github.com/moritzbrantner/video-to-3d/issues/76), [#77](https://github.com/moritzbrantner/video-to-3d/issues/77)).
