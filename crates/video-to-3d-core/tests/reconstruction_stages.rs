//! Acceptance contract of #172: public staged reconstruction entry points with
//! versioned stage artifacts.
//!
//! The classic pipeline is split into three public stages, each producing its own
//! versioned, serializable artifact that records the content hash of its inputs:
//!
//! ```text
//! reconstruct_cameras(&request)                    -> CameraStageArtifact
//! reconstruct_dense(&request, &cameras)            -> DenseStageArtifact
//! reconstruct_mesh(&cameras, &dense)               -> MeshStageArtifact
//! compose_reconstruction(&cameras, &dense, &mesh)  -> ReconstructionResult
//! ```
//!
//! `reconstruct` and `reconstruct_browser` keep producing exactly the same result;
//! these tests pin that by comparing the composed stages with the single call on
//! deterministic multi-view clips. The browser contract tests in
//! `src/browser_contract.rs` are deliberately not touched.
//!
//! Until #172 lands, `src/reconstruction_stages.rs` holds compiling stubs
//! (`// contract: implemented by #172`) that return `NotImplemented`, so every
//! test below except `stage_fixtures_exercise_the_single_call_pipeline` fails.

use serde_json::Value;
use video_to_3d_core::scene_project::ContentHash;
use video_to_3d_core::{
    compose_reconstruction, reconstruct, reconstruct_cameras, reconstruct_dense, reconstruct_mesh,
    request_content_hash, CameraStageArtifact, DenseStageArtifact, FrameInput, MeshStageArtifact,
    ReconstructionOptions, ReconstructionRequest, ReconstructionResult, ReconstructionStage,
    ReconstructionStageError, StageInput, CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION,
    DENSE_STAGE_ARTIFACT_SCHEMA_VERSION, MESH_STAGE_ARTIFACT_SCHEMA_VERSION,
};

// ---------------------------------------------------------------------------
// Deterministic fixtures: the lateral point-scene renderer of the in-crate
// reconstruction tests (`src/reconstruction.rs`, `scene_points`/`render_scene`).
// ---------------------------------------------------------------------------

const SCENE_WIDTH: u32 = 640;
const SCENE_HEIGHT: u32 = 480;
const SCENE_FOCAL: f64 = 520.0;
const UNKNOWN_SCHEMA_VERSION: u32 = 9_999;
const RECOVERY_WARNING: &str = "The Rust core selected";

fn scene_points(seed: u64) -> Vec<([f64; 3], u8)> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1_u64 << 53) as f64)
    };
    (0..420)
        .map(|index| {
            let point = [
                (next() - 0.5) * 5.2,
                (next() - 0.5) * 3.5,
                4.0 + next() * 6.0,
            ];
            (point, 35 + ((index * 47) % 190) as u8)
        })
        .collect()
}

/// One frame of a lateral camera move through `points`, on a flat `background`.
fn render_scene(points: &[([f64; 3], u8)], camera_x: f64, background: u8) -> FrameInput {
    let (width, height) = (SCENE_WIDTH, SCENE_HEIGHT);
    let mut rgba = vec![background; (width * height * 4) as usize];
    for pixel in rgba.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    for (index, (point, shade)) in points.iter().enumerate() {
        let (x, z) = (point[0] - camera_x, point[2]);
        if z <= 0.5 {
            continue;
        }
        let px = (x / z * SCENE_FOCAL + width as f64 * 0.5).round() as i32;
        let py = (point[1] / z * SCENE_FOCAL + height as f64 * 0.5).round() as i32;
        let radius = 3 + (index % 2) as i32;
        if px < radius + 1
            || py < radius + 1
            || px >= width as i32 - radius - 1
            || py >= height as i32 - radius - 1
        {
            continue;
        }
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let offset = (((py + dy) as u32 * width + (px + dx) as u32) * 4) as usize;
                let value = if ((dx + dy + index as i32) & 1) == 0 {
                    *shade
                } else {
                    255_u8.saturating_sub(shade / 2)
                };
                rgba[offset] = value;
                rgba[offset + 1] = value.saturating_add((index % 17) as u8);
                rgba[offset + 2] = value.saturating_sub((index % 13) as u8);
            }
        }
    }
    FrameInput {
        width,
        height,
        rgba,
    }
}

fn lateral_span(points: &[([f64; 3], u8)], background: u8) -> Vec<FrameInput> {
    (0..6)
        .map(|frame| render_scene(points, -0.42 + frame as f64 * 0.12, background))
        .collect()
}

fn scene_request(frames: Vec<FrameInput>) -> ReconstructionRequest {
    ReconstructionRequest {
        frames,
        options: ReconstructionOptions {
            min_feature_distance: 6,
            match_radius: 80,
            focal_length_pixels: Some(SCENE_FOCAL as f32),
            ..ReconstructionOptions::default()
        },
    }
}

/// Six-frame lateral clip: calibrated seed, registered views, accepted dense depth
/// and a non-empty mesh (the in-crate analogue of the golden lateral case).
fn lateral_request() -> ReconstructionRequest {
    scene_request(lateral_span(&scene_points(0x5eed), 236))
}

/// Another scene of the same shape: a different request, same dimensions/options.
fn other_lateral_request() -> ReconstructionRequest {
    scene_request(lateral_span(&scene_points(0xbadc0de), 236))
}

/// A lateral clip followed by a hard cut into a wide pan: the first solve leaves the
/// secondary segment unseeded and `reconstruct` selects a bounded recovery retry with
/// different effective options (in-crate `wide_pan_split_request`).
fn recovery_request() -> ReconstructionRequest {
    let mut frames = lateral_span(&scene_points(0x5eed), 236);
    let wide = scene_points(0xc0ffee);
    frames.extend((0..6).map(|frame| render_scene(&wide, -0.42 + frame as f64 * 1.2, 64)));
    scene_request(frames)
}

/// Two flat frames: no calibrated seed, the conservative uncalibrated preview, no
/// dense points and no mesh.
fn uncalibrated_request() -> ReconstructionRequest {
    let flat = FrameInput {
        width: 48,
        height: 32,
        rgba: [120, 120, 120, 255].repeat(48 * 32),
    };
    ReconstructionRequest {
        frames: vec![flat.clone(), flat],
        options: ReconstructionOptions::default(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct Chain {
    cameras: CameraStageArtifact,
    dense: DenseStageArtifact,
    mesh: MeshStageArtifact,
}

fn run_stages(request: &ReconstructionRequest) -> Chain {
    let cameras = reconstruct_cameras(request).expect("camera stage");
    let dense = reconstruct_dense(request, &cameras).expect("dense stage");
    let mesh = reconstruct_mesh(&cameras, &dense).expect("mesh stage");
    Chain {
        cameras,
        dense,
        mesh,
    }
}

/// `ReconstructionResult` has no `PartialEq`; its serialized form is the observable
/// result (the browser and WASM adapter ship exactly this), so equality of the JSON
/// trees is exact equality of every field, f32 values included.
fn result_json(result: &ReconstructionResult) -> Value {
    serde_json::to_value(result).expect("reconstruction result serializes")
}

fn assert_same_result(composed: &ReconstructionResult, single: &ReconstructionResult) {
    // Field-level checks first, for a readable failure.
    assert_eq!(composed.cameras.len(), single.cameras.len(), "cameras");
    assert_eq!(composed.points, single.points, "sparse points");
    assert_eq!(composed.dense_points, single.dense_points, "dense points");
    assert_eq!(
        composed.mesh_triangles.len(),
        single.mesh_triangles.len(),
        "mesh triangles"
    );
    assert_eq!(
        composed.warnings, single.warnings,
        "warnings and their order"
    );
    let (composed, single) = (result_json(composed), result_json(single));
    if composed != single {
        let Value::Object(composed) = composed else {
            unreachable!()
        };
        let Value::Object(single) = single else {
            unreachable!()
        };
        let differing: Vec<_> = single
            .keys()
            .chain(composed.keys())
            .filter(|key| composed.get(*key) != single.get(*key))
            .collect();
        panic!("composed result differs from reconstruct() in fields {differing:?}");
    }
}

fn assert_valid_hash(hash: &ContentHash) {
    // `ContentHash` already enforces `sha256:` + 64 lowercase hex digits.
    assert!(hash.as_str().starts_with("sha256:"), "{}", hash.as_str());
}

/// The document with its top-level `schema_version` replaced.
fn with_schema_version(document: &str, version: Value) -> String {
    let mut value: Value = serde_json::from_str(document).expect("artifact document is JSON");
    value["schema_version"] = version;
    serde_json::to_string(&value).unwrap()
}

/// The document without its top-level `schema_version`.
fn without_schema_version(document: &str) -> String {
    let mut value: Value = serde_json::from_str(document).expect("artifact document is JSON");
    value
        .as_object_mut()
        .expect("artifact documents are objects")
        .remove("schema_version");
    serde_json::to_string(&value).unwrap()
}

const IDENTITY_FIELDS: [&str; 4] = [
    "schema_version",
    "request_hash",
    "cameras_hash",
    "dense_hash",
];

/// Edit the first non-zero number of the artifact's own payload (never its identity
/// fields), as a corrupted or hand-edited store would. Panics if the artifact owns no
/// numeric payload, which a stage artifact always does.
fn edit_payload(document: &str) -> String {
    fn nudge(value: &mut Value) -> bool {
        match value {
            Value::Number(number) => {
                let edited = if let Some(unsigned) = number.as_u64() {
                    if unsigned == 0 {
                        return false;
                    }
                    Value::from(unsigned + 1)
                } else if let Some(float) = number.as_f64() {
                    if float == 0.0 {
                        return false;
                    }
                    Value::from(float * 0.5)
                } else {
                    return false;
                };
                *value = edited;
                true
            }
            Value::Array(items) => items.iter_mut().any(nudge),
            Value::Object(fields) => fields.values_mut().any(nudge),
            _ => false,
        }
    }
    let mut value: Value = serde_json::from_str(document).expect("artifact document is JSON");
    let edited = value
        .as_object_mut()
        .expect("artifact documents are objects")
        .iter_mut()
        .filter(|(key, _)| !IDENTITY_FIELDS.contains(&key.as_str()))
        .any(|(_, field)| nudge(field));
    assert!(
        edited,
        "a stage artifact owns a numeric payload: {document:.200}"
    );
    serde_json::to_string(&value).unwrap()
}

fn assert_mismatch(
    error: ReconstructionStageError,
    expected_stage: ReconstructionStage,
    expected_input: StageInput,
) {
    match error {
        ReconstructionStageError::InputMismatch {
            stage,
            input,
            recorded,
            actual,
        } => {
            assert_eq!((stage, input), (expected_stage, expected_input));
            assert_ne!(
                recorded, actual,
                "a mismatch names two different identities"
            );
        }
        other => {
            panic!("expected InputMismatch({expected_stage:?}, {expected_input:?}), got {other:?}")
        }
    }
}

fn assert_unsupported(
    error: ReconstructionStageError,
    expected_artifact: ReconstructionStage,
    expected_supported: u32,
) {
    assert_eq!(
        error,
        ReconstructionStageError::UnsupportedSchemaVersion {
            artifact: expected_artifact,
            found: UNKNOWN_SCHEMA_VERSION,
            supported: expected_supported,
        }
    );
}

// ---------------------------------------------------------------------------
// Fixture sanity (passes before #172: it uses only the existing single call)
// ---------------------------------------------------------------------------

#[test]
fn stage_fixtures_exercise_the_single_call_pipeline() {
    let lateral = reconstruct(&lateral_request()).expect("lateral reconstruction");
    assert!(lateral.calibrated_pair.is_some());
    assert!(!lateral.registered_views.is_empty());
    assert!(!lateral.dense_points.is_empty());
    assert!(!lateral.mesh_triangles.is_empty(), "lateral clip meshes");

    let recovery = reconstruct(&recovery_request()).expect("recovery reconstruction");
    assert!(
        recovery
            .warnings
            .iter()
            .any(|warning| warning.starts_with(RECOVERY_WARNING)),
        "the recovery fixture selects a retry: {:?}",
        recovery.warnings
    );
    assert!(!recovery.mesh_triangles.is_empty(), "recovered clip meshes");

    let uncalibrated = reconstruct(&uncalibrated_request()).expect("uncalibrated preview");
    assert!(uncalibrated.calibrated_pair.is_none());
    assert!(uncalibrated.mesh_triangles.is_empty());
}

// ---------------------------------------------------------------------------
// (a) Equivalence with the single call
// ---------------------------------------------------------------------------

#[test]
fn composed_stages_equal_the_single_call_on_a_lateral_clip() {
    let request = lateral_request();
    let chain = run_stages(&request);
    let composed = compose_reconstruction(&chain.cameras, &chain.dense, &chain.mesh)
        .expect("compose one chain");
    let single = reconstruct(&request).expect("single call");
    assert!(!single.mesh_triangles.is_empty());
    assert_same_result(&composed, &single);
}

/// Recovery ranks sparse evidence only, so the camera stage selects the same solve;
/// the dense stage must then use that solve's effective options, not the request's.
#[test]
fn composed_stages_equal_the_single_call_after_registration_recovery() {
    let request = recovery_request();
    let chain = run_stages(&request);
    let composed = compose_reconstruction(&chain.cameras, &chain.dense, &chain.mesh)
        .expect("compose one chain");
    let single = reconstruct(&request).expect("single call");
    assert_same_result(&composed, &single);
}

#[test]
fn composed_stages_equal_the_single_call_without_a_calibrated_seed() {
    let request = uncalibrated_request();
    let chain = run_stages(&request);
    let composed = compose_reconstruction(&chain.cameras, &chain.dense, &chain.mesh)
        .expect("an uncalibrated chain still composes");
    let single = reconstruct(&request).expect("single call");
    assert_same_result(&composed, &single);
}

#[test]
fn camera_stage_reports_the_single_call_request_error() {
    let mut request = uncalibrated_request();
    request.frames[1].rgba.pop();
    let single = reconstruct(&request).expect_err("malformed RGBA is rejected");
    let staged = reconstruct_cameras(&request).expect_err("malformed RGBA is rejected");
    assert_eq!(staged, ReconstructionStageError::Reconstruction(single));
}

// ---------------------------------------------------------------------------
// Identity and determinism
// ---------------------------------------------------------------------------

#[test]
fn stage_artifacts_record_their_schema_and_input_identities() {
    let request = lateral_request();
    let chain = run_stages(&request);
    let request_hash = request_content_hash(&request);
    assert_valid_hash(&request_hash);

    assert_eq!(
        chain.cameras.schema_version,
        CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION
    );
    assert_eq!(chain.cameras.request_hash, request_hash);

    assert_eq!(
        chain.dense.schema_version,
        DENSE_STAGE_ARTIFACT_SCHEMA_VERSION
    );
    assert_eq!(chain.dense.request_hash, request_hash);
    assert_eq!(chain.dense.cameras_hash, chain.cameras.content_hash());

    assert_eq!(
        chain.mesh.schema_version,
        MESH_STAGE_ARTIFACT_SCHEMA_VERSION
    );
    assert_eq!(chain.mesh.dense_hash, chain.dense.content_hash());

    // The identity is the hash of the stored document, so the scene store's recorded
    // artifact hash and the downstream record agree.
    for (hash, document) in [
        (chain.cameras.content_hash(), chain.cameras.to_json()),
        (chain.dense.content_hash(), chain.dense.to_json()),
        (chain.mesh.content_hash(), chain.mesh.to_json()),
    ] {
        assert_valid_hash(&hash);
        assert_eq!(hash, ContentHash::of_bytes(document.as_bytes()));
    }
    assert_ne!(chain.cameras.content_hash(), chain.dense.content_hash());
    assert_ne!(chain.dense.content_hash(), chain.mesh.content_hash());
}

#[test]
fn stages_are_deterministic() {
    let request = lateral_request();
    let (first, second) = (run_stages(&request), run_stages(&request));
    assert_eq!(first.cameras, second.cameras);
    assert_eq!(first.dense, second.dense);
    assert_eq!(first.mesh, second.mesh);
    assert_eq!(first.cameras.to_json(), second.cameras.to_json());
    assert_eq!(first.dense.to_json(), second.dense.to_json());
    assert_eq!(first.mesh.to_json(), second.mesh.to_json());
}

#[test]
fn request_identity_covers_frames_order_and_options() {
    let request = lateral_request();
    let hash = request_content_hash(&request);
    assert_eq!(hash, request_content_hash(&request.clone()));

    assert_ne!(
        hash,
        request_content_hash(&other_lateral_request()),
        "pixels"
    );

    let mut reordered = request.clone();
    reordered.frames.swap(0, 1);
    assert_ne!(hash, request_content_hash(&reordered), "frame order");

    let mut one_pixel = request.clone();
    one_pixel.frames[3].rgba[0] ^= 1;
    assert_ne!(hash, request_content_hash(&one_pixel), "one pixel value");

    let mut options = request.clone();
    options.options.max_dense_working_set_bytes /= 2;
    assert_ne!(hash, request_content_hash(&options), "options");
}

// ---------------------------------------------------------------------------
// (b) Round trip of every stage artifact
// ---------------------------------------------------------------------------

#[test]
fn stage_artifacts_round_trip_through_their_versioned_documents() {
    let chain = run_stages(&lateral_request());

    let cameras = CameraStageArtifact::from_json(&chain.cameras.to_json()).expect("cameras");
    let dense = DenseStageArtifact::from_json(&chain.dense.to_json()).expect("dense");
    let mesh = MeshStageArtifact::from_json(&chain.mesh.to_json()).expect("mesh");
    assert_eq!(cameras, chain.cameras);
    assert_eq!(dense, chain.dense);
    assert_eq!(mesh, chain.mesh);
    // Lossless: re-serialization reproduces the stored bytes and therefore the identity.
    assert_eq!(cameras.to_json(), chain.cameras.to_json());
    assert_eq!(dense.to_json(), chain.dense.to_json());
    assert_eq!(mesh.to_json(), chain.mesh.to_json());
    assert_eq!(cameras.content_hash(), chain.cameras.content_hash());
    assert_eq!(dense.content_hash(), chain.dense.content_hash());
    assert_eq!(mesh.content_hash(), chain.mesh.content_hash());

    // The artifacts are ordinary serde types too (e.g. embedded in another document).
    let via_serde: CameraStageArtifact =
        serde_json::from_value(serde_json::to_value(&chain.cameras).unwrap()).unwrap();
    assert_eq!(via_serde, chain.cameras);
    let via_serde: DenseStageArtifact =
        serde_json::from_value(serde_json::to_value(&chain.dense).unwrap()).unwrap();
    assert_eq!(via_serde, chain.dense);
    let via_serde: MeshStageArtifact =
        serde_json::from_value(serde_json::to_value(&chain.mesh).unwrap()).unwrap();
    assert_eq!(via_serde, chain.mesh);
}

/// The recovered solve's artifacts (effective options differ from the request's)
/// round-trip as well.
#[test]
fn recovered_stage_artifacts_round_trip() {
    let chain = run_stages(&recovery_request());
    let cameras = CameraStageArtifact::from_json(&chain.cameras.to_json()).expect("cameras");
    let dense = DenseStageArtifact::from_json(&chain.dense.to_json()).expect("dense");
    let mesh = MeshStageArtifact::from_json(&chain.mesh.to_json()).expect("mesh");
    assert_eq!(
        (cameras, dense, mesh),
        (chain.cameras, chain.dense, chain.mesh)
    );
}

// ---------------------------------------------------------------------------
// (c) Re-running later stages from stored artifacts
// ---------------------------------------------------------------------------

#[test]
fn later_stages_rerun_from_stored_artifacts() {
    for request in [lateral_request(), recovery_request()] {
        let in_memory = run_stages(&request);

        let stored_cameras =
            CameraStageArtifact::from_json(&in_memory.cameras.to_json()).expect("stored cameras");
        let dense = reconstruct_dense(&request, &stored_cameras).expect("dense from storage");
        assert_eq!(dense, in_memory.dense, "dense stage from stored cameras");

        let stored_dense =
            DenseStageArtifact::from_json(&in_memory.dense.to_json()).expect("stored dense");
        let mesh = reconstruct_mesh(&stored_cameras, &stored_dense).expect("mesh from storage");
        assert_eq!(mesh, in_memory.mesh, "mesh stage from stored artifacts");

        let stored_mesh =
            MeshStageArtifact::from_json(&in_memory.mesh.to_json()).expect("stored mesh");
        let composed = compose_reconstruction(&stored_cameras, &stored_dense, &stored_mesh)
            .expect("compose stored chain");
        assert_same_result(&composed, &reconstruct(&request).expect("single call"));
    }
}

/// Re-running only the dense stage on a fresh copy of the request (no state shared
/// with the camera stage's request value) gives the same artifact.
#[test]
fn dense_stage_depends_only_on_the_request_content() {
    let request = lateral_request();
    let cameras = reconstruct_cameras(&request).expect("camera stage");
    let first = reconstruct_dense(&request, &cameras).expect("dense");
    let rebuilt = lateral_request();
    let second = reconstruct_dense(&rebuilt, &cameras).expect("dense from an equal request");
    assert_eq!(first, second);
}

// ---------------------------------------------------------------------------
// (d) Rejection: unknown schema versions and mismatched input identities
// ---------------------------------------------------------------------------

#[test]
fn readers_reject_unknown_schema_versions() {
    let chain = run_stages(&lateral_request());
    let unknown = Value::from(UNKNOWN_SCHEMA_VERSION);

    assert_unsupported(
        CameraStageArtifact::from_json(&with_schema_version(
            &chain.cameras.to_json(),
            unknown.clone(),
        ))
        .expect_err("unknown camera schema"),
        ReconstructionStage::Cameras,
        CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION,
    );
    assert_unsupported(
        DenseStageArtifact::from_json(&with_schema_version(
            &chain.dense.to_json(),
            unknown.clone(),
        ))
        .expect_err("unknown dense schema"),
        ReconstructionStage::Dense,
        DENSE_STAGE_ARTIFACT_SCHEMA_VERSION,
    );
    assert_unsupported(
        MeshStageArtifact::from_json(&with_schema_version(&chain.mesh.to_json(), unknown))
            .expect_err("unknown mesh schema"),
        ReconstructionStage::Mesh,
        MESH_STAGE_ARTIFACT_SCHEMA_VERSION,
    );
}

#[test]
fn readers_reject_documents_without_a_schema_version_or_of_another_stage() {
    let chain = run_stages(&lateral_request());
    for error in [
        CameraStageArtifact::from_json(&without_schema_version(&chain.cameras.to_json()))
            .expect_err("missing schema version"),
        CameraStageArtifact::from_json("not json").expect_err("not JSON"),
        // A dense document is not a camera document, even at the same version number.
        CameraStageArtifact::from_json(&chain.dense.to_json()).expect_err("dense as cameras"),
    ] {
        assert!(
            matches!(
                error,
                ReconstructionStageError::Malformed {
                    artifact: ReconstructionStage::Cameras,
                    ..
                }
            ),
            "{error:?}"
        );
    }
    let error =
        MeshStageArtifact::from_json(&chain.cameras.to_json()).expect_err("cameras as mesh");
    assert!(
        matches!(
            error,
            ReconstructionStageError::Malformed {
                artifact: ReconstructionStage::Mesh,
                ..
            }
        ),
        "{error:?}"
    );
}

/// `schema_version` is a public field, so the stages also check artifacts that were
/// built or altered in memory rather than read through `from_json`.
#[test]
fn stages_reject_in_memory_artifacts_of_unknown_schema_versions() {
    let request = lateral_request();
    let chain = run_stages(&request);

    let mut cameras = chain.cameras.clone();
    cameras.schema_version = UNKNOWN_SCHEMA_VERSION;
    assert_unsupported(
        reconstruct_dense(&request, &cameras).expect_err("dense stage"),
        ReconstructionStage::Cameras,
        CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION,
    );
    assert_unsupported(
        reconstruct_mesh(&cameras, &chain.dense).expect_err("mesh stage"),
        ReconstructionStage::Cameras,
        CAMERA_STAGE_ARTIFACT_SCHEMA_VERSION,
    );

    let mut dense = chain.dense.clone();
    dense.schema_version = UNKNOWN_SCHEMA_VERSION;
    assert_unsupported(
        reconstruct_mesh(&chain.cameras, &dense).expect_err("mesh stage"),
        ReconstructionStage::Dense,
        DENSE_STAGE_ARTIFACT_SCHEMA_VERSION,
    );

    let mut mesh = chain.mesh.clone();
    mesh.schema_version = UNKNOWN_SCHEMA_VERSION;
    assert_unsupported(
        compose_reconstruction(&chain.cameras, &chain.dense, &mesh).expect_err("compose"),
        ReconstructionStage::Mesh,
        MESH_STAGE_ARTIFACT_SCHEMA_VERSION,
    );
}

#[test]
fn dense_stage_rejects_cameras_of_another_request() {
    let request = lateral_request();
    let cameras = reconstruct_cameras(&request).expect("camera stage");

    assert_mismatch(
        reconstruct_dense(&other_lateral_request(), &cameras).expect_err("other frames"),
        ReconstructionStage::Cameras,
        StageInput::Request,
    );

    let mut other_options = request.clone();
    other_options.options.max_dense_working_set_bytes /= 2;
    assert_mismatch(
        reconstruct_dense(&other_options, &cameras).expect_err("other options"),
        ReconstructionStage::Cameras,
        StageInput::Request,
    );
}

#[test]
fn mesh_stage_rejects_dense_evidence_of_another_chain() {
    let ours = run_stages(&lateral_request());
    let theirs = run_stages(&other_lateral_request());
    assert_mismatch(
        reconstruct_mesh(&ours.cameras, &theirs.dense).expect_err("foreign dense evidence"),
        ReconstructionStage::Dense,
        StageInput::Cameras,
    );
}

#[test]
fn compose_rejects_artifacts_of_mixed_chains() {
    let ours = run_stages(&lateral_request());
    let theirs = run_stages(&other_lateral_request());
    assert_mismatch(
        compose_reconstruction(&ours.cameras, &ours.dense, &theirs.mesh).expect_err("foreign mesh"),
        ReconstructionStage::Mesh,
        StageInput::Dense,
    );
    assert_mismatch(
        compose_reconstruction(&ours.cameras, &theirs.dense, &ours.mesh)
            .expect_err("foreign dense evidence"),
        ReconstructionStage::Dense,
        StageInput::Cameras,
    );
}

/// An upstream artifact whose payload changed after a downstream artifact recorded it
/// (a corrupted or hand-edited store) no longer matches the recorded content hash.
#[test]
fn stages_reject_edited_upstream_artifacts() {
    let chain = run_stages(&lateral_request());

    let edited_cameras = CameraStageArtifact::from_json(&edit_payload(&chain.cameras.to_json()))
        .expect("an edited payload still parses");
    assert_ne!(edited_cameras, chain.cameras);
    assert_mismatch(
        reconstruct_mesh(&edited_cameras, &chain.dense).expect_err("edited cameras"),
        ReconstructionStage::Dense,
        StageInput::Cameras,
    );

    let edited_dense = DenseStageArtifact::from_json(&edit_payload(&chain.dense.to_json()))
        .expect("an edited payload still parses");
    assert_ne!(edited_dense, chain.dense);
    assert_mismatch(
        compose_reconstruction(&chain.cameras, &edited_dense, &chain.mesh)
            .expect_err("edited dense evidence"),
        ReconstructionStage::Mesh,
        StageInput::Dense,
    );
}
