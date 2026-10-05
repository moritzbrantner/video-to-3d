use super::*;
use serde_json::{json, Value};

const CLIP_HASH: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";
const OTHER_HASH: &str = "sha256:2222222222222222222222222222222222222222222222222222222222222222";

fn standard_document() -> Value {
    json!({
        "schema_version": 1,
        "project_id": "living-room",
        "quality_mode": "standard",
        "inputs": [
            { "id": "clip", "path": "media/clip.webm", "content_hash": CLIP_HASH, "byte_length": 4096 }
        ],
        "provider_policy": {
            "execution": "allow_paid_cloud",
            "providers": [
                {
                    "id": "local-depth",
                    "location": "local",
                    "capabilities": ["learned_reconstruction"],
                    "revision": "v2"
                },
                {
                    "id": "cloud-world",
                    "location": "cloud",
                    "capabilities": ["generative_completion"],
                    "revision": "2026-09",
                    "credential_capability": "providers.cloud-world",
                    "cost_per_attempt": 40
                }
            ],
            "fallback_order": ["local-depth", "cloud-world"],
            "cloud_upload_allowlist": ["keyframes", "surface_mesh"],
            "max_cost_per_operation": 100,
            "max_total_cost": 200
        },
        "operations": [
            { "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] },
            { "id": "sparse", "kind": "sparse_reconstruction", "inputs": [{ "operation": "ingest" }] },
            { "id": "learned", "kind": "learned_reconstruction", "inputs": [{ "operation": "ingest" }, { "operation": "sparse" }], "provider": "local-depth" },
            { "id": "dense", "kind": "dense_reconstruction", "inputs": [{ "operation": "sparse" }, { "operation": "learned" }] },
            { "id": "mesh", "kind": "surface_mesh", "inputs": [{ "operation": "dense" }] },
            { "id": "complete", "kind": "generative_completion", "inputs": [{ "operation": "ingest" }, { "operation": "mesh" }], "provider": "cloud-world", "max_attempts": 2 },
            { "id": "assemble", "kind": "scene_assembly", "inputs": [{ "operation": "mesh" }, { "operation": "complete" }] }
        ],
        "requested_outputs": ["assembled_scene", "surface_mesh"],
        "exports": [
            { "id": "web", "format": "web_bundle", "path": "exports/web", "includes": ["assembled_scene"] }
        ]
    })
}

fn parse(document: &Value) -> Result<SceneProjectManifest, SceneProjectError> {
    SceneProjectManifest::from_json(&document.to_string())
}

fn expect_invalid(document: &Value, needle: &str) {
    match parse(document) {
        Err(SceneProjectError::Invalid(message)) => assert!(
            message.contains(needle),
            "expected `{needle}` in `{message}`"
        ),
        other => panic!("expected invalid manifest containing `{needle}`, got {other:?}"),
    }
}

fn with(mut document: Value, pointer: &str, value: Value) -> Value {
    let (parent, key) = pointer.rsplit_once('/').expect("fixture pointer");
    match document.pointer_mut(parent).expect("fixture parent") {
        Value::Object(object) => {
            object.insert(key.to_owned(), value);
        }
        Value::Array(array) => array[key.parse::<usize>().expect("index")] = value,
        _ => panic!("fixture parent must be a container"),
    }
    document
}

#[test]
fn valid_manifest_round_trips_canonically() {
    let manifest = parse(&standard_document()).expect("valid manifest");
    let canonical = manifest.to_canonical_json().expect("canonical json");
    let reloaded = SceneProjectManifest::from_json(&canonical).expect("reload");
    assert_eq!(reloaded, manifest);
    assert_eq!(reloaded.to_canonical_json().unwrap(), canonical);
}

#[test]
fn declaration_order_does_not_change_canonical_form_or_identities() {
    let mut shuffled = standard_document();
    shuffled["operations"].as_array_mut().unwrap().reverse();
    shuffled["provider_policy"]["providers"]
        .as_array_mut()
        .unwrap()
        .reverse();
    shuffled["requested_outputs"]
        .as_array_mut()
        .unwrap()
        .reverse();
    shuffled["operations"][4]["inputs"]
        .as_array_mut()
        .unwrap()
        .reverse();
    let original = parse(&standard_document()).unwrap();
    let shuffled = parse(&shuffled).unwrap();
    assert_eq!(
        original.to_canonical_json().unwrap(),
        shuffled.to_canonical_json().unwrap()
    );
    assert_eq!(
        original.operation_identities().unwrap(),
        shuffled.operation_identities().unwrap()
    );
}

#[test]
fn manifest_survives_restart_through_a_file() {
    let manifest = parse(&standard_document()).unwrap();
    let identities = manifest.operation_identities().unwrap();
    let mut resumed = manifest.clone();
    resumed.artifacts.push(ArtifactRecord {
        id: "sparse-output".into(),
        kind: ArtifactKind::SparseReconstruction,
        produced_by: "sparse".into(),
        operation_identity: identities["sparse"].clone(),
        path: ProjectPath::new("artifacts/sparse.bin").unwrap(),
        content_hash: ContentHash::of_bytes(b"sparse"),
    });
    let directory =
        std::env::temp_dir().join(format!("video-to-3d-scene-project-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let file = directory.join("project.json");
    std::fs::write(&file, resumed.to_canonical_json().unwrap()).unwrap();
    let reloaded = SceneProjectManifest::load(&file).unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    assert_eq!(reloaded, resumed);
    assert!(reloaded.stale_artifacts().unwrap().is_empty());
    assert_eq!(
        reloaded.artifacts[0]
            .path
            .resolve(Path::new("/projects/room")),
        Path::new("/projects/room/artifacts/sparse.bin")
    );
}

#[test]
fn input_change_invalidates_only_descendant_identities() {
    let base = parse(&standard_document()).unwrap();
    let mut changed_media = standard_document();
    changed_media["inputs"][0]["content_hash"] = json!(OTHER_HASH);
    let changed_media = parse(&changed_media).unwrap();
    let a = base.operation_identities().unwrap();
    let b = changed_media.operation_identities().unwrap();
    for id in a.keys() {
        assert_ne!(a[id], b[id], "every operation descends from the clip");
    }

    let mut changed_provider = standard_document();
    changed_provider["provider_policy"]["providers"][1]["revision"] = json!("2026-10");
    let c = parse(&changed_provider)
        .unwrap()
        .operation_identities()
        .unwrap();
    for id in ["ingest", "sparse", "learned", "dense", "mesh"] {
        assert_eq!(a[id], c[id], "`{id}` does not depend on the cloud provider");
    }
    for id in ["complete", "assemble"] {
        assert_ne!(
            a[id], c[id],
            "`{id}` depends on the cloud provider revision"
        );
    }

    let mut renamed_credential = standard_document();
    renamed_credential["provider_policy"]["providers"][1]["credential_capability"] =
        json!("providers.other-account");
    let d = parse(&renamed_credential)
        .unwrap()
        .operation_identities()
        .unwrap();
    assert_eq!(a, d, "credential selection does not change results");
}

#[test]
fn stale_artifacts_are_reported_after_upstream_changes() {
    let base = parse(&standard_document()).unwrap();
    let identities = base.operation_identities().unwrap();
    let mut document = standard_document();
    document["inputs"][0]["content_hash"] = json!(OTHER_HASH);
    document["artifacts"] = json!([{
        "id": "mesh-output",
        "kind": "surface_mesh",
        "produced_by": "mesh",
        "operation_identity": identities["mesh"].as_str(),
        "path": "artifacts/mesh.bin",
        "content_hash": OTHER_HASH
    }]);
    let manifest = parse(&document).unwrap();
    let stale = manifest.stale_artifacts().unwrap();
    assert_eq!(stale.len(), 1);
    assert_eq!(stale[0].id, "mesh-output");
}

#[test]
fn unknown_schema_versions_and_fields_fail_closed() {
    assert_eq!(
        parse(&with(standard_document(), "/schema_version", json!(2))),
        Err(SceneProjectError::UnsupportedSchemaVersion(2))
    );
    let mut missing = standard_document();
    missing.as_object_mut().unwrap().remove("schema_version");
    assert!(matches!(
        parse(&missing),
        Err(SceneProjectError::Malformed(_))
    ));
    let mut extra = standard_document();
    extra["provider_policy"]["providers"][1]["api_key"] = json!("sk-secret");
    assert!(matches!(
        parse(&extra),
        Err(SceneProjectError::Malformed(_))
    ));
    assert!(matches!(
        SceneProjectManifest::from_json("{"),
        Err(SceneProjectError::Malformed(_))
    ));
}

#[test]
fn non_portable_paths_are_rejected() {
    for path in [
        "/abs/clip.webm",
        "../clip.webm",
        "media/../clip.webm",
        "media//clip.webm",
        "./clip.webm",
        "C:/clip.webm",
        "media\\clip.webm",
        "",
    ] {
        assert!(ProjectPath::new(path).is_err(), "{path} must be rejected");
        assert!(
            matches!(
                parse(&with(standard_document(), "/inputs/0/path", json!(path))),
                Err(SceneProjectError::Malformed(_))
            ),
            "{path} must be rejected while loading"
        );
    }
    expect_invalid(
        &with(
            standard_document(),
            "/exports/0/path",
            json!("media/clip.webm"),
        ),
        "used more than once",
    );
}

#[test]
fn credentials_must_be_capability_names() {
    let document = with(
        standard_document(),
        "/provider_policy/providers/1/credential_capability",
        json!("sk-ABCDEF0123456789"),
    );
    expect_invalid(&document, "never a secret");
}

#[test]
fn quality_mode_gates_declared_operations() {
    expect_invalid(
        &with(standard_document(), "/quality_mode", json!("preview")),
        "requires quality mode Standard",
    );
    let mut splat = standard_document();
    splat["provider_policy"]["providers"][0]["capabilities"] =
        json!(["learned_reconstruction", "splat_optimization"]);
    splat["operations"].as_array_mut().unwrap().push(json!({
        "id": "splat", "kind": "splat_optimization",
        "inputs": [{ "operation": "sparse" }], "provider": "local-depth"
    }));
    expect_invalid(&splat, "requires quality mode Production");
    let production = with(splat, "/quality_mode", json!("production"));
    parse(&production).expect("production allows splat optimization");
}

#[test]
fn provider_policy_combinations_fail_closed() {
    expect_invalid(
        &with(
            standard_document(),
            "/provider_policy/execution",
            json!("local_only"),
        ),
        "not allowed by the local_only",
    );
    expect_invalid(
        &with(
            standard_document(),
            "/provider_policy/execution",
            json!("local_first"),
        ),
        "requires the allow_paid_cloud",
    );
    let mut unbounded = standard_document();
    unbounded["provider_policy"]
        .as_object_mut()
        .unwrap()
        .remove("max_total_cost");
    expect_invalid(&unbounded, "requires max_total_cost");
    expect_invalid(
        &with(
            standard_document(),
            "/provider_policy/max_cost_per_operation",
            json!(50),
        ),
        "exceeds max_cost_per_operation",
    );
    expect_invalid(
        &with(
            standard_document(),
            "/provider_policy/max_total_cost",
            json!(79),
        ),
        "exceeds max_total_cost",
    );
    expect_invalid(
        &with(
            standard_document(),
            "/provider_policy/cloud_upload_allowlist",
            json!(["keyframes"]),
        ),
        "would upload SurfaceMesh",
    );
    expect_invalid(
        &with(
            standard_document(),
            "/provider_policy/fallback_order",
            json!(["local-depth"]),
        ),
        "every declared provider",
    );

    let mut free_cloud = standard_document();
    free_cloud["provider_policy"]["execution"] = json!("local_first");
    free_cloud["provider_policy"]["providers"][1]["cost_per_attempt"] = json!(0);
    parse(&free_cloud).expect("local_first allows free cloud providers");
    free_cloud["provider_policy"]["fallback_order"] = json!(["cloud-world", "local-depth"]);
    expect_invalid(&free_cloud, "before cloud providers");
}

#[test]
fn operation_graph_invariants_fail_closed() {
    expect_invalid(
        &with(
            standard_document(),
            "/operations/1/inputs",
            json!([{ "media": "clip" }]),
        ),
        "only ingest_video may",
    );
    expect_invalid(
        &with(
            standard_document(),
            "/operations/1/provider",
            json!("local-depth"),
        ),
        "must not declare provider",
    );
    let mut missing_provider = standard_document();
    missing_provider["operations"][2]
        .as_object_mut()
        .unwrap()
        .remove("provider");
    expect_invalid(&missing_provider, "requires a declared provider");
    expect_invalid(
        &with(
            standard_document(),
            "/operations/2/provider",
            json!("cloud-world"),
        ),
        "does not declare capability learned_reconstruction",
    );
    expect_invalid(
        &with(
            standard_document(),
            "/operations/4/inputs",
            json!([{ "operation": "sparse" }]),
        ),
        "cannot consume `sparse`",
    );
    expect_invalid(
        &with(
            standard_document(),
            "/operations/4/inputs",
            json!([{ "operation": "missing" }]),
        ),
        "unknown operation `missing`",
    );
    expect_invalid(
        &with(standard_document(), "/operations/5/max_attempts", json!(0)),
        "attempts",
    );
    expect_invalid(
        &with(standard_document(), "/operations/5/max_attempts", json!(6)),
        "attempts",
    );
    expect_invalid(
        &with(standard_document(), "/operations/1/id", json!("ingest")),
        "declared more than once",
    );
    expect_invalid(
        &with(
            standard_document(),
            "/requested_outputs",
            json!(["splat_field"]),
        ),
        "not produced by any declared operation",
    );
    expect_invalid(
        &with(
            standard_document(),
            "/exports/0/includes",
            json!(["dense_evidence"]),
        ),
        "not a requested output",
    );
}

#[test]
fn cycles_are_rejected() {
    let mut document = standard_document();
    document["operations"][3]["inputs"] =
        json!([{ "operation": "sparse" }, { "operation": "learned" }]);
    document["operations"][2]["inputs"] = json!([{ "operation": "sparse" }]);
    parse(&document).expect("acyclic variant is valid");

    let operations = vec![
        OperationDeclaration {
            id: "a".into(),
            kind: OperationKind::DenseReconstruction,
            inputs: vec![OperationInput::Operation("b".into())],
            provider: None,
            max_attempts: 1,
        },
        OperationDeclaration {
            id: "b".into(),
            kind: OperationKind::LearnedReconstruction,
            inputs: vec![OperationInput::Operation("a".into())],
            provider: None,
            max_attempts: 1,
        },
    ];
    match topological_order(&operations) {
        Err(SceneProjectError::Invalid(message)) => assert!(message.contains("cycle")),
        other => panic!("expected cycle error, got {other:?}"),
    }
}

#[test]
fn artifacts_must_match_their_producing_operation() {
    let identities = parse(&standard_document())
        .unwrap()
        .operation_identities()
        .unwrap();
    let artifact = |kind: &str, produced_by: &str| {
        json!({
            "id": "out", "kind": kind, "produced_by": produced_by,
            "operation_identity": identities["mesh"].as_str(),
            "path": "artifacts/out.bin", "content_hash": OTHER_HASH
        })
    };
    expect_invalid(
        &with(
            standard_document(),
            "/artifacts",
            json!([artifact("splat_field", "mesh")]),
        ),
        "produces SurfaceMesh",
    );
    expect_invalid(
        &with(
            standard_document(),
            "/artifacts",
            json!([artifact("surface_mesh", "missing")]),
        ),
        "unknown operation",
    );
    assert!(matches!(
        parse(&with(
            standard_document(),
            "/artifacts",
            json!([{
                "id": "out", "kind": "surface_mesh", "produced_by": "mesh",
                "operation_identity": "md5:abc", "path": "artifacts/out.bin",
                "content_hash": OTHER_HASH
            }]),
        )),
        Err(SceneProjectError::Malformed(_))
    ));
}

#[test]
fn every_operation_kind_produces_a_distinct_artifact_kind() {
    let outputs: BTreeSet<ArtifactKind> = OperationKind::ALL
        .iter()
        .map(|kind| kind.output())
        .collect();
    assert_eq!(outputs.len(), OperationKind::ALL.len());
}

#[test]
fn replaced_upstream_output_invalidates_recorded_descendants() {
    let mut manifest = parse(&standard_document()).unwrap();
    let identities = manifest.operation_identities().unwrap();
    manifest.artifacts.push(ArtifactRecord {
        id: "mesh-output".into(),
        kind: ArtifactKind::SurfaceMesh,
        produced_by: "mesh".into(),
        operation_identity: identities["mesh"].clone(),
        path: ProjectPath::new("artifacts/mesh.bin").unwrap(),
        content_hash: ContentHash::of_bytes(b"mesh v1"),
    });
    let identities = manifest.operation_identities().unwrap();
    manifest.artifacts.push(ArtifactRecord {
        id: "completion-output".into(),
        kind: ArtifactKind::GenerativeCompletion,
        produced_by: "complete".into(),
        operation_identity: identities["complete"].clone(),
        path: ProjectPath::new("artifacts/completion.bin").unwrap(),
        content_hash: ContentHash::of_bytes(b"completion"),
    });
    assert!(manifest.stale_artifacts().unwrap().is_empty());

    // Re-running `mesh` with the same declaration produced different bytes.
    manifest.artifacts[0].content_hash = ContentHash::of_bytes(b"mesh v2");
    let stale: Vec<&str> = manifest
        .stale_artifacts()
        .unwrap()
        .into_iter()
        .map(|artifact| artifact.id.as_str())
        .collect();
    assert_eq!(stale, ["completion-output"]);
}

#[test]
fn cloud_fallbacks_are_checked_for_uploads_and_cost() {
    let mut document = standard_document();
    document["provider_policy"]["providers"][1]["capabilities"] =
        json!(["generative_completion", "learned_reconstruction"]);
    // `learned` names the local provider but can fall back to the cloud one,
    // which would upload the sparse reconstruction.
    expect_invalid(&document, "can reach cloud provider `cloud-world`");

    document["provider_policy"]["cloud_upload_allowlist"] =
        json!(["keyframes", "sparse_reconstruction", "surface_mesh"]);
    let manifest = parse(&document).expect("allowlisted fallback is valid");
    let learned = manifest
        .operations
        .iter()
        .find(|operation| operation.id == "learned")
        .unwrap();
    let reachable: Vec<&str> = manifest
        .reachable_providers(learned)
        .into_iter()
        .map(|provider| provider.id.as_str())
        .collect();
    assert_eq!(reachable, ["local-depth", "cloud-world"]);

    // Worst case now includes the paid fallback: 2 x 40 + 1 x 40 = 120.
    expect_invalid(
        &with(document, "/provider_policy/max_total_cost", json!(119)),
        "exceeds max_total_cost",
    );
}

#[test]
fn filesystem_aliases_are_rejected() {
    for path in [
        "media/clip.webm.",
        "media/con.webm",
        "LPT1",
        "media/clip webm",
    ] {
        assert!(ProjectPath::new(path).is_err(), "{path} must be rejected");
    }
    assert!(ProjectPath::new("media/Clip.WEBM").is_ok());
    expect_invalid(
        &with(
            standard_document(),
            "/exports/0/path",
            json!("MEDIA/Clip.webm"),
        ),
        "used more than once",
    );
}

#[test]
fn local_first_rejects_cloud_primary_when_local_can_serve() {
    let mut document = standard_document();
    document["provider_policy"]["execution"] = json!("local_first");
    document["provider_policy"]["providers"][1]["cost_per_attempt"] = json!(0);
    document["provider_policy"]["providers"][1]["capabilities"] =
        json!(["generative_completion", "learned_reconstruction"]);
    document["provider_policy"]["cloud_upload_allowlist"] =
        json!(["keyframes", "sparse_reconstruction", "surface_mesh"]);
    parse(&document).expect("local primary with cloud fallback is valid");
    document["operations"][2]["provider"] = json!("cloud-world");
    expect_invalid(
        &document,
        "although local provider `local-depth` can serve it",
    );
}

#[test]
fn built_in_operations_carry_an_implementation_revision() {
    let identity = OperationIdentityInput {
        schema_version: SCENE_PROJECT_SCHEMA_VERSION,
        kind: OperationKind::SparseReconstruction,
        implementation: Some((
            OperationKind::SparseReconstruction.implementation_revision(),
            env!("CARGO_PKG_VERSION"),
        )),
        providers: Vec::new(),
        max_attempts: 1,
        inputs: Vec::new(),
    };
    let encoded = serde_json::to_vec(&identity).unwrap();
    let bumped = OperationIdentityInput {
        implementation: Some((
            OperationKind::SparseReconstruction.implementation_revision() + 1,
            env!("CARGO_PKG_VERSION"),
        )),
        ..identity
    };
    assert_ne!(encoded, serde_json::to_vec(&bumped).unwrap());
}

#[test]
fn switching_between_identical_providers_changes_identity() {
    let mut document = standard_document();
    document["provider_policy"]["providers"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "id": "local-depth-alt",
            "location": "local",
            "capabilities": ["learned_reconstruction"],
            "revision": "v2"
        }));
    document["provider_policy"]["fallback_order"] =
        json!(["local-depth", "local-depth-alt", "cloud-world"]);
    let base = parse(&document).unwrap().operation_identities().unwrap();
    document["operations"][2]["provider"] = json!("local-depth-alt");
    let switched = parse(&document).unwrap().operation_identities().unwrap();
    assert_ne!(base["learned"], switched["learned"]);
    assert_eq!(base["sparse"], switched["sparse"]);
}
