use super::*;
use serde_json::json;
use std::sync::Mutex;

const CLIP_HASH: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";

/// Two independent branches after sparse: `learned -> dense -> mesh` needs a
/// provider; `decompose` is independent; `assemble` joins them.
fn manifest_document() -> serde_json::Value {
    json!({
        "schema_version": 1,
        "project_id": "runner",
        "quality_mode": "standard",
        "inputs": [
            { "id": "clip", "path": "media/clip.webm", "content_hash": CLIP_HASH, "byte_length": 10 }
        ],
        "provider_policy": {
            "execution": "local_only",
            "providers": [
                { "id": "depth-a", "location": "local", "capabilities": ["learned_reconstruction"] },
                { "id": "depth-b", "location": "local", "capabilities": ["learned_reconstruction"] },
                { "id": "segmenter", "location": "local", "capabilities": ["scene_decomposition"] }
            ],
            "fallback_order": ["depth-a", "depth-b", "segmenter"]
        },
        "operations": [
            { "id": "ingest", "kind": "ingest_video", "inputs": [{ "media": "clip" }] },
            { "id": "sparse", "kind": "sparse_reconstruction", "inputs": [{ "operation": "ingest" }] },
            { "id": "learned", "kind": "learned_reconstruction", "inputs": [{ "operation": "sparse" }], "provider": "depth-a", "max_attempts": 3 },
            { "id": "dense", "kind": "dense_reconstruction", "inputs": [{ "operation": "sparse" }, { "operation": "learned" }], "max_attempts": 2 },
            { "id": "mesh", "kind": "surface_mesh", "inputs": [{ "operation": "dense" }] },
            { "id": "decompose", "kind": "scene_decomposition", "inputs": [{ "operation": "ingest" }], "provider": "segmenter" },
            { "id": "assemble", "kind": "scene_assembly", "inputs": [{ "operation": "mesh" }, { "operation": "decompose" }] }
        ],
        "requested_outputs": ["assembled_scene"]
    })
}

fn manifest() -> SceneProjectManifest {
    SceneProjectManifest::from_json(&manifest_document().to_string()).unwrap()
}

type Script = dyn Fn(&OperationRequest) -> Option<AttemptOutcome> + Sync;

/// Deterministic executor: output bytes are a function of the request identity
/// and provider, so equal identities produce equal artifacts. A script may
/// override outcomes per attempt.
struct FakeExecutor {
    calls: Mutex<Vec<(String, Option<String>, u32)>>,
    script: Box<Script>,
}

impl FakeExecutor {
    fn new() -> Self {
        Self::scripted(|_| None)
    }

    fn scripted(
        script: impl Fn(&OperationRequest) -> Option<AttemptOutcome> + Sync + 'static,
    ) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            script: Box::new(script),
        }
    }

    fn calls(&self) -> Vec<(String, Option<String>, u32)> {
        let mut calls = self.calls.lock().unwrap().clone();
        calls.sort();
        calls
    }

    fn executed_operations(&self) -> BTreeSet<String> {
        self.calls().into_iter().map(|(id, _, _)| id).collect()
    }
}

impl OperationExecutor for FakeExecutor {
    fn execute(&self, request: &OperationRequest, _cancel: &CancellationToken) -> AttemptOutcome {
        self.calls.lock().unwrap().push((
            request.operation.id.clone(),
            request
                .provider
                .as_ref()
                .map(|provider| provider.id.clone()),
            request.attempt,
        ));
        if let Some(outcome) = (self.script)(request) {
            return outcome;
        }
        let mut bytes = request.identity.as_str().as_bytes().to_vec();
        for input in &request.inputs {
            if let ResolvedInput::Artifact(artifact) = input {
                bytes.extend_from_slice(artifact.content_hash.as_str().as_bytes());
            }
        }
        AttemptOutcome::Succeeded(ProducedArtifact {
            path: ProjectPath::new(format!("artifacts/{}.bin", request.operation.id)).unwrap(),
            content_hash: ContentHash::of_bytes(&bytes),
        })
    }
}

fn run_with(
    manifest: &mut SceneProjectManifest,
    executor: &FakeExecutor,
    max_concurrency: usize,
) -> RunReport {
    run(
        manifest,
        executor,
        RunOptions { max_concurrency },
        &CancellationToken::new(),
    )
    .unwrap()
}

#[test]
fn runs_the_declared_graph_and_records_every_artifact() {
    let mut manifest = manifest();
    let executor = FakeExecutor::new();
    let report = run_with(&mut manifest, &executor, 1);
    assert!(report.is_complete());
    assert_eq!(report.executed(), 7);
    assert_eq!(manifest.artifacts.len(), 7);
    assert!(manifest.stale_artifacts().unwrap().is_empty());
    assert_eq!(
        report.states["learned"],
        OperationState::Succeeded {
            attempts: 1,
            provider: Some("depth-a".into())
        }
    );
    // The recorded manifest survives a canonical round trip.
    let reloaded = SceneProjectManifest::from_json(&manifest.to_canonical_json().unwrap()).unwrap();
    assert_eq!(reloaded, manifest);
}

#[test]
fn rerunning_a_completed_graph_executes_nothing() {
    let mut manifest = manifest();
    run_with(&mut manifest, &FakeExecutor::new(), 1);
    let snapshot = manifest.clone();
    let executor = FakeExecutor::new();
    let report = run_with(&mut manifest, &executor, 4);
    assert!(executor.calls().is_empty());
    assert!(report
        .states
        .values()
        .all(|state| *state == OperationState::Reused));
    assert_eq!(manifest, snapshot);
}

#[test]
fn concurrency_and_declaration_order_do_not_change_artifacts() {
    let mut sequential = manifest();
    run_with(&mut sequential, &FakeExecutor::new(), 1);

    let mut document = manifest_document();
    document["operations"].as_array_mut().unwrap().reverse();
    let mut concurrent = SceneProjectManifest::from_json(&document.to_string()).unwrap();
    run_with(&mut concurrent, &FakeExecutor::new(), 4);

    assert_eq!(
        sequential.to_canonical_json().unwrap(),
        concurrent.to_canonical_json().unwrap()
    );
}

#[test]
fn independent_operations_run_concurrently_in_one_wave() {
    use std::sync::Barrier;
    // `sparse` and `decompose` are both ready after `ingest`; a two-party
    // barrier only releases if they really execute at the same time.
    let barrier = Arc::new(Barrier::new(2));
    let script_barrier = barrier.clone();
    let executor = FakeExecutor::scripted(move |request| {
        if matches!(request.operation.id.as_str(), "sparse" | "decompose") {
            script_barrier.wait();
        }
        None
    });
    let mut manifest = manifest();
    let report = run_with(&mut manifest, &executor, 2);
    assert!(report.is_complete());
}

#[test]
fn partial_failure_blocks_only_descendants_and_resumes() {
    let mut manifest = manifest();
    let failing = FakeExecutor::scripted(|request| {
        (request.operation.id == "dense").then(|| AttemptOutcome::Failed("no texture".into()))
    });
    let report = run_with(&mut manifest, &failing, 2);
    assert!(!report.is_complete());
    assert_eq!(
        report.states["dense"],
        OperationState::Failed {
            attempts: 1,
            message: "no texture".into()
        }
    );
    assert_eq!(
        report.states["mesh"],
        OperationState::Blocked {
            upstream: vec!["dense".into()]
        }
    );
    assert!(matches!(
        report.states["assemble"],
        OperationState::Blocked { .. }
    ));
    assert!(report.states["decompose"].is_complete());
    assert_eq!(manifest.artifacts.len(), 4);

    let executor = FakeExecutor::new();
    let resumed = run_with(&mut manifest, &executor, 2);
    assert!(resumed.is_complete());
    assert_eq!(
        executor.executed_operations(),
        BTreeSet::from(["assemble".into(), "dense".into(), "mesh".into()])
    );

    let mut cold = self::manifest();
    run_with(&mut cold, &FakeExecutor::new(), 1);
    assert_eq!(
        cold.to_canonical_json().unwrap(),
        manifest.to_canonical_json().unwrap()
    );
}

#[test]
fn retryable_failures_are_retried_within_the_declared_bound() {
    let flaky = FakeExecutor::scripted(|request| {
        (request.operation.id == "learned" && request.attempt < 3)
            .then(|| AttemptOutcome::RetryableFailure("busy".into()))
    });
    let mut manifest = manifest();
    let report = run_with(&mut manifest, &flaky, 1);
    assert_eq!(
        report.states["learned"],
        OperationState::Succeeded {
            attempts: 3,
            provider: Some("depth-a".into())
        }
    );

    let always_busy = FakeExecutor::scripted(|request| {
        (request.operation.id == "learned").then(|| AttemptOutcome::RetryableFailure("busy".into()))
    });
    let mut manifest = self::manifest();
    let report = run_with(&mut manifest, &always_busy, 1);
    assert_eq!(
        report.states["learned"],
        OperationState::RetryableFailure {
            attempts: 3,
            message: "busy".into()
        }
    );
    let learned_calls = always_busy
        .calls()
        .into_iter()
        .filter(|(id, _, _)| id == "learned")
        .count();
    assert_eq!(learned_calls, 3, "attempt bound is never exceeded");
}

#[test]
fn unsupported_provider_falls_back_in_declared_order() {
    let executor = FakeExecutor::scripted(|request| {
        let provider = request
            .provider
            .as_ref()
            .map(|provider| provider.id.as_str());
        (request.operation.id == "learned" && provider == Some("depth-a"))
            .then(|| AttemptOutcome::Unsupported("no GPU".into()))
    });
    let mut manifest = manifest();
    let report = run_with(&mut manifest, &executor, 1);
    assert_eq!(
        report.states["learned"],
        OperationState::Succeeded {
            attempts: 2,
            provider: Some("depth-b".into())
        }
    );

    let nothing_supported = FakeExecutor::scripted(|request| {
        (request.operation.id == "learned").then(|| AttemptOutcome::Unsupported("no GPU".into()))
    });
    let mut manifest = self::manifest();
    let report = run_with(&mut manifest, &nothing_supported, 1);
    assert_eq!(
        report.states["learned"],
        OperationState::Unsupported {
            attempts: 2,
            message: "no GPU".into()
        }
    );
}

#[test]
fn cancellation_stops_new_work_and_resume_completes() {
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    let executor = FakeExecutor::scripted(move |request| {
        if request.operation.id == "sparse" {
            trigger.cancel();
        }
        None
    });
    let mut manifest = manifest();
    let report = run(&mut manifest, &executor, RunOptions::default(), &cancel).unwrap();
    assert!(!report.is_complete());
    assert!(report.states["sparse"].is_complete());
    for id in ["learned", "dense", "mesh", "assemble"] {
        assert!(
            matches!(
                report.states[id],
                OperationState::Canceled { attempts: 0 } | OperationState::Blocked { .. }
            ),
            "{id}: {:?}",
            report.states[id]
        );
    }
    assert!(!executor.executed_operations().contains("learned"));

    let executor = FakeExecutor::new();
    let resumed = run_with(&mut manifest, &executor, 2);
    assert!(resumed.is_complete());
    assert!(!executor.executed_operations().contains("sparse"));
}

#[test]
fn changed_inputs_rerun_only_affected_operations() {
    let mut manifest = manifest();
    run_with(&mut manifest, &FakeExecutor::new(), 1);
    // Re-declare the segmenter with a new revision: only `decompose` and its
    // descendant `assemble` must run again.
    manifest.provider_policy.providers[2].revision = Some("v2".into());
    let executor = FakeExecutor::new();
    let report = run_with(&mut manifest, &executor, 2);
    assert!(report.is_complete());
    assert_eq!(
        executor.executed_operations(),
        BTreeSet::from(["assemble".into(), "decompose".into()])
    );
    assert!(manifest.stale_artifacts().unwrap().is_empty());
}

#[test]
fn zero_concurrency_is_rejected() {
    let mut manifest = manifest();
    assert!(run(
        &mut manifest,
        &FakeExecutor::new(),
        RunOptions { max_concurrency: 0 },
        &CancellationToken::new()
    )
    .is_err());
}

#[test]
fn attempt_budget_persists_across_resumed_runs() {
    let failing = || {
        FakeExecutor::scripted(|request| {
            (request.operation.id == "dense").then(|| AttemptOutcome::Failed("no texture".into()))
        })
    };
    let mut manifest = manifest();
    run_with(&mut manifest, &failing(), 1);
    assert_eq!(manifest.attempt_usage.len(), 1);
    assert_eq!(manifest.attempt_usage[0].attempts, 1);

    // The usage survives persistence.
    let mut manifest =
        SceneProjectManifest::from_json(&manifest.to_canonical_json().unwrap()).unwrap();
    let second = failing();
    let report = run_with(&mut manifest, &second, 1);
    assert_eq!(second.calls(), [("dense".into(), None, 2)]);
    assert_eq!(report.states["dense"].attempts(), 2);

    let third = failing();
    let report = run_with(&mut manifest, &third, 1);
    assert!(
        third.calls().is_empty(),
        "exhausted budget must not execute"
    );
    assert!(
        report
            .diagnostics()
            .iter()
            .any(|line| line.starts_with("dense: failed")
                && line.contains("attempt budget exhausted"))
    );

    // Success clears the usage record.
    let mut manifest = self::manifest();
    run_with(&mut manifest, &failing(), 1);
    run_with(&mut manifest, &FakeExecutor::new(), 1);
    assert!(manifest.attempt_usage.is_empty());
}

#[test]
fn rejected_records_are_failures_that_keep_the_attempt_budget() {
    let mut manifest = manifest();
    let before = manifest.clone();
    let clashing = FakeExecutor::scripted(|request| {
        (request.operation.id == "ingest").then(|| {
            AttemptOutcome::Succeeded(ProducedArtifact {
                path: ProjectPath::new("media/clip.webm").unwrap(),
                content_hash: ContentHash::of_bytes(b"x"),
            })
        })
    });
    let report = run_with(&mut manifest, &clashing, 1);
    assert!(matches!(
        &report.states["ingest"],
        OperationState::Failed { attempts: 1, message } if message.contains("used more than once")
    ));
    assert!(manifest.artifacts.is_empty());
    assert_eq!(manifest.operations, before.operations);
    assert_eq!(manifest.attempt_usage.len(), 1);
    manifest.validate().unwrap();

    // The single declared attempt is spent: a rerun does not call the executor.
    let rerun = FakeExecutor::new();
    let report = run_with(&mut manifest, &rerun, 1);
    assert!(rerun.calls().is_empty());
    assert_eq!(report.executed(), 0);
}

#[test]
fn executor_panics_are_counted_attempts() {
    let panicking = || {
        FakeExecutor::scripted(|request| {
            if request.operation.id == "dense" {
                panic!("boom");
            }
            None
        })
    };
    let mut manifest = manifest();
    for (concurrency, expected) in [(2, 1), (1, 2)] {
        let report = run_with(&mut manifest, &panicking(), concurrency);
        assert_eq!(report.states["dense"].attempts(), expected);
    }
    let third = panicking();
    run_with(&mut manifest, &third, 2);
    assert!(!third.executed_operations().contains("dense"));
}

#[test]
fn generated_artifact_ids_are_valid_and_unique() {
    let mut document = manifest_document();
    let long = "a".repeat(60);
    document["operations"][0]["id"] = json!(long);
    document["operations"][1]["inputs"] = json!([{ "operation": long }]);
    document["operations"][5]["inputs"] = json!([{ "operation": long }]);
    // An operation already named like the readable artifact id of `sparse`.
    document["operations"][6]["id"] = json!("sparse.output");
    let mut manifest = SceneProjectManifest::from_json(&document.to_string()).unwrap();
    let report = run_with(&mut manifest, &FakeExecutor::new(), 1);
    assert!(report.is_complete(), "{:?}", report.diagnostics());
    let ids: BTreeSet<&str> = manifest
        .artifacts
        .iter()
        .map(|artifact| artifact.id.as_str())
        .collect();
    assert_eq!(ids.len(), 7);
    assert!(ids.iter().all(|id| id.len() <= 64));
    assert!(!ids.contains("sparse.output"));
}
