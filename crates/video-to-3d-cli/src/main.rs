//! `video-to-3d` command line: build, status, inspect, and export a scene
//! project from its manifest.
//!
//! Every command takes the manifest path explicitly; the project root is the
//! manifest's directory, never the working directory. Every invocation prints
//! exactly one JSON document on stdout (results, or `{command, error,
//! exit_code}` on failure); human-readable diagnostics go to stderr.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use video_to_3d_core::scene_artifacts::BuiltInExecutor;
use video_to_3d_core::scene_model::AssembledScene;
use video_to_3d_core::scene_project::{
    ArtifactKind, ExportFormat, OperationExecution, SceneProjectError, SceneProjectManifest,
};
use video_to_3d_core::scene_runner::{CancellationToken, OperationState, RunOptions, RunReport};
use video_to_3d_core::scene_store::{Invalidation, ProjectStore, ReconcileReport, VerifyMode};

/// Process exit codes. Scripts can rely on these values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Exit {
    Success = 0,
    /// I/O or other unexpected failure.
    Internal = 1,
    /// The manifest is malformed, unsupported, or invalid.
    InvalidProject = 2,
    /// A required capability (operation executor, provider, exporter) is not
    /// available in this build.
    Unsupported = 3,
    /// An operation or provider failed.
    OperationFailure = 4,
    /// The build is not complete (blocked, canceled, or not yet run).
    Incomplete = 5,
    /// Bad command-line usage.
    Usage = 64,
}

const USAGE: &str = "usage:
  video-to-3d build <project.json> [--jobs <n>]
  video-to-3d status <project.json>
  video-to-3d inspect <project.json>
  video-to-3d export <project.json> --target web

exit codes: 0 success, 1 internal error, 2 invalid project, 3 unsupported capability,
            4 operation/provider failure, 5 incomplete build, 64 usage error";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (output, exit) = match dispatch(&args) {
        Ok((output, exit)) => (output, exit),
        Err((message, exit)) => {
            eprintln!("error: {message}");
            if exit == Exit::Usage {
                eprintln!("{USAGE}");
            }
            // Errors are part of the machine-readable contract too.
            (
                json!({
                    "command": args.first(),
                    "error": message,
                    "exit_code": exit as u8,
                }),
                exit,
            )
        }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("JSON values serialize")
    );
    ExitCode::from(exit as u8)
}

type Outcome = Result<(Value, Exit), (String, Exit)>;

fn dispatch(args: &[String]) -> Outcome {
    let usage = |message: &str| Err((message.to_owned(), Exit::Usage));
    let Some(command) = args.first() else {
        return usage("missing command");
    };
    let Some(manifest) = args.get(1) else {
        return usage("missing project manifest path");
    };
    let manifest = PathBuf::from(manifest);
    let options = &args[2..];
    match command.as_str() {
        "build" => {
            let jobs = match options {
                [] => 1,
                [flag, value] if flag == "--jobs" => match value.parse::<usize>() {
                    Ok(jobs) if jobs > 0 => jobs,
                    _ => return usage("--jobs expects a positive integer"),
                },
                _ => return usage("build accepts only --jobs <n>"),
            };
            build(&manifest, jobs)
        }
        "status" if options.is_empty() => status(&manifest),
        "inspect" if options.is_empty() => inspect(&manifest),
        "export" => match options {
            [flag, target] if flag == "--target" && target == "web" => {
                export(&manifest, ExportFormat::WebBundle)
            }
            [flag, target] if flag == "--target" => {
                usage(&format!("unknown export target `{target}`"))
            }
            _ => usage("export requires --target web"),
        },
        "status" | "inspect" => usage(&format!("{command} takes no options")),
        _ => usage(&format!("unknown command `{command}`")),
    }
}

fn project_error(error: SceneProjectError) -> (String, Exit) {
    let exit = match &error {
        SceneProjectError::Malformed(message) if message.starts_with("cannot ") => Exit::Internal,
        _ => Exit::InvalidProject,
    };
    (error.to_string(), exit)
}

fn open(
    manifest: &std::path::Path,
) -> Result<(ProjectStore, SceneProjectManifest), (String, Exit)> {
    ProjectStore::open(manifest).map_err(project_error)
}

fn run_exit(report: &RunReport) -> Exit {
    if report.is_complete() {
        return Exit::Success;
    }
    let states = report.states.values();
    if states.clone().any(|state| {
        matches!(
            state,
            OperationState::Failed { .. } | OperationState::RetryableFailure { .. }
        )
    }) {
        Exit::OperationFailure
    } else if states
        .clone()
        .any(|state| matches!(state, OperationState::Unsupported { .. }))
    {
        Exit::Unsupported
    } else {
        Exit::Incomplete
    }
}

fn state_json(state: &OperationState) -> Value {
    let name = match state {
        OperationState::Reused => "reused",
        OperationState::Succeeded { .. } => "succeeded",
        OperationState::RetryableFailure { .. } => "retryable_failure",
        OperationState::Failed { .. } => "failed",
        OperationState::Unsupported { .. } => "unsupported",
        OperationState::Canceled { .. } => "canceled",
        OperationState::Blocked { .. } => "blocked",
    };
    let mut value = json!({ "state": name, "attempts": state.attempts() });
    match state {
        OperationState::Succeeded {
            provider: Some(provider),
            ..
        } => value["provider"] = json!(provider),
        OperationState::RetryableFailure { message, .. }
        | OperationState::Failed { message, .. }
        | OperationState::Unsupported { message, .. } => value["message"] = json!(message),
        OperationState::Blocked { upstream } => value["blocked_by"] = json!(upstream),
        _ => {}
    }
    value
}

fn reconcile_json(report: &ReconcileReport) -> Value {
    json!({
        "verified": report.verified,
        "invalidated": report.invalidated.iter().map(|(operation, reason)| json!({
            "operation": operation,
            "reason": invalidation_code(reason),
            "message": reason.describe(),
        })).collect::<Vec<_>>(),
        "orphan_receipts": report.orphan_receipts,
    })
}

fn invalidation_code(reason: &Invalidation) -> &'static str {
    match reason {
        Invalidation::MissingOutput => "missing_output",
        Invalidation::CorruptOutput { .. } => "corrupt_output",
        Invalidation::MissingReceipt => "missing_receipt",
        Invalidation::UnreadableReceipt(_) => "unreadable_receipt",
        Invalidation::ReceiptMismatch(_) => "receipt_mismatch",
        Invalidation::CorruptSidecar(_) => "corrupt_sidecar",
    }
}

fn build(manifest_path: &std::path::Path, jobs: usize) -> Outcome {
    let (store, mut manifest) = open(manifest_path)?;
    // Built-in executors exist for texture_bake; every other kind reports an
    // explicit unsupported outcome.
    // Reservations come from the reconciled manifest the build persists.
    let executor = BuiltInExecutor::for_project(store.root(), manifest_path);
    let report = store
        .build(
            &mut manifest,
            &executor,
            RunOptions {
                max_concurrency: jobs,
            },
            &CancellationToken::new(),
        )
        .map_err(project_error)?;
    for line in report
        .reconcile
        .diagnostics()
        .into_iter()
        .chain(report.run.diagnostics())
    {
        eprintln!("{line}");
    }
    let exit = run_exit(&report.run);
    let mut operations: BTreeMap<&String, Value> = BTreeMap::new();
    for (operation, state) in &report.run.states {
        let mut value = state_json(state);
        // Executed work reports its receipt observations (bake diagnostics,
        // reused/rewritten texture references) on stdout and stderr.
        if matches!(state, OperationState::Succeeded { .. }) {
            if let Ok(Some(receipt)) = store.read_receipt(operation) {
                for (key, observation) in &receipt.observations {
                    if matches!(key.as_str(), "diagnostic" | "appearance") {
                        eprintln!("{operation}: {observation}");
                    }
                }
                value["observations"] = json!(receipt.observations);
            }
        }
        operations.insert(operation, value);
    }
    Ok((
        json!({
            "command": "build",
            "project_id": manifest.project_id,
            "complete": report.run.is_complete(),
            "executed": report.run.executed,
            "reconcile": reconcile_json(&report.reconcile),
            "operations": operations,
        }),
        exit,
    ))
}

/// Per-operation status derived without writing anything.
struct Status {
    manifest: SceneProjectManifest,
    verified: SceneProjectManifest,
    reconcile: ReconcileReport,
    operations: BTreeMap<String, Value>,
    complete: bool,
}

fn compute_status(
    manifest_path: &std::path::Path,
) -> Result<(ProjectStore, Status), (String, Exit)> {
    let (store, manifest) = open(manifest_path)?;
    let mut verified = manifest.clone();
    let reconcile = store
        .reconcile(&mut verified, VerifyMode::Cached)
        .map_err(project_error)?;
    // status, inspect and export all explain verification problems.
    for line in reconcile.diagnostics() {
        eprintln!("{line}");
    }
    let identities = verified.operation_identities().map_err(project_error)?;
    let mut operations = BTreeMap::new();
    let mut complete = true;
    for operation in &verified.operations {
        let identity = &identities[&operation.id];
        let artifact = verified
            .artifacts
            .iter()
            .find(|artifact| artifact.produced_by == operation.id);
        let invalid = reconcile
            .invalidated
            .iter()
            .find(|(invalidated, _)| *invalidated == operation.id);
        let used = verified
            .attempt_usage
            .iter()
            .find(|usage| usage.operation == operation.id && usage.operation_identity == *identity)
            .map_or(0, |usage| usage.attempts);
        let mut value = match (artifact, invalid) {
            (Some(artifact), _) if artifact.operation_identity == *identity => {
                json!({ "state": "current" })
            }
            // Exhaustion of the current identity outranks an old artifact.
            _ if used >= operation.max_attempts => json!({ "state": "exhausted" }),
            (Some(_), _) => json!({ "state": "stale" }),
            (None, Some((_, reason))) => json!({
                "state": "invalid",
                "reason": invalidation_code(reason),
                "message": reason.describe(),
            }),
            (None, None) => json!({ "state": "pending" }),
        };
        if value["state"] != "current" {
            complete = false;
        }
        if used > 0 {
            value["attempts_used"] = json!(used);
            value["max_attempts"] = json!(operation.max_attempts);
        }
        operations.insert(operation.id.clone(), value);
    }
    Ok((
        store,
        Status {
            manifest,
            verified,
            reconcile,
            operations,
            complete,
        },
    ))
}

fn status(manifest_path: &std::path::Path) -> Outcome {
    let (_, status) = compute_status(manifest_path)?;
    let exit = if status.complete {
        Exit::Success
    } else if status
        .operations
        .values()
        .any(|value| value["state"] == "exhausted")
    {
        Exit::OperationFailure
    } else {
        Exit::Incomplete
    };
    Ok((
        json!({
            "command": "status",
            "project_id": status.manifest.project_id,
            "complete": status.complete,
            "verification": "cached",
            "reconcile": reconcile_json(&status.reconcile),
            "operations": status.operations,
        }),
        exit,
    ))
}

fn inspect(manifest_path: &std::path::Path) -> Outcome {
    let (store, status) = compute_status(manifest_path)?;
    let manifest = &status.verified;
    let identities = manifest.operation_identities().map_err(project_error)?;
    let mut operations = BTreeMap::new();
    let mut scene = Value::Null;
    for operation in &manifest.operations {
        let mut value = status.operations[&operation.id].clone();
        value["kind"] = serde_json::to_value(operation.kind).unwrap_or(Value::Null);
        value["execution"] = json!(match operation.kind.execution() {
            OperationExecution::BuiltIn => "built_in",
            OperationExecution::Provider => "provider",
        });
        value["identity"] = json!(identities[&operation.id].as_str());
        value["inputs"] = serde_json::to_value(&operation.inputs).unwrap_or(Value::Null);
        value["reachable_providers"] = json!(manifest
            .reachable_providers(operation)
            .iter()
            .map(|provider| json!({
                "id": provider.id,
                "location": provider.location,
                "revision": provider.revision,
            }))
            .collect::<Vec<_>>());
        if let Some(artifact) = manifest
            .artifacts
            .iter()
            .find(|artifact| artifact.produced_by == operation.id)
        {
            value["artifact"] = json!({
                "id": artifact.id,
                "kind": artifact.kind,
                "path": artifact.path.as_str(),
                "content_hash": artifact.content_hash.as_str(),
            });
            if artifact.kind == ArtifactKind::AssembledScene
                && artifact.operation_identity == identities[&operation.id]
            {
                scene = scene_summary(&store, artifact.path.as_str(), &artifact.path);
            }
        }
        if let Ok(Some(receipt)) = store.read_receipt(&operation.id) {
            value["receipt"] = json!({
                "provider": receipt.provider.as_ref().map(|provider| json!({
                    "id": provider.id,
                    "revision": provider.revision,
                })),
                "reproducibility": receipt.reproducibility,
                "observations": receipt.observations,
                "output_bytes": receipt.output.byte_length,
            });
        }
        operations.insert(operation.id.clone(), value);
    }
    Ok((
        json!({
            "command": "inspect",
            "project_id": manifest.project_id,
            "quality_mode": manifest.quality_mode,
            "execution_policy": manifest.provider_policy.execution,
            "requested_outputs": manifest.requested_outputs,
            "complete": status.complete,
            "reconcile": reconcile_json(&status.reconcile),
            "operations": operations,
            "assembled_scene": scene,
        }),
        Exit::Success,
    ))
}

/// Provenance summary of a current assembled-scene artifact, when readable.
fn scene_summary(
    store: &ProjectStore,
    label: &str,
    path: &video_to_3d_core::scene_project::ProjectPath,
) -> Value {
    let document = match std::fs::read_to_string(store.resolve(path)) {
        Ok(document) => document,
        Err(error) => return json!({ "path": label, "error": error.to_string() }),
    };
    let scene = match AssembledScene::from_json(&document) {
        Ok(scene) => scene,
        Err(error) => return json!({ "path": label, "error": error.to_string() }),
    };
    let assets: Vec<Value> = scene
        .assets
        .iter()
        .map(|asset| {
            let provenance = scene.asset_provenance(&asset.id);
            json!({
                "id": asset.id,
                "role": asset.role,
                "visual_provenance": provenance.as_ref().map(|p| &p.visual),
                "collision_provenance": provenance.as_ref().map(|p| &p.collision),
                "contains_generative": provenance.as_ref().is_some_and(|p| p.contains_generative()),
            })
        })
        .collect();
    json!({ "path": label, "unit": scene.frame.unit, "assets": assets })
}

fn export(manifest_path: &std::path::Path, format: ExportFormat) -> Outcome {
    let (_, status) = compute_status(manifest_path)?;
    let declared: Vec<&str> = status
        .manifest
        .exports
        .iter()
        .filter(|export| export.format == format)
        .map(|export| export.id.as_str())
        .collect();
    if declared.is_empty() {
        return Err((
            "the project declares no web_bundle export".into(),
            Exit::InvalidProject,
        ));
    }
    if !status.complete {
        return Err((
            "the build is incomplete; run `video-to-3d build` first".into(),
            Exit::Incomplete,
        ));
    }
    Err((
        format!(
            "web bundle export is not available in this build (declared exports: {})",
            declared.join(", ")
        ),
        Exit::Unsupported,
    ))
}
