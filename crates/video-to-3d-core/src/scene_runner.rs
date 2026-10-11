//! Deterministic executor for the operation graph declared in a
//! [`SceneProjectManifest`].
//!
//! The runner is deliberately small: it derives work from the manifest, reuses
//! recorded artifacts whose operation identity is still current, runs ready
//! operations in deterministic waves (concurrently when allowed), applies the
//! declared attempt bound and provider fallback order, honours cancellation,
//! and records every produced artifact back into the manifest. It never guesses
//! a dependent result: an operation runs only after every upstream operation
//! has a current recorded artifact.
//!
//! Accepted artifact identity depends only on content (see
//! [`SceneProjectManifest::operation_identities`]), so wave composition,
//! concurrency, and declaration order cannot change it.
//!
//! Recorded artifacts are trusted here; verifying their bytes on disk is the
//! job of receipt-backed reconciliation.

use crate::scene_project::{
    ArtifactKind, ArtifactRecord, AttemptUsage, ContentHash, MediaInput, OperationDeclaration,
    OperationInput, ProjectPath, ProviderDeclaration, SceneProjectError, SceneProjectManifest,
};
pub use crate::scene_store::Reproducibility;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Shared cancellation flag. Cancelling stops new attempts from starting;
/// executors should also poll it to abandon long-running work.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_canceled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// A resolved input handed to an executor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedInput {
    Media(MediaInput),
    Artifact(ArtifactRecord),
}

/// Everything an executor needs to run one attempt of one operation.
#[derive(Clone, Debug)]
pub struct OperationRequest {
    pub operation: OperationDeclaration,
    /// Identity the produced artifact will be recorded under.
    pub identity: ContentHash,
    /// Provider for this attempt (`None` for built-in operations).
    pub provider: Option<ProviderDeclaration>,
    /// 1-based attempt number across all providers and resumed runs.
    pub attempt: u32,
    /// Inputs in declaration order.
    pub inputs: Vec<ResolvedInput>,
}

/// Produced output of a successful attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProducedArtifact {
    pub path: ProjectPath,
    pub content_hash: ContentHash,
    /// How reproducible this output is; recorded in the operation receipt.
    pub reproducibility: Reproducibility,
    /// Deterministic observations worth keeping with the receipt (counts,
    /// acceptance diagnostics, seeds). Never secrets.
    pub observations: BTreeMap<String, String>,
}

impl ProducedArtifact {
    pub fn new(
        path: ProjectPath,
        content_hash: ContentHash,
        reproducibility: Reproducibility,
    ) -> Self {
        Self {
            path,
            content_hash,
            reproducibility,
            observations: BTreeMap::new(),
        }
    }
}

/// An output recorded during a run, handed to a [`RunObserver`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedOutput {
    pub artifact: ArtifactRecord,
    pub produced: ProducedArtifact,
    pub provider: Option<ProviderDeclaration>,
}

/// Persistence hooks. `reserve_attempt` and `verify_output` may be called from
/// worker threads (serialized by the runner); `wave_recorded` is called after
/// every wave with the manifest already containing that wave's records.
/// Returning an error from `wave_recorded` stops the run; the in-memory
/// manifest stays valid.
pub trait RunObserver: Send {
    /// Durably record the cumulative attempt count for an operation identity.
    /// Called before each dispatch with the count including the attempt about
    /// to start (so an interrupted process cannot spend it twice), and again
    /// with the lower count when an unsupported outcome releases it. `0` means
    /// no attempts are charged. A rejection before dispatch prevents the
    /// attempt.
    fn reserve_attempt(
        &mut self,
        _operation: &str,
        _identity: &ContentHash,
        _attempts: u32,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Check a produced output of artifact `kind` before it is accepted (for
    /// example by hashing the written file and the sidecars it lists). A
    /// rejection is a failed attempt for that provider, so fallback providers
    /// still apply.
    fn verify_output(
        &mut self,
        _kind: ArtifactKind,
        _produced: &ProducedArtifact,
    ) -> Result<(), String> {
        Ok(())
    }

    fn wave_recorded(
        &mut self,
        manifest: &SceneProjectManifest,
        recorded: &[RecordedOutput],
    ) -> Result<(), SceneProjectError>;
}

struct NoObserver;

impl RunObserver for NoObserver {
    fn wave_recorded(
        &mut self,
        _manifest: &SceneProjectManifest,
        _recorded: &[RecordedOutput],
    ) -> Result<(), SceneProjectError> {
        Ok(())
    }
}

/// Result of one attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttemptOutcome {
    Succeeded(ProducedArtifact),
    /// Transient failure; the same provider may be retried within the bound.
    RetryableFailure(String),
    /// Permanent failure for this provider; the next fallback provider (if
    /// any) is tried within the bound.
    Failed(String),
    /// The provider/build cannot perform this operation; the next fallback
    /// provider (if any) is tried within the bound.
    Unsupported(String),
    /// The attempt observed cancellation and stopped.
    Canceled,
}

/// Executes single operation attempts. Implementations must be safe to call
/// concurrently for independent operations.
pub trait OperationExecutor: Sync {
    fn execute(&self, request: &OperationRequest, cancel: &CancellationToken) -> AttemptOutcome;
}

/// Final state of one operation after a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperationState {
    /// A current recorded artifact existed; nothing was executed.
    Reused,
    /// Executed and recorded.
    Succeeded {
        attempts: u32,
        provider: Option<String>,
    },
    /// The last attempt failed with a transient error and the attempt bound is
    /// exhausted; a later run may retry.
    RetryableFailure { attempts: u32, message: String },
    /// Every reachable provider failed permanently or the bound is exhausted.
    Failed { attempts: u32, message: String },
    /// No reachable provider/build supports the operation.
    Unsupported { attempts: u32, message: String },
    /// Canceled before or during execution.
    Canceled { attempts: u32 },
    /// Not executed because an upstream operation did not complete.
    Blocked { upstream: Vec<String> },
}

impl OperationState {
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Reused | Self::Succeeded { .. })
    }

    /// Attempts spent on the operation's current identity, including attempts
    /// from earlier runs.
    pub fn attempts(&self) -> u32 {
        match self {
            Self::Succeeded { attempts, .. }
            | Self::RetryableFailure { attempts, .. }
            | Self::Failed { attempts, .. }
            | Self::Unsupported { attempts, .. }
            | Self::Canceled { attempts } => *attempts,
            Self::Reused | Self::Blocked { .. } => 0,
        }
    }

    /// One-line human-readable diagnostic for status/inspect output.
    pub fn diagnostic(&self, operation: &str) -> String {
        match self {
            Self::Reused => format!("{operation}: reused current artifact"),
            Self::Succeeded { attempts, provider } => match provider {
                Some(provider) => format!(
                    "{operation}: succeeded via provider `{provider}` after {attempts} attempt(s)"
                ),
                None => format!("{operation}: succeeded after {attempts} attempt(s)"),
            },
            Self::RetryableFailure { attempts, message } => format!(
                "{operation}: transient failure after {attempts} attempt(s): {message}"
            ),
            Self::Failed { attempts, message } => {
                format!("{operation}: failed after {attempts} attempt(s): {message}")
            }
            Self::Unsupported { attempts, message } => format!(
                "{operation}: unsupported by every reachable provider after {attempts} attempt(s): {message}"
            ),
            Self::Canceled { attempts } => {
                format!("{operation}: canceled after {attempts} attempt(s)")
            }
            Self::Blocked { upstream } => format!(
                "{operation}: blocked by incomplete upstream {}",
                upstream.join(", ")
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunReport {
    pub states: BTreeMap<String, OperationState>,
    /// Operations that made at least one executor attempt during this run.
    pub executed: BTreeSet<String>,
}

impl RunReport {
    pub fn is_complete(&self) -> bool {
        self.states.values().all(OperationState::is_complete)
    }

    /// Diagnostics for every operation that did not complete, in operation-id
    /// order; empty when the build is complete.
    pub fn diagnostics(&self) -> Vec<String> {
        self.states
            .iter()
            .filter(|(_, state)| !state.is_complete())
            .map(|(operation, state)| state.diagnostic(operation))
            .collect()
    }

    /// Number of operations that were actually executed (at least one attempt).
    pub fn executed(&self) -> usize {
        self.executed.len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunOptions {
    /// Upper bound on concurrently executing operations. `1` runs inline on
    /// the calling thread (required on targets without threads).
    pub max_concurrency: usize,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self { max_concurrency: 1 }
    }
}

/// Run every declared operation that is not already current, recording
/// produced artifacts into `manifest`. The manifest stays valid throughout and
/// can be persisted after the call to resume later.
pub fn run(
    manifest: &mut SceneProjectManifest,
    executor: &dyn OperationExecutor,
    options: RunOptions,
    cancel: &CancellationToken,
) -> Result<RunReport, SceneProjectError> {
    run_observed(manifest, executor, options, cancel, &mut NoObserver)
}

/// [`run`] with a persistence hook called after each recorded wave.
pub fn run_observed(
    manifest: &mut SceneProjectManifest,
    executor: &dyn OperationExecutor,
    options: RunOptions,
    cancel: &CancellationToken,
    observer: &mut dyn RunObserver,
) -> Result<RunReport, SceneProjectError> {
    if options.max_concurrency == 0 {
        return Err(SceneProjectError::Invalid(
            "runner max_concurrency must be positive".into(),
        ));
    }
    manifest.validate()?;
    let mut states: BTreeMap<String, OperationState> = BTreeMap::new();
    let mut executed: BTreeSet<String> = BTreeSet::new();

    loop {
        let identities = manifest.operation_identities()?;
        let current: BTreeSet<&str> = manifest
            .artifacts
            .iter()
            .filter(|artifact| identities[&artifact.produced_by] == artifact.operation_identity)
            .map(|artifact| artifact.produced_by.as_str())
            .collect();

        // Classify every unresolved operation; collect the ready wave.
        let mut wave: Vec<OperationRequest> = Vec::new();
        let mut newly_resolved: Vec<(String, OperationState)> = Vec::new();
        for operation in &manifest.operations {
            if states.contains_key(&operation.id) {
                continue;
            }
            if current.contains(operation.id.as_str()) {
                newly_resolved.push((operation.id.clone(), OperationState::Reused));
                continue;
            }
            let upstream: Vec<&str> = operation
                .inputs
                .iter()
                .filter_map(|input| match input {
                    OperationInput::Operation(id) => Some(id.as_str()),
                    OperationInput::Media(_) => None,
                })
                .collect();
            let failed: Vec<String> = upstream
                .iter()
                .filter(|id| states.get(**id).is_some_and(|state| !state.is_complete()))
                .map(|id| (*id).to_owned())
                .collect();
            if !failed.is_empty() {
                newly_resolved.push((
                    operation.id.clone(),
                    OperationState::Blocked { upstream: failed },
                ));
                continue;
            }
            if upstream.iter().all(|id| current.contains(id)) {
                let used = prior_attempts(manifest, &operation.id, &identities[&operation.id]);
                if used >= operation.max_attempts {
                    newly_resolved.push((
                        operation.id.clone(),
                        OperationState::Failed {
                            attempts: used,
                            message: format!(
                                "attempt budget exhausted by earlier runs ({used} of {})",
                                operation.max_attempts
                            ),
                        },
                    ));
                    continue;
                }
                let mut request = request_for(manifest, operation, &identities);
                request.attempt = used;
                wave.push(request);
            }
        }
        wave.sort_by(|a, b| a.operation.id.cmp(&b.operation.id));
        let progressed = !newly_resolved.is_empty() || !wave.is_empty();
        states.extend(newly_resolved);
        if wave.is_empty() {
            if progressed {
                continue;
            }
            break;
        }

        let outcomes = {
            let observer = Mutex::new(&mut *observer);
            execute_wave(manifest, executor, options, cancel, &observer, &wave)
        };
        let mut recorded_outputs = Vec::new();
        // Record in operation-id order so the manifest is independent of
        // completion order.
        for (request, (mut state, produced, charged)) in wave.iter().zip(outcomes) {
            let id = &request.operation.id;
            if state.attempts() > request.attempt {
                executed.insert(id.clone());
            }
            manifest
                .attempt_usage
                .retain(|usage| usage.operation != *id);
            let provider_id = match &state {
                OperationState::Succeeded { provider, .. } => provider.clone(),
                _ => None,
            };
            let recorded = match produced {
                Some(produced) => match record(
                    manifest,
                    id,
                    &request.identity,
                    produced.clone(),
                    provider_id,
                ) {
                    Ok(()) => {
                        let provider = match &state {
                            OperationState::Succeeded {
                                provider: Some(provider),
                                ..
                            } => manifest
                                .provider_policy
                                .providers
                                .iter()
                                .find(|candidate| candidate.id == *provider)
                                .cloned(),
                            _ => None,
                        };
                        let artifact = manifest
                            .artifacts
                            .iter()
                            .find(|artifact| artifact.produced_by == *id)
                            .expect("just recorded")
                            .clone();
                        recorded_outputs.push(RecordedOutput {
                            artifact,
                            produced,
                            provider,
                        });
                        true
                    }
                    Err(error) => {
                        // The attempt was spent even though its output was
                        // rejected; keep it in the budget.
                        state = OperationState::Failed {
                            attempts: state.attempts(),
                            message: format!("output could not be recorded: {error}"),
                        };
                        false
                    }
                },
                None => false,
            };
            // Replace the reservations with the attempts that actually count.
            let used = request.attempt + charged;
            if !recorded && used > 0 {
                manifest.attempt_usage.push(AttemptUsage {
                    operation: id.clone(),
                    operation_identity: request.identity.clone(),
                    attempts: used,
                });
            }
            states.insert(id.clone(), state);
        }
        manifest.canonicalize();
        observer.wave_recorded(manifest, &recorded_outputs)?;
    }

    // Anything left unresolved was never reachable because of cancellation.
    for operation in &manifest.operations {
        states
            .entry(operation.id.clone())
            .or_insert(OperationState::Canceled { attempts: 0 });
    }
    manifest.canonicalize();
    manifest.validate()?;
    Ok(RunReport { states, executed })
}

fn prior_attempts(manifest: &SceneProjectManifest, operation: &str, identity: &ContentHash) -> u32 {
    manifest
        .attempt_usage
        .iter()
        .find(|usage| usage.operation == operation && usage.operation_identity == *identity)
        .map_or(0, |usage| usage.attempts)
}

fn request_for(
    manifest: &SceneProjectManifest,
    operation: &OperationDeclaration,
    identities: &BTreeMap<String, ContentHash>,
) -> OperationRequest {
    let inputs = operation
        .inputs
        .iter()
        .map(|input| match input {
            OperationInput::Media(id) => ResolvedInput::Media(
                manifest
                    .inputs
                    .iter()
                    .find(|media| &media.id == id)
                    .expect("validated media reference")
                    .clone(),
            ),
            OperationInput::Operation(id) => ResolvedInput::Artifact(
                manifest
                    .artifacts
                    .iter()
                    .find(|artifact| {
                        &artifact.produced_by == id && artifact.operation_identity == identities[id]
                    })
                    .expect("upstream is current")
                    .clone(),
            ),
        })
        .collect();
    OperationRequest {
        operation: operation.clone(),
        identity: identities[&operation.id].clone(),
        provider: None,
        attempt: 0,
        inputs,
    }
}

type OperationResult = (OperationState, Option<ProducedArtifact>, u32);

fn execute_wave(
    manifest: &SceneProjectManifest,
    executor: &dyn OperationExecutor,
    options: RunOptions,
    cancel: &CancellationToken,
    observer: &Mutex<&mut (dyn RunObserver + '_)>,
    wave: &[OperationRequest],
) -> Vec<OperationResult> {
    let providers: Vec<Vec<ProviderDeclaration>> = wave
        .iter()
        .map(|request| {
            manifest
                .reachable_providers(&request.operation)
                .into_iter()
                .cloned()
                .collect()
        })
        .collect();
    if options.max_concurrency == 1 || wave.len() == 1 {
        return wave
            .iter()
            .zip(&providers)
            .map(|(request, providers)| {
                execute_operation(executor, cancel, observer, request, providers)
            })
            .collect();
    }
    let mut results: Vec<Option<OperationResult>> = vec![None; wave.len()];
    let jobs: Vec<usize> = (0..wave.len()).collect();
    for chunk in jobs.chunks(options.max_concurrency) {
        std::thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|&index| {
                    let request = &wave[index];
                    let providers = &providers[index];
                    (
                        index,
                        scope.spawn(move || {
                            execute_operation(executor, cancel, observer, request, providers)
                        }),
                    )
                })
                .collect();
            for (index, handle) in handles {
                results[index] = Some(handle.join().unwrap_or_else(|_| {
                    (
                        OperationState::Failed {
                            attempts: wave[index].attempt + 1,
                            message: "executor panicked".into(),
                        },
                        None,
                        1,
                    )
                }));
            }
        });
    }
    results
        .into_iter()
        .map(|result| result.expect("every wave job completes"))
        .collect()
}

/// Run one operation within its attempt bound, walking reachable providers in
/// order: retryable failures retry the same provider, permanent failures,
/// rejected outputs and unsupported outcomes advance to the next provider.
/// Returns the final state, the accepted output, and the number of attempts
/// that count against the budget (unsupported outcomes do not: no provider or
/// build performed work).
fn execute_operation(
    executor: &dyn OperationExecutor,
    cancel: &CancellationToken,
    observer: &Mutex<&mut (dyn RunObserver + '_)>,
    base: &OperationRequest,
    providers: &[ProviderDeclaration],
) -> OperationResult {
    let max_attempts = base.operation.max_attempts;
    let lock = || {
        observer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };
    let mut provider_index = 0;
    let mut attempts = base.attempt;
    let mut charged = 0;
    let mut last: Option<AttemptOutcome> = None;
    // The budget bounds charged attempts; unsupported probes advance the
    // provider walk without consuming it.
    while base.attempt + charged < max_attempts {
        if cancel.is_canceled() {
            return (OperationState::Canceled { attempts }, None, charged);
        }
        let provider = if providers.is_empty() {
            None
        } else if let Some(provider) = providers.get(provider_index) {
            Some(provider.clone())
        } else {
            break;
        };
        if let Err(message) = lock().reserve_attempt(
            &base.operation.id,
            &base.identity,
            base.attempt + charged + 1,
        ) {
            last = Some(AttemptOutcome::Failed(format!(
                "attempt could not be reserved: {message}"
            )));
            break;
        }
        attempts += 1;
        let mut request = base.clone();
        request.provider = provider.clone();
        request.attempt = attempts;
        let mut outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.execute(&request, cancel)
        }))
        .unwrap_or_else(|_| AttemptOutcome::Failed("executor panicked".into()));
        if matches!(outcome, AttemptOutcome::Unsupported(_)) {
            // Release the provisional reservation: nothing was performed.
            // A failure to persist the release only over-counts, which is safe.
            let _ =
                lock().reserve_attempt(&base.operation.id, &base.identity, base.attempt + charged);
        } else {
            charged += 1;
        }
        if let AttemptOutcome::Succeeded(produced) = &outcome {
            if let Err(message) = lock().verify_output(base.operation.kind.output(), produced) {
                outcome = AttemptOutcome::Failed(format!("output verification failed: {message}"));
            }
        }
        match &outcome {
            AttemptOutcome::Succeeded(produced) => {
                return (
                    OperationState::Succeeded {
                        attempts,
                        provider: provider.map(|provider| provider.id),
                    },
                    Some(produced.clone()),
                    charged,
                );
            }
            AttemptOutcome::Canceled => {
                return (OperationState::Canceled { attempts }, None, charged)
            }
            AttemptOutcome::RetryableFailure(_) => {}
            AttemptOutcome::Failed(_) | AttemptOutcome::Unsupported(_) => {
                if providers.len() <= 1 {
                    last = Some(outcome);
                    break;
                }
                provider_index += 1;
            }
        }
        last = Some(outcome);
    }
    let state = match last {
        Some(AttemptOutcome::RetryableFailure(message)) => {
            OperationState::RetryableFailure { attempts, message }
        }
        Some(AttemptOutcome::Unsupported(message)) => {
            OperationState::Unsupported { attempts, message }
        }
        Some(AttemptOutcome::Failed(message)) => OperationState::Failed { attempts, message },
        _ => OperationState::Failed {
            attempts,
            message: "no attempt was possible".into(),
        },
    };
    (state, None, charged)
}

fn record(
    manifest: &mut SceneProjectManifest,
    operation_id: &str,
    identity: &ContentHash,
    produced: ProducedArtifact,
    provider: Option<String>,
) -> Result<(), SceneProjectError> {
    let kind = manifest
        .operations
        .iter()
        .find(|operation| operation.id == operation_id)
        .expect("declared operation")
        .kind
        .output();
    // Validate a candidate so a rejected record leaves the manifest untouched.
    let mut candidate = manifest.clone();
    let previous = candidate
        .artifacts
        .iter()
        .position(|artifact| artifact.produced_by == operation_id);
    let id = match previous {
        Some(index) => candidate.artifacts.remove(index).id,
        None => artifact_id(&candidate, operation_id, identity),
    };
    candidate.artifacts.push(ArtifactRecord {
        id,
        kind,
        produced_by: operation_id.to_owned(),
        operation_identity: identity.clone(),
        path: produced.path,
        content_hash: produced.content_hash,
        provider,
    });
    candidate.validate().map_err(|error| {
        SceneProjectError::Invalid(format!(
            "recording the output of operation `{operation_id}` would make the manifest invalid: {error}"
        ))
    })?;
    *manifest = candidate;
    Ok(())
}

/// Deterministic, valid, collision-free artifact id for a new record.
pub(crate) fn artifact_id(
    manifest: &SceneProjectManifest,
    operation_id: &str,
    identity: &ContentHash,
) -> String {
    let taken = |id: &str| {
        manifest.inputs.iter().any(|item| item.id == id)
            || manifest.operations.iter().any(|item| item.id == id)
            || manifest
                .provider_policy
                .providers
                .iter()
                .any(|item| item.id == id)
            || manifest.artifacts.iter().any(|item| item.id == id)
            || manifest.exports.iter().any(|item| item.id == id)
    };
    let readable = format!("{operation_id}.output");
    if readable.len() <= 64 && !taken(&readable) {
        return readable;
    }
    let digest = &identity.as_str()["sha256:".len()..];
    let mut suffix = 0usize;
    loop {
        let candidate = if suffix == 0 {
            format!("artifact-{}", &digest[..16])
        } else {
            format!("artifact-{}-{suffix}", &digest[..16])
        };
        if !taken(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

#[cfg(test)]
#[path = "scene_runner_tests.rs"]
mod tests;
