//! Receipt-backed build state on disk: verification, reconciliation, and
//! resumable builds.
//!
//! A project lives in the directory containing its manifest. Every recorded
//! artifact has an operation receipt under `.video-to-3d/receipts/` that states
//! the request identity, provider and revision, input hashes, output hash and
//! size, deterministic observations, and a reproducibility class.
//!
//! Before a build resumes, [`ProjectStore::reconcile`] checks every recorded
//! artifact against its file and receipt instead of assuming the previous
//! process finished cleanly. Records whose output is missing or corrupt, or
//! whose receipt is missing or disagrees, are dropped; because downstream
//! identities include upstream content hashes, exactly the affected operation
//! and the descendants whose inputs actually change are rebuilt.
//!
//! The content-hash cache under `.video-to-3d/cache/` only skips re-hashing
//! unchanged files in [`VerifyMode::Cached`]; deleting or corrupting it never
//! changes a build result, and [`VerifyMode::Full`] ignores it.

use crate::scene_project::{
    ArtifactRecord, AttemptUsage, ContentHash, OperationInput, OperationKind, ProjectPath,
    SceneProjectError, SceneProjectManifest, STATE_DIRECTORY,
};
use crate::scene_runner::{
    run_observed, CancellationToken, OperationExecutor, ProducedArtifact, RecordedOutput,
    RunObserver, RunOptions, RunReport,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub const RECEIPT_SCHEMA_VERSION: u32 = 1;
const CACHE_SCHEMA_VERSION: u32 = 1;

/// How reproducible an operation's output is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reproducibility {
    /// Same request identity always yields the same bytes.
    Deterministic,
    /// Stochastic but reproducible from recorded seeds/observations.
    SeededStochastic,
    /// May differ between runs (for example remote generative providers).
    NonDeterministic,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptProvider {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptInput {
    pub input: OperationInput,
    pub content_hash: ContentHash,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptOutput {
    pub path: ProjectPath,
    pub content_hash: ContentHash,
    pub byte_length: u64,
}

/// Authoritative record of one executed operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationReceipt {
    pub schema_version: u32,
    pub operation: String,
    pub kind: OperationKind,
    pub operation_identity: ContentHash,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ReceiptProvider>,
    pub inputs: Vec<ReceiptInput>,
    pub output: ReceiptOutput,
    pub reproducibility: Reproducibility,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub observations: BTreeMap<String, String>,
}

impl OperationReceipt {
    pub fn path_for(operation: &str) -> ProjectPath {
        ProjectPath::new(format!("{STATE_DIRECTORY}/receipts/{operation}.json"))
            .expect("operation ids are valid path components")
    }
}

/// Why a recorded artifact was dropped during reconciliation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Invalidation {
    MissingOutput,
    CorruptOutput {
        expected: ContentHash,
        actual: ContentHash,
    },
    MissingReceipt,
    UnreadableReceipt(String),
    ReceiptMismatch(String),
}

impl Invalidation {
    pub fn describe(&self) -> String {
        match self {
            Self::MissingOutput => "output file is missing".into(),
            Self::CorruptOutput { expected, actual } => format!(
                "output hash {} does not match recorded {}",
                actual.as_str(),
                expected.as_str()
            ),
            Self::MissingReceipt => "operation receipt is missing".into(),
            Self::UnreadableReceipt(message) => {
                format!("operation receipt is unreadable: {message}")
            }
            Self::ReceiptMismatch(message) => format!("operation receipt disagrees: {message}"),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReconcileReport {
    /// Operations whose recorded artifact, file, and receipt all agree.
    pub verified: Vec<String>,
    /// Operations whose record was dropped, with the reason.
    pub invalidated: Vec<(String, Invalidation)>,
    /// Receipts with no matching record (for example a crash between writing
    /// the receipt and saving the manifest). They are ignored and overwritten
    /// when the operation runs again.
    pub orphan_receipts: Vec<String>,
}

impl ReconcileReport {
    pub fn diagnostics(&self) -> Vec<String> {
        let mut lines: Vec<String> = self
            .invalidated
            .iter()
            .map(|(operation, reason)| {
                format!(
                    "{operation}: recorded artifact invalidated: {}",
                    reason.describe()
                )
            })
            .collect();
        lines.extend(
            self.orphan_receipts.iter().map(|operation| {
                format!("{operation}: ignoring receipt without a recorded artifact")
            }),
        );
        lines
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyMode {
    /// Hash every file. Used before building.
    Full,
    /// Reuse cached hashes for files whose size and modification time are
    /// unchanged. Suitable for read-only status reporting.
    Cached,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildReport {
    pub reconcile: ReconcileReport,
    pub run: RunReport,
}

/// A project directory: the manifest file plus everything it references.
#[derive(Clone, Debug)]
pub struct ProjectStore {
    root: PathBuf,
    manifest_path: PathBuf,
}

impl ProjectStore {
    /// Open the project whose manifest is at `manifest_path`. The project root
    /// is the manifest's directory; the working directory is never consulted.
    pub fn open(manifest_path: &Path) -> Result<(Self, SceneProjectManifest), SceneProjectError> {
        let manifest_path = fs::canonicalize(manifest_path).map_err(|error| {
            SceneProjectError::Malformed(format!(
                "cannot open {}: {error}",
                manifest_path.display()
            ))
        })?;
        let root = manifest_path
            .parent()
            .ok_or_else(|| SceneProjectError::Malformed("manifest has no parent directory".into()))?
            .to_path_buf();
        let manifest = SceneProjectManifest::load(&manifest_path)?;
        Ok((
            Self {
                root,
                manifest_path,
            },
            manifest,
        ))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn resolve(&self, path: &ProjectPath) -> PathBuf {
        path.resolve(&self.root)
    }

    /// Atomically replace the manifest file with the canonical document.
    pub fn save_manifest(&self, manifest: &SceneProjectManifest) -> Result<(), SceneProjectError> {
        write_atomic(
            &self.manifest_path,
            manifest.to_canonical_json()?.as_bytes(),
        )
    }

    pub fn read_receipt(&self, operation: &str) -> Result<Option<OperationReceipt>, String> {
        let path = self.resolve(&OperationReceipt::path_for(operation));
        let document = match fs::read_to_string(&path) {
            Ok(document) => document,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.to_string()),
        };
        let receipt: OperationReceipt =
            serde_json::from_str(&document).map_err(|error| error.to_string())?;
        if receipt.schema_version != RECEIPT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported receipt schema version {}",
                receipt.schema_version
            ));
        }
        Ok(Some(receipt))
    }

    pub fn write_receipt(&self, receipt: &OperationReceipt) -> Result<(), SceneProjectError> {
        let mut document = serde_json::to_string_pretty(receipt)
            .map_err(|error| SceneProjectError::Malformed(error.to_string()))?;
        document.push('\n');
        write_atomic(
            &self.resolve(&OperationReceipt::path_for(&receipt.operation)),
            document.as_bytes(),
        )
    }

    /// Verify media inputs and every recorded artifact. Media that does not
    /// match its declared identity is a hard error (the declared input changed);
    /// artifacts that fail verification are dropped from `manifest`.
    pub fn reconcile(
        &self,
        manifest: &mut SceneProjectManifest,
        mode: VerifyMode,
    ) -> Result<ReconcileReport, SceneProjectError> {
        manifest.validate()?;
        let mut cache = HashCache::load(&self.root, mode);
        for media in &manifest.inputs {
            match cache.hash(&self.resolve(&media.path), media.path.as_str()) {
                Some((hash, length))
                    if hash == media.content_hash && length == media.byte_length => {}
                Some(_) => {
                    return Err(SceneProjectError::Invalid(format!(
                    "media input `{}` at `{}` does not match its declared content hash and size",
                    media.id,
                    media.path.as_str()
                )))
                }
                None => {
                    return Err(SceneProjectError::Invalid(format!(
                        "media input `{}` at `{}` is missing or unreadable",
                        media.id,
                        media.path.as_str()
                    )))
                }
            }
        }

        let mut report = ReconcileReport::default();
        let mut keep = Vec::with_capacity(manifest.artifacts.len());
        for artifact in &manifest.artifacts {
            match self.verify_artifact(manifest, artifact, &mut cache) {
                Ok(()) => {
                    report.verified.push(artifact.produced_by.clone());
                    keep.push(artifact.clone());
                }
                Err(reason) => report
                    .invalidated
                    .push((artifact.produced_by.clone(), reason)),
            }
        }
        manifest.artifacts = keep;

        let receipts_dir = self.root.join(STATE_DIRECTORY).join("receipts");
        if let Ok(entries) = fs::read_dir(&receipts_dir) {
            let mut orphans: Vec<String> = entries
                .filter_map(Result::ok)
                .filter_map(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .and_then(|name| name.strip_suffix(".json"))
                        .map(str::to_owned)
                })
                .filter(|operation| {
                    !manifest
                        .artifacts
                        .iter()
                        .any(|artifact| artifact.produced_by == *operation)
                        && !report
                            .invalidated
                            .iter()
                            .any(|(invalidated, _)| invalidated == operation)
                })
                .collect();
            orphans.sort();
            report.orphan_receipts = orphans;
        }
        // Only full verification refreshes the cache; cached verification is
        // read-only so status queries never mutate the project.
        if mode == VerifyMode::Full {
            cache.save(&self.root);
        }
        manifest.validate()?;
        Ok(report)
    }

    fn verify_artifact(
        &self,
        manifest: &SceneProjectManifest,
        artifact: &ArtifactRecord,
        cache: &mut HashCache,
    ) -> Result<(), Invalidation> {
        let receipt = match self.read_receipt(&artifact.produced_by) {
            Ok(Some(receipt)) => receipt,
            Ok(None) => return Err(Invalidation::MissingReceipt),
            Err(message) => return Err(Invalidation::UnreadableReceipt(message)),
        };
        if receipt.operation != artifact.produced_by {
            return Err(Invalidation::ReceiptMismatch(format!(
                "receipt names operation `{}`",
                receipt.operation
            )));
        }
        if receipt.operation_identity != artifact.operation_identity {
            return Err(Invalidation::ReceiptMismatch(
                "operation identity differs from the recorded artifact".into(),
            ));
        }
        if receipt.output.path != artifact.path
            || receipt.output.content_hash != artifact.content_hash
        {
            return Err(Invalidation::ReceiptMismatch(
                "output path or hash differs from the recorded artifact".into(),
            ));
        }
        verify_receipt_provenance(manifest, artifact, &receipt)?;
        let Some((actual, length)) =
            cache.hash(&self.resolve(&artifact.path), artifact.path.as_str())
        else {
            return Err(Invalidation::MissingOutput);
        };
        if actual != artifact.content_hash || length != receipt.output.byte_length {
            return Err(Invalidation::CorruptOutput {
                expected: artifact.content_hash.clone(),
                actual,
            });
        }
        Ok(())
    }

    /// Reconcile, then run every operation that is not current, persisting
    /// receipts and the manifest after each recorded wave so an interrupted
    /// build resumes from the last verified boundary.
    pub fn build(
        &self,
        manifest: &mut SceneProjectManifest,
        executor: &dyn OperationExecutor,
        options: RunOptions,
        cancel: &CancellationToken,
    ) -> Result<BuildReport, SceneProjectError> {
        let reconcile = self.reconcile(manifest, VerifyMode::Full)?;
        self.save_manifest(manifest)?;
        let mut persister = Persister {
            store: self,
            snapshot: manifest.clone(),
        };
        let run = run_observed(manifest, executor, options, cancel, &mut persister)?;
        self.save_manifest(manifest)?;
        Ok(BuildReport { reconcile, run })
    }
}

/// Every receipt field derivable from the manifest must agree with it, so a
/// copied or edited receipt cannot misstate kind, provider, or inputs.
fn verify_receipt_provenance(
    manifest: &SceneProjectManifest,
    artifact: &ArtifactRecord,
    receipt: &OperationReceipt,
) -> Result<(), Invalidation> {
    let mismatch = |message: &str| Err(Invalidation::ReceiptMismatch(message.into()));
    let Some(operation) = manifest
        .operations
        .iter()
        .find(|operation| operation.id == artifact.produced_by)
    else {
        return mismatch("operation is not declared");
    };
    if receipt.kind != operation.kind {
        return mismatch("operation kind differs from the declaration");
    }
    if receipt.provider.as_ref().map(|provider| &provider.id) != artifact.provider.as_ref() {
        return mismatch("provider differs from the provider recorded with the artifact");
    }
    let reachable = manifest.reachable_providers(operation);
    match (&receipt.provider, reachable.is_empty()) {
        (None, true) => {}
        (Some(provider), false) => {
            if !reachable.iter().any(|candidate| {
                candidate.id == provider.id && candidate.revision == provider.revision
            }) {
                return mismatch("provider or revision is not reachable for this operation");
            }
        }
        (None, false) => return mismatch("provider operation receipt names no provider"),
        (Some(_), true) => return mismatch("built-in operation receipt names a provider"),
    }
    let mut expected = Vec::with_capacity(operation.inputs.len());
    for input in &operation.inputs {
        let content_hash = match input {
            OperationInput::Media(id) => manifest
                .inputs
                .iter()
                .find(|media| media.id == *id)
                .map(|media| media.content_hash.clone()),
            OperationInput::Operation(id) => manifest
                .artifacts
                .iter()
                .find(|upstream| upstream.produced_by == *id)
                .map(|upstream| upstream.content_hash.clone()),
        };
        let Some(content_hash) = content_hash else {
            return mismatch("an input has no recorded content");
        };
        expected.push(ReceiptInput {
            input: input.clone(),
            content_hash,
        });
    }
    let mut actual = receipt.inputs.clone();
    actual.sort_by(|a, b| a.input.cmp(&b.input));
    expected.sort_by(|a, b| a.input.cmp(&b.input));
    if actual != expected {
        return mismatch("input hashes differ from the recorded inputs");
    }
    Ok(())
}

/// Persists reservations, receipts and the manifest. `snapshot` mirrors the
/// last persisted manifest so reservations can be saved between waves.
struct Persister<'a> {
    store: &'a ProjectStore,
    snapshot: SceneProjectManifest,
}

impl RunObserver for Persister<'_> {
    fn reserve_attempt(
        &mut self,
        operation: &str,
        identity: &ContentHash,
        attempts: u32,
    ) -> Result<(), String> {
        let usage = &mut self.snapshot.attempt_usage;
        usage.retain(|usage| usage.operation != operation);
        usage.push(AttemptUsage {
            operation: operation.to_owned(),
            operation_identity: identity.clone(),
            attempts,
        });
        self.store
            .save_manifest(&self.snapshot)
            .map_err(|error| error.to_string())
    }

    fn verify_output(&mut self, produced: &ProducedArtifact) -> Result<(), String> {
        let Some((hash, _)) = hash_file(&self.store.resolve(&produced.path)) else {
            return Err(format!(
                "output `{}` is missing or unreadable",
                produced.path.as_str()
            ));
        };
        if hash != produced.content_hash {
            return Err(format!(
                "output `{}` hashes to {}, not the reported {}",
                produced.path.as_str(),
                hash.as_str(),
                produced.content_hash.as_str()
            ));
        }
        Ok(())
    }

    fn wave_recorded(
        &mut self,
        manifest: &SceneProjectManifest,
        recorded: &[RecordedOutput],
    ) -> Result<(), SceneProjectError> {
        // Receipts first: a crash before the manifest is saved leaves only
        // orphan receipts, never a record without a receipt.
        for output in recorded {
            let receipt = receipt_for(self.store, manifest, output)?;
            self.store.write_receipt(&receipt)?;
        }
        self.snapshot = manifest.clone();
        self.store.save_manifest(manifest)
    }
}

fn receipt_for(
    store: &ProjectStore,
    manifest: &SceneProjectManifest,
    output: &RecordedOutput,
) -> Result<OperationReceipt, SceneProjectError> {
    let artifact = &output.artifact;
    let operation = manifest
        .operations
        .iter()
        .find(|operation| operation.id == artifact.produced_by)
        .expect("recorded operation is declared");
    let identities = manifest.operation_identities()?;
    let inputs = operation
        .inputs
        .iter()
        .map(|input| {
            let content_hash = match input {
                OperationInput::Media(id) => manifest
                    .inputs
                    .iter()
                    .find(|media| media.id == *id)
                    .map(|media| media.content_hash.clone()),
                OperationInput::Operation(id) => manifest
                    .artifacts
                    .iter()
                    .find(|upstream| {
                        upstream.produced_by == *id && upstream.operation_identity == identities[id]
                    })
                    .map(|upstream| upstream.content_hash.clone()),
            };
            content_hash
                .map(|content_hash| ReceiptInput {
                    input: input.clone(),
                    content_hash,
                })
                .ok_or_else(|| {
                    SceneProjectError::Invalid(format!(
                        "input {input:?} of `{}` has no current content",
                        operation.id
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let byte_length = fs::metadata(store.resolve(&artifact.path))
        .map(|metadata| metadata.len())
        .map_err(|error| {
            SceneProjectError::Invalid(format!(
                "output of `{}` at `{}` cannot be read: {error}",
                operation.id,
                artifact.path.as_str()
            ))
        })?;
    Ok(OperationReceipt {
        schema_version: RECEIPT_SCHEMA_VERSION,
        operation: operation.id.clone(),
        kind: operation.kind,
        operation_identity: artifact.operation_identity.clone(),
        provider: output.provider.as_ref().map(|provider| ReceiptProvider {
            id: provider.id.clone(),
            revision: provider.revision.clone(),
        }),
        inputs,
        output: ReceiptOutput {
            path: artifact.path.clone(),
            content_hash: artifact.content_hash.clone(),
            byte_length,
        },
        reproducibility: output.produced.reproducibility,
        observations: output.produced.observations.clone(),
    })
}

/// Stream a file through SHA-256. Returns `None` when it cannot be read.
pub fn hash_file(path: &Path) -> Option<(ContentHash, u64)> {
    let mut file = fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 16];
    let mut length = 0u64;
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        length += read as u64;
    }
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Some((ContentHash::new(format!("sha256:{digest}")).ok()?, length))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), SceneProjectError> {
    let io = |error: std::io::Error| {
        SceneProjectError::Malformed(format!("cannot write {}: {error}", path.display()))
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(io)?;
    }
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    {
        let mut file = fs::File::create(&temporary).map_err(io)?;
        file.write_all(bytes).map_err(io)?;
        file.sync_all().map_err(io)?;
    }
    fs::rename(&temporary, path).map_err(io)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CacheEntry {
    byte_length: u64,
    modified_nanos: u128,
    content_hash: ContentHash,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct CacheDocument {
    schema_version: u32,
    entries: BTreeMap<String, CacheEntry>,
}

/// Size/mtime-keyed hash cache. Acceleration only: any problem reading or
/// writing it is ignored and falls back to hashing.
struct HashCache {
    mode: VerifyMode,
    document: CacheDocument,
}

impl HashCache {
    fn path(root: &Path) -> PathBuf {
        root.join(STATE_DIRECTORY)
            .join("cache")
            .join("content-hashes.json")
    }

    fn load(root: &Path, mode: VerifyMode) -> Self {
        let document = fs::read_to_string(Self::path(root))
            .ok()
            .and_then(|document| serde_json::from_str::<CacheDocument>(&document).ok())
            .filter(|document| document.schema_version == CACHE_SCHEMA_VERSION)
            .unwrap_or(CacheDocument {
                schema_version: CACHE_SCHEMA_VERSION,
                entries: BTreeMap::new(),
            });
        Self { mode, document }
    }

    fn hash(&mut self, path: &Path, key: &str) -> Option<(ContentHash, u64)> {
        let metadata = fs::metadata(path).ok()?;
        let modified_nanos = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_nanos());
        if self.mode == VerifyMode::Cached {
            if let Some(entry) = self.document.entries.get(key) {
                if entry.byte_length == metadata.len() && entry.modified_nanos == modified_nanos {
                    return Some((entry.content_hash.clone(), entry.byte_length));
                }
            }
        }
        let (hash, length) = hash_file(path)?;
        self.document.entries.insert(
            key.to_owned(),
            CacheEntry {
                byte_length: length,
                modified_nanos,
                content_hash: hash.clone(),
            },
        );
        Some((hash, length))
    }

    fn save(&self, root: &Path) {
        if let Ok(document) = serde_json::to_vec(&self.document) {
            let _ = write_atomic(&Self::path(root), &document);
        }
    }
}

#[cfg(test)]
#[path = "scene_store_tests.rs"]
mod tests;
