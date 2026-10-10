//! Versioned scene-project manifest.
//!
//! A manifest is the single declarative description of a whole-scene build: the
//! input media identities, the quality mode, the provider policy, the declared
//! operation graph, the scene artifacts already produced, and the export targets.
//! It is sufficient to start a build or to resume one after a process restart,
//! because every artifact produced so far is recorded with its project-relative
//! path and content hash.
//!
//! The manifest is fail-closed: unknown schema versions, unknown fields, unsafe
//! paths, unbounded paid execution, and invalid quality/provider/operation
//! combinations are rejected instead of being interpreted heuristically.
//! Credentials are never stored; providers reference an externally resolved
//! capability name instead.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

pub const SCENE_PROJECT_SCHEMA_VERSION: u32 = 1;
/// Project-relative directory reserved for build state (receipts, caches).
/// Declared manifest paths may not point into it.
pub const STATE_DIRECTORY: &str = ".video-to-3d";
/// Upper bound for declared attempts of one operation (first attempt plus retries).
pub const MAX_OPERATION_ATTEMPTS: u32 = 5;
const MAX_IDENTIFIER_LENGTH: usize = 64;
const MAX_PATH_LENGTH: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SceneProjectError {
    /// The document is not well-formed JSON or does not match the schema shape.
    Malformed(String),
    /// The document declares a schema version this build does not understand.
    UnsupportedSchemaVersion(u64),
    /// The document is well-formed but violates a manifest invariant.
    Invalid(String),
}

impl fmt::Display for SceneProjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(message) => write!(f, "malformed scene-project manifest: {message}"),
            Self::UnsupportedSchemaVersion(version) => write!(
                f,
                "unsupported scene-project schema version {version}; this build supports version {SCENE_PROJECT_SCHEMA_VERSION}"
            ),
            Self::Invalid(message) => write!(f, "invalid scene-project manifest: {message}"),
        }
    }
}

impl std::error::Error for SceneProjectError {}

fn invalid<T>(message: impl Into<String>) -> Result<T, SceneProjectError> {
    Err(SceneProjectError::Invalid(message.into()))
}

/// Declared quality mode. Modes only gate which operation kinds may be declared;
/// they never change the behavior of an operation that is declared.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityMode {
    /// Fast local reconstruction and coarse outputs.
    Preview,
    /// Learned escalation, scene decomposition, and usable completion.
    Standard,
    /// Expensive provider escalation and higher-quality representations.
    Production,
}

/// Kind of a declared build operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    IngestVideo,
    SparseReconstruction,
    DenseReconstruction,
    SurfaceMesh,
    LearnedReconstruction,
    SceneDecomposition,
    GenerativeCompletion,
    TextureBake,
    SplatOptimization,
    SceneAssembly,
}

/// Whether an operation runs the built-in Rust pipeline or a declared provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationExecution {
    /// Built-in, deterministic, local; a provider must not be declared.
    BuiltIn,
    /// Requires a declared provider from the provider policy.
    Provider,
}

impl OperationKind {
    pub const ALL: [OperationKind; 10] = [
        Self::IngestVideo,
        Self::SparseReconstruction,
        Self::DenseReconstruction,
        Self::SurfaceMesh,
        Self::LearnedReconstruction,
        Self::SceneDecomposition,
        Self::GenerativeCompletion,
        Self::TextureBake,
        Self::SplatOptimization,
        Self::SceneAssembly,
    ];

    /// Lowest quality mode in which this operation may be declared.
    pub fn minimum_quality(self) -> QualityMode {
        match self {
            Self::IngestVideo
            | Self::SparseReconstruction
            | Self::DenseReconstruction
            | Self::SurfaceMesh
            | Self::SceneAssembly => QualityMode::Preview,
            Self::LearnedReconstruction
            | Self::SceneDecomposition
            | Self::GenerativeCompletion
            | Self::TextureBake => QualityMode::Standard,
            Self::SplatOptimization => QualityMode::Production,
        }
    }

    pub fn execution(self) -> OperationExecution {
        match self {
            Self::LearnedReconstruction
            | Self::SceneDecomposition
            | Self::GenerativeCompletion
            | Self::SplatOptimization => OperationExecution::Provider,
            _ => OperationExecution::BuiltIn,
        }
    }

    /// Artifact kind this operation produces.
    pub fn output(self) -> ArtifactKind {
        match self {
            Self::IngestVideo => ArtifactKind::Keyframes,
            Self::SparseReconstruction => ArtifactKind::SparseReconstruction,
            Self::DenseReconstruction => ArtifactKind::DenseEvidence,
            Self::SurfaceMesh => ArtifactKind::SurfaceMesh,
            Self::LearnedReconstruction => ArtifactKind::LearnedEvidence,
            Self::SceneDecomposition => ArtifactKind::SceneDecomposition,
            Self::GenerativeCompletion => ArtifactKind::GenerativeCompletion,
            Self::TextureBake => ArtifactKind::SurfaceTextures,
            Self::SplatOptimization => ArtifactKind::SplatField,
            Self::SceneAssembly => ArtifactKind::AssembledScene,
        }
    }

    /// Upstream operation kinds this operation may consume. `IngestVideo` is the
    /// only operation that consumes source media and consumes nothing else.
    pub fn allowed_upstream(self) -> &'static [OperationKind] {
        use OperationKind::*;
        match self {
            IngestVideo => &[],
            SparseReconstruction => &[IngestVideo],
            DenseReconstruction => &[IngestVideo, SparseReconstruction, LearnedReconstruction],
            SurfaceMesh => &[DenseReconstruction, LearnedReconstruction],
            LearnedReconstruction => &[IngestVideo, SparseReconstruction],
            SceneDecomposition => &[IngestVideo, SparseReconstruction],
            GenerativeCompletion => &[
                IngestVideo,
                SparseReconstruction,
                SurfaceMesh,
                SceneDecomposition,
            ],
            TextureBake => &[IngestVideo, SurfaceMesh],
            SplatOptimization => &[IngestVideo, SparseReconstruction, DenseReconstruction],
            SceneAssembly => &[
                SparseReconstruction,
                DenseReconstruction,
                SurfaceMesh,
                LearnedReconstruction,
                SceneDecomposition,
                GenerativeCompletion,
                TextureBake,
                SplatOptimization,
            ],
        }
    }

    /// Revision of the built-in implementation of this operation kind. Bump
    /// it whenever the built-in semantics change so previously recorded
    /// artifacts become stale. Provider operations are versioned by their
    /// provider declaration instead.
    pub fn implementation_revision(self) -> u32 {
        match self {
            // 2: the built-in executor and the versioned surface-textures format.
            Self::TextureBake => 2,
            _ => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::IngestVideo => "ingest_video",
            Self::SparseReconstruction => "sparse_reconstruction",
            Self::DenseReconstruction => "dense_reconstruction",
            Self::SurfaceMesh => "surface_mesh",
            Self::LearnedReconstruction => "learned_reconstruction",
            Self::SceneDecomposition => "scene_decomposition",
            Self::GenerativeCompletion => "generative_completion",
            Self::TextureBake => "texture_bake",
            Self::SplatOptimization => "splat_optimization",
            Self::SceneAssembly => "scene_assembly",
        }
    }
}

/// Kind of a scene artifact produced by an operation. Source media is not an
/// artifact kind, so it can never be named in a cloud-upload allowlist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Keyframes,
    SparseReconstruction,
    DenseEvidence,
    SurfaceMesh,
    LearnedEvidence,
    SceneDecomposition,
    GenerativeCompletion,
    SurfaceTextures,
    SplatField,
    AssembledScene,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    WebBundle,
    Glb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderLocation {
    /// Runs on the user's device (browser, Tauri, or a local native process).
    Local,
    /// Runs on a remote service.
    Cloud,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPolicy {
    /// Only local providers may be referenced.
    LocalOnly,
    /// Cloud providers are allowed when free; local providers are preferred in
    /// fallback order.
    LocalFirst,
    /// Paid cloud providers are allowed within a declared total budget.
    AllowPaidCloud,
}

/// Portable project-relative path: forward slashes, no root, no `.`/`..`
/// components, no empty components, no drive prefixes or backslashes.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ProjectPath(String);

impl ProjectPath {
    pub fn new(path: impl Into<String>) -> Result<Self, SceneProjectError> {
        let path = path.into();
        if path.is_empty() || path.len() > MAX_PATH_LENGTH {
            return invalid(format!(
                "project path must contain 1..={MAX_PATH_LENGTH} bytes"
            ));
        }
        if path.contains('\\') || path.contains(':') || path.chars().any(char::is_control) {
            return invalid(format!(
                "project path `{path}` must use portable forward-slash separators without drive prefixes"
            ));
        }
        if path.starts_with('/') {
            return invalid(format!("project path `{path}` must be project-relative"));
        }
        for component in path.split('/') {
            if component.is_empty() || component == "." || component == ".." {
                return invalid(format!(
                    "project path `{path}` must not contain empty, `.` or `..` components"
                ));
            }
            if !component
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
            {
                return invalid(format!(
                    "project path `{path}` components may only contain ASCII letters, digits, `-`, `_` and `.`"
                ));
            }
            if component.ends_with('.') {
                return invalid(format!(
                    "project path `{path}` components must not end with `.`"
                ));
            }
            let stem = component
                .split('.')
                .next()
                .unwrap_or_default()
                .to_ascii_lowercase();
            let reserved = matches!(stem.as_str(), "con" | "prn" | "aux" | "nul")
                || ((stem.starts_with("com") || stem.starts_with("lpt"))
                    && stem.len() == 4
                    && stem.as_bytes()[3].is_ascii_digit());
            if reserved {
                return invalid(format!("project path `{path}` uses a reserved device name"));
            }
        }
        Ok(Self(path))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Resolve against an explicit project root; never against the ambient
    /// working directory.
    pub fn resolve(&self, project_root: &Path) -> PathBuf {
        self.0
            .split('/')
            .fold(project_root.to_path_buf(), |path, component| {
                path.join(component)
            })
    }
}

impl TryFrom<String> for ProjectPath {
    type Error = SceneProjectError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ProjectPath> for String {
    fn from(value: ProjectPath) -> Self {
        value.0
    }
}

/// `sha256:` followed by 64 lowercase hexadecimal digits.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ContentHash(String);

impl ContentHash {
    pub fn new(value: impl Into<String>) -> Result<Self, SceneProjectError> {
        let value = value.into();
        let valid = value.strip_prefix("sha256:").is_some_and(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        });
        if !valid {
            return invalid(format!(
                "content hash `{value}` must be `sha256:` followed by 64 lowercase hex digits"
            ));
        }
        Ok(Self(value))
    }

    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(format!("sha256:{}", hex(&Sha256::digest(bytes))))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ContentHash {
    type Error = SceneProjectError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ContentHash> for String {
    fn from(value: ContentHash) -> Self {
        value.0
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediaInput {
    pub id: String,
    pub path: ProjectPath,
    pub content_hash: ContentHash,
    pub byte_length: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderDeclaration {
    pub id: String,
    pub location: ProviderLocation,
    /// Operation kinds this provider may serve.
    pub capabilities: Vec<OperationKind>,
    /// Model or service revision; part of the identity of operations using it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Name of an externally resolved credential capability. The secret itself
    /// is never part of the manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_capability: Option<String>,
    /// Declared worst-case cost of one attempt, in the policy's minor cost unit.
    /// Required for cloud providers (`0` declares them free explicitly); local
    /// providers must omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_per_attempt: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderPolicy {
    pub execution: ExecutionPolicy,
    #[serde(default)]
    pub providers: Vec<ProviderDeclaration>,
    /// Provider preference order used for mechanical provider selection.
    #[serde(default)]
    pub fallback_order: Vec<String>,
    /// Artifact kinds that may be sent to cloud providers. Source media can never
    /// be listed.
    #[serde(default)]
    pub cloud_upload_allowlist: Vec<ArtifactKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_cost_per_operation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_total_cost: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum OperationInput {
    Media(String),
    Operation(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationDeclaration {
    pub id: String,
    pub kind: OperationKind,
    pub inputs: Vec<OperationInput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// First attempt plus bounded retries.
    #[serde(default = "default_attempts")]
    pub max_attempts: u32,
}

fn default_attempts() -> u32 {
    1
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRecord {
    pub id: String,
    pub kind: ArtifactKind,
    pub produced_by: String,
    /// Identity of the producing operation at the time the artifact was written.
    pub operation_identity: ContentHash,
    pub path: ProjectPath,
    pub content_hash: ContentHash,
    /// Provider that actually produced the output (`None` for built-in
    /// operations). Receipts must agree with it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportDeclaration {
    pub id: String,
    pub format: ExportFormat,
    pub path: ProjectPath,
    pub includes: Vec<ArtifactKind>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneProjectManifest {
    pub schema_version: u32,
    pub project_id: String,
    pub quality_mode: QualityMode,
    pub inputs: Vec<MediaInput>,
    pub provider_policy: ProviderPolicy,
    pub operations: Vec<OperationDeclaration>,
    pub requested_outputs: Vec<ArtifactKind>,
    #[serde(default)]
    pub artifacts: Vec<ArtifactRecord>,
    #[serde(default)]
    pub exports: Vec<ExportDeclaration>,
    /// Attempts already spent on operations that have not produced a current
    /// artifact. Persisting this keeps `max_attempts` (and therefore the
    /// validated worst-case cost) binding across resumed runs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attempt_usage: Vec<AttemptUsage>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptUsage {
    pub operation: String,
    /// Identity the attempts were spent on; a changed declaration or input
    /// yields a new identity and a fresh budget.
    pub operation_identity: ContentHash,
    pub attempts: u32,
}

/// Canonical identity input for one operation. Upstream operations contribute
/// their identities and media contribute their content hashes, so a change to
/// any input, provider, or revision changes exactly the affected descendants.
#[derive(Serialize)]
struct OperationIdentityInput {
    schema_version: u32,
    kind: OperationKind,
    /// Built-in implementation revision and core crate version; `None` for
    /// provider operations, which are versioned by their declarations.
    implementation: Option<(u32, &'static str)>,
    /// Declared provider followed by reachable fallbacks, in order.
    providers: Vec<ProviderDeclaration>,
    max_attempts: u32,
    inputs: Vec<String>,
}

impl SceneProjectManifest {
    /// Parse, validate and canonicalize a manifest document.
    pub fn from_json(document: &str) -> Result<Self, SceneProjectError> {
        let value: serde_json::Value = serde_json::from_str(document)
            .map_err(|error| SceneProjectError::Malformed(error.to_string()))?;
        match value.get("schema_version").map(serde_json::Value::as_u64) {
            Some(Some(version)) if version == u64::from(SCENE_PROJECT_SCHEMA_VERSION) => {}
            Some(Some(version)) => {
                return Err(SceneProjectError::UnsupportedSchemaVersion(version))
            }
            _ => {
                return Err(SceneProjectError::Malformed(
                    "`schema_version` must be a non-negative integer".into(),
                ))
            }
        }
        let mut manifest: Self = serde_json::from_value(value)
            .map_err(|error| SceneProjectError::Malformed(error.to_string()))?;
        manifest.validate()?;
        manifest.canonicalize();
        Ok(manifest)
    }

    pub fn load(path: &Path) -> Result<Self, SceneProjectError> {
        let document = std::fs::read_to_string(path).map_err(|error| {
            SceneProjectError::Malformed(format!("cannot read {}: {error}", path.display()))
        })?;
        Self::from_json(&document)
    }

    /// Validate and serialize in canonical order. Equivalent manifests produce
    /// byte-identical documents.
    pub fn to_canonical_json(&self) -> Result<String, SceneProjectError> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.canonicalize();
        let mut document = serde_json::to_string_pretty(&canonical)
            .map_err(|error| SceneProjectError::Malformed(error.to_string()))?;
        document.push('\n');
        Ok(document)
    }

    /// Order-insensitive collections are sorted; ordered collections
    /// (`fallback_order`) keep their declared order.
    pub fn canonicalize(&mut self) {
        self.inputs.sort_by(|a, b| a.id.cmp(&b.id));
        self.provider_policy
            .providers
            .sort_by(|a, b| a.id.cmp(&b.id));
        for provider in &mut self.provider_policy.providers {
            provider.capabilities.sort();
        }
        self.provider_policy.cloud_upload_allowlist.sort();
        self.operations.sort_by(|a, b| a.id.cmp(&b.id));
        for operation in &mut self.operations {
            operation.inputs.sort();
        }
        self.requested_outputs.sort();
        self.artifacts.sort_by(|a, b| a.id.cmp(&b.id));
        self.exports.sort_by(|a, b| a.id.cmp(&b.id));
        self.attempt_usage
            .sort_by(|a, b| a.operation.cmp(&b.operation));
        for export in &mut self.exports {
            export.includes.sort();
        }
    }

    pub fn validate(&self) -> Result<(), SceneProjectError> {
        if self.schema_version != SCENE_PROJECT_SCHEMA_VERSION {
            return Err(SceneProjectError::UnsupportedSchemaVersion(u64::from(
                self.schema_version,
            )));
        }
        validate_identifier("project_id", &self.project_id)?;

        let mut ids = BTreeSet::new();
        let mut paths = BTreeSet::new();
        let mut unique_id = |scope: &str, id: &str| -> Result<(), SceneProjectError> {
            validate_identifier(scope, id)?;
            if !ids.insert(id.to_owned()) {
                return invalid(format!("identifier `{id}` is declared more than once"));
            }
            Ok(())
        };

        if self.inputs.is_empty() {
            return invalid("at least one media input is required");
        }
        for input in &self.inputs {
            unique_id("media input id", &input.id)?;
            if input.byte_length == 0 {
                return invalid(format!("media input `{}` has zero byte length", input.id));
            }
            unique_path(&mut paths, &input.path)?;
        }

        let providers = self.validate_provider_policy(&mut unique_id)?;

        let media: BTreeSet<&str> = self.inputs.iter().map(|input| input.id.as_str()).collect();
        let operations: BTreeMap<&str, &OperationDeclaration> = self
            .operations
            .iter()
            .map(|operation| (operation.id.as_str(), operation))
            .collect();
        if self.operations.is_empty() {
            return invalid("at least one operation is required");
        }
        for operation in &self.operations {
            unique_id("operation id", &operation.id)?;
        }
        for operation in &self.operations {
            self.validate_operation(operation, &media, &operations, &providers)?;
        }
        topological_order(&self.operations)?;
        self.validate_costs()?;

        let produced: BTreeSet<ArtifactKind> = self
            .operations
            .iter()
            .map(|operation| operation.kind.output())
            .collect();
        let mut requested = BTreeSet::new();
        for output in &self.requested_outputs {
            if !requested.insert(*output) {
                return invalid(format!(
                    "requested output {output:?} is listed more than once"
                ));
            }
            if !produced.contains(output) {
                return invalid(format!(
                    "requested output {output:?} is not produced by any declared operation"
                ));
            }
        }
        if requested.is_empty() {
            return invalid("at least one requested output is required");
        }

        let mut recorded_operations = BTreeSet::new();
        for artifact in &self.artifacts {
            unique_id("artifact id", &artifact.id)?;
            unique_path(&mut paths, &artifact.path)?;
            let Some(operation) = operations.get(artifact.produced_by.as_str()) else {
                return invalid(format!(
                    "artifact `{}` references unknown operation `{}`",
                    artifact.id, artifact.produced_by
                ));
            };
            if operation.kind.output() != artifact.kind {
                return invalid(format!(
                    "artifact `{}` is {:?} but operation `{}` produces {:?}",
                    artifact.id,
                    artifact.kind,
                    operation.id,
                    operation.kind.output()
                ));
            }
            if !recorded_operations.insert(artifact.produced_by.as_str()) {
                return invalid(format!(
                    "operation `{}` has more than one recorded artifact",
                    artifact.produced_by
                ));
            }
            // A stale artifact is not invalid: its recorded identity no longer
            // matches and it must be rebuilt (see `stale_artifacts`).
        }

        for export in &self.exports {
            unique_id("export id", &export.id)?;
            unique_path(&mut paths, &export.path)?;
            if export.includes.is_empty() {
                return invalid(format!("export `{}` includes no artifacts", export.id));
            }
            let mut included = BTreeSet::new();
            for kind in &export.includes {
                if !included.insert(*kind) {
                    return invalid(format!(
                        "export `{}` includes {kind:?} more than once",
                        export.id
                    ));
                }
                if !requested.contains(kind) {
                    return invalid(format!(
                        "export `{}` includes {kind:?}, which is not a requested output",
                        export.id
                    ));
                }
            }
        }

        let mut usage_operations = BTreeSet::new();
        for usage in &self.attempt_usage {
            let Some(operation) = operations.get(usage.operation.as_str()) else {
                return invalid(format!(
                    "attempt usage references unknown operation `{}`",
                    usage.operation
                ));
            };
            if !usage_operations.insert(usage.operation.as_str()) {
                return invalid(format!(
                    "attempt usage for operation `{}` is recorded more than once",
                    usage.operation
                ));
            }
            if usage.attempts == 0 || usage.attempts > MAX_OPERATION_ATTEMPTS {
                return invalid(format!(
                    "attempt usage for operation `{}` must be within 1..={MAX_OPERATION_ATTEMPTS}",
                    operation.id
                ));
            }
        }
        Ok(())
    }

    fn validate_provider_policy<'a>(
        &'a self,
        unique_id: &mut impl FnMut(&str, &str) -> Result<(), SceneProjectError>,
    ) -> Result<BTreeMap<&'a str, &'a ProviderDeclaration>, SceneProjectError> {
        let policy = &self.provider_policy;
        let mut providers = BTreeMap::new();
        for provider in &policy.providers {
            unique_id("provider id", &provider.id)?;
            providers.insert(provider.id.as_str(), provider);
            if provider.capabilities.is_empty() {
                return invalid(format!(
                    "provider `{}` declares no capabilities",
                    provider.id
                ));
            }
            let mut capabilities = BTreeSet::new();
            for kind in &provider.capabilities {
                if kind.execution() != OperationExecution::Provider {
                    return invalid(format!(
                        "provider `{}` cannot serve built-in operation kind {}",
                        provider.id,
                        kind.name()
                    ));
                }
                if !capabilities.insert(*kind) {
                    return invalid(format!(
                        "provider `{}` lists capability {} more than once",
                        provider.id,
                        kind.name()
                    ));
                }
            }
            if let Some(revision) = &provider.revision {
                if revision.trim().is_empty() || revision.chars().any(char::is_control) {
                    return invalid(format!("provider `{}` has an empty revision", provider.id));
                }
            }
            if let Some(capability) = &provider.credential_capability {
                validate_capability_name(&provider.id, capability)?;
            }
            match provider.location {
                ProviderLocation::Local if provider.cost_per_attempt.is_some() => {
                    return invalid(format!(
                        "local provider `{}` must not declare a cost",
                        provider.id
                    ))
                }
                ProviderLocation::Cloud if provider.cost_per_attempt.is_none() => {
                    return invalid(format!(
                        "cloud provider `{}` must declare cost_per_attempt (0 when free)",
                        provider.id
                    ))
                }
                ProviderLocation::Cloud if policy.execution == ExecutionPolicy::LocalOnly => {
                    return invalid(format!(
                        "cloud provider `{}` is not allowed by the local_only execution policy",
                        provider.id
                    ))
                }
                ProviderLocation::Cloud
                    if provider.cost_per_attempt.unwrap_or(0) > 0
                        && policy.execution != ExecutionPolicy::AllowPaidCloud =>
                {
                    return invalid(format!(
                        "paid cloud provider `{}` requires the allow_paid_cloud execution policy",
                        provider.id
                    ))
                }
                _ => {}
            }
        }

        let mut seen = BTreeSet::new();
        let mut seen_cloud = false;
        for id in &policy.fallback_order {
            let Some(provider) = providers.get(id.as_str()) else {
                return invalid(format!("fallback order references unknown provider `{id}`"));
            };
            if !seen.insert(id.as_str()) {
                return invalid(format!(
                    "fallback order lists provider `{id}` more than once"
                ));
            }
            match provider.location {
                ProviderLocation::Cloud => seen_cloud = true,
                ProviderLocation::Local
                    if seen_cloud && policy.execution == ExecutionPolicy::LocalFirst =>
                {
                    return invalid(format!(
                        "local_first policy must order local provider `{id}` before cloud providers"
                    ))
                }
                ProviderLocation::Local => {}
            }
        }
        if seen.len() != providers.len() {
            return invalid("fallback order must list every declared provider exactly once");
        }

        let mut uploads = BTreeSet::new();
        for kind in &policy.cloud_upload_allowlist {
            if !uploads.insert(*kind) {
                return invalid(format!(
                    "cloud upload allowlist lists {kind:?} more than once"
                ));
            }
        }
        if policy.execution == ExecutionPolicy::LocalOnly && !uploads.is_empty() {
            return invalid("local_only execution policy must not allow cloud uploads");
        }
        if policy.execution == ExecutionPolicy::AllowPaidCloud && policy.max_total_cost.is_none() {
            return invalid("allow_paid_cloud execution policy requires max_total_cost");
        }
        Ok(providers)
    }

    fn validate_operation(
        &self,
        operation: &OperationDeclaration,
        media: &BTreeSet<&str>,
        operations: &BTreeMap<&str, &OperationDeclaration>,
        providers: &BTreeMap<&str, &ProviderDeclaration>,
    ) -> Result<(), SceneProjectError> {
        let kind = operation.kind;
        if kind.minimum_quality() > self.quality_mode {
            return invalid(format!(
                "operation `{}` ({}) requires quality mode {:?} or higher, but the project declares {:?}",
                operation.id,
                kind.name(),
                kind.minimum_quality(),
                self.quality_mode
            ));
        }
        if operation.max_attempts == 0 || operation.max_attempts > MAX_OPERATION_ATTEMPTS {
            return invalid(format!(
                "operation `{}` must declare 1..={MAX_OPERATION_ATTEMPTS} attempts",
                operation.id
            ));
        }
        if operation.inputs.is_empty() {
            return invalid(format!("operation `{}` declares no inputs", operation.id));
        }
        let mut seen = BTreeSet::new();
        let mut upstream_kinds = Vec::new();
        for input in &operation.inputs {
            if !seen.insert(input) {
                return invalid(format!(
                    "operation `{}` lists input {input:?} more than once",
                    operation.id
                ));
            }
            match input {
                OperationInput::Media(id) => {
                    if kind != OperationKind::IngestVideo {
                        return invalid(format!(
                            "operation `{}` ({}) must not consume source media directly; only ingest_video may",
                            operation.id,
                            kind.name()
                        ));
                    }
                    if !media.contains(id.as_str()) {
                        return invalid(format!(
                            "operation `{}` references unknown media input `{id}`",
                            operation.id
                        ));
                    }
                }
                OperationInput::Operation(id) => {
                    let Some(upstream) = operations.get(id.as_str()) else {
                        return invalid(format!(
                            "operation `{}` references unknown operation `{id}`",
                            operation.id
                        ));
                    };
                    if !kind.allowed_upstream().contains(&upstream.kind) {
                        return invalid(format!(
                            "operation `{}` ({}) cannot consume `{id}` ({})",
                            operation.id,
                            kind.name(),
                            upstream.kind.name()
                        ));
                    }
                    upstream_kinds.push(upstream.kind.output());
                }
            }
        }
        if kind == OperationKind::TextureBake {
            let count = |wanted: ArtifactKind| {
                upstream_kinds
                    .iter()
                    .filter(|output| **output == wanted)
                    .count()
            };
            if operation.inputs.len() != 2
                || count(ArtifactKind::Keyframes) != 1
                || count(ArtifactKind::SurfaceMesh) != 1
            {
                return invalid(format!(
                    "operation `{}` (texture_bake) must consume exactly one ingest_video and one surface_mesh operation",
                    operation.id
                ));
            }
        }
        if kind == OperationKind::IngestVideo && operation.inputs.len() != 1 {
            return invalid(format!(
                "operation `{}` (ingest_video) must consume exactly one media input",
                operation.id
            ));
        }

        match (kind.execution(), &operation.provider) {
            (OperationExecution::BuiltIn, Some(provider)) => invalid(format!(
                "built-in operation `{}` ({}) must not declare provider `{provider}`",
                operation.id,
                kind.name()
            )),
            (OperationExecution::Provider, None) => invalid(format!(
                "operation `{}` ({}) requires a declared provider",
                operation.id,
                kind.name()
            )),
            (OperationExecution::BuiltIn, None) => Ok(()),
            (OperationExecution::Provider, Some(provider_id)) => {
                let Some(provider) = providers.get(provider_id.as_str()) else {
                    return invalid(format!(
                        "operation `{}` references unknown provider `{provider_id}`",
                        operation.id
                    ));
                };
                if !provider.capabilities.contains(&kind) {
                    return invalid(format!(
                        "provider `{provider_id}` does not declare capability {} required by operation `{}`",
                        kind.name(),
                        operation.id
                    ));
                }
                if self.provider_policy.execution == ExecutionPolicy::LocalFirst
                    && provider.location == ProviderLocation::Cloud
                {
                    if let Some(local) = self.provider_policy.providers.iter().find(|candidate| {
                        candidate.location == ProviderLocation::Local
                            && candidate.capabilities.contains(&kind)
                    }) {
                        return invalid(format!(
                            "local_first policy: operation `{}` names cloud provider `{provider_id}` although local provider `{}` can serve it",
                            operation.id, local.id
                        ));
                    }
                }
                for reachable in self.reachable_providers(operation) {
                    if reachable.location != ProviderLocation::Cloud {
                        continue;
                    }
                    for artifact in &upstream_kinds {
                        if !self
                            .provider_policy
                            .cloud_upload_allowlist
                            .contains(artifact)
                        {
                            return invalid(format!(
                                "operation `{}` can reach cloud provider `{}`, which would upload {artifact:?} not permitted by the cloud upload allowlist",
                                operation.id, reachable.id
                            ));
                        }
                    }
                }
                Ok(())
            }
        }
    }

    /// Providers that may execute an operation: its declared provider first,
    /// then every other provider in `fallback_order` that declares the
    /// operation's capability. `max_attempts` bounds attempts across all of
    /// them, so upload and cost rules are checked against every reachable one.
    pub fn reachable_providers(
        &self,
        operation: &OperationDeclaration,
    ) -> Vec<&ProviderDeclaration> {
        let Some(primary) = operation.provider.as_deref() else {
            return Vec::new();
        };
        let find = |id: &str| {
            self.provider_policy
                .providers
                .iter()
                .find(|provider| provider.id == id)
        };
        let mut reachable: Vec<&ProviderDeclaration> = find(primary).into_iter().collect();
        for id in &self.provider_policy.fallback_order {
            if id == primary {
                continue;
            }
            if let Some(provider) = find(id) {
                if provider.capabilities.contains(&operation.kind) {
                    reachable.push(provider);
                }
            }
        }
        reachable
    }

    fn validate_costs(&self) -> Result<(), SceneProjectError> {
        let policy = &self.provider_policy;
        let mut total: u64 = 0;
        for operation in &self.operations {
            let reachable = self.reachable_providers(operation);
            if reachable.is_empty() {
                continue;
            }
            let worst_case = reachable
                .iter()
                .map(|provider| provider.cost_per_attempt.unwrap_or(0))
                .max()
                .unwrap_or(0)
                .checked_mul(u64::from(operation.max_attempts))
                .ok_or_else(|| {
                    SceneProjectError::Invalid(format!(
                        "operation `{}` worst-case cost overflows",
                        operation.id
                    ))
                })?;
            if let Some(limit) = policy.max_cost_per_operation {
                if worst_case > limit {
                    return invalid(format!(
                        "operation `{}` worst-case cost {worst_case} exceeds max_cost_per_operation {limit}",
                        operation.id
                    ));
                }
            }
            total = total.checked_add(worst_case).ok_or_else(|| {
                SceneProjectError::Invalid("project worst-case cost overflows".into())
            })?;
        }
        if let Some(limit) = policy.max_total_cost {
            if total > limit {
                return invalid(format!(
                    "project worst-case cost {total} exceeds max_total_cost {limit}"
                ));
            }
        }
        Ok(())
    }

    /// Deterministic identity of every declared operation. Identities ignore
    /// operation ids and declaration order and depend only on kind, reachable
    /// provider declarations (id, revision, cost; not credentials), attempt bound,
    /// media hashes, and for each upstream operation its identity plus the
    /// content hash of its current recorded artifact (`pending` when none).
    /// Replacing an upstream artifact therefore invalidates its descendants.
    pub fn operation_identities(&self) -> Result<BTreeMap<String, ContentHash>, SceneProjectError> {
        self.validate()?;
        self.compute_operation_identities()
    }

    fn compute_operation_identities(
        &self,
    ) -> Result<BTreeMap<String, ContentHash>, SceneProjectError> {
        let media: BTreeMap<&str, &MediaInput> = self
            .inputs
            .iter()
            .map(|input| (input.id.as_str(), input))
            .collect();
        let mut identities: BTreeMap<String, ContentHash> = BTreeMap::new();
        for operation in topological_order(&self.operations)? {
            let mut inputs: Vec<String> = operation
                .inputs
                .iter()
                .map(|input| match input {
                    OperationInput::Media(id) => format!(
                        "media:{}:{}",
                        media[id.as_str()].content_hash.as_str(),
                        media[id.as_str()].byte_length
                    ),
                    OperationInput::Operation(id) => {
                        let identity = &identities[id];
                        // Downstream work depends on the bytes actually
                        // produced upstream, not only on how they were requested.
                        let content = self
                            .artifacts
                            .iter()
                            .find(|artifact| {
                                artifact.produced_by == *id
                                    && artifact.operation_identity == *identity
                            })
                            .map_or("pending", |artifact| artifact.content_hash.as_str());
                        format!("operation:{}:{content}", identity.as_str())
                    }
                })
                .collect();
            inputs.sort();
            let providers: Vec<ProviderDeclaration> = self
                .reachable_providers(operation)
                .into_iter()
                .map(|provider| {
                    let mut provider = provider.clone();
                    provider.capabilities.sort();
                    // The provider id is kept: it is the stable backend key that
                    // distinguishes otherwise identical declarations.
                    // Credentials select an account, not a result.
                    provider.credential_capability = None;
                    provider
                })
                .collect();
            let identity_input = OperationIdentityInput {
                schema_version: SCENE_PROJECT_SCHEMA_VERSION,
                kind: operation.kind,
                implementation: (operation.kind.execution() == OperationExecution::BuiltIn).then(
                    || {
                        (
                            operation.kind.implementation_revision(),
                            env!("CARGO_PKG_VERSION"),
                        )
                    },
                ),
                providers,
                max_attempts: operation.max_attempts,
                inputs,
            };
            let bytes = serde_json::to_vec(&identity_input)
                .map_err(|error| SceneProjectError::Malformed(error.to_string()))?;
            identities.insert(operation.id.clone(), ContentHash::of_bytes(&bytes));
        }
        Ok(identities)
    }

    /// Recorded artifacts whose producing operation identity changed since they
    /// were written. These, and everything downstream of them, must be rebuilt.
    pub fn stale_artifacts(&self) -> Result<Vec<&ArtifactRecord>, SceneProjectError> {
        let identities = self.operation_identities()?;
        Ok(self
            .artifacts
            .iter()
            .filter(|artifact| identities[&artifact.produced_by] != artifact.operation_identity)
            .collect())
    }
}

fn validate_identifier(scope: &str, id: &str) -> Result<(), SceneProjectError> {
    let valid = !id.is_empty()
        && id.len() <= MAX_IDENTIFIER_LENGTH
        && id
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        && id.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
        });
    if !valid {
        return invalid(format!(
            "{scope} `{id}` must be 1..={MAX_IDENTIFIER_LENGTH} lowercase ASCII letters, digits, `-`, `_` or `.`, starting with a letter or digit"
        ));
    }
    Ok(())
}

/// Capability names are short dotted identifiers such as `providers.marble`.
/// The lowercase-only alphabet and length bound make it impossible to paste a
/// typical API key or token into this field by mistake.
fn validate_capability_name(provider: &str, capability: &str) -> Result<(), SceneProjectError> {
    let valid = !capability.is_empty()
        && capability.len() <= MAX_IDENTIFIER_LENGTH
        && capability
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_lowercase())
        && capability.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
        });
    if !valid {
        return invalid(format!(
            "provider `{provider}` credential capability must be a capability name (lowercase identifier), never a secret"
        ));
    }
    Ok(())
}

/// Paths are compared case-insensitively so that two declarations cannot alias
/// one file on case-insensitive filesystems.
fn unique_path(paths: &mut BTreeSet<String>, path: &ProjectPath) -> Result<(), SceneProjectError> {
    let folded = path.as_str().to_ascii_lowercase();
    if folded == STATE_DIRECTORY || folded.starts_with(&format!("{STATE_DIRECTORY}/")) {
        return invalid(format!(
            "project path `{}` is inside the reserved `{STATE_DIRECTORY}` state directory",
            path.as_str()
        ));
    }
    if !paths.insert(folded) {
        return invalid(format!(
            "project path `{}` is used more than once",
            path.as_str()
        ));
    }
    Ok(())
}

/// Deterministic topological order (ties broken by operation id). Fails on cycles.
fn topological_order(
    operations: &[OperationDeclaration],
) -> Result<Vec<&OperationDeclaration>, SceneProjectError> {
    let by_id: BTreeMap<&str, &OperationDeclaration> = operations
        .iter()
        .map(|operation| (operation.id.as_str(), operation))
        .collect();
    let mut pending: BTreeMap<&str, usize> = BTreeMap::new();
    let mut dependents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for operation in operations {
        let mut count = 0;
        for input in &operation.inputs {
            if let OperationInput::Operation(upstream) = input {
                if by_id.contains_key(upstream.as_str()) {
                    count += 1;
                    dependents
                        .entry(upstream.as_str())
                        .or_default()
                        .push(operation.id.as_str());
                }
            }
        }
        pending.insert(operation.id.as_str(), count);
    }
    let mut ready: BTreeSet<&str> = pending
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(id, _)| *id)
        .collect();
    let mut order = Vec::with_capacity(operations.len());
    while let Some(id) = ready.pop_first() {
        order.push(by_id[id]);
        for dependent in dependents.get(id).into_iter().flatten() {
            let count = pending.get_mut(dependent).expect("dependent is declared");
            *count -= 1;
            if *count == 0 {
                ready.insert(dependent);
            }
        }
    }
    if order.len() != operations.len() {
        let cyclic: Vec<&str> = pending
            .iter()
            .filter(|(_, count)| **count > 0)
            .map(|(id, _)| *id)
            .collect();
        return invalid(format!(
            "operation graph contains a cycle through {}",
            cyclic.join(", ")
        ));
    }
    Ok(order)
}

#[cfg(test)]
#[path = "scene_project_tests.rs"]
mod tests;
