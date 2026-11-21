//! Private local file exchange for native model and camera providers.
use super::*;
use std::process::{Command, Stdio};
use std::time::Duration;

const MAX_RESPONSE_BYTES: u64 = 512 << 20;

/// A scratch directory for one sidecar exchange, removed on drop.
pub(super) struct Exchange(pub(super) PathBuf);

impl Exchange {
    pub(super) fn new(operation: &str) -> Result<Self, String> {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "video-to-3d-{operation}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        #[cfg(unix)]
        let result =
            std::os::unix::fs::DirBuilderExt::mode(&mut fs::DirBuilder::new(), 0o700).create(&path);
        #[cfg(not(unix))]
        let result = fs::create_dir(&path);
        result.map_err(|error| {
            format!("cannot create sidecar exchange {}: {error}", path.display())
        })?;
        Ok(Self(path))
    }
}

impl Drop for Exchange {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(super) enum SidecarRun {
    Response(Vec<u8>),
    NotStarted(String),
    Canceled,
}

/// Run the sidecar to completion, killing and reaping it on cancellation.
pub(super) fn run_sidecar(
    label: &str,
    program: &Path,
    request: &Path,
    response: &Path,
    scratch: &Path,
    cancel: &CancellationToken,
) -> Result<SidecarRun, String> {
    let stderr_path = scratch.join("stderr.log");
    let stderr = fs::File::create(&stderr_path).map_err(|error| error.to_string())?;
    let mut child = match Command::new(program)
        .arg(request)
        .arg(response)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr)
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return Ok(SidecarRun::NotStarted(format!(
            "{label} sidecar is unavailable: the local sidecar `{}` could not be started ({error})",
            program.display()
        )))
        }
    };
    let status = loop {
        if cancel.is_canceled() {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(SidecarRun::Canceled);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{label} sidecar wait failed: {error}"));
            }
        }
    };
    if !status.success() {
        let log = fs::read_to_string(&stderr_path).unwrap_or_default();
        let tail: String = log
            .lines()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" / ");
        return Err(format!("{label} sidecar exited with {status}: {tail}"));
    }
    let length = fs::metadata(response)
        .map_err(|error| format!("{label} sidecar wrote no response: {error}"))?
        .len();
    if length > MAX_RESPONSE_BYTES {
        return Err(format!(
            "{label} sidecar response of {length} bytes exceeds the {MAX_RESPONSE_BYTES}-byte limit"
        ));
    }
    fs::read(response)
        .map(SidecarRun::Response)
        .map_err(|error| format!("{label} sidecar response is unreadable: {error}"))
}
