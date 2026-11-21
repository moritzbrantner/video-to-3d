// Keep the desktop window and its loopback Studio (including native provider children)
// in the same development session. The packaged application uses its static frontend.
import { spawn, type ChildProcess } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startLocalStudio } from "./local-studio";

const root = join(import.meta.dir, "..");

async function serve() {
  process.chdir(root);
  const studio = await startLocalStudio();
  // startLocalStudio removes its temporary frame directory on process exit.
  process.once("SIGTERM", () => process.exit(0));
  process.once("disconnect", () => stopTree(process.pid));
  if (process.send) {
    process.send(studio.url.href);
  }
}

function stopTree(pid: number | undefined) {
  if (!pid) {
    return;
  }
  if (process.platform === "win32") {
    Bun.spawnSync(["taskkill", "/PID", String(pid), "/T", "/F"], { stdout: "ignore", stderr: "ignore" });
    return;
  }
  try {
    // Both children have their own process groups. Native providers inherit Studio's group.
    process.kill(-pid, "SIGTERM");
  } catch (error) {
    if (!(error instanceof Error && "code" in error && error.code === "ESRCH")) {
      throw error;
    }
  }
}

async function launch() {
  // Providers may stop mid-run when the window closes. Keep their private exchange
  // directories inside this session so the supervisor can clean them up together.
  const scratch = await mkdtemp(join(tmpdir(), "video-to-3d-desktop-"));
  const studio = spawn(process.execPath, [import.meta.path, "--serve-studio"], {
    cwd: root, detached: true, stdio: ["ignore", "inherit", "inherit", "ipc"],
    env: { ...process.env, TMPDIR: scratch, TEMP: scratch, TMP: scratch },
  });
  let desktop: ChildProcess | undefined;
  const stop = () => {
    stopTree(desktop?.pid);
    stopTree(studio.pid);
  };
  let interrupted = false;
  const interrupt = () => {
    interrupted = true;
    stop();
  };
  process.once("SIGINT", interrupt);
  process.once("SIGTERM", interrupt);
  try {
    const studioExit = once(studio, "exit");
    const address = await Promise.race([
      once(studio, "message").then(([url]) => new URL(url)),
      studioExit.then(([code]) => { throw new Error(`Local Studio exited before opening a port (${code})`); }),
    ]);
    if (address.protocol !== "http:" || address.hostname !== "127.0.0.1") {
      throw new Error("Desktop Studio requires a local loopback address");
    }
    console.log(`Desktop Studio: ${address.href}`);
    const config = JSON.stringify({ build: { beforeDevCommand: null, devUrl: address.href } });
    const workspace = join(root, "apps/desktop");
    desktop = spawn(process.execPath, [
      require.resolve("@tauri-apps/cli/tauri.js", { paths: [workspace] }),
      "dev", "--no-watch", "--config", config, ...process.argv.slice(2),
    ], { cwd: workspace, detached: true, stdio: "inherit" });
    const [code] = await Promise.race([
      once(desktop, "exit"),
      studioExit.then(() => { throw new Error("The local Studio server stopped"); }),
    ]);
    process.exitCode = code ?? 1;
  } catch (error) {
    if (!interrupted) {
      throw error;
    }
    process.exitCode = 1;
  } finally {
    stop();
    process.removeListener("SIGINT", interrupt);
    process.removeListener("SIGTERM", interrupt);
    await rm(scratch, { recursive: true, force: true });
  }
}

if (process.argv[2] === "--serve-studio" && process.send) {
  await serve();
} else {
  await launch();
}
