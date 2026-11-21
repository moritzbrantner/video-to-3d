import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "bun:test";

test("the shipped browser app exports a surface from the real Trevi video", async () => {
  const directory = await mkdtemp(join(tmpdir(), "video-to-scene-"));
  try {
    const process = Bun.spawn(["bun", "run", "reconstruct", "trevi.webm", directory], {
      env: { ...Bun.env,
        VIDEO_TO_3D_SEMANTIC_SIDECAR: join(directory, "unavailable-sidecar"),
        VIDEO_TO_3D_CAMERA_SIDECAR: join(directory, "unavailable-camera-sidecar"),
      },
      stdout: "pipe",
      stderr: "inherit",
    });
    const [code, output] = await Promise.all([process.exited, new Response(process.stdout).text()]);
    assert.equal(code, 0, output);
    assert.match(output, /Using browser camera reconstruction\. Local COLMAP registration failed/);
    const evidence = JSON.parse(await readFile(join(directory, "trevi.objects.json"), "utf8"));
    assert.equal(evidence.status, "unavailable");
    assert(evidence.frames.length > 1);
    assert.equal(evidence.observations.length, 0);
    const bytes = await readFile(join(directory, "trevi.glb"));
    assert.equal(bytes.toString("ascii", 0, 4), "glTF");
    assert.equal(bytes.readUInt32LE(4), 2);
    assert.equal(bytes.readUInt32LE(8), bytes.length);
    const json = JSON.parse(bytes.toString("utf8", 20, 20 + bytes.readUInt32LE(12)));
    assert(json.images.some((image: { bufferView?: number }) => image.bufferView !== undefined),
      "the Trevi surface should include a baked reference image");
    const visual = json.scenes[json.scene];
    const meshNode = visual.nodes.map((index: number) => json.nodes[index])
      .find((node: { name: string }) => node.name === "accepted-surface");
    assert(meshNode, "default scene must contain the accepted visual surface");
    const mesh = json.meshes[meshNode.mesh];
    assert(mesh.primitives.some((primitive: { mode?: number; indices: number }) =>
      (primitive.mode ?? 4) === 4 && json.accessors[primitive.indices].count >= 3));
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}, 720_000);
