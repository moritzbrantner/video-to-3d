"""Render a static scene from 18 known camera poses and encode a local test video."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--travel", required=True, type=float, help="horizontal travel in Blender units")
    parser.add_argument("--focus-distance", required=True, type=float, help="distance to look-at target")
    parser.add_argument("--forward", type=float, default=0, help="additional forward travel")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if (not all(map(math.isfinite, (args.travel, args.focus_distance, args.forward)))
            or args.focus_distance <= max(0, args.forward) or args.travel == 0):
        parser.error("provide finite travel and a look-at target ahead of the camera path")

    scene = bpy.context.scene
    scene.frame_set(1)
    original = scene.camera
    if original is None or original.data.type != "PERSP":
        parser.error("the active scene needs a perspective camera")
    pose = original.matrix_world.copy()
    center = pose.translation
    right = pose.to_quaternion() @ Vector((1, 0, 0))
    forward = pose.to_quaternion() @ Vector((0, 0, -1))
    target = center + forward * args.focus_distance
    camera = bpy.data.objects.new("Benchmark capture", original.data.copy())
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.shift_x = camera.data.shift_y = 0
    camera.data.dof.use_dof = False

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("frames", "cameras.json", "scene.webm")):
        parser.error("capture files already exist in the output directory")
    frames_dir = output / "frames"
    frames_dir.mkdir()
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.use_border = scene.render.use_compositing = scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = 24
        scene.cycles.use_denoising = True
    frames = []
    for index in range(18):
        fraction = index / 17
        camera.location = center + right * (args.travel * (fraction - 0.5)) + forward * (args.forward * fraction)
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        frames.append({"matrix_world": [list(row) for row in camera.matrix_world]})
        scene.render.filepath = str(frames_dir / f"{index:02d}.png")
        bpy.ops.render.render(write_still=True)

    (output / "cameras.json").write_text(json.dumps({
        "width": 640, "height": 480, "scene": scene.name, "scene_frame": scene.frame_current,
        "horizontal_fov_radians": camera.data.angle_x, "frames": frames,
    }, indent=2) + "\n")
    # At 18 s the default browser plan samples at 0.9..17.1 s, hitting each
    # held pose once. Reconstruct from encoded RGB; poses are for scoring only.
    subprocess.run([
        "ffmpeg", "-v", "error", "-n", "-framerate", "1", "-i", str(frames_dir / "%02d.png"),
        "-vf", "fps=10", "-c:v", "libvpx-vp9", "-crf", "16", "-b:v", "0",
        "-pix_fmt", "yuv420p", str(output / "scene.webm"),
    ], check=True)


if __name__ == "__main__":
    main()
