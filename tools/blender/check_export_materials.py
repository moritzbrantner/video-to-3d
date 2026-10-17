"""Check actual #84 production-export material/texture fidelity against input oracle."""
import argparse
import json
import sys
from pathlib import Path

import bpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    report = {"schema": "video-to-3d/export-material-acceptance/v1", "checks": []}
    try:
        root = Path(args.inputs)
        expected = json.loads((root / "scene.json").read_text())
        bpy.ops.wm.read_factory_settings(use_empty=True)
        assert bpy.ops.import_scene.gltf(filepath=str(Path(args.glb).resolve())) == {"FINISHED"}
        textures = [image for image in bpy.data.images if image.source == "FILE" or image.packed_file]
        assert textures, "canonical texture is present"
        original = bpy.data.images.load(str(root / "resources/checker.png"), check_existing=False)
        truth_pixels = list(original.pixels)
        assert any(tuple(image.size) == tuple(original.size) and len(image.pixels) == len(truth_pixels) and all(abs(a-b)<1e-5 for a,b in zip(image.pixels,truth_pixels)) for image in textures), "embedded checker pixels preserved"
        report["checks"].append({"check": "texture_pixels", "passed": True})
        for material in expected["materials"]:
            found = bpy.data.materials.get(material["id"])
            assert found is not None, f"material identity {material['id']}"
            principled = next(node for node in found.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
            # With a textured base color glTF's factors may use a multiplication
            # node; source factors are independently checked in the GLB contract.
            assert abs(principled.inputs["Metallic"].default_value-material["metallic"])<1e-5
            assert abs(principled.inputs["Roughness"].default_value-material["roughness"])<1e-5
            assert any(node.type == "TEX_IMAGE" and node.image for node in found.node_tree.nodes)
            report["checks"].append({"check": f"material:{material['id']}", "passed": True})
        report["passed"] = True
    except Exception as error:  # every import/data failure is nonpassing
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


sys.exit(main())
