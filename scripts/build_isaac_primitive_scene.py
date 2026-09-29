#!/usr/bin/env python3
"""Build the Isaac stage of a MuJoCo scene drawn only with primitive boxes (box_platform), in the layout stage.py expects.

The warehouse's stage wraps a Blender USDC (scripts/build_isaac_warehouse_scene.py) and the table scenes'
come from OBJ meshes and a PLY room (scripts/build_isaac_assets.sh). A generated scene such as `box_platform`
(locomani.tasks.crate.box_platform_scene) has neither: its platform and its carton are MJCF boxes. So its
stage is written straight from the compiled MJCF, as two small USD layers:

    <assets>/scene/<label>/scene.usda              /World/static/<geom>: every visible non-robot box that does not
                                                   move, at its world pose, plus /World/static/ground -- the MJCF
                                                   floor is an invisible plane and Isaac has no void to fall into
    <assets>/scene/<label>/usd/<target>/           the movable target: its boxes in the BODY frame, so stage.py's
        <target>.usd, <target>.json                pose-driven /World/<target> follows the MJCF body directly
    <assets>/scene/<label>/source.json             the MJCF's sha256: run_context.isaac_scene refuses a stage built
                                                   from another scene (a rebuilt box, a moved platform)

Colours are the geoms' rgba as UsdPreviewSurface materials, converted from sRGB to linear: an MJCF rgba is a display
value (MuJoCo applies no gamma) and RTX reads diffuseColor as linear, so taken as-is the carton's 0.86/0.78/0.62
rendered near white (2026-09-29; the robot's colours had the same fault, stage.G1_LOOKS). No lights:
stage.assemble_stage adds its dome and sun when the scene brings none. The look (exposure) is the scene's
`isaac.rtx_settings` in configs/scenes.yaml.

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/build_isaac_primitive_scene.py \\
        outputs/scene/box_platform [--label box_platform] [--assets $ISAAC_ASSETS]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

GROUND_RGB = (0.32, 0.32, 0.34)  # linear already: a matte concrete grey chosen in Isaac; MuJoCo draws no floor at all
GROUND_HALF = (10.0, 10.0, 0.01)  # m; top face at z = 0, where the MJCF plane is
ROUGHNESS = 0.8
BOX, PLANE = 6, 0  # mjtGeom


def _f(values) -> str:
    return "(" + ", ".join(f"{float(v):.6g}" for v in values) + ")"


def _material(path: str, rgb) -> str:
    name = path.rsplit("/", 1)[1]
    return "\n".join(
        [
            f'        def Material "{name}"',
            "        {",
            f"            token outputs:surface.connect = <{path}/shader.outputs:surface>",
            '            def Shader "shader"',
            "            {",
            '                uniform token info:id = "UsdPreviewSurface"',
            f"                color3f inputs:diffuseColor = {_f(rgb)}",
            f"                float inputs:roughness = {ROUGHNESS}",
            "                float inputs:metallic = 0",
            "                token outputs:surface",
            "            }",
            "        }",
        ]
    )


def _cube(name: str, pos, quat, half, material: str, indent: str) -> str:
    """A USD Cube (edge 2) scaled to MuJoCo box half-extents, at `pos`/`quat` (w, x, y, z) in its parent's frame."""
    return "\n".join(
        [
            f'{indent}def Cube "{name}" (',
            f'{indent}    prepend apiSchemas = ["MaterialBindingAPI"]',
            f"{indent})",
            f"{indent}{{",
            f"{indent}    double size = 2",
            f"{indent}    double3 xformOp:translate = {_f(pos)}",
            f"{indent}    quatf xformOp:orient = {_f(quat)}",
            f"{indent}    float3 xformOp:scale = {_f(half)}",
            f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
            f"{indent}    rel material:binding = <{material}>",
            f"{indent}}}",
        ]
    )


def srgb_to_linear(rgb) -> tuple:
    """An sRGB display colour (0-1 per channel) as the linear value a renderer's albedo takes."""
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)


def _key(rgb) -> str:
    return "m_" + "_".join(f"{int(round(c * 255)):03d}" for c in rgb)


def scene_boxes(scene_xml: Path, target_body: str):
    """(static, movable): visible non-robot boxes as (name, pos, quat, half, linear rgb), static in world, movable in
    body frame.

    Robot geoms (the pelvis's subtree) are the robot USD's. Any other visible geom that is not a box is refused:
    this builder draws primitives only, and a silently missing object would be a scene the policy never trained on.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pelvis = model.body("pelvis").id
    target = model.body(target_body).id

    def under(body: int, root: int) -> bool:
        while body not in (0, root):
            body = int(model.body_parentid[body])
        return body == root

    static, movable = [], []
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        if under(body, pelvis) or model.geom_rgba[g][3] == 0:
            continue
        name = model.geom(g).name or f"geom_{g}"
        rgb = tuple(round(c, 5) for c in srgb_to_linear(float(c) for c in model.geom_rgba[g][:3]))
        if model.geom_type[g] != BOX:
            raise SystemExit(f"{scene_xml}: visible geom {name!r} is type {model.geom_type[g]}, not a box")
        half = model.geom_size[g].tolist()
        if under(body, target):
            if body != target:
                raise SystemExit(f"{scene_xml}: {name!r} hangs below the target body; only direct geoms are drawn")
            movable.append((name, model.geom_pos[g].tolist(), model.geom_quat[g].tolist(), half, rgb))
        else:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, data.geom_xmat[g])
            static.append((name, data.geom_xpos[g].tolist(), quat.tolist(), half, rgb))
    if not movable:
        raise SystemExit(f"{scene_xml}: the target body {target_body!r} has no visible box")
    return static, movable


def _layer(default_prim: str, doc: str, body: list[str]) -> str:
    return "\n".join(
        ["#usda 1.0", "(", f'    defaultPrim = "{default_prim}"', "    metersPerUnit = 1", '    upAxis = "Z"']
        + [f'    doc = "{doc}"', ")", ""]
        + body
        + [""]
    )


def build(scene_dir: Path, out: Path) -> dict:
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    key = meta["target_object"]
    xml = scene_dir / "scene_locomani.xml"
    static, movable = scene_boxes(xml, meta["objects"][key]["body"])
    static = [("ground", (0.0, 0.0, -GROUND_HALF[2]), (1, 0, 0, 0), GROUND_HALF, GROUND_RGB)] + static

    looks = "/World/Looks"
    scene = ['def Xform "World"', "{", '    def Scope "Looks"', "    {"]
    scene += [_material(f"{looks}/{_key(rgb)}", rgb) for rgb in sorted({s[4] for s in static})]
    scene += ["    }", '    def Xform "static"', "    {"]
    scene += [_cube(n, p, q, h, f"{looks}/{_key(rgb)}", "        ") for n, p, q, h, rgb in static]
    scene += ["    }", "}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.usda").write_text(
        _layer("World", f"{out.name}: primitive boxes from {xml} (scripts/build_isaac_primitive_scene.py)", scene)
    )

    obj_looks = f"/{key}/Looks"
    obj = [f'def Xform "{key}"', "{", '    def Scope "Looks"', "    {"]
    obj += [_material(f"{obj_looks}/{_key(rgb)}", rgb) for rgb in sorted({m[4] for m in movable})]
    obj += ["    }"]
    obj += [_cube(n, p, q, h, f"{obj_looks}/{_key(rgb)}", "    ") for n, p, q, h, rgb in movable]
    obj += ["}"]
    obj_dir = out / "usd" / key
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / f"{key}.usd").write_text(_layer(key, f"movable {key}: its MJCF boxes in the body frame", obj))
    (obj_dir / f"{key}.json").write_text(
        json.dumps({"scale": [1.0, 1.0, 1.0], "offset_pos": [0.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0]})
        + "\n"
    )
    source = {
        "scene_xml": str(xml.resolve()),
        "scene_xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "builder": "scripts/build_isaac_primitive_scene.py",
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "static": [s[0] for s in static],
        "movable": {key: [m[0] for m in movable]},
    }
    (out / "source.json").write_text(json.dumps(source, indent=1) + "\n")
    return source


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path, help="a MuJoCo scene directory: scene_locomani.xml + scene_meta.json")
    ap.add_argument("--label", help="stage directory name under <assets>/scene; default: the scene directory's name")
    ap.add_argument("--assets", default=os.environ.get("ISAAC_ASSETS", "assets/isaac"))
    args = ap.parse_args(argv)
    out = Path(args.assets).resolve() / "scene" / (args.label or args.scene_dir.resolve().name)
    source = build(args.scene_dir.resolve(), out)
    print(f"wrote {out}/scene.usda ({len(source['static'])} static boxes) and usd/: {source['movable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
