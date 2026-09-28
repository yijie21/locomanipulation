#!/usr/bin/env python3
"""Build the Isaac scene layer for a Blender-exported room (the warehouse), in the layout stage.py expects.

The crate/bottle scenes are assembled by scripts/build_isaac_assets.sh from table meshes and a PLY room;
a Blender room already ships as one USDC with materials, so this writes two small USD layers around it:

    <assets>/scene/<label>/scene.usda                 /World/room -> the USDC's /root, with its env DomeLight
                                                      (RTX rejects the exported EXR) and the static carton
                                                      deactivated; stage.assemble_stage then adds dome + sun
    <assets>/scene/<label>/usd/nor_cal_box/           the movable carton: the same /root with every other
        nor_cal_box.usd, nor_cal_box.json             child deactivated, so its /root/_materials bindings still
                                                      resolve, plus the body->geom offset for sync_replay

Both layers also block every UsdPreviewSurface `normal` connection that a single-channel texture feeds. A
tangent-space normal map has three channels, so a grey one there can only be a Bump height map. Blender's exporter
wires those in, and the v1 USDC predates blender_to_usd.py dropping them. Left in, the carton photos' heights
(0.50 +- 0.03) become an about-zero normal whose sign flips texel to texel, and every carton renders under white
speckle (2026-09-28, docs/ledger/blender_scenes.md).

Why the offset: the Blender carton mesh is baked in world coordinates at the scene's rest pose (floor centre
at (0, -0.39, 0.371)); the MJCF body nor_cal_box_object has its origin at that floor centre. The json's
offset_pos moves the mesh so the wrapper origin is the body origin; stage.py applies it.

A resized carton (`--scene`, a four-edge scene from locomani.tasks.crate.warehouse_box_scene): the stage is the same
room with the carton's mesh scale read from that MuJoCo scene, so both draw one box. The scale applies to the baked
world coordinates before the offset, so the offset scales with it; source.json records the MJCF's sha256, and
run_context.isaac_scene refuses the stage once that scene changes.

Needs the `pxr` module (pip install usd-core, or any Isaac/Blender python) to list the room's children.

    python3 scripts/build_isaac_warehouse_scene.py [--usdc assets/scene/logistics_room_box_v1/isaac/logistics_room_box.usdc]
                                                   [--label warehouse_textured] [--assets $ISAAC_ASSETS]
                                                   [--scene outputs/scene/warehouse_box_45x20x18]  (label: its name)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

CARTON = "NOR_CAL_carton___45_x_32_x_33_cm"
ENV_LIGHT = "env_light"
OFFSET_POS = [0.0, 0.39, -0.371]  # world floor centre of the baked carton mesh -> body origin
PNG_CHANNELS = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}  # PNG colour type -> channel count


def carton_scale(scene_xml: Path) -> list[float]:
    """The one `scale` every carton photo mesh (`nor_cal_*`) carries in a MuJoCo scene; [1, 1, 1] when unset."""
    meshes = [m for m in ET.parse(scene_xml).getroot().iter("mesh") if (m.get("name") or "").startswith("nor_cal_")]
    scales = {tuple(float(v) for v in (m.get("scale") or "1 1 1").split()) for m in meshes}
    if len(scales) != 1:
        raise SystemExit(f"{scene_xml}: expected one scale on the nor_cal_* meshes, found {sorted(scales) or 'none'}")
    return list(scales.pop())


def texture_channels(path: str) -> int | None:
    """Channel count read from a PNG header; None for anything that is not a readable PNG (left connected)."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
    except OSError:
        return None
    return PNG_CHANNELS.get(head[25]) if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) == 26 else None


def height_normal_shaders(stage, usdc: Path) -> list[str]:
    """Shader paths relative to /root whose UsdPreviewSurface `normal` is fed by a single-channel texture."""
    from pxr import UsdShade  # noqa: PLC0415

    found = []
    for prim in stage.Traverse():
        shader = UsdShade.Shader(prim)
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        normal = shader.GetInput("normal")
        if not normal or not normal.HasConnectedSource():
            continue
        file = UsdShade.Shader(normal.GetConnectedSource()[0].GetPrim()).GetInput("file")
        asset = file.Get() if file else None
        if asset is None:
            continue
        path = asset.resolvedPath or str(usdc.parent / asset.path)
        if texture_channels(path) in (1, 2):
            found.append(str(prim.GetPath().MakeRelativePath("/root")))
    return found


def blocked_normal_overs(shaders: list[str], pad: str) -> list[str]:
    """Nested `over`s authoring an explicit empty `inputs:normal` connection on each shader path."""
    tree: dict = {}
    for path in shaders:
        node = tree
        for part in path.split("/"):
            node = node.setdefault(part, {})

    def emit(node: dict, pad: str) -> list[str]:
        lines = []
        for name, sub in node.items():
            lines += [f'{pad}over "{name}"', f"{pad}{{"]
            lines += emit(sub, pad + "    ") if sub else [f"{pad}    normal3f inputs:normal.connect = None"]
            lines.append(f"{pad}}}")
        return lines

    return emit(tree, pad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usdc", default="assets/scene/logistics_room_box_v1/isaac/logistics_room_box.usdc")
    ap.add_argument("--label", default=None, help="default: the --scene directory's name, else warehouse_textured")
    ap.add_argument("--assets", default=os.environ.get("ISAAC_ASSETS", "assets/isaac"))
    ap.add_argument("--scene", type=Path, default=None, help="MuJoCo scene dir whose carton size the stage takes")
    args = ap.parse_args()
    try:
        from pxr import Usd  # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            f"{sys.executable} has no pxr module: pip install usd-core into it, or run this with a python that has one "
            "(scripts/build_scene.sh <scene> --isaac picks one; USD_PY overrides)"
        ) from None

    scene_xml = args.scene / "scene_locomani.xml" if args.scene else None
    scale = carton_scale(scene_xml) if scene_xml else [1.0, 1.0, 1.0]
    args.label = args.label or (args.scene.resolve().name if args.scene else "warehouse_textured")
    usdc = Path(args.usdc).resolve()
    stage = Usd.Stage.Open(str(usdc))
    children = [c.GetName() for c in stage.GetPrimAtPath("/root").GetChildren()]
    if CARTON not in children or ENV_LIGHT not in children:
        raise SystemExit(f"{usdc} has no /root/{CARTON} or /root/{ENV_LIGHT}; is this the logistics room?")
    heights = height_normal_shaders(stage, usdc)
    out = Path(args.assets).resolve() / "scene" / args.label
    (out / "usd" / "nor_cal_box").mkdir(parents=True, exist_ok=True)

    def over_inactive(name: str) -> str:
        return f'    over "{name}" (\n        active = false\n    )\n    {{\n    }}'

    scene = "\n".join(
        [
            "#usda 1.0",
            "(",
            '    defaultPrim = "World"',
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            f'    doc = "{args.label}: the Blender-exported logistics room as /World/room (generated by '
            "scripts/build_isaac_warehouse_scene.py). The USDC env DomeLight and the static carton are "
            'deactivated; stage.assemble_stage adds dome + distant lights and drives the carton from usd/nor_cal_box."',
            ")",
            "",
            'def Xform "World"',
            "{",
            '    def Xform "room" (',
            f"        references = @{usdc}@</root>",
            "    )",
            "    {",
            "    " + over_inactive(ENV_LIGHT).replace("\n", "\n    "),
            "    " + over_inactive(CARTON).replace("\n", "\n    "),
            *blocked_normal_overs(heights, "        "),
            "    }",
            "}",
            "",
        ]
    )
    (out / "scene.usda").write_text(scene)
    keep = {CARTON, "_materials"}
    carton = "\n".join(
        [
            "#usda 1.0",
            "(",
            '    defaultPrim = "nor_cal_box"',
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            '    doc = "movable NOR-CAL carton for the warehouse Isaac replay: references the Blender room root so the '
            "carton meshes keep their /root/_materials bindings, with every other room child deactivated "
            '(generated by scripts/build_isaac_warehouse_scene.py)"',
            ")",
            "",
            'def Xform "nor_cal_box" (',
            f"    references = @{usdc}@</root>",
            ")",
            "{",
            *[over_inactive(n) for n in children if n not in keep],
            *blocked_normal_overs(heights, "    "),
            "}",
            "",
        ]
    )
    (out / "usd" / "nor_cal_box" / "nor_cal_box.usd").write_text(carton)
    (out / "usd" / "nor_cal_box" / "nor_cal_box.json").write_text(
        json.dumps(
            {
                "scale": scale,
                "offset_pos": [o * s for o, s in zip(OFFSET_POS, scale)],
                "offset_quat": [1.0, 0.0, 0.0, 0.0],
                "note": "the USDC carton mesh is baked in world coordinates at the scene's rest pose (bbox x +-0.225, "
                "y -0.55..-0.23, z 0.371..0.701); this offset moves its floor centre to the wrapper origin so "
                "/World/nor_cal_box follows the MJCF body nor_cal_box_object (origin at the floor centre)",
            },
            indent=1,
        )
        + "\n"
    )
    if scene_xml:
        source = {
            "scene_xml": str(scene_xml.resolve()),
            "scene_xml_sha256": hashlib.sha256(scene_xml.read_bytes()).hexdigest(),
            "builder": "scripts/build_isaac_warehouse_scene.py",
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "carton_scale": scale,
        }
        (out / "source.json").write_text(json.dumps(source, indent=1) + "\n")
    print(
        f"wrote {out}/scene.usda and usd/nor_cal_box/ ({len(children) - len(keep)} room children deactivated, "
        f"{len(heights)} height-map normal connections blocked, carton scale {scale})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
