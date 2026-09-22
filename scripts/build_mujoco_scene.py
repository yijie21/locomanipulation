#!/usr/bin/env python3
"""Build a MuJoCo scene that holds the same layout as the Isaac stage.

    python scripts/build_mujoco_scene.py outputs/scene/<name> [--base <scene_locomani.xml>]

This is surgery on an existing scene_locomani.xml rather than a scene written
from scratch, so the G1 body tree, the joint defaults and the actuators come
across untouched: a rebuilt robot would be a new calibration problem, and this
line already has one.

What it adds, and the reasoning behind the split:

* the room, one visual geom per material, contype=0 -- scenery, never contact.
* one box collider for the pallet stack. That stack IS the table: its top face
  is the work surface.
* the carton as a body with a free joint and FIVE box colliders (floor and four
  walls). A convex hull would be a solid block no hand could enter, which is the
  same reason the blue crate is eleven boxes rather than one.
* the ground plane keeps colliding but stops being drawn. A plane is the exact,
  cheap, well-behaved collider for a flat floor, but the base scene's blue-grey
  checker material was fighting the room's own floor mesh.
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "outputs/scene/crate_table30_depth04/scene_locomani.xml"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("scene_dir", help="the directory blender_to_mujoco.py wrote")
parser.add_argument("--base", default=str(DEFAULT_BASE), help="scene_locomani.xml to inherit the robot from")
parser.add_argument("--platform-top", type=float, default=0.371)
parser.add_argument("--near-edge-y", type=float, default=-0.555)
args = parser.parse_args()

MJ = os.path.abspath(args.scene_dir)  # MuJoCo resolves asset paths from its own cwd
BASE = args.base
TEX = f"{MJ}/textures"
OUT = f"{MJ}/scene_locomani.xml"
PLATFORM_TOP, NEAR_EDGE_Y = args.platform_top, args.near_edge_y
PLAT_X, PLAT_Y = (-0.60, 0.60), (NEAR_EDGE_Y, NEAR_EDGE_Y + 0.90)
BOX = dict(x=0.45, y=0.32, z=0.33, wall=0.005)
BOX_POS = (0.0, NEAR_EDGE_Y + BOX["y"] / 2 + 0.005, PLATFORM_TOP)
PELVIS = (0.0, -1.205, 0.793)

tree = ET.parse(BASE)
root = tree.getroot()
asset = root.find("asset")
world = root.find("worldbody")

# drop the old scene furniture; keep floor, light, and the whole robot tree.
# Also drop everything a previous run of this script added: the base may be our
# own output (a re-run, or --base pointing at the scene being built), and
# MuJoCo rejects a second mesh named room_00.
OWN = ("room_", "t_room_", "m_room_", "vis_room_", "nor_cal", "m_nor_cal", "platform_col", "reverse_room", "room_light")


def ours(el):
    return (el.get("name") or "").startswith(OWN)


for body in list(world.findall("body")):
    if body.get("name") != "pelvis":
        world.remove(body)
for el in list(world):
    if el.tag in ("geom", "camera", "light") and ours(el):
        world.remove(el)
for tag in ("texture", "material", "mesh"):
    for el in list(asset.findall(tag)):
        n = el.get("name") or ""
        if any(k in n for k in ("dining_table", "counter", "blue_crate")) or ours(el):
            asset.remove(el)

parts = json.load(open(f"{MJ}/room_parts.json"))
texmap = json.load(open(f"{MJ}/texture_map.json"))
for _p in parts:
    _p["texture"] = texmap.get(_p["texture"])  # MuJoCo reads PNG only
used = set()
for p in parts:
    name, tex, rgb = p["name"], p["texture"], p["rgb"]
    ET.SubElement(asset, "mesh", name=name, file=f"mesh/{name}.obj")
    if tex and os.path.exists(f"{TEX}/{tex}") and tex not in used:
        ET.SubElement(asset, "texture", type="2d", colorspace="linear", name=f"t_{name}", file=f"textures/{tex}")
        used.add(tex)
    if tex and os.path.exists(f"{TEX}/{tex}"):
        ET.SubElement(asset, "material", name=f"m_{name}", texture=f"t_{name}", specular="0.1", shininess="0.1")
    else:
        ET.SubElement(
            asset, "material", name=f"m_{name}", rgba=f"{rgb[0]} {rgb[1]} {rgb[2]} 1", specular="0.1", shininess="0.1"
        )
ET.SubElement(asset, "mesh", name="nor_cal_box", file="mesh/nor_cal_box.obj")
ET.SubElement(asset, "material", name="m_nor_cal", rgba="0.86 0.78 0.62 1", specular="0.05", shininess="0.05")

# room: visual only. Walls and racks are scenery; the robot must not collide
# with 489k triangles, and nothing in the task touches them.
for p in parts:
    ET.SubElement(
        world,
        "geom",
        name=f"vis_{p['name']}",
        type="mesh",
        mesh=p["name"],
        material=f"m_{p['name']}",
        contype="0",
        conaffinity="0",
        group="1",
        density="0",
    )

# the pallet stack IS the table: one collider box under the visual boxes
hx, hy = (PLAT_X[1] - PLAT_X[0]) / 2, (PLAT_Y[1] - PLAT_Y[0]) / 2
cx, cy = (PLAT_X[0] + PLAT_X[1]) / 2, (PLAT_Y[0] + PLAT_Y[1]) / 2
ET.SubElement(
    world,
    "geom",
    name="platform_col",
    type="box",
    pos=f"{cx} {cy} {PLATFORM_TOP / 2}",
    size=f"{hx} {hy} {PLATFORM_TOP / 2}",
    rgba="0.2 0.4 0.9 0",
    friction="1.0 0.02 0.001",
)

# the carton: hollow, like the crate asset -- a convex hull would be a solid
# block the hands could not enter. Floor plus four walls.
body = ET.SubElement(world, "body", name="nor_cal_box_object", pos=f"{BOX_POS[0]} {BOX_POS[1]} {BOX_POS[2]}")
ET.SubElement(body, "freejoint", name="nor_cal_box_free")
ET.SubElement(body, "inertial", pos=f"0 0 {BOX['z'] / 2}", mass="1.2", diaginertia="0.02 0.03 0.03")
ET.SubElement(
    body,
    "geom",
    name="nor_cal_vis",
    type="mesh",
    mesh="nor_cal_box",
    material="m_nor_cal",
    contype="0",
    conaffinity="0",
    group="1",
    density="0",
)
w, hxb, hyb, hzb = BOX["wall"], BOX["x"] / 2, BOX["y"] / 2, BOX["z"] / 2
walls = [
    ("floor", f"0 0 {w}", f"{hxb} {hyb} {w}"),
    ("xp", f"{hxb - w} 0 {hzb}", f"{w} {hyb} {hzb}"),
    ("xm", f"{-(hxb - w)} 0 {hzb}", f"{w} {hyb} {hzb}"),
    ("yp", f"0 {hyb - w} {hzb}", f"{hxb} {w} {hzb}"),
    ("ym", f"0 {-(hyb - w)} {hzb}", f"{hxb} {w} {hzb}"),
]
for n, pos, size in walls:
    ET.SubElement(
        body,
        "geom",
        name=f"nor_cal_{n}",
        type="box",
        pos=pos,
        size=size,
        friction="1.4 0.02 0.001",
        solref="0.002",
        solimp="0.998 0.998",
        rgba="0.86 0.78 0.62 0",
    )

# The floor: a plane is the exact, cheap, numerically well-behaved collider for
# a flat surface, so it stays -- but it must not be SEEN. The base scene's
# groundplane checker is blue-grey (rgb1 0.2 0.3 0.4) and was fighting the
# room's own floor mesh. Hide the plane and let the room supply every pixel.
floor = world.find("geom[@name='floor']")
if floor is not None:
    floor.attrib.pop("material", None)
    floor.set("rgba", "0.5 0.5 0.5 0")  # collides, never drawn
    floor.set("group", "3")  # and out of the default render groups
    floor.set("pos", "0 0 0.001")  # sit on the room floor's own top face
# the blue gradient skybox lit the room through the same reflection; this room
# has its own ceiling and walls, so a flat dark surround is honest here.
for t in list(asset.findall("texture")):
    if t.get("type") == "skybox":
        t.set("rgb1", "0.08 0.08 0.09")
        t.set("rgb2", "0.02 0.02 0.03")
vis_rgba = root.find("visual/rgba")
if vis_rgba is not None:
    vis_rgba.set("haze", "0.1 0.1 0.1 1")

pel = world.find("body[@name='pelvis']")
pel.set("pos", f"{PELVIS[0]} {PELVIS[1]} {PELVIS[2]}")
pel.set("quat", "0.70710678 0 0 0.70710678")  # +x forward -> face +y

# the Isaac "02 Reverse room" camera, same pose and the same 73.74 deg fovy
ET.SubElement(
    world,
    "camera",
    name="reverse_room",
    pos="-1.4 1.33 1.71",
    quat="-0.232547 -0.184719 0.593921 0.747702",
    fovy="73.740",
)

vis = root.find("visual")
g = vis.find("global")
if g is not None:
    g.set("offwidth", "960")
    g.set("offheight", "720")
ET.SubElement(
    world,
    "light",
    name="room_light",
    pos="0 0 2.9",
    dir="0 0 -1",
    directional="false",
    diffuse="0.6 0.6 0.6",
    specular="0.1 0.1 0.1",
)

# The calibrated base is generated under outputs/ and historically contains
# checkout-specific absolute references to the standard SONIC mesh install.
# Make inherited references portable from the generated scene directory too.
for element in root.iter():
    file_path = element.get("file")
    if file_path and os.path.isabs(file_path):
        element.set("file", os.path.relpath(file_path, MJ))

tree.write(OUT, encoding="unicode")
print(f"[mjcf] wrote {OUT}")
print(f"[mjcf] {len(parts)} room parts, platform {2 * hx:.2f}x{2 * hy:.2f} top {PLATFORM_TOP}, carton at {BOX_POS}")
