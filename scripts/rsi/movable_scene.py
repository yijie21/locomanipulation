"""TODO 33.24.1 (user 2026-10-24: chairs and stools in front of a counter can be pushed aside before a grasp; a scene
variant, the scene itself stays frozen): build a copy of a scene in which some floor furniture is free to move.

The ADT apartment's chairs and stools are part of the fixed environment body (`adt_env`): a visual mesh (`adt_fix_K`),
a collision mesh (`adt_fixcol_K`), upper collision boxes (`adt_upcol_K_###`), the seat's slab (`adt_slab_K__0`, a
surface) and its panels (`adt_panel_K__0_*`); the step stool has one collision mesh (`adt_obj_StepStool_col`). For
each key in MOVABLE this moves those geoms into a new body `K_object` at the object's pose, with a free joint, the
scene file's mass (`semantic.mass_kg`) and a box inertia from its half extents, and makes every collider of it collide
with the floor, the walls, the support slabs and panels, the free objects, each other and the robot -- not with the
other furniture's collision meshes, whose convex hulls fill the space under a table top (contype CONTYPE,
conaffinity CONAFFINITY; the robot's conaffinity is 5, the furniture's contype 4, the walls' 6, the free objects'
conaffinity 3).
The scene file marks them free and movable and drops the seat surfaces they carried (a seat that moves is not a
support the planners may count on). Everything else is the source scene's: the small files are copied, the mesh
directory linked; SceneWorld builds the physics copy (`<dir>_physics`) on first use.

    python3 scripts/rsi/movable_scene.py build [--src outputs/scene_discovery/adt_apartment_mars] [--dst ...] [--keys K ...]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars"
DST = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars_movable"
MOVABLE = (
    "BlackKitchenChair",
    "BlackKitchenChair_B",
    "YellowDiningChair",
    "WhiteChair",
    "BlackBarStool_B",
    "StepStool",
)
CONTYPE, CONAFFINITY = "5", "3"  # not 7: the other furniture collides by its collision mesh's convex hull, solid under
# a table top -- a dining chair tucked under the table starts 20-30 cm inside it and is thrown out (2026-10-24)
FRICTION = "0.6 0.005 0.0001"
SMALL_FILES = ("colliders.json", "floorplan.png", "panels.json", "settle.json", "shadowed.json", "surface_heights.json",
               "upright.json")  # fmt: skip


def _q(s: str | None) -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0]) if s is None else np.array([float(x) for x in s.split()])


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])  # fmt: skip


def _qinv(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]]) / float(np.dot(q, q))


def _rot(q) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def owned(name: str, key: str) -> bool:
    """A geom of object `key`: adt_<kind>_<key> exactly, or followed by __<n>[_side], _<nnn> or _col -- never another
    key that only starts with this one (BlackKitchenChair vs BlackKitchenChair_B)."""
    return re.fullmatch(rf"adt_[a-z]+_{re.escape(key)}(?:__\d+(?:_[a-z]+)?|_\d{{3}}|_col)?", name or "") is not None


def build(src: Path = SRC, dst: Path = DST, keys=MOVABLE) -> dict:
    meta = json.loads((src / "scene_meta.json").read_text())
    tree = ET.parse(src / "scene_locomani.xml")
    world = tree.getroot().find("worldbody")
    env = world.find("body[@name='adt_env']")
    if env is None or env.get("pos") not in (None, "0 0 0") or env.get("quat") not in (None, "1 0 0 0"):
        raise SystemExit("expected a fixed environment body adt_env at the origin")
    done = {}
    for key in keys:
        o = meta["objects"][key]
        geoms = [g for g in list(env) if g.tag == "geom" and owned(g.get("name"), key)]
        if not geoms:
            raise SystemExit(f"{key}: no geoms of it in adt_env")
        p_b, q_b = np.asarray(o["init_pos"], float), np.asarray(o["init_quat"], float)
        R_b = _rot(q_b)
        body = ET.SubElement(world, "body", name=f"{key}_object", pos=_fmt(p_b), quat=_fmt(q_b))
        ET.SubElement(body, "freejoint", name=f"{key}_free")
        mass = float((o.get("semantic") or {}).get("mass_kg") or 5.0)
        hx, hy, hz = (2 * np.asarray(o["box"]["half_extent"], float)) ** 2
        ET.SubElement(body, "inertial", mass=f"{mass:.4g}", pos=_fmt(o["box"].get("centre", (0, 0, 0))),
                      diaginertia=_fmt([mass / 12 * (hy + hz), mass / 12 * (hx + hz), mass / 12 * (hx + hy)]))  # fmt: skip
        colliders = 0
        for g in geoms:
            env.remove(g)
            p_g = np.array([float(x) for x in (g.get("pos") or "0 0 0").split()])
            g.set("pos", _fmt(R_b.T @ (p_g - p_b)))
            g.set("quat", _fmt(_qmul(_qinv(q_b), _q(g.get("quat")))))
            if not (g.get("contype") == "0" and g.get("conaffinity") == "0"):  # a collider (visual geoms stay visual)
                g.set("contype", CONTYPE)
                g.set("conaffinity", CONAFFINITY)
                g.set("density", "0")
                g.set("friction", FRICTION)
                colliders += 1
            body.append(g)
        o.update(free=True, body=f"{key}_object", movable=True)
        o.pop("body_id", None)
        o.pop("qpos_adr", None)
        o.setdefault("semantic", {})["movable"] = True
        dropped = [s for s in list(meta["surfaces"]) if s.startswith(f"{key}__")]
        for s in dropped:
            meta["surfaces"].pop(s)
        done[key] = dict(geoms=len(geoms), colliders=colliders, mass_kg=mass, surfaces_dropped=dropped)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    tree.write(dst / "scene_locomani.xml")
    meta["movable_variant_of"] = str(src)
    (dst / "scene_meta.json").write_text(json.dumps(meta))
    for name in SMALL_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    (dst / "mesh").symlink_to(src / "mesh")
    (dst / "movable.json").write_text(json.dumps(done, indent=1))
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", type=Path, default=SRC)
    b.add_argument("--dst", type=Path, default=DST)
    b.add_argument("--keys", nargs="+", default=list(MOVABLE))
    a = ap.parse_args(argv)
    print(json.dumps(build(a.src.resolve(), a.dst.resolve(), a.keys), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
