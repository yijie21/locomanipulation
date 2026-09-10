"""Isolated scene variants and physics setup for expert/evaluation experiments.

Only reset/setup and audit code may use this module. It is not a VLA input or
action-executor dependency. Shared scene files and running simulations are never
modified; taller tables are vertically scaled copies, including their legs.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from locomani.tasks.bottle_visual import scene_legacy as B

STRATA = {
    "low_near": (0.6211385812222395, (0.04, 0.08)),
    "high_near": (0.7611385812222395, (0.04, 0.08)),
    "low_deep": (0.6211385812222395, (0.18, 0.24)),
}
BASE_SCENE = Path(B.R.SCENE)


def scene_directory(root, stratum):
    height, _ = STRATA[stratum]
    meta = json.loads((BASE_SCENE / "scene_meta.json").read_text())
    old_height = meta["surfaces"]["A"]["top"]
    if abs(height - old_height) < 1e-8:
        return BASE_SCENE
    out = Path(root) / f"table_{height:.6f}"
    tree = ET.parse(BASE_SCENE / "scene_locomani.xml")
    scale = height / old_height
    for mesh in tree.findall("./asset/mesh"):
        if mesh.get("name", "").startswith("dining_table_"):
            xyz = [float(x) for x in mesh.get("scale", "1 1 1").split()]
            xyz[2] *= scale
            mesh.set("scale", " ".join(str(x) for x in xyz))
    body = tree.find(".//body[@name='dining_table_dining_table']")
    xyz = [float(x) for x in body.get("pos").split()]
    xyz[2] *= scale
    body.set("pos", " ".join(str(x) for x in xyz))
    # The visible table mesh is non-colliding; an invisible box is the actual
    # contact surface. Scale its local placement and thickness as well.
    for geom in body.findall("geom"):
        if geom.get("type") == "box":
            for attribute in ("pos", "size"):
                xyz = [float(x) for x in geom.get(attribute).split()]
                xyz[2] *= scale
                geom.set(attribute, " ".join(str(x) for x in xyz))
    meta["surfaces"]["A"]["top"] = height
    meta["objects"]["water_bottle"]["init_pos"][2] += height - old_height
    meta["scene_xml"] = str((out / "scene_locomani.xml").resolve())
    values = {
        "scene_locomani.xml": ET.tostring(tree.getroot(), encoding="unicode"),
        "scene_meta.json": json.dumps(meta, indent=2),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        path = out / name
        if path.exists() and path.read_text() != value:
            raise ValueError(f"Scene asset changed: {path}")
        if not path.exists():
            path.write_text(value)
    return out


def make_sim(policy, directory=None):
    previous = B.R.SCENE
    try:
        B.R.SCENE = str(directory or BASE_SCENE)
        sim = B.Sim(policy.meta, video=False)
    finally:
        B.R.SCENE = previous
    for g in range(sim.m.ngeom):
        name = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, sim.m.geom_bodyid[g]) or ""
        if "dex1_finger" in name and sim.m.geom_contype[g]:
            sim.m.geom_priority[g] = 1
            sim.m.geom_condim[g] = 4
            sim.m.geom_solref[g] = (0.002, 1)
            sim.m.geom_solimp[g] = (0.995, 0.9999, 0.001, 0.5, 2)
    return sim
