"""Prepare the fixed-footprint 55 cm carry experiment; starts no GPU process.

This dated experiment deliberately supports only the inspected 62 cm scene.
The generic builder scales the whole table; this comparison holds XY constant.
"""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, destination):
    meta = json.loads((source / "scene_meta.json").read_text())
    table = meta["surfaces"]["A"]
    crate = meta["objects"]["blue_crate"]
    if table["top"] != 0.62 or crate["surface"] != "A":
        raise ValueError("requires the 62 cm blue-crate table scene")
    xml = source / "scene_locomani.xml"
    tree = ET.parse(xml)
    root = tree.getroot()
    body = root.find(f"./worldbody/body[@name='{table['body']}']")
    top = body.find(f"geom[@name='{table['body']}_top']")
    basket = root.find(f"./worldbody/body[@name='{crate['body']}']")
    assert body.get("pos") == "0 0 0.31"
    assert not any(body.get(k) for k in ("quat", "euler", "axisangle", "xyaxes", "zaxis"))
    assert np.allclose(np.fromstring(top.get("pos"), sep=" "), [0, 0, 0.29])
    assert float(top.get("size").split()[2]) == 0.02
    assert len(list(body)) == 3 and basket is not None
    body.set("pos", "0 0 0.275")
    top.set("pos", "0 0 0.255")
    mesh_names = [g.get("mesh") for g in body.findall("geom") if g.get("mesh")]
    assert len(mesh_names) == 2
    for name in mesh_names:
        assert len(root.findall(f".//geom[@mesh='{name}']")) == 1
        mesh = root.find(f"./asset/mesh[@name='{name}']")
        scale = np.fromstring(mesh.get("scale"), sep=" ")
        scale[2] *= 0.55 / 0.62
        mesh.set("scale", " ".join(f"{v:.15g}" for v in scale))
    pos = np.fromstring(basket.get("pos"), sep=" ")
    pos[2] -= 0.07
    basket.set("pos", " ".join(f"{v:.15g}" for v in pos))
    table["top"] = 0.55
    crate["init_pos"][2] -= 0.07
    meta["scene_xml"] = str(destination / xml.name)

    # copytree refuses to overwrite evidence, including partially prepared scenes.
    shutil.copytree(source, destination)
    tree.write(destination / xml.name, encoding="unicode")
    (destination / "scene_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    original = mujoco.MjModel.from_xml_path(str(xml))
    lowered = mujoco.MjModel.from_xml_path(str(destination / xml.name))
    same_fields = (
        "jnt_range",
        "jnt_type",
        "dof_damping",
        "dof_frictionloss",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_gear",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_priority",
        "eq_data",
        "body_mass",
        "body_quat",
    )
    for field in same_fields:
        np.testing.assert_array_equal(getattr(original, field), getattr(lowered, field), err_msg=field)
    tid = original.body(table["body"]).id
    cid = original.body(crate["body"]).id
    other_bodies = np.ones(original.nbody, dtype=bool)
    other_bodies[[tid, cid]] = False
    np.testing.assert_array_equal(original.body_pos[other_bodies], lowered.body_pos[other_bodies])
    np.testing.assert_array_equal(original.body_inertia[other_bodies], lowered.body_inertia[other_bodies])
    active = (original.geom_contype != 0) | (original.geom_conaffinity != 0)
    np.testing.assert_array_equal(original.geom_size[active], lowered.geom_size[active])
    qpos = original.qpos0.copy()
    qpos[crate["qpos_adr"] + 2] -= 0.07
    np.testing.assert_allclose(qpos, lowered.qpos0, atol=1e-12, rtol=0)
    top_id = original.geom(top.get("name")).id
    geometry = []
    for model in (original, lowered):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        geometry.append(
            {
                "top_world_z": float(data.geom_xpos[top_id, 2] + model.geom_size[top_id, 2]),
                "table_half_xy": model.geom_size[top_id, :2].tolist(),
                "crate_initial_xyz": data.xpos[cid].tolist(),
            }
        )
    np.testing.assert_allclose([g["top_world_z"] for g in geometry], [0.62, 0.55], atol=1e-12)
    np.testing.assert_array_equal(geometry[0]["table_half_xy"], geometry[1]["table_half_xy"])
    report = {
        "source": str(source),
        "destination": str(destination),
        "source_xml_sha256": digest(xml),
        "lowered_xml_sha256": digest(destination / xml.name),
        "identical_compiled_fields": list(same_fields),
        "geometry": geometry,
        "changes": "Static table body Z and tabletop local Z; table mesh Z scale only; "
        "crate initial Z and metadata top/initial Z. Table XY, active collision sizes, "
        "robot, crate shape and all controls retained. Static table mesh inertia changes.",
    }
    (destination / "height_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.destination.resolve())
