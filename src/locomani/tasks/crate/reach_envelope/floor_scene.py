"""Floor-only G1 scene for ghost-reach probes: the bimanual crate scene with every object removed.

The physics (robot model, finger pads, gripper gear) is exactly the collection stack's; only the
table, crate and their reset metadata go. Ghost targets are synthetic task data, never scene bodies.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from locomani.tasks.crate import bimanual_scene

KEEP_WORLDBODY = {"geom": {"floor"}, "light": None, "body": {"pelvis"}, "camera": None}  # None = keep all of that tag
OVERVIEW = {
    "name": "reach_overview",
    "pos": "2.0621212 -1.7821211 1.5788951",
    "quat": "0.7474342435443343 0.543042762745792 0.22493567727786193 0.3095974006581382",
    "fovy": "45",
}
SIDE = {
    "name": "reach_side",
    "pos": "0.35 -2.1290762 1.0192221",
    "quat": "0.7547095870921283 0.6560590210882168 0 0",
    "fovy": "45",
}


def strip_to_floor(scene_dir: Path) -> dict:
    xml_path, meta_path = scene_dir / "scene_locomani.xml", scene_dir / "scene_meta.json"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    removed = []
    for child in list(worldbody):
        keep = KEEP_WORLDBODY.get(child.tag)
        if keep is None and child.tag in KEEP_WORLDBODY:
            continue
        if keep is not None and child.get("name") in keep:
            continue
        worldbody.remove(child)
        removed.append((child.tag, child.get("name")))
    for cam in (OVERVIEW, SIDE):
        if worldbody.find(f"camera[@name='{cam['name']}']") is None:
            ET.SubElement(worldbody, "camera", cam)
    # Equality/contact/keyframe entries that referenced removed bodies would fail to compile.
    for section in ("equality", "contact", "keyframe"):
        node = root.find(section)
        if node is not None and section != "equality":
            root.remove(node)
    tree.write(xml_path, encoding="unicode")
    meta = json.loads(meta_path.read_text())
    meta.update(
        objects={},
        surfaces={},
        video_camera="reach_overview",
        purpose="floor-only physical G1 reach; ghost targets are synthetic task data and offline overlay only",
    )
    meta["scene_xml"] = str(xml_path.resolve())
    meta["robot_start"]["xy"] = [0.0, 0.0]  # no table to walk to; probes reset the robot where it stands
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))  # must compile
    meta.update(nq=model.nq, nv=model.nv)
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"removed": removed, "scene_xml": str(xml_path), "nq": model.nq, "nv": model.nv}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    bimanual_scene.main([str(out)])
    result = strip_to_floor(out)
    print(
        f"[floor_scene] removed {len(result['removed'])} worldbody entries; nq={result['nq']} nv={result['nv']} -> {result['scene_xml']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
