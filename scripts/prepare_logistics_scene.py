#!/usr/bin/env python3
"""Install the packaged logistics MJCF as a box-probe runtime scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "assets/scene/logistics_room_box_v1/mujoco"
SONIC_MESHES = ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/meshes"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_references(tree: ET.ElementTree, source: Path, destination: Path) -> None:
    for element in tree.getroot().iter():
        value = element.get("file")
        if value is None:
            continue
        resolved = (source / value).resolve()
        if not resolved.is_file():
            sonic_candidate = SONIC_MESHES / Path(value).name
            if not sonic_candidate.is_file():
                raise FileNotFoundError(f"unresolved source asset: {value}")
            resolved = sonic_candidate.resolve()
        # Assets copied along with the scene (mesh/, textures/) are written relative to the scene dir itself
        # ("mesh/room_00.obj"), so any copy of the dir resolves them; anything outside the scene dir
        # (the robot meshes under third_party) is written absolute: a run copies the scene dir one level
        # deeper (outputs/runs/<run>/<scene>/), where "../../../third_party" no longer resolves (2026-09-24).
        if resolved.is_relative_to(source):
            element.set("file", resolved.relative_to(source).as_posix())  # travels with the copied dir
        else:
            element.set("file", str(resolved))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="new runtime directory under outputs/")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.out_dir.resolve()
    if destination.exists():
        raise SystemExit(f"refusing to replace existing runtime scene: {destination}")
    source_xml = source / "scene_locomani.xml"
    if not source_xml.is_file():
        raise SystemExit(f"source MJCF does not exist: {source_xml}")

    shutil.copytree(source, destination)
    runtime_xml = destination / "scene_locomani.xml"
    tree = ET.parse(runtime_xml)
    _relative_references(tree, source, destination)
    world = tree.getroot().find("worldbody")
    assert world is not None
    platform_geom = world.find("geom[@name='platform_col']")
    if platform_geom is None:
        raise AssertionError("source scene has no platform_col")
    world.remove(platform_geom)
    platform = ET.SubElement(world, "body", name="logistics_platform", pos="0 0 0")
    platform.append(platform_geom)
    tree.write(runtime_xml, encoding="unicode")

    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def object_id(kind, name: str) -> int:
        value = int(mujoco.mj_name2id(model, kind, name))
        if value < 0:
            raise AssertionError(f"compiled scene has no {name}")
        return value

    box_body = object_id(mujoco.mjtObj.mjOBJ_BODY, "nor_cal_box_object")
    platform_body = object_id(mujoco.mjtObj.mjOBJ_BODY, "logistics_platform")
    platform_geom_id = object_id(mujoco.mjtObj.mjOBJ_GEOM, "platform_col")
    qpos_address = int(model.jnt_qposadr[model.body_jntadr[box_body]])
    half = model.geom_size[platform_geom_id, :2].tolist()
    xy = data.geom_xpos[platform_geom_id, :2].tolist()
    top = float(data.geom_xpos[platform_geom_id, 2] + model.geom_size[platform_geom_id, 2])
    quat = model.qpos0[3:7]
    yaw_deg = math.degrees(
        math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
    )
    try:
        source_scene = str(source_xml.relative_to(ROOT))
    except ValueError:
        source_scene = str(source_xml)
    meta = {
        "scene_xml": str(runtime_xml),
        "target_object": "nor_cal_box",
        "target_pool": ["nor_cal_box"],
        "object_phrase": {"nor_cal_box": "open cardboard box"},
        "goal_object": None,
        "robot_start": {
            "xy": model.qpos0[:2].tolist(),
            "yaw_deg": yaw_deg,
            "pelvis_z": float(model.qpos0[2]),
        },
        "surfaces": {
            "A": {
                "name": "HP pallet platform",
                "body": "logistics_platform",
                "body_id": platform_body,
                "xy": xy,
                "top": top,
                "half_extent": half,
                "flat_half_extent": half,
                "circumradius": math.hypot(*half),
            }
        },
        "objects": {
            "nor_cal_box": {
                "body": "nor_cal_box_object",
                "body_id": box_body,
                "qpos_adr": qpos_address,
                "free": True,
                "surface": "A",
                "init_pos": model.qpos0[qpos_address : qpos_address + 3].tolist(),
                "init_quat": model.qpos0[qpos_address + 3 : qpos_address + 7].tolist(),
                "height": 0.33,
                "scale": 1.0,
                "bimanual": True,
                "container": True,
                "box": {"half_extent": [0.225, 0.16, 0.165], "centre": [0.0, 0.0, 0.165]},
            }
        },
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nq": int(model.nq),
        "source_scene": source_scene,
        "source_sha256": _sha256(source_xml),
        "snapshot_change": (
            "Wrapped unchanged platform_col in a named identity body for privileged surface "
            "contact diagnostics; generated metadata from the compiled model."
        ),
        "video_camera": "reverse_room",
    }
    meta_path = destination / "scene_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        json.dumps(
            {
                "scene": str(destination),
                "scene_xml": str(runtime_xml),
                "scene_meta": str(meta_path),
                "target_key": "nor_cal_box",
                "surface_top_m": top,
                "box_qpos_adr": qpos_address,
                "compiled": {"nbody": model.nbody, "ngeom": model.ngeom, "nq": model.nq},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
