#!/usr/bin/env python3
"""Install the packaged logistics MJCF as a box-probe runtime scene; `--rsi` adds the scene-file v2 fields and
`env_only.xml` the scene RSI needs (TODO 38.1, docs/handoff/box-bimanual-rsi-plan-20261023.md)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # `python scripts/prepare_logistics_scene.py` from the repository root

from locomani.sim import make_dex1_model  # noqa: E402

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


ROOM = "logistics_room"  # the one room; tasks and spawn regions name it
BOX_KEY, BOX_MASS_KG = "nor_cal_box", 1.2  # assets/scene/logistics_room_box_v1/README.md


def rsi_fields(meta: dict, model, data, body_radius: float, source_scene: str) -> None:
    """Extend the v1 meta in place to the scene-file v2 fields the scene RSI reads (TODO 38.1): `room` from the
    meshes (scan_room), the platform's yaw / approach edges / room, the box's `semantic`, `randomization`."""
    from datetime import datetime, timezone

    from locomani.scene_discovery import scan_room

    occ, lo = scan_room.slice_occupancy(model, data)
    room = scan_room.room_block(occ, lo, scan_room.RES_M, meta["robot_start"]["xy"], body_radius=body_radius, name=ROOM)
    # the scene RSI films from a camera that follows the robot at the default 45 deg fovy (report_media.track), as on
    # the ADT apartment; the pack's fixed `reverse_room` shot (73.74 deg) made the Unreal level's film camera disagree
    # with the replays (TODO 38.20)
    meta.pop("video_camera", None)
    meta["schema_version"] = 2
    meta["source"] = {
        "kind": "scan",
        "pack": source_scene,
        "built_by": "scripts/prepare_logistics_scene.py --rsi",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta["room"] = room
    surf = meta["surfaces"]["A"]
    surf.update(yaw=0.0, approach_edges=["-x", "+x", "-y", "+y"], room=ROOM, phrase="pallet platform")
    surf["provenance"] = {"yaw": "rule", "approach_edges": "rule", "room": "rule", "top": "measured"}
    box = meta["objects"][BOX_KEY]
    box["semantic"] = {
        "phrase": meta["object_phrase"][BOX_KEY],
        "room": ROOM,
        "movable": True,
        "graspable": "bimanual",  # 0.32 m across its narrow side: wider than a Dex1 jaw (9.5 cm) either way
        "mass_kg": BOX_MASS_KG,
        "upright_axis": [0.0, 0.0, 1.0],
        "container": {"is": True, "opening_half_extent": [0.215, 0.15], "rim_height": 0.33},
    }
    box["provenance"] = {"semantic": "rule", "bimanual": "rule", "container": "rule", "mass_kg": "measured"}
    meta["randomization"] = {
        "movable_pool": [BOX_KEY],
        "placements": {BOX_KEY: ["A"]},
        "distractor_pool": [],
        "spawn_rooms": [ROOM],
    }


def write_env_only(tree: ET.ElementTree, destination: Path) -> None:
    """`env_only.xml`: the scene without the robot, for scene_settle.py's and scene_lint.py's rest tests. The robot's
    equalities, actuators and sensors name robot elements, so the pelvis subtree and those sections go; assets nothing
    references any more are pruned (the physics copy does the same, tasks/scene/world.physics_scene)."""
    import copy

    root = tree.getroot()
    env = ET.Element("mujoco", model="logistics_env_only")
    for tag in ("compiler", "option", "statistic", "visual", "default", "asset"):
        el = root.find(tag)
        if el is not None:
            env.append(copy.deepcopy(el))
    world = ET.SubElement(env, "worldbody")
    for el in root.find("worldbody"):
        if not (el.tag == "body" and el.get("name") == "pelvis"):
            world.append(copy.deepcopy(el))
    geoms = list(env.iter("geom"))
    meshes = {g.get("mesh") for g in geoms if g.get("mesh")}
    mats = {g.get("material") for g in geoms if g.get("material")}
    asset = env.find("asset")
    for el in list(asset):
        if (el.tag == "mesh" and el.get("name") not in meshes) or (el.tag == "material" and el.get("name") not in mats):
            asset.remove(el)
    texs = {m.get("texture") for m in asset.iter("material") if m.get("texture")}
    for el in list(asset):
        if el.tag == "texture" and el.get("type") != "skybox" and el.get("name") not in texs:
            asset.remove(el)
    path = destination / "env_only.xml"
    ET.ElementTree(env).write(path, encoding="unicode")
    mujoco.MjModel.from_xml_path(str(path))  # it must compile on its own


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="new runtime directory under outputs/")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--robot",
        default="stock",
        choices=("stock", "mars"),
        help="the pack carries SONIC's G1 with Dex1 (stock); mars rewrites it into the lab's real G1 "
        "(make_dex1_model.apply_mars: kinematics, inertia, meshes, torque limits, the ego and D405 wrist cameras)",
    )
    parser.add_argument(
        "--ego-camera",
        choices=sorted(make_dex1_model.MARS_EGO_LINKS),
        default=None,
        help=f"mars only: which head camera is the ego view (head_camera); default {make_dex1_model.MARS_EGO_DEFAULT}, "
        "the D455 on the pan/tilt bracket; d435_head is the D435 fixed in the face. Recorded as scene_meta.json "
        "`ego_camera`",
    )
    parser.add_argument(
        "--rsi",
        action="store_true",
        help="also write what the scene RSI needs (TODO 38.1): the v2 scene-file fields -- `room` sliced from the "
        "meshes (scene_discovery.scan_room: bounds, obstacles, the spawn region), the surface's yaw and approach "
        "edges, the object's `semantic`, `randomization` -- and `env_only.xml` (the scene without the robot, for "
        "scene_settle.py and scene_lint.py). Put the output under outputs/scene_discovery/ (the sandbox links that "
        "directory and the leak audit expects a scene there)",
    )
    parser.add_argument(
        "--body-radius",
        type=float,
        default=0.35,
        help="--rsi: the navigation body radius the spawn region keeps from any obstacle "
        "(locomani.tasks.scene.world.BODY_RADIUS_M, not imported: it pulls in the simulator)",
    )
    args = parser.parse_args()
    if args.ego_camera and args.robot != "mars":
        raise SystemExit(f"--ego-camera picks one of mars's head cameras; robot {args.robot} has one ego camera")
    ego_camera = args.ego_camera or make_dex1_model.MARS_EGO_DEFAULT
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
    if args.robot == "mars":
        make_dex1_model.write_mars()  # the meshes the scene references (and the robot model) are gitignored
        tree = make_dex1_model.apply_mars(tree, make_dex1_model.MARS_MESHES, ego_camera)
    if args.rsi:  # scene_settle.py rewrites a free body's `pos="..." quat="..."` pair; the pack's box has no quat
        box_body = tree.getroot().find("worldbody").find("body[@name='nor_cal_box_object']")
        box_body.set("quat", box_body.get("quat") or "1 0 0 0")
    tree.write(runtime_xml, encoding="unicode")
    if args.robot == "mars":
        make_dex1_model.check_mars(str(runtime_xml))

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
        "robot": args.robot,
        **({"ego_camera": ego_camera} if args.robot == "mars" else {}),
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
    if args.rsi:
        rsi_fields(meta, model, data, args.body_radius, source_scene)
        write_env_only(tree, destination)
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
                **(
                    {
                        "rsi": {
                            "bounds": meta["room"]["bounds"],
                            "obstacles": len(meta["room"]["obstacles"]),
                            "spawn_rectangles": len(meta["room"]["spawn"][ROOM]),
                            "env_only": str(destination / "env_only.xml"),
                        }
                    }
                    if args.rsi
                    else {}
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
