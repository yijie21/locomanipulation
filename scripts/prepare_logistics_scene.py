#!/usr/bin/env python3
"""Install the packaged logistics MJCF as a box-probe runtime scene; `--rsi` adds the scene-file v2 fields and
`env_only.xml` the scene RSI needs (TODO 38.1, docs/handoff/box-bimanual-rsi-plan-20261023.md); `--gso-seed N` adds
N-drawn Google Scanned Objects on the platform next to the carton (the `logistics_room_1030_s<N>` scenes, 2026-11-03,
docs/rsi_logistic_room_1030.md): the objects from `catalog.json`, the carton and every object at a random spot and
yaw on the platform, a top-down camera `platform_top`, and the scene file's `gso` block naming the draw."""

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

from locomani.sim import gso_objects as G  # noqa: E402
from locomani.sim import make_dex1_model  # noqa: E402

DEFAULT_SOURCE = ROOT / "assets/scene/logistics_room_box_v1/mujoco"
SONIC_MESHES = ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/meshes"
GSO_ASSETS = ROOT / "assets/mujoco_scanned_objects"


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
BOX_BODY = "nor_cal_box_object"
BOX_HALF = [0.225, 0.16, 0.165]  # the carton's outer 0.45 x 0.32 x 0.33 m; its body origin is the bottom centre
BOX_PHRASE = "open cardboard box"
PLATFORM_TOP_CAMERA = "platform_top"  # straight down on the platform, for the scene previews (scripts/gso_preview.py)
PLATFORM_TOP_HEIGHT_M = 2.0  # above the platform top: at fovy 45 the 4:3 frame spans about 1.7 x 2.2 m of it
DRAW_ATTEMPTS = 20  # draws of one seed tried before giving up on fitting its objects on the platform
# where the carton stands (user 2026-11-05 Q7): the two-hand hug's stance band, its long side along the edge it is
# nearest to, that edge drawn at random (box_bimanual's `box_long_side` placement: depth 0.18-0.31 m, yaw 85-90 deg);
# the put-in's reach (skills/into.MAX_DEPTH_M 0.40) covers it from the same edge; the task rows repeat it
CARTON_PLACEMENT = {"depth_m": [0.18, 0.31], "yaw_deg": [85, 90]}
REACH_DEPTH_M = 0.35  # a target's centre at most this far inside the nearest approach edge (D13: forward reach
# 0.6-0.7 m from a pelvis standing about 0.3 m outside the edge; the feasibility checker says the rest)


def rsi_fields(meta: dict, model, data, body_radius: float, source_scene: str, built_by: str) -> None:
    """Extend the v1 meta in place to the scene-file v2 fields the scene RSI reads (TODO 38.1): `room` from the
    meshes (scan_room), the platform's yaw / approach edges / room, the box's `semantic`, `randomization` over every
    free object (the GSO objects carry their `semantic` already, gso_objects.object_record)."""
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
        "built_by": built_by,
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
        "mass_kg": (meta.get("gso") or {}).get("carton_mass_kg", BOX_MASS_KG),
        "upright_axis": [0.0, 0.0, 1.0],
        "container": {"is": True, "opening_half_extent": [0.215, 0.15], "rim_height": 0.33},
    }
    box["provenance"] = {  # one source per semantic field (scene_file.validate); `bimanual` is the legacy flag's
        "semantic": "rule", "phrase": "rule", "room": "rule", "movable": "rule", "graspable": "rule",
        "upright_axis": "rule", "bimanual": "rule", "container": "rule", "mass_kg": "measured",
    }  # fmt: skip
    free = [k for k, o in meta["objects"].items() if o.get("free")]
    meta["randomization"] = {
        "movable_pool": free,
        "placements": {k: ["A"] for k in free},
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


def platform_surface(platform_geom: ET.Element) -> dict:
    """The platform's top as the scene file describes a surface (from the pack's `platform_col` box attributes)."""
    pos = [float(v) for v in platform_geom.get("pos").split()]
    size = [float(v) for v in platform_geom.get("size").split()]
    return {"xy": pos[:2], "top": pos[2] + size[2], "half_extent": size[:2], "flat_half_extent": size[:2], "yaw": 0.0,
            "approach_edges": ["-x", "+x", "-y", "+y"]}  # fmt: skip


def carton_mass(box_body: ET.Element, cap_kg: float) -> float:
    """The carton of a GSO instance weighs no more than the catalog's heaviest object (user 2026-11-07: no object
    over 0.3 kg): its `<inertial>` mass and inertia scaled down to `cap_kg` when the pack's BOX_MASS_KG is above it.
    Returns the carton's mass. The pack's own scenes (no --gso-seed) keep BOX_MASS_KG."""
    inertial = box_body.find("inertial")
    mass0 = float(inertial.get("mass"))
    mass = min(mass0, float(cap_kg))
    if mass < mass0:
        inertial.set("mass", f"{mass:.6g}")
        inertia = [float(x) * mass / mass0 for x in inertial.get("diaginertia").split()]
        inertial.set("diaginertia", " ".join(f"{x:.6g}" for x in inertia))
    return mass


def add_gso_objects(tree: ET.ElementTree, destination: Path, surface: dict, args) -> dict:
    """Draw the objects, place them and the carton, write them into the MJCF; returns the `gso` block of the scene
    file ({seed, pool, catalog, keys, placed}) with each object's catalog record under `records`."""
    catalog = G.load_catalog(args.catalog)
    for attempt in range(DRAW_ATTEMPTS):  # a set that does not fit on the platform is drawn again (deterministic)
        keys = G.sample_keys(
            catalog, args.gso_seed, n=args.objects, n_bimanual=args.bimanual, pool=args.pool, attempt=attempt
        )
        records = {k: catalog["objects"][k] for k in keys}
        boxes = {k: G.object_box(r)["half_extent"] for k, r in records.items()}
        boxes[BOX_KEY] = BOX_HALF
        try:
            placed = G.place_objects(boxes, surface, seed=args.gso_seed, reach_depth_m=args.reach_depth,
                                     constraints={BOX_KEY: CARTON_PLACEMENT})  # fmt: skip
            break
        except RuntimeError as e:
            print(f"[gso] draw {attempt} of seed {args.gso_seed} does not fit on the platform ({e}); drawing again")
    else:
        raise SystemExit(
            f"seed {args.gso_seed}: no draw of {args.objects} objects fits on the platform in {DRAW_ATTEMPTS} tries"
        )
    root = tree.getroot()
    for k in keys:  # pos: the scan's origin is its bottom centre (bbox_min z ~ 0); 1 mm up, scene_settle.py seats it
        rec = records[k]
        z = surface["top"] - rec["bbox_min_m"][2] + 0.001
        G.add_to_scene(root, destination, args.gso_models, k, rec, [*placed[k]["xy"], z], placed[k]["yaw"])
    box_body = root.find("worldbody").find(f"body[@name='{BOX_BODY}']")
    carton_kg = carton_mass(box_body, catalog["thresholds"]["mass_range_kg"][1])
    box_z = float(box_body.get("pos").split()[2])
    box_body.set("pos", " ".join(f"{v:.6g}" for v in (*placed[BOX_KEY]["xy"], box_z)))
    box_body.set("quat", " ".join(f"{v:.6g}" for v in G.yaw_quat(placed[BOX_KEY]["yaw"])))
    cam = ET.SubElement(root.find("worldbody"), "camera", name=PLATFORM_TOP_CAMERA)
    cam.set("pos", " ".join(f"{v:.6g}" for v in (*surface["xy"], surface["top"] + PLATFORM_TOP_HEIGHT_M)))
    cam.set("quat", "1 0 0 0")  # MuJoCo cameras look down their -z: identity looks straight down, image x = world x
    cam.set("fovy", "45")
    return {
        "seed": args.gso_seed, "pool": args.pool, "objects": args.objects, "bimanual": args.bimanual,
        "reach_depth_m": args.reach_depth, "carton_placement": CARTON_PLACEMENT, "catalog": str(args.catalog.relative_to(ROOT)) if args.catalog.is_relative_to(ROOT) else str(args.catalog),
        "catalog_sha256": G.sha256_file(args.catalog), "thresholds": catalog["thresholds"], "carton_mass_kg": carton_kg,
        "draw_attempt": attempt, "keys": keys, "placed": placed, "records": records,
    }  # fmt: skip


def main(argv=None) -> int:
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
    gso = parser.add_argument_group("Google Scanned Objects (the logistics_room_1030 scenes)")
    gso.add_argument("--gso-seed", type=int, default=None, help="draw and place the objects with this seed (the scene "
                     "instance's name carries it: logistics_room_1030_s<seed>); without it the scene is the carton alone")  # fmt: skip
    gso.add_argument("--objects", type=int, default=10, help="how many objects the instance gets")
    gso.add_argument(
        "--bimanual", type=int, default=4, help="how many of them are two-hand objects (the rest single-hand)"
    )
    gso.add_argument("--pool", choices=G.POOLS, default="train", help="train: the catalog's training split; eval_unseen: "
                     "the held-out tenth no training instance draws; all: both")  # fmt: skip
    gso.add_argument("--catalog", type=Path, default=GSO_ASSETS / "catalog.json", help="scripts/gso_catalog.py's file")
    gso.add_argument("--gso-models", type=Path, default=GSO_ASSETS / "models", help="the models/<name>/ directories")
    gso.add_argument("--reach-depth", type=float, default=REACH_DEPTH_M, help="m; an object is placed within this "
                     "depth of an approach edge when a spot exists (the task rows' `placement.depth_m` repeat it)")  # fmt: skip
    args = parser.parse_args(argv)
    if args.ego_camera and args.robot != "mars":
        raise SystemExit(f"--ego-camera picks one of mars's head cameras; robot {args.robot} has one ego camera")
    if args.gso_seed is not None and not args.catalog.is_file():
        raise SystemExit(f"no catalog {args.catalog}: bash scripts/install_gso.sh, then scripts/gso_catalog.py")
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
    if args.rsi or args.gso_seed is not None:  # scene_settle.py rewrites a free body's `pos="..." quat="..."` pair
        box_body = tree.getroot().find("worldbody").find(f"body[@name='{BOX_BODY}']")
        box_body.set("quat", box_body.get("quat") or "1 0 0 0")
    gso_block = (
        add_gso_objects(tree, destination, platform_surface(platform_geom), args) if args.gso_seed is not None else None
    )
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

    def free_pose(body_name: str) -> tuple[int, int, list, list]:
        body = object_id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        adr = int(model.jnt_qposadr[model.body_jntadr[body]])
        return body, adr, model.qpos0[adr : adr + 3].tolist(), model.qpos0[adr + 3 : adr + 7].tolist()

    box_body, qpos_address, box_pos, box_quat = free_pose(BOX_BODY)
    platform_body = object_id(mujoco.mjtObj.mjOBJ_BODY, "logistics_platform")
    platform_geom_id = object_id(mujoco.mjtObj.mjOBJ_GEOM, "platform_col")
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
    objects = {
        BOX_KEY: {
            "body": BOX_BODY,
            "body_id": box_body,
            "qpos_adr": qpos_address,
            "free": True,
            "surface": "A",
            "init_pos": box_pos,
            "init_quat": box_quat,
            "height": 0.33,
            "scale": 1.0,
            "bimanual": True,
            "container": True,
            "box": {"half_extent": BOX_HALF, "centre": [0.0, 0.0, 0.165]},
        }
    }
    phrases = {BOX_KEY: BOX_PHRASE}
    if gso_block:
        for k in gso_block["keys"]:
            rec = gso_block["records"][k]
            body_id, adr, pos, q = free_pose(f"{k}_object")
            objects[k] = G.object_record(rec, ROOM, gso_block["placed"][k], body_id, adr, pos, q)
            phrases[k] = rec["phrase"]
    built_by = (
        "scripts/prepare_logistics_scene.py"
        + (" --rsi" if args.rsi else "")
        + (f" --gso-seed {args.gso_seed}" if gso_block else "")
    )
    meta = {
        "scene_xml": str(runtime_xml),
        "target_object": BOX_KEY,
        "target_pool": list(objects),
        "object_phrase": phrases,
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
        "objects": objects,
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
    if gso_block:
        meta["gso"] = {k: v for k, v in gso_block.items() if k not in ("records", "placed")}
        meta["gso"]["in_reach_band"] = {k: v["in_reach_band"] for k, v in gso_block["placed"].items()}
    if args.rsi:
        rsi_fields(meta, model, data, args.body_radius, source_scene, built_by)
        write_env_only(tree, destination)
    meta_path = destination / "scene_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        json.dumps(
            {
                "scene": str(destination),
                "scene_xml": str(runtime_xml),
                "scene_meta": str(meta_path),
                "target_key": BOX_KEY,
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
                **(
                    {
                        "gso": {
                            "seed": args.gso_seed,
                            "objects": {k: gso_block["records"][k]["name"] for k in gso_block["keys"]},
                            "in_reach_band": meta["gso"]["in_reach_band"],
                        }
                    }
                    if gso_block
                    else {}
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
