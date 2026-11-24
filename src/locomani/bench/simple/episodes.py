"""SIMPLE's practice episodes as MuJoCo scenes for ScaleBFM, and the episode pools (TODO 49.2).

Moved from simple_eval_20261112/episodes.py (the scene builder), escalation_20261116/bed_simple.py (`load`, `ids_of`, `check_id`).
Each episode of a practice pool stores its layout: the table slab, the target and the distractors with their poses, the
robot's start. `build` writes that layout as a scene directory the ScaleBFM runtime loads (scene_locomani.xml,
scene_meta.json), with SIMPLE's own robot (G1 with Dex3 hands), object and table contact settings and solver options
(third_party/SIMPLE/src/simple/engines/mujoco.py), and SIMPLE's head camera on the torso.

Ids are "<Task>/<kind>/<n>": kind Q = SIMPLE's own draws (practice2, the loops' pool), kind P = the development set. A released
evaluation episode ("<Task>/L<level>/e<i>") is the exam: this module has no code that opens those files, and an id of a kind the
run's `SimpleConfig.kinds` does not list is refused (the old SIMPLE_LOOP=1 / EXAM_OPEN rules, as one rule).
"""

import glob
import json
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from locomani.bench.simple.config import SimpleConfig
from locomani.perception import head

RES = {"graspnet1b": "graspnet", "objects": "objects"}  # an asset's res_id -> its folder under data/assets
OBJ_PULL = (
    -9.81 * 0.1
)  # N on every object besides its weight (SIMPLE's "pseudo gravity"; a 0.1 kg object weighs double)
AMO_PELVIS = (
    0.785  # m, engines/mujoco.py: the ground lies this far under the AMO robot's layout height (robots/g1_wholebody.py)
)
PICKS = (
    "BendPickMP",
    "TabletopGraspMP",
    "XMovePickTeleop",
    "XMoveBendPickTeleop",
    "LocomotionPickBetweenTablesTeleop",
    "HandoverTeleop",
)
ARTICULATED = ("OpenOvenTeleop", "CloseDoorTeleop", "OpenFaucetTeleop", "OpenTrashCanTeleop")
NO_CACHE = "/dev/null/no_model_cache"


class EpisodeRefused(ValueError):
    """An id the bed does not run: a released (exam) episode, or a pool this run was not given."""


@contextmanager
def no_model_cache():
    """Compile scenes without the runtime's binary model cache: every episode is a scene of its own, ~180 MB compiled, and an
    unwritable cache makes `runtime.load_model` compile without saving. The runtime reads `locomani.paths.MJB_CACHE` when it
    loads, and has no argument for this, so the choice is made for the duration of the `with` and undone after it (the old
    bed rebound the name once at import and for good)."""
    import locomani.paths as paths

    old, paths.MJB_CACHE = paths.MJB_CACHE, NO_CACHE
    try:
        yield
    finally:
        paths.MJB_CACHE = old


def parse_id(episode_id: str) -> tuple[str, str, int]:
    parts = episode_id.split("/")
    if len(parts) != 3:
        raise EpisodeRefused(f"{episode_id}: an episode id is <Task>/<kind>/<n>")
    task, kind, n = parts
    if not n.isdigit():
        raise EpisodeRefused(
            f"{episode_id}: released evaluation episodes are the exam and never run here; practice ids are <Task>/Q/<n> or <Task>/P/<n>"
        )
    return task, kind, int(n)


def check_id(episode_id: str, cfg: SimpleConfig) -> None:
    """Refuse an id that is not of a practice pool this run may open (released ids never are)."""
    task, kind, _ = parse_id(episode_id)
    if kind not in ("P", "Q"):
        raise EpisodeRefused(f"{episode_id}: released evaluation episodes are the exam and never run here")
    if kind not in cfg.kinds:
        raise EpisodeRefused(f"{episode_id}: this run opens pool(s) {cfg.kinds!r} only (SimpleConfig.kinds)")
    if task not in PICKS + ARTICULATED:
        raise EpisodeRefused(f"{episode_id}: unknown task {task!r}")


@lru_cache(maxsize=64)
def _lines(path: str) -> list[str]:
    return open(path).read().splitlines()


def load(episode_id: str, cfg: SimpleConfig) -> dict:
    """The episode's dict(task, level, index, instruction, cfg). Only the asked line is parsed (a sandbox copy blanks the
    hidden ones: a blank line is an error, not an episode)."""
    check_id(episode_id, cfg)
    task, kind, n = parse_id(episode_id)
    f = Path(cfg.practice_dir if kind == "Q" else cfg.dev_dir) / f"{task}.jsonl"
    line = _lines(str(f))[n]
    if not line.strip():
        raise EpisodeRefused(f"{episode_id}: the line is blank in {f}")
    return dict(json.loads(line), task=task, level=kind, index=n)


def ids_of(spec: str) -> list[str]:
    """'Task/P/0-49,Task/Q/3' or a file of ids -> the ids."""
    if os.path.exists(spec):
        return [s.strip() for s in open(spec) if s.strip()]
    out = []
    for part in spec.split(","):
        task, kind, n = part.split("/")
        if "-" in n:
            lo, hi = n.split("-")
            out += [f"{task}/{kind}/{i}" for i in range(int(lo), int(hi) + 1)]
        else:
            out.append(part)
    return out


def build(ep: dict, out: Path, simple_root: str, visual_dir: str) -> dict:
    """Write the episode's scene into `out`; -> dict(dir, start xy, start yaw, target body, objects, table top). `simple_root`:
    SIMPLE's data (robot, assets); `visual_dir`: the objects' look meshes and textures (the local model only; Isaac renders the camera)."""
    cfg = ep["cfg"]
    SIMPLE, ROBOT = Path(simple_root), Path(simple_root) / "data" / "robots" / "g1_sonic"
    out.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec.from_file(str(ROBOT / "g1_29dof_with_hand.xml"))
    spec.meshdir = str(ROBOT / "meshes")
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 2
    for g in spec.geoms:  # SIMPLE: the hands' contacts are made stiffer
        if "hand" in (g.name or g.meshname):
            g.solimp[:3] = [0.9, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]
    q = np.zeros(4)  # SIMPLE's head camera (engines/mujoco.py, mount eye_in_head): on the torso, pitched down 47.6 deg
    mujoco.mju_mulQuat(q, np.array(head.HEAD_CAM_QUAT), np.array([0.5, 0.5, -0.5, -0.5]))
    spec.body("torso_link").add_camera(name=head.HEAD_CAM, pos=head.HEAD_CAM_POS, quat=q, fovy=head.HEAD_CAM_FOVY)
    w = spec.worldbody
    w.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1], friction=[1.0, 0.005, 0.0001],
               rgba=[0.75, 0.75, 0.72, 1])  # fmt: skip
    info = dict(dir=str(out), objects={}, tables={})
    looks = set()
    lift = 0.0
    if cfg["robot_cfg"]["uid"] != "g1_sonic":  # SIMPLE's AMO tasks put the table top at z 0 and the ground below it
        t, r = cfg["layout"]["actors"]["table"], cfg["dr_state_dict"]["spatial"][cfg["robot_cfg"]["uid"]]
        lift = AMO_PELVIS - r["position"][2] - (t["pose"]["position"][2] + 0.5 * t["size"][2])
    info["lift"] = lift
    for key, a in cfg["layout"]["actors"].items():
        pos, quat = list(a["pose"]["position"]), a["pose"]["quaternion"]
        pos[2] += lift
        if a.get("uid") == "box":  # a table: a slab
            b = w.add_body(name=key, pos=pos, quat=quat)
            b.add_geom(name=f"{key}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=0.5 * np.array(a["size"]), condim=6,
                       friction=[2, 0.04, 0.0005], priority=10, rgba=[0.72, 0.6, 0.42, 1])  # fmt: skip
            info["tables"][key] = dict(top=pos[2] + 0.5 * a["size"][2], pos=pos, size=a["size"], quat=quat)
        elif "asset" in a and a["asset"].get("res_id") in RES:  # an object, or the basket things are placed in
            label, uid = a["asset"]["label"], int(a["asset"]["uid"])
            models = SIMPLE / "data/assets" / RES[a["asset"]["res_id"]] / "models" / f"{uid:03d}" / "coacd_0.05"
            pieces = sorted(glob.glob(str(models / "*.obj")))
            assert pieces, f"no collision pieces for object {uid}"
            b = w.add_body(name=label, pos=pos, quat=quat)
            look = Path(visual_dir) / f"{RES[a['asset']['res_id']]}_{uid:03d}"  # visuals.py: what the camera sees
            if look.with_suffix(".obj").exists() and f"{label}_look" not in looks:
                looks.add(f"{label}_look")
                spec.add_mesh(name=f"{label}_look", file=str(look.with_suffix(".obj")))
                spec.add_texture(
                    name=f"{label}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(look.with_suffix(".png"))
                )
                mat = spec.add_material(name=f"{label}_mat")
                mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{label}_tex"
            seen = f"{label}_look" in looks
            if seen:
                b.add_geom(name=f"{label}_look", meshname=f"{label}_look", type=mujoco.mjtGeom.mjGEOM_MESH, density=0,
                           contype=0, conaffinity=0, group=1, material=f"{label}_mat")  # fmt: skip
            for i, f in enumerate(pieces):
                spec.add_mesh(name=f"{label}_mesh_convex{i}", file=f)
                b.add_geom(name=f"{label}_convex_{i}", meshname=f"{label}_mesh_convex{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                           condim=4, mass=0.1 / len(pieces), friction=[0.8, 0.05, 0.005], solref=[0.005, 2],
                           group=3 if seen else 0, rgba=[0.55, 0.55, 0.55, 1])  # fmt: skip
            b.add_freejoint(name=f"{label}_joint")
            info["objects"][key] = dict(body=label, uid=uid, pos=pos, quat=quat)
        elif key == cfg["robot_cfg"]["uid"] or "robot" in key:
            info["start"] = dict(xy=pos[:2], quat=quat)
        else:
            info.setdefault("skipped", []).append(key)
    if "start" not in info:
        s = cfg["dr_state_dict"]["spatial"][cfg["robot_cfg"]["uid"]]
        info["start"] = dict(xy=s["position"][:2], quat=s["quaternion"])
    q = info["start"]["quat"]
    info["start"]["yaw"] = float(2 * np.arctan2(q[3], q[0]))
    info["target"] = info["objects"]["target"]["body"]
    spec.compile()
    for name, text in (("scene_locomani.xml", spec.to_xml()), ("scene_meta.json", json.dumps(dict(surfaces={}, objects={}, simple=info)))):  # fmt: skip
        tmp = (
            out / f"{name}.{os.getpid()}.tmp"
        )  # two runs of one task build the same scene: never leave a half-written file
        tmp.write_text(text)
        os.replace(tmp, out / name)
    return info


def pull_objects(m: mujoco.MjModel, d: mujoco.MjData, info: dict) -> None:
    """SIMPLE's extra downward force on every object; call after each reset (a reset clears it)."""
    for o in info["objects"].values():
        d.xfrc_applied[m.body(o["body"]).id, 2] = OBJ_PULL
