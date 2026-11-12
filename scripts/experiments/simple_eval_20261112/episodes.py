"""SIMPLE's released evaluation episodes rebuilt as MuJoCo scenes for ScaleBFM (2026-11-12).

Each episode of `USC-PSI-Lab/psi-data` simple-eval stores its layout: the table slab, the target and the distractors with
their poses, the robot's start. `build` writes that layout as a scene directory the ScaleBFM runtime loads
(scene_locomani.xml, scene_meta.json), with SIMPLE's own robot (G1 with Dex3 hands), object and table contact settings and
solver options (third_party/SIMPLE/src/simple/engines/mujoco.py), and SIMPLE's head camera on the torso. `lifted` is SIMPLE's success check for the pick tasks.

    python episodes.py XMovePickTeleop        # lists the task's 30 episodes
"""

import glob
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

try:  # every episode is a scene of its own, ~180 MB compiled: keep them out of the runtime's model cache (an unwritable
    import locomani.paths as _paths  # cache makes runtime.load_model compile without saving)

    _paths.MJB_CACHE = "/dev/null/no_model_cache"
except ImportError:  # listing episodes needs no runtime
    pass

REPO = Path(__file__).resolve().parents[3]
SIMPLE = Path(os.environ.get("SIMPLE_ROOT", "/workspace/code/locomanipulation/third_party/SIMPLE"))
OUT = Path(os.environ.get("SIMPLE_EVAL_OUT", REPO / "outputs" / "simple_eval_20261112"))
ROBOT = SIMPLE / "data" / "robots" / "g1_sonic"
LEVELS = ("0", "1", "2")
OBJ_PULL = (
    -9.81 * 0.1
)  # N on every object besides its weight (SIMPLE's "pseudo gravity"; a 0.1 kg object weighs double)
LIFT = {"XMovePickTeleop": 0.09, "XMoveBendPickTeleop": 0.09, "BendPickMP": 0.09, "TabletopGraspMP": 0.018}  # m


AMO_PELVIS = (
    0.785  # m, engines/mujoco.py: the ground lies this far under the AMO robot's layout height (robots/g1_wholebody.py)
)
RES = {"graspnet1b": "graspnet", "objects": "objects"}  # an asset's res_id -> its folder under data/assets
HEAD_CAM = "head"
HEAD_CAM_POS = [0.05366004 + 0.0039635, 0.01752999, 0.4738702 - 0.044]  # in torso_link
HEAD_CAM_QUAT = [0.91496, 0.0, 0.40355, 0.0]  # the camera's x forward, z up frame in torso_link
HEAD_CAM_WH = (640, 360)
HEAD_CAM_FOVY = float(np.degrees(2 * np.arctan(np.tan(np.radians(110) / 2) * 360 / 640)))  # 110 deg across the width


def episodes(task: str) -> list[dict]:
    """The task's evaluation episodes, 10 per randomisation level: dict(task, level, index, cfg)."""
    out = []
    for lv in LEVELS:
        (f,) = glob.glob(str(OUT / "x" / task / "*" / f"dr-level-{lv}" / "meta" / "episodes.jsonl"))
        for line in open(f):
            e = json.loads(line)
            c = e["environment_config"]
            out.append(dict(task=task, level=lv, index=e["episode_index"], instruction=e["tasks"][0],
                            cfg=json.loads(c) if isinstance(c, str) else c))  # fmt: skip
    return out


def build(ep: dict, out: Path | None = None) -> dict:
    """Write the episode's scene; -> dict(dir, start xy, start yaw, target body, objects, table top)."""
    cfg = ep["cfg"]
    out = out or OUT / "scene" / f"{ep['task']}_L{ep['level']}_e{ep['index']}"
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
    mujoco.mju_mulQuat(q, np.array(HEAD_CAM_QUAT), np.array([0.5, 0.5, -0.5, -0.5]))
    spec.body("torso_link").add_camera(name=HEAD_CAM, pos=HEAD_CAM_POS, quat=q, fovy=HEAD_CAM_FOVY)
    w = spec.worldbody
    w.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1], friction=[1.0, 0.005, 0.0001],
               rgba=[0.75, 0.75, 0.72, 1])  # fmt: skip
    info = dict(dir=str(out), objects={}, tables={})
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
            for i, f in enumerate(pieces):
                spec.add_mesh(name=f"{label}_mesh_convex{i}", file=f)
                b.add_geom(name=f"{label}_convex_{i}", meshname=f"{label}_mesh_convex{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                           condim=4, mass=0.1 / len(pieces), friction=[0.8, 0.05, 0.005], solref=[0.005, 2],
                           rgba=[0.85, 0.3, 0.25, 1] if key == "target" else [0.35, 0.5, 0.8, 1])  # fmt: skip
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
    (out / "scene_locomani.xml").write_text(spec.to_xml())
    json.dump(dict(surfaces={}, objects={}, simple=info), open(out / "scene_meta.json", "w"))
    return info


def pull_objects(m: mujoco.MjModel, d: mujoco.MjData, info: dict) -> None:
    """SIMPLE's extra downward force on every object; call after each reset (a reset clears it)."""
    for o in info["objects"].values():
        d.xfrc_applied[m.body(o["body"]).id, 2] = OBJ_PULL


def lifted(task: str, z0: float, z: float) -> bool:
    """SIMPLE's pick success: the target's body origin rose by the task's lift."""
    return z - z0 >= LIFT[task]


if __name__ == "__main__":
    for e in episodes(sys.argv[1]):
        i = build(e)
        t = i["objects"]["target"]
        print(e["level"], e["index"], e["instruction"], "| start", np.round(i["start"]["xy"], 2), round(i["start"]["yaw"], 2),
              "| target", t["body"], np.round(t["pos"], 2), "| tables", {k: round(v["top"], 2) for k, v in i["tables"].items()},
              "| skipped", i.get("skipped"))  # fmt: skip
