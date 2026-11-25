"""TODO 27.2.1 (plan docs/handoff/scene-rsi-plan-20261016.md D5): live check of the ScaleBFM motor layer
(src/locomani/backends/scalebfm/motor.py) -- what each primitive delivers, before any skill is built on it.

`loco`: walk_mars (flat floor), each case from a fresh reset at the origin: straight walk, walk to the side with turns,
in-place turns, sidestep, step back, posture (pelvis height, torso lean), reaches in front at three heights.
`pick`: the ADT apartment (mars), the large coffee canister on the utility cart: reset at a stance in front of it (or,
--walk, at the scene's robot start and walk there), reach the right jaw to a pre-grasp 12 cm short, to the canister's
centre, grip, lift 10 cm, hold 2 s.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=1 $PY scripts/rsi/motor_check.py loco
    CUDA_VISIBLE_DEVICES=1 $PY scripts/rsi/motor_check.py pick [--walk]
`--controller amo` runs the same cases on AMO (backends/amo, TODO 28); results go to motor_check/amo/; `--controller
gr00t_wbc` / `gr00t_wbc_h066_p012` on NVIDIA's gear WBC (backends/gr00t_wbc, TODO 34), into motor_check/<controller>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src")]

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.backends.scalebfm.motor import Motor, MotorConfig  # noqa: E402

OUT = REPO / "outputs" / "scene_rsi_20261016" / "motor_check"
WALK = REPO / "outputs" / "scene" / "walk_mars"
ADT = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars"


CONTROLLER = "scalebfm"


def make(policy, scene: Path, track=()):
    cfg = ScaleBfmBackendConfig(scene_dir=str(scene), object_name="")
    if CONTROLLER.startswith("gr00t_wbc"):  # TODO 34
        from locomani.backends.gr00t_wbc.motor import VARIANTS, Gr00tMotor
        from locomani.backends.gr00t_wbc.runtime import Gr00tSim

        return Gr00tMotor(Gr00tSim(cfg), policy, None, model_xml=str(scene / "scene_locomani.xml"), track=track,
                          gr00t=VARIANTS[CONTROLLER])  # fmt: skip
    if CONTROLLER == "amo":
        from locomani.backends.amo.motor import AmoMotor
        from locomani.backends.amo.runtime import AmoSim

        return AmoMotor(AmoSim(cfg), policy, None, model_xml=str(scene / "scene_locomani.xml"), track=track)
    sim = R.Sim(policy.meta, backend=cfg)
    return Motor(sim, policy, MotorConfig(), model_xml=str(scene / "scene_locomani.xml"), track=track)


def summary(out) -> dict:
    return dict(ok=out.ok, why=out.why, seconds=round(out.t1 - out.t0, 2), **out.info)


def loco(policy) -> dict:
    res = {}

    def case(name, steps):
        m = make(policy, WALK)
        m.reset((0.0, 0.0), 0.0)
        rows = [summary(m.hold(1.0))]
        for fn in steps:
            out = fn(m)
            rows.append(summary(out))
            if not out.ok and out.why.startswith("fell"):
                break
        p, q = m.pelvis()
        res[name] = dict(
            steps=rows, end_xy=[round(float(v), 3) for v in p[:2]], end_yaw_deg=round(float(np.degrees(R.yaw_of(q))), 1)
        )
        print(name, json.dumps(res[name]), flush=True)

    case("walk_2m", [lambda m: m.walk_to((2.0, 0.0), 0.0)])
    case("walk_side_turn", [lambda m: m.walk_to((1.0, 1.5), np.pi / 2)])
    case("turn_180", [lambda m: m.turn(np.pi)])
    case("turn_90_right", [lambda m: m.turn(-np.pi / 2)])
    case("sidestep_left_0p3", [lambda m: m.sidestep(left=0.3)])
    case("step_back_0p3", [lambda m: m.sidestep(forward=-0.3)])
    case("posture_low", [lambda m: m.posture(0.62), lambda m: m.posture(0.52), lambda m: m.posture(0.74)])
    case("posture_lean", [lambda m: m.posture(torso_pitch=0.35), lambda m: m.posture(torso_pitch=0.0)])

    def reach_rel(m, x, y, z, side="right"):
        p, q = m.pelvis()
        yaw = R.yaw_of(q)
        tgt = np.array([p[0], p[1], 0.0]) + R.rot_z(yaw) @ np.array([x, y, z])
        return m.reach(side, tgt, R.rot_z(yaw))

    case("reach_mid", [lambda m: reach_rel(m, 0.38, -0.18, 0.90)])
    case("reach_high", [lambda m: reach_rel(m, 0.35, -0.18, 1.15)])
    case("reach_low", [lambda m: reach_rel(m, 0.38, -0.18, 0.55)])
    case("reach_floorish", [lambda m: reach_rel(m, 0.38, -0.18, 0.35)])
    return res


def pick(policy, walk: bool) -> dict:
    meta = json.loads((ADT / "scene_meta.json").read_text())
    obj = meta["objects"]["CoffeeCanisterLarge"]
    surf = meta["surfaces"][obj["surface"]]
    c = np.asarray(obj["init_pos"], float)
    Rs = R.rot_z(float(surf["yaw"]))[:2, :2]
    local = Rs.T @ (c[:2] - np.asarray(surf["xy"], float))
    half = np.asarray(surf["half_extent"], float)
    normals = {"-x": (-1, 0), "+x": (1, 0), "-y": (0, -1), "+y": (0, 1)}
    # the approachable edge nearest the canister (the surface's own frame): its depth from that edge
    depth = {e: float(half[0 if "x" in e else 1] - np.dot(normals[e], local)) for e in surf["approach_edges"]}
    edge = min(depth, key=depth.get)
    n = Rs @ np.array(normals[edge], float)  # outward normal of the approach edge
    face = float(np.arctan2(-n[1], -n[0]))  # the robot faces into the surface
    Rf = R.rot_z(face)
    stance = c[:2] - (Rf @ np.array([0.40, -0.12, 0.0]))[:2]  # canister 40 cm ahead, 12 cm right of the pelvis
    m = make(policy, ADT, track=(obj["body"],))
    start = meta["robot_start"]
    if walk:
        m.reset(start["xy"], np.radians(start["yaw_deg"]))
    else:
        m.reset(stance, face)
    # the canister fits the jaw (9.5 cm) only across its 8.2 cm side (local x): turn it so that side is lateral
    m.place(obj["body"], c, R.quat_wxyz(R.rot_z(face + np.pi / 2)))
    steps = [("hold", m.hold(1.0))]
    if walk:
        steps.append(("walk_to", m.walk_to(stance, face)))
    pre = c - Rf @ np.array([0.12, 0.0, 0.0])
    lift = c + np.array([0.0, 0.0, 0.10])
    for name, fn in (
        ("pregrasp", lambda: m.reach("right", pre, Rf)),
        ("grasp", lambda: m.reach("right", c + Rf @ np.array([0.01, 0, 0]), Rf, speed=0.08)),
        ("grip", lambda: m.grip("right")),
        ("lift", lambda: m.reach("right", lift, Rf, speed=0.08)),
        ("hold", lambda: m.hold(2.0)),
    ):
        out = fn()
        steps.append((name, out))
        if not out.ok and (out.why.startswith("fell") or name == "grip"):
            break
    o = m.sim.d.xpos[m.sim.m.body(obj["body"]).id]
    res = dict(
        steps=[dict(name=k, **summary(v)) for k, v in steps],
        rise_cm=round(100 * float(o[2] - c[2]), 1),
        held=sorted(m.sim.m.body(b).name for b in m.between_fingers("right")),
        stance=[round(float(v), 3) for v in stance],
        face_deg=round(float(np.degrees(face)), 1),
    )
    np.savez_compressed(OUT / f"pick{'_walk' if walk else ''}.npz", **m.arrays())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("loco", "pick"))
    ap.add_argument("--walk", action="store_true", help="pick: start at the scene's robot start and walk to the stance")
    ap.add_argument("--controller", choices=("scalebfm", "amo", "gr00t_wbc", "gr00t_wbc_h066_p012"), default="scalebfm")
    a = ap.parse_args(argv)
    global CONTROLLER, OUT
    CONTROLLER = a.controller
    if a.controller == "amo":
        from locomani.backends.amo.runtime import AmoPolicy

        OUT = OUT / "amo"
        policy = AmoPolicy("cuda")
    elif a.controller.startswith("gr00t_wbc"):
        from locomani.backends.gr00t_wbc.runtime import Gr00tPolicy

        OUT = OUT / a.controller
        policy = Gr00tPolicy()
    else:
        policy = R.Policy("cuda", ScaleBfmBackendConfig())
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = loco(policy) if a.what == "loco" else pick(policy, a.walk)
    name = a.what + ("_walk" if a.walk else "")
    (OUT / f"{name}.json").write_text(json.dumps(dict(result=res, wall_s=round(time.time() - t0, 1)), indent=1))
    print(json.dumps(res, indent=1) if a.what == "pick" else "", f"\nwrote {OUT / name}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
