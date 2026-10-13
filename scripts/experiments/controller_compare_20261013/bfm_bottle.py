"""Bottle L1, ScaleBFM side (TODO 25.2, plan §10): two programs per SONIC production round, written as bottle_states.py
round files so bottle_measure.py measures both controllers with the same code.

* `walk` (B1 B2 B8): from the round's spawn (xy, yaw), the stock bottle scene with the bottle where it stood, walk on
  a trapezoid pelvis reference (SONIC's walk speed, acc 0.6 m/s^2) along the guard walk's own straight direction
  (edge_guard.straight_walk) and stop where the table-top band's frontmost robot point sits `--margin` (SONIC's
  guard margin, 3 cm) before the edge -- the band reach measured standing at the stand heading before the walk. The
  yaw reference turns from the spawn yaw to psi over the first 1.5 s. Hands at the side. `brake` = the reference's
  deceleration start.
* `replay` (B4 B5 B6, and the mode sweep): ScaleBFM tracks what SONIC's body actually did. Reset at SONIC's pelvis
  pose 0.5 s before `ready`, the bottle at its pose then; 2 s blend to SONIC's keypoints; then SONIC's measured
  keypoint trajectory (FK of its recorded state) to the end of its record (handback, the 0.35 m carry back-off and the
  terminal hold included), the right jaw commanded closed at SONIC's `close`. `--mode` picks which keypoints are
  targets: 1 wrists, 2 pelvis + wrists, 6 pelvis + shoulders, elbows, wrists, 7 all 14. With `--carry`, everything
  after SONIC's handback is ScaleBFM's own carry (carry_profile: the targets of SONIC's carry plans), every keypoint
  held rigidly in the pelvis frame; only rounds SONIC completed. Hand law = the SONIC line's Dex1 law (kp 1.5, kd 0.1, +-20 N, closed -0.02).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \\
        scripts/experiments/controller_compare_20261013/bfm_bottle.py --program replay --mode 2 \\
        --sonic outputs/controller_compare_20261013/bottle/sonic/states --out outputs/controller_compare_20261013/bottle/bfm_replay_m2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "experiments" / "wbc_compare_20261012")]

from bfm_brake import Profile  # noqa: E402
from bottle_states import SCENE, Fk, save  # noqa: E402
from w0_admission import seat  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.primitives.foot_watch import FootWatch  # noqa: E402
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402
from locomani.tasks.bottle.edge_guard import straight_walk  # noqa: E402

JAW_OPEN, JAW_CLOSED = 0.0245, -0.02  # agent_server._hand_poses (the SONIC line)
HAND_KP, HAND_KD, HAND_FORCE = 1.5, 0.1, 20.0  # grasp_cache.py's law, as bfm_object_scene.finger_config
MODE_BODIES = {
    1: ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    2: ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"),
    6: (
        "pelvis",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ),
}
SETTLE = 2.0
YAW_TURN_S = 1.5
CARRY_SPEED = 0.25  # m/s, the bottle line's slow walk
CARRY_YAW_RATE = 0.5  # rad/s on the spot


def slerp(q0, q1, u):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    c = float(q0 @ q1)
    if c < 0:
        q1, c = -q1, -c
    if c > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    a = np.arccos(c)
    return (np.sin((1 - u) * a) * q0 + np.sin(u * a) * q1) / np.sin(a)


class Recorder:
    """Per tick: the bottle_states.py arrays from the ScaleBFM sim."""

    def __init__(self, sim):
        self.sim = sim
        self.mj = np.array([sim.m.jnt_qposadr[sim.m.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.rows = {k: [] for k in ("t", "pelvis", "joints", "bottle", "cmd_pelvis", "cmd_wrist", "jaw")}

    def add(self, t, targets):
        d = self.sim.d
        self.rows["t"].append(t)
        self.rows["pelvis"].append(d.qpos[:7].copy())
        self.rows["joints"].append(d.qpos[self.mj].copy())
        self.rows["bottle"].append(d.qpos[self.sim.crate_q : self.sim.crate_q + 7].copy())
        pel = targets.get("pelvis")
        self.rows["cmd_pelvis"].append(np.full(7, np.nan) if pel is None else np.concatenate([pel[0][0], pel[1][0]]))
        w = np.full((2, 7), np.nan)
        for k, s in enumerate(("left", "right")):
            x = targets.get(f"{s}_wrist_yaw_link")
            if x is not None:
                w[k] = np.concatenate([x[0][0], x[1][0]])
        self.rows["cmd_wrist"].append(w)
        self.rows["jaw"].append(float(d.qpos[self.sim.gq[2]]))

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.rows.items()}


def place_bottle(sim, pose):
    sim.d.qpos[sim.crate_q : sim.crate_q + 7] = pose
    sim.d.qvel[sim.m.jnt_dofadr[sim.m.body_jntadr[sim.crate_b]] :][:6] = 0
    mujoco.mj_forward(sim.m, sim.d)


def walk_round(pol, sim, z, f, a):
    edge_axis, line = np.asarray(f["edge"]["axis"], float), float(f["edge"]["line"])
    band = f["band"]
    psi, yaw0 = float(f["psi"]), float(f["spawn_yaw"])
    p0 = np.asarray(f["robot_xy"], float)
    watch = FootWatch(model=sim.m)
    # band reach at the stand heading, standing (FK only)
    sim.reset(p0, psi, 0.0)
    seat(sim)
    place_bottle(sim, z["bottle"][0])
    rec = Recorder(sim)
    st = lambda: dict(pelvis=dict(pos=sim.d.qpos[:3], quat=sim.d.qpos[3:7]), joints=sim.d.qpos[rec.mj])
    reach = watch.frontmost_band(st(), np.zeros(2), edge_axis, *band)[0] - float(edge_axis @ sim.d.qpos[:2])
    d = straight_walk(p0, f["stand_xy"], edge_axis)
    dist = (line - a.margin - reach - float(edge_axis @ p0)) / float(edge_axis @ d)
    prof = Profile(a.walk_speed, R.WALK_ACC, dist)
    sim.reset(p0, yaw0, 0.0)
    seat(sim)
    place_bottle(sim, z["bottle"][0])
    refs = R.Refs(pol, sim)
    pol.reset(sim.state())

    def ref(t):
        s = prof.s(t - R.SETTLE_S)
        p = p0 + d * s
        u = R.smoothstep((t - R.SETTLE_S) / YAW_TURN_S)
        yaw = yaw0 + u * R.wrap(psi - yaw0)
        pel = np.array([p[0], p[1], R.PELVIS_Z])
        out = {"pelvis": (pel, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))}
        out.update(refs.hands_at_side(pel, yaw))
        return out

    t_end = R.SETTLE_S + prof.t_end + a.settle
    k, fell = 0, None
    while k * R.DT < t_end:
        t = k * R.DT
        mujoco.mj_kinematics(sim.m, sim.d)
        fut = refs.futures(ref, t)
        rec.add(t, fut)
        if sim.fallen():
            fell = t
            break
        sim.step(pol(sim.state(), fut))
        k += 1
    facts = dict(
        f,
        controller="scalebfm",
        program=f"walk_{round(100 * a.margin)}cm_v{a.walk_speed:.2f}{a.tag}",
        mode=int(pol.mode.item()),
        events=dict(walk=R.SETTLE_S, brake=R.SETTLE_S + prof.t_dec),
        walk_target=dict(distance=dist, direction=d.tolist(), band_reach=reach, margin=a.margin, speed=a.walk_speed),
        ok=fell is None,
        why="" if fell is None else f"fell at {fell:.2f} s",
    )
    return rec.arrays(), facts


def carry_profile(kind: str, yaw_from: float):
    """ScaleBFM's carry after the grasp, the targets of configs/plans/bottle_carry_<kind>.json: (pelvis offset along
    the start heading (m, + forward), yaw (rad)) as a function of seconds since the carry started, and its length.
    back1 / back2: hold 1 s, walk backwards 1 / 2 m; turn90 / turn180slow: hold 1 s, turn on the spot to the plan's
    absolute yaw (0 / +pi/2) at CARRY_YAW_RATE, then 1.5 m forward along the new heading; all walks are trapezoids at
    CARRY_SPEED. A hold of 3 s ends every carry."""
    if kind in ("back1", "back2"):
        prof = Profile(CARRY_SPEED, R.WALK_ACC, 1.0 if kind == "back1" else 2.0)
        return (lambda s: (-prof.s(s - 1.0), 0.0, yaw_from)), 1.0 + prof.t_end + 3.0
    turn = R.wrap({"turn90slow": 0.0, "turn180slow": np.pi / 2}[kind] - yaw_from)
    t_turn = abs(turn) / CARRY_YAW_RATE
    prof = Profile(CARRY_SPEED, R.WALK_ACC, 1.5)

    def at(s):
        yaw = yaw_from + turn * R.smoothstep((s - 1.0) / t_turn)
        return prof.s(s - 1.0 - t_turn), 1.0, yaw  # the walk runs along the NEW heading (flag 1.0)

    return at, 1.0 + t_turn + prof.t_end + 3.0


def replay_round(pol, sim, z, f, a, fk):
    t_s = z["t"]
    t0 = f["events"]["ready"] - 0.5
    # to the end of SONIC's record (handback, the carry back-off and the terminal hold included); a carry replaces
    # everything after SONIC's handback with ScaleBFM's own carry
    t1 = t_s[-1] if a.carry is None else f["events"]["handback"]
    i0 = int(np.searchsorted(t_s, t0))
    bodies = MODE_BODIES.get(a.mode) or tuple(pol.sel)
    # SONIC's measured keypoints: FK of its recorded state, every tick of the replay span
    span = np.arange(i0, int(np.searchsorted(t_s, t1)) + 1)
    kp = {b: np.empty((len(span), 7)) for b in bodies}
    for n, i in enumerate(span):
        fk.put(z["pelvis"][i], z["joints"][i])
        for b in bodies:
            kp[b][n] = fk.pose(b)
    ts = t_s[span] - t0  # replay clock: 0 at the replay start, the blend runs before it

    def sonic_at(b, tt):
        j = int(np.clip(np.searchsorted(ts, tt), 0, len(ts) - 1))
        return kp[b][j, :3], kp[b][j, 3:]

    pel0 = z["pelvis"][i0]
    sim.reset(pel0[:2], R.yaw_of(pel0[3:]), 0.0)
    seat(sim)
    place_bottle(sim, z["bottle"][i0])
    refs = R.Refs(pol, sim)
    pol.reset(sim.state())
    yaw0 = R.yaw_of(pel0[3:])
    start = {"pelvis": (np.array([pel0[0], pel0[1], R.PELVIS_Z]), np.array([np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)]))}
    start.update(refs.hands_at_side(start["pelvis"][0], yaw0))
    close_t = f["events"]["close"] - t0

    def ref(t):
        tt = t - SETTLE
        if tt >= 0:
            return {b: sonic_at(b, tt) for b in bodies}
        u = R.smoothstep(t / SETTLE)
        out = {}
        for b in bodies:
            p1, q1 = sonic_at(b, 0.0)
            p0_, q0_ = start.get(b, (p1, q1))
            out[b] = (p0_ + u * (p1 - p0_), slerp(q0_, q1, u))
        return out

    t_end = SETTLE + (t1 - t0)
    if a.carry is not None:
        # from the handback on: every keypoint rigidly attached to the pelvis as it was then, the pelvis on the carry
        pel_end = sonic_at("pelvis", t1 - t0)
        yaw_c = R.yaw_of(pel_end[1])
        Rc = R.rot_z(yaw_c)
        rel = {
            b: (Rc.T @ (sonic_at(b, t1 - t0)[0] - pel_end[0]), Rc.T @ R.mat(sonic_at(b, t1 - t0)[1])) for b in bodies
        }
        carry_at, carry_s = carry_profile(a.carry, yaw_c)
        replay_ref = ref

        t_carry = t_end  # the carry starts here (t_end grows below; the closure must not read it)

        def ref(t):
            s_ = t - t_carry
            if s_ < 0:
                return replay_ref(t)
            dist, along_new, yaw = carry_at(s_)
            if along_new:  # turn first on the spot, then walk along the new heading
                xy = pel_end[0][:2] + dist * np.array([np.cos(yaw), np.sin(yaw)])
            else:
                xy = pel_end[0][:2] + dist * np.array([np.cos(yaw_c), np.sin(yaw_c)])
            Rn = R.rot_z(yaw)
            P = np.array([xy[0], xy[1], pel_end[0][2]])
            return {b: (P + Rn @ rel[b][0], R.quat_wxyz(Rn @ rel[b][1])) for b in bodies}

        t_end += carry_s
    rec = Recorder(sim)
    k, fell = 0, None
    while k * R.DT < t_end:
        t = k * R.DT
        if t - SETTLE >= close_t:
            sim.grip_target[2:] = JAW_CLOSED
        mujoco.mj_kinematics(sim.m, sim.d)
        fut = refs.futures(ref, t)
        rec.add(t, fut)
        if sim.fallen():
            fell = t
            break
        sim.step(pol(sim.state(), fut))
        k += 1
    shift = SETTLE - t0
    events = {k_: v + shift for k_, v in f["events"].items() if k_ not in ("walk", "brake")}
    if a.carry is not None:
        events = {k_: v for k_, v in events.items() if v <= SETTLE + (t1 - t0) + 1e-6}
        events["carry"] = SETTLE + (t1 - t0)
    facts = dict(
        f,
        controller="scalebfm",
        program=f"replay_m{a.mode}" if a.carry is None else f"carry_{a.carry}_m{a.mode}",
        mode=a.mode,
        events=events,
        replay=dict(sonic_t0=t0, sonic_t1=t1, bodies=list(bodies)),
        sonic_ok=f["ok"],
        sonic_why=f["why"],
        ok=fell is None,
        why="" if fell is None else f"fell at {fell:.2f} s",
    )
    return rec.arrays(), facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", choices=("walk", "replay"), required=True)
    ap.add_argument("--mode", type=int, default=2)
    ap.add_argument(
        "--carry",
        choices=("back1", "back2", "turn90slow", "turn180slow"),
        default=None,
        help="replay: after SONIC's handback, ScaleBFM's own carry (configs/plans/bottle_carry_<carry>.json targets)",
    )
    ap.add_argument("--sonic", type=Path, required=True, help="SONIC round files (bottle_states.py)")
    ap.add_argument("--walk-speed", type=float, default=0.25, help="walk: m/s (bottle_simple_200 guard walk_speed)")
    ap.add_argument("--margin", type=float, default=0.03, help="walk: band clearance kept before the edge (m)")
    ap.add_argument("--settle", type=float, default=3.0, help="walk: simulated seconds after the reference stops")
    ap.add_argument("--tag", default="", help="appended to the walk program's name (e.g. _yaw40 for those rounds)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = ScaleBfmBackendConfig(
        scene_dir=str(SCENE), object_name="water_bottle", grip_kp=HAND_KP, grip_kd=HAND_KD, grip_clip=HAND_FORCE
    )
    pol = R.Policy(device=a.device, backend=cfg)
    pol.mode = torch.tensor([a.mode], dtype=torch.long, device=a.device)
    sim = R.Sim(pol.meta, backend=cfg)
    fk = Fk()
    for npz in sorted(a.sonic.glob("*.npz")):
        z = dict(np.load(npz))
        f = json.loads(npz.with_suffix(".json").read_text())
        if (a.out / "STOP").exists():
            break
        if a.program == "replay" and not {"ready", "close"} <= set(f["events"]):
            print(npz.stem, "skipped: SONIC never reached the reach")
            continue
        if a.carry is not None and not f["ok"]:
            print(npz.stem, "skipped: a carry starts from a grasp SONIC completed")
            continue
        w0 = time.time()
        arrays, facts = walk_round(pol, sim, z, f, a) if a.program == "walk" else replay_round(pol, sim, z, f, a, fk)
        facts["wall_s"] = round(time.time() - w0, 1)
        label = npz.stem.replace("sonic", f"bfm_{facts['program']}")
        save(a.out, label, arrays, facts)
        print(label, "ok" if facts["ok"] else facts["why"], f"{facts['wall_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
