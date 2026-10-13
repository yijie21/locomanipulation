"""Warehouse L1, ScaleBFM side (TODO 25.3, plan §15): programs per SONIC round, written as warehouse_states.py round
files so warehouse_measure.py measures both controllers with the same code. The mars warehouse scene, the box where
SONIC's round had it; the jaws on the SONIC line's Dex1 law (kp 1.5, kd 0.1, +-20 N) with the warehouse expert's
closed target (-0.06).

* `walk` (W1): from the round's start pose, a trapezoid pelvis reference (SONIC's walk speed 0.25 m/s, acc 0.6 m/s^2)
  on the straight line to the stance `--standoff` metres before the edge line (pelvis, along the walk-in axis),
  centred on the box across the edge, facing square to the box; the yaw reference turns over the first 1.5 s. Hands
  at the side. `--standoff` defaults to the median stance of SONIC's completed rounds in `--sonic`. `brake` = the
  reference's deceleration start. Open loop ScaleBFM's pelvis trails the moving reference by ~10 cm and at the stop
  either closes to ~3 cm or stays ~10 cm short (no last step); `--walk-servo` then adds the measured pelvis error,
  integrated (Ki 1/s, capped at 15 cm), to the pelvis target once the reference has stopped -- the re-walk SONIC's
  expert does on a short stance.
* `replay` (W2-W4, W7, W8, and the mode sweep): ScaleBFM tracks what SONIC's body actually did. Reset at SONIC's
  pelvis pose at the replay start (production: 1 s into `settle`, at most 0.5 s before `handoff`; plan rounds: 1 s
  after the walk-in ends), the box at its pose then; 2 s blend to SONIC's keypoints; then SONIC's measured keypoint
  trajectory (FK of its recorded state) to the end of its record; the jaws follow SONIC's jaw commands tick by tick.
  `--mode` picks the keypoints: 1 wrists, 2 pelvis + wrists, 6 pelvis + shoulders, elbows, wrists, 7 all 14.
  `--pelvis-shift` (mode 2; W1's stand-off sweep) stands the robot that much farther from the platform (m along the
  walk-in axis, - = closer): the reset and the pelvis keypoint move, the hand targets stay where SONIC's hands were.
  `--box-mass` sets the box's mass (inertia scaled with it) -- W4's 2.4 kg tier. `--servo` closes the loop on the
  hands the way SONIC's expert closes it on its tool points (visual.tracking_gain), integral, toward SONIC's measured
  hands (Ki 2/s, each axis capped at 8 cm -- the 2026-09-08 spike's setting, with which ScaleBFM's mode-2 crate carry
  went 10/10): `wrist` on the wrist origin; `pad` on the jaw centre 15 cm ahead of it (runtime.OFF) plus the wrist's
  orientation (Ki 2/s, capped at 0.5 rad) -- ScaleBFM's wrist attitude is 10-30 deg off at the warehouse grasp, which
  puts the jaws 2-9 cm off with the wrist on target. The arm's elbow / shoulder targets move with their wrist's offset.
  `--servo-kp` adds a proportional term on the same errors (SONIC's expert corrects proportionally, tracking_gain 1.0).
* `--carry` (W5 W6; with `replay`, rounds SONIC completed): after SONIC's record ends (the loaded hold), ScaleBFM's
  own carry, every keypoint held rigidly in the pelvis frame: back1 = 1 m backwards, heading kept; turn180 = turn on
  the spot by pi at 0.5 rad/s, then 1 m forward. Walks are 0.25 m/s trapezoids; 1 s hold before, 3 s after.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \\
        scripts/experiments/controller_compare_20261013/bfm_warehouse.py --program replay --mode 6 \\
        --sonic outputs/controller_compare_20261013/warehouse/sonic/states --out outputs/controller_compare_20261013/warehouse/bfm_replay_m6
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

from bfm_bottle import HAND_FORCE, HAND_KD, HAND_KP, MODE_BODIES, SETTLE, YAW_TURN_S, slerp  # noqa: E402
from bfm_brake import Profile  # noqa: E402
from bottle_states import save  # noqa: E402
from w0_admission import seat  # noqa: E402
from warehouse_states import BOX, JAW_OPEN, SCENE, WRISTS, Fk, square_yaw  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

CARRY_SPEED = 0.25
CARRY_YAW_RATE = 0.5
SERVO_KI, SERVO_CAP = 2.0, 0.08  # 1/s, m per axis
SERVO_KI_ROT, SERVO_CAP_ROT = 2.0, 0.5  # 1/s, rad
WALK_KI, WALK_CAP = 1.0, 0.15  # 1/s, m per axis
ARM_BODIES = {
    side: (f"{side}_shoulder_roll_link", f"{side}_elbow_link", f"{side}_wrist_yaw_link") for side in ("left", "right")
}


class Recorder:
    """Per tick: the warehouse_states.py arrays from the ScaleBFM sim."""

    def __init__(self, sim):
        self.sim = sim
        self.mj = np.array([sim.m.jnt_qposadr[sim.m.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.rows = {k: [] for k in ("t", "qpos", "pelvis", "joints", "box", "cmd_pelvis", "cmd_wrist", "cmd_jaw")}

    def add(self, t, targets):
        d = self.sim.d
        self.rows["t"].append(t)
        self.rows["qpos"].append(d.qpos.copy())
        self.rows["pelvis"].append(d.qpos[:7].copy())
        self.rows["joints"].append(d.qpos[self.mj].copy())
        self.rows["box"].append(d.qpos[self.sim.crate_q : self.sim.crate_q + 7].copy())
        pel = targets.get("pelvis")
        self.rows["cmd_pelvis"].append(np.full(7, np.nan) if pel is None else np.concatenate([pel[0][0], pel[1][0]]))
        w = np.full((2, 7), np.nan)
        for k, b in enumerate(WRISTS):
            x = targets.get(b)
            if x is not None:
                w[k] = np.concatenate([x[0][0], x[1][0]])
        self.rows["cmd_wrist"].append(w)
        self.rows["cmd_jaw"].append(self.sim.grip_target[[0, 2]].copy())

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.rows.items()}


def rotvec(Rm):
    """Rotation vector of a rotation matrix."""
    q = R.quat_wxyz(Rm)
    q = q if q[0] >= 0 else -q
    s_ = np.linalg.norm(q[1:])
    return np.zeros(3) if s_ < 1e-9 else 2 * np.arctan2(s_, q[0]) * q[1:] / s_


def expmap(v):
    a = float(np.linalg.norm(v))
    if a < 1e-9:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def place_box(sim, pose):
    sim.d.qpos[sim.crate_q : sim.crate_q + 7] = pose
    sim.d.qvel[sim.m.jnt_dofadr[sim.m.body_jntadr[sim.crate_b]] :][:6] = 0
    mujoco.mj_forward(sim.m, sim.d)


def set_jaws(sim, left, right):
    sim.grip_target[:2] = JAW_OPEN if not np.isfinite(left) else left
    sim.grip_target[2:] = JAW_OPEN if not np.isfinite(right) else right


def stance_of(z, f, i) -> tuple[float, float]:
    """(stand-off: pelvis before the edge line along the walk-in axis, lateral: pelvis left of the box centre) at tick i."""
    axis = np.asarray(f["edge"]["axis"], float)
    perp = np.array([-axis[1], axis[0]])
    p = z["pelvis"][i][:2]
    return float(f["edge"]["line"] - axis @ p), float(perp @ (p - z["box"][i][:2]))


def stance_tick(z, f) -> int:
    """The tick SONIC's stance is read at: its `handoff` (the walk-in and the settle done), else 1 s after the walk-in."""
    ev = f["events"]
    if "handoff" in ev:
        return int(np.searchsorted(z["t"], ev["handoff"]))
    return int(np.searchsorted(z["t"], walk_end(z) + 1.0))


def walk_end(z) -> float:
    """The last tick before the first jaw closure at which the pelvis moved faster than 0.1 m/s."""
    v = np.linalg.norm(np.diff(z["pelvis"][:, :2], axis=0), axis=1) / np.diff(z["t"])
    shut = np.flatnonzero(np.isfinite(z["cmd_jaw"][:, 0]) & (z["cmd_jaw"][:, 0] < JAW_OPEN - 0.003))
    stop = shut[0] if len(shut) else len(v)
    moving = np.flatnonzero(v[:stop] > 0.1)
    return float(z["t"][moving[-1] + 1]) if len(moving) else float(z["t"][0])


def walk_round(pol, sim, z, f, a):
    axis = np.asarray(f["edge"]["axis"], float)
    perp = np.array([-axis[1], axis[0]])
    line = float(f["edge"]["line"])
    box0 = z["box"][0]
    psi = square_yaw(float(f["heading"]), R.yaw_of(box0[3:]))
    goal = (line - a.standoff) * axis + float(perp @ box0[:2]) * perp
    p0, yaw0 = z["pelvis"][0][:2].copy(), R.yaw_of(z["pelvis"][0][3:])
    dist = float(np.linalg.norm(goal - p0))
    d = (goal - p0) / dist
    prof = Profile(a.walk_speed, R.WALK_ACC, dist)
    sim.reset(p0, yaw0, 0.0)
    seat(sim)
    place_box(sim, box0)
    refs = R.Refs(pol, sim)
    pol.reset(sim.state())

    def ref(t):
        p = p0 + d * prof.s(t - R.SETTLE_S)
        yaw = yaw0 + R.smoothstep((t - R.SETTLE_S) / YAW_TURN_S) * R.wrap(psi - yaw0)
        pel = np.array([p[0], p[1], R.PELVIS_Z])
        out = {"pelvis": (pel, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))}
        out.update(refs.hands_at_side(pel, yaw))
        return out

    t_end = R.SETTLE_S + prof.t_end + a.settle
    rec = Recorder(sim)
    push = np.zeros(2)
    k, fell = 0, None
    while k * R.DT < t_end:
        t = k * R.DT
        mujoco.mj_kinematics(sim.m, sim.d)
        fut = refs.futures(ref, t)
        rec.add(t, fut)
        if a.walk_servo and t >= R.SETTLE_S + prof.t_end:
            push = np.clip(push + WALK_KI * R.DT * (goal - sim.d.qpos[:2]), -WALK_CAP, WALK_CAP)
            fut = {b: (p_ + np.append(push, 0.0), q_) for b, (p_, q_) in fut.items()}
        if sim.fallen():
            fell = t
            break
        sim.step(pol(sim.state(), fut))
        k += 1
    facts = dict(
        f,
        controller="scalebfm",
        program=f"walk_so{round(100 * a.standoff)}{'s' if a.walk_servo else ''}{a.tag}",
        walk_servo=dict(ki=WALK_KI, cap=WALK_CAP) if a.walk_servo else None,
        mode=int(pol.mode.item()),
        events=dict(walk=R.SETTLE_S, brake=R.SETTLE_S + prof.t_dec, end=t_end),
        walk_target=dict(standoff=a.standoff, xy=goal.tolist(), yaw=psi, distance=dist, speed=a.walk_speed),
        ok=fell is None,
        why="" if fell is None else f"fell at {fell:.2f} s",
    )
    return rec.arrays(), facts


def carry_profile(kind: str, yaw_from: float):
    """(pelvis offset along the heading in use (m), walk along the new heading?, yaw) vs seconds since the carry
    started, and its length (module docstring)."""
    prof = Profile(CARRY_SPEED, R.WALK_ACC, 1.0)
    if kind == "back1":
        return (lambda s: (-prof.s(s - 1.0), False, yaw_from)), 1.0 + prof.t_end + 3.0
    t_turn = np.pi / CARRY_YAW_RATE

    def at(s):
        return prof.s(s - 1.0 - t_turn), True, yaw_from + np.pi * R.smoothstep((s - 1.0) / t_turn)

    return at, 1.0 + t_turn + prof.t_end + 3.0


def replay_start(z, f) -> float:
    ev = f["events"]
    if "settle" in ev and "handoff" in ev:
        return min(ev["settle"] + 1.0, ev["handoff"] - 0.5)
    return walk_end(z) + 1.0


def replay_round(pol, sim, z, f, a, fk):
    t_s = z["t"]
    t0 = replay_start(z, f)
    t1 = float(t_s[-1])
    i0 = int(np.searchsorted(t_s, t0))
    bodies = MODE_BODIES.get(a.mode) or tuple(pol.sel)
    span = np.arange(i0, len(t_s))
    kp = {b: np.empty((len(span), 7)) for b in bodies}
    for n, i in enumerate(span):
        fk.put(z["pelvis"][i], z["joints"][i])
        for b in bodies:
            kp[b][n] = fk.pose(b)
    ts = t_s[span] - t0
    jaw = z["cmd_jaw"][span]

    def at(tt):
        return int(np.clip(np.searchsorted(ts, tt), 0, len(ts) - 1))

    pel0 = z["pelvis"][i0].copy()
    if a.pelvis_shift:
        back = -a.pelvis_shift * np.append(np.asarray(f["edge"]["axis"], float), 0.0)
        kp["pelvis"][:, :3] += back
        pel0[:3] += back
    yaw0 = R.yaw_of(pel0[3:])
    sim.reset(pel0[:2], yaw0, 0.0)
    seat(sim)
    place_box(sim, z["box"][i0])
    refs = R.Refs(pol, sim)
    pol.reset(sim.state())
    start = {"pelvis": (np.array([pel0[0], pel0[1], R.PELVIS_Z]), np.array([np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)]))}
    start.update(refs.hands_at_side(start["pelvis"][0], yaw0))

    def ref(t):
        tt = t - SETTLE
        if tt >= 0:
            j = at(tt)
            return {b: (kp[b][j, :3], kp[b][j, 3:]) for b in bodies}
        u = R.smoothstep(t / SETTLE)
        out = {}
        for b in bodies:
            p1, q1 = kp[b][0, :3], kp[b][0, 3:]
            p0_, q0_ = start.get(b, (p1, q1))
            out[b] = (p0_ + u * (p1 - p0_), slerp(q0_, q1, u))
        return out

    t_end = SETTLE + (t1 - t0)
    if a.carry is not None:
        pe = kp["pelvis"][-1] if "pelvis" in kp else np.concatenate(fk_pelvis(fk, z, len(t_s) - 1))
        yaw_c = R.yaw_of(pe[3:])
        Rc = R.rot_z(yaw_c)
        rel = {b: (Rc.T @ (kp[b][-1, :3] - pe[:3]), Rc.T @ R.mat(kp[b][-1, 3:])) for b in bodies}
        carry_at, carry_s = carry_profile(a.carry, yaw_c)
        replay_ref = ref
        t_carry = t_end  # the carry starts here (t_end grows below; the closure must not read it)

        def ref(t):
            s_ = t - t_carry
            if s_ < 0:
                return replay_ref(t)
            dist, along_new, yaw = carry_at(s_)
            h = yaw if along_new else yaw_c
            xy = pe[:2] + dist * np.array([np.cos(h), np.sin(h)])
            Rn = R.rot_z(yaw)
            P = np.array([xy[0], xy[1], pe[2]])
            return {b: (P + Rn @ rel[b][0], R.quat_wxyz(Rn @ rel[b][1])) for b in bodies}

        t_end += carry_s
    rec = Recorder(sim)
    servo = {side: np.zeros(3) for side in ARM_BODIES}
    turn = {side: np.zeros(3) for side in ARM_BODIES}  # rotation vector, world frame
    k, fell = 0, None
    while k * R.DT < t_end:
        t = k * R.DT
        j = at(t - SETTLE) if t >= SETTLE else 0
        set_jaws(sim, *jaw[j]) if t >= SETTLE else set_jaws(sim, np.nan, np.nan)
        mujoco.mj_kinematics(sim.m, sim.d)
        fut = refs.futures(ref, t)
        rec.add(t, fut)  # the reference (SONIC's keypoints), not the servo-shifted targets
        if a.servo and t >= SETTLE:
            for side, arm in ARM_BODIES.items():
                w = arm[-1]
                if w not in fut:
                    continue
                p_ref, R_ref = fut[w][0][0], R.mat(fut[w][1][0])
                p_meas, q_meas = sim.body(w)
                R_meas = R.mat(q_meas)
                shift = np.zeros_like(fut[w][0])
                if a.servo == "wrist":
                    err = p_ref - p_meas
                else:
                    err = (p_ref + R_ref @ R.OFF) - (p_meas + R_meas @ R.OFF)
                    e_rot = rotvec(R_ref @ R_meas.T)
                    turn[side] = turn[side] + SERVO_KI_ROT * R.DT * e_rot
                    n = np.linalg.norm(turn[side])
                    if n > SERVO_CAP_ROT:
                        turn[side] *= SERVO_CAP_ROT / n
                    lead_rot = turn[side] + a.servo_kp * e_rot
                    n = np.linalg.norm(lead_rot)
                    Rt = expmap(lead_rot if n <= SERVO_CAP_ROT else lead_rot * SERVO_CAP_ROT / n)
                    quats = np.array([R.quat_wxyz(Rt @ R.mat(q)) for q in fut[w][1]])
                    # keep the jaw centre where the reference has it under the turned attitude
                    shift = np.array([R.mat(q0) @ R.OFF - Rt @ R.mat(q0) @ R.OFF for q0 in fut[w][1]])
                    fut[w] = (fut[w][0], quats)
                servo[side] = np.clip(servo[side] + SERVO_KI * R.DT * err, -SERVO_CAP, SERVO_CAP)
                lead = servo[side] + np.clip(a.servo_kp * err, -SERVO_CAP, SERVO_CAP)
                for b in arm:
                    if b in fut:
                        fut[b] = (fut[b][0] + lead + (shift if b == w else 0.0), fut[b][1])
        if sim.fallen():
            fell = t
            break
        sim.step(pol(sim.state(), fut))
        k += 1
    shift = SETTLE - t0
    events = {
        k_: v + shift for k_, v in f["events"].items() if k_ not in ("walk", "brake", "settle") and v + shift >= 0
    }
    if a.carry is not None:
        events["carry"] = SETTLE + (t1 - t0)
    name = (
        (f"replay_m{a.mode}" if a.carry is None else f"carry_{a.carry}_m{a.mode}")
        + {None: "", "wrist": "s", "pad": "p"}[a.servo]
        + ("" if not a.servo_kp else f"k{a.servo_kp:g}")
    )
    facts = dict(
        f,
        controller="scalebfm",
        program=name
        + ("" if a.box_mass is None else f"_kg{a.box_mass:g}")
        + ("" if not a.pelvis_shift else f"_shift{round(100 * a.pelvis_shift):+d}cm")
        + a.tag,
        pelvis_shift=a.pelvis_shift,
        mode=a.mode,
        servo=None
        if a.servo is None
        else dict(kind=a.servo, ki=SERVO_KI, kp=a.servo_kp, cap=SERVO_CAP, ki_rot=SERVO_KI_ROT, cap_rot=SERVO_CAP_ROT),
        box_mass=a.box_mass,
        events=events,
        replay=dict(sonic_t0=t0, sonic_t1=t1, bodies=list(bodies)),
        sonic_ok=f["ok"],
        sonic_why=f["why"],
        ok=fell is None,
        why="" if fell is None else f"fell at {fell:.2f} s",
    )
    return rec.arrays(), facts


def fk_pelvis(fk, z, i):
    fk.put(z["pelvis"][i], z["joints"][i])
    p = fk.pose("pelvis")
    return p[:3], p[3:]


def sonic_standoff(sonic: Path) -> float:
    """Median stance of SONIC's completed rounds (stance_of at stance_tick)."""
    so = []
    for npz in sorted(sonic.glob("*.npz")):
        f = json.loads(npz.with_suffix(".json").read_text())
        if f["ok"]:
            z = dict(np.load(npz))
            so.append(stance_of(z, f, stance_tick(z, f))[0])
    return float(np.median(so))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", choices=("walk", "replay"), required=True)
    ap.add_argument("--mode", type=int, default=2)
    ap.add_argument("--carry", choices=("back1", "turn180"), default=None)
    ap.add_argument(
        "--servo", choices=("wrist", "pad"), default=None, help="replay: integral hand servo on SONIC's measured hands"
    )
    ap.add_argument("--servo-kp", type=float, default=0.0, help="replay: proportional gain added to the hand servo")
    ap.add_argument("--pelvis-shift", type=float, default=0.0, help="replay, mode 2: stand this much farther (m)")
    ap.add_argument("--box-mass", type=float, default=None, help="replay: the box's mass (kg; the scene's is 1.2)")
    ap.add_argument("--sonic", type=Path, required=True, help="SONIC round files (warehouse_states.py)")
    ap.add_argument("--walk-speed", type=float, default=0.25, help="walk: m/s (warehouse_mars_200 visual.walk_speed)")
    ap.add_argument(
        "--standoff", type=float, default=None, help="walk: pelvis before the edge line (m); default SONIC's median"
    )
    ap.add_argument("--settle", type=float, default=3.0, help="walk: simulated seconds after the reference stops")
    ap.add_argument("--walk-servo", action="store_true", help="walk: integral pelvis servo after the reference stops")
    ap.add_argument("--only-ok", action="store_true", help="only the rounds SONIC completed")
    ap.add_argument("--tag", default="", help="appended to the program's name")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    assert not a.pelvis_shift or a.mode == 2, "--pelvis-shift moves only the pelvis keypoint: mode 2"
    if a.program == "walk" and a.standoff is None:
        a.standoff = sonic_standoff(a.sonic)
        print(f"stand-off: SONIC's median {a.standoff:.3f} m")
    cfg = ScaleBfmBackendConfig(
        scene_dir=str(SCENE), object_name="nor_cal_box", grip_kp=HAND_KP, grip_kd=HAND_KD, grip_clip=HAND_FORCE
    )
    pol = R.Policy(device=a.device, backend=cfg)
    pol.mode = torch.tensor([a.mode], dtype=torch.long, device=a.device)
    sim = R.Sim(pol.meta, backend=cfg)
    if a.box_mass is not None:
        b = sim.crate_b
        sim.m.body_inertia[b] *= a.box_mass / sim.m.body_mass[b]
        sim.m.body_mass[b] = a.box_mass
    assert sim.m.body(sim.crate_b).name == BOX
    fk = Fk(SCENE)
    for npz in sorted(a.sonic.glob("*.npz")):
        z = dict(np.load(npz))
        f = json.loads(npz.with_suffix(".json").read_text())
        if (a.out / "STOP").exists():
            break
        if (a.only_ok or a.carry is not None) and not f["ok"]:
            print(npz.stem, "skipped: SONIC did not complete it")
            continue
        if a.program == "replay" and not np.any(np.isfinite(z["cmd_jaw"]) & (z["cmd_jaw"] < JAW_OPEN - 0.003)):
            print(npz.stem, "skipped: SONIC never closed the jaws")
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
