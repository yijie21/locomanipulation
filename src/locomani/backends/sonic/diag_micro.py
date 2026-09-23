"""Can SONIC make SMALL moves -- a few centimetres, a few degrees -- and is the limit the controller or us?

Four trial kinds, each from a fresh reset facing +x in the walk scene (G1 alone on the floor), each
read AFTER a quiet IDLE settle so the stop carry is inside the number:

    face        raw: command a facing yaw0+delta with a zero movement vector (IDLE or SLOW_WALK)
    burst       raw: SLOW_WALK at `speed` along forward / left / back for `seconds`, facing held
    goal_walk   ours: Primitives.walk_to a point dx, dy away (stop_dist 2 cm, heading held)
    goal_turn   ours: Primitives.turn_to yaw0+delta at a 2 deg tolerance
    idle        control: IDLE for the same settle, the drift every other row is read against

`face` / `burst` say what the planner can deliver at all; `goal_*` say what our closed loop gets
out of it. Raw small moves that exist while the goals miss is a program problem; raw moves that do
not exist is the controller's floor.

    LOCOMANI_SIM_DIR=outputs/scene/walk python -m locomani.backends.sonic.diag_micro --json out.json
                                                                   (needs a live stack on that scene)
Touch <json dir>/STOP to end after the current trial; rows are written after every trial.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from locomani import paths
from locomani.backends.sonic.primitives import DT, LocomotionMode, Primitives
from locomani.sim.privileged_pub import EpisodeControlClient

SETTLE = 3.0  # quiet IDLE after every trial before it is measured
FACE_HOLD = 4.0  # how long a raw facing command is held
FACES_DEG = (3, 5, 10, 15, 20)
BURST_SPEEDS = (0.20, 0.35)
BURST_DIRS_DEG = (0, 90, 180)
BURST_SECONDS = (0.3, 0.6, 1.0, 1.5)
GOALS_M = ((0.05, 0), (0.10, 0), (0.20, 0), (0, 0.05), (0, 0.10), (0, 0.20), (-0.05, 0))
TURNS_DEG = (3, 5, 10, 20)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def pose(p):
    s = p.obs()
    return p.pelvis_xy(s).copy(), float(p.pelvis_yaw(s)), bool(s["fallen"])


def fresh_start(p, reset):
    """Quiesce, teleport home, stand. The same preamble as diag_shuffle.reset_and_face, minus its
    turn to open floor: the walk scene is all open floor, and a turn would be a move of its own."""
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label="micro")
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    xy, yaw, fallen = pose(p)
    if fallen:
        return None
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=(np.cos(yaw), np.sin(yaw), 0.0))
    return pose(p)


def settle(p, yaw):
    p.cmd.hold(SETTLE, mode=LocomotionMode.IDLE, facing=(np.cos(yaw), np.sin(yaw), 0.0))


def run_trial(p, kind, arg, start):
    xy0, yaw0, _ = start
    fv0 = (np.cos(yaw0), np.sin(yaw0), 0.0)
    want_xy, want_yaw, extra = xy0, yaw0, {}
    t0 = time.time()
    if kind == "face":
        mode, deg = arg
        want_yaw = yaw0 + np.radians(deg)
        fv = (np.cos(want_yaw), np.sin(want_yaw), 0.0)
        while time.time() - t0 < FACE_HOLD:
            p.cmd.frame(mode=mode, movement=(0.0, 0.0, 0.0), facing=fv)
            time.sleep(DT)
        settle(p, want_yaw)
    elif kind == "burst":
        speed, dir_deg, seconds = arg
        heading = yaw0 + np.radians(dir_deg)
        mv = (np.cos(heading), np.sin(heading), 0.0)
        while time.time() - t0 < seconds:
            p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=speed, movement=mv, facing=fv0)
            time.sleep(DT)
        settle(p, yaw0)
    elif kind == "goal_walk":
        dx, dy = arg
        c, s = np.cos(yaw0), np.sin(yaw0)
        want_xy = xy0 + np.array([c * dx - s * dy, s * dx + c * dy])
        extra["reached"] = bool(p.walk_to(want_xy, stop_dist=0.02, timeout=15.0, face_target=False, face_yaw=yaw0))
        settle(p, yaw0)
    elif kind == "goal_turn":
        want_yaw = yaw0 + np.radians(arg)
        extra["reached"] = bool(p.turn_to(want_yaw, tol=np.radians(2.0), timeout=10.0, mode=LocomotionMode.SLOW_WALK))
        settle(p, want_yaw)
    elif kind == "idle":
        settle(p, yaw0)
    extra["seconds"] = round(time.time() - t0, 2)
    xy1, yaw1, fallen = pose(p)
    c, s = np.cos(yaw0), np.sin(yaw0)
    d = xy1 - xy0
    e = xy1 - want_xy
    return dict(
        fell=fallen,
        fwd_cm=100 * float(c * d[0] + s * d[1]),
        lat_cm=100 * float(-s * d[0] + c * d[1]),
        dyaw_deg=float(np.degrees(wrap(yaw1 - yaw0))),
        pos_err_cm=100 * float(np.linalg.norm(e)),
        yaw_err_deg=float(np.degrees(wrap(yaw1 - want_yaw))),
        **extra,
    )


def grid(repeats):
    rows = []
    for _ in range(repeats):
        rows += [("idle", None)]
        rows += [
            ("face", (m, sg * d))
            for m in (LocomotionMode.IDLE, LocomotionMode.SLOW_WALK)
            for d in FACES_DEG
            for sg in (1, -1)
        ]
        rows += [("burst", (sp, dd, sec)) for sp in BURST_SPEEDS for dd in BURST_DIRS_DEG for sec in BURST_SECONDS]
        rows += [("goal_walk", g) for g in GOALS_M]
        rows += [("goal_turn", sg * d) for d in TURNS_DEG for sg in (1, -1)]
    return rows


def label(kind, arg):
    if kind == "face":
        return f"{arg[0].name}:{arg[1]:+d}deg"
    if kind == "burst":
        return f"v{arg[0]:.2f}@{arg[1]}deg x{arg[2]}s"
    if kind == "goal_walk":
        return f"dx{arg[0]:+.2f} dy{arg[1]:+.2f}"
    if kind == "goal_turn":
        return f"{arg:+d}deg"
    return "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--kinds", default="idle,face,burst,goal_walk,goal_turn")
    a = ap.parse_args(argv)
    kinds = set(a.kinds.split(","))
    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    out = dict(policy=paths.SONIC_POLICY, scene=paths.SIM_DIR, settle_s=SETTLE, face_hold_s=FACE_HOLD, rows=[])
    falls = 0
    print(f"policy {paths.SONIC_POLICY}  scene {paths.SIM_DIR}")
    print(f"{'kind':>10} {'setting':>22} {'fwd cm':>7} {'lat cm':>7} {'dyaw':>6} {'pos err':>8} {'yaw err':>8}")
    for kind, arg in [r for r in grid(a.repeats) if r[0] in kinds]:
        if (a.json.parent / "STOP").exists() or falls >= 3:
            print("stopping:", "STOP file" if falls < 3 else "three consecutive falls")
            break
        start = fresh_start(p, reset)
        row = dict(kind=kind, setting=label(kind, arg))
        if start is None:
            falls += 1
            out["rows"].append(dict(row, fell="reset"))
            continue
        r = run_trial(p, kind, arg, start)
        falls = falls + 1 if r["fell"] else 0
        out["rows"].append(dict(row, **r))
        print(
            f"{kind:>10} {row['setting']:>22} {r['fwd_cm']:>+7.1f} {r['lat_cm']:>+7.1f} {r['dyaw_deg']:>+6.1f} "
            f"{r['pos_err_cm']:>8.1f} {r['yaw_err_deg']:>+8.1f}{'  FELL' if r['fell'] else ''}",
            flush=True,
        )
        a.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
