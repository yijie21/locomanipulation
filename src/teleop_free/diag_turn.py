"""How fast can the robot turn on the spot, and in which mode?

Written when the spawn-scatter work (roadmap 2b) tried to give the robot an
initial heading error and turn it out before the walk. `Primitives.turn_to`
already carried the warning -- "an IDLE facing more than ~60 deg away stalls
the controller (measured: 68 deg turns timing out while 30 deg turns finish
in a second)" -- and stepping the commanded facing in <=0.5 rad increments,
which is turn_to's answer to it, was not enough: a -70 deg turn timed out at
18 s.

So measure it rather than tune the timeout blind. For each mode and each
commanded error this drives the same stepped-facing loop turn_to uses and
records the yaw against time: time to converge, and the sustained rate.

The second question is whether IDLE is simply the wrong mode. A person turns
on the spot by STEPPING, and the planner has gait modes; if SLOW_WALK with
zero translation turns several times faster than IDLE, the spawn turn should
use it and turn_to's ceiling is a mode choice rather than a policy limit.

    python src/teleop_free/diag_turn.py [--max-deg 120]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths  # noqa: E402
from agent_server import LocomotionMode  # noqa: E402
from primitives import DT, Primitives, wrap  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402

STEP = 0.5  # rad, the cap turn_to uses on the commanded facing
TOL = 0.12  # rad, "arrived"
BUDGET = 25.0  # s per attempt
OUT = f"{paths.DIAG_DIR}/turn.json"


def turn_probe(p, mode, target_err, settle_first=True):
    """Command a turn of `target_err` rad from the current yaw; log the trace.

    Returns (converged, seconds, trace) where trace is [(t, yaw_err)].
    """
    if settle_first:
        y = p.pelvis_yaw(p.obs())
        end = time.time() + 1.5
        while time.time() < end:
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=(np.cos(y), np.sin(y), 0.0))
            time.sleep(DT)
    yaw0 = p.pelvis_yaw(p.obs())
    goal = wrap(yaw0 + target_err)
    t0 = time.time()
    trace = []
    while time.time() - t0 < BUDGET:
        s = p.obs()
        if s["fallen"]:
            return False, time.time() - t0, trace
        err = wrap(goal - p.pelvis_yaw(s))
        trace.append((round(time.time() - t0, 2), round(float(err), 4)))
        if abs(err) < TOL:
            return True, time.time() - t0, trace
        # Identical to turn_to: never command more than STEP away from where
        # the robot actually is.
        cmd = p.pelvis_yaw(s) + np.clip(err, -STEP, STEP)
        p.cmd.frame(mode=mode, facing=(np.cos(cmd), np.sin(cmd), 0.0), movement=(0.0, 0.0, 0.0))
        time.sleep(DT)
    return False, time.time() - t0, trace


def rate_of(trace):
    """Sustained turn rate, deg/s, over the middle of the attempt."""
    if len(trace) < 10:
        return float("nan")
    a, b = trace[len(trace) // 5], trace[-1]
    dt = b[0] - a[0]
    if dt <= 0.2:
        return float("nan")
    return float(np.degrees(abs(a[1]) - abs(b[1])) / dt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-deg", type=float, default=120.0)
    args = ap.parse_args()

    p = Primitives()
    reset = EpisodeControlClient()
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label="diag_turn")
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    if p.obs()["fallen"]:
        sys.exit("did not stand")

    import stream_reach

    stream_reach.reanchor(p)

    angles = [a for a in (15, 30, 45, 60, 90, 120) if a <= args.max_deg]
    modes = [("IDLE", LocomotionMode.IDLE), ("SLOW_WALK", LocomotionMode.SLOW_WALK)]
    results = []
    print(f"{'mode':10s} {'cmd':>6} {'converged':>10} {'seconds':>8} {'rate deg/s':>11} {'residual':>9}")
    for name, mode in modes:
        for a in angles:
            for sign in (+1, -1):
                ok, secs, tr = turn_probe(p, mode, np.deg2rad(sign * a))
                res = np.degrees(tr[-1][1]) if tr else float("nan")
                r = rate_of(tr)
                print(f"{name:10s} {sign * a:+6.0f} {str(ok):>10} {secs:8.1f} {r:11.1f} {res:+8.1f}")
                results.append(
                    dict(
                        mode=name,
                        cmd_deg=sign * a,
                        ok=bool(ok),
                        seconds=round(secs, 2),
                        rate_deg_s=None if np.isnan(r) else round(r, 2),
                        residual_deg=round(float(res), 1),
                    )
                )
                if p.obs()["fallen"]:
                    print("  fell — resetting")
                    p.cmd.reset_pose()
                    reset.reset(label="diag_turn")
                    time.sleep(0.3)
                    p.cmd.start_control()
                    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
                    stream_reach.reanchor(p)

    os.makedirs(paths.DIAG_DIR, exist_ok=True)
    import json

    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
