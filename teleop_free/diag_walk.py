"""Measure how the deploy's `facing`/`movement` frame relates to MuJoCo world.

`facing` is an absolute heading vector, but absolute *in whose frame*? The C++
deploy takes it in its state-estimator frame, and pico_manager's YawAccumulator
resets that frame to (1, 0, 0) whenever the operator recentres. If that frame is
rotated relative to MuJoCo world, a closed loop that computes headings from
privileged world positions steers into a constant error and spirals — which is
exactly what the first episode did.

This commands a fixed heading and measures what the robot actually does, so the
offset is a measured number rather than an assumption.

    python teleop_free/diag_walk.py [--seconds 6] [--heading-deg 0]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_server import AgentCommander, LocomotionMode  # noqa: E402
from privileged_pub import PrivilegedSubscriber  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--heading-deg", type=float, default=0.0)
    # The deploy normalises `movement` to a unit direction (the log shows [1,0,0]
    # for a 0.6-length vector), so the actual pace comes from `target_vel`, i.e.
    # the planner message's `speed` field. -1 asks for the mode's default.
    ap.add_argument("--speed", type=float, default=-1.0)
    ap.add_argument("--mode", type=int, default=int(LocomotionMode.WALK))
    ap.add_argument("--settle", type=float, default=2.0)
    args = ap.parse_args()

    cmd = AgentCommander()
    state = PrivilegedSubscriber()
    th = np.deg2rad(args.heading_deg)
    facing = (float(np.cos(th)), float(np.sin(th)), 0.0)

    cmd.start_control()
    # Stand still first: the measurement is of *walking*, so any transient from
    # entering control should be over before the clock starts.
    cmd.hold(args.settle, mode=LocomotionMode.IDLE, facing=facing)

    s0 = state.wait()
    p0 = np.array(s0["pelvis"]["pos"][:2])
    y0 = s0["pelvis"]["yaw"]

    deadline = time.time() + args.seconds
    next_sample = time.time()
    print("\n  t     pelvis xy            yaw")
    while time.time() < deadline:
        cmd.frame(mode=LocomotionMode(args.mode), movement=facing,
                  facing=facing, speed=args.speed)
        if time.time() >= next_sample:
            s = state.latest(0) or s0
            print(f"  {time.time() - (deadline - args.seconds):4.1f}  "
                  f"{np.round(s['pelvis']['pos'][:2], 3)}  "
                  f"{np.rad2deg(s['pelvis']['yaw']):+7.1f}  z={s['pelvis']['pos'][2]:.3f}")
            next_sample += 0.5
        time.sleep(0.02)
    cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=facing)

    s1 = state.wait()
    p1 = np.array(s1["pelvis"]["pos"][:2])
    y1 = s1["pelvis"]["yaw"]

    d = p1 - p0
    dist = float(np.linalg.norm(d))
    moved_deg = float(np.rad2deg(np.arctan2(d[1], d[0]))) if dist > 0.05 else float("nan")

    print(f"\nmode {LocomotionMode(args.mode).name} speed {args.speed}")
    print(f"commanded heading   {args.heading_deg:+7.1f} deg")
    print(f"start pelvis        {p0.round(3)}  yaw {np.rad2deg(y0):+7.1f} deg")
    print(f"end   pelvis        {p1.round(3)}  yaw {np.rad2deg(y1):+7.1f} deg")
    print(f"displacement        {d.round(3)}  |d| {dist:.3f} m")
    print(f"actual travel dir   {moved_deg:+7.1f} deg")
    if dist > 0.05:
        print(f"OFFSET (travel - commanded) {moved_deg - args.heading_deg:+7.1f} deg")
        print(f"OFFSET (yaw    - commanded) "
              f"{np.rad2deg(y1) - args.heading_deg:+7.1f} deg")
    print("fallen:", s1["fallen"])

    cmd.pause_control()
    cmd.close()
    state.close()


if __name__ == "__main__":
    main()
