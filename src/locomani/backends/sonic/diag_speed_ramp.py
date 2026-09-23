"""Commanded vs delivered walking speed: a staircase from standstill up to the top speed and back to a stop.

One run in the walk scene (G1 alone on the floor). The robot walks straight ahead on its start heading;
every PLATEAU seconds the planner speed field steps by STEP, from 0 up to --top and back down to 0,
then IDLE. Level 0 is sent as IDLE (a SLOW_WALK with speed 0 is not a documented command).

The sim writes the qpos trace (SIM_STATE_TRACE_DIR) between start_video/end_video; this process
writes, per control tick, the privileged `seq` it acted on and what it commanded, so the offline
renderer (scripts/experiments/render_speed_ramp_20260923.py) joins the two on `seq` exactly.

    LOCOMANI_SIM_DIR=outputs/scene/walk SIM_STATE_TRACE_DIR=<dir> \
        python -m locomani.backends.sonic.diag_speed_ramp --out <dir>        (needs a live stack)
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

LABEL = "speed_ramp"
SIM_HZ = 50  # privileged state publishes per simulated second; `seq` counts them


def levels(top: float, step: float) -> list[float]:
    """0, step, ..., top, ..., step, 0 -- rounded so the subtitle shows the value actually sent."""
    up = [round(v, 3) for v in np.arange(0.0, top + 1e-9, step)]
    return up + up[-2::-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--top", type=float, default=0.8, help="highest speed command (SLOW_WALK's documented range ends at 0.8)"
    )
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument(
        "--mode",
        type=int,
        default=int(LocomotionMode.SLOW_WALK),
        help="planner locomotion mode for every non-zero level (1 SLOW_WALK, 18 stealth, 20 careful/LEDGE_WALKING ...)",
    )
    ap.add_argument("--plateau", type=float, default=3.0, help="simulated seconds each speed command is held")
    a = ap.parse_args(argv)
    walk_mode = LocomotionMode(a.mode)
    a.out.mkdir(parents=True, exist_ok=True)
    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label=LABEL)
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    s = p.obs()
    if s["fallen"]:
        raise SystemExit("robot fell during the reset")
    yaw = float(p.pelvis_yaw(s))
    fv = (np.cos(yaw), np.sin(yaw), 0.0)
    ticks = []
    reset.start_video(LABEL)
    try:
        for level in levels(a.top, a.step) + [0.0]:  # the trailing 0 is the settle after the stop
            # Held for `plateau` SIMULATION seconds: the sim runs well below real time on this box, so a
            # wall-clock plateau would give each level a fraction of a stride.
            first = None
            mode = walk_mode if level > 0 else LocomotionMode.IDLE
            while first is None or s["seq"] - first < a.plateau * SIM_HZ:
                s = p.obs()
                first = s["seq"] if first is None else first
                p.cmd.frame(
                    mode=mode, speed=level if level > 0 else -1.0, movement=fv if level > 0 else (0, 0, 0), facing=fv
                )
                ticks.append(
                    dict(seq=s.get("seq"), speed=level, mode=int(mode), mode_name=mode.name, fallen=bool(s["fallen"]))
                )
                if s["fallen"]:
                    raise SystemExit(f"fell at speed command {level}")
                time.sleep(DT)
    finally:
        reset.end_video()
        (a.out / "commands.json").write_text(
            json.dumps(
                dict(
                    policy=paths.SONIC_POLICY,
                    scene=paths.SIM_DIR,
                    mode=int(walk_mode),
                    heading=yaw,
                    plateau_s=a.plateau,
                    ticks=ticks,
                )
            )
        )
    print(f"wrote {a.out / 'commands.json'} ({len(ticks)} ticks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
