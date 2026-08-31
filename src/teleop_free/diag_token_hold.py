"""Net pelvis drift while the deploy holds ONE motion token.

The eval prelude holds initial_pose.initial_token() from a keepalive thread
while the first chunk is computed, then blends into it. vla_round's comment
claims the token channel is "rock-stable (0.013 m / 30 s)" -- but that was
measured in the NVIDIA-token era, and the probes disagree about what the
switch to OUR medoid cost:

    probe_initpose (NVIDIA token)  tstart_spawn_off median 0.108
    initpose_ours  (our medoid)    tstart_spawn_off median 0.254

tstart_spawn_off mixes four segments (planner settle, hold, blend, chunk-0
walking), so it cannot say WHICH token creeps. This isolates the hold:
teleport, settle, prime, enter streamed mode, stream one fixed token for
--secs, and measure the pelvis. Joint stillness in the medoid's source
frames (< 0.05 rad/s) does not imply zero base intent -- the token also
encodes velocities and anchor, and a 1/16-quantised "about to walk" lean
integrates into exactly this kind of creep.

Needs the stack up with NO_ENCODER=1 (same as any VLA eval).

    python src/teleop_free/diag_token_hold.py [--secs 30] [--reps 3] \
        [--tokens ours official]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tools"))

from agent_server import LocomotionMode, StreamMode  # noqa: E402
from primitives import Primitives                # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402
from sonic_token import GRIPPER_MAX, expand_hands  # noqa: E402

RATE = 25.0


def get_token(name: str) -> np.ndarray:
    os.environ["TF_INIT_TOKEN"] = name
    import importlib

    import initial_pose
    importlib.reload(initial_pose)
    return np.asarray(initial_pose.initial_token(), dtype=np.float32)


def trial(p, reset, tok: np.ndarray, secs: float) -> dict:
    l_open, r_open = expand_hands(GRIPPER_MAX)
    # Same reset + settle every eval round gets (grasp_gym's spawn path,
    # abbreviated): back to the planner channel, teleport, pin, settle.
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label="tokhold")
    time.sleep(0.3)
    p.cmd.start_control()
    time.sleep(2.0)
    s = p.obs()
    yaw0 = float(p.pelvis_yaw(s))
    fv = (float(np.cos(yaw0)), float(np.sin(yaw0)), 0.0)
    p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=fv)

    # Prime BEFORE the mode switch (streamed over an empty buffer stops
    # control), then hold: the exact prelude sequence from vla_round.
    for _ in range(30):
        p.cmd.token_frame(tok, right_hand=r_open, left_hand=l_open,
                          frame_index=0)
        time.sleep(0.02)
    p.cmd.enter_streamed()

    s0 = p.obs()
    xy0 = p.pelvis_xy(s0).copy()
    yaw_a = float(p.pelvis_yaw(s0))
    marks = []
    t0 = time.time()
    next_mark = 5.0
    while time.time() - t0 < secs:
        p.cmd.token_frame(tok, right_hand=r_open, left_hand=l_open,
                          frame_index=0)
        if time.time() - t0 >= next_mark:
            sm = p.obs()
            marks.append((next_mark,
                          float(np.linalg.norm(p.pelvis_xy(sm) - xy0))))
            next_mark += 5.0
        time.sleep(1.0 / RATE)
    s1 = p.obs()
    d = float(np.linalg.norm(p.pelvis_xy(s1) - xy0))
    dyaw = float(np.degrees(
        (float(p.pelvis_yaw(s1)) - yaw_a + np.pi) % (2 * np.pi) - np.pi))
    # Hand the deploy back to the planner so the next trial's reset is clean.
    p.cmd.exit_streamed()
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=fv)
    return dict(drift=d, dyaw=dyaw, fell=bool(s1["fallen"]), marks=marks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=30.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", nargs="+", default=["ours", "official"])
    args = ap.parse_args()

    p = Primitives()
    reset = EpisodeControlClient()
    for name in args.tokens:
        tok = get_token(name)
        ds = []
        for r in range(args.reps):
            out = trial(p, reset, tok, args.secs)
            ds.append(out["drift"])
            prog = " ".join(f"{t:.0f}s:{v:.3f}" for t, v in out["marks"])
            print(f"[{name}] rep {r + 1}/{args.reps}: net {out['drift']:.3f} m "
                  f"dyaw {out['dyaw']:+.1f} deg fell={out['fell']}  ({prog})",
                  flush=True)
        ds = np.array(ds)
        print(f"[{name}] {args.secs:.0f} s hold: median {np.median(ds):.3f} m "
              f"max {ds.max():.3f} m over {args.reps} reps", flush=True)


if __name__ == "__main__":
    main()
