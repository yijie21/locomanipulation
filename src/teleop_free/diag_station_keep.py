"""Can planner mode hold a position at all, and what sets the floor?

Context: the 18-D prelude has to keep the robot where the gym handed it over.
A first attempt (IDLE + a world movement vector aimed at the anchor) diverged
to 207 cm against planner IDLE's 23 cm, and was written off as "the movement
channel is not frame-compensated". diag_movement.py then measured the frame
and that explanation was WRONG: movement shares facing's compensated frame
(errors +16.1 / -4.7 / -4.1 deg, not the +49.6 a missing compensation would
give). What it does show is that the gait drags a commanded direction toward
the nose, hard, once the direction is far off it -- i.e. the planner cannot
strafe.

So this measures the three numbers that actually decide whether holding is
possible, and in a form where "unsolvable" would be a MEASUREMENT:

  A  IDLE            what the drift is if nothing is done            (baseline)
  B  stop distance   how far the robot travels AFTER the walk command stops.
                     This is the floor: a controller whose smallest possible
                     move is larger than the error it is correcting cannot
                     settle, it can only hunt.
  C  turn-and-walk   the recipe that DOES work elsewhere in this repo
                     (_nudge_pass / _creep_to_band): steer with facing, send
                     movement == facing, never strafe. Reports the residual
                     radius AND the heading excursion it costs, because for a
                     prelude the heading is the thing that must not move.

    INSTANCE=3 python src/teleop_free/diag_station_keep.py [--secs 20]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_server import LocomotionMode  # noqa: E402
from primitives import DT, Primitives, wrap  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402


def _fv(yaw):
    return (math.cos(yaw), math.sin(yaw), 0.0)


OFFSET_MODE = "yaw"  # see main(): which anchor convention to test


def _anchor(p, reset, label, settle=2.5):
    """Reset, wait out the teleport transient, CALIBRATE the anchor, then
    capture the reference pose.

    Two traps, both of which produced wrong numbers here before:

    1. The teleport lands the robot in model.qpos0 -- straight-legged, pelvis
       z 0.892 -- and it drops ~17 cm in 0.2 s catching itself. The base spins
       up to 90 deg during that catch (station_keep_watch2.txt). Anything
       measured, or anchored, inside that window is measuring the drop.
    2. `heading_offset` must equal the deploy's psi0, which is neither 0 nor
       the landing yaw. It is measurable directly: command a KNOWN facing with
       zero compensation, let the heading settle, and the yaw it settles at IS
       psi0 (the same quantity diag_heading.py reports). Measured -106 deg in a
       fresh planner-only process -- so both of the conventions this file first
       tried were wrong by ~60-90 deg, which is exactly what their dyaw
       columns showed.
    """
    reset.reset(robot={"dxy": (0.0, 0.0), "yaw": 0.0}, label=label)
    p.cmd.heading_offset = 0.0
    p.cmd.hold(settle, mode=LocomotionMode.IDLE, facing=(1.0, 0.0, 0.0))
    psi0 = float(p.pelvis_yaw(p.obs()))  # commanded 0 -> settles at psi0
    p.cmd.heading_offset = psi0
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=_fv(psi0))
    s = p.obs()
    return np.asarray(p.pelvis_xy(s), float).copy(), float(p.pelvis_yaw(s))


def trial_idle(p, reset, secs):
    a_xy, a_yaw = _anchor(p, reset, "sk_idle")
    d = []
    t_end = time.time() + secs
    while time.time() < t_end:
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=_fv(a_yaw))
        time.sleep(DT)
        d.append(np.linalg.norm(p.pelvis_xy(p.obs()) - a_xy))
    s = p.obs()
    return dict(net=d[-1], mx=max(d), dyaw=math.degrees(wrap(p.pelvis_yaw(s) - a_yaw)))


def trial_stop(p, reset, walk_s=2.0):
    """How much travel is left after the stop command -- the controller's quantum."""
    a_xy, a_yaw = _anchor(p, reset, "sk_stop")
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=_fv(a_yaw))
    t_end = time.time() + walk_s
    while time.time() < t_end:
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK, facing=_fv(a_yaw), movement=_fv(a_yaw), speed=0.215)
        time.sleep(DT)
    xy_cmd = np.asarray(p.pelvis_xy(p.obs()), float)  # at the stop command
    t_end = time.time() + 3.0
    while time.time() < t_end:
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=_fv(a_yaw))
        time.sleep(DT)
    xy_end = np.asarray(p.pelvis_xy(p.obs()), float)
    return dict(walked=float(np.linalg.norm(xy_cmd - a_xy)), coast=float(np.linalg.norm(xy_end - xy_cmd)))


def trial_hold(p, reset, secs, band_in=0.08, band_out=0.20, strafe=False):
    """Closed station-keeping. strafe=False steers with facing (the recipe that
    works); strafe=True keeps the heading pinned and sends a world movement
    vector (the one that diverged) -- both measured the same way."""
    a_xy, a_yaw = _anchor(p, reset, "sk_hold")
    walking = False
    d = []
    t_end = time.time() + secs
    while time.time() < t_end:
        s = p.obs()
        xy = np.asarray(p.pelvis_xy(s), float)
        yaw = float(p.pelvis_yaw(s))
        e = a_xy - xy
        r = float(np.linalg.norm(e))
        d.append(r)
        walking = r > band_out if not walking else r > band_in
        if not walking:
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=_fv(a_yaw))
        elif strafe:
            p.cmd.frame(
                mode=LocomotionMode.SLOW_WALK, facing=_fv(a_yaw), movement=(e[0] / r, e[1] / r, 0.0), speed=0.215
            )
        else:
            b = math.atan2(e[1], e[0])
            face = yaw + float(np.clip(wrap(b - yaw), -0.4, 0.4))
            p.cmd.frame(mode=LocomotionMode.SLOW_WALK, facing=_fv(face), movement=_fv(face), speed=0.215)
        time.sleep(DT)
    # give the heading back
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE, facing=_fv(a_yaw))
    s = p.obs()
    return dict(
        net=float(np.linalg.norm(p.pelvis_xy(s) - a_xy)),
        mx=max(d),
        mean=float(np.mean(d)),
        dyaw=math.degrees(wrap(p.pelvis_yaw(s) - a_yaw)),
    )


def trial_anchor_sweep(p, reset, secs=10.0):
    """WHICH yaw is the deploy's planner anchor? Hold IDLE at a pinned facing
    under each candidate convention and report the heading that survives.

    The compensation in AgentCommander.frame() subtracts `heading_offset` from
    every facing/movement command; the deploy then adds its own anchor psi0.
    So the residual yaw error IS (psi0 - heading_offset), and whichever
    convention drives it to zero names psi0. Candidates, in the order the
    codebase has assumed them:

      zero      what a fresh AgentCommander defaults to
      landing   the yaw at the instant the teleport lands
      settled   the yaw after the settle hold  (grasp_gym.py:924)
      reanchor  force the anchor with a streamed toggle (stream_reach.reanchor)
    """
    import stream_reach

    out = []
    for mode in ("zero", "landing", "settled", "reanchor"):
        reset.reset(label=f"sk_anchor_{mode}")
        time.sleep(0.3)
        yaw_land = float(p.pelvis_yaw(p.obs()))
        if mode == "zero":
            p.cmd.heading_offset = 0.0
        elif mode == "landing":
            p.cmd.heading_offset = yaw_land
        p.cmd.hold(1.5, mode=LocomotionMode.IDLE, facing=_fv(yaw_land))
        if mode == "settled":
            p.cmd.heading_offset = float(p.pelvis_yaw(p.obs()))
        elif mode == "reanchor":
            stream_reach.reanchor(p, side="right")
        s = p.obs()
        a_xy, a_yaw = np.asarray(p.pelvis_xy(s), float).copy(), float(p.pelvis_yaw(s))
        t_end = time.time() + secs
        while time.time() < t_end:
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=_fv(a_yaw))
            time.sleep(DT)
        s = p.obs()
        dy = math.degrees(wrap(p.pelvis_yaw(s) - a_yaw))
        d = float(np.linalg.norm(p.pelvis_xy(s) - a_xy))
        out.append((mode, dy, d, math.degrees(p.cmd.heading_offset)))
        print(
            f"  anchor {mode:9s}: offset {math.degrees(p.cmd.heading_offset):+7.1f} deg"
            f" -> dyaw {dy:+6.1f} deg, moved {d * 100:5.1f} cm",
            flush=True,
        )
    return out


def trial_watch(p, reset, secs=20.0):
    """Just watch the yaw. Does the robot ACTUALLY turn, or was the dyaw column
    in the earlier runs an artefact of this probe's own bookkeeping?

    Pins the start explicitly (robot yaw 0, the only value a reset to
    model.qpos0 can produce) and commands the matching facing with
    heading_offset 0, then prints the raw state. A base that really rotates
    ramps smoothly while upright; a bad reading jumps, or comes with a
    collapsing pelvis z (a fallen robot's yaw from atan2 is meaningless).
    """
    # High-rate window across the reset itself. A yaw that JUMPS between two
    # consecutive 20 ms samples is a bookkeeping/timing artefact (nothing
    # physical yaws 290 deg/s); one that RAMPS is the robot really turning.
    print(f"  pre-reset  yaw {math.degrees(p.pelvis_yaw(p.obs())):+7.1f}", flush=True)
    reset.reset(robot={"dxy": (0.0, 0.0), "yaw": 0.0}, label="sk_watch")
    p.cmd.heading_offset = 0.0
    t_r = time.time()
    while time.time() - t_r < 2.0:
        st = p.obs()
        print(
            f"  +{time.time() - t_r:4.2f}s  yaw {math.degrees(p.pelvis_yaw(st)):+7.1f}  "
            f"z {st['pelvis']['pos'][2]:.3f}  simt {st['t']:.2f}",
            flush=True,
        )
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=(1.0, 0.0, 0.0))
        time.sleep(0.1)
    t0 = time.time()
    nxt = 0.0
    while time.time() - t0 < secs:
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=(1.0, 0.0, 0.0))
        time.sleep(DT)
        if time.time() - t0 >= nxt:
            s = p.obs()
            xy = p.pelvis_xy(s)
            print(
                f"  t {time.time() - t0:5.1f}s  yaw {math.degrees(p.pelvis_yaw(s)):+7.1f}  "
                f"xy [{xy[0]:+.3f} {xy[1]:+.3f}]  pelvis_z {s['pelvis']['pos'][2]:.3f}  "
                f"fallen {bool(s['fallen'])}",
                flush=True,
            )
            nxt += 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--offset", choices=("yaw", "zero", "sweep", "watch"), default="yaw")
    a = ap.parse_args()
    global OFFSET_MODE
    OFFSET_MODE = a.offset
    print(f"heading_offset convention: {a.offset}", flush=True)
    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    time.sleep(1.0)
    p.cmd.start_control()
    p.cmd.hold(2.0, mode=LocomotionMode.IDLE)

    if a.offset == "watch":
        print("raw yaw trace, start pinned at yaw 0, heading_offset 0", flush=True)
        trial_watch(p, reset)
        p.cmd.pause_control()
        reset.close()
        p.close()
        return
    if a.offset == "sweep":
        print("anchor convention sweep (which yaw is the deploy's psi0?)", flush=True)
        trial_anchor_sweep(p, reset)
        p.cmd.pause_control()
        reset.close()
        p.close()
        return
    r = trial_stop(p, reset)
    print(
        f"B stop distance : walked {r['walked'] * 100:5.1f} cm, then coasted "
        f"{r['coast'] * 100:5.1f} cm after the stop command",
        flush=True,
    )
    r = trial_idle(p, reset, a.secs)
    print(
        f"A idle          : net {r['net'] * 100:5.1f} cm  max {r['mx'] * 100:5.1f} cm  dyaw {r['dyaw']:+5.1f} deg",
        flush=True,
    )
    r = trial_hold(p, reset, a.secs, strafe=True)
    print(
        f"C strafe hold   : net {r['net'] * 100:5.1f} cm  max {r['mx'] * 100:5.1f} cm  "
        f"mean {r['mean'] * 100:5.1f}  dyaw {r['dyaw']:+5.1f} deg",
        flush=True,
    )
    r = trial_hold(p, reset, a.secs, strafe=False)
    print(
        f"D turn-and-walk : net {r['net'] * 100:5.1f} cm  max {r['mx'] * 100:5.1f} cm  "
        f"mean {r['mean'] * 100:5.1f}  dyaw {r['dyaw']:+5.1f} deg",
        flush=True,
    )
    p.cmd.pause_control()
    reset.close()
    p.close()


if __name__ == "__main__":
    main()
