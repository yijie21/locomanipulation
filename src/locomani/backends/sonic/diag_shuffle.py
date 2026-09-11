"""Can SONIC shuffle? Step frequency, per-step advance, and STOP CARRY.

The stand keeps missing its planned point because a walk that is commanded
to stop finishes the stride it is in (measured 9-21 cm after walk_to
returns). The fix everyone uses on G1 demos is to stay IN the gait and
shuffle — feet alternating, a couple of centimetres per step — so there is
no stop transient to absorb. That is only useful here if SONIC's low-speed
regime really is a shuffle rather than a stall, so this measures, per
(mode, speed):

    net m/s      how fast the pelvis actually travels
    steps/s      gait cycles, counted from the hip-pitch difference
    cm/step      net advance per cycle — the shuffle's quantum
    carry cm     how far it keeps going after IDLE is commanded
    carry steps  how many more cycles that takes

Also probes SLOW_WALK with a zero movement vector (march in place?) and, since
2026-09-11, a pure IDLE leg: planner-mode IDLE drifts on its own (8-24 cm/s has
been measured), and the carry is a DISPLACEMENT over 3.5 s of commanded IDLE, so
without that control the carry column cannot be read.

    python -m locomani.backends.sonic.diag_shuffle --speeds 0.20,0.26 --repeats 3
                                                   (needs a live stack)

RESTORED 2026-09-11: this file was deleted by refactor step 2 (ec0eb1c, "delete
unimported diag/tool scripts") and brought back for the v1.1 re-calibration --
the speed command is NOT m/s, the mapping is a property of the CHECKPOINT, and
every walking constant in the repo (primitives_tuning.NAV_SPEED*,
round_phases_approach.APPR_*, stream_reach._creep_to_band) was picked off
release's mapping. The measurement logic is unchanged from the deleted version;
only the imports (teleop_free -> locomani.*), the scene-meta path, an IDLE
control leg, the recorded policy provenance and --json are new.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from locomani import paths
from locomani.backends.sonic.primitives import DT, LocomotionMode, Primitives
from locomani.sim.privileged_pub import EpisodeControlClient

META = paths.SCENE_META

LEG = 5.0  # seconds per setting
WARMUP = 1.5  # let the gait build before counting
L_HIP_P, R_HIP_P = 0, 6  # MuJoCo joint order: hip pitch


def gait_phase(s):
    """One scalar whose zero crossings are gait cycles."""
    j = s.get("joints") or []
    if len(j) < 12:
        return 0.0
    return float(j[L_HIP_P] - j[R_HIP_P])


def count_steps(trace):
    """Zero crossings of the hip-pitch difference, ignoring dither."""
    x = np.asarray(trace)
    if len(x) < 5 or np.ptp(x) < 0.02:  # never lifted a foot
        return 0
    sign = np.sign(x - np.mean(x))
    sign = sign[sign != 0]
    return int(np.sum(sign[1:] != sign[:-1]))


def free_heading(meta, xy):
    """Heading with the most clear floor: the tables are what a 5 s leg
    walks into, and a blocked leg reads as a stalled gait (the first
    sweep measured 0.01 m/s at speed 0.45 because the robot was already
    leaning on the table)."""
    rects = []
    for k in ("A", "B"):
        sf = meta["surfaces"].get(k)
        if sf:
            rects.append((np.asarray(sf["xy"], float), np.asarray(sf.get("half_extent", [0.6, 0.6])[:2], float)))
    best, best_d = 0.0, -1.0
    for th in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        v = np.array([np.cos(th), np.sin(th)])
        d = min(
            (
                min(np.linalg.norm(np.maximum(np.abs(xy + v * r - c) - h, 0)) for c, h in rects)
                for r in np.linspace(0.3, 2.0, 12)
            ),
            default=9.0,
        )
        if d > best_d:
            best, best_d = float(th), d
    return best, best_d


def reset_and_face(p, reset, meta):
    """One clean start per leg: quiesce, teleport home, stand, turn to open
    floor. Without the quiesce the deploy carries the previous leg's intent
    through the reset."""
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label="shuffle")
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    s = p.obs()
    if s["fallen"]:
        return None
    th, clear = free_heading(meta, p.pelvis_xy(s))
    p.turn_to(th, tol=0.12, timeout=12.0)
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=(np.cos(th), np.sin(th), 0.0))
    return th


def leg(p, mode, speed, heading, move=True):
    fv = (float(np.cos(heading)), float(np.sin(heading)), 0.0)
    mv = fv if move else (0.0, 0.0, 0.0)
    t0 = time.time()
    start_xy, start_t, trace = None, None, []
    while time.time() - t0 < LEG:
        s = p.obs()
        if s["fallen"]:
            return None
        if time.time() - t0 > WARMUP:
            if start_xy is None:
                start_xy, start_t = p.pelvis_xy(s).copy(), time.time()
            trace.append(gait_phase(s))
        p.cmd.frame(mode=mode, speed=speed, movement=mv, facing=fv)
        time.sleep(DT)
    s = p.obs()
    walk_end_xy, walk_end_t = p.pelvis_xy(s).copy(), time.time()
    net = float(np.linalg.norm(walk_end_xy - start_xy)) / (walk_end_t - start_t)
    steps = count_steps(trace) / (walk_end_t - start_t)

    # -- the carry: command IDLE and watch how far it keeps going ---------
    c_trace, t1 = [], time.time()
    while time.time() - t1 < 3.5:
        s = p.obs()
        c_trace.append(gait_phase(s))
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=fv)
        time.sleep(DT)
    carry = float(np.linalg.norm(p.pelvis_xy(p.obs()) - walk_end_xy))
    return dict(
        net=net,
        steps=steps,
        per_step=(net / steps * 100 if steps > 0.05 else float("nan")),
        carry=carry * 100,
        carry_steps=count_steps(c_trace) / 2.0,
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=str, default="")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--json", default=None, help="write every leg's numbers here")
    a = ap.parse_args()
    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label="shuffle")
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    if p.obs()["pelvis"]["pos"][2] < 0.5:
        sys.exit("robot is not standing")

    meta = json.load(open(META))
    if a.speeds:
        sps = [float(x) for x in a.speeds.split(",")]
        grid = [(LocomotionMode.SLOW_WALK, sp, True) for sp in sps for _ in range(a.repeats)]
    else:
        grid = [(LocomotionMode.SLOW_WALK, sp, True) for sp in (0.18, 0.22, 0.26, 0.30, 0.40)]
        grid += [(LocomotionMode.SLOW_WALK, 0.30, False), (LocomotionMode.WALK, 0.35, True)]
    # The control: commanded IDLE from a standstill. Its `net` is the drift
    # rate the carry column has to be read against, not zero.
    grid += [(LocomotionMode.IDLE, 0.0, False) for _ in range(max(1, a.repeats))]
    rows, consecutive_falls = [], 0
    print(f"\npolicy {paths.SONIC_POLICY}")
    print(
        f"\n{'mode':>10} {'speed':>6} {'move':>5} {'net m/s':>8} "
        f"{'steps/s':>8} {'cm/step':>8} {'carry cm':>9} {'carry steps':>11}"
    )
    for mode, sp, mv in grid:
        # A fall costs its own leg, not the rest of the grid: every leg starts
        # from a full reset, so the next one is independent. Three in a row is
        # a broken stack, not a data point (the deleted version broke out of
        # the sweep on the first fall and lost every remaining speed).
        if consecutive_falls >= 3:
            print("three consecutive falls -- stopping, the stack is not standing the robot up")
            break
        heading = reset_and_face(p, reset, meta)
        if heading is None:
            print(f"{mode.name:>10} {sp:>6.2f} {str(mv):>5}   FELL during the reset")
            rows.append(dict(mode=mode.name, speed=sp, move=mv, fell="reset"))
            consecutive_falls += 1
            continue
        r = leg(p, mode, sp, heading, move=mv)
        if r is None:
            print(f"{mode.name:>10} {sp:>6.2f} {str(mv):>5}   FELL")
            rows.append(dict(mode=mode.name, speed=sp, move=mv, fell=True))
            consecutive_falls += 1
            continue
        consecutive_falls = 0
        print(
            f"{mode.name:>10} {sp:>6.2f} {str(mv):>5} {r['net']:>8.3f} "
            f"{r['steps']:>8.2f} {r['per_step']:>8.1f} {r['carry']:>9.1f} "
            f"{r['carry_steps']:>11.1f}"
        )
        rows.append(dict(mode=mode.name, speed=sp, move=mv, fell=False, **r))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(policy=paths.SONIC_POLICY, leg_s=LEG, warmup_s=WARMUP, rows=rows), fh, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
