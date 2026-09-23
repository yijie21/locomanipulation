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
    python -m locomani.backends.sonic.diag_shuffle --march 0.30,0.35 --crab 0,90 --speeds 0.32
                                                   (needs a live stack)

2026-09-22 (ledger crate.md): `--march` legs (zero movement vector) do NOT step
in place -- 0-0.3 steps/s, same as IDLE -- and the v1.1 speed map measured that
day was CONTINUOUS from 0.20 (0.05 m/s) to 0.35 (0.25 m/s), not the 0.28->0.30
step of 2026-09-11. Re-measure before trusting either table.

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
from locomani.backends.sonic.agent_server import StreamMode
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
    if not rects:  # the walk scene: nothing to walk into, keep the start heading
        return 0.0, 9.0
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


def hold_upper(p, upper):
    """Carry a fixed 17-joint upper body (waist y/r/p, L/R-interleaved arms) through the legs, the way the
    crate expert's approach does; reset_pose() clears it, so it is set again after every reset."""
    if upper is not None:
        p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
        p.cmd.set_upper_body(upper)


def reset_and_face(p, reset, meta, crab_deg: float = 0.0, upper=None):
    """One clean start per leg: quiesce, teleport home, stand, turn to open
    floor. Without the quiesce the deploy carries the previous leg's intent
    through the reset.

    `crab_deg` is the angle between the movement vector and the FACING. The
    robot travels along the CLEARED heading either way -- the facing is turned
    back by that angle -- so a crab leg keeps a forward leg's obstacle
    clearance. Returns (travel heading, facing), both radians.
    """
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label="shuffle")
    time.sleep(0.3)
    p.cmd.start_control()
    hold_upper(p, upper)
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    s = p.obs()
    if s["fallen"]:
        return None
    th, clear = free_heading(meta, p.pelvis_xy(s))
    face = float(th - np.deg2rad(crab_deg))
    p.turn_to(face, tol=0.12, timeout=12.0)
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=(np.cos(face), np.sin(face), 0.0))
    return th, face


def leg(p, mode, speed, heading, move=True, face=None):
    """One 5 s leg. `heading` is where the robot is asked to TRAVEL, `face`
    where it is asked to look; they differ by the crab angle."""
    face = heading if face is None else face
    fv = (float(np.cos(face)), float(np.sin(face)), 0.0)
    mv = (float(np.cos(heading)), float(np.sin(heading)), 0.0) if move else (0.0, 0.0, 0.0)
    t0 = time.time()
    yaw_start = float(p.pelvis_yaw(p.obs()))
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
    # WHERE it went, not just how fast. `net` is a magnitude, so on a crab leg
    # it cannot tell "moved sideways" from "turned to face the movement vector
    # and walked forward" -- the first pass (2026-09-11) read 90 deg as FASTER
    # than forward, which is only possible if the robot reoriented. Resolve the
    # displacement in the frame the robot was FACING when the leg started, and
    # record how far the yaw travelled.
    disp = walk_end_xy - start_xy
    face_vec = np.array([np.cos(face), np.sin(face)])
    perp_vec = np.array([-np.sin(face), np.cos(face)])
    want = np.array([np.cos(heading), np.sin(heading)])
    travel = float(np.degrees(np.arctan2(disp[1], disp[0]) - np.arctan2(want[1], want[0])))
    travel = (travel + 180.0) % 360.0 - 180.0
    yaw_end = float(p.pelvis_yaw(s))

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
        fwd=float(disp @ face_vec) / (walk_end_t - start_t),
        lat=float(disp @ perp_vec) / (walk_end_t - start_t),
        travel_err_deg=travel,
        yaw_turned_deg=float(np.degrees((yaw_end - yaw_start + np.pi) % (2 * np.pi) - np.pi)),
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
    ap.add_argument(
        "--crab",
        type=str,
        default="0",
        help="angles (deg) between the movement vector and the facing. The tabletop approach's last "
        "0.3 m is a crab walk -- it steers at the planned stand while the facing stays on the "
        "approach heading -- and the deploy normalises the movement vector's magnitude away, so "
        "forward delivery does not imply lateral delivery (ledger bottle.md 24.5).",
    )
    ap.add_argument(
        "--march",
        type=str,
        default="",
        help="speeds to probe with a ZERO movement vector in SLOW_WALK (does the gait step in place?). "
        "Each is repeated --repeats times; net/lat then read as drift, steps/s as the answer.",
    )
    ap.add_argument(
        "--upper",
        default="",
        help="17 comma-separated upper-body joint targets held through every leg (waist y/r/p, then L/R-interleaved "
        "arms); default: the deploy's own arms. 2026-09-23: the warehouse walk ends with the arms prepared forward, "
        "and the stop carry measured with default arms did not transfer.",
    )
    ap.add_argument("--json", default=None, help="write every leg's numbers here")
    a = ap.parse_args()
    upper = [float(x) for x in a.upper.split(",")] if a.upper else None
    if upper is not None and len(upper) != 17:
        ap.error("--upper takes 17 values")
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
    crabs = [float(x) for x in a.crab.split(",")]
    if a.speeds:
        sps = [float(x) for x in a.speeds.split(",")]
        grid = [(LocomotionMode.SLOW_WALK, sp, True, cd) for cd in crabs for sp in sps for _ in range(a.repeats)]
    else:
        grid = [(LocomotionMode.SLOW_WALK, sp, True, 0.0) for sp in (0.18, 0.22, 0.26, 0.30, 0.40)]
        grid += [(LocomotionMode.SLOW_WALK, 0.30, False, 0.0), (LocomotionMode.WALK, 0.35, True, 0.0)]
    if a.march:
        grid += [(LocomotionMode.SLOW_WALK, float(x), False, 0.0) for x in a.march.split(",") for _ in range(a.repeats)]
    # The control: commanded IDLE from a standstill. Its `net` is the drift
    # rate the carry column has to be read against, not zero.
    grid += [(LocomotionMode.IDLE, 0.0, False, 0.0) for _ in range(max(1, a.repeats))]
    rows, consecutive_falls = [], 0
    print(f"\npolicy {paths.SONIC_POLICY}")
    print(
        f"\n{'mode':>10} {'speed':>6} {'crab':>5} {'move':>5} {'net m/s':>8} {'fwd':>8} {'lat':>8} "
        f"{'off deg':>7} {'turned':>7} {'steps/s':>7} {'cm/step':>8} {'carry cm':>9}"
    )
    for mode, sp, mv, crab in grid:
        # A fall costs its own leg, not the rest of the grid: every leg starts
        # from a full reset, so the next one is independent. Three in a row is
        # a broken stack, not a data point (the deleted version broke out of
        # the sweep on the first fall and lost every remaining speed).
        if consecutive_falls >= 3:
            print("three consecutive falls -- stopping, the stack is not standing the robot up")
            break
        placed = reset_and_face(p, reset, meta, crab_deg=crab, upper=upper)
        tag = f"{mode.name:>10} {sp:>6.2f} {crab:>5.0f} {str(mv):>5}"
        if placed is None:
            print(f"{tag}   FELL during the reset")
            rows.append(dict(mode=mode.name, speed=sp, crab_deg=crab, move=mv, fell="reset"))
            consecutive_falls += 1
            continue
        heading, face = placed
        r = leg(p, mode, sp, heading, move=mv, face=face)
        if r is None:
            print(f"{tag}   FELL")
            rows.append(dict(mode=mode.name, speed=sp, crab_deg=crab, move=mv, fell=True))
            consecutive_falls += 1
            continue
        consecutive_falls = 0
        print(
            f"{tag} {r['net']:>8.3f} {r['fwd']:>+8.3f} {r['lat']:>+8.3f} {r['travel_err_deg']:>+7.0f} "
            f"{r['yaw_turned_deg']:>+7.0f} {r['steps']:>7.2f} {r['per_step']:>8.1f} {r['carry']:>9.1f}"
        )
        rows.append(dict(mode=mode.name, speed=sp, crab_deg=crab, move=mv, fell=False, **r))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(policy=paths.SONIC_POLICY, upper=upper, leg_s=LEG, warmup_s=WARMUP, rows=rows), fh, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
