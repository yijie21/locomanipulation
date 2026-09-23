"""Walk to a marker, brake, and measure where the robot actually stops -- per speed command, repeated.

Walk scene (G1 alone on the floor). Each trial: reset, stand, then SLOW_WALK straight along the start
heading; the tick the pelvis has advanced --marker metres, IDLE is commanded (facing held) and kept for
--settle simulated seconds. Recorded per trial: the pelvis at the brake tick, its speed there (from the
last 0.5 s), the settled pelvis, and the yaw. `carry` = settled minus brake point along the heading;
its spread is what a calibrated brake margin cannot remove.

The sim writes one qpos trace per trial (SIM_STATE_TRACE_DIR, label brake_<speed>_<i>) for the renderer.

    LOCOMANI_SIM_DIR=outputs/scene/walk python -m locomani.backends.sonic.diag_brake --out <dir>  (live stack)
Touch <out>/STOP to end after the current trial; results.json is rewritten after every trial.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from locomani import paths
from locomani.backends.sonic.primitives import DT, LocomotionMode, Primitives
from locomani.sim.privileged_pub import EpisodeControlClient

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


class FootWatch:
    """Planted-foot bookkeeping from the privileged state, by forward kinematics of the walk-scene model.

    A foot is planted while its ankle is within 1 cm of its standing height. A landing is the tick a foot
    becomes planted; the front planted ankle and the last stride (distance between consecutive landings)
    are what the guard brake predicts from.
    """

    def __init__(self):
        import mujoco

        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(paths.SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.qadr = [int(self.model.jnt_qposadr[self.model.joint(n).id]) for n in MUJOCO_JOINT_NAMES]
        self.feet = [self.model.body(n).id for n in FEET]
        self.stand_z = None
        self.planted = [True, True]
        self.landings = []
        self.last_strides = []

    def feet_now(self, s):
        q = self.data.qpos
        q[:3] = s["pelvis"]["pos"]
        q[3:7] = s["pelvis"]["quat"]
        q[self.qadr] = s["joints"]
        self.mujoco.mj_kinematics(self.model, self.data)
        return [self.data.xpos[b].copy() for b in self.feet]

    def update(self, s, x0, h):
        """Returns (landed this tick, front planted ankle along h, last stride or None)."""
        feet = self.feet_now(s)
        if self.stand_z is None:
            self.stand_z = [f[2] for f in feet]
        planted = [f[2] < z + 0.01 for f, z in zip(feet, self.stand_z)]
        landed = any(p and not was for p, was in zip(planted, self.planted))
        self.planted = planted
        along = [float((f[:2] - x0) @ h) for f in feet]
        if landed:
            self.landings.append(max(a for a, p in zip(along, planted) if p))
        front = max((a for a, p in zip(along, planted) if p), default=float("nan"))
        stride = self.landings[-1] - self.landings[-2] if len(self.landings) >= 2 else None
        self.last_strides = list(np.diff(self.landings[-3:]))  # the most recent (up to) two strides
        return landed, front, stride


SIM_HZ = 50  # privileged state publishes per simulated second; `seq` counts them


def stagger(watch, s, h):
    """Fore-aft distance between the two ankles along h (m, absolute)."""
    left, right = watch.feet_now(s)
    return abs(float((left[:2] - right[:2]) @ h))


def trial(
    p,
    reset,
    speed,
    marker,
    settle,
    label,
    final_speed=None,
    final_distance=0.0,
    guard=None,
    brake_delay=0.0,
    resquare=0.0,
    waypoint=None,
    timeout_s=40.0,
):
    """`guard` = dict(extra, margin, stride0): the marker is then a LINE the body must not cross. At every
    foot landing the stop after one more stride is predicted as front ankle + stride + extra (`extra` =
    worst observed reach of the robot's frontmost low-body point past the front planted ankle after IDLE),
    and IDLE is sent on the first landing where that would pass line - margin. Only on landings: an IDLE
    mid-swing is the one case (+19 cm) the foot model does not cover."""
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label=label)
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    s = p.obs()
    if s["fallen"]:
        return dict(fell="reset")
    yaw0 = float(p.pelvis_yaw(s))
    h = np.array([np.cos(yaw0), np.sin(yaw0)])
    n = np.array([-h[1], h[0]])
    fv = (h[0], h[1], 0.0)
    xy0 = p.pelvis_xy(s).copy()
    watch = FootWatch()
    reset.start_video(label)
    recent = deque(maxlen=SIM_HZ // 2)  # (seq, forward) over the last 0.5 simulated s
    brake, pending = None, None
    moved, still_since = False, None
    rounds, target = [], None if waypoint is None else (marker, 0.0, 0.0, waypoint["id"] * 10)
    try:
        first = first_seq = s["seq"]
        while True:
            s = p.obs()
            if s["fallen"]:
                return dict(fell="walk")
            fwd = float((p.pelvis_xy(s) - xy0) @ h)
            recent.append((s["seq"], fwd))
            trigger = fwd >= marker
            landed, front, stride = watch.update(s, xy0, h)
            if waypoint is not None:
                # WAYPOINT_TARGET_V1: the planner walks to the latched target itself; "brake" = the moment the
                # pelvis has been still for 1 s after moving (or the timeout), then the target is cleared
                (q0, f0), (q1, f1) = recent[0], recent[-1]
                v = abs(f1 - f0) * SIM_HZ / max(q1 - q0, 1)
                moved = moved or fwd > 0.03
                still_since = (s["seq"] if still_since is None else still_since) if (moved and v < 0.03) else None
                trigger = still_since is not None and s["seq"] - still_since >= SIM_HZ
                lat_now = float((p.pelvis_xy(s) - xy0) @ n)
                remaining = (marker - fwd, -lat_now)
                if trigger and len(rounds) < waypoint["rounds"] and abs(remaining[0]) > waypoint["tol"]:
                    # closed loop: the planner walks its REFERENCE to the target but the body follows only part of
                    # it, so re-issue what is still missing, measured on the body, as a new target id
                    rounds.append(dict(seq=s["seq"], fwd=fwd, lat=lat_now))
                    target = (remaining[0], remaining[1], 0.0, waypoint["id"] * 10 + len(rounds))
                    moved, still_since, trigger = False, None, False
                trigger = trigger or s["seq"] - first > waypoint["timeout"] * SIM_HZ
            if guard is not None:
                if guard.get("stride_mode") == "recent2":
                    # the larger of the last two strides; stride0 only until two strides exist. At 0.20-0.25 the
                    # strides run 2-22 cm, and the fixed 14 cm floor bound on every brake (2026-09-23)
                    strides = watch.last_strides
                    stride = max(strides) if len(strides) >= 2 else guard["stride0"]
                else:
                    stride = guard["stride0"] if stride is None else max(stride, guard["stride0"])
                if pending is None and landed and front + stride + guard["extra"] >= marker - guard["margin"]:
                    # C: the brake may be held back `brake_delay` simulated seconds after the landing that
                    # decided it, so the feet are further into the stride when IDLE arrives
                    pending = s["seq"] + int(brake_delay * SIM_HZ)
                trigger = pending is not None and s["seq"] >= pending
            if trigger:
                (q0, f0), (q1, f1) = recent[0], recent[-1]
                brake = dict(
                    seq=s["seq"],
                    fwd=fwd,
                    lat=float((p.pelvis_xy(s) - xy0) @ n),
                    speed=(f1 - f0) * SIM_HZ / max(q1 - q0, 1),
                )
                brake.update(front_ankle=front, stride=stride)
                break
            if waypoint is None and s["seq"] - first > timeout_s * SIM_HZ:
                return dict(fell=None, timeout=True, fwd=fwd)
            if waypoint is not None:
                p.cmd.frame(
                    mode=waypoint["mode"],
                    speed=speed if waypoint["mode"] != LocomotionMode.IDLE else -1.0,
                    movement=fv if waypoint["mode"] != LocomotionMode.IDLE else (0.0, 0.0, 0.0),
                    facing=fv,
                    target=target,
                )
                time.sleep(DT)
                continue
            slow_from = marker - final_distance - (0 if guard is None else guard["extra"] + guard["margin"])
            now = final_speed if final_speed is not None and fwd >= slow_from else speed
            p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=now, movement=fv, facing=fv)
            time.sleep(DT)
        while s["seq"] - brake["seq"] < settle * SIM_HZ:
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=fv)
            time.sleep(DT)
            s = p.obs()
            if s["fallen"]:
                return dict(fell="brake")
        extra = dict(stagger_settled=stagger(watch, s, h), pelvis_settled=float((p.pelvis_xy(s) - xy0) @ h))
        if resquare > 0:
            # B: a mode change is one of the three things that make a static mode replan (planner_onnx.md,
            # Replan Logic); SLOW_WALK with no movement for `resquare` s, then IDLE again, then settle
            first = s["seq"]
            while s["seq"] - first < resquare * SIM_HZ:
                p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=-1.0, movement=(0.0, 0.0, 0.0), facing=fv)
                time.sleep(DT)
                s = p.obs()
            first = s["seq"]
            while s["seq"] - first < settle * SIM_HZ:
                p.cmd.frame(mode=LocomotionMode.IDLE, facing=fv)
                time.sleep(DT)
                s = p.obs()
                if s["fallen"]:
                    return dict(fell="resquare")
    finally:
        reset.end_video()
    d = p.pelvis_xy(s) - xy0
    return dict(
        fell=None,
        origin_xy=xy0.tolist(),
        heading=yaw0,
        brake_fwd=brake["fwd"],
        brake_lat=brake["lat"],
        brake_speed=brake["speed"],
        stop_fwd=float(d @ h),
        stop_lat=float(d @ n),
        carry=float(d @ h) - brake["fwd"],
        stagger=stagger(watch, s, h),
        **extra,
        brake_front_ankle=brake.get("front_ankle"),
        seconds_to_still=(brake["seq"] - first_seq) / SIM_HZ,
        waypoint_rounds=rounds,
        brake_stride=brake.get("stride"),
        yaw_deg=float(np.degrees((p.pelvis_yaw(s) - yaw0 + np.pi) % (2 * np.pi) - np.pi)),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--speeds", default="0.55,0.35,0.25")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--marker", type=float, default=1.5, help="metres along the start heading where IDLE is commanded")
    ap.add_argument(
        "--final-speed",
        type=float,
        default=None,
        help="two-stage approach: drop to this speed command for the last --final-distance before the marker. "
        "2026-09-23: after IDLE the robot takes exactly one more step and settles ~8 cm past its planted front "
        "foot, so the stop is quantised by the step length; shorter final steps are the only way to shrink it",
    )
    ap.add_argument("--final-distance", type=float, default=0.3, help="metres before the marker to switch speed")
    ap.add_argument("--guard", action="store_true", help="treat --marker as a line the body must not cross")
    ap.add_argument(
        "--extra",
        type=float,
        default=0.33,
        help="guard: frontmost low-body reach past the planted front ankle after IDLE, worst case (m)",
    )
    ap.add_argument("--margin", type=float, default=0.03, help="guard: clearance kept before the line (m)")
    ap.add_argument(
        "--stride-mode",
        choices=("floor", "recent2"),
        default="floor",
        help="guard stride prediction: floor = max(last stride, stride0); recent2 = max of the last two strides",
    )
    ap.add_argument("--stride0", type=float, default=0.14, help="guard: stride assumed until two landings are seen (m)")
    ap.add_argument(
        "--brake-delay", type=float, default=0.0, help="guard: simulated s between the deciding landing and IDLE"
    )
    ap.add_argument(
        "--resquare", type=float, default=0.0, help="after settling: s of SLOW_WALK with no movement, then IDLE"
    )
    ap.add_argument(
        "--waypoint",
        choices=("slow", "idle"),
        default=None,
        help="WAYPOINT_TARGET_V1 deploy: send a root waypoint --marker metres ahead (heading kept) in SLOW_WALK "
        "(speed from --speeds) or IDLE, instead of walking and braking; the marker is then the target",
    )
    ap.add_argument("--waypoint-rounds", type=int, default=0, help="re-issue the measured remainder up to N times")
    ap.add_argument("--waypoint-tol", type=float, default=0.015, help="stop re-issuing once within this (m)")
    ap.add_argument("--waypoint-timeout", type=float, default=25.0, help="simulated s before giving up")
    ap.add_argument("--first", type=int, default=0, help="index of the first trial (labels continue a run)")
    ap.add_argument("--settle", type=float, default=3.0, help="simulated seconds of IDLE before the stop is read")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    rows, falls = [], 0
    print(f"{'speed':>6} {'i':>3} {'brake v':>8} {'carry cm':>9} {'stop cm':>8} {'lat cm':>7} {'yaw':>6}")
    for speed in [float(x) for x in a.speeds.split(",")]:
        for i in range(a.first, a.first + a.repeats):
            if (a.out / "STOP").exists():
                break
            if falls >= 3:
                # A deploy that lost LowState leaves every later reset "fallen"; 33 junk trials came of
                # not stopping here (2026-09-23, a vision worker started on GPU0 mid-run).
                print("three consecutive falls -- stopping, the stack is not standing the robot up", flush=True)
                break
            tag = f"{speed:.2f}" if a.final_speed is None else f"{speed:.2f}to{a.final_speed:.2f}"
            label = f"brake_{tag}_{i:02d}"
            r = dict(speed=speed, final_speed=a.final_speed, final_distance=a.final_distance, i=i, label=label)
            guard = (
                dict(extra=a.extra, margin=a.margin, stride0=a.stride0, stride_mode=a.stride_mode) if a.guard else None
            )
            r.update(
                trial(
                    p,
                    reset,
                    speed,
                    a.marker,
                    a.settle,
                    label,
                    a.final_speed,
                    a.final_distance,
                    guard,
                    a.brake_delay,
                    a.resquare,
                    None
                    if a.waypoint is None
                    else dict(
                        mode=LocomotionMode.SLOW_WALK if a.waypoint == "slow" else LocomotionMode.IDLE,
                        id=int(time.time()) % 10_000 * 100 + i,  # x10 per round, still exact in f32
                        timeout=a.waypoint_timeout,
                        rounds=a.waypoint_rounds,
                        tol=a.waypoint_tol,
                    ),
                )
            )
            rows.append(r)
            falls = falls + 1 if r.get("fell") else 0
            if r.get("fell") is None and "carry" in r:
                print(
                    f"{speed:>6.2f} {i:>3d} {r['brake_speed']:>8.2f} {100 * r['carry']:>+9.1f} "
                    f"{100 * (r['stop_fwd'] - a.marker):>+8.1f} {100 * r['stop_lat']:>+7.1f} {r['yaw_deg']:>+6.1f}"
                    f"  stagger {100 * r['stagger_settled']:4.1f}->{100 * r['stagger']:4.1f} cm",
                    flush=True,
                )
            else:
                print(f"{speed:>6.2f} {i:>3d}  {r}", flush=True)
            (a.out / "results.json").write_text(
                json.dumps(
                    dict(policy=paths.SONIC_POLICY, marker=a.marker, settle_s=a.settle, guard=guard, rows=rows),
                    indent=1,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
