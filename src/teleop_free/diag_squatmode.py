"""Does LocomotionMode::IDEL_SQUAT give back the pelvis height the all-planner
architecture otherwise loses?

M3 showed the planner message's `height` field is inert (0.2 cm of authority)
and M4 showed the policy delivers ~3% of a commanded waist pitch, so in the
all-planner architecture there is no way left to lower the hand -- except the
k-planner's own squat gait, which the pipeline has never used (our
LocomotionMode enum only ever declared IDLE/SLOW_WALK/WALK/RUN, while the
deploy has 27 modes).

The scene's table needs ~10 cm: the can centre sits 0.106 m below the pelvis
and the reach core bottoms out at 0.037 m below it.

Measures the depth reached, how long the transition takes, whether the
17-joint upper-body target still applies while squatting, and whether the
robot comes back up cleanly.

    python src/teleop_free/diag_squatmode.py
Output: outputs/diag/squatmode.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stream_reach  # noqa: E402
from primitives import (  # noqa: E402
    LocomotionMode,
    Primitives,
    StreamMode,
    upper_body_from_mj,
)
from privileged_pub import EpisodeControlClient  # noqa: E402

SCENE_META = "/workspace/code/locomanipulation/outputs/scene/sim/scene_meta.json"
OUT = "/workspace/code/locomanipulation/outputs/diag/squatmode.json"
SQUAT = 4                        # LocomotionMode::IDEL_SQUAT in the deploy enum
PHASE_S = 6.0


def restand(p, reset, label):
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.set_upper_body_vel(None)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label=label)
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    stream_reach.reanchor(p, side="right")
    return p.obs()


def run_phase(p, mode_val, face, seconds, upper=None, height=-1.0):
    """Stream one mode and return the per-tick pelvis-z / palm-z trace."""
    trace = []
    end = time.time() + seconds
    while time.time() < end:
        if upper is not None:
            p.cmd.set_upper_body(upper)
        p.cmd.frame(mode=mode_val, facing=face, height=height,
                    movement=(0.0, 0.0, 0.0))
        time.sleep(0.02)
        st = p.obs()
        trace.append(dict(t=time.time(), z=float(st["pelvis"]["pos"][2]),
                          palm_z=float(st["palms"]["right"]["pos"][2]),
                          knee=[float(st["joints"][3]), float(st["joints"][9])],
                          fallen=bool(st["fallen"])))
    return trace


def summarise(tag, trace, z0):
    z = np.array([t["z"] for t in trace])
    lo = float(z.min())
    settled = float(np.median(z[-int(len(z) * 0.3):]))
    # time to reach 90% of the final excursion
    tt = None
    if abs(settled - z0) > 0.01:
        thr = z0 + 0.9 * (settled - z0)
        for t in trace:
            if (settled < z0 and t["z"] <= thr) or (settled > z0 and t["z"] >= thr):
                tt = t["t"] - trace[0]["t"]
                break
    fell = any(t["fallen"] for t in trace)
    print(f"[sq] {tag:22} pelvis {z0:.3f} -> {settled:.3f} "
          f"(drop {(z0 - settled) * 100:5.1f} cm, min {lo:.3f})  "
          f"t90 {('%.1f s' % tt) if tt else '  -  '}"
          f"{'  FELL' if fell else ''}")
    return dict(tag=tag, z0=z0, settled=settled, min=lo, t90=tt, fell=fell)


def sweep(p, reset, face, z_stand):
    """Calibrate height -> delivered squat depth inside IDEL_SQUAT."""
    rows = []
    for h in (-1.0, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.40, 0.20):
        st = p.obs()
        if st["fallen"]:
            st = restand(p, reset, "restand"); face = p._facing(st)
        p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
        upper = upper_body_from_mj(st["joints"])
        tr = run_phase(p, SQUAT, face, 5.0, upper=upper, height=h)
        r = summarise(f"height {h:+.2f}", tr, z_stand)
        r["height"] = h
        rows.append(r)
        run_phase(p, LocomotionMode.IDLE, face, 3.5, upper=upper)
    return rows


def main() -> None:
    meta = json.load(open(SCENE_META))
    p = Primitives(meta=meta, verbose=False)
    reset = EpisodeControlClient()
    s = restand(p, reset, "squatmode")
    if s["fallen"]:
        sys.exit("robot did not stand")
    face = p._facing(s)
    z_stand = float(s["pelvis"]["pos"][2])
    print(f"[sq] standing pelvis z {z_stand:.3f}; the scene needs ~0.10 m")

    if "--sweep" in sys.argv:
        rows = sweep(p, reset, face, z_stand)
        json.dump(rows, open(OUT.replace(".json", "_sweep.json"), "w"), indent=1)
        print("[sq] target: 0.10 m of drop -> pelvis 0.686")
        for r in rows:
            print(f"    height {r['height']:+.2f} -> drop "
                  f"{(r['z0'] - r['settled']) * 100:5.1f} cm")
        return

    out = []
    # 1. bare squat mode, no upper-body target
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    tr = run_phase(p, SQUAT, face, PHASE_S)
    out.append(summarise("squat, no upper", tr, z_stand))
    tr = run_phase(p, LocomotionMode.IDLE, face, PHASE_S)
    out.append(summarise("back to idle", tr, out[-1]["settled"]))

    # 2. squat mode WITH the frozen 17-joint upper-body target held
    s = p.obs()
    if s["fallen"]:
        s = restand(p, reset, "restand"); face = p._facing(s)
    z0 = float(s["pelvis"]["pos"][2])
    upper = upper_from = upper_body_from_mj(s["joints"])
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    tr = run_phase(p, SQUAT, face, PHASE_S, upper=upper)
    out.append(summarise("squat + frozen upper", tr, z0))
    arm_drift = float(np.max(np.abs(
        np.array(p.obs()["joints"])[15:29] - np.array(upper_from)[3:] * 0)) * 0)
    # how far did the arm joints move from their held target?
    st = p.obs()
    held = upper_body_from_mj(st["joints"])
    dev = float(np.max(np.abs(np.asarray(held) - np.asarray(upper))))
    print(f"[sq] upper-body target deviation while squatting: {dev:.3f} rad")
    out[-1]["upper_dev"] = dev
    tr = run_phase(p, LocomotionMode.IDLE, face, PHASE_S, upper=upper)
    out.append(summarise("back to idle (frozen)", tr, out[-1]["settled"]))

    # 3. squat with height also asked for, in case the two interact
    s = p.obs()
    if s["fallen"]:
        s = restand(p, reset, "restand"); face = p._facing(s)
    z0 = float(s["pelvis"]["pos"][2])
    tr = run_phase(p, SQUAT, face, PHASE_S, upper=upper, height=0.60)
    out.append(summarise("squat + height 0.60", tr, z0))
    run_phase(p, LocomotionMode.IDLE, face, 3.0, upper=upper)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[sq] wrote {OUT}")


if __name__ == "__main__":
    main()
