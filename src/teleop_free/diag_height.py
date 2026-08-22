"""M3: is the planner message's `height` field actually honoured?

An earlier probe concluded it was inert, but `height` is a real input
tensor of the k-planner ONNX model (localmotion_kplanner_onnx.hpp:268,
default 0.788740, -1 = "use the mode default"), so "inert" was probably a
value-range or stream-mode artefact rather than a missing wire.

It matters now: with the arm driven through the planner channel the legs
belong entirely to the k-planner, so `height` is the ONLY pelvis-height
knob left. A working one is the fallback for low objects if waist lean
alone cannot cover the table.

Sweeps `height` under both stream modes (VR-3pt drives encoder mode 1,
frozen-upper-body drives mode 0) and both a static and a stepping gait,
since any of those could gate it. Unit is unknown, so the sweep spans
[0, 1], which covers both the "absolute metres" reading (default 0.79)
and a normalised one.

    python src/teleop_free/diag_height.py
Output: outputs/diag/height.json
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
import paths

SCENE_META = f"{paths.REPO}/outputs/scene/sim/scene_meta.json"
OUT = f"{paths.REPO}/outputs/diag/height.json"
HEIGHTS = [-1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0]
HOLD_S = 5.0
KNEE_MJ = (3, 9)                 # left / right knee in MuJoCo joint order


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


def main() -> None:
    meta = json.load(open(SCENE_META))
    p = Primitives(meta=meta, verbose=False)
    reset = EpisodeControlClient()
    s = restand(p, reset, "height")
    if s["fallen"]:
        sys.exit("robot did not stand")
    print(f"[h] baseline pelvis z {s['pelvis']['pos'][2]:.3f}")

    out = []
    for sm in (StreamMode.PLANNER_VR_3PT, StreamMode.PLANNER_FROZEN_UPPER_BODY):
        for loco in (LocomotionMode.IDLE, LocomotionMode.SLOW_WALK):
            for h in HEIGHTS:
                s = p.obs()
                if s["fallen"]:
                    s = restand(p, reset, "restand")
                    print("[h] re-stood")
                p.cmd.stream_mode = sm
                if sm == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                    p.cmd.set_upper_body(upper_body_from_mj(s["joints"]))
                else:
                    p.cmd.upper_body = None
                face = p._facing(s)
                # settle at this height, then average the last second
                t_end = time.time() + HOLD_S
                zs, ks, ps = [], [], []
                while time.time() < t_end:
                    p.cmd.frame(mode=loco, facing=face, height=h,
                                movement=(0.0, 0.0, 0.0))
                    time.sleep(0.02)
                    if t_end - time.time() < 1.0:
                        st = p.obs()
                        zs.append(st["pelvis"]["pos"][2])
                        ks.append([st["joints"][i] for i in KNEE_MJ])
                        ps.append(st["palms"]["right"]["pos"][2])
                st = p.obs()
                rec = dict(stream_mode=sm.name, loco=loco.name, height=h,
                           pelvis_z=float(np.mean(zs)) if zs else None,
                           knee=list(map(float, np.mean(ks, axis=0))) if ks else None,
                           palm_z=float(np.mean(ps)) if ps else None,
                           fallen=bool(st["fallen"]))
                out.append(rec)
                print(f"[h] {sm.name:26} {loco.name:9} h={h:5.2f}  "
                      f"pelvis {rec['pelvis_z']:.3f}  "
                      f"knee {rec['knee'][0]:+.2f}/{rec['knee'][1]:+.2f}  "
                      f"palm {rec['palm_z']:.3f}"
                      f"{'  FELL' if rec['fallen'] else ''}")
            # back to default before changing regime
            s = p.obs()
            p.cmd.hold(2.0, mode=LocomotionMode.IDLE, height=-1.0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    base = [r for r in out if r["height"] == -1.0]
    print(f"[h] wrote {OUT}")
    for r in base:
        print(f"    default ({r['stream_mode']}, {r['loco']}): "
              f"pelvis {r['pelvis_z']:.3f}")
    span = {}
    for r in out:
        key = (r["stream_mode"], r["loco"])
        span.setdefault(key, []).append(r["pelvis_z"])
    for key, v in span.items():
        print(f"    {key}: pelvis range {min(v):.3f}..{max(v):.3f} "
              f"({(max(v) - min(v)) * 100:.1f} cm of authority)")


if __name__ == "__main__":
    main()
