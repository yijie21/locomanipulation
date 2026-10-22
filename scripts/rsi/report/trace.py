"""TODO 32.6: a round's 50 Hz trace as a time series -- per sample: the primitive running, the pelvis (xy, yaw, tilt),
each wrist's commanded vs measured position and the error, the jaw opening, the finger contacts, and each traced
object's height change and its offset from the nearer wrist (creep in the hand shows as that offset drifting). Step
boundaries from the round's JSON are marked. What 36 of 44 Reviewers rebuilt by hand (`trace.py`, `creep.py`, `fk.py`).

    python3 scripts/rsi/report/trace.py <rounds dir> <label> [--from 12.5] [--to 20] [--every 0.2] [--obj Mug_2]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def yaw_tilt(q) -> tuple[float, float]:
    w, x, y, z = (float(v) for v in q)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    up_z = 1 - 2 * (x * x + y * y)  # world z of the body's z axis
    return math.degrees(yaw), math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("label")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=1e9)
    ap.add_argument("--every", type=float, default=0.5, help="seconds between printed samples")
    ap.add_argument("--obj", default="", help="only this object (key or body name)")
    a = ap.parse_args(argv)
    z = np.load(a.dir / f"{a.label}.npz")
    rec = json.loads((a.dir / f"{a.label}.json").read_text())
    t = z["t"]
    names = [str(x) for x in z["primitives"]] if "primitives" in z.files else []
    objs = [k for k in z.files if k.startswith("obj_") and (not a.obj or a.obj in k)]
    print(
        f"{a.label}: {'OK' if rec.get('ok') else 'FAIL'} t_end {rec.get('t_end')}  {rec.get('plan_why') or rec.get('why')}"
    )
    marks = sorted((s.get("t0", 0.0), s["step"], s.get("ok", True)) for s in rec.get("steps") or [])
    start = {k: z[k][0, :3] for k in objs}
    cols = "t      prim     pelvis xy        yaw   tilt | L wrist err cm  jaw cm | R wrist err cm  jaw cm | fingers"
    print(cols + "".join(f" | {k[4:]:.18s} dz cm, off from wrist cm" for k in objs))
    nxt, mi = -1e9, 0
    for i in range(len(t)):
        if t[i] < a.t0 or t[i] > a.t1:
            continue
        while mi < len(marks) and marks[mi][0] <= t[i]:
            print(f"--- {marks[mi][0]:.2f} s: step {marks[mi][1]}{'' if marks[mi][2] else ' (failed)'}")
            mi += 1
        if t[i] < nxt and i != len(t) - 1:
            continue
        nxt = t[i] + a.every
        r = z["robot"][i]
        yaw, tilt = yaw_tilt(r[3:7])
        p = int(z["primitive"][i]) if "primitive" in z.files else -1
        prim = names[p] if 0 <= p < len(names) else str(p)
        line = f"{t[i]:6.2f} {prim:8.8s} ({r[0]:+.3f},{r[1]:+.3f}) {yaw:+6.1f} {tilt:5.1f}"
        for s in range(2):
            err = 100 * np.linalg.norm(z["wrists"][i, s] - z["ref_wrists"][i, s])
            line += f" | {err:5.1f} {100 * float(z['jaw'][i, s]):12.1f}"
        line += " | " + "".join(str(int(v)) for v in z["fingers"][i])
        for k in objs:
            o = z[k][i, :3]
            d = [np.linalg.norm(o - z["wrists"][i, s]) for s in range(2)]
            s = int(np.argmin(d))
            off = 100 * (o - z["wrists"][i, s])
            line += f" | {100 * (o[2] - start[k][2]):+6.1f} {'LR'[s]} ({off[0]:+.1f},{off[1]:+.1f},{off[2]:+.1f})"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
