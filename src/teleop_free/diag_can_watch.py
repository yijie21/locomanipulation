"""Bystander: sample the privileged stream and log the can's life story.

Run alongside a gym round. Prints a line whenever the can moves, tilts,
or anything touches it — the missing timeline for topples that happen
between servo status prints.

Usage: HAND=dex1 python src/teleop_free/diag_can_watch.py [seconds]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from privileged_pub import PrivilegedSubscriber  # noqa: E402


def main() -> None:
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    sub = PrivilegedSubscriber()
    t_end = time.time() + dur
    last = None
    p0 = None
    while time.time() < t_end:
        s = sub.latest(300)
        if s is None or "can" not in s.get("objects", {}):
            continue
        o = s["objects"]["can"]
        pos = np.asarray(o["pos"])
        q = o["quat"]
        # rot[2,2] from quat: 1-2(qx^2+qy^2)
        up = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
        tl = o.get("touch_links") or []
        ft = s.get("furniture_touch") or []
        if p0 is None:
            p0 = pos.copy()
        moved = float(np.linalg.norm(pos - p0))
        key = (round(moved, 2), round(up, 1), tuple(tl), tuple(ft))
        if key != last:
            print(f"t={s['t']:.2f} moved={moved * 100:5.1f}cm up={up:+.2f} touch={tl} furn={ft}", flush=True)
            last = key
        time.sleep(0.1)


if __name__ == "__main__":
    main()
