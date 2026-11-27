"""selfmodel_hacollect_20261126 (development, 2026-11-26): settle waits from hc_ha's --vtrace reset logs. Read only.

    python hc_vtrace.py DIR [DIR ...]     (DIR/data/w*_log.json written with --vtrace)

The trace holds per frame [settle speed, pelvis, right hand point, left hand point (m/s, displacement over 0.1 s), tested]
where `tested` marks the frames on which d3_collect's wait asked rig.quiet(); a wait is a run of tested frames (gaps
<= 10 frames), its open is the first tested frame minus SETTLE_F (25 frames). Quiet on a frame: the last 25 speeds all
below 2 cm/s. Per record: waits that turn quiet within 3 s / within 6 s of the open, by place (first wait of a reset
in place / after a walk / later waits); the time to quiet (median); for waits still moving at 3 s, the 0.5 s window max
per point at the 3 s mark (median) and which point is the fastest.
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np

FPS, SETTLE_F, V = 50, 25, 0.02


def waits(tr: list) -> list[tuple[int, int]]:
    t = [i for i, r in enumerate(tr) if r[-1]]
    out, s = [], None
    for j, i in enumerate(t):
        if s is None:
            s = i
        if j + 1 == len(t) or t[j + 1] - i > 10:
            out.append((s, i))
            s = None
    return out


def main() -> None:
    for d in sys.argv[1:]:
        R = []
        for f in sorted(glob.glob(f"{d}/data/w*_log.json")):
            R += json.load(open(f))
        R = [r for r in R if r.get("vtrace")]
        G = {"first in place": [], "first after walk": [], "later": []}
        for r in R:
            tr = r["vtrace"]
            v = np.array([np.inf if x[0] is None else x[0] for x in tr])
            P = np.array([[np.nan if u is None else u for u in x[1:4]] for x in tr], float)
            for j, (s, e) in enumerate(waits(tr)):
                o = s - SETTLE_F
                q = None
                for i in range(s, e + 1):
                    if i >= SETTLE_F - 1 and np.all(v[i - SETTLE_F + 1 : i + 1] < V):
                        q = i
                        break
                g = "later" if j > 0 else ("first after walk" if r.get("walked") else "first in place")
                m3 = o + 3 * FPS
                w3 = np.nanmax(P[max(m3 - SETTLE_F + 1, 0) : m3 + 1], axis=0) if m3 < len(P) else np.full(3, np.nan)
                G[g].append(dict(t=None if q is None else (q - o) / FPS, len=(e - o) / FPS, w3=w3))
        print(f"== {d}: {len(R)} resets with traces")
        for g, W in G.items():
            if not W:
                continue
            t = [w["t"] for w in W if w["t"] is not None]
            in3 = np.mean([w["t"] is not None and w["t"] <= 3.0 + 1e-9 for w in W])
            in6 = np.mean([w["t"] is not None for w in W])
            slow = [w["w3"] for w in W if (w["t"] is None or w["t"] > 3.0) and np.isfinite(w["w3"]).all()]
            msg = ""
            if slow:
                S = np.array(slow)
                top = np.bincount(np.argmax(S, 1), minlength=3)
                msg = (f"; moving at 3 s: {len(slow)}, 0.5 s max at 3 s median pelvis / right / left "
                       f"{np.round(100 * np.median(S, 0), 1).tolist()} cm/s, fastest {dict(zip(['pelvis', 'right', 'left'], top.tolist()))}")
            print(f"  {g:16s} waits {len(W):3d}: quiet by 3 s {100 * in3:5.1f} %, by the end (<= 6 s) {100 * in6:5.1f} %; "
                  f"time to quiet median {np.median(t) if t else float('nan'):.2f} s, p90 {np.percentile(t, 90) if t else float('nan'):.2f} s{msg}")


if __name__ == "__main__":
    main()
