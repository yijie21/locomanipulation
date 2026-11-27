"""selfmodel_xfer_20261125, read-only analysis written after the freeze (2026-11-25): how still the body is at the end
label, from a record's W rows (10 Hz): the 0.1 s displacement speeds of the pelvis and both hand points (13 cm ahead of
wrist_yaw_link) over the last 0.5 s before the label (5 intervals per segment, the settle rule's window on the record's
grid), and the settle timeouts (skipped slots before a command, end labels not settled).

    $VPY sx_speed.py DIR [DIR ...]          (one-hand data/ and two-hand data_both/ together; in-place resets only)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def logs(d: Path) -> list[dict]:
    out = []
    for p in sorted((d / "data").glob("w*_log.json")):
        out += json.loads(p.read_text())
    return out


def main() -> None:
    for d in map(Path, sys.argv[1:]):
        L = logs(d)
        L = [r for r in L if not r.get("walked")]
        slots = [s for r in L for s in r["slots"] if s["status"] != "not reached"]
        skipped = sum(s["status"] == "unsettled" for s in slots)
        V = {k: [] for k in ("pelvis", "acting", "other", "rule")}
        n_seg = n_set = 0
        for sub in ("data", "data_both"):
            p = d / sub / "all.npz"
            if not p.exists():
                continue
            z = np.load(p)
            n = z["n_steps"].astype(int)
            so = np.r_[0, np.cumsum(n)[:-1]]
            last = so + n - 1
            ok = (z["fell"] == 0) & (z["walked"] == 0) & (n >= 6)
            W = z["W"].astype(float)
            n_seg += int(ok.sum())
            n_set += int((ok & (z["settled_end"] > 0)).sum())
            for i in np.flatnonzero(ok):
                r = np.arange(last[i] - 5, last[i] + 1)
                P = {"pelvis": W[r, 0:3], "right": W[r, 4:7], "left": W[r, 7:10]}
                v = {k: 100 * np.linalg.norm(np.diff(x, axis=0), axis=1) / 0.1 for k, x in P.items()}
                act, oth = ("right", "left") if z["side"][i] > 0 else ("left", "right")
                V["pelvis"] += list(v["pelvis"])
                V["acting"] += list(v[act])
                V["other"] += list(v[oth])
                V["rule"] += list(np.max([v["pelvis"], v["right"], v["left"]], axis=0))
        q = {k: (float(np.median(x)), float(np.percentile(x, 90))) for k, x in V.items() if x}
        print(f"{d}: in-place resets {len(L)}, reaches tried {len(slots)}, skipped as unsettled before the command "
              f"{skipped} ({100 * skipped / max(len(slots), 1):.1f} %); segments (one- and two-hand, not fallen) {n_seg}, "
              f"end label not settled {n_seg - n_set} ({100 * (n_seg - n_set) / max(n_seg, 1):.1f} %)")  # fmt: skip
        print("  speed over the last 0.5 s before the end label (0.1 s displacements, cm/s), median / p90: "
              + "; ".join(f"{k} {m:.2f} / {p9:.2f}" for k, (m, p9) in q.items())
              + "  (rule = the largest of the three, the settle rule's quantity; threshold 2 cm/s)")


if __name__ == "__main__":
    main()
