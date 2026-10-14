"""How the hands get from the walk's hang to the pregrasp point (user, 2026-10-14: natural motion; visual.reach_route).

Per round, from `settle` (the walk is over) to the first of `align` / `descend` (the hands are over the rim), on the
measured wrists (the mean of the two, world frame, from the recorded joints and pelvis):
* approach_s -- the window's length;
* segments -- separate movements in the wrists' mean speed (smoothed over 0.1 s): one, plus one more each time the
  speed falls under 20% of its peak and rises again over 50% of it (lift in place, stop, reach = 2; one sweep = 1);
  a fixed slow-speed threshold read the start of a smoothstep ramp as a pause (2026-10-14);
* peak_speed (m/s); path_m -- the wrists' mean path length;
* contacts -- ticks with a hand or an arm on the box or anything on the platform (warehouse_measure.Scene), and in
  how many rounds.
Then the round's outcome. Reads round directories; prints one line per directory and writes a JSON.

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/scalebfm_warehouse_rsi_20261014/approach_motion.py \\
        DIR [DIR ...] --out approach.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "scripts" / "experiments" / "controller_compare_20261013"), str(REPO / "src")]

from warehouse_measure import Round, Scene  # noqa: E402

WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
LOW, HIGH = 0.2, 0.5  # fractions of the peak speed that end and start a segment
SMOOTH = 5  # ticks (0.1 s)


def approach(rd: Round) -> dict | None:
    ev = rd.ev
    end = min((ev[k] for k in ("align", "descend") if k in ev), default=None)
    if "settle" not in ev or end is None:
        return None
    i0, i1 = int(np.searchsorted(rd.t, ev["settle"])), int(np.searchsorted(rd.t, end))
    if i1 - i0 < 3:
        return None
    w = np.mean([rd.body(n)[i0:i1, :3] for n in WRISTS], axis=0)
    dt = np.diff(rd.t[i0:i1])
    speed = np.linalg.norm(np.diff(w, axis=0), axis=1) / np.maximum(dt, 1e-6)
    smooth = np.convolve(speed, np.ones(SMOOTH) / SMOOTH, mode="same")
    peak = float(smooth.max())
    segments, armed = 0, True
    for v in smooth:
        if armed and v > HIGH * peak:
            segments, armed = segments + 1, False
        elif not armed and v < LOW * peak:
            armed = True
    hits = 0
    for i in range(i0, i1):
        pairs, touch = rd.sc.contacts(rd.z["qpos"][i])
        hits += bool(touch[0] or touch[1] or pairs & {"arm_box", "platform"})
    return dict(
        approach_s=float(rd.t[i1] - rd.t[i0]),
        segments=segments,
        peak_speed=float(speed.max()),
        path_m=float(np.sum(np.linalg.norm(np.diff(w, axis=0), axis=1))),
        contact_ticks=hits,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    sc = Scene()
    out = {}
    for d in a.dirs:
        rows = {}
        for npz in sorted(d.glob("bfm_*.npz")):
            rd = Round(npz, sc)
            m = approach(rd)
            if m is not None:
                rows[npz.stem] = dict(m, ok=bool(rd.f["ok"]), why=rd.f.get("expert_reason", ""))
        med = lambda k: round(st.median(r[k] for r in rows.values()), 3) if rows else None  # noqa: E731
        summary = dict(
            rounds=len(rows),
            ok=sum(r["ok"] for r in rows.values()),
            approach_s=med("approach_s"),
            segments_mean=round(float(np.mean([r["segments"] for r in rows.values()])), 2) if rows else None,
            one_segment=sum(r["segments"] == 1 for r in rows.values()),
            peak_speed=med("peak_speed"),
            path_m=med("path_m"),
            rounds_with_contact=sum(r["contact_ticks"] > 0 for r in rows.values()),
        )
        out[str(d)] = dict(summary=summary, rounds=rows)
        print(d, json.dumps(summary))
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
