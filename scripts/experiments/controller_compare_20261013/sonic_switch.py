"""S1, SONIC side (TODO 25.4, plan §4.4): the interface switch transient in SONIC's own L1 rounds.

The warehouse expert hands the arms from the planner to the streamed channel at `handoff` (robot standing at the
edge); the bottle expert takes them on the streamed channel at `ready` (stopped at the table; its `handback` is
followed at once by the carry walk, so it is not read). Read from the 25.2 / 25.3 state
files over +-0.5 s around the event: the pelvis displacement after it and its peak speed, the wrists' displacement
after it (FK), and the feet's (a foot that moves while it stays below 4 cm counts as slide), falls.

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/controller_compare_20261013/sonic_switch.py \\
        --out outputs/controller_compare_20261013/switch/sonic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from bottle_states import Fk  # noqa: E402

OUT = REPO / "outputs" / "controller_compare_20261013"
SETS = {
    "warehouse_handoff": (
        OUT / "warehouse" / "sonic" / "states",
        "handoff",
        REPO / "outputs" / "scene" / "warehouse_box_45x32x33_mars",
    ),
    "bottle_ready": (OUT / "bottle" / "sonic" / "states", "ready", REPO / "outputs" / "scene" / "sim_lowcom"),
}
WIN = 0.5
WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def one(z, t_event, fk):
    t = z["t"]
    i_s = int(np.searchsorted(t, t_event))
    i0, i1 = int(np.searchsorted(t, t_event - WIN)), int(np.searchsorted(t, t_event + WIN))
    i1 = min(i1, len(t) - 1)
    if i_s >= len(t) - 1 or i0 >= i_s:
        return None
    pel = z["pelvis"]
    pos = {}
    for i in range(i0, i1 + 1):
        fk.put(pel[i], z["joints"][i])
        pos[i] = {b: fk.pose(b)[:3] for b in (*WRISTS, *FEET)}
    vel = np.linalg.norm(np.diff(pel[i0 : i1 + 1, :3], axis=0), axis=1) / np.diff(t[i0 : i1 + 1])
    slide = 0.0
    for f in FEET:
        low = [i for i in range(i_s, i1 + 1) if pos[i][f][2] < 0.04]
        if len(low) > 1:
            slide = max(slide, float(np.linalg.norm(pos[low[-1]][f][:2] - pos[low[0]][f][:2])))
    return dict(
        fell=bool(pel[i0 : i1 + 1, 2].min() < 0.40),
        pelvis_move_cm=100 * float(np.linalg.norm(pel[i1, :3] - pel[i_s, :3])),
        pelvis_speed_max=float(vel.max()),
        wrist_move_cm=100 * float(max(np.linalg.norm(pos[i1][w] - pos[i_s][w]) for w in WRISTS)),
        foot_slide_cm=100 * slide,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT / "switch" / "sonic.json")
    a = ap.parse_args(argv)
    res = {}
    for name, (d, event, scene) in SETS.items():
        fk = Fk(scene)
        rows = []
        for p in sorted(d.glob("*.npz")):
            f = json.loads(p.with_suffix(".json").read_text())
            if event not in f.get("events", {}):
                continue
            r = one(dict(np.load(p)), f["events"][event], fk)
            if r:
                rows.append(dict(r, round=p.stem))
        keys = ("pelvis_move_cm", "pelvis_speed_max", "wrist_move_cm", "foot_slide_cm")
        res[name] = dict(
            n=len(rows),
            falls=sum(r["fell"] for r in rows),
            **{
                k: [round(float(np.mean([r[k] for r in rows])), 2), round(float(np.max([r[k] for r in rows])), 2)]
                for k in keys
                if rows
            },
            rows=rows,
        )
        print(name, {k: v for k, v in res[name].items() if k != "rows"})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
