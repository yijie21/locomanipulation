"""selfmodel_hadata_20261126 revision 2a (2026-11-26): the forward range of the training data's reach commands, from the
plan generators alone (d3_collect.reset_plan / reset_plan_b with the formal collection's seeds; no collected data is
read), against the real episodes' command targets (hr_real logs: tgt_h, the hand-point target in the pelvis heading
frame, and in_table).

Training side: every drawn standing command (pelvis_z None) of data3 (3000 resets from seed 89,100,000) and data3b (1500
from 89,600,000, b_config of the collection): data3's targets, data3b's detour targets and pre / entry pairs' pre-grasp
and grasp points (rel slots are relative moves: no absolute target). Per side: x percentiles (p0.5, p5, p50, p95,
p99.5). Real side: x per reach, the share outside the drawn [p0.5, p99.5] of its side, the share outside data3's
standing reach table at its height (in_table), per controller and task.

    python hr_train_range.py --b-config B.json REAL_DIR [REAL_DIR ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfmodel_data3_20261125"))
import d3_collect as D3  # noqa: E402


def drawn(b_config: Path) -> dict:
    table = json.loads(Path(D3.TABLE).read_text())
    cfg = D3.b_config(b_config)
    xs = {"right": [], "left": []}
    for i in range(3000):
        plan = D3.reset_plan(np.random.default_rng(89_100_000 + i), table, True)
        for c in plan["cmds"]:
            if c["pelvis_z"] is None:
                for s, hd in c["hands"].items():
                    xs[s].append(float(hd["target"][0]))
    for i in range(1500):
        plan = D3.reset_plan_b(np.random.default_rng(89_600_000 + i), table, True, cfg)
        for sl in plan["slots"]:
            if sl["pelvis_z"] is not None:
                continue
            for s, hd in sl["hands"].items():
                if sl["kind"] == "detour":
                    xs[s].append(float(hd["target"][0]))
                elif sl["kind"] == "pre":
                    xs[s].append(float(hd["pre"][0]))
                elif sl["kind"] == "entry":
                    xs[s].append(float(hd["grasp"][0]))
    return {s: np.array(v) for s, v in xs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-config", type=Path, required=True)
    ap.add_argument("real", nargs="*", type=Path)
    a = ap.parse_args()
    X = drawn(a.b_config)
    band = {}
    for s, v in X.items():
        q = np.percentile(v, [0.5, 5, 50, 95, 99.5])
        band[s] = (q[0], q[-1])
        print(f"training draws, standing, {s}: n {len(v)}; x p0.5/p5/p50/p95/p99.5 " + "/".join(f"{t:.3f}" for t in q) + " m")
    for d in a.real:
        xs, out, tab, n = [], 0, 0, 0
        for f in sorted(glob.glob(str(d / "log" / "*.json"))):
            for r in json.loads(Path(f).read_text()).get("reaches", []):
                x = float(r["tgt_h"][0])
                lo, hi = band[r["side"]]
                xs.append(x)
                out += not (lo <= x <= hi)
                tab += not bool(r.get("in_table"))
                n += 1
        if n:
            q = np.percentile(xs, [0, 50, 100])
            print(f"{d.name}: reaches {n}; x min/p50/max {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f} m; outside the drawn x band {out}/{n}; "
                  f"outside data3's standing table {tab}/{n}")  # fmt: skip
        else:
            print(f"{d.name}: no reaches")


if __name__ == "__main__":
    main()
