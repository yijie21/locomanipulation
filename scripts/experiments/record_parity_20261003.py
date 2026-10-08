"""Does `plan_episode --record` produce the dataset the bottle collection line does? (TODO 19.4 acceptance)

    PYTHONPATH=src $VENV_SIM_PY scripts/experiments/record_parity_20261003.py \\
        --plan-run outputs/runs/sd_record_bottle_plan_20261003 \\
        --ref-run outputs/runs/bottle_simple_guard_lifted_20260925 --out <dir>

Converts both runs' sessions with today's converter into `<out>/{plan,ref}` (the reference run directory is
not touched), then compares field by field: the feature schema (names, dtypes, shapes), fps, the per-episode
task strings, episode lengths, the verification trims, and per-dimension value ranges of every numeric
feature. Writes `<out>/parity.json` and prints the differences. Same converter, same expert config: what
differs is the recording path (and, by design, the plan runner's spawn: at the planned stand, no jitter).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]


def convert(run: Path, out: Path) -> Path:
    from locomani.pipeline import run_convert

    entry = run_convert.collected(run)
    dataset = out
    if dataset.exists():
        return dataset
    argv = [
        str(a)
        for a in run_convert.convert_argv(entry, {"train": {"holdout": 0}}, dataset, out.parent / f"{out.name}_val")
    ]
    subprocess.run(argv, check=True, env=run_convert.CHILD_ENV)
    return dataset


def describe(ds: Path) -> dict:
    info = json.loads((ds / "meta/info.json").read_text())
    tasks = [json.loads(line)["task"] for line in (ds / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
    lengths, ranges = [], {}
    for f in sorted((ds / "data").rglob("episode_*.parquet")):
        t = pq.read_table(f)
        lengths.append(t.num_rows)
        for name in t.column_names:
            col = t.column(name).to_pylist()
            arr = np.asarray(col, dtype=float) if col and not isinstance(col[0], (str, dict)) else None
            if arr is None or arr.size == 0:
                continue
            arr = arr.reshape(len(col), -1)
            lo, hi = arr.min(axis=0), arr.max(axis=0)
            if name in ranges:
                ranges[name] = (np.minimum(ranges[name][0], lo), np.maximum(ranges[name][1], hi))
            else:
                ranges[name] = (lo, hi)
    trims = ds / "meta/verification_trims.jsonl"
    return {
        "fps": info.get("fps"),
        "features": {k: {"dtype": v.get("dtype"), "shape": v.get("shape")} for k, v in info["features"].items()},
        "episodes": len(lengths),
        "tasks": tasks,
        "length_s": [round(n / info["fps"], 1) for n in lengths],
        "trims": sum(1 for _ in trims.open()) if trims.exists() else 0,
        "ranges": {k: [lo.round(3).tolist(), hi.round(3).tolist()] for k, (lo, hi) in ranges.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-run", type=Path, required=True)
    ap.add_argument("--ref-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = describe(convert(REPO / args.plan_run, args.out / "plan"))
    ref = describe(convert(REPO / args.ref_run, args.out / "ref"))
    diff = {
        "fps": [plan["fps"], ref["fps"]],
        "features_only_plan": sorted(set(plan["features"]) - set(ref["features"])),
        "features_only_ref": sorted(set(ref["features"]) - set(plan["features"])),
        "features_differ": sorted(
            k for k in set(plan["features"]) & set(ref["features"]) if plan["features"][k] != ref["features"][k]
        ),
        "tasks": [sorted(set(plan["tasks"])), sorted(set(ref["tasks"]))],
        "episodes": [plan["episodes"], ref["episodes"]],
        "trims": [plan["trims"], ref["trims"]],
        "length_s_median": [float(np.median(plan["length_s"])), float(np.median(ref["length_s"]))],
        "length_s_range": [
            [min(plan["length_s"]), max(plan["length_s"])],
            [min(ref["length_s"]), max(ref["length_s"])],
        ],
    }
    widths = {}
    for k in set(plan["ranges"]) & set(ref["ranges"]):
        p_lo, p_hi = (np.asarray(v) for v in plan["ranges"][k])
        r_lo, r_hi = (np.asarray(v) for v in ref["ranges"][k])
        widths[k] = round(float(np.max(np.abs(p_lo - r_lo) + np.abs(p_hi - r_hi))), 3)
    diff["range_shift_max"] = dict(sorted(widths.items(), key=lambda kv: -kv[1]))
    (args.out / "parity.json").write_text(json.dumps({"diff": diff, "plan": plan, "ref": ref}, indent=1))
    print(json.dumps({k: v for k, v in diff.items() if k != "range_shift_max"}, indent=1))
    print("largest per-feature range shift (sum of |lo| and |hi| moves, worst dimension):")
    for k, v in list(diff["range_shift_max"].items())[:12]:
        print(f"  {k:48s} {v}")


if __name__ == "__main__":
    main()
