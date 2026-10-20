"""TODO 27.11 (plan docs/handoff/scene-rsi-plan-20261016.md §13): how many tasks the scene RSI loop has made work. Per
goal, its measured rounds under the catalog as it is now are counted. These are every registered family's sealed
pool at its last regress tag (regress.py), pooled when several families ran the same goal, and the calibration
samples' rounds; no family or sample is left out. A goal is `works` at >= BUCKET1_RATE (the checker's bucket-1
line), `low` when it has a success but stays below that (user 2026-10-19: these count as cases the loop got, marked
as low-rate), else `none`.

The seed column is the family's own `seed` tag, paired by start, so it is the catalog that family started from (later
families start after earlier write-backs). A calibration sample has no seed. The very first catalog (pick_single's
seed) won 0 of 80.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/successes.py

Writes outputs/scene_rsi_20261016/outer/successes.json and prints the counts and the works / low lists.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import batch as B  # noqa: E402
import regress as R  # noqa: E402
from calibrate import BUCKET1_RATE, kind, norm  # noqa: E402
from gate import wilson  # noqa: E402

OUT = R.REGISTRY.parent / "successes.json"


def tasks_of(family: str) -> dict:
    fam = json.loads((HERE / "families" / f"{family}.json").read_text())
    rows = [json.loads(x) for x in (REPO / fam["tasks_file"]).read_text().splitlines() if x.strip()]
    return {r["id"]: r for r in rows}


def goals() -> dict:
    """{goal: row}: pooled rounds now (and at the family's seed) over every registered family and sample."""
    out, reg = {}, R.load()
    for family, row in sorted(reg.items()):
        tasks = tasks_of(family)
        now = B.outcomes(R.rounds_dir(family, row, row["last"]))
        seed = B.outcomes(R.rounds_dir(family, row, "seed")) if row["kind"] != "calibration" else {}
        for label, rec in now.items():
            t = tasks[rec["task"]]
            g = out.setdefault(norm(t["goal"]), dict(goal=list(norm(t["goal"])), kind=kind(t["goal"]), ok=0, n=0,
                                                     seed_ok=0, seed_n=0, runs=[]))  # fmt: skip
            g["ok"] += bool(rec["ok"])
            g["n"] += 1
            if label in seed:
                g["seed_ok"] += bool(seed[label]["ok"])
                g["seed_n"] += 1
            if (family, rec["task"]) not in g["runs"]:
                g["runs"].append((family, rec["task"]))
    for g in out.values():
        rate = g["ok"] / g["n"]
        g.update(rate=round(rate, 3), ci=wilson(g["ok"], g["n"]),
                 status="works" if rate >= BUCKET1_RATE else "low" if g["ok"] else "none",
                 trained=any(reg[f]["kind"] != "calibration" for f, _ in g["runs"]))  # fmt: skip
    return out


def main() -> int:
    gs = goals()
    rows = sorted(
        gs.values(), key=lambda g: (not g["trained"], {"works": 0, "low": 1, "none": 2}[g["status"]], -g["rate"])
    )
    counts = collections.Counter((g["trained"], g["status"]) for g in rows)
    summary = {("trained" if t else "untrained"): {s: counts[(t, s)] for s in ("works", "low", "none")} for t in (1, 0)}
    OUT.write_text(json.dumps(dict(bucket1_rate=BUCKET1_RATE, summary=summary, goals=rows), indent=1))
    print(json.dumps(summary))
    for g in rows:
        if g["status"] != "none":
            seed = f"seed {g['seed_ok']}/{g['seed_n']}  " if g["seed_n"] else ""
            print(f"{'trained' if g['trained'] else 'untrained':9s} {g['status']:5s} {g['ok']:3d}/{g['n']:<3d} {seed}"
                  f"{g['kind']:12s} {' + '.join(t for _, t in g['runs'])}")  # fmt: skip
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
