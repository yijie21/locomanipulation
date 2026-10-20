"""TODO 28.3: scene tasks under AMO vs ScaleBFM, same starts, same skills (the current catalog), paired by start.

For each (family, task) in TASKS, the first N starts of that task in the family's sealed pool run twice through
batch.py / scene_round.py with --npz: once with --controller amo, once with --controller scalebfm (deterministic; its
outcomes must equal the registry's `colliders_20261019` rounds, which this checks). Outputs under
outputs/amo_compare_20261020/rounds/<controller>/<family>/; `report` writes report.json (success, falls, drops, per
task, exact McNemar) and the stability numbers per round from the traces (stability.py).

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/amo_compare_20261020/tasks.py run [--only amo] [--procs 6]
    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/amo_compare_20261020/tasks.py report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCENE = REPO / "scripts/experiments/scene_rsi_20261016"
sys.path[:0] = [str(HERE), str(SCENE), str(REPO / "src")]

import batch as B  # noqa: E402
import regress as RG  # noqa: E402
from gate import mcnemar_one_sided  # noqa: E402

OUT = REPO / "outputs/amo_compare_20261020"
ROUNDS = OUT / "rounds"
N = 10
# (family, task): 7 ScaleBFM mostly does, 3 it mostly misses (colliders_20261019)
TASKS = [
    ("pick_single", "orange_pickup"),
    ("pick_single", "tv_remote_pickup"),
    ("pick_single", "mug_pickup"),
    ("r1_held_single", "pick_car_key"),
    ("r1_on_single", "tv_remote_to_tv_stand"),
    ("r1_on_single", "jam_to_dining_table"),
    ("r2_in_single", "apple_in_black_bowl"),
    ("r1_held_near_single", "bring_remote_to_sofa"),
    ("r1_slide_then_grasp", "pick_thick_novel"),
    ("r2_on_single", "coffee_can_to_cart"),
]


def baseline(family: str) -> dict:
    reg = RG.load()
    return B.outcomes(RG.rounds_dir(family, reg[family], "colliders_20261019"))


def labels_of(family: str, task: str, n: int = N) -> list[str]:
    base = baseline(family)
    return sorted(k for k, r in base.items() if r["task"] == task)[:n]


def run(only: str | None, procs: int, gpus: str, smoke: bool) -> None:
    reg = RG.load()
    fams: dict[str, list[str]] = {}
    for f, t in TASKS:
        fams.setdefault(f, []).extend(labels_of(f, t, 1 if smoke else N))
    for ctrl in [c for c in ("amo", "scalebfm") if only in (None, c)]:
        for f, labels in fams.items():
            out = ROUNDS / ctrl / f
            job = B.Job(RG.CATALOG, out, {}, tuple(labels), npz=True, controller=ctrl)
            B.run_jobs([job], f, RG.pool_of(f, reg[f]), procs, out / "_batch.log", gpus)
            print(ctrl, f, len(labels), "rounds", flush=True)


def report() -> dict:
    rows, tot = {}, dict(n=0, amo=0, bfm=0, b=0, c=0, amo_fell=0, bfm_fell=0, amo_drop=0, bfm_drop=0, mismatch=[])
    for f, t in TASKS:
        a, s, base = B.outcomes(ROUNDS / "amo" / f), B.outcomes(ROUNDS / "scalebfm" / f), baseline(f)
        keys = [k for k in labels_of(f, t) if k in a and k in s]
        tot["mismatch"] += [f"{f}/{k}" for k in keys if bool(s[k]["ok"]) != bool(base[k]["ok"])]
        r = dict(n=len(keys), amo=sum(bool(a[k]["ok"]) for k in keys), bfm=sum(bool(s[k]["ok"]) for k in keys),
                 b=sum(bool(a[k]["ok"]) and not s[k]["ok"] for k in keys), c=sum(bool(s[k]["ok"]) and not a[k]["ok"] for k in keys),
                 amo_fell=sum(bool(a[k].get("fell")) for k in keys), bfm_fell=sum(bool(s[k].get("fell")) for k in keys),
                 amo_drop=sum(bool(a[k].get("dropped")) for k in keys), bfm_drop=sum(bool(s[k].get("dropped")) for k in keys),
                 amo_why=sorted({(a[k].get("why") or a[k].get("plan_why") or a[k].get("error") or "")[:70] for k in keys if not a[k]["ok"]}))  # fmt: skip
        rows[f"{f}/{t}"] = r
        for k in ("n", "amo", "bfm", "b", "c", "amo_fell", "bfm_fell", "amo_drop", "bfm_drop"):
            tot[k] += r[k]
    tot["p_amo_better"] = round(mcnemar_one_sided(tot["b"], tot["c"]), 4)
    tot["p_amo_worse"] = round(mcnemar_one_sided(tot["c"], tot["b"]), 4)
    out = dict(tasks=rows, total=tot)
    (OUT / "report.json").write_text(json.dumps(out, indent=1))
    for k, r in rows.items():
        print(f"{k:44s} AMO {r['amo']:2d}/{r['n']}  ScaleBFM {r['bfm']:2d}/{r['n']}  fell {r['amo_fell']}/{r['bfm_fell']}  "
              f"drop {r['amo_drop']}/{r['bfm_drop']}  {r['amo_why'][:2]}")  # fmt: skip
    print(
        "total",
        {k: v for k, v in tot.items() if k != "mismatch"},
        "ScaleBFM rerun != registry:",
        tot["mismatch"] or "none",
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--only", choices=("amo", "scalebfm"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--smoke", action="store_true", help="one start per task")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.only, a.procs, a.gpus, a.smoke)
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
