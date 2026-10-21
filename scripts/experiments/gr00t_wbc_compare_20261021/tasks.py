"""TODO 34.3: scene tasks under the gear WBC (default posture and rhoyn's h066_p012) vs ScaleBFM -- the same ten tasks as
the AMO comparison (TODO 28.3, amo_compare_20261020/tasks.py), the same starts, the current skills catalog, paired by
start. All controllers run on one machine (round outcomes are machine-specific).

  pools    copy each task's first N starts from the registry's sealed pools into outputs/gr00t_wbc_compare_20261021/
           pools/<family>/ (so the remote box needs no sandbox)
  run      every controller on those starts through batch.py / scene_round.py --npz
  report   success per task and controller, falls, drops, exact one-sided McNemar of each against ScaleBFM -> report.json

    python3 scripts/experiments/gr00t_wbc_compare_20261021/tasks.py pools
    PYTHONPATH=src $PY scripts/experiments/gr00t_wbc_compare_20261021/tasks.py run [--procs 48] [--gpus 0,1,2,3]
    python3 scripts/experiments/gr00t_wbc_compare_20261021/tasks.py report
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "scripts/rsi"), str(REPO / "src")]

OUT = REPO / "outputs/gr00t_wbc_compare_20261021"
POOLS, ROUNDS = OUT / "pools", OUT / "rounds"
CONTROLLERS = ("scalebfm", "gr00t_wbc", "gr00t_wbc_h066_p012")
N = 10
TASKS = [  # amo_compare_20261020/tasks.py: 7 ScaleBFM mostly does, 3 it mostly misses
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


def pools() -> None:
    import regress as RG

    reg = RG.load()
    for fam, task in TASKS:
        src = RG.pool_of(fam, reg[fam])
        dst = POOLS / fam
        dst.mkdir(parents=True, exist_ok=True)
        labels = sorted(p.stem for p in src.glob(f"{task}_s*.json"))[:N]
        for x in labels:
            shutil.copy2(src / f"{x}.json", dst / f"{x}.json")
        print(fam, task, len(labels))


def labels_of(fam: str, task: str) -> list[str]:
    return sorted(p.stem for p in (POOLS / fam).glob(f"{task}_s*.json"))


def run(procs: int, gpus: str, only: list[str]) -> None:
    import batch as B

    catalog = REPO / "src/locomani/tasks/scene/skills"
    for ctrl in only:
        for fam in sorted({f for f, _ in TASKS}):
            labels = [x for f, t in TASKS if f == fam for x in labels_of(f, t)]
            out = ROUNDS / ctrl / fam
            job = B.Job(catalog, out, {}, tuple(labels), npz=True, controller=ctrl)
            miss = B.run_jobs([job], fam, POOLS / fam, procs, out / "_batch.log", gpus)
            print(ctrl, fam, "missing", miss[0], flush=True)


def report() -> dict:
    from gate import mcnemar_one_sided

    out = dict(tasks={}, total={})
    rd = lambda c, f, x: json.loads((ROUNDS / c / f / f"{x}.json").read_text())  # noqa: E731
    for fam, task in TASKS:
        row = {}
        for c in CONTROLLERS:
            rs = [rd(c, fam, x) for x in labels_of(fam, task) if (ROUNDS / c / fam / f"{x}.json").exists()]
            row[c] = dict(ok=sum(bool(r["ok"]) for r in rs), n=len(rs), fell=sum(bool(r.get("fell")) for r in rs),
                          dropped=sum(bool(r.get("dropped")) for r in rs))  # fmt: skip
        out["tasks"][f"{fam}/{task}"] = row
    for c in CONTROLLERS:
        b = cc = ok = n = fell = 0
        for fam, task in TASKS:
            for x in labels_of(fam, task):
                pa, pb = ROUNDS / "scalebfm" / fam / f"{x}.json", ROUNDS / c / fam / f"{x}.json"
                if not (pa.exists() and pb.exists()):
                    continue
                a, r = json.loads(pa.read_text()), json.loads(pb.read_text())
                n += 1
                ok += bool(r["ok"])
                fell += bool(r.get("fell"))
                b += bool(r["ok"]) and not a["ok"]
                cc += bool(a["ok"]) and not r["ok"]
        out["total"][c] = dict(ok=ok, n=n, fell=fell, b_vs_scalebfm=b, c_vs_scalebfm=cc,
                               p_better=mcnemar_one_sided(b, cc), p_worse=mcnemar_one_sided(cc, b))  # fmt: skip
    (OUT / "tasks_report.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pools", "run", "report"))
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--only", nargs="+", default=list(CONTROLLERS))
    a = ap.parse_args()
    if a.cmd == "pools":
        pools()
    elif a.cmd == "run":
        run(a.procs, a.gpus, a.only)
    else:
        print(json.dumps(report(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
