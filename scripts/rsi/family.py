"""TODO 27.2.4 (plan docs/handoff/scene-rsi-plan-20261016.md D7): a task family (families/<name>.json -- the scene, the
tasks file, the task ids, the pools' seeds and sizes) and its start pools. A pool is a directory of one JSON per start,
`<task>_s<seed>.json` = {"task", "start"}: start i of a pool takes the family's task i mod n and seed `seed + i`
(start.sample_start; a start the scene cannot give -- no free spot, no reachable spawn -- or one where the goal already
holds (scene_discovery.lint.goal_holds, plan §12) moves on to the next seed and is recorded). The diagnostic pool lives in the loop's tree, the acceptance and sealed pools in its hidden tree, and the
sealed pool is drawn only after the loop ends.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/family.py draw pick_single diag <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src")]

from locomani.primitives.navigator import build  # noqa: E402
from locomani.scene_discovery.lint import goal_holds  # noqa: E402
from locomani.tasks.scene.start import sample_start  # noqa: E402
from locomani.tasks.scene.world import BODY_RADIUS_M, Task  # noqa: E402

SKIP_LIMIT = 20  # seeds tried per start before the pool gives up on that task


def load(name: str) -> dict:
    fam = json.loads((HERE / "families" / f"{name}.json").read_text())
    fam["scene_dir"] = REPO / fam["scene"]
    fam["meta"] = json.loads((fam["scene_dir"] / "scene_meta.json").read_text())
    rows = {
        json.loads(x)["id"]: json.loads(x) for x in (REPO / fam["tasks_file"]).read_text().splitlines() if x.strip()
    }
    fam["rows"] = {t: rows[t] for t in fam["tasks"]}
    return fam


def task(fam: dict, task_id: str) -> Task:
    return Task.from_row(fam["rows"][task_id], fam["meta"])


def draw(fam: dict, pool: str, out: Path) -> dict:
    """Write the pool's starts into `out`; returns {"written": n, "skipped": [(task, seed, why)]}."""
    spec = fam["pools"][pool]
    occ, nav = build(fam["meta"], body_radius=BODY_RADIUS_M)
    out.mkdir(parents=True, exist_ok=True)
    tasks = list(fam["tasks"])
    skipped, unavailable, written, seed, i = [], {}, 0, int(spec["seed"]), 0
    while written < int(spec["n"]) and tasks:
        t = task(fam, tasks[i % len(tasks)])
        st = None
        for _ in range(SKIP_LIMIT):
            try:
                st = sample_start(
                    fam["meta"], occ, nav, list(t.objects), np.random.default_rng(seed), seed=seed, room=t.room
                )
                if goal_holds(fam["rows"][t.id]["goal"], fam["meta"], st.to_dict()):
                    raise RuntimeError("the goal holds at the start")
                break
            except RuntimeError as e:
                st = None
                skipped.append((t.id, seed, str(e)))
                seed += 1
        if st is None:  # the scene cannot give this task a start (no reachable spawn): out of the pool, recorded
            unavailable[t.id] = skipped[-1][2]
            tasks.remove(t.id)
            continue
        (out / f"{t.id}_s{seed}.json").write_text(json.dumps(dict(task=t.id, start=st.to_dict()), indent=1))
        written += 1
        seed += 1
        i += 1
    summary = dict(family=fam["name"], pool=pool, written=written, unavailable=unavailable, skipped=skipped)
    (out / "_pool.json").write_text(json.dumps(summary, indent=1))
    return summary


def labels(pool: Path) -> list[str]:
    return sorted(p.stem for p in pool.glob("*.json") if not p.name.startswith("_"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("family")
    d.add_argument("pool", choices=("diag", "accept", "sealed"))
    d.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    res = draw(load(a.family), a.pool, a.out)
    print(f"{res['written']} starts, {len(res['skipped'])} seeds skipped, unavailable {res['unavailable']} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
