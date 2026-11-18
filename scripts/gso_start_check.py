#!/usr/bin/env python3
"""Can each task of a family start? (user 2026-11-17, Q11: the hand-picked two-hand scene may be too crowded for the
hug's free corridors.) For every task of the family, draws starts as its pools do (start.sample_start with the
family's spawn and the row's placement: the task object on the reach band, the carton on its band, HANDS_CLEAR_M free
beside a hugged object's walls, the other objects shuffled) over `--seeds` seeds, and reports how many started; a
task with none names its object, the one to swap out in scripts/gso_gallery.py --pick --serve. CPU only.

    PYTHONPATH=src $VENV_SIM_PY scripts/gso_start_check.py scripts/rsi/families/gso_bimanual_lift.json [--seeds 20]
        [--out outputs/gso_preview/starts_gso_bimanual_lift.json]

Exit 1 when a task found no start in any seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts" / "rsi")]


def load(path: Path) -> dict:
    """A family file as family.load reads it, from any directory: its tasks file is looked up beside it first."""
    fam = json.loads(path.read_text())
    fam["scene_dir"] = REPO / fam["scene"]
    fam["meta"] = json.loads((fam["scene_dir"] / "scene_meta.json").read_text())
    tasks = path.parent / Path(fam["tasks_file"]).name
    tasks = tasks if tasks.is_file() else REPO / fam["tasks_file"]
    rows = {json.loads(x)["id"]: json.loads(x) for x in tasks.read_text().splitlines() if x.strip()}
    fam["rows"] = {t: rows[t] for t in fam["tasks"]}
    return fam


def check(fam: dict, seeds: int) -> dict:
    """{task: {"started": n, "seeds": seeds, "first": seed or None, "why": the last refusal}}."""
    from family import Task

    from locomani.primitives.navigator import build
    from locomani.tasks.scene.start import sample_start
    from locomani.tasks.scene.world import BODY_RADIUS_M

    occ, nav = build(fam["meta"], body_radius=BODY_RADIUS_M)
    out = {}
    for t in fam["tasks"]:
        task = Task.from_row(fam["rows"][t], fam["meta"])
        got, first, why = 0, None, None
        for i in range(seeds):
            try:
                sample_start(fam["meta"], occ, nav, list(task.objects), np.random.default_rng(i), seed=i,
                             spawn=fam["spawn"], placement=fam["rows"][t]["placement"])  # fmt: skip
            except RuntimeError as e:
                why = str(e)
                continue
            got += 1
            first = i if first is None else first
        out[t] = {"started": got, "seeds": seeds, "first": first, "why": None if got else why}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family", type=Path, help="a family file, e.g. scripts/rsi/families/gso_bimanual_lift.json")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    fam = load(a.family)
    res = check(fam, a.seeds)
    phrase = {k: (o.get("semantic") or {}).get("phrase") for k, o in fam["meta"]["objects"].items()}
    for t, r in res.items():
        key = t.split("_", 1)[1]
        print(
            f"[starts] {t} ({phrase.get(key)}): {r['started']} / {r['seeds']}" + (f" -- {r['why']}" if r["why"] else "")
        )
    stuck = [t for t, r in res.items() if not r["started"]]
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"family": fam["name"], "tasks": res, "stuck": stuck}, indent=1) + "\n")
    print(json.dumps({"family": fam["name"], "stuck": stuck}))
    return 1 if stuck else 0


if __name__ == "__main__":
    raise SystemExit(main())
