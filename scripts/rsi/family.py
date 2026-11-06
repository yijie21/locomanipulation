"""TODO 27.2.4 (plan docs/handoff/scene-rsi-plan-20261016.md D7): a task family (families/<name>.json -- the scene, the
tasks file, the task ids, the pools' seeds and sizes) and its start pools. A pool is a directory of one JSON per start,
`<task>_s<seed>.json` = {"task", "start"}: start i of a pool takes the family's task i mod n and seed `seed + i`
(start.sample_start, with the family's `spawn` option; a start the scene cannot give -- no free spot, no reachable spawn -- or one where the goal already
holds (scene_discovery.lint.goal_holds, plan §12) moves on to the next seed and is recorded). The diagnostic pool lives in the loop's tree, the acceptance and sealed pools in its hidden tree, and the
sealed pool is drawn only after the loop ends. In a sandbox the acceptance and sealed seeds are blank in the family file
and read from outputs/scene_rsi_20261016_hidden/<family>/seeds.json (sandbox.py hide_seeds, TODO 36, review 2026-10-22).

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


def hidden_seeds(name: str) -> dict:
    """The seeds a sandbox keeps out of its family file (sandbox.py hide_seeds); {} when it holds none -- or when the
    hidden tree cannot be read: while a role runs it has mode 000 (loop.py hidden_locked), and Path.exists raises
    PermissionError there on Python 3.10, so every scene_round.py a role started in a sandbox crashed in load() and
    the roles wrote their own wrappers that stub this function (run C's round1.py, TODO 37.1, 2026-10-23). The
    diagnostic pool's seed is in the family file; only drawing a hidden pool needs these, and no role draws one."""
    p = REPO / "outputs" / "scene_rsi_20261016_hidden" / name / "seeds.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except PermissionError:
        return {}


def load(name: str) -> dict:
    fam = json.loads((HERE / "families" / f"{name}.json").read_text())
    for pool, spec in fam.get("pools", {}).items():  # TODO 36 (review 2026-10-22): blank seeds -> the hidden file's
        if spec.get("seed") is None and pool in (hid := hidden_seeds(name)):
            fam["pools"][pool] = dict(spec, seed=hid[pool])
    fam["scene_dir"] = REPO / fam["scene"]
    fam["meta"] = json.loads((fam["scene_dir"] / "scene_meta.json").read_text())
    rows = {
        json.loads(x)["id"]: json.loads(x) for x in (REPO / fam["tasks_file"]).read_text().splitlines() if x.strip()
    }
    fam["rows"] = {t: rows[t] for t in fam["tasks"]}
    return fam


def task(fam: dict, task_id: str) -> Task:
    return Task.from_row(fam["rows"][task_id], fam["meta"])


def timing(fam: dict, task_id: str) -> dict:
    """A round's success timing, `hold_s` (the end-of-plan hold) and `judge_s` (the judging window, TODO 38.19): the
    task row's, else the family's; a key neither sets is absent (the caller's default). gso_all mixes one-hand picks,
    judged at the end, with two-hand lifts and packs judged over the last 5 s (user 2026-11-06)."""
    row = fam["rows"][task_id]
    return {k: row.get(k, fam.get(k)) for k in ("hold_s", "judge_s") if row.get(k, fam.get(k)) is not None}


def draw(fam: dict, pool: str, out: Path, seed=None, n=None, tasks=None, start_i: int = 0) -> dict:
    """Write the pool's starts into `out`; returns {"written": n, "skipped": [(task, seed, why)], "next_seed": s}.
    `seed`, `n` and `tasks` override the family's pool spec and `start_i` is where the task rotation starts (TODO
    33.23: the diagnostic pool's redraw continues from the seed after the last one the pool used, over the unparked
    tasks only, so it never touches the hidden pools' seed ranges)."""
    spec = fam["pools"][pool]
    if seed is None and spec.get("seed") is None:
        raise SystemExit(f"{fam['name']}: the {pool} pool's seed is blank and the hidden seeds.json has none")
    occ, nav = build(fam["meta"], body_radius=BODY_RADIUS_M)
    out.mkdir(parents=True, exist_ok=True)
    tasks = list(fam["tasks"] if tasks is None else tasks)
    n = int(spec["n"]) if n is None else int(n)
    skipped, unavailable, written, seed, i = [], {}, 0, int(spec["seed"] if seed is None else seed), int(start_i)
    while written < n and tasks:
        t = task(fam, tasks[i % len(tasks)])
        st = None
        for _ in range(SKIP_LIMIT):
            try:
                st = sample_start(
                    fam["meta"],
                    occ,
                    nav,
                    list(t.objects),
                    np.random.default_rng(seed),
                    seed=seed,
                    room=t.room,
                    spawn=fam.get("spawn"),
                    placement={**(fam.get("placement") or {}), **(fam["rows"][t.id].get("placement") or {})} or None,
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
    summary = dict(family=fam["name"], pool=pool, written=written, unavailable=unavailable, skipped=skipped,
                   next_seed=seed)  # fmt: skip
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
