#!/usr/bin/env python3
"""TODO 27.4 (plan docs/handoff/scene-rsi-plan-20261016.md §2, §10, D7, D9): the outer loop's program steps around
Claude's discovery (scripts/scene_discover_tasks.py) and the inner loops (loop.py).

`select <round dir>`: the discovered tasks (`<round dir>/discovery/tasks.jsonl`, a link to the discovery's directory
under outputs/scene_discovery/<scene>/discovery/, already checked against the mars +
ScaleBFM table and the stance overlay) -> gaps to close. A gap (D7: one gap = one group of tasks, one inner loop) is a
goal shape and hand count: every bucket-2 task whose goal is held(X) with a single-hand object is one gap
(`held_single`: the pick skill outside its measured settings), on(X, S) another (`on_single`: no put-down skill),
held(X) + near(X, Y) a third (`held+near_single`: carry it there), and so on; upright rides along. The inner loops' gap proposals (`loop/gaps.jsonl` of every
finished family under --loops) are gaps too, with the tasks they name. Ranked by tasks unlocked / estimated cost:
EST_USD_EXTEND for a gap whose skill exists (the pick_single loop spent 73 USD of model fees on its first five
iterations), EST_USD_NEW per skill the checker says is missing; the first MAX_GAPS within ROUND_BUDGET_USD are
selected (D9: two rounds, 2000 USD), and they run one after another, each on the catalog the previous one wrote back
(no merge of two loops' edits to one skill). Writes `<round dir>/selection.json`.

`family <round dir> <gap>`: families/<name>.json for a selected gap -- at most MAX_TASKS tasks, spread over the gap's
range (sorted by how far outside the measured settings they are, then every k-th), pools 50 / 60 / 80 with seeds in a
block of their own.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/scene_rsi_20261016/outer.py select outputs/scene_rsi_20261016/outer/round1 \\
        --loops /workspace/code/rsi-scene/pick-a
    $PY scripts/experiments/scene_rsi_20261016/outer.py family outputs/scene_rsi_20261016/outer/round1 on_single
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src")]

from locomani.scene_discovery.pilot_groups import extrapolation  # noqa: E402
from locomani.scene_discovery.predicates import parse  # noqa: E402

SCENE = "outputs/scene_discovery/adt_apartment_mars"
EST_USD_EXTEND = 75.0
EST_USD_NEW = 150.0
MAX_GAPS = 6  # D9
ROUND_BUDGET_USD = 1000.0  # D9: 2000 USD over two rounds
MAX_TASKS = 10  # per family: 8 sealed starts per task at 80
POOLS = {"diag": 50, "accept": 60, "sealed": 80}
SEED_BLOCK = 10000  # family k of round r: diag 100000 + block, accept 200000 + block, sealed 300000 + block
# gaps the checker reports when no skill exists (feasibility.py: an untested skill or a step without one)
NEW_SKILL = ("place_on_surface.single", "place_into.single", "grasp_bimanual.skill", "put_down_bimanual.skill")
CONSTRAINTS = {"upright", "in_room"}  # predicates that ride along with a move (near is one: held + near = a carry)


def shape(row: dict, meta: dict) -> str:
    """`<move>_<hands>`: the goal's move predicates (held / on / in, joined when several) and the hand count of its
    objects (single / bimanual / mixed)."""
    moves, hands = [], set()
    for g in row["goal"]:
        p = parse(g)
        if p.name in CONSTRAINTS:
            continue
        if p.name not in moves:
            moves.append(p.name)
        x = p.args[0]
        if isinstance(x, str) and x in meta["objects"]:
            hands.add(((meta["objects"][x].get("semantic") or {}).get("graspable")) or "none")
    h = hands.pop() if len(hands) == 1 else ("mixed" if hands else "none")
    return f"{'+'.join(sorted(moves)) or 'none'}_{h}"


def select(round_dir: Path, loops: list[Path]) -> dict:
    meta = json.loads((REPO / SCENE / "scene_meta.json").read_text())
    rows = [json.loads(x) for x in (round_dir / "discovery" / "tasks.jsonl").read_text().splitlines() if x.strip()]
    buckets = collections.Counter(r["bucket"] for r in rows)
    groups: dict = collections.defaultdict(list)
    for r in rows:
        if r["bucket"] == 2:
            groups[shape(r, meta)].append(r)
    gaps = []
    for name, tasks in groups.items():
        gap_count = collections.Counter(g for t in tasks for g in t["gaps"])
        new = sorted(g for g in gap_count if g in NEW_SKILL)
        cost = EST_USD_NEW * len(new) if new else EST_USD_EXTEND
        gaps.append(dict(name=name, source="checker", tasks=sorted(t["id"] for t in tasks), n=len(tasks),
                         new_skills=new, est_usd=cost, per_usd=round(len(tasks) / cost, 4),
                         checker_gaps=dict(gap_count.most_common())))  # fmt: skip
    goals = {json.dumps(sorted(r["goal"])): r["id"] for r in rows}
    for root in loops:
        for f in sorted((root / "outputs" / "scene_rsi_20261016").glob("*/loop/gaps.jsonl")):
            fam = json.loads(
                (root / "scripts/experiments/scene_rsi_20261016/families" / f"{f.parents[1].name}.json").read_text()
            )
            frows = {
                json.loads(x)["id"]: json.loads(x)
                for x in (REPO / fam["tasks_file"]).read_text().splitlines()
                if x.strip()
            }
            for line in f.read_text().splitlines():
                p = json.loads(line)
                named = [goals.get(json.dumps(sorted(frows[t]["goal"]))) for t in p["tasks"] if t in frows]
                gaps.append(dict(name=p["name"], source=str(f), tasks=sorted(t for t in named if t), n=len(p["tasks"]),
                                 new_skills=[p["name"]], est_usd=EST_USD_NEW * 2, per_usd=round(len(p["tasks"]) / (2 * EST_USD_NEW), 4),
                                 capability=p.get("capability", ""), loop_estimate=p.get("est_cost", "")))  # fmt: skip
    gaps.sort(key=lambda g: (-g["per_usd"], -g["n"], g["name"]))
    chosen, spent = [], 0.0
    for g in gaps:
        if len(chosen) < MAX_GAPS and spent + g["est_usd"] <= ROUND_BUDGET_USD and g["n"] > 0:
            chosen.append(g["name"])
            spent += g["est_usd"]
    out = dict(round=round_dir.name, discovered=len(rows), buckets=dict(sorted(buckets.items())), gaps=gaps,
               selected=chosen, est_usd=spent, rule=f"tasks / estimated USD (extend {EST_USD_EXTEND}, new skill "
               f"{EST_USD_NEW} each); first {MAX_GAPS} within {ROUND_BUDGET_USD} USD")  # fmt: skip
    (round_dir / "selection.json").write_text(json.dumps(out, indent=1))
    return out


def family(round_dir: Path, gap: str) -> Path:
    sel = json.loads((round_dir / "selection.json").read_text())
    g = next(x for x in sel["gaps"] if x["name"] == gap)
    k = sel["selected"].index(gap) + 1 if gap in sel["selected"] else len(sel["selected"]) + 1
    rows = {
        json.loads(x)["id"]: json.loads(x)
        for x in (round_dir / "discovery" / "tasks.jsonl").read_text().splitlines()
        if x.strip()
    }
    tasks = sorted(g["tasks"], key=lambda t: (extrapolation(rows[t]), t))
    if len(tasks) > MAX_TASKS:
        step = (len(tasks) - 1) / (MAX_TASKS - 1)
        tasks = [tasks[round(i * step)] for i in range(MAX_TASKS)]
    r = int("".join(c for c in round_dir.name if c.isdigit()) or 1)
    block = SEED_BLOCK * (10 * (r - 1) + k)
    name = f"r{r}_{gap.replace('+', '_')}"
    fam = dict(
        name=name,
        about=f"outer round {r}, gap {gap} ({g['source']}): {len(tasks)} of its {g['n']} discovered tasks, spread over "
        f"how far they are from the measured settings; new skills {g['new_skills'] or 'none'}",
        scene=SCENE,
        # the discovery lives in the scene's tree (a sandbox links outputs/scene_discovery only); <round dir>/discovery
        # links to it
        tasks_file=str((round_dir / "discovery" / "tasks.jsonl").resolve().relative_to(REPO)),
        tasks=tasks,
        pools={p: {"seed": base + block, "n": n} for (p, n), base in zip(POOLS.items(), (100000, 200000, 300000))},
    )
    path = HERE / "families" / f"{name}.json"
    path.write_text(json.dumps(fam, indent=1) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("round_dir", type=Path)
    s.add_argument("--loops", type=Path, nargs="*", default=[], help="finished loop sandboxes (their gaps.jsonl)")
    f = sub.add_parser("family")
    f.add_argument("round_dir", type=Path)
    f.add_argument("gap")
    a = ap.parse_args(argv)
    rd = a.round_dir if a.round_dir.is_absolute() else REPO / a.round_dir
    if a.cmd == "select":
        out = select(rd, a.loops)
        print(f"[outer] {out['discovered']} tasks, buckets {out['buckets']}; selected {out['selected']} "
              f"(~{out['est_usd']} USD)")  # fmt: skip
        for g in out["gaps"]:
            print(
                f"   {g['name']:28s} n {g['n']:3d}  new {','.join(g['new_skills']) or '-':40s} ~{g['est_usd']:5.0f} USD"
            )
    else:
        print(family(rd, a.gap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
