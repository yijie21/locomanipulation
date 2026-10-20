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
iterations), EST_USD_NEW per skill the checker says is missing; the first MAX_GAPS are selected (D9; its 2000 USD
budget was lifted by the user on 2026-10-19, so the estimate only ranks), and they run one after another, each on
the catalog the previous one wrote back (no merge of two loops' edits to one skill). Writes `<round dir>/selection.json`.

`family <round dir> <gap>`: families/<name>.json for a selected gap -- at most MAX_TASKS tasks, spread over the gap's
range (sorted by how far outside the measured settings they are, then every k-th), pools 50 / 60 / 80 with seeds in a
block of their own.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/rsi/outer.py select outputs/scene_rsi_20261016/outer/round1 \\
        --loops /workspace/code/rsi-scene/pick-a
    $PY scripts/rsi/outer.py family outputs/scene_rsi_20261016/outer/round1 on_single
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD_SCRIPTS = "scripts/experiments/scene_rsi_20261016"  # these scripts' place until 2026-10-20
sys.path[:0] = [str(REPO / "src")]

from locomani.scene_discovery.pilot_groups import extrapolation  # noqa: E402
from locomani.scene_discovery.predicates import parse  # noqa: E402

SCENE = "outputs/scene_discovery/adt_apartment_mars"
EST_USD_EXTEND = 75.0
EST_USD_NEW = 150.0
MAX_GAPS = 6  # D9
MAX_TASKS = 10  # per family: 8 sealed starts per task at 80
POOLS = {"diag": 50, "accept": 60, "sealed": 80}
SEED_BLOCK = 10000  # family k: diag 100000 + k block, accept 200000 + .., sealed 300000 + ..; round r adds ROUND_SEEDS
ROUND_SEEDS = 1_000_000  # per round after the first (until 2026-10-18 round r added 10 blocks per round: round 2's diag
# seeds were round 1's acceptance seeds, its acceptance seeds round 1's sealed ones)
# gaps the checker reports when no skill exists (feasibility.py: an untested skill or a step without one)
NEW_SKILL = (
    "place_on_surface.single",
    "place_into.single",
    "grasp_bimanual.skill",
    "put_down_bimanual.skill",
    "goal.in_room",
)
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
    if not moves:  # a goal of constraints only (in_room(robot, bedroom)): named by them
        moves = sorted({parse(g).name for g in row["goal"]})
    return f"{'+'.join(sorted(moves))}_{h}"


def group(row: dict, meta: dict) -> str:
    """The gap a bucket-2 task waits on: its pick's, `held_<hands>`, when the checker finds a gap in the grasp step
    (the first step of every move), else its goal's shape. Until 2026-10-19 every task went by its goal's shape:
    r2_in_single took the wallet and the car key, which no pick could lift (too flat for the pads), and parked both
    after 80 USD (TODO 27.10.4, plan §12)."""
    for c in row.get("checks") or ():
        if c.get("outcome") != "ok" and c.get("gap") and c["step"].startswith("grasp_"):
            hands = c["step"].split("(")[0].removeprefix("grasp_")
            return f"held_{hands}"
    return shape(row, meta)


def family_skills(fam: dict) -> list[str]:
    """The new skills a family was made to build: its `new_skills`, or (round 1's files) the list its `about` names."""
    if "new_skills" in fam:
        return list(fam["new_skills"])
    m = re.search(r"new skills (\[[^\]]*\])", fam.get("about", ""))
    return list(ast.literal_eval(m.group(1))) if m else []


def select(round_dir: Path, loops: list[Path]) -> dict:
    meta = json.loads((REPO / SCENE / "scene_meta.json").read_text())
    rows = [json.loads(x) for x in (round_dir / "discovery" / "tasks.jsonl").read_text().splitlines() if x.strip()]
    buckets = collections.Counter(r["bucket"] for r in rows)
    groups: dict = collections.defaultdict(list)
    for r in rows:
        if r["bucket"] == 2:
            groups[group(r, meta)].append(r)
    gaps = []
    for name, tasks in groups.items():
        gap_count = collections.Counter(g for t in tasks for g in t["gaps"])
        new = sorted(g for g in gap_count if g in NEW_SKILL)
        cost = EST_USD_NEW * len(new) if new else EST_USD_EXTEND
        gaps.append(dict(name=name, source="checker", tasks=sorted(t["id"] for t in tasks), n=len(tasks),
                         new_skills=new, est_usd=cost, per_usd=round(len(tasks) / cost, 4),
                         checker_gaps=dict(gap_count.most_common())))  # fmt: skip
    goals = {json.dumps(sorted(r["goal"])): r["id"] for r in rows}
    bucket = {r["id"]: r["bucket"] for r in rows}
    # a skill an earlier round's family was made to build is not proposed again; a proposal counts only this round's
    # tasks still in bucket 2, and the same skill proposed by several loops is one gap (until 2026-10-18 each loop's
    # proposal stood alone and counted its own round's tasks: round 2 re-selected slide_then_grasp, which round 1 had
    # built, for tasks two of which were in bucket 1 already)
    built = {k for f in sorted((HERE / "families").glob("*.json")) for k in family_skills(json.loads(f.read_text()))}
    proposals: dict = {}
    for root in loops:
        for f in sorted((root / "outputs" / "scene_rsi_20261016").glob("*/loop/gaps.jsonl")):
            # sandboxes built before 2026-10-20 keep the scripts under scripts/experiments/scene_rsi_20261016/
            name = f"{f.parents[1].name}.json"
            ff = [
                d / name
                for d in (root / "scripts/rsi/families", root / OLD_SCRIPTS / "families")
                if (d / name).exists()
            ]
            fam = json.loads(ff[0].read_text())
            frows = {
                json.loads(x)["id"]: json.loads(x)
                for x in (REPO / fam["tasks_file"]).read_text().splitlines()
                if x.strip()
            }
            for line in f.read_text().splitlines():
                p = json.loads(line)
                named = [goals.get(json.dumps(sorted(frows[t]["goal"]))) for t in p["tasks"] if t in frows]
                g = proposals.setdefault(p["name"], dict(name=p["name"], source=[], tasks=set(), proposed=0,
                                                         capability=[], loop_estimate=[]))  # fmt: skip
                g["source"].append(str(f))
                g["tasks"] |= {t for t in named if t and bucket[t] == 2}
                g["proposed"] += len(p["tasks"])
                g["capability"].append(p.get("capability", ""))
                g["loop_estimate"].append(p.get("est_cost", ""))
    skipped = []
    for name, g in sorted(proposals.items()):
        if name in built or not g["tasks"]:
            skipped.append(dict(name=name, why="built by an earlier family" if name in built else "no bucket-2 task here",
                                proposed=g["proposed"]))  # fmt: skip
            continue
        n = len(g["tasks"])
        gaps.append(dict(g, tasks=sorted(g["tasks"]), n=n, new_skills=[name], est_usd=EST_USD_NEW * 2,
                         per_usd=round(n / (2 * EST_USD_NEW), 4)))  # fmt: skip
    gaps.sort(key=lambda g: (-g["per_usd"], -g["n"], g["name"]))
    chosen, spent = [], 0.0
    for g in gaps:
        if len(chosen) < MAX_GAPS and g["n"] > 0:
            chosen.append(g["name"])
            spent += g["est_usd"]
    out = dict(round=round_dir.name, discovered=len(rows), buckets=dict(sorted(buckets.items())), gaps=gaps,
               skipped_proposals=skipped, selected=chosen, est_usd=spent, rule=f"tasks / estimated USD (extend {EST_USD_EXTEND}, new skill "
               f"{EST_USD_NEW} each); first {MAX_GAPS}")  # fmt: skip
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
    block = SEED_BLOCK * k + ROUND_SEEDS * (r - 1)
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
        new_skills=g["new_skills"],
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
