"""TODO 33.23 (plan section 3.8, D21; user 2026-10-23 "先做 33.23"): re-draw the diagnostic pool when the incumbent
saturates it. Run C's incumbent (iteration 11) passed all 30 diagnostic starts while 19 of its 90 hidden-pool starts
failed, so the roles saw no failure at all: iteration 12's Reviewer read 30 successes for 33 minutes, the Manager had
only the gate's per-class counts, the Engineers made up their own starts. A pool without failing starts cannot measure
direction (rules.md 8): a candidate can fix at most 0 of them.

The driver, and only the driver, does this after it has run the new incumbent on the pool (loop.refresh_diag_pool):
when the incumbent fails fewer than `min_fails` starts, it draws fresh starts from the family's diagnostic seed range
(family.draw from the seed after the last one the pool used, over the unparked tasks, so the hidden pools' seed ranges
are never touched), runs the incumbent on them, and swaps the failing ones in for passing starts until `target` starts
fail or `draw_max` fresh starts have been drawn. The pool keeps its size; every task keeps PER_TASK_MIN starts and one
passing start (a pool of failures only could not measure "broken"), no task grows past PER_TASK_MAX, and parked starts
and parked tasks are left alone (rule 26). Retired starts are not deleted: their start files move to
`pool_diag_retired/` and their round files in every diag/* directory become `_retired_<label>.*` (batch.outcomes,
batch.pool_labels and digest skip names starting with "_"); the added starts come in with the incumbent's rounds on
them, so nothing is run twice. `_pool.json` records every redraw and the ledger gets a `diag_pool` row. This is the one
exception, besides rule 26's acceptance-pool redraw, to rule 2 "a start is never removed from a pool" (version 17).
"""

from __future__ import annotations

import collections
import json
import re
import shutil
from pathlib import Path

MIN_FAILS_DEFAULT = 10  # redraw when the incumbent fails fewer than this many of the pool's starts
TARGET_FAILS_DEFAULT = 15  # ... until this many fail (about half of a 30-start pool)
DRAW_MAX_DEFAULT = 90  # fresh starts a redraw may draw and run (about 3 minutes of simulation)
PER_TASK_MIN = 2  # every task keeps at least this many starts ...
PER_TASK_MAX = 6  # ... and takes no more than this many, so the pool stays representative
BATCH = 30  # fresh starts drawn and run at a time
RETIRED_DIR = "pool_diag_retired"


def task_of(label: str) -> str:
    """'pick_Mug_2_s410003' -> 'pick_Mug_2'."""
    return re.sub(r"_s\d+$", "", label)


def seed_of(label: str) -> int:
    m = re.search(r"_s(\d+)$", label)
    return int(m.group(1)) if m else -1


def labels(pool: Path) -> list[str]:
    return sorted(p.stem for p in pool.glob("*.json") if not p.name.startswith("_"))


def outcomes(d: Path) -> dict:
    """{label: round record} of a rounds directory (batch.outcomes without importing batch)."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if not p.name.startswith("_")}


def next_seed(pool: Path) -> int:
    """One past the largest seed the pool has used: its starts, its retired starts, the seeds its draws skipped and the
    end of every earlier redraw."""
    meta = json.loads((pool / "_pool.json").read_text()) if (pool / "_pool.json").exists() else {}
    seeds = [seed_of(x) for x in labels(pool)]
    retired = pool.parent / RETIRED_DIR
    if retired.exists():
        seeds += [seed_of(x) for x in labels(retired)]
    seeds += [int(s) for _, s, _ in meta.get("skipped") or []]
    seeds += [int(r.get("seed_to", -1)) for r in meta.get("redraws") or []]
    seeds += [int(meta.get("next_seed", 0)) - 1]
    return max(seeds) + 1


def pick_retire(ok_by_label: dict, task: str, frozen: set, per_task_min: int = PER_TASK_MIN) -> str | None:
    """The passing start that makes room for a new failing start of `task`: one of the same task when the task keeps
    `per_task_min` starts and a passing start after it goes, else one from the task with the most starts (ties: the most
    passing, then the name); never a failing start or a frozen (parked) one. None when nothing can go."""
    by_task = collections.defaultdict(list)
    for x in ok_by_label:
        by_task[task_of(x)].append(x)

    def room(t: str) -> list[str]:
        xs = by_task.get(t, [])
        passing = [x for x in xs if ok_by_label[x]]
        if len(xs) - 1 < per_task_min or len(passing) - 1 < 1:
            return []
        return sorted(x for x in passing if x not in frozen)

    c = room(task)
    if c:
        return c[0]
    order = sorted(by_task, key=lambda t: (-len(by_task[t]), -sum(ok_by_label[x] for x in by_task[t]), t))
    for t in order:
        c = room(t)
        if c:
            return c[0]
    return None


def plan(ok_by_label: dict, new_fails: list[str], frozen: set, target: int, per_task_min: int = PER_TASK_MIN,
         per_task_max: int = PER_TASK_MAX) -> tuple[list[str], list[str]]:  # fmt: skip
    """Which of the fresh failing starts enter and which passing starts leave, the pool's size kept: in order, until
    `target` of the pool's starts fail or nothing can be retired. A start whose task already holds `per_task_max`
    starts is skipped. Returns (added, retired), aligned."""
    state = dict(ok_by_label)
    added, retired = [], []
    for x in new_fails:
        if sum(not v for v in state.values()) >= target:
            break
        t = task_of(x)
        if sum(task_of(y) == t for y in state) >= per_task_max:
            continue
        state[x] = False
        r = pick_retire(state, t, frozen, per_task_min)
        if r is None:
            del state[x]
            continue
        del state[r]
        added.append(x)
        retired.append(r)
    return added, retired


def apply(pool: Path, inc_dir: Path, new_pool: Path, new_rounds: Path, added: list[str], retired: list[str]) -> None:
    """Move the retired starts to pool_diag_retired/, set their rounds aside in every diag/* directory as
    `_retired_<label>.*`, and bring the added starts in with the incumbent's rounds on them from the redraw's run."""
    out = pool.parent / RETIRED_DIR
    out.mkdir(parents=True, exist_ok=True)
    for x in retired:
        shutil.move(str(pool / f"{x}.json"), str(out / f"{x}.json"))
        for d in inc_dir.parent.iterdir():
            if d.is_dir():
                for f in d.glob(f"{x}.*"):
                    f.rename(d / f"_retired_{f.name}")
    for x in added:
        shutil.copy(new_pool / f"{x}.json", pool / f"{x}.json")
        for f in new_rounds.glob(f"{x}.*"):
            shutil.copy(f, inc_dir / f.name)


def record(pool: Path, entry: dict) -> None:
    p = pool / "_pool.json"
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta.setdefault("redraws", []).append(entry)
    meta["written"] = len(labels(pool))
    meta["next_seed"] = int(entry["seed_to"]) + 1
    p.write_text(json.dumps(meta, indent=1))


def refresh(fam: dict, pool: Path, inc_dir: Path, run, scratch: Path, parked_tasks: set, frozen: set,
            min_fails: int = MIN_FAILS_DEFAULT, target: int = TARGET_FAILS_DEFAULT, draw_max: int = DRAW_MAX_DEFAULT,
            draw=None, batch: int = BATCH) -> dict | None:  # fmt: skip
    """The redraw. `run(pool_dir, out_dir)` runs the incumbent on every start of `pool_dir` into `out_dir` (the
    driver's batch queue); `draw` is family.draw. Returns the ledger row, or None when the pool keeps `min_fails`
    failing starts or no redraw is possible."""
    if draw is None:
        import family as F

        draw = F.draw
    inc = outcomes(inc_dir)
    ok_by = {x: bool(inc[x]["ok"]) for x in labels(pool) if x in inc}
    before = sum(not v for v in ok_by.values())
    live = [t for t in fam["tasks"] if t not in parked_tasks]
    frozen = set(frozen) | {x for x in ok_by if task_of(x) in parked_tasks}  # a parked task's starts stay too
    if before >= min_fails or not live or draw_max <= 0:
        return None
    seed = seed_from = next_seed(pool)
    drawn, added, retired, batches = 0, [], [], 0
    while sum(not v for v in ok_by.values()) < target and drawn < draw_max:
        d = scratch / f"draw_{batches:02d}"
        s = draw(fam, "diag", d, seed=seed, n=min(batch, draw_max - drawn), tasks=live, start_i=drawn)
        seed, batches = int(s["next_seed"]), batches + 1
        drawn += int(s["written"])
        if not s["written"]:
            break
        rounds = scratch / f"rounds_{batches - 1:02d}"
        run(d, rounds)
        res = outcomes(rounds)
        new_fails = sorted(x for x in labels(d) if x in res and not res[x]["ok"])
        a, r = plan(ok_by, new_fails, frozen, target)
        apply(pool, inc_dir, d, rounds, a, r)
        for x, y in zip(a, r):
            ok_by[x] = False
            ok_by.pop(y, None)
        added += a
        retired += r
        if new_fails and not a:
            break  # nothing can be retired any more
    entry = dict(step="diag_pool", fails_before=before, fails_after=sum(not v for v in ok_by.values()),
                 n=len(labels(pool)), drawn=drawn, added=added, retired=retired, seed_from=seed_from,
                 seed_to=seed - 1)  # fmt: skip
    record(pool, {k: entry[k] for k in ("fails_before", "fails_after", "drawn", "added", "retired", "seed_from",
                                        "seed_to")})  # fmt: skip
    return entry
