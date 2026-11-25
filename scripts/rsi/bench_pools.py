"""The pools of a benchmark family: which episodes the diagnostic, acceptance and sealed pools hold. This file is the
anti-leaderboard wall (what a role may see and what the hidden split seed chooses), so it is the one place that draws them.

Two kinds of adapter (TODO 54 phase 3; code moved from backend_simple.py and backend_bench.py, the draws unchanged):

- practice files (SIMPLE, LocoManiLab): `<Task>/Q/<line>` over `<Task>.jsonl` files whose rows carry `src.level`. The first 10
  lines of a level are the diagnostic pool; the others are hidden and split by a hidden seed into two halves (acceptance,
  sealed), each level shuffled on its own;
- seed pools (`pool_mode` "seeds": BiGym 2.0, HumanoidArena): no practice files, the episodes are `<task>/<P|A|S>/<n>`. The
  diagnostic pool is the practice pool P's first `per_task` indices; the acceptance and sealed pools are a block of `per_task`
  consecutive indices of the pool A / S, at an offset (at least OFFSET_FLOOR, so no label repeats the practice pool's) that a
  hidden seed chooses for each task.

`draw(fam, ad, pool, out, n, tasks)` writes the pool's files `<task>_s<n>.json` (`task`, `episode_id`) and `_pool.json`."""

from __future__ import annotations

import json
import random
from pathlib import Path

import bench_common as C

LEVELS = (0, 1, 2)
DIAG_PER_LEVEL = 10
PER_LEVEL = 50
POOL_LETTER = {
    "diag": "P",
    "accept": "A",
    "sealed": "S",
}  # the loop's pools -> the seed-pool adapter's (practice, acceptance, sealed)
PROBE_LETTERS = "P"  # a diagnostic round or a probe opens the practice pool only
GATE_LETTERS = "PAS"  # the gate (accept, and the check half's diagnostic starts) and the sealed set
OFFSET_FLOOR = (
    100  # the first index a hidden pool block may start at: the practice pool's labels (`s0`..) never repeat there
)


def seeded(ad: dict) -> bool:
    """True for an adapter whose episodes are seed pools (BiGym 2.0, HumanoidArena), not practice files."""
    return ad.get("pool_mode") == "seeds"


# ------------------------------------------------------------------------------------------------ practice files
def practice_dir(fam: dict, full: bool = True) -> Path:
    """The directory of the practice files. `full`: the hidden tree's copy when it can be listed (a sandbox), else
    the family file's `practice_dir` (which, in a sandbox, holds blanked rows)."""
    if full:
        h = C.hidden_family_dir(fam["name"]) / "practice"
        try:
            if h.is_dir() and any(h.iterdir()):
                return h
        except PermissionError:
            pass
    return C._abs(fam["practice_dir"])


def rows(path: Path) -> list[dict]:
    """The practice file's rows as {level (None for a blanked line)} in file order."""
    out = []
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line) if line.strip() else {}
        except ValueError:
            d = {}
        lv = (d.get("src") or {}).get("level") if isinstance(d, dict) else None
        out.append(dict(level=lv if lv in LEVELS else None))
    return out


def levels_of(fam: dict, task: str, full: bool = True) -> list[list[int]]:
    """For each level 0, 1, 2 the line numbers of that level in file order."""
    f = practice_dir(fam, full) / f"{task}.jsonl"
    by: list[list[int]] = [[], [], []]
    if f.exists():
        for i, r in enumerate(rows(f)):
            if r["level"] is not None:
                by[r["level"]].append(i)
    return by


def diag_lines(fam: dict, task: str) -> list[int]:
    """The diagnostic pool's lines: the first 10 of each level in file order."""
    return sorted(i for lv in levels_of(fam, task) for i in lv[:DIAG_PER_LEVEL])


def split(fam: dict, task: str, seed: int) -> dict[str, list[int]]:
    """{"accept": lines, "sealed": lines}: each level's lines after the first 10, shuffled by the seed, half each.
    A pool's `per_level` in the family file keeps that many of its half per level (an official round costs minutes:
    the gate's pool is 10 per level; the lines left over stay hidden and unused, and can be added later)."""
    acc: list[int] = []
    sea: list[int] = []
    keep = {p: (fam.get("pools", {}).get(p) or {}).get("per_level") for p in ("accept", "sealed")}
    for k, lines in enumerate(levels_of(fam, task)):
        rest = lines[DIAG_PER_LEVEL:]
        random.Random(f"{seed}:{task}:{k}").shuffle(rest)
        h = len(rest) // 2
        acc += rest[:h][: keep["accept"]]
        sea += rest[h:][: keep["sealed"]]
    return dict(accept=sorted(acc), sealed=sorted(sea))


def hidden_ids(fam: dict) -> set[str]:
    """Every episode id outside the diagnostic pool (accept and sealed together), as `<Task>/Q/<line>`. Needs only the
    levels in the file, so it works on the full file and on the blanked one."""
    out = set()
    for task in fam["tasks"]:
        f = practice_dir(fam, True) / f"{task}.jsonl"
        if not f.exists():
            continue
        diag = set(diag_lines(fam, task))
        out |= {f"{task}/Q/{i}" for i in range(len(rows(f))) if i not in diag}
    return out


def in_practice(pdir: Path, eid: str) -> bool:
    """True when line n of <pdir>/<Task>.jsonl, for the id `<Task>/Q/<n>`, is a real (not blank) row."""
    try:
        task, _, n = eid.split("/")
        return rows(pdir / f"{task}.jsonl")[int(n)]["level"] is not None
    except (ValueError, IndexError, OSError):
        return False


def pool_seed(fam: dict) -> int | None:
    a = fam["pools"].get("accept", {}).get("seed")
    s = fam["pools"].get("sealed", {}).get("seed")
    if a is not None and s is not None and a != s:
        raise SystemExit(f"{fam['name']}: accept and sealed must share one split seed ({a} != {s})")
    return a if a is not None else s


# ------------------------------------------------------------------------------------------------- seed pools
def parse_id(eid: str) -> tuple[str, str, int]:
    """(task, pool letter, n) of `<task>/<pool>/<n>`; ValueError for anything else."""
    parts = str(eid).split("/")
    if len(parts) != 3 or not (parts[2].isascii() and parts[2].isdigit()):
        raise ValueError(f"{eid!r} is not <task>/<pool>/<n>")
    return parts[0], parts[1], int(parts[2])


def id_refusal(ad: dict, eid: str, letters: str = GATE_LETTERS) -> str | None:
    """Why an episode id is not one this family may name or this call may run (None: fine). The evaluation episodes are no
    letter of the pools: an `E` id, a pool the call does not open and an index outside the pool are refused here, the adapter
    refuses the same again (its CLI, its episodes module and its server)."""
    try:
        task, letter, n = parse_id(eid)
    except ValueError as e:
        return str(e)
    if task not in ad["tasks"]:
        return f"{eid}: task {task!r} is not one of the adapter's"
    if letter not in ad["pool_sizes"]:
        return (
            f"{eid}: pool {letter!r} is not one of {''.join(ad['pool_sizes'])} (the evaluation episodes are never run)"
        )
    if letter not in letters:
        return f"{eid}: pool {letter} is not open to this call (opens {letters})"
    if n >= ad["pool_sizes"][letter]:
        return f"{eid}: pool {letter} has {ad['pool_sizes'][letter]} episodes"
    return None


def check_seeded(fam: dict, ad: dict) -> None:
    """A family of a seed-pool adapter names its pools by size and split seed only: each pool takes its letter from its name
    (diag P, accept A, sealed S) and `per_task` episodes of every task; a `letter` or `ids` entry may only repeat that. The
    evaluation episodes cannot be named: they are no letter and no index of any pool (id_refusal)."""
    for name, spec in (fam.get("pools") or {}).items():
        if name not in POOL_LETTER:
            raise ValueError(
                f"{fam.get('name')}: pool {name!r}: a seed-pool family has the pools {sorted(POOL_LETTER)}"
            )
        letter = POOL_LETTER[name]
        if spec.get("letter", letter) != letter:
            raise ValueError(f"{fam.get('name')}: pool {name} is the adapter's pool {letter}, not {spec['letter']!r}")
        per = int(spec.get("per_task", 0) or 0)
        room = ad["pool_sizes"][letter] - (0 if name == "diag" else OFFSET_FLOOR)
        if per < 1 or per > (OFFSET_FLOOR if name == "diag" else room):
            raise ValueError(f"{fam.get('name')}: pool {name}: per_task {per} is not a count of 1 to {OFFSET_FLOOR if name == 'diag' else room}")  # fmt: skip
        for eid in spec.get("ids") or []:
            bad = id_refusal(ad, eid, letter)
            if bad:
                raise ValueError(f"{fam.get('name')}: pool {name}: {bad}")
    for t in fam.get("tasks") or []:
        if t not in ad["tasks"]:
            raise ValueError(f"{fam.get('name')}: task {t!r} is not one of the adapter's: {', '.join(ad['tasks'])}")


# ------------------------------------------------------------------------------------------------------ draw
def _practice_lines(fam: dict, pool: str, tasks, ids, seed, unavailable: dict) -> dict[str, list[int]]:
    per_task: dict[str, list[int]] = {}
    for task in tasks:
        if ids:  # the pool is a list of episode ids (a hidden seeds.json entry {"ids": [...]})
            per_task[task] = sorted(int(x.rsplit("/", 1)[1]) for x in ids if x.startswith(f"{task}/Q/"))
            continue
        lv = levels_of(fam, task)
        need = DIAG_PER_LEVEL if pool == "diag" else PER_LEVEL
        short = [k for k in LEVELS if len(lv[k]) < need]
        if short:
            unavailable[task] = f"level {short[0]} has {len(lv[short[0]])} lines, the {pool} pool needs {need}"
            continue
        per_task[task] = diag_lines(fam, task) if pool == "diag" else split(fam, task, seed)[pool]
    return per_task


def _seeded_blocks(fam: dict, ad: dict, pool: str, tasks, ids, seed) -> dict[str, list[int]]:
    letter, per = POOL_LETTER[pool], int(fam["pools"][pool]["per_task"])
    per_task: dict[str, list[int]] = {}
    for task in tasks:
        if ids:
            per_task[task] = sorted(int(x.rsplit("/", 1)[1]) for x in ids if x.startswith(f"{task}/{letter}/"))
            continue
        if pool == "diag":
            first = 0
        else:
            room = ad["pool_sizes"][letter] - OFFSET_FLOOR - per + 1
            first = OFFSET_FLOOR + random.Random(f"{seed}:{task}:{letter}").randrange(room)
        per_task[task] = list(range(first, first + per))
    return per_task


def draw(fam: dict, ad: dict, pool: str, out, n: int | None = None, tasks: list[str] | None = None) -> dict:
    """Write the pool's files `<task>_s<line>.json` into `out`; returns {written, skipped, unavailable, next_seed}. `n` caps
    the total (tasks taken in turn); `tasks` restricts the tasks. Practice files: a task whose file lacks the lines the pool needs
    (diag: 10 per level; accept / sealed: 50 per level, from the full file) is `unavailable` with the reason. Seed pools: see the
    module doc."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    by_seed = seeded(ad)
    letter = POOL_LETTER[pool] if by_seed else "Q"
    ids = fam["pools"].get(pool, {}).get("ids") if pool != "diag" else None
    seed = None
    if pool != "diag" and not ids:
        seed = pool_seed(fam)
        if seed is None:
            raise SystemExit(f"{fam['name']}: the {pool} pool's seed is blank and the hidden seeds.json has none")
    todo = fam["tasks"] if tasks is None else tasks
    unavailable: dict[str, str] = {}
    if by_seed:
        per_task = _seeded_blocks(fam, ad, pool, todo, ids, seed)
    else:
        per_task = _practice_lines(fam, pool, todo, ids, seed, unavailable)
    order, k = [], 0  # one line of each task in turn, until `n` or nothing is left
    while any(per_task.values()) and (n is None or len(order) < n):
        t = list(per_task)[k % len(per_task)]
        k += 1
        if per_task[t]:
            order.append((t, per_task[t].pop(0)))
    for t, i in order:
        (out / f"{t}_s{i}.json").write_text(json.dumps(dict(task=t, episode_id=f"{t}/{letter}/{i}"), indent=1))
    summary = dict(family=fam["name"], pool=pool, written=len(order), skipped=[], unavailable=unavailable,
                   next_seed=(seed if seed is not None else 0))  # fmt: skip
    (out / "_pool.json").write_text(json.dumps(summary, indent=1))
    return summary
