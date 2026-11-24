"""Episode ids, the practice rows behind them and the reset message (TODO 49.4; from bed_lml.py check_id / practice_row / reset_msg
and bed_simple.ids_of).

An id is "<Task>/<kind>/<n>":
    Q   line n of the task's practice file: a jitter seed (at least `seed_min`) and the two jitter radii; the episode index is 0 and
        the row's seed is the whole seed (the loops' pool)
    P   development: the benchmark's own randomisation with jitter seed 1000, episode n
    W   development, wide: seed 2000, episode n, the robot drawn in 0.30 m and the objects in 0.08 m
    E   the benchmark's own evaluation episode n (jitter seed 0): NEVER run. No code path of the bed makes its start state.

Four layers refuse an evaluation start, each on its own (tests/test_lml_episodes.py tries every one):
    1. the session's runner (`__main__`): `check_id` of every id before a slot or a server is taken;
    2. `check_id` itself (also called by the bed's setup): kind and pool (`LocoManiLabConfig.kinds`), the row's seed and radii;
    3. `reset_msg`: it builds a message from a checked id only, and re-checks the seed it sends against `seed_floor`;
    4. the environment server: it is started with the same floor as an argument and refuses any `jitter seed + episode` under it.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from locomani.bench.locomanilab import families
from locomani.bench.locomanilab.config import PRACTICE_SEED, WIDE, LocoManiLabConfig


class EpisodeRefused(ValueError):
    """An id the bed does not run: the benchmark's own episodes, a pool this run was not given, a bad practice row."""


def parse_id(episode_id: str) -> tuple[str, str, int]:
    parts = episode_id.split("/")
    if len(parts) != 3:
        raise EpisodeRefused(f"{episode_id!r} is not <Task>/<kind>/<n>")
    task, kind, n = parts
    if not (n.isascii() and n.isdigit()):
        raise EpisodeRefused(f"{episode_id}: n must be a plain non-negative integer")
    return task, kind, int(n)


@lru_cache(maxsize=256)
def _lines(path: str) -> tuple[str, ...]:
    return tuple(Path(path).read_text().splitlines())


def practice_row(cfg: LocoManiLabConfig, task: str, n: int) -> dict:
    """Line n of the task's practice file, checked: {seed >= cfg.seed_min, robot > 0, objects > 0}. Refuses (EpisodeRefused) a
    missing file or line, a blank row (a sandbox copy blanks the hidden ones: only the asked line is parsed), a seed under
    `seed_min`, a missing or non-positive radius."""
    f = Path(families.practice_dir(cfg, task)) / f"{task}.jsonl"
    if not f.is_file():
        raise EpisodeRefused(f"no practice file for {task}")
    lines = _lines(str(f))
    if n >= len(lines):
        raise EpisodeRefused(f"{task} has no practice row {n}")
    row = json.loads(lines[n] or "{}")
    if not row:
        raise EpisodeRefused(f"{task}/Q/{n} is a blank row")
    seed = row.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < cfg.seed_min:
        raise EpisodeRefused(f"{task}/Q/{n}: seed {seed!r} is not an integer of at least {cfg.seed_min}")
    for k in ("robot", "objects"):
        v = row.get(k)
        if not isinstance(v, int | float) or isinstance(v, bool) or not v > 0:
            raise EpisodeRefused(f"{task}/Q/{n}: radius {k} {v!r} is missing")
    return row


def check_id(cfg: LocoManiLabConfig, episode_id: str) -> dict | None:
    """Refuse an id that is not of a pool this run may open. -> the practice row of a Q id, None for P and W."""
    cfg.validate_scope()
    task, kind, n = parse_id(episode_id)
    if kind == "E" or kind not in "QPW" or len(kind) != 1:
        raise EpisodeRefused(
            f"{episode_id}: kind {kind!r}: the benchmark's own episodes are the exam and never run here"
        )
    if kind not in cfg.kinds:
        raise EpisodeRefused(f"{episode_id}: this run opens kind(s) {cfg.kinds!r} only (LocoManiLabConfig.kinds)")
    if task not in families.TASKS:
        raise EpisodeRefused(f"{episode_id}: task is not one of the bed's {len(families.TASKS)}")
    return practice_row(cfg, task, n) if kind == "Q" else None


def reset_msg(cfg: LocoManiLabConfig, episode_id: str, cameras: bool) -> tuple:
    """The server message for an episode's reset. Q: episode index 0, the row's seed as the whole seed, its radii."""
    row = check_id(cfg, episode_id)
    task, kind, n = parse_id(episode_id)
    if row is not None:
        msg = ("reset", task, 0, row["seed"], cameras, float(row["robot"]), float(row["objects"]))
    elif kind == "W":
        msg = ("reset", task, n, WIDE["seed"], cameras, WIDE["robot"], WIDE["objects"])
    else:
        msg = ("reset", task, n, PRACTICE_SEED, cameras)
    if msg[2] + msg[3] < cfg.seed_floor():
        raise EpisodeRefused(f"{episode_id}: seed {msg[3]} + episode {msg[2]} is under the floor {cfg.seed_floor()}")
    return msg


def ids_of(spec: str) -> list[str]:
    """'Task/P/0-49,Task/Q/3' or a file of ids -> the ids."""
    if os.path.exists(spec):
        return [s.strip() for s in open(spec) if s.strip()]
    out = []
    for part in spec.split(","):
        task, kind, n = part.split("/")
        if "-" in n:
            lo, hi = n.split("-")
            out += [f"{task}/{kind}/{i}" for i in range(int(lo), int(hi) + 1)]
        else:
            out.append(part)
    return out
