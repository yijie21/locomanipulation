"""Episode ids, the seeds behind them, and the refusal of HumanoidArena's evaluation episodes (TODO 49.5b).

An id is "<task>/<pool>/<n>": task one of pp_box, double_desk, open_door; pool P (practice), A (acceptance), S (sealed); n the index
in the pool (config.POOLS: group = first group + n // 20, repeat = n % 20). The episode seed is the benchmark's own formula
(`seed_of`). E, the evaluation groups {0, 1, 2}, is never an id. Every layer checks:

    1. `HumanoidArenaConfig.validate_scope`   the pools' groups against {0, 1, 2}
    2. the CLI (`__main__`)                   `check_id` of every id before a server is started
    3. `check_id`                             pool letter, task, n inside the pool, the derived seed against the 60
    4. `reset_msg`                            builds a message from a checked id only and re-checks the group and the seed it sends
    5. the server (server.py)                 derives the seed itself and refuses a group of {0, 1, 2} or a seed of the 60
"""

from __future__ import annotations

import hashlib
import os

from locomani.bench.humanoidarena.config import EVAL_GROUPS, EVAL_REPEATS, PER_GROUP, POOLS, HumanoidArenaConfig

GYM = {  # task key -> the benchmark's registered task name (what its seed formula hashes)
    "pp_box": "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby",
    "double_desk": "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody",
    "open_door": "Isaac-Move-Open-Door-G129-Dex3-Wholebody",
}
TASKS = tuple(GYM)


class EpisodeRefused(ValueError):
    """An id the bed does not run: an evaluation episode, a pool this run was not given, a bad index."""


def derive_seed(task: str, group: int, repeat: int) -> int:
    """The benchmark's formula: low 31 bits of the last four bytes of sha256("<gym task>|<group>|<repeat>")."""
    payload = f"{GYM[task]}|{int(group)}|{int(repeat)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], "big", signed=False) & 0x7FFFFFFF


def eval_seeds(task: str) -> frozenset[int]:
    """The 60 derived evaluation seeds of a task."""
    return frozenset(derive_seed(task, g, r) for g in EVAL_GROUPS for r in range(EVAL_REPEATS))


def collides(task: str, seed: int) -> bool:
    return seed in eval_seeds(task)


def parse_id(episode_id: str) -> tuple[str, str, int]:
    parts = episode_id.split("/")
    if len(parts) != 3:
        raise EpisodeRefused(f"{episode_id!r} is not <task>/<pool>/<n>")
    task, pool, n = parts
    if not (n.isascii() and n.isdigit()):
        raise EpisodeRefused(f"{episode_id}: n must be a plain non-negative integer")
    return task, pool, int(n)


def group_repeat(episode_id: str) -> tuple[int, int]:
    """(group, repeat) of an id (not checked: `check_id` does that)."""
    _, pool, n = parse_id(episode_id)
    if pool not in POOLS:
        raise EpisodeRefused(f"{episode_id}: pool {pool!r}")
    return POOLS[pool][0] + n // PER_GROUP, n % PER_GROUP


def seed_of(episode_id: str) -> int:
    task, _, _ = parse_id(episode_id)
    g, r = group_repeat(episode_id)
    return derive_seed(task, g, r)


def check_id(cfg: HumanoidArenaConfig, episode_id: str) -> int:
    """Refuse an id that is not of a pool this run may open or whose seed is one of the 60. -> its seed."""
    cfg.validate_scope()
    task, pool, n = parse_id(episode_id)
    if task not in GYM:
        raise EpisodeRefused(f"{episode_id}: task {task!r} is not one of {TASKS}")
    if pool == "E" or pool not in POOLS or len(pool) != 1:
        raise EpisodeRefused(f"{episode_id}: pool {pool!r}: the evaluation groups {EVAL_GROUPS} are never run here")
    if pool not in cfg.pools:
        raise EpisodeRefused(f"{episode_id}: this run opens pool(s) {cfg.pools!r} only (HumanoidArenaConfig.pools)")
    first, groups = POOLS[pool]
    if n >= groups * PER_GROUP:
        raise EpisodeRefused(f"{episode_id}: pool {pool} has {groups * PER_GROUP} episodes")
    g, r = group_repeat(episode_id)
    seed = derive_seed(task, g, r)
    if g in EVAL_GROUPS or collides(task, seed):
        raise EpisodeRefused(f"{episode_id}: group {g} / seed {seed} is an evaluation episode")
    return seed


def reset_msg(cfg: HumanoidArenaConfig, episode_id: str) -> tuple:
    """The server message for an episode's reset."""
    seed = check_id(cfg, episode_id)
    task, _, _ = parse_id(episode_id)
    g, r = group_repeat(episode_id)
    if g in EVAL_GROUPS or collides(task, seed):
        raise EpisodeRefused(f"{episode_id}: group {g} / seed {seed} is an evaluation episode")
    return ("reset", task, g, r)


def ids_of(spec: str) -> list[str]:
    """'task/P/0-49,task/A/3' or a file of ids -> the ids."""
    if os.path.exists(spec):
        return [s.strip() for s in open(spec) if s.strip()]
    out = []
    for part in spec.split(","):
        task, pool, n = part.split("/")
        if "-" in n:
            lo, hi = n.split("-")
            out += [f"{task}/{pool}/{i}" for i in range(int(lo), int(hi) + 1)]
        else:
            out.append(part)
    return out
