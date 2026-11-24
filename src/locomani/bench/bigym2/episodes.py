"""Episode ids, the seeds behind them, and the refusal of BiGym's evaluation block (TODO 49.5).

An id is "<task>/<pool>/<n>": pool P (practice), A (acceptance), S (sealed), n the index in the pool (config.POOLS gives each pool's
first seed and size). E, the official block, is never an id. The reset message carries the seed; every layer checks it:

    1. `BiGym2Config.validate_scope`   the pools' ranges against the block
    2. the CLI (`__main__`)            `check_id` of every id before a server is started
    3. `check_id`                      pool letter, task, n inside the pool, the seed outside the block
    4. `reset_msg`                     builds a message from a checked id only and re-checks the seed it sends
    5. the server (server.py)          started with the block as arguments, refuses any seed inside it
"""

from __future__ import annotations

import os

from locomani.bench.bigym2.config import DEMO_SEEDS, EVAL_BLOCK, POOLS, SEED_IGNORING, BiGym2Config

PRACTICE_TASKS = (
    "reach_target_single",
    "drawer_top_open",
)  # what the seed package runs; the id check takes any registered name below
BENCHMARK_TASKS = (  # the 20 benchmark tasks (docs/tasks.md of the clone)
    "reach_target_single", "reach_target_multi_modal", "reach_target_dual", "move_plate", "move_two_plates", "flip_cup",
    "flip_cutlery", "stack_blocks", "dishwasher_close", "dishwasher_load_cups", "dishwasher_load_cutlery",
    "dishwasher_load_plates", "drawer_top_open", "drawer_top_close", "pick_box", "put_cups", "saucepan_to_hob",
    "sandwich_remove", "wall_cupboard_open", "wall_cupboard_close",
)  # fmt: skip


class EpisodeRefused(ValueError):
    """An id the bed does not run: the evaluation block, a pool this run was not given, a bad index."""


def in_eval_block(seed: int) -> bool:
    return EVAL_BLOCK[0] <= seed < EVAL_BLOCK[1]


def parse_id(episode_id: str) -> tuple[str, str, int]:
    parts = episode_id.split("/")
    if len(parts) != 3:
        raise EpisodeRefused(f"{episode_id!r} is not <task>/<pool>/<n>")
    task, pool, n = parts
    if not (n.isascii() and n.isdigit()):
        raise EpisodeRefused(f"{episode_id}: n must be a plain non-negative integer")
    return task, pool, int(n)


def seed_of(episode_id: str) -> int:
    """The seed of an id (not checked: `check_id` does that)."""
    _, pool, n = parse_id(episode_id)
    if pool not in POOLS:
        raise EpisodeRefused(f"{episode_id}: pool {pool!r}")
    return POOLS[pool][0] + n


def check_id(cfg: BiGym2Config, episode_id: str) -> int:
    """Refuse an id that is not of a pool this run may open or whose seed is in the evaluation block. -> its seed."""
    cfg.validate_scope()
    task, pool, n = parse_id(episode_id)
    if pool == "E" or pool not in POOLS or len(pool) != 1:
        raise EpisodeRefused(f"{episode_id}: pool {pool!r}: BiGym's evaluation seeds {EVAL_BLOCK} are never run here")
    if pool not in cfg.pools:
        raise EpisodeRefused(f"{episode_id}: this run opens pool(s) {cfg.pools!r} only (BiGym2Config.pools)")
    first, size = POOLS[pool]
    if n >= size:
        raise EpisodeRefused(f"{episode_id}: pool {pool} has {size} episodes")
    seed = first + n
    if in_eval_block(seed) or DEMO_SEEDS[0] <= seed < DEMO_SEEDS[1]:
        raise EpisodeRefused(f"{episode_id}: seed {seed} is an evaluation or demonstration seed")
    return seed


def reset_msg(cfg: BiGym2Config, episode_id: str) -> tuple:
    """The server message for an episode's reset."""
    seed = check_id(cfg, episode_id)
    task, _, _ = parse_id(episode_id)
    if in_eval_block(seed):
        raise EpisodeRefused(f"{episode_id}: seed {seed} is in the evaluation block")
    return ("reset", task, seed)


def ignores_seed(task: str) -> bool:
    """Tasks whose reset does not consume the seed (their pools repeat one start; see config.SEED_IGNORING)."""
    return task in SEED_IGNORING


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
