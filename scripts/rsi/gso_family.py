#!/usr/bin/env python3
"""Write the task families of a `logistics_room_1030_s<seed>` scene instance (user Q13, 2026-11-03; the pack line, user
2026-11-05): one task per object, generated from the scene file instead of written by hand.

- `gso_grasp_s<seed>`: one hand; every `semantic.graspable == "single"` object; goal `held(X)` + `lifted(X, 0.10, 60)`
  (user Q23: held and 10 cm off the platform); the seed is the empty skills package (`seeds/atomic`), as grasp_atomic's.
- `gso_bimanual_s<seed>`: two hands; every `bimanual` object and the carton; goal `held(X)` + `lifted(X, 0.20, 60)`,
  judged over the round's last 5 s (box_bimanual's `hold_s` / `judge_s`); the skills are the main library's, whose
  two-hand hug pick (`skills/bimanual.py`) is the box line's seed ported on 2026-11-05 (user Q18: one library).
- `gso_pack_s<seed>` / `gso_pack_bimanual_s<seed>` (user 2026-11-05): every single-hand / two-hand object X of the
  instance; goal `in(X, nor_cal_box)` + `held(nor_cal_box)` + `lifted(nor_cal_box, 0.20, 60)`: X put into the carton,
  then the carton lifted with both hands, judged over the last 5 s; the main library's `skills/pack.py` chain. Each
  episode stands alone (the carton starts empty); its start moves X and the carton only.
- `gso_all_s<seed>` (user 2026-11-06): every task of the four families in one family, so one inner loop iteration
  diagnoses, changes and gates all the skills together on one instance (21 tasks on s0-s9); the main library as the
  start, pools of PER_TASK starts per task. Each two-hand or pack row carries its own `hold_s` / `judge_s`
  (family.timing), so a one-hand pick is judged as in `gso_grasp` and the rest over the last 5 s.

Every family's pools (user 2026-11-24, every logistics room family): PER_TASK starts per task, 3 diagnostic / 10
acceptance / 10 sealed by default (`--per-task`; the acceptance pool at least gate.STAGE1 + 1), and the stops STOPS: the
incumbent wins 95 % of the acceptance pool, 20 h, 5 iterations.

Every task row carries a `placement`: the task object within REACH (the scene's `gso.reach_depth_m`) of an approach edge
(D13: the middle of the platform is out of reach). The pack rows add per-object conditions (`objects`): the carton on
the two-hand hug's stance band with its long side along its edge (`gso.carton_placement`, the scene builder's
CARTON_PLACEMENT, box_bimanual's `box_long_side`), and a two-hand X with its wide side along its edge, so the hug
squeezes it across that side and it goes into the carton the way it fits (the carton's long inner side lies along the
same edge). Every object a task hugs (a two-hand X, the carton) also asks for HANDS_CLEAR_M free beside each squeezed
wall and in front of it (`hands_clear_m`, start.hand_corridors; user 2026-11-06: a neighbour there stopped both hands
15 cm short of the pre-grasp), and every row moves the other objects of the platform too (`shuffle_others`,
start.sample_start; user 2026-11-06: with the nine others fixed, 13 of s0's 21 tasks found no start). Pool seeds: 44/45/46/47 x 10^4 per family, +10^5 per pool, disjoint from every other family's (D7).

    PYTHONPATH=src $VENV_SIM_PY scripts/rsi/gso_family.py outputs/scene_discovery/logistics_room_1030_s0
        [--families-dir scripts/rsi/families]

A hand-picked scene (`logistics_room_gso_single` / `_bimanual`, user 2026-11-17; prepare_logistics_scene.py --picks)
gets one family instead: `gso_single_grasp` (every picked one-hand object) or `gso_bimanual_lift` (every picked
two-hand object and the carton), goal `held(X)` + `lifted(X, 0.20, 60)` (`--lift` another height) judged over the
last 5 s for both, the main skill library as the start, the same pools and stops; files `gso_single_grasp.json` +
`gso_single_grasp_tasks.jsonl`.

Writes `<families-dir>/gso_grasp_s<seed>.json`, `gso_bimanual_s<seed>.json`, `gso_pack_s<seed>.json`,
`gso_pack_bimanual_s<seed>.json`, `gso_all_s<seed>.json` and the shared `gso_s<seed>_tasks.jsonl`. A family's `sandbox.keep` lists its file,
the tasks file and the plan, as the existing families do; `sandbox.seed` null keeps the main skill library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FAMILIES = HERE / "families"
# family kind -> family name stem; the object kind each draws its tasks from; what the goal asks
KINDS = {"single": "gso_grasp", "bimanual": "gso_bimanual", "pack": "gso_pack", "pack_bimanual": "gso_pack_bimanual"}
OBJECT_KIND = {"single": "single", "bimanual": "bimanual", "pack": "single", "pack_bimanual": "bimanual"}
PLANS = {
    "gso_grasp": "scripts/rsi/plans/gso_grasp.md",
    "gso_bimanual": "scripts/rsi/plans/gso_bimanual.md",
    "gso_pack": "scripts/rsi/plans/gso_pack.md",
    "gso_pack_bimanual": "scripts/rsi/plans/gso_pack.md",
    "gso_all": "scripts/rsi/plans/gso_all.md",
    "gso_pack_all": "scripts/rsi/plans/gso_pack_all.md",
}
SEEDS = {"gso_grasp": "scripts/rsi/seeds/atomic", "gso_bimanual": None, "gso_pack": None, "gso_pack_bimanual": None,
         "gso_all": None, "gso_pack_all": None}  # fmt: skip
POOL_SEEDS = {"gso_grasp": 440_000, "gso_bimanual": 450_000, "gso_pack": 460_000, "gso_pack_bimanual": 470_000,
              "gso_all": 480_000, "gso_pack_all": 490_000}  # fmt: skip
POOLS = ("diag", "accept", "sealed")  # pool seeds: POOL_SEEDS[family] + 0 / 100000 / 200000 in this order
ALL = "gso_all"
PACK_ALL = "gso_pack_all"
PACK_KINDS = ("pack", "pack_bimanual")
# the two-hand and pack families' success timing (box_bimanual's): the end of the plan, 2 s still, judged over 5 s
TIMING = {"hold_s": 2.0, "judge_s": 5.0}
LIFT_M = {"gso_grasp": 0.10, "gso_bimanual": 0.20, "gso_pack": 0.20, "gso_pack_bimanual": 0.20}
TILT_DEG = 60
# the hand-picked test scenes (user 2026-11-17; prepare_logistics_scene.py --picks): one family each, every object
# lifted PICKED_LIFT_M and held, judged over the last 5 s, from the main skill library; 0.20 m since 2026-11-18 (user;
# 0.30 m before), `--lift` another height
PICKED_LIFT_M = 0.20
# every family's pools: starts per task, the default of `--per-task` (user 2026-11-18 for the picked families: by the
# task count; with a fixed 30 / 40 / 50 the 20 one-hand tasks had 1-2 diagnostic and 2 acceptance starts each, so a
# change that helped one object could never win the gate's net 3). The family file records the numbers it was written
# with (`pool_per_task`) and each pool's `n`; the sandbox, the loop and sealed.py read `n` from there.
# 3 / 10 / 10 for every logistics room family since 2026-11-24 (user; the picked families 3 / 4 / 5, the drawn
# instances a fixed 30 / 40 / 50, gso_all 2 / 2 / 5 and gso_pack_all 3 / 4 / 5 before): the diagnostic pool only feeds
# the Reviewer failures and is topped up by the redraw, so it stays small; the acceptance pool decides the gate, where
# a 32-start pool let two or three chance flips of a contact round decide the verdict; the sealed pool is run once and
# sets the report's interval.
PER_TASK = {"diag": 3, "accept": 10, "sealed": 10}


def gate_stage1() -> int:
    """gate.py's STAGE1, read from its source (importing gate.py here would put a module named `gate` in sys.modules,
    a name the scalebfm experiment's scripts share): an acceptance pool must hold more starts than the gate's stage 1."""
    import ast

    tree = ast.parse((HERE / "gate.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STAGE1" for t in node.targets):
            return int(ast.literal_eval(node.value))
    raise SystemExit("gate.py has no STAGE1")


def pools(pool_seed: int, n_tasks: int, per_task: dict[str, int] | None = None) -> dict:
    """A family's pools: `per_task[pool]` (default PER_TASK) starts per task, the acceptance pool at least
    gate.STAGE1 + 1 starts; seeds `pool_seed` + 0 / 100000 / 200000."""
    per_task = per_task or PER_TASK
    floor = {"accept": gate_stage1() + 1}
    return {
        pool: {"seed": pool_seed + i * 100_000, "n": max(int(per_task[pool]) * n_tasks, floor.get(pool, 0))}
        for i, pool in enumerate(POOLS)
    }


def parse_per_task(items: list[str] | None) -> dict[str, int]:
    """`--per-task diag=3 accept=10 sealed=10` (any of them; the rest PER_TASK) -> {pool: starts per task}."""
    out = dict(PER_TASK)
    for item in items or []:
        key, _, value = item.partition("=")
        if key not in out or not value.isdigit() or int(value) < 1:
            raise SystemExit(f"--per-task {item!r}: want POOL=N with POOL one of {sorted(out)} and N >= 1")
        out[key] = int(value)
    return out


# where the robot starts in the picked scenes (user 2026-11-17): 0.5-4 m from the task object, turned towards it
# within +-80 deg (uniform)
PICKED_SPAWN = {"rooms": "all", "distance_m": [0.5, 4.0], "face_target_deg": [-80, 80]}
PICKED = {
    "single": {"name": "gso_single_grasp", "kind": "single", "plan": "scripts/rsi/plans/gso_single_grasp.md",
               "base_plan": "scripts/rsi/plans/gso_grasp.md", "pool_seed": 9_000_000,
               "skills_note": "the main skill library as it stands: its one-hand pick (skills/pick.py); sandbox.seed null"},
    "bimanual": {"name": "gso_bimanual_lift", "kind": "bimanual", "plan": "scripts/rsi/plans/gso_bimanual_lift.md",
                 "base_plan": "scripts/rsi/plans/gso_bimanual.md", "pool_seed": 9_500_000,
                 "skills_note": "the main skill library as it stands: its two-hand hug pick (skills/bimanual.py); sandbox.seed null"},
}  # fmt: skip
CARTON = "nor_cal_box"
# where the pack rows put the carton when the scene file has no `gso.carton_placement` (an instance built before
# 2026-11-05): prepare_logistics_scene.CARTON_PLACEMENT, the hug's stance band, long side along the edge
CARTON_BAND = {"depth_m": [0.18, 0.31], "yaw_deg": [85, 90]}
ALONG_EDGE = {True: [85, 90], False: [0, 5]}  # the box x axis along the edge when it is the wide one, else across
# free width beside each wall a hug squeezes (and in front of the box, from the edge): the hug's open jaw centre comes in
# FLANK + CLEAR + PRE_OUT = 0.13 m outside the wall (skills/bimanual.py, 2026-11-05) plus ~5 cm of hand; the 11-06
# probe (s0, gso_0945 beside the carton) missed by 15 cm with neighbours there and by 1-2 cm with the platform clear
HANDS_CLEAR_M = 0.18
# when a family's inner loop stops (loop.py resolve_settings): the incumbent wins 95 % of the acceptance pool, the run
# has been going 20 h, or it has taken 5 iterations, whichever comes first (user 2026-11-24, every logistics room
# family; 80 % and 5 h before, user 2026-11-06: the main library already wins 87 % of the picked families' acceptance
# pools, the written-back incumbents of gso-alt-a, so an 80 % line stopped a run at its first gate; 15 h until the
# evening of 11-24). The hours are the inner loop's own cap (modes 1 and 2); in mode 3 alternate.py's --max-hours (also
# 20) counts for both families together.
STOPS = {"stop_rate": 0.95, "max_wall_h": 20.0, "max_run_iters": 5}

# the sandbox block every scene RSI family carries (box_bimanual.json, grasp_atomic.json): what the archive keeps,
# the src import closure, what is stripped; only the family's own files differ
KEEP_COMMON = [
    "Makefile", "pyproject.toml", ".gitignore", ".gitattributes", "assets", "configs", "scripts/paths.sh",
    "scripts/rsi/loop.py", "scripts/rsi/batch.py", "scripts/rsi/digest.py", "scripts/rsi/family.py", "scripts/rsi/gate.py",
    "scripts/rsi/diag_pool.py", "scripts/rsi/try.py", "scripts/rsi/scene_round.py", "scripts/rsi/sealed.py",
    "scripts/rsi/claim_check.py", "scripts/rsi/failure_class.py", "scripts/rsi/graspability.py", "scripts/rsi/body.py",
    "scripts/rsi/choose.py", "scripts/rsi/records.py", "scripts/rsi/isolate.py", "scripts/rsi/design.py",
    "scripts/rsi/prep.py", "scripts/rsi/past.py", "scripts/rsi/experience", "scripts/rsi/rules.md", "scripts/rsi/prompts",
    "tests/conftest.py", "tests/test_scene_task_runner.py",
]  # fmt: skip
CLOSURE = [
    "loop", "gate", "digest", "sealed", "claim_check", "family", "diag_pool", "batch", "scene_round",
    "locomani.tasks.scene.world", "locomani.tasks.scene.episode", "locomani.tasks.scene.navigate",
    "locomani.tasks.scene.start", "locomani.backends.scalebfm.runtime", "locomani.backends.scalebfm.motor",
    "locomani.tasks.scene.graspability", "locomani.tasks.scene.body", "locomani.tasks.scene.choose",
    "locomani.tasks.scene.stage", "locomani.tasks.scene.prep", "locomani.tasks.scene.past",
]  # fmt: skip
STRIP = [
    "configs/experiments", "configs/collection", "configs/deployment", "configs/eval", "configs/sweeps",
    "configs/injections", "configs/cases", "configs/plans", "configs/scene_discovery",
]  # fmt: skip
SCENE_EXCLUDE = ["lint.json", "settle.json", "upright.json", "feasibility", "preview"]


def rows_for(meta: dict, kind: str, reach: float, lift: float | None = None, timed: bool = False) -> list[dict]:
    """The task rows of one family kind (KINDS) from the scene file: `held` + `lifted` on each object of its kind, or
    -- the pack kinds -- `in(X, carton)` + `held` + `lifted` of the carton for each such X but the carton itself.
    `lift` (m) replaces the kind's LIFT_M; `timed` judges a one-hand row over the last 5 s too (the picked scenes)."""
    room = meta["room"]["rooms"] and next(iter(meta["room"]["rooms"]))
    pack = kind.startswith("pack")
    carton_band = (meta.get("gso") or {}).get("carton_placement") or CARTON_BAND
    lift = LIFT_M[KINDS[kind]] if lift is None else lift
    band = {"depth_m": [0.0, round(reach, 3)]}
    out = []
    for key, o in meta["objects"].items():
        sem = o.get("semantic") or {}
        if not o.get("free") or sem.get("graspable") != OBJECT_KIND[kind] or (pack and key == CARTON):
            continue
        p = sem["phrase"]
        placement: dict = {**band, "shuffle_others": True}
        if kind == "single" and timed:
            prompts = [f"pick up the {p} and hold it {int(round(lift * 100))} cm above the platform",
                       f"grab the {p}, lift it off the platform and hold it there"]  # fmt: skip
            tid, goal = f"pick_{key}", [f"held({key})", f"lifted({key}, {lift:.2f}, {TILT_DEG})"]
        elif kind == "single":
            prompts = [f"pick up the {p}", f"grab the {p} and lift it off the platform"]
            tid, goal = f"pick_{key}", [f"held({key})", f"lifted({key}, {lift:.2f}, {TILT_DEG})"]
        elif kind == "bimanual":
            prompts = [f"lift the {p} {int(round(lift * 100))} cm off the platform with both hands and hold it",
                       f"pick up the {p} with both hands and hold it level"]  # fmt: skip
            tid, goal = f"lift_{key}", [f"held({key})", f"lifted({key}, {lift:.2f}, {TILT_DEG})"]
            placement["hands_clear_m"] = HANDS_CLEAR_M
        else:
            hands = "both hands" if kind == "pack_bimanual" else "one hand"
            prompts = [f"put the {p} into the cardboard box with {hands}, then lift the box with both hands",
                       f"pack the {p} into the open box and pick the box up"]  # fmt: skip
            tid = f"pack_{key}"
            goal = [f"in({key}, {CARTON})", f"held({CARTON})", f"lifted({CARTON}, {lift:.2f}, {TILT_DEG})"]
            own = {CARTON: {**carton_band, "hands_clear_m": HANDS_CLEAR_M}}
            if kind == "pack_bimanual":
                half = o["box"]["half_extent"]
                own[key] = {
                    **band,
                    "yaw_deg": ALONG_EDGE[float(half[0]) >= float(half[1])],
                    "hands_clear_m": HANDS_CLEAR_M,
                }
            placement["objects"] = own
        timing = (
            {} if kind == "single" and not timed else TIMING
        )  # per row too: gso_all mixes the kinds (family.timing)
        out.append({"id": tid, "goal": goal, "prompts": prompts, "room": room, "placement": placement, **timing})
    return out


def carton_mass(meta: dict) -> float:
    """The carton's mass in the scene file (0.3 kg since 2026-11-07, prepare_logistics_scene.carton_mass); 1.2, the
    pack's own, when the file does not say."""
    return float(((meta["objects"].get(CARTON) or {}).get("semantic") or {}).get("mass_kg", 1.2))


def _about(kind: str, seed: int, reach: float, names: dict, carton_kg: float = 1.2) -> str:
    stem, lift = KINDS[kind], LIFT_M[KINDS[kind]]
    objects = "; objects: " + ", ".join(f"{k} = {v}" for k, v in names.items())
    if kind in ("single", "bimanual"):
        hands = "one hand" if kind == "single" else "both hands"
        return (
            f"logistics_room_1030 (user 2026-11-03): held(X) and lifted(X, {lift:.2f}, {TILT_DEG}) with {hands} on the "
            f"Google Scanned Objects drawn with seed {seed} onto the 0.371 m pallet platform (catalog.json {kind} "
            f"objects{f', plus the {carton_kg:g} kg open carton' if kind == 'bimanual' else ''}); every task object starts within "
            f"{reach} m of a platform edge"
            + ("" if kind == "single" else f", {HANDS_CLEAR_M} m free beside the walls the hug squeezes and in front")
            + ", the robot anywhere in the room it can walk from"
            + objects
        )
    hands = "two-hand" if kind == "pack_bimanual" else "single-hand"
    return (
        f"logistics_room_1030 pack line (user 2026-11-05): in(X, {CARTON}), then held({CARTON}) and lifted({CARTON}, "
        f"{lift:.2f}, {TILT_DEG}) with both hands -- X, a {hands} Google Scanned Object of the instance drawn with seed "
        f"{seed} (every drawn object fits the carton's 0.43 x 0.30 x 0.33 m inside), is put into the {carton_kg:g} kg open carton "
        f"on the 0.371 m pallet platform, which is then lifted; each episode starts with the carton empty, X within "
        f"{reach} m of a platform edge{' with its wide side along it' if kind == 'pack_bimanual' else ''}, the carton "
        f"0.18-0.31 m in from an edge with its long side along it, {HANDS_CLEAR_M} m free beside the walls each hug "
        f"squeezes and in front, the robot anywhere in the room it can walk from"
        + objects
    )  # fmt: skip


SKILLS_NOTE = {
    "gso_grasp": "the incumbent's skills; the seed was empty (`__init__.py`: plan_for covers no goal)",
    "gso_bimanual": "the main skill library as it stands: its two-hand hug pick (skills/bimanual.py, the box line's "
                    "seed ported on 2026-11-05; sandbox.seed null)",
    "gso_pack": "the main skill library as it stands: skills/pack.py (into.place_into with one hand, then "
                "bimanual.pick of the carton; sandbox.seed null)",
    "gso_pack_bimanual": "the main skill library as it stands: skills/pack.py (bimanual.place_into with both hands, "
                         "then bimanual.pick of the carton; sandbox.seed null)",
    "gso_all": "the main skill library as it stands: the one-hand pick (skills/pick.py), the two-hand hug pick "
               "(skills/bimanual.py) and the pack chain (skills/pack.py) together; sandbox.seed null",
    "gso_pack_all": "the main skill library as it stands: skills/pack.py, the chain of the one-hand (into.place_into) "
                    "or two-hand (bimanual.place_into) put-in and bimanual.pick of the carton; sandbox.seed null",
}  # fmt: skip


def family(
    name: str, kind: str, scene_rel: str, seed: int, tasks_file: str, rows: list[dict], meta: dict, per_task=None
) -> dict:
    stem = KINDS[kind]
    names = {r["id"].split("_", 1)[1]: meta["objects"][r["id"].split("_", 1)[1]]["semantic"]["phrase"] for r in rows}
    return {
        "name": name,
        "about": _about(kind, seed, rows[0]["placement"]["depth_m"][1], names, carton_mass(meta)),
        "scene": scene_rel,
        "scene_note": (
            "in a simulated logistics room (a scanned warehouse room: storage racks along the walls, one 1.2 x 0.9 m "
            "pallet platform 0.371 m high with an open cardboard box and ten scanned household objects on it)"
        ),  # fmt: skip
        "tasks_file": tasks_file,
        "tasks": [r["id"] for r in rows],
        "spawn": {"rooms": "all", "distance_m": [0.5, None]},
        **(TIMING if kind != "single" else {}),
        **STOPS,
        "pool_per_task": dict(per_task or PER_TASK),
        "pools": pools(POOL_SEEDS[stem], len(rows), per_task),
        "plan": PLANS[stem],
        "skills_note": SKILLS_NOTE[stem],
        "sandbox": {
            "seed": SEEDS[stem],
            "keep": KEEP_COMMON + [PLANS[stem], f"scripts/rsi/families/{name}.json", tasks_file],
            "closure": CLOSURE,
            "strip": STRIP,
            "scene_exclude": SCENE_EXCLUDE,
        },
    }


def family_all(name: str, scene_rel: str, seed: int, tasks_file: str, rows: dict, meta: dict, per_task=None) -> dict:
    """gso_all: the four kinds' rows in one family, in KINDS order; no family-level timing (each row has its own)."""
    ids = [r["id"] for kind in KINDS for r in rows[kind]]
    counts = ", ".join(f"{len(rows[k])} {KINDS[k]}" for k in KINDS)
    keys = dict.fromkeys(r["id"].split("_", 1)[1] for kind in KINDS for r in rows[kind])
    objects = "; objects: " + ", ".join(f"{k} = {meta['objects'][k]['semantic']['phrase']}" for k in keys)
    return {
        "name": name,
        "about": (
            f"logistics_room_1030, the whole instance in one family (user 2026-11-06): every task of {KINDS['single']}, "
            f"{KINDS['bimanual']}, {KINDS['pack']} and {KINDS['pack_bimanual']}_s{seed} ({counts}) on the Google "
            f"Scanned Objects drawn with seed {seed} onto the 0.371 m pallet platform, each with its own family's goal, "
            f"placement and judging (a one-hand pick at the end of the plan, the rest over the last 5 s), so one "
            f"iteration evolves all the skills together" + objects
        ),  # fmt: skip
        "scene": scene_rel,
        "scene_note": (
            "in a simulated logistics room (a scanned warehouse room: storage racks along the walls, one 1.2 x 0.9 m "
            "pallet platform 0.371 m high with an open cardboard box and ten scanned household objects on it)"
        ),  # fmt: skip
        "tasks_file": tasks_file,
        "tasks": ids,
        "spawn": {"rooms": "all", "distance_m": [0.5, None]},
        **STOPS,
        "pool_per_task": dict(per_task or PER_TASK),
        "pools": pools(POOL_SEEDS[ALL], len(ids), per_task),
        "plan": PLANS[ALL],
        "skills_note": SKILLS_NOTE[ALL],
        "sandbox": {
            "seed": SEEDS[ALL],
            # every gso plan: gso_all.md points at the three kinds' plans for their measurements
            "keep": KEEP_COMMON + [*dict.fromkeys(PLANS.values()), f"scripts/rsi/families/{name}.json", tasks_file],
            "closure": CLOSURE,
            "strip": STRIP,
            "scene_exclude": SCENE_EXCLUDE,
        },
    }


def kind_name(kind: str) -> str:
    return KINDS[kind]


def family_pack_all(
    name: str, scene_rel: str, seed: int, tasks_file: str, rows: dict, meta: dict, per_task=None
) -> dict:
    """gso_pack_all (user 2026-11-07): the two pack kinds' rows in one family -- every object but the carton goes into
    the carton, which is then lifted --, judged by the gate on the progress stage (gso_pack_all.md D15)."""
    ids = [r["id"] for kind in PACK_KINDS for r in rows[kind]]
    keys = dict.fromkeys(r["id"].split("_", 1)[1] for kind in PACK_KINDS for r in rows[kind])
    objects = "; objects: " + ", ".join(f"{k} = {meta['objects'][k]['semantic']['phrase']}" for k in keys)
    return {
        "name": name,
        "about": (
            f"logistics_room_1030, the whole pack task on one instance (user 2026-11-07): every task of {KINDS['pack']} "
            f"and {KINDS['pack_bimanual']}_s{seed} ({len(rows['pack'])} one-hand, {len(rows['pack_bimanual'])} two-hand) "
            f"on the Google Scanned Objects drawn with seed {seed} onto the 0.371 m pallet platform: pick the object, "
            f"carry it to the open carton, put it in, then lift the carton with both hands (judged over the last 5 s). "
            f"The gate compares the two versions start by start on the round's progress stage (0 none, 1 picked, 2 over "
            f"the opening, 3 in the carton, 4 the goal), so a change that carries starts further is accepted before "
            f"any start completes the chain" + objects
        ),  # fmt: skip
        "scene": scene_rel,
        "scene_note": (
            "in a simulated logistics room (a scanned warehouse room: storage racks along the walls, one 1.2 x 0.9 m "
            "pallet platform 0.371 m high with an open cardboard box and ten scanned household objects on it)"
        ),  # fmt: skip
        "tasks_file": tasks_file,
        "tasks": ids,
        "spawn": {"rooms": "all", "distance_m": [0.5, None]},
        **STOPS,
        **TIMING,
        "pool_per_task": dict(per_task or PER_TASK),
        "pools": pools(POOL_SEEDS[PACK_ALL], len(ids), per_task),
        "plan": PLANS[PACK_ALL],
        "skills_note": SKILLS_NOTE[PACK_ALL],
        "sandbox": {
            "seed": SEEDS[PACK_ALL],
            # every gso plan: gso_pack_all.md points at gso_pack.md and gso_bimanual.md for their measurements
            "keep": KEEP_COMMON + [*dict.fromkeys(PLANS.values()), f"scripts/rsi/families/{name}.json", tasks_file],
            "closure": CLOSURE,
            "strip": STRIP,
            "scene_exclude": SCENE_EXCLUDE,
        },
    }


def write_picked(
    scene_dir: Path,
    meta: dict,
    families_dir: Path = FAMILIES,
    lift: float = PICKED_LIFT_M,
    per_task: dict[str, int] | None = None,
) -> dict:
    """The one family of a hand-picked scene (`gso.picks`, prepare_logistics_scene.py --picks): gso_single_grasp or
    gso_bimanual_lift and its tasks file; returns {family name: task count}."""
    gso = meta["gso"]
    which = gso["picks"]
    spec = PICKED[which]
    name, kind = spec["name"], spec["kind"]
    reach = float(gso["reach_depth_m"])
    scene_rel = (
        str(scene_dir.resolve().relative_to(REPO)) if scene_dir.resolve().is_relative_to(REPO) else str(scene_dir)
    )
    tasks_file = f"scripts/rsi/families/{name}_tasks.jsonl"
    rows = rows_for(meta, kind, reach, lift=lift, timed=True)
    if not rows:
        raise SystemExit(f"{scene_dir}: no {kind} object for {name}")
    if kind == "bimanual" and not any(r["id"] == f"lift_{CARTON}" for r in rows):
        raise SystemExit(f"{CARTON} is not a bimanual object of {scene_dir}")
    families_dir.mkdir(parents=True, exist_ok=True)
    (families_dir / Path(tasks_file).name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    names = {r["id"].split("_", 1)[1]: meta["objects"][r["id"].split("_", 1)[1]]["semantic"]["phrase"] for r in rows}
    hands = "one hand" if kind == "single" else "both hands"
    n_obj = sum(1 for k, o in meta["objects"].items() if o.get("free") and k != CARTON)
    fam = {
        "name": name,
        "about": (
            f"{gso['scene']} (user 2026-11-17): held(X) and lifted(X, {lift:.2f}, {TILT_DEG}) with {hands}, judged "
            f"over the round's last {TIMING['judge_s']:g} s, on the {len(rows)} {'objects' if kind == 'single' else 'objects and the carton'}"
            f" picked by hand (assets/mujoco_scanned_objects/scene_picks.json, catalog.json {kind} objects of the training "
            f"split{f', plus the {carton_mass(meta):g} kg open carton' if kind == 'bimanual' else ''}) on the 0.371 m "
            f"pallet platform, placed with seed {gso['seed']}; every task object starts within {reach} m of a platform "
            f"edge" + ("" if kind == "single" else f", {HANDS_CLEAR_M} m free beside the walls the hug squeezes and in front")
            + ", the robot 0.5-4 m away where it can walk from, turned towards the task object within 80 deg; objects: " + ", ".join(f"{k} = {v}" for k, v in names.items())
        ),
        "scene": scene_rel,
        "scene_note": (
            "in a simulated logistics room (a scanned warehouse room: storage racks along the walls, one 1.2 x 0.9 m "
            f"pallet platform 0.371 m high with {'an open cardboard box and ' if CARTON in meta['objects'] else ''}"
            f"{n_obj} scanned household objects on it)"
        ),
        "tasks_file": tasks_file,
        "tasks": [r["id"] for r in rows],
        "spawn": PICKED_SPAWN,
        **TIMING,
        **STOPS,
        "pool_per_task": dict(per_task or PER_TASK),
        "pools": pools(spec["pool_seed"], len(rows), per_task),
        "plan": spec["plan"],
        "skills_note": spec["skills_note"],
        "sandbox": {
            "seed": None,
            # the knowledge library and its code (user 2026-11-17: gso_single_grasp's entries reach gso_bimanual_lift;
            # the loop evaluates the library before each Reviewer, loop.knowledge_step)
            "keep": KEEP_COMMON + ["scripts/rsi/knowledge.py", "scripts/rsi/knowledge", spec["plan"], spec["base_plan"],
                                   f"scripts/rsi/families/{name}.json", tasks_file],  # fmt: skip
            "closure": CLOSURE,
            "strip": STRIP,
            "scene_exclude": SCENE_EXCLUDE,
        },
    }  # fmt: skip
    (families_dir / f"{name}.json").write_text(json.dumps(fam, indent=1) + "\n")
    return {name: len(rows)}


def write(
    scene_dir: Path, families_dir: Path = FAMILIES, lift: float | None = None, per_task: dict[str, int] | None = None
) -> dict:
    """The four families, gso_all, gso_pack_all and the tasks file for the scene, or a picked scene's one family
    (`write_picked`); returns {family name: task count}."""
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    gso = meta.get("gso")
    if not gso:
        raise SystemExit(f"{scene_dir} is not a GSO scene (no `gso` block in scene_meta.json)")
    if gso.get("picks"):
        return write_picked(scene_dir, meta, families_dir, PICKED_LIFT_M if lift is None else lift, per_task)
    if lift is not None:
        raise SystemExit("--lift is for a hand-picked scene; the drawn instances keep LIFT_M")
    seed, reach = int(gso["seed"]), float(gso["reach_depth_m"])
    scene_rel = (
        str(scene_dir.resolve().relative_to(REPO)) if scene_dir.resolve().is_relative_to(REPO) else str(scene_dir)
    )
    tasks_file = f"scripts/rsi/families/gso_s{seed}_tasks.jsonl"
    families_dir.mkdir(parents=True, exist_ok=True)
    rows = {kind: rows_for(meta, kind, reach) for kind in KINDS}
    if CARTON in meta["objects"] and not any(r["id"] == f"lift_{CARTON}" for r in rows["bimanual"]):
        raise SystemExit(f"{CARTON} is not a bimanual object of {scene_dir}")
    (families_dir / Path(tasks_file).name).write_text(
        "".join(json.dumps(r) + "\n" for kind in KINDS for r in rows[kind])
    )
    out = {}
    for kind in KINDS:
        if not rows[kind]:
            raise SystemExit(f"{scene_dir}: no {OBJECT_KIND[kind]} object for the {KINDS[kind]} family")
        name = f"{KINDS[kind]}_s{seed}"
        fam = family(name, kind, scene_rel, seed, tasks_file, rows[kind], meta, per_task)
        (families_dir / f"{name}.json").write_text(json.dumps(fam, indent=1) + "\n")
        out[name] = len(rows[kind])
    name = f"{ALL}_s{seed}"
    fam = family_all(name, scene_rel, seed, tasks_file, rows, meta, per_task)
    (families_dir / f"{name}.json").write_text(json.dumps(fam, indent=1) + "\n")
    out[name] = len(fam["tasks"])
    name = f"{PACK_ALL}_s{seed}"
    fam = family_pack_all(name, scene_rel, seed, tasks_file, rows, meta, per_task)
    (families_dir / f"{name}.json").write_text(json.dumps(fam, indent=1) + "\n")
    out[name] = len(fam["tasks"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--families-dir", type=Path, default=FAMILIES)
    ap.add_argument("--lift", type=float, default=None, help=f"m; a hand-picked scene's lift (default {PICKED_LIFT_M}; "
                    "user 2026-11-18: 0.20, 0.30 before)")  # fmt: skip
    ap.add_argument("--per-task", nargs="+", metavar="POOL=N", default=None, help="starts per task, any of diag / "
                    f"accept / sealed (default {PER_TASK}); the acceptance pool holds at least gate.STAGE1 + 1")  # fmt: skip
    a = ap.parse_args(argv)
    out = write(a.scene_dir, a.families_dir, a.lift, parse_per_task(a.per_task) if a.per_task else None)
    print(json.dumps({"families_dir": str(a.families_dir), "tasks": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
