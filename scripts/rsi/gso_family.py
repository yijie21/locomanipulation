#!/usr/bin/env python3
"""Write the two task families of a `logistics_room_1030_s<seed>` scene instance (user Q13, 2026-11-03): one task per
object, generated from the scene file instead of written by hand.

- `gso_grasp_s<seed>`: one hand; every `semantic.graspable == "single"` object; goal `held(X)` + `lifted(X, 0.10, 60)`
  (user Q23: held and 10 cm off the platform); the seed is the empty skills package (`seeds/atomic`), as grasp_atomic's.
- `gso_bimanual_s<seed>`: two hands; every `bimanual` object and the carton; goal `held(X)` + `lifted(X, 0.20, 60)`,
  judged over the round's last 5 s (box_bimanual's `hold_s` / `judge_s`); the seed is the box line's two-hand hug
  (`seeds/box_strict_start`).

Every task row carries `placement: {"depth_m": [0, REACH]}`: the start sampler puts the task object within REACH of
an approach edge (D13: the middle of the platform is out of reach), REACH being the scene's `gso.reach_depth_m`.

    PYTHONPATH=src $VENV_SIM_PY scripts/rsi/gso_family.py outputs/scene_discovery/logistics_room_1030_s0
        [--families-dir scripts/rsi/families]

Writes `<families-dir>/gso_grasp_s<seed>.json`, `gso_bimanual_s<seed>.json` and the shared `gso_s<seed>_tasks.jsonl`.
The family's `sandbox.keep` lists these three files and the plan, as the existing families do.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FAMILIES = HERE / "families"
PLANS = {"gso_grasp": "scripts/rsi/plans/gso_grasp.md", "gso_bimanual": "scripts/rsi/plans/gso_bimanual.md"}
SEEDS = {"gso_grasp": "scripts/rsi/seeds/atomic", "gso_bimanual": "scripts/rsi/seeds/box_strict_start"}
POOL_SEEDS = {"gso_grasp": 440_000, "gso_bimanual": 450_000}  # disjoint from every other family's (D7)
POOL_SIZES = {"diag": 30, "accept": 40, "sealed": 50}
LIFT_M = {"gso_grasp": 0.10, "gso_bimanual": 0.20}
TILT_DEG = 60
CARTON = "nor_cal_box"

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


def rows_for(meta: dict, kind: str, reach: float) -> list[dict]:
    """The task rows of one family from the scene file: `held` + `lifted` on each of its objects."""
    room = meta["room"]["rooms"] and next(iter(meta["room"]["rooms"]))
    out = []
    for key, o in meta["objects"].items():
        sem = o.get("semantic") or {}
        if not o.get("free") or sem.get("graspable") != kind:
            continue
        p = sem["phrase"]
        if kind == "single":
            prompts = [f"pick up the {p}", f"grab the {p} and lift it off the platform"]
            tid, lift = f"pick_{key}", LIFT_M["gso_grasp"]
        else:
            prompts = [f"lift the {p} {int(LIFT_M['gso_bimanual'] * 100)} cm off the platform with both hands and hold it",
                       f"pick up the {p} with both hands and hold it level"]  # fmt: skip
            tid, lift = f"lift_{key}", LIFT_M["gso_bimanual"]
        out.append({"id": tid, "goal": [f"held({key})", f"lifted({key}, {lift:.2f}, {TILT_DEG})"], "prompts": prompts,
                    "room": room, "placement": {"depth_m": [0.0, round(reach, 3)]}})  # fmt: skip
    return out


def family(name: str, kind: str, scene_rel: str, seed: int, tasks_file: str, rows: list[dict], meta: dict) -> dict:
    names = {r["id"].split("_", 1)[1]: meta["objects"][r["id"].split("_", 1)[1]]["semantic"]["phrase"] for r in rows}
    hands = "one hand" if kind == "single" else "both hands"
    fam = {
        "name": name,
        "about": (
            f"logistics_room_1030 (user 2026-11-03): held(X) and lifted(X, {LIFT_M[kind_name(kind)]:.2f}, {TILT_DEG}) "
            f"with {hands} on the Google Scanned Objects drawn with seed {seed} onto the 0.371 m pallet platform "
            f"(catalog.json {kind} objects{', plus the 1.2 kg open carton' if kind == 'bimanual' else ''}); every task "
            f"object starts within {rows[0]['placement']['depth_m'][1]} m of a platform edge, the robot anywhere in the "
            f"room it can walk from; objects: " + ", ".join(f"{k} = {v}" for k, v in names.items())
        ),  # fmt: skip
        "scene": scene_rel,
        "scene_note": (
            "in a simulated logistics room (a scanned warehouse room: storage racks along the walls, one 1.2 x 0.9 m "
            "pallet platform 0.371 m high with an open cardboard box and ten scanned household objects on it)"
        ),  # fmt: skip
        "tasks_file": tasks_file,
        "tasks": [r["id"] for r in rows],
        "spawn": {"rooms": "all", "distance_m": [0.5, None]},
        **({"hold_s": 2.0, "judge_s": 5.0} if kind == "bimanual" else {}),
        "pools": {
            pool: {"seed": POOL_SEEDS[kind_name(kind)] + i * 100_000, "n": n}
            for i, (pool, n) in enumerate(POOL_SIZES.items())
        },
        "plan": PLANS[kind_name(kind)],
        "skills_note": (
            "the incumbent's skills; the seed was empty (`__init__.py`: plan_for covers no goal)"
            if kind == "single"
            else "the incumbent's skills; the seed was the box line's two-hand hug pick (seeds/box_strict_start)"
        ),  # fmt: skip
        "sandbox": {
            "seed": SEEDS[kind_name(kind)],
            "keep": KEEP_COMMON + [PLANS[kind_name(kind)], f"scripts/rsi/families/{name}.json", tasks_file],
            "closure": CLOSURE,
            "strip": STRIP,
            "scene_exclude": SCENE_EXCLUDE,
        },
    }
    return fam


def kind_name(kind: str) -> str:
    return "gso_grasp" if kind == "single" else "gso_bimanual"


def write(scene_dir: Path, families_dir: Path = FAMILIES) -> dict:
    """Both families and the tasks file for the scene; returns {family name: task count}."""
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    gso = meta.get("gso")
    if not gso:
        raise SystemExit(f"{scene_dir} is not a GSO scene (no `gso` block in scene_meta.json)")
    seed, reach = int(gso["seed"]), float(gso["reach_depth_m"])
    scene_rel = (
        str(scene_dir.resolve().relative_to(REPO)) if scene_dir.resolve().is_relative_to(REPO) else str(scene_dir)
    )
    tasks_file = f"scripts/rsi/families/gso_s{seed}_tasks.jsonl"
    families_dir.mkdir(parents=True, exist_ok=True)
    rows = {kind: rows_for(meta, kind, reach) for kind in ("single", "bimanual")}
    if CARTON in meta["objects"] and not any(r["id"] == f"lift_{CARTON}" for r in rows["bimanual"]):
        raise SystemExit(f"{CARTON} is not a bimanual object of {scene_dir}")
    (families_dir / Path(tasks_file).name).write_text(
        "".join(json.dumps(r) + "\n" for r in rows["single"] + rows["bimanual"])
    )
    out = {}
    for kind in ("single", "bimanual"):
        if not rows[kind]:
            raise SystemExit(f"{scene_dir}: no {kind} object for a family")
        name = f"{kind_name(kind)}_s{seed}"
        fam = family(name, kind, scene_rel, seed, tasks_file, rows[kind], meta)
        (families_dir / f"{name}.json").write_text(json.dumps(fam, indent=1) + "\n")
        out[name] = len(rows[kind])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--families-dir", type=Path, default=FAMILIES)
    a = ap.parse_args(argv)
    out = write(a.scene_dir, a.families_dir)
    print(json.dumps({"families_dir": str(a.families_dir), "tasks": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
