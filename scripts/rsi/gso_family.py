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

Every task row carries a `placement`: the task object within REACH (the scene's `gso.reach_depth_m`) of an approach edge
(D13: the middle of the platform is out of reach). The pack rows add per-object conditions (`objects`): the carton on
the two-hand hug's stance band with its long side along its edge (`gso.carton_placement`, the scene builder's
CARTON_PLACEMENT, box_bimanual's `box_long_side`), and a two-hand X with its wide side along its edge, so the hug
squeezes it across that side and it goes into the carton the way it fits (the carton's long inner side lies along the
same edge). Pool seeds: 44/45/46/47 x 10^4 per family, +10^5 per pool, disjoint from every other family's (D7).

    PYTHONPATH=src $VENV_SIM_PY scripts/rsi/gso_family.py outputs/scene_discovery/logistics_room_1030_s0
        [--families-dir scripts/rsi/families]

Writes `<families-dir>/gso_grasp_s<seed>.json`, `gso_bimanual_s<seed>.json`, `gso_pack_s<seed>.json`,
`gso_pack_bimanual_s<seed>.json` and the shared `gso_s<seed>_tasks.jsonl`. A family's `sandbox.keep` lists its file,
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
}
SEEDS = {"gso_grasp": "scripts/rsi/seeds/atomic", "gso_bimanual": None, "gso_pack": None, "gso_pack_bimanual": None}
POOL_SEEDS = {"gso_grasp": 440_000, "gso_bimanual": 450_000, "gso_pack": 460_000, "gso_pack_bimanual": 470_000}
POOL_SIZES = {"diag": 30, "accept": 40, "sealed": 50}
LIFT_M = {"gso_grasp": 0.10, "gso_bimanual": 0.20, "gso_pack": 0.20, "gso_pack_bimanual": 0.20}
TILT_DEG = 60
CARTON = "nor_cal_box"
# where the pack rows put the carton when the scene file has no `gso.carton_placement` (an instance built before
# 2026-11-05): prepare_logistics_scene.CARTON_PLACEMENT, the hug's stance band, long side along the edge
CARTON_BAND = {"depth_m": [0.18, 0.31], "yaw_deg": [85, 90]}
ALONG_EDGE = {True: [85, 90], False: [0, 5]}  # the box x axis along the edge when it is the wide one, else across

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
    """The task rows of one family kind (KINDS) from the scene file: `held` + `lifted` on each object of its kind, or
    -- the pack kinds -- `in(X, carton)` + `held` + `lifted` of the carton for each such X but the carton itself."""
    room = meta["room"]["rooms"] and next(iter(meta["room"]["rooms"]))
    pack = kind.startswith("pack")
    carton_band = (meta.get("gso") or {}).get("carton_placement") or CARTON_BAND
    lift = LIFT_M[KINDS[kind]]
    band = {"depth_m": [0.0, round(reach, 3)]}
    out = []
    for key, o in meta["objects"].items():
        sem = o.get("semantic") or {}
        if not o.get("free") or sem.get("graspable") != OBJECT_KIND[kind] or (pack and key == CARTON):
            continue
        p = sem["phrase"]
        placement: dict = dict(band)
        if kind == "single":
            prompts = [f"pick up the {p}", f"grab the {p} and lift it off the platform"]
            tid, goal = f"pick_{key}", [f"held({key})", f"lifted({key}, {lift:.2f}, {TILT_DEG})"]
        elif kind == "bimanual":
            prompts = [f"lift the {p} {int(lift * 100)} cm off the platform with both hands and hold it",
                       f"pick up the {p} with both hands and hold it level"]  # fmt: skip
            tid, goal = f"lift_{key}", [f"held({key})", f"lifted({key}, {lift:.2f}, {TILT_DEG})"]
        else:
            hands = "both hands" if kind == "pack_bimanual" else "one hand"
            prompts = [f"put the {p} into the cardboard box with {hands}, then lift the box with both hands",
                       f"pack the {p} into the open box and pick the box up"]  # fmt: skip
            tid = f"pack_{key}"
            goal = [f"in({key}, {CARTON})", f"held({CARTON})", f"lifted({CARTON}, {lift:.2f}, {TILT_DEG})"]
            own = {CARTON: dict(carton_band)}
            if kind == "pack_bimanual":
                half = o["box"]["half_extent"]
                own[key] = {**band, "yaw_deg": ALONG_EDGE[float(half[0]) >= float(half[1])]}
            placement["objects"] = own
        out.append({"id": tid, "goal": goal, "prompts": prompts, "room": room, "placement": placement})
    return out


def _about(kind: str, seed: int, reach: float, names: dict) -> str:
    stem, lift = KINDS[kind], LIFT_M[KINDS[kind]]
    objects = "; objects: " + ", ".join(f"{k} = {v}" for k, v in names.items())
    if kind in ("single", "bimanual"):
        hands = "one hand" if kind == "single" else "both hands"
        return (
            f"logistics_room_1030 (user 2026-11-03): held(X) and lifted(X, {lift:.2f}, {TILT_DEG}) with {hands} on the "
            f"Google Scanned Objects drawn with seed {seed} onto the 0.371 m pallet platform (catalog.json {kind} "
            f"objects{', plus the 1.2 kg open carton' if kind == 'bimanual' else ''}); every task object starts within "
            f"{reach} m of a platform edge, the robot anywhere in the room it can walk from" + objects
        )
    hands = "two-hand" if kind == "pack_bimanual" else "single-hand"
    return (
        f"logistics_room_1030 pack line (user 2026-11-05): in(X, {CARTON}), then held({CARTON}) and lifted({CARTON}, "
        f"{lift:.2f}, {TILT_DEG}) with both hands -- X, a {hands} Google Scanned Object of the instance drawn with seed "
        f"{seed} (every drawn object fits the carton's 0.43 x 0.30 x 0.33 m inside), is put into the 1.2 kg open carton "
        f"on the 0.371 m pallet platform, which is then lifted; each episode starts with the carton empty, X within "
        f"{reach} m of a platform edge{' with its wide side along it' if kind == 'pack_bimanual' else ''}, the carton "
        f"0.18-0.31 m in from an edge with its long side along it, the robot anywhere in the room it can walk from"
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
}  # fmt: skip


def family(name: str, kind: str, scene_rel: str, seed: int, tasks_file: str, rows: list[dict], meta: dict) -> dict:
    stem = KINDS[kind]
    names = {r["id"].split("_", 1)[1]: meta["objects"][r["id"].split("_", 1)[1]]["semantic"]["phrase"] for r in rows}
    return {
        "name": name,
        "about": _about(kind, seed, rows[0]["placement"]["depth_m"][1], names),
        "scene": scene_rel,
        "scene_note": (
            "in a simulated logistics room (a scanned warehouse room: storage racks along the walls, one 1.2 x 0.9 m "
            "pallet platform 0.371 m high with an open cardboard box and ten scanned household objects on it)"
        ),  # fmt: skip
        "tasks_file": tasks_file,
        "tasks": [r["id"] for r in rows],
        "spawn": {"rooms": "all", "distance_m": [0.5, None]},
        **({"hold_s": 2.0, "judge_s": 5.0} if kind != "single" else {}),
        "pools": {
            pool: {"seed": POOL_SEEDS[stem] + i * 100_000, "n": n} for i, (pool, n) in enumerate(POOL_SIZES.items())
        },
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


def kind_name(kind: str) -> str:
    return KINDS[kind]


def write(scene_dir: Path, families_dir: Path = FAMILIES) -> dict:
    """The four families and the tasks file for the scene; returns {family name: task count}."""
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
