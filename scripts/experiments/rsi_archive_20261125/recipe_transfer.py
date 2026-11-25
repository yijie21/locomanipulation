"""TODO 33.13.1 / 33.13.2 (plan section 2.3, D10, D15): does a flow under recipes/ serve similar objects, and may it be
promoted to them?

A recipe ([branch]) declares the features it is for (`SIGNATURE`) and scores an object by them (`match`); nothing
names its object. This check, run by a person (or the driver) over a skills package after a run:

1. Similar objects (`similar`): every free one-hand object of the scene on a support gets the package's own features
   (`recipes.features`, the side grasp's view along SIMILAR_APPROACHES yaws at the jaw height) and the recipe's best
   `match` over them. Birth objects are the family's own objects it matches; similar objects are the others, those it
   matches first (by score), then by feature distance to the birth objects (height, width across the jaw, flat,
   container) -- SIMILAR of them. A similar object it does not match (score 0) is listed as such: the recipe would not
   be used on it.
2. The transfer pool: STARTS starts for each birth and each similar object, drawn by the `pick_any` family (held(X) of
   every one-hand free object, `tasks` writes its file) from TRANSFER_SEED on.
3. Two arms on that pool: the package as it is, and a copy whose `recipes.RECIPES` leaves this recipe out. Reported:
   birth objects x/n and similar objects y/m with the recipe; on the similar objects the pairs it wins (b) and loses
   (c) against the arm without it; how many rounds used it (their "recipe" note). It is promotable when c = 0 on at
   least PROMOTE_MIN similar objects that it matches and that used it in a round (the plan's "at least two similar
   objects, not one round worse"); else it stays local -- its SIGNATURE should stay as tight as its birth objects.
The transfer pool is never a gate pool and no role reads it during a run; the verdict goes to the report and, for a
sandbox, a `recipe_transfer` row in its ledger (D15: "推广记进账本").

    python3 scripts/rsi/recipe_transfer.py tasks                      # (re)write pick_any's task file
    PYTHONPATH=src $PY scripts/rsi/recipe_transfer.py similar <skills dir> --family grasp_atomic
    PYTHONPATH=src $PY scripts/rsi/recipe_transfer.py run <skills dir> --family grasp_atomic --out <dir> [--ledger L]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "rsi"), str(REPO / "scripts" / "rsi" / "scene"), str(REPO / "src")]

ANY = "pick_any"
TRANSFER_SEED = 710000
STARTS = 2
SIMILAR = 3
PROMOTE_MIN = 2
SIMILAR_APPROACHES = 8


def _inside(poly, xy) -> bool:
    poly = np.asarray(poly, float)
    x, y = xy
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def write_tasks(scene_dir: Path, out: Path) -> int:
    """pick_any's tasks: held(X) of every free, one-hand object on a support, in the room it lies in."""
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    rooms = (meta.get("room") or {}).get("rooms") or {}
    rows = []
    for key, o in sorted(meta["objects"].items()):
        if not o.get("free") or o.get("bimanual") or not o.get("surface") or "body" not in o:
            continue
        room = next((r for r, v in rooms.items() if _inside(v["polygon"], o["init_pos"][:2])), None)
        rows.append(dict(id=f"pick_{key}", goal=[f"held({key})"], prompts=[], room=room))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def _world(static):
    """A view of a StaticScene that a skills package's recipes.features can read (sim.m, free, object_pose, meta)."""
    m = static.m
    free = {k: m.body(o["body"]).id for k, o in static.meta["objects"].items() if o.get("free") and "body" in o}
    return SimpleNamespace(meta=static.meta, object_pose=static.object_pose, sim=SimpleNamespace(m=m), free=free)


def _features(pkg, world, key: str) -> list:
    from locomani.tasks.scene import graspability as G

    flat = G.resting_half_height(world, key) < G.FLAT_HALF_M
    z = G.grasp_height(world, key)
    out = []
    for t in np.radians(np.arange(0.0, 360.0, 360.0 / SIMILAR_APPROACHES)):
        try:
            out.append(pkg.recipes.features(world, key, np.array([math.cos(t), math.sin(t), 0.0]), z, flat=flat,
                                            cover=False))  # fmt: skip
        except Exception:  # noqa: BLE001 -- a package's features() is the loop's code; an object it cannot read is skipped
            continue
    return out


def _vec(objs: list) -> np.ndarray | None:
    """height, the narrowest width across the jaw over the approaches, flat, container -- for the feature distance."""
    if not objs:
        return None
    o = objs[0]
    widths = [x.width for x in objs if x.width is not None]
    return np.array([o.height, min(widths) if widths else 0.2, float(o.flat), float(o.container)])


def recipe_name(r) -> str:
    return r.__name__.rsplit(".", 1)[-1]


def similar(pkg, static, family_objects: list[str], n: int = SIMILAR) -> dict:
    """{recipe: {birth: [..], similar: [{key, score}], ...}} (module docstring, step 1)."""
    world = _world(static)
    recipes = list(getattr(getattr(pkg, "recipes", None), "RECIPES", ()) or ())
    feats = {k: _features(pkg, world, k) for k, o in static.meta["objects"].items()
             if o.get("free") and not o.get("bimanual") and o.get("surface") and "body" in o}  # fmt: skip
    out = {}
    for r in recipes:
        score = {k: max([float(r.match(f)) for f in fs] or [0.0]) for k, fs in feats.items()}
        birth = [k for k in family_objects if score.get(k, 0.0) > 0]
        bv = [v for v in (_vec(feats.get(k) or []) for k in birth) if v is not None]
        scale = np.array([0.1, 0.02, 1.0, 1.0])  # 10 cm of height ~ 2 cm of width ~ flat vs not

        def dist(k):
            v = _vec(feats[k])
            return min(float(np.linalg.norm((v - b) / scale)) for b in bv) if v is not None and bv else math.inf

        others = [k for k in feats if k not in family_objects]
        ranked = sorted(others, key=lambda k: (-score[k], dist(k), k))[:n]
        out[recipe_name(r)] = dict(birth=birth, similar=[dict(key=k, score=round(score[k], 3),
                                                              dist=None if dist(k) == math.inf else round(dist(k), 3))
                                                         for k in ranked])  # fmt: skip
    return out


def without(skills: Path, name: str, out: Path) -> Path:
    """A copy of the package whose recipes.RECIPES leaves `name` out (the arm without the recipe)."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(skills, out, ignore=shutil.ignore_patterns("__pycache__"))
    init = out / "recipes" / "__init__.py"
    init.write_text(init.read_text() + f"\n\nRECIPES = tuple(r for r in RECIPES if r.__name__.rsplit('.', 1)[-1] != {name!r})"
                    "  # recipe_transfer.py: the arm without it\n")  # fmt: skip
    return out


def _used(r: dict, name: str) -> bool:
    return any(
        isinstance(s, dict) and s.get("step") == "recipe" and s.get("recipe") == name for s in r.get("steps") or []
    )


def verdict(per_similar: list[dict]) -> tuple[bool, str]:
    """Promotable when no similar object lost a round with the recipe (c = 0) and at least PROMOTE_MIN similar objects
    it matches actually used it in a round (a static match the rounds never reach tests nothing: run D's tall_narrow
    matched the jam statically and no jam round took it, 2026-10-24); else local, and why."""
    lost = sum(int(p.get("c") or 0) for p in per_similar)
    clean = sum(bool(p.get("score", 0) > 0 and p.get("used") and not p.get("c")) for p in per_similar)
    if lost:
        return False, f"local: {lost} round(s) lost with it on similar objects"
    if clean < PROMOTE_MIN:
        return False, f"local: used on only {clean} similar object(s) without a loss (needs {PROMOTE_MIN})"
    return True, "promotable: used on similar objects and none lost a round with it"


def run(skills: Path, family: str, out: Path, starts: int = STARTS, n_similar: int = SIMILAR, procs: int = 12,
        gpus: str = "0,1", overrides: dict | None = None) -> dict:  # fmt: skip
    import batch as B
    import family as F
    from scene_round import load_skills

    from locomani.tasks.scene import graspability as G

    fam = F.load(family)
    objs = sorted({o for t in fam["tasks"] for o in F.task(fam, t).objects})
    static = G.StaticScene(fam["scene_dir"])
    pkg = load_skills(skills)
    sim = similar(pkg, static, objs, n_similar)
    anyf = F.load(ANY)
    report = dict(family=family, skills=str(skills), t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), recipes={})
    out.mkdir(parents=True, exist_ok=True)
    for name, s in sim.items():
        keys = s["birth"] + [x["key"] for x in s["similar"]]
        tasks = [f"pick_{k}" for k in keys if f"pick_{k}" in anyf["rows"]]
        pool = out / name / "pool"
        drawn = F.draw(anyf, "transfer", pool, seed=TRANSFER_SEED, n=starts * len(tasks), tasks=tasks)
        labels = sorted(p.stem for p in pool.glob("*.json") if not p.name.startswith("_"))
        off = without(skills, name, out / name / "skills_without")
        jobs = [B.Job(skills, out / name / "with", overrides or {}, labels),
                B.Job(off, out / name / "without", overrides or {}, labels)]  # fmt: skip
        B.run_jobs(jobs, ANY, pool, procs, out / name / "run.log", gpus)
        on, offr = B.outcomes(out / name / "with"), B.outcomes(out / name / "without")

        def tally(keys_):
            ls = [lab for lab in on if any(lab.startswith(f"pick_{k}_s") for k in keys_)]
            return dict(
                ok=sum(bool(on[lab]["ok"]) for lab in ls), n=len(ls), used=sum(_used(on[lab], name) for lab in ls)
            )

        per_similar = []
        for x in s["similar"]:
            ls = [lab for lab in on if lab.startswith(f"pick_{x['key']}_s") and lab in offr]
            per_similar.append(dict(x, n=len(ls), ok=sum(bool(on[lab]["ok"]) for lab in ls),
                                    ok_without=sum(bool(offr[lab]["ok"]) for lab in ls),
                                    b=sum(bool(on[lab]["ok"]) and not offr[lab]["ok"] for lab in ls),
                                    c=sum(bool(offr[lab]["ok"]) and not on[lab]["ok"] for lab in ls),
                                    used=sum(_used(on[lab], name) for lab in ls)))  # fmt: skip
        promotable, why = verdict(per_similar)
        report["recipes"][name] = dict(birth=s["birth"], birth_rounds=tally(s["birth"]), similar=per_similar,
                                       similar_rounds=tally([x["key"] for x in s["similar"]]),
                                       unavailable=drawn.get("unavailable"), promotable=promotable, verdict=why)  # fmt: skip
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks")
    s = sub.add_parser("similar")
    s.add_argument("skills", type=Path)
    s.add_argument("--family", required=True)
    r = sub.add_parser("run")
    r.add_argument("skills", type=Path)
    r.add_argument("--family", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--set", default="{}", help="the package's parameter overrides (a candidate's set)")
    r.add_argument("--starts", type=int, default=STARTS)
    r.add_argument("--similar", type=int, default=SIMILAR)
    r.add_argument("--procs", type=int, default=12)
    r.add_argument("--gpus", default="0,1")
    r.add_argument("--ledger", type=Path, default=None, help="a sandbox's loop ledger to append the verdicts to")
    a = ap.parse_args(argv)
    if a.cmd == "tasks":
        fam = json.loads((REPO / "scripts" / "rsi" / "families" / f"{ANY}.json").read_text())
        n = write_tasks(REPO / fam["scene"], REPO / fam["tasks_file"])
        rows = [json.loads(x) for x in (REPO / fam["tasks_file"]).read_text().splitlines() if x.strip()]
        fam["tasks"] = [r["id"] for r in rows]
        (REPO / "scripts" / "rsi" / "families" / f"{ANY}.json").write_text(
            json.dumps(fam, indent=1, ensure_ascii=False) + "\n"
        )
        print(f"{n} tasks -> {fam['tasks_file']} and the family's task list")
        return 0
    if a.cmd == "similar":
        import family as F
        from scene_round import load_skills

        from locomani.tasks.scene import graspability as G

        fam = F.load(a.family)
        objs = sorted({o for t in fam["tasks"] for o in F.task(fam, t).objects})
        print(json.dumps(similar(load_skills(a.skills.resolve()), G.StaticScene(fam["scene_dir"]), objs), indent=1))
        return 0
    rep = run(a.skills.resolve(), a.family, a.out.resolve(), a.starts, a.similar, a.procs, a.gpus, json.loads(a.set))
    for name, v in rep["recipes"].items():
        print(f"{name}: birth {v['birth_rounds']['ok']}/{v['birth_rounds']['n']}, similar {v['similar_rounds']['ok']}/"
              f"{v['similar_rounds']['n']} -- {v['verdict']}")  # fmt: skip
        if a.ledger is not None:
            with a.ledger.open("a") as fh:
                fh.write(json.dumps(dict(t=rep["t"], step="recipe_transfer", recipe=name, skills=rep["skills"],
                                         birth=v["birth_rounds"], similar=v["similar_rounds"],
                                         promotable=v["promotable"], verdict=v["verdict"])) + "\n")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
