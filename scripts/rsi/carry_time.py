"""TODO 27.10.1.1 (plan docs/handoff/scene-rsi-plan-20261016.md §12.3, user 2026-10-19): the planned lift-to-release
time of a put-down as the checker can know it, from the route it measures.

The put-down's pick is refused before contact when the object's pad margin is short of place.CREEP_RATE_M_S times
T_est, the planned lift-to-release time (place._refusal; the round's `plan_ahead` step logs `t_est_s`). The skill
plans T_est from its own route and spot; the checker only has its route length d from the pick stand to the target's
stand. This fits, over every logged plan_ahead (the registered families' last sealed rounds and the calibration
rounds), per task the median T_est against d:

    T = base + per_m * d + put_in * [in(X, C)]      a carry
    T = near_drop                                   a put-in on X's own surface (into's near drop: no carry)

and writes outputs/scene_rsi_20261016/capability/carry_time/fit.json; `section` turns it and the catalog's constants
into capability_write.py's `carry.single_arm.creep` (feasibility's `carry_single.creep` reads it). The file also
lists every task with its d, its T_est and the skill's refusals.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/carry_time.py [--rounds <dir> ...]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

from locomani.scene_discovery import feasibility, predicates, scene_file  # noqa: E402

OUT = REPO / "outputs" / "scene_rsi_20261016" / "capability" / "carry_time"
REGISTRY = REPO / "outputs" / "scene_rsi_20261016" / "outer" / "registry.json"
HIDDEN = "outputs/scene_rsi_20261016_hidden"
CALIBRATION = REPO / HIDDEN / "calibration"
SCENE = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars" / "scene_meta.json"
EDGES = REPO / "outputs" / "scene_rsi_20261016" / "capability" / "stance" / "adt_apartment_mars.json"
CAP = REPO / "configs" / "scene_discovery" / "capability_mars_scalebfm.yaml"
MOVES = ("on", "in")
CATALOG = REPO / "src" / "locomani" / "tasks" / "scene" / "skills"


def skill_constant(module: str, name: str, catalog: Path = CATALOG):
    """A module-level literal of the skill catalog (`PAD_Y, PAD_Z = 0.0475, 0.014` too), read without importing it."""
    for node in ast.parse((catalog / f"{module}.py").read_text()).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
                if isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for k, v in zip(t.elts, node.value.elts):
                        if isinstance(k, ast.Name) and k.id == name:
                            return ast.literal_eval(v)
    raise KeyError(f"{module}.{name}")


def section(catalog: Path = CATALOG, fit: Path | None = None) -> dict | None:
    """capability_write.py's `carry.single_arm.creep`: the put-down's refusal before the pick (place._refusal) as the
    checker can apply it -- the catalog's creep rate, pad half height and top-pinch exemption, and this fit's T_est
    from the route. None before this script has run (`fit`: its fit.json, default OUT's)."""
    fit = OUT / "fit.json" if fit is None else fit
    if not fit.exists():
        return None
    m = json.loads(fit.read_text())["model"]
    return dict(
        rule="a side-grasped object's pad margin (box half height + centre z - pad_half_m) under rate_m_s x T_est: the put-down's pick is refused before contact (place._refusal); T_est = base + per_m x route + put_in for in(X, C), near_drop for a put-in on X's own surface; carry.single_arm.top_pinch is exempt",
        rate_m_s=skill_constant("place", "CREEP_RATE_M_S", catalog),
        pad_half_m=skill_constant("grasp_program", "PAD_Z", catalog),
        put_in=bool(skill_constant("into", "CREEP", catalog)),
        t_est_s={k: m[k] for k in ("base", "per_m", "put_in", "near_drop")},
        t_est_fit=dict(tasks=m["tasks"], near_tasks=m["near_tasks"], residual_sd_s=m["residual_sd_s"]),
        source="src/locomani/tasks/scene/skills/{place,grasp_program,pick,into}.py; outputs/scene_rsi_20261016/capability/carry_time/fit.json (carry_time.py, plan §12.3)",
    )  # fmt: skip


def sources() -> list[tuple[Path, dict]]:
    """(rounds dir, {task id: goal}) for every registered family's last rounds and every calibration sample not
    registered."""
    out = []
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    for family, row in sorted(reg.items()):
        fam = json.loads((HERE / "families" / f"{family}.json").read_text())
        goals = {t["id"]: t["goal"] for t in map(json.loads, (REPO / fam["tasks_file"]).read_text().splitlines()) if t}
        base = Path(row["dir"]) if row.get("dir") else Path(row["sandbox"]) / HIDDEN / family / "sealed"
        out.append((base / row["last"], goals))
    registered = {d.resolve() for d, _ in out}
    for d in sorted(CALIBRATION.glob("*/rounds")):
        if d.resolve() in registered:
            continue
        goals = {t["id"]: t["goal"] for t in map(json.loads, (d.parent / "tasks.jsonl").read_text().splitlines()) if t}
        out.append((d, goals))
    return out


def logged(rounds: Path, goals: dict) -> dict:
    """{goal key: dict(goal, t=[T_est...], refused=n)} of the rounds' plan_ahead steps."""
    out = {}
    for p in sorted(rounds.glob("*.json")):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text())
        step = next((s for s in r.get("steps") or () if s["step"] == "plan_ahead"), None)
        if step is None or step.get("t_est_s") is None or r["task"] not in goals:
            continue
        goal = goals[r["task"]]
        e = out.setdefault(json.dumps(sorted(goal)), dict(goal=goal, task=r["task"], t=[], refused=0, top=step["top"]))
        e["t"].append(float(step["t_est_s"]))
        e["refused"] += "creep margin" in (step.get("why") or "")
    return out


def route_of(checker: feasibility.Checker, goal) -> tuple[str | None, float | None, bool]:
    """(the move's predicate, the checker's carry route d (None: no carry), whether X stays on its surface)."""
    move = next((p for p in map(predicates.parse, goal) if p.name in MOVES), None)
    if move is None:
        return None, None, False
    x, target = move.args[0], move.args[1]
    same = move.name == "in" and checker.objects[target].get("surface") == checker.objects[x].get("surface")
    r = checker.check(dict(id="carry_time", goal=goal))
    d = None
    for name in ("carry_distance", "measured_task", "route"):
        d = next((c.measured for c in r.checks if c.step.startswith("carry(") and c.name == name
                  and isinstance(c.measured, (int, float))), None)  # fmt: skip
        if d is not None:
            break
    return move.name, d, same


def fit(rows: list[dict]) -> dict:
    carries = [r for r in rows if r["d"] is not None]
    near = [r["t_med"] for r in rows if r["same_surface"]]
    a = np.array([[1.0, r["d"], float(r["move"] == "in")] for r in carries])
    t = np.array([r["t_med"] for r in carries])
    coef, *_ = np.linalg.lstsq(a, t, rcond=None)
    res = t - a @ coef
    return dict(base=round(float(coef[0]), 1), per_m=round(float(coef[1]), 2), put_in=round(float(coef[2]), 1),
                near_drop=round(float(np.median(near)), 1) if near else None, residual_sd_s=round(float(res.std()), 1),
                tasks=len(carries), near_tasks=len(near))  # fmt: skip


def run(extra: list[Path]) -> dict:
    meta = scene_file.with_approach_edges(scene_file.load(str(SCENE)), str(EDGES))
    cap = yaml.safe_load(CAP.read_text())
    cap.get("carry", {}).get("single_arm", {}).pop("creep", None)  # the route alone, whatever an older fit said
    checker = feasibility.Checker(meta, cap)
    seen = {}
    srcs = sources() + [(d, {t["id"]: t["goal"] for t in map(json.loads, (d.parent / "tasks.jsonl").read_text().splitlines()) if t}) for d in extra]  # fmt: skip
    for rounds, goals in srcs:
        for key, e in logged(rounds, goals).items():
            s = seen.setdefault(key, dict(e, t=[], refused=0, sources=[]))
            s["t"] += e["t"]
            s["refused"] += e["refused"]
            s["sources"].append(str(rounds.relative_to(REPO) if rounds.is_relative_to(REPO) else rounds))
    rows = []
    for e in seen.values():
        try:
            move, d, same = route_of(checker, e["goal"])
        except (KeyError, ValueError):  # a goal over what the scene no longer has (the deleted counters)
            continue
        if move is None or (d is None and not same):
            continue
        rows.append(dict(task=e["task"], goal=e["goal"], move=move, d=d, same_surface=same, n=len(e["t"]),
                         t_med=round(float(np.median(e["t"])), 1), t_range=[min(e["t"]), max(e["t"])],
                         refused=e["refused"], top=e["top"], sources=e["sources"]))  # fmt: skip
    out = dict(model=fit(rows), rounds=sum(r["n"] for r in rows), rows=sorted(rows, key=lambda r: r["task"]),
               source="scripts/rsi/carry_time.py (plan §12.3)")  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit.json").write_text(json.dumps(out, indent=1))
    m = out["model"]
    print(f"[carry_time] {m['tasks']} carries + {m['near_tasks']} near drops, {out['rounds']} rounds: T = {m['base']} + "
          f"{m['per_m']} d + {m['put_in']} [in]; near drop {m['near_drop']} s; residual sd {m['residual_sd_s']} s")  # fmt: skip
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=Path, nargs="*", default=[], help="more rounds dirs with a tasks.jsonl beside")
    run(ap.parse_args(argv).rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
