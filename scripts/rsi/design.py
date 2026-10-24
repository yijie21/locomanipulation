"""TODO 33.19 (plan section 1.3, D16): design probes -- one factor, a few levels, the same starts at every level, the
condition forced by the driver. A probe answers how the factor changes the outcome; it is never evidence that a change
should be accepted (the gate decides that, rule 7). A Reviewer's hypothesis may carry a `design` {factor, levels,
starts, tasks?, name?}; the driver runs at most MAX_PROBES per iteration, each at most MAX_ROUNDS rounds, under the
incumbent, on the diagnostic pool's starts (the same base starts at every level, so the levels pair up).

Factors (the forcing is written here, by people; what to measure is the Reviewer's):
  param           the incumbent with one overridable parameter (`name`) set to each level -- the stance's minimum
                  distance, an entry depth, ...
  spawn_bearing   the robot turned on its spawn so the object lies `level` degrees off its heading (0 ahead, 90 left)
  spawn_distance  the robot moved along the line from the object to its spawn until `level` metres from the object,
                  facing the object as it did; a start whose new spot is not free and reachable is left out
  object_yaw      the task's object turned by `level` degrees about the vertical, then slid by the smallest shift (at
                  most MAX_SHIFT_M) that keeps its footprint on its surface's top and clear of its neighbours; left out
                  when no spot that close fits

Each level reports how many starts were left out and why (`skipped_why`), an object_yaw level how far the objects were
slid (`shift_cm_max`), and the result a `note` naming the levels that lost more than half their starts. Run D turned a
0.31 m spoon in place twice (iterations 15 and 17): 11 of 16 starts left out, no reason given, and the Reviewer asked
again (TODO 37.9.1).

    python3 scripts/rsi/design.py <family> --pool <starts> --skills <dir> --set '{...}' --out <dir> \\
        --factor spawn_bearing --levels 0 90 180 --starts 4 [--task T ...] [--name pick.X]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import batch as B  # noqa: E402
import family as F  # noqa: E402

from locomani.tasks.scene import start as ST  # noqa: E402

FACTORS = ("param", "spawn_bearing", "spawn_distance", "object_yaw")
MAX_PROBES = 2  # per iteration
MAX_ROUNDS = 300  # per probe: levels x starts x tasks
MAX_LEVELS = 8
MAX_SHIFT_M = 0.15  # object_yaw: how far a turned object may slide on its top to fit (TODO 37.9.1)
SHIFT_STEP_M = 0.01


def check(spec: dict, tasks_all: list[str], params: dict | None = None) -> str | None:
    """Why a design spec cannot run (None when it can)."""
    if spec.get("factor") not in FACTORS:
        return f"factor must be one of {', '.join(FACTORS)}"
    levels = spec.get("levels") or []
    if not 1 <= len(levels) <= MAX_LEVELS or not all(isinstance(x, (int, float)) for x in levels):
        return f"levels: 1 to {MAX_LEVELS} numbers"
    if spec["factor"] == "param" and (params is None or spec.get("name") not in params):
        return f"param: `name` must be one of the incumbent's overridable parameters, not {spec.get('name')!r}"
    tasks = spec.get("tasks") or tasks_all
    if set(tasks) - set(tasks_all):
        return f"unknown tasks {sorted(set(tasks) - set(tasks_all))}"
    n = len(levels) * max(1, int(spec.get("starts") or 1)) * len(tasks)
    return f"{n} rounds > {MAX_ROUNDS}" if n > MAX_ROUNDS else None


def _object_of(fam: dict, task_id: str) -> str:
    t = F.task(fam, task_id)
    return t.objects[0]


def _shifts() -> list[np.ndarray]:
    """The candidate slides of a turned object, nearest first (a fixed order, so a start always lands the same way)."""
    k = int(round(MAX_SHIFT_M / SHIFT_STEP_M))
    grid = [(i, j) for i in range(-k, k + 1) for j in range(-k, k + 1) if math.hypot(i, j) <= k + 1e-9]
    return [SHIFT_STEP_M * np.array(ij, float) for ij in sorted(grid, key=lambda ij: (math.hypot(*ij), ij))]


def _rotated(meta: dict, key: str, start: dict, deg: float) -> tuple[list | None, dict]:
    """The object's placement turned by `deg` about the vertical and slid by the nearest shift (MAX_SHIFT_M at most)
    that keeps its footprint on its top and clear of its neighbours, with {"shift_m"}; (None, {"why"}) when no spot
    fits -- `off_top` or `neighbour`, what the turn in place ran into."""
    obj = meta["objects"][key]
    pos, quat = start["placements"][key]
    q = ST._qmul(ST._yaw_quat(math.radians(deg)), np.asarray(quat, float))
    surf = meta["surfaces"][obj["surface"]]
    flat = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)[:2]
    top = (np.asarray(surf["xy"], float), flat - ST.SURFACE_MARGIN_M, float(surf.get("yaw", 0.0)))
    others = []
    for k, o in meta["objects"].items():
        if k == key or o.get("surface") != obj["surface"]:
            continue
        p, qq = start["placements"].get(k, (o["init_pos"], o["init_quat"]))
        others.append(ST.footprint(o, np.asarray(p, float)[:2], qq))

    def clash(xy) -> str | None:
        fp = ST.footprint(obj, xy, q)
        if not ST.inside_top(fp, top):
            return "off_top"
        return "neighbour" if any(ST.overlap(fp, t, ST.OBJECT_GAP_M) for t in others) else None

    xy0 = np.asarray(pos, float)[:2]
    why = clash(xy0)
    for d in _shifts() if why else [np.zeros(2)]:
        if clash(xy0 + d) is None:
            xy = xy0 + d
            return [[float(xy[0]), float(xy[1]), float(pos[2])], q.tolist()], dict(shift_m=float(np.hypot(*d)))
    return None, dict(why=why)


def transform(factor: str, level: float, row: dict, key: str, meta: dict, nav=None) -> tuple[dict | None, dict]:
    """The start under one level of a start factor, with what the forcing did (an object_yaw `shift_m`); (None,
    {"why"}) when the forced start is not valid."""
    s = json.loads(json.dumps(row["start"]))
    obj = np.asarray(s["placements"][key][0], float)[:2]
    robot = np.asarray(s["robot_xy"], float)
    to_obj = math.atan2(obj[1] - robot[1], obj[0] - robot[0])
    info: dict = {}
    if factor == "spawn_bearing":
        s["robot_yaw"] = to_obj - math.radians(level)
    elif factor == "spawn_distance":
        u = robot - obj
        u = u / (np.linalg.norm(u) or 1.0)
        xy = obj + u * float(level)
        if nav is not None:
            occ, navi = nav
            navi.set_goal_ring(obj, *ST.APPROACH_RING_M)
            if not occ.is_free(xy):
                return None, dict(why="not_free")
            if not navi.reachable(xy):
                return None, dict(why="unreachable")
        bearing = s["robot_yaw"] - to_obj  # keep the object where it was in the robot's view
        s["robot_xy"] = [float(xy[0]), float(xy[1])]
        s["robot_yaw"] = math.atan2(obj[1] - xy[1], obj[0] - xy[0]) + bearing
    elif factor == "object_yaw":
        spot, info = _rotated(meta, key, s, float(level))
        if spot is None:
            return None, info
        s["placements"][key] = spot
    return dict(row, start=s), info


def base_starts(pool: Path, tasks: list[str], per_task: int) -> list[tuple[str, dict]]:
    """The diagnostic pool's first `per_task` starts of each task (label order)."""
    by = collections.defaultdict(list)
    for p in sorted(pool.glob("*.json")):
        if p.name.startswith("_"):
            continue
        row = json.loads(p.read_text())
        if row.get("task") in tasks and len(by[row["task"]]) < per_task:
            by[row["task"]].append((p.stem, row))
    return [x for t in tasks for x in by[t]]


def run(spec: dict, family: str, pool: Path, skills: Path, overrides: dict, out: Path, procs: int, gpus: str) -> dict:
    """One design probe: the forced starts (or parameter sets), the rounds on the queue, the tally per level."""
    fam = F.load(family)
    tasks = list(spec.get("tasks") or fam["tasks"])
    starts = base_starts(pool, tasks, max(1, int(spec.get("starts") or 1)))
    levels = [float(x) for x in spec["levels"]]
    factor = spec["factor"]
    nav = None
    if factor == "spawn_distance":
        from locomani.primitives.navigator import build as build_navigation
        from locomani.tasks.scene.world import BODY_RADIUS_M

        nav = build_navigation(fam["meta"], body_radius=BODY_RADIUS_M)
    dpool = out / "pool"
    dpool.mkdir(parents=True, exist_ok=True)
    jobs, skipped, shifts = [], collections.defaultdict(collections.Counter), collections.defaultdict(list)
    for i, lv in enumerate(levels):
        labels = []
        for label, row in starts:
            new, info = (
                (row, {})
                if factor == "param"
                else transform(factor, lv, row, _object_of(fam, row["task"]), fam["meta"], nav)
            )
            if new is None:
                skipped[i][info.get("why", "invalid")] += 1
                continue
            if "shift_m" in info:
                shifts[i].append(info["shift_m"])
            lab = f"{label}_L{i}"
            (dpool / f"{lab}.json").write_text(json.dumps(new))
            labels.append(lab)
        ov = dict(overrides, **({spec["name"]: lv} if factor == "param" else {}))
        jobs.append(B.Job(skills, out / f"level_{i}", ov, labels))
    B.run_jobs([j for j in jobs if j.labels], family, dpool, procs, out / "run.log", gpus)
    per = []
    for i, (lv, job) in enumerate(zip(levels, jobs, strict=True)):
        rounds = list(B.outcomes(job.out).values()) if job.out.exists() else []
        n, ok = len(rounds), sum(bool(r["ok"]) for r in rounds)
        per.append(dict(level=lv, n=n, ok=ok, rate=round(ok / n, 3) if n else None, skipped=sum(skipped[i].values()),
                        skipped_why=dict(skipped[i]),
                        classes=dict(collections.Counter(r.get("failure_class") for r in rounds if not r["ok"])),
                        **({"shift_cm_max": round(100 * max(shifts[i]), 1)} if shifts[i] else {})))  # fmt: skip
    res = dict(factor=factor, name=spec.get("name"), tasks=tasks, starts=len(starts), levels=per, dir=str(out))
    ran = [r for job in jobs if job.out.exists() for r in B.outcomes(job.out).values()]
    for k in ("controller", "motor"):  # what the curve was measured under (TODO 33.1.7)
        res[k] = next((r.get(k) for r in ran if r.get(k)), None)
    thin = [p for p in per if 2 * p["skipped"] > len(starts)]
    if thin:
        why = "; ".join(f"{p['level']:g}: " + ", ".join(f"{k} {v}" for k, v in p["skipped_why"].items()) for p in thin)
        res["note"] = (
            f"levels {', '.join(format(p['level'], 'g') for p in thin)} left out more than half of the {len(starts)} "
            f"starts ({why}); their rates say little -- pick other levels, other starts or another factor rather than "
            "asking again"
        )
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--skills", type=Path, required=True)
    ap.add_argument("--set", default="{}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--factor", choices=FACTORS, required=True)
    ap.add_argument("--levels", type=float, nargs="+", required=True)
    ap.add_argument("--starts", type=int, default=4, help="starts per task, the same at every level")
    ap.add_argument("--task", action="append")
    ap.add_argument("--name", help="the parameter, for factor param")
    ap.add_argument("--procs", type=int, default=B.SIM_SLOTS or 12)
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args(argv)
    spec = dict(factor=a.factor, levels=a.levels, starts=a.starts, tasks=a.task, name=a.name)
    fam = F.load(a.family)
    overrides = json.loads(a.set)
    why = check(spec, list(fam["tasks"]), {a.name: None} if a.factor == "param" and a.name else None)
    if why:
        raise SystemExit(f"design: {why}")
    print(json.dumps(run(spec, a.family, a.pool, a.skills, overrides, a.out, a.procs, a.gpus)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
