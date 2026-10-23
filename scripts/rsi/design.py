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
  object_yaw      the task's object turned in place by `level` degrees about the vertical; left out when its footprint
                  then leaves its surface's top or meets a neighbour

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


def _rotated(meta: dict, key: str, start: dict, deg: float):
    """The object's placement turned by `deg` about the vertical, or None when it leaves its top or meets a neighbour."""
    obj = meta["objects"][key]
    pos, quat = start["placements"][key]
    q = ST._qmul(ST._yaw_quat(math.radians(deg)), np.asarray(quat, float))
    surf = meta["surfaces"][obj["surface"]]
    flat = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)[:2]
    top = (np.asarray(surf["xy"], float), flat - ST.SURFACE_MARGIN_M, float(surf.get("yaw", 0.0)))
    fp = ST.footprint(obj, np.asarray(pos, float)[:2], q)
    others = []
    for k, o in meta["objects"].items():
        if k == key or o.get("surface") != obj["surface"]:
            continue
        p, qq = start["placements"].get(k, (o["init_pos"], o["init_quat"]))
        others.append(ST.footprint(o, np.asarray(p, float)[:2], qq))
    if not ST.inside_top(fp, top) or any(ST.overlap(fp, t, ST.OBJECT_GAP_M) for t in others):
        return None
    return [list(map(float, pos)), q.tolist()]


def transform(factor: str, level: float, row: dict, key: str, meta: dict, nav=None) -> dict | None:
    """The start under one level of a start factor; None when the forced start is not valid."""
    s = json.loads(json.dumps(row["start"]))
    obj = np.asarray(s["placements"][key][0], float)[:2]
    robot = np.asarray(s["robot_xy"], float)
    to_obj = math.atan2(obj[1] - robot[1], obj[0] - robot[0])
    if factor == "spawn_bearing":
        s["robot_yaw"] = to_obj - math.radians(level)
    elif factor == "spawn_distance":
        u = robot - obj
        u = u / (np.linalg.norm(u) or 1.0)
        xy = obj + u * float(level)
        if nav is not None:
            occ, navi = nav
            navi.set_goal_ring(obj, *ST.APPROACH_RING_M)
            if not occ.is_free(xy) or not navi.reachable(xy):
                return None
        bearing = s["robot_yaw"] - to_obj  # keep the object where it was in the robot's view
        s["robot_xy"] = [float(xy[0]), float(xy[1])]
        s["robot_yaw"] = math.atan2(obj[1] - xy[1], obj[0] - xy[0]) + bearing
    elif factor == "object_yaw":
        spot = _rotated(meta, key, s, float(level))
        if spot is None:
            return None
        s["placements"][key] = spot
    return dict(row, start=s)


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
    jobs, skipped = [], collections.Counter()
    for i, lv in enumerate(levels):
        labels = []
        for label, row in starts:
            new = (
                row if factor == "param" else transform(factor, lv, row, _object_of(fam, row["task"]), fam["meta"], nav)
            )
            if new is None:
                skipped[i] += 1
                continue
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
        per.append(dict(level=lv, n=n, ok=ok, rate=round(ok / n, 3) if n else None, skipped=skipped[i],
                        classes=dict(collections.Counter(r.get("failure_class") for r in rounds if not r["ok"]))))  # fmt: skip
    res = dict(factor=factor, name=spec.get("name"), tasks=tasks, starts=len(starts), levels=per, dir=str(out))
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
