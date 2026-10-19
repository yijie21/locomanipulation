#!/usr/bin/env python3
"""Check a built scene before any loop spends on it (plan docs/handoff/scene-rsi-plan-20261016.md §12, user
2026-10-19). Runs in the sim venv; no model calls.

    $VENV_SIM_PY scripts/scene_lint.py --scene outputs/scene_discovery/adt_apartment_mars \\
        [--tasks <tasks.jsonl> ...] [--starts 8] [--rest-s 30] [--out <dir>]

Scene checks (scene_discovery.lint): upright axes off the resting pose, objects resting tilted off every body axis,
shadowed surfaces, objects too flat for a side grasp. The rest test loads `<scene>/env_only.xml` (no robot) at the runner's timestep, lets every free object
rest 1 s and then `--rest-s`, and reports the ones that leave their surface or drop (`fell`, an error) and the ones
that slide more than scene_settle.DRIFT_M or turn more than scene_settle.DRIFT_DEG (`drifts`, a warning). Task
checks, per task of every `--tasks` file: does the goal already hold at the scene pose, and at how many of
`--starts` starts drawn as a family pool draws them (start.sample_start, seeds from LINT_SEED)? Writes
`<out>/lint.json` (default `<scene>/lint.json`); exits 1 when there is an error:

- error: an upright mismatch, a shadowed surface, an object that falls at rest, a task whose goal holds at the
  scene pose or at every drawn start;
- warning: drift at rest, an object resting tilted, a flat object, a task whose goal holds at some drawn starts (family.draw skips those).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

import scene_settle as settle  # noqa: E402

from locomani.scene_discovery import lint  # noqa: E402

LINT_SEED = 9_000_000  # task j, start i: LINT_SEED + 1000 * j + i (no pool's block comes near it)
FALL_DZ_M = 0.02


def rest_test(scene: Path, meta: dict, seconds: float) -> dict:
    import mujoco

    m = settle.load(scene)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    joints = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    name = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])).removesuffix("_object")  # noqa: E731
    pose = lambda j: (d.qpos[m.jnt_qposadr[j] : m.jnt_qposadr[j] + 3].copy(), d.qpos[m.jnt_qposadr[j] + 3 : m.jnt_qposadr[j] + 7].copy())  # noqa: E731  # fmt: skip
    for _ in range(int(1.0 / settle.DT)):
        mujoco.mj_step(m, d)
    start = {j: pose(j) for j in joints}
    for _ in range(int(seconds / settle.DT)):
        mujoco.mj_step(m, d)
    fell, drifts = {}, {}
    for j in joints:
        (p0, q0), (p1, q1) = start[j], pose(j)
        key = name(j)
        row = dict(slid_mm=round(1000 * float(np.linalg.norm(p1 - p0)), 1), turned_deg=round(settle.angle_deg(q0, q1), 2),
                   dz_mm=round(1000 * float(p1[2] - p0[2]), 1))  # fmt: skip
        if key in meta["objects"] and (not settle.on_surface(meta, key, p1) or p1[2] - p0[2] < -FALL_DZ_M):
            fell[key] = row
        elif row["slid_mm"] > 1000 * settle.DRIFT_M or row["turned_deg"] > settle.DRIFT_DEG:
            drifts[key] = row
    return dict(seconds=seconds, n_free=len(joints), fell=fell, drifts=drifts)


def task_checks(meta: dict, files: list[Path], n_starts: int) -> dict:
    from locomani.primitives.navigator import build
    from locomani.tasks.scene.start import sample_start
    from locomani.tasks.scene.world import BODY_RADIUS_M, Task

    occ, nav = build(meta, body_radius=BODY_RADIUS_M)
    out = {}
    j = 0
    for f in files:
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        res = {}
        for row in rows:
            r = dict(scene_pose=False, starts=0, holds=[], unavailable="")
            try:
                r["scene_pose"] = lint.goal_holds(row["goal"], meta)
                t = Task.from_row(row, meta)
            except (ValueError, KeyError) as e:  # the checker calls such a goal invalid already
                res[row["id"]] = dict(r, unavailable=f"goal does not compile: {e}")
                continue
            for i in range(n_starts if not r["scene_pose"] else 0):
                seed = LINT_SEED + 1000 * j + i
                try:
                    st = sample_start(
                        meta, occ, nav, list(t.objects), np.random.default_rng(seed), seed=seed, room=t.room
                    )
                except RuntimeError as e:
                    r["unavailable"] = str(e)
                    continue
                r["starts"] += 1
                if lint.goal_holds(row["goal"], meta, st.to_dict()):
                    r["holds"].append(seed)
            res[row["id"]] = r
            j += 1
        out[str(f)] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, nargs="*", default=[])
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--rest-s", type=float, default=30.0)
    ap.add_argument("--no-rest", action="store_true", help="skip the rest test (it needs the simulator)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    meta = json.loads((a.scene / "scene_meta.json").read_text())
    t0 = time.time()
    rep = dict(scene=str(a.scene), upright_mismatch=lint.upright_mismatch(meta), shadowed=lint.shadowed(meta),
               tilted_at_rest=lint.tilted_at_rest(meta), flat_objects=lint.flat_objects(meta))  # fmt: skip
    if not a.no_rest:
        rep["rest"] = rest_test(a.scene, meta, a.rest_s)
    rep["tasks"] = task_checks(meta, a.tasks, a.starts)
    errors, warnings = [], []
    errors += [f"upright_axis of {k} is {v[0]}, it rests with {v[1]} up" for k, v in rep["upright_mismatch"].items()]
    errors += [f"surface {k} lies under {v}" for k, v in rep["shadowed"].items()]
    warnings += [f"{k} rests {v:.0f} deg off its upright axis: upright({k}) fails at the start"
                 for k, v in rep["tilted_at_rest"].items() if k not in rep["upright_mismatch"]]  # fmt: skip
    warnings += [f"{k} rests {v['half_height'] * 200:.1f} cm flat on {v['support']} (top {v['support_top']} m; pad "
                 f"margin {v['margin'] * 100:+.2f} cm)" for k, v in rep["flat_objects"].items()]  # fmt: skip
    if "rest" in rep:
        errors += [f"{k} falls or leaves its surface at rest: {v}" for k, v in rep["rest"]["fell"].items()]
        warnings += [f"{k} drifts at rest: {v}" for k, v in rep["rest"]["drifts"].items()]
    scene_errors = list(errors)
    for f, res in rep["tasks"].items():
        for tid, r in res.items():
            if r["scene_pose"]:
                errors.append(f"{Path(f).parent.name}/{tid}: the goal holds at the scene pose")
            elif r["starts"] and len(r["holds"]) == r["starts"]:
                errors.append(f"{Path(f).parent.name}/{tid}: the goal holds at all {r['starts']} drawn starts")
            elif r["holds"]:
                warnings.append(f"{Path(f).parent.name}/{tid}: the goal holds at {len(r['holds'])} of {r['starts']} drawn starts")  # fmt: skip
    rep.update(errors=errors, scene_errors=scene_errors, task_errors=errors[len(scene_errors) :], warnings=warnings,
               seconds=round(time.time() - t0))  # fmt: skip
    out = a.out or a.scene
    out.mkdir(parents=True, exist_ok=True)
    (out / "lint.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"[lint] {len(errors)} errors, {len(warnings)} warnings ({rep['seconds']} s) -> {out / 'lint.json'}")
    for e in errors:
        print("  error:", e)
    for w in warnings:
        print("  warning:", w)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
