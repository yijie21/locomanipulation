"""TODO 33.18: the geometric graspability check for roles (src/locomani/tasks/scene/graspability.py). Prints, for a
task's object at a start's placement (or the scene file's), the jaw's fit at an approach, the smallest rotation of the
approach that fits, a container's rim pinch, a flat object's top pinch, and an ordered list of recommendations. No
simulation: it loads the scene's physics model once (about 7 s).

    PYTHONPATH=src python3 scripts/rsi/graspability.py grasp_atomic pick_DishSoap_1 \\
        [--start outputs/scene_rsi_20261016/grasp_atomic/pool_diag/pick_DishSoap_1_s410012.json] [--approach-deg 0.6] \\
        [--object DishSoap_1] [--json]

`--approach-deg` is the horizontal direction from the shoulder to the object (a record's `geom.approach_deg`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src")]

from locomani.tasks.scene import graspability as G  # noqa: E402


def task_objects(family: str, task: str, meta: dict) -> list[str]:
    fam = json.loads((HERE / "families" / f"{family}.json").read_text())
    rows = [json.loads(x) for x in (REPO / fam["tasks_file"]).read_text().splitlines() if x.strip()]
    row = next((r for r in rows if r["id"] == task), None)
    if row is None:
        raise SystemExit(f"{task}: not in {fam['tasks_file']}")
    keys = []
    for atom in row["goal"]:
        for arg in re.findall(r"\(([^)]*)\)", atom):
            keys += [a.strip() for a in arg.split(",") if a.strip() in meta["objects"] and a.strip() not in keys]
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("task")
    ap.add_argument("--start", type=Path, help="a pool start (its placements); default: the scene file's poses")
    ap.add_argument("--approach-deg", type=float, help="the side grasp's approach yaw, degrees")
    ap.add_argument("--object", help="the object (default: the task goal's first object)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    fam = json.loads((HERE / "families" / f"{a.family}.json").read_text())
    scene = G.StaticScene(REPO / fam["scene"])
    key = a.object or task_objects(a.family, a.task, scene.meta)[0]
    if a.start:
        row = json.loads(a.start.read_text())
        scene.place((row.get("start") or row)["placements"])
    res = G.assess(scene, key, None if a.approach_deg is None else np.radians(a.approach_deg))
    if a.json:
        print(json.dumps(res, indent=1))
        return 0
    print(f"{key}: grasp height {res['z']:.3f} m, resting half-height {res['resting_half_height_m'] * 100:.1f} cm, "
          f"jaw {G.JAW_OPEN_M * 100:.1f} cm open")  # fmt: skip
    for k in ("side", "narrowest", "rotate", "rim", "top_pinch"):
        if res.get(k) is not None:
            print(f"  {k}: {json.dumps(res[k])}")
    print("recommend:")
    for r in res["recommend"] or ["(nothing)"]:
        print("  -", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
