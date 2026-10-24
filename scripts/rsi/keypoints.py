"""TODO 33.9: the keypoint-constraint grasp from the command line (locomani.tasks.scene.keypoints), without a simulation --
for a role or a person asking which points an object offers and what grasps a set of constraints gives, ranked by the
chooser and the body model.

    PYTHONPATH=src .venv_sim/bin/python scripts/rsi/keypoints.py <scene dir> <object> --points
    PYTHONPATH=src .venv_sim/bin/python scripts/rsi/keypoints.py <scene dir> <object> \\
        --spec '{"pinch": ["hull:-y", "hull:+y"], "twist": 45}' [--start START.json] [--top N] [--json]

`--points` lists the named points (world xyz and normal). `--spec` is a JSON constraint set (the module's docstring:
pinch, centre, approach, closing, depth, z, twist); the answer lists the solved poses' count and the options best first
(attitude, hand, facing, distance, posture, odds of a miss, fit margin, how far the jaw was raised off the support).
`--start` is a pool's start file (its placements and robot_xy). The scene's physics model loads once (about 7 s). A
prediction is for choosing what to try; only a closed-loop round is evidence (rules.md 7)."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from locomani.tasks.scene import choose as C  # noqa: E402
from locomani.tasks.scene import graspability as G  # noqa: E402
from locomani.tasks.scene import keypoints as K  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="the scene directory (scene_meta.json, scene_locomani.xml)")
    ap.add_argument("object", help="the object key in the scene file, e.g. Mug_2")
    ap.add_argument("--points", action="store_true", help="list the named points")
    ap.add_argument("--spec", default=None, help="a JSON constraint set")
    ap.add_argument("--start", type=Path, default=None, help="a pool start file (placements, robot_xy)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    scene = G.StaticScene(a.scene)
    start_xy = None
    if a.start is not None:
        st = json.loads(a.start.read_text())["start"]
        st = ast.literal_eval(st) if isinstance(st, str) else st
        scene.place(st.get("placements") or {})
        start_xy = st.get("robot_xy")
    if a.points or not a.spec:
        pts = {n: dict(xyz=[round(float(v), 4) for v in p.xyz],
                       normal=None if p.normal is None else [round(float(v), 3) for v in p.normal])
               for n, p in K.points(scene, a.object).items()}  # fmt: skip
        print(json.dumps(pts) if a.json else "\n".join(f"{n}: {v['xyz']} normal {v['normal']}" for n, v in pts.items()))
        if not a.spec:
            return 0
    spec = json.loads(a.spec)
    poses = K.solve(scene, a.object, spec)
    opts = K.plan(scene, a.object, spec, start_xy)
    if a.json:
        print(
            json.dumps(dict(object=a.object, spec=spec, poses=len(poses), options=[o.as_dict() for o in opts[: a.top]]))
        )
        return 0
    print(f"{a.object}: {len(poses)} poses, {len(opts)} options" + (f", from {start_xy}" if start_xy else ""))
    print(C.explain(opts[: a.top]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
