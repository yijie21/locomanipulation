"""TODO 33.12: the stance-and-grasp chooser from the command line (locomani.tasks.scene.choose), without a simulation --
for a role or a person asking where to stand, which hand and which grasp for an object at a start, and why.

    PYTHONPATH=src .venv_sim/bin/python scripts/rsi/choose.py <scene dir> <object> [--start START.json] [--top N] [--json]

`--start` is a pool's start file (its `placements` put the free objects where that round began, its `robot_xy` is
where the walk starts); without it the scene file's poses are used and the walk is not scored. The scene's physics
model loads once (about 7 s). The answer lists the options best first, the one `choose()` would take, and what it
would say when none predicts a delivered reach. A prediction is for choosing what to try; only a closed-loop round is
evidence ([evidence])."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from locomani.tasks.scene import choose as C  # noqa: E402
from locomani.tasks.scene import graspability as G  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="the scene directory (scene_meta.json, scene_locomani.xml)")
    ap.add_argument("object", help="the object key in the scene file, e.g. Mug_2")
    ap.add_argument("--start", type=Path, default=None, help="a pool start file (placements, robot_xy)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="print the options as JSON instead of text")
    ap.add_argument("--two-hand", action="store_true", help="a two-hand clamp's options instead (TODO 33.1.5)")
    a = ap.parse_args(argv)
    scene = G.StaticScene(a.scene)
    start_xy = None
    if a.start is not None:
        st = json.loads(a.start.read_text())["start"]
        st = ast.literal_eval(st) if isinstance(st, str) else st
        scene.place(st.get("placements") or {})
        start_xy = st.get("robot_xy")
    if a.two_hand:
        two = C.two_hand(scene, a.object, start_xy)[: a.top]
        if a.json:
            print(json.dumps(dict(object=a.object, two_hand=two), ensure_ascii=False))
            return 0
        print(f"{a.object}: two-hand clamp, {len(two)} options shown")
        for i, o in enumerate(two):
            pz = "standing" if o["pelvis_z"] is None else f"pelvis {o['pelvis_z']:.2f} m"
            print(f"{i + 1}. approach {o['approach_deg']:+.0f} deg, {o['distance_m']:.2f} m ahead, {pz}, span "
                  f"{100 * o['span_m']:.1f} cm: miss odds left {o['p_left']:.0%} right {o['p_right']:.0%} either "
                  f"{o['p_both']:.0%}{'' if o['inside'] else ' (outside the measured grid)'}, clear {o['clear_m']:.2f} m, "
                  f"score {o['score']:.2f}")  # fmt: skip
        return 0
    opts = C.options(scene, a.object, start_xy)
    best = C.choose(scene, a.object, start_xy)
    if a.json:
        print(json.dumps(dict(object=a.object, chosen=best.as_dict() if best else None,
                              options=[o.as_dict() for o in opts[: a.top]]), ensure_ascii=False))  # fmt: skip
        return 0
    print(f"{a.object}: {len(opts)} options" + (f", from the start at {start_xy}" if start_xy else ""))
    print(C.explain(opts[: a.top]))
    if best is None:
        print("chosen: none -- nothing can be listed even with the clearance relaxed")
    else:
        print("chosen:\n" + C.explain([best]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
