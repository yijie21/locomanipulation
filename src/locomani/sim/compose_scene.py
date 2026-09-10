"""Render the two-surface scene as a placement check.

The arrangement itself lives in scene_lib.py, which build_sim_scene.py and
isaac_scene.py also read, so what this renders is what the sim runs. Objects
are frozen here (free joints stripped) because a placement check should show
where things were *put*, not where they settled after a physics step.

Usage:
    python src/locomani/sim/compose_scene.py <scene_root> <mjcf_root_ignored> <out_dir>
    python src/locomani/sim/compose_scene.py <scene_root> <out_dir>
    python src/locomani/sim/compose_scene.py --preset pick_place <scene_root> <out_dir>
"""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np
from PIL import Image

from locomani.sim import scene_lib  # noqa: E402


def render_views(model, data, out_dir: str, width=1280, height=960):
    renderer = mujoco.Renderer(model, height=height, width=width)
    views = {
        "front": dict(azimuth=90, elevation=-12, distance=6.5),
        "side": dict(azimuth=0, elevation=-12, distance=6.5),
        "top": dict(azimuth=90, elevation=-65, distance=6.5),
    }
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    # Centre on the middle of the walk, not on surface A.
    cam.lookat[:] = [0.5 * (scene_lib.ROBOT_START_XY[0] + scene_lib.SURFACE_B_XY[0]), 0.0, 0.75]
    for name, params in views.items():
        cam.azimuth, cam.elevation, cam.distance = (params["azimuth"], params["elevation"], params["distance"])
        renderer.update_scene(data, camera=cam)
        path = os.path.join(out_dir, f"scene_{name}.png")
        Image.fromarray(renderer.render()).save(path)
        print(f"  wrote {path}")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    # --preset NAME narrows the table like build_sim_scene.py's flag (scene_lib.SCENE_PRESETS).
    preset = ""
    if "--preset" in args:
        i = args.index("--preset")
        preset = args[i + 1]
        del args[i : i + 2]
    scene_root, out_dir = args[0], args[-1]
    os.makedirs(out_dir, exist_ok=True)

    spec, meta = scene_lib.build(scene_root, freeze_objects=True, preset=preset)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print(f"\ncompiled: {model.nbody} bodies, {model.ngeom} geoms")
    render_views(model, data, out_dir)


if __name__ == "__main__":
    main()
