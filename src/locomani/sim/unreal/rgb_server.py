"""Unreal's RGB of a MuJoCo state for vision RSI L3 (locomani.tasks.scene.services): the open editor's level of the
scene, read through the robot's three cameras (`UnrealView`), served to round workers on services.UNREAL_ADDRESS.
A request {"qpos": the scene's full qpos} gets {"images": {stream: HxWx3 uint8}} for ego_view (head_camera), left_wrist
and right_wrist, 640 x 480, captured fresh (`repeats` captures for Lumen and TSR to settle). Runs in the bridge venv,
with the editor up (scripts/start_unreal.sh --gpu 1):

    PYTHONPATH=src "$BRIDGE_PY" -m locomani.sim.unreal.rgb_server \\
        --scene outputs/scene_discovery/adt_apartment_mars_physics/scene_locomani.xml --level /Game/Levels/adt_apartment_mars

Measured 2026-10-23 on the apartment: 22 s to open the level and start PIE, then about 0.18 s a request (8 captures)."""

from __future__ import annotations

import argparse

import numpy as np

from locomani.sim.unreal.view import UnrealView
from locomani.tasks.scene.services import UNREAL_ADDRESS, serve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="the scene's MuJoCo XML (joints are matched by name)")
    ap.add_argument("--level", required=True, help="the editor level, e.g. /Game/Levels/adt_apartment_mars")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--port", type=int, default=UNREAL_ADDRESS[1])
    a = ap.parse_args(argv)
    view = UnrealView(a.scene, a.level, a.repeats)
    try:
        serve((UNREAL_ADDRESS[0], a.port), lambda req: dict(images=view.render(np.asarray(req["qpos"], float))))
    finally:
        view.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
