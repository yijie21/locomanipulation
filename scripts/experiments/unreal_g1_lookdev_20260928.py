"""Full-body G1 look-dev views for the Unreal renderer (same camera in MuJoCo and Unreal).

Adds two world cameras to the exported dynamic.xml and scene.xml (unreal_export_mjcf_20260928.py) that frame the
standing robot the way MuJoCo's free camera does at azimuth +-20 deg around its front (distance 2.2 m, elevation
-8 deg, fovy 45 -- the framing of the URLab G1 comparison render), and writes a one-row "trace" that puts the robot
in its default standing pose on open floor, so unreal_smoke_20260928.py replay renders Unreal | MuJoCo for it.

    PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/unreal_g1_lookdev_20260928.py outputs/unreal_warehouse_20260928/mjcf
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

ROBOT_XY = (-1.2, 1.7)  # open floor in front of the far shelves (top-down check in g1_lookdev/topdown.png)
ROBOT_YAW_DEG = -90.0  # facing -y, towards the camera side
VIEWS = {"g1_view_a": -20.0, "g1_view_b": 20.0}  # azimuth offsets from straight-on
DISTANCE, ELEVATION_DEG, FOVY = 2.2, -8.0, 45.0


def free_camera(lookat, azimuth_deg, elevation_deg, distance):
    """MuJoCo free-camera pose as an MJCF camera: position and xyaxes (x right, y up, looking along -z)."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = np.asarray(lookat) - distance * fwd
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, np.concatenate([right, up])


def standing_qpos(model) -> np.ndarray:
    q = model.qpos0.copy()
    adr = model.jnt_qposadr[model.joint("floating_base_joint").id]
    half = math.radians(ROBOT_YAW_DEG) / 2
    q[adr : adr + 3] = [ROBOT_XY[0], ROBOT_XY[1], q[adr + 2]]
    q[adr + 3 : adr + 7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mjcf_dir", type=Path)
    args = ap.parse_args()
    scene = mujoco.MjModel.from_xml_path(str(args.mjcf_dir / "scene.xml"))
    data = mujoco.MjData(scene)
    data.qpos[:] = standing_qpos(scene)
    mujoco.mj_forward(scene, data)
    lookat = data.xpos[scene.body("torso_link").id] - np.array([0.0, 0.0, 0.25])
    front = math.degrees(math.atan2(-math.sin(math.radians(ROBOT_YAW_DEG)), -math.cos(math.radians(ROBOT_YAW_DEG))))
    for name in ("dynamic.xml", "scene.xml"):
        spec = mujoco.MjSpec.from_file(str(args.mjcf_dir / name))
        for cam in list(spec.worldbody.cameras):
            if cam.name in VIEWS:
                spec.delete(cam)
        for view, offset in VIEWS.items():
            pos, xyaxes = free_camera(lookat, front + offset, ELEVATION_DEG, DISTANCE)
            cam = spec.worldbody.add_camera(name=view, pos=pos, fovy=FOVY)
            cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
            cam.alt.xyaxes = xyaxes
        spec.modelfiledir = str(args.mjcf_dir)
        spec.compile()
        (args.mjcf_dir / name).write_text(spec.to_xml())
    np.savez(
        args.mjcf_dir / "lookdev_trace.npz",
        seq=np.array([0]),
        qpos=data.qpos[None].copy(),
        qvel=np.zeros((1, scene.nv)),
    )
    print(json.dumps({"lookat": lookat.round(3).tolist(), "front_azimuth": front, "views": list(VIEWS)}))


if __name__ == "__main__":
    main()
