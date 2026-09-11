"""Offline EGL RGB localization check against known scene placements (no stack)."""

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_ik import ArmIK
from locomani.tasks.bottle.image_bottle import CameraGeometry, fit_bottle
from locomani.tasks.bottle.ready_grasp_policy import ready_q
from locomani.tasks.bottle.recoverable_grasp_policy import retracted_q


def main():
    out = Path("outputs/experiments/image_servo_20260911/calibration")
    out.mkdir(parents=True, exist_ok=True)
    ik = ArmIK()
    m, d = ik.model, ik.data
    obj = m.body("water_bottle_object")
    adr = m.jnt_qposadr[m.body_jntadr[obj.id]]
    renderer = mujoco.Renderer(m, height=480, width=640)
    results = []
    cases = [(d, False) for d in (1.2, 0.7, 0.42, 0.31)] + [(0.31, True), (0.4, True)]
    for index, (depth, retracted) in enumerate(cases):
        mujoco.mj_resetData(m, d)
        target = d.qpos[adr : adr + 3].copy()
        d.qpos[:3] = target + [0.27, depth, 0.793 - target[2]]
        d.qpos[3:7] = Rotation.from_euler("z", -np.pi / 2).as_quat(scalar_first=True)
        d.qpos[ik.qadr] = retracted_q() if retracted else ready_q()
        mujoco.mj_forward(m, d)
        rotation = d.xmat[m.body("pelvis").id].reshape(3, 3)
        base = d.xpos[m.body("pelvis").id]
        cameras, images = {}, {}
        for key, name in (("ego_view", "head_camera"), ("right_wrist", "right_wrist")):
            camera = m.camera(name).id
            renderer.update_scene(d, camera=camera)
            images[key] = renderer.render().copy()
            cameras[key] = CameraGeometry(
                rotation.T @ (d.cam_xpos[camera] - base),
                rotation.T @ d.cam_xmat[camera].reshape(3, 3),
                float(m.cam_fovy[camera]),
            )
            cv2.imwrite(str(out / f"{index}_{key}.png"), cv2.cvtColor(images[key], cv2.COLOR_RGB2BGR))
        fit = fit_bottle(images, cameras, rotation.T @ [0, 0, 1])
        truth = rotation.T @ (target - base)
        row = dict(
            depth=depth,
            retracted=retracted,
            truth=truth.tolist(),
            estimate=None if fit is None else fit["centre"].tolist(),
        )
        if fit:
            row.update(
                error_m=float(np.linalg.norm(fit["centre"] - truth)),
                selected=fit["selected"],
                views=[
                    dict(camera=v["camera"], error_px=v["error_px"], pixels=v["pixels"], centre=v["centre"].tolist())
                    for v in fit["views"]
                ],
            )
        results.append(row)
    renderer.close()
    (out / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
