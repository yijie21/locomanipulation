#!/usr/bin/env python3
"""Render one frame of a MuJoCo scene from one of its own cameras.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 "$VENV_SIM_PY" \
        scripts/render_mujoco_frame.py <scene_locomani.xml> <out.png> --camera reverse_room

Pair it with the Isaac render of the same camera to check that the two
simulators really hold the same layout; scripts/compare_frames.py reports how
far apart they are.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xml", help="the scene_locomani.xml to load")
    parser.add_argument("out", help="PNG to write")
    parser.add_argument("--camera", required=True, help="name of a camera in the model")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args(argv)

    import mujoco
    import numpy as np
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if cam_id < 0:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
        print(f"render: no camera named {args.camera}; this model has {names}")
        return 2

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = cam_id
    renderer.update_scene(data, camera=camera)
    image = np.asarray(renderer.render())
    Image.fromarray(image).save(args.out)
    print(f"render: mean luma {image.mean():.1f} wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
