"""Does IsaacView draw the warehouse right? Five states of one trace, every dataset camera, Isaac beside MuJoCo.

The timing in bench_render says nothing about the picture: a scene that failed to compose renders fast too.
Two passes, because Kit and MuJoCo's EGL context do not share a process well:

    CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src "$RENDER_PY" \
        scripts/experiments/isaac_view_check_20260924.py isaac <out_dir>
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src "$RENDER_PY" \
        scripts/experiments/isaac_view_check_20260924.py mujoco <out_dir>

The second pass also writes view_check_grid.png: one row per state, Isaac | MuJoCo per camera.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from locomani import paths  # noqa: E402
from locomani.sim.dataset_cameras import CAMERA_CONFIGS  # noqa: E402

RUN = REPO / "outputs" / "runs" / "docrepro_20260924_warehouse_offline"
TRACE = RUN / "trace" / "visual_crate_s7000_00_state_trace.npz"
SCENE_XML = RUN / "warehouse_textured" / "scene_locomani.xml"
SCENE_DIR = Path(paths.ISAAC_ASSETS) / "scene" / "warehouse_textured"
RTX = ["/rtx/post/histogram/enabled=false", "/rtx/post/tonemap/filmIso=30"]  # scripts/render_isaac_videos.sh
STREAMS = list(CAMERA_CONFIGS)


def picks(n: int) -> np.ndarray:
    return np.linspace(0, n - 1, 5).astype(int)


def isaac(out: Path) -> None:
    from locomani.sim.isaac import stage as S
    from locomani.sim.isaac.view import IsaacView

    tr = np.load(TRACE)
    qpos, qvel = tr["qpos"], tr["qvel"]
    cfg = S.StageCfg(scene_dir=str(SCENE_DIR), robot_mjcf=str(SCENE_XML), absolute_camera=True)
    app = S.start_app(cfg)
    try:
        view = IsaacView(cfg, str(SCENE_XML), STREAMS, RTX)
        rows = picks(len(qpos))
        view.warm_up(qpos[rows[0]], qvel[rows[0]])
        imgs = {k: [] for k in STREAMS}
        for i in rows:  # jump between states, as the on-demand mirror does
            got = view.render(qpos[i], qvel[i])
            for k in STREAMS:
                imgs[k].append(got[k])
        np.savez_compressed(out / "isaac.npz", rows=rows, **{k: np.stack(v) for k, v in imgs.items()})
        print(f"wrote {out / 'isaac.npz'}", flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(1)  # before close(): close() ends the process with status 0
    app.close()


def mujoco_pass(out: Path) -> None:
    import cv2
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    qpos = np.load(TRACE)["qpos"]
    isa = np.load(out / "isaac.npz")
    rows = isa["rows"]
    rs = {k: mujoco.Renderer(model, CAMERA_CONFIGS[k]["height"], CAMERA_CONFIGS[k]["width"]) for k in STREAMS}
    grid = []
    for j, i in enumerate(rows):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        cells = []
        for k in STREAMS:
            rs[k].update_scene(data, camera=CAMERA_CONFIGS[k]["mjcf_name"])
            mj = rs[k].render()
            pair = np.concatenate([isa[k][j], mj], axis=1)
            cv2.putText(pair, f"row {i} {k}: isaac | mujoco", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cells.append(pair)
        grid.append(np.concatenate(cells, axis=1))
    img = np.concatenate(grid, axis=0)
    img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "view_check_grid.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"wrote {out / 'view_check_grid.png'} {img.shape[1]}x{img.shape[0]}", flush=True)


if __name__ == "__main__":
    mode, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    isaac(out_dir) if mode == "isaac" else mujoco_pass(out_dir)
