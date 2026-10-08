"""ADT apartment through the dataset cameras: does each renderer draw it, what VRAM and time does it cost (TODO 19.6.5b).

Plan decision 16 picks the renderer by three things: it fits beside the other projects already on the card, its
time per frame, and how ADT's PBR textures come out (the user judges the images). Three stand poses square to
three surfaces the robot can approach (kitchen island, dining table, long kitchen counter), robot in its zero
pose. Two passes, because Kit and MuJoCo's EGL context do not share a process (isaac_view_check_20260924.py):

    CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES TMPDIR=/workspace/tmp PYTHONPATH=src "$RENDER_PY" \\
        scripts/experiments/adt_render_check_20261003.py isaac <out_dir>
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH=src "$VENV_SIM_PY" \\
        scripts/experiments/adt_render_check_20261003.py mujoco <out_dir>

The isaac pass writes isaac.npz and isaac_cost.json (process VRAM after warm-up and at the end, per-render
wall time over `REPEATS` jumps between the poses); the mujoco pass writes the same cameras and the grid
render_check_grid.png, one row per pose, Isaac | MuJoCo per camera.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from locomani.sim.dataset_cameras import CAMERA_CONFIGS  # noqa: E402

SCENE = REPO / "outputs" / "scene_discovery" / "adt_apartment"
SCENE_XML = SCENE / "scene_locomani.xml"
SCENE_DIR = Path("/workspace/code/locomanipulation/assets/isaac/scene/adt_apartment")
STANDS = ("KitchIsland__0", "ExtendableTable__0", "ApartmentEnv__6")
STANDOFF = 0.55  # m from the edge to the pelvis
PELVIS_Z = 0.793
REPEATS = 30
# auto-exposure off as in scripts/render_isaac_videos.sh; the film ISO is this scene's own (env RTX_FILM_ISO while tuning)
RTX = ["/rtx/post/histogram/enabled=false", f"/rtx/post/tonemap/filmIso={os.environ.get('RTX_FILM_ISO', '100')}"]
STREAMS = list(CAMERA_CONFIGS)


def stand_states() -> tuple[list[str], np.ndarray]:
    """qpos per stand: the model's own qpos0 with the pelvis square to the first open edge of each surface."""
    import mujoco

    meta = json.loads((SCENE / "scene_meta.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    states = []
    for key in STANDS:
        s = meta["surfaces"][key]
        edge = s["approach_edges"][0]
        ax, sgn = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1)}[edge]
        normal_local = np.zeros(2)
        normal_local[ax] = sgn
        c, sn = np.cos(s["yaw"]), np.sin(s["yaw"])
        normal = np.array([c * normal_local[0] - sn * normal_local[1], sn * normal_local[0] + c * normal_local[1]])
        xy = np.asarray(s["xy"]) + normal * (s["half_extent"][ax] + STANDOFF)
        yaw = float(np.arctan2(-normal[1], -normal[0]))  # facing the surface
        q = model.qpos0.copy()
        q[:3] = [xy[0], xy[1], PELVIS_Z]
        q[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        states.append(q)
    return list(STANDS), np.stack(states)


def gpu_mib(pid: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout  # fmt: skip
    for line in out.splitlines():
        p, mib = (v.strip() for v in line.split(","))
        if int(p) == pid:
            return int(mib)
    return -1


def isaac(out: Path) -> None:
    from locomani.sim.isaac import stage as S
    from locomani.sim.isaac.view import IsaacView

    names, qpos = stand_states()
    t0 = time.time()
    cfg = S.StageCfg(scene_dir=str(SCENE_DIR), robot_mjcf=str(SCENE_XML), absolute_camera=True)
    app = S.start_app(cfg)
    try:
        view = IsaacView(cfg, str(SCENE_XML), STREAMS, RTX)
        build_s = time.time() - t0
        steps = view.warm_up(qpos[0])
        warm_mib = gpu_mib(os.getpid())
        imgs = {k: [] for k in STREAMS}
        for q in qpos:
            view.warm_up(q)  # a still per pose, settled: this is the picture the user judges
            got = view.render(q)
            for k in STREAMS:
                imgs[k].append(got[k])
        times = []
        for i in range(REPEATS):  # jump between poses, as the on-demand mirror does
            t = time.time()
            view.render(qpos[i % len(qpos)])
            times.append(time.time() - t)
        cost = {
            "stage_build_s": round(build_s, 1),
            "warm_up_steps": steps,
            "vram_mib_after_warm_up": warm_mib,
            "vram_mib_end": gpu_mib(os.getpid()),
            "render_s_all_streams": {
                "p50": round(float(np.median(times)), 3),
                "p90": round(float(np.percentile(times, 90)), 3),
            },
            "streams": STREAMS,
            "poses": names,
        }
        np.savez_compressed(out / "isaac.npz", **{k: np.stack(v) for k, v in imgs.items()})
        (out / "isaac_cost.json").write_text(json.dumps(cost, indent=1))
        print(f"wrote {out / 'isaac.npz'}: {json.dumps(cost)}", flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(1)  # before close(): close() ends the process with status 0
    app.close()


def mujoco_pass(out: Path) -> None:
    import cv2
    import mujoco

    names, qpos = stand_states()
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    isa = np.load(out / "isaac.npz") if (out / "isaac.npz").exists() else None
    rs = {k: mujoco.Renderer(model, CAMERA_CONFIGS[k]["height"], CAMERA_CONFIGS[k]["width"]) for k in STREAMS}
    grid = []
    for j, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cells = []
        for k in STREAMS:
            rs[k].update_scene(data, camera=CAMERA_CONFIGS[k]["mjcf_name"])
            mj = rs[k].render()
            pair = np.concatenate([isa[k][j], mj], axis=1) if isa is not None else mj
            cv2.putText(
                pair, f"{names[j]} {k}: isaac | mujoco", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            cells.append(pair)
        grid.append(np.concatenate(cells, axis=1))
    img = np.concatenate(grid, axis=0)
    cv2.imwrite(str(out / "render_check_grid.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"wrote {out / 'render_check_grid.png'} {img.shape}", flush=True)


if __name__ == "__main__":
    which, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    {"isaac": isaac, "mujoco": mujoco_pass}[which](out)
