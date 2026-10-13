"""Side-by-side video of one warehouse round (TODO 25.3, plan §8): left SONIC, right ScaleBFM, both rendered from their
warehouse_states.py round files on the same mars scene and the same camera, the commanded jaw centres drawn as
translucent ghost spheres (magenta left hand, cyan right hand); the room's dressing meshes are hidden -- SONIC's from its streamed frame, ScaleBFM's from the
keypoints it was given (SONIC's measured hands). The two clocks are aligned on `close`. Writes the full clip and a
<= 500 KB preview.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/controller_compare_20261013/warehouse_video.py \\
        outputs/controller_compare_20261013/warehouse/sonic/states/sonic_s7000_00.npz \\
        outputs/controller_compare_20261013/warehouse/bfm_replay_m2p/bfm_replay_m2p_s7000_00.npz \\
        --out outputs/controller_compare_20261013/warehouse/videos/s7000_00_m2p.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from warehouse_states import SCENE  # noqa: E402

OFF = np.array([0.1515, 0.0, 0.0])  # wrist frame -> jaw centre (dex1; backends/scalebfm/runtime.OFF)
W, H, FPS = 640, 480, 25
GHOST = ((1.0, 0.2, 1.0, 0.45), (0.2, 1.0, 1.0, 0.45))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def quat_mat(q):
    r = np.empty(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, float))
    return r.reshape(3, 3)


class View:
    """A free camera behind and to the left of the robot's walk-in, looking at the box."""

    def __init__(self, m, edge_axis, box_xy):
        self.m, self.d = m, mujoco.MjData(m)
        self.r = mujoco.Renderer(m, height=H, width=W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth = float(np.degrees(np.arctan2(edge_axis[1], edge_axis[0]))) + 180 - 55
        self.cam.elevation = -25
        self.cam.distance = 2.0
        self.cam.lookat[:] = [box_xy[0] - 0.3 * edge_axis[0], box_xy[1] - 0.3 * edge_axis[1], 0.55]
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0  # the sim's com_marker

    def frame(self, q, ghosts):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        self.r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        scn = self.r.scene
        for k, p in enumerate(ghosts):
            if p is None or scn.ngeom >= scn.maxgeom:
                continue
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.035, 0, 0]),
                np.asarray(p, float),
                np.eye(3).reshape(9),
                np.asarray(GHOST[k], np.float32),
            )
            scn.ngeom += 1
        return self.r.render()


def ghosts_at(z, i):
    out = []
    for k in range(2):
        c = z["cmd_wrist"][i, k]
        out.append(None if not np.isfinite(c[0]) else c[:3] + quat_mat(c[3:]) @ OFF)
    return out


def caption(f):
    who = "SONIC" if f["controller"] == "sonic" else f"ScaleBFM {f['program']}"
    b = f.get("_rise")
    return f"{who}  box rise at end {100 * b:.1f} cm"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--before", type=float, default=8.0, help="seconds shown before `close`")
    ap.add_argument("--after", type=float, default=9.0, help="seconds shown after `close`")
    a = ap.parse_args(argv)
    m = mujoco.MjModel.from_xml_path(str(SCENE / "scene_locomani.xml"))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
    for g in range(m.ngeom):  # the room's dressing meshes block every view of the platform: hidden (group 5)
        if m.geom(g).name.startswith("vis_room"):
            m.geom_group[g] = 5
    m.geom_rgba[m.geom("platform_col").id] = (0.45, 0.45, 0.5, 1.0)  # the platform's own visual is a room mesh
    sides = []
    for p in (a.left, a.right):
        z = dict(np.load(p))
        f = json.loads(p.with_suffix(".json").read_text())
        f["_rise"] = float(z["box"][-1, 2] - z["box"][0, 2])
        sides.append((z, f, View(m, np.asarray(f["edge"]["axis"], float), z["box"][0, :2])))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    raw = a.out.with_suffix(".raw.mp4")
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{2 * W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(raw),
        ],
        stdin=subprocess.PIPE,
    )
    for s in np.arange(-a.before, a.after, 1.0 / FPS):
        tiles = []
        for z, f, v in sides:
            i = int(np.clip(np.searchsorted(z["t"], f["events"]["close"] + s), 0, len(z["t"]) - 1))
            tiles.append(v.frame(z["qpos"][i], ghosts_at(z, i)))
        ff.stdin.write(np.hstack(tiles).tobytes())
    ff.stdin.close()
    ff.wait()
    font = f"fontfile={FONT}:" if Path(FONT).exists() else ""
    txt = lambda text, x: f"drawtext={font}text='{text}':x={x}:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5"
    vf = ",".join([txt(caption(sides[0][1]), 8), txt(caption(sides[1][1]), W + 8)])
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(raw),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(a.out),
        ],
        check=True,
    )
    raw.unlink()
    preview = a.out.with_name(a.out.stem + "_preview.mp4")
    for crf, scale in ((30, 960), (34, 800), (38, 640)):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(a.out),
                "-vf",
                f"scale={scale}:-2",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(preview),
            ],
            check=True,
        )
        if preview.stat().st_size <= 500_000:
            break
    print(a.out, preview, preview.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
