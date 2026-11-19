"""simple_bench.py --film states rendered in SIMPLE's own Isaac scene (2026-11-19, the SIMPLE report's videos).

The physics is our bench's (the MuJoCo replica of the episode, episodes.build); SIMPLE only draws it. The scene is SIMPLE's:
gym.make("simple/<env>", sim_mode="mujoco_isaac") as official_server.py makes it, reset to the episode's
environment_config, so the table, the objects' USD looks, the lights and the robot are what SIMPLE's evaluation renders.
Each frame, as SIMPLE's own offline renderer does it (cli/replay_wbc.py _render_state_trace_with_isaac): the film's
state written into SIMPLE's MuJoCo model by joint name (the robot's root and joints, every object's free joint; heights
moved down by the replica's lift on SIMPLE's AMO tasks, whose table top is at z 0), mj_forward, isaac.step (its
sync_states), then two cameras read: SIMPLE's head camera, and a still world camera on the episode (CAM: the target
seen from the robot's right front, in the robot's heading at the start).

Run in SIMPLE's venv from its root:
    cd third_party/SIMPLE && MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 \
        .venv/bin/python <this file> <episodes root (outputs/simple_eval_20261112/x)> <out dir> FILM.npz ...
-> <out>/<film stem>_world.mp4 and _head.mp4 (H.264, 25 fps). --probe: the model's names and one frame per camera, no video.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("xdir")
ap.add_argument("out")
ap.add_argument("films", nargs="+")
ap.add_argument("--cam", default="-55,28,1.6", help="the world camera: azimuth deg (0 ahead, - right), elevation deg, m")
ap.add_argument("--look", default="-0.12,0,-0.03", help="m, the camera's aim from the target, heading frame at the start")
ap.add_argument("--size", default="960,540")
ap.add_argument("--probe", action="store_true")
a = ap.parse_args()
W, H = map(int, a.size.split(","))
os.makedirs(a.out, exist_ok=True)

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import simple.envs  # noqa: E402, F401
from simple.cli.eval_decoupled_wbc import _make_sonic_config  # noqa: E402

FREE = 0  # mjtJoint.mjJNT_FREE


def load(f):
    z = np.load(f, allow_pickle=False)
    return dict(t=z["t"], phase=z["phase"], qpos=z["qpos"], jnt=[str(n) for n in z["jnt"]], adr=z["jnt_adr"],
                typ=z["jnt_type"], meta=json.loads(str(z["meta"])), stem=Path(f).stem)  # fmt: skip


def look_at_wxyz(pos, target):
    """USD camera (looks along its -z, y up) at pos aimed at target -> its world quaternion, w x y z."""
    import transforms3d as t3d

    fwd = (target - pos) / np.linalg.norm(target - pos)
    z = -fwd
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return t3d.quaternions.mat2quat(np.column_stack([x, y, z]))


def yaw_of_wxyz(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Writer:
    def __init__(self, path, w, h):
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                   f"{w}x{h}", "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf",
                                   "20", "-preset", "medium", path], stdin=subprocess.PIPE)  # fmt: skip

    def write(self, rgb):
        self.p.stdin.write(np.ascontiguousarray(rgb[..., :3], dtype=np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


films = [load(f) for f in a.films]
groups = defaultdict(list)  # one scene per episode
for f in films:
    m = f["meta"]
    groups[(m["task"], str(m["level"]), int(m["index"]))].append(f)

env = env_task = None
for (task, level, index), fs in groups.items():
    if env_task != task:
        if env is not None:
            env.close()
        (name,) = os.listdir(f"{a.xdir}/{task}")
        env = gym.make(f"simple/{name}", sim_mode="mujoco_isaac", render_hz=50, headless=True,
                       sonic_config=_make_sonic_config())  # fmt: skip
        u = env.unwrapped
        u.sync_isaac_every_substep = False
        env_task = task
        cam = None
    (fj,) = glob.glob(f"{a.xdir}/{task}/*/dr-level-{level}/meta/episodes.jsonl")
    ep = [json.loads(line) for line in open(fj)][index]
    conf = ep["environment_config"]
    env.reset(options={"state_dict": json.loads(conf) if isinstance(conf, str) else conf})
    mm, dd = u.mujoco.mjModel, u.mujoco.mjData
    names = [mm.joint(j).name for j in range(mm.njnt)]
    free = [j for j in range(mm.njnt) if mm.jnt_type[j] == FREE]
    root = free[0]  # get_states: the robot's root is qpos[:7]
    assert mm.jnt_qposadr[root] == 0
    f0 = fs[0]
    ours_free = [j for j in range(len(f0["jnt"])) if f0["typ"][j] == FREE]
    # the replica's objects: "<label>_joint" on the body named after the label; SIMPLE's: its object free joints, by body
    obj_bodies = {mm.body(mm.jnt_bodyid[j]).name: j for j in free[1:]}
    if a.probe:
        print("SIMPLE joints", mm.njnt, "nq", mm.nq, "free:", [(names[j], mm.body(mm.jnt_bodyid[j]).name) for j in free])
        print("ours free:", [f0["jnt"][j] for j in ours_free], "objects", f0["meta"]["objects"], "lift", f0["meta"]["lift"])
        print("SIMPLE hinge names not in ours:", sorted(set(names) - set(f0["jnt"]) - {names[j] for j in free})[:20])
        print("ours hinge names not in SIMPLE:", sorted(set(f0["jnt"]) - set(names) - {f0["jnt"][j] for j in ours_free})[:20])
        print("isaac cameras", list(getattr(u.isaac, "cameras", {}).keys()), "mj_objects", list(u.mujoco.mj_objects.keys()))
    pairs = []  # (our qpos address, SIMPLE's, width)
    for j, n in enumerate(f0["jnt"]):
        if f0["typ"][j] == FREE:
            continue
        if n in names:
            k = names.index(n)
            pairs.append((int(f0["adr"][j]), int(mm.jnt_qposadr[k]), 1))
    ours_root = ours_free[0]
    pairs.append((int(f0["adr"][ours_root]), 0, 7))
    lift = float(f0["meta"]["lift"])
    zfix = [2]  # the qpos heights moved down by the lift: the root's, then each object's
    for key, label in f0["meta"]["objects"].items():
        jo = f0["jnt"].index(f"{label}_joint")
        k = obj_bodies.get(label, obj_bodies.get(key))
        if k is None:
            cand = [b for b in obj_bodies if label in b or key in b]
            k = obj_bodies[cand[0]] if len(cand) == 1 else None
        if k is None:
            print("no SIMPLE object for", key, label, "of", list(obj_bodies), flush=True)
            continue
        pairs.append((int(f0["adr"][jo]), int(mm.jnt_qposadr[k]), 7))
        zfix.append(int(mm.jnt_qposadr[k]) + 2)
    print(task, level, index, "mapped", len(pairs), "joint blocks, lift", round(lift, 3), flush=True)

    # the still world camera: on the target's place at the start, in the robot's heading then
    q0 = f0["qpos"][0]
    yaw = yaw_of_wxyz(q0[int(f0["adr"][ours_root]) + 3:int(f0["adr"][ours_root]) + 7])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    jt = f0["jnt"].index(f"{f0['meta']['target']}_joint")
    p_t = q0[int(f0["adr"][jt]):int(f0["adr"][jt]) + 3] - np.r_[0, 0, lift]
    aim = p_t + Rz @ np.array([float(v) for v in a.look.split(",")])
    az, el, dist = (float(v) for v in a.cam.split(","))
    az, el = np.radians(az), np.radians(el)
    pos = aim + Rz @ (dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]))
    from omni.isaac.sensor import Camera
    from pxr import UsdGeom

    if cam is None or not cam.prim.IsValid():  # a reset may rebuild the stage
        cam = Camera(prim_path="/World/film_cam", resolution=(W, H))
        cam.initialize()
        g = UsdGeom.Camera(cam.prim)
        g.GetFocalLengthAttr().Set(24.0)
        g.GetHorizontalApertureAttr().Set(20.955)
        g.GetVerticalApertureAttr().Set(20.955 * H / W)
        g.GetClippingRangeAttr().Set((0.05, 30.0))
    cam.set_world_pose(position=pos, orientation=look_at_wxyz(pos, aim), camera_axes="usd")
    heads = [k for k in getattr(u.isaac, "cameras", {}) if "head" in k and "right" not in k]

    def draw(q):
        for src, dst, w in pairs:
            dd.qpos[dst:dst + w] = q[src:src + w]
        for i in zfix:
            dd.qpos[i] -= lift
        dd.qvel[:] = 0
        mujoco.mj_forward(mm, dd)
        u.isaac.step(u.mujoco)
        return cam.get_rgba(), (u.isaac.cameras[heads[0]].get_rgba() if heads else None)

    for f in fs:
        for _ in range(8):  # the renderer's accumulation, before the first frame kept
            draw(f["qpos"][0])
        if a.probe:
            import cv2

            for i in (0, len(f["qpos"]) // 2, len(f["qpos"]) - 1):
                wv, hv = draw(f["qpos"][i])
                cv2.imwrite(f"{a.out}/{f['stem']}_probe{i}_world.png", cv2.cvtColor(wv[..., :3], cv2.COLOR_RGB2BGR))
                if hv is not None:
                    cv2.imwrite(f"{a.out}/{f['stem']}_probe{i}_head.png", cv2.cvtColor(hv[..., :3], cv2.COLOR_RGB2BGR))
            print("probe frames", f["stem"], flush=True)
            continue
        ww = Writer(f"{a.out}/{f['stem']}_world.mp4", W, H)
        hw = None
        for q in f["qpos"]:
            wv, hv = draw(q)
            ww.write(wv)
            if hv is not None:
                hw = hw or Writer(f"{a.out}/{f['stem']}_head.mp4", hv.shape[1], hv.shape[0])
                hw.write(hv)
        ww.close()
        if hw:
            hw.close()
        print("wrote", f["stem"], len(f["qpos"]), "frames", flush=True)
env.close()
sys.stdout.flush()
os._exit(0)  # Kit's shutdown can hang after the work is done
