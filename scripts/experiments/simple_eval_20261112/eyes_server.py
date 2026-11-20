"""SIMPLE's Isaac scene as the eyes of the replica (2026-11-19): the replica (episodes.build) is the physics, SIMPLE draws it.

A client (blind.py --eyes) sends the replica's state. This server writes it into SIMPLE's MuJoCo model by joint name, as
SIMPLE's own offline renderer does (cli/replay_wbc.py _render_state_trace_with_isaac) and as the report's videos were
drawn: the robot's root and joints, every object's free joint, heights moved down by the replica's lift on SIMPLE's AMO
tasks (their table top is at z 0). Then mj_forward, isaac.step, and the head stereo pair comes back. The scene is SIMPLE's
(gym.make("simple/<env>", sim_mode="mujoco_isaac") reset to the episode's environment_config): the room, the table, the
objects' looks and the lights are what SIMPLE's evaluation renders. One task per process: closing an environment ends
the Kit app.

    cd third_party/SIMPLE && MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 [SIMPLE_OLD_TASK=<task file>] \
        .venv/bin/python <this file> <socket path> <episodes root (outputs/simple_eval_20261112/x)>

messages: ("scene", task, level, index, dict(jnt, adr, typ, objects, lift)) -> dict(mapped, cameras, instruction)
          ("eyes", qpos) -> dict(left, right)    ("quit",)
SIMPLE_OLD_TASK: XMoveBendPickTeleop left SIMPLE in eef474f; its task file from eef474f^ (git show), registered here.
"""

import faulthandler
import glob
import json
import os
import sys
from multiprocessing.connection import Listener

import numpy as np

faulthandler.dump_traceback_later(900, repeat=True)
sock, xdir = sys.argv[1], sys.argv[2]

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import simple.envs  # noqa: E402, F401
from simple.cli.eval_decoupled_wbc import _make_sonic_config  # noqa: E402

FREE = 0  # mjtJoint.mjJNT_FREE
SETTLE = 8  # renders before the frame kept: the renderer's accumulation

old = os.environ.get("SIMPLE_OLD_TASK")
if old and "simple/G1WholebodyXMoveBendPickTeleop-v0" not in gym.registry:
    import importlib.util

    spec = importlib.util.spec_from_file_location("simple.tasks.g1_wholebody_xmove_bend_pick_teleop", old)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # registers itself in TaskRegistry
    gym.register(id="simple/G1WholebodyXMoveBendPickTeleop-v0", entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv",
                 kwargs={"task": "g1_wholebody_xmove_bend_pick_teleop"})  # fmt: skip


class Scene:
    def __init__(self, task):
        self.task = task
        (name,) = os.listdir(f"{xdir}/{task}")  # the episodes' folder carries SIMPLE's name of the environment
        self.env = gym.make(f"simple/{name}", sim_mode="mujoco_isaac", render_hz=50, headless=True,
                            sonic_config=_make_sonic_config())  # fmt: skip
        self.u = self.env.unwrapped
        self.u.sync_isaac_every_substep = False

    def reset(self, level, index, ours):
        (f,) = glob.glob(f"{xdir}/{self.task}/*/dr-level-{level}/meta/episodes.jsonl")
        ep = [json.loads(line) for line in open(f)][index]
        conf = ep["environment_config"]
        self.env.reset(options={"state_dict": json.loads(conf) if isinstance(conf, str) else conf})
        mm = self.u.mujoco.mjModel
        names = [mm.joint(j).name for j in range(mm.njnt)]
        free = [j for j in range(mm.njnt) if mm.jnt_type[j] == FREE]
        assert mm.jnt_qposadr[free[0]] == 0  # the robot's root is qpos[:7]
        obj_bodies = {mm.body(mm.jnt_bodyid[j]).name: j for j in free[1:]}
        jnt, adr, typ = list(ours["jnt"]), np.asarray(ours["adr"]), np.asarray(ours["typ"])
        ours_free = [j for j in range(len(jnt)) if typ[j] == FREE]
        pairs = [(int(adr[j]), int(mm.jnt_qposadr[names.index(n)]), 1) for j, n in enumerate(jnt)
                 if typ[j] != FREE and n in names]  # fmt: skip
        pairs.append((int(adr[ours_free[0]]), 0, 7))
        zfix, missing = [2], []  # the qpos heights moved down by the lift: the root's, then each object's
        for key, label in ours["objects"].items():
            k = obj_bodies.get(label, obj_bodies.get(key))
            if k is None:
                cand = [b for b in obj_bodies if label in b or key in b]
                k = obj_bodies[cand[0]] if len(cand) == 1 else None
            if k is None:
                missing.append((key, label))
                continue
            pairs.append((int(adr[jnt.index(f"{label}_joint")]), int(mm.jnt_qposadr[k]), 7))
            zfix.append(int(mm.jnt_qposadr[k]) + 2)
        self.pairs, self.zfix, self.lift = pairs, zfix, float(ours["lift"])
        cams = list(getattr(self.u.isaac, "cameras", {}).keys())
        self.left = next(k for k in cams if "head" in k and "right" not in k)
        self.right = next(k for k in cams if "head" in k and "right" in k)
        return dict(mapped=len(pairs), missing=missing, cameras=cams, left=self.left, right=self.right,
                    unmatched=sorted(set(n for j, n in enumerate(jnt) if typ[j] != FREE) - set(names))[:10],
                    instruction=self.u.task.instruction)  # fmt: skip

    def eyes(self, q):
        mm, dd = self.u.mujoco.mjModel, self.u.mujoco.mjData
        for _ in range(SETTLE):
            for src, dst, w in self.pairs:
                dd.qpos[dst : dst + w] = q[src : src + w]
            for i in self.zfix:
                dd.qpos[i] -= self.lift
            dd.qvel[:] = 0
            mujoco.mj_forward(mm, dd)
            self.u.isaac.step(self.u.mujoco)
        cam = self.u.isaac.cameras
        return dict(left=np.asarray(cam[self.left].get_rgba())[..., :3].copy(),
                    right=np.asarray(cam[self.right].get_rgba())[..., :3].copy())  # fmt: skip


sc = None
with Listener(sock, family="AF_UNIX") as lis:
    print("listening", sock, flush=True)
    while True:
        with lis.accept() as c:
            try:
                while True:
                    msg = c.recv()
                    if msg[0] == "scene":
                        if sc is None or sc.task != msg[1]:
                            assert sc is None, "one task per process: closing an environment ends the Kit app"
                            sc = Scene(msg[1])
                        c.send(sc.reset(msg[2], msg[3], msg[4]))
                    elif msg[0] == "eyes":
                        c.send(sc.eyes(np.asarray(msg[1], float)))
                    elif msg[0] == "quit":
                        sys.stdout.flush()
                        os._exit(0)  # Kit's shutdown can hang after the work is done
            except EOFError:
                pass
