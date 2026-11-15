"""SIMPLE's own environment behind a socket: MuJoCo physics, Isaac Sim rendering, the task's step limit and success check, all
as in third_party/SIMPLE/src/simple/cli/eval_decoupled_wbc.py. The client (official.py) sends joint targets and gains for the
43 motors each 20 ms step and gets back the joint readings, the base state (IMU) and the episode's flags; the head stereo pair
on request. Run in SIMPLE's venv from its root:

    cd third_party/SIMPLE && MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python <this file> <socket path> <episodes root>
"""

import faulthandler
import glob
import json
import os
import sys
import types
from multiprocessing.connection import Listener

import numpy as np

faulthandler.dump_traceback_later(900, repeat=True)
sock, xdir = sys.argv[1], sys.argv[2]

import gymnasium as gym  # noqa: E402
import simple.envs  # noqa: E402, F401
from gymnasium.wrappers import TimeLimit  # noqa: E402
from simple.cli.eval_decoupled_wbc import _make_sonic_config  # noqa: E402
from simple.core.action import ActionCmd  # noqa: E402


class Episode:
    def __init__(self, task):
        self.task = task
        (name,) = os.listdir(f"{xdir}/{task}")  # the episodes' folder carries SIMPLE's name of the environment
        raw = gym.make(f"simple/{name}", sim_mode="mujoco_isaac", render_hz=50, headless=True,
                       sonic_config=_make_sonic_config())  # fmt: skip
        self.u = raw.unwrapped
        self.limit = self.u.task.metadata.get("max_episode_steps")
        self.env = TimeLimit(raw, max_episode_steps=self.limit)
        self.u.sync_isaac_every_substep = (
            False  # the renderer follows the physics once per 20 ms step, as replay_wbc.py does
        )
        self.rob = self.u.task.robot
        n = self.rob.num_body_dof
        cmd = lambda k: types.SimpleNamespace(
            motor_cmd=[types.SimpleNamespace(q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0) for _ in range(k)]
        )  # noqa: E731
        self.low, self.lh, self.rh = cmd(n), cmd(7), cmd(7)
        self.act = ActionCmd(
            "wbc_torque", low_cmd=self.low, use_sensor=False, left_hand_cmd=self.lh, right_hand_cmd=self.rh
        )
        self.frames, self.steps, self.obs = [], 0, None

    def state(self):
        m, d = self.u.mujoco.mjModel, self.u.mujoco.mjData
        return dict(qpos=d.qpos[: 7 + 43].copy(), qvel=d.qvel[: 6 + 43].copy())

    def reset(self, level, idx):
        (f,) = glob.glob(f"{xdir}/{self.task}/*/dr-level-{level}/meta/episodes.jsonl")
        ep = [json.loads(line) for line in open(f)][idx]
        conf = ep["environment_config"]
        self.obs, _ = self.env.reset(options={"state_dict": json.loads(conf) if isinstance(conf, str) else conf})
        m = self.u.mujoco.mjModel
        self.frames, self.steps = [], 0
        idx = lambda a: (np.asarray(a) - 1).tolist()  # noqa: E731
        return dict(self.state(), names=[m.joint(i).name for i in range(1, 44)], body=idx(self.rob.body_joint_index),
                    left=idx(self.rob.left_hand_index), right=idx(self.rob.right_hand_index), limit=self.limit,
                    torque_limit=np.asarray(self.rob.torque_limit), instruction=self.u.task.instruction)  # fmt: skip

    def step(self, q, kp, kd, counted, film):
        b, lft, rgt = (
            np.asarray(a) - 1 for a in (self.rob.body_joint_index, self.rob.left_hand_index, self.rob.right_hand_index)
        )
        for cmd, ids in ((self.low, b), (self.lh, lft), (self.rh, rgt)):
            for c, i in zip(cmd.motor_cmd, ids):
                c.q, c.kp, c.kd = float(q[i]), float(kp[i]), float(kd[i])
        if counted:
            self.obs, _, term, trunc, _ = self.env.step(self.act)
            self.steps += 1
        else:  # stabilising before the episode: the raw environment, as eval_decoupled_wbc.py does
            self.obs, _, term, trunc, _ = self.u.step(self.act)
            term = False
        if film and (counted and self.steps % 2 == 0 or not counted and len(self.frames) < 2):
            self.frames.append(np.asarray(self.obs["head_stereo_left"]).copy())
        return dict(
            self.state(), terminated=bool(term), truncated=bool(trunc), success=bool(self.u._success), steps=self.steps
        )

    def video(self, path):
        import cv2

        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
        for f in self.frames:
            w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        w.release()
        return len(self.frames)


ep = None
with Listener(sock, family="AF_UNIX") as lis:
    print("listening", sock, flush=True)
    while True:
        with lis.accept() as c:
            try:
                while True:
                    msg = c.recv()
                    k = msg[0]
                    if k == "reset":
                        if ep is None or ep.task != msg[1]:
                            if ep is not None:
                                ep.env.close()
                            ep = Episode(msg[1])
                        c.send(ep.reset(msg[2], msg[3]))
                    elif k == "step":
                        c.send(ep.step(*msg[1:]))
                    elif k == "eyes":
                        c.send(
                            dict(
                                left=np.asarray(ep.obs["head_stereo_left"]),
                                right=np.asarray(ep.obs["head_stereo_right"]),
                            )
                        )
                    elif k == "video":
                        c.send(ep.video(msg[1]))
                    elif k == "quit":
                        raise SystemExit
            except EOFError:
                pass
