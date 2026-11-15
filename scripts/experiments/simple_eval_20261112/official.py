"""The ScaleBFM runtime's Sim with SIMPLE's own environment (official_server.py) as the physics: `step` sends ScaleBFM's joint
targets with its gains to SIMPLE's motors and writes the joint readings and base state that come back into this model, which
then serves as the robot's own kinematic model. The scene's objects in this model never move: nothing may be read from them.
`eyes` gives the head camera's colour image (Isaac Sim) and the depth worked out from its stereo pair (depth_server.py)."""

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R


class EpisodeOver(Exception):
    pass


class OffSim(R.Sim):
    def __init__(self, meta, backend, link, depth_link, film: bool = False):
        super().__init__(meta, backend=backend)
        self.link, self.depth_link = link, depth_link  # multiprocessing connections, shared by the run's episodes
        self.counted, self.over, self.film, self.first, self.steps = False, None, film, None, 0

    def begin(self, task: str, level: int, index: int) -> dict:
        """SIMPLE's reset of the episode; the motors' order and limits come back with the first readings."""
        self.link.send(("reset", task, level, index))
        st = self.link.recv()
        ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in st["names"]]
        assert min(ids) >= 0, "a SIMPLE joint is missing in the robot's model"
        self.oq, self.ov = self.m.jnt_qposadr[ids], self.m.jnt_dofadr[ids]
        slot = {int(a): k for k, a in enumerate(self.oq)}  # this model's qpos address -> SIMPLE's motor
        self.body_slot, self.hand_slot = (
            np.array([slot[int(a)] for a in self.qadr]),
            np.array([slot[int(a)] for a in self.gq]),
        )
        self.kp43, self.kd43 = np.zeros(43), np.zeros(43)
        self.kp43[self.body_slot], self.kd43[self.body_slot] = self.kp, self.kd
        self.kp43[self.hand_slot], self.kd43[self.hand_slot] = np.tile(R.DEX3_KP, 2), R.DEX3_KD
        self.counted, self.over, self.first = False, None, st
        self.limits = dict(theirs=st["torque_limit"][self.body_slot], ours=self.tlim)
        return st

    def pull(self, st: dict | None = None) -> None:
        st = st or self.first
        d = self.d
        d.qpos[:7], d.qpos[self.oq] = st["qpos"][:7], st["qpos"][7:]
        d.qvel[:6], d.qvel[self.ov] = st["qvel"][:6], st["qvel"][6:]
        mujoco.mj_forward(self.m, d)

    def reset(self, spawn_xy, spawn_yaw, crate_dy):
        super().reset(spawn_xy, spawn_yaw, crate_dy)
        self.pull()

    def step(self, tgt):
        if self.over is not None:
            raise EpisodeOver
        q = np.zeros(43)
        q[self.body_slot], q[self.hand_slot] = tgt, self.grip_target
        self.link.send(("step", q, self.kp43, self.kd43, self.counted, self.film))
        st = self.link.recv()
        self.pull(st)
        if st["terminated"] or st["truncated"]:
            self.over = dict(success=st["success"], steps=st["steps"], truncated=st["truncated"])
        self.steps = st["steps"]

    def eyes(self):
        self.link.send(("eyes",))
        e = self.link.recv()
        self.depth_link.send((e["left"], e["right"]))
        depth, sg, lb = self.depth_link.recv()
        self.last_eyes = dict(e, depth=lb, sg=sg, lb=lb)
        return e[
            "left"
        ], lb  # LingBot-Depth's output given the matched depth: raw matching scatters 2 cm at 0.85 m (a quarter pixel)

    def video(self, path: str) -> int:
        self.link.send(("video", path))
        return self.link.recv()
