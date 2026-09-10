"""Shared collection/inference observation adapter: RGB and robot state only."""

import sys
from pathlib import Path

import mujoco
import numpy as np

from locomani.backends.scalebfm.state import ImageHistory, state_features
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

TASK = "pick up the water bottle"
CAMERAS = {"ego_view": "head_camera", "right_wrist": "right_wrist"}


class RobotCameraObserver:
    def __init__(self, sim, width=320, height=240):
        self.sim = sim
        self.renderer = mujoco.Renderer(sim.m, height=height, width=width)
        self.cameras = {key: sim.m.camera(name).id for key, name in CAMERAS.items()}
        self.q29 = np.array([sim.m.jnt_qposadr[sim.m.joint(name).id] for name in MUJOCO_JOINT_NAMES])
        self.history = ImageHistory()

    def close(self):
        self.renderer.close()

    def capture(self, t, previous_action, blackout=False):
        sim = self.sim
        images = {}
        for key, cid in self.cameras.items():
            self.renderer.update_scene(sim.d, camera=cid)
            frame = self.renderer.render().copy()
            images[key] = np.zeros_like(frame) if blackout else frame
        velocities = {}
        for name in ("pelvis", "right_wrist_yaw_link"):
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(sim.m, sim.d, mujoco.mjtObj.mjOBJ_BODY, sim.bid[name], velocity, 0)
            velocities[name] = velocity
        wp, wq = sim.body("right_wrist_yaw_link")
        state = state_features(
            sim.d.qpos[self.q29],
            sim.d.qpos[sim.gq],
            sim.d.qpos[:3],
            sim.d.qpos[3:7],
            wp,
            wq,
            velocities["pelvis"][3:],
            velocities["pelvis"][:3],
            velocities["right_wrist_yaw_link"][3:],
            previous_action,
        )
        history = self.history.push(t, images)
        observation = {f"state.{key}": value[None] for key, value in state.items()}
        observation.update({f"video.{key}": value for key, value in history.items()})
        observation["annotation.human.task_description"] = TASK
        return observation, state, images
