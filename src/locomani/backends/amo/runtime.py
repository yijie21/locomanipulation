"""AMO runtime (TODO 28, user 2026-10-20: would AMO beat ScaleBFM?): the in-process sim + policy pair.

AMO (Li, Cheng, Huang et al., RSS 2025, arXiv 2505.03738) is an RL policy for the G1's legs and waist (15 joints) that
takes a command -- forward / sideways speed, a target heading, base height, torso yaw / pitch / roll -- and the arms'
shoulder and elbow angles as input. The arms and wrists are not driven by the policy at all: the caller sets their
joint targets and our PD holds them. The checkpoint is the one SIMPLE vendors (third_party/SIMPLE/third_party/AMO,
`amo_jit.pt` + the `adapter_jit.pt` MLP that turns height / torso commands into reference leg and waist angles).

`AmoPolicy` reproduces SIMPLE's `AMO_Policy` (src/simple/robots/policy/AMO_Policy.py, the dex3 29-joint branch):
the 93-number proprio frame, 10 frames of history in the observation, 25 more for the policy's own encoder, the
two-phase gait clock at 1.3 Hz that freezes at 0.25 / 0.25 when the speed command is below 0.1 m/s.
`AmoSim` is the ScaleBFM `Sim` with AMO's PD gains (SIMPLE g1_wholebody.py) and AMO's 2 ms physics step, ten of them
per 50 Hz policy tick. The torque limits stay the scene model's (mars: ankle 35, waist roll / pitch 50 N m).
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.paths import REPO
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

AMO_DIR = os.path.join(REPO, "third_party/SIMPLE/third_party/AMO")
DT = R.DT  # 50 Hz policy tick, as ScaleBFM
SIM_DT, DECIM = 0.002, 10
HEIGHT0 = 0.75  # the command's base height when its offset is 0
ARM_IN = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]  # fmt: skip
ARM_IDX = np.array([MUJOCO_JOINT_NAMES.index(n) for n in ARM_IN])
POL_IDX = np.r_[np.arange(15), ARM_IDX]  # the 23 joints the policy observes
ARMS = np.arange(15, 29)  # both arms and wrists, set by the caller
# SIMPLE g1_wholebody.py: legs, waist, left arm, right arm
KP = np.array([150, 150, 150, 300, 80, 20] * 2 + [400] * 3 + [80, 80, 40, 60, 40, 40, 40] * 2, float)
KD = np.array([2, 2, 2, 4, 2, 1] * 2 + [15] * 3 + [2, 2, 1, 1, 1, 1, 1] * 2, float)
# AMO's command ranges (README "I.D. range"); the base height is HEIGHT0 + offset
VX_MAX, VY_MAX = 0.5, 0.4
TORSO_PITCH = (-0.52, 1.57)
TORSO_ROLL = 0.7
TORSO_YAW = 1.57
STAND_V = 0.1  # below this |vx| and |vy| (and no turning flag) the gait clock stops: standing in place


@dataclass
class Command:
    vx: float = 0.0  # m/s, pelvis heading frame
    vy: float = 0.0
    yaw: float = 0.0  # rad, world heading the policy turns to while it steps
    height: float = HEIGHT0  # m, AMO's base-height command
    torso_yaw: float = 0.0
    torso_pitch: float = 0.0
    torso_roll: float = 0.0
    turning: bool = False  # step in place (to turn) although the speed is below STAND_V

    def clipped(self) -> "Command":
        return Command(
            float(np.clip(self.vx, -VX_MAX, VX_MAX)),
            float(np.clip(self.vy, -VY_MAX, VY_MAX)),
            self.yaw,
            self.height,
            float(np.clip(self.torso_yaw, -TORSO_YAW, TORSO_YAW)),
            float(np.clip(self.torso_pitch, *TORSO_PITCH)),
            float(np.clip(self.torso_roll, -TORSO_ROLL, TORSO_ROLL)),
            self.turning,
        )


def euler(q) -> np.ndarray:
    """(roll, pitch, yaw) of a wxyz quaternion, AMO's quatToEuler."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = 2 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, s) if abs(s) >= 1 else np.arcsin(s)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def sim_meta(model_xml: str) -> dict:
    """The joint table `Sim` wants: AMO's gains, the scene model's torque limits, ScaleBFM's standing pose (the reset
    pose only; AMO stands up from it)."""
    m = mujoco.MjModel.from_xml_path(model_xml)
    lim = []
    for n in MUJOCO_JOINT_NAMES:
        j = m.joint(n).id
        a = next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == j)
        lim.append(float(m.actuator_ctrlrange[a, 1]))
    default = json.load(open(R.META))["default_dof_pos"]
    return dict(joint_names=list(MUJOCO_JOINT_NAMES), stiffness=KP.tolist(), damping=KD.tolist(), torque_limit=lim,
                default_dof_pos=default)  # fmt: skip


class AmoSim(R.Sim):
    """`runtime.Sim` on AMO's PD and 2 ms step (10 per tick); the grippers, bodies and fall test are the same."""

    def __init__(self, backend: ScaleBfmBackendConfig, video=False, perturb: dict | None = None):
        xml = os.path.join(backend.scene_path(), "scene_locomani.xml")
        super().__init__(sim_meta(xml), video=video, backend=backend, perturb=perturb)
        self.low_dt, self.decim = SIM_DT, DECIM
        self.m.opt.timestep = SIM_DT
        if self.pert is not None:  # its delay queue counts physics steps
            from locomani.sim import perturb as perturbation

            self.pert = perturbation.Perturbation(perturb, len(MUJOCO_JOINT_NAMES), SIM_DT)


class AmoPolicy:
    """AMO on our 29 joints (MuJoCo order). `__call__(state, command, arm_targets)` -> the 29 PD targets."""

    N_PROPRIO = 3 + 2 + 2 + 23 * 3 + 2 + 15
    HIST, EXTRA = 10, 25

    def __init__(self, device: str = "cuda"):  # the traced net has CUDA constants baked in
        self.dev = device
        self.net = torch.jit.load(os.path.join(AMO_DIR, "amo_jit.pt"), map_location=device).eval()
        self.adapter = torch.jit.load(os.path.join(AMO_DIR, "adapter_jit.pt"), map_location=device).eval()
        st = torch.load(os.path.join(AMO_DIR, "adapter_norm_stats.pt"), weights_only=False)
        f = lambda k: torch.as_tensor(np.asarray(st[k]), dtype=torch.float32, device=device)  # noqa: E731
        self.in_mean, self.in_std, self.out_mean, self.out_std = (
            f("input_mean"),
            f("input_std"),
            f("output_mean"),
            f("output_std"),
        )
        self.default = np.zeros(23)  # AMO's default pose: every joint 0
        # the bodies backends.scalebfm.motor.Keypoints computes for the motor (FK of a reference pose)
        self.sel = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link",
                    "right_ankle_roll_link"]  # fmt: skip
        self.reset()

    def reset(self, st=None) -> None:
        self.prop = deque([np.zeros(self.N_PROPRIO)] * self.HIST, maxlen=self.HIST)
        self.extra = deque([np.zeros(self.N_PROPRIO)] * self.EXTRA, maxlen=self.EXTRA)
        self.last = np.zeros(23)
        self.gait = np.array([0.25, 0.25])
        self.standing = True
        self.adapter_out = np.zeros(15)

    def _adapter(self, cmd: Command, arm_q) -> np.ndarray:
        x = np.r_[cmd.height, cmd.torso_yaw, cmd.torso_pitch, cmd.torso_roll, arm_q]
        x = (torch.as_tensor(x, dtype=torch.float32, device=self.dev)[None] - self.in_mean) / (self.in_std + 1e-8)
        with torch.inference_mode():
            y = self.adapter(x) * self.out_std + self.out_mean
        return y[0].cpu().numpy()

    def __call__(self, st: dict, cmd: Command, arm_targets) -> np.ndarray:
        cmd = cmd.clipped()
        q, dq = st["dof_pos"][POL_IDX], st["dof_vel"][POL_IDX]
        rpy = euler(st["root_quat"])
        dyaw = 0.0 if self.standing else float(R.wrap(rpy[2] - cmd.yaw))
        self.adapter_out = self._adapter(cmd, q[15:])
        prop = np.concatenate([
            st["ang_vel"] * 0.25, rpy[:2], (np.sin(dyaw), np.cos(dyaw)),
            q - self.default, dq * 0.05, self.last, np.sin(self.gait * 2 * np.pi), self.adapter_out,
        ])  # fmt: skip
        demo = np.r_[q[15:], cmd.vx, cmd.vy, 0.0, cmd.torso_yaw, cmd.torso_pitch, cmd.torso_roll, [cmd.height] * 3]
        self.standing = abs(cmd.vx) < STAND_V and abs(cmd.vy) < STAND_V and not cmd.turning
        obs = np.concatenate([prop, demo, np.zeros(3), np.array(self.prop).ravel()])
        self.prop.append(prop)
        self.extra.append(prop)
        t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=self.dev)[None]  # noqa: E731
        with torch.inference_mode():
            raw = self.net(t(obs), t(np.array(self.extra).ravel()))[0].cpu().numpy()
        raw = np.clip(raw, -40.0, 40.0)
        self.last = np.r_[raw, (q - self.default)[15:] / 0.25]
        self.gait = np.remainder(self.gait + DT * 1.3, 1.0)
        near = np.abs(self.gait - 0.25) < 0.05
        if self.standing and near.any():
            self.gait = np.array([0.25, 0.25])
        if not self.standing and near.all():
            self.gait = np.array([0.25, 0.75])
        out = np.zeros(29)
        out[:15] = raw * 0.25 + self.default[:15]
        out[ARMS] = arm_targets
        return out
