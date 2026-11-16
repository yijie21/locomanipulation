"""GR00T-WholeBodyControl runtime (TODO 34, user 2026-10-21: the top of rhoyn's walk-under-disturbance board, "会比
scalebfm 更稳定更流畅吗"): the in-process sim + policy pair.

NVIDIA's gear WBC (third_party/GR00T-WholeBodyControl/decoupled_wbc, `G1GearWbcPolicy`) is an RL policy for the G1's
legs and waist (15 joints): two ONNX nets, Balance below a 0.05 command norm and Walk above it, each taking six frames
of an 86-number observation -- the velocity command (vx, vy, yaw rate, scaled 2 / 2 / 0.5), the base-height command,
the torso roll / pitch / yaw command, the base angular velocity x 0.5, gravity in the base frame, all 29 joints'
positions minus the default pose and velocities x 0.05, and its last 15 actions. The action x 0.25 plus the default
pose is the 15 joints' PD target. The arms are not the policy's: the caller sets their targets (our arm IK), as
GR00T's own decoupled controller does. rhoyn's board ranks it first at a 0.66 m height and 0.12 rad pitch command
(`gr00t_wbc_h066_p012`); its default is 0.74 m upright.

`Gr00tPolicy(state, command, arm_targets)` -> the 29 PD targets, taking AMO's `Command` (backends.amo.runtime) so the
AMO motor adapter drives it: the command's target heading becomes a yaw rate here. `Gr00tSim` is the ScaleBFM `Sim`
with the gear WBC's gains (the sim2mujoco config's legs and waist, the deployed MOTOR_KP / KD arms) on the same 5 ms
step, four per 50 Hz tick.
"""

from __future__ import annotations

import json
import os
from collections import deque

import mujoco
import numpy as np

from locomani.backends.amo.runtime import Command
from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.paths import REPO
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

G1_DIR = os.path.join(REPO, "third_party/GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco/resources/robots/g1")
BALANCE = os.path.join(G1_DIR, "policy/GR00T-WholeBodyControl-Balance.onnx")
WALK = os.path.join(G1_DIR, "policy/GR00T-WholeBodyControl-Walk.onnx")
DT = R.DT  # 50 Hz policy tick
ARMS = np.arange(15, 29)
# g1_gear_wbc.yaml (sim2mujoco) for the legs and waist; g1_29dof_gear_wbc.yaml MOTOR_KP / KD for the arms
KP = np.array([150, 150, 150, 200, 40, 40] * 2 + [250] * 3 + [100, 100, 40, 40, 20, 20, 20] * 2, float)
KD = np.array([2, 2, 2, 4, 2, 2] * 2 + [5] * 3 + [5, 5, 2, 2, 2, 2, 2] * 2, float)
DEFAULT = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0] * 3)
CMD_SCALE = np.array([2.0, 2.0, 0.5])
ANG_VEL_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.5, 0.05, 0.25
HIST, SINGLE = 6, 86
HEIGHT0 = 0.74  # the default base-height command
STAND_NORM = 0.05  # below this command norm the Balance net runs (G1GearWbcPolicy.get_action)
VX_MAX, VY_MAX, YAW_RATE_MAX = 0.8, 0.5, 1.0  # the teleop's reach (steps of 0.2); clipped here
YAW_KP = 1.5  # 1/s: heading error -> yaw-rate command


def gravity_in_base(q) -> np.ndarray:
    """(0, 0, -1) in the frame of the wxyz quaternion q (gear_wbc_utils.get_gravity_orientation)."""
    w, x, y, z = (float(v) for v in q)
    u = np.array([x, y, z])
    g = np.array([0.0, 0.0, -1.0])
    return g * (2 * w * w - 1) - np.cross(u, g) * 2 * w + u * np.dot(u, g) * 2


def in_order(meta: dict, order: str) -> dict:
    """`meta` (MuJoCo joint order) with its per-joint lists in `order`: "mujoco" as is, "scalebfm" in ScaleBFM's
    metadata order (what scalebfm.runtime.Sim and the self-model features use), so every controller's sim exchanges
    state and targets in the same columns."""
    if order == "mujoco":
        return meta
    assert order == "scalebfm", order
    names = json.load(open(R.META))["joint_names"]
    idx = [meta["joint_names"].index(n) for n in names]
    out = dict(meta, joint_names=list(names))
    for k in ("stiffness", "damping", "torque_limit", "default_dof_pos"):
        out[k] = [meta[k][i] for i in idx]
    return out


def sim_meta(model_xml: str) -> dict:
    """The joint table `Sim` wants: the gear WBC's gains, the scene model's torque limits, ScaleBFM's standing pose."""
    m = mujoco.MjModel.from_xml_path(model_xml)
    lim = []
    for n in MUJOCO_JOINT_NAMES:
        j = m.joint(n).id
        a = next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == j)
        lim.append(float(m.actuator_ctrlrange[a, 1]))
    default = json.load(open(R.META))["default_dof_pos"]
    return dict(joint_names=list(MUJOCO_JOINT_NAMES), stiffness=KP.tolist(), damping=KD.tolist(), torque_limit=lim,
                default_dof_pos=default)  # fmt: skip


class Gr00tSim(R.Sim):
    """`runtime.Sim` on the gear WBC's PD (same 5 ms step, four per tick); grippers, bodies and the fall test as is."""

    def __init__(self, backend: ScaleBfmBackendConfig, video=False, perturb: dict | None = None, order: str = "mujoco"):
        """`order`: as AmoSim's."""
        from locomani.backends.amo.runtime import in_order

        xml = os.path.join(backend.scene_path(), "scene_locomani.xml")
        meta = in_order(sim_meta(xml), order)
        super().__init__(meta, video=video, backend=backend, perturb=perturb)
        self.joint_names = list(meta["joint_names"])


class Gr00tPolicy:
    """The gear WBC on our 29 joints (MuJoCo order). `__call__(state, command, arm_targets)` -> the 29 PD targets.
    `last_net` says which net ran ("balance" / "walk")."""

    def __init__(self, device: str = "cpu"):  # ONNX on the CPU: two small MLPs
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.nets = {k: ort.InferenceSession(p, opts, providers=["CPUExecutionProvider"])
                     for k, p in (("balance", BALANCE), ("walk", WALK))}  # fmt: skip
        self.inputs = {k: s.get_inputs()[0].name for k, s in self.nets.items()}
        self.dev = device
        # the bodies backends.scalebfm.motor.Keypoints computes for the motor (FK of a reference pose)
        self.sel = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link",
                    "right_ankle_roll_link"]  # fmt: skip
        self.reset()

    def reset(self, st=None) -> None:
        self.hist = deque([np.zeros(SINGLE, np.float32)] * HIST, maxlen=HIST)
        self.action = np.zeros(15, np.float32)
        self.last_net = "balance"
        self.last_cmd = np.zeros(3)

    def velocity(self, st: dict, cmd: Command) -> np.ndarray:
        """(vx, vy, yaw rate): the command's speeds, and its target heading as a rate while stepping (not AMO's
        `Command.clipped`: those are AMO's ranges)."""
        rate = 0.0
        if cmd.turning or abs(cmd.vx) > 1e-6 or abs(cmd.vy) > 1e-6:
            yaw = R.yaw_of(st["root_quat"])
            rate = float(np.clip(YAW_KP * R.wrap(cmd.yaw - yaw), -YAW_RATE_MAX, YAW_RATE_MAX))
        return np.array([np.clip(cmd.vx, -VX_MAX, VX_MAX), np.clip(cmd.vy, -VY_MAX, VY_MAX), rate])

    def __call__(self, st: dict, cmd: Command, arm_targets) -> np.ndarray:
        v = self.velocity(st, cmd)
        obs = np.zeros(SINGLE, np.float32)
        obs[0:3] = v * CMD_SCALE
        obs[3] = cmd.height
        obs[4:7] = (cmd.torso_roll, cmd.torso_pitch, cmd.torso_yaw)
        obs[7:10] = st["ang_vel"] * ANG_VEL_SCALE
        obs[10:13] = gravity_in_base(st["root_quat"])
        q = st["dof_pos"] - np.r_[DEFAULT, np.zeros(14)]
        obs[13:42] = q
        obs[42:71] = st["dof_vel"] * DOF_VEL_SCALE
        obs[71:86] = self.action
        self.hist.append(obs)
        x = np.concatenate(self.hist)[None].astype(np.float32)
        self.last_net = "balance" if float(np.linalg.norm(v)) < STAND_NORM else "walk"
        self.action = self.nets[self.last_net].run(None, {self.inputs[self.last_net]: x})[0][0].astype(np.float32)
        self.last_cmd = v
        out = np.zeros(29)
        out[:15] = self.action * ACTION_SCALE + DEFAULT
        out[ARMS] = arm_targets
        return out
