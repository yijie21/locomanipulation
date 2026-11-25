"""HEFT whole-body tracker (Liu et al., arXiv 2607.02332; LocoManiLab's `heft/pmg` policy) behind the Motor's reference.

TODO 51. A port of LocoManiLab's observation builder (external/g1_backend/sim2real/rl_policy/observations/heft.py,
`heft_policy_obs._build_obs_parts` and `update`) and action mapping (rl_policy/base_policy.py `setup_policy`: target =
default + action * scale; the manifest's joint_kp / joint_kd go with it). `HeftPolicy` has the surface the Motor needs of
`runtime.Policy` (`sel`, `fidx`, `F`, `mode`, `meta`, `reset`, `__call__`, `root_source`), so the Motor, the skills and the
bed run unchanged above it; only the tracker that turns the reference into joint targets and gains differs.

What HEFT reads is not what ScaleBFM reads. ScaleBFM takes keypoint targets (pelvis and wrists); HEFT takes the whole-body
reference: 29 joint angles plus the pelvis pose, at the 13 steps of the manifest (0..6, -1, -2, -4, -8, -12, -16 ticks of
20 ms). The Motor's reference is a `Pose` (pelvis position, quaternion, 29 joints) per instant; `Motor._futures` hands the
samples to `HeftPolicy` as `fut.poses`. No absolute pelvis position is read: the command is the reference's own displacement
in the reference frame, its height, and its attitude against the robot's IMU quaternion.

Weights are downloaded data (`checkpoints/heft/pmg/policy.onnx`, GammaRobotics/LocoManiLab, MIT), loaded with onnxruntime.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from locomani.paths import REPO
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

HEFT_DIR = os.path.join(REPO, "third_party/LocoManiLab/external/g1_backend/checkpoints/heft/pmg")
SCALEBFM_META = os.path.join(REPO, "assets/scalebfm/model_22200_tensorrt_metadata.json")
OBS_KEYS = ("context", "motion_command", "target_motion", "proprioception")
DOWN = np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32)


# -- LocoManiLab's math (sim2real/utils/math.py), wxyz quaternions, copied so the port needs none of its package --------------
def quat_rotate_inverse_numpy(q, v):
    shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    q_w, q_vec = q[:, 0], q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    c = q_vec * np.sum(q_vec * v, axis=1, keepdims=True) * 2.0
    return (a - b + c).reshape(shape)


def matrix_from_quat(quaternions):
    original_shape = quaternions.shape[:-1]
    quaternions = quaternions.reshape(-1, 4)
    r, i, j, k = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)
    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(original_shape + (3, 3))


def quat_mul(q1, q2):
    shape = q1.shape
    q1, q2 = q1.reshape(-1, 4), q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def quat_conjugate(q):
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[:, 0:1], -q[:, 1:]), axis=-1).reshape(shape)


def match_values(patterns: dict, names: list[str]) -> np.ndarray:
    """sim2real.utils.strings.resolve_matching_names_values: full regex match, one pattern per name, 0 where none matches."""
    out = np.zeros(len(names))
    for i, name in enumerate(names):
        hits = [k for k in patterns if re.fullmatch(k, name)]
        if len(hits) > 1:
            raise ValueError(f"Multiple matches for {name!r}: {hits}")
        if hits:
            out[i] = patterns[hits[0]]
    return out


class HeftManifest:
    """policy.yaml of the checkpoint: names, step lists, gains, scales, default pose. `joint_names` is the policy order."""

    def __init__(self, directory: str = HEFT_DIR):
        import yaml

        self.dir = directory
        y = yaml.safe_load(open(os.path.join(directory, "policy.yaml")))
        comp = y["observation"]["context"]["context"]
        self.future_steps = [int(s) for s in comp["future_steps"]]
        self.history_steps = [int(s) for s in comp["root_angvel_history_steps"]]
        for k in ("projected_gravity_history_steps", "joint_pos_history_steps", "joint_vel_history_steps"):
            if [int(s) for s in comp[k]] != self.history_steps:
                raise ValueError(f"{k} differs from root_angvel_history_steps")
        self.prev_action_steps = int(comp["prev_action_steps"])
        self.boot_indicator_max = int(comp["boot_indicator_max"])
        self.joint_names = list(y["policy_joint_names"])
        if list(comp["joint_names"]) != self.joint_names:
            raise ValueError("observation joint_names differ from policy_joint_names")
        self.sim_joint_names = list(y["joint_names_simulation"])
        n = self.joint_names
        self.default = match_values(y["default_joint_pos"], n)
        self.scale = match_values(y["action_scale"], n)
        self.kp = match_values(y["joint_kp"], n)
        self.kd = match_values(y["joint_kd"], n)
        self.action_clip = y.get("action_clip")  # not applied by the runtime for HEFT (the clip is ALMI's)


class HeftObs:
    """`heft_policy_obs` without the registry, env and motion buffer: the same four input groups from the same numbers.

    `update` takes the robot state (root quaternion wxyz, body-frame angular velocity, joint position and velocity, all in
    policy joint order), the reference at the manifest's future steps (joint angles in policy order, pelvis position and
    quaternion) and the previous raw action; it appends the histories first, as the runtime does, then builds the groups.
    """

    def __init__(self, man: HeftManifest):
        self.man = man
        self.future_steps = np.asarray(man.future_steps, dtype=np.int64)
        self.h = list(man.history_steps)
        n = len(man.joint_names)
        L = max(self.h) + 1
        self.hist = dict(
            angvel=np.zeros((L, 3), np.float32),
            gravity=np.zeros((L, 3), np.float32),
            pos=np.zeros((L, n), np.float32),
            vel=np.zeros((L, n), np.float32),
        )
        self.prev = np.zeros((man.prev_action_steps, n), np.float32)
        self.boot = man.boot_indicator_max
        self.parts: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        for v in self.hist.values():
            v[:] = 0.0
        self.prev[:] = 0.0
        self.boot = self.man.boot_indicator_max
        self.parts = {}

    @staticmethod
    def _push(history, value) -> None:
        history[:] = np.roll(history, 1, axis=0)
        history[0] = value

    def update(
        self, root_quat, ang_vel_b, joint_pos, joint_vel, ref_joint_pos, ref_root_pos, ref_root_quat, prev_action
    ):
        self.boot = max(self.boot - 1, 0)
        root_quat = np.asarray(root_quat, dtype=np.float32)
        gravity = quat_rotate_inverse_numpy(root_quat.reshape(1, 4), DOWN)[0]
        gravity = (gravity / (np.linalg.norm(gravity) + 1.0e-8)).astype(np.float32)
        cur = np.asarray(joint_pos, dtype=np.float32)
        self._push(self.hist["angvel"], np.asarray(ang_vel_b, dtype=np.float32))
        self._push(self.hist["gravity"], gravity)
        self._push(self.hist["pos"], cur)
        self._push(self.hist["vel"], np.asarray(joint_vel, dtype=np.float32))
        self.prev[:] = np.roll(self.prev, 1, axis=0)
        self.prev[0, :] = np.asarray(prev_action, dtype=np.float32).reshape(-1)

        ref_joint_pos = np.asarray(ref_joint_pos, dtype=np.float32)
        ref_root_pos_w = np.asarray(ref_root_pos, dtype=np.float32)
        ref_root_quat_w = np.asarray(ref_root_quat, dtype=np.float32)
        pos_diff_w = ref_root_pos_w[1:] - ref_root_pos_w[0:1]
        base_ref_quat = np.broadcast_to(ref_root_quat_w[0:1], (pos_diff_w.shape[0], 4))
        pos_diff_b = quat_rotate_inverse_numpy(base_ref_quat, pos_diff_w)
        robot_q = np.broadcast_to(root_quat.reshape(1, 4), ref_root_quat_w.shape)
        rel_rot = matrix_from_quat(quat_mul(quat_conjugate(robot_q), ref_root_quat_w))
        rot6d = rel_rot[:, :, :2].transpose(0, 2, 1).reshape(-1).astype(np.float32)
        command = np.concatenate([pos_diff_b.reshape(-1), rot6d], axis=0)
        target_joint = np.concatenate(
            [ref_joint_pos.reshape(-1), (ref_joint_pos - cur.reshape(1, -1)).reshape(-1)], axis=0
        ).astype(np.float32)
        down = np.broadcast_to(DOWN, (ref_root_quat_w.shape[0], 3))
        target_gravity = quat_rotate_inverse_numpy(ref_root_quat_w, down).reshape(-1)
        h = self.h
        self.parts = {
            "context": np.asarray([self.boot / self.man.boot_indicator_max], dtype=np.float32),
            "motion_command": command.astype(np.float32),
            "target_motion": np.concatenate(
                [target_joint, ref_root_pos_w[:, 2].astype(np.float32), target_gravity.astype(np.float32)], axis=0
            ).astype(np.float32),
            "proprioception": np.concatenate(
                [
                    self.hist["angvel"][h].reshape(-1),
                    self.hist["gravity"][h].reshape(-1),
                    self.hist["pos"][h].reshape(-1),
                    self.hist["vel"][h].reshape(-1),
                    self.prev.reshape(-1),
                ],
                axis=0,
            ).astype(np.float32),
        }
        return self.parts

    def inputs(self) -> dict[str, np.ndarray]:
        return {k: self.parts[k][None, :] for k in OBS_KEYS}


def target_of(man: HeftManifest, action) -> np.ndarray:
    """Joint position targets (policy order) of one raw action: default + action * scale (BasePolicy.setup_policy)."""
    return man.default + np.asarray(action, dtype=np.float32) * man.scale


class HeftPolicy:
    """Drop-in for `runtime.Policy` under the Motor: HEFT reads the reference's joints, not its keypoints."""

    wants_poses = True
    mode = None  # set in __init__ (a torch tensor the Motor replaces; HEFT has no modes)

    def __init__(self, device="cpu", directory: str = HEFT_DIR, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        import torch

        self.man = HeftManifest(directory)
        self.obs = HeftObs(self.man)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1  # one sim per slot, 15 cores shared by three
        self.session = ort.InferenceSession(os.path.join(directory, "policy.onnx"), so, providers=list(providers))
        meta = json.load(open(SCALEBFM_META))
        if meta["joint_names"] != self.man.joint_names:
            raise ValueError("HEFT policy joint order differs from the scalebfm meta's")
        # the surface runtime.Sim reads: joint names (policy order), gains, default pose; the torque limits stay the robot's
        self.meta = dict(
            meta,
            stiffness=self.man.kp.tolist(),
            damping=self.man.kd.tolist(),
            default_dof_pos=self.man.default.tolist(),
            action_scale=self.man.scale.tolist(),
        )
        self.sel = meta["selected_body_names"]  # the keypoints the Motor still computes (wrists for the hands)
        self.fidx = list(self.man.future_steps)
        self.F = len(self.fidx)
        self.mode = torch.zeros(1, dtype=torch.long)
        self.root_source = (
            None  # HEFT reads no absolute pelvis position, so the controller_root switch has nothing to set
        )
        self.mj_to_policy = np.array([MUJOCO_JOINT_NAMES.index(n) for n in self.man.joint_names])
        self.prev_action = np.zeros(len(self.man.joint_names), np.float32)
        self.last_action = self.prev_action
        self.joint_correction = (
            None  # optional (fut, poses) -> poses: arm joints that realise a wrist the Motor shifted
        )

    def attach(self, motor) -> None:
        """Called by the Motor that drives this policy: wrist shifts become arm joints (ArmCorrection)."""
        self.joint_correction = ArmCorrection(motor)

    def reset(self, st) -> None:
        if self.joint_correction is not None:
            self.joint_correction.last.clear()
        self.obs.reset()
        self.prev_action = np.zeros_like(self.prev_action)

    def _reference(self, poses):
        pos = np.array([p[0] for p in poses])
        quat = np.array([p[1] for p in poses])
        joints = np.array([np.asarray(p[2], dtype=float)[self.mj_to_policy] for p in poses])
        return joints, pos, quat

    def __call__(self, st, fut):
        """`fut` is the Motor's futures dict with `.poses` (one (pelvis pos, quat, 29 joints in MuJoCo order) per step of
        `fidx`); with none (the Motor's warm-up calls) the joint targets are the measured joints."""
        poses = getattr(fut, "poses", None)
        if not poses:
            return np.asarray(st["dof_pos"], dtype=float).copy()
        if self.joint_correction is not None:
            poses = self.joint_correction(fut, poses)
        ref_joint, ref_pos, ref_quat = self._reference(poses)
        self.obs.update(
            st["root_quat"], st["ang_vel"], st["dof_pos"], st["dof_vel"], ref_joint, ref_pos, ref_quat, self.prev_action
        )
        out = self.session.run(["action"], self.obs.inputs())[0][0].astype(np.float32)
        self.prev_action = out
        self.last_action = out
        return target_of(self.man, out).astype(float)


class ArmCorrection:
    """Wrist shifts as arm joints. HEFT has no keypoint input, so what the Motor does to the wrist keypoints after the
    reference (the pad servo's integral lead, `Motor.hand` overrides) is turned into arm joint angles: the arm IK, seeded at
    the last solution, takes the shifted wrist of the first future step to the pelvis frame, and the arm joints' change from
    the reference's own solution is added to every future step (the lead moves slowly over 0.32 s). One side is corrected
    only while its wrist differs from the reference's by more than `eps`."""

    def __init__(self, motor, eps_m: float = 2e-4, iters: int = 15):
        self.motor, self.eps, self.iters = motor, eps_m, iters
        self.last: dict = {}
        self.active = {"left": False, "right": False}

    def __call__(self, fut, poses):
        from scipy.spatial.transform import Rotation

        from locomani.primitives.g1_joints import ARM_MJ

        base = getattr(fut, "base_wrists", None)
        ik = self.motor.ik
        if base is None or ik is None:
            return poses
        poses = [(p, q, np.array(j, dtype=float)) for p, q, j in poses]
        pos0, quat0, joints0 = poses[0]
        Rp = Rotation.from_quat(quat0, scalar_first=True).as_matrix()
        for side in ("left", "right"):
            w = f"{side}_wrist_yaw_link"
            (bp, bq), (fp, fq) = base[w], fut[w]
            moved = (
                float(np.linalg.norm(fp[0] - bp[0])) > self.eps
                or float(
                    (
                        Rotation.from_quat(fq[0], scalar_first=True)
                        * Rotation.from_quat(bq[0], scalar_first=True).inv()
                    ).magnitude()
                )
                > 1e-3
            )
            if not moved:
                self.last.pop(side, None)
                continue
            target_p = Rp.T @ (fp[0] - pos0)
            target_R = Rp.T @ Rotation.from_quat(fq[0], scalar_first=True).as_matrix()
            q_ref = joints0.copy()
            q0 = q_ref.copy()  # this tick's waist and legs under the arm seeded at the last solution
            if side in self.last:
                q0[list(ARM_MJ[side])] = self.last[side][list(ARM_MJ[side])]
            sol, _, _ = ik.solve(side, target_p, target_R, q0, iters=self.iters)
            self.last[side] = sol.copy()
            delta = np.zeros(29)
            idx = list(ARM_MJ[side])
            delta[idx] = sol[idx] - joints0[idx]
            poses = [(p, q, j + delta) for p, q, j in poses]
        return poses
