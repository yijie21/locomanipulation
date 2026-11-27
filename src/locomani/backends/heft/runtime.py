"""HEFT runtime: LocoManiLab's whole-body motion tracker (g1_backend `checkpoints/heft/pmg`) on our MuJoCo sim.

`HeftPolicy` is g1_backend's generic `Tracking` policy reduced to what HEFT uses (`sim2real/rl_policy/base_policy.py`
at 25d262a): the four ONNX inputs from the vendored observation (obs.py), one onnxruntime call, and
`q_target = default + action * action_scale` (policy order scattered into MuJoCo order; their yaml's `action_clip` is
not applied by their generic policy either). Its reference is whole-body qpos frames -- pelvis position, pelvis wxyz,
29 joints in MuJoCo order -- at HEFT's 13 reference steps: frames 0..6 ahead and 1, 2, 4, 8, 12, 16 behind, 50 Hz.
Who builds them (and so whether the policy sees the frames LocoManiLab's RealtimeMotionBuffer would, 0.16 s late) is
the motor's business (motor.py).

`HeftSim` is `scalebfm.runtime.Sim` -- the same scene, robot, joint order, 5 ms x 4 step and hand PD -- with HEFT's
joint PD from policy.yaml (`joint_kp`, `joint_kd`), LocoManiLab's stance as the reset pose (HeftConfig.reset_pose), and
the scene's actuator ranges as torque limits (g1_backend's `joint_effort_limit` table holds the same numbers). The scene's own joint
dynamics (armature, damping, frictionloss as its XML has them) are kept, as ScaleBFM and SONIC keep them on the same
bed: policy.yaml's `joint_armature` / `joint_damping` / `joint_frictionloss` (HEFT's training values) are recorded in
`training_dynamics` and not applied. LocoManiLab's benchmark host does not apply them either (lml_host.py).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import yaml

from locomani import paths
from locomani.backends.heft.obs import HeftObs
from locomani.backends.scalebfm import runtime as R
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

DT = R.DT  # 50 Hz policy step = HEFT's motion frame step (their buffer's dt_s 0.02)
# LocoManiLab's stance (lml_selfmodel.DEFAULT_Q: robot_fk's default pose; MuJoCo order): the pose every LocoManiLab drive
# holds HEFT in before and between its reaches (lml_host.start_frames)
LML_STANCE = np.array([-0.2, 0, 0, 0.4, -0.2, 0, -0.2, 0, 0, 0.4, -0.2, 0, 0, 0, 0,
                       0, 0.4, 0, 1.2, 0, 0, 0, 0, -0.4, 0, 1.2, 0, 0, 0])  # fmt: skip


def match(patterns: dict, names, strict: bool = True) -> np.ndarray:
    """g1_backend's resolve_matching_names_values for one table: each name takes the value of the one regex key that
    fully matches it; more than one is an error, none is an error when `strict` (else 0)."""
    out = np.zeros(len(names))
    for i, n in enumerate(names):
        hits = [v for k, v in patterns.items() if re.fullmatch(k, n)]
        if len(hits) > 1:
            raise ValueError(f"multiple matches for {n!r} in {list(patterns)}")
        if not hits and strict:
            raise ValueError(f"no match for {n!r} in {list(patterns)}")
        out[i] = float(hits[0]) if hits else 0.0
    return out


@dataclass(frozen=True)
class HeftConfig:
    policy_dir: str = paths.HEFT_POLICY_DIR  # policy.yaml / .onnx / .json of g1_backend's checkpoints/heft/pmg
    # s the reference is sampled behind the motor's clock: 0, the motor's own frames from now on (as ScaleBFM's
    # keypoints are on this bed); 0.16, LocoManiLab's RealtimeMotionBuffer (6 frames + 0.04 s tolerance), where the
    # policy's frame 0 is the frame pushed 0.16 s ago and nothing ahead of the newest pushed frame is seen
    ref_delay_s: float = 0.0
    threads: int = 1  # onnxruntime CPU threads (g1_backend's SIM2REAL_ORT_NUM_THREADS)
    # the sim's reset pose, which the motor then holds as its stance: "lml", LML_STANCE; "heft", policy.yaml's
    # default_joint_pos. On SIMPLE's G1 + Dex3 HEFT's default puts both thumbs 30 mm inside the hip pitch links
    # (2026-11-25): the arms start wedged and stay there (a 0.5 rad shoulder ramp moved 0.05-0.22 of it); from LML_STANCE
    # they follow it as in LocoManiLab (0.68-0.77, LocoManiLab 0.66-0.76). ScaleBFM's and SONIC's reset poses touch nothing.
    reset_pose: str = "lml"

    @property
    def yaml_path(self) -> str:
        return os.path.join(self.policy_dir, "policy.yaml")


def load_config(cfg: HeftConfig) -> dict:
    with open(cfg.yaml_path) as f:
        return yaml.safe_load(f)


def obs_config(conf: dict) -> dict:
    """The observation's arguments: the four groups of policy.yaml carry the same core arguments (one core is built)."""
    groups = [next(iter(g.values())) for g in conf["observation"].values()]
    core = {k: v for k, v in groups[0].items() if k not in ("_target_", "component")}
    for g in groups[1:]:
        other = {k: v for k, v in g.items() if k not in ("_target_", "component")}
        assert other == core, "HEFT's observation groups disagree on the core arguments"
    assert [g["component"] for g in groups] == list(HeftObs.PARTS), [g["component"] for g in groups]
    assert all(g["_target_"] == "heft.heft_component_obs" for g in groups)
    return core


class HeftPolicy:
    """HEFT PMG from its ONNX file. `joint_names`: the sim's joint order, in which state and targets are exchanged."""

    # keypoint bodies Motor's FK computes (Keypoints(sim.m, policy.sel)): the motor's trace and commanded_jaw
    sel = ("pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link",
           "right_ankle_roll_link")  # fmt: skip

    def __init__(self, joint_names, cfg: HeftConfig | None = None):
        import onnxruntime as ort

        self.cfg = cfg or HeftConfig()
        self.conf = conf = load_config(self.cfg)
        assert list(conf["joint_names_simulation"]) == MUJOCO_JOINT_NAMES, "HEFT's simulation order is MuJoCo's"
        self.policy_joint_names = list(conf["policy_joint_names"])
        oc = obs_config(conf)
        self.future_steps = np.asarray(oc["future_steps"], int)
        self.obs = HeftObs(state_joint_names=MUJOCO_JOINT_NAMES, **oc)
        self.to_mj = np.array([list(joint_names).index(n) for n in MUJOCO_JOINT_NAMES])  # sim index of each MJ joint
        self.from_mj = np.array([MUJOCO_JOINT_NAMES.index(n) for n in joint_names])  # MuJoCo index of each sim joint
        self.ctrl = np.array([MUJOCO_JOINT_NAMES.index(n) for n in self.policy_joint_names])  # controlled_joint_indices
        self.default = match(conf["default_joint_pos"], MUJOCO_JOINT_NAMES, strict=False)  # MuJoCo order
        self.scale = match(conf["action_scale"], self.policy_joint_names)  # policy order
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = self.cfg.threads, 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(
            os.path.join(self.cfg.policy_dir, "policy.onnx"), so, providers=["CPUExecutionProvider"]
        )
        names = [i.name for i in self.sess.get_inputs()]
        assert names == list(HeftObs.PARTS), names
        self.out_action = [o.name for o in self.sess.get_outputs()].index("action")
        self.root_source = None  # leg odometry (reachwm leg_odo.OdoRoot) sets this; called once a tick
        self.reset_state()

    def reset_state(self) -> None:
        self.root_est = None
        self.action = np.zeros(29, np.float32)  # the previous raw action (policy order): their state_dict["action"]
        self.obs.reset()

    def reset(self, st: dict) -> None:
        self.reset_state()

    def __call__(self, st: dict, frames: np.ndarray) -> np.ndarray:
        """One 50 Hz step. `st`: Sim.state(); `frames` (13, 36): the reference qpos frame at each of `future_steps`.
        Returns the joint targets in the sim's order."""
        if self.root_source is not None:  # HEFT reads no pelvis position; the odometry still integrates for its users
            self.root_est = np.asarray(self.root_source(st, None), float)
        q = np.asarray(st["dof_pos"], float)[self.to_mj]
        dq = np.asarray(st["dof_vel"], float)[self.to_mj]
        self.obs.set_state(q, dq, st["root_quat"], st["ang_vel"])
        f = np.asarray(frames, float)
        self.obs.set_reference(f[:, 7:], f[:, :3], f[:, 3:7])
        self.obs.update({"action": self.action})
        out = self.sess.run(None, self.obs.inputs())
        self.action = np.asarray(out[self.out_action], np.float32).reshape(-1)
        q_target = self.default.copy()
        q_target[self.ctrl] += self.action * self.scale
        return q_target[self.from_mj]


def heft_meta(meta: dict, conf: dict, reset_pose: str = "lml") -> dict:
    """ScaleBFM's metadata (joint order kept) with HEFT's gains (policy.yaml) and the reset pose (HeftConfig.reset_pose)."""
    names = meta["joint_names"]
    out = dict(meta)
    out["stiffness"] = match(conf["joint_kp"], names).tolist()
    out["damping"] = match(conf["joint_kd"], names).tolist()
    if reset_pose == "heft":
        out["default_dof_pos"] = match(conf["default_joint_pos"], names, strict=False).tolist()
    else:
        assert reset_pose == "lml", reset_pose
        out["default_dof_pos"] = [float(LML_STANCE[MUJOCO_JOINT_NAMES.index(n)]) for n in names]
    return out


class HeftSim(R.Sim):
    """`scalebfm.runtime.Sim` with HEFT's joint PD and the reset pose; the scene's joint dynamics and actuator ranges."""

    def __init__(self, meta: dict, heft: HeftConfig | None = None, video=False, backend=None, perturb=None):
        self.heft = heft or HeftConfig()
        conf = load_config(self.heft)
        super().__init__(heft_meta(meta, conf, self.heft.reset_pose), video=video, backend=backend, perturb=perturb)
        # torque limits: the scene's actuator ranges (SIMPLE's G1: 88 / 139 / 50 / 25 / 5 N m, g1_backend's table); an
        # actuator without a range keeps the metadata's limit
        limited = self.m.actuator_ctrllimited[self.act].astype(bool)
        self.tlim = np.where(limited, self.m.actuator_ctrlrange[self.act, 1], self.tlim)
        self.training_dynamics = {k: conf[k] for k in ("joint_armature", "joint_damping", "joint_frictionloss")}


__all__ = ["DT", "LML_STANCE", "HeftConfig", "HeftPolicy", "HeftSim", "heft_meta", "load_config", "match", "obs_config"]
