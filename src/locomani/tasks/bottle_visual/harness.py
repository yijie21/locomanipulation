"""CPU-only, restartable paired experiments for the visual expert Goal.

Does not render, record a dataset, open ports or initialize CUDA. Each case stores
the exact config/source hashes and applied commands; completed cases are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.reference import MotionLimits, PoseReference, PoseTransition
from locomani.tasks.bottle_visual import geometry
from locomani.tasks.bottle_visual.pipeline.provenance import line_sources


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def pose_angles(sim):
    values = []
    for name in ("pelvis", "torso_link"):
        r = R.mat(sim.body(name)[1])
        values.extend(np.degrees([np.arctan2(r[2, 1], r[2, 2]), np.arcsin(np.clip(-r[2, 0], -1, 1))]))
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        values.append(float(np.degrees(sim.d.qpos[sim.m.jnt_qposadr[sim.m.joint(name).id]])))
    return values


class AppliedPolicy:
    def __init__(self, policy, sim, limits=None, transition=False):
        self.policy, self.sim, self.limits = policy, sim, limits
        self.reference = None
        self.rows = []
        self.phase = "settle"
        self.use_transition = transition
        self.previous_command = None
        self.previous_phase = None
        self.transition = None
        self.transition_tick = 0

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def reset(self, state):
        self.policy.reset(state)
        self.reference = None
        self.rows = []
        self.previous_command = None
        self.previous_phase = None
        self.transition = None

    def __call__(self, state, targets):
        wrist = "right_wrist_yaw_link"
        if self.limits is not None:
            if self.reference is None:
                self.reference = PoseReference(*self.sim.body(wrist), self.limits)
            targets = dict(targets)
            targets[wrist] = self.reference.futures(*targets[wrist], R.DT)
        elif self.use_transition:
            raw_p, raw_q = targets[wrist]
            if self.phase != self.previous_phase:
                previous = self.previous_command or self.sim.body(wrist)
                self.transition = PoseTransition(previous, (raw_p[0], raw_q[0]), 2.5 if self.phase == "reach" else 0.5)
                self.transition_tick = len(self.rows)
            elapsed = (len(self.rows) - self.transition_tick) * R.DT
            pairs = [self.transition.sample(p, q, elapsed + i * R.DT) for i, (p, q) in enumerate(zip(raw_p, raw_q))]
            targets = dict(targets)
            targets[wrist] = np.stack([v[0] for v in pairs]), np.stack([v[1] for v in pairs])
            self.previous_command = targets[wrist][0][0].copy(), targets[wrist][1][0].copy()
            self.previous_phase = self.phase
        self.rows.append(
            dict(
                t=len(self.rows) * R.DT,
                phase=self.phase,
                angles=pose_angles(self.sim),
                wrist_position=targets[wrist][0][0].tolist(),
                wrist_quaternion=targets[wrist][1][0].tolist(),
                pelvis_position=state["root_pos"].tolist(),
            )
        )
        return self.policy(state, targets)

    def tick(self, sim, state, command, t, phase, cmd_next=None):
        # run_round recorder hook supplies phase only. Applied targets are captured
        # at the policy boundary AFTER the reference adapter, not from raw ref(t).
        self.phase = phase


def summarize(rows):
    angles = np.array([r["angles"] for r in rows])
    pos = np.array([r["wrist_position"] for r in rows])
    rot = Rotation.from_quat(np.array([r["wrist_quaternion"] for r in rows]), scalar_first=True)
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dr = np.degrees((rot[1:] * rot[:-1].inv()).magnitude())
    phases = {}
    for phase in dict.fromkeys(r["phase"] for r in rows):
        mask = np.array([r["phase"] == phase for r in rows])
        phases[phase] = np.round(abs(angles[mask]).max(0), 4).tolist()
    return dict(
        columns=["pelvis_roll", "pelvis_pitch", "torso_roll", "torso_pitch", "waist_yaw", "waist_roll", "waist_pitch"],
        phase_max_abs_deg=phases,
        torso_roll_max_deg=float(abs(angles[:, 2]).max()),
        torso_roll_above15_s=float((abs(angles[:, 2]) > 15).sum() * R.DT),
        waist_yaw_max_deg=float(abs(angles[:, 4]).max()),
        wrist_step_max_cm=float(dp.max(initial=0) * 100),
        wrist_step_max_deg=float(dr.max(initial=0)),
    )
