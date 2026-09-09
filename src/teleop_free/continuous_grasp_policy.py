"""Opt-in overlap of deceleration, arm preparation and measured capture."""

import time
from copy import deepcopy

import numpy as np
from coupled_grasp_policy import CoupledGraspPolicy
from primitives import Primitives
from ready_grasp_policy import OFFSET, PRE_GAP


class ContinuousGraspPolicy(CoupledGraspPolicy):
    def walk_tick(self, state):
        distance = np.linalg.norm(np.asarray(self.task.grip(state))[:2] - state["pelvis"]["pos"][:2])
        if not self.retracted and distance < 0.55:
            return self.prepare_arm(state)
        return super().walk_tick(state)

    def prepare_arm(self, state):
        palm = np.asarray(state["palms"][self.side]["pos"])
        if not hasattr(self, "_arrival_ref"):
            self._arrival_ref = palm.copy()
            self._arrival_rot = Primitives._rot(state["palms"][self.side]["quat"])
            self._arrival_z = float(palm[2])
        target = self.task.grip(state) - self._arrival_rot @ OFFSET - PRE_GAP * self._arrival_rot[:, 0]
        target[2] = max(target[2] + 0.12, self._arrival_z - 0.06)
        delta = target - self._arrival_ref
        ref = self._arrival_ref + delta * min(1.0, 0.0015 / max(np.linalg.norm(delta), 1e-9))
        ref = palm + (ref - palm) * min(1.0, 0.025 / max(np.linalg.norm(ref - palm), 1e-9))
        seed = self.arm.q.copy()
        seed[:12] = state["joints"][:12]
        q, pe, re = self.arm.ik.solve(
            self.side,
            self.p.world_to_root_full(ref, state),
            Primitives._rot(state["pelvis"]["quat"]).T @ self._arrival_rot,
            q_mj29=seed,
            use_waist=False,
            iters=30,
        )
        q = seed + np.clip(q - seed, -0.02, 0.02)
        stats = self.execution.metrics.setdefault("arrival_overlap", dict(command_ticks=0, blocked_ticks=0))
        for u in (0.0, 1.0):
            trial = deepcopy(state)
            trial["pelvis"]["pos"] = (
                np.asarray(state["pelvis"]["pos"]) + u * 0.1 * np.asarray(state["pelvis"]["vel"][:3])
            ).tolist()
            if pe > 0.02 or re > 0.20 or self.reference_check(trial, q, "prepare"):
                stats["blocked_ticks"] += 1
                # Keep the previous checked reference; do not force the hand
                # through an obstruction or reset it to the ready pose.
                return self._push(state, self.arm.q)
        self._arrival_ref = ref
        stats["command_ticks"] += 1
        return self._push(state, q)

    def brake(self):
        self._prepared = None
        self._prepared_once = False
        self.execution.enter("recovery_brake")
        end = time.monotonic() + 3.0
        stable = None
        while time.monotonic() < end and not self.expired():
            state = self.p.obs()
            if state["fallen"] or not self.prepare_arm(state):
                return False
            self.hold()
            now = time.monotonic()
            speed = np.linalg.norm(state["pelvis"]["vel"][:2])
            stable = (stable if stable is not None else now) if speed < 0.06 else None
            if stable is not None and now - stable >= 0.25:
                return True
            time.sleep(0.02)
        return False
