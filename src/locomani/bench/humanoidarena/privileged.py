"""HumanoidArena's privileged state for the truth view: bench/channels.py `Privileged` on the environment server (TODO 49.5b).

Object poses and the task's own facts (the shelf's support surfaces and the target layer's height) come from the server's "truth"
reply. Everything is in Isaac Lab's world frame, which is also the frame of the robot's own readings (self_pose=truth). Contacts are
not read here: the server keeps them for the judge (judge.py).

Only the truth View may hold one; a skill package cannot import this module (guard rule R8).
"""

from __future__ import annotations

import numpy as np


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


class HumanoidArenaPrivileged:
    def __init__(self, link, task: str, instruction: str):
        self.link, self.task, self.instruction = link, task, instruction

    def _truth(self) -> dict:
        return self.link.ask("truth")

    def pelvis(self) -> tuple[np.ndarray, float]:
        r = np.asarray(self._truth()["root"], float)
        R = quat_to_mat(r[3:7])
        return r[:3].copy(), float(np.arctan2(R[1, 0], R[0, 0]))

    def body_pose(self, name: str) -> np.ndarray:
        """(4, 4) world pose of a scene object by its scene key (`box`, `shelf` is static and has none)."""
        o = np.asarray(self._truth()["objects"][name], float)
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(o[3:7])
        T[:3, 3] = o[:3]
        return T

    def body_vertices(self, name: str) -> np.ndarray:
        raise NotImplementedError(
            "HumanoidArena's server keeps no collision meshes; objects are boxes of known size (facts)"
        )

    def facts(self) -> dict:
        """The task's own: `box_center`, `box_half`, `surfaces` (x_lo, x_hi, y_lo, y_hi, top_z), `target_top_z`."""
        return self._truth()["facts"]

    def objects(self) -> dict:
        return self._truth()["objects"]

    def episode(self) -> dict:
        return dict(instruction=self.instruction, task=self.task)
