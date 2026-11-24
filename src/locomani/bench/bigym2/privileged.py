"""BiGym 2.0's privileged state for the truth view: bench/channels.py `Privileged` on the environment server (TODO 49.5).

Object poses, articulated joint angles and the task's own facts (the target sphere, the drawer's state) come from the server's
"truth" reply, collision vertices from "verts". Everything is in BiGym's world frame, which is also the frame of the robot's own
readings (self_pose=truth). Contacts are not read here: the server keeps them for the judge (judge.py).

Only the truth View may hold one; a skill package cannot import this module (guard rule R8).
"""

from __future__ import annotations

import numpy as np


class BiGymPrivileged:
    def __init__(self, link, task: str, instruction: str, names: list[str], movable: list[str]):
        self.link, self.task, self.instruction = link, task, instruction
        self.names, self.movable = names, movable

    def _truth(self) -> dict:
        return self.link.ask("truth")

    def pelvis(self) -> tuple[np.ndarray, float]:
        T = self._truth()["pelvis"]
        R = T[:3, :3]
        return T[:3, 3].copy(), float(np.arctan2(R[1, 0], R[0, 0]))

    def body_pose(self, name: str) -> np.ndarray:
        return np.asarray(self._truth()["bodies"][name], float)

    def body_vertices(self, name: str) -> np.ndarray:
        return np.asarray(self.link.ask("verts", name)["v"], float)

    def joints(self) -> dict:
        """Every scene joint: {name: {q, range, body}} (drawers, doors, trays; free bodies have none listed)."""
        return self._truth()["joints"]

    def facts(self) -> dict:
        """The task's own: `targets` (the reach spheres: position and the hands that count), `drawers` (normalised 0..1)."""
        return self._truth()["facts"]

    def episode(self) -> dict:
        return dict(instruction=self.instruction, task=self.task, movable=list(self.movable))
