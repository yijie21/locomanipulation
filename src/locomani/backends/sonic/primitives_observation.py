"""`Primitives`'s state-reading mixin, split out of primitives.py (refactor
step 4b family 4, pure extraction; decision 9's file-size cap). Every
statement here is unchanged from the original inline body -- only the
packaging moved.
"""

from __future__ import annotations

import numpy as np


class PrimitivesObservationMixin:
    """`obs`/`pelvis_xy`/`pelvis_yaw`/`object_pos`/`world_to_root`/`_rot`/
    `world_to_root_full`/`log`/`_facing` -- reading and transforming the
    privileged state, no actuation. Needs nothing from `primitives_tuning.py`."""

    def obs(self) -> dict:
        return self.state.wait()

    def pelvis_xy(self, s=None):
        s = s or self.obs()

        return np.array(s["pelvis"]["pos"][:2])

    def pelvis_yaw(self, s=None) -> float:
        s = s or self.obs()

        return float(s["pelvis"]["yaw"])

    def object_pos(self, key: str, s=None):
        s = s or self.obs()

        return np.array(s["objects"][key]["pos"])

    def world_to_root(self, p_world, s=None):
        """World point -> pelvis frame, yaw-only (the pelvis stays near upright)."""
        s = s or self.obs()
        root = np.array(s["pelvis"]["pos"])
        yaw = float(s["pelvis"]["yaw"])
        d = np.asarray(p_world, dtype=float) - root
        c, sn = np.cos(-yaw), np.sin(-yaw)

        return np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1], d[2]])

    @staticmethod
    def _rot(quat) -> np.ndarray:
        w, x, y, z = quat

        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def world_to_root_full(self, p_world, s=None):
        """World point -> pelvis frame using the pelvis's *full* orientation.

        The yaw-only version is fine for walking, where the pelvis stays near
        upright, but not for reaching: the robot pitches noticeably as it
        extends, and a yaw-only frame then disagrees with the frame the
        controller interprets the wrist target in. Feeding an error measured in
        one frame into a command expressed in another leaves a standing offset
        that no amount of integrating removes.
        """
        s = s or self.obs()
        d = np.asarray(p_world, dtype=float) - np.array(s["pelvis"]["pos"])

        return self._rot(s["pelvis"]["quat"]).T @ d

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}")

    def _facing(self, s) -> tuple:
        yaw = self.hold_yaw if self.hold_yaw is not None else self.pelvis_yaw(s)

        return (float(np.cos(yaw)), float(np.sin(yaw)), 0.0)
