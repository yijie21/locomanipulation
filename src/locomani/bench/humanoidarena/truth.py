"""The truth view for target=truth: what a task's goal is, read from Isaac Lab's state (TODO 49.5b).

A skill package asks `view.goal()` and gets plain numbers in the benchmark's world frame (the robot's own readings share it,
self_pose=truth):

    pp_box   {"kind": "pp_box", "box": (3,) the box's root (its bottom centre), "box_center": (3,), "yaw": rad, "half": m,
              "place_xy": (2,) a spot the benchmark's own check accepts, "place_bottom_z": m the box's bottom must be at (+-0.06),
              "place_rect": (4,) x_lo, x_hi, y_lo, y_hi of the surface it lies on,
              "table": (2, 3) the table's min and max corners (the box stands on it), "shelf": (2, 3) the shelf's}

It holds the Privileged (bench/channels.py) and nothing of the simulator beyond it; only the bed creates it. DoubleDesk and OpenDoor
have no truth goal yet (the bed runs their environment and judge).
"""

from __future__ import annotations

import numpy as np


class UnsupportedTask(ValueError):
    """The truth view has no goal for this task (the seed package covers P&PBox)."""


def yaw_of_quat(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class TruthView:
    def __init__(self, priv):
        self.priv = priv
        self.task = priv.episode()["task"]

    def goal(self) -> dict:
        if self.task == "pp_box":
            return self._pp_box()
        raise UnsupportedTask(f"no truth goal for {self.task}")

    def _pp_box(self) -> dict:
        f = self.priv.facts()
        objs = self.priv.objects()
        b = np.asarray(objs["box"], float)
        half = float(f["box_half"][2])
        spots = [
            a
            for a in f["accepted"]
            if 0.3 < a["bottom_z"] < 1.2
            and (a["rect"][1] - a["rect"][0]) >= 0.26
            and (a["rect"][3] - a["rect"][2]) >= 0.26
        ]
        if not spots:
            raise UnsupportedTask("the benchmark's check accepts no spot with room for the box")
        near = min(spots, key=lambda a: float(np.hypot(a["center_xy"][0] - b[0], a["center_xy"][1] - b[1])))
        return dict(kind="pp_box", box=b[:3], box_center=np.asarray(f["box_center"], float), yaw=yaw_of_quat(b[3:7]), half=half,
                    place_xy=np.asarray(near["center_xy"], float), place_bottom_z=float(near["bottom_z"]), place_rect=np.asarray(near["rect"], float),
                    table=None if f.get("table") is None else np.asarray(f["table"], float),
                    shelf=None if f.get("shelf") is None else np.asarray(f["shelf"], float))  # fmt: skip
