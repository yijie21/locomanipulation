"""The truth view for target=truth: what a task's goal is, read from BiGym's state (TODO 49.5).

A skill package asks `view.goal()` and gets plain numbers in BiGym's world frame (the robot's own readings share it,
self_pose=truth):

    reach    {"kind": "reach", "side", "point"}                      the sphere the pinch centre has to enter, and the hand that counts
    drawer   {"kind": "drawer", "handle": (3,), "bar_axis": (3,), "pull": (3,), "travel": m, "open_frac": 0..1, "name"}
             the handle bar's centre, the bar's long axis, the direction the drawer slides out, its travel and how open it is now

It holds the Privileged (bench/channels.py) and nothing of the simulator beyond it; only the bed creates it.
"""

from __future__ import annotations

import numpy as np


class UnsupportedTask(ValueError):
    """The truth view has no goal for this task (the seed package covers the reach and the drawer)."""


class TruthView:
    def __init__(self, priv):
        self.priv = priv
        self.task = priv.episode()["task"]

    def goal(self) -> dict:
        if self.task.startswith("reach_target"):
            return self._reach()
        if self.task.startswith("drawer_top"):
            return self._drawer()
        raise UnsupportedTask(f"no truth goal for {self.task}")

    def _reach(self) -> dict:
        targets = self.priv.facts()["targets"]
        t = targets[0]
        return dict(kind="reach", side=t["hands"][0], point=np.asarray(t["pos"], float))

    def _drawer(self) -> dict:
        f = self.priv.facts()
        joints = self.priv.joints()
        # the top drawer is the drawer with the greatest z among the sliding joints of the drawers (the last of cabinet_drawers.get_state())
        sliding = {n: j for n, j in joints.items() if j["type"] == 2 and "drawer" in n}
        if not sliding:
            raise UnsupportedTask("no drawer joint in the scene")
        top = max(sliding, key=lambda n: self.priv.body_pose(sliding[n]["body"])[2, 3])
        j = sliding[top]
        bars = f["handles"].get(j["body"], [])
        if not bars:
            raise UnsupportedTask(f"{top} has no handle")
        lo, hi = np.asarray(bars[0]["lo"]), np.asarray(bars[0]["hi"])
        ext = hi - lo
        axis = np.zeros(3)
        axis[int(np.argmax(ext))] = 1.0  # the bar is longest along its own axis
        rng = j["range"]
        return dict(kind="drawer", name=top, handle=(lo + hi) / 2, bar_axis=axis, pull=np.asarray(j["axis_world"], float),
                    travel=float(rng[1] - rng[0]), open_frac=float(f["drawers"][-1]))  # fmt: skip
