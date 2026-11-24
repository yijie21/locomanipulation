"""SIMPLE's privileged state for the truth view (TODO 49.2 step 6): bench/channels.py `Privileged` on the official bed.

Everything comes from the environment's server, never from the local MuJoCo mirror's object bodies (those stay at the
start layout on this bed): object poses from the "truth" reply, collision vertices from "verts" / "art_verts" (world
frame, per collision geom), the pelvis from the base state the server last sent (the mirror's free joint, pulled each
step), the episode's facts (instruction, names, tables, start) from the episode info. All positions are in the mirror's
frame: the server's world plus `zoff` in z (the AMO tasks' world lies that far lower than the mirror's). The robot's own
frame (the Belief's) is the TruthView's business.

Only the truth View may hold one; a skill package cannot import this module (guard rule R8).
"""

from __future__ import annotations

import numpy as np

from locomani.backends.scalebfm import runtime as R


class SimplePrivileged:
    def __init__(self, sim, info: dict, task: str, instruction: str = ""):
        self.sim, self.info, self.task, self.instruction = sim, info, task, instruction

    def _ask(self, *msg):
        self.sim.link.send(msg)
        return self.sim.link.recv()

    def _z(self, p) -> np.ndarray:
        p = np.array(p, float)
        p[..., 2] += self.sim.zoff
        return p

    def pelvis(self) -> tuple[np.ndarray, float]:
        q = self.sim.d.qpos
        return q[:3].copy(), float(R.yaw_of(q[3:7]))

    def body_pose(self, name: str) -> np.ndarray:
        o = self._ask("truth")["objects"][name]
        T = np.eye(4)
        T[:3, :3] = R.mat(np.asarray(o[3:7], float))
        T[:3, 3] = self._z(o[:3])
        return T

    def body_vertices(self, name: str) -> np.ndarray:
        return np.vstack([self._z(v) for v in self._ask("verts", [name])[name]])

    def episode(self) -> dict:
        i = self.info
        return dict(
            instruction=self.instruction, task=self.task, target=i["target"], container=i["objects"].get("container"),
            tables=i["tables"], start=i["start"], objects=i["objects"],
        )  # fmt: skip

    def art_geoms(self) -> dict:
        """The articulated asset, per collision geom: {"obj": [vertices of every geom of the asset], "moving": {body: [vertices]}, "depth": {body: moving ancestors}}."""
        r = self._ask("art_verts")
        return dict(
            obj=[self._z(v) for v in r["obj"]],
            moving={n: [self._z(v) for v in g] for n, g in r["moving"].items()},
            depth=r.get("depth", {}),
        )
