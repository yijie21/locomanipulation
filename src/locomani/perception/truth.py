"""The truth view: the scene read from the simulator (moved from seed_pkg skill_simple.py, TODO 49.2).

`TruthView` answers the view questions (target, grasp, table edge, basket, heading) from the simulator's state; it is the
privileged counterpart of the camera view and is built only when the run's options say target=truth (not part of 49.2's
scope: the SIMPLE bed refuses that switch for now, TODO 49.2 step 6). Fixed infrastructure: a skill package cannot define
or import it.
"""

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.perception.grasps import load_library


def _T(p, q) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3], M[:3, :3] = p, R.mat(np.asarray(q, float))
    return M


def body_verts(sim, body: int) -> np.ndarray:
    """World positions of the vertices of a body's collision meshes."""
    m, d = sim.m, sim.d
    V = []
    for g in np.flatnonzero(m.geom_bodyid == body):
        k = m.geom_dataid[g]
        if k < 0 or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
            continue
        v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        V.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
    return np.vstack(V)


class TruthView:
    """The scene read from the simulator, in the robot's frame (which is the simulator's on this track). Privileged: the truth
    track (target=truth), never built by the camera track. Fixed infrastructure, not part of a skill package."""

    search = False  # the basket's place is known without turning to look for it

    def __init__(self, env, choose_grasp, lib_dir):
        self.env, self.choose_grasp, self.lib_dir = (
            env,
            choose_grasp,
            lib_dir,
        )  # choose_grasp: the skill's (it is editable)

    def heading(self) -> float:
        return float(self.env.info["start"]["yaw"])

    def target(self):
        sim, info = self.env.sim, self.env.info
        b = sim.m.body(info["target"]).id
        V = body_verts(sim, b)
        return dict(T_wo=_T(sim.d.xpos[b], sim.d.xquat[b]), lo=V.min(0), hi=V.max(0), top=float(V[:, 2].min()))

    def grasp(self, tgt, yaw: float):
        # the task's library (grasp_lib.py): the most typical grasp first
        return self.choose_grasp(
            load_library(self.lib_dir, self.env.task),
            tgt["T_wo"],
            yaw,
            tgt["top"],
            self.env.K,
            high=float(tgt["hi"][2]),
        )

    def edge(self, ahead, lateral: float, table: str = "table"):
        """Where the table begins along `ahead` (the coordinate p . ahead of its nearest corner)."""
        t = self.env.info["tables"].get(table)
        if t is None:
            return None
        Rt = R.mat(np.asarray(t["quat"], float))
        h = 0.5 * np.asarray(t["size"], float)
        corners = [np.asarray(t["pos"]) + Rt @ (h * [sx, sy, 1]) for sx in (-1, 1) for sy in (-1, 1)]
        return float(min(c[:2] @ ahead for c in corners))

    def basket(self, yaw2: float):
        """The basket's centre (xy), rim height and near side along the heading `yaw2`."""
        sim = self.env.sim
        V = body_verts(sim, sim.m.body(self.env.info["objects"]["container"]["body"]).id)
        ahead2 = np.array([np.cos(yaw2), np.sin(yaw2)])
        return 0.5 * (V.min(0) + V.max(0))[:2], float(V[:, 2].max()), float((V[:, :2] @ ahead2).min())
