"""The truth view: the scene as the simulator's own state gives it (TODO 49.2 step 6; was seed_pkg skill_simple.py's TruthView).

`TruthView` answers every question the seed skill asks of a view (heading, look, target, grasp, edge, basket, basket_near,
find_basket, and for the four articulated tasks art()) from a `Privileged` (bench/channels.py: object poses, collision
vertices, the pelvis, the episode's facts), expressed in the robot's frame through the Belief's tie to the simulator: a
position the Privileged gives in the simulator's frame goes through `belief.from_sim`, a rotation through
rot_z(yb - yt), exactly what the camera view's points have been through. The grasp library is the truth track's (`load_library`);
the skill's `choose_grasp` is handed in because it is loop-editable.

Privileged: used with target=truth only, never built for the camera track. Fixed infrastructure, not part of a skill package
(guard rule R8 forbids a package to import it).
"""

import numpy as np

from locomani.perception.grasps import load_library

PART_MAX_EXTENT = 0.5  # m: a moving body's collision geom is a candidate "part" (handle, lid) when its box is no larger


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TruthView:
    search = False  # the container's place is known without turning to look for it

    def __init__(self, priv, belief, choose_grasp, lib_dir, K: dict, task: str):
        self.priv, self.bel, self.choose_grasp, self.lib_dir, self.K, self.task = (
            priv,
            belief,
            choose_grasp,
            lib_dir,
            K,
            task,
        )
        self.ep = priv.episode()

    # ---- frames

    def pos(self, p) -> np.ndarray:
        """Simulator-frame position(s) -> the robot's frame."""
        return self.bel.from_sim(p)

    def rot(self, Rm) -> np.ndarray:
        return rot_z(self.bel.yb - self.bel.yt) @ Rm

    # ---- the view's questions

    def heading(self) -> float:
        return float(self.ep["start"]["yaw"]) + (self.bel.yb - self.bel.yt)

    def look(self) -> dict:
        return {}  # nothing to look at: the simulator is read when asked

    def target(self):
        T = self.priv.body_pose("target")
        V = self.pos(self.priv.body_vertices("target"))
        Tr = np.eye(4)
        Tr[:3, :3], Tr[:3, 3] = self.rot(T[:3, :3]), self.pos(T[:3, 3])
        return dict(T_wo=Tr, lo=V.min(0), hi=V.max(0), top=float(V[:, 2].min()))

    def grasp(self, tgt, yaw: float):
        # the task's library (grasp_lib.py): the most typical grasp first
        return self.choose_grasp(
            load_library(self.lib_dir, self.task), tgt["T_wo"], yaw, tgt["top"], self.K, high=float(tgt["hi"][2])
        )

    def edge(self, ahead, lateral: float, table: str = "table"):
        """Where the table begins along `ahead` (the coordinate p . ahead of its nearest corner)."""
        t = self.ep["tables"].get(table)
        if t is None:
            return None
        Rt = self.priv_R(t["quat"])
        h = 0.5 * np.asarray(t["size"], float)
        corners = [self.pos(np.asarray(t["pos"], float) + Rt @ (h * [sx, sy, 1])) for sx in (-1, 1) for sy in (-1, 1)]
        return float(min(c[:2] @ ahead for c in corners))

    @staticmethod
    def priv_R(quat) -> np.ndarray:
        from locomani.backends.scalebfm import runtime as R

        return R.mat(np.asarray(quat, float))

    def basket(self, yaw2: float):
        """The basket's centre (xy), rim height and near side along the heading `yaw2`."""
        V = self.pos(self.priv.body_vertices("container"))
        ahead2 = np.array([np.cos(yaw2), np.sin(yaw2)])
        return 0.5 * (V.min(0) + V.max(0))[:2], float(V[:, 2].max()), float((V[:, :2] @ ahead2).min())

    def basket_near(self, yaw2: float):
        """The second look at the basket from near (the object held aside): the same truth, (centre, rim)."""
        c, rim, _ = self.basket(yaw2)
        return c, rim

    def find_basket(self, yaw: float):
        yaw2 = yaw + 0.0
        return yaw2, self.basket(yaw2)

    # ---- the articulated tasks

    def art(self, after: float = 0.2) -> dict:
        """What CameraView.art() returns, from truth: the asset's points (collision vertices of every geom) and its moving
        parts, one candidate per collision geom of a moving body (the hinged leaf, the lid, the handle), points `c`, centre `p`,
        corners `lo` / `hi`, `n` points, score 1 + the body's depth (the number of moving bodies above it: a knob hinged on a hinged leaf
        outranks the leaf's own geoms, which is what a SAM "handle" phrase finds); best first by K["art_pick"] as the camera view sorts them. The asset's own
        geometry does not name a handle, so a part is a geom no larger than PART_MAX_EXTENT (all geoms when none is)."""
        K, bel = self.K, self.bel
        g = self.priv.art_geoms()
        pb, yaw = bel.pelvis()
        ahead = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-ahead[1], ahead[0]])
        out = dict(rgb=None, pts=None, ok=None, pb=pb, yaw=yaw, obj=None, parts=[])
        if not g["obj"]:
            return out
        out.update(obj=self.pos(np.vstack(g["obj"])), obj_score=1.0)
        cand = []
        for body, geoms in g["moving"].items():
            for V in geoms:
                c = self.pos(V)
                cand.append(dict(c=c, p=np.median(c, axis=0), lo=np.percentile(c, 5, axis=0), hi=np.percentile(c, 95, axis=0),
                                 n=int(len(c)), score=1.0 + float(g.get("depth", {}).get(body, 0)), phrase=body, extent=float(np.ptp(c, axis=0).max())))  # fmt: skip
        small = [r for r in cand if r["extent"] <= PART_MAX_EXTENT]
        out["parts"] = small or cand
        pick = {"score": lambda r: -r["score"], "high": lambda r: -r["p"][2], "low": lambda r: r["p"][2],
                "left": lambda r: -(r["p"][:2] @ left), "right": lambda r: r["p"][:2] @ left,
                "near": lambda r: (r["p"][:2] - pb[:2]) @ ahead, "big": lambda r: -r["n"]}[K["art_pick"]]  # fmt: skip
        out["parts"].sort(key=pick)
        return out
