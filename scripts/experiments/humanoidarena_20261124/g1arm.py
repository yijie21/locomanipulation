"""The G1's arms as a kinematic chain from the pelvis, read from a URDF (numpy only: imported by the Isaac side and by
cuRobo's venv). Forward kinematics, the point Jacobian, and damped least-squares IK on one arm's 7 joints with the
waist and everything else held. kin_compare.py checks it (and cuRobo's) against Isaac Lab's kinematics of
HumanoidArena's G1; the URDF it uses is the one that check names (SIMPLE's g1_29dof_with_dex3.urdf by default).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SIMPLE_URDF = Path("/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE/data/robots/g1/g1_29dof_with_dex3.urdf")
HA_URDF = (
    Path(__file__).resolve().parent / "curobo/g1_29dof_with_dex3_ha.urdf"
)  # kin_compare.py: 0.001 mm off Isaac Lab
ARM = {
    s: [f"{s}_shoulder_pitch_joint", f"{s}_shoulder_roll_joint", f"{s}_shoulder_yaw_joint", f"{s}_elbow_joint",
        f"{s}_wrist_roll_joint", f"{s}_wrist_pitch_joint", f"{s}_wrist_yaw_joint"]
    for s in ("left", "right")
}  # fmt: skip


def _rpy(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])  # fmt: skip


def _axis_rot(a, q) -> np.ndarray:
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def rotvec(Rm) -> np.ndarray:
    c = np.clip((np.trace(Rm) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    if np.pi - th < 1e-6:  # near pi: from the symmetric part
        B = (Rm + np.eye(3)) / 2
        ax = np.sqrt(np.clip(np.diag(B), 0, None))
        i = int(np.argmax(ax))
        ax = B[i] / max(ax[i], 1e-12)
        return th * ax / np.linalg.norm(ax)
    return th * v / (2 * np.sin(th))


class Chain:
    """pelvis -> `tip` through the URDF's joints. q is a {joint name: angle} dict (missing joints are 0)."""

    def __init__(self, urdf=HA_URDF, tip="right_hand_palm_link", base="pelvis"):
        root = ET.parse(str(urdf)).getroot()
        by_child = {}
        self.limits = {}
        for j in root.iter("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [1.0, 0.0, 0.0]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
            lim = j.find("limit")
            if lim is not None and lim.get("lower") is not None:
                self.limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
            by_child[j.find("child").get("link")] = dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"), T=T, axis=np.array(axis)
            )
        path, link = [], tip
        while link != base:
            jt = by_child[link]
            path.append(jt)
            link = jt["parent"]
        self.path = path[::-1]
        self.tip = tip
        self.movable = [j["name"] for j in self.path if j["type"] in ("revolute", "continuous")]

    def fk(self, q: dict, frames: bool = False):
        """The tip's 4x4 in the pelvis frame; with frames=True also {joint: (origin, axis)} in the pelvis frame."""
        T = np.eye(4)
        fr = {}
        for j in self.path:
            T = T @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                fr[j["name"]] = (T[:3, 3].copy(), T[:3, :3] @ (j["axis"] / np.linalg.norm(j["axis"])))
                M = np.eye(4)
                M[:3, :3] = _axis_rot(j["axis"], float(q.get(j["name"], 0.0)))
                T = T @ M
        return (T, fr) if frames else T

    def point(self, q: dict, off=(0.0, 0.0, 0.0)):
        """A point fixed in the tip frame (`off`) and the tip's rotation, in the pelvis frame."""
        T = self.fk(q)
        return T[:3, 3] + T[:3, :3] @ np.asarray(off, float), T[:3, :3]

    def jacobian(self, q: dict, joints, off=(0.0, 0.0, 0.0)):
        """6 x len(joints): the point's linear velocity and the tip's angular velocity per joint rate (pelvis frame)."""
        T, fr = self.fk(q, frames=True)
        p = T[:3, 3] + T[:3, :3] @ np.asarray(off, float)
        J = np.zeros((6, len(joints)))
        for i, n in enumerate(joints):
            if n in fr:
                o, a = fr[n]
                J[:3, i] = np.cross(a, p - o)
                J[3:, i] = a
        return J, p, T[:3, :3]

    def ik(self, q0: dict, joints, p_goal, R_goal=None, off=(0.0, 0.0, 0.0), iters=60, damp=0.05, rot_w=0.3,
           step_max=0.2, tol=1e-4, margin=0.02):  # fmt: skip
        """Damped least squares on `joints` from q0 (others held): the point `off` (tip frame) to p_goal and, with weight
        rot_w (m per rad), the tip's rotation to R_goal. Joint limits (less `margin`) clamp every step.
        Returns (q dict, position error m, rotation error rad)."""
        q = dict(q0)
        lim = np.array([self.limits.get(n, (-np.pi, np.pi)) for n in joints])
        lo, hi = lim[:, 0] + margin, lim[:, 1] - margin
        x = np.clip(np.array([q.get(n, 0.0) for n in joints]), lo, hi)
        for _ in range(iters):
            for n, v in zip(joints, x):
                q[n] = float(v)
            J, p, Rt = self.jacobian(q, joints, off)
            e = np.asarray(p_goal, float) - p
            if R_goal is not None and rot_w > 0:
                er = rotvec(np.asarray(R_goal) @ Rt.T)
                E = np.r_[e, rot_w * er]
                Jw = np.r_[J[:3], rot_w * J[3:]]
            else:
                E, Jw = e, J[:3]
            if np.linalg.norm(E) < tol:
                break
            dx = Jw.T @ np.linalg.solve(Jw @ Jw.T + damp**2 * np.eye(len(E)), E)
            nd = np.abs(dx).max()
            if nd > step_max:
                dx *= step_max / nd
            x = np.clip(x + dx, lo, hi)
        for n, v in zip(joints, x):
            q[n] = float(v)
        p, Rt = self.point(q, off)
        er = 0.0 if R_goal is None else float(np.linalg.norm(rotvec(np.asarray(R_goal) @ Rt.T)))
        return q, float(np.linalg.norm(np.asarray(p_goal) - p)), er
