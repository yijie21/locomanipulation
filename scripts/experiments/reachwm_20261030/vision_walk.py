"""Where the walk ended, from the head camera instead of the leg odometry: the scene in front (furniture, objects) seen before
and after the walk, laid on itself by a planar registration. The legs' estimate drifts about 2 cm per metre and does not see
the feet slide; the furniture does not move.

Eye(env).cloud()        the head camera's points in the pelvis' heading frame (x ahead, y left, z above the floor): RealSense-like
                        depth (vision_depth.py), the camera's mounting error of VISION_CALIB, no floor, none of the robot's body.
                        The camera's pose in the pelvis comes from the simulator's kinematics (joint readings and the gravity
                        reading on a robot), not from where the robot is in the room.
register(A, B, init)    (dx, dy, dyaw) of the pelvis between the two clouds, in A's frame, from `init` (the leg odometry's).
"""

import math

import mujoco
import numpy as np
import vision_depth as VD
from scipy.spatial import cKDTree

W, H, N_PTS, FLOOR = 640, 480, 6000, 0.04


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Eye:
    def __init__(self, env, calib_deg: float = 0.5, calib_scale: float = 0.01):
        sm = env.sim.m
        self.env, self.rd, self.rs = env, mujoco.Renderer(sm, H, W), mujoco.Renderer(sm, H, W)
        self.rd.enable_depth_rendering()
        self.rs.enable_segmentation_rendering()
        root = sm.body_rootid[sm.body("pelvis").id]
        self.robot = np.array([g for g in range(sm.ngeom) if sm.body_rootid[sm.geom_bodyid[g]] == root])
        self.cid = sm.camera("head_camera").id
        self.calib_deg, self.calib_scale = calib_deg, calib_scale
        self.new_episode(0)

    def new_episode(self, seed: int) -> None:
        """One camera for the episode: its mounting error and depth scale stay, the noise is drawn per frame."""
        self.rng = np.random.default_rng(8100 + seed)
        w = self.rng.normal(0.0, math.radians(self.calib_deg) / math.sqrt(3), 3)
        q, R = np.zeros(4), np.zeros(9)
        mujoco.mju_axisAngle2Quat(q, w / max(1e-12, np.linalg.norm(w)), float(np.linalg.norm(w)))
        mujoco.mju_quat2Mat(R, q)
        self.mount, self.scale = R.reshape(3, 3), 1.0 + float(self.rng.normal(0.0, self.calib_scale))

    def _depth(self):
        m, d, c = self.env.sim.m, self.env.sim.d, self.cid
        pos0 = m.cam_pos[c].copy()
        mujoco.mj_forward(m, d)
        m.cam_pos[c] = pos0 + d.xmat[m.cam_bodyid[c]].reshape(3, 3).T @ (
            d.cam_xmat[c].reshape(3, 3)[:, 0] * VD.BASELINE
        )
        mujoco.mj_forward(m, d)
        self.rd.update_scene(d, camera=c)
        right = self.rd.render().copy()
        m.cam_pos[c] = pos0
        mujoco.mj_forward(m, d)
        self.rd.update_scene(d, camera=c)
        depth = self.rd.render().copy()
        self.rs.update_scene(d, camera=c)
        seg = self.rs.render()[..., 0].copy()
        return depth, right, seg

    def cloud(self) -> np.ndarray:
        env, d, c = self.env, self.env.sim.d, self.cid
        depth, right, seg = self._depth()
        fovy = float(env.sim.m.cam_fovy[c])
        rs = VD.realsense(depth, right, fovy, self.rng) * self.scale
        ok = (rs > 0) & (rs < 3.0) & ~np.isin(seg, self.robot)
        f = 0.5 * H / math.tan(math.radians(fovy) / 2)
        v, u = np.nonzero(ok)
        z = rs[v, u]
        pc = np.c_[(u - W / 2 + 0.5) * z / f, (v - H / 2 + 0.5) * z / f, z]
        Rc = d.cam_xmat[c].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]) @ self.mount
        pv, Rp = env.pelvis()
        yaw = math.atan2(Rp[1, 0], Rp[0, 0])
        P = (d.cam_xpos[c] + pc @ Rc.T - np.r_[pv[:2], 0.0]) @ rot_z(yaw)  # world -> the pelvis' heading frame, z stays
        P = P[P[:, 2] > FLOOR]
        if len(P) > N_PTS:
            P = P[self.rng.choice(len(P), N_PTS, replace=False)]
        return P


def register(A: np.ndarray, B: np.ndarray, init, iters: int = 40):
    """Planar (dx, dy, dyaw) with A ~ Rz(dyaw) B + (dx, dy): trimmed point-to-point ICP, matches in 3-D, the solve in the plane.
    -> (dx, dy, dyaw), info (pairs kept, their mean distance); the init comes back when too little overlaps."""
    x, y, th = (float(v) for v in init)
    if len(A) < 200 or len(B) < 200:
        return (x, y, th), dict(pairs=0, dist=float("nan"), ok=False)
    tree = cKDTree(A)
    kept, dist = 0, float("nan")
    for it in range(iters):
        Bt = B @ rot_z(th).T + np.r_[x, y, 0.0]
        dd, idx = tree.query(Bt)
        gate = max(0.02, 0.10 * (1.0 - it / (0.6 * iters)))
        keep = dd < min(gate, np.quantile(dd, 0.7))
        kept = int(keep.sum())
        if kept < 150:
            return tuple(float(v) for v in init), dict(pairs=kept, dist=float("nan"), ok=False)
        p, q = Bt[keep, :2], A[idx[keep], :2]
        pc_, qc_ = p.mean(0), q.mean(0)
        Hm = (p - pc_).T @ (q - qc_)
        dth = math.atan2(Hm[0, 1] - Hm[1, 0], Hm[0, 0] + Hm[1, 1])
        c, s = math.cos(dth), math.sin(dth)
        Rd = np.array([[c, -s], [s, c]])
        t = qc_ - Rd @ pc_
        x, y = (Rd @ np.r_[x, y] + t).tolist()
        th += dth
        dist = float(dd[keep].mean())
        if abs(dth) < 1e-5 and np.linalg.norm(t) < 1e-5:
            break
    return (x, y, th), dict(pairs=kept, dist=dist, ok=True)
