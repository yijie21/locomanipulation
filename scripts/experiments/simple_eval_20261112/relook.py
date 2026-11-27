"""A second look at the stance (simple_relook_20261126): localize by registering the furniture's points, step again when
the stance is off, then plan the grasp from the new pose.

The bench's perception column looks once, at the start (blind.py look, cached: cloud_blind, look_frame.py's
look_dump_blind), and after the walk takes its pelvis from leg odometry. simple_bench.py --relook adds, once the walk has
ended:

  look      the head camera's depth image in the replica (see.head_frame, the same camera and calibration as the first
            look; the simulation does not advance while it renders), each pixel's point placed on the pelvis by the joint
            readings (see.HeadKin) and the gravity reading (leg_odo.level), and the pelvis where the robot believes it is
  furniture both looks without the robot's own body (its geoms by forward kinematics from its joint readings, each
            geom's bounding sphere plus SELF_PAD), without the floor (under FLOOR_Z) and farther than MAX_RANGE from the
            camera; the first look also without the target (SAM 3's mask of that look, cloud_blind's "mask"), the second
            without its points within TGT_PAD of the first look's target points as the current estimate places them
  register  the second look's furniture onto the first's: point-to-plane ICP over x, y, z and yaw about the believed
            pelvis (the gravity reading gives roll and pitch), Huber weights, a shrinking correspondence radius (SCHED),
            and the odometry as a prior (PRIOR_SIG): a direction the furniture does not pin (a long table edge) stays
            near the odometry instead of sliding. Starts from the odometry, after a coarse search over x, y and yaw
            (COARSE) scored by the truncated point distance. -> the corrected pelvis (x, y, yaw), rounded to 0.1 mm and
            1e-5 rad so the arms of one episode (A, J, S) take the same steps
  step      the target's place in the body frame by the corrected pose, against its planned place (the planned stance's
            frame); when the two are more than step_cm apart, walk to the planned stance again (gen_reach.open_walk's
            way with the walk's end test reading nothing of the simulator: stop_tolerance -1, a fixed extra time),
            leg odometry over the step, look again; at most max_steps steps

Nothing here reads the simulator's state for a decision: the truth goes into the record only (fields ending in _true).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.spatial import cKDTree

import episodes as EP
import see as SEE

VOX_REF, VOX_MOV = 0.01, 0.015  # m, the voxel grids the two looks are thinned to
FLOOR_Z = 0.04  # m: points lower than this are the floor
SELF_PAD = 0.03  # m added to each of the robot's geoms' bounding spheres
TGT_PAD = 0.04  # m around the first look's target points
MAX_RANGE = 3.0  # m from the camera
NORMAL_K = 12
SIG_R = 0.005  # m, the point-to-plane residual taken as noise
HUBER = 0.01  # m
PRIOR_SIG = (0.05, 0.05, 0.05, math.radians(3.0))  # the odometry prior: x, y, z (m), yaw (rad)
SCHED = (0.12, 0.08, 0.05, 0.035, 0.025, 0.02, 0.02, 0.02)  # m, the correspondence radius, ITER_STAGE iterations each
ITER_STAGE = 3
COARSE = dict(xy=0.24, xy_step=0.03, yaw_deg=4.0, yaw_step_deg=1.0, trunc=0.03, n=1500)
LEVER = 0.5  # m: a yaw is reported as the displacement it makes this far out
FEET_Z = True  # the believed pelvis height is the feet's (Body.height), not leg odometry's z (2026-11-26 development:
# SONIC's pelvis drops 5-6 cm at some stances, odometry keeps the standing height, the floor rose over FLOOR_Z and
# swamped the registration, inliers 12-14 %)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def voxel(P: np.ndarray, size: float) -> np.ndarray:
    """One point a voxel (the voxel's first point in array order), the voxels in sorted order: deterministic."""
    if len(P) == 0:
        return P
    k = np.floor(P / size).astype(np.int64)
    _, i = np.unique(k, axis=0, return_index=True)
    return P[np.sort(i)]


def normals(P: np.ndarray, tree: cKDTree) -> np.ndarray:
    _, idx = tree.query(P, min(NORMAL_K, len(P)))
    Q = P[idx] - P[idx].mean(axis=1, keepdims=True)
    _, v = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))
    return v[:, :, 0]


class Body:
    """The robot's own geoms in the pelvis frame from its joint readings (its own model in a private MjData): spheres."""

    def __init__(self, model: mujoco.MjModel, qadr: np.ndarray):
        self.m, self.d, self.qadr = model, mujoco.MjData(model), np.asarray(qadr)
        root = model.body_rootid[model.body("pelvis").id]
        self.g = np.flatnonzero(model.body_rootid[model.geom_bodyid] == root)
        self.r = model.geom_rbound[self.g] + SELF_PAD

    def outside(self, P_pel: np.ndarray, joints: np.ndarray) -> np.ndarray:
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qpos[:3], d.qpos[3:7], d.qpos[self.qadr] = 0.0, (1.0, 0.0, 0.0, 0.0), joints
        mujoco.mj_kinematics(self.m, d)
        keep = np.ones(len(P_pel), bool)
        for c, r in zip(d.geom_xpos[self.g], self.r):
            keep &= ((P_pel - c) ** 2).sum(1) > r * r
        return keep

    def height(self, joints: np.ndarray, L: np.ndarray) -> float:
        """The pelvis' height over the floor the feet stand on: minus the lowest corner of the feet's collision boxes
        (motor.lowest_foot_point's) in the level pelvis frame, from the joint readings and the gravity reading (L)."""
        from locomani.backends.scalebfm.motor import lowest_foot_point

        m, d = self.m, self.d
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, np.ascontiguousarray(L, float).reshape(-1))  # the body's attitude in the level frame
        d.qpos[:] = m.qpos0
        d.qpos[:3], d.qpos[3:7], d.qpos[self.qadr] = 0.0, q, joints
        mujoco.mj_kinematics(m, d)
        feet = {m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")}
        return -lowest_foot_point(m, d, feet)


def level_R(grav: np.ndarray) -> np.ndarray:
    """Body -> level frame (forward on the floor, left, up) from the gravity reading in the body frame (leg_odo.level)."""
    up = -np.asarray(grav, float)
    up /= np.linalg.norm(up)
    x = np.array([1.0, 0.0, 0.0]) - up[0] * up
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up])


class Reference:
    """The first look's furniture, in the robot's frame at the start (the simulator's then: blind.py's convention)."""

    def __init__(self, dump: dict, mask: np.ndarray, target: np.ndarray, body: Body, joints0: np.ndarray, L0: np.ndarray):
        """dump: look_dump_blind's arrays (pts, ok, pb, yaw, cam_p); mask: the target's pixels (cloud_blind); target: the
        target's points; body, joints0, L0 (level frame at the start): to leave out the robot's own body."""
        ok = np.asarray(dump["ok"], bool) & ~np.asarray(mask, bool)
        P = np.asarray(dump["pts"], float)[ok]
        P = P[(P[:, 2] > FLOOR_Z) & (np.linalg.norm(P - np.asarray(dump["cam_p"], float), axis=1) < MAX_RANGE)]
        Rp = rot_z(float(dump["yaw"])) @ L0  # pelvis frame -> robot's frame (level by the gravity reading)
        P_pel = (P - np.asarray(dump["pb"], float)) @ Rp
        P = P[body.outside(P_pel, joints0)]
        if len(target):
            d, _ = cKDTree(target).query(P, distance_upper_bound=0.02)
            P = P[~np.isfinite(d)]
        self.P = voxel(P, VOX_REF)
        self.tree = cKDTree(self.P)
        self.N = normals(self.P, self.tree)
        self.target = np.asarray(target, float)
        self.target_tree = cKDTree(self.target) if len(self.target) else None


def second_look(env, kin: SEE.HeadKin, body: Body) -> tuple[np.ndarray, int]:
    """The head camera's points in the level pelvis frame (forward on the floor, left, up; origin the pelvis), without the
    robot's own body; also the number of pixels with a depth."""
    sim = env.sim
    _, depth = SEE.head_frame(sim)
    joints = sim.d.qpos[sim.qadr].copy()
    rel_p, rel_R = kin(joints)
    ok = ((depth > 0.2) & (depth < MAX_RANGE)).reshape(-1)
    c = SEE.cam_points(depth).reshape(-1, 3)[ok]
    P_pel = rel_p + c @ rel_R.T
    P_pel = P_pel[body.outside(P_pel, joints)]
    L = level_R(gravity(env))
    return P_pel @ L.T, int(ok.sum()), body.height(joints, L)


def gravity(env) -> np.ndarray:
    """The gravity reading in the body frame, as env.Env.self_state gives it to leg odometry (the IMU's), without
    calling self_state."""
    return env.pelvis()[1].T @ np.array([0.0, 0.0, -1.0])


def place(P_lev: np.ndarray, pb: np.ndarray, yb: float) -> np.ndarray:
    return np.asarray(pb, float) + P_lev @ rot_z(yb).T


def apply(x: np.ndarray, P: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The correction x = (tx, ty, tz, yaw) about the point c."""
    R = rot_z(float(x[3]))
    return (P - c) @ R.T + c + x[:3]


def coarse(ref: Reference, M: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The grid of x, y, yaw corrections whose moved points lie nearest the reference (mean truncated distance); z is
    left to the ICP. -> (x, its score, the score of no correction)."""
    k = COARSE
    sub = M[np.linspace(0, len(M) - 1, min(k["n"], len(M))).astype(int)]
    best, s0 = None, None
    g = np.arange(-k["xy"], k["xy"] + 1e-9, k["xy_step"])
    for a in np.radians(np.arange(-k["yaw_deg"], k["yaw_deg"] + 1e-9, k["yaw_step_deg"])):
        Pa = apply(np.array([0.0, 0.0, 0.0, a]), sub, c)
        for dx in g:
            for dy in g:
                d, _ = ref.tree.query(Pa + np.array([dx, dy, 0.0]), distance_upper_bound=k["trunc"])
                s = float(np.minimum(d, k["trunc"]).mean())
                if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(a) < 1e-9:
                    s0 = s
                if best is None or s < best[1] - 1e-12:
                    best = (np.array([dx, dy, 0.0, a]), s)
    return best[0], best[1], s0


def register(ref: Reference, M_bel: np.ndarray, c: np.ndarray, use_coarse: bool = True) -> dict:
    """M_bel: the second look's furniture placed by the believed pelvis; c: the believed pelvis. -> dict(x = the
    correction (tx, ty, tz, yaw), the record)."""
    Lam = np.diag(1.0 / np.square(PRIOR_SIG))
    out = dict(n_mov=len(M_bel), n_ref=len(ref.P))
    x = np.zeros(4)
    if use_coarse:
        x, s, s0 = coarse(ref, M_bel, c)
        out.update(coarse=np.round(x, 5).tolist(), coarse_mm=round(1000 * s, 2), coarse0_mm=round(1000 * s0, 2))
    it, H_d = 0, np.zeros((4, 4))
    for it in range(ITER_STAGE * len(SCHED) + 10):
        dmax = SCHED[min(it // ITER_STAGE, len(SCHED) - 1)]
        P = apply(x, M_bel, c)
        d, j = ref.tree.query(P, distance_upper_bound=dmax)
        ok = np.isfinite(d)
        if ok.sum() < 50:
            out["why"] = "too few matches"
            break
        p, q, n = P[ok], ref.P[j[ok]], ref.N[j[ok]]
        r = np.einsum("ij,ij->i", n, p - q)
        c2 = c + x[:3]
        J = np.c_[n, n[:, 1] * (p[:, 0] - c2[0]) - n[:, 0] * (p[:, 1] - c2[1])]
        a = np.abs(r)
        w = np.where(a <= HUBER, 1.0, HUBER / np.maximum(a, 1e-12)) / SIG_R**2
        H_d = (J * w[:, None]).T @ J
        g = (J * w[:, None]).T @ r + Lam @ x
        dx = -np.linalg.solve(H_d + Lam, g)
        x = x + dx
        if it >= ITER_STAGE * (len(SCHED) - 1) and np.abs(dx[:3]).max() < 1e-5 and abs(dx[3]) < 1e-6:
            break
    P = apply(x, M_bel, c)
    d, _ = ref.tree.query(P, distance_upper_bound=0.02)
    inl = np.isfinite(d)
    # how well the furniture alone pins each direction: the std of x, y (cm) and yaw (deg) from the data's information
    cov = np.linalg.pinv(H_d) if np.abs(H_d).sum() > 0 else np.full((4, 4), np.inf)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    out.update(x=np.round(x, 5).tolist(), iters=it + 1, inlier=round(float(inl.mean()), 3),
               rms_mm=round(1000 * float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("nan"), 2),
               sd_cm=[round(100 * float(sd[0]), 3), round(100 * float(sd[1]), 3), round(100 * float(sd[2]), 3)],
               sd_deg=round(math.degrees(float(sd[3])), 4))  # fmt: skip
    return out


def localize(env, ref: Reference, kin, body: Body, pb: np.ndarray, yb: float, use_coarse: bool = True, keep=None):
    """One look and its registration from the believed pelvis (pb, yb). -> (pelvis xy z, yaw, record). keep: a dict
    that gets the look's points (level pelvis frame), for development replays."""
    P_lev, n_px, h = second_look(env, kin, body)
    if keep is not None:
        keep["P_lev"], keep["h"] = P_lev, h
    if FEET_Z:  # the believed pelvis height from the feet (leg odometry's z drifts when the pelvis drops)
        pb = np.r_[np.asarray(pb, float)[:2], h]
    M = place(P_lev, pb, yb)
    M = M[M[:, 2] > FLOOR_Z]
    est = dict(pb=np.asarray(pb, float), yb=float(yb))
    for _ in range(2):  # the target's points out, by where the current estimate puts the first look's target
        if ref.target_tree is not None:
            Mw = apply(np.r_[est["pb"] - pb, est["yb"] - yb], M, np.asarray(pb, float))
            d, _ = ref.target_tree.query(Mw, distance_upper_bound=TGT_PAD)
            keep = ~np.isfinite(d)
        else:
            keep = np.ones(len(M), bool)
        Mv = voxel(M[keep], VOX_MOV)
        rec = register(ref, Mv, np.asarray(pb, float), use_coarse)
        x = np.asarray(rec["x"])
        est = dict(pb=np.asarray(pb, float) + x[:3], yb=float(yb) + float(x[3]))
    rec.update(n_px=n_px, n_target_out=int((~keep).sum()), h_feet=round(float(h), 4))
    p = np.round(est["pb"], 4)
    return p, round(est["yb"], 5), rec
