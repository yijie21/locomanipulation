"""Reference clips for the complex motions C1-C7 (TODO 25.4, plan §4.3), on the mars robot.

One clip = 50 Hz whole-body references in a canonical frame (pelvis at the origin facing +x at t = 0): `qpos` (T, 36)
= pelvis position, pelvis quaternion (wxyz), 29 joints in MuJoCo order; `grip` (T, 2) = left / right jaw command
(0 open, 1 closed); plus keyframes (a hand's jaw centre pose to hold between t0 and t1). Both controllers are fed the
same clip anchored at the robot's settled start pose: SONIC its 29 joints and the root yaw on the streamed channel,
ScaleBFM the 14 keypoints that FK of the clip gives on the mars model (the mode picks which it sees).

Sources (plan §4.3):
* in distribution -- Unitree's G1 retarget of LAFAN1 (datasets/lafan1_g1_retarget, 30 fps CSV: pelvis xyz,
  quaternion xyzw, 29 joints), resampled to 50 Hz, floor-aligned on the mars legs. `L0` a forward walk; C4 sidestep
  left / right, walk backward, turn in place (searched in the walk takes by the pelvis velocity in its heading frame),
  each also `_load` = the same legs with the arms holding a carry pose (run on the walk_mars_load12 scene).
* out of distribution -- whole-body IK keyframes on the mars model (feet fixed, centre of mass over the feet, jaw
  centre targets): C1 floor pick, C2 high reach, C5 deep lean, C6 bimanual floor lift, C7 right-to-left hand-over;
  C3 = the L0 walk with the right arm re-solved to grab a fixed world point on the way.
  LAFAN1's G1 takes have no floor pick or high reach, so C1 / C2 are keyframed too (DECISIONS 2026-10-13, 25.4).

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/controller_compare_20261013/clips.py \\
        --out outputs/controller_compare_20261013/clips
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

SCENE = REPO / "outputs" / "scene" / "walk_mars"
LAFAN = REPO / "datasets" / "lafan1_g1_retarget" / "g1"
META = REPO / "assets" / "scalebfm" / "model_22200_tensorrt_metadata.json"
HZ = 50
OFF = np.array([0.1515, 0.0, 0.0])  # wrist yaw link -> jaw centre (backends/scalebfm/runtime.OFF)
WRIST = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
KEYPOINTS = tuple(json.loads(META.read_text())["selected_body_names"])  # ScaleBFM's 14 bodies
POS_TOL, ROT_TOL_DEG, HOLD_S = 0.05, 15.0, 0.5  # plan §4.3 keyframe rule


# --- small rotation helpers -----------------------------------------------------------------------------------------
def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def quat_of(Rm):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(Rm, float).reshape(9))
    return q


def mat_of(q):
    r = np.empty(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, float))
    return r.reshape(3, 3)


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rot_err(R_tgt, R_cur):
    """Axis-angle vector (world) that turns R_cur into R_tgt."""
    q = quat_of(R_tgt @ R_cur.T)
    v = np.empty(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def slerp(q0, q1, u):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    c = float(q0 @ q1)
    if c < 0:
        q1, c = -q1, -c
    if c > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    a = np.arccos(c)
    return (np.sin((1 - u) * a) * q0 + np.sin(u * a) * q1) / np.sin(a)


def smooth(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


def jaw_attitude(elev_deg, yaw_deg=0.0, roll_deg=0.0):
    """Wrist rotation whose x-axis (the approach) is turned yaw_deg left of +x and pitched elev_deg down; roll about
    the approach (probe_cells.attitude_in_heading's convention)."""
    return rz(np.radians(yaw_deg)) @ ry(np.radians(elev_deg)) @ rx(np.radians(roll_deg))


# --- the mars model ---------------------------------------------------------------------------------------------------
class Body:
    """FK and whole-body IK on the mars walk scene (pelvis free joint + 29 joints; the jaws stay open)."""

    def __init__(self, scene=SCENE):
        self.m = mujoco.MjModel.from_xml_path(str(Path(scene) / "scene_locomani.xml"))
        self.d = mujoco.MjData(self.m)
        jid = [self.m.joint(n).id for n in MUJOCO_JOINT_NAMES]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in jid])
        self.lo = self.m.jnt_range[jid, 0].copy()
        self.hi = self.m.jnt_range[jid, 1].copy()
        self.bid = {n: self.m.body(n).id for n in (*KEYPOINTS, "torso_link")}
        self.pelvis = self.m.body("pelvis").id
        self.foot_geoms = [
            g
            for g in range(self.m.ngeom)
            if self.m.geom_bodyid[g] in [self.m.body(f).id for f in FEET] and self.m.geom_contype[g]
        ]
        meta = json.loads(META.read_text())
        default = dict(zip(meta["joint_names"], meta["default_dof_pos"]))
        self.default = np.array([default[n] for n in MUJOCO_JOINT_NAMES])
        self.mask = np.zeros(self.m.nv, bool)  # the IK's variables: the pelvis and the 29 body joints
        self.mask[:6] = True
        self.mask[self.dadr] = True

    def put(self, qpos36):
        self.d.qpos[:7] = qpos36[:7]
        self.d.qpos[self.qadr] = qpos36[7:]
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def get(self):
        return np.concatenate([self.d.qpos[:7], self.d.qpos[self.qadr]])

    def pose(self, body):
        b = self.bid[body] if body in self.bid else self.m.body(body).id
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def jaw(self, side):
        p, Rm = self.pose(WRIST[side])
        return p + Rm @ OFF, Rm

    def keypoints(self):
        """(14, 7): ScaleBFM's bodies, position + wxyz quaternion."""
        return np.array([np.concatenate([self.d.xpos[self.bid[b]], self.d.xquat[self.bid[b]]]) for b in KEYPOINTS])

    def sole_z(self):
        """Lowest point of the foot collision geoms (their world AABBs)."""
        lo = np.inf
        for g in self.foot_geoms:
            Rm = self.d.geom_xmat[g].reshape(3, 3)
            c = self.d.geom_xpos[g] + Rm @ self.m.geom_aabb[g, :3]
            half = np.abs(Rm) @ self.m.geom_aabb[g, 3:]
            lo = min(lo, c[2] - half[2])
        return lo

    def standing(self):
        """The default (ScaleBFM) standing pose, pelvis at the origin facing +x, soles on the floor."""
        q = np.concatenate([[0, 0, 0.78, 1, 0, 0, 0], self.default])
        self.put(q)
        q[2] -= self.sole_z()
        self.put(q)
        return q

    # -- IK ------------------------------------------------------------------------------------------------------------
    def solve(self, q, tasks, iters=30, posture=None, w_posture=0.05, damping=1e-3):
        """Damped least squares over the pelvis and the 29 joints. tasks: list of dicts
        kind 'pose' (body, pos, R, w_pos, w_rot[, point]), 'com' (xy, w), 'height' (body, z, w), 'tilt' (body, R, w: the
        body's z-axis only).
        Returns the solution and the final per-task errors."""
        m, d = self.m, self.d
        self.put(q)
        nv = m.nv
        cols = np.flatnonzero(self.mask)
        post = self.default if posture is None else posture
        jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
        jcom = np.zeros((3, nv))
        for _ in range(iters):
            rows, res = [], []
            for t in tasks:
                if t["kind"] == "pose":
                    b = self.m.body(t["body"]).id
                    Rb = d.xmat[b].reshape(3, 3)
                    point = d.xpos[b] + Rb @ t.get("point", np.zeros(3))
                    mujoco.mj_jac(m, d, jacp, jacr, point, b)
                    if t.get("w_pos", 0) > 0:
                        rows.append(np.sqrt(t["w_pos"]) * jacp[:, cols])
                        res.append(np.sqrt(t["w_pos"]) * (t["pos"] - point))
                    if t.get("w_rot", 0) > 0:
                        rows.append(np.sqrt(t["w_rot"]) * jacr[:, cols])
                        res.append(np.sqrt(t["w_rot"]) * rot_err(t["R"], Rb))
                elif t["kind"] == "com":
                    mujoco.mj_jacSubtreeCom(m, d, jcom, self.pelvis)
                    rows.append(np.sqrt(t["w"]) * jcom[:2, cols])
                    res.append(np.sqrt(t["w"]) * (np.asarray(t["xy"]) - d.subtree_com[self.pelvis][:2]))
                elif t["kind"] == "height":
                    b = self.m.body(t["body"]).id
                    mujoco.mj_jac(m, d, jacp, jacr, d.xpos[b], b)
                    rows.append(np.sqrt(t["w"]) * jacp[2:3, cols])
                    res.append(np.sqrt(t["w"]) * np.array([t["z"] - d.xpos[b][2]]))
                elif t["kind"] == "tilt":
                    b = self.m.body(t["body"]).id
                    Rb = d.xmat[b].reshape(3, 3)
                    mujoco.mj_jac(m, d, jacp, jacr, d.xpos[b], b)
                    e = np.cross(Rb[:, 2], t["R"][:, 2])  # turns the body's z onto the target's z
                    rows.append(np.sqrt(t["w"]) * jacr[:2, cols])
                    res.append(np.sqrt(t["w"]) * e[:2])
            J = np.vstack(rows)
            r = np.concatenate(res)
            free = [k for k, a in enumerate(self.dadr) if self.mask[a]]  # posture rows on the free joints only
            P = np.zeros((len(free), cols.size))
            for row, k in enumerate(free):
                P[row, np.searchsorted(cols, self.dadr[k])] = 1.0
            qj = d.qpos[self.qadr[free]]
            J = np.vstack([J, np.sqrt(w_posture) * P])
            r = np.concatenate([r, np.sqrt(w_posture) * (post[free] - qj)])
            dx = np.linalg.solve(J.T @ J + damping * np.eye(cols.size), J.T @ r)
            step = np.max(np.abs(dx))
            if step > 0.2:
                dx *= 0.2 / step
            dq = np.zeros(nv)
            dq[cols] = dx
            qpos = d.qpos.copy()
            mujoco.mj_integratePos(m, qpos, dq, 1.0)
            qpos[self.qadr] = np.clip(qpos[self.qadr], self.lo, self.hi)
            d.qpos[:] = qpos
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
        return self.get(), self.errors(tasks)

    def errors(self, tasks):
        out = []
        for t in tasks:
            if t["kind"] == "pose":
                b = self.m.body(t["body"]).id
                Rb = self.d.xmat[b].reshape(3, 3)
                point = self.d.xpos[b] + Rb @ t.get("point", np.zeros(3))
                out.append(
                    (
                        t["body"],
                        float(np.linalg.norm(t["pos"] - point)),
                        float(np.degrees(np.linalg.norm(rot_err(t["R"], Rb)))),
                    )
                )
            elif t["kind"] == "com":
                out.append(
                    ("com", float(np.linalg.norm(np.asarray(t["xy"]) - self.d.subtree_com[self.pelvis][:2])), 0.0)
                )
        return out


# --- clip container ----------------------------------------------------------------------------------------------------
class Clip:
    def __init__(self, name, item, source, in_dist, qpos, grip=None, keyframes=(), rule="keyframes", scene="walk_mars"):
        self.name, self.item, self.source, self.in_dist = name, item, source, in_dist
        self.qpos = np.asarray(qpos, float)
        self.grip = np.zeros((len(self.qpos), 2)) if grip is None else np.asarray(grip, float)
        self.keyframes = list(keyframes)
        self.rule, self.scene = rule, scene

    @property
    def t(self):
        return np.arange(len(self.qpos)) / HZ

    def save(self, out: Path, body: Body):
        out.mkdir(parents=True, exist_ok=True)
        kp = []
        for q in self.qpos:
            body.put(q)
            kp.append(body.keypoints())
        np.savez_compressed(out / f"{self.name}.npz", t=self.t, qpos=self.qpos, grip=self.grip, keypoints=np.array(kp))
        facts = dict(
            name=self.name,
            item=self.item,
            source=self.source,
            in_distribution=self.in_dist,
            rule=self.rule,
            scene=self.scene,
            duration_s=round(len(self.qpos) / HZ, 2),
            keyframes=[
                dict(k, pos=[round(float(v), 4) for v in k["pos"]], R=np.asarray(k["R"]).round(5).tolist())
                for k in self.keyframes
            ],
            keypoints=list(KEYPOINTS),
        )
        (out / f"{self.name}.json").write_text(json.dumps(facts, indent=1))
        return facts


def load(clip_dir: Path, name: str):
    z = dict(np.load(clip_dir / f"{name}.npz"))
    f = json.loads((clip_dir / f"{name}.json").read_text())
    for k in f["keyframes"]:
        k["pos"], k["R"] = np.asarray(k["pos"]), np.asarray(k["R"])
    return z, f


def anchored(z, xy, yaw):
    """The clip's qpos and keypoints moved from the canonical frame to a start pose (xy, yaw); z is the floor's."""
    Rz = rz(yaw)
    qz = quat_of(Rz)
    q = z["qpos"].copy()
    q[:, :2] = (Rz[:2, :2] @ q[:, :2].T).T + xy
    q[:, 3:7] = np.array([quat_mul(qz, v) for v in q[:, 3:7]])
    kp = z["keypoints"].copy()
    kp[..., :2] = np.einsum("ij,tkj->tki", Rz[:2, :2], kp[..., :2]) + xy
    kp[..., 3:7] = np.array([[quat_mul(qz, v) for v in row] for row in kp[..., 3:7]])
    return q, kp


def anchor_keyframe(k, xy, yaw):
    Rz = rz(yaw)
    return Rz @ k["pos"] + np.array([xy[0], xy[1], 0.0]), Rz @ k["R"]


def quat_mul(a, b):
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


# --- LAFAN1 ---------------------------------------------------------------------------------------------------------------
def read_lafan(name):
    raw = np.loadtxt(LAFAN / f"{name}.csv", delimiter=",")
    n = len(raw)
    t30 = np.arange(n) / 30.0
    t50 = np.arange(0, t30[-1], 1.0 / HZ)
    pos = np.stack([np.interp(t50, t30, raw[:, i]) for i in range(3)], 1)
    quat30 = raw[:, [6, 3, 4, 5]]  # xyzw -> wxyz
    for i in range(1, n):
        if quat30[i] @ quat30[i - 1] < 0:
            quat30[i] = -quat30[i]
    quat = np.stack([np.interp(t50, t30, quat30[:, i]) for i in range(4)], 1)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    joints = np.stack([np.interp(t50, t30, raw[:, 7 + i]) for i in range(29)], 1)
    return np.concatenate([pos, quat, joints], 1)


def heading_motion(q, win=25):
    """Per frame: forward and lateral pelvis speed in its own heading frame, and the yaw rate (0.5 s windows)."""
    yaw = np.unwrap([yaw_of(v) for v in q[:, 3:7]])
    T = len(q)
    vf, vl, wz = np.zeros(T), np.zeros(T), np.zeros(T)
    for i in range(T):
        a, b = max(0, i - win // 2), min(T - 1, i + win // 2)
        dt = (b - a) / HZ
        d = q[b, :2] - q[a, :2]
        c, s = np.cos(yaw[i]), np.sin(yaw[i])
        vf[i] = (c * d[0] + s * d[1]) / dt
        vl[i] = (-s * d[0] + c * d[1]) / dt
        wz[i] = (yaw[b] - yaw[a]) / dt
    return vf, vl, wz


def floor_align(body: Body, q):
    """Shift the pelvis height so the soles touch the floor in stance (10th percentile of the lowest sole over the
    take = 0) on the mars legs."""
    lows = []
    for v in q[::5]:
        body.put(v)
        lows.append(body.sole_z())
    q = q.copy()
    q[:, 2] -= np.percentile(lows, 10)
    return q


def canonical(q):
    """Pelvis at the origin facing +x at frame 0."""
    q = q.copy()
    yaw0 = yaw_of(q[0, 3:7])
    Rz = rz(-yaw0)
    q[:, :2] = (Rz[:2, :2] @ (q[:, :2] - q[0, :2]).T).T
    qz = quat_of(Rz)
    q[:, 3:7] = np.array([quat_mul(qz, v) for v in q[:, 3:7]])
    return q


def find_segment(kind, takes, length_s):
    """Best window of `length_s` of one kind of motion over the walk takes, by the mean of its defining speed."""
    best = None
    n = int(length_s * HZ)
    for name, q in takes.items():
        vf, vl, wz = heading_motion(q)
        for i in range(0, len(q) - n, 10):
            f, lat, w = vf[i : i + n], vl[i : i + n], wz[i : i + n]
            # upright the whole window (the takes also crawl and roll on the floor) and no yaw glitch
            if q[i : i + n, 2].min() < 0.72 or np.abs(w).max() > np.radians(300):
                continue
            if kind == "forward":
                ok = 0.35 <= f.mean() <= 0.75 and np.abs(lat).max() < 0.25 and np.abs(w).max() < 0.4
                score = -abs(f.mean() - 0.5)
            elif kind in ("side_left", "side_right"):
                sgn = 1 if kind == "side_left" else -1
                ok = (
                    np.mean(sgn * lat > 0.05) > 0.8
                    and sgn * lat.mean() > 0.12
                    and np.abs(f).mean() < 0.2
                    and np.abs(w).max() < 1.0
                )
                score = sgn * lat.mean()
            elif kind == "backward":
                ok = np.mean(f < -0.05) > 0.9 and np.abs(lat).mean() < 0.12 and np.abs(w).max() < 0.6
                score = -f.mean()
            elif kind == "turn":
                speed = np.hypot(f, lat)
                ok = speed.mean() < 0.15 and abs(w.sum() / HZ) > np.radians(90)
                score = abs(w.sum() / HZ)
            else:
                raise ValueError(kind)
            if ok and (best is None or score > best[0]):
                best = (score, name, i)
    return best


MIRROR_SIGN = {"roll": -1.0, "yaw": -1.0, "pitch": 1.0, "knee": 1.0, "elbow": 1.0}


def mirror(q):
    """Left-right mirror of a clip: y and the yaw / roll of the pelvis negated, left and right joints swapped, roll and
    yaw joints negated."""
    out = q.copy()
    out[:, 1] = -q[:, 1]
    out[:, 3:7] = q[:, 3:7] * np.array([1.0, -1.0, 1.0, -1.0])
    names = list(MUJOCO_JOINT_NAMES)
    for i, n in enumerate(names):
        other = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
        kind = next(k for k in MIRROR_SIGN if k in n)
        out[:, 7 + i] = MIRROR_SIGN[kind] * q[:, 7 + names.index(other)]
    return out


def lead_in(q, i, before_s=1.5):
    """Start the window up to `before_s` earlier, at the slowest frame (the clip begins from a near-still pose)."""
    a = max(0, i - int(before_s * HZ))
    speed = np.linalg.norm(np.diff(q[a : i + 1, :2], axis=0), axis=1) * HZ
    return a + int(np.argmin(speed)) if len(speed) else i


def hold_ends(q, start_s=0.5, end_s=1.0):
    """Repeat the first frame start_s and the last end_s (the controller starts and ends standing on the clip's pose)."""
    return np.concatenate([np.repeat(q[:1], int(start_s * HZ), 0), q, np.repeat(q[-1:], int(end_s * HZ), 0)])


# --- keyframed clips -------------------------------------------------------------------------------------------------------
def feet_tasks(body, q_stand):
    body.put(q_stand)
    tasks = []
    for f in FEET:
        p, Rm = body.pose(f)
        tasks.append(dict(kind="pose", body=f, pos=p, R=Rm, w_pos=400.0, w_rot=100.0))
    mid = np.mean([body.pose(f)[0][:2] for f in FEET], axis=0)
    return tasks, mid


def hand_task(side, pos, Rm, w_pos=100.0, w_rot=10.0):
    return dict(
        kind="pose",
        body=WRIST[side],
        pos=np.asarray(pos, float),
        R=np.asarray(Rm, float),
        w_pos=w_pos,
        w_rot=w_rot,
        point=OFF,
    )


def keyframed(body, segments, extra=None, com_w=50.0, iters=30, hand_w=(100.0, 10.0), posture=None):
    """segments: list of (duration_s, {side: (pos, R) or None}, grip (l, r)) -- each hand's jaw target moves
    (smoothstep position, slerp attitude) from where the previous segment left it to this one's; None = keep the hand
    where the IK leaves it (no task). extra(t_s) -> extra tasks. Feet fixed, centre of mass over the feet's midpoint."""
    q = body.standing()
    feet, mid = feet_tasks(body, q)
    body.put(q)
    cur = {s: body.jaw(s) for s in ("left", "right")}
    frames, grips, worst = [], [], {}
    t_s = 0.0
    for dur, targets, grip in segments:
        n = max(1, int(round(dur * HZ)))
        start = dict(cur)
        for i in range(n):
            u = smooth((i + 1) / n)
            tasks = list(feet) + [dict(kind="com", xy=mid, w=com_w)]
            for side in ("left", "right"):
                tgt = targets.get(side)
                if tgt is None:
                    continue
                p0, R0 = start[side]
                p = p0 + u * (np.asarray(tgt[0]) - p0)
                Rm = mat_of(slerp(quat_of(R0), quat_of(tgt[1]), u))
                tasks.append(hand_task(side, p, Rm, *hand_w))
                cur[side] = (p, Rm)
            if extra is not None:
                tasks += extra(t_s)
            post, w_post = posture(t_s) if posture is not None else (None, 0.05)
            q, errs = body.solve(q, tasks, iters=iters if frames else 200, posture=post, w_posture=w_post)
            for name, e_pos, e_rot in errs:
                w = worst.setdefault(name, [0.0, 0.0])
                w[0], w[1] = max(w[0], e_pos), max(w[1], e_rot)
            frames.append(q)
            grips.append(grip)
            t_s += 1.0 / HZ
        for side in ("left", "right"):
            if targets.get(side) is None:
                body.put(q)
                cur[side] = body.jaw(side)
    return (
        np.array(frames),
        np.array(grips, float),
        {k: [round(100 * v[0], 2), round(v[1], 2)] for k, v in worst.items()},
    )


def kf(side, t0, t1, pos, Rm, what):
    return dict(
        side=side, t0=round(t0, 3), t1=round(t1, 3), pos=np.asarray(pos, float), R=np.asarray(Rm, float), what=what
    )


def on_reference(body, q, keyframes):
    """Each keyframe's target becomes the reference's own jaw pose in the middle of its hold (the criterion is then
    'tracks the commanded hand', whatever the IK could not reach); the nominal target and the IK's miss are kept."""
    for k in keyframes:
        i = int(round(0.5 * (k["t0"] + k["t1"]) * HZ))
        body.put(q[i])
        p, Rm = body.jaw(k["side"])
        k["nominal_pos"] = np.round(k["pos"], 4).tolist()
        k["ik_miss_cm_deg"] = [
            round(100 * float(np.linalg.norm(p - k["pos"])), 2),
            round(float(np.degrees(np.linalg.norm(rot_err(k["R"], Rm)))), 2),
        ]
        k["pos"], k["R"] = p, Rm
    return keyframes


def torso_pitch_deg(body, q):
    body.put(q)
    Rt = body.pose("torso_link")[1]
    return float(np.degrees(np.arctan2(Rt[0, 2], Rt[2, 2])))


def ramp(t, a, b, up_s, down_s):
    """0 outside [a, b]; smoothstep up over up_s after a, down over down_s before b."""
    if t < a or t > b:
        return 0.0
    return smooth(min((t - a) / up_s, (b - t) / down_s, 1.0))


def build_keyframed(body):
    clips, report = [], {}
    q0 = body.standing()
    body.put(q0)
    hang = {s: body.jaw(s) for s in ("left", "right")}
    # C1 floor pick. Flat-footed with the centre of mass over the feet the mars legs bottom out at a 0.55 m pelvis
    # (ankle dorsiflexion limit) and the jaw at ~0.15 m (IK search, 25.4): target 0.30 ahead, 0.15 right, z 0.15,
    # approach straight down, the pelvis pulled down to 0.62 m on the way (a squat, not only a stoop)
    down = jaw_attitude(90)
    tgt = np.array([0.30, -0.15, 0.15])
    seg = [
        (1.0, {}, (0, 0)),
        (2.5, {"right": (tgt + [0, 0, 0.10], down)}, (0, 0)),
        (0.8, {"right": (tgt, down)}, (0, 0)),
        (1.2, {"right": (tgt, down)}, (0, 1)),
        (2.5, {"right": hang["right"]}, (0, 1)),
        (1.0, {"right": hang["right"]}, (0, 1)),
    ]
    t0 = 1.0 + 2.5 + 0.8

    def squat(t):
        u = ramp(t, 1.0, t0 + 1.2 + 2.5, 2.5, 2.0)
        return [dict(kind="height", body="pelvis", z=float(q0[2] - u * (q0[2] - 0.62)), w=20.0 * u + 1e-6)]

    # the per-frame IK from standing stoops (jaw 0.21 m); a static multi-start solve of the hold pose squats, and a
    # posture blend toward it keeps the trajectory in that basin
    feet, mid = feet_tasks(body, q0)
    hold, best = None, np.inf
    for seed_z in (None, 0.62, 0.56):
        seed = q0.copy()
        if seed_z is not None:
            seed, _ = body.solve(
                q0.copy(),
                feet + [dict(kind="com", xy=mid, w=50.0), dict(kind="height", body="pelvis", z=seed_z, w=200.0)],
                iters=400,
            )
        cand, errs = body.solve(
            seed, feet + [dict(kind="com", xy=mid, w=20.0), hand_task("right", tgt, down)], iters=500
        )
        if max(e[1] for e in errs[:2]) < 0.01 and errs[-1][1] < best:
            hold, best = cand, errs[-1][1]

    def guide(t):
        u = ramp(t, 1.0, t0 + 1.2 + 2.5, 3.3, 2.5)
        return (1 - u) * body.default + u * hold[7:], 0.05 + 2.0 * u

    q, g, w = keyframed(body, seg, extra=squat, posture=guide)
    kfs = on_reference(body, q, [kf("right", t0, t0 + 1.2, tgt, down, "jaw at the low point")])
    clips.append(Clip("C1_floor_pick", "C1", "ik keyframes", False, q, g, kfs))
    report["C1_floor_pick"] = dict(
        worst=w, jaw_z=round(float(kfs[0]["pos"][2]), 3), pelvis_z_min=round(float(q[:, 2].min()), 3)
    )
    # C2 high reach: right jaw to 1.40 m, 0.30 ahead, 0.20 right, approach forward and 20 deg up
    up = jaw_attitude(-20)
    tgt = np.array([0.30, -0.20, 1.40])
    seg = [
        (1.0, {}, (0, 0)),
        (2.5, {"right": (tgt - [0.08, 0, 0], up)}, (0, 0)),
        (0.8, {"right": (tgt, up)}, (0, 0)),
        (1.2, {"right": (tgt, up)}, (0, 1)),
        (2.5, {"right": hang["right"]}, (0, 1)),
        (1.0, {"right": hang["right"]}, (0, 1)),
    ]
    q, g, w = keyframed(body, seg)
    t0 = 1.0 + 2.5 + 0.8
    kfs = on_reference(body, q, [kf("right", t0, t0 + 1.2, tgt, up, "jaw at the high point")])
    clips.append(Clip("C2_high_reach", "C2", "ik keyframes", False, q, g, kfs))
    report["C2_high_reach"] = dict(worst=w, jaw_z=round(float(kfs[0]["pos"][2]), 3))
    # C5 deep lean across a table: torso tilt 70 deg asked, the right jaw at 0.62 ahead, 0.70 m high, approach 75 deg
    # down. Flat-footed with the centre of mass over the feet the IK delivers ~56 deg of torso pitch (search, 25.4)
    lean = jaw_attitude(75)
    tgt = np.array([0.62, -0.10, 0.70])
    seg = [
        (1.0, {}, (0, 0)),
        (3.5, {"right": (tgt + [-0.06, 0, 0.06], lean)}, (0, 0)),
        (1.5, {"right": (tgt, lean)}, (0, 0)),
        (1.2, {"right": (tgt, lean)}, (0, 1)),
        (3.0, {"right": hang["right"]}, (0, 1)),
        (1.0, {"right": hang["right"]}, (0, 1)),
    ]
    t0 = 1.0 + 3.5 + 1.5

    def tilt(t):
        u = ramp(t, 1.0, t0 + 1.2 + 2.0, 3.0, 2.0)
        return [dict(kind="tilt", body="torso_link", R=ry(np.radians(70 * u)), w=50.0 * u + 1e-6)]

    q, g, w = keyframed(body, seg, extra=tilt, com_w=20.0, hand_w=(30.0, 3.0))
    kfs = on_reference(body, q, [kf("right", t0, t0 + 1.2, tgt, lean, "jaw across the table, torso leaning")])
    clips.append(Clip("C5_deep_lean", "C5", "ik keyframes", False, q, g, kfs))
    report["C5_deep_lean"] = dict(worst=w, torso_pitch_deg=round(max(torso_pitch_deg(body, v) for v in q), 1))
    # C6 bimanual floor lift: jaws 60 deg down on the rims of a 0.45 m-wide, 0.33 m-high box on the floor, then lifted
    box_x, half_w = 0.45, 0.225
    rim = {s: (np.array([box_x, sgn * half_w, 0.33]), jaw_attitude(60)) for s, sgn in (("left", 1), ("right", -1))}
    lifted = {s: (np.array([0.32, sgn * half_w, 0.75]), jaw_attitude(60)) for s, sgn in (("left", 1), ("right", -1))}
    above = {s: (rim[s][0] + [-0.05, 0, 0.10], rim[s][1]) for s in rim}
    seg = [
        (1.0, {}, (0, 0)),
        (2.5, above, (0, 0)),
        (0.8, rim, (0, 0)),
        (1.0, rim, (1, 1)),
        (2.5, lifted, (1, 1)),
        (1.2, lifted, (1, 1)),
    ]
    q, g, w = keyframed(body, seg)
    t_rim = 1.0 + 2.5 + 0.8
    t_up = t_rim + 1.0 + 2.5
    kfs = [kf(s, t_rim, t_rim + 1.0, rim[s][0], rim[s][1], "jaw on the box rim") for s in ("left", "right")]
    kfs += [kf(s, t_up, t_up + 1.2, lifted[s][0], lifted[s][1], "box lifted") for s in ("left", "right")]
    kfs = on_reference(body, q, kfs)
    clips.append(Clip("C6_floor_box_lift", "C6", "ik keyframes", False, q, g, kfs))
    report["C6_floor_box_lift"] = dict(worst=w, torso_pitch_deg=round(max(torso_pitch_deg(body, v) for v in q), 1))
    # C7 hand-over: the right jaw brings an object to the midline at 1.0 m, the left jaw meets it 7 cm to its left,
    # both hold; the right lets go and goes back
    give = (np.array([0.36, -0.035, 1.00]), jaw_attitude(0, 30, 90))
    take = (np.array([0.36, 0.035, 1.00]), jaw_attitude(0, -30, -90))
    seg = [
        (1.0, {}, (0, 0)),
        (2.5, {"right": give}, (0, 1)),
        (2.0, {"right": give, "left": take}, (0, 1)),
        (1.2, {"right": give, "left": take}, (1, 1)),
        (0.5, {"right": give, "left": take}, (1, 0)),
        (2.0, {"right": hang["right"], "left": take}, (1, 0)),
        (1.0, {"right": hang["right"], "left": hang["left"]}, (1, 0)),
    ]
    q, g, w = keyframed(body, seg)
    t_meet = 1.0 + 2.5 + 2.0
    kfs = [
        kf("right", t_meet, t_meet + 1.7, *give, "giving jaw at the midline"),
        kf("left", t_meet, t_meet + 1.7, *take, "taking jaw at the midline"),
    ]
    kfs = on_reference(body, q, kfs)
    clips.append(Clip("C7_handover", "C7", "ik keyframes", False, q, g, kfs))
    report["C7_handover"] = dict(worst=w)
    return clips, report


# --- arm overlays on LAFAN1 legs ---------------------------------------------------------------------------------------------
ARM_JOINTS = {
    s: [
        MUJOCO_JOINT_NAMES.index(n)
        for n in MUJOCO_JOINT_NAMES
        if n.startswith(f"{s}_") and ("shoulder" in n or "elbow" in n or "wrist" in n)
    ]
    for s in ("left", "right")
}
WAIST = [MUJOCO_JOINT_NAMES.index(n) for n in ("waist_yaw_joint", "waist_pitch_joint")]


def arm_ik(body, q, side, pos, Rm, iters=40, extra_joints=()):
    """Re-solve one arm (7 joints, plus `extra_joints`) so its jaw centre reaches (pos, Rm); pelvis and legs held."""
    keep = body.mask.copy()
    body.mask[:] = False
    body.mask[body.dadr[ARM_JOINTS[side] + list(extra_joints)]] = True
    try:
        post = q[7:].copy()
        q, errs = body.solve(q, [hand_task(side, pos, Rm)], iters=iters, posture=post, w_posture=0.01)
    finally:
        body.mask[:] = keep
    return q, errs[0]


def carry_overlay(body, q):
    """Both jaws in a box-carry pose fixed in the torso's heading frame (0.30 ahead of the torso link, +-0.21, 0.06
    above it, approach forward and 30 deg down): the upper body leads a turn, so the pelvis frame would put the box
    behind the shoulders."""
    out = []
    prev = None
    worst = [0.0, 0.0]
    for v in q:
        v = v.copy()
        if prev is not None:
            for s in ("left", "right"):
                v[7 + np.array(ARM_JOINTS[s])] = prev[7 + np.array(ARM_JOINTS[s])]
        body.put(v)
        tp, tR = body.pose("torso_link")
        Rh = rz(float(np.arctan2(tR[1, 0], tR[0, 0])))
        for s, sgn in (("left", 1), ("right", -1)):
            v, e = arm_ik(
                body,
                v,
                s,
                tp + Rh @ np.array([0.30, sgn * 0.21, 0.06]),
                Rh @ jaw_attitude(30),
                iters=200 if prev is None else 12,
            )
            worst = [max(worst[0], e[1]), max(worst[1], e[2])]
        out.append(v)
        prev = v
    return np.array(out), worst


def walk_reach(body, q, t_k, hold_s=0.6):
    """C3: the right jaw grabs a fixed world point while the legs keep walking -- reach out over 1.2 s, hold the point
    from t_k for hold_s (the jaw closes at t_k + 0.1), return over 1.2 s; arm plus waist yaw / pitch re-solved."""
    k = int(t_k * HZ)
    pel, Rh = q[k, :3], rz(yaw_of(q[k, 3:7]))
    target = pel + Rh @ np.array([0.52, -0.22, 0.20])
    att = Rh @ jaw_attitude(20)
    body.put(q[max(0, k - int(1.2 * HZ))])
    start = body.jaw("right")
    out, grip, worst = q.copy(), np.zeros((len(q), 2)), [0.0, 0.0]
    a, b, c = k - int(1.2 * HZ), k + int(hold_s * HZ), k + int((hold_s + 1.2) * HZ)
    prev = None
    for i in range(a, min(c, len(q))):
        v = q[i].copy()
        if prev is not None:
            v[7 + np.array(ARM_JOINTS["right"] + WAIST)] = prev[7 + np.array(ARM_JOINTS["right"] + WAIST)]
        body.put(v)
        if i < k:
            u = smooth((i - a) / (k - a))
            p, Rm = start[0] + u * (target - start[0]), mat_of(slerp(quat_of(start[1]), quat_of(att), u))
        elif i < b:
            p, Rm = target, att
        else:
            body.put(q[i])
            back = body.jaw("right")
            u = smooth((i - b) / (c - b))
            p, Rm = target + u * (back[0] - target), mat_of(slerp(quat_of(att), quat_of(back[1]), u))
        v, e = arm_ik(body, v, "right", p, Rm, iters=40 if prev is None else 12, extra_joints=WAIST if i < c else ())
        if k <= i < b:
            worst = [max(worst[0], e[1]), max(worst[1], e[2])]
        out[i] = v
        prev = v
        grip[i, 1] = 1.0 if k + int(0.1 * HZ) <= i < b + int(0.3 * HZ) else 0.0
    # past the return the waist goes back to the take's over 0.5 s
    for j, i in enumerate(range(c, min(c + 25, len(q)))):
        u = smooth((j + 1) / 25)
        out[i, 7 + np.array(WAIST)] = (1 - u) * out[c - 1, 7 + np.array(WAIST)] + u * q[i, 7 + np.array(WAIST)]
    kfs = on_reference(
        body, out, [kf("right", t_k, t_k + hold_s, target, att, "jaw holds a world point while walking")]
    )
    return out, grip, kfs[0], worst


def standing_clip(seconds=12.0):
    """S0: the default standing pose held (the push test g of 25.5 runs on it)."""
    body = Body()
    q = body.standing()
    return Clip(
        "S0_stand", "S0", "default standing pose", True, np.repeat(q[None], int(seconds * HZ), 0), rule="gmpkpe"
    )


# --- the build --------------------------------------------------------------------------------------------------------------
def build(out: Path) -> dict:
    body = Body()
    report = dict(ik_worst_cm_deg={}, lafan={})
    clips, w = build_keyframed(body)
    report["ik_worst_cm_deg"].update(w)
    takes = {p.stem: read_lafan(p.stem) for p in sorted(LAFAN.glob("walk*.csv"))}
    segments = {
        "L0_walk_forward": ("forward", 6.0),
        "C4_side_left": ("side_left", 2.5),
        "C4_backward": ("backward", 3.0),
        "C4_turn": ("turn", 3.0),
    }
    for name, (kind, length) in segments.items():
        best = find_segment(kind, takes, length)
        if best is None:
            report["lafan"][name] = "no window found"
            continue
        score, take, i = best
        q = takes[take]
        a = lead_in(q, i)
        seg = floor_align(body, canonical(q[a : i + int(length * HZ)]))
        seg = hold_ends(seg)
        src = f"lafan1 {take} frames {a}-{i + int(length * HZ)} at 50 Hz"
        report["lafan"][name] = dict(
            take=take, start_s=round(a / HZ, 2), end_s=round((i + length * HZ) / HZ, 2), score=round(float(score), 3)
        )
        clips.append(Clip(name, "L0" if name.startswith("L0") else "C4", src, True, seg, rule="gmpkpe"))
        if name == "C4_side_left":
            # the walk takes have no clean right side-step (best window -0.16 m/s, half the frames): mirror the left
            seg_r = mirror(seg)
            clips.append(Clip("C4_side_right", "C4", src + ", mirrored", True, seg_r, rule="gmpkpe"))
            carried, _ = carry_overlay(body, seg_r)
            clips.append(
                Clip(
                    "C4_side_right_load",
                    "C4",
                    src + ", mirrored + carry arms (IK)",
                    False,
                    carried,
                    np.ones((len(carried), 2)),
                    rule="gmpkpe",
                    scene="walk_mars_load12",
                )
            )
        if name.startswith("C4"):
            carried, worst = carry_overlay(body, seg)
            g = np.ones((len(carried), 2))
            clips.append(
                Clip(
                    f"{name}_load",
                    "C4",
                    src + " + carry arms (IK)",
                    False,
                    carried,
                    g,
                    rule="gmpkpe",
                    scene="walk_mars_load12",
                )
            )
            report["ik_worst_cm_deg"][f"{name}_load"] = [round(100 * worst[0], 2), round(worst[1], 2)]
        if name == "L0_walk_forward":
            t_k = 0.5 + 3.0
            q3, g3, kf3, worst = walk_reach(body, seg, t_k)
            clips.append(
                Clip("C3_walk_reach", "C3", src + " + right arm grabs a world point (IK)", False, q3, g3, [kf3])
            )
            report["ik_worst_cm_deg"]["C3_walk_reach"] = [round(100 * worst[0], 2), round(worst[1], 2)]
    clips.append(standing_clip())
    report["clips"] = [c.save(out, body) for c in clips]
    (out / "clips.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "controller_compare_20261013" / "clips")
    a = ap.parse_args(argv)
    r = build(a.out)
    for c in r["clips"]:
        print(f"{c['name']:24s} {c['item']:3s} {c['duration_s']:5.1f} s  in-dist {c['in_distribution']}  {c['source']}")
    print("IK worst (cm, deg):", json.dumps(r["ik_worst_cm_deg"]))
    print("LAFAN1 windows:", json.dumps(r["lafan"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
