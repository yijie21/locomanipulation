"""selfmodel_jref 38.10.3: one data sample from a SIMPLE stance (simple_bench.py --collect calls `sample` where the
pregrasp reach would start). TODO 38.10.3; the planner is jref_plan_server.py.

The start is SIMPLE's: the robot walks to the stance with the scene as it is, opens the hand to the grasp's pre-shape and
stands gen_reach.PRE_S, the right arm wherever the table's edge leaves it; the state at the command is recorded then. Only
at the command does the scene stop colliding with the robot (every geom that is neither the robot's nor the floor's: the
tables, the objects), so the reach is the body alone, as the self model is meant to be (obstacles are not its input). A
goal and some virtual boxes are drawn in the heading frame at the command:
  goal   half the draws near the episode's own pregrasp (5 cm a coordinate), half anywhere in front of the right
         shoulder between 4 cm over the table top (at least 0.62 m) and 1.12 m: the band a standing arm reaches with the
         waist held (a pregrasp below it, the low tables of the Bend tasks, is never drawn near); the wrist turned from
         the grasp's by up to 35 deg yaw and 25 deg pitch and roll
  slab   in 60 % of draws a board at the table's top (4 cm thick) whose near edge is 4 cm behind to 12 cm ahead of the
         hand, pushed forward until no start sphere is within 3.5 cm of it: paths that rise over an edge
  boxes  0, 1 or 2 (30 / 45 / 25 %) between the hand and the goal, 4-20 cm a side, turned up to 45 deg, none within
         3.5 cm of a start sphere or 8 cm of the goal (the start spheres with the hand opened and with the planner's
         flat hand)
The server plans the right arm (waist at 0, the world in the virtual pelvis's frame); up to ATTEMPTS draws. The plan is
then run by Motor.reach_joints (the rest of the body held, 0.15 m/s), the hand already opened to the grasp's pre-shape
as in the bench, and the robot holds gen_reach.POST_S. --collect-k samples a stance: after each the arm goes back along
its plan reversed and holds 1 s. The draws come from one generator seeded by the stance's id.
"""

from __future__ import annotations

import math
import os
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfmodel_via_20261120"))

ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
ATTEMPTS, CLEAR, GOAL_CLEAR, LOW, HIGH = 20, 0.035, 0.08, 0.62, 1.12
_FK, _CONN, _T0 = {}, {}, {}


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def quat_wxyz(R: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def box_clear(P: np.ndarray, c: np.ndarray, R: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Signed distance of the points P to the box (centre c, rotation R, full sizes dims)."""
    d = np.abs((P - c) @ R) - 0.5 * dims
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)


def fk():
    if "fk" not in _FK:
        from arm_fk import ArmFK

        _FK["fk"] = ArmFK(Path(os.environ.get("REACH_COMP_SCENE", "outputs/scene/simple_g1_sonic")))
    return _FK["fk"]


def conn(path: str):
    if path not in _CONN:
        _CONN[path] = Client(path, "AF_UNIX", authkey=b"jref")
    return _CONN[path]


def virtual_pelvis(sim) -> tuple[np.ndarray, np.ndarray]:
    """The world pose of a pelvis that carries the actual torso with the waist at 0 (the planner's frame)."""
    import mujoco

    m, d = sim.m, sim.d
    t = m.body("torso_link").id
    if "pt0" not in _T0:  # the torso in the pelvis's frame with every joint at 0
        d0 = mujoco.MjData(m)
        d0.qpos[:] = 0
        d0.qpos[3] = 1.0
        mujoco.mj_kinematics(m, d0)
        p = m.body("pelvis").id
        Rp = d0.xmat[p].reshape(3, 3)
        _T0["pt0"] = (Rp.T @ (d0.xpos[t] - d0.xpos[p]), Rp.T @ d0.xmat[t].reshape(3, 3))
    p_t0, R_t0 = _T0["pt0"]
    R_v = d.xmat[t].reshape(3, 3) @ R_t0.T
    return d.xpos[t] - R_v @ p_t0, R_v


def start_spheres(sim, side: str, fingers) -> np.ndarray:
    """The acting arm's 24 sphere centres now (world), by the scene robot's own state."""
    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

    m, d = sim.m, sim.d
    adr = [int(m.jnt_qposadr[m.joint(n).id]) for n in MUJOCO_JOINT_NAMES]
    X, _ = fk().spheres(side, d.qpos[0:3], d.qpos[3:7], d.qpos[adr], "mj", fingers)
    return X


def free_scene(sim) -> int:
    """Every geom that is neither the robot's nor the world body's (the floor) stops colliding; -> how many."""
    m = sim.m
    root = m.body_rootid[m.body("pelvis").id]
    k = 0
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b != 0 and m.body_rootid[b] != root and (m.geom_contype[g] or m.geom_conaffinity[g]):
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
            k += 1
    return k


def draw(rng, s_h, pre_h, top: float, R_grasp_h, X_h, radius) -> dict:
    """A goal (jaw centre and wrist rotation), a slab and boxes, all in the heading frame at the command."""
    low = max(top + 0.04, LOW)
    near = rng.random() < 0.5 and pre_h[2] >= low - 0.03
    if near:
        g = pre_h + rng.normal(0.0, 0.05, 3)
    else:
        g = np.array([rng.uniform(0.20, 0.55), rng.uniform(-0.40, 0.10), rng.uniform(low, max(low + 0.10, HIGH))])
    yaw, pit, rol = np.radians([rng.uniform(-35, 35), rng.uniform(-25, 25), rng.uniform(-25, 25)])
    Rg = rot_z(yaw) @ rot_y(pit) @ rot_x(rol) @ R_grasp_h
    boxes = []
    if rng.random() < 0.6:  # a board at the table's top whose near edge the hand must rise over
        xe = s_h[0] + rng.uniform(-0.04, 0.12)
        dims = np.array([0.8, 1.4, 0.04])
        for _ in range(40):
            c = np.array([xe + 0.4, 0.0, top - 0.02])
            if (box_clear(X_h, c, np.eye(3), dims) - radius).min() >= CLEAR:
                boxes.append(("slab", c, 0.0, dims))
                break
            xe += 0.01
    for _ in range(int(rng.choice(3, p=[0.30, 0.45, 0.25]))):
        for _ in range(20):
            c = s_h + rng.uniform(0.25, 0.75) * (g - s_h) + rng.normal(0.0, 0.04, 3)
            dims, a = rng.uniform(0.04, 0.20, 3), math.radians(rng.uniform(-45, 45))
            Rb = rot_z(a)
            if (box_clear(X_h, c, Rb, dims) - radius).min() >= CLEAR and box_clear(g[None], c, Rb, dims)[
                0
            ] >= GOAL_CLEAR:
                boxes.append(("box", c, a, dims))
                break
    return dict(goal=g, R=Rg, boxes=boxes, near=bool(near))


def sample(
    a, sid: int, env, m, sim, rec, side: str, info: dict, pw, R_grasp, f_pre, look: dict, t0: float
) -> tuple[list, dict]:
    """--collect-k samples from this stance, each its own reach (phase jref<k>); between them the arm goes back along the
    last plan reversed and holds 1 s (phase back<k>). -> (stage dicts for the row, extra arrays for the film)."""
    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

    assert side == "right", "the planner plans the right arm"
    rng = np.random.default_rng([a.collect_seed, sid])
    stages, extra = [], {}
    for k in range(a.collect_k):
        s, Q, names = one(a, rng, k, env, m, sim, rec, side, info, pw, R_grasp, f_pre, look, t0)
        stages.append(s)
        if Q is None:
            continue
        extra[f"plan_Q{k}"], extra["plan_names"] = Q.astype(np.float32), np.array(names)
        if m.fell_at is not None or k == a.collect_k - 1:
            break
        look["phase"] = f"back{k}"
        m.reach_joints(side, Q[::-1], [MUJOCO_JOINT_NAMES.index(n) for n in names], speed=None, servo=False)
        m.hold(1.0)
        if m.fell_at is not None:
            break
    return stages, extra


def one(a, rng, k: int, env, m, sim, rec, side: str, info: dict, pw, R_grasp, f_pre, look: dict, t0: float):
    """One sample: -> (stage dict, the plan's joints or None, their names)."""
    import env as E
    import gen_reach as GR

    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

    look["phase"] = f"jref{k}"
    rec.start(side)
    m.hold(GR.PRE_S)
    rec.on = False
    s = dict(stage="jref", k=k, S3=[np.round(np.asarray(x, float), 5).tolist() for x in rec.S[-3:][::-1]],
             W=np.round(np.asarray(rec.W[-1], float), 5).tolist(), J=np.round(np.asarray(rec.J[-1], float), 5).tolist())  # fmt: skip
    s["n_free"] = free_scene(sim)  # at the command (0 after the first sample of a stance)
    p1, Rp1 = env.pelvis()
    Rh = E.rot_z(E.yaw_of(Rp1))
    to_h = lambda X: (np.asarray(X, float) - np.r_[p1[:2], 0.0]) @ Rh  # world -> heading frame (z stays world)  # noqa: E731
    X_h = to_h(np.concatenate([start_spheres(sim, side, f_pre), start_spheres(sim, side, np.zeros(7))]))
    s_h = to_h(m.jaw_centre(side)[0])
    top = max((t["top"] for t in info["tables"].values()), default=0.80)
    pv, Rv = virtual_pelvis(sim)
    qadr = {n: int(sim.m.jnt_qposadr[sim.m.joint(n).id]) for n in ARM}
    q = {n: float(sim.d.qpos[qadr[n]]) for n in ARM}
    OFF = fk().off  # the jaw centre in the wrist's frame (REACH_HAND_OFF)
    tries, rep = [], None
    for k in range(ATTEMPTS):
        dr = draw(rng, s_h, to_h(pw), top, Rh.T @ R_grasp, X_h, np.tile(fk().radius, 2))
        g_w = np.r_[p1[:2], 0.0] + Rh @ dr["goal"]
        Rg_w = Rh @ dr["R"]
        wp = g_w - Rg_w @ OFF
        boxes = []
        for _, c, ang, dims in dr["boxes"]:
            c_w = np.r_[p1[:2], 0.0] + Rh @ c
            boxes.append((Rv.T @ (c_w - pv), quat_wxyz(Rv.T @ Rh @ rot_z(ang)), dims))
        t_ = time.time()
        c_ = conn(a.collect)
        c_.send(dict(q=q, boxes=boxes, goal_p=Rv.T @ (wp - pv), goal_q=quat_wxyz(Rv.T @ Rg_w)))
        rep = c_.recv()
        tries.append(dict(status=rep["status"], t=round(time.time() - t_, 2), n_box=len(boxes), near=dr["near"],
                          slab=any(b[0] == "slab" for b in dr["boxes"])))  # fmt: skip
        if rep["ok"]:
            break
    s.update(tries=tries, top=round(float(top), 4))
    if not rep or not rep["ok"]:
        s["skip"] = "no plan"
        return s, None, None
    # the planner's start wrist against the scene's (virtual pelvis frame), mm
    w = sim.m.body("right_wrist_yaw_link").id
    s["fk_mm"] = round(1000 * float(np.linalg.norm(rep["fk_p"] - Rv.T @ (sim.d.xpos[w] - pv))), 2)
    Q = np.asarray(rep["Q"], float)
    names = [str(n) for n in rep["names"]]
    s.update(goal_h=np.round(dr["goal"], 4).tolist(), goal_R_h=np.round(dr["R"], 5).tolist(),
             boxes=[(kind, np.round(c, 4).tolist(), round(ang, 4), np.round(d_, 4).tolist()) for kind, c, ang, d_ in dr["boxes"]],
             plan_T=len(Q), t_cmd=round(float(m.t - t0), 3))  # fmt: skip
    out = m.reach_joints(side, Q, [MUJOCO_JOINT_NAMES.index(n) for n in names], speed=None, servo=False)
    s.update(jaw_err_cm=out.info.get("jaw_err_cm"), path_cm=out.info.get("path_cm"), dur_s=out.info.get("dur_s"),
             gap_deg=out.info.get("start_gap_deg"), fell=m.fell_at)  # fmt: skip
    if m.fell_at is None:
        m.hold(GR.POST_S)
    return s, Q, names
