"""The seed skill of the LocoManiLab package: the pick as functions of one parameter dict (moved from seed_pkg_lml/skill_simple.py, TODO 49.4).

The motions are those of simple_eval_20261112/{pick,place,blind}.py: walk to a stance, reach behind the pregrasp, to the
pregrasp, to the grasp, close three fingers, lift. Every speed, wait and offset those scripts fix is a key of K here. `K_ORIG`
reproduces the 11-12 scripts; `defaults(task)` is the seed the escalation experiments start from. The placing and the
navigation flows of LocoManiLab are in flows.py.

What the robot knows of the scene comes through a view: `camera_view.LmlCameraView` is the head camera's. The simulator-side
classes of the old file (TruthView, Belief, body_verts) are fixed modules of locomani now (bench/locomanilab/truth.py, body.py).
The fingers are commanded and read through `env.hand`; the old contact-based closing (K["close"] == "contact", which read the
simulator's contacts) and SIMPLE's own placing flow are not in this package.

A stance override (level 2) is a dict: d (m, added to the stand-off), t (m, a shift to the robot's left along the
table's edge), yaw (deg, added to the heading at the stance); d2, t2, yaw2 the same for the placing stance; here
(true: no walk, the pick is made from where the robot stands).
"""

import numpy as np

from locomani.bench.simple import body as PL

from . import pick as P

SIDE = "right"

# The 11-12 scripts' values. Units: m, m/s, s, rad/s unless a comment says otherwise.
K_ORIG = dict(
    # stance
    stand=0.27,  # the palm's target ahead of the pelvis at the stance
    side_off=-0.22,  # and to the left of the pelvis
    place_side=-0.32,  # the same at the release: further right, the idle left hand clear of the basket
    table_gap=0.2,  # the pelvis stays this far before the table's edge
    # which grasp
    grasp_k=0,  # the k-th best-scored grasp the hand can take
    ahead_cos=0.7,  # the fingers within this cosine of straight ahead of the robot
    roll_lo=-30.0,  # deg, the palm's facing: 0 left, 90 down
    roll_hi=110.0,
    clear=0.06,  # the palm link above the object's foot
    pre=0.08,  # the pregrasp: back from the grasp against the palm's normal
    back=0.06,  # the reach starts this far behind the pregrasp
    # walking
    start_hold=1.0,  # standing before the clock starts (SIMPLE's stabilising phase)
    walk_speed=0.25,
    long_walk_speed=0.5,  # the walk between the tables
    turn_rate=0.8,
    settle_s=0.5,  # standing after a walk before it returns
    walk_hold=0.5,  # and a second wait after that
    stop_timeout=3.0,  # how long the stop servo may run (truth track; odometry runs without it)
    # reaching
    reach1_speed=0.15,  # hanging hand to behind the pregrasp
    pre_speed=0.1,
    grasp_speed=0.08,
    reach_min_s=0.6,  # shortest leg
    reach_settle_s=1.5,  # longest wait after a reach for the palm to come within reach_tolerance
    reach_tolerance=0.01,
    # the hand
    close="feel",  # "feel": finger joints stop following ("contact", the simulator's contacts, is not in this package)
    close_s=1.0,
    close_early=False,  # end the closing as soon as every finger has stopped
    press=0.3,  # rad past where a finger stopped
    grip_hold=0.5,
    lift=0.15,
    lift_speed=0.1,
    lift_hold=1.0,
    # carrying and placing
    carry_x=0.25,  # the palm ahead and to the left of the pelvis while walking with the object
    carry_y=-0.22,
    carry_speed=0.15,
    # a second look at the basket, the object held aside (the camera view sets it): True stops look_from short of the
    # place stance, looks and walks the rest; "stance" walks all the way, looks there and moves only the hand's goals
    look_near=False,
    look_from=0.5,
    aside_x=0.20,
    aside_y=-0.45,
    aside_speed=0.2,
    over_rim=0.02,  # the object's foot above the rim when it arrives over the basket,
    over_extra=0.06,  # plus this
    seat=-0.02,  # under the rim (negative: above) where the foot is let go
    over_speed=0.15,
    release_speed=0.05,
    open_s=0.3,
    open_hold=1.0,
    back_off=0.04,  # off the object against the palm's normal
    back_speed=0.05,
    away_out=0.08,  # then further away
    away_up=0.08,  # and up
    away_speed=0.1,
    monitor_s=1.5,  # standing by after the hand is away
    # Motor.hold(0.02) runs one or two 50 Hz steps (k * 0.02 - k0 * 0.02 >= 0.02 fails by rounding for about nine k in
    # ten), so the 11-12 scripts' finger ramps ran at about half their written rate: closing 1.9 s for close_s 1.0.
    # False keeps that; True steps the ramps exactly one step at a time
    exact_ticks=False,
)
# The same motions without the three waits that buy nothing (plan, stage A-S step 4b): the second wait after a walk, the
# finger ramps at half speed and running on after every finger has stopped, the wait after the lift.
K_CUT = dict(K_ORIG, exact_ticks=True, close_early=True, walk_hold=0.0, lift_hold=0.0)
MOTOR_KEYS = ("walk_speed", "turn_rate", "settle_s", "stop_timeout", "reach_min_s", "reach_settle_s", "reach_tolerance")


def defaults(task: str, base: dict | None = None) -> dict:
    """The parameters a task starts from: `base` (K_ORIG). (`task` is kept for the package's `parameters(task)`: no LocoManiLab
    task has parameters of its own.)"""
    return dict(base or K_ORIG)


def grasp_desc(Tw: np.ndarray, centre, yaw: float) -> np.ndarray:
    """A palm pose as the robot sees it, whatever frame the candidates came in: its place from the middle of the object
    in the heading's axes (3), then the palm's normal and its x axis in the same axes (6)."""
    Rz = PL.rot_z(-yaw)[:3, :3]
    return np.r_[Rz @ (Tw[:3, 3] - centre), Rz @ Tw[:3, 1], Rz @ Tw[:3, 0]]


def choose_grasp(z, T_wo: np.ndarray, yaw: float, support: float, K: dict, high=None):
    """The K["grasp_k"]-th of the palm poses z["T_op"] (in their file's order) that the right hand can take at heading
    `yaw` with the object's foot at height `support` (pick.graspgen_grasp's test), as (grasp, pregrasp in the object's
    frame, fingers closed, fingers open, score)."""
    ahead, left = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    Tw = T_wo @ z["T_op"]
    Rw = Tw[:, :3, :3]
    roll = np.degrees(np.arctan2(-Rw[:, 2, 1], Rw[:, :, 1] @ left))
    ok = (
        (Rw[:, :, 0] @ ahead >= K["ahead_cos"])
        & (roll >= K["roll_lo"])
        & (roll <= K["roll_hi"])
        & (Tw[:, 2, 3] >= support + K["clear"])
    )
    idx = np.flatnonzero(ok)
    if len(idx) <= K["grasp_k"]:
        return None
    i = idx[int(K["grasp_k"])]
    T_og = z["T_op"][i]
    T_op = T_og.copy()
    T_op[:3, 3] -= K["pre"] * T_og[:3, 1]
    return T_og, T_op, z["f_close"], z["f_open"], float(z["score"][i])


def one_step(mo, K: dict) -> None:
    mo.hold(0.01 if K["exact_ticks"] else 0.02)


def close_by_feel(hand, side: str, f_close, f_pre, K: dict, lag: float = 0.4, slow: float = 0.3, log=None):
    """pick.close_by_feel with the closing time and the press from K; K["close_early"] ends it once the fingers stopped."""
    f_close, f_pre = np.asarray(f_close, float), np.asarray(f_pre, float)
    cmd, free, hist, seconds = f_pre.copy(), dict(P.FINGERS), [], K["close_s"]
    for k, u in enumerate(np.linspace(0, 1, int(seconds / 0.02))):
        now = hand.q(side)  # the finger encoders
        hist.append(now)
        if k >= 10 and free:
            moved = np.abs(now - hist[k - 5])
            want = np.abs(f_close - f_pre) / seconds * 0.1 * (1 - np.exp(-(k - 2.5) * 0.02 / lag))
            for f, j in list(free.items()):
                jj = [i for i in j if want[i] > 2e-3]
                if log is not None:
                    log.append((k, f, float(moved[jj].sum() / max(1e-9, want[jj].sum())) if jj else None))
                if jj and moved[jj].sum() < slow * want[jj].sum():
                    for j in free.values():
                        cmd[j] = now[j] + K["press"] * np.sign(f_close[j] - f_pre[j])
                    free.clear()
                    break
        for j in free.values():
            cmd[j] = f_pre[j] + u * (f_close[j] - f_pre[j])
        hand.cmd[side] = cmd.copy()
        yield
        if not free and K["close_early"]:
            return


def pick(env, Tg, Tp, f_close, f_pre) -> None:
    """Behind the pregrasp, the pregrasp, the grasp, close, lift (pick.pick). Tg, Tp: the palm in the simulator's frame."""
    mo, hand, K, row, phase = env.mo, env.hand, env.K, env.row, env.clock.phase
    if K["close"] != "feel":
        raise ValueError(
            f'K["close"] = {K["close"]!r}: only "feel" is in this package (the contact closing read the simulator)'
        )
    step = Tp[:3, 3] - Tg[:3, 3]
    hand.cmd[SIDE] = f_pre
    behind, more = Tp[:3, 3] + K["back"] * step / max(1e-6, np.linalg.norm(step)), {}
    if K["back"] > 0:
        with phase("reach_behind"):
            o1 = mo.reach(SIDE, behind, Tp[:3, :3], speed=K["reach1_speed"], **more)
        more = P.cont(mo, SIDE)
    with phase("reach_pregrasp"):  # back = 0: straight to the pregrasp, one reach and one settle less
        o2 = mo.reach(SIDE, Tp[:3, 3], Tp[:3, :3], speed=K["pre_speed"] if K["back"] > 0 else K["reach1_speed"], **more)
        o1 = o1 if K["back"] > 0 else o2
    with phase("reach_grasp"):
        o3 = mo.reach(SIDE, Tg[:3, 3], Tg[:3, :3], speed=K["grasp_speed"], **P.cont(mo, SIDE))
    p, Rm = mo.jaw_centre(SIDE)  # record: the true palm against the grasp
    row["palm_err_cm"] = float(100 * np.linalg.norm(p - Tg[:3, 3]))
    row["palm_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(Tg[:3, :3].T @ Rm) - 1) / 2, -1, 1))))
    row["reach_ok"] = [bool(o.ok) for o in (o1, o2, o3)]
    env.mark("before_close")
    feel = []
    with phase("close"):
        for _ in close_by_feel(hand, SIDE, f_close, f_pre, K, log=feel):
            one_step(mo, K)
    row["stopped"] = sorted({f for _, f, r in feel if r is not None and r < 0.3})
    with phase("grip_hold"):
        mo.hold(K["grip_hold"])
    p, Rm = (getattr(mo, "jaw_source", None) or mo.jaw_centre)(SIDE)
    with phase("lift"):
        mo.reach(SIDE, p + [0, 0, K["lift"]], Rm, speed=K["lift_speed"], **P.cont(mo, SIDE))
    env.mark("after_lift")
    with phase("lift_hold"):
        mo.hold(K["lift_hold"])


def episode(env) -> None:
    """One episode's motions; returns when the script is through (the clock ends it earlier at SIMPLE's success)."""
    K, bel, row, view, phase = env.K, env.bel, env.row, env.view, env.clock.phase
    st = env.stance or {}
    yaw = view.heading()
    ahead, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    tgt = view.target()
    if tgt is None:
        row["why"] = "no target seen"
        return
    yaw_s = yaw + np.radians(st.get("yaw", 0.0))
    found = view.grasp(tgt, yaw_s)
    if found is None:
        row["why"] = "no grasp"
        return
    T_og, T_op, f_close, f_pre, row["score"] = found
    if "lo" in tgt:  # record: the chosen palm pose
        row["grasp_desc"] = grasp_desc(tgt["T_wo"] @ T_og, 0.5 * (np.asarray(tgt["lo"]) + np.asarray(tgt["hi"])), yaw_s).round(3).tolist()  # fmt: skip
    Tg, Tp = tgt["T_wo"] @ T_og, tgt["T_wo"] @ T_op
    stance = (
        Tg[:2, 3] - PL.rot_z(yaw_s)[:2, :2] @ [K["stand"] + st.get("d", 0.0), K["side_off"]] + st.get("t", 0.0) * left
    )
    edge = view.edge(ahead, float(stance @ left))
    if edge is not None:  # no nearer to the table than the gap
        stance = stance - max(0.0, stance @ ahead - (edge - K["table_gap"])) * ahead
    row["edge"] = edge
    here = bool(st.get("here"))  # a stance choice: reach from where the robot stands, no walk to the stance
    if here:
        stance = bel.pelvis()[0][:2].copy()
    row["stance_used"] = dict(d=float((Tg[:2, 3] - stance) @ ahead), t=float((Tg[:2, 3] - stance) @ left), yaw=float(st.get("yaw", 0.0)))  # fmt: skip
    if here:
        row["walk_cm"] = 0.0
    else:
        with phase("walk"):
            row["walk_cm"], _ = bel.walk(stance, yaw_s, face="fixed")
    to_sim = lambda M: np.block([[bel.to_sim_R(M[:3, :3]), bel.to_sim(M[:3, 3])[:, None]], [np.eye(4)[3:]]])  # noqa: E731
    pick(env, to_sim(Tg), to_sim(Tp), f_close, f_pre)
