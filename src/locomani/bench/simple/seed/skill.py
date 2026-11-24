"""The seed skill for SIMPLE's six tasks as functions of one parameter dict (escalation plan, stage A-S, 2026-11-17).

Moved from escalation_20261116/seed_pkg_art/skill_simple.py (TODO 49.2): TruthView and body_verts are now
locomani.perception.truth and Belief locomani.bench.simple.body (fixed infrastructure, not loop-editable); the skill is the
episode logic and its parameters.

The motions are those of simple_eval_20261112/{pick,place,blind}.py: walk to a stance, reach behind the pregrasp, to the
pregrasp, to the grasp, close three fingers, lift; for the two placing tasks carry, walk to the basket, lower the object
onto its rim and let go. Every speed, wait and offset those scripts fix is a key of K here. `K_ORIG` reproduces the
11-12 scripts; `defaults(task)` is the seed the escalation experiments start from.

What the robot knows of the scene comes through a view: `camera_view.CameraView` is the head camera's (target=perception);
perception.truth.TruthView reads the simulator (target=truth, not in 49.2's scope).

A stance override (level 2) is a dict: d (m, added to the stand-off), t (m, a shift to the robot's left along the
table's edge), yaw (deg, added to the heading at the stance); d2, t2, yaw2 the same for the placing stance; here
(true: no walk, the pick is made from where the robot stands).
"""

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.bench.simple import body as PL

from . import pick as P

SIDE = "right"
PLACE = ("LocomotionPickBetweenTablesTeleop", "HandoverTeleop")
TURN = {"LocomotionPickBetweenTablesTeleop": np.pi, "HandoverTeleop": 0.0}  # rad: where the basket's table lies

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
    close="feel",  # "feel": finger joints stop following; "contact": the simulator's contacts
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
PLACING = dict(clear=0.10, roll_lo=-30.0, roll_hi=45.0)  # held from the side, high enough to be lowered to its seat
MOTOR_KEYS = ("walk_speed", "turn_rate", "settle_s", "stop_timeout", "reach_min_s", "reach_settle_s", "reach_tolerance")


# The four articulated teleop tasks (simple_art, 2026-11-23). The robot finds the object the instruction names and its part
# (SAM 3 on the head camera, depth), walks to a stance in front of it, takes the part with the right palm (the clock's phases
# walk, reach, close), moves it along a path about a hinge guessed from the picture or in a line (move), lets go and backs off
# (release). Frames: the "approach frame" has a = horizontal, from the stance toward the part, l = to the robot's left of a,
# u = up. Angles in deg, lengths in m, speeds in m/s. Palm attitude = Rz(art_az) Ry(art_pitch) Rx(-art_roll) in the approach
# frame: fingers along its x (pitch > 0: fingers down), palm normal along its y (roll 0: palm to the left, 90: down, -90: up).
K_ART = dict(
    art_obj="",  # SAM 3 phrases for the object ("|" between alternatives)
    art_part="",  # and for its part
    art_pick="score",  # which part instance: score, high, low, left, right, near, big
    art_floor=0.3,  # lowest SAM 3 score taken (the faucet is small and scores low)
    art_min_px=30,  # fewest valid-depth pixels of a part
    art_relook=1,  # look at the part again from the stance and take that place
    art_face=1,  # a = the object's face normal from its points (0: the line to the part)
    art_at="centre",  # the point of the part to take: centre, near, far, high, low (of its seen points)
    art_off_a=0.0,  # shift of that point in the approach frame
    art_off_l=0.0,
    art_off_u=0.0,
    art_stand=0.40,  # the part this far ahead of the pelvis at the stance
    art_side=-0.22,  # and this far to its left (negative: to the right)
    art_gap=0.30,  # the pelvis stays this far before the object's face
    art_az=0.0,  # palm attitude
    art_pitch=0.0,
    art_roll=0.0,
    art_gx=0.07,  # where the part lies in the palm's frame at the grasp: along the fingers
    art_gy=0.03,  # along the palm's normal
    art_gz=0.0,  # across
    art_app_az=0.0,  # direction of the final approach in the approach frame (az from a toward l, el from the horizontal)
    art_app_el=0.0,
    art_pre=0.10,  # the pregrasp: back from the grasp against the approach
    art_pre_speed=0.12,
    art_grasp_speed=0.06,
    art_grip=1,  # close the fingers on the part (0: push with the open hand)
    art_path="arc",  # arc: about a hinge; line: straight
    art_hinge="fixed",  # fixed: art_ha / art_hl / art_hu from the part; obj: the object's middle (at the part's height); leaf: the object's end opposite the part
    art_ha=0.0,
    art_hl=0.0,
    art_hu=-0.20,
    art_ax="l",  # the hinge's axis: l, u or a
    art_dv_a=-1.0,  # the way the part should start to go (approach frame); the sweep's sign follows
    art_dv_l=0.0,
    art_dv_u=0.0,
    art_sweep=85.0,  # deg
    art_chunk=30.0,  # deg per Motor.reach leg
    art_line=0.30,  # m, the line's length
    art_speed=0.10,  # m/s of the hand along the path
    art_hold=0.3,  # s held at the end of the path
    art_back=0.08,  # m the hand leaves the part against the approach
    art_away=0.10,  # then further, upward
    art_back_speed=0.08,
    art_open_s=0.4,
)
ART_TASK = dict(
    OpenOvenTeleop=dict(
        art_obj="oven",
        art_part="oven handle|handle",
        art_pick="high",
        art_pitch=90.0,
        art_roll=90.0,
        art_gx=0.07,
        art_gy=0.02,
        art_app_el=-90.0,
        art_hu=-0.22,
        art_dv_a=-1.0,
        art_sweep=85.0,
    ),
    CloseDoorTeleop=dict(
        art_obj="door",
        art_part="door handle|handle",
        art_grip=0,
        art_path="arc",
        art_hinge="leaf",
        art_ax="u",
        art_hu=0.0,
        art_dv_a=1.0,
        art_sweep=80.0,
        art_gx=0.10,
        art_gy=0.0,
        art_stand=0.30,
        art_pre=0.0,
        art_grasp_speed=0.35,
        art_speed=0.45,
        art_chunk=40.0,
        walk_speed=0.5,
        reach_min_s=0.3,
        reach_settle_s=0.5,
        reach_tolerance=0.02,
    ),  # the limit is 9 s
    OpenFaucetTeleop=dict(
        art_obj="faucet|sink faucet|spout",
        art_part="faucet|sink faucet|spout",
        art_floor=0.1,
        art_at="high",
        art_hinge="obj",
        art_ax="u",
        art_hu=0.0,
        art_dv_a=0.0,
        art_dv_l=1.0,
        art_sweep=50.0,
        art_gx=0.06,
        art_gy=0.02,
        art_roll=0.0,
        art_stand=0.30,
    ),
    OpenTrashCanTeleop=dict(
        art_obj="trash can|green box|green bin",
        art_part="green lid|lid|trash can lid",
        art_at="near",
        art_roll=-90.0,
        art_gx=0.07,
        art_gy=0.02,
        art_ha=0.30,
        art_hu=0.0,
        art_dv_a=0.0,
        art_dv_u=1.0,
        art_sweep=45.0,
        art_off_u=-0.08,
    ),
)


def defaults(task: str, base: dict | None = None) -> dict:
    """The parameters a task starts from: `base` (K_ORIG) with the placing tasks' grasp limits; for the four articulated
    tasks also the `art_*` keys (K_ART with the task's own values)."""
    art = {**K_ART, **ART_TASK[task]} if task in ART_TASK else {}
    return {**(base or K_ORIG), **(PLACING if task in PLACE else {}), **art}


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


def close_by_feel(sim, side: str, f_close, f_pre, K: dict, lag: float = 0.4, slow: float = 0.3, log=None):
    """pick.close_by_feel with the closing time and the press from K; K["close_early"] ends it once the fingers stopped."""
    q = sim.gq[7 * ("left", "right").index(side) :][:7]
    f_close, f_pre = np.asarray(f_close, float), np.asarray(f_pre, float)
    cmd, free, hist, seconds = f_pre.copy(), dict(P.FINGERS), [], K["close_s"]
    for k, u in enumerate(np.linspace(0, 1, int(seconds / 0.02))):
        now = sim.d.qpos[q].copy()
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
        sim.fingers_cmd[side] = cmd.copy()
        yield
        if not free and K["close_early"]:
            return


def close_on(sim, side: str, body: int, f_close, f_pre, K: dict):
    """pick.close_on (each finger stops past where the simulator says it touches the body) with K's time and press."""
    m, d = sim.m, sim.d
    q = sim.gq[7 * ("left", "right").index(side) :][:7]
    cmd, free = np.array(f_pre, float), dict(P.FINGERS)
    for u in np.linspace(0, 1, int(K["close_s"] / 0.02)):
        touched = set()
        for i in range(d.ncon):
            a, b = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            other = b if a == body else a if b == body else None
            if other is not None:
                touched |= {f for f in free if f"{side}_hand_{f}_" in m.body(other).name}
        for f in touched:
            j = free.pop(f)
            cmd[j] = d.qpos[q[j]] + K["press"] * np.sign(f_close[j] - f_pre[j])
        for j in free.values():
            cmd[j] = f_pre[j] + u * (f_close[j] - f_pre[j])
        sim.fingers_cmd[side] = cmd.copy()
        yield
        if not free and K["close_early"]:
            return


def pick(env, Tg, Tp, f_close, f_pre) -> None:
    """Behind the pregrasp, the pregrasp, the grasp, close, lift (pick.pick). Tg, Tp: the palm in the simulator's frame."""
    mo, sim, K, row, phase = env.mo, env.sim, env.K, env.row, env.clock.phase
    b = sim.m.body(env.info["target"]).id
    step = Tp[:3, 3] - Tg[:3, 3]
    sim.fingers_cmd[SIDE] = f_pre
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
        gen = close_by_feel(sim, SIDE, f_close, f_pre, K, log=feel) if K["close"] == "feel" else close_on(sim, SIDE, b, f_close, f_pre, K)  # fmt: skip
        for _ in gen:
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


def place(env, tgt, Tg, f_close, yaw: float, edge) -> None:
    """Carry, walk to the basket, the object over it, its foot to the rim, let go, the hand away (blind.run's second half).
    Tg: the palm at the grasp in the robot's frame."""
    mo, sim, bel, K, row, view, phase = env.mo, env.sim, env.bel, env.K, env.row, env.view, env.clock.phase
    st = env.stance or {}
    foot = Tg[:3, :3].T @ (np.r_[0.5 * (tgt["lo"] + tgt["hi"])[:2], tgt["top"]] - Tg[:3, 3])  # in the palm's frame
    pb, yb = bel.pelvis()
    palm, Rm = bel.palm(SIDE)
    carry = pb[:2] + PL.rot_z(yb)[:2, :2] @ [K["carry_x"], K["carry_y"]]
    turn, dy = TURN[env.task], np.radians(st.get("yaw2", 0.0))
    far = bool(turn)  # the basket stands on another table
    if far and view.search:  # the other table is out of view: carry the object and turn to look for the basket
        with phase("carry"):
            bel.reach(SIDE, np.r_[carry, palm[2]], Rm, speed=K["carry_speed"])
        with phase("search"):
            found = view.find_basket(yaw)
        if found is None:
            row["why"] = "no basket seen"
            return
        yaw2, (centre, rim, near) = found
        yaw2 += dy
    else:
        yaw2 = yaw + turn + dy
        found = view.basket(yaw2)
        if found is None:
            row["why"] = "no basket seen"
            return
        centre, rim, near = found
    turn = yaw2 - yaw
    ahead2 = np.array([np.cos(yaw2), np.sin(yaw2)])
    left2 = np.array([-ahead2[1], ahead2[0]])
    Rr = PL.rot_z(turn) @ Tg[:3, :3]  # the same hand attitude seen from the robot
    over = (
        np.r_[centre, rim + K["over_rim"] + K["over_extra"]] - Rr @ foot
    )  # the palm, the object upright over the basket
    release = np.r_[centre, rim - K["seat"]] - Rr @ foot  # and with its foot at the rim
    stance2 = release[:2] - PL.rot_z(yaw2)[:2, :2] @ [K["stand"] + st.get("d2", 0.0), K["place_side"]] + st.get("t2", 0.0) * left2  # fmt: skip
    edge2 = view.edge(ahead2, float(stance2 @ left2), "table2" if far else "table") if (far or dy) else edge
    limit = min(near, edge2) if edge2 is not None else near
    stance2 = stance2 - max(0.0, stance2 @ ahead2 - (limit - K["table_gap"])) * ahead2
    row["place"] = dict(stance2=stance2.round(3).tolist(), near=round(near, 3), edge2=edge2, centre=centre.round(3).tolist(), rim=round(rim, 3))  # fmt: skip
    row["stance2_used"] = dict(d=float((release[:2] - stance2) @ ahead2), t=float((release[:2] - stance2) @ left2), yaw=float(st.get("yaw2", 0.0)))  # fmt: skip
    if not (far and view.search):
        with phase("carry"):
            bel.reach(SIDE, np.r_[carry, max(palm[2], over[2])], Rm, speed=K["carry_speed"])
    short = K["look_from"] if (far and K["look_near"] is True) else 0.0  # after a long walk: stop short, look again
    with phase("walk2"):
        row["walk2_cm"], row["walk2_deg"] = bel.walk(stance2 - short * ahead2, yaw2, face="travel" if far else "fixed",
                                                     speed=K["long_walk_speed"] if far else None)  # fmt: skip
    row["carried"] = bool(P.touching(sim, sim.m.body(env.info["target"]).id))  # record
    if K["look_near"]:  # the object held aside, a look at the basket from near
        pb, yb = bel.pelvis()
        aside = pb[:2] + PL.rot_z(yb)[:2, :2] @ [K["aside_x"], K["aside_y"]]
        if np.linalg.norm(bel.palm(SIDE)[0][:2] - aside) > 0.03:  # carried there already: no reach
            with phase("aside"):
                bel.reach(SIDE, np.r_[aside, max(bel.palm(SIDE)[0][2], over[2])], Rr, speed=K["aside_speed"])
        with phase("look"):
            seen = view.basket_near(yaw2)
        if seen is not None:
            shift = np.r_[seen[0] - centre, seen[1] - rim]
            row["look_cm"] = [float(v) for v in 100 * shift]
            over, release, stance2 = over + shift, release + shift, stance2 + shift[:2]
        if short:
            with phase("walk3"):
                row["walk3_cm"], _ = bel.walk(stance2, yaw2, face="fixed")
    with phase("over"):
        bel.reach(SIDE, over, Rr, speed=K["over_speed"])
    with phase("release"):
        bel.reach(SIDE, release, Rr, speed=K["release_speed"])
    row["release_cm"] = float(100 * np.linalg.norm(mo.jaw_centre(SIDE)[0] - bel.to_sim(release)))  # record
    env.mark("before_open")
    with phase("open"):
        for u in np.linspace(0, 1, max(1, int(K["open_s"] / 0.02))):
            sim.fingers_cmd[SIDE] = f_close + u * (PL.WIDE - f_close)
            one_step(mo, K)
        mo.hold(K["open_hold"])
    with phase("away"):
        p, Rm = bel.palm(SIDE)
        bel.reach(
            SIDE, p - K["back_off"] * Rm[:, 1], Rm, speed=K["back_speed"]
        )  # off the object, against the palm's normal
        p, Rm = bel.palm(SIDE)
        bel.reach(SIDE, p - K["away_out"] * Rm[:, 1] + [0, 0, K["away_up"]], Rm, speed=K["away_speed"])
    with phase("monitor"):
        mo.hold(K["monitor_s"])


def go(env, xy, yaw: float) -> float:
    """Walk the pelvis to xy facing yaw (the heading held on the way); -> the odometry's error at the end (cm)."""
    return env.bel.walk(xy, yaw, face="fixed")[0]


def episode(env) -> None:
    """One episode's motions; returns when the script is through (the clock ends it earlier at SIMPLE's success)."""
    if env.task in ART_TASK:
        return art_episode(env)
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
            row["walk_cm"] = go(env, stance, yaw_s)
    to_sim = lambda M: np.block([[bel.to_sim_R(M[:3, :3]), bel.to_sim(M[:3, 3])[:, None]], [np.eye(4)[3:]]])  # noqa: E731
    pick(env, to_sim(Tg), to_sim(Tp), f_close, f_pre)
    if env.task in PLACE:
        place(env, tgt, Tg, f_close, yaw, edge)


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_about(axis, a: float) -> np.ndarray:
    """Rodrigues: the rotation by `a` about the unit vector `axis`."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def art_frame(a2) -> np.ndarray:
    """The approach frame's axes a, l, u as columns, a the horizontal unit vector `a2`."""
    return np.array([[a2[0], -a2[1], 0.0], [a2[1], a2[0], 0.0], [0.0, 0.0, 1.0]])


def art_point(part: dict, F: np.ndarray, K: dict) -> np.ndarray:
    """The point of the part to take: its seen points' median, with the coordinate K["art_at"] names (in the approach frame)
    replaced by that extreme; then shifted by art_off_*."""
    c = part["c"]
    p = np.median(c, axis=0)
    q = (c - p) @ F
    at = K["art_at"]
    if at in ("near", "far"):
        p = p + F[:, 0] * float(np.percentile(q[:, 0], 5 if at == "near" else 95))
    elif at in ("high", "low"):
        p = p + F[:, 2] * float(np.percentile(q[:, 2], 95 if at == "high" else 5))
    return p + F @ np.array([K["art_off_a"], K["art_off_l"], K["art_off_u"]])


def art_hinge(K: dict, F, p, seen: dict) -> np.ndarray:
    """The hinge point: offset from the part, or the object's middle, or the object's end opposite the part (a leaf)."""
    mode, obj = K["art_hinge"], seen["obj"]
    if mode == "fixed" or obj is None:
        return p + F @ np.array([K["art_ha"], K["art_hl"], K["art_hu"]])
    q = (obj - p) @ F  # the object's points in the approach frame, the part at the origin
    if mode == "obj":
        c = np.median(q, axis=0)
        return p + F @ np.array([c[0], c[1], 0.0])
    xy = q[:, :2]  # leaf: the leaf's own horizontal direction (the thin direction of its points is its normal)
    w = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))[1][:, 1]
    s = xy @ w
    far = np.percentile(s, 97) if abs(np.percentile(s, 97)) > abs(np.percentile(s, 3)) else np.percentile(s, 3)
    return p + F @ np.array([far * w[0], far * w[1], 0.0])


def art_episode(env) -> None:
    """The articulated tasks: look, walk, reach, close, move along a path, release (the clock's phases)."""
    K, bel, row, view, phase, mo, sim = env.K, env.bel, env.row, env.view, env.clock.phase, env.mo, env.sim
    seen = view.art()
    if seen["obj"] is None:
        row["why"] = "no target seen"
        return
    if not seen["parts"]:
        row["why"] = "no part seen"
        return
    pb, yaw = seen["pb"], seen["yaw"]
    part = seen["parts"][0]
    p0 = part["p"]
    ahead = np.array([np.cos(yaw), np.sin(yaw)])
    to_part = p0[:2] - pb[:2]
    a2 = to_part / max(1e-6, np.linalg.norm(to_part))
    if K["art_face"]:  # the object's face: the thin direction of its points, turned to face the robot
        xy = seen["obj"][:, :2]
        w, v = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))
        n = v[:, 0] * (1.0 if v[:, 0] @ (pb[:2] - xy.mean(0)) > 0 else -1.0)
        if (-n) @ a2 > np.cos(np.radians(45.0)):
            a2 = -n
    yaw_a = float(np.arctan2(a2[1], a2[0]))
    F = art_frame(a2)
    pt = art_point(part, F, K)
    q = (seen["obj"] - pt) @ F
    face = float(np.percentile(q[np.abs(q[:, 1]) < 0.6][:, 0], 2)) if (np.abs(q[:, 1]) < 0.6).sum() > 50 else 0.0
    stance = pt[:2] - K["art_stand"] * a2 - K["art_side"] * F[:2, 1]
    stance = stance - max(0.0, (stance - pt[:2]) @ a2 - (face - K["art_gap"])) * a2
    row["art_seen"] = dict(part=pt.round(3).tolist(), n_parts=len(seen["parts"]), score=round(part["score"], 3),
                      px=part["n"], a=a2.round(3).tolist(), face=round(face, 3), stance=stance.round(3).tolist())  # fmt: skip
    with phase("walk"):
        row["walk_cm"] = go(env, stance, yaw_a)
    with phase("reach"):
        if K["art_relook"]:
            again = view.art()
            near = [r for r in again["parts"] if np.linalg.norm(r["p"][:2] - part["p"][:2]) < 0.25]
            if near:
                P2 = art_point(near[0], F, K)
                row["art_seen"]["relook_cm"] = (100 * (P2 - pt)).round(1).tolist()
                pt = P2
            else:
                row["art_seen"]["relook_cm"] = None
        Rm = (
            F
            @ PL.rot_z(np.radians(K["art_az"]))
            @ PL.rot_y(np.radians(K["art_pitch"]))
            @ rot_x(-np.radians(K["art_roll"]))
        )
        el, az = np.radians(K["art_app_el"]), np.radians(K["art_app_az"])
        app = F @ np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        palm = pt - Rm @ np.array([K["art_gx"], K["art_gy"], K["art_gz"]])
        f_open = np.zeros(7)
        sim.fingers_cmd[SIDE] = f_open
        if K["art_pre"] > 0:
            o1 = mo.reach(SIDE, bel.to_sim(palm - K["art_pre"] * app), bel.to_sim_R(Rm), speed=K["art_pre_speed"])
            o2 = bel.reach(SIDE, palm, Rm, speed=K["art_grasp_speed"])
        else:
            o1 = o2 = mo.reach(SIDE, bel.to_sim(palm), bel.to_sim_R(Rm), speed=K["art_grasp_speed"])
        got, Rg = bel.palm(SIDE)
        row["palm_err_cm"] = float(100 * np.linalg.norm(got - palm))
        row["reach_ok"] = [bool(o1.ok), bool(o2.ok)]
    env.mark("before_close")
    if K["art_grip"]:
        with phase("close"):
            for _ in close_by_feel(sim, SIDE, R.DEX3_CLOSE[SIDE], f_open, K):
                one_step(mo, K)
    with phase("move"):
        H = art_hinge(K, F, pt, seen)
        ax = {"l": F[:, 1], "u": F[:, 2], "a": F[:, 0]}[K["art_ax"]]
        dv = F @ np.array([K["art_dv_a"], K["art_dv_l"], K["art_dv_u"]])
        palm, Rm = bel.palm(SIDE)
        if K["art_path"] == "line":
            end = palm + K["art_line"] * dv / max(1e-6, np.linalg.norm(dv))
            row["art_seen"]["hinge"] = None
            bel.reach(SIDE, end, Rm, speed=K["art_speed"])
        else:
            sign = 1.0 if np.cross(ax, pt - H) @ dv >= 0 else -1.0
            row["art_seen"]["hinge"] = H.round(3).tolist()
            row["art_seen"]["sign"] = sign
            n = max(1, int(np.ceil(K["art_sweep"] / K["art_chunk"])))
            for k in range(1, n + 1):
                T = rot_about(ax, sign * np.radians(K["art_sweep"]) * k / n)
                bel.reach(SIDE, H + T @ (palm - H), T @ Rm, speed=K["art_speed"])
        mo.hold(K["art_hold"])
    env.mark("after_move")
    with phase("release"):
        for u in np.linspace(0, 1, max(1, int(K["art_open_s"] / 0.02))):
            sim.fingers_cmd[SIDE] = f_open + u * (PL.WIDE - f_open)
            one_step(mo, K)
        p, Rm = bel.palm(SIDE)
        bel.reach(SIDE, p - K["art_back"] * app, Rm, speed=K["art_back_speed"])
        p, Rm = bel.palm(SIDE)
        bel.reach(SIDE, p - K["art_away"] * app + [0, 0, 0.05], Rm, speed=K["art_back_speed"])
