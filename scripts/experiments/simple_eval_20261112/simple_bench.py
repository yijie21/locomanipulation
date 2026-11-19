"""SIMPLE's pick tasks with one command per reach: direct, lookup and the self model (2026-11-19, simple_bench_20261119).

The "truth" column of the comparison. The target's pose is read from the simulator at the start and again at the stance
(a perfect second look), so every method aims at the same exact goal relative to the body and only the commands differ.
The grasp is SIMPLE's stored Dex3 grasp for the target object (pick.stored_grasp). The robot is SIMPLE's G1 with Dex3 hands
under ScaleBFM, told its pelvis by leg odometry (BFM_ROOT=odo), with the Dex3 palm as the hand point (REACH_HAND_OFF): the
body and settings the self model's data was recorded with (simple_selfmodel_20261119, gen_reach.py REACH_GRASP_ROOM).

An episode:
  - settle 1 s at the episode's start (SIMPLE's episodes start after their settle);
  - the open walk to the stance (gen_reach.open_walk), the grasp STAND ahead of the pelvis and SIDE_OFF to its left;
  - three reaches, each as the data recorded it (PRE_S hold, the reach with the servo off, POST_S hold): a waypoint
    CLEAR_OUT outside the table's front edge and CLEAR_UP over its top (room_grasp.py's, and the data's), the pregrasp PRE
    back from the grasp against the palm's normal (the palm link's y; SIMPLE's stored pregrasp is the open hand 1.7 cm
    from the grasp, and reached directly its fingers swept the box over in 26 of 27 trial episodes), and the entry to the
    grasp along that normal at 0.08 m/s, starting halfway between the hand and the pregrasp command, with its IK from the
    reference pose;
  - close the fingers, lift LIFT_M from where the last command's reference ended (LIFT_CROUCH_M from a crouch, the
    pelvis rising LIFT_M), hold 1 s.
One command per reach; nothing measured of the hand corrects it.

Methods (the command for the pregrasp and the entry; the waypoint is always sent as it is):
  direct  the wanted palm position
  lookup  the wanted position minus the mean miss of the K training reaches of the same kind with the nearest commands
          (the self model's own training split), iterated to a fixed point the same way
  model   the self model's fixed point (room_grasp.py's model arm: wrist_invert.fixed_point on the exported ensemble)
  model_pose  (2026-11-19, added after the first results: model left the hand 7-17 deg off) the same model's position and
          wrist fixed point, wrist_invert.refine's pose3g as it is: the position target and the wrist command it picks
  inverse the direct inverse model's command from the state and the wanted position (inverse_model.py, same data),
          one shot
  servo   reference only, not a one-command method: the wanted position with the Motor's palm servo on, which reads the
          true palm while it reaches; a check that the grasp, the lift and the success test work when the palm arrives
By default the waypoint is left out and the pregrasp is the first reach after the walk (chain 0); --clear puts it back,
about 2 s longer. The defaults (no waypoint, the grasp 0.30 m ahead) are the best direct had on 3 of the 30 episodes of
three tasks (2026-11-19 smoke5: 0.40 m and the waypoint were worse for direct); every method runs with the same.

The "perception" column (--column perception): the robot's own look at the start, cached (blind.py look: SAM 3 on the head
camera's image, the target's points from depth, GraspGen-X's grasps on them, grasps_graspgen.py), the grasp chosen as
pick.graspgen_grasp chooses it, the table's edge as blind.py saw it (its rows' "edge"); no second look: after the walk
the robot takes its pelvis from leg odometry (leg_odo.TickFeed, as room_grasp's feedback arm), and a goal goes to the
Motor kept where it lies from that believed pelvis (place.Belief.to_sim), heights as seen. Every method gets the same.

Success is SIMPLE's (episodes.lifted), checked at every motor tick. A row records when it first held, counted from the end
of the settle, so one run gives both the official time limit (LIMIT_S) and HITCH's 45 s.

    cd <repo>; export REACH_HAND_OFF=0.0415,0,0 BFM_ROOT=odo REACHWM_GOAL_REF=1 PYTHONPATH=src
    $PY scripts/experiments/simple_eval_20261112/simple_bench.py XMovePickTeleop --method model --out rows.jsonl
-> one json line per episode appended to --out, and a summary line
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "reachwm_20261030"))
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
import env as E  # noqa: E402
import episodes as EP  # noqa: E402
import gen_reach as GR  # noqa: E402
import pick as P  # noqa: E402

# SIMPLE's max_episode_steps at the evaluation's 50 Hz (papers/simple_protocols_20261119.md, 1.3)
LIMIT_S = {"XMovePickTeleop": 16.0, "XMoveBendPickTeleop": 16.0, "BendPickMP": 8.0, "TabletopGraspMP": 16.0,
           "LocomotionPickBetweenTablesTeleop": 24.0, "HandoverTeleop": 16.0}  # fmt: skip
HITCH_S = 45.0
FILM_DT = 0.04  # s, --film's frames (25 Hz)
PICK_TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
CLEAR_OUT, CLEAR_UP = 0.10, 0.10  # m, room_grasp.CLEAR_OUT; the data's waypoint is 0.10 over the surface
ENTRY_SPEED, ENTRY_CONT = 0.08, 0.5  # the data's REACH_GRASP_SPEED and REACH_GRASP_CONT
LIFT_M = 0.15
LIFT_CROUCH_M = 0.25  # from a crouch (pelvis ~0.56 m) the hand rises ~60 % of the command: 0.15 lifted the box 5-8 cm
GRASPR_ID0 = 1300000  # data_graspr_odo's reset ids: three reaches each, the third the entry
WM = "outputs/reachwm_20261030/invert/wrist_graspr_odo/wm_ref.pt"
LOOKUP_DATA = ",".join(
    f"outputs/reachwm_20261030/data_{k}/all.npz"
    for k in ("ref_odo", "approach_odo_reach", "approach0_odo_reach", "graspr_odo")
)


class Lookup:
    """The calibration-table baseline: the controller's mean miss near a command, from the self model's own training
    reaches. Separate tables for the slow entries and for every other reach, for each hand, standing or crouched. A
    command's miss is the mean of (final palm - command) over the K training reaches nearest in the command (heading
    frame, and the pelvis height when crouched)."""

    def __init__(self, spec: str, k: int, cache: str | None = None):
        self.k = k
        if cache and Path(cache).exists():  # the tables of an earlier build of the same spec
            z = np.load(cache, allow_pickle=True)
            assert str(z["spec"]) == spec, (str(z["spec"]), spec)
            self.tabs = {tuple(key): (z[f"X{i}"], z[f"Y{i}"]) for i, key in enumerate(z["keys"].tolist())}
            self.n = {key: len(v[0]) for key, v in self.tabs.items()}
            return
        import data as DT

        D = DT.load(spec)
        tr = np.asarray(DT.splits(D)["train"])
        r = D["reset"]
        pos = np.zeros(len(r), int)
        for j in range(1, len(r)):
            pos[j] = pos[j - 1] + 1 if r[j] == r[j - 1] else 0
        entry = (r >= GRASPR_ID0) & (r < GRASPR_ID0 + 100000) & (pos == 2)
        u, self.tabs = D["u"], {}
        for e in (False, True):
            for sd in (1.0, -1.0):
                for stand in (True, False):
                    i = tr[(entry[tr] == e) & (u[tr, 0] == sd) & ((u[tr, 10] > 0.5) == stand)]
                    X = np.c_[D["target"][i], np.zeros(len(i)) if stand else u[i, 11]].astype(np.float64)
                    self.tabs[(e, sd, stand)] = (X, (D["pos_f"][i] - D["target"][i]).astype(np.float64))
        self.n = {key: len(v[0]) for key, v in self.tabs.items()}
        if cache:
            keys = list(self.tabs)
            arrs = {f"X{i}": self.tabs[key][0] for i, key in enumerate(keys)}
            arrs.update({f"Y{i}": self.tabs[key][1] for i, key in enumerate(keys)})
            tmp = f"{cache}.{os.getpid()}.npz"
            np.savez(tmp, spec=spec, keys=np.array(keys, dtype=object), **arrs)
            os.replace(tmp, cache)

    def miss(self, t: np.ndarray, sd: float, pz: float, entry: bool) -> np.ndarray:
        X, Y = self.tabs[(entry, sd, pz < 0)]
        x = np.r_[t, 0.0 if pz < 0 else pz]
        i = np.argpartition(((X - x) ** 2).sum(1), self.k)[: self.k]
        return Y[i].mean(0)

    def invert(self, goal: np.ndarray, sd: float, pz: float, entry: bool, iters: int = 6):
        """The command whose predicted palm (command + its miss) is the goal: -> (command, predicted miss in cm)."""
        t, best = goal.copy(), None
        for _ in range(iters + 1):
            pred = t + self.miss(t, sd, pz, entry)
            err = float(np.linalg.norm(pred - goal))
            if best is None or err < best[1]:
                best = (t.copy(), err)
            t = t + (goal - pred)
        return best[0], 100 * best[1]


def contacts(sim, body: int) -> list:
    """Every body in contact with `body` (the robot's links, tables, other objects)."""
    m, d, out = sim.m, sim.d, set()
    for i in range(d.ncon):
        a, b = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
        other = b if a == body else a if b == body else None
        if other is not None:
            out.add(m.body(other).name)
    return sorted(out)


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def episode(ep: dict, method: str, a, models: dict) -> dict:
    import approach_eval as AE
    import plan_invert as PI
    import room_grasp as RGP
    import wrist_invert as WI

    task, side = ep["task"], a.side
    info = EP.build(ep)
    os.environ["REACH_SCENE_DIR"] = info["dir"]
    env = E.Env("open")
    m, sim = env.m, env.sim
    rec = GR.Recorder(env)
    b = sim.m.body(info["target"]).id
    m.reset(info["start"]["xy"], info["start"]["yaw"])
    EP.pull_objects(sim.m, sim.d, info)
    m.hold(1.0)
    z0, t0, first = float(sim.d.xpos[b][2]), m.t, [None]
    tick0 = m.tick
    look = dict(at=None, phase="walk", knocked=None, touched=None)  # the target at the second look; when it first moved
    # 1 cm; when a robot link first touched it
    robot = sim.m.body_rootid == sim.m.body_rootid[sim.m.body("pelvis").id]

    def tick():  # SIMPLE's success check at every motor tick: when it first holds
        ok = tick0()
        if first[0] is None and EP.lifted(task, z0, float(sim.d.xpos[b][2])):
            first[0] = float(m.t - t0)
        if look["touched"] is None and look["at"] is not None:
            links = [n for n in contacts(sim, b) if robot[sim.m.body(n).id]]
            if links:
                look["touched"] = dict(t=round(float(m.t - t0), 2), phase=look["phase"], links=links,
                                       hand=(100 * (m.jaw_centre(side)[0] - look["at"])).round(1).tolist())  # fmt: skip
        if look["knocked"] is None and look["at"] is not None and np.linalg.norm(sim.d.xpos[b] - look["at"]) > 0.01:
            look["knocked"] = dict(t=round(float(m.t - t0), 2), phase=look["phase"], touch=contacts(sim, b))
        if trace is not None and look["phase"] in ("pregrasp", "entry", "close"):
            # 2026-11-19 (stance/contact diagnosis): every tick of the reaches and the close, jaw and target in mm (world),
            # the robot links on the target and on a table
            on_t = sorted({sim.m.body(k).name for k in robot_on(tables)})
            trace.append([round(float(m.t - t0), 3), look["phase"], (1000 * m.jaw_centre(side)[0]).round(1).tolist(),
                          (1000 * sim.d.xpos[b]).round(1).tolist(), [n for n in contacts(sim, b) if robot[sim.m.body(n).id]],
                          on_t])  # fmt: skip
        if film is not None and m.t - t0 >= len(film) * FILM_DT - 1e-9:  # the whole state, for isaac_film.py
            film.append((float(m.t - t0), look["phase"], sim.d.qpos.copy()))
        return ok

    def robot_on(bodies: set) -> set:  # robot bodies in contact with any of `bodies`
        out = set()
        for i in range(sim.d.ncon):
            x, y = sim.m.geom_bodyid[sim.d.contact[i].geom1], sim.m.geom_bodyid[sim.d.contact[i].geom2]
            if robot[x] and y in bodies:
                out.add(int(x))
            elif robot[y] and x in bodies:
                out.add(int(y))
        return out

    tables = {sim.m.body(n).id for n in info["tables"]} if a.trace else set()
    trace = [] if a.trace else None
    film = [] if a.film else None

    m.tick = tick
    row = dict(task=task, level=ep["level"], index=ep["index"], method=method, stand=a.stand, side_off=a.side_off,
               clear=a.clear, column=a.column)  # fmt: skip
    p0, Rp0 = env.pelvis()
    yaw0 = E.yaw_of(Rp0)
    R0 = E.rot_z(yaw0)
    bel = dict(pb=p0.copy(), yb=yaw0, pt=p0.copy(), yt=yaw0)  # believed and true pelvis, tied at the start
    if a.column == "truth":
        T_og, _, f_close, f_pre = P.stored_grasp(side, str(info["objects"]["target"]["uid"]))

        def grasp_now():
            return P.T(sim.d.xpos[b], sim.d.xquat[b]) @ T_og

        edge = min(t["pos"][0] - 0.5 * t["size"][0] for t in info["tables"].values())  # the tables lie square to x
    else:
        name = f"{task}_L{ep['level']}_e{ep['index']}"
        cloud = EP.OUT / "cloud_blind"
        found = None
        if (cloud / f"grasp_{name}.npz").exists():
            T_wo = np.eye(4)
            T_wo[:3, 3] = np.load(cloud / f"{name}.npz")["p_o"]  # the frame the grasps were worked out in
            found = P.graspgen_grasp(name, T_wo, yaw0, cloud="cloud_blind")
        if found is None:
            return dict(row, why="no grasp", stages=[], palm_cm=None, palm_deg=None, success_end=False, t_success=None,
                        success_official=False, success_45=False, fell=None, knocked=None, touched=None)  # fmt: skip
        T_og, _, f_close, f_pre, row["score"] = found
        Tg_seen = T_wo @ T_og  # the robot's frame at the start, which is the simulator's then
        row["seen_cm"] = round(
            100 * float(np.linalg.norm(T_wo[:3, 3] - sim.d.xpos[b])), 2
        )  # record: cloud median - body
        edge = a.edges.get((str(ep["level"]), int(ep["index"])))

        def grasp_now():  # place.Belief.to_sim: kept where it lies from the believed pelvis
            Rz = E.rot_z(bel["yt"] - bel["yb"])
            M = np.eye(4)
            M[:3, :3] = Rz @ Tg_seen[:3, :3]
            M[:2, 3] = bel["pt"][:2] + (Rz @ np.r_[Tg_seen[:2, 3] - bel["pb"][:2], 0.0])[:2]
            M[2, 3] = Tg_seen[2, 3]
            return M

        import leg_odo as LO

        odo = LO.TickFeed(env)
        odo.begin()
    Tg = grasp_now()
    stance = Tg[:2, 3] - (R0 @ np.r_[a.stand, a.side_off, 0.0])[:2]
    if edge is not None:
        stance[0] = min(stance[0], edge - a.table_gap)
    w = (R0.T @ np.r_[stance - p0[:2], 0.0])[:2]
    lo, hi = GR.APPROACH_BOX
    walk = np.r_[np.clip(w[0], lo[0], 9.0), np.clip(w[1], lo[1], hi[1])]
    legs = max(1, math.ceil(walk[0] / hi[0]))
    if np.linalg.norm(walk) > 0.02:
        for _ in range(legs):
            GR.open_walk(m, float(walk[0]) / legs, float(walk[1]) / legs, 0.0)
            if m.fell_at is not None:
                break
    p1, Rp1 = env.pelvis()
    row.update(walk=walk.round(3).tolist(), walk_err_cm=(100 * ((R0.T @ (p1 - p0))[:2] - walk)).round(2).tolist())
    if a.column == "perception":
        odo.on = False
        dx, dy, dth = odo.odo.pose()
        bel.update(pb=p0 + R0 @ np.r_[dx, dy, 0.0], yb=yaw0 + float(dth), pt=p1, yt=E.yaw_of(Rp1))
        row["odo_cm"] = round(100 * float(np.linalg.norm((bel["pb"] - bel["pt"])[:2])), 2)
        row["odo_deg"] = round(math.degrees(bel["yb"] - bel["yt"]), 2)
    Tg = grasp_now()  # truth: the second look, from the stance; perception: the start's look, by odometry
    o_look = look["at"] = sim.d.xpos[b].copy()
    gw = Tg[:3, 3]
    pw = gw - a.pre * Tg[:3, 1]  # the palm link's y is its normal, towards the object
    top = max(t["top"] for t in info["tables"].values())
    ch = R0.T @ (pw - p1)  # heading frame at the start, from the pelvis at the stance
    ch[0] = min(ch[0], float((R0.T @ np.r_[edge - p1[0], 0.0, 0.0])[0]) - CLEAR_OUT)
    cw = p1 + R0 @ ch
    cw[2] = max(float(pw[2]), top + CLEAR_UP)
    pz = AE.pelvis_for(float(gw[2]))
    row.update(grasp_h=(R0.T @ (gw - p1))[:2].round(3).tolist() + [round(float(gw[2]), 3)], pz=pz,
               pre_dir=(100 * R0.T @ (pw - gw)).round(1).tolist(), obj_from_grasp=(100 * R0.T @ (o_look - gw)).round(1).tolist())  # fmt: skip
    sim.fingers_cmd[side] = f_pre
    last_cmd = [None]
    stages = []

    def moved() -> float:  # cm, the target since the second look
        return round(100 * float(np.linalg.norm(sim.d.xpos[b] - o_look)), 2)

    def reach(stage: str, goal_w: np.ndarray, rot_w: np.ndarray, chain: int, invert: bool, entry: bool = False):
        look["phase"] = stage
        rec.start(side)
        m.hold(GR.PRE_S)
        rec.on = False
        kw = {}
        if entry:  # as gen_reach records the entry: REACH_GRASP_SPEED, _CONT (from the measured palm), _IKREF
            j = m.jaw_centre(side)[0]
            kw = dict(speed=ENTRY_SPEED, start=j + ENTRY_CONT * (last_cmd[0] - j), ik_from_ref=True)
        s = dict(stage=stage)
        target, rot_cmd = goal_w, rot_w
        if invert and method in ("lookup", "model", "model_pose", "inverse"):
            st_ = RGP.state_now(env, rec, side, goal_w, rot_w, chain)
            if method in ("model", "model_pose"):
                plan0 = PI.plan_of
                PI.plan_of = lambda *a_: plan0(*a_, **kw)
                try:
                    if method == "model":
                        evs = WI.fixed_point(models["model"], m, st_, st_["goal"], pz, None)
                        best = min(evs, key=lambda ev: ev["l_pos"])
                    else:
                        best = WI.refine(models["model"], m, st_, st_["goal"], pz)["pose3g"][0]
                        rot_cmd = st_["Rh"] @ WI.cmd_h(st_, best["cmd"])
                        s.update(pred_att=round(float(best["pred_att"]), 2), att_spread=round(float(best["att_spread"]), 2),
                                 cmd_dev_deg=round(float(best["dev"]), 2), picked=f'{best.get("stage")}{best.get("k")}')  # fmt: skip
                finally:
                    PI.plan_of = plan0
                t_h = best["t"]
                s.update(pred_err=round(float(best["pred_err"]), 3), pred_std=round(float(best["pred_std"]), 3))
            elif method == "inverse":
                import inverse_model as IM

                x = IM.features(st_["x0b"][None], st_["goal"][None], st_["rot6"][None], [st_["sd"]], [float(pz < 0)],
                                [0.0 if pz < 0 else pz], [float(entry)])  # fmt: skip
                t_h = st_["goal"] + IM.predict(models["inverse"], x)[0]
            else:
                t_h, pe = models["lookup"].invert(st_["goal"], st_["sd"], pz, entry)
                s.update(pred_err=round(pe, 3))
            target = np.r_[st_["p"][:2] + (st_["Rh"] @ np.r_[t_h[:2], 0.0])[:2], t_h[2]]
            s["shift_cm"] = round(100 * float(np.linalg.norm(target - goal_w)), 3)
        servo = method == "servo"
        out = m.reach(side, np.asarray(target, float), rot_cmd, pelvis_z=None if pz < 0 else float(pz), servo=servo, **kw)
        s["ik_cm"] = out.info.get(
            "ik_err_cm"
        )  # the whole-body IK's own miss of the command (the reference's, not the body's)
        last_cmd[0] = np.asarray(target, float)
        if m.fell_at is None:
            m.hold(GR.POST_S)
        p, Rm = m.jaw_centre(side)
        s.update(err_cm=round(100 * float(np.linalg.norm(p - goal_w)), 3), err_deg=round(angle_deg(rot_w, Rm), 2),
                 miss=(100 * (p - goal_w)).round(2).tolist(), obj_cm=moved())  # fmt: skip
        if trace is not None:  # the wanted and the settled jaw pose, world (mm; rotation matrix rows)
            s.update(goal_mm=(1000 * goal_w).round(1).tolist(), jaw_mm=(1000 * p).round(1).tolist(),
                     rot_want=np.asarray(rot_w).round(4).tolist(), rot_jaw=np.asarray(Rm).round(4).tolist())  # fmt: skip
        stages.append(s)
        return m.fell_at is None

    ok = m.fell_at is None
    if a.clear:
        ok = ok and reach("clear", cw, Tg[:3, :3], 0, False)
    ok = ok and reach("pregrasp", pw, Tg[:3, :3], 1 if a.clear else 0, True)
    ok = ok and reach("entry", gw, Tg[:3, :3], 1, True, entry=True)
    if ok:
        look["phase"] = "close"
        for u in np.linspace(0, 1, int(0.5 / 0.02)):
            sim.fingers_cmd[side] = f_pre + u * (f_close + a.squeeze * (f_close - f_pre) - f_pre)
            m.hold(0.02)
        m.hold(0.5)
        row["obj_cm_closed"] = moved()
        kw = P.cont(m, side)
        h0 = float(m.jaw_centre(side)[0][2])
        look["phase"] = "lift"
        # from a crouch the pelvis rises with the hand, and the hand is sent higher (LIFT_CROUCH_M)
        m.reach(side, kw["start"] + np.r_[0.0, 0.0, LIFT_M if pz < 0 else LIFT_CROUCH_M], kw["start_rot"],
                pelvis_z=None if pz < 0 else float(pz) + LIFT_M, speed=0.1, servo=False, **kw)  # fmt: skip
        m.hold(1.0)
        row["hand_rise_cm"] = round(100 * (float(m.jaw_centre(side)[0][2]) - h0), 2)
    z = float(sim.d.xpos[b][2])
    e = stages[-1] if stages and stages[-1]["stage"] == "entry" else {}
    row.update(stages=stages, palm_cm=e.get("err_cm"), palm_deg=e.get("err_deg"), rise_cm=round(100 * (z - z0), 2),
               success_end=bool(EP.lifted(task, z0, z)), t_success=first[0], fell=m.fell_at, t=round(float(m.t - t0), 2),
               touch=P.touching(sim, b), obj_cm_end=moved(), knocked=look["knocked"], touched=look["touched"])  # fmt: skip
    if trace is not None:
        row["trace"] = trace
    row["success_official"] = first[0] is not None and first[0] <= LIMIT_S[task]
    row["success_45"] = first[0] is not None and first[0] <= HITCH_S
    if film is not None:
        J = range(sim.m.njnt)
        meta = dict({k: row[k] for k in ("task", "level", "index", "method", "column", "success_45", "success_official",
                                         "t_success", "palm_cm", "palm_deg")}, lift=info["lift"], target=info["target"],
                    objects={k: o["body"] for k, o in info["objects"].items()}, dt=FILM_DT)  # fmt: skip
        os.makedirs(a.film, exist_ok=True)
        np.savez_compressed(f"{a.film}/{task}_L{ep['level']}_e{ep['index']}_{a.column}_{method}.npz",
                            t=np.array([f[0] for f in film]), phase=np.array([f[1] for f in film]),
                            qpos=np.array([f[2] for f in film]), jnt=np.array([sim.m.joint(j).name for j in J]),
                            jnt_adr=sim.m.jnt_qposadr.copy(), jnt_type=sim.m.jnt_type.copy(), meta=json.dumps(meta))  # fmt: skip
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=PICK_TASKS)
    ap.add_argument("--method", choices=("direct", "lookup", "model", "model_pose", "inverse", "servo"), required=True)
    ap.add_argument("--clear", action="store_true", help="a waypoint outside the table's edge before the pregrasp")
    ap.add_argument("--episodes", default="0-29", help="indices into the task's 30 episodes, 'a-b' or 'i,j,k'")
    ap.add_argument("--side", default="right")
    ap.add_argument("--stand", type=float, default=0.30, help="m, the grasp ahead of the pelvis at the stance")
    ap.add_argument("--side-off", type=float, default=-0.20, help="m, the grasp to the left of the pelvis")
    ap.add_argument("--table-gap", type=float, default=0.2, help="m, the pelvis stays this far before the table's edge")
    ap.add_argument(
        "--pre", type=float, default=0.08, help="m, the pregrasp back from the grasp against the palm normal"
    )
    ap.add_argument("--squeeze", type=float, default=0.6, help="how far past the closed posture the fingers are driven")
    ap.add_argument("--wm", default=WM, help="the exported self model (method model)")
    ap.add_argument("--lookup-data", default=LOOKUP_DATA)
    ap.add_argument(
        "--lookup-cache", default="outputs/reachwm_20261030/lookup_tables.npz", help="built once, then read"
    )
    ap.add_argument("--inverse", default="outputs/reachwm_20261030/inverse/inverse.pt", help="method inverse's model")
    ap.add_argument("--k", type=int, default=32, help="lookup's neighbours")
    ap.add_argument("--column", choices=("truth", "perception"), default="truth")
    ap.add_argument("--trace", action="store_true", help="every tick of the reaches and the close in the row (contacts)")
    ap.add_argument("--film", help="a folder for each episode's whole MuJoCo state at 25 Hz (isaac_film.py renders it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.edges = {}
    if a.column == "perception":  # the table's edge as blind.py's look saw it
        for r in json.load(open(EP.OUT / f"blind_f1_{a.task}.json")):
            a.edges[(str(r["level"]), int(r["index"]))] = r.get("edge")
    assert os.environ.get("BFM_ROOT") == "odo" and os.environ.get("REACH_HAND_OFF"), "the self model's data settings"
    models = {}
    if a.method in ("model", "model_pose"):
        import torch

        models["model"] = torch.load(a.wm, map_location="cpu", weights_only=False).eval()
        print("self model", a.wm, flush=True)
    elif a.method == "inverse":
        import inverse_model as IM

        models["inverse"] = IM.load(a.inverse)
        print("inverse model", a.inverse, len(models["inverse"]), "members", flush=True)
    elif a.method == "lookup":
        models["lookup"] = Lookup(a.lookup_data, a.k, a.lookup_cache)
        print("lookup tables", models["lookup"].n, flush=True)
    eps = EP.episodes(a.task)
    if "-" in a.episodes:
        i0, i1 = map(int, a.episodes.split("-"))
        idx = range(i0, i1 + 1)
    else:
        idx = [int(i) for i in a.episodes.split(",")]
    for i in idx:
        t_ = time.time()
        row = episode(eps[i], a.method, a, models)
        row.update(ep=i, wall_s=round(time.time() - t_, 1))
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        short = {k: row[k] for k in ("ep", "level", "index", "palm_cm", "success_end", "t_success", "fell", "wall_s")}
        print(short, [(s["stage"], s["err_cm"], s.get("shift_cm")) for s in row["stages"]], flush=True)


if __name__ == "__main__":
    main()
