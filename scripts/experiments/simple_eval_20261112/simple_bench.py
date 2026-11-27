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

The "perception" column (target=perception): the robot's own look at the start, cached (blind.py look: SAM 3 on the head
camera's image, the target's points from depth, GraspGen-X's grasps on them, grasps_graspgen.py), the grasp chosen as
pick.graspgen_grasp chooses it, the table's edge as blind.py saw it (its rows' "edge"); no second look: after the walk
the robot takes its pelvis from leg odometry (leg_odo.TickFeed, as room_grasp's feedback arm), and a goal goes to the
Motor kept where it lies from that believed pelvis (place.Belief.to_sim), heights as seen. Every method gets the same.
The sensor switch picks the look: sensor=rgbd cloud_blind, the replica's MuJoCo images (the 11-19 tables); sensor=stereo
cloud_eyes_final, SIMPLE's Isaac images (blind.py look --eyes, 2026-11-20; docs/handoff/simple-isaac-eyes-20261120.md). An episode without a grasp there
fails ("no grasp"); the table's edge stays the MuJoCo look's (blind_f1) with either.

--avoid with --comp <model dir> (2026-11-23, selfmodel_bench_20261123): the planned path's via points and the pregrasp goal
are moved by the controller's multi-waypoint self model before they are sent (via_comp.Comp.compensate, as gen_reach.py's
REACH_VIA_COMP; REACH_VIA_COMP_MODE, default position only); the entry is sent as it is. A row's "hits": between a reach's
command and the end of its settle, every (stage, robot link, table or other object) pair in penetration and its deepest
penetration (cm); "bel": the believed and the true pelvis at the stance (perception column).

--avoid with --shift x,y,z (cm; 2026-11-24, the entry's control arm D): every via point and the pregrasp goal are moved by
this one vector in the heading frame at the command; no model, the episode is not looked at.

--avoid with --joint (2026-11-24, TODO 38.10.2, selfmodel_jref_20261124): the pregrasp reach does not go through the via
points and our whole-body IK; the right arm follows the planned path's own joint trajectory (the via file's src plan,
plan2.py's Q and names), the rest of the body held at the reference pose of the command (Motor.reach_joints). The entry
is sent as usual. The stage records joint_path_cm, joint_dur_s and joint_gap_deg (the reference's arm against the plan's
start, faded out along the path).

--relook (2026-11-26, simple_relook_20261126; perception column, sensor=rgbd; off by default, and off nothing changes): a
second look at the stance (relook.py): the head camera's depth in the replica, its furniture registered onto the start
look's furniture (look_dump_blind without the target and the robot) with the odometry as the prior, the pelvis belief
replaced by the result; when the target's place in the body frame is more than --relook-step-cm from its planned place,
a step back to the planned stance (--relook-extra-s more walking time, the walk's end reading nothing of the simulator),
leg odometry over it, and another look; at most --relook-steps steps. The grasp, the route's goal and every column's
command then come from that pose. A row's "relook": each look's prior, estimate, the furniture's constraint, the
decision and the step, and (record only) the errors against the simulator; "bel" is the final belief, the odometry's is
"relook.bel_odo". --relook-only ends the episode there (development).

--correct <model D dir> (2026-11-26, selfmodel_simplecmp_20261126; method direct, ScaleBFM; off by default, and off
nothing changes): the pregrasp's and the entry's straight commands go through lmlcmp3's inversion on the SIMPLE-trained
self model D before they are sent (--correct-rule R: candidates shrunk to IK <= 2 cm; S: no shrink; dryR: R recorded,
the goal sent); a joint-path pregrasp (--joint) is sent as it is with D's prediction of it recorded. Each such stage gets
"correct" (candidates, choice, D's prediction, the landing); the record goes to --correct-out. scmp_correct.py.

--joint-dur <json> (2026-11-26, selfmodel_simplecmp2_20261126 V1; with --joint; off by default, and off nothing changes): the
joint-path pregrasp is retimed to the episode's duration in the file ("<task>_L<level>_e<index>": seconds, or another
plan whose reach_joints duration at this command is taken, live): reach_joints' speed is the path's jaw arc length over
that duration (the same arc reach_joints times), so its leg lasts that long; the path itself is unchanged. The stage
records joint_dur_set (the asked duration, the arc, the speed, the other plan). scmp2_jdur.py.

Success is SIMPLE's (episodes.lifted), checked at every motor tick. A row records when it first held, counted from the end
of the settle, so one run gives both the official time limit (LIMIT_S) and HITCH's 45 s.

    cd <repo>; export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 PYTHONPATH=src
    $PY scripts/experiments/simple_eval_20261112/simple_bench.py XMovePickTeleop --method model --out rows.jsonl \
        [--bench-set controller=sonic] [--bench-set target=truth | --bench-set sensor=stereo]
-> one json line per episode appended to --out, and a summary line

Run switches (2026-11-24, TODO 38.11; the bench-options branch's way, locomani.bench): --bench simple and repeatable
--bench-set key=value over configs/bench/simple.yaml, resolved once here and checked against SIMPLE's capability table
(python -m locomani.bench show --bench simple). The profile is the default combination (the user's, 2026-11-24):
target=perception, sensor=rgbd (the replica's MuJoCo look, cloud_blind), self_pose=odometry, controller_root=odometry
(BFM_ROOT=odo), controller=scalebfm; the rows recorded from 11-20 to 11-24 are sensor=stereo (cloud_eyes_final). They replace --column (target), --cloud (sensor: stereo cloud_eyes_final, rgbd cloud_blind) and the
REACH_CONTROLLER / BFM_ROOT variables: the controller and the root go to env.Env explicitly, the two variables are set
from the switches for modules that still read them, and one already set to something else stops the run. Every row
carries the resolved switches under "bench". sonic_vr_frame (2026-11-26, simple_sonic_vrref_20261126) picks the frame of
SONIC's VR hand targets, SonicMotorConfig.vr_frame: robot (the default, every SONIC row before it) or reference.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "reachwm_20261030"))
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
sys.path.insert(0, str(HERE.parent / "selfmodel_via_20261120"))
import env as E  # noqa: E402
import episodes as EP  # noqa: E402
import gen_reach as GR  # noqa: E402
import pick as P  # noqa: E402

from locomani.bench import options as BO  # noqa: E402
from locomani.bench import simple as BS  # noqa: E402

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


_COMP: dict = {}


def comp_model(model: str):
    """--comp: the multi-waypoint self model, loaded once a process. Its arm kinematics come from the robot alone on an open
    floor (REACH_COMP_SCENE): the episodes' scenes hold the same robot."""
    if model not in _COMP:
        import via_comp as VC

        scene = os.environ.get("REACH_COMP_SCENE", "outputs/reachwm_20261030/scene/simple_g1_sonic")
        _COMP[model] = VC.Comp(Path(model), Path(scene))
    return _COMP[model]


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def sample_id(a, task: str, ep: dict, rep: int) -> int:
    """--collect: a sample's id, unique over the four tasks' 30 layouts and the reps."""
    return a.collect_id0 + 1000 * (30 * PICK_TASKS.index(task) + 10 * int(ep["level"]) + int(ep["index"])) + rep


def held_out(task: str, ep: dict, k: int) -> bool:
    """--collect: the layouts left out of the data (the check's unseen ones), a third by crc32."""
    return zlib.crc32(f"{task}:{ep['level']}:{ep['index']}".encode()) % k == 0


def step_walk(m, forward: float, left: float, turn: float, extra_s: float, idle_end: bool = False, face: str = "fixed"):
    """--relook's step: gen_reach.open_walk (the heading kept, then the turn, the stop correction off) with extra_s more
    time before the walk ends, and its end test reading nothing of the simulator (stop_tolerance -1: it ends at the
    profile's end + settle_s + extra_s). A relative command: forward and left in the robot's heading (the world goal is
    formed from the simulator's pelvis as open_walk's is; the motor's walk starts its path there and its pelvis
    estimate was anchored there at the walk's last stand, so only the relative step reaches the controller).
    idle_end (SONIC, --relook-idle-end): end earlier, once the reference has reached its end pose and SONIC's planner
    command has been IDLE for settle_s -- the motor's own rule, its pelvis estimate within stop_radius of the goal.
    face "travel" (--relook-face travel): turn toward the step first when it is off the heading by more than the
    motor's face_travel_deg, walk forward, turn back (Motor.walk_to's face "travel" with face_travel_min_m 0)."""
    import dataclasses

    from locomani.backends.scalebfm.motor import Outcome

    yaw = m.yaw()
    c, s = math.cos(yaw), math.sin(yaw)
    goal = m.sim.d.qpos[:2] + np.array([c * forward - s * left, s * forward + c * left])
    cfg = m.cfg
    m.cfg = dataclasses.replace(cfg, stop_ki=0.0, stop_timeout=cfg.settle_s + extra_s, stop_tolerance=-1.0,
                                face_travel_min_m=0.0 if face == "travel" else cfg.face_travel_min_m)
    run0 = m._run

    def run(name, update, timeout):
        since = [None]

        def upd(t):
            out = update(t)
            if out is not None:
                return out
            r, r_end = m.ref(t), m.ref(t + 1e3)
            done = np.array_equal(r.pos, r_end.pos) and np.array_equal(r.quat, r_end.quat)
            cmd = getattr(m, "cmd", None)
            if done and cmd is not None and cmd.movement.mode == 0:  # sonic_onnx.runtime.IDLE
                since[0] = t if since[0] is None else since[0]
                if t - since[0] >= m.cfg.settle_s - 1e-9:
                    return Outcome(True, "relook step: idle")
            else:
                since[0] = None
            return None

        return run0(name, upd, timeout)

    if idle_end:
        m._run = run
    try:
        return m.walk_to(goal, yaw + turn, face=face)
    finally:
        m.cfg = cfg
        if idle_end:
            del m._run


TUCK = (0.30, 0.30, 0.0, 0.60, 0.0, 0.0, 0.0)  # --relook-tuck: one arm's 7 joints (shoulder pitch back, roll out, elbow)


def tuck_arms(m, seconds: float, hold: float = 0.3):
    """--relook-tuck: bring both arms to TUCK (the hands by the hips, off a table in front) over `seconds`, standing,
    before a step; the walk then carries that arm pose (Motor.walk_to keeps the reference's joints)."""
    from locomani.backends.scalebfm.motor import Outcome, Pose
    from locomani.primitives.g1_joints import ARM_MJ

    start = m.ref(m.t)
    q = start.joints.copy()
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        q[ARM_MJ[side]] = [TUCK[0], sgn * TUCK[1], sgn * TUCK[2], TUCK[3], sgn * TUCK[4], TUCK[5], sgn * TUCK[6]]
    goal = Pose(start.pos.copy(), start.quat.copy(), q, np.zeros(29))
    at, total = m._blend([start, goal], [seconds])
    m._stand(start)
    m.ref = at
    t_0 = m.t
    return m._run("posture", lambda t: Outcome(True) if t - t_0 >= total + hold - 1e-9 else None, total + hold + 1.0)


def relook_stage(a, env, m, odo, bel: dict, r0: dict, p0, yaw0: float, walk, t0: float) -> dict:
    """--relook (relook.py): look, register, step when the stance is off, look again; `bel` ends as the last look's
    pose (pb, yb) tied to the simulator's pelvis then (pt, yt). The decisions read the looks and the odometry only;
    fields ending in _true are the record."""
    import relook as RL
    import see as SEE

    sim = env.sim
    rk = dict(step_cm=a.relook_step_cm, max_steps=a.relook_steps, extra_s=a.relook_extra_s, coarse=not a.relook_no_coarse,
              turn=not a.relook_no_turn, max_step=a.relook_max_step, settle_s=a.relook_settle_s, idle_end=a.relook_idle_end, face=a.relook_face, back=a.relook_back, tuck=a.relook_tuck, tuck_hold=a.relook_tuck_hold,
              looks=[])  # fmt: skip
    f = EP.OUT / a.look / f"{r0['name']}.npz"
    if not f.exists():
        rk["why"] = f"no look dump {a.look}"
        return rk
    t_ = time.time()
    cl = np.load(EP.OUT / a.cloud / f"{r0['name']}.npz")
    body = RL.Body(sim.m, sim.qadr)
    ref = RL.Reference(dict(np.load(f)), cl["mask"], np.asarray(cl["pts"], float), body, r0["joints"], r0["L"])
    kin = SEE.HeadKin(sim.m, sim.qadr)
    R0 = E.rot_z(yaw0)
    plan = np.asarray(p0[:2], float) + (R0 @ np.r_[walk, 0.0])[:2]  # where the walk was to end (the planned stance)
    g = np.asarray(r0["Tg_seen"][:2, 3], float)  # the grasp as the start look saw it
    want = (R0.T @ np.r_[g - plan, 0.0])[:2]  # its place in the planned stance's frame
    rk.update(plan=np.round(plan, 4).tolist(), want_cm=(100 * want).round(2).tolist(), n_ref=len(ref.P))
    for k in range(a.relook_steps + 1):
        pb_, yb_ = np.asarray(bel["pb"], float).copy(), float(bel["yb"])
        kept = {} if os.environ.get("RELOOK_DUMP") else None
        p_c, y_c, rec = RL.localize(env, ref, kin, body, pb_, yb_, not a.relook_no_coarse, kept)
        have = (RL.rot_z(y_c).T @ np.r_[g - p_c[:2], 0.0])[:2]
        off = have - want
        pt_, Rt_ = env.pelvis()
        yt_ = E.yaw_of(Rt_)
        rec.update(t=round(float(m.t - t0), 2), prior=[*np.round(pb_, 4).tolist(), round(yb_, 5)],
                   est=[*p_c.tolist(), y_c], off_cm=(100 * off).round(2).tolist(), off_n_cm=round(100 * float(np.linalg.norm(off)), 2),
                   prior_err_true_cm=round(100 * float(np.linalg.norm(pb_[:2] - pt_[:2])), 2),
                   prior_err_true_deg=round(math.degrees(E.wrap(yb_ - yt_)), 3),
                   est_err_true_cm=(100 * (R0.T @ np.r_[p_c[:2] - pt_[:2], 0.0])[:2]).round(2).tolist(),
                   est_err_true_deg=round(math.degrees(E.wrap(y_c - yt_)), 3),
                   z_err_true_cm=[round(100 * float(pb_[2] - pt_[2]), 2), round(100 * float(rec.get("h_feet", np.nan) - pt_[2]), 2),
                                  round(100 * float(p_c[2] - pt_[2]), 2)],  # odometry, feet, estimate
                   stance_err_true_cm=(100 * (R0.T @ np.r_[pt_[:2] - plan, 0.0])[:2]).round(2).tolist(),
                   yaw_err_true_deg=round(math.degrees(E.wrap(yt_ - yaw0)), 3))  # fmt: skip
        mv = np.linalg.norm(sim.d.xpos - r0["xpos"], axis=1)  # record: has anything but the robot moved since the start
        mv[r0["robot"]] = 0.0
        mv[0] = 0.0
        tgt_mv, mv[r0["target"]] = float(mv[r0["target"]]), 0.0
        rec.update(scene_moved_true_cm=[round(100 * float(mv.max()), 2), sim.m.body(int(mv.argmax())).name],
                   target_moved_true_cm=round(100 * tgt_mv, 2))  # fmt: skip
        if kept is not None:  # development: the look and its inputs, to replay the registration offline
            os.makedirs(os.environ["RELOOK_DUMP"], exist_ok=True)
            np.savez_compressed(f"{os.environ['RELOOK_DUMP']}/{a.controller}_{r0['name']}_k{k}.npz", P_lev=kept["P_lev"], h=kept["h"],
                                pb=pb_, yb=yb_, pt=pt_, yt=yt_, joints0=r0["joints"], L0=r0["L"], name=r0["name"],
                                qadr=sim.qadr)  # fmt: skip
        if k > 0:  # the step before this look: what it was told and what it did (heading frame of the start; record)
            d_true = (R0.T @ np.r_[pt_[:2] - rk["looks"][-1]["_pt"], 0.0])[:2]
            rk["looks"][-1]["step_done_true_cm"] = (100 * d_true).round(2).tolist()
        rec["_pt"] = pt_[:2].copy()
        bel.update(pb=p_c, yb=y_c, pt=pt_, yt=yt_)
        rk["looks"].append(rec)
        if k == a.relook_steps or 100 * float(np.linalg.norm(off)) <= a.relook_step_cm or m.fell_at is not None:
            break
        d = (RL.rot_z(y_c).T @ np.r_[plan - p_c[:2], 0.0])[:2]
        d = d * min(1.0, a.relook_max_step / max(float(np.linalg.norm(d)), 1e-9))
        turn = 0.0 if a.relook_no_turn else float(np.clip(E.wrap(yaw0 - y_c), -math.radians(20), math.radians(20)))
        rec["step"] = [round(float(d[0]), 4), round(float(d[1]), 4), round(turn, 5)]
        rec["step_h_cm"] = (100 * (R0.T @ RL.rot_z(y_c) @ np.r_[d, 0.0])[:2]).round(2).tolist()  # in the start's heading
        t_s = m.t
        if a.relook_settle_s > 0:  # stand still first: the step and its odometry start from rest
            m.hold(a.relook_settle_s)
        odo.begin()  # the odometry from the look on: over the arms' tuck too
        if a.relook_tuck > 0 and not rk.get("tucked"):  # the hands off the table before the first step
            tuck_arms(m, a.relook_tuck, a.relook_tuck_hold)
            rk["tucked"] = True
        k_tr, est0 = len(m.trace), np.asarray(getattr(m.pol, "root_est", None) if getattr(m.pol, "root_est", None) is not None else [np.nan] * 3, float)
        if a.relook_back > 0:  # back off first (SONIC's hands drag on the table when it steps beside it), then the rest
            step_walk(m, -a.relook_back, 0.0, 0.0, a.relook_extra_s, a.relook_idle_end, a.relook_face)
            bx, by, bth = odo.odo.pose()  # what the odometry says the back-off did
            rem = (RL.rot_z(-float(bth)) @ np.r_[d[0] - bx, d[1] - by, 0.0])[:2]
            step_walk(m, float(rem[0]), float(rem[1]), turn - float(bth), a.relook_extra_s, a.relook_idle_end, a.relook_face)
        else:
            step_walk(m, float(d[0]), float(d[1]), turn, a.relook_extra_s, a.relook_idle_end, a.relook_face)
        odo.on = False
        if kept is not None:  # development: the step's ticks (the simulator's pelvis, the motor's reference and command)
            tr = m.trace[k_tr:]
            np.savez_compressed(f"{os.environ['RELOOK_DUMP']}/{a.controller}_{r0['name']}_k{k}_step.npz", est0=est0,
                                robot=np.array([r["robot"] for r in tr]), ref=np.array([r["ref_pelvis"] for r in tr]),
                                t=np.array([r["t"] for r in tr]), mode=np.array([r["mode"] for r in tr]),
                                joints=np.array([r["joints"] for r in tr]), wrists=np.array([r["wrists"] for r in tr]),
                                sonic=np.array([r.get("sonic", np.zeros(1)) for r in tr]), d=d, turn=turn,
                                odo=np.array(odo.odo.pose()))  # fmt: skip
        rec["step_s"] = round(float(m.t - t_s), 2)
        dx, dy, dth = odo.odo.pose()
        bel.update(pb=p_c + RL.rot_z(y_c) @ np.r_[dx, dy, 0.0], yb=y_c + float(dth))
    for r in rk["looks"]:
        r.pop("_pt", None)
    rk.update(steps=sum("step" in r for r in rk["looks"]), wall_s=round(time.time() - t_, 1),
              t_end=round(float(m.t - t0), 2))  # fmt: skip
    return rk


def episode(ep: dict, method: str, a, models: dict, rep: int | None = None) -> dict:
    import approach_eval as AE
    import plan_invert as PI
    import room_grasp as RGP
    import wrist_invert as WI

    task, side = ep["task"], a.side
    sid = sample_id(a, task, ep, rep) if rep is not None else None
    info = EP.build(ep)
    os.environ["REACH_SCENE_DIR"] = info["dir"]
    env = E.Env("open", controller=a.controller, root=a.root, sonic_vr_frame=a.sonic_vr_frame)
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
        if live[0]:
            hit_scan()
        if film is not None and m.t - t0 >= len(film) * FILM_DT - 1e-9:  # the whole state, for isaac_film.py
            ref = None
            if a.film_ref:  # the motor's reference pose (MuJoCo order) and the acting hand's finger command
                r_ = m.ref(m.t)
                fc = sim.fingers_cmd.get(side)
                ref = np.r_[r_.pos, r_.quat, r_.joints, np.full(7, np.nan) if fc is None else fc].astype(np.float32)
            film.append((float(m.t - t0), look["phase"], sim.d.qpos.copy(), ref))
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
    # 2026-11-23: what a reach's arm runs into. Bodies that are neither the robot, the world (the floor) nor the target
    scenery = np.array([k > 0 and not robot[k] and k != b for k in range(sim.m.nbody)])
    hits, live = {}, [False]

    def hit_scan():
        gb = sim.m.geom_bodyid
        for i in range(sim.d.ncon):
            c = sim.d.contact[i]
            if c.dist >= 0:
                continue
            x, y = int(gb[c.geom1]), int(gb[c.geom2])
            if robot[x] and scenery[y]:
                key = (look["phase"], sim.m.body(x).name, sim.m.body(y).name)
            elif robot[y] and scenery[x]:
                key = (look["phase"], sim.m.body(y).name, sim.m.body(x).name)
            else:
                continue
            hits[key] = max(hits.get(key, 0.0), -float(c.dist))

    trace = [] if a.trace else None
    film = [] if a.film else None

    m.tick = tick
    row = dict(task=task, level=ep["level"], index=ep["index"], method=method, stand=a.stand, side_off=a.side_off,
               clear=a.clear, column=a.column, avoid=a.avoid or None, joint=a.joint, bench=a.bench_record)  # fmt: skip
    if sid is not None:
        row["sid"] = sid
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
        cloud = EP.OUT / a.cloud
        row["cloud"] = a.cloud
        found = None
        if (cloud / f"grasp_{name}.npz").exists():  # no grasp (or no look) for the episode: it fails, nothing stands in
            T_wo = np.eye(4)
            T_wo[:3, 3] = np.load(cloud / f"{name}.npz")["p_o"]  # the frame the grasps were worked out in
            found = P.graspgen_grasp(name, T_wo, yaw0, cloud=a.cloud)
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
        if a.relook:  # the readings at the start: the start look's own body is left out of its furniture with them
            import relook as RL

            relook0 = dict(joints=sim.d.qpos[sim.qadr].copy(), L=RL.level_R(RL.gravity(env)), name=name, Tg_seen=Tg_seen,
                           xpos=sim.d.xpos.copy(), robot=robot, target=b)  # the last three: the record's scene check
    Tg = grasp_now()
    stance = Tg[:2, 3] - (R0 @ np.r_[a.stand, a.side_off, 0.0])[:2]
    if rep is not None:  # --collect: the stance moved by up to --jitter ahead and to the side (the sample's own draw)
        jit = np.random.default_rng([a.collect_seed, sid, 7]).uniform(-a.jitter, a.jitter, 2)  # ahead, left (m)
        stance = stance + R0[:2, :2] @ jit
        row["jitter"] = jit.round(4).tolist()
    elif a.stance_jitter > 0:
        # selfmodel_precision_20261124: the stance moved by up to --stance-jitter ahead and aside, the same draw for every
        # arm of an episode (keyed by the task, level, index and --jitter-rep), so the reach varies
        key = zlib.crc32(f"{task}:{ep['level']}:{ep['index']}".encode())
        rng = np.random.default_rng([a.jitter_seed, key, a.jitter_rep, 11])
        jit = rng.uniform(-a.stance_jitter, a.stance_jitter, 2)
        stance = stance + R0[:2, :2] @ jit  # ahead, left (m)
        row.update(jitter=jit.round(4).tolist(), jitter_rep=a.jitter_rep)
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
        row["bel"] = dict(pb=np.round(bel["pb"], 5).tolist(), yb=round(float(bel["yb"]), 6),
                          pt=np.round(bel["pt"], 5).tolist(), yt=round(float(bel["yt"]), 6))  # fmt: skip
        if a.relook:
            look["phase"] = "relook"
            live[0] = True  # record: the robot's contacts with the scenery over the looks and steps (row "hits", phase relook)
            row["relook"] = relook_stage(a, env, m, odo, bel, relook0, p0, yaw0, walk, t0)
            live[0] = False
            row["relook"]["bel_odo"] = row["bel"]
            row["bel"] = dict(pb=np.round(bel["pb"], 5).tolist(), yb=round(float(bel["yb"]), 6),
                              pt=np.round(bel["pt"], 5).tolist(), yt=round(float(bel["yt"]), 6))  # fmt: skip
            p1, Rp1 = env.pelvis()
            if a.relook_only:
                return dict(row, stages=[], palm_cm=None, success_end=False, success_45=False, success_official=False,
                            t_success=None, fell=m.fell_at, t=round(float(m.t - t0), 2), controller=a.controller,
                            hits=[[*k, round(100 * v, 2)] for k, v in sorted(hits.items())])  # fmt: skip
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

    film_extra: dict = {}

    def finish() -> dict:  # the row's outcome and the film
        z = float(sim.d.xpos[b][2])
        e = stages[-1] if stages and stages[-1]["stage"] == "entry" else {}
        row.update(stages=stages, palm_cm=e.get("err_cm"), palm_deg=e.get("err_deg"), rise_cm=round(100 * (z - z0), 2),
                   success_end=bool(EP.lifted(task, z0, z)), t_success=first[0], fell=m.fell_at, t=round(float(m.t - t0), 2),
                   touch=P.touching(sim, b), obj_cm_end=moved(), knocked=look["knocked"], touched=look["touched"],
                   hits=[[*k, round(100 * v, 2)] for k, v in sorted(hits.items())], comp=a.comp or None,
                   controller=a.controller)  # fmt: skip
        if trace is not None:
            row["trace"] = trace
        row["success_official"] = first[0] is not None and first[0] <= LIMIT_S[task]
        row["success_45"] = first[0] is not None and first[0] <= HITCH_S
        if film is not None:
            J = range(sim.m.njnt)
            meta = dict({k: row[k] for k in ("task", "level", "index", "method", "column", "success_45", "success_official",
                                             "t_success", "palm_cm", "palm_deg")}, lift=info["lift"], target=info["target"],
                        objects={k: o["body"] for k, o in info["objects"].items()}, dt=FILM_DT, **({} if sid is None else dict(sid=sid)))  # fmt: skip
            fr = [f for f in film if f[1].startswith("jref")] if a.collect else film  # --collect: the samples' reaches
            more = dict(film_extra, ref=np.array([f[3] for f in fr])) if a.film_ref else film_extra
            os.makedirs(a.film, exist_ok=True)
            tag = "" if rep is None else f"_r{rep}"
            np.savez_compressed(f"{a.film}/{task}_L{ep['level']}_e{ep['index']}_{a.column}_{method}{tag}.npz",
                                t=np.array([f[0] for f in fr]), phase=np.array([f[1] for f in fr]),
                                qpos=np.array([f[2] for f in fr]), jnt=np.array([sim.m.joint(j).name for j in J]),
                                jnt_adr=sim.m.jnt_qposadr.copy(), jnt_type=sim.m.jnt_type.copy(), meta=json.dumps(meta), **more)  # fmt: skip
        return row

    if a.collect:  # 38.10.3's data (jref_collect.py): the body alone from this stance, then the episode ends
        import jref_collect as JC

        s_, extra = JC.sample(a, sid, env, m, sim, rec, side, info, pw, Tg[:3, :3], f_pre, look, t0)
        stages.extend(s_)
        film_extra.update(extra)
        return finish()

    def reach(
        stage: str, goal_w: np.ndarray, rot_w: np.ndarray, chain: int, invert: bool, entry: bool = False, via=None
    ):
        look["phase"] = stage
        rec.start(side)
        m.hold(GR.PRE_S)
        rec.on = False
        kw = {}
        if entry:  # as gen_reach records the entry: REACH_GRASP_SPEED, _CONT (from the measured palm), _IKREF
            j = m.jaw_centre(side)[0]
            kw = dict(speed=ENTRY_SPEED, start=j + ENTRY_CONT * (last_cmd[0] - j), ik_from_ref=True)
        s = dict(stage=stage, S0=np.round(np.asarray(rec.S[-1], float)[:29], 4).tolist())  # the joints at the command
        if a.film_ref:  # 38.10.3's check: the self model's inputs at the command, and when it was sent
            s.update(S3=[np.round(np.asarray(x, float), 5).tolist() for x in rec.S[-3:][::-1]], t_cmd=round(float(m.t - t0), 3),
                     W=np.round(np.asarray(rec.W[-1], float), 5).tolist(), J=np.round(np.asarray(rec.J[-1], float), 5).tolist())  # fmt: skip
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
        if getattr(a, "correct", "") and invert and via is None and stage in a.corr.stages:  # --correct (simplecmp)
            target, s["correct"] = a.corr.command(stage, goal_w, rot_w, pz, kw)
        servo = method == "servo"
        if via is not None:  # a planned path's jaw points and wrist rotations up to the last leg (--avoid)
            kw = dict(kw, via=list(via["p"]), via_rot=list(via["R"]))
            if "speed" in via:  # the legs' speed (curobo_attrib_20261120: a straight line timed as a planned path)
                kw["speed"] = float(via["speed"])
            if a.shift:  # a control: one fixed displacement of the whole path, no model
                dv = E.rot_z(E.yaw_of(env.pelvis()[1])) @ (np.array([float(v) for v in a.shift.split(",")]) / 100)
                kw["via"] = [np.asarray(q, float) + dv for q in via["p"]]
                target = np.asarray(target, float) + dv
                s["fixed_shift_cm"] = [float(v) for v in a.shift.split(",")]
            if a.comp:  # the path's points and goal moved by the multi-waypoint self model (via_comp.py)
                p_c, Rp_c = env.pelvis()
                cp, cR, target, rot_cmd, ci = comp_model(a.comp).compensate(
                    m, rec, side, np.asarray(target, float), rot_cmd, None if pz < 0 else float(pz), via["p"], via["R"],
                    kw.get("speed"), p_c, E.rot_z(E.yaw_of(Rp_c)), GR.hand_q(m, side),
                )  # fmt: skip
                kw.update(via=list(cp), via_rot=list(cR))
                sh = np.linalg.norm(ci["shift"], axis=1)
                cm_ = comp_model(a.comp)
                wr_ = [k for k, n in enumerate(cm_.fk.links) if "wrist" in n]
                # what the model expected of the original command: the palm's end offset (heading frame, cm), and how high
                # the wanted sweep takes the wrist (m; over about 1.10 m the controller does not follow)
                s.update(comp_goal_cm=round(100 * float(sh[-1]), 2), comp_via_cm=round(100 * float(sh[:-1].max()), 2),
                         comp_refused=int(ci["refused"]), comp_pred0=(100 * ci["off0"][-1, cm_.palm]).round(2).tolist(),
                         comp_shift_goal=(100 * ci["shift"][-1]).round(2).tolist(),
                         comp_wrist_top=round(float((ci["ref0"][:, wr_, 2] + cm_.fk.radius[wr_]).max()), 3))  # fmt: skip
        live[0] = True
        if via is not None and a.joint:  # the plan's right-arm joint trajectory itself as the reference (no IK of ours)
            from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

            name = f"{task}_L{ep['level']}_e{ep['index']}"
            pl = np.load(Path(str(via["src"])) / f"plan_{name}.npz")
            jn = [str(n) for n in pl["names"]]
            jsp = kw.get("speed")
            if getattr(a, "joint_dur", ""):  # --joint-dur (simplecmp2 V1): the path retimed to the asked duration
                sys.path.insert(0, str(HERE.parent / "selfmodel_simplecmp2_20261126"))
                import scmp2_jdur as JDUR

                jsp, s["joint_dur_set"] = JDUR.speed_for(m, side, pl["Q"], [MUJOCO_JOINT_NAMES.index(n) for n in jn], jn,
                                                         a.joint_dur, name)  # fmt: skip
            if getattr(a, "correct", ""):  # --correct: D's prediction of the path's end recorded; the path as it is
                s["correct"] = a.corr.joint(stage, goal_w, rot_w, pl["Q"], [MUJOCO_JOINT_NAMES.index(n) for n in jn],
                                            jsp)  # fmt: skip
            out = m.reach_joints(
                side, pl["Q"], [MUJOCO_JOINT_NAMES.index(n) for n in jn], speed=jsp, servo=servo
            )
            s.update(joint_src=str(via["src"]), joint_names=jn, joint_path_cm=out.info.get("path_cm"),
                     joint_dur_s=out.info.get("dur_s"), joint_gap_deg=out.info.get("start_gap_deg"))  # fmt: skip
        else:
            out = m.reach(
                side, np.asarray(target, float), rot_cmd, pelvis_z=None if pz < 0 else float(pz), servo=servo, **kw
            )
        s["ik_cm"] = out.info.get(
            "ik_err_cm"
        )  # the whole-body IK's own miss of the command (the reference's, not the body's)
        last_cmd[0] = np.asarray(target, float)
        if m.fell_at is None:
            m.hold(GR.POST_S)
        live[0] = False
        p, Rm = m.jaw_centre(side)
        s.update(err_cm=round(100 * float(np.linalg.norm(p - goal_w)), 3), err_deg=round(angle_deg(rot_w, Rm), 2),
                 miss=(100 * (p - goal_w)).round(2).tolist(), obj_cm=moved())  # fmt: skip
        if "correct" in s:  # --correct: the landing beside D's prediction (record only)
            a.corr.landed(s["correct"], p, Rm)
        if trace is not None:  # the wanted and the settled jaw pose, world (mm; rotation matrix rows)
            s.update(goal_mm=(1000 * goal_w).round(1).tolist(), jaw_mm=(1000 * p).round(1).tolist(),
                     rot_want=np.asarray(rot_w).round(4).tolist(), rot_jaw=np.asarray(Rm).round(4).tolist())  # fmt: skip
        stages.append(s)
        return m.fell_at is None

    ok = m.fell_at is None
    if a.clear:
        ok = ok and reach("clear", cw, Tg[:3, :3], 0, False)
    via = None
    if (
        a.avoid
    ):  # curobo_avoid_20261120: a path planned around the look's obstacles from this reach's start (plan2.py, via2.py)
        f = Path(a.avoid) / f"via_{task}_L{ep['level']}_e{ep['index']}.npz"
        if f.exists():
            via = dict(np.load(f))
            row["avoid"] = dict(n_via=len(via["p"]), goal_cm=round(100 * float(np.linalg.norm(via["goal"] - pw)), 2))
        else:
            row["avoid"] = dict(n_via=0, why="no plan")
    ok = ok and reach("pregrasp", pw, Tg[:3, :3], 1 if a.clear else 0, True, via=via)
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
    return finish()


def switches(a) -> None:
    """Resolve the run switches once (locomani.bench) and set this bed's own settings from them: a.column, a.cloud,
    a.controller, a.root, a.sonic_vr_frame (None unless SONIC; 2026-11-26), and the record every row carries.
    REACH_CONTROLLER and BFM_ROOT are set from the switches for the modules that still read them; one already set to
    another value is an error, not a silent second source."""
    r = BO.resolve(a.bench or "simple", a.bench_set)
    bed = BS.bed_settings(r.options)
    a.column, a.cloud, a.controller, a.root = bed["column"], bed["cloud"], bed["controller"], bed["root"]
    a.look = bed["look"]  # the cached look's dump (--relook's first look)
    a.sonic_vr_frame = bed["sonic_vr_frame"] if a.controller == "sonic" else None
    a.bench_record = r.record()
    for var, value in (("REACH_CONTROLLER", a.controller), ("BFM_ROOT", a.root)):
        if os.environ.get(var, value) != value:
            raise SystemExit(f"{var}={os.environ[var]} but the switches say {value} ({r.options.signature()}); "
                             f"unset it and use --bench-set")  # fmt: skip
        os.environ[var] = value
    print("switches", r.options.signature(), "explicit", list(r.explicit), "normalized", list(r.normalized), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=PICK_TASKS)
    ap.add_argument("--method", choices=("direct", "lookup", "model", "model_pose", "inverse", "servo"), required=True)
    ap.add_argument("--clear", action="store_true", help="a waypoint outside the table's edge before the pregrasp")
    ap.add_argument("--avoid", default="", help="dir of via_<task>_L<l>_e<i>.npz: the pregrasp reach goes through a planned "
                    "path's points (curobo_avoid_20261120); an episode without one reaches straight")  # fmt: skip
    ap.add_argument("--comp", default="", help="with --avoid: a multi-waypoint self model's dir (via_point.pt); the path's via "
                    "points and goal are moved by it before they are sent")  # fmt: skip
    ap.add_argument("--shift", default="", help="with --avoid: 'x,y,z' cm in the heading frame at the command; every via point "
                    "and the pregrasp goal are moved by it (a control without a model)")  # fmt: skip
    ap.add_argument("--joint", action="store_true", help="with --avoid: the pregrasp reach follows the planned path's own "
                    "right-arm joint trajectory (the via file's plan, Motor.reach_joints), not its via points through our IK")  # fmt: skip
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
    BO.add_arguments(ap)  # --bench simple, --bench-set target=|sensor=|controller=... (the docstring's run switches)
    ap.add_argument(
        "--trace", action="store_true", help="every tick of the reaches and the close in the row (contacts)"
    )
    ap.add_argument("--film", help="a folder for each episode's whole MuJoCo state at 25 Hz (isaac_film.py renders it)")
    ap.add_argument("--film-ref", action="store_true", help="with --film: also the motor's reference pose and the acting "
                    "hand's finger command at every frame, and the self model's inputs at each reach's command")  # fmt: skip
    ap.add_argument("--collect", default="", help="38.10.3's data (jref_collect.py): the planner's socket (jref_plan_server.py). "
                    "Each episode walks to its stance (moved by --jitter), takes --collect-k samples and ends")  # fmt: skip
    ap.add_argument("--reps", default="0", help="--collect: the stances a layout, 'a-b' or 'i,j,k'")
    ap.add_argument("--collect-k", type=int, default=1, help="--collect: samples a stance")
    ap.add_argument("--collect-seed", type=int, default=0)
    ap.add_argument("--collect-id0", type=int, default=0)
    ap.add_argument(
        "--jitter", type=float, default=0.06, help="--collect: m, the stance moved by up to this ahead and aside"
    )
    ap.add_argument("--stance-jitter", type=float, default=0.0, help="not --collect: m, the stance moved by up to this ahead "
                    "and aside, one draw per task, level, index and --jitter-rep (selfmodel_precision_20261124); 0: off")  # fmt: skip
    ap.add_argument("--jitter-rep", type=int, default=0)
    ap.add_argument("--jitter-seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=3, help="--collect: layouts with crc32 % this == 0 are left out")
    ap.add_argument("--skip-done", default="", help="--collect: a rows dir; the task's stances already in any of its files are "
                    "skipped (a stance's run is fixed by its id, so a stopped collection is completed, not redone)")  # fmt: skip
    ap.add_argument("--relook", action="store_true", help="perception column, sensor=rgbd: a second look at the stance, "
                    "the pose from the furniture's registration, a step when the stance is off (relook.py; the docstring)")  # fmt: skip
    ap.add_argument("--relook-step-cm", type=float, default=8.0, help="--relook: step when the target's place in the body "
                    "frame is farther than this from its planned place")  # fmt: skip
    ap.add_argument("--relook-steps", type=int, default=2, help="--relook: at most this many steps")
    ap.add_argument("--relook-extra-s", type=float, default=1.0, help="--relook: s of walking time added to a step")
    ap.add_argument("--relook-max-step", type=float, default=0.4, help="--relook: m, a step's length at most")
    ap.add_argument("--relook-settle-s", type=float, default=0.0, help="--relook: s standing still before a step")
    ap.add_argument("--relook-tuck", type=float, default=0.0,
                    help="--relook: s to bring both arms to TUCK (hands by the hips) before the first step; 0: no")
    ap.add_argument("--relook-tuck-hold", type=float, default=0.3, help="--relook-tuck: s to stand after the tuck")
    ap.add_argument("--relook-back", type=float, default=0.0,
                    help="--relook: m a step first walks straight back, then to the planned stance (by the odometry)")
    ap.add_argument("--relook-face", choices=("fixed", "travel"), default="fixed",
                    help="--relook: a step keeps the heading (fixed) or turns toward the step, walks and turns back")
    ap.add_argument("--relook-idle-end", action="store_true",
                    help="--relook, SONIC: a step ends once SONIC's planner has idled settle_s (else at the timeout)")
    ap.add_argument("--relook-no-turn", action="store_true", help="--relook: a step keeps the heading (no turn back)")
    ap.add_argument("--relook-no-coarse", action="store_true", help="--relook: the ICP from the odometry, no coarse search")
    ap.add_argument("--relook-only", action="store_true", help="--relook: the episode ends after the looks (development)")
    ap.add_argument("--correct", default="", help="(selfmodel_simplecmp_20261126) model D's directory (E3_ensemble_f1): the "
                    "straight reaches' commands through lmlcmp3's inversion on it (scmp_correct.py); off by default")  # fmt: skip
    ap.add_argument("--correct-rule", choices=("R", "S", "dryR"), default="R",
                    help="--correct: R (shrunk to IK <= 2 cm), S (no shrink), dryR (R recorded, the goal sent)")
    ap.add_argument("--correct-stages", default="pregrasp,entry", help="--correct: the straight reaches it changes")
    ap.add_argument("--correct-point", choices=("palm", "p13"), default="palm",
                    help="--correct: the hand point the inversion aims (the palm from D's pose, or D's 13 cm point)")
    ap.add_argument("--correct-out", default="", help="--correct: the record's dir (data/, log/)")
    ap.add_argument("--correct-template", default="", help="--correct: a data3-layout npz D is rebuilt on (shapes only)")
    ap.add_argument("--correct-arm-id", type=int, default=0, help="--correct: the arm's number in the record's reset ids")
    ap.add_argument("--joint-dur", default="", help="--joint: a json {episode name: s}, the joint-path pregrasp retimed to "
                    "that duration (selfmodel_simplecmp2_20261126 V1); empty: off")  # fmt: skip
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    switches(a)
    assert not (a.comp and a.shift), "--comp or --shift, not both"
    assert not (a.collect and a.stance_jitter), "--collect moves the stance by --jitter"
    assert not (a.joint and (a.comp or a.shift or not a.avoid)), "--joint goes with --avoid alone"
    assert not a.joint_dur or a.joint, "--joint-dur goes with --joint"
    assert not a.relook or (a.column == "perception" and a.cloud == "cloud_blind"), "--relook: perception, sensor=rgbd"
    assert not a.relook_only or a.relook, "--relook-only goes with --relook"
    if a.collect:
        assert a.film and a.method == "direct" and not (a.avoid or a.comp or a.shift or a.clear), (
            "--collect: --film, direct"
        )
        a.film_ref = True
    a.edges = {}
    if a.column == "perception":  # the table's edge as blind.py's look saw it
        for r in json.load(open(EP.OUT / f"blind_f1_{a.task}.json")):
            a.edges[(str(r["level"]), int(r["index"]))] = r.get("edge")
    assert a.root == "odo" and os.environ.get("REACH_HAND_OFF"), "the self model's data settings"
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
    a.corr = None
    if a.correct:  # selfmodel_simplecmp_20261126: off, nothing of scmp_correct is imported
        sys.path.insert(0, str(HERE.parent / "selfmodel_simplecmp_20261126"))
        import scmp_correct as CORR

        a.corr = CORR.install(sys.modules[__name__], a)
    eps = EP.episodes(a.task)
    if "-" in a.episodes:
        i0, i1 = map(int, a.episodes.split("-"))
        idx = range(i0, i1 + 1)
    else:
        idx = [int(i) for i in a.episodes.split(",")]
    reps = [None]
    if a.collect:  # stance by stance, every layout once before the next stance (a stopped run covers them evenly)
        r0, r1 = map(int, a.reps.split("-")) if "-" in a.reps else (None, None)
        reps = list(range(r0, r1 + 1)) if r0 is not None else [int(r) for r in a.reps.split(",")]
        idx = [i for i in idx if not held_out(a.task, eps[i], a.holdout)]
        print(
            f"collect: {len(idx)} layouts of {a.task} (held out: crc32 % {a.holdout} == 0), stances {a.reps}",
            flush=True,
        )
    todo = [(r, i) for r in reps for i in idx]
    if a.collect and a.skip_done:
        import glob

        done = {(int(r["rep"]), int(r["ep"])) for f in glob.glob(f"{a.skip_done}/*.jsonl") for r in map(json.loads, open(f))
                if r["task"] == a.task}  # fmt: skip
        todo = [p for p in todo if p not in done]
        print(f"skip-done: {len(done)} stances of {a.task} done in {a.skip_done}, {len(todo)} to run", flush=True)
    for rep, i in todo:
        t_ = time.time()
        row = episode(eps[i], a.method, a, models, rep)
        row.update(
            ep=i, wall_s=round(time.time() - t_, 1), **({} if rep is None else dict(rep=rep, sid=row.get("sid")))
        )
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        short = {k: row[k] for k in ("ep", "level", "index", "palm_cm", "success_end", "t_success", "fell", "wall_s")}
        print(short, [(s["stage"], s.get("err_cm"), s.get("shift_cm") if rep is None else s.get("jaw_err_cm")) for s in row["stages"]],
              flush=True)  # fmt: skip
        # every episode builds a new Env; the last one's objects sit in reference cycles that the cyclic collector
        # rarely reaches, and with SONIC's networks on the GPU each holds ~1 GB of onnxruntime arena (2026-11-24, the
        # 4090: two of four processes died allocating after 16-17 episodes). Collect them now.
        gc.collect()


if __name__ == "__main__":
    main()
