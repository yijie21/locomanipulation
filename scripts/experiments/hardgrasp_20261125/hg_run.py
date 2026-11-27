"""selfmodel_hardgrasp_20261125, LocoManiLab part: one look, a stance, an optional walk, a two-stage reach (pre-grasp,
then straight in to the grasp), close, lift, hold -- the direct command (`alone`) against the self model's command
(`selfmodel`), ScaleBFM in the benchmark host, Dex3 right hand. 2026-11-25.

Everything up to the reach commands is the same for both groups of an episode and decided from the look (hg_look:
head RGB-D, SAM 3, GraspGen-X) and the robot's own readings; simulator truth (object poses, the host pelvis after the
look) only scores the episode afterwards. The look's world frame is anchored at the host pelvis reading at the look
(the map's origin is where the robot stands; nothing measured about the world); after it, the robot's pose is leg
odometry (hg_walk.LegOdo: encoders + IMU orientation + foot contact force).

Per episode (frames at 50 Hz):
  stance START_HOLD_S -> look (cached; both groups read it) -> plan: grasp + stance (fixed rules below) ->
  walk (hg_walk.walk_frames, turn-walk-turn) when the stance is >= MOVE_MIN away or the heading off by >= TURN_MIN ->
  reach 1 to the pre-grasp point (PRE_D back along GraspGen-X's approach, the palm normal) -> reach 2 straight in to
  the grasp point -> close the fingers (CLOSE_S) -> lift (LIFT m up, standing up, the same rule for both groups) ->
  hold HOLD_S.  Each reach is one command, planned from the measured state (lml_tasks protocol), never corrected.

The two groups differ only in reaches 1 and 2's commands: `alone` = the goal itself; `selfmodel` = lml_tasks'
self-model fixed point (6 steps, least predicted miss + spread, models_scalebfm_s0 of lml_hostdata_20261118 -- the
model of this host's body, the one lml_tasks.py `selfmodel` loads).

    python hg_run.py look --task T --eps 0:30 --shard K --shards N       # GPU services on 4090b
    python hg_run.py run  --task T --eps 0:30 --methods alone,selfmodel --shard K --shards N
    python hg_run.py stats --out DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "locomanilab_20261117"))
import lml_selfmodel as L  # noqa: E402
import lml_tasks as LT  # noqa: E402

OUT = Path(os.environ.get("HG_OUT", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/hardgrasp_20261125/locomanilab"))
HOSTDATA = LT.HOSTDATA
WM = OUT / "wm_s0.pt"  # pickled from HOSTDATA/models_scalebfm_s0 (lml_tasks.table)

# task: scene id, kind (approach: walk first; near: start in front), lift family (own 30 cm / 1 s judgement) or
# pick-only (our lift >= 10 cm held 1 s), extra SAM 3 prompts
TASKS = {
    "IThorKitchenApproachTableAndLiftApple": dict(scene=13, kind="approach", own=True),
    "IThorKitchenApproachTableAndLiftBowl": dict(scene=13, kind="approach", own=True),
    "IThorKitchenApproachTableAndLiftSaltShaker": dict(scene=13, kind="approach", own=True),
    "IThorLivingRoomApproachTableAndLiftDumbbell": dict(scene=207, kind="approach", own=True),
    "IThorLivingRoomApproachTableAndLiftStatue": dict(scene=209, kind="approach", own=True),
    "IThorLivingRoomApproachTableAndLiftWateringCan": dict(scene=207, kind="approach", own=True),
    "IThorKitchenApproachTableAndStandCannedDrinkUpright": dict(scene=13, kind="approach", own=False, obj="canned_drink",
                                                                prompts=["canned drink", "soda can"]),
    "IThorKitchenTakeStatueFromCabinet": dict(scene=13, kind="near", own=False, obj="statue", prompts=["statue"]),
    "IThorKitchenOpenLowMicrowaveAndLiftApple": dict(scene=13, kind="door", own=True),
    "IThorKitchenOpenDrawerTakeApple": dict(scene=13, kind="door", own=False),
    "IThorBathroomLiftSoapBottleFromCabinet": dict(scene=None, kind="near", own=True),
    "IThorBathroomLiftCandleFromFloor": dict(scene=401, kind="near", own=True, prompts=["candle", "pillar candle"]),
}
METHODS = ("alone", "selfmodel")
DESIGN_EP0 = 1000  # design episodes 1000.. ; formal 0..29 ; OFFICIAL = the unjittered start
OFFICIAL = -1

# -- fixed rules (frozen with the design) -----------------------------------------------------------------------------
J_ROBOT, J_ROBOT_YAW = 0.05, math.radians(10)  # our jitter: robot disc radius, yaw half-range
J_OBJ, J_OBJ_YAW = 0.03, math.radians(30)  # target object disc radius, yaw about z half-range
PRE_D = 0.08  # pre-grasp: this far back along the grasp's approach (palm normal)
ROLL_MAX = math.radians(float(os.environ.get("HG_ROLL_MAX", "40")))  # |roll| the command form cannot express
STAND_AHEAD = (0.42, -0.18)  # approach stance: grasp point at this (x ahead, y left) in the stance heading frame
FOOT_R = 0.32  # stance footprint radius kept clear of furniture points (z 0.05..1.2)
BACKOFF, BACKOFF_MAX = 0.05, 0.40
MOVE_MIN, TURN_MIN = 0.15, math.radians(12)
WALK_FIX, FIX_MIN = 1, 0.10
# high targets (grasp point >= VIA_Z): a first reach to a via point VIA_BACK nearer the body along the stance heading and
# VIA_UP above the pre-grasp, the same direct command in both groups (design: on the 0.93 m kitchen counter the straight
# line from the hand at the hip to the pre-grasp ran through the counter front, 20-37 cm off in both groups)
VIA_Z, VIA_BACK, VIA_UP, VIA_XMIN = 0.80, 0.22, 0.05, 0.15  # at most one correction walk, when the odometry says the stance is >= FIX_MIN away
HAND_CLEAR = 3  # a grasp (or its pre-grasp) with more furniture points than this inside the hand box is dropped
LIFT, CLOSE_S, SETTLE_S, HOLD_S = 0.45, 0.6, 0.3, 2.0  # design: 0.38 left lifts at 29.6-30.2 cm
GRIP = 1.0
# approach tasks look with the torso leaned back (waist pitch, rad; blended in over LEAN_S, the look LOOK_AFTER_S after the
# lean is reached): with the default stance the head camera (pitched 44 deg down, top of the image 14 deg below the
# horizon) does not see a table top beyond ~2.2 m (design 2026-11-25: kitchen counter objects at 2.4 m, z 0.96, unseen)
LOOK_LEAN, LEAN_S, LOOK_AFTER_S = -0.30, 0.5, 0.5
WAIST_PITCH = 14  # MuJoCo joint order index
LIFT_MIN, LIFT_HOLD = 0.10, 50  # pick-only success: object >= 10 cm above its start, held 50 steps (1 s)


def pelvis_rule(z: float) -> float | None:
    """Pelvis height asked for a grasp point at world height z: standing above 0.70 m, else lower (data range)."""
    if z >= 0.70:
        return None
    return float(np.clip(z + 0.36, 0.52, 0.76))


# -- the driver ---------------------------------------------------------------------------------------------------
class HGDriver(LT.TaskDriver):
    def __init__(self, task: str):
        sc = TASKS[task]["scene"]
        if sc is None:
            sys.path[:0] = [str(LT.H.HB), str(L.G1B)]
            import robocasa  # noqa: F401
            from robosuite.environments.base import REGISTERED_ENVS

            sc = int(REGISTERED_ENVS[task].SUPPORTED_SCENE_IDS[0])
        LT.TASKS[task] = dict(scene=sc, max_steps=0, genuine=0)
        self.scene = sc
        super().__init__(task, seed=0)  # one env seed everywhere: the start is a function of the episode alone
        self.env.deterministic_reset = True  # no env-side reset jitter: ours only (J_*)
        self.grip = 0.0
        name = TASKS[task].get("obj") or getattr(self.env, "TARGET_NAME", None)
        if name not in self.env.obj_body_id:
            name = next(iter(self.env.obj_body_id))
        self.obj_name = name
        self.obj_bid = self.env.obj_body_id[name]
        self.obj_tree = set(self.env._body_tree_ids([self.obj_bid]))
        from robocasa.g1_host.dex3_grip import grip_to_joint_targets  # noqa: F401

    def push(self, fw: np.ndarray) -> None:
        from sim2real.utils.dex3 import grip_to_joint_targets

        f = self.to_frame(fw) if self.buf is not None else fw.astype(np.float64)
        self.comp.set_vla_action(dict(ref_root_pos=f[:3].astype(np.float32), ref_root_quat=f[3:7].astype(np.float32),
                                      ref_joint_pos=f[7:].astype(np.float32), left_hand=np.zeros(7, np.float32),
                                      right_hand=grip_to_joint_targets(self.grip, "right").astype(np.float32)))  # fmt: skip

    def reset_hg(self, ep: int) -> dict:
        """The episode's start: the scene's fixed placement, then (ep != OFFICIAL) our jitter drawn from the episode
        index alone: robot xy in a J_ROBOT disc + yaw, the target object xy in a J_OBJ disc + yaw about z, kept on its
        support (the env's own penetration check); 32 attempts, else the unjittered start (recorded)."""
        from robocasa.utils import scene_initial_jitter as SJ

        env = self.env
        self.ep = ep
        self.grip = 0.0
        env.robot_base_pos[:] = self.base0[0]
        env.robot_base_ori[:] = self.base0[1]
        env.rng = np.random.default_rng(1_000_003 * (ep + 2))
        env.reset()
        self.comp = env.robots[0].composite_controller
        self.buf = None
        self.mj.mj_forward(self.m, self.d)
        info = dict(jitter=ep != OFFICIAL, jitter_ok=True, attempts=0)
        if ep != OFFICIAL:
            rj = env._get_robot_free_joint()
            oj = env.objects[self.obj_name].joints[0] if self.obj_name in env.objects else None
            base = {j: env.sim.data.get_joint_qpos(j).copy() for j in (rj, oj) if j is not None}
            rng = np.random.default_rng([ep, 4747])
            ok = False
            for a in range(32):
                for j, q0 in base.items():
                    q = q0.copy()
                    rad, yr = (J_ROBOT, J_ROBOT_YAW) if j == rj else (J_OBJ, J_OBJ_YAW)
                    q[:2] += SJ._sample_disk_offset(rng, rad)
                    dz = rng.uniform(-yr, yr)
                    qz = np.array([math.cos(dz / 2), 0, 0, math.sin(dz / 2)])
                    qq = np.zeros(4)
                    self.mj.mju_mulQuat(qq, qz, q0[3:7])
                    q[3:7] = qq
                    env.sim.data.set_joint_qpos(j, q)
                env.sim.forward()
                info["attempts"] = a + 1
                good = SJ._default_candidate_is_valid(env) and env._initial_jitter_candidate_is_valid()
                if good and oj is not None:
                    good = not SJ._object_has_illegal_penetration(env, self.obj_name)
                if good:
                    ok = True
                    break
            if not ok:
                for j, q0 in base.items():
                    env.sim.data.set_joint_qpos(j, q0)
                env.sim.forward()
            info["jitter_ok"] = ok
            env.robot_base_pos[:2] = env._robot_base_xy()
            env.robot_base_ori[2] = env._robot_base_yaw()
        self.mj.mj_forward(self.m, self.d)
        self.obj_z0 = float(self.d.xpos[self.obj_bid][2])
        return info

    # -- truth, for scoring only --------------------------------------------------------------------------------------
    def obj_pos(self) -> np.ndarray:
        return self.d.xpos[self.obj_bid].copy()

    def obj_contacts(self) -> tuple[bool, bool]:
        """(object touches the robot's right hand, object touches a non-robot body other than itself)."""
        mj, m, d = self.mj, self.m, self.d
        hand, world = False, False
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in self.obj_tree) == (b2 in self.obj_tree):
                continue
            o = b2 if b1 in self.obj_tree else b1
            n = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, o) or ""
            if n.startswith("robot0_right_hand") or n.startswith("robot0_right_wrist"):
                hand = True
            elif not n.startswith("robot0_"):
                world = True
        return hand, world

    def arm_scene_contact(self) -> str | None:
        """A non-target scene body the right hand / wrist / elbow touches now (furniture hit), or None."""
        mj, m, d = self.mj, self.m, self.d
        own = ("robot0_right_hand", "robot0_right_wrist", "robot0_right_elbow")
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            n1 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b1) or ""
            n2 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b2) or ""
            if n1.startswith(own) and not n2.startswith("robot0_") and b2 not in self.obj_tree:
                return n2
            if n2.startswith(own) and not n1.startswith("robot0_") and b1 not in self.obj_tree:
                return n1
        return None

    def hand_point(self, x: dict) -> np.ndarray:
        pw, Rw = x["wr"]["right"]
        return pw + Rw @ L.HAND_OFF


# -- the plan from the look (no truth) --------------------------------------------------------------------------------
def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def plan_from_look(lk: dict, kind: str, p0: np.ndarray, yaw0: float) -> dict:
    """Grasp + stance by the fixed rules. p0 / yaw0: the robot's pelvis at the look (look frame). Returns the plan or
    dict(fail=<reason>)."""
    import hg_look as HL

    if not bool(lk["found"]) or len(lk["obj_pts"]) < 15:
        return dict(fail="perception", why=str(lk["meta"].get("status")))
    if len(lk["grasp_T"]) == 0:
        return dict(fail="perception", why="no grasps")
    obj = np.asarray(lk["obj_pts"], float)
    furn = np.asarray(lk["furn_pts"], float)
    c = np.median(obj, axis=0)
    # stance: approach -> face the object's centroid from the start, the grasp side at STAND_AHEAD; backed off along the
    # line while furniture is inside the footprint. near / door -> stand where the robot is.
    if kind == "approach":
        psi = math.atan2(c[1] - p0[1], c[0] - p0[0])
        Rz = L.rot_z(psi)[:2, :2]
        s = c[:2] - Rz @ np.array(STAND_AHEAD)
        f_low = furn[(furn[:, 2] > 0.05) & (furn[:, 2] < 1.2)]
        back = 0.0
        while back < BACKOFF_MAX - 1e-9 and len(f_low) and np.min(np.linalg.norm(f_low[:, :2] - s, axis=1)) < FOOT_R:
            s = s - BACKOFF * np.array([math.cos(psi), math.sin(psi)])
            back += BACKOFF
        stance = dict(xy=s, yaw=psi, back=back)
    else:
        stance = dict(xy=np.asarray(p0[:2], float), yaw=float(yaw0), back=0.0)
    Ts = np.asarray(lk["grasp_T"], float)
    G = np.asarray(lk["grasp_hand"], float)
    sc = np.asarray(lk["grasp_score"], float)
    n_pal = Ts[:, :3, 1]  # GraspGen-X's approach: the palm normal (wrist +y, hg_look)
    cf = HL.command_form(Ts)
    dyaw = np.array([wrap(a - stance["yaw"]) for a in cf["yaw"]])
    pitch = cf["pitch"]
    roll = cf["roll"]
    Rh = L.rot_z(stance["yaw"])
    th = (G[:, :2] - stance["xy"]) @ Rh[:2, :2]  # heading-frame xy of the grasp point
    pre = G - PRE_D * n_pal
    tp = (pre[:, :2] - stance["xy"]) @ Rh[:2, :2]
    lo, hi = L.WIDE
    inbox = lambda t, z: (t[:, 0] >= lo[0]) & (t[:, 0] <= hi[0]) & (t[:, 1] >= lo[1]) & (t[:, 1] <= hi[1]) & (z >= lo[2]) & (z <= hi[2])  # noqa: E731
    keep = {}
    keep["form"] = (np.abs(dyaw) <= math.radians(60)) & (pitch >= math.radians(-5)) & (pitch <= math.radians(90)) & (np.abs(roll) <= ROLL_MAX)
    keep["reach"] = keep["form"] & inbox(th, G[:, 2]) & inbox(tp, pre[:, 2])
    Tp = Ts.copy()
    Tp[:, :3, 3] -= PRE_D * n_pal
    hits = HL.hand_box_hits(Ts, furn) if len(furn) else np.zeros(len(Ts), int)
    hits_p = HL.hand_box_hits(Tp, furn) if len(furn) else np.zeros(len(Ts), int)
    keep["clear"] = keep["reach"] & (hits <= HAND_CLEAR) & (hits_p <= HAND_CLEAR)
    counts = {k: int(v.sum()) for k, v in keep.items()}
    if counts["clear"] == 0:
        return dict(fail="reach", why=f"no grasp left: {counts}", counts=counts, stance=_js(stance))
    i = int(np.flatnonzero(keep["clear"])[np.argmax(sc[keep["clear"]])])
    g = G[i]
    pz = pelvis_rule(float(g[2]))
    ad = np.array([math.cos(pitch[i]) * math.cos(cf["yaw"][i]), math.cos(pitch[i]) * math.sin(cf["yaw"][i]), -math.sin(pitch[i])])
    via = None
    if g[2] >= VIA_Z:
        hd = np.array([math.cos(stance["yaw"]), math.sin(stance["yaw"])])
        xh = float((pre[i][:2] - stance["xy"]) @ hd)
        back = min(VIA_BACK, max(0.0, xh - VIA_XMIN))
        via = np.r_[pre[i][:2] - back * hd, pre[i][2] + VIA_UP]
    return dict(fail=None, idx=i, score=float(sc[i]), goal=g, pre=pre[i], n_pal=n_pal[i], ad=ad, roll=float(roll[i]),
                pz=pz, stance=stance, counts=counts, obj_c=c, via=via)  # fmt: skip


def _js(d):
    return json.loads(json.dumps(d, default=LT._json))


# -- the command (belief frame) -------------------------------------------------------------------------------------
def command_b(pose: tuple, goal: np.ndarray, ad: np.ndarray, pz) -> tuple[dict, np.ndarray]:
    """lml_tasks.command with the believed pelvis (xy, yaw) instead of the simulator's."""
    p, yaw = np.asarray(pose[0], float), float(pose[1])
    Rh = L.rot_z(yaw)
    gh = np.r_[(Rh.T @ np.r_[goal[:2] - p[:2], 0.0])[:2], goal[2]]
    dyaw = wrap(math.atan2(ad[1], ad[0]) - yaw)
    pitch = math.asin(float(np.clip(-ad[2], -1.0, 1.0)))
    c = dict(side="right", target=gh.copy(), dyaw=float(np.clip(dyaw, -math.radians(60), math.radians(60))),
             pitch=float(np.clip(pitch, 0.0, math.pi / 2)), pelvis_z=pz)  # fmt: skip
    return c, gh


# -- training-data ranges ---------------------------------------------------------------------------------------------
_RANGE = None


def train_range() -> tuple[np.ndarray, np.ndarray]:
    """0.5 / 99.5 % of each of the 29 joints at the command rows of the self model's training segments."""
    global _RANGE
    if _RANGE is None:
        f = OUT / "train_q_range.npz"
        if f.exists():
            z = np.load(f)
            _RANGE = (z["lo"], z["hi"])
        else:
            z = np.load(HOSTDATA / "data_scalebfm" / "all.npz")
            n, k0 = z["n_steps"], z["k0"]
            start = np.r_[0, np.cumsum(n)[:-1]]
            q = z["S"][start + k0, :29]
            lo, hi = np.percentile(q, 0.5, axis=0), np.percentile(q, 99.5, axis=0)
            np.savez(f, lo=lo, hi=hi)
            _RANGE = (lo, hi)
    return _RANGE


# -- one episode ------------------------------------------------------------------------------------------------------
def look_path(task: str, ep: int) -> Path:
    return OUT / "looks" / task / f"ep{ep:+05d}"


def episode(dr: HGDriver, ep: int, method: str, P, do_look: bool = False) -> dict:
    import hg_look as HL
    import hg_walk as HW

    t0 = time.time()
    spec = TASKS[dr.task]
    jit = dr.reset_hg(ep)
    frames = dr.start_frames()
    f_look = int(round(L.START_HOLD_S * L.FPS))
    if spec["kind"] == "approach":  # the look lean (same for both groups)
        n_l, n_a = int(round(LEAN_S * L.FPS)), int(round(LOOK_AFTER_S * L.FPS))
        f0 = 5
        frames = np.concatenate([frames, np.repeat(frames[-1:], n_l + n_a + 10, axis=0)])
        for j in range(f0, len(frames)):
            frames[j, 7 + WAIST_PITCH] = L.DEFAULT_Q[WAIST_PITCH] + LOOK_LEAN * L.minjerk(min(1.0, (j - f0) / n_l))
        f_look = max(f_look, f0 + n_l + n_a)
    rows: dict[int, dict] = {}
    R = dict(task=dr.task, ep=ep, method=method, jitter=jit, obj_name=dr.obj_name, scene=dr.scene)
    f = 0
    phase, fell, why = "stance", None, ""
    odo, plan, lk = None, None, None
    sched: dict[str, int] = {}
    reaches, contact_hits = [], []
    lift_run, lift_best, rise_max, official_at, own_at = 0, 0, 0.0, None, None
    anchor = None
    walk_info: dict = {}
    n_walks = 0
    first = None
    reset_id = 9_000_000 + max(ep, 0)
    rlo, rhi = train_range()
    while True:
        dr.push(frames[min(f, len(frames) - 1)])
        if dr.buf is None:
            dr.buf = dr._find_buffer()
        dr.step()
        x = rows[f] = dr.sample()
        if odo is not None:
            odo.update(dr, x)
        if float(x["root"][2]) < L.FALL_Z:
            fell, why = f, f"fell during {phase}"
            break
        if anchor is not None:
            dev = (x["root"][:2] - anchor[0]) - (frames[min(f, len(frames) - 1), :2] - anchor[1])
            if float(np.linalg.norm(dev)) > L.LOST_M:
                fell, why = f, f"lost during {phase}"
                break
        # scoring (truth)
        z = float(dr.d.xpos[dr.obj_bid][2]) - dr.obj_z0
        if "close" in sched and f >= sched["close"]:
            hand_c, world_c = dr.obj_contacts()
            rise_max = max(rise_max, z)
            lift_run = lift_run + 1 if (z >= LIFT_MIN and not world_c) else 0
            lift_best = max(lift_best, lift_run)
            try:
                if official_at is None and bool(dr.env._check_success()):
                    official_at = f
            except Exception:
                pass
        if first and first in sched and sched[first] <= f < sched.get("close", 10**9):
            h = dr.arm_scene_contact()
            if h is not None:
                contact_hits.append((f, h))
        # -- the look ---------------------------------------------------------------------------------------------
        if f == f_look:
            lp = look_path(dr.task, ep)
            if do_look:
                HL.look(dr, HL.prompt_of(dr), lp, prompts=spec.get("prompts"))
                return dict(task=dr.task, ep=ep, look=str(lp), wall_s=time.time() - t0)
            lk = HL.load(lp)
            pel = np.asarray(lk["pelvis"], float)
            R["look_state_dev_mm"] = 1000 * float(np.abs(pel[:3] - x["root"][:3]).max())
            R["look"] = dict(status=lk["meta"].get("status"), mask_score=float(lk["mask_score"]), n_obj=len(lk["obj_pts"]),
                             n_grasps=len(lk["grasp_T"]), prompt=lk["meta"].get("prompt_used"))  # fmt: skip
            tp = dr.obj_pos()
            R["look"]["obj_center_err_cm"] = 100 * float(np.linalg.norm(np.median(lk["obj_pts"], 0) - tp)) if len(lk["obj_pts"]) else None
            p0, yaw0 = x["root"][:2].copy(), L.yaw_of(L._qmat(x["root"][3:]))
            odo = HW.LegOdo(dr, x, (p0, yaw0))
            plan = plan_from_look(lk, spec["kind"], x["root"][:3], yaw0)
            R["plan"] = _js({k: v for k, v in plan.items() if k not in ("stance",)})
            R["plan"]["stance"] = _js(plan.get("stance"))
            if plan["fail"]:
                why = plan["fail"]
                break
            st = plan["stance"]
            d_move = float(np.linalg.norm(st["xy"] - p0))
            d_turn = abs(wrap(st["yaw"] - yaw0))
            R["move"] = dict(dist=d_move, turn_deg=math.degrees(d_turn))
            if d_move >= MOVE_MIN or d_turn >= TURN_MIN:
                wf = HW.walk_frames(dr, x, frames[f], st["xy"], st["yaw"], pose=odo.pose(), info=walk_info)
                frames = np.concatenate([frames[: f + 1], wf])
                anchor = (x["root"][:2].copy(), frames[f, :2].copy())
                first = "cmd0" if plan["via"] is not None else "cmd1"
                sched[first] = f + len(wf) - 25
                phase = "walk"
                n_walks = 1
            else:
                frames = np.concatenate([frames[: f + 1], L.hold((frames[f][:3], frames[f][3:7], frames[f][7:]), 2.0)])
                first = "cmd0" if plan["via"] is not None else "cmd1"
                sched[first] = f + int(round(L.PRE_S * L.FPS))
        # -- reaches ----------------------------------------------------------------------------------------------
        if first and sched.get(first) == f and phase == "walk" and n_walks < 1 + WALK_FIX:
            # a correction walk while the leg odometry puts the stance >= FIX_MIN away (design: ScaleBFM stopped 17-29 cm
            # short of 2.2-2.3 m walks); the same rule for both groups, odometry only
            pose = odo.pose()
            st = plan["stance"]
            d_left = float(np.linalg.norm(st["xy"] - pose[0]))
            R.setdefault("walk_left_cm", []).append(100 * d_left)
            if d_left >= FIX_MIN and n_walks < WALK_FIX + 1:
                wf = HW.walk_frames(dr, x, frames[f], st["xy"], st["yaw"], pose=pose)
                frames = np.concatenate([frames[: f + 1], wf])
                anchor = (x["root"][:2].copy(), frames[f, :2].copy())
                sched[first] = f + len(wf) - 25
                n_walks += 1
        for k, key in ((-1, "cmd0"), (0, "cmd1"), (1, "cmd2")):
            if sched.get(key) == f:
                phase = f"reach{k + 1}"
                if k == 0 and walk_info is not None and "walk" in R.get("phase_log", ["walk"]):
                    pass
                goal = plan["via"] if k == -1 else plan["pre"] if k == 0 else plan["goal"]
                kc = 0 if k == -1 else min(1, k + (plan["via"] is not None))  # the chain index the self model sees
                pose = odo.pose()
                c, gh = command_b(pose, goal, plan["ad"], plan["pz"])
                info = dict(k=k, goal_w=goal.tolist(), goal_h=gh.tolist(), dyaw_deg=math.degrees(c["dyaw"]),
                            pitch_deg=math.degrees(c["pitch"]), pz=c["pelvis_z"], pose_odo=[pose[0].tolist(), pose[1]],
                            pose_true=[x["root"][:2].tolist(), L.yaw_of(L._qmat(x["root"][3:]))])  # fmt: skip
                qx = x["q"]
                out_j = int(((qx < rlo) | (qx > rhi)).sum())
                info.update(q_out=out_j, q_in_range=out_j == 0)
                tic = time.time()
                best = None
                if method == "selfmodel" and k >= 0:
                    t, evs = gh.copy(), []
                    for it in range(LT.ITERS + 1):
                        cc = dict(c, target=t.copy())
                        try:
                            pl_, D = P.features(dr, rows, f, frames, cc, kc, reset_id)
                        except (np.linalg.LinAlgError, ValueError):
                            evs.append(dict(t=t.copy(), l=np.inf))
                            break
                        pos, std = P.predict(D)
                        evs.append(dict(t=t.copy(), pos=pos, std=std, l=float(((pos - gh) ** 2).sum() + LT.LAMBDA * (std**2).sum()), plan=pl_))
                        t = LT.clip_t(t + (gh - pos), "right")
                    best = min(evs, key=lambda e: e["l"])
                    if "plan" in best:
                        c["target"] = best["t"]
                        info.update(pred0_h=evs[0]["pos"].tolist(), pred_miss0_cm=100 * float(np.linalg.norm(evs[0]["pos"] - gh)),
                                    pred_miss_cm=100 * float(np.linalg.norm(best["pos"] - gh)), pred_exec_h=best["pos"].tolist())  # fmt: skip
                        blend, last, rec = best["plan"]
                    else:
                        info["model_fallback"] = True
                        best = None
                if best is None:
                    try:
                        (blend, last, rec), Dx = P.features(dr, rows, f, frames, c, kc, reset_id)
                        info["pred_exec_h"] = P.predict(Dx)[0].tolist()
                    except (np.linalg.LinAlgError, ValueError) as e:
                        why = f"reach: plan failed {type(e).__name__}"
                        info["plan_error"] = str(e)[:200]
                        reaches.append(info)
                        fell = None
                        break
                info["think_s"] = time.time() - tic
                rec.update(chain=kc, f_cmd=f, f_end=f + len(blend))
                post = int(round(L.POST_S * L.FPS))
                frames = np.concatenate([frames[: f + 1], blend, L.hold(last, L.POST_S + L.PRE_S + L.MARGIN_S + 0.5)])
                anchor = (x["root"][:2].copy(), frames[f, :2].copy())
                info.update(cmd_h=np.asarray(c["target"]).tolist(), f_cmd=f, f_end=rec["f_end"], f_win=rec["f_end"] + post,
                            ik_err_cm=100 * float(rec["ik_err"]), p_cmd=x["root"][:3].tolist(), yaw_cmd=L.yaw_of(L._qmat(x["root"][3:])))  # fmt: skip
                reaches.append(info)
                if k == -1:
                    sched["cmd1"] = rec["f_end"] + post + int(round(L.PRE_S * L.FPS))
                elif k == 0:
                    sched["cmd2"] = rec["f_end"] + post + int(round(L.PRE_S * L.FPS))
                else:
                    sched["close"] = rec["f_end"] + post
        if why.startswith("reach: plan failed"):
            break
        if sched.get("close") == f:
            phase = "close"
            hp = dr.hand_point(x)
            R["close_dist_cm"] = 100 * float(np.linalg.norm(hp - plan["goal"]))  # hand point vs the grasp target (look frame)
            R["close_obj_cm"] = 100 * float(np.linalg.norm(hp - dr.obj_pos()))
            sched["lift"] = f + int(round((CLOSE_S + SETTLE_S) * L.FPS))
        if "close" in sched and f >= sched["close"] and dr.grip < GRIP:
            dr.grip = min(GRIP, (f - sched["close"] + 1) / (0.3 * L.FPS))
        if sched.get("lift") == f:
            phase = "lift"
            pose = odo.pose()
            goal = plan["goal"] + np.array([0.0, 0.0, LIFT])
            c, gh = command_b(pose, goal, plan["ad"], None)
            try:
                blend, last, rec = dr.plan(x, frames[f], c)
            except (np.linalg.LinAlgError, ValueError):
                why = "lift: plan failed"
                break
            frames = np.concatenate([frames[: f + 1], blend, L.hold(last, HOLD_S + L.MARGIN_S + 0.5)])
            anchor = (x["root"][:2].copy(), frames[f, :2].copy())
            sched["end"] = f + len(blend) + int(round(HOLD_S * L.FPS))
            R["lift_cmd"] = dict(goal_h=gh.tolist(), ik_err_cm=100 * float(rec["ik_err"]))
        if sched.get("end") is not None and f >= sched["end"]:
            break
        if f >= len(frames) - 1:
            why = why or "motion ended"
            break
        if f > 60 * L.FPS:
            why = "timeout"
            break
        f += 1
    # -- post-hoc --------------------------------------------------------------------------------------------------
    for w in reaches:
        if "f_win" not in w:
            continue
        fw = min(w["f_win"], max(rows))
        hand = dr.hand_point(rows[fw]) if fw in rows else None
        if hand is None:
            continue
        p0c, Rh = np.asarray(w["p_cmd"]), L.rot_z(w["yaw_cmd"])
        act_h = np.r_[(Rh.T @ np.r_[hand[:2] - p0c[:2], 0.0])[:2], hand[2]]
        w["err_cm"] = 100 * float(np.linalg.norm(hand - np.asarray(w["goal_w"])))
        w["act_h"] = act_h.tolist()
        if "pred_exec_h" in w:
            w["pred_err_cm"] = 100 * float(np.linalg.norm(np.asarray(w["pred_exec_h"]) - act_h))
        w["complete"] = fw == w["f_win"]
    if walk_info:
        R["walk"] = _js({k: v for k, v in walk_info.items() if k in ("walk_s", "n_steps", "ik_err", "knots")})
    if first and first in sched and odo is not None and 0 < sched[first] <= max(rows):
        xr = rows[min(sched[first], max(rows))]
        st = plan["stance"]
        R["arrive"] = dict(pos_err_cm=100 * float(np.linalg.norm(xr["root"][:2] - st["xy"])),
                           yaw_err_deg=math.degrees(abs(wrap(L.yaw_of(L._qmat(xr["root"][3:])) - st["yaw"]))))  # fmt: skip
    own_ok = official_at is not None
    lift10 = lift_best >= LIFT_HOLD
    success = own_ok if spec["own"] else lift10
    R.update(success=bool(success), official=own_ok, official_step=official_at, lift10=bool(lift10), rise_max_cm=100 * rise_max,
             fell=fell, why=why, steps=len(rows), n_walks=n_walks, reaches=reaches, n_contacts=len(contact_hits),
             first_contact=contact_hits[0] if contact_hits else None, wall_s=round(time.time() - t0, 1))  # fmt: skip
    R["reason"] = classify(R, phase)
    return R


def classify(R: dict, phase: str) -> str:
    """Failure reason, in the entry's order of priority (success -> 'ok')."""
    if R.get("success"):
        return "ok"
    p = R.get("plan") or {}
    if p.get("fail") == "perception":
        return "perception"
    if p.get("fail") == "reach" or str(R.get("why", "")).startswith("reach") or any(w.get("ik_err_cm", 0) > 3.0 for w in R.get("reaches", [])):
        return "stance_or_ik"
    if R.get("fell") is not None and "walk" in str(R.get("why")):
        return "fell_walking"
    if R.get("n_contacts", 0) > 0:
        return "hit_furniture"
    if R.get("rise_max_cm", 0) < 2.0:
        return "missed"
    if R.get("rise_max_cm", 0) >= 2.0:
        return "dropped"
    return "other"


# -- runners ----------------------------------------------------------------------------------------------------------
def ep_path(task: str, method: str, ep: int) -> Path:
    return OUT / "episodes" / task / method / f"ep{ep:+05d}.json"


def run(task: str, eps: list[int], methods: list[str], shard: int, shards: int, phase: str) -> None:
    import torch

    torch.set_num_threads(1)
    if phase == "look":
        jobs = [e for e in eps][shard::shards]
        todo = [e for e in jobs if not Path(str(look_path(task, e)) + ".npz").exists()]
    else:
        jobs = [(m, e) for e in eps for m in methods][shard::shards]
        todo = [(m, e) for m, e in jobs if not ep_path(task, m, e).exists()]
    print(f"hg {phase} {task} shard {shard}/{shards}: {len(jobs) - len(todo)}/{len(jobs)} done", flush=True)
    if not todo:
        return
    dr = HGDriver(task)
    P = None if phase == "look" else LT.Predictor(WM)
    for job in todo:
        m, e = (None, job) if phase == "look" else job
        try:
            r = episode(dr, e, m or "look", P, do_look=phase == "look")
        except Exception as err:  # recorded, not retried
            r = dict(task=task, ep=e, method=m, error=f"{type(err).__name__}: {err}", tb=traceback.format_exc()[-2000:],
                     success=False, reason="other")  # fmt: skip
            print(f"hg {task} {m} ep {e}: ERROR {err}", flush=True)
            dr = HGDriver(task)
        if phase == "look":
            print(f"hg look {task} ep {e}: {r.get('wall_s', 0):.1f} s", flush=True)
            continue
        p = ep_path(task, m, e)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, default=LT._json))
        os.replace(tmp, p)
        rw = r.get("reaches") or [{}]
        print(f"hg {task} {m} ep {e}: success {r.get('success')} reason {r.get('reason')} close {r.get('close_dist_cm', float('nan')):.1f} cm "
              f"rise {r.get('rise_max_cm', 0):.1f} cm reach errs {[round(w.get('err_cm', -1), 1) for w in rw]} {r.get('why')} {r.get('wall_s')} s", flush=True)  # fmt: skip


def mcnemar_exact(b: int, c: int) -> float:
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def stats(tag_eps: str) -> None:
    lo, hi = (int(v) for v in tag_eps.split(":"))
    rs = {}
    for p in glob.glob(str(OUT / "episodes" / "*" / "*" / "ep*.json")):
        r = json.loads(Path(p).read_text())
        if lo <= r["ep"] < hi:
            rs[(r["task"], r["method"], r["ep"])] = r
    tasks = sorted({k[0] for k in rs})
    allb = allc = 0
    alld = []
    print(f"episodes {lo}:{hi}, {len(rs)} records")
    for t in tasks:
        eps = sorted({k[2] for k in rs if k[0] == t and (t, "alone", k[2]) in rs and (t, "selfmodel", k[2]) in rs})
        A = [rs[(t, "alone", e)] for e in eps]
        S = [rs[(t, "selfmodel", e)] for e in eps]
        b = sum(a["success"] and not s["success"] for a, s in zip(A, S))
        c = sum(s["success"] and not a["success"] for a, s in zip(A, S))
        allb, allc = allb + b, allc + c
        d = [s["close_dist_cm"] - a["close_dist_cm"] for a, s in zip(A, S) if "close_dist_cm" in a and "close_dist_cm" in s]
        alld += d
        cd = lambda G: np.median([g["close_dist_cm"] for g in G if "close_dist_cm" in g]) if any("close_dist_cm" in g for g in G) else float("nan")  # noqa: E731
        from collections import Counter

        print(f"== {t}  pairs {len(eps)}")
        for name, G in (("alone", A), ("selfmodel", S)):
            pe = [w["pred_err_cm"] for g in G for w in g.get("reaches", []) if w.get("k") == 1 and "pred_err_cm" in w]
            qin = [w["q_in_range"] for g in G for w in g.get("reaches", []) if w.get("k") == 0 and "q_in_range" in w]
            print(f"  {name:9s} success {sum(g['success'] for g in G):2d}/{len(G)}  official {sum(g.get('official', False) for g in G):2d}  "
                  f"lift10 {sum(g.get('lift10', False) for g in G):2d}  close-dist median {cd(G):.1f} cm  "
                  f"reasons {dict(Counter(g['reason'] for g in G))}  start in range {sum(qin)}/{len(qin)}  "
                  f"grasp-reach pred err median {np.median(pe) if pe else float('nan'):.1f} cm")  # fmt: skip
        print(f"  discordant: alone-only {b}, selfmodel-only {c}, McNemar exact p {mcnemar_exact(b, c):.3g}; "
              f"paired close-dist diff (S - A) median {np.median(d) if d else float('nan'):+.1f} cm (n {len(d)})")
    print(f"== pooled: alone-only {allb}, selfmodel-only {allc}, McNemar exact p {mcnemar_exact(allb, allc):.3g}; "
          f"paired close-dist diff median {np.median(alld) if alld else float('nan'):+.1f} cm (n {len(alld)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["look", "run", "stats", "range"])
    ap.add_argument("--task", default="IThorBathroomLiftCandleFromFloor")
    ap.add_argument("--eps", default="1000:1010")
    ap.add_argument("--methods", default="alone,selfmodel")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.eps.split(":"))
    if a.phase == "stats":
        stats(a.eps)
    elif a.phase == "range":
        print(train_range())
    else:
        run(a.task, list(range(lo, hi)), a.methods.split(","), a.shard, a.shards, a.phase)
