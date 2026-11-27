"""selfmodel_hadata_20261126 revision 2a (2026-11-26): real HumanoidArena (HA) task episodes for ScaleBFM / HEFT (and
SONIC, selfmodel_hadata_sonic_20261126), every reach recorded in data3's segment layout, for the entries' evaluation 3
(the model predicts offline; nothing here reads a model, and no command depends on one).

The flow is humanoidarena_20261124/ha_bench.py's "look, walk to the stand, reach", moved onto the collection's hosts
(hc_ha's: sx_ha.HAG1Host for ScaleBFM / HEFT, HA's own SonicActionProvider with the native stance for SONIC), without
hc_ha's site move and start noise (HA's own episode reset). What is ha_bench's, imported and used as it is: the looks (one front-camera RGB-D frame, SAM 3 masks for the
task's prompts, perceive.Look), each task's target rule (find_box / find_hammer / find_handle, the stand rule on the
support -- with the clearance per controller, CLEAR_BY below --, box_faces / topdown_R / the door hook), the stand point
and heading, the search turns, up to 3 approach rounds, the final look at the stand, the pregrasp (8 cm out along the approach) and the grasp rotation. What changes
with the controller: the locomotion and the reach go through LocoManiLab's path, as data3 collected them --
  walk   one hardgrasp hg_walk.walk_frames plan (turn, walk, turn; short moves side-step) to the stand point and
         heading (from the latest look), then data3's settle wait; a search turn is a walk to the same point. A walk
         to a stand more than SHORT_M + 0.12 m away stops SHORT_M short of it (0.25 m back along the stand heading,
         away from the support; coordinator 2026-11-26: ScaleBFM coasted 7-21 cm past the door stands in the smoke);
         the next round, after a look, walks the rest (short steps)
  belief the robot's planar position is leg odometry (hg_walk.LegOdo, hardgrasp's: the joint encoders, the IMU's
         orientation, the loaded foot anchored; the feet's load from the task's ContactSensor on the ankle links),
         started where the robot stands before the first look (the map's origin; nothing measured about the world).
         Every look's world points, the stand rule's kept edge, the walks' start pose and every command use it;
         heading and height are the IMU's / the legs' (the sim's values stand in for them, as in hardgrasp). The
         simulator's base position only scores (odo_err_cm, the drift) and aborts (lost)
  stand  ha_bench's CLEAR (pelvis to the support's near edge, 0.16 m, set for SONIC's arm-only reach) per controller:
         its hands' forward extent at its own stance (hand_front, the robot's kinematics) + 5 cm. At 0.16 m ScaleBFM's
         hands (0.19 m ahead, 0.54-0.83 m high) were over the 0.74 m table edge before any reach (12 of 13 smoke
         episodes touched while walking or standing)
  reach  d3_collect.plan_reach (LocoManiLab's whole-body IK, pelvis_z None = standing) to the command: the hand point
         (L.HAND_OFF on wrist_yaw_link) and wrist rotation that put ha_bench's grasp point (its palm-frame offset,
         G_BOX / G_FIST) at the pregrasp with its grasp rotation (palm frame = wrist_yaw frame + d3's PALM_OFF, no
         rotation: checked against HA's bodies at start-up), in the measured pelvis heading frame at the command
         (target x, y; z world; dyaw / pitch / roll the intrinsic Z-Y-X angles of the wrist rotation relative to the
         heading, as d3's commands are drawn)
  via    (--via 1, dev 2026-11-26) hardgrasp hg_run's via point before a pre-grasp on a perceived support (the box's
         table, the hammer's desk; --via-rule z: hg_run's grasp >= 0.80 m): first a reach to the pre-grasp pulled back
         toward the body along the heading (by up to 0.22 m, not nearer than 0.15 m ahead of the pelvis) and 5 cm up,
         same rotation, then the reach to the pre-grasp; --via-rule lift: straight up from the hand to LIFT_CLEAR over
         the perceived support (at least VIA_UP over the pre-grasp), then the pre-grasp. The support height is ha_bench's
         perceive.support_level from the look's points, never the scene's geometry
  ppbox  per hand (the left, then the right: ha_bench's order) its via reach and its pre-grasp reach
Settling (data3's rule with hc_ha's fixed parameters: 2 cm/s -- SONIC 5 cm/s -- for 0.5 s, up to 6 s): before each command and for each
reach's end label. A command waits for it; if the wait runs out the command goes anyway (pre_settled 0; data3 would
skip it: a task cannot) and the label search is data3's. The episode ends after the last reach's label (the task's
grasp, carry and place are not run: the self-model covers empty-hand reaching), at a fall (d3's 0.45 m) or lost (d3's
0.5 m, off during walks).

Record per episode (data/<task>_s<seed>_r<repeat>.npz + arm/, as hc_ha's chunks; log/<same>.json): d3_collect.segments
(walked 1, chain = reaches before in the episode, the walk fields from the last walk before the first reach), sx_collect's
d_this / d_prev, hc_ha's EXTRA, sx_ha's scene contact, and per segment: task_id, seed, repeat, reach_no (0 via point,
1 pre-grasp; lmlreal's numbering), hand_no (the hand's order in the episode), pre_settled, cmd_in_range (d3's dyaw /
pitch / roll / z ranges), and in the log per reach the command's target in the heading frame (tgt_h) and in_table
(inside data3's standing reach table). reset = RID0 + 1000 task_id + 10 (seed - SEED0) + repeat (the same numbers for
every controller; the controllers' records are separate directories). Diagnostics in the log: hand_lead (link
origins), hand_front, clear.

    SX_DEPTH=1 SX_HA_TASK=ppbox HC_SCRIPT=hr_real.py bash hc_ha.sh <gpu> --controller scalebfm --out DIR \
        --seeds 200-209 --repeats 0
"""

from __future__ import annotations

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
import hc_ha as HC  # noqa: E402  (sx_ha, ha_sim)

SXH, ha_sim = HC.SXH, HC.ha_sim
sys.path.insert(0, str(Path(ha_sim.__file__).resolve().parent))
import ha_bench as B  # noqa: E402
import perceive as P  # noqa: E402

TASK_ID = {"ppbox": 0, "doubledesk": 1, "opendoor": 2}
RID0, SEED0 = 95_000_000, 200  # record ids (data3 91M / 92M, SONIC 93M / 94M, xfer 40M); dev seeds from 200
SEARCH = {"doubledesk": (-45, -90), "opendoor": (45, -45)}  # ha_bench's search turns (deg from the heading now)
STAND_TOL_M, STAND_TOL_DEG = 0.12, 5.0  # ha_bench: within 0.12 m of the stand point (its turns land to 3 deg)
# hardgrasp hg_run's via point (--via 1): a first reach to a point VIA_BACK nearer the body along the heading (at most
# to VIA_XMIN ahead of the pelvis) and VIA_UP above the pre-grasp, same rotation. hg_run gives it to grasp points >=
# VIA_Z (LocoManiLab's 0.93 m counter); here (--via-rule support, dev 2026-11-26) to every grasp on a perceived support
# (the box on its table, the hammer on its desk: HA's are 0.74 m, under VIA_Z, and the straight line from the hip ran
# into their front edge in the smoke); --via-rule z keeps hg_run's
VIA_Z, VIA_BACK, VIA_UP, VIA_XMIN = 0.80, 0.22, 0.05, 0.15
LIFT_CLEAR = 0.15  # m over the perceived support, --via-rule lift (dev)
TOUCH_ALL_N = 1.0  # every non-foot body over this force is recorded per frame (log "contacts"; scoring only, revision 2a)
ROUNDS = 3
SHORT_M = 0.25  # a long walk stops this far short of the stand (back along the stand heading); the next round corrects
# the stand's clearance (ha_bench's CLEAR, pelvis to the support's near edge; its 0.16 m was set for SONIC) per
# controller = its hands' forward extent at its own stance (hand_front: the robot's kinematics) + CLEAR_MARGIN
# (coordinator 2026-11-26), rounded up to the cm. --probe-lead on PPBox dev seeds 296-299 (2026-11-26, the same on all
# four: each controller's stance after HA's reset is deterministic): hand_front max over the hands ScaleBFM 0.193 m
# (lml stance), HEFT 0.162 m (lml), SONIC 0.289 m (native); --clear overrides (dev)
CLEAR_MARGIN = 0.05
CLEAR_BY = {"scalebfm": 0.25, "heft": 0.22, "sonic": 0.34}


class End(Exception):
    pass


class HRG1(HC.HCG1):
    """hc_ha's host with HA's own reset: no site move, no start noise."""

    def place(self) -> None:
        self.placed = {}


def ha_odo(HW, dr, x0, pose0):
    """hg_walk.LegOdo with the feet's load read from HA's ContactSensor: the vertical net force on each ankle_roll_link
    (what a foot force sensor reads) instead of MuJoCo's contacts (hg_walk is imported in run())."""

    class HAOdo(HW.LegOdo):
        def _foot_geoms(self, dr):
            names = list(dr.cs.body_names)
            self.i_foot = {s: names.index(f"{s}_ankle_roll_link") for s in ("left", "right")}

        def load(self, dr):
            F = dr.cs.data.net_forces_w[0].detach().cpu().numpy().astype(np.float64)
            return {s: abs(float(F[i, 2])) for s, i in self.i_foot.items()}

    return HAOdo(dr, x0, pose0)


# -- --heft-gait (off by default): HEFT's walks as LocoManiLab's task episodes walk it ------------------------------------
HEFT_GAIT_DIR = HERE.parent / "selfmodel_lmlcmp_20261126" / "heft_gait"  # lc3_run's LMLCMP_DIR / heft_gait (byte copies)


def heft_gait_modules():
    """lc3_run.heft_gait: locomani.backends.heft.{gait, gait_heft} from the byte-identical copies in HEFT_GAIT_DIR."""
    import types

    name = "locomani.backends.heft"
    if name not in sys.modules:
        import locomani.backends as LB

        pkg = types.ModuleType(name)
        pkg.__path__ = [str(HEFT_GAIT_DIR)]
        pkg.__package__ = name
        sys.modules[name] = pkg
        LB.heft = pkg
    from locomani.backends.heft import gait as G
    from locomani.backends.heft import gait_heft as GH

    return G, GH


def heft_walk_frames(dr, ref_frame, goal_xy, goal_yaw: float, pose, info, L, HW):
    """lc3_run.walk_ctrl's HEFT branch (lines 199-214): gait_heft.HEFT's footsteps on its own LegModel, the upper body
    (joints 12+) blended from the reference frame's to DEFAULT_Q over hg_walk's BLEND_S (min-jerk)."""
    G, GH = heft_gait_modules()
    if not hasattr(dr, "_heft_legs"):
        dr._heft_legs = G.LegModel(dr.pl.ik)
        print(f"heft-gait: {GH.__file__}, {G.__file__}", flush=True)
    wf = GH.walk_frames(dr._heft_legs, ref_frame, pose, goal_xy, goal_yaw, GH.HEFT, info)
    q0 = np.asarray(ref_frame[7:], float)
    t = (np.arange(len(wf)) + 1) / L.FPS
    bl = np.array([L.minjerk(min(1.0, v / HW.BLEND_S)) for v in t])
    wf[:, 7 + 12 :] = (q0[12:][None] + bl[:, None] * (L.DEFAULT_Q[12:] - q0[12:])[None]).astype(np.float32)
    if info is not None:
        info["gait"] = "gait_heft.HEFT"
    return wf


class HRSonic(HC.HCSonic):
    """hc_ha's SONIC host (HA's own SonicActionProvider, stance native, 5 cm/s) with HA's own reset: the collection's
    walk and reach generation, not ha_bench's arm-only SONIC line (coordinator 2026-11-26; sonic entry's revision)."""

    def place(self) -> None:
        self.placed = {}


def parser():
    p = ha_sim.build_parser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--controller", choices=["scalebfm", "heft", "sonic"], required=True)
    p.add_argument("--seeds", default="200-209")
    p.add_argument("--repeats", default="0")
    p.add_argument("--save_looks", type=int, default=1)
    p.add_argument("--via", type=int, default=1, help="hardgrasp's via point before the pre-grasp")
    p.add_argument("--via-rule", choices=["support", "z", "lift"], default="support",
                   help="support / z: hg_run's via point (on a support / grasp >= 0.80 m); lift (dev): straight up from the hand to "
                   "LIFT_CLEAR over the support (at least VIA_UP over the pre-grasp), then the pre-grasp")
    p.add_argument("--clear", type=float, default=None,
                   help="ha_bench's CLEAR (pelvis to the support's near edge at the stand, its 0.16 m) for this run (dev)")
    p.add_argument("--probe-lead", action="store_true", help="per seed: HA's reset, the stance wait, hand_front; nothing else")
    p.add_argument("--no-camera", action="store_true")  # hc_ha.sh's arguments are shared; looks need the camera
    p.add_argument("--heft-gait", action="store_true",
                   help="HEFT only: walks on gait_heft.HEFT as LocoManiLab's task episodes (lc3_run.walk_ctrl); off: hg_walk")
    p.add_argument("--replay-looks", type=Path, default=None,
                   help="dev check only (revision 2a): each look's colour image from DIR's saved look of the same name")
    p.add_argument("--hrvid", type=Path, default=None, help="diagnosis only: a third-person video and per-step record (Video)")
    return p


def rng_list(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


class Look(B.Bench):
    """ha_bench's looking and target rules on the running Sim (its stepping, locomotion and reaches are not used)."""

    def __init__(self, sim, a, look_dir: Path):
        self.a, self.sim, self.W = a, sim, None
        self.chain = {s: B.g1arm.Chain(tip=f"{s}_hand_palm_link") for s in ("left", "right")}
        self.torso = B.g1arm.Chain(tip="torso_link")
        self.sam = P.Sam()
        self.plan_conn = None
        self.look_dir = str(look_dir)

    def new_episode(self, name: str, ep):
        self.ep_name, self.looks, self.edges, self.target, self.stand_err = name, [], {}, None, None
        self.ep = ep

    # the robot's own pose: the odometry's planar position (Episode.delta), the IMU's rotation and the legs' height
    def pelvis(self):
        pp, pR = self.sim.pelvis()
        return pp + self.ep.delta(), pR

    def own_points(self):  # its own links relative to the pelvis (kinematics)
        pp, pR = self.sim.pelvis()
        Pw = self.sim.robot.data.body_link_pos_w[0].cpu().numpy().astype(np.float64)
        return (Pw - pp) @ pR

    def look(self, prompts, tag):
        """ha_bench.Bench.look with the frame placed at the believed pelvis (the camera's pose on the pelvis is the
        kinematics')."""
        Lk = self.sim.look()
        if self.a.replay_looks is not None:  # dev check only (revision 2a): an earlier run's colour image, the rest live
            f = Path(self.a.replay_looks) / f"{self.ep_name}_{len(self.looks) + 1:02d}_{tag}.npz"
            if f.exists():
                Lk["rgb"] = np.load(f)["rgb"]
                print(f"replay-looks: rgb from {f.name}", flush=True)
            else:
                print(f"replay-looks: {f.name} missing; live rgb", flush=True)
        Lk["pelvis_p"] = Lk["pelvis_p"] + self.ep.delta()
        lk = P.Look(Lk, own_links=self.own_points(), own_r=0.10)
        ms = self.sam.masks(Lk["rgb"], prompts, floor=0.3)
        self.looks.append(dict(tag=tag, k=self.sim.k, prompts=prompts, n=len(ms), best=ms[0][0] if ms else 0.0,
                               odo_err_cm=round(100 * float(np.linalg.norm(self.ep.delta())), 1)))  # fmt: skip
        if self.a.save_looks:
            np.savez_compressed(f"{self.look_dir}/{self.ep_name}_{len(self.looks):02d}_{tag}.npz", rgb=Lk["rgb"],
                                depth=Lk["depth"], K=Lk["K"], cam_p=Lk["cam_p"], cam_R=Lk["cam_R"], pelvis_p=Lk["pelvis_p"],
                                pelvis_R=Lk["pelvis_R"], masks=np.array([m for _, m, _ in ms[:4]]) if ms else np.zeros((0,)))  # fmt: skip
        return lk, ms


class Episode:
    """data3's run_reset loop (d3_collect), driven by the task flow instead of a drawn plan."""

    def __init__(self, h, rig, SX, D3, L, HW):
        self.h, self.rig, self.SX, self.D3, self.L, self.HW = h, rig, SX, D3, L, HW
        h.reset()
        rig.clear()
        fr = h.start_frames()
        self.frames = np.concatenate([fr, np.repeat(fr[-1:], int(D3.HOLD_S * D3.FPS), 0)])
        self.rows, self.reaches, self.walks = {}, [], []
        self.f, self.x = -1, None
        self.fell = self.lost = None
        self.why = ""
        self.anchor, self.cur_delta, self.walking = None, np.zeros(3), False
        self.delay, self.plan_s, self.t0 = None, 0.0, time.time()
        self.ready, self.wait_s = (None, False), 0.0
        self.odo = None
        self.contacts = {}  # frame -> [(body, N, origin in the belief frame, lowest point z or None)]
        self.vid = None  # --hrvid only (hr_video.Video); None: nothing below that names it runs
        self.heft_gait = False  # --heft-gait with HEFT only: walks on gait_heft.HEFT (heft_walk_frames)

    # -- the belief: leg odometry's planar position ------------------------------------------------------------------
    def start_odo(self) -> None:
        x = self.x
        self.odo = ha_odo(self.HW, self.h, x, (x["root"][:2].astype(float).copy(), self.L.yaw_of(self.L._qmat(x["root"][3:]))))

    def delta(self) -> np.ndarray:
        """believed - simulated pelvis position (planar), the shift from the sim's world to the belief's."""
        if self.odo is None:
            return np.zeros(3)
        return np.r_[self.odo.p[:2] - self.x["root"][:2], 0.0]

    def pose(self) -> tuple[np.ndarray, float]:
        return self.x["root"][:2] + self.delta()[:2], self.L.yaw_of(self.L._qmat(self.x["root"][3:]))

    def xb(self) -> dict:
        """This row with the pelvis and wrists at the belief's position."""
        d = self.delta()
        x = dict(self.x)
        x["root"] = np.r_[self.x["root"][:2] + d[:2], self.x["root"][2:]]
        x["wr"] = {s: (np.asarray(p, float) + d, R) for s, (p, R) in self.x["wr"].items()}
        return x

    def note_contacts(self, f: int) -> None:
        """Every non-foot body over TOUCH_ALL_N now (read only, after the sample; nothing reads it back in the episode):
        its name, force (N), origin in the belief frame and, for wrist and hand links, the lowest point of its geoms in
        the host's mirror (the robot's kinematics). For the sensitivity group (revision 2a): touches only on the target."""
        h = self.h
        F = h.cs.data.net_forces_w[0, h.cs_keep].detach().cpu().numpy().astype(np.float64)
        n = np.linalg.norm(F, axis=1)
        idx = np.flatnonzero(n > TOUCH_ALL_N)
        if not len(idx):
            return
        d = self.delta()
        out = []
        for i in idx:
            nm = h.cs_names[i]
            pos = h.sim.robot.data.body_link_pos_w[0, h.cs_body[i]].detach().cpu().numpy().astype(np.float64) + d
            low = link_low(h, nm) if ("_hand" in nm or "_wrist" in nm) else None
            out.append((nm, float(n[i]), pos, low))
        self.contacts[f] = out

    # -- one control step on the reference ----------------------------------------------------------------------------
    def tick(self) -> dict:
        D3, L, h = self.D3, self.L, self.h
        self.f += 1
        f = self.f
        h.push(self.frames[min(f, len(self.frames) - 1)])
        h.step()
        if self.delay is None:
            self.delay = D3.ref_delay_f(h)
        x = self.rows[f] = h.sample()
        self.rig.record(x)
        self.x = x
        if self.odo is not None:
            self.odo.update(h, x)
        self.note_contacts(f)
        if self.vid is not None:
            self.vid.step(self, f, x)
        h.sim.k = f + 1  # ha_sim.Sim's step count (its look records it; Sim.step is not used here)
        if float(x["root"][2]) < L.FALL_Z:
            self.fell, self.why = f, "height"
            raise End("fell")
        if self.anchor is not None and not self.walking:
            fr = self.frames[min(max(f - self.delay, self.anchor[2]), len(self.frames) - 1)]
            dev = float(np.linalg.norm((x["root"][:2] - self.anchor[0]) - (fr[:2] - self.anchor[1])))
            if dev > L.LOST_M:
                self.fell = self.lost = f
                self.why = "lost"
                raise End("lost")
        if f >= len(self.frames) - 1:
            self.why = "motion ended"
            raise End("motion ended")
        return x

    def settle(self, open_f: int, grid0: int | None = None) -> tuple[int, bool]:
        """data3's wait: the first row (on the 10 Hz grid of the active reach) from open_f + 0.5 s where pelvis and both
        hand points have been quiet for 0.5 s, else the row at open_f + SETTLE_MAX_F. Ends on that row."""
        D3 = self.D3
        if self.f < 0:
            self.tick()
        while True:
            f = self.f
            on_grid = grid0 is None or (f - grid0) % D3.TICKS == 0
            if f >= open_f + D3.SETTLE_F and on_grid and self.rig.quiet():
                ok = True
                break
            if f >= open_f + D3.SETTLE_MAX_F and on_grid:
                ok = False
                break
            self.tick()
        self.ready = (f, ok)
        return f, ok

    def stand_still(self) -> bool:
        f, ok = self.settle(self.D3.FIRST_OPEN_F)
        self.wait_s = (f - self.D3.FIRST_OPEN_F) / self.D3.FPS
        self.start_odo()
        return ok

    def walk_to(self, xy, yaw, tag: str) -> dict:
        D3, L, HW, x, f = self.D3, self.L, self.HW, self.x, self.f
        p0, y0 = self.pose()
        p0t = x["root"][:2].astype(float).copy()
        info = {}
        t0 = time.time()
        if self.heft_gait:
            wf = heft_walk_frames(self.h, self.frames[f], np.asarray(xy, float), float(yaw), (p0, y0), info, L, HW)
        else:
            wf = HW.walk_frames(self.h, x, self.frames[f], np.asarray(xy, float), float(yaw), pose=(p0, y0), info=info)
        info["plan_s"] = time.time() - t0
        self.plan_s += info["plan_s"]
        nh = int(round(HW.HOLD_END_S * D3.FPS))
        self.frames = np.concatenate([self.frames[: f + 1], wf, np.repeat(wf[-1:], int(D3.HOLD_S * D3.FPS), 0)])
        self.cur_delta = np.r_[self.frames[f, :2] - x["root"][:2], 0.0]
        dl = L.rot_z(y0).T @ np.r_[np.asarray(xy, float) - p0, 0.0]
        w = dict(tag=tag, f=f, asked=[float(dl[0]), float(dl[1]), float(B.wrap(yaw - y0))], goal=[float(xy[0]), float(xy[1]), float(yaw)],
                 plan=dict(n_steps=info.get("n_steps"), walk_s=info.get("walk_s"), ik_err=info.get("ik_err")), dev_cm=0.0)  # fmt: skip
        if "gait" in info:  # --heft-gait only (hg_walk's info has no "gait")
            w["plan"]["gait"] = info["gait"]
        open_f = f + len(wf) - nh + self.delay
        self.walking = True
        anchor = (x["root"][:2].copy(), self.frames[f, :2].copy(), f)
        while self.f < open_f:
            xx = self.tick()
            fr = self.frames[min(max(self.f - self.delay, anchor[2]), len(self.frames) - 1)]
            w["dev_cm"] = max(w["dev_cm"], 100 * float(np.linalg.norm((xx["root"][:2] - anchor[0]) - (fr[:2] - anchor[1]))))
        self.walking = False
        self.anchor = (self.x["root"][:2].copy(), self.frames[self.f, :2].copy(), self.f)
        fs, ok = self.settle(open_f)
        self.wait_s = (fs - open_f) / D3.FPS
        pe, ye = self.pose()
        de = L.rot_z(y0).T @ np.r_[pe - p0, 0.0]
        dt_ = L.rot_z(y0).T @ np.r_[self.x["root"][:2] - p0t, 0.0]
        w.update(real=[float(de[0]), float(de[1]), float(B.wrap(ye - y0))], real_true=[float(dt_[0]), float(dt_[1])],
                 odo_err_cm=100 * float(np.linalg.norm(self.delta())), settled=ok, f_end=fs, wait_s=(fs - open_f) / D3.FPS)  # fmt: skip
        self.walks.append(w)
        return w

    def reach(self, cmd: dict, extra: dict) -> dict:
        """A command on the row the last wait ended on (quiet, or the wait ran out: pre_settled 0; looks take no
        sim time), its frames, then its end label (data3's: the next quiet 10 Hz row, else the wait's last)."""
        D3, L = self.D3, self.L
        f_ready, pre_ok = self.ready
        x, f = self.x, self.f
        assert f == f_ready, (f, f_ready)
        try:
            blend, last, rec = D3.plan_reach(self.h, x, self.frames[f], cmd)
        except (np.linalg.LinAlgError, ValueError) as e:
            self.why = f"ik {type(e).__name__}: {e}"
            raise End("ik")
        self.plan_s += rec["plan_s"]
        rec.update(slot=len(self.reaches), chain=len(self.reaches), f_cmd=f, f_end=f + len(blend), pre_wait_s=self.wait_s,
                   d_prev=self.cur_delta.copy(), f_label=None, settled_end=False, settle_s=-1.0, pre_settled=bool(pre_ok), **extra)  # fmt: skip
        self.frames = np.concatenate([self.frames[: f + 1], blend, L.hold(last, D3.HOLD_S)])
        self.anchor = (x["root"][:2].copy(), self.frames[f, :2].copy(), f)
        self.cur_delta = rec["delta"]
        self.reaches.append(rec)
        fl, ok = self.settle(rec["f_end"] + self.delay, grid0=f)
        rec.update(f_label=fl, settled_end=ok, settle_s=(fl - rec["f_end"]) / D3.FPS)
        self.wait_s = 0.0  # the next command goes on this label row (data3: the label is the next command's row)
        return rec

    def info(self) -> dict:
        w = None
        if self.reaches:
            before = [w for w in self.walks if w["f"] < self.reaches[0]["f_cmd"]]
            w = before[-1] if before else None
        walk = dict(asked=w["asked"] if w else None, real=w.get("real") if w else None, dev_cm=w["dev_cm"] if w else 0.0,
                    plan=w["plan"] if w else None)  # fmt: skip
        return dict(fell=self.fell, lost=self.lost, why=self.why, steps=len(self.rows), walked=bool(self.walks), walk=walk,
                    walks=self.walks, planned=len(self.reaches), wall_s=time.time() - self.t0, plan_s=self.plan_s,
                    delay_f=self.delay, slots=[dict(slot=r["slot"], mode=r["mode"], status="done") for r in self.reaches])  # fmt: skip


# -- the command --------------------------------------------------------------------------------------------------------
def command(L, D3, x: dict, side: str, pg_w, Rg_w, off) -> tuple[dict, dict]:
    """d3's command putting the palm-frame point `off` at pg_w (world) with the palm rotation Rg_w (world)."""
    from scipy.spatial.transform import Rotation

    p = x["root"][:3].astype(float)
    yaw = L.yaw_of(L._qmat(x["root"][3:]))
    Rw = np.asarray(Rg_w, float)  # wrist_yaw_link = palm rotation (d3's PALM_OFF has none)
    hp = np.asarray(pg_w, float) + Rw @ (L.HAND_OFF - D3.PALM_OFF[side] - np.asarray(off, float))
    Rh = L.rot_z(yaw)
    tgt = np.r_[(Rh.T @ np.r_[hp[:2] - p[:2], 0.0])[:2], hp[2]]
    dyaw, pitch, roll = Rotation.from_matrix(Rh.T @ Rw).as_euler("ZYX")
    back = L.rot_z(yaw + dyaw) @ L.rot_y(pitch) @ D3.rot_x(roll)
    assert np.abs(back - Rw).max() < 1e-6, "Z-Y-X decomposition"
    deg = np.degrees([dyaw, pitch, roll])
    rng = dict(dyaw=bool(D3.DYAW_DEG[0] <= deg[0] <= D3.DYAW_DEG[1]), pitch=bool(D3.PITCH_DEG[0] <= deg[1] <= D3.PITCH_DEG[1]),
               roll=bool(D3.ROLL_DEG[0] <= deg[2] <= D3.ROLL_DEG[1]), z=bool(D3.Z_RANGE[0] <= hp[2] <= D3.Z_RANGE[1]))  # fmt: skip
    cmd = dict(mode=side, pelvis_z=None, hands={side: dict(target=tgt, core=False, dyaw=float(dyaw), pitch=float(pitch), roll=float(roll))})
    return cmd, dict(hand_point_w=hp.round(4).tolist(), tgt_h=np.round(tgt, 4).tolist(), deg=np.round(deg, 2).tolist(), in_range=rng,
                     in_table=in_table(D3, side, tgt), grasp_off=list(off))  # fmt: skip


# -- the task flows (ha_bench's, with our walk and reach) ----------------------------------------------------------------
def look(ep: Episode, lk: Look, prompts, tag):
    return lk.look(prompts, tag)


def stand_of(t) -> tuple[np.ndarray, float]:
    S = np.asarray(t["p"][:2]) - B.heading_R(t["yaw"])[:2, :2] @ np.array([t["x"], t["y"]])
    return S, float(t["yaw"])


def approach(ep: Episode, lk: Look, find, prompts, log: dict):
    """ha_bench.Bench.approach: per round the stand point from the latest estimate; within 0.12 m and 5 deg: done;
    else one walk there (hg_walk), a look; up to 3 rounds; a final look at the stand."""
    tgt = lk.target
    for it in range(ROUNDS):
        S, yaw = stand_of(tgt)
        b, by = ep.pose()
        d = S - b
        e = B.wrap(yaw - by)
        if np.linalg.norm(d) < STAND_TOL_M and abs(e) < math.radians(STAND_TOL_DEG):
            break
        short = float(np.linalg.norm(d)) > SHORT_M + STAND_TOL_M
        goal = S - SHORT_M * B.heading_R(yaw)[:2, 0] if short else S
        if ep.vid is not None:
            ep.vid.phase(f"walk approach{it}: " + ("long walk, stops 0.25 m short" if short else "short-step correction")
                         + f" (asked {np.linalg.norm(goal - b):.2f} m)")  # fmt: skip
        w = ep.walk_to(goal, yaw, f"approach{it}")
        w["short"] = short
        lkk, ms = look(ep, lk, prompts, f"approach{it}")
        t = find(lkk, ms, tgt)
        tgt = tgt if t is None else t
        lk.target = tgt
    lkk, ms = look(ep, lk, prompts, "stand")
    t = find(lkk, ms, tgt)
    lk.target = tgt if t is None else t
    S, yaw = stand_of(lk.target)
    b, by = ep.pose()
    lk.stand_err = (B.heading_R(yaw)[:2, :2].T @ (b - S)).round(3).tolist()
    # err is relative to the latest estimate: when the final look saw the target, perception only (the odometry cancels);
    # otherwise it carries the odometry's drift since that look. odo_err_cm (scoring only): the drift now
    log["stand"] = dict(err=lk.stand_err, yaw_err_deg=round(math.degrees(B.wrap(yaw - by)), 2), seen=t is not None,
                        rounds=len(ep.walks), odo_err_cm=round(100 * float(np.linalg.norm(ep.delta())), 1))  # fmt: skip
    return lkk, t is not None


def search(ep: Episode, lk: Look, find, prompts, steps_deg, tag):
    """ha_bench.Bench.turn_until_seen: look; not found: turn (hg_walk, in place) to each heading and look again."""
    lkk, ms = look(ep, lk, prompts, f"{tag}0")
    t = find(lkk, ms, None)
    _, y0 = ep.pose()
    for i, dy in enumerate(steps_deg):
        if t is not None:
            break
        if ep.vid is not None:
            ep.vid.phase(f"search turn {i} ({dy:+.0f} deg)")
        ep.walk_to(ep.pose()[0], y0 + math.radians(dy), f"search{i}")
        lkk, ms = look(ep, lk, prompts, f"{tag}{i + 1}")
        t = find(lkk, ms, None)
    return t


def run_ppbox(ep, lk, log):
    prom = ["cardboard box", "box"]
    lkk, ms = look(ep, lk, prom, "reset")
    t = lk.find_box(lkk, ms, None)
    if t is None:
        return "box not seen"
    lk.target = t
    log["first"] = dict(p=np.round(t["p"], 3).tolist(), h=round(t["h"], 3), stand_x=round(t["x"], 3), edge=t["edge"])
    approach(ep, lk, lk.find_box, prom, log)
    t = lk.target
    faces, a, Rw, half_a = lk.box_faces(t)
    pre = {"right": faces["right"] - B.PRE * a, "left": faces["left"] + B.PRE * a}
    log["perceived"] = dict(box_p=np.round(t["p"], 3).tolist(), pregrasp={s: np.round(v, 3).tolist() for s, v in pre.items()},
                            half_a=round(half_a, 3), h=round(t["h"], 3))  # fmt: skip
    return [("left", pre["left"], Rw, B.G_BOX["left"], faces["left"], t["support"]), ("right", pre["right"], Rw, B.G_BOX["right"], faces["right"], t["support"])]


def run_doubledesk(ep, lk, log):
    prom = ["hammer"]
    t = search(ep, lk, lk.find_hammer, prom, SEARCH["doubledesk"], "hammer")
    if t is None:
        return "hammer not seen"
    lk.target = t
    log["first"] = dict(p=np.round(t["p"], 3).tolist(), stand_x=round(t["x"], 3), edge=t["edge"])
    approach(ep, lk, lk.find_hammer, prom, log)
    t = lk.target
    y = lk.yaw()
    Rw = lk.topdown_R(t["axis"], lk.face(y), lk.face(y + np.pi / 2))
    pre = np.asarray(t["p"], float) + B.PRE * np.array([0, 0, 1.0])
    log["perceived"] = dict(grasp=np.round(t["p"], 3).tolist(), pregrasp=np.round(pre, 3).tolist(), support=t["support"])
    return [("right", pre, Rw, B.G_FIST, np.asarray(t["p"], float), t["support"])]


def run_opendoor(ep, lk, log):
    prom = ["door handle", "door"]
    t = search(ep, lk, lk.find_handle, prom, SEARCH["opendoor"], "handle")
    if t is None:
        return "handle not seen"
    lk.target = t
    log["first"] = dict(p=np.round(t["p"], 3).tolist())
    approach(ep, lk, lk.find_handle, prom, log)
    t = lk.target
    xx = -t["n"]
    yy = np.array([0, 0, -1.0])
    Rw = np.stack([xx, yy, np.cross(xx, yy)], axis=1)  # a hook over the lever: fingers into the door, grasp side down
    pre = np.asarray(t["p"], float) + B.PRE * np.asarray(t["n"], float)
    log["perceived"] = dict(grasp=np.round(t["p"], 3).tolist(), pregrasp=np.round(pre, 3).tolist(), door_plane=t["door_plane"])
    return [("right", pre, Rw, B.G_FIST, np.asarray(t["p"], float), None)]


FLOWS = dict(ppbox=run_ppbox, doubledesk=run_doubledesk, opendoor=run_opendoor)


# -- run ----------------------------------------------------------------------------------------------------------------
def hand_lead(sim) -> dict:
    """Diagnostics: how far ahead of the pelvis (heading frame) the hand and wrist link origins are now, and their
    heights (m)."""
    rb = sim.robot.data
    names = list(rb.body_names)
    pp, pR = sim.pelvis()
    yaw = math.atan2(pR[1, 0], pR[0, 0])
    c, s_ = math.cos(yaw), math.sin(yaw)
    P_ = rb.body_link_pos_w[0].cpu().numpy().astype(np.float64)
    out = {}
    for side in ("left", "right"):
        idx = [i for i, n in enumerate(names) if n.startswith(f"{side}_") and ("hand" in n or "wrist" in n)]
        d = P_[idx] - pp
        x = d[:, 0] * c + d[:, 1] * s_
        out[side] = dict(x_max=round(float(x.max()), 3), body=names[idx[int(np.argmax(x))]], z_min=round(float(P_[idx, 2].min()), 3),
                         z_max=round(float(P_[idx, 2].max()), 3))  # fmt: skip
    return out


def hand_front(h) -> dict:
    """Each hand's forward extent from the robot's own kinematics: the host's MuJoCo mirror (LocoManiLab's G1 + Dex3
    model, posed at the measured joints and fingers by every sample()) -- the most forward point of every geom on the
    side's wrist and hand bodies (mesh vertices; other geoms their bounding sphere) along the pelvis heading, from
    the pelvis origin (m), and those points' height range."""
    mj, m, d = h.mj, h.m, h.d
    pp = d.xpos[h.pelvis].copy()
    Rp = d.xmat[h.pelvis].reshape(3, 3)
    yaw = math.atan2(Rp[1, 0], Rp[0, 0])
    fw = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    out = {}
    for side in ("left", "right"):
        best, body, zlo, zhi = -np.inf, None, np.inf, -np.inf
        for g in range(m.ngeom):
            bn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
            if f"{side}_hand" not in bn and f"{side}_wrist" not in bn:
                continue
            c, Rg = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            if int(m.geom_type[g]) == int(mj.mjtGeom.mjGEOM_MESH):
                a, n = int(m.mesh_vertadr[m.geom_dataid[g]]), int(m.mesh_vertnum[m.geom_dataid[g]])
                V = c + m.mesh_vert[a : a + n].astype(np.float64) @ Rg.T
                xf, z0, z1 = float(((V - pp) @ fw).max()), float(V[:, 2].min()), float(V[:, 2].max())
            else:
                r = float(m.geom_rbound[g])
                xf, z0, z1 = float((c - pp) @ fw) + r, float(c[2]) - r, float(c[2]) + r
            zlo, zhi = min(zlo, z0), max(zhi, z1)
            if xf > best:
                best, body = xf, bn
        out[side] = dict(x_front=round(best, 3), body=body, z_lo=round(zlo, 3), z_hi=round(zhi, 3))
    return out


def link_low(h, name: str):
    """The lowest point (world z, m) of the mirror geoms on body `name` at the mirror's current pose (sample() poses
    it every row), or None when the mirror has no such body."""
    mj, m, d = h.mj, h.m, h.d
    cache = h.__dict__.setdefault("_hr_low", {})
    if name not in cache:
        b = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name)
        cache[name] = [] if b < 0 else [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b]
    gs = cache[name]
    if not gs:
        return None
    lo = np.inf
    for g in gs:
        c, Rg = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        if int(m.geom_type[g]) == int(mj.mjtGeom.mjGEOM_MESH):
            a, k = int(m.mesh_vertadr[m.geom_dataid[g]]), int(m.mesh_vertnum[m.geom_dataid[g]])
            lo = min(lo, float((c + m.mesh_vert[a : a + k].astype(np.float64) @ Rg.T)[:, 2].min()))
        else:
            lo = min(lo, float(c[2]) - float(m.geom_rbound[g]))
    return lo


def reach_contacts(ep, log: dict) -> list[dict]:
    """Per reach (ep.reaches order = the segments' chain), from the command to the end label (the last frame when
    there is none): every body over TOUCH_ALL_N with its origin at >= 0.2 m (the main rule's height), its peak force,
    frames, the lowest point's least height over the perceived support, and the largest horizontal distance of its
    origin from the perceived object (PPBox: the box centre; DoubleDesk / OpenDoor: the grasp point). Scoring only."""
    pc = log.get("perceived") or {}
    obj = pc.get("box_p", pc.get("grasp"))
    last = max(ep.contacts) if ep.contacts else -1
    out = []
    for rc in ep.reaches:
        f0 = int(rc["f_cmd"])
        f1 = int(rc["f_label"]) if rc.get("f_label") is not None else max(last, f0)
        sup = rc.get("support")
        bodies = {}
        for f in range(f0, f1 + 1):
            for nm, n, pos, low in ep.contacts.get(f, ()):
                if pos[2] < 0.2:
                    continue
                b = bodies.setdefault(nm, dict(n=0.0, frames=0, low_over_support=None, dxy_obj=None))
                b["n"] = max(b["n"], round(n, 2))
                b["frames"] += 1
                if low is not None and sup is not None:
                    v = round(low - float(sup), 4)
                    b["low_over_support"] = v if b["low_over_support"] is None else min(b["low_over_support"], v)
                if obj is not None:
                    v = round(float(np.linalg.norm(pos[:2] - np.asarray(obj[:2], float))), 4)
                    b["dxy_obj"] = v if b["dxy_obj"] is None else max(b["dxy_obj"], v)
        out.append(dict(chain=int(rc["chain"]), side=list(rc["hands"])[0], reach_no=rc["reach_no"], support=sup,
                        obj=obj, bodies=bodies))  # fmt: skip
    return out


def in_table(D3, side: str, tgt) -> bool:
    """The hand-point target (pelvis heading frame) inside data3's standing reach-table box at the nearest probed
    height: the region data3's standing targets are drawn from (d3_collect.sample_target)."""
    if not hasattr(D3, "_hr_table"):
        D3._hr_table = json.loads(Path(D3.TABLE).read_text())
    row = D3._hr_table["rows"][side]["stand"]
    zs = np.array(row["z"], float)
    ok = np.array([b is not None for b in row["box"]])
    if not (zs[ok].min() - 0.05 <= tgt[2] <= zs[ok].max() + 0.05):
        return False
    j = int(np.argmin(np.where(ok, np.abs(zs - tgt[2]), np.inf)))
    xlo, xhi, ylo, yhi = row["box"][j]
    return bool(xlo <= tgt[0] <= xhi and ylo <= tgt[1] <= yhi)


def palm_check(sim) -> float:
    """HA's palm link vs wrist_yaw_link + d3's PALM_OFF (no rotation): the largest deviation (m or rad)."""
    import sx_collect as SX

    D3 = SX.D3
    dev = 0.0
    for s in ("left", "right"):
        w, pl = sim.body(f"{s}_wrist_yaw_link"), sim.body(f"{s}_hand_palm_link")
        Rw, Rp = ha_sim.mat_of(w[3:]), ha_sim.mat_of(pl[3:])
        dev = max(dev, float(np.abs(Rw.T @ (pl[:3] - w[:3]) - D3.PALM_OFF[s]).max()), float(np.abs(Rw.T @ Rp - np.eye(3)).max()))
    return dev


def run(a) -> int:
    _orig = ha_sim.sev._configure_renderer_gpu

    def _cfg(args):
        _orig(args)
        SXH.kit_single_gpu(args)

    ha_sim.sev._configure_renderer_gpu = _cfg
    a = ha_sim.finish_args(a, a.ha_task)
    a.no_walker = True
    if a.clear is not None:
        B.CLEAR = float(a.clear)  # stand_on_support reads the module's CLEAR
    elif a.controller in CLEAR_BY:
        B.CLEAR = CLEAR_BY[a.controller]
    seeds, repeats = rng_list(a.seeds), rng_list(a.repeats)
    for sd in seeds:
        ha_sim.check_seed(sd)
    for k in ("data", "data/arm", "log", "looks"):
        (a.out / k).mkdir(parents=True, exist_ok=True)
    t_up = time.time()
    sim = ha_sim.Sim(a)
    SXH.setup_paths()
    sys.path.insert(0, str(HC.W_REPO / "src"))
    import sx_collect as SX

    import hg_walk as HW

    D3, L = SX.D3, SX.L
    D3.SETTLE_MAX_F = int(round(HC.SETTLE_MAX_S * D3.FPS))
    D3.HOLD_S = HC.HOLD_S
    D3.SETTLE_V = HC.SETTLE_V[a.controller]
    if a.controller == "sonic":
        if a.no_camera:
            sim.provider._should_refresh_lerobot_visuals_next_step = lambda: False  # as the collection (looks render)
        h = HRSonic(sim, L, D3, SXH.SONIC_DELAY_F, "native", 0.0, None)
    else:
        h = HRG1(sim, L, D3, a.controller, "lml", None, 0.0, None)
    rig = SX.SRig(h)
    lk = Look(sim, a, a.out / "looks")
    vid = None
    if a.hrvid is not None:  # diagnosis only (hr_video.py, imported only here)
        import hr_video

        vid = hr_video.Video(sim, a.hrvid)
    h.sample()
    tag = f"hr_real {a.controller} {a.ha_task}"
    print(f"{tag}: up in {time.time() - t_up:.0f} s; dt {sim.dt:.3f}; palm vs wrist + PALM_OFF max dev {palm_check(sim):.1e}; "
          f"delay_f {h.ref_delay_f}; root check {rig.check_root():.1e} m; CLEAR {B.CLEAR}", flush=True)  # fmt: skip
    scalars = D3.SCALARS + HC.EXTRA + ("task_id", "seed", "repeat", "reach_no", "hand_no", "pre_settled", "cmd_in_range")
    for sd in seeds:
        for rp in repeats:
            name = f"{a.ha_task}_s{sd}_r{rp}"
            if (a.out / "log" / f"{name}.json").exists():
                continue
            es = ha_sim.episode_seed(a.task, sd, rp)
            rid = RID0 + 1000 * TASK_ID[a.ha_task] + 10 * (sd - SEED0) + rp
            h.ep_seed = es
            h.dev = dict(xy=0.0, z=0.0, rot_deg=0.0)
            log = dict(task=a.ha_task, seed=sd, repeat=rp, episode_seed=es, reset=rid, controller=a.controller)
            ep = Episode(h, rig, SX, D3, L, HW)
            lk.new_episode(name, ep)
            ep.heft_gait = bool(a.heft_gait) and a.controller == "heft"
            if vid is not None:
                ep.vid = vid
                vid.begin(f"{a.controller}_{name}", ep)
            if a.probe_lead:
                ok = ep.stand_still()
                print(f"{tag} {name}: probe-lead settled={ok} wait_s={ep.wait_s:.2f} hand_front={hand_front(h)} "
                      f"link_origins={hand_lead(sim)}", flush=True)  # fmt: skip
                continue
            end = None
            try:
                ep.stand_still()
                log["hand_lead"] = hand_lead(sim)
                log["hand_front"] = hand_front(h)
                log["clear"] = B.CLEAR
                plan = FLOWS[a.ha_task](ep, lk, log)
                if isinstance(plan, str):
                    end = plan
                else:
                    for hand_no, (side, pg, Rw, off, g, support) in enumerate(plan):
                        goals = [(1, np.asarray(pg, float))]
                        if a.via and a.via_rule == "lift" and support is not None:
                            x = ep.xb()
                            pw, Rw0 = x["wr"][side]
                            hp0 = np.asarray(pw, float) + np.asarray(Rw0, float) @ L.HAND_OFF
                            # the via is the hand point's; command() takes the grasp point: shift by the same offset
                            hp_pg = np.asarray(pg, float) + np.asarray(Rw, float) @ (L.HAND_OFF - D3.PALM_OFF[side] - np.asarray(off, float))
                            via_h = np.r_[hp0[:2], max(float(support) + LIFT_CLEAR, float(hp_pg[2]) + VIA_UP)]
                            goals.insert(0, (0, via_h - (hp_pg - np.asarray(pg, float))))
                        elif a.via and a.via_rule != "lift" and (support is not None if a.via_rule == "support" else float(g[2]) >= VIA_Z):
                            x = ep.xb()
                            yaw = L.yaw_of(L._qmat(x["root"][3:]))
                            hd = np.array([math.cos(yaw), math.sin(yaw)])
                            xh = float((np.asarray(pg[:2], float) - x["root"][:2]) @ hd)
                            back = min(VIA_BACK, max(0.0, xh - VIA_XMIN))
                            goals.insert(0, (0, np.r_[np.asarray(pg[:2], float) - back * hd, float(pg[2]) + VIA_UP]))
                        for no, goal in goals:
                            cmd, cinfo = command(L, D3, ep.xb(), side, goal, Rw, off)
                            ok = all(cinfo["in_range"].values())
                            if ep.vid is not None:
                                ep.vid.phase(f"reach {len(ep.reaches)} ({side} hand, {'lift' if no == 0 else 'pre-grasp'})"
                                             + (": first reach" if not ep.reaches else ""))  # fmt: skip
                            rec = ep.reach(cmd, dict(reach_no=no, hand_no=hand_no, cmd_info=cinfo, cmd_in_range=ok, support=support))
                            log.setdefault("reaches", []).append(dict(side=side, reach_no=no, f_cmd=rec["f_cmd"], f_label=rec["f_label"],
                                                                      pre_settled=rec["pre_settled"], settled_end=rec["settled_end"],
                                                                      ik_err_cm=round(100 * rec["hands"][side]["ik_err"], 2), **cinfo))  # fmt: skip
                    end = "reaches done"
            except End as e:
                end = str(e)
            except Exception as e:  # a harness error: record it, keep Kit for the next episode
                traceback.print_exc()
                end = f"error {type(e).__name__}: {e}"
            info = ep.info()
            segs = []
            if ep.reaches:
                segs = D3.segments(ep.rows, info, ep.frames, ep.reaches, rid, h.pl, rig)
                SX.add_fields(segs, ep.reaches)
                HC.settle_fields(segs, ep.rows, ep.reaches, L, D3)
                info["touch"] = SXH.add_touch(h, segs, ep.reaches)
                log["contacts"] = reach_contacts(ep, log)
                for sg in segs:
                    rc = ep.reaches[sg["chain"]]
                    sg.update(task_id=TASK_ID[a.ha_task], seed=sd, repeat=rp, reach_no=rc["reach_no"], hand_no=rc["hand_no"],
                              pre_settled=rc["pre_settled"], cmd_in_range=rc["cmd_in_range"])  # fmt: skip
                HC.save_chunk(SX, D3, segs, a.out / "data", name, rig, scalars)
            log.update(end=end, why=info["why"], segments=len(segs), looks=lk.looks, info=info)
            if vid is not None:
                vid.end(str(end))
            (a.out / "log" / f"{name}.json").write_text(json.dumps(log, default=D3._json, indent=1))
            print(f"{tag} {name}: end={end!r} why={info['why']!r} walks={len(ep.walks)} reaches={len(ep.reaches)} segs={len(segs)} "
                  f"stand={log.get('stand')} reach={[(r['side'], r['reach_no'], r['pre_settled'], r['settled_end'], r['ik_err_cm'], all(r['in_range'].values())) for r in log.get('reaches', [])]} "
                  f"steps={info['steps']} wall={info['wall_s']:.0f} s", flush=True)  # fmt: skip
    print(f"{tag}: DONE", flush=True)
    return 0


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        rc = run(args)
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    os._exit(rc)
