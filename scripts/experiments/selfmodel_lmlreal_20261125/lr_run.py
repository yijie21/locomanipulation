"""selfmodel_lmlreal_20261125: hardgrasp's LocoManiLab task episodes, the direct group only (hg_run `alone`), with every
reach recorded in selfmodel_data3's segment layout, so any end model can be evaluated offline. 2026-11-25.

What runs is hg_run.episode (selfmodel_hardgrasp_20261125, frozen file, imported, not edited) copied here with additions
marked "lmlreal:". Same tasks (hardgrasp's 7 formal ones), episode ids (0..29), our jitter, the cached look (hardgrasp's
looks/: the same perception for every controller), stance, walk, via point, pre-grasp, entry, close, lift. The command of
every reach is hg_run's direct one (the goal itself); no model changes anything. hg_run plans through
lml_tasks.Predictor.features -> HostDriver.plan (lml_selfmodel.OnlineDriver._plan); `plan_rec` below is that function
line for line, plus the IK waypoints d3_collect records (RT, RQ, RP). The hostdata model (hardgrasp's wm_s0.pt) still
predicts each reach as in hardgrasp (pred_exec_h; it changes nothing), so a ScaleBFM episode can be compared with
hardgrasp's archived one field for field.

Controllers: scalebfm (HostDriver's default), heft (HostDriver backend "heft"), sonic (d3_sonic.SonicHost, CPU inference
unless D3_SONIC_INFER says otherwise) -- the same task driver (hg_run.HGDriver) over each host.

Delay alignment (data3 revision 1: the robot executes a pushed frame ref_delay_f later -- ScaleBFM 7, HEFT 8, SONIC 47
frames). hg_run's schedule was written for ScaleBFM; here every wait that follows a reference motion is lengthened by
shift = ref_delay_f - 7 frames (ScaleBFM 0: hardgrasp's schedule exactly): the look, the first reach after a walk, the
next reach, the close, the end of the hold; and the lost rule compares the robot with the reference shift frames behind
(ScaleBFM: hg_run's rule exactly). The hand (Dex3) is not delayed (written straight to its joint targets).

Recording (d3_collect's Rig on every row; d3_collect.segments for the layout): per reach, rows every 5 steps from 0.3 s
before the command to the end label -- S, W, J, C, HQ, KP -- the command (target in the measured pelvis heading frame,
rot6, pelvis_z), the reference (ref_q, ref_pel, ref_s, RT / RQ / RP), and the label: the first 10 Hz row (from the
command) at least 0.5 s after the reference end + ref_delay_f where pelvis and both hand points have been slower than
2 cm/s for 0.5 s (data3's settle rule, d3_collect.Rig.quiet), searched up to the end of the reach's window -- the next
reach's command row (via, pre-grasp) or the close row (entry). Not settled by then: the last 10 Hz row inside the window,
settled_end 0. Added per segment: reach_no (0 via point, 1 pre-grasp = 第一下, 2 entry = 第二下; also in `slot`), task_id
(TASKS index), ep, pre_settled (data3's rule true at the command row), pre_v_pel_cms / pre_v_hand_cms (acting = right) /
pre_v_other_cms / pre_v_max_cms (each point's largest 0.1 s speed over the 0.5 s before the command), live_err_cm (the
hand point to the planned world target at the label row, from the episode's own row, for the check), backend_id,
furn_rows / obj_rows (50 Hz rows after the command up to the label with the right hand / wrist / elbow on a non-target
scene body -- hg_run's furniture contact -- / the right hand or wrist on the target object; observation only).
`walked` = 1 on every segment of an episode that walked (data3's meaning); `chain` = reaches done before in the episode
(data3's meaning; the entry after a via point is 2 -- an evaluator clips it to its training range); reset =
RESET0 + 1000 task_id + ep; pre_wait_s -1 (no settle wait: hardgrasp's fixed schedule).

    cd $G; PYTHONPATH=$G:$W/src python lr_run.py run --backend scalebfm --task T --eps 0:30 --shard K --shards N
    python lr_run.py merge --backend scalebfm
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
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(EXP / "selfmodel_data3_20261125"), str(EXP / "hardgrasp_20261125"), str(EXP / "locomanilab_20261117")]
import d3_collect as D3  # noqa: E402
import hg_run as HG  # noqa: E402
import lml_host as LH  # noqa: E402
import lml_selfmodel as L  # noqa: E402
import lml_tasks as LT  # noqa: E402

OUT = Path(os.environ.get("LR_OUT", "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlreal_20261125"))
HG_OUT = HG.OUT  # hardgrasp's output (read only here: looks/, wm_s0.pt, train_q_range.npz, episodes/ for the check)
TASKS = (  # hardgrasp's formal tasks (revision 1, point 3), in this fixed order (task_id)
    "IThorBathroomLiftCandleFromFloor",
    "IThorKitchenApproachTableAndLiftApple",
    "IThorKitchenApproachTableAndLiftBowl",
    "IThorKitchenApproachTableAndLiftSaltShaker",
    "IThorKitchenTakeStatueFromCabinet",
    "IThorLivingRoomApproachTableAndLiftDumbbell",
    "IThorLivingRoomApproachTableAndLiftWateringCan",
)
BACKENDS = ("scalebfm", "heft", "sonic")
BASE_DELAY_F = 7  # ScaleBFM's buffer delay (0.14 s): hg_run's schedule is written for it
REACH_NO = {-1: 0, 0: 1, 1: 2}  # hg_run's k -> 0 via point, 1 pre-grasp, 2 entry
RESET0 = 9_100_000
EXTRA = ("reach_no", "task_id", "ep", "pre_settled", "pre_v_pel_cms", "pre_v_hand_cms", "pre_v_other_cms",
         "pre_v_max_cms", "live_err_cm", "backend_id", "furn_rows", "obj_rows")  # fmt: skip
SCALARS_LR = D3.SCALARS + EXTRA


# -- the drivers ----------------------------------------------------------------------------------------------------
class HeftBase(LH.HostDriver):
    """HostDriver with the host's own HEFT backend (heft/pmg), as d3_collect.make_host builds it."""

    def __init__(self, env_name: str = "EmptyEnv", seed: int = 0, **kw):
        super().__init__(env_name=env_name, backend="heft", seed=seed, **kw)


def plan_rec(self, x: dict, ref_w: np.ndarray, c: dict):
    """lml_selfmodel.OnlineDriver._plan (HostDriver.plan), line for line, plus the IK waypoints: RT (each waypoint's time
    after the command), RQ (joints, MuJoCo order), RP (pelvis, robot_fk convention), measured coordinates."""
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    yaw = L.yaw_of(L._qmat(quat))
    Rh = L.rot_z(yaw)
    target_w = np.r_[p[:2] + (Rh @ np.r_[c["target"][:2], 0.0])[:2], c["target"][2]]
    rot_w = L.rot_z(yaw + c["dyaw"]) @ L.rot_y(c["pitch"])
    q_mj = x["q"][self.to_mj].astype(float)
    t0 = time.time()
    poses, durs, ik_err, dur = self.pl.reach((p, quat, q_mj), c["side"], target_w, rot_w, c["pelvis_z"])
    self.plan_s += time.time() - t0
    delta = np.r_[ref_w[:2] - p[:2], 0.0]
    ref_pose = (ref_w[:3].astype(float), ref_w[3:7].astype(float), ref_w[7:].astype(float))
    shifted = [ref_pose] + [(pp + delta, qq, jj) for pp, qq, jj in poses[1:]]
    last = poses[-1]
    rec = dict(side=c["side"], target_w=target_w.tolist(), rot_w=rot_w.tolist(), dyaw=c["dyaw"], pitch=c["pitch"],
               pelvis_z=c["pelvis_z"], cmd_target=c["target"].tolist(), ik_err=ik_err, reach_s=dur,
               ref_q=last[2].tolist(), ref_pel=np.r_[last[0], last[1]].tolist(), delta=delta.tolist(),
               RT=np.r_[0.0, np.cumsum(durs)], RQ=np.array([pp[2] for pp in poses]),
               RP=np.array([np.r_[pp[0], pp[1]] for pp in poses]))  # fmt: skip
    return L.blend_frames(shifted, durs), shifted[-1], rec


def make_driver(backend: str, task: str):
    if backend == "sonic":
        import d3_sonic

        bases = (HG.HGDriver, d3_sonic.SonicHost)
    elif backend == "heft":
        bases = (HG.HGDriver, HeftBase)
    else:
        bases = (HG.HGDriver,)
    cls = type(f"LR_{backend}", bases, {"plan": plan_rec})
    return cls(task)


# -- one episode (hg_run.episode + lmlreal) ---------------------------------------------------------------------------
def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def episode(dr, rig, ep: int, P, backend: str) -> tuple[dict, list[dict]]:
    import hg_look as HL
    import hg_walk as HW

    t0 = time.time()
    spec = HG.TASKS[dr.task]
    method = "alone"
    jit = dr.reset_hg(ep)
    rig.clear()  # lmlreal
    frames = dr.start_frames()
    f_look = int(round(L.START_HOLD_S * L.FPS))
    if spec["kind"] == "approach":  # the look lean (same for both groups)
        n_l, n_a = int(round(HG.LEAN_S * L.FPS)), int(round(HG.LOOK_AFTER_S * L.FPS))
        f0 = 5
        frames = np.concatenate([frames, np.repeat(frames[-1:], n_l + n_a + 10, axis=0)])
        for j in range(f0, len(frames)):
            frames[j, 7 + HG.WAIST_PITCH] = L.DEFAULT_Q[HG.WAIST_PITCH] + HG.LOOK_LEAN * L.minjerk(min(1.0, (j - f0) / n_l))
        f_look = max(f_look, f0 + n_l + n_a)
    rows: dict[int, dict] = {}
    R = dict(task=dr.task, ep=ep, method=method, backend=backend, jitter=jit, obj_name=dr.obj_name, scene=dr.scene)
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
    reset_id = 9_000_000 + max(ep, 0)  # hg_run's (the hostdata model's features)
    rlo, rhi = HG.train_range()
    # lmlreal: delay alignment, the d3 reach records, per-point settle speeds, the walk
    delay = shift = None
    t_max = 60 * L.FPS
    d3r: list[dict] = []
    cur_delta = np.zeros(3)
    active = None
    vq: deque = deque(maxlen=D3.SETTLE_F)
    lost = None
    walking = False
    wrec = dict(asked=None, real=None, dev_cm=0.0)
    x_walk0 = None

    def close_window(fe: int) -> None:
        """The pending reach's window ends at row fe: no settled label yet -> the last 10 Hz row inside it."""
        nonlocal active
        if active is not None and active["f_label"] is None:
            g = active["f_cmd"] + ((fe - active["f_cmd"]) // D3.TICKS) * D3.TICKS
            hw = rows[g]["wr"]["right"]
            active.update(f_label=g, settled_end=False, settle_s=(g - active["f_end"]) / L.FPS,
                          live_err_cm=100 * float(np.linalg.norm(hw[0] + hw[1] @ L.HAND_OFF - active["hands"]["right"]["target_w"])))  # fmt: skip
        active = None

    while True:
        dr.push(frames[min(f, len(frames) - 1)])
        if dr.buf is None:
            dr.buf = dr._find_buffer()
        dr.step()
        if delay is None:  # lmlreal
            delay = D3.ref_delay_f(dr)
            shift = delay - BASE_DELAY_F
            if shift > 0:
                frames = np.concatenate([frames, np.repeat(frames[-1:], shift + 10, axis=0)])
                f_look += shift
                t_max += 6 * shift
        if f == 1 and getattr(getattr(dr, "comp", None), "_settling", False):
            raise RuntimeError("the bridge is still settling: our reference is not being tracked")
        x = rows[f] = dr.sample()
        rig.record(x)  # lmlreal
        vq.append(np.linalg.norm(rig.pts[-1] - rig.pts[0], axis=1) / (D3.TICKS * L.DT) if len(rig.pts) > D3.TICKS else np.full(3, np.inf))
        if odo is not None:
            odo.update(dr, x)
        if float(x["root"][2]) < L.FALL_Z:
            fell, why = f, f"fell during {phase}"
            break
        if anchor is not None:  # lmlreal: the reference `shift` frames behind (0 for ScaleBFM = hg_run)
            fr = frames[min(max(f - shift, anchor[2]), len(frames) - 1)]
            dev = (x["root"][:2] - anchor[0]) - (fr[:2] - anchor[1])
            if walking:
                wrec["dev_cm"] = max(wrec["dev_cm"], 100 * float(np.linalg.norm(dev)))
            if float(np.linalg.norm(dev)) > L.LOST_M:
                fell, why = f, f"lost during {phase}"
                lost = f
                break
        # lmlreal: the pending reach's end label (data3's settle rule, on its 10 Hz grid)
        if active is not None and active["f_label"] is None:
            if f >= active["f_end"] + delay + D3.SETTLE_F and (f - active["f_cmd"]) % D3.TICKS == 0 and rig.quiet():
                hw = x["wr"]["right"]
                active.update(f_label=f, settled_end=True, settle_s=(f - active["f_end"]) / L.FPS,
                              live_err_cm=100 * float(np.linalg.norm(hw[0] + hw[1] @ L.HAND_OFF - active["hands"]["right"]["target_w"])))  # fmt: skip
        # lmlreal: contacts during the reach (after its command row, up to its label row): the right hand / wrist / elbow on
        # a non-target scene body (hg_run's furniture hit), the right hand / wrist on the target object
        if active is not None and f > active["f_cmd"] and (active["f_label"] is None or active["f_label"] == f):
            active["furn_rows"] += int(dr.arm_scene_contact() is not None)
            active["obj_rows"] += int(dr.obj_contacts()[0])
        # scoring (truth)
        z = float(dr.d.xpos[dr.obj_bid][2]) - dr.obj_z0
        if "close" in sched and f >= sched["close"]:
            hand_c, world_c = dr.obj_contacts()
            rise_max = max(rise_max, z)
            lift_run = lift_run + 1 if (z >= HG.LIFT_MIN and not world_c) else 0
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
            lp = HG.look_path(dr.task, ep)
            lk = HL.load(lp)
            pel = np.asarray(lk["pelvis"], float)
            R["look_state_dev_mm"] = 1000 * float(np.abs(pel[:3] - x["root"][:3]).max())
            R["look"] = dict(status=lk["meta"].get("status"), mask_score=float(lk["mask_score"]), n_obj=len(lk["obj_pts"]),
                             n_grasps=len(lk["grasp_T"]), prompt=lk["meta"].get("prompt_used"))  # fmt: skip
            tp = dr.obj_pos()
            R["look"]["obj_center_err_cm"] = 100 * float(np.linalg.norm(np.median(lk["obj_pts"], 0) - tp)) if len(lk["obj_pts"]) else None
            p0, yaw0 = x["root"][:2].copy(), L.yaw_of(L._qmat(x["root"][3:]))
            odo = HW.LegOdo(dr, x, (p0, yaw0))
            plan = HG.plan_from_look(lk, spec["kind"], x["root"][:3], yaw0)
            R["plan"] = HG._js({k: v for k, v in plan.items() if k not in ("stance",)})
            R["plan"]["stance"] = HG._js(plan.get("stance"))
            if plan["fail"]:
                why = plan["fail"]
                break
            st = plan["stance"]
            d_move = float(np.linalg.norm(st["xy"] - p0))
            d_turn = abs(wrap(st["yaw"] - yaw0))
            R["move"] = dict(dist=d_move, turn_deg=math.degrees(d_turn))
            if d_move >= HG.MOVE_MIN or d_turn >= HG.TURN_MIN:
                wf = HW.walk_frames(dr, x, frames[f], st["xy"], st["yaw"], pose=odo.pose(), info=walk_info)
                frames = np.concatenate([frames[: f + 1], wf] + ([np.repeat(wf[-1:], shift + 10, 0)] if shift > 0 else []))
                anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                first = "cmd0" if plan["via"] is not None else "cmd1"
                sched[first] = f + len(wf) - 25 + shift
                phase = "walk"
                n_walks = 1
                # lmlreal: the walk asked (start heading frame), the reference shift, the walk's deviation
                dp = L.rot_z(yaw0).T @ np.r_[st["xy"] - p0, 0.0]
                wrec["asked"] = [float(dp[0]), float(dp[1]), wrap(float(st["yaw"]) - yaw0)]
                cur_delta = np.r_[frames[f, :2] - x["root"][:2], 0.0]
                x_walk0, walking = x, True
            else:
                frames = np.concatenate([frames[: f + 1], L.hold((frames[f][:3], frames[f][3:7], frames[f][7:]), 2.0 + shift / L.FPS)])
                first = "cmd0" if plan["via"] is not None else "cmd1"
                sched[first] = f + int(round(L.PRE_S * L.FPS))
        # -- reaches ----------------------------------------------------------------------------------------------
        if first and sched.get(first) == f and phase == "walk" and n_walks < 1 + HG.WALK_FIX:
            pose = odo.pose()
            st = plan["stance"]
            d_left = float(np.linalg.norm(st["xy"] - pose[0]))
            R.setdefault("walk_left_cm", []).append(100 * d_left)
            if d_left >= HG.FIX_MIN and n_walks < HG.WALK_FIX + 1:
                wf = HW.walk_frames(dr, x, frames[f], st["xy"], st["yaw"], pose=pose)
                frames = np.concatenate([frames[: f + 1], wf] + ([np.repeat(wf[-1:], shift + 10, 0)] if shift > 0 else []))
                anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                sched[first] = f + len(wf) - 25 + shift
                n_walks += 1
                cur_delta = np.r_[frames[f, :2] - x["root"][:2], 0.0]  # lmlreal
        for k, key in ((-1, "cmd0"), (0, "cmd1"), (1, "cmd2")):
            if sched.get(key) == f:
                phase = f"reach{k + 1}"
                close_window(f)  # lmlreal: the previous reach's window ends at this command
                walking = False
                if x_walk0 is not None and wrec["real"] is None:
                    pw0, yw0 = x_walk0["root"][:2].astype(float), L.yaw_of(L._qmat(x_walk0["root"][3:]))
                    dp = L.rot_z(yw0).T @ np.r_[x["root"][:2] - pw0, 0.0]
                    wrec["real"] = [float(dp[0]), float(dp[1]), wrap(L.yaw_of(L._qmat(x["root"][3:])) - yw0)]
                goal = plan["via"] if k == -1 else plan["pre"] if k == 0 else plan["goal"]
                kc = 0 if k == -1 else min(1, k + (plan["via"] is not None))  # the chain index the self model sees
                pose = odo.pose()
                c, gh = HG.command_b(pose, goal, plan["ad"], plan["pz"])
                info = dict(k=k, goal_w=goal.tolist(), goal_h=gh.tolist(), dyaw_deg=math.degrees(c["dyaw"]),
                            pitch_deg=math.degrees(c["pitch"]), pz=c["pelvis_z"], pose_odo=[pose[0].tolist(), pose[1]],
                            pose_true=[x["root"][:2].tolist(), L.yaw_of(L._qmat(x["root"][3:]))])  # fmt: skip
                qx = x["q"]
                out_j = int(((qx < rlo) | (qx > rhi)).sum())
                info.update(q_out=out_j, q_in_range=out_j == 0)
                tic = time.time()
                try:
                    (blend, last, rec), Dx = P.features(dr, rows, f, frames, c, kc, reset_id)
                    info["pred_exec_h"] = P.predict(Dx)[0].tolist()
                    info["feat_x0"], info["feat_u"] = Dx["x0"][0].tolist(), Dx["u"][0].tolist()  # lmlreal: for the check
                except (np.linalg.LinAlgError, ValueError) as e:
                    why = f"reach: plan failed {type(e).__name__}"
                    info["plan_error"] = str(e)[:200]
                    reaches.append(info)
                    fell = None
                    break
                info["think_s"] = time.time() - tic
                rec.update(chain=kc, f_cmd=f, f_end=f + len(blend))
                post = int(round(L.POST_S * L.FPS))
                frames = np.concatenate([frames[: f + 1], blend, L.hold(last, L.POST_S + L.PRE_S + L.MARGIN_S + 0.5 + shift / L.FPS)])
                anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                info.update(cmd_h=np.asarray(c["target"]).tolist(), f_cmd=f, f_end=rec["f_end"], f_win=rec["f_end"] + post + shift,
                            ik_err_cm=100 * float(rec["ik_err"]), p_cmd=x["root"][:3].tolist(), yaw_cmd=L.yaw_of(L._qmat(x["root"][3:])))  # fmt: skip
                reaches.append(info)
                # lmlreal: the reach in d3_collect's record form (one hand, roll 0)
                vv = np.array(vq)
                rc = dict(mode="right", pelvis_z=rec["pelvis_z"], reach_s=rec["reach_s"], ref_q=np.asarray(rec["ref_q"]),
                          ref_pel=np.asarray(rec["ref_pel"]), delta=np.asarray(rec["delta"]), RT=rec["RT"], RQ=rec["RQ"], RP=rec["RP"],
                          hands={"right": dict(target_w=np.asarray(rec["target_w"]), rot_w=np.asarray(rec["rot_w"]), dyaw=rec["dyaw"],
                                               pitch=rec["pitch"], roll=0.0, core=False, cmd_target=np.asarray(rec["cmd_target"]),
                                               ik_err=rec["ik_err"])},
                          slot=REACH_NO[k], chain=len(d3r), f_cmd=f, f_end=rec["f_end"], pre_wait_s=-1.0, d_prev=cur_delta.copy(),
                          f_label=None, settled_end=False, settle_s=-1.0, live_err_cm=-1.0, pre_settled=bool(rig.quiet()),
                          pre_v_pel_cms=100 * float(vv[:, 0].max()), pre_v_hand_cms=100 * float(vv[:, 1].max()),
                          pre_v_other_cms=100 * float(vv[:, 2].max()), pre_v_max_cms=100 * float(vv.max()), furn_rows=0, obj_rows=0)  # fmt: skip
                cur_delta = rc["delta"].copy()
                d3r.append(rc)
                active = rc
                if k == -1:
                    sched["cmd1"] = rec["f_end"] + post + int(round(L.PRE_S * L.FPS)) + shift
                elif k == 0:
                    sched["cmd2"] = rec["f_end"] + post + int(round(L.PRE_S * L.FPS)) + shift
                else:
                    sched["close"] = rec["f_end"] + post + shift
        if why.startswith("reach: plan failed"):
            break
        if sched.get("close") == f:
            close_window(f)  # lmlreal: the entry's window ends at the close
            phase = "close"
            hp = dr.hand_point(x)
            R["close_dist_cm"] = 100 * float(np.linalg.norm(hp - plan["goal"]))
            R["close_obj_cm"] = 100 * float(np.linalg.norm(hp - dr.obj_pos()))
            sched["lift"] = f + int(round((HG.CLOSE_S + HG.SETTLE_S) * L.FPS))
        if "close" in sched and f >= sched["close"] and dr.grip < HG.GRIP:
            dr.grip = min(HG.GRIP, (f - sched["close"] + 1) / (0.3 * L.FPS))
        if sched.get("lift") == f:
            phase = "lift"
            pose = odo.pose()
            goal = plan["goal"] + np.array([0.0, 0.0, HG.LIFT])
            c, gh = HG.command_b(pose, goal, plan["ad"], None)
            try:
                blend, last, rec = dr.plan(x, frames[f], c)
            except (np.linalg.LinAlgError, ValueError):
                why = "lift: plan failed"
                break
            frames = np.concatenate([frames[: f + 1], blend, L.hold(last, HG.HOLD_S + L.MARGIN_S + 0.5 + shift / L.FPS)])
            anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
            sched["end"] = f + len(blend) + int(round(HG.HOLD_S * L.FPS)) + shift
            R["lift_cmd"] = dict(goal_h=gh.tolist(), ik_err_cm=100 * float(rec["ik_err"]))
        if sched.get("end") is not None and f >= sched["end"]:
            break
        if f >= len(frames) - 1:
            why = why or "motion ended"
            break
        if f > t_max:
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
        R["walk"] = HG._js({k: v for k, v in walk_info.items() if k in ("walk_s", "n_steps", "ik_err", "knots")})
    if first and first in sched and odo is not None and 0 < sched[first] <= max(rows):
        xr = rows[min(sched[first], max(rows))]
        st = plan["stance"]
        R["arrive"] = dict(pos_err_cm=100 * float(np.linalg.norm(xr["root"][:2] - st["xy"])),
                           yaw_err_deg=math.degrees(abs(wrap(L.yaw_of(L._qmat(xr["root"][3:])) - st["yaw"]))))  # fmt: skip
    own_ok = official_at is not None
    lift10 = lift_best >= HG.LIFT_HOLD
    success = own_ok if spec["own"] else lift10
    R.update(success=bool(success), official=own_ok, official_step=official_at, lift10=bool(lift10), rise_max_cm=100 * rise_max,
             fell=fell, why=why, steps=len(rows), n_walks=n_walks, reaches=reaches, n_contacts=len(contact_hits),
             first_contact=contact_hits[0] if contact_hits else None, wall_s=round(time.time() - t0, 1))  # fmt: skip
    R["reason"] = HG.classify(R, phase)
    # lmlreal: the segments
    info3 = dict(fell=fell, lost=lost, walk=wrec, walked=n_walks > 0, delay_f=delay)
    segs = D3.segments(rows, info3, frames, d3r, RESET0 + 1000 * TASKS.index(dr.task) + max(ep, 0), dr.pl, rig)
    for s in segs:
        rc = d3r[s["chain"]]
        s.update(reach_no=rc["slot"], task_id=TASKS.index(dr.task), ep=ep, backend_id=BACKENDS.index(backend),
                 **{k: rc[k] for k in ("pre_settled", "pre_v_pel_cms", "pre_v_hand_cms", "pre_v_other_cms", "pre_v_max_cms", "live_err_cm",
                                       "furn_rows", "obj_rows")})  # fmt: skip
    R.update(delay_f=delay, shift_f=shift, walk_rec=wrec, n_segments=len(segs),
             d3=[dict(reach_no=rc["slot"], chain=rc["chain"], f_cmd=rc["f_cmd"], f_end=rc["f_end"], f_label=rc["f_label"],
                      settled_end=rc["settled_end"], settle_s=rc["settle_s"], pre_settled=rc["pre_settled"],
                      pre_v_max_cms=rc["pre_v_max_cms"], live_err_cm=rc["live_err_cm"], furn_rows=rc["furn_rows"],
                      obj_rows=rc["obj_rows"]) for rc in d3r])  # fmt: skip
    for s in segs:
        R["d3"][s["chain"]].update(jaw_err_cm=float(s["jaw_err_cm"]), seg_fell=bool(s["fell"]))
    return R, segs


# -- runners ----------------------------------------------------------------------------------------------------------
def ep_path(backend: str, task: str, ep: int) -> Path:
    return OUT / backend / "episodes" / task / f"ep{ep:+05d}.json"


def run(backend: str, task: str, eps: list[int], shard: int, shards: int) -> None:
    import torch

    torch.set_num_threads(1)
    jobs = eps[shard::shards]
    todo = [e for e in jobs if not ep_path(backend, task, e).exists()]
    print(f"lr {backend} {task} shard {shard}/{shards}: {len(jobs) - len(todo)}/{len(jobs)} done", flush=True)
    if not todo:
        return
    dr = make_driver(backend, task)
    rig = D3.Rig(dr)
    print(f"lr {backend} {task}: pelvis offset c {np.round(dr.c, 4).tolist()}, root check {rig.check_root():.2e} m", flush=True)
    P = LT.Predictor(HG.WM)
    segd = OUT / backend / "segs" / task
    for e in todo:
        try:
            r, segs = episode(dr, rig, e, P, backend)
        except Exception as err:  # recorded, not retried
            r, segs = dict(task=task, ep=e, method="alone", backend=backend, error=f"{type(err).__name__}: {err}",
                           tb=traceback.format_exc()[-2000:], success=False, reason="other"), []  # fmt: skip
            print(f"lr {backend} {task} ep {e}: ERROR {err}", flush=True)
            dr = make_driver(backend, task)
            rig = D3.Rig(dr)
        D3.save_chunk(segs, segd, f"ep{e:+05d}", rig, SCALARS_LR)
        p = ep_path(backend, task, e)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, default=LT._json))
        os.replace(tmp, p)
        print(f"lr {backend} {task} ep {e}: success {r.get('success')} reason {r.get('reason')} why {r.get('why')} segments "
              f"{r.get('n_segments')} labels {[(d['reach_no'], d['settled_end'], round(d.get('jaw_err_cm', -1), 1)) for d in r.get('d3', [])]} "
              f"{r.get('wall_s')} s", flush=True)  # fmt: skip


def merge(backend: str) -> None:
    """segs/<task>/ep*.npz (+ arm/) -> data/all.npz and data/arm.npz (data.py's layout), tasks in TASKS order, then ep."""
    base = OUT / backend
    names = []
    for t in TASKS:
        names += sorted((base / "segs" / t).glob("ep*.npz"), key=lambda p: int(p.stem[2:]))
    d = base / "data"
    d.mkdir(parents=True, exist_ok=True)
    for src, dst in ((None, d / "all.npz"), ("arm", d / "arm.npz")):
        zs = [dict(np.load(p if src is None else p.parent / "arm" / p.name)) for p in names]
        keep = ("radius", "link")
        allz = {k: (zs[0][k] if k in keep else np.concatenate([z[k] for z in zs])) for k in zs[0]}
        tmp = dst.with_name(f".{dst.stem}.tmp.npz")
        np.savez_compressed(tmp, **allz)
        os.replace(tmp, dst)
    z = np.load(d / "all.npz")
    print(f"{backend}: {len(names)} episode files, {len(z['n_steps'])} segments, {int(z['n_steps'].sum())} rows, fell/lost "
          f"{int(z['fell'].sum())}, settled at the label {int(z['settled_end'].sum())}; reach_no counts "
          f"{dict(zip(*[v.tolist() for v in np.unique(z['reach_no'], return_counts=True)]))}")  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "merge"])
    ap.add_argument("--backend", choices=BACKENDS, default="scalebfm")
    ap.add_argument("--task", default=TASKS[0])
    ap.add_argument("--eps", default="0:30")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    if a.phase == "merge":
        merge(a.backend)
    else:
        lo, hi = (int(v) for v in a.eps.split(":"))
        run(a.backend, a.task, list(range(lo, hi)), a.shard, a.shards)
