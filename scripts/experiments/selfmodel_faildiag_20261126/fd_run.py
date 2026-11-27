"""selfmodel_faildiag_20261126: failure diagnosis on lmlcmp3's LocoManiLab task episodes (measurement only).

Everything imports lmlcmp3's frozen private copy (loco_outputs/selfmodel_lmlcmp3_20261126/code, LC_CODE_ROOT, its code
check stays on); nothing in it is edited. Output: $FD_OUT (default loco_outputs/selfmodel_faildiag_20261126).

Groups (`--group`):
  rerun     lmlcmp3's episode (lc3_run.episode, unchanged) for one of lmlcmp3's methods, with a step log (below) added
            through the driver's step / push (read only: the simulation is untouched). The record and the segments are
            compared with lmlcmp3's own, field for field, by `fd_run.py check` (wall_s and think_s aside).
  servo     MEASUREMENT TOOL. The perception grasp, the hand brought to it by a truth servo: up to and including the
            first pre-grasp command, exactly lmlcmp3's `alone` (so the episode is bit-identical to alone until then);
            after each pre-grasp / entry command has landed, the simulator's hand point and wrist frame are compared with
            the planned pre-grasp / grasp pose (the look's GraspGen-X wrist frame, roll included); while the error is
            > SERVO_P (0.5 cm) or > SERVO_R (5 deg), at most SERVO_N (4) more commands of that reach: the world target
            moved by the remaining position error, the wrist target rotated by the remaining rotation (a full rotation
            given to the planner's IK, which the command form cannot express), planned from the measured state as
            always. Damped: each correction moves the target by at most SERVO_STEP_P (4 cm) and turns it by at most
            SERVO_STEP_R (15 deg); a correction that leaves the error worse than the best try (score cm + deg / 5,
            by > 0.5) is answered by re-sending the best try once, then no more corrections of that reach. Then the
            close and the lift, unchanged. The time limit is lengthened by 10 x 6 s.
  gtgrasp   MEASUREMENT TOOL. lmlcmp3's `reachable` (R), but the look's grasps replaced by GraspGen-X grasps on the true
            object (fd_gt.py: its visual meshes sampled at the simulator pose, 2048 points, seed 0, the look's request),
            moved to the object's simulator pose at the look. The stance, the walk and every filter are the look's
            (the stance from the perceived cloud, the furniture check on the perceived furniture), so up to the grasp
            choice the episode is the R episode.
  servopos  MEASUREMENT TOOL. servo, but position only: the wrist target stays the command form's rotation (alone's
            rot_z(yaw + dyaw) rot_y(pitch), no roll) and only the position error decides (<= SERVO_P).
  gtservo   MEASUREMENT TOOL. gtgrasp's grasps, servo's execution.
  tuck*     revision 6: SONIC's arms tucked for the walks (make_walk_tuck; TUCK_MODES).
  noturn    revision 6 (N): lmlcmp3's episode with HG_SHORT_M=0.40 (hg_walk's own dev override, read at its import): a
            walk shorter than 0.40 m is one omnidirectional shuffle (sideways / backward, no turn in place) instead of
            turn-walk-turn; nothing else changes (the first walk, ~2.25 m, is planned as before). Set here from --group
            before anything imports hg_walk; any other group asserts HG_SHORT_M unset.
  noturn_tuck1, noturn_tuck1b   N + T1a / N + T1b (the tuck of tuck1 / tuck1b with HG_SHORT_M=0.40).

Step log (every group): one row per policy step after the step -- pelvis pose and free-joint velocity, the right
wrist_yaw_link pose, the left wrist position, the object pose, the right hand's joint angles and targets, right-hand /
object contacts (count, normal force, which finger), object / world normal force, the right hand / wrist / elbow on
furniture, each foot's normal force and position, the 29 joint angles, the frame pushed this step (world qpos frame);
with --qpos also the full qpos (for fd_render.py). Contact forces from mj_contactForce (read only).

    cd $G; LC_CODE_ROOT=$C0 PYTHONPATH=$G:$C0/src python fd_run.py run --group rerun --backend sonic --task T \
        --method alone --ep-list 30,31 --shard 0 --shards 1
    python fd_run.py check --group rerun --backend sonic
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

LC3 = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126")
C0 = Path(os.environ.get("LC_CODE_ROOT", str(LC3 / "code")))
sys.path[:0] = [str(C0 / "scripts" / "experiments" / "selfmodel_lmlcmp3_20261126")]
os.environ.setdefault("LC_OUT", os.environ.get("FD_OUT", "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126"))
# revision 6 (N): hg_walk reads HG_SHORT_M once, at its import, so it is set from --group before lc3_run is imported
NOTURN_GROUPS, NOTURN_SHORT_M = ("noturn", "noturn_tuck1", "noturn_tuck1b"), "0.40"


def _argv_group() -> str:
    for i, v in enumerate(sys.argv):
        if v == "--group" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if v.startswith("--group="):
            return v.split("=", 1)[1]
    return "rerun"


if _argv_group() in NOTURN_GROUPS:
    os.environ.setdefault("HG_SHORT_M", NOTURN_SHORT_M)
    assert os.environ["HG_SHORT_M"] == NOTURN_SHORT_M, "noturn groups: HG_SHORT_M must be 0.40"
else:
    assert "HG_SHORT_M" not in os.environ, "HG_SHORT_M is set: only the noturn groups may set it"
import lc3_run as LC  # noqa: E402

LR, D3, HG, L, LT = LC.LR, LC.D3, LC.HG, LC.L, LC.LT
FD_OUT = Path(os.environ.get("FD_OUT", "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126"))
GT_DIR = FD_OUT / "gt_looks"
GROUPS = ("rerun", "servo", "servopos", "gtgrasp", "gtservo", "tuck0", "tuck1", "tuck2", "tuck1b", "tuck2b") + NOTURN_GROUPS
SERVO_P, SERVO_R, SERVO_N = 0.005, math.radians(5.0), 4
SERVO_STEP_P, SERVO_STEP_R = 0.04, math.radians(15.0)  # the largest change of the target per correction (damping)
SERVO_W_R = 5.0  # degrees counted as 1 cm when comparing tries (best so far)


def rot_cap(Rm: np.ndarray, cap: float) -> np.ndarray:
    """Rm with its rotation angle limited to cap (same axis)."""
    a = math.acos(float(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))
    if a <= cap or a < 1e-9:
        return Rm
    w = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    n = np.linalg.norm(w)
    if n < 1e-9:  # near 180 deg: axis from the symmetric part
        k = np.sqrt(np.clip((np.diag(Rm) + 1) / 2, 0, None))
        k = k / max(np.linalg.norm(k), 1e-9)
    else:
        k = w / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(cap) * K + (1 - math.cos(cap)) * K @ K
SERVO_EXTRA_S = 10 * 6.0
IGNORE = {"wall_s", "think_s"}


# -- the step log -----------------------------------------------------------------------------------------------------
FIX_ARMATURE = 1e8


class StepLog:
    def __init__(self, dr, qpos: bool = False):
        mj, m = dr.mj, dr.m
        self.dr, self.mj, self.m, self.qpos = dr, mj, m, qpos
        bn = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
        jn = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i) or "" for i in range(m.njnt)]
        self.robot = np.array([n.startswith("robot0_") for n in bn])
        self.hand = np.array([n.startswith(("robot0_right_hand", "robot0_right_wrist")) for n in bn])
        self.arm = np.array([n.startswith(("robot0_right_hand", "robot0_right_wrist", "robot0_right_elbow")) for n in bn])
        # finger groups of the right Dex3 hand by body name (thumb / index / middle / palm or wrist)
        grp = np.full(m.nbody, -1)
        for i, n in enumerate(bn):
            if n.startswith("robot0_right_hand"):
                grp[i] = 0 if "thumb" in n else 1 if "index" in n else 2 if "middle" in n else 3
            elif n.startswith("robot0_right_wrist"):
                grp[i] = 3
        self.grp = grp
        self.foot = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"robot0_{s}_ankle_roll_link") for s in ("left", "right")]
        foot_tree = np.zeros(m.nbody, int) - 1
        for k, b in enumerate(self.foot):
            for i in range(m.nbody):
                j = i
                while j > 0:
                    if j == b:
                        foot_tree[i] = k
                        break
                    j = m.body_parentid[j]
        self.foot_tree = foot_tree
        self.fj = [j for j, n in enumerate(jn) if n.startswith("robot0_right_hand")]
        self.fq = np.array([m.jnt_qposadr[j] for j in self.fj], int)
        acts = [a for a in range(m.nu) if int(m.actuator_trnid[a, 0]) in self.fj]
        self.fa = np.array(acts, int)
        self.names = dict(bodies=bn, finger_joints=[jn[j] for j in self.fj],
                          finger_acts=[mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, a) or "" for a in acts])  # fmt: skip
        self.f6 = np.zeros(6)
        # FD_HITLOG=1 (question 1, the object's disturbances): every contact of the object's bodies with another body,
        # (row, other body id, normal force), saved as obj_hits; read only, the rows are unchanged
        self.hitlog = os.environ.get("FD_HITLOG") in ("1", "2")
        # FD_HITLOG=2 also: the object's supports (the root bodies of the non-robot, non-world bodies it touches at
        # its first such contact) -- their root position every row (sup_pos) and every robot contact with them
        # (sup_hits: row, robot body, other body, normal force)
        self.suplog = os.environ.get("FD_HITLOG") == "2"
        self.root = np.asarray(m.body_rootid, int)
        # FD_FIX_SUPPORT=<name part> (revision 4, the fixed coffee table): every root body whose name contains it and
        # has joints gets armature FIX_ARMATURE on all its DOFs -- effectively immovable (contacts unchanged)
        self.fixed, self.fixed_dofs = [], []
        fx = os.environ.get("FD_FIX_SUPPORT", "")
        if fx:
            for i, n in enumerate(bn):
                if fx in n and int(m.body_rootid[i]) == i and int(m.body_jntnum[i]) > 0:
                    da, dn = int(m.body_dofadr[i]), int(m.body_dofnum[i])
                    self.fixed_dofs += list(range(da, da + dn))
                    self.fixed.append(n)
            m.dof_armature[self.fixed_dofs] = FIX_ARMATURE
            self.names["fixed_support"] = self.fixed
            print(f"fd FD_FIX_SUPPORT={fx}: armature {FIX_ARMATURE:g} on {self.fixed}", flush=True)
        self.clear()

    def clear(self) -> None:
        self.rows, self.qp, self.push_fw, self.hits = [], [], None, []
        self.sup_roots, self.sup_pos, self.sup_hits = None, [], []
        self.obj = set(self.dr.obj_tree)
        self.obj_b = int(self.dr.obj_bid)

    def push(self, fw: np.ndarray) -> None:
        self.push_fw = np.asarray(fw, np.float64).copy()

    def record(self) -> None:
        dr, mj, m = self.dr, self.mj, self.m
        d = dr.d
        if self.fixed_dofs and not self.rows:  # each episode's first step (after the env's reset): keep the fix
            m.dof_armature[self.fixed_dofs] = FIX_ARMATURE
        nc = int(d.ncon)
        hand_n, hand_f, obj_world_f, arm_furn = 0, 0.0, 0.0, 0
        fing = np.zeros(4)
        foot_f = np.zeros(2)
        if nc:
            g1 = np.asarray(d.contact.geom1[:nc])
            g2 = np.asarray(d.contact.geom2[:nc])
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if self.suplog:
                if self.sup_roots is None:
                    sr = set()
                    for a, b in zip(b1.tolist(), b2.tolist()):
                        if (a in self.obj) != (b in self.obj):
                            o = b if a in self.obj else a
                            if o != 0 and not self.robot[o]:
                                sr.add(int(self.root[o]))
                    if sr:
                        self.sup_roots = sorted(sr)
                if self.sup_roots:
                    for i in range(nc):
                        a, b = int(b1[i]), int(b2[i])
                        if self.robot[a] != self.robot[b]:
                            o, rb = (b, a) if self.robot[a] else (a, b)
                            if int(self.root[o]) in self.sup_roots:
                                mj.mj_contactForce(m, d, i, self.f6)
                                self.sup_hits.append((len(self.rows), rb, o, abs(float(self.f6[0]))))
            for i in range(nc):
                a, b = int(b1[i]), int(b2[i])
                ao, bo = a in self.obj, b in self.obj
                ha, hb = bool(self.hand[a]), bool(self.hand[b])
                fa_, fb_ = self.foot_tree[a], self.foot_tree[b]
                need = (ao != bo) or (fa_ >= 0) != (fb_ >= 0) or (self.arm[a] and not self.robot[b] and not bo) or (self.arm[b] and not self.robot[a] and not ao)
                if not need:
                    continue
                mj.mj_contactForce(m, d, i, self.f6)
                fn = abs(float(self.f6[0]))
                if ao != bo:
                    o = b if ao else a
                    if self.hitlog:
                        self.hits.append((len(self.rows), o, fn))
                    if self.hand[o]:
                        hand_n += 1
                        hand_f += fn
                        if self.grp[o] >= 0:
                            fing[self.grp[o]] += fn
                    elif not self.robot[o]:
                        obj_world_f += fn
                if (fa_ >= 0) != (fb_ >= 0):
                    k, o = (fa_, b) if fa_ >= 0 else (fb_, a)
                    if not self.robot[o]:
                        foot_f[k] += fn
                if (self.arm[a] and not self.robot[b] and not bo) or (self.arm[b] and not self.robot[a] and not ao):
                    arm_furn = 1
        pw = dr.wrist["right"]
        row = np.r_[
            d.xpos[dr.pelvis], d.xquat[dr.pelvis], d.qvel[dr.rv : dr.rv + 6],
            d.xpos[pw], d.xquat[pw], d.xpos[dr.wrist["left"]],
            d.xpos[self.obj_b], d.xquat[self.obj_b],
            d.qpos[self.fq], d.ctrl[self.fa] if len(self.fa) else np.zeros(0), float(getattr(dr, "grip", 0.0)),
            hand_n, hand_f, fing, obj_world_f, arm_furn, foot_f, d.xpos[self.foot[0]], d.xpos[self.foot[1]],
            d.qpos[dr.qadr],
            self.push_fw if self.push_fw is not None else np.full(36, np.nan),
        ]  # fmt: skip
        self.rows.append(row.astype(np.float32))
        if self.suplog:
            self.sup_pos.append(np.concatenate([d.xpos[r] for r in self.sup_roots]) if self.sup_roots else np.zeros(0))
        if self.qpos:
            self.qp.append(d.qpos.astype(np.float32).copy())

    def cols(self) -> list[str]:
        nf, na = len(self.fq), len(self.fa)
        c = [f"pel_p{i}" for i in "xyz"] + [f"pel_q{i}" for i in "wxyz"] + [f"pel_v{i}" for i in range(6)]
        c += [f"wr_p{i}" for i in "xyz"] + [f"wr_q{i}" for i in "wxyz"] + [f"wl_p{i}" for i in "xyz"]
        c += [f"obj_p{i}" for i in "xyz"] + [f"obj_q{i}" for i in "wxyz"]
        c += [f"fq{i}" for i in range(nf)] + [f"fc{i}" for i in range(na)] + ["grip"]
        c += ["hand_obj_n", "hand_obj_f", "f_thumb", "f_index", "f_middle", "f_palm", "obj_world_f", "arm_furn", "foot_l_f", "foot_r_f"]
        c += [f"foot_l_p{i}" for i in "xyz"] + [f"foot_r_p{i}" for i in "xyz"] + [f"q{i}" for i in range(29)] + [f"push{i}" for i in range(36)]
        return c

    def save(self, path: Path, extra: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        z = dict(rows=np.array(self.rows), cols=np.array(self.cols()), **extra)
        if self.qpos:
            z["qpos"] = np.array(self.qp)
        if self.hitlog:
            z["obj_hits"] = np.array(self.hits, np.float32).reshape(-1, 3)
        if self.suplog:
            z["sup_roots"] = np.array(self.sup_roots or [], np.int32)
            wd = 3 * len(self.sup_roots or [])  # rows before the supports were known: NaN
            z["sup_pos"] = np.array([v if len(v) == wd else np.full(wd, np.nan) for v in self.sup_pos], np.float32).reshape(len(self.sup_pos), wd)
            z["sup_hits"] = np.array(self.sup_hits, np.float32).reshape(-1, 4)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **z)
        os.replace(tmp, path)


def attach(dr, log: StepLog) -> None:
    """Instance-level wrappers: push records the frame, step records the state after the step (both read only)."""
    push0, step0 = dr.push, dr.step

    def push(fw):
        log.push(fw)
        push0(fw)

    def step():
        step0()
        log.record()

    dr.push, dr.step = push, step


# -- the servo's planner: plan_rec with an optional full wrist rotation ---------------------------------------------------
def plan_rot(self, x: dict, ref_w: np.ndarray, c: dict):
    """lr_run.plan_rec, except that c["rot_w"] (a world rotation), when given, replaces rot_z(yaw + dyaw) rot_y(pitch)."""
    if "rot_w" not in c:
        return LR.plan_rec(self, x, ref_w, c)
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    yaw = L.yaw_of(L._qmat(quat))
    Rh = L.rot_z(yaw)
    target_w = np.r_[p[:2] + (Rh @ np.r_[c["target"][:2], 0.0])[:2], c["target"][2]]
    rot_w = np.asarray(c["rot_w"], float)
    q_mj = x["q"][self.to_mj].astype(float)
    t0 = time.time()
    poses, durs, ik_err, dur = self.pl.reach((p, quat, q_mj), c["side"], target_w, rot_w, c["pelvis_z"])
    self.plan_s += time.time() - t0
    delta = np.r_[ref_w[:2] - p[:2], 0.0]
    ref_pose = (ref_w[:3].astype(float), ref_w[3:7].astype(float), ref_w[7:].astype(float))
    shifted = [ref_pose] + [(pp + delta, qq, jj) for pp, qq, jj in poses[1:]]
    last = poses[-1]
    rec = dict(side=c["side"], target_w=target_w.tolist(), rot_w=rot_w.tolist(), dyaw=c["dyaw"], pitch=c["pitch"],
               pelvis_z=c["pelvis_z"], cmd_target=np.asarray(c["target"]).tolist(), ik_err=ik_err, reach_s=dur,
               ref_q=last[2].tolist(), ref_pel=np.r_[last[0], last[1]].tolist(), delta=delta.tolist(),
               RT=np.r_[0.0, np.cumsum(durs)], RQ=np.array([pp[2] for pp in poses]),
               RP=np.array([np.r_[pp[0], pp[1]] for pp in poses]))  # fmt: skip
    return L.blend_frames(shifted, durs), shifted[-1], rec


def make_driver(backend: str, task: str, group: str):
    if group == "rerun":
        return LR.make_driver(backend, task)  # lmlcmp3's driver exactly
    dr = LR.make_driver(backend, task)
    dr.plan = plan_rot.__get__(dr)  # the same driver; plan_rot is plan_rec unless a full rotation is asked
    return dr


# -- revision 6: SONIC's arms tucked for the walks (simple_relook_20261126's tuck) ----------------------------------------
# simple_eval_20261112/simple_bench.py TUCK / tuck_arms with simple_relook's --relook-tuck 0.8 --relook-tuck-hold 1.0:
# standing, both arms' reference joints min-jerk to TUCK over TUCK_S, then held TUCK_HOLD_S, then the walk, which carries
# that arm pose (the hands by the hips); the reach then starts from it. Only the arm columns of the walk frames change.
TUCK = (0.30, 0.30, 0.0, 0.60, 0.0, 0.0, 0.0)  # one arm's 7 joints: shoulder pitch (back), roll (out), yaw, elbow, wrist x3
TUCK_S, TUCK_HOLD_S = 0.8, 1.0
TUCK_IDX = np.array(list(range(15, 22)) + list(range(22, 29)))  # g1_joints.ARM_MJ left, right (MuJoCo order)
TUCK_Q = np.array([TUCK[0], TUCK[1], TUCK[2], TUCK[3], TUCK[4], TUCK[5], TUCK[6],
                   TUCK[0], -TUCK[1], -TUCK[2], TUCK[3], -TUCK[4], TUCK[5], -TUCK[6]])  # fmt: skip
# the new pose (T1b / T2b): fd_tuckpose.py's pick by revision 6's rule (left arm; the right mirrored as TUCK's):
# `--objective radial --zmax-cm 0` -> tuckpose_radial_z0.json (hand points 13.8 cm from the pelvis's vertical axis,
# (-11.9, +-6.9, -17.1) cm in the pelvis frame, elbow bend 29.6 deg; 3 tied within 0.5 cm, the least bend kept)
NEW_L = (0.48676000000000297, 0.19827000000000133, -1.074399999999999, 0.94696, 0.15750000000000008, -0.015, 0.045)
NEW_Q = np.array(list(NEW_L) + [NEW_L[0], -NEW_L[1], -NEW_L[2], NEW_L[3], -NEW_L[4], NEW_L[5], -NEW_L[6]])
# group -> (mode, pose); mode 0: no tuck (the copy's identity check); 1: the correction walks; 2: every walk
TUCK_MODES = {"tuck0": (0, TUCK_Q), "tuck1": (1, TUCK_Q), "tuck2": (2, TUCK_Q), "tuck1b": (1, NEW_Q), "tuck2b": (2, NEW_Q),
              "noturn_tuck1": (1, TUCK_Q), "noturn_tuck1b": (1, NEW_Q)}


def make_walk_tuck(orig, st: dict):
    """lc3_run.walk_ctrl with the tuck: call k = 1 is the first walk (after the look), k >= 2 the correction walks. The
    standing tuck once per episode (later tucked walks start tucked); st['extra_f'] counts the inserted frames."""

    def walk(dr, backend, x, ref_frame, goal_xy, goal_yaw, pose, info=None):
        st["n"] += 1
        k = st["n"]
        if not (st["mode"] == 2 or (st["mode"] == 1 and k >= 2)):
            st["walks"].append(dict(k=k, tucked=False))
            return orig(dr, backend, x, ref_frame, goal_xy, goal_yaw, pose=pose, info=info)
        rf = np.asarray(ref_frame).copy()
        pre = np.zeros((0, len(rf)), np.float32)
        if not st["tucked"]:
            nb, nh = int(round(TUCK_S * L.FPS)), int(round(TUCK_HOLD_S * L.FPS))
            q0 = rf[7 + TUCK_IDX].astype(float)
            pre = np.repeat(rf[None].astype(np.float32), nb + nh, 0)
            for i in range(nb + nh):
                pre[i, 7 + TUCK_IDX] = q0 + L.minjerk(min(1.0, (i + 1) / nb)) * (st["q"] - q0)
            st["tucked"] = True
            st["extra_f"] += nb + nh
            rf = pre[-1].copy()
        wf = orig(dr, backend, x, rf, goal_xy, goal_yaw, pose=pose, info=info)
        wf[:, 7 + TUCK_IDX] = st["q"]
        st["walks"].append(dict(k=k, tucked=True, pre_f=len(pre), walk_f=len(wf)))
        return np.concatenate([pre.astype(wf.dtype), wf])

    return walk


def episode_tuck_fn(st: dict):
    """lc3_run.episode, its source unchanged except that the time limit grows by the inserted tuck frames, run in a copy
    of lc3_run's globals whose walk_ctrl is make_walk_tuck's (the frozen copy on disk is untouched)."""
    import inspect

    src = inspect.getsource(LC.episode)
    old = "        if f > t_max:\n"
    assert src.count(old) == 1, "episode_tuck_fn: the time-limit line moved"
    src = src.replace(old, "        if f > t_max + _FD_TUCK['extra_f']:  # faildiag revision 6\n")
    ns = dict(LC.__dict__)
    ns.update(walk_ctrl=make_walk_tuck(LC.walk_ctrl, st), _FD_TUCK=st)
    exec(compile(src, LC.__file__, "exec"), ns)
    return ns["episode"]


# -- the true object's grasps -------------------------------------------------------------------------------------------
def obj_T(dr) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = dr.d.xmat[dr.obj_bid].reshape(3, 3)
    T[:3, 3] = dr.d.xpos[dr.obj_bid]
    return T


def gt_load(task: str, ep: int) -> dict:
    z = np.load(GT_DIR / task / f"ep{ep:+05d}.npz")
    return {k: z[k] for k in z.files}


def patched_load(dr, task: str, ep: int, info: dict):
    """hg_look.load for the gt groups: the cached look with its grasps replaced by the true object's, moved to the
    object's simulator pose now (the look's frame is the simulator world)."""
    import hg_look as HL

    load0 = HL.load

    def load(path):
        lk = load0(path)
        g = gt_load(task, ep)
        Tn = obj_T(dr)
        Gw = Tn @ g["grasp_G_obj"]  # grasp frames, object frame -> world now
        Tw = HL.wrist_of(Gw)
        info.update(gt_n=len(Gw), gt_obj_move_mm=1000 * float(np.linalg.norm(Tn[:3, 3] - g["obj_T_reset"][:3, 3])),
                    gt_obj_rot_deg=float(np.degrees(np.arccos(np.clip((np.trace(Tn[:3, :3].T @ g["obj_T_reset"][:3, :3]) - 1) / 2, -1, 1)))))  # fmt: skip
        lk = dict(lk)
        lk["grasp_T"] = Tw
        lk["grasp_hand"] = Tw[:, :3, 3] + Tw[:, :3, :3] @ HL.HAND_OFF if len(Tw) else np.zeros((0, 3))
        lk["grasp_score"] = np.asarray(g["grasp_score"], float)
        return lk

    return HL, load0, load


# -- one servo episode: lc3_run.episode with the servo (marked "faildiag:") ------------------------------------------------
def episode_servo(dr, rig, ps, ep: int, MD, backend: str, gt: bool, R_extra: dict, full_rot: bool = True) -> tuple[dict, list[dict]]:
    """lc3_run.episode (method alone) with the truth servo after each pre-grasp / entry command (module docstring)."""
    import hg_look as HL

    method = "alone"
    t0 = time.time()
    spec = HG.TASKS[dr.task]
    jit = dr.reset_hg(ep)
    rig.clear()
    ps.clear()
    frames = dr.start_frames()
    f_look = int(round(L.START_HOLD_S * L.FPS))
    if spec["kind"] == "approach":
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
    lift_run, lift_best, rise_max, official_at = 0, 0, 0.0, None
    anchor = None
    walk_info: dict = {}
    n_walks = 0
    first = None
    rlo, rhi = HG.train_range()
    delay = shift = None
    t_max = 60 * L.FPS + int(SERVO_EXTRA_S * L.FPS)  # faildiag: the servo's extra commands
    d3r: list[dict] = []
    cur_delta = np.zeros(3)
    active = None
    from collections import deque

    vq: deque = deque(maxlen=D3.SETTLE_F)
    lost = None
    walking = False
    wrec = dict(asked=None, real=None, dev_cm=0.0)
    x_walk0 = None
    waits: dict[str, dict] = {}
    wlog: list[dict] = []
    agree = dict(both=0, neither=0, own_only=0, truth_only=0)
    servo = dict(cmd1=[], cmd2=[])  # faildiag: per reach key, the servo's commands (0 = alone's)
    servo_next: dict[str, dict] = {}  # faildiag: key -> the corrected world target / rotation for its next command
    R_goal = None

    def close_window(fe: int, kind: int = 2) -> None:
        nonlocal active
        if active is not None and active["f_label"] is None:
            g = active["f_cmd"] + ((fe - active["f_cmd"]) // D3.TICKS) * D3.TICKS
            hw = rows[g]["wr"]["right"]
            active.update(f_label=g, settled_end=False, settle_s=(g - active["f_end"]) / L.FPS, label_kind=kind,
                          live_err_cm=100 * float(np.linalg.norm(hw[0] + hw[1] @ L.HAND_OFF - active["hands"]["right"]["target_w"])))  # fmt: skip
        active = None

    def gate(key: str) -> bool:
        w = waits.setdefault(key, dict(open=f, tf=None, n=dict(both=0, neither=0, own_only=0, truth_only=0)))
        qo, qt = ps.quiet(), rig.quiet()
        cat = "both" if qo and qt else "neither" if not (qo or qt) else "own_only" if qo else "truth_only"
        w["n"][cat] += 1
        agree[cat] += 1
        if qt and w["tf"] is None:
            w["tf"] = f
        if qo or f - w["open"] >= LC.SETTLE_MAX_F:
            vh, vl = ps.vmax()
            tv = 100 * float(max(rig.v)) if len(rig.v) else float("inf")
            wlog.append(dict(key=key, open=w["open"], fire=f, wait_s=(f - w["open"]) / L.FPS, prop_settled=bool(qo), truth_quiet=bool(qt), truth_v_max_cms=tv,
                             truth_first_s=-1.0 if w["tf"] is None else (w["tf"] - w["open"]) / L.FPS, steps=w["n"],
                             prop_v_hand_cms=100 * vh, prop_v_leg_rads=vl, phase=phase))  # fmt: skip
            del waits[key]
            return True
        sched[key] = f + 1
        return False

    while True:
        dr.push(frames[min(f, len(frames) - 1)])
        if dr.buf is None:
            dr.buf = dr._find_buffer()
        dr.step()
        if delay is None:
            delay = D3.ref_delay_f(dr)
            shift = delay - LR.BASE_DELAY_F
            if shift > 0:
                frames = np.concatenate([frames, np.repeat(frames[-1:], shift + 10, axis=0)])
                f_look += shift
                t_max += 6 * shift
        if f == 1 and getattr(getattr(dr, "comp", None), "_settling", False):
            raise RuntimeError("the bridge is still settling: our reference is not being tracked")
        x = rows[f] = dr.sample()
        rig.record(x)
        ps.update(x)
        vq.append(np.linalg.norm(rig.pts[-1] - rig.pts[0], axis=1) / (D3.TICKS * L.DT) if len(rig.pts) > D3.TICKS else np.full(3, np.inf))
        if odo is not None:
            odo.update(dr, x)
        if float(x["root"][2]) < L.FALL_Z:
            fell, why = f, f"fell during {phase}"
            break
        if anchor is not None:
            fr = frames[min(max(f - shift, anchor[2]), len(frames) - 1)]
            dev = (x["root"][:2] - anchor[0]) - (fr[:2] - anchor[1])
            if walking:
                wrec["dev_cm"] = max(wrec["dev_cm"], 100 * float(np.linalg.norm(dev)))
            if float(np.linalg.norm(dev)) > L.LOST_M:
                fell, why = f, f"lost during {phase}"
                lost = f
                break
        if active is not None and active["f_label"] is None and (f - active["f_cmd"]) % D3.TICKS == 0:
            if f >= active["f_end"] + delay + D3.SETTLE_F and rig.quiet():
                hw = x["wr"]["right"]
                active.update(f_label=f, settled_end=True, settle_s=(f - active["f_end"]) / L.FPS, label_kind=0,
                              live_err_cm=100 * float(np.linalg.norm(hw[0] + hw[1] @ L.HAND_OFF - active["hands"]["right"]["target_w"])))  # fmt: skip
            elif f >= active["f_end"] + delay + LC.SETTLE_MAX_F:
                close_window(f, kind=1)
        if active is not None and f > active["f_cmd"] and (active["f_label"] is None or active["f_label"] == f):
            active["furn_rows"] += int(dr.arm_scene_contact() is not None)
            active["obj_rows"] += int(dr.obj_contacts()[0])
            active["win_rows"] += 1
        z = float(dr.d.xpos[dr.obj_bid][2]) - dr.obj_z0
        if "close" in sched and f >= sched["close"] and phase in ("close", "lift"):
            hand_c, world_c = dr.obj_contacts()
            rise_max = max(rise_max, z)
            lift_run = lift_run + 1 if (z >= HG.LIFT_MIN and not world_c) else 0
            lift_best = max(lift_best, lift_run)
            try:
                if official_at is None and bool(dr.env._check_success()):
                    official_at = f
            except Exception:
                pass
        if first and first in sched and sched[first] <= f and phase not in ("close", "lift"):
            h = dr.arm_scene_contact()
            if h is not None:
                contact_hits.append((f, h))
        if f == f_look:
            lp = LC.look_path(dr.task, ep)
            lk = HL.load(lp)  # (gt groups: patched by the caller)
            pel = np.asarray(lk["pelvis"], float)
            R["look_state_dev_mm"] = 1000 * float(np.abs(pel[:3] - x["root"][:3]).max())
            R["look"] = dict(status=lk["meta"].get("status"), mask_score=float(lk["mask_score"]), n_obj=len(lk["obj_pts"]),
                             n_grasps=len(lk["grasp_T"]), prompt=lk["meta"].get("prompt_used"))  # fmt: skip
            tp = dr.obj_pos()
            R["look"]["obj_center_err_cm"] = 100 * float(np.linalg.norm(np.median(lk["obj_pts"], 0) - tp)) if len(lk["obj_pts"]) else None
            p0, yaw0 = x["root"][:2].copy(), L.yaw_of(L._qmat(x["root"][3:]))
            import hg_walk as HW

            odo = HW.LegOdo(dr, x, (p0, yaw0))
            plan = HG.plan_from_look(lk, spec["kind"], x["root"][:3], yaw0)
            R["plan"] = HG._js({k: v for k, v in plan.items() if k not in ("stance",)})
            R["plan"]["stance"] = HG._js(plan.get("stance"))
            if plan["fail"]:
                why = plan["fail"]
                break
            R_goal = np.asarray(lk["grasp_T"], float)[plan["idx"]][:3, :3].copy()  # faildiag: the planned wrist frame
            st = plan["stance"]
            d_move = float(np.linalg.norm(st["xy"] - p0))
            d_turn = abs(wrap(st["yaw"] - yaw0))
            R["move"] = dict(dist=d_move, turn_deg=math.degrees(d_turn))
            if d_move >= HG.MOVE_MIN or d_turn >= HG.TURN_MIN:
                wf = LC.walk_ctrl(dr, backend, x, frames[f], st["xy"], st["yaw"], pose=odo.pose(), info=walk_info)
                frames = np.concatenate([frames[: f + 1], wf, np.repeat(wf[-1:], max(shift, 0) + 10 + int(LC.EXTRA_HOLD_S * L.FPS), 0)])
                anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                first = "cmd0" if plan["via"] is not None else "cmd1"
                sched[first] = f + len(wf) - 25 + shift
                phase = "walk"
                n_walks = 1
                dp = L.rot_z(yaw0).T @ np.r_[st["xy"] - p0, 0.0]
                wrec["asked"] = [float(dp[0]), float(dp[1]), wrap(float(st["yaw"]) - yaw0)]
                cur_delta = np.r_[frames[f, :2] - x["root"][:2], 0.0]
                x_walk0, walking = x, True
            else:
                frames = np.concatenate([frames[: f + 1], L.hold((frames[f][:3], frames[f][3:7], frames[f][7:]), 2.0 + shift / L.FPS + LC.EXTRA_HOLD_S)])
                first = "cmd0" if plan["via"] is not None else "cmd1"
                sched[first] = f + int(round(L.PRE_S * L.FPS))
        fired = set()
        if first and sched.get(first) == f and phase == "walk" and n_walks < 1 + HG.WALK_FIX and gate(first):
            pose = odo.pose()
            st = plan["stance"]
            d_left = float(np.linalg.norm(st["xy"] - pose[0]))
            R.setdefault("walk_left_cm", []).append(100 * d_left)
            if d_left >= HG.FIX_MIN and n_walks < HG.WALK_FIX + 1:
                wlog[-1]["then"] = "correction walk"
                wf = LC.walk_ctrl(dr, backend, x, frames[f], st["xy"], st["yaw"], pose=pose)
                frames = np.concatenate([frames[: f + 1], wf, np.repeat(wf[-1:], max(shift, 0) + 10 + int(LC.EXTRA_HOLD_S * L.FPS), 0)])
                anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                sched[first] = f + len(wf) - 25 + shift
                n_walks += 1
                cur_delta = np.r_[frames[f, :2] - x["root"][:2], 0.0]
            else:
                fired.add(first)
        # faildiag: the servo's check, after a pre-grasp / entry command's window (where alone's next command would go)
        for key in ("cmd1", "cmd2"):
            sk = "chk_" + key
            if sched.get(sk) != f:
                continue
            del sched[sk]
            goal = plan["pre"] if key == "cmd1" else plan["goal"]
            hw_ = x["wr"]["right"]
            hand = hw_[0] + hw_[1] @ L.HAND_OFF
            e_p = goal - hand
            Rerr = R_goal @ hw_[1].T
            e_r = math.acos(float(np.clip((np.trace(Rerr) - 1) / 2, -1, 1)))
            last_s = servo[key][-1]
            last_s.update(err_p_cm=100 * float(np.linalg.norm(e_p)), err_r_deg=math.degrees(e_r), f_chk=f)
            bad_r = e_r > SERVO_R if full_rot else False  # servopos: the rotation is only measured

            def _score(s_):
                return s_["err_p_cm"] + (s_["err_r_deg"] / SERVO_W_R if full_rot else 0.0)

            tried = [s_ for s_ in servo[key] if "err_p_cm" in s_]
            best = min(tried, key=_score)
            worse = best is not last_s and _score(last_s) > _score(best) + 0.5
            stop = bool(last_s.get("revert"))  # the best try was re-sent once: go on whatever it gives
            if not stop and worse:  # the last correction made it worse: back to the best try, then stop
                servo_next[key] = dict(target_w=np.asarray(best["target_w"]), rot_w=np.asarray(best["rot_w"]), revert=True)
                sched[key] = f
            elif not stop and (np.linalg.norm(e_p) > SERVO_P or bad_r) and len(servo[key]) <= SERVO_N:
                dp = e_p * min(1.0, SERVO_STEP_P / max(float(np.linalg.norm(e_p)), 1e-9))
                rw = rot_cap(Rerr, SERVO_STEP_R) @ np.asarray(last_s["rot_w"]) if full_rot else np.asarray(last_s["rot_w"])
                servo_next[key] = dict(target_w=np.asarray(last_s["target_w"]) + dp, rot_w=rw)
                sched[key] = f
            elif key == "cmd1":
                sched["cmd2"] = f
            else:
                sched["close"] = f
        for k, key in ((-1, "cmd0"), (0, "cmd1"), (1, "cmd2")):
            if sched.get(key) != f or (phase == "walk" and n_walks < 1 + HG.WALK_FIX and key == first and key not in fired):
                continue
            if key not in fired and not gate(key):
                continue
            wl = wlog[-1]
            wl["reach_no"] = LR.REACH_NO[k]
            phase = f"reach{k + 1}"
            close_window(f)
            walking = False
            if x_walk0 is not None and wrec["real"] is None:
                pw0, yw0 = x_walk0["root"][:2].astype(float), L.yaw_of(L._qmat(x_walk0["root"][3:]))
                dp = L.rot_z(yw0).T @ np.r_[x["root"][:2] - pw0, 0.0]
                wrec["real"] = [float(dp[0]), float(dp[1]), wrap(L.yaw_of(L._qmat(x["root"][3:])) - yw0)]
            goal = plan["via"] if k == -1 else plan["pre"] if k == 0 else plan["goal"]
            chain = len(d3r)
            pose = odo.pose()
            c, gh = HG.command_b(pose, goal, plan["ad"], plan["pz"])
            sv = servo_next.pop(key, None)  # faildiag: a servo correction of this reach
            if sv is not None:
                p_, Rh_ = x["root"][:3].astype(float), L.rot_z(L.yaw_of(L._qmat(x["root"][3:].astype(float))))
                tw = sv["target_w"]
                c = dict(c, target=np.r_[(Rh_.T @ np.r_[tw[:2] - p_[:2], 0.0])[:2], tw[2]], rot_w=sv["rot_w"])
            info = dict(k=k, reach_no=LR.REACH_NO[k], chain=chain, goal_w=goal.tolist(), goal_h=gh.tolist(),
                        dyaw_deg=math.degrees(c["dyaw"]), pitch_deg=math.degrees(c["pitch"]), pz=c["pelvis_z"],
                        pose_odo=[pose[0].tolist(), pose[1]], pose_true=[x["root"][:2].tolist(), L.yaw_of(L._qmat(x["root"][3:]))],
                        wait_s=wl["wait_s"], prop_settled=wl["prop_settled"], truth_settled=bool(rig.quiet()),
                        servo_i=None if k < 0 else len(servo[key]))  # fmt: skip
            qx = x["q"]
            out_j = int(((qx < rlo) | (qx > rhi)).sum())
            info.update(q_out=out_j, q_in_range=out_j == 0)
            tic = time.time()
            feat = lambda cc: MD.features(dr, rig, rows, f, frames, cc, chain, n_walks > 0, cur_delta, wrec, delay)  # noqa: E731
            try:
                (blend, last, rec), zz = feat(c)
            except (np.linalg.LinAlgError, ValueError) as e:
                why = f"reach: plan failed {type(e).__name__}"
                info["plan_error"] = str(e)[:200]
                reaches.append(info)
                fell = None
                break
            pr = MD.predict(zz)
            info["pred_h"] = {n: v[0].tolist() for n, v in pr.items()}
            info["pred_std"] = {n: v[1].tolist() for n, v in pr.items()}
            info["think_s"] = time.time() - tic
            rec.update(chain=chain, f_cmd=f, f_end=f + len(blend))
            post = int(round(L.POST_S * L.FPS))
            frames = np.concatenate([frames[: f + 1], blend, L.hold(last, L.POST_S + L.PRE_S + L.MARGIN_S + 0.5 + shift / L.FPS + LC.EXTRA_HOLD_S)])
            anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
            info.update(cmd_h=np.asarray(c["target"]).tolist(), f_cmd=f, f_end=rec["f_end"], f_win=rec["f_end"] + post + shift,
                        ik_err_cm=100 * float(rec["ik_err"]), p_cmd=x["root"][:3].tolist(), yaw_cmd=L.yaw_of(L._qmat(x["root"][3:])),
                        rot_w=np.asarray(rec["rot_w"]).tolist(), target_w=np.asarray(rec["target_w"]).tolist())  # fmt: skip
            info.update(LC.ref_end(dr, rec, x))
            reaches.append(info)
            if k >= 0:
                servo[key].append(dict(f_cmd=f, target_w=list(rec["target_w"]), rot_w=np.asarray(rec["rot_w"]).tolist(), ik_err_cm=100 * float(rec["ik_err"]),
                                       revert=bool(sv is not None and sv.get("revert"))))
            vv = np.array(vq)
            vh, vl = ps.vmax()
            rc = LC.reach_rc(rec, f, rec["f_end"], chain, cur_delta, slot=LR.REACH_NO[k])
            rc.update(pre_wait_s=wl["wait_s"], live_err_cm=-1.0, pre_settled=bool(rig.quiet()),
                      pre_v_pel_cms=100 * float(vv[:, 0].max()), pre_v_hand_cms=100 * float(vv[:, 1].max()),
                      pre_v_other_cms=100 * float(vv[:, 2].max()), pre_v_max_cms=100 * float(vv.max()), furn_rows=0, obj_rows=0,
                      win_rows=0, prop_settled=wl["prop_settled"], prop_v_hand_cms=100 * vh, prop_v_leg_rads=vl,
                      truth_first_s=wl["truth_first_s"], label_kind=-1)  # fmt: skip
            cur_delta = rc["delta"].copy()
            d3r.append(rc)
            active = rc
            nxt = rec["f_end"] + post + int(round(L.PRE_S * L.FPS)) + shift
            if k == -1:
                sched["cmd1"] = nxt
            elif k == 0:
                sched["chk_cmd1"] = nxt  # faildiag: check, then servo again or the entry
            else:
                sched["chk_cmd2"] = rec["f_end"] + post + shift  # faildiag: check, then servo again or the close
            break
        if why.startswith("reach: plan failed"):
            break
        if sched.get("close") == f and phase == "reach2" and gate("close"):
            wlog[-1]["reach_no"] = -1
            close_window(f)
            contact_hits = [h for h in contact_hits if h[0] < f]
            phase = "close"
            hp = dr.hand_point(x)
            R["close_dist_cm"] = 100 * float(np.linalg.norm(hp - plan["goal"]))
            R["close_obj_cm"] = 100 * float(np.linalg.norm(hp - dr.obj_pos()))
            Rw = x["wr"]["right"][1]
            R["close_rot_err_deg"] = math.degrees(math.acos(float(np.clip((np.trace(R_goal @ Rw.T) - 1) / 2, -1, 1))))  # faildiag
            sched["lift"] = f + int(round((HG.CLOSE_S + HG.SETTLE_S) * L.FPS))
        if phase in ("close", "lift") and dr.grip < HG.GRIP:
            dr.grip = min(HG.GRIP, (f - sched["close"] + 1) / (0.3 * L.FPS))
        if sched.get("lift") == f and phase == "close" and gate("lift"):
            wlog[-1]["reach_no"] = -2
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
    for w, rc in zip([w for w in reaches if "f_win" in w], d3r):
        fl = rc["f_label"] if rc["f_label"] is not None else min(w["f_win"], max(rows))
        hand = dr.hand_point(rows[fl]) if fl in rows else None
        if hand is None:
            continue
        p0c, Rh = np.asarray(w["p_cmd"]), L.rot_z(w["yaw_cmd"])
        act_h = np.r_[(Rh.T @ np.r_[hand[:2] - p0c[:2], 0.0])[:2], hand[2]]
        w["err_cm"] = 100 * float(np.linalg.norm(hand - np.asarray(w["goal_w"])))
        w["act_h"], w["f_label"], w["settled_end"] = act_h.tolist(), fl, bool(rc["settled_end"])
        w["pred_err_cm"] = {n: 100 * float(np.linalg.norm(np.asarray(v) - act_h)) for n, v in w.get("pred_h", {}).items()}
    if walk_info:
        R["walk"] = HG._js({k: v for k, v in walk_info.items() if k in ("walk_s", "n_steps", "ik_err", "knots", "ik_err_cm", "kinds", "gait")})
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
    R["waits"], R["agree_steps"] = wlog, agree
    R["servo"] = servo  # faildiag
    R["servo_done"] = {k: (bool(v) and v[-1].get("err_p_cm", 1e9) <= 100 * SERVO_P and (not full_rot or v[-1].get("err_r_deg", 1e9) <= math.degrees(SERVO_R))) for k, v in servo.items()}
    R["servo_full_rot"] = full_rot
    R.update(R_extra)
    info3 = dict(fell=fell, lost=lost, walk=wrec, walked=n_walks > 0, delay_f=delay)
    segs = D3.segments(rows, info3, frames, d3r, LC.reset_id(dr.task, method, ep), dr.pl, rig)
    for s in segs:
        rc = d3r[s["chain"]]
        s.update(reach_no=rc["slot"], task_id=LC.TASKS.index(dr.task), ep=ep, backend_id=LC.BACKENDS.index(backend), method_id=LC.METHODS.index(method),
                 d_prev_x=float(rc["d_prev"][0]), d_prev_y=float(rc["d_prev"][1]), d_this_x=float(rc["delta"][0]), d_this_y=float(rc["delta"][1]),
                 **{k: rc[k] for k in ("pre_settled", "pre_v_pel_cms", "pre_v_hand_cms", "pre_v_other_cms", "pre_v_max_cms", "live_err_cm",
                                       "furn_rows", "obj_rows", "win_rows", "prop_settled", "prop_v_hand_cms", "prop_v_leg_rads",
                                       "truth_first_s", "label_kind")})  # fmt: skip
    R.update(delay_f=delay, shift_f=shift, walk_rec=wrec, n_segments=len(segs))
    return R, segs


wrap = LR.wrap


# -- runner -----------------------------------------------------------------------------------------------------------
def out_dir(group: str) -> Path:
    return FD_OUT / group


def ep_path(group: str, backend: str, task: str, method: str, ep: int) -> Path:
    return out_dir(group) / backend / "episodes" / task / method / f"ep{ep:+05d}.json"


def run(group: str, backend: str, task: str, method: str, eps: list[int], shard: int, shards: int, qpos: bool) -> None:
    mtag = method if group in ("rerun", "gtgrasp", "noturn") or group in TUCK_MODES else group
    jobs = eps[shard::shards]
    todo = [e for e in jobs if not ep_path(group, backend, task, mtag, e).exists()]
    print(f"fd {group} {backend} {task} {mtag} shard {shard}/{shards}: {len(jobs) - len(todo)}/{len(jobs)} done", flush=True)
    if not todo:
        return
    dr = make_driver(backend, task, group)
    rig, ps = D3.Rig(dr), LC.PropSettle(dr)
    MD = LC.Models(backend)
    log = StepLog(dr, qpos=qpos)
    attach(dr, log)
    LC.code_check("start")
    import hg_walk as HW  # noqa: E402 (lc3_run's own import path)

    short_m = float(NOTURN_SHORT_M) if group in NOTURN_GROUPS else 0.0
    assert HW.SHORT_M == short_m, f"hg_walk.SHORT_M {HW.SHORT_M} != {short_m} for {group}"
    for e in todo:
        log.clear()
        t0 = time.time()
        extra: dict = {}
        try:
            if group in ("rerun", "noturn"):
                r, segs = LC.episode(dr, rig, ps, e, MD, backend, method)
            elif group in TUCK_MODES:
                md, tq = TUCK_MODES[group]
                st = dict(mode=md, q=tq, n=0, tucked=False, extra_f=0, walks=[])
                r, segs = episode_tuck_fn(st)(dr, rig, ps, e, MD, backend, method)
                r["fd_tuck"] = dict(mode=md, extra_f=st["extra_f"], walks=st["walks"], q=tq.tolist(), tuck_s=TUCK_S, tuck_hold_s=TUCK_HOLD_S)
            elif group == "gtgrasp":
                HL, load0, load = patched_load(dr, task, e, extra)
                HL.load = load
                try:
                    r, segs = LC.episode(dr, rig, ps, e, MD, backend, method)
                finally:
                    HL.load = load0
                r.update(extra)
            else:
                if group == "gtservo":
                    HL, load0, load = patched_load(dr, task, e, extra)
                    HL.load = load
                try:
                    r, segs = episode_servo(dr, rig, ps, e, MD, backend, gt=group == "gtservo", R_extra=extra, full_rot=group != "servopos")
                finally:
                    if group == "gtservo":
                        HL.load = load0
        except Exception as err:
            r, segs = dict(task=task, ep=e, method=mtag, backend=backend, error=f"{type(err).__name__}: {err}",
                           tb=traceback.format_exc()[-2000:], success=False, reason="other"), []  # fmt: skip
            print(f"fd {group} {backend} {task} ep {e}: ERROR {err}", flush=True)
            dr = make_driver(backend, task, group)
            rig, ps = D3.Rig(dr), LC.PropSettle(dr)
            log = StepLog(dr, qpos=qpos)
            attach(dr, log)
        r["fd_group"] = group
        if group in NOTURN_GROUPS:
            r["fd_noturn"] = dict(HG_SHORT_M=os.environ["HG_SHORT_M"], SHORT_M=HW.SHORT_M, MOVE_MIN=HW.MOVE_MIN)
        base = out_dir(group) / backend
        D3.save_chunk(segs, base / "segs" / task / mtag, f"ep{e:+05d}", rig, LC.SCALARS_LC)
        log.save(base / "steplog" / task / mtag / f"ep{e:+05d}.npz",
                 dict(names=np.array(json.dumps(log.names)), wall_s=np.float32(time.time() - t0)))  # fmt: skip
        p = ep_path(group, backend, task, mtag, e)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, default=LT._json))
        os.replace(tmp, p)
        print(f"fd {group} {backend} {task} {mtag} ep {e}: success {r.get('success')} reason {r.get('reason')} why {r.get('why')} "
              f"close {r.get('close_dist_cm', float('nan')):.1f} cm steps {r.get('steps')} {time.time() - t0:.1f} s", flush=True)  # fmt: skip
        LC.code_check(f"after {group} ep {e}")


# -- the bit-identity check of the reruns ------------------------------------------------------------------------------
def diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'new' if k in a else 'orig'}")
                continue
            out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    if a != b and not (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
        return [f"{path}: {a!r} vs {b!r}"]
    return []


def check(group: str, backend: str) -> None:
    base = out_dir(group) / backend / "episodes"
    files = sorted(base.glob("*/*/ep*.json"))
    n_same, bad = 0, []
    for f in files:
        task, method = f.parent.parent.name, f.parent.name
        o = LC3 / backend / "episodes" / task / method / f.name
        a, b = json.loads(f.read_text()), json.loads(o.read_text())
        a.pop("fd_group", None)
        if (a.get("fd_tuck") or {}).get("mode") == 0:  # tuck0: the copy's identity check; the tuck record is the only extra
            a.pop("fd_tuck")
        dj = diff(a, b)
        sa, so = out_dir(group) / backend / "segs" / task / method / f"{f.stem}.npz", LC3 / backend / "segs" / task / method / f"{f.stem}.npz"
        ds = []
        for s1, s2 in ((sa, so), (sa.parent / "arm" / sa.name, so.parent / "arm" / so.name)):
            if s1.exists() != s2.exists():
                ds.append(f"{s1.name}: exists {s1.exists()} vs {s2.exists()}")
                continue
            if not s1.exists():
                continue
            z1, z2 = np.load(s1), np.load(s2)
            if set(z1.files) != set(z2.files):
                ds.append(f"{s1}: keys differ")
            for k in set(z1.files) & set(z2.files):
                if z1[k].shape != z2[k].shape or not np.array_equal(z1[k], z2[k], equal_nan=z1[k].dtype.kind == "f"):
                    ds.append(f"{s1.parent.name}/{k}")
        if dj or ds:
            bad.append((task, method, f.stem, dj[:4], ds[:4]))
        else:
            n_same += 1
    print(f"fd check {group} {backend}: {len(files)} episodes; identical to lmlcmp3 (record except {sorted(IGNORE)}, every segment array): {n_same}; differing {len(bad)}")
    for b in bad[:20]:
        print("  DIFF", b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "check"])
    ap.add_argument("--group", choices=GROUPS, default="rerun")
    ap.add_argument("--backend", choices=LC.BACKENDS, default="sonic")
    ap.add_argument("--task", default=LC.TASKS[0])
    ap.add_argument("--method", default="alone")
    ap.add_argument("--ep-list", default="30")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--qpos", action="store_true")
    a = ap.parse_args()
    if a.phase == "check":
        check(a.group, a.backend)
    else:
        run(a.group, a.backend, a.task, a.method, [int(v) for v in a.ep_list.split(",")], a.shard, a.shards, a.qpos)
