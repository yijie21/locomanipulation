"""Self-model data in LocoManiLab's simulator (2026-11-17): the reach recipe of reachwm_20261030/gen_reach.py (ref
dataset: chained standing / crouching reaches, same command distribution) driven through LocoManiLab's own G1
controllers (g1_backend: HEFT, ScaleBFM, SONIC v1.1 -- the motion-reference ones) on its physics model, and written in
gen_reach's segment layout so reachwm_20261030/data.py and train.py read it unchanged.

How a reach becomes a reference: our whole-body IK (locomani.primitives.whole_body_ik, as Motor.reach with ik_from_ref)
on LocoManiLab's own FK model (g1_backend assets/g1_unified/robot_fk.xml), the hand point LocoManiLab's pipeline uses
for Dex3 (13 cm ahead of wrist_yaw_link, humanoid_benchmark rl/kimodo/heft/scene_frame.PALM_OFFSET), the min-jerk
blend between IK waypoints, written as one any4hdmi motion (qpos[T, 36] at 50 Hz) per reset. Each reset starts from
LocoManiLab's default standing pose; the runner puts the robot exactly on the motion's first frame (its
_reset_playback), so every reset is re-anchored like the self model's data.

How it is driven: g1_backend's IntegratedSim2Sim (through lml_run_integrated's two fixes: the physics model robot.xml
with Dex3 hands and the controller's joint dynamics), built once per shard; for each reset the shard's motion is swapped
in and played to its end, and the robot state is sampled every policy step. Nothing in their checkout is edited.

Not in this first dataset (differences from the ScaleBFM recipe): no approach walks (a walking reference would need a
gait generator) and no grasp-room reaches; the pelvis height in S is the simulator's (the recipe read leg odometry).

    PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
    G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
    export PYTHONPATH=$G:/workspace/code/loco/src
    $PY lml_selfmodel.py gen   --out DIR --resets 20000 --seed 41000000 --id0 4000000 --shards 100 --shard K
    $PY lml_selfmodel.py drive --out DIR --shard K --policy-config $G/checkpoints/heft/pmg/policy.yaml --tag heft
    $PY lml_selfmodel.py merge --out DIR --tag heft
    # protocol 2 (each reach planned from the measured state; no gen phase):
    $PY lml_selfmodel.py drive2 --out DIR --resets 30000 --shard K --policy-config $G/checkpoints/heft/pmg/policy.yaml --tag heft
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

G1B = Path(
    os.environ.get("LML_G1_BACKEND", "/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend")
)
FK_XML = G1B / "assets" / "g1_unified" / "robot_fk.xml"
SCALEBFM_ORDER = (  # the order of S's joints in the self model's data (ScaleBFM metadata; = HEFT/ScaleBFM yaml order)
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)  # fmt: skip
HAND_OFF = np.array([0.13, 0.0, 0.0])  # wrist_yaw_link -> the point a Dex3 hand encloses (LocoManiLab PALM_OFFSET)
FPS, DT = 50, 0.02  # motion frames = policy steps
TICKS = 5  # one data row every 5 policy steps (10 Hz, as gen_reach)
PRE_S, POST_S, START_HOLD_S, CHAIN = 0.3, 1.0, 0.7, 2  # gen_reach.PRE_S / POST_S / CHAIN
REACH_SPEED, REACH_MIN_S, REACH_STEP_M, ROT_W = 0.15, 0.6, 0.03, 0.6  # scalebfm MotorConfig
CORE = (np.array([0.15, -0.40, 0.15]), np.array([0.70, 0.15, 1.35]))  # gen_reach.CORE / WIDE (right hand; y mirrored)
WIDE = (np.array([0.05, -0.60, 0.02]), np.array([1.00, 0.35, 1.50]))
IK_CFG = SimpleNamespace(waist_pitch_max=0.52, waist_dofs="all", pelvis_height_min=0.40, pelvis_pitch_max=0.70)
# LocoManiLab's G1 default standing pose (g1_backend sim2real/config/robots/g1.py default_qpos), MuJoCo joint order
DEFAULT_Q = np.array([-0.2, 0, 0, 0.4, -0.2, 0, -0.2, 0, 0, 0.4, -0.2, 0, 0, 0, 0,
                      0, 0.4, 0, 1.2, 0, 0, 0, 0, -0.4, 0, 1.2, 0, 0, 0])  # fmt: skip
FALL_Z = 0.45


# -- commands and references ----------------------------------------------------------------------------------------
def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def yaw_of(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def sample_command(rng) -> dict:
    """gen_reach.sample_command (no perturbation, no wrist deviation)."""
    side = "right" if rng.random() < 0.5 else "left"
    lo, hi = CORE if rng.random() < 0.7 else WIDE
    t = rng.uniform(lo, hi)
    if side == "left":
        t[1] = -t[1]
    dyaw = float(np.radians(rng.uniform(-60.0, 60.0)))
    pitch = 0.0 if rng.random() < 0.35 else float(np.radians(rng.uniform(0.0, 90.0)))
    pel = None if rng.random() < 0.45 else float(rng.uniform(0.52, 0.76))
    return dict(side=side, target=t, dyaw=dyaw, pitch=pitch, pelvis_z=pel)


class Planner:
    """Whole-body IK on LocoManiLab's FK model and the reach references it gives (Motor.reach, ik_from_ref)."""

    def __init__(self):
        from locomani.primitives.arm_ik import ArmIK

        self.ik = ArmIK(str(FK_XML))
        m, d = self.ik.model, self.ik.data
        self.mj = self.ik._mj
        self.wrist = {s: m.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")}
        self.feet = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
        # standing height: the default pose with its lowest foot point (ankle_roll body origin minus its sole offset as
        # the FK model gives it) where LocoManiLab's default puts it (pelvis 0.8 m)
        self.stand_z = 0.8
        self.d = d

    def fk_world(self, root_pos, root_quat, q_mj):
        """World (pos, R) of both wrist_yaw_links and of the hand points for a reference pose."""
        mj, m, d = self.mj, self.ik.model, self.d
        d.qpos[:] = 0
        d.qpos[self.ik.base_qadr : self.ik.base_qadr + 3] = root_pos
        d.qpos[self.ik.base_qadr + 3 : self.ik.base_qadr + 7] = root_quat
        d.qpos[self.ik.qadr] = q_mj
        mj.mj_kinematics(m, d)
        out = {}
        for s, b in self.wrist.items():
            p, R = d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()
            out[s] = (p + R @ HAND_OFF, R)
        return out

    def start_pose(self):
        return np.array([0.0, 0.0, self.stand_z]), np.array([1.0, 0.0, 0.0, 0.0]), DEFAULT_Q.copy()

    def reach(self, pose, side: str, target_w, rot_w, pz):
        """IK waypoints from `pose` (root pos, quat wxyz, 29 joints) to the hand at target_w / rot_w; returns the poses
        (the first is `pose`), leg durations and the last waypoint's hand error (m)."""
        from scipy.spatial.transform import Rotation

        from locomani.primitives.whole_body_ik import WholeBodyIK

        pos, quat, q = pose
        solver = WholeBodyIK(self.ik, dict(joints=q, pelvis=dict(pos=pos, quat=quat)), IK_CFG, tool_offset=tuple(HAND_OFF),
                             knee_max=1.5, pelvis_forward_max=0.20, pelvis_height_target=pz, jacobian="analytic")  # fmt: skip
        if pz is not None:
            solver.hi[2] = max(solver.hi[2], pz + 0.01)
        solver.seed = np.clip(solver.initial, solver.lo, solver.hi)
        cur = self.fk_world(pos, quat, q)
        a, R0 = cur[side]
        b = np.asarray(target_w, float)
        length = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(length / REACH_STEP_M)))
        dur = max(REACH_MIN_S, length / REACH_SPEED)
        rv = Rotation.from_matrix(R0.T @ rot_w).as_rotvec()
        poses, durs, err = [pose], [], np.inf
        weights = np.array([1.0 if s == side else 0.3 for s in ("left", "right")])
        for k in range(1, n + 1):
            u = k / n
            tp, tr = [], []
            for s in ("left", "right"):
                if s == side:
                    tp.append(a + u * (b - a))
                    tr.append(R0 @ Rotation.from_rotvec(u * rv).as_matrix())
                else:
                    tp.append(cur[s][0])
                    tr.append(cur[s][1])
            solver.hand_weights = weights
            sol = solver.solve(np.array(tp), np.array(tr), hand_rotation_weight=ROT_W)
            qw = Rotation.from_matrix(np.asarray(sol.rotation)).as_quat()[[3, 0, 1, 2]]
            poses.append((np.asarray(sol.position, float), qw, np.asarray(sol.joints, float).copy()))
            durs.append(dur / n)
            err = float(max(sol.hand_errors)) if side == "both" else float(sol.hand_errors[0 if side == "left" else 1])
        return poses, durs, err, dur


def minjerk(u: float) -> float:
    return u * u * u * (10 - 15 * u + 6 * u * u)


def blend_frames(poses, durs) -> np.ndarray:
    """qpos frames (root xyz, wxyz, 29 joints) at FPS through `poses`, min-jerk per leg (Motor._blend); the first frame
    is poses[0], the last poses[-1]."""
    starts = np.cumsum([0.0] + list(durs))
    T = int(round(starts[-1] * FPS))
    out = np.empty((T, 36), np.float32)
    for f in range(1, T + 1):
        tt = min(f / FPS, starts[-1])
        i = int(np.clip(np.searchsorted(starts, tt, side="right") - 1, 0, len(durs) - 1))
        u = (tt - starts[i]) / max(durs[i], 1e-9)
        s = minjerk(min(max(u, 0.0), 1.0))
        (pa, qa, ja), (pb, qb, jb) = poses[i], poses[i + 1]
        qb = qb if float(np.dot(qa, qb)) >= 0 else -qb
        qq = (1 - s) * qa + s * qb
        out[f - 1] = np.r_[pa + s * (pb - pa), qq / np.linalg.norm(qq), ja + s * (jb - ja)]
    return out


def hold(pose, seconds: float) -> np.ndarray:
    p, q, j = pose
    return np.tile(np.r_[p, q, j].astype(np.float32), (int(round(seconds * FPS)), 1))


def plan_reset(pl: Planner, rng) -> tuple[np.ndarray, list[dict]]:
    """One reset's reference: the default stance held START_HOLD_S, then CHAIN reaches, each followed by POST_S (+ the
    next reach's PRE_S). Returns the qpos frames and per reach its frame indices and command (world)."""
    pose = pl.start_pose()
    frames = [hold(pose, START_HOLD_S)]
    reaches = []
    for k in range(CHAIN):
        c = sample_command(rng)
        p, q, _ = pose
        R = rot_z(yaw_of(_qmat(q)))  # heading of the reference pelvis at the command
        target_w = np.r_[p[:2] + (R @ np.r_[c["target"][:2], 0.0])[:2], c["target"][2]]
        rot_w = rot_z(yaw_of(R) + c["dyaw"]) @ rot_y(c["pitch"])
        f0 = sum(len(x) for x in frames)
        try:
            poses, durs, ik_err, dur = pl.reach(pose, c["side"], target_w, rot_w, c["pelvis_z"])
        except (np.linalg.LinAlgError, ValueError) as e:  # gen_reach drops such a reset (degenerate IK)
            return None, [dict(error=f"{type(e).__name__}: {e}")]
        fr = blend_frames(poses, durs)
        frames.append(fr)
        pose = poses[-1]
        f1 = f0 + len(fr)
        # after the last reach a little more than POST_S, so its segment's last row (POST_S after the reach) is played
        frames.append(hold(pose, POST_S + (PRE_S if k < CHAIN - 1 else 0.2)))
        reaches.append(dict(chain=k, side=c["side"], target_w=target_w.tolist(), rot_w=rot_w.tolist(), dyaw=c["dyaw"],
                            pitch=c["pitch"], pelvis_z=c["pelvis_z"], cmd_target=c["target"].tolist(), f_cmd=f0, f_end=f1,
                            ik_err=ik_err, reach_s=dur, ref_q=pose[2].tolist(), ref_pel=np.r_[pose[0], pose[1]].tolist()))  # fmt: skip
    return np.concatenate(frames), reaches


def _qmat(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def gen(out: Path, resets: int, seed: int, id0: int, shards: int, shard: int) -> None:
    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

    d = out / f"motions_{shard:03d}"
    (d / "motions").mkdir(parents=True, exist_ok=True)
    pl = Planner()
    plans, t0 = {}, time.time()
    ids = list(range(shard, resets, shards))
    for n, i in enumerate(ids):
        rng = np.random.default_rng(seed + i)
        fr, reaches = plan_reset(pl, rng)
        if fr is None:
            plans[str(id0 + i)] = dict(skipped=reaches[0]["error"])
            continue
        name = f"r{id0 + i:08d}"
        np.savez_compressed(d / "motions" / f"{name}.npz", qpos=fr)
        plans[name] = dict(reset=id0 + i, n_frames=len(fr), reaches=reaches)
        if (n + 1) % 50 == 0:
            print(f"gen shard {shard}: {n + 1}/{len(ids)} {time.time() - t0:.0f} s", flush=True)
    manifest = dict(format_version=2, dataset_name=f"lml_reach_{shard:03d}", mjcf=str(FK_XML), motions_subdir="motions",
                    timestep=DT, qpos_dim=36,
                    qpos_names=["root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz", *MUJOCO_JOINT_NAMES],
                    num_motions=sum(1 for k in plans if k.startswith("r")))  # fmt: skip
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (d / "plans.json").write_text(json.dumps(plans))
    print(
        f"gen shard {shard}: {manifest['num_motions']} motions, {len(ids) - manifest['num_motions']} skipped, {time.time() - t0:.0f} s",
        flush=True,
    )


# -- driving LocoManiLab's controllers ------------------------------------------------------------------------------
class Driver:
    def __init__(self, policy_config: str, motion_dir: Path):
        # their ONNXModule leaves onnxruntime's intra-op threads at its default (one per core) unless this is set: 100
        # shards then ran ~6 runnable threads each (load 650 on 128 cores) and a policy step took 72 ms instead of ~5
        os.environ.setdefault("SIM2REAL_ORT_NUM_THREADS", "1")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import lml_run_integrated  # patches load_sim_model and the root joint name before construction
        import mujoco as mj
        import sim2real.sim_env.integrated_sim2sim as I
        from sim2real.rl_policy.utils.motion import MotionDataset

        lml_run_integrated.set_dynamics(policy_config)
        args = I.IntegratedSim2SimArgs(policy_config=policy_config, motion_path=str(motion_dir), robot="g1",
                                       headless=True, initial_pause_s=0.5, run_once=True)  # fmt: skip
        self.runner = r = I.IntegratedSim2Sim(args)
        sp = r.state_processor
        self.full = MotionDataset.create_from_path(
            str(motion_dir), robot_cfg=r.robot_cfg, mjcf_path=sp.motion_config.get("mjcf_path")
        )
        self.m, self.d = m, d = r.sim.mj_model, r.sim.mj_data
        jid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self.qadr = np.array([m.jnt_qposadr[jid(n)] for n in SCALEBFM_ORDER])
        self.vadr = np.array([m.jnt_dofadr[jid(n)] for n in SCALEBFM_ORDER])
        root = jid("pelvis")
        self.rq, self.rv = int(m.jnt_qposadr[root]), int(m.jnt_dofadr[root])
        bid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.wrist = {s: bid(f"{s}_wrist_yaw_link") for s in ("left", "right")}
        self.torso = bid("torso_link")
        self.mj_joint_names = list(self.full.joint_names)  # the dataset's joint order, for the motion check

    def sample(self) -> dict:
        """The robot this policy step, compactly: root, joints (ScaleBFM order), both wrists, the torso."""
        d = self.d
        return dict(root=d.qpos[self.rq : self.rq + 7].copy(), q=d.qpos[self.qadr].copy(), dq=d.qvel[self.vadr].copy(),
                    w=d.qvel[self.rv + 3 : self.rv + 6].copy(),
                    wr={s: (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for s, b in self.wrist.items()},
                    torso=(d.xpos[self.torso].copy(), d.xmat[self.torso].reshape(3, 3).copy()))  # fmt: skip

    def check_motion(self, frames: np.ndarray) -> float:
        """Max |joint| difference between the selected motion and `frames` at three frames: the selected index is the
        file (all resets share the first frame, so the start pose alone cannot tell). Their loader keeps motions in
        float16, so a match is ~1e-3 rad."""
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        sp = self.runner.state_processor
        cols = [MUJOCO_JOINT_NAMES.index(n) for n in self.mj_joint_names]
        worst = 0.0
        for f in (len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1):
            md = sp.motion_dataset.get_slice(np.array([0]), np.array([f]), np.array([0]))
            jp = np.asarray(md.joint_pos).reshape(-1)[: len(cols)]
            worst = max(worst, float(np.abs(jp - frames[f][7:][cols]).max()))
        return worst

    def run_reset(self, idx: int, frames: np.ndarray) -> tuple[dict, dict]:
        r, sp = self.runner, self.runner.state_processor
        sp.motion_dataset = self.full.select_motions([idx])
        sp.motion_length = sp.motion_dataset.num_steps
        sp.motion_t[:] = 0
        sp._update_motion_data()
        r._reset_playback()
        info = dict(motion_check=self.check_motion(frames), length_ok=int(sp.motion_length) == len(frames),
                    first_qpos_err=float(np.abs(np.r_[self.d.qpos[self.rq : self.rq + 7], self.d.qpos[self.qadr]][:7] - frames[0][:7]).max()))  # fmt: skip
        rows, fell, sim_count = {}, None, 0
        while True:  # until the motion's last frame has been reached
            r._maybe_start_motion(sim_elapsed_s=sim_count * r.args.sim_dt)
            r._policy_step()
            r.policy.total_inference_cnt += 1
            for _ in range(r.args.decimation):
                r.sim.sim_step()
                sim_count += 1
            if not r.playback_started:
                continue
            f = int(sp.motion_t[0])
            rows[f] = self.sample()
            if self.d.qpos[self.rq + 2] < FALL_Z:
                fell = f
                break
            if f >= len(frames) - 1:
                break
        info.update(fell=fell, steps=len(rows), sim_s=sim_count * r.args.sim_dt)
        return rows, info


def segments(rows: dict, fell, frames: np.ndarray, plan: dict, pl: Planner) -> list[dict]:
    """gen_reach segments from one reset's per-step samples: a row every TICKS steps from PRE_S before the command to
    POST_S after the reference reach ends; a fall ends the reset (that segment kept with fell=True, if it has a start)."""
    out = []
    for rc in plan["reaches"]:
        f_cmd = rc["f_cmd"]
        f_pre, f_end = f_cmd - int(round(PRE_S * FPS)), rc["f_end"] + int(round(POST_S * FPS))
        want = list(range(f_pre, f_end + 1, TICKS))
        seg_fell = fell is not None and fell <= f_end
        if seg_fell:
            want = [f for f in want if f <= fell]
        if f_cmd not in want or any(f not in rows for f in want) or len(want) < want.index(f_cmd) + 2:
            break
        side = rc["side"]
        S, Wr, Jr, Cr = [], [], [], []
        for f in want:
            x = rows[f]
            p, Rp = x["root"][:3], _qmat(x["root"][3:])
            hands = {s: (pw + Rw @ HAND_OFF, Rw) for s, (pw, Rw) in x["wr"].items()}
            rf = frames[min(f, len(frames) - 1)]
            ref_hand = pl.fk_world(rf[:3], rf[3:7], rf[7:])[side][0]
            S.append(np.r_[x["q"], x["dq"], Rp.T @ np.array([0.0, 0.0, -1.0]), x["w"], p[2]].astype(np.float32))
            Wr.append(
                np.r_[
                    p,
                    yaw_of(Rp),
                    hands["right"][0],
                    hands["left"][0],
                    x["torso"][0] + x["torso"][1] @ np.array([0, 0, 0.35]),
                ].astype(np.float32)
            )
            Rj = hands[side][1]
            Jr.append(np.r_[Rj[:, 0], Rj[:, 1]].astype(np.float32))
            Cr.append(np.r_[Rp.T @ (ref_hand - p), -1.0 if f < f_cmd else (f - f_cmd) / FPS].astype(np.float32))
        k0 = want.index(f_cmd)
        p0, yaw0 = Wr[k0][:3].astype(float), float(Wr[k0][3])
        Rh = rot_z(yaw0)
        tw, rw = np.array(rc["target_w"]), np.array(rc["rot_w"])
        tg = Rh.T @ np.r_[tw[:2] - p0[:2], 0.0]
        rot_h = Rh.T @ rw
        last = rows[want[-1]]
        pw, Rw = last["wr"][side]
        land = pw + Rw @ HAND_OFF
        att = float(np.degrees(np.arccos(np.clip((np.trace(rw.T @ Rw) - 1) / 2, -1, 1))))
        jaw_err = 100 * float(np.linalg.norm(land - tw))
        out.append(dict(S=np.array(S), W=np.array(Wr), J=np.array(Jr), C=np.array(Cr), k0=k0, side=1 if side == "right" else -1,
                        target=np.r_[tg[:2], tw[2]].astype(np.float32), dyaw=rc["dyaw"], pitch=rc["pitch"],
                        rot6=np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32),
                        pelvis_z=-1.0 if rc["pelvis_z"] is None else rc["pelvis_z"], start=np.r_[p0, yaw0].astype(np.float32),
                        walked=False, chain=rc["chain"], homed=False, fell=seg_fell, ok=jaw_err < 2.0, jaw_err_cm=jaw_err,
                        att_err_deg=att, ik_err_cm=100 * rc["ik_err"], reach_s=rc["reach_s"], reset=plan["reset"],
                        ref_s=rc["reach_s"], n_sidesteps=0, ref_q=np.array(rc["ref_q"], np.float32),
                        ref_pel=np.array(rc["ref_pel"], np.float32), sidestep=np.zeros(2, np.float32),
                        wrist_dev=np.zeros(3, np.float32)))  # fmt: skip
        if seg_fell:
            break
    return out


SEG_SCALARS = ("k0", "side", "dyaw", "pitch", "pelvis_z", "walked", "chain", "homed", "fell", "ok", "jaw_err_cm",
               "att_err_deg", "ik_err_cm", "reach_s", "reset", "ref_s", "n_sidesteps")  # fmt: skip
SEG_SCALARS2 = SEG_SCALARS + ("lost",)  # protocol 2
SEG_VECTORS = ("target", "rot6", "start", "ref_q", "ref_pel", "sidestep", "wrist_dev")


def pack(segs: list[dict]) -> dict:
    """gen_reach.save's layout, in memory."""
    out = {k: np.concatenate([s[k] for s in segs]) for k in ("S", "W", "J", "C")}
    out["n_steps"] = np.array([len(s["S"]) for s in segs])
    for k in SEG_SCALARS2 if "lost" in segs[0] else SEG_SCALARS:
        out[k] = np.array([s[k] for s in segs])
    for k in SEG_VECTORS:
        out[k] = np.stack([s[k] for s in segs])
    return out


def save(segs: list[dict], path: Path) -> None:
    np.savez_compressed(path, **pack(segs))


CHUNK = 50  # resets per saved file: a rerun skips finished chunks, so a container restart loses at most one


def drive(out: Path, shard: int, policy_config: str, tag: str, limit: int | None) -> None:
    md = out / f"motions_{shard:03d}"
    plans = json.loads((md / "plans.json").read_text())
    names = sorted(k for k in plans if k.startswith("r"))
    n = min(limit or len(names), len(names))
    od = out / f"data_{tag}"
    od.mkdir(parents=True, exist_ok=True)
    chunks = [range(c, min(c + CHUNK, n)) for c in range(0, n, CHUNK)]
    done = lambda c: (od / f"w{shard:03d}_{c:03d}_log.json").exists()  # noqa: E731 -- written after the chunk's data
    before = sum(len(r) for c, r in enumerate(chunks) if done(c))
    print(f"drive {tag} shard {shard}: {before}/{len(names)} resets already driven", flush=True)
    if before == n:
        return
    dr, pl = Driver(policy_config, md), Planner()
    assert dr.full.num_motions == len(names), (dr.full.num_motions, len(names))
    t0, nres, nseg, nfall = time.time(), 0, 0, 0
    for c, ids in enumerate(chunks):
        if done(c):
            continue
        segs, log = [], []
        for idx in ids:  # any4hdmi's loader sorts by name; check_motion verifies
            name = names[idx]
            frames = np.load(md / "motions" / f"{name}.npz")["qpos"]
            rows, info = dr.run_reset(idx, frames)
            # their dataset stores motions in float16 (any4hdmi storage_dtype): 1e-3 rad of rounding; another reset's
            # motion differs by tenths of a radian
            assert info["length_ok"] and info["motion_check"] < 5e-3, (
                f"{name}: the runner plays another motion ({info})"
            )
            s = segments(rows, info["fell"], frames, plans[name], pl)
            segs += s
            log.append(dict(name=name, n_segs=len(s), **info))
            nres, nseg, nfall = nres + 1, nseg + len(s), nfall + (info["fell"] is not None)
            if nres % 10 == 0:
                print(f"drive {tag} shard {shard}: {before + nres}/{len(names)} resets, {nseg} segments, falls {nfall}, "
                      f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip
        if segs:
            tmp = od / f".w{shard:03d}_{c:03d}.tmp.npz"  # not matched by merge's glob until renamed
            save(segs, tmp)
            os.replace(tmp, od / f"w{shard:03d}_{c:03d}.npz")
        (od / f"w{shard:03d}_{c:03d}_log.json").write_text(json.dumps(log))
    print(f"drive {tag} shard {shard} done: {before + nres} resets ({nres} this run, {nseg} segments, falls {nfall}), "
          f"{time.time() - t0:.0f} s", flush=True)  # fmt: skip


# -- protocol 2 (2026-11-17, locomanilab_selfmodel2): every reach planned from the measured state ------------------
# Protocol 1 above planned a reset's whole reference offline, so the second reach started from the first reach's
# REFERENCE end while the robot was elsewhere; the recipe (gen_reach -> Motor.reach, servo off) plans each reach when it
# is commanded: the command is placed with the measured pelvis, the IK path starts at the measured hand and is solved on
# the measured posture, and the reference blends into it from where the reference is. Here that happens inside the
# drive: at each command row the reach is planned from the robot's state and spliced into the runner's motion after the
# current frame (the frames already played are unchanged, so the controller sees a new command, not a jump). Both
# controllers read the reference root only as displacements (HEFT: future root minus current root; ScaleBFM:
# reference_forcing) but track its heading against the robot's, so the plan is shifted in x/y onto the current
# reference root and keeps the measured heading (as Motor.reach builds its poses on the measured yaw): the
# reference-to-robot heading offset is not a hidden input of the self model.
LOST_M = (
    0.5  # a reach whose pelvis displacement departs this far from the reference's ends the reset (protocol 1 kept them)
)
# No tilt criterion: in protocol 1's data 11 % (ScaleBFM) of the segments that did not fall had the pelvis tilted
# > 45 deg at some row (deep crouches, IK pelvis pitch up to 0.70 rad) with ordinary errors (median 12.8 cm vs 10.6).
MARGIN_S = 0.5  # extra hold after each spliced reach so the motion never runs out before the next splice


class MotionBuilder:
    """The runner's in-memory motion from qpos frames, exactly as their loader builds it (any4hdmi compute_motion_qvel
    + FKRunner on the motion's MJCF, MotionDataset's joint reorder and float16 storage; checked equal to the loader's
    arrays, max |diff| 0) -- without the loader's on-disk FK cache, which would keep one entry per splice."""

    def __init__(self, template):
        import torch  # noqa: F401 -- FKRunner takes tensors
        from any4hdmi.fk.runner import FKRunner

        self.fk = FKRunner(FK_XML, batch_size=4096, device="cpu")
        self.t = template

    def build(self, frames: np.ndarray):
        import torch
        from any4hdmi.utils.dataset import compute_motion_qvel
        from sim2real.rl_policy.utils.motion import MotionDataset

        t = self.t
        qv = compute_motion_qvel(self.fk.model, frames, float(FPS))
        out = self.fk.forward_kinematics(
            torch.as_tensor(frames, dtype=torch.float32), torch.as_tensor(qv, dtype=torch.float32)
        )
        n = len(frames)
        st = {"motion_id": np.zeros(n, np.int64), "step": np.arange(n, dtype=np.int64)}
        for k, v in out.items():
            if k not in t._storage:
                continue
            v = v.numpy()
            if k in ("joint_pos", "joint_vel"):
                r = np.zeros((n, t._storage[k].shape[1]), v.dtype)
                r[:, t._dest_joint_indices] = v[:, t._src_joint_indices]
                v = r
            st[k] = v.astype(t._storage[k].dtype)
        ds = object.__new__(MotionDataset)
        ds._dataset, ds.body_names, ds.joint_names = t._dataset, t.body_names, t.joint_names
        ds._src_joint_indices, ds._dest_joint_indices = t._src_joint_indices, t._dest_joint_indices
        ds._source_starts, ds._source_ends = np.array([0], np.int64), np.array([n], np.int64)
        ds._storage, ds._motion_ids = st, np.array([0], np.int64)
        ds._refresh_selected_bounds()
        return ds


class OnlineDriver(Driver):
    """Driver for protocol 2: one reset = the default stance, then CHAIN reaches each planned at its command row."""

    def __init__(self, policy_config: str, init_dir: Path):
        super().__init__(policy_config, init_dir)
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        self.builder = MotionBuilder(self.full)
        self.to_mj = np.array([SCALEBFM_ORDER.index(n) for n in MUJOCO_JOINT_NAMES])  # S order -> MuJoCo order
        self.plan_s = 0.0

    def _swap(self, frames: np.ndarray) -> None:
        sp = self.runner.state_processor
        sp.motion_dataset = self.builder.build(frames)
        sp.motion_length = sp.motion_dataset.num_steps
        sp._update_motion_data()

    def _plan(self, pl: "Planner", x: dict, ref_frame: np.ndarray, c: dict):
        """Plan reach `c` from the measured sample `x` (as gen_reach + Motor.reach, servo off); returns the blended
        frames (to follow `ref_frame`, the current reference frame) and the reach record."""
        p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
        yaw = yaw_of(_qmat(quat))
        Rh = rot_z(yaw)
        target_w = np.r_[p[:2] + (Rh @ np.r_[c["target"][:2], 0.0])[:2], c["target"][2]]
        rot_w = rot_z(yaw + c["dyaw"]) @ rot_y(c["pitch"])
        q_mj = x["q"][self.to_mj].astype(float)
        t0 = time.time()
        poses, durs, ik_err, dur = pl.reach((p, quat, q_mj), c["side"], target_w, rot_w, c["pelvis_z"])
        self.plan_s += time.time() - t0
        delta = np.r_[ref_frame[:2] - p[:2], 0.0]  # measured world -> reference coordinates (x/y shift)
        ref_pose = (ref_frame[:3].astype(float), ref_frame[3:7].astype(float), ref_frame[7:].astype(float))
        shifted = [ref_pose] + [(pp + delta, qq, jj) for pp, qq, jj in poses[1:]]
        last = poses[-1]
        rec = dict(side=c["side"], target_w=target_w.tolist(), rot_w=rot_w.tolist(), dyaw=c["dyaw"], pitch=c["pitch"],
                   pelvis_z=c["pelvis_z"], cmd_target=c["target"].tolist(), ik_err=ik_err, reach_s=dur,
                   ref_q=last[2].tolist(), ref_pel=np.r_[last[0], last[1]].tolist(), delta=delta.tolist())  # fmt: skip
        return blend_frames(shifted, durs), shifted[-1], rec

    def run_reset_online(self, pl: "Planner", rng) -> tuple[dict, dict, np.ndarray, list[dict]]:
        import mujoco as mj

        r, sp, m, d = self.runner, self.runner.state_processor, self.m, self.d
        cmds = [sample_command(rng) for _ in range(CHAIN)]  # same draws, same order as protocol 1's plan_reset
        start = pl.start_pose()
        frames = hold(start, START_HOLD_S + MARGIN_S + PRE_S)
        mj.mj_resetData(m, d)  # fingers, velocities, warm start: nothing carried over from the previous reset
        mj.mj_forward(m, d)
        self._swap(frames)
        sp.motion_t[:] = 0
        sp._update_motion_data()
        r._reset_playback()
        reaches, rows, fell, lost, why, sim_count = [], {}, None, None, "", 0
        next_cmd, end_frame, k = int(round(START_HOLD_S * FPS)), None, 0
        anchor = None  # (measured pelvis xy, reference root xy) at the active reach's command row
        while True:
            r._maybe_start_motion(sim_elapsed_s=sim_count * r.args.sim_dt)
            r._policy_step()
            r.policy.total_inference_cnt += 1
            for _ in range(r.args.decimation):
                r.sim.sim_step()
                sim_count += 1
            if not r.playback_started:
                continue
            f = int(sp.motion_t[0])
            x = rows[f] = self.sample()
            if float(x["root"][2]) < FALL_Z:
                fell, why = f, "height"
                break
            if anchor is not None:
                dev = (x["root"][:2] - anchor[0]) - (frames[min(f, len(frames) - 1), :2] - anchor[1])
                if float(np.linalg.norm(dev)) > LOST_M:
                    fell, lost, why = f, f, "lost"
                    break
            if k < CHAIN and f == next_cmd:
                try:
                    blend, last, rec = self._plan(pl, x, frames[f], cmds[k])
                except (np.linalg.LinAlgError, ValueError) as e:  # gen_reach drops such a reset (degenerate IK)
                    why = f"ik {type(e).__name__}: {e}"
                    break
                rec.update(chain=k, f_cmd=f, f_end=f + len(blend))
                tail = POST_S + (PRE_S if k < CHAIN - 1 else 0.2) + MARGIN_S
                frames = np.concatenate([frames[: f + 1], blend, hold(last, tail)])
                self._swap(frames)
                reaches.append(rec)
                anchor = (x["root"][:2].copy(), frames[f, :2].copy())
                post = int(round(POST_S * FPS))
                next_cmd = rec["f_end"] + post + int(round(PRE_S * FPS))
                end_frame = rec["f_end"] + post
                k += 1
            if k == CHAIN and end_frame is not None and f >= end_frame:
                break
            if f >= len(frames) - 1:  # the motion ran out (must not happen: MARGIN_S)
                why = "motion ended"
                break
        info = dict(fell=fell, lost=lost, why=why, steps=len(rows), sim_s=sim_count * r.args.sim_dt, planned=len(reaches),
                    motion_check=self.check_motion(frames) if len(frames) > 3 else 0.0)  # fmt: skip
        return rows, info, frames, reaches


def segments2(rows: dict, fell, lost, frames: np.ndarray, reaches: list[dict], reset: int, pl: Planner) -> list[dict]:
    """segments() for protocol 2: the same rows and fields; the reference hand in C is taken back to measured
    coordinates with the shift of the reach whose reference is playing; target / ref_pel are measured-world (as
    gen_reach); a lost reach is kept like a fall (fell=True, lost=True)."""
    out = []
    for i, rc in enumerate(reaches):
        f_cmd = rc["f_cmd"]
        f_pre, f_end = f_cmd - int(round(PRE_S * FPS)), rc["f_end"] + int(round(POST_S * FPS))
        want = list(range(f_pre, f_end + 1, TICKS))
        seg_fell = fell is not None and fell <= f_end
        if seg_fell:
            want = [f for f in want if f <= fell]
        if f_cmd not in want or any(f not in rows for f in want) or len(want) < want.index(f_cmd) + 2:
            break
        side = rc["side"]
        d_prev = np.array(reaches[i - 1]["delta"]) if i > 0 else np.zeros(3)
        d_this = np.array(rc["delta"])
        S, Wr, Jr, Cr = [], [], [], []
        for f in want:
            x = rows[f]
            p, Rp = x["root"][:3], _qmat(x["root"][3:])
            hands = {s: (pw + Rw @ HAND_OFF, Rw) for s, (pw, Rw) in x["wr"].items()}
            rf = frames[min(f, len(frames) - 1)]
            ref_hand = pl.fk_world(rf[:3], rf[3:7], rf[7:])[side][0] - (d_this if f > f_cmd else d_prev)
            S.append(np.r_[x["q"], x["dq"], Rp.T @ np.array([0.0, 0.0, -1.0]), x["w"], p[2]].astype(np.float32))
            Wr.append(
                np.r_[
                    p,
                    yaw_of(Rp),
                    hands["right"][0],
                    hands["left"][0],
                    x["torso"][0] + x["torso"][1] @ np.array([0, 0, 0.35]),
                ].astype(np.float32)
            )
            Rj = hands[side][1]
            Jr.append(np.r_[Rj[:, 0], Rj[:, 1]].astype(np.float32))
            Cr.append(np.r_[Rp.T @ (ref_hand - p), -1.0 if f < f_cmd else (f - f_cmd) / FPS].astype(np.float32))
        k0 = want.index(f_cmd)
        p0, yaw0 = Wr[k0][:3].astype(float), float(Wr[k0][3])
        Rh = rot_z(yaw0)
        tw, rw = np.array(rc["target_w"]), np.array(rc["rot_w"])
        tg = Rh.T @ np.r_[tw[:2] - p0[:2], 0.0]
        rot_h = Rh.T @ rw
        pw, Rw = rows[want[-1]]["wr"][side]
        land = pw + Rw @ HAND_OFF
        att = float(np.degrees(np.arccos(np.clip((np.trace(rw.T @ Rw) - 1) / 2, -1, 1))))
        jaw_err = 100 * float(np.linalg.norm(land - tw))
        seg_lost = seg_fell and lost is not None and lost == fell
        out.append(dict(S=np.array(S), W=np.array(Wr), J=np.array(Jr), C=np.array(Cr), k0=k0, side=1 if side == "right" else -1,
                        target=np.r_[tg[:2], tw[2]].astype(np.float32), dyaw=rc["dyaw"], pitch=rc["pitch"],
                        rot6=np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32),
                        pelvis_z=-1.0 if rc["pelvis_z"] is None else rc["pelvis_z"], start=np.r_[p0, yaw0].astype(np.float32),
                        walked=False, chain=rc["chain"], homed=False, fell=seg_fell, lost=seg_lost, ok=jaw_err < 2.0,
                        jaw_err_cm=jaw_err, att_err_deg=att, ik_err_cm=100 * rc["ik_err"], reach_s=rc["reach_s"], reset=reset,
                        ref_s=rc["reach_s"], n_sidesteps=0, ref_q=np.array(rc["ref_q"], np.float32),
                        ref_pel=np.array(rc["ref_pel"], np.float32), sidestep=np.zeros(2, np.float32),
                        wrist_dev=np.zeros(3, np.float32)))  # fmt: skip
        if seg_fell:
            break
    return out


def drive2(out: Path, shard: int, policy_config: str, tag: str, resets: int, seed: int, id0: int, shards: int,
           limit: int | None) -> None:  # fmt: skip
    ids = list(range(shard, resets, shards))
    n = min(limit or len(ids), len(ids))
    od = out / f"data_{tag}"
    od.mkdir(parents=True, exist_ok=True)
    chunks = [range(c, min(c + CHUNK, n)) for c in range(0, n, CHUNK)]
    done = lambda c: (od / f"w{shard:03d}_{c:03d}_log.json").exists()  # noqa: E731 -- written after the chunk's data
    before = sum(len(r) for c, r in enumerate(chunks) if done(c))
    print(f"drive2 {tag} shard {shard}: {before}/{len(ids)} resets already driven", flush=True)
    if before == n:
        return
    init = out / "init" / f"s{shard:03d}"  # the stance motion the runner is built with (one small loader cache entry)
    if not (init / "manifest.json").exists():
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        (init / "motions").mkdir(parents=True, exist_ok=True)
        pl0 = Planner()
        np.savez_compressed(
            init / "motions" / "stance.npz", qpos=hold(pl0.start_pose(), START_HOLD_S + MARGIN_S + PRE_S)
        )
        (init / "manifest.json").write_text(json.dumps(dict(format_version=2, dataset_name="lml_stance", mjcf=str(FK_XML),
            motions_subdir="motions", timestep=DT, qpos_dim=36, num_motions=1,
            qpos_names=["root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz", *MUJOCO_JOINT_NAMES])))  # fmt: skip
    dr, pl = OnlineDriver(policy_config, init), Planner()
    t0, nres, nseg, nfall, nlost = time.time(), 0, 0, 0, 0
    for c, rng_ids in enumerate(chunks):
        if done(c):
            continue
        segs, log = [], []
        for j in rng_ids:
            i = ids[j]
            rows, info, frames, reaches = dr.run_reset_online(pl, np.random.default_rng(seed + i))
            assert info["motion_check"] < 5e-3, f"reset {id0 + i}: the runner plays another motion ({info})"
            s = segments2(rows, info["fell"], info["lost"], frames, reaches, id0 + i, pl)
            segs += s
            log.append(dict(reset=id0 + i, n_segs=len(s), **info))
            nres, nseg = nres + 1, nseg + len(s)
            nfall, nlost = (
                nfall + (info["fell"] is not None and info["lost"] is None),
                nlost + (info["lost"] is not None),
            )
            if nres % 10 == 0:
                print(f"drive2 {tag} shard {shard}: {before + nres}/{len(ids)} resets, {nseg} segments, falls {nfall}, "
                      f"lost {nlost}, {time.time() - t0:.0f} s (planning {dr.plan_s:.0f} s)", flush=True)  # fmt: skip
        if segs:
            tmp = od / f".w{shard:03d}_{c:03d}.tmp.npz"
            save(segs, tmp)
            os.replace(tmp, od / f"w{shard:03d}_{c:03d}.npz")
        (od / f"w{shard:03d}_{c:03d}_log.json").write_text(json.dumps(log))
    print(f"drive2 {tag} shard {shard} done: {before + nres} resets ({nres} this run, {nseg} segments, falls {nfall}, "
          f"lost {nlost}), {time.time() - t0:.0f} s", flush=True)  # fmt: skip


def merge(out: Path, tag: str) -> None:
    od = out / f"data_{tag}"
    zs = [dict(np.load(p)) for p in sorted(od.glob("w[0-9]*_[0-9]*.npz"))]
    allz = {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}
    np.savez_compressed(od / "all.npz", **allz)
    print(f"{tag}: {len(allz['n_steps'])} segments, {int(allz['n_steps'].sum())} rows, fell {int(allz['fell'].sum())}, "
          f"jaw_err median {np.median(allz['jaw_err_cm'][~allz['fell'].astype(bool)]):.2f} cm", flush=True)  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gen", "drive", "drive2", "merge"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resets", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=41000000)
    ap.add_argument("--id0", type=int, default=4000000)
    ap.add_argument("--shards", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--policy-config", default="")
    ap.add_argument("--tag", default="heft")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.phase == "gen":
        gen(a.out, a.resets, a.seed, a.id0, a.shards, a.shard)
    elif a.phase == "drive":
        drive(a.out, a.shard, a.policy_config, a.tag, a.limit)
    elif a.phase == "drive2":
        drive2(a.out, a.shard, a.policy_config, a.tag, a.resets, a.seed, a.id0, a.shards, a.limit)
    else:
        merge(a.out, a.tag)
