"""Closed-loop evaluation of a GR00T policy trained on record_bottle.py data, in-process.

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl $VENV_SIM_PY src/teleop_free/scalebfm/eval_bfm_bottle.py \
        --model-path outputs/experiments/<run>/model/checkpoint-3000 --rounds 60 --seed 6000 \
        --out outputs/experiments/<run>/eval_bfm --video

Same scene sampling, settle and success predicate as run_bottle.py (the expert); the
expert's tick is replaced by the VLA:
  * every --replan-s of sim time an observation is rendered (ego_view + right_wrist,
    the 53-d state assembled exactly as record_bottle.py / vla_round.build) and the
    policy predicts a 40-row chunk at DATASET_FPS (25 Hz) in the pelvis frame of
    THAT observation;
  * the chunk becomes usable --latency-ticks later (inference wall time is measured
    and reported; the default models it as 4 ticks = 80 ms) and is executed from the
    latency-compensated row, NVIDIA-client style: ScaleBFM's 6 future frames at tick
    k are chunk rows ((k - k0 + j) // 2), converted back to world with the pelvis
    pose stored at prediction time;
  * the left wrist (not in the action) follows the predicted pelvis with the same
    at-the-side offset the expert used; the gripper target is the chunk's scalar.
No SONIC, no ZMQ, no wall clock: a round is reproducible for a fixed seed and
policy seed. Traces carry the fields compare_arms.py / classify_eval.py read.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
import run_spike as R  # noqa: E402
import run_bottle as B  # noqa: E402
from run_spike import DT, GRIP_OPEN, PELVIS_Z, SETTLE_S, mat, quat_wxyz, rot_z, yaw_of, wrap, Sim, Policy, Refs  # noqa: E402
from g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

W, H = 640, 480
DATASET_FPS = 25
ROWS_PER_TICK = DATASET_FPS / 50.0          # 0.5 chunk rows per 50 Hz tick
VEL_SCALE = np.ones(7)
TASK = "pick up the water bottle"


def rot6d_to_mat(d6):
    a1, a2 = d6[:3], d6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


class Observer:
    """Renders the two cameras and assembles the 53-d state the way the data was recorded."""

    def __init__(self, sim: Sim):
        m = sim.m
        self.sim = sim
        self.r = mujoco.Renderer(m, height=H, width=W)
        self.cam = {"ego_view": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"),
                    "right_wrist": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "right_wrist")}
        self.q29 = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MUJOCO_JOINT_NAMES])

    def __call__(self):
        sim, d = self.sim, self.sim.d
        f32 = lambda a: np.asarray(a, dtype=np.float32).reshape(1, -1)  # noqa: E731
        root_pos = d.qpos[:3].copy(); root_q = d.qpos[3:7].copy(); Rp = mat(root_q)
        q = d.qpos[self.q29]
        lh = np.zeros(7); lh[4:6] = d.qpos[sim.gq[:2]]
        rh = np.zeros(7); rh[4:6] = d.qpos[sim.gq[2:]]
        wp, wq = sim.body("right_wrist_yaw_link")
        obs = {
            "state.left_leg": f32(q[0:6]), "state.right_leg": f32(q[6:12]), "state.waist": f32(q[12:15]),
            "state.left_arm": f32(q[15:22]), "state.right_arm": f32(q[22:29]),
            "state.left_hand": f32(lh), "state.right_hand": f32(rh),
            "state.projected_gravity": f32(Rp.T @ np.array([0.0, 0.0, -1.0])),
            "state.right_wrist_pos": f32(Rp.T @ (wp - root_pos)),
            "state.right_wrist_abs_quat": f32(quat_wxyz(Rp.T @ mat(wq))),
            "annotation.human.task_description": TASK,
        }
        for key, cid in self.cam.items():
            self.r.update_scene(d, camera=cid)
            obs[f"video.{key}"] = np.ascontiguousarray(self.r.render())[None].astype(np.uint8)
        return obs, (root_pos, Rp)


def load_replay_body(dataset, episode):
    """Round-trip test in the POLICY's own terms: the recorded rows in their own pelvis frame
    plus the 7 velocity columns of bfm_action.velocity_columns, executed like a prediction
    (no recorded root pose at execution time). Passes only if the executor can walk from
    per-row body-frame targets, which is what a trained policy will hand it."""
    import pyarrow.parquet as pq
    from bfm_action import velocity_columns
    t = pq.read_table(os.path.join(dataset, f"data/chunk-000/episode_{episode:06d}.parquet"))
    act = np.stack(t["action.bfm"].to_pylist()).astype(np.float64)
    grip = np.stack(t["teleop.right_hand_joints"].to_pylist())[:, 0].astype(np.float64)
    rq = np.stack(t["observation.root_orientation"].to_pylist()); rp = np.stack(t["observation.root_position"].to_pylist())
    rows = np.concatenate([act, velocity_columns(act, rp, rq).astype(np.float64)], axis=1)
    n = len(act)
    state = dict(k=0)

    def predict(obs):
        r0 = min(n - 1, state["k"] // 2)
        idx = np.clip(np.arange(r0, r0 + 40), 0, n - 1)
        return rows[idx], grip[idx]
    predict.state = state
    return predict


def load_replay(dataset, episode):
    """Round-trip test: the recorded action.bfm of one episode, decoded with the recorded
    root pose per row, in place of the VLA. Run with the recorder's seed and the same
    round index so the scene is identical; a faithful pipeline reproduces the grasp."""
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(dataset, f"data/chunk-000/episode_{episode:06d}.parquet"))
    act = np.stack(t["action.bfm"].to_pylist()).astype(np.float64)
    grip = np.stack(t["teleop.right_hand_joints"].to_pylist())[:, 0].astype(np.float64)
    rq = np.stack(t["observation.root_orientation"].to_pylist()); rp = np.stack(t["observation.root_position"].to_pylist())
    # pre-convert every row to WORLD with its own recorded pelvis pose, then re-express in
    # a fixed identity frame so the caller's (root_pos0, R0) conversion is a no-op
    world = np.zeros_like(act)
    for r in range(len(act)):
        Rr = mat(rq[r]); p = rp[r]
        world[r, 0:3] = p + Rr @ act[r, 0:3]; world[r, 3:9] = (Rr @ rot6d_to_mat(act[r, 3:9]))[:, :2].T.reshape(-1)
        world[r, 9:12] = p + Rr @ act[r, 9:12]; world[r, 12:18] = (Rr @ rot6d_to_mat(act[r, 12:18]))[:, :2].T.reshape(-1)
    n = len(act)
    state = dict(k=0)

    def predict(obs):
        # rows from the current tick on (the recorder wrote one row per 2 ticks from tick 0)
        r0 = min(n - 1, state["k"] // 2)
        idx = np.clip(np.arange(r0, r0 + 40), 0, n - 1)
        return world[idx], grip[idx]
    predict.world_frame = True
    predict.state = state
    return predict


def load_vla(model_path, modality_config, denoising_steps):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    cfg_path = os.path.abspath(modality_config)
    sys.path.insert(0, os.path.dirname(cfg_path))
    __import__(os.path.splitext(os.path.basename(cfg_path))[0])
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.eval.open_loop_eval import parse_action_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    tag = EmbodimentTag.UNITREE_G1_SONIC
    cfg = MODALITY_CONFIGS[tag.value]
    policy = Gr00tPolicy(model_path=model_path, embodiment_tag=tag, device="cuda")
    policy.model.action_head.num_inference_timesteps = denoising_steps

    def predict(obs):
        chunk, _ = policy.get_action(parse_observation_gr00t(obs, cfg))
        out = {k: np.asarray(v) for k, v in parse_action_gr00t(chunk).items()}
        act = np.asarray(out["action.bfm"], dtype=np.float64)
        if "action.bfm_vel" in out:      # gr00t_bfm_vel_2cam: 18 + 7 velocity columns (bfm_action.py)
            act = np.concatenate([act, np.asarray(out["action.bfm_vel"], dtype=np.float64)], axis=1)
        return act, np.asarray(out["action.right_gripper"], dtype=np.float64).reshape(-1)
    return predict


def run_round(sim, pol, refs_cls, observer, predict, rnd, rng, a, out_dir):
    m = dict(round=rnd, ok=False, why="")
    # --- scene sample: the recorder's own (run_bottle.sample_scene; BFM_SCENE selects the geometry) ---
    bottle, stand_xy, spawn_xy, yaw0, meta = B.sample_scene(sim, rng)
    m.update(meta)
    sim.reset(spawn_xy, yaw0, 0.0)
    sim.d.qpos[sim.crate_q:sim.crate_q + 3] = bottle; mujoco.mj_forward(sim.m, sim.d)
    sim.grip_clip = B.GRIP_CLIP
    pol.reset(sim.state()); refs = refs_cls(pol, sim)
    bz0 = float(sim.crate()[0][2])
    ffm = None
    if a.video:
        ffm = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480", "-r", "25", "-i", "-",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", os.path.join(out_dir, f"round_{rnd:03d}.mp4")], stdin=subprocess.PIPE)
        ego_r = mujoco.Renderer(sim.m, height=H, width=W); ego_cam = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    trace = []; T = 0.0; k = 0
    plan_pose = [None]      # (world pos, yaw) of the last executed pelvis target, for --pelvis-anchor plan
    exec_summary = [None]   # last executed world targets (frame 0 and frame 5) for pelvis / right wrist
    pred_summary = [None]   # last prediction: pelvis-frame pelvis xy at rows 0/19/39, wrist x at 0/39, grip min -- what the policy asked for
    first_move = None; closed_once = False; max_rise = -1.0
    infer_wall = []
    # --- settle exactly like the expert (pelvis pinned, hands at the side) --------------------
    p0 = sim.d.qpos[:2].copy()
    side0 = refs.hands_at_side(np.array([p0[0], p0[1], PELVIS_Z]), yaw0)
    settle_ref = lambda t: dict(refs.pelvis(p0, yaw0), **side0)  # noqa: E731

    def record(phase):
        d_ = sim.d; bp, bq = sim.crate(); pel = d_.qpos[:3]; pad = sim.pad("right")
        tilt = float(np.degrees(np.arccos(np.clip(mat(bq)[2, 2], -1, 1))))
        trace.append(dict(t=round(T, 3), phase=phase, dist=float(np.linalg.norm(bp[:2] - pel[:2])), rise=float(bp[2] - bz0),
                          fallen=bool(sim.fallen()), grip=float(sim.grip_target[2]), pelvis_xy=[float(pel[0]), float(pel[1])],
                          pelvis_yaw=float(yaw_of(d_.qpos[3:7])), pelvis_z=float(pel[2]), torso_pitch=float(np.radians(B.torso_pitch_deg(sim))),
                          obj_xy=[float(bp[0]), float(bp[1])], obj_z=float(bp[2]), palm_z=float(pad[2]),
                          palm_obj=float(np.linalg.norm(pad - bp)), obj_tilt=tilt, touch=[], touch_links=[],
                          pred=pred_summary[0], exec=exec_summary[0],
                          wrist_w=[round(float(v), 3) for v in sim.body("right_wrist_yaw_link")[0]]))

    def step(targets, phase):
        nonlocal T, k, first_move, max_rise
        st = sim.state()
        sim.step(pol(st, targets))
        T += DT; k += 1
        bp, _ = sim.crate()
        if first_move is None and np.linalg.norm(bp[:2] - bottle[:2]) > 0.005:
            first_move = (round(T, 2), phase)
        max_rise = max(max_rise, float(bp[2] - bz0))
        if k % 2 == 0:
            record(phase)
        if ffm is not None and k % 2 == 0:
            ego_r.update_scene(sim.d, camera=ego_cam)
            ffm.stdin.write(np.concatenate([sim.frame(), ego_r.render()], axis=1).tobytes())
        return sim.fallen()

    for _ in range(int(SETTLE_S / DT)):
        if step(refs.futures(settle_ref, T), "settle"):
            m["why"] = "[settle] fell"; break
    # --- VLA control ---------------------------------------------------------------------------
    chunk = None          # (k_ready, k0, act[40,18], grip[40], root_pos0, R0)
    live = None           # the chunk being executed; a fresh prediction replaces it only once its
                          # latency window has elapsed. Falling back to hold_targets (the settle pose:
                          # pelvis at the spawn, hands at the side) inside that window sent the
                          # controller 1-2 m behind the robot for 4 of every 20 ticks -- the replay
                          # round-trip of a recorded grasp failed (bottle knocked) until this was fixed.
    k_last_obs = -10 ** 9
    replan_ticks = int(round(a.replan_s / DT))
    hold_targets = refs.futures(settle_ref, T)
    budget_ticks = int(a.budget / DT)
    while not m["why"] and k < int(SETTLE_S / DT) + budget_ticks:
        if k - k_last_obs >= replan_ticks:
            obs, (root_pos0, R0) = observer()
            if hasattr(predict, "state"):
                predict.state["k"] = k
            if getattr(predict, "world_frame", False):
                root_pos0, R0 = np.zeros(3), np.eye(3)
            t0 = time.time(); act, grip = predict(obs); infer_wall.append(time.time() - t0)
            if not getattr(predict, "world_frame", False):
                pred_summary[0] = dict(pel0=[round(float(v), 3) for v in act[0, 0:2]], pel19=[round(float(v), 3) for v in act[19, 0:2]],
                                       pel39=[round(float(v), 3) for v in act[39, 0:2]], wr0=round(float(act[0, 9]), 3),
                                       wr39=round(float(act[39, 9]), 3), grip_min=round(float(grip.min()), 3),
                                       row0=[round(float(v), 3) for v in act[0]])
            k_last_obs = k
            chunk = (k + a.latency_ticks, k, act, grip, root_pos0, R0)
        if chunk is not None and k >= chunk[0]:
            live = chunk
        if live is not None:
            k_ready, k0, act, grip, root_pos0, R0 = live
            targets = {}
            pel_list, rw_list = ([], []), ([], [])
            root_pos_pred, R_pred = root_pos0, R0          # pelvis pose at prediction time
            if a.anchor == "live" and not getattr(predict, "world_frame", False):
                # the row is a target relative to the pelvis it will be applied from: use the
                # pelvis NOW, not the one at prediction time (0-0.4 s stale during a walk)
                root_pos0, R0 = sim.d.qpos[:3].copy(), mat(sim.d.qpos[3:7])
            has_vel = act.shape[1] >= 22 and not a.no_vel and not getattr(predict, "world_frame", False)
            if a.pelvis_anchor == "plan" and has_vel:
                # Anchor on where the previous chunk's plan says the pelvis SHOULD be (bounded residual
                # over the live pose), so the controller's tracking lag (~10 cm during a walk) is not
                # re-zeroed at every replan. The training offset is plan - actual, so the policy's row
                # offset already contains the expert's version of this residual; carrying it bounded
                # (+-0.10 m, +-0.2 rad) trades a little overshoot for not stopping 10 cm short.
                live_p = sim.d.qpos[:3].copy(); live_yaw = yaw_of(sim.d.qpos[3:7])
                if plan_pose[0] is None:
                    plan_pose[0] = (live_p.copy(), live_yaw)
                res = np.clip(plan_pose[0][0] - live_p, -0.10, 0.10); res[2] = 0.0
                res_yaw = float(np.clip(wrap(plan_pose[0][1] - live_yaw), -0.2, 0.2))
                root_pos_pred = live_p + res; R_pred = rot_z(live_yaw + res_yaw)
            if a.pelvis_anchor in ("world", "plan") and has_vel:
                # Pelvis target as a WORLD point fixed at prediction time: integrate the chunk's plan
                # velocity from the prediction-time pelvis, then add the row's offset. Live anchoring
                # has zero stiffness against the backward drift ScaleBFM shows while the torso leans
                # into the reach (v4 eval: stand 0.55 -> 0.65 m after arrival, then the bottle is out of
                # reach); the expert's stand is a world point, which is what this reproduces.
                cum = np.cumsum(act[:, 18:21], axis=0) / DATASET_FPS
                cum = np.vstack([np.zeros((1, 3)), cum[:-1]])           # displacement before row r
                cum_yaw = np.concatenate([[0.0], np.cumsum(act[:-1, 21]) / DATASET_FPS])
            for j in range(pol.F):
                rf = min(39.0, (k - k0 + (0 if has_vel else j)) * ROWS_PER_TICK)
                if a.interp:
                    # the chunk is 25 Hz, the controller's future frames are 50 Hz: interpolate
                    # between rows instead of repeating each row twice (rot6d is re-orthonormalised
                    # by rot6d_to_mat, so a linear blend of the two 6-vectors is a valid rotation)
                    r_lo = int(rf); w = rf - r_lo; r_hi = min(39, r_lo + 1)
                    row = (1 - w) * act[r_lo] + w * act[r_hi]
                else:
                    row = act[int(rf)]
                pp = root_pos0 + R0 @ row[0:3]; pR = R0 @ rot6d_to_mat(row[3:9])
                if a.pelvis_anchor in ("world", "plan") and has_vel:
                    ri = int(rf)
                    pp = root_pos_pred + R_pred @ (cum[ri] + row[0:3])
                    pR = rot_z(float(cum_yaw[ri])) @ R_pred @ rot6d_to_mat(row[3:9])
                if a.level_pelvis:
                    # the expert's pelvis reference is yaw-only (Refs.pelvis); the row stores it RELATIVE to a
                    # pelvis that was pitched forward ~0.15 rad by the reach, i.e. as "pitch back a little".
                    # Applied to a level pelvis that is a backward lean, and it compounds every replan
                    # (probe_vel: torso -0.4 rad, palm 20 cm above the bottle). Keep only the yaw.
                    pR = rot_z(yaw_of(quat_wxyz(pR)))
                wpos = root_pos0 + R0 @ row[9:12]; wR = R0 @ rot6d_to_mat(row[12:18])
                if has_vel:
                    # future frame j = the plan j*DT later: the expert filled it from plan(t + j*DT)
                    # and ScaleBFM reads its speed from that spacing (bfm_action.py)
                    tau = j * DT; vel = row[18:25] * VEL_SCALE[:len(row) - 18]
                    Rv = R_pred if (a.pelvis_anchor in ("world", "plan")) else R0
                    pp = pp + Rv @ vel[0:3] * tau; pR = rot_z(float(vel[3]) * tau) @ pR
                    if len(vel) >= 7:            # offline 7-column layout carries a wrist velocity too
                        wpos = wpos + R0 @ vel[4:7] * tau
                pel_list[0].append(pp); pel_list[1].append(quat_wxyz(pR))
                rw_list[0].append(wpos); rw_list[1].append(quat_wxyz(wR))
            targets["pelvis"] = (np.array(pel_list[0]), np.array(pel_list[1]))
            targets["right_wrist_yaw_link"] = (np.array(rw_list[0]), np.array(rw_list[1]))
            plan_pose[0] = (np.array(pel_list[0][0]), yaw_of(pel_list[1][0]))   # executed frame-0 target = the plan now
            exec_summary[0] = dict(pel_w=[round(float(v), 3) for v in pel_list[0][0]], pel_w5=[round(float(v), 3) for v in pel_list[0][-1]],
                                   wr_w=[round(float(v), 3) for v in rw_list[0][0]], wr_w5=[round(float(v), 3) for v in rw_list[0][-1]])
            # left wrist: at the side of the PREDICTED pelvis, the expert's own rule
            lp, lq = [], []
            for pp, pq in zip(pel_list[0], pel_list[1]):
                side = refs.hands_at_side(pp, yaw_of(pq))["left_wrist_yaw_link"]
                lp.append(side[0]); lq.append(side[1])
            targets["left_wrist_yaw_link"] = (np.array(lp), np.array(lq))
            g = float(grip[min(39, int((k - k0) * ROWS_PER_TICK))])
            sim.grip_target[2:] = g
            if g < 0.0:
                closed_once = True
            phase = "vla"
        else:
            targets = hold_targets; phase = "hold"
        if step(targets, phase):
            m["why"] = "[vla] fell"; break
    bp, bq = sim.crate()
    rise = float(bp[2] - bz0); tilt = float(np.degrees(np.arccos(np.clip(mat(bq)[2, 2], -1, 1))))
    m.update(rise_end_cm=round(rise * 100, 1), max_rise_cm=round(max_rise * 100, 1), tilt_end_deg=round(tilt, 1),
             closed_once=closed_once, first_move=first_move, len_s=round(T, 2),
             infer_ms=dict(median=round(1e3 * float(np.median(infer_wall)), 1), p95=round(1e3 * float(np.percentile(infer_wall, 95)), 1)) if infer_wall else None,
             min_dist=round(min(x["dist"] for x in trace), 3))
    if not m["why"]:
        if rise >= B.LIFT_OK and tilt <= B.TILT_OK_DEG:
            m["ok"] = True
        elif first_move is not None and not closed_once:
            m["why"] = f"[vla] bottle knocked at t={first_move[0]} before any close (min dist {m['min_dist']} m)"
        elif not closed_once:
            m["why"] = f"[vla] reach: min dist {m['min_dist']} m but never commanded a close"
        elif tilt > B.TILT_OK_DEG:
            m["why"] = f"[vla] close: bottle tilted {tilt:.0f} deg (rise {rise * 100:.1f} cm)"
        else:
            m["why"] = f"[vla] close: commanded a grasp but no lift ({rise * 100:.1f} cm, max {max_rise * 100:.1f})"
    if ffm is not None:
        ffm.stdin.close(); ffm.wait()
    json.dump(dict(ok=m["ok"], why=m["why"], meta=m, trace=trace), open(os.path.join(out_dir, "traces", f"round_{rnd:03d}.json"), "w"))
    return m


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--replay-dataset", default=None, help="round-trip test: replay this dataset's recorded actions instead of a model")
    ap.add_argument("--replay-episode", type=int, default=0)
    ap.add_argument("--replay-mode", choices=("world", "body"), default="world",
                    help="world: rows pre-converted with the recorded root pose (tests the sim/controller only); "
                         "body: rows in their own pelvis frame + velocity columns, executed like a prediction (tests the executor)")
    ap.add_argument("--no-vel", action="store_true", help="ignore the velocity columns even when present (ablation)")
    ap.add_argument("--level-pelvis", action="store_true", help="pelvis orientation target = yaw only (the expert's own convention)")
    ap.add_argument("--pelvis-anchor", choices=("live", "world", "plan"), default="live",
                    help="world: pelvis target = prediction-time pelvis + integrated plan velocity + row offset (a fixed world point "
                         "per chunk, stiff against drift); plan: like world but anchored on the previous plan's pose (bounded "
                         "residual over the live pelvis, carries the tracking lag across replans); live: row offset from the live pelvis (default)")
    ap.add_argument("--vel-scale", default="1,1,1,1,1,1,1", help="per-column scale on the 7 velocity columns (ablation)")
    ap.add_argument("--anchor", choices=("pred", "live"), default="pred",
                    help="pelvis pose the chunk rows are expressed in: at prediction time (pred) or the live pose each tick (live)")
    ap.add_argument("--replay-round", type=int, default=None,
                    help="collection round that produced --replay-episode (train_dataset/manifest.jsonl: round of the row "
                         "with that episode_index and split); default = the episode index, which is only right when no "
                         "earlier round was rejected or held out")
    ap.add_argument("--modality-config", default=os.path.join(ROOT, "configs/gr00t_bfm_2cam.py"))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--policy-seed", type=int, default=0, help="torch seed per round (0 = torch's own stream)")
    ap.add_argument("--replan-s", type=float, default=0.4)
    ap.add_argument("--latency-ticks", type=int, default=4, help="ticks (20 ms) between observation and first executed row")
    ap.add_argument("--budget", type=float, default=20.0, help="sim seconds of VLA control after the settle")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--interp", action="store_true", help="interpolate the 25 Hz chunk rows to the 50 Hz future frames (default: repeat each row twice)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    global VEL_SCALE
    VEL_SCALE = np.array([float(v) for v in a.vel_scale.split(",")], float); assert VEL_SCALE.shape == (7,)
    os.makedirs(os.path.join(a.out, "traces"), exist_ok=True)
    pol = Policy(); sim = Sim(pol.meta, video=a.video)
    if os.environ.get("PAD_HARDEN", "1") == "1":
        for g in range(sim.m.ngeom):
            b = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, sim.m.geom_bodyid[g]) or ""
            if "dex1_finger" in b and sim.m.geom_contype[g] != 0:
                sim.m.geom_priority[g] = 1; sim.m.geom_condim[g] = 4
                sim.m.geom_solref[g] = (0.002, 1.0); sim.m.geom_solimp[g] = (0.995, 0.9999, 0.001, 0.5, 2.0)
    observer = Observer(sim)
    predict = ((load_replay_body if a.replay_mode == "body" else load_replay)(a.replay_dataset, a.replay_episode) if a.replay_dataset
               else load_vla(a.model_path, a.modality_config, a.denoising_steps))
    import torch
    rng = np.random.default_rng(a.seed)
    man = open(os.path.join(a.out, "closed_loop.manifest.jsonl"), "a")
    oks = 0; t0 = time.time()
    for r in range(a.rounds):
        if a.replay_dataset and r != (a.replay_round if a.replay_round is not None else a.replay_episode):
            B.sample_scene(sim, rng)   # scene draws must stay in step with the recorder: burn this round's draws
            continue
        if a.policy_seed:
            torch.manual_seed(a.policy_seed + r)
        m = run_round(sim, pol, Refs, observer, predict, r, rng, a, a.out)
        oks += int(m["ok"]); man.write(json.dumps(m) + "\n"); man.flush()
        print(f"round {r:3d} {'OK ' if m['ok'] else 'no '} {m['why'][:70]:70s} | min dist {m.get('min_dist')} rise {m.get('rise_end_cm')} "
              f"infer {m.get('infer_ms', {}).get('median') if m.get('infer_ms') else '-'} ms | {oks}/{r + 1} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(model=a.model_path, rounds=a.rounds, ok=oks, seed=a.seed, replan_s=a.replan_s, latency_ticks=a.latency_ticks,
                   budget=a.budget, denoising_steps=a.denoising_steps), open(os.path.join(a.out, "closed_loop.json"), "w"), indent=1)
    print(f"OK {oks}/{a.rounds}")


if __name__ == "__main__":
    main()
