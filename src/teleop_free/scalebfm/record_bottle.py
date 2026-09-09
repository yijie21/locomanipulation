"""Record the ScaleBFM v3 bottle expert straight into LeRobot v2.1 datasets (25 Hz).

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl $VENV_SIM_PY src/teleop_free/scalebfm/record_bottle.py \
        --successes 200 --holdout 10 --seed 1 --out outputs/datasets/bfm_bottle_v3

No SONIC stack, no exporter, no converter: the expert process renders the two
cameras at tick boundaries, assembles the same observation columns the SONIC
exporter wrote (so configs/gr00t_bfm_2cam.py reads a 53-d state identical in
meaning to the SONIC-era datasets), and writes the ScaleBFM action:

  action.bfm (18)        pelvis target pos3+rot6d and right-wrist target pos3+rot6d,
                         both expressed in the pelvis frame of the SAME tick's
                         observation -- the command actually sent (bias +
                         proportional term included), never the nominal geometry.
  teleop.right_hand_joints[0] the dex1 jaw target (GRIP_OPEN..GRIP_CLOSED).

Conventions copied from vla_round.py / convert_to_hf_format.py / g1_joints.py:
  observation.state (43)  MuJoCo joint order; hand groups 7 wide with the dex1 jaw
                          joints at slots 4,5 (raw columns 26/27 and 40/41).
  observation.eef_state (14) left then right wrist_yaw_link pos3 + quat4 (wxyz), pelvis frame.
  observation.projected_gravity  R(root)^T [0,0,-1].
  observation.root_orientation   root quaternion wxyz.
  observation.root_position      world pelvis position (privileged; with root_orientation it lets an
                                 episode's actions be replayed exactly through eval_bfm_bottle.py --replay).
  observation.water_bottle_pose  pos3 + rot6d, pelvis frame (privileged; not in modality).
  rot6d = first two columns of the rotation matrix, row-major (GR00T).

Only episodes that pass run_bottle's success predicate are kept; failed rounds are
logged in manifest.jsonl (with the expert's own metrics) and their videos deleted.
Holdout episodes go to <out>_val, spread evenly over the accepted episodes.
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
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_spike as R  # noqa: E402
import run_bottle as B  # noqa: E402
from run_spike import DT, mat, quat_wxyz, yaw_of, wrap, Sim, Policy  # noqa: E402
from g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

FPS, TICK_HZ = 25, 50
W, H = 640, 480
TASK = "pick up the water bottle"
SKIP_SETTLE = os.environ.get("BFM_RECORD_SETTLE", "0") != "1"   # BFM_RECORD_SETTLE=1 restores the v3 behaviour
STREAMS = ("observation.images.ego_view", "observation.images.right_wrist")
CAMS = {"observation.images.ego_view": "head_camera", "observation.images.right_wrist": "right_wrist"}


def mat_to_rot6d(m):
    return m[:, :2].T.reshape(-1)


def fsl(arr2d, k):
    flat = pa.array(np.asarray(arr2d, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, k)


def feat_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [int(a.shape[0])]}


def video_stats(frames):
    px = np.stack(frames).astype(np.float64) / 255.0        # [k,H,W,3] RGB
    agg = lambda fn: fn(px, axis=(0, 1, 2)).reshape(3, 1, 1).tolist()  # noqa: E731
    return {"min": agg(np.min), "max": agg(np.max), "mean": agg(np.mean), "std": agg(np.std),
            "count": [int(px.shape[0])]}


# --- schema (subset of the SONIC exporter's columns, same names/dtypes/shapes) ------------
STATE_NAMES = (MUJOCO_JOINT_NAMES[:22]
               + [f"left_hand_{n}" for n in ("index_0_joint", "index_1_joint", "middle_0_joint", "middle_1_joint", "thumb_0_joint", "thumb_1_joint", "thumb_2_joint")]
               + MUJOCO_JOINT_NAMES[22:29]
               + [f"right_hand_{n}" for n in ("index_0_joint", "index_1_joint", "middle_0_joint", "middle_1_joint", "thumb_0_joint", "thumb_1_joint", "thumb_2_joint")])
COLS = [  # (name, width, dtype)
    ("observation.state", 43, "float64"), ("observation.eef_state", 14, "float64"),
    ("observation.water_bottle_pose", 9, "float32"), ("observation.root_orientation", 4, "float64"),
    ("observation.projected_gravity", 3, "float64"), ("observation.root_position", 3, "float64"),
    ("action.bfm", 18, "float32"),
    ("action.bfm_vel", 4, "float32"),   # plan velocity: pelvis target linear v (3, row's pelvis frame) + yaw rate, from ref(t+DT)-ref(t)
    ("teleop.left_hand_joints", 7, "float32"), ("teleop.right_hand_joints", 7, "float32"),
]
MODALITY = {
    "state": {
        "left_leg": {"start": 0, "end": 6}, "right_leg": {"start": 6, "end": 12}, "waist": {"start": 12, "end": 15},
        "left_arm": {"start": 15, "end": 22}, "left_hand": {"start": 22, "end": 29},
        "right_arm": {"start": 29, "end": 36}, "right_hand": {"start": 36, "end": 43},
        "left_wrist_pos": {"start": 0, "end": 3, "original_key": "observation.eef_state"},
        "left_wrist_abs_quat": {"start": 3, "end": 7, "original_key": "observation.eef_state", "rotation_type": "quaternion"},
        "right_wrist_pos": {"start": 7, "end": 10, "original_key": "observation.eef_state"},
        "right_wrist_abs_quat": {"start": 10, "end": 14, "original_key": "observation.eef_state", "rotation_type": "quaternion"},
        "root_orientation": {"start": 0, "end": 4, "original_key": "observation.root_orientation", "rotation_type": "quaternion"},
        "projected_gravity": {"start": 0, "end": 3, "original_key": "observation.projected_gravity"},
    },
    "action": {
        "bfm": {"start": 0, "end": 18, "original_key": "action.bfm"},
        "bfm_vel": {"start": 0, "end": 4, "original_key": "action.bfm_vel"},
        "left_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.left_hand_joints"},
        "right_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.right_hand_joints"},
        "right_gripper": {"start": 0, "end": 1, "original_key": "teleop.right_hand_joints"},
        "left_gripper": {"start": 0, "end": 1, "original_key": "teleop.left_hand_joints"},
    },
    "video": {"ego_view": {"original_key": STREAMS[0]}, "right_wrist": {"original_key": STREAMS[1]}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}


class Sink:
    def __init__(self, out):
        self.out = out
        os.makedirs(os.path.join(out, "data/chunk-000"), exist_ok=True)
        os.makedirs(os.path.join(out, "meta"), exist_ok=True)
        for st in STREAMS:
            os.makedirs(os.path.join(out, f"videos/chunk-000/{st}"), exist_ok=True)
        self.episodes, self.ep_stats, self.n, self.gidx = [], [], 0, 0

    def video_path(self, st, ep):
        return os.path.join(self.out, f"videos/chunk-000/{st}/episode_{ep:06d}.mp4")

    def finalize(self, extra):
        total = sum(e["length"] for e in self.episodes)
        feats = {}
        for st in STREAMS:
            feats[st] = {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channel"],
                         "info": {"video.height": H, "video.width": W, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                  "video.is_depth_map": False, "video.fps": float(FPS), "video.channels": 3, "has_audio": False}}
        for name, k, dt in COLS:
            feats[name] = {"dtype": dt, "shape": [k], "names": STATE_NAMES if name == "observation.state" else None}
        for name in ("timestamp",):
            feats[name] = {"dtype": "float32", "shape": [1], "names": None}
        for name in ("frame_index", "episode_index", "index", "task_index"):
            feats[name] = {"dtype": "int64", "shape": [1], "names": None}
        info = {"codebase_version": "v2.1", "robot_type": None, "total_episodes": self.n, "total_frames": total,
                "total_tasks": 1, "total_videos": self.n * len(STREAMS), "total_chunks": 1, "chunks_size": 1000,
                "fps": FPS, "splits": {"train": f"0:{self.n}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": feats, "discarded_episode_indices": [],
                "script_config": dict(source="teleop_free/scalebfm/record_bottle.py", controller="ScaleBFM mode 2", **extra)}
        json.dump(info, open(os.path.join(self.out, "meta/info.json"), "w"), indent=4)
        json.dump(MODALITY, open(os.path.join(self.out, "meta/modality.json"), "w"), indent=4)
        with open(os.path.join(self.out, "meta/tasks.jsonl"), "w") as f:
            f.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")
        with open(os.path.join(self.out, "meta/episodes.jsonl"), "w") as f:
            for e in self.episodes:
                f.write(json.dumps(e) + "\n")
        with open(os.path.join(self.out, "meta/episodes_stats.jsonl"), "w") as f:
            for e in self.ep_stats:
                f.write(json.dumps(e) + "\n")


class Recorder:
    """Called by run_bottle.run_round's tick BEFORE the sim step: obs at t, command at t."""

    def __init__(self, sim: Sim, sink: Sink, ep: int):
        self.sim, self.sink, self.ep = sim, sink, ep
        m = sim.m
        self.r = mujoco.Renderer(m, height=H, width=W)
        self.cam = {st: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for st, c in CAMS.items()}
        assert min(self.cam.values()) >= 0, "camera missing in the scene"
        self.q29 = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MUJOCO_JOINT_NAMES])
        self.ffm = {st: subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                          sink.video_path(st, ep)], stdin=subprocess.PIPE) for st in STREAMS}
        self.rows = {name: [] for name, _, _ in COLS}
        self.sample_frames = {st: [] for st in STREAMS}
        self.k = 0

    def tick(self, sim: Sim, st, cmd, t, phase, cmd_next=None):
        k = self.k; self.k += 1
        if k % (TICK_HZ // FPS):
            return
        if phase == "settle" and SKIP_SETTLE:
            # the 1.5 s settle is 46 identical frames per episode labelled "stand still" (15% of the
            # frames): an absorbing state in closed loop. The VLA is engaged right after the settle.
            return
        d = sim.d
        root_pos = d.qpos[:3].copy(); root_q = d.qpos[3:7].copy(); Rp = mat(root_q)
        to_pelvis = lambda p_w: Rp.T @ (np.asarray(p_w) - root_pos)  # noqa: E731
        # --- images ---
        for stream, cid in self.cam.items():
            self.r.update_scene(d, camera=cid)
            frame = self.r.render()
            self.ffm[stream].stdin.write(np.ascontiguousarray(frame).tobytes())
            if k % (TICK_HZ * 4) == 0:                    # ~one frame every 4 s for the stats
                self.sample_frames[stream].append(frame.copy())
        # --- state ---
        q = d.qpos[self.q29]
        gl = d.qpos[sim.gq[:2]]; gr = d.qpos[sim.gq[2:]]
        lh = np.zeros(7); lh[4:6] = gl
        rh = np.zeros(7); rh[4:6] = gr
        state43 = np.concatenate([q[:22], lh, q[22:29], rh])
        eef = []
        for side in ("left", "right"):
            p_w, q_w = sim.body(f"{side}_wrist_yaw_link")
            eef += list(to_pelvis(p_w)) + list(quat_wxyz(Rp.T @ mat(q_w)))
        b_p, b_q = sim.crate()
        bottle9 = np.concatenate([to_pelvis(b_p), mat_to_rot6d(Rp.T @ mat(b_q))])
        gravity = Rp.T @ np.array([0.0, 0.0, -1.0])
        # --- action: the command sent this tick, in this tick's pelvis frame ---
        pp, pqw = cmd["pelvis"]; wp, wqw = cmd["right_wrist_yaw_link"]
        act = np.concatenate([to_pelvis(pp), mat_to_rot6d(Rp.T @ mat(pqw)), to_pelvis(wp), mat_to_rot6d(Rp.T @ mat(wqw))])
        # plan velocity of the pelvis target: the spacing ScaleBFM reads its speed from (bfm_action.py).
        # From the reference itself (ref(t+DT) - ref(t)), so it is as smooth as the plan, unlike a
        # finite difference of the recorded rows (bfm_pilot_v3: spikes to 2.5 m/s cost position accuracy).
        vel = np.zeros(4, np.float32)
        if cmd_next is not None:
            pp2, pqw2 = cmd_next["pelvis"]
            vel[0:3] = Rp.T @ ((np.asarray(pp2) - np.asarray(pp)) / DT)
            vel[3] = wrap(yaw_of(pqw2) - yaw_of(pqw)) / DT
        grip = float(sim.grip_target[2])
        rhj = np.zeros(7, np.float32); rhj[0] = rhj[1] = grip
        rows = self.rows
        rows["observation.state"].append(state43); rows["observation.eef_state"].append(np.array(eef))
        rows["observation.water_bottle_pose"].append(bottle9); rows["observation.root_orientation"].append(root_q)
        rows["observation.projected_gravity"].append(gravity); rows["observation.root_position"].append(root_pos); rows["action.bfm"].append(act)
        rows["action.bfm_vel"].append(vel)
        rows["teleop.left_hand_joints"].append(np.zeros(7, np.float32)); rows["teleop.right_hand_joints"].append(rhj)

    def finish(self, ok: bool):
        for p in self.ffm.values():
            p.stdin.close(); p.wait()
        self.r.close()
        n = len(self.rows["observation.state"])
        if not ok or n < FPS:
            for st in STREAMS:
                try: os.remove(self.sink.video_path(st, self.ep))
                except FileNotFoundError: pass
            return None
        sink = self.sink
        cols = {}
        for name, k, dt in COLS:
            arr = np.asarray(self.rows[name])
            cols[name] = fsl(arr, k) if dt == "float32" else pa.FixedSizeListArray.from_arrays(
                pa.array(arr.astype(np.float64).reshape(-1), type=pa.float64()), k)
        cols["timestamp"] = pa.array((np.arange(n) / FPS).astype(np.float32), type=pa.float32())
        cols["frame_index"] = pa.array(np.arange(n, dtype=np.int64)); cols["episode_index"] = pa.array(np.full(n, self.ep, dtype=np.int64))
        cols["index"] = pa.array(np.arange(sink.gidx, sink.gidx + n, dtype=np.int64)); cols["task_index"] = pa.array(np.zeros(n, dtype=np.int64))
        table = pa.table(cols)
        pq.write_table(table, os.path.join(sink.out, f"data/chunk-000/episode_{self.ep:06d}.parquet"))
        stats = {name: feat_stats(np.asarray(self.rows[name])) for name, _, _ in COLS}
        stats["timestamp"] = feat_stats(np.arange(n) / FPS); stats["frame_index"] = feat_stats(np.arange(n))
        stats["episode_index"] = feat_stats(np.full(n, self.ep)); stats["index"] = feat_stats(np.arange(sink.gidx, sink.gidx + n))
        stats["task_index"] = feat_stats(np.zeros(n))
        for st in STREAMS:
            stats[st] = video_stats(self.sample_frames[st] or [np.zeros((H, W, 3), np.uint8)])
        sink.ep_stats.append({"episode_index": self.ep, "stats": stats})
        sink.episodes.append({"episode_index": self.ep, "tasks": [TASK], "length": n})
        sink.gidx += n; sink.n += 1
        return n


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--out", required=True, help="train dataset dir; the holdout goes to <out>_val")
    ap.add_argument("--successes", type=int, default=200)
    ap.add_argument("--holdout", type=int, default=10)
    ap.add_argument("--max-rounds", type=int, default=0, help="0 = successes*2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--video", action="store_true", help="also write the expert's side-by-side review mp4s")
    a = ap.parse_args()
    max_rounds = a.max_rounds or a.successes * 2
    os.makedirs(a.out, exist_ok=True)
    review = os.path.join(a.out, "review"); os.makedirs(os.path.join(review, "trace"), exist_ok=True)
    pol = Policy(); sim = Sim(pol.meta, video=a.video)
    # Pad contact hardening exactly as run_bottle.main (mandatory for the bottle: without it
    # the bottle pivots 8-18 deg in the jaw).
    if os.environ.get("PAD_HARDEN", "1") == "1":
        for g in range(sim.m.ngeom):
            b = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, sim.m.geom_bodyid[g]) or ""
            if "dex1_finger" in b and sim.m.geom_contype[g] != 0:
                sim.m.geom_priority[g] = 1; sim.m.geom_condim[g] = 4
                sim.m.geom_solref[g] = (0.002, 1.0); sim.m.geom_solimp[g] = (0.995, 0.9999, 0.001, 0.5, 2.0)
    rng = np.random.default_rng(a.seed)
    train, val = Sink(a.out), Sink(a.out + "_val")
    # holdout: every (successes // holdout)-th accepted episode, so it spreads over the run
    stride = max(1, a.successes // max(1, a.holdout)) if a.holdout else 0
    kept = 0; rounds = 0; t0 = time.time()
    man = open(os.path.join(a.out, "manifest.jsonl"), "a")
    while kept < a.successes and rounds < max_rounds:
        to_val = bool(a.holdout) and (kept % stride == stride - 1) and val.n < a.holdout
        sink = val if to_val else train
        rec = Recorder(sim, sink, sink.n)
        m = B.run_round(sim, pol, rounds, rng, review, a.video, None, None, recorder=rec)
        n = rec.finish(bool(m.get("ok")))
        m.update(produced=n is not None, split=("val" if to_val else "train") if n else None,
                 episode_index=(sink.n - 1) if n else None, frames=n)
        man.write(json.dumps(m) + "\n"); man.flush()
        rounds += 1; kept += int(n is not None)
        print(f"round {rounds:3d} {'OK ' if m.get('ok') else 'no '} {m.get('why', '')[:50]:50s} kept {kept}/{a.successes} "
              f"({time.time() - t0:.0f}s)", flush=True)
    extra = dict(seed=a.seed, rounds=rounds, successes=kept, expert="run_bottle.py v3", stand_d=list(B.STAND_D),
                 grip_clip=B.GRIP_CLIP, bias_cm=(np.asarray(B.BIAS) * 100).round(1).tolist())
    train.finalize(extra); val.finalize(extra)
    print(f"done: train {train.n} episodes / {sum(e['length'] for e in train.episodes)} frames -> {a.out}; "
          f"val {val.n} -> {a.out}_val; {rounds} rounds, {kept} accepted", flush=True)


if __name__ == "__main__":
    main()
