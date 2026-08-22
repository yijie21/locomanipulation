"""Convert SONIC data-exporter sessions into a LeRobot v2.1 dataset whose format
matches hojjunekim/g1_sonic_orange_to_plate_locomani exactly.

Reference deltas vs the raw exporter output (fps-50, 32 features):
  * fps 30 (rows resampled by nearest source frame, video re-encoded frame-exact)
  * two extra columns:
      observation.state_psi0  f32[15] = eef_state (pos/quat x2) rotated into the
                                        initial-heading (psi0) frame + delta_heading
      action.psi0_18          f32[18] = wrist targets from vr_3pt (pos3 + rot6d)
                                        x2 in the psi0 frame
    The reference dataset's meta gives only slices for these (modality_psi0.json),
    not semantics; the derivation above is our documented best reconstruction.
  * meta/: episodes.jsonl, tasks.jsonl, episodes_stats.jsonl, modality.json,
    modality_gr00t_sonic.json, modality_psi0.json copied/generated to match.

Episodes listed in a session's `discarded_episode_indices` (the 'x' key) are
skipped unless --include-discarded.

Usage:
  .venv_data_collection/bin/python convert_to_hf_format.py \
      --sessions "$LOCOMANI_WORKSPACE"/tmp/p03/outputs/<ts> [<ts2> ...] \
      --out outputs/lerobot/g1_sonic_orange_to_plate_locomani \
      [--include-discarded]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq_mod

# Fallback only. It is what every episode gets when no --manifest is given,
# i.e. when the whole run had one target. It used to read "place the orange
# on the plate", which described neither the pipeline (which picks the object
# up and holds it) nor the object (a can), and every episode ever converted
# carried it.
TASK = "pick up the red can"


def load_manifest(path):
    """grasp_gym --manifest -> the session's episodes, in order.

    Only lines with `produced` are episodes: a round that failed before the
    exporter started recording leaves nothing in the session, and a stand
    retry that got as far as recording leaves one even though it failed.
    Order is the join -- the exporter numbers episodes in the order it wrote
    them, which is the order the gym produced them.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("produced"):
                rows.append(r)
    return rows


def check_manifest(rows, src_eps, discarded, sess_name):
    """Refuse to guess. A mislabelled instruction is invisible in the data
    and poisons exactly the thing the target randomisation exists to teach,
    so a join that does not line up is a hard error, never a warning."""
    if len(rows) != len(src_eps):
        raise SystemExit(
            f"[manifest] {sess_name}: {len(rows)} produced rounds but "
            f"{len(src_eps)} episodes on disk. The manifest and the session "
            f"do not describe the same run -- pass the manifest written by "
            f"the gym invocation that filled this session.")
    for r, ep in zip(rows, src_eps):
        if (ep in discarded) != (not r["ok"]):
            raise SystemExit(
                f"[manifest] {sess_name}: episode {ep} is "
                f"{'discarded' if ep in discarded else 'kept'} on disk but "
                f"the manifest says ok={r['ok']} (target {r['target']}). The "
                f"join is out of phase; do not convert this session.")
SRC_FPS, DST_FPS = 50, 30
W, H = 640, 480

MODALITY = {
    "state": {
        "left_leg": {"start": 0, "end": 6},
        "right_leg": {"start": 6, "end": 12},
        "waist": {"start": 12, "end": 15},
        "left_arm": {"start": 15, "end": 22},
        "left_hand": {"start": 22, "end": 29},
        "right_arm": {"start": 29, "end": 36},
        "right_hand": {"start": 36, "end": 43},
        "left_wrist_pos": {"start": 0, "end": 3, "original_key": "observation.eef_state"},
        "left_wrist_abs_quat": {"start": 3, "end": 7, "original_key": "observation.eef_state", "rotation_type": "quaternion"},
        "right_wrist_pos": {"start": 7, "end": 10, "original_key": "observation.eef_state"},
        "right_wrist_abs_quat": {"start": 10, "end": 14, "original_key": "observation.eef_state", "rotation_type": "quaternion"},
        "root_orientation": {"start": 0, "end": 4, "original_key": "observation.root_orientation", "rotation_type": "quaternion"},
        "projected_gravity": {"start": 0, "end": 3, "original_key": "observation.projected_gravity"},
        "cpp_rotation_offset": {"start": 0, "end": 4, "original_key": "observation.cpp_rotation_offset", "rotation_type": "quaternion"},
        "init_base_quat": {"start": 0, "end": 4, "original_key": "observation.init_base_quat", "rotation_type": "quaternion"},
        "can_pos": {"start": 0, "end": 3, "original_key": "observation.can_pose"},
        "can_rot6d": {"start": 3, "end": 9, "original_key": "observation.can_pose", "rotation_type": "rotation_6d"},
        "plate_pos": {"start": 0, "end": 3, "original_key": "observation.plate_pose"},
        "plate_rot6d": {"start": 3, "end": 9, "original_key": "observation.plate_pose", "rotation_type": "rotation_6d"},
    },
    "action": {
        "delta_heading": {"start": 0, "end": 1, "original_key": "teleop.delta_heading"},
        "motion_token": {"start": 0, "end": 64, "original_key": "action.motion_token"},
        "smpl_joints": {"start": 0, "end": 72, "original_key": "teleop.smpl_joints"},
        "smpl_pose": {"start": 0, "end": 63, "original_key": "teleop.smpl_pose"},
        "body_quat_w": {"start": 0, "end": 4, "original_key": "teleop.body_quat_w", "rotation_type": "quaternion"},
        "target_body_orientation": {"start": 0, "end": 6, "original_key": "teleop.target_body_orientation", "rotation_type": "rotation_6d"},
        "left_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.left_hand_joints"},
        "right_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.right_hand_joints"},
        "left_wrist_joints": {"start": 0, "end": 3, "original_key": "teleop.left_wrist_joints"},
        "right_wrist_joints": {"start": 0, "end": 3, "original_key": "teleop.right_wrist_joints"},
        "stream_mode": {"start": 0, "end": 1, "original_key": "teleop.stream_mode"},
        "planner_mode": {"start": 0, "end": 1, "original_key": "teleop.planner_mode"},
        "planner_movement": {"start": 0, "end": 3, "original_key": "teleop.planner_movement"},
        "planner_facing": {"start": 0, "end": 3, "original_key": "teleop.planner_facing"},
        "planner_speed": {"start": 0, "end": 1, "original_key": "teleop.planner_speed"},
        "planner_height": {"start": 0, "end": 1, "original_key": "teleop.planner_height"},
        "vr_3pt_position": {"start": 0, "end": 9, "original_key": "teleop.vr_3pt_position"},
        "vr_3pt_orientation": {"start": 0, "end": 18, "original_key": "teleop.vr_3pt_orientation", "rotation_type": "rotation_6d"},
    },
    "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}
MODALITY_PSI0 = {
    "state": {"all": {"start": 0, "end": 15, "original_key": "observation.state_psi0"}},
    "action": {"all": {"start": 0, "end": 18, "original_key": "action.psi0_18"}},
    "video": {"egocentric": {"original_key": "observation.images.egocentric"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}


def yaw_of_quat_wxyz(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_to_rot6d(q):
    w, x, y, z = q
    m = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return m[:, :2].T.reshape(-1)  # first two columns, row-major like GR00T


def rot6d_to_mat(d6):
    a1, a2 = d6[:3], d6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def mat_to_rot6d(m):
    return m[:, :2].T.reshape(-1)


def fsl(arr2d, k):
    """np [n,k] float array -> arrow fixed_size_list<float>[k] (matches exporter)."""
    flat = pa.array(np.asarray(arr2d, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, k)


def compute_psi0_columns(df):
    n = len(df)
    eef = np.stack(df["observation.eef_state"].values).astype(np.float64)  # [n,14]
    dh = np.stack(df["teleop.delta_heading"].values).reshape(n, -1)[:, 0]
    ibq = np.asarray(df["observation.init_base_quat"].values[0], dtype=np.float64)
    psi0 = float(yaw_of_quat_wxyz(ibq))
    R = rz(-psi0)
    qz = np.array([np.cos(-psi0 / 2), 0, 0, np.sin(-psi0 / 2)])

    state = np.zeros((n, 15), dtype=np.float32)
    for i in range(n):
        lp, lq = eef[i, 0:3], eef[i, 3:7]
        rp, rq = eef[i, 7:10], eef[i, 10:14]
        state[i, 0:3] = R @ lp
        state[i, 3:7] = quat_mul_wxyz(qz, lq)
        state[i, 7:10] = R @ rp
        state[i, 10:14] = quat_mul_wxyz(qz, rq)
        state[i, 14] = dh[i]

    vp = np.stack(df["teleop.vr_3pt_position"].values).astype(np.float64)      # [n,9]
    vo = np.stack(df["teleop.vr_3pt_orientation"].values).astype(np.float64)   # [n,18]
    act = np.zeros((n, 18), dtype=np.float32)
    for i in range(n):
        for j, (p_sl, o_sl, out) in enumerate(
            (((0, 3), (0, 6), 0), ((3, 6), (6, 12), 9))
        ):
            p = vp[i, p_sl[0]:p_sl[1]]
            m = rot6d_to_mat(vo[i, o_sl[0]:o_sl[1]])
            act[i, out:out + 3] = R @ p
            act[i, out + 3:out + 9] = mat_to_rot6d(R @ m)
    return state, act


def convert_video(src, dst, sel_idx):
    cap = cv2.VideoCapture(str(src))
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(DST_FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(dst)],
        stdin=subprocess.PIPE,
    )
    want = set(sel_idx.tolist())
    order = {}
    i = 0
    frames = {}
    max_idx = int(sel_idx.max())
    while i <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = frame
        i += 1
    cap.release()
    last = None
    for k in sel_idx:
        f = frames.get(int(k), last)
        if f is None:
            f = np.zeros((H, W, 3), np.uint8)
        proc.stdin.write(f.tobytes())
        last = f
    proc.stdin.close()
    proc.wait()
    return frames


def feat_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "count": [int(a.shape[0])],
    }


def video_stats(frames_dict):
    fs = list(frames_dict.values())
    if len(fs) > 100:
        fs = fs[:: max(1, len(fs) // 100)][:100]
    px = np.stack(fs).astype(np.float64) / 255.0          # [k,H,W,3] BGR
    px = px[..., ::-1]                                     # RGB
    def agg(fn):
        return fn(px, axis=(0, 1, 2)).reshape(3, 1, 1).tolist()
    return {"min": agg(np.min), "max": agg(np.max),
            "mean": agg(np.mean), "std": agg(np.std),
            "count": [int(px.shape[0])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-discarded", action="store_true")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="keep only these source episode indices")
    ap.add_argument("--manifest", nargs="+", default=None,
                    help="grasp_gym --manifest files, one per --sessions "
                         "entry and in the same order. Gives each episode "
                         "its own instruction, which the exporter cannot: "
                         "its --task-prompt is fixed for a whole session.")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "videos/chunk-000/observation.images.ego_view").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    src_info = None
    episodes, ep_stats = [], []
    new_ep, global_index, total_videos = 0, 0, 0

    if args.manifest and len(args.manifest) != len(args.sessions):
        raise SystemExit(f"--manifest takes one file per --sessions entry "
                         f"({len(args.manifest)} vs {len(args.sessions)})")
    task_ids = {}          # task string -> task_index, in first-seen order

    for si, sess in enumerate(args.sessions):
        sess = Path(sess)
        info = json.load(open(sess / "meta/info.json"))
        src_info = src_info or info
        discarded = set(info.get("discarded_episode_indices", []))
        pqs = sorted((sess / "data/chunk-000").glob("episode_*.parquet"))
        src_eps = [int(q.stem.split("_")[1]) for q in pqs]
        man = None
        if args.manifest:
            man = load_manifest(args.manifest[si])
            check_manifest(man, src_eps, discarded, sess.name)
            man = dict(zip(src_eps, man))     # keyed by the episode's own id
        for pq in pqs:
            src_ep = int(pq.stem.split("_")[1])
            if src_ep in discarded and not args.include_discarded:
                continue
            if args.only is not None and src_ep not in args.only:
                continue
            table = pq_mod.read_table(pq)
            n50 = table.num_rows
            n30 = int(np.floor((n50 - 1) * DST_FPS / SRC_FPS)) + 1
            sel = np.clip(np.round(np.arange(n30) * SRC_FPS / DST_FPS).astype(int), 0, n50 - 1)
            sub = table.take(pa.array(sel))

            pdf = sub.select([
                "observation.eef_state", "teleop.delta_heading",
                "observation.init_base_quat", "teleop.vr_3pt_position",
                "teleop.vr_3pt_orientation"]).to_pandas()
            state_psi0, psi0_18 = compute_psi0_columns(pdf)

            def set_col(t, name, arr, typ):
                i = t.schema.get_field_index(name)
                return t.set_column(i, pa.field(name, typ), pa.array(arr, type=typ))

            sub = set_col(sub, "timestamp", (np.arange(n30) / DST_FPS).astype(np.float32), pa.float32())
            sub = set_col(sub, "frame_index", np.arange(n30, dtype=np.int64), pa.int64())
            sub = set_col(sub, "episode_index", np.full(n30, new_ep, dtype=np.int64), pa.int64())
            sub = set_col(sub, "index", np.arange(global_index, global_index + n30, dtype=np.int64), pa.int64())
            ep_task = man[src_ep]["task"] if man else TASK
            ti = task_ids.setdefault(ep_task, len(task_ids))
            sub = set_col(sub, "task_index",
                          np.full(n30, ti, dtype=np.int64), pa.int64())
            ts_pos = sub.schema.get_field_index("timestamp")
            sub = sub.add_column(ts_pos, pa.field("observation.state_psi0", pa.list_(pa.float32(), 15)), fsl(state_psi0, 15))
            sub = sub.add_column(ts_pos + 1, pa.field("action.psi0_18", pa.list_(pa.float32(), 18)), fsl(psi0_18, 18))
            pq_mod.write_table(sub, out / f"data/chunk-000/episode_{new_ep:06d}.parquet")

            vsrc = sess / f"videos/chunk-000/observation.images.ego_view/episode_{src_ep:06d}.mp4"
            vdst = out / f"videos/chunk-000/observation.images.ego_view/episode_{new_ep:06d}.mp4"
            frames = convert_video(vsrc, vdst, sel)

            stats = {}
            for name in sub.schema.names:
                stats[name] = feat_stats(np.asarray(sub[name].to_pylist()))
            stats["observation.images.ego_view"] = video_stats(frames)
            ep_stats.append({"episode_index": new_ep, "stats": stats})
            episodes.append({"episode_index": new_ep, "tasks": [ep_task],
                             "length": n30})
            print(f"[{new_ep}] {sess.name}/ep{src_ep}: {n50} rows @50Hz -> "
                  f"{n30} @30Hz  task[{ti}] \"{ep_task}\"")
            new_ep += 1
            global_index += n30
            total_videos += 1

    total_frames = sum(e["length"] for e in episodes)
    features = dict(src_info["features"])
    feats = {}
    for k, v in features.items():
        if k == "timestamp":
            feats["observation.state_psi0"] = {"dtype": "float32", "shape": [15], "names": None}
            feats["action.psi0_18"] = {"dtype": "float32", "shape": [18], "names": None}
        feats[k] = v
    vi = feats["observation.images.ego_view"]
    if "info" in vi and vi["info"]:
        vi["info"]["video.fps"] = float(DST_FPS)

    info_out = dict(src_info)
    info_out.update({
        "fps": DST_FPS,
        "total_episodes": new_ep,
        "total_frames": total_frames,
        "total_tasks": max(1, len(task_ids)),
        "total_videos": total_videos,
        "total_chunks": 1,
        "splits": {"train": f"0:{new_ep}"},
        "features": feats,
        "discarded_episode_indices": [],
        "script_config": {
            "source": "teleop_free/tools/convert_to_hf_format.py",
            "sessions": [str(s) for s in args.sessions],
            "resample": f"{SRC_FPS}->{DST_FPS} nearest",
        },
    })
    json.dump(info_out, open(out / "meta/info.json", "w"), indent=4)
    json.dump(MODALITY, open(out / "meta/modality.json", "w"), indent=4)
    json.dump(MODALITY, open(out / "meta/modality_gr00t_sonic.json", "w"), indent=4)
    json.dump(MODALITY_PSI0, open(out / "meta/modality_psi0.json", "w"), indent=4)
    with open(out / "meta/tasks.jsonl", "w") as f:
        for t, i in sorted(task_ids.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
        if not task_ids:
            f.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    with open(out / "meta/episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    with open(out / "meta/episodes_stats.jsonl", "w") as f:
        for e in ep_stats:
            f.write(json.dumps(e) + "\n")
    print(f"done: {new_ep} episodes, {total_frames} frames, "
          f"{max(1, len(task_ids))} task(s) -> {out}")
    for t, i in sorted(task_ids.items(), key=lambda kv: kv[1]):
        n = sum(1 for e in episodes if e["tasks"] == [t])
        print(f"  task[{i}] {n:4d} episodes  \"{t}\"")


if __name__ == "__main__":
    main()
