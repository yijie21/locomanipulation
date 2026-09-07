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
from verification_tail import training_prefix_length

# Fallback only. It is what every episode gets when no --manifest is given,
# i.e. when the whole run had one target. It used to read "place the orange
# on the plate", which described neither the pipeline (which picks the object
# up and holds it) nor the object (a can), and every episode ever converted
# carried it.
TASK = "pick up the red can"


def video_streams(info):
    """Which image features this session actually carries.

    The exporter adds observation.images.{left,right}_wrist when it is run
    with --record-wrist-cameras, so the set is a property of the recording,
    not of this script. Deriving it means a session with or without the wrist
    streams converts with the same command.
    """
    return [k for k, v in info.get("features", {}).items()
            if k.startswith("observation.images.")
            and v.get("dtype") in ("image", "video")]


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
# 25, not 30, since 2026-08-28. The video frames are discrete 50 Hz renders and
# the state has to stay aligned with them, so a stored sample time can only be
# PICKED from {k/50} -- never interpolated. A uniform stored stream therefore
# needs a target rate that DIVIDES 50, and 30 does not: the picks come out
# spaced 2,1,2,2,1,2 while meta/info.json claims a uniform 1/30. That mismatch
# is a periodic timing error of period 3 frames = a 10 Hz line, a vibration the
# robot never performed. Measured: it carried about half of our excess >3 Hz
# power against the human teleop session, and after the upper-body ramp landed
# it became the largest single term (st_r_arm_10hz_ratio 2.41 -> 4.33).
#
# The cost is a lower Nyquist, 12.5 Hz against 30 fps's 15. Measured on real
# 50 Hz source data (task arm joint velocity, 3 episodes), the band only 25
# folds -- 12.5 to 15 Hz -- carries 0.75% of the power, while 6.07% sits above
# 15 Hz and was already folding at 30 fps. So this trades 0.75 points of extra
# aliasing for a measured artefact.
#
# 50 fps is the option that dominates on signal fidelity (no resampling at all,
# and NVIDIA's own client is 50 on both sides) but it is a different change:
# 1.67x the frames, and an action chunk of 16 frames stops meaning 533 ms and
# starts meaning 320 ms, so the horizon has to be re-tuned.
#
# CHANGING THIS IS A TWO-PLACE EDIT: the deploy has to publish at the same rate
# or motions replay at the wrong speed. See TF_DATASET_FPS in vla_round.py.
SRC_FPS, DST_FPS = 50, 25
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
        # observation.can_pose / observation.plate_pose are NOT here on
        # purpose. They stay in the parquet -- they are what an analysis reads
        # and what the stage predicates were written against -- but a model
        # that gets the object's true pose as an input is being trained on a
        # channel that does not exist at deployment: the deploy has an ego
        # camera and proprioception, and nothing else. Declaring them as state
        # is the one mistake here that would still show a falling loss.
    },
    # The wire protocol, not taste, fixes the action. The deploy's pose
    # protocol v4 is token-only streaming: REQUIRED token_state, OPTIONAL
    # frame_index / left_hand_joints / right_hand_joints / body_quat_w. So a
    # deployed VLA emits a token plus a gripper and nothing else has a channel
    # to travel on. This mirrors the patched exporter's own declaration
    # (features_sonic_vla.py, patches/vla_action_space.md); the columns this
    # used to slice -- planner mode/movement/facing/speed/height, and 135
    # SMPL slots that never change without a Pico headset -- are still WRITTEN
    # to the parquet, so using them as auxiliary supervision later needs no
    # re-recording. They are just not the action.
    "action": {
        "motion_token": {"start": 0, "end": 64, "original_key": "action.motion_token"},
        # These names are load-bearing, and the width is too. Isaac-GR00T's
        # unitree_g1_sonic config (gr00t/configs/data/embodiment_configs.py)
        # selects action groups BY NAME -- exactly motion_token,
        # left_hand_joints, right_hand_joints -- and a name it cannot find is a
        # `Warning: Joint group 'left_hand_joints' not found`, not an error.
        # Training then runs to completion on a 64-dim action with both hands
        # silently dropped. Calling these left_gripper/right_gripper, or
        # slicing them to the single component dex1 actually moves, does that.
        # The 7 wide is what makes the total 78 = 64 + 7 + 7, the action space
        # the SONIC inference client expects. dex1 is a parallel jaw: measured
        # over this dataset, components 0 and 1 move together (-0.020..0.024)
        # and 2..6 are constant, which is the same sparsity a real Dex3 shows
        # when only part of the hand is driven. The left column never varies
        # here at all -- the gym is right-handed throughout.
        "left_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.left_hand_joints"},
        "right_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.right_hand_joints"},
        # The group the 65-dim action space actually selects
        # (configs/gr00t_sonic_2cam.py). Same column, sliced to the one
        # component that carries information: measured over 23,103 frames,
        # left_hand_joints has std exactly 0 in all 7 components and
        # right_hand_joints is rank 1 -- components 0 and 1 are the same
        # mirrored jaw value and 2..6 are identically 0. So component 0 IS the
        # gripper, and the other 13 columns are constants that would be
        # predicted perfectly while carrying nothing, diluting every MSE.
        # modality.json is a menu, not a selection: the 7-wide groups above
        # stay declared, so going back to the 78-dim action is a config edit
        # and needs no re-conversion.
        "right_gripper": {"start": 0, "end": 1, "original_key": "teleop.right_hand_joints"},
        # Same slice of the left column, for the two-handed crate task
        # (configs/gr00t_sonic_3cam_bimanual.py). On right-handed datasets it
        # is a constant 0 like left_hand_joints above -- declared, not selected.
        "left_gripper": {"start": 0, "end": 1, "original_key": "teleop.left_hand_joints"},
        "a18": {"start": 0, "end": 18, "original_key": "action.a18"},
    },
    # Filled from the session's own features -- see video_streams(). Fixing it
    # to ego_view here is what made the wrist cameras invisible to this
    # converter even after the sim published them and the exporter recorded
    # them.
    "video": {},
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


def _video_job(job):
    """One (source, destination, frame selection) video, done in a worker.

    Returns only the STATS. convert_video hands back every decoded frame --
    ~550 MB for a 600-frame episode -- so the stats must be computed here;
    shipping frames back through the pool would cost more than the encode.
    """
    vsrc, vdst, sel = job
    return video_stats(convert_video(Path(vsrc), Path(vdst), np.asarray(sel)))


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




def _active_frame_mask(table, max_pause_s: float, hz: float = SRC_FPS):
    """True where the frame stays; False inside over-long quasi-static runs.

    Quasi-static = body joints barely moving AND the gripper command not
    changing. Runs longer than max_pause_s keep only their last max_pause_s
    (the tail carries the transition INTO motion, which is the part worth
    learning); the excess middle is dropped. Cutting whole runs to zero was
    deliberately rejected -- the policy still needs to see a beat of standing
    before acting, or arrival itself becomes out-of-distribution.
    """
    import numpy as np

    st = np.stack(table["observation.state"].to_pylist())
    g = np.stack(table["teleop.right_hand_joints"].to_pylist())[:, 0]
    body = np.linalg.norm(np.diff(st[:, :43], axis=0, prepend=st[:1, :43]), axis=1)
    gmove = np.abs(np.diff(g, prepend=g[0]))
    # thresholds from the walking phase's own scale, so the mask adapts to
    # the episode rather than to a magic constant
    thr = max(0.15 * np.median(body[body > 1e-6]), 1e-4)
    quiet = (body < thr) & (gmove < 1e-4)
    keep = np.ones(len(quiet), dtype=bool)
    limit = int(max_pause_s * hz)
    i = 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < len(quiet) and quiet[j]:
            j += 1
        if j - i > limit:                 # keep the TAIL of the run
            keep[i:j - limit] = False
        i = j
    return keep


class Sink:
    """One output dataset: its directory, its counters, its metadata.

    The train/val split writes two of these from one pass over the sessions,
    so an episode is decoded, resampled and re-encoded exactly once and lands
    in exactly one dataset. Each sink numbers its own episodes from 0 and
    keeps its own task table, so both come out as self-contained LeRobot v2.1
    datasets rather than one dataset with a hole in its episode indices.
    """

    def __init__(self, out, streams):
        self.out = Path(out)
        self.streams = streams
        (self.out / "data/chunk-000").mkdir(parents=True, exist_ok=True)
        (self.out / "meta").mkdir(parents=True, exist_ok=True)
        for st in streams:
            (self.out / f"videos/chunk-000/{st}").mkdir(parents=True, exist_ok=True)
        self.episodes, self.ep_stats = [], []
        self.new_ep = self.global_index = self.total_videos = 0
        self.task_ids = {}


def plan_holdout(items, n, seed):
    """Choose `n` episodes to hold out, spread evenly across sessions.

    Stratified by session, not random over the pool, because each session is
    its own exporter process: a clock drift or a dropped-frame bug is a
    property of one session, and letting it concentrate in the holdout would
    silently make the validation set the odd one out. Taking the same share
    from every chunk averages that away.

    Deterministic in `seed` so a re-conversion reproduces the split -- the
    checkpoint's normalization statistics are computed from the train half,
    so a split that moved between conversions would invalidate every metric
    already measured against it.
    """
    if n <= 0:
        return set()
    if n > len(items):
        raise SystemExit(f"--holdout {n} exceeds {len(items)} kept episodes")
    by_sess = {}
    for it in items:
        by_sess.setdefault(it["si"], []).append(it)
    rng = np.random.default_rng(seed)
    order = sorted(by_sess)
    # Largest-remainder apportionment: every session gives up the same
    # fraction, and the leftover goes to the sessions with the largest
    # fractional part rather than always to the first ones.
    exact = {si: n * len(by_sess[si]) / len(items) for si in order}
    take = {si: int(exact[si]) for si in order}
    for si in sorted(order, key=lambda s: (-(exact[s] - take[s]), s))[:n - sum(take.values())]:
        take[si] += 1
    chosen = set()
    for si in order:
        pool = by_sess[si]
        k = min(take[si], len(pool))
        for i in rng.choice(len(pool), size=k, replace=False):
            chosen.add((si, pool[int(i)]["src_ep"]))
    return chosen


def finalize(sink, src_info, args, feats_extra=True):
    """Write one sink's meta/ and print its summary."""
    total_frames = sum(e["length"] for e in sink.episodes)
    feats = {}
    for k, v in dict(src_info["features"]).items():
        if (getattr(args, "strip_pose_columns", False)
                and k.startswith("observation.") and k.endswith("_pose")):
            continue
        if k == "timestamp":
            feats["observation.state_psi0"] = {"dtype": "float32", "shape": [15], "names": None}
            feats["action.psi0_18"] = {"dtype": "float32", "shape": [18], "names": None}
            if getattr(args, "a18", None):
                feats["action.a18"] = {"dtype": "float32", "shape": [18], "names": None}
        feats[k] = v
    for st in sink.streams:
        vi = feats.get(st, {})
        if vi.get("info"):
            vi["info"]["video.fps"] = float(DST_FPS)
    modality = json.loads(json.dumps(MODALITY))
    modality["video"] = {st.split(".")[-1]: {"original_key": st} for st in sink.streams}

    info_out = dict(src_info)
    info_out.update({
        "fps": DST_FPS,
        "total_episodes": sink.new_ep,
        "total_frames": total_frames,
        "total_tasks": max(1, len(sink.task_ids)),
        "total_videos": sink.total_videos * len(sink.streams),
        "total_chunks": 1,
        "splits": {"train": f"0:{sink.new_ep}"},
        "features": feats,
        "discarded_episode_indices": [],
        "script_config": {
            "source": "teleop_free/tools/convert_to_hf_format.py",
            "sessions": [str(s) for s in args.sessions],
            "resample": f"{SRC_FPS}->{DST_FPS} nearest",
            "holdout": args.holdout,
            "holdout_seed": args.holdout_seed,
        },
    })
    json.dump(info_out, open(sink.out / "meta/info.json", "w"), indent=4)
    json.dump(modality, open(sink.out / "meta/modality.json", "w"), indent=4)
    json.dump(modality, open(sink.out / "meta/modality_gr00t_sonic.json", "w"), indent=4)
    json.dump(MODALITY_PSI0, open(sink.out / "meta/modality_psi0.json", "w"), indent=4)
    with open(sink.out / "meta/tasks.jsonl", "w") as f:
        for t, i in sorted(sink.task_ids.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
        if not sink.task_ids:
            f.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    with open(sink.out / "meta/episodes.jsonl", "w") as f:
        for e in sink.episodes:
            f.write(json.dumps(e) + "\n")
    with open(sink.out / "meta/episodes_stats.jsonl", "w") as f:
        for e in sink.ep_stats:
            f.write(json.dumps(e) + "\n")
    print(f"done: {sink.new_ep} episodes, {total_frames} frames, "
          f"{max(1, len(sink.task_ids))} task(s) -> {sink.out}")
    for t, i in sorted(sink.task_ids.items(), key=lambda kv: kv[1]):
        n = sum(1 for e in sink.episodes if e["tasks"] == [t])
        print(f"  task[{i}] {n:4d} episodes  \"{t}\"")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-discarded", action="store_true")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="keep only these source episode indices")
    ap.add_argument('--trim-verification-tail', action='store_true',
                    help='Use manifest training_tail sequence boundaries; keep raw episodes intact')
    ap.add_argument("--manifest", nargs="+", default=None,
                    help="grasp_gym --manifest files, one per --sessions "
                         "entry and in the same order. Gives each episode "
                         "its own instruction, which the exporter cannot: "
                         "its --task-prompt is fixed for a whole session.")
    ap.add_argument("--a18", default=None, metavar="NPZ",
                    help="seq -> 18-D commanded action table from "
                         "tools/export_action18.py --seq-out. Adds an "
                         "action.a18 column (wrist SE(3) + grip per arm + "
                         "4-D base), joined on the privileged `seq` each "
                         "source frame carries. Additive: the 65-D "
                         "motion_token action is unchanged.")
    ap.add_argument("--holdout", type=int, default=0,
                    help="hold this many episodes out of --out and write them "
                         "to --holdout-out instead, taking the same share from "
                         "every session. GR00T has no validation split of its "
                         "own -- DatasetFactory.build asserts eval_strategy "
                         "== 'no' -- so the split has to happen here.")
    ap.add_argument("--holdout-out", default=None,
                    help="where the held-out episodes go (required with "
                         "--holdout)")
    ap.add_argument("--holdout-seed", type=int, default=0,
                    help="seed for the holdout draw; recorded in info.json")
    ap.add_argument("--drop-idle", type=float, default=None, metavar="SECONDS",
                    help="drop runs of quasi-static frames longer than this "
                         "many seconds, keeping that much of each run's tail. "
                         "THE PAUSE PROBLEM: a pause makes the demo "
                         "non-Markovian -- 'standing at second 1' and "
                         "'standing at second 4' are the same observation "
                         "with different correct actions -- and action "
                         "chunking only absorbs pauses shorter than one "
                         "chunk (1.33 s here). Measured on the bottle data, "
                         "arrival->reach stood at median 4.6 s (the human-"
                         "teleop reference the official recipe trains from: "
                         "0.2 s), and the trained policy died in exactly the "
                         "standing-still attractor this predicts. Source-"
                         "side trims got it to ~4.2 s; this filter removes "
                         "what machinery remains. 1.0 is a sensible value: "
                         "pauses stay sub-chunk, and the kept tail still "
                         "teaches a beat of settling.")
    ap.add_argument("--jobs", "-j", type=int, default=1, metavar="N",
                    help="parallel video encoders (default 1). Only the "
                         "decode+encode is parallelised; every table decision "
                         "stays serial and in order, so any N produces "
                         "byte-identical parquet and meta.")
    ap.add_argument("--strip-pose-columns", action="store_true",
                    help="drop every observation.*_pose column while "
                         "converting. The pose channel is privileged analysis "
                         "data, never a model input, and it is the one column "
                         "set that legitimately differs between recording "
                         "eras (the TF_POSE_OBJECTS restart bug); stripping "
                         "it lets a mixed-era collection convert as one "
                         "dataset without weakening the schema check for "
                         "every other column.")
    args = ap.parse_args()

    if args.holdout and not args.holdout_out:
        raise SystemExit("--holdout needs --holdout-out")
    if args.trim_verification_tail and (not args.manifest or args.drop_idle or args.include_discarded):
        ap.error('--trim-verification-tail requires manifests, successful episodes, and no --drop-idle')
    if args.manifest and len(args.manifest) != len(args.sessions):
        raise SystemExit(f"--manifest takes one file per --sessions entry "
                         f"({len(args.manifest)} vs {len(args.sessions)})")

    streams = video_streams(json.load(open(Path(args.sessions[0]) / "meta/info.json")))
    if not streams:
        raise SystemExit("no observation.images.* features in the first session")
    print(f"video streams: {', '.join(s.split('.')[-1] for s in streams)}")

    # Pass 1 -- enumerate what will be kept, and validate every manifest join,
    # before a single frame is decoded. The split has to be decided over the
    # whole pool, and finding a broken join after two hours of transcoding is
    # a bad way to find out.
    items, src_info, mans = [], None, {}
    for si, sess in enumerate(args.sessions):
        sess = Path(sess)
        info = json.load(open(sess / "meta/info.json"))
        src_info = src_info or info
        discarded = set(info.get("discarded_episode_indices", []))
        pqs = sorted((sess / "data/chunk-000").glob("episode_*.parquet"))
        src_eps = [int(q.stem.split("_")[1]) for q in pqs]
        if args.manifest:
            man = load_manifest(args.manifest[si])
            check_manifest(man, src_eps, discarded, sess.name)
            mans[si] = dict(zip(src_eps, man))
        for pq in pqs:
            src_ep = int(pq.stem.split("_")[1])
            if src_ep in discarded and not args.include_discarded:
                continue
            if args.only is not None and src_ep not in args.only:
                continue
            items.append({"si": si, "sess": sess, "src_ep": src_ep, "pq": pq})

    # Every session must describe the same columns. info.json is copied from
    # the FIRST session and becomes the dataset's declared schema, so a session
    # that carries different columns produces a dataset whose metadata is a
    # confident description of 10% of its own parquet files -- and nothing
    # downstream notices until gr00t.data.stats concatenates them and dies on a
    # width mismatch, an hour of transcoding later.
    #
    # This is not hypothetical. The first bottle collection restarted the
    # exporter between chunks (it leaks threads), the restart did not carry
    # TF_POSE_OBJECTS, and sessions 2..10 silently recorded observation.can_pose
    # = 0.0 instead of observation.water_bottle_pose. Refuse rather than guess:
    # a mixed dataset is worth catching before the videos are re-encoded.
    def _is_pose(c):
        return c.startswith("observation.") and c.endswith("_pose")

    schemas = {}
    for si, sess in enumerate(args.sessions):
        cols = tuple(sorted(
            c for c in json.load(open(Path(sess) / "meta/info.json"))
            .get("features", {})
            if not (args.strip_pose_columns and _is_pose(c))))
        schemas.setdefault(cols, []).append(Path(sess).name)
    if len(schemas) > 1:
        print("\nsessions do NOT share a schema:")
        ref = max(schemas.items(), key=lambda kv: len(kv[1]))[0]
        for cols, names in schemas.items():
            extra, missing = set(cols) - set(ref), set(ref) - set(cols)
            print(f"  {len(names):3d} session(s) {names[0]}..."
                  f"  +{sorted(extra) or '-'}  -{sorted(missing) or '-'}")
        raise SystemExit(
            "refusing to build a dataset whose info.json would describe only "
            "some of its parquet files. Re-record the odd sessions, or pass "
            "only the sessions that agree.")

    held = plan_holdout(items, args.holdout, args.holdout_seed)
    train = Sink(args.out, streams)
    val = Sink(args.holdout_out, streams) if args.holdout else None
    if val is not None:
        per_sess = {}
        for si, _ in held:
            per_sess[si] = per_sess.get(si, 0) + 1
        print(f"holdout: {len(held)} of {len(items)} episodes "
              f"(seed {args.holdout_seed}), per session "
              f"{[per_sess.get(i, 0) for i in range(len(args.sessions))]}")

    # Video decode+x264 encode is ~95% of the wall clock (2.5 GB of video vs
    # 231 MB of parquet on cover400) and the serial pipe keeps one core busy on
    # a 32-core box. Only the videos are parallelised: every table decision --
    # episode order, `index`, task ids, the idle filter -- stays in this loop,
    # in this order, so the parquet output is bit-identical to --jobs 1.
    jobs = max(1, int(args.jobs))
    pending = []          # (stats_dict, stream_name, src, dst, sel)

    a18_lut = None
    if args.a18:
        z = np.load(args.a18)
        a18_lut = (z["seq"].astype(np.int64), z["a18"].astype(np.float32))
        print(f"a18: {len(a18_lut[0])} seq rows from {args.a18}")

    # Pass 2 -- convert, routing each episode to its sink.
    for it in items:
        si, sess, src_ep, pq = it["si"], it["sess"], it["src_ep"], it["pq"]
        sink = val if (si, src_ep) in held else train
        man = mans.get(si)

        table = pq_mod.read_table(pq)
        if a18_lut is not None:
            # Join on the privileged `seq` the sim stamps on every frame:
            # priv_seq/episode_XXXXXX.json lists one seq per SOURCE row, and
            # the sidecar rows carry the same counter. Attaching here (before
            # the idle filter and the 50->25 take) means every later row
            # selection carries the action with it, so no separate resample
            # can drift.
            ps = json.load(open(sess / f"priv_seq/episode_{src_ep:06d}.json"))
            seqs = np.asarray(ps["priv_seq"], dtype=np.int64)
            if len(seqs) != table.num_rows:
                raise SystemExit(f"a18: {sess.name}/ep{src_ep} has "
                                 f"{len(seqs)} priv_seq for {table.num_rows} rows")
            idx = np.searchsorted(a18_lut[0], seqs)
            idx = np.clip(idx, 0, len(a18_lut[0]) - 1)
            miss = int((a18_lut[0][idx] != seqs).sum())
            if miss > len(seqs) * 0.02:
                raise SystemExit(f"a18: {sess.name}/ep{src_ep} misses "
                                 f"{miss}/{len(seqs)} seq -- wrong --a18 file?")
            col = a18_lut[1][idx]
            table = table.append_column(
                pa.field("action.a18", pa.list_(pa.float32(), 18)), fsl(col, 18))
        if args.trim_verification_tail:
            ps = json.loads((sess / f'priv_seq/episode_{src_ep:06d}.json').read_text())
            original_rows = table.num_rows
            end = training_prefix_length(ps['priv_seq'], original_rows, man[src_ep])
            table = table.slice(0, end)
            with (sink.out/'meta/verification_trims.jsonl').open('a') as receipt:
                receipt.write(json.dumps(dict(source_session=str(sess), source_episode=src_ep,
                    episode=sink.new_ep, source_rows=original_rows, kept_source_rows=end,
                    boundary=man[src_ep]['execution']['metrics']['training_tail']))+'\n')
            print(f'    verification tail: kept {end}/{original_rows} source frames')
        if args.strip_pose_columns:
            table = table.drop(
                [c for c in table.column_names if _is_pose(c)])
        if args.drop_idle:
            keep = _active_frame_mask(table, args.drop_idle)
            if keep is not None and not keep.all():
                dropped = int((~keep).sum())
                table = table.take(pa.array(np.flatnonzero(keep)))
                print(f"    idle filter: dropped {dropped} of "
                      f"{len(keep)} frames "
                      f"({dropped / len(keep) * 100:.0f}%)")
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

        new_ep = sink.new_ep
        sub = set_col(sub, "timestamp", (np.arange(n30) / DST_FPS).astype(np.float32), pa.float32())
        sub = set_col(sub, "frame_index", np.arange(n30, dtype=np.int64), pa.int64())
        sub = set_col(sub, "episode_index", np.full(n30, new_ep, dtype=np.int64), pa.int64())
        sub = set_col(sub, "index",
                      np.arange(sink.global_index, sink.global_index + n30, dtype=np.int64), pa.int64())
        ep_task = man[src_ep]["task"] if man else TASK
        ti = sink.task_ids.setdefault(ep_task, len(sink.task_ids))
        sub = set_col(sub, "task_index", np.full(n30, ti, dtype=np.int64), pa.int64())
        ts_pos = sub.schema.get_field_index("timestamp")
        sub = sub.add_column(ts_pos, pa.field("observation.state_psi0", pa.list_(pa.float32(), 15)), fsl(state_psi0, 15))
        sub = sub.add_column(ts_pos + 1, pa.field("action.psi0_18", pa.list_(pa.float32(), 18)), fsl(psi0_18, 18))
        pq_mod.write_table(sub, sink.out / f"data/chunk-000/episode_{new_ep:06d}.parquet")

        stats = {}
        for name in sub.schema.names:
            stats[name] = feat_stats(np.asarray(sub[name].to_pylist()))
        for st in streams:
            vsrc = sess / f"videos/chunk-000/{st}/episode_{src_ep:06d}.mp4"
            vdst = sink.out / f"videos/chunk-000/{st}/episode_{new_ep:06d}.mp4"
            if jobs > 1:
                # `stats` is appended to ep_stats below and filled in after the
                # pool runs -- same dict object, so order of ep_stats is
                # unaffected by which video finishes first.
                pending.append((stats, st, str(vsrc), str(vdst), sel))
            else:
                stats[st] = video_stats(convert_video(vsrc, vdst, sel))
        sink.ep_stats.append({"episode_index": new_ep, "stats": stats})
        sink.episodes.append({"episode_index": new_ep, "tasks": [ep_task], "length": n30})
        tag = "val " if sink is val else "train"
        print(f"[{tag} {new_ep:3d}] {sess.name}/ep{src_ep}: {n50} rows "
              f"@{SRC_FPS}Hz -> {n30} @{DST_FPS}Hz  task[{ti}] \"{ep_task}\"")
        sink.new_ep += 1
        sink.global_index += n30
        sink.total_videos += 1

    if pending:
        import multiprocessing as mp
        print(f"videos: {len(pending)} clips on {jobs} workers")
        with mp.get_context("fork").Pool(jobs) as pool:
            out = pool.map(_video_job, [(p[2], p[3], p[4]) for p in pending],
                           chunksize=1)
        for (st_dict, st, _, _, _), vs in zip(pending, out):
            st_dict[st] = vs

    finalize(train, src_info, args)
    if val is not None:
        finalize(val, src_info, args)


if __name__ == "__main__":
    main()
