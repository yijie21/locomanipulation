"""The data half of the re-render: what a recorded episode says the robot was.

Assembles per-frame kinematic state from what collection already persists --
no re-execution, no new recording:

  * joints  : parquet `observation.state` (43-dim WBC layout; JOINT_MAP picks
              out our dex1 model's 33 actuated joints)
  * pelvis  : pose sidecar jsonl (world pos + quat, the one thing the parquet
              never stored -- see PoseSidecar in privileged_pub.py)
  * objects : same sidecar rows (world poses, keyed by object name)

No isaacsim, no pxr, no mujoco: this module has to import in the SIM venv (for
compute_cam_traj) and in the RENDER venv (for isaac_render) alike.
"""

from __future__ import annotations

import json
import os

import numpy as np

from locomani import paths

SCENE_XML = paths.SCENE_XML

# 43-dim observation.state -> our dex1 MuJoCo joint names.
# Verified 2026-08-31 against the exporter layout (motion_smoothness.py:83,
# g1_joints.py, primitives.py:56): 43 = [legs 12 | waist 3 | left arm 7 |
# left hand 7 | right arm 7 | right hand 7]; dex1 grippers occupy the first
# two slots of each dex3 hand block, the rest are zero-padding.
# Body 29 follow the WBC order [legs 12, waist 3, arms 14]; the two dex1
# gripper prismatic joints per hand live inside the 7-slot dex3 hand blocks.
JOINT_MAP: list[tuple[str, int]] = []  # filled by build_joint_map()

WBC_BODY_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# 43-layout: [left leg 6 | right leg 6 | waist 3 | left arm 7 + left hand 7 |
#             right arm 7 + right hand 7] -- verified against the exporter
# before first run (see build_joint_map's asserts).
# dex1 prismatic joints inside the 7-slot dex3 hand blocks. They sit at offset
# 4 and 5 of each block, NOT 0 and 1 as this file asserted until 2026-09-06.
# The wrong slots (22/23 and 36/37) are identically zero across every frame of
# every episode, so both grippers rendered frozen half-open no matter what the
# robot did -- which is what "the bottle is embedded in the gripper" looks like.
# Verified against the data, not the layout comment: of the 43 columns only
# [0..21, 26, 27, 29..35, 40, 41] are ever non-zero, slot 40 tracks the
# commanded teleop.right_hand_joints[:,0] at corr 0.994, and slots 40/41 agree
# with each other to 6e-3 as two symmetric fingers must.
GRIPPER_SLOTS = {
    "left_hand_gripper_1_joint": 26,
    "left_hand_gripper_2_joint": 27,
    "right_hand_gripper_1_joint": 40,
    "right_hand_gripper_2_joint": 41,
}


def build_joint_map() -> list[tuple[str, int]]:
    """43-vector layout: legs+waist 0..14, left arm 15..21, left hand 22..28,
    right arm 29..35, right hand 36..42 (indices asserted, not assumed --
    update GRIPPER_SLOTS from the exporter recon before trusting output)."""
    m: list[tuple[str, int]] = []
    for i, name in enumerate(WBC_BODY_ORDER[:15]):  # legs + waist
        m.append((name, i))
    for k, name in enumerate(WBC_BODY_ORDER[15:22]):  # left arm 7
        m.append((name, 15 + k))
    for k, name in enumerate(WBC_BODY_ORDER[22:29]):  # right arm 7
        m.append((name, 29 + k))
    for name, idx in GRIPPER_SLOTS.items():
        m.append((name, idx))
    assert len(m) == 33, len(m)
    return m


# ---------------------------------------------------------------------------
def load_frames(session: str, episode: int, sidecar_glob: str):
    """Join parquet joints with sidecar poses via the exact per-frame key:
    parquet row i used privileged message seq = priv_seq[i] (recorded by the
    exporter in priv_seq/episode_*.json; see run_data_exporter.py:587-594).
    The sidecar jsonl rows carry that seq; seq resets per gym process (= per
    collection batch), so pass the batch's sidecar files as a glob."""
    import glob as _glob

    import pandas as pd

    pq = os.path.join(session, "data/chunk-000", f"episode_{episode:06d}.parquet")
    df = pd.read_parquet(pq, columns=["observation.state"])
    joints = np.stack(df["observation.state"].to_numpy())  # [T, 43]

    seq_file = os.path.join(session, "priv_seq", f"episode_{episode:06d}.json")
    priv = json.load(open(seq_file))
    priv_seq = priv["priv_seq"]
    assert len(priv_seq) == len(joints), (len(priv_seq), len(joints))

    by_seq: dict[int, dict] = {}
    files = sorted(_glob.glob(sidecar_glob))
    if not files:
        raise SystemExit(f"no sidecar files match {sidecar_glob}")
    for f in files:
        for line in open(f):
            row = json.loads(line)
            if row.get("seq") is not None:
                by_seq[row["seq"]] = row
    seqs = sorted(by_seq)
    log_miss = 0
    pel = np.zeros((len(joints), 7))
    objs: dict[str, np.ndarray] = {}
    first = by_seq[seqs[0]]
    for key in first.get("objects", {}):
        objs[key] = np.zeros((len(joints), 7))
    import bisect

    for i, s in enumerate(priv_seq):
        if s is None or s not in by_seq:
            # dropped sidecar row (ZMQ SUB): take nearest recorded seq
            j = bisect.bisect_left(seqs, s if s is not None else seqs[0])
            j = min(max(j, 0), len(seqs) - 1)
            row = by_seq[seqs[j]]
            log_miss += 1
        else:
            row = by_seq[s]
        pel[i] = row["pelvis"]
        for key in objs:
            objs[key][i] = row["objects"].get(key, objs[key][i])
    if log_miss:
        print(f"NOTE: {log_miss}/{len(joints)} frames used nearest-seq fallback (sidecar drops)")
    return joints, pel, objs


def head_camera_spec(scene_xml: str = ""):
    """Read the MJCF-authored ego camera verbatim from the scene XML."""
    import re

    xml = open(scene_xml or SCENE_XML).read()
    mcam = re.search(
        r'<camera name="head_camera" pos="([^"]+)" '
        r'quat="([^"]+)" fovy="([^"]+)"',
        xml,
    )
    if mcam is None:
        raise SystemExit(f"no head_camera in {scene_xml or SCENE_XML}")
    pos = [float(v) for v in mcam.group(1).split()]
    quat = [float(v) for v in mcam.group(2).split()]  # wxyz
    return pos, quat, float(mcam.group(3))


def load_state_trace(trace_path: str, session: str = "", episode: int = 0):
    """SIMPLE's replay contract: the recorded qpos/qvel, not a reconstruction.

    Returns (seq [N], qpos [N, nq], qvel [N, nv]). If `session` is given, the
    rows are SELECTED and ORDERED to match that episode's parquet rows, using
    the priv_seq the exporter recorded -- so trace row i is the state the
    dataset's row i was written from, exactly, with nothing interpolated.

    Why this replaces load_frames for rendering: load_frames joins joints from
    the parquet (deploy proprioception at 25 fps) with pelvis and object poses
    from the pose sidecar (MuJoCo truth on its own cadence). Two provenances,
    two clocks. Re-rendering that mixture with MuJoCo ITSELF does not reproduce
    the recorded video (measured 2026-09-06), so it cannot be the input to a
    pixel-aligned Isaac render either. load_frames stays for data that predates
    the trace, where an approximation is the only option.
    """
    npz = np.load(trace_path)
    seq, qpos, qvel = npz["seq"], npz["qpos"], npz["qvel"]
    if not session:
        return seq, qpos, qvel
    sq = os.path.join(session, "priv_seq", f"episode_{episode:06d}.json")
    priv = json.load(open(sq))["priv_seq"]
    index = {int(s): i for i, s in enumerate(seq)}
    rows, missing = [], 0
    for s in priv:
        i = index.get(int(s)) if s is not None else None
        if i is None:
            missing += 1
            i = rows[-1] if rows else 0
        rows.append(i)
    if missing:
        raise SystemExit(
            f"{missing}/{len(priv)} parquet rows have no state in {trace_path}. "
            f"The trace and the episode are not the same run -- rendering a "
            f"held-over state would silently misalign the whole episode."
        )
    r = np.asarray(rows)
    return seq[r], qpos[r], qvel[r]
