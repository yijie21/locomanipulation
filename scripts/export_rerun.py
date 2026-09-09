#!/usr/bin/env python3
"""Export episodes of a LeRobot v2.1 dataset to a standalone Rerun .rrd file.

Rerun is the backend LeRobot's own visualize_dataset.py uses, but going through
a file instead of a viewer server means this needs no free port -- every one on
this box is taken -- and nothing is uploaded anywhere. Open the result with:

    pip install rerun-sdk && rerun <file>.rrd

The mp4s are logged as video assets (rr.AssetVideo), so the file carries the
encoded bytes rather than decoded frames and stays small.

Usage:
    python scripts/export_rerun.py <dataset_dir> <out.rrd> [--episodes 0 1 2]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb

# (parquet column, component index, entity path) -- the signals that say
# whether an episode is a real grasp, in the order a reviewer reads them.
SCALARS = [
    ("teleop.right_hand_joints", 0, "action/gripper_right"),
    ("teleop.left_hand_joints", 0, "action/gripper_left"),
    ("action.motion_token", 0, "action/motion_token_0"),
    ("action.motion_token", 1, "action/motion_token_1"),
    ("action.motion_token", 2, "action/motion_token_2"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    root = Path(args.dataset)
    info = json.loads((root / "meta/info.json").read_text())
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in (root / "meta/tasks.jsonl").open()}
    vid_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    chunk, fps = info["chunks_size"], info["fps"]

    rr.init(f"lerobot::{root.name}", spawn=False)
    rr.save(args.out)

    for ep in args.episodes:
        base = f"episode_{ep:03d}"
        t = pq.read_table(root / info["data_path"].format(episode_chunk=ep // chunk, episode_index=ep))
        n = t.num_rows
        times = np.arange(n) / fps

        # Cameras: log the mp4 once, then one frame reference per timestamp.
        for vk in vid_keys:
            vp = root / info["video_path"].format(episode_chunk=ep // chunk, video_key=vk, episode_index=ep)
            path = f"{base}/{vk.split('.')[-1]}"
            rr.log(path, rr.AssetVideo(path=vp), static=True)
            stamps = rr.AssetVideo(path=vp).read_frame_timestamps_nanos()
            rr.send_columns(
                path,
                indexes=[rr.TimeColumn("time", duration=1e-9 * np.asarray(stamps))],
                columns=rr.VideoFrameReference.columns_nanos(stamps),
            )

        col = {
            c: np.stack(t[c].to_pylist())
            for c in [
                "observation.can_pose",
                "observation.eef_state",
                "teleop.right_hand_joints",
                "teleop.left_hand_joints",
                "action.motion_token",
            ]
        }

        for name, idx, path in SCALARS:
            rr.send_columns(
                f"{base}/{path}",
                indexes=[rr.TimeColumn("time", duration=times)],
                columns=rr.Scalars.columns(scalars=col[name][:, idx]),
            )

        # The grasp evidence: can-to-wrist distance collapsing and then holding
        # a constant offset is what separates a grasp from a knock-over.
        can = col["observation.can_pose"][:, :3]
        wrist_r = col["observation.eef_state"][:, 7:10]
        dist = np.linalg.norm(can - wrist_r, axis=1)
        rr.send_columns(
            f"{base}/derived/can_to_right_wrist_m",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=rr.Scalars.columns(scalars=dist),
        )
        rr.send_columns(
            f"{base}/derived/can_height_robot_frame_m",
            indexes=[rr.TimeColumn("time", duration=times)],
            columns=rr.Scalars.columns(scalars=can[:, 2]),
        )

        # 3D, robot frame: can_pose and both wrists are all pelvis-relative.
        for i in range(0, n, 2):
            rr.set_time("time", duration=times[i])
            rr.log(f"{base}/scene/can", rr.Points3D(can[i : i + 1], colors=[[220, 60, 50]], radii=0.033))
            rr.log(f"{base}/scene/wrist_right", rr.Points3D(wrist_r[i : i + 1], colors=[[60, 140, 220]], radii=0.02))
            rr.log(
                f"{base}/scene/wrist_left",
                rr.Points3D(col["observation.eef_state"][i : i + 1, 0:3], colors=[[120, 120, 120]], radii=0.02),
            )

        rr.log(
            f"{base}/task",
            rr.TextDocument(f"{tasks[int(t['task_index'][0].as_py())]}\n{n} frames / {n / fps:.1f} s @ {fps} fps"),
            static=True,
        )
        print(f"  episode {ep}: {n} frames, {len(vid_keys)} cameras")

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
