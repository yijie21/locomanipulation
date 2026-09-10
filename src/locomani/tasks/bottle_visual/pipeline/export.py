"""Export a frozen list of raw attempts into a LeRobot v2.1 VLA dataset.

No simulation or episode selection is performed here. Inputs must be an explicit
JSON list; improved-expert attempts must pass audit.json. No object truth columns
are exported. Existing exports are immutable and identified by input hashes.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from locomani.backends.scalebfm.state import STATE_DIMS
from locomani.tasks.bottle_visual.visibility import initial_visibility

CAMERAS = ("ego_view", "right_wrist")
TASK = "pick up the water bottle"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def stats(array):
    return {name: getattr(array, name)(axis=0).tolist() for name in ("min", "max", "mean", "std")} | {
        "count": [len(array)]
    }


def export(inputs, out, baseline_control=False):
    inputs, out = [Path(p).resolve() for p in inputs], Path(out)
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("Input manifest must contain distinct, nonempty attempts")
    manifest = {
        "baseline_control": baseline_control,
        "inputs": [
            dict(path=str(p), student_sha256=hashlib.sha256((p / "student.npz").read_bytes()).hexdigest())
            for p in inputs
        ],
    }
    if (out / "meta/info.json").exists():
        if json.loads((out / "input_manifest.json").read_text()) != manifest:
            raise ValueError("Dataset already exists with different inputs")
        return json.loads((out / "meta/info.json").read_text())
    if (out / "input_manifest.json").exists() and json.loads((out / "input_manifest.json").read_text()) != manifest:
        raise ValueError("Incomplete export has a different frozen manifest")
    (out / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    atomic_json(out / "input_manifest.json", manifest)
    episodes, episode_stats = [], []
    global_index = 0
    for index, source in enumerate(inputs):
        result = json.loads((source / "result.json").read_text())
        if not result["ok"]:
            raise ValueError(f"Failed attempt cannot enter accepted dataset: {source}")
        if not baseline_control and not json.loads((source / "audit.json").read_text())["accepted"]:
            raise ValueError(f"Unqualified improved-expert attempt: {source}")
        data = np.load(source / "student.npz")
        state, action, times = data["state"], data["action"], data["simulation_time"]
        n = len(state)
        observations = json.loads((source / "observations.json").read_text())
        initial = initial_visibility(
            observations, result.get("first_record_evidence"), float(times[0]) if len(times) else float("nan")
        )
        if not all(initial.values()):
            raise ValueError(f"Initial head-camera visibility unverified: {source}: {initial}")
        if n < 40 or state.shape != (n, 91) or action.shape != (n, 29):
            raise ValueError(f"Unexpected observation/action shape or stub episode: {source}")
        if (
            not np.isfinite(state).all()
            or not np.isfinite(action).all()
            or not np.allclose(np.diff(times), 0.04, atol=1e-8)
        ):
            raise ValueError(f"Nonfinite or mistimed recording: {source}")
        columns = {}
        for key, array in [("observation.proprio", state), ("action.visual_command", action)]:
            columns[key] = pa.FixedSizeListArray.from_arrays(pa.array(array.astype(np.float32).ravel()), array.shape[1])
        columns["timestamp"] = pa.array(np.arange(n, dtype=np.float32) / 25)
        for key, value in [
            ("frame_index", np.arange(n)),
            ("episode_index", np.full(n, index)),
            ("index", np.arange(global_index, global_index + n)),
            ("task_index", np.zeros(n)),
        ]:
            columns[key] = pa.array(value.astype(np.int64))
        path = out / f"data/chunk-000/episode_{index:06d}.parquet"
        temporary = path.with_suffix(".partial.parquet")
        pq.write_table(pa.table(columns), temporary)
        temporary.replace(path)
        per_stats = {
            "observation.proprio": stats(state.astype(float)),
            "action.visual_command": stats(action.astype(float)),
        }
        for camera in CAMERAS:
            video = source / f"{camera}.mp4"
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-threads",
                        "1",
                        "-select_streams",
                        "v:0",
                        "-count_frames",
                        "-show_entries",
                        "stream=width,height,nb_read_frames",
                        "-of",
                        "json",
                        str(video),
                    ]
                )
            )["streams"][0]
            if [probe["width"], probe["height"], int(probe["nb_read_frames"])] != [320, 240, n]:
                raise ValueError(f"Video/frame mismatch: {video}")
            dest = out / f"videos/chunk-000/observation.images.{camera}/episode_{index:06d}.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if not os.path.samefile(video, dest):
                    raise ValueError(f"Video destination differs: {dest}")
            else:
                os.link(video, dest)
            # Images are already decoded by the training processor; RGB channel
            # normalization is fixed there. Dataset video stats are informational.
        episodes.append(dict(episode_index=index, tasks=[TASK], length=n))
        episode_stats.append(dict(episode_index=index, stats=per_stats))
        global_index += n
    state_modality = {}
    cursor = 0
    for key, width in STATE_DIMS.items():
        state_modality[key] = dict(start=cursor, end=cursor + width, original_key="observation.proprio")
        if key == "right_wrist_abs_quat":
            state_modality[key]["rotation_type"] = "quaternion"
        cursor += width
    modality = dict(
        state=state_modality,
        action={"visual_command": dict(start=0, end=29, original_key="action.visual_command")},
        video={key: dict(original_key=f"observation.images.{key}") for key in CAMERAS},
        annotation={"human.task_description": dict(original_key="task_index")},
    )
    atomic_json(out / "meta/modality.json", modality)
    features = {
        key: dict(dtype="float32", shape=[width], names=None)
        for key, width in [("observation.proprio", 91), ("action.visual_command", 29)]
    }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = dict(dtype="float32" if key == "timestamp" else "int64", shape=[1], names=None)
    for key in CAMERAS:
        features[f"observation.images.{key}"] = dict(
            dtype="video",
            shape=[240, 320, 3],
            names=["height", "width", "channel"],
            info={
                "video.height": 240,
                "video.width": 320,
                "video.fps": 25.0,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        )
    for filename, rows in [
        ("tasks.jsonl", [dict(task_index=0, task=TASK)]),
        ("episodes.jsonl", episodes),
        ("episodes_stats.jsonl", episode_stats),
    ]:
        path = out / "meta" / filename
        temporary = path.with_suffix(".partial")
        temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        temporary.replace(path)
    info = dict(
        codebase_version="v2.1",
        robot_type=None,
        total_episodes=len(inputs),
        total_frames=global_index,
        total_tasks=1,
        total_videos=2 * len(inputs),
        total_chunks=1,
        chunks_size=1000,
        fps=25,
        splits={"train": f"0:{len(inputs)}"},
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        features=features,
        discarded_episode_indices=[],
        script_config=dict(source=__file__, privileged_policy_inputs=False),
    )
    atomic_json(out / "meta/info.json", info)  # Commit marker written last.
    return info


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-control", action="store_true")
    args = parser.parse_args()
    info = export(json.loads(Path(args.inputs).read_text()), args.out, args.baseline_control)
    print(json.dumps(dict(episodes=info["total_episodes"], frames=info["total_frames"])))


if __name__ == "__main__":
    main()
