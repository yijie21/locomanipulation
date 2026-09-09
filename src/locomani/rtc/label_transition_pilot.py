"""Add offline event indices to a converted pilot without changing actions.

Joins source frames to pose sidecars by exact sequence. Metadata is intended
for auditing and future transition sampling; the current trainer does not
automatically consume this extra JSONL file.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from locomani.rtc.transition_events import measure


def annotate_session(session, capture, fps):
    records = [json.loads(s) for s in (capture / "manifest.jsonl").read_text().splitlines()]
    produced = [r for r in records if r["produced"]]
    result = []
    for source_episode, record in enumerate(produced):
        if not record["ok"]:
            continue
        choices = list(
            (capture / "pose_sidecar").glob(f"*_round_{record['round']:03d}_attempt_{record['attempt']}.jsonl")
        )
        if len(choices) != 1:
            raise ValueError(f"Ambiguous sidecar: {choices}")
        side = {r["seq"]: r for r in map(json.loads, choices[0].read_text().splitlines())}
        seq = json.loads((session / f"priv_seq/episode_{source_episode:06d}.json").read_text())["priv_seq"]
        data = pq.read_table(session / f"data/chunk-000/episode_{source_episode:06d}.parquet").to_pydict()
        valid = np.array([i for i, n in enumerate(seq) if n in side])
        if len(valid) / len(seq) < 0.95:
            raise ValueError("Insufficient exact frame matches")
        matched = [side[seq[i]] for i in valid]
        pelvis = np.array([r["pelvis"] for r in matched])
        root = np.array(data["observation.root_orientation"])[valid]
        error = np.minimum(
            np.linalg.norm(root - pelvis[:, 3:], axis=1), np.linalg.norm(root + pelvis[:, 3:], axis=1)
        ).mean()
        if error > 0.02:
            raise ValueError(f"Wrong sidecar: quaternion error {error}")
        obj = np.array([r["objects"][record["target"]] for r in matched])
        t = np.array(data["timestamp"])[valid] - data["timestamp"][0]
        events, _ = measure(
            t,
            pelvis,
            obj,
            np.array(data["teleop.right_hand_joints"])[valid, 0],
            wrist=np.array(data["observation.eef_state"])[valid, 7:14],
            cmd_squat=np.array(data["teleop.planner_mode"])[valid] == 4,
        )
        indices = {
            k: None if events.get(k) is None else round(events[k] * fps)
            for k in ["near45", "squat_command", "actual_drop6cm", "grip_centre_near4cm", "close", "lift6cm"]
        }
        result.append(
            dict(
                episode_index=len(result),
                source_episode=source_episode,
                source_session=str(session),
                exact_seq_fraction=len(valid) / len(seq),
                quaternion_error=float(error),
                events=events,
                event_frame_indices=indices,
                collector_transition=record["transition"],
            )
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, nargs="+", required=True)
    ap.add_argument("--capture", type=Path, nargs="+", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args()
    if len(args.session) != len(args.capture):
        ap.error("pair each session with its capture")
    fps = json.loads((args.dataset / "meta/info.json").read_text())["fps"]
    result = []
    for session, capture in zip(args.session, args.capture):
        for record in annotate_session(session, capture, fps):
            record["episode_index"] = len(result)
            result.append(record)
    expected = json.loads((args.dataset / "meta/info.json").read_text())["total_episodes"]
    if len(result) != expected:
        raise ValueError("Dataset/manifest episode mismatch")
    out = args.dataset / "meta/transition_events.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in result))
    print(f"Wrote {len(result)} exact-sequence event annotations to {out}")


if __name__ == "__main__":
    main()
