"""Summarize fixed-reference/pulse diagnostics; gain is local and empirical."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm_ik import ArmIK
from g1_joints import MUJOCO_JOINT_NAMES
from primitives import Primitives


def rotation_error(a, b):
    return float(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1.0, 1.0)))


def analyze(root):
    run = json.loads((root / "run.json").read_text())
    samples = [json.loads(l) for l in (root / "trace.jsonl").read_text().splitlines()]
    ik = ArmIK()
    segments = []
    for row in samples:
        if not segments or row["phase"] != segments[-1][0]["phase"]:
            segments.append([])
        segments[-1].append(row)
    rows = []
    baseline = None
    for segment in segments:
        end = segment[-1]["wall"]
        tail = [r for r in segment if r["wall"] >= end - 1.0]
        goal = np.array(segment[-1]["reference"])
        settled = [r for r in segment if np.max(np.abs(np.asarray(r["reference"])[12:] - goal[12:])) < 1e-5]
        first = [r for r in settled if r["wall"] <= settled[0]["wall"] + 0.5]
        q = np.array([r["state"]["joints"] for r in tail])
        median = np.median(q, axis=0)
        base = np.array([r["state"]["pelvis"]["pos"] for r in tail])
        ref_pos, ref_rot = ik.fk(goal, "right")
        pose = []
        fk = []
        for r in tail:
            st = r["state"]
            rot = Primitives._rot(st["pelvis"]["quat"])
            actual_pos = rot.T @ (np.asarray(st["palms"]["right"]["pos"]) - st["pelvis"]["pos"])
            actual_rot = rot.T @ Primitives._rot(st["palms"]["right"]["quat"])
            pose.append([np.linalg.norm(actual_pos - ref_pos), rotation_error(actual_rot, ref_rot)])
            measured_pos, measured_rot = ik.fk(np.asarray(st["joints"]), "right")
            fk.append([np.linalg.norm(actual_pos - measured_pos), rotation_error(actual_rot, measured_rot)])
        entry = dict(
            phase=segment[0]["phase"],
            samples=len(segment),
            hold_s=settled[-1]["wall"] - settled[0]["wall"],
            reference=goal.tolist(),
            measured=median.tolist(),
            error=(median - goal).tolist(),
            early_wrist_error=np.median(
                np.array([r["state"]["joints"] for r in first])[:, 26:29] - goal[26:29], axis=0
            ).tolist(),
            late_wrist_error=(median - goal)[26:29].tolist(),
            root_pose_error_median=np.median(pose, axis=0).tolist(),
            fk_error_max=np.max(fk, axis=0).tolist(),
            pelvis_median=np.median(base, axis=0).tolist(),
            base_motion=float(
                np.max(
                    np.linalg.norm(
                        np.array([r["state"]["pelvis"]["pos"] for r in segment]) - segment[0]["state"]["pelvis"]["pos"],
                        axis=1,
                    )
                )
            ),
        )
        entry["pose_delivered"] = (
            entry["root_pose_error_median"][0] < 0.045 and entry["root_pose_error_median"][1] < 0.25
        )
        if entry["phase"] == "baseline":
            baseline = entry
        elif baseline is not None:
            joint_name, sign = entry["phase"].split(":")
            joint = MUJOCO_JOINT_NAMES.index(joint_name)
            amplitude = goal[joint] - baseline["reference"][joint]
            entry["gain"] = ((median - np.asarray(baseline["measured"])) / amplitude).tolist()
            entry["joint"] = joint
            entry["base_delta"] = (np.median(base, axis=0) - baseline["pelvis_median"]).tolist()
        rows.append(entry)
    result = dict(
        run=run,
        segments=rows,
        samples=len(samples),
        command_period_median=float(np.median(np.diff([r["command_wall"] for r in samples]))),
        max_state_age=max(r["wall"] - r["state"]["wall"] for r in samples),
        max_reference_step=float(np.max(np.abs(np.diff(np.array([r["reference"] for r in samples]), axis=0)))),
    )
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(root.name, "error:", run["error"], "samples", len(samples))
    for r in rows:
        if r["phase"] == "baseline":
            print(
                "  hold wrist error",
                np.round(r["late_wrist_error"], 3),
                "pose cm/deg",
                np.round(np.array(r["root_pose_error_median"]) * [100, 180 / np.pi], 2),
            )
        else:
            print(" ", r["phase"], "wrist gain", np.round(r["gain"][26:29], 2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path, nargs="+")
    ap.add_argument(
        "--require-pose",
        action="store_true",
        help="Exit nonzero when baseline steady pose misses existing transport thresholds",
    )
    ap.add_argument("--require-rate", action="store_true", help="Exit nonzero when median command period exceeds 30 ms")
    a = ap.parse_args()
    failed = False
    slow = False
    for root in a.capture:
        result = analyze(root)
        slow |= result["command_period_median"] > 0.03
        failed |= result["run"]["error"] is not None or any(
            not r["pose_delivered"] for r in result["segments"] if r["phase"] == "baseline"
        )
    if (a.require_pose and failed) or (a.require_rate and slow):
        raise SystemExit(1)
