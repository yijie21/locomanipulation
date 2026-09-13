"""Summarize all low-table attempts and measured crouch poses; no simulator step."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from crate_table_joint_report_20260913 import extent, rows


def summarize(run):
    manifests = sorted(run.glob("manifest_*.jsonl"))
    attempts = [r for path in manifests for r in rows(path)]
    (scene,) = run.glob("*/scene_locomani.xml")
    inputs = manifests + [scene, run / "config_snapshot.json", run / "deploy_receipts.jsonl"]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    torso = model.body("torso_link").id
    result = []
    for attempt in attempts:
        record = dict(attempt)
        path = run / f"{attempt['video']}_motion.jsonl"
        if not path.exists():
            result.append(record)
            continue
        motion = rows(path)
        inputs.append(path)
        crouch = [r for r in motion if r["phase"] == "crouch"]
        body = [r["body_height"] for r in motion if r.get("body_height")]
        if len(body) > 1:
            dt = np.diff([r["feedback_wall"] for r in body])
            dc = np.diff([r["command"] for r in body])
            valid = (dt > 0) & (dt <= 0.1)
            record["height_command_rate_max"] = float(np.max(np.abs(dc[valid] / dt[valid]))) if np.any(valid) else None
        if crouch:
            first = [r for r in crouch if r["t"] <= crouch[0]["t"] + 0.2]
            last = list({r["seq"]: r for r in crouch if r["t"] >= crouch[-1]["t"] - 0.3}.values())
            initial = np.median([r["base_position"] for r in first], axis=0)
            final = np.median([r["base_position"] for r in last], axis=0)
            q = np.median([r["joints"] for r in last], axis=0)
            inclination = []
            trace_path = run / "trace" / f"{attempt['video']}_state_trace.npz"
            inputs.append(trace_path)
            with np.load(trace_path) as trace:
                lookup = {int(seq): i for i, seq in enumerate(trace["seq"])}
                for r in last:
                    data.qpos[:] = trace["qpos"][lookup[r["seq"]]]
                    mujoco.mj_kinematics(model, data)
                    inclination.append(float(np.degrees(np.arccos(np.clip(data.xmat[torso, 8], -1, 1)))))
            record["crouch_pose"] = dict(
                samples=len(last),
                pelvis_xyz=final.tolist(),
                pelvis_drop_m=float(initial[2] - final[2]),
                retreat_x_m=float(initial[0] - final[0]),
                lateral_shift_m=float(final[1] - initial[1]),
                waist_deg=float(np.degrees(q[14])),
                knee_deg=np.degrees(q[[3, 9]]).tolist(),
                torso_inclination_deg=float(np.median(inclination)),
            )
        result.append(record)
    poses = [r["crouch_pose"] for r in result if "crouch_pose" in r]
    return dict(
        revision=json.loads((run / "config_snapshot.json").read_text())["git_sha"],
        input_sha256={str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        attempts=len(result),
        successes=sum(r["ok"] for r in result),
        crouch_started=len(poses),
        crouch_ready=sum(r.get("crouch_ready", False) for r in result),
        grasp_started=sum("reach_start_s" in r for r in result),
        failures=dict(Counter(r["why"] for r in result if not r["ok"])),
        crouch_metrics={
            key: extent([p[key] for p in poses])
            for key in ["pelvis_drop_m", "retreat_x_m", "lateral_shift_m", "waist_deg", "torso_inclination_deg"]
        },
        rounds=result,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = {
        name: summarize(args.root / name)
        for name in ("crouch_smoke", "crouch_pilot", "standing30", "default55_smoke")
        if (args.root / name / "manifest_00.jsonl").exists()
    }
    report["scope"] = "All attempts counted separately by protocol/revision. Crouch end poses use the final "
    report["scope"] += "0.3 s median, deduplicated by seq; torso inclination uses FK of matching recorded qpos. "
    report["scope"] += "Retreat compares first 0.2 s and last 0.3 s pelvis X. Positive means away from the table. "
    report["scope"] += (
        "Height command slew uses its recorded feedback clock; no success claim is inferred from a height command."
    )
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, value in report.items():
        if isinstance(value, dict):
            print(name, json.dumps({k: v for k, v in value.items() if k not in ("rounds", "input_sha256")}))


if __name__ == "__main__":
    main()
