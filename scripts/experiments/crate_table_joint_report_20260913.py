"""Measured joint angles near capture; all attempted episodes remain in the report."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extent(values):
    return (
        None if not values else dict(n=len(values), min=min(values), median=float(np.median(values)), max=max(values))
    )


def summarize(run):
    attempts = [r for p in sorted(run.glob("manifest_*.jsonl")) for r in rows(p)]
    details = []
    hashes = {}
    scenes = list(run.glob("*/scene_locomani.xml"))
    assert len(scenes) == 1
    model = mujoco.MjModel.from_xml_path(str(scenes[0]))
    data = mujoco.MjData(model)
    torso = model.body("torso_link").id
    for entry in attempts:
        result = dict(entry)
        motion = run / f"{entry['video']}_motion.jsonl"
        if motion.exists():
            hashes[motion.name] = hashlib.sha256(motion.read_bytes()).hexdigest()
            grasp = [r for r in rows(motion) if r["phase"] == "grasp"]
            if grasp:
                # Deduplicate repeated observations; retain the final 0.3 s of grasp.
                last = {r["seq"]: r for r in grasp if r["t"] >= grasp[-1]["t"] - 0.3}
                sample = list(last.values())
                q = np.degrees([r["joints"] for r in sample])
                result["end_grasp_window"] = {
                    "n": len(sample),
                    "span_s": sample[-1]["t"] - sample[0]["t"],
                    "median_deg": dict(
                        zip(
                            ("waist", "left_elbow", "right_elbow", "left_wrist", "right_wrist"),
                            np.median(q[:, [14, 18, 25, 20, 27]], axis=0).tolist(),
                        )
                    ),
                    "base_position_median": np.median([r["base_position"] for r in sample], axis=0).tolist(),
                }
                trace_path = run / "trace" / f"{entry['video']}_state_trace.npz"
                with np.load(trace_path) as trace:
                    indices = {int(seq): i for i, seq in enumerate(trace["seq"])}
                    inclination = []
                    for state in sample:
                        if state["seq"] not in indices:
                            continue
                        data.qpos[:] = trace["qpos"][indices[state["seq"]]]
                        mujoco.mj_kinematics(model, data)
                        inclination.append(float(np.degrees(np.arccos(np.clip(data.xmat[torso, 8], -1, 1)))))
                    result["end_grasp_window"]["torso_samples"] = len(inclination)
                    result["end_grasp_window"]["torso_inclination_deg"] = extent(inclination)
        dump = run / f"{entry['video']}_servo_end.json"
        if dump.exists():
            result["servo_end_touch_links"] = json.loads(dump.read_text())["crate"].get("touch_links", [])
        details.append(result)
    groups = {}
    for name, population in (
        ("reached", [r for r in details if "reach_start_s" in r]),
        ("closed", [r for r in details if "close_trigger" in r]),
        ("success", [r for r in details if r["ok"]]),
    ):
        groups[name] = {
            "count": len(population),
            "joint_degrees": {
                key: extent([r["end_grasp_window"]["median_deg"][key] for r in population if "end_grasp_window" in r])
                for key in ("waist", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
            },
        }
        groups[name]["torso_inclination_degrees"] = extent(
            [
                r["end_grasp_window"]["torso_inclination_deg"]["median"]
                for r in population
                if r.get("end_grasp_window", {}).get("torso_inclination_deg")
            ]
        )
        groups[name]["servo_end_wrist_link_contacts"] = sum(
            any("wrist" in name for name in r.get("servo_end_touch_links", [])) for r in population
        )
    snapshot = json.loads((run / "config_snapshot.json").read_text())
    return dict(
        revision=snapshot["git_sha"],
        attempts=len(attempts),
        groups=groups,
        failures=dict(Counter(r["why"] for r in attempts if not r["ok"])),
        actual_start_edge_m=extent([r["initial_edge_distance"] for r in attempts]),
        rounds=details,
        motion_sha256=hashes,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = {name: summarize(args.root / name) for name in ("table55", "table62")}
    by_label = {name: {r["video"]: r for r in report[name]["rounds"]} for name in report}
    labels = set(by_label["table55"]) & set(by_label["table62"])
    assert len(labels) == report["table55"]["attempts"] == report["table62"]["attempts"]
    paired = []
    for label in sorted(labels):
        low, high = [by_label[n][label] for n in ("table55", "table62")]
        for key in ("start_xy", "start_yaw", "crate_dy"):
            assert low[key] == high[key], (label, key)
        if "close_trigger" in low and "close_trigger" in high:
            paired.append(dict(label=label, low=low["end_grasp_window"], high=high["end_grasp_window"]))
    report["protocol"] = {
        "matched_sampling_labels": len(labels),
        "both_closed": paired,
        "measurement": "Per-episode median of measured joint angles in the final 0.3 s of grasp, "
        "deduplicated by observation seq. Reached failures are not capture poses; "
        "closed and success populations are reported separately. Joint angles are "
        "not palm world orientation. Sequential asynchronous runs are not paired "
        "realized poses, and a small sample does not establish causality or high reliability.",
    }
    (args.root / "joint_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ("table55", "table62"):
        print(name, json.dumps({k: v for k, v in report[name].items() if k not in ("rounds", "motion_sha256")}))


if __name__ == "__main__":
    main()
