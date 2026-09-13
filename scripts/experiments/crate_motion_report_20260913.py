"""Compare carry attempts without discarding failures or pooling controller versions.

Run with the simulation Python: this script uses numpy but starts no simulator.
The first 3 s of grasping is an observation window, not a contact-free guarantee.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distribution(values):
    values = np.asarray(values, dtype=float)
    return (
        None
        if not len(values)
        else {
            "n": len(values),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    )


def attempt(run, row):
    result = {
        k: row.get(k)
        for k in (
            "video",
            "ok",
            "why",
            "initial_edge_distance",
            "brake_seconds",
            "handoff_seconds",
            "edge_clear",
            "reach_start_s",
            "capture_err_mm",
            "waist_pitch_meas_deg",
        )
    }
    result["initial_distance_admissible"] = 1.2 <= row.get("initial_edge_distance", -1) <= 1.5
    motion_path = run / f"{row['video']}_motion.jsonl"
    if not motion_path.exists():
        result["missing_motion"] = True
        return result
    motion = read_rows(motion_path)
    grasp = [r for r in motion if r["phase"] == "grasp"]
    if not grasp:
        return result
    result["grasp_samples"] = len(grasp)
    t = np.array([r["t"] for r in grasp])
    v = np.array([r["velocity"] for r in grasp])
    result["sent_speed_max"] = float(np.max(np.abs(v)))
    dt = np.diff(t)
    # Velocity is a per-interval average: acceleration belongs at interval centres.
    interval = (dt[:-1] + dt[1:]) / 2
    valid = (dt[:-1] >= 0.005) & (dt[:-1] <= 0.1) & (dt[1:] >= 0.005) & (dt[1:] <= 0.1)
    if np.any(valid):
        acc = np.max(np.abs(np.diff(v, axis=0)[1:][valid] / interval[valid, None]), axis=1)
        result["sent_acceleration"] = distribution(acc)
    result["irregular_intervals"] = int(np.sum((dt < 0.005) | (dt > 0.1)))
    early = [r for r in grasp if r["t"] <= row["reach_start_s"] + 3]
    result["early_grasp_seconds"] = early[-1]["t"] - early[0]["t"] if early else 0
    trace_path = run / "trace" / f"{row['video']}_state_trace.npz"
    if early and trace_path.exists():
        with np.load(trace_path) as trace:
            mask = (trace["seq"] >= early[0]["seq"]) & (trace["seq"] <= early[-1]["seq"])
            # MuJoCo free-base qvel: translation 0:3, angular velocity 3:6.
            result["early_base_angular_speed"] = distribution(np.linalg.norm(trace["qvel"][mask, 3:6], axis=1))
    return result


def summarize(run):
    rows = [r for p in sorted(run.glob("manifest_*.jsonl")) for r in read_rows(p)]
    metrics = [attempt(run, r) for r in rows]
    snapshot = json.loads((run / "config_snapshot.json").read_text())
    result = {
        "revision": snapshot["git_sha"],
        "attempts": len(rows),
        "successes": sum(bool(r["ok"]) for r in rows),
        "initial_distance_admissible": sum(r["initial_distance_admissible"] for r in metrics),
        "reached_grasp": sum("reach_start_s" in r for r in rows),
        "failures_by_stage": dict(Counter(r["why"].split("]")[0] + "]" for r in rows if not r["ok"])),
        "rounds": metrics,
        "sampled_starts": {r["video"]: [r[k] for k in ("start_xy", "start_yaw", "crate_dy")] for r in rows},
    }
    for field in ("brake_seconds", "handoff_seconds", "sent_speed_max"):
        result[field] = distribution([r[field] for r in metrics if r.get(field) is not None])
    result["brake_plus_handoff_seconds"] = distribution(
        [r["brake_seconds"] + r["handoff_seconds"] for r in rows if "handoff_seconds" in r]
    )
    for field in ("sent_acceleration", "early_base_angular_speed"):
        result[f"{field}_per_attempt_p95"] = distribution([r[field]["p95"] for r in metrics if r.get(field)])
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--variants", nargs="+", default=["baseline", "continuous_handoff", "bounded_motion"])
    args = parser.parse_args()
    report = {name: summarize(args.root / name) for name in args.variants}
    names = args.variants
    common = set.intersection(*(set(report[n]["sampled_starts"]) for n in names))
    for label in common:
        if any(report[n]["sampled_starts"][label] != report[names[0]]["sampled_starts"][label] for n in names):
            raise ValueError(f"sampling mismatch for {label}")
    report["protocol"] = {
        "matched_sampling_labels": len(common),
        "window_seconds": 3,
        "limits": "Asynchronous sequential runs; realized starting poses differ. Early reach may include contact. "
        "Command acceleration uses recorded publish times and excludes intervals outside 5–100 ms. "
        "Base angular velocity is a body-motion proxy, not a full torso or VLA metric. All failures remain counted.",
    }
    (args.root / "motion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in names:
        print(name, json.dumps({k: v for k, v in report[name].items() if k not in ("rounds", "sampled_starts")}))


if __name__ == "__main__":
    main()
