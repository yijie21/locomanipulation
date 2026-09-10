"""Offline acceptance gates for a recorded visual-expert attempt.

May read simulator truth for measurement. This module is never in the policy
or deployment execution path. Every failed gate is retained in the report.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.tasks.bottle_visual import geometry
from locomani.tasks.bottle_visual.harness import atomic_json
from locomani.tasks.bottle_visual.scene import make_sim
from locomani.tasks.bottle_visual.visibility import initial_visibility


def audit_case(case):
    case = Path(case)
    result = json.loads((case / "result.json").read_text())
    if result.get("stratum") not in ("high_near", "low_near", "low_deep") or not result.get("scene_directory"):
        raise ValueError(f"Missing explicit reach stratum/scene metadata: {case}")
    commands = json.loads((case / "commands.json").read_text())
    observations = json.loads((case / "observations.json").read_text())
    qpos = np.load(case / "trace.npz")["qpos"]
    angles = np.asarray([r["angles"] for r in commands])
    active = [r["t"] for r in commands if r["phase"] != "settle"]
    close = [r["t"] for r in commands if r["phase"] == "close"]
    decision_observations = [
        r
        for r in observations
        if r["t"] + 0.08 >= min(active, default=np.inf) - 1e-8 and r["t"] + 0.08 <= min(close, default=np.inf) + 1e-8
    ]
    visible = [any(p >= 32 and f >= 0.2 for p, f in zip(r["pixels"], r["fractions"])) for r in decision_observations]
    close_angles = [r["angles"][3] for r in commands if r["phase"] == "close"]
    pitch_close = float(np.median(abs(np.array(close_angles)))) if close_angles else None
    m = result["metrics"]
    first_record_time = None
    if (case / "student.npz").exists():
        with np.load(case / "student.npz") as data:
            times = data["simulation_time"]
            first_record_time = float(times[0]) if len(times) else float("nan")
    initial = initial_visibility(observations, result.get("first_record_evidence"), first_record_time)
    gates = dict(
        initial_visibility=all(initial.values()),
        task=bool(result["ok"]),
        physics=not result["physics_warnings"] and np.isfinite(qpos).all().item() and float(qpos[:, 2].min()) >= 0.45,
        wrist_position=m["wrist_step_max_cm"] <= 2.0001,
        wrist_rotation=m["wrist_step_max_deg"] <= 5.0001,
        side_lean=m["torso_roll_max_deg"] <= 20 and m["torso_roll_above15_s"] <= 0.200001,
        twist=m["waist_yaw_max_deg"] <= 25,
        forward_bend=float(abs(angles[:, 3]).max()) <= 40,
        visibility=bool(visible) and float(np.mean(visible)) >= 0.9,
        upright=(pitch_close is not None and pitch_close <= 15) if result.get("stratum") == "high_near" else True,
    )
    # Actual robot FK detects arm extension/raising, not a phase-name assertion.
    reach = [i for i, r in enumerate(commands) if r["phase"] == "reach"]
    onset_t = speed = None
    if reach:
        sim = make_sim(SimpleNamespace(meta=json.loads(Path(R.META).read_text())), result.get("scene_directory"))
        start = max(0, reach[0] - 1)
        reference = None
        for i in range(start, min(len(qpos), reach[-1] + 1)):
            sim.d.qpos[:] = qpos[i]
            mujoco.mj_forward(sim.m, sim.d)
            delta = sim.pad("right") - qpos[i, :3]
            yaw = R.yaw_of(qpos[i, 3:7])
            local = R.rot_z(yaw).T @ delta
            if reference is None:
                reference = local
            if local[0] - reference[0] >= 0.03 or local[2] - reference[2] >= 0.01:
                j = max(0, i - 5)
                speed = float(np.linalg.norm(qpos[i, :2] - qpos[j, :2]) / max(0.02, (i - j) * 0.02))
                onset_t = (i + 1) * 0.02
                break
    return dict(
        case=str(case),
        seed=result["seed"],
        stratum=result.get("stratum", "low_near"),
        accepted=all(gates.values()),
        gates=gates,
        initial_visibility=initial,
        visibility_fraction=float(np.mean(visible)) if visible else 0.0,
        visual_decision_updates=len(visible),
        close_pitch_median_deg=pitch_close,
        torso_pitch_peak_deg=float(abs(angles[:, 3]).max()),
        arm_extension_onset_s=onset_t,
        pelvis_speed_at_extension=speed,
        walk_reach_overlap=speed is not None and speed >= 0.05,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    reports = []
    for case in sorted(root.glob("seed*")):
        if (case / "result.json").exists():
            report = audit_case(case)
            atomic_json(case / "audit.json", report)
            reports.append(report)
    successes = [r for r in reports if r["gates"]["task"]]
    atomic_json(
        root / "audit_summary.json",
        dict(
            count=len(reports),
            accepted=sum(r["accepted"] for r in reports),
            successful_walk_reach_overlap_fraction=float(np.mean([r["walk_reach_overlap"] for r in successes]))
            if successes
            else 0.0,
            cases=reports,
        ),
    )
    print(json.dumps(dict(count=len(reports), accepted=sum(r["accepted"] for r in reports))), flush=True)


if __name__ == "__main__":
    main()
