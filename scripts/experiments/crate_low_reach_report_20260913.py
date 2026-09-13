"""Summarize full-body carry pilots, preserving failures and measured pose windows."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(run):
    manifests = sorted(run.glob("manifest_*.jsonl"))
    if not manifests:
        return dict(attempts=None, status="no manifest; inspect command exit and preserved runtime failure")
    attempts = [r for p in manifests for r in rows(p)]
    (scene,) = run.glob("*/scene_locomani.xml")
    model, data = mujoco.MjModel.from_xml_path(str(scene)), None
    data = mujoco.MjData(model)
    torso = model.body("torso_link").id
    crate = model.body("blue_crate_object").id
    feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    base = model.jnt_dofadr[model.joint("floating_base_joint").id]
    details = []
    hashes = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests + [scene]}
    for attempt in attempts:
        result = dict(attempt)
        path = run / f"{attempt['video']}_motion.jsonl"
        if not path.exists():
            details.append(result)
            continue
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        motion = rows(path)
        ref = [r for r in motion if isinstance(r.get("root_reference"), dict)]
        if ref:
            result["full_reference_speed_max"] = float(np.max(np.abs([r["root_reference"]["velocity"] for r in ref])))
            t = np.array([r["t"] for r in ref])
            rotations = Rotation.from_quat(np.array([r["root_reference"]["quat"] for r in ref])[:, [1, 2, 3, 0]])
            dt = np.diff(t)
            good = (dt > 0) & (dt < 0.1)
            omega = (rotations[1:] * rotations[:-1].inv()).magnitude()
            result["root_reference_angular_speed_max"] = float(np.max(omega[good] / dt[good])) if good.any() else None
        trace_path = run / "trace" / f"{attempt['video']}_state_trace.npz"
        with np.load(trace_path) as trace:
            lookup = {int(seq): i for i, seq in enumerate(trace["seq"])}
            for phase in ("whole_body", "grasp", "close", "lift"):
                samples = list({r["seq"]: r for r in motion if r["phase"] == phase}.values())
                if not samples:
                    continue
                last = [r for r in samples if r["t"] >= samples[-1]["t"] - 0.3]
                torso_angles = []
                torso_pitch = []
                foot_positions = []
                base_angular = []
                crate_heights = []
                for r in samples:
                    i = lookup[r["seq"]]
                    data.qpos[:] = trace["qpos"][i]
                    mujoco.mj_kinematics(model, data)
                    foot_positions.append(data.xpos[feet, :2].copy())
                    crate_heights.append(float(data.xpos[crate, 2]))
                    base_angular.append(float(np.linalg.norm(trace["qvel"][i, base + 3 : base + 6])))
                    if r["t"] >= samples[-1]["t"] - 0.3:
                        torso_angles.append(float(np.degrees(np.arccos(np.clip(data.xmat[torso, 8], -1, 1)))))
                        torso_pitch.append(
                            float(Rotation.from_matrix(data.xmat[torso].reshape(3, 3)).as_euler("xyz", degrees=True)[1])
                        )
                q = np.median([r["joints"] for r in last], axis=0)
                rbase = Rotation.from_quat(np.array([r["base_quat"] for r in last])[:, [1, 2, 3, 0]])
                initial = np.median([r["base_position"] for r in samples if r["t"] <= samples[0]["t"] + 0.2], axis=0)
                final = np.median([r["base_position"] for r in last], axis=0)
                final_z = np.median(crate_heights[-len(last) :])
                hold_z = [z for r, z in zip(samples, crate_heights) if r["t"] >= samples[-1]["t"] - 1.0]
                result[phase + "_measured"] = dict(
                    window_samples=len(last),
                    span_s=samples[-1]["t"] - samples[0]["t"],
                    torso_deg=float(np.median(torso_angles)),
                    torso_forward_pitch_deg=float(np.median(torso_pitch)),
                    pelvis_pitch_deg=float(np.median(rbase.as_euler("xyz")[:, 1]) * 180 / np.pi),
                    waist_deg=np.degrees(q[12:15]).tolist(),
                    knee_deg=np.degrees(q[[3, 9]]).tolist(),
                    elbow_deg=np.degrees(q[[18, 25]]).tolist(),
                    wrist_pitch_deg=np.degrees(q[[20, 27]]).tolist(),
                    pelvis_height=float(final[2]),
                    pelvis_drop_m=float(initial[2] - final[2]),
                    retreat_m=float(initial[0] - final[0]),
                    foot_xy_excursion_max_m=np.max(
                        np.linalg.norm(np.array(foot_positions) - foot_positions[0], axis=2), axis=0
                    ).tolist(),
                    base_angular_speed_p95=float(np.percentile(base_angular, 95)),
                    crate_peak_to_final_m=float(max(crate_heights) - final_z),
                    crate_final_one_second_span_m=float(np.ptp(hold_z)),
                )
        hashes[str(trace_path.relative_to(run))] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        details.append(result)
    return dict(
        source=json.loads((run / "config_snapshot.json").read_text())["git_sha"],
        attempts=len(attempts),
        successes=sum(r["ok"] for r in attempts),
        grasp_started=sum("reach_start_s" in r for r in attempts),
        failures=dict(Counter(r["why"] for r in attempts if not r["ok"])),
        rounds=details,
        input_sha256=hashes,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    result = {name: summarize(args.root / name) for name in args.runs}
    result["scope"] = (
        "All attempts kept by run/revision. End poses: last 0.3 s, seq-deduplicated, matched trace FK. Foot excursion is motion, not necessarily slipping. Angular velocity covers the motion-sampled interval of each phase and can include contact; the initial StreamedArm.start handshake is not sampled by MotionRecording."
    )
    result["report_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.root / "body_report.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, value in result.items():
        if isinstance(value, dict):
            print(name, json.dumps({k: v for k, v in value.items() if k not in ("rounds", "input_sha256")}))


if __name__ == "__main__":
    main()
