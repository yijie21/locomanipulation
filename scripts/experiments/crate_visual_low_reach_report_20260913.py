"""Measure visual low-reach phases against their matching full simulator traces.

Truth is used here only after each attempt. Measurements do not alter its verdict.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(run):
    attempts = [row for path in sorted(run.glob("manifest_*.jsonl")) for row in rows(path)]
    (scene,) = run.glob("*/scene_locomani.xml")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    pelvis = model.body("pelvis").id
    torso = model.body("torso_link").id
    crate = model.body("blue_crate_object").id
    base = model.jnt_dofadr[model.joint("floating_base_joint").id]
    report = []
    for attempt in attempts:
        label = attempt["label"]
        observation_path = run / "observations" / label / "observations.jsonl"
        observations = rows(observation_path) if observation_path.exists() else []
        result = dict(attempt, phases={})
        if not observations:
            report.append(result)
            continue
        trace = np.load(run / "trace" / f"{label}_state_trace.npz")
        lookup = {int(seq): index for index, seq in enumerate(trace["seq"])}
        started = observations[0]["wall_time"]
        endpoints = []
        for row in (observations[0], observations[-1]):
            data.qpos[:] = trace["qpos"][lookup[row["state_seq"]]]
            mujoco.mj_kinematics(model, data)
            endpoints.append(
                dict(
                    state_seq=row["state_seq"],
                    crate_position=data.xpos[crate].copy().tolist(),
                    crate_tilt_deg=float(np.degrees(np.arccos(np.clip(data.xmat[crate, 8], -1, 1)))),
                )
            )
        result["recorded_end_state"] = dict(
            initial=endpoints[0],
            final=endpoints[1],
            rise=endpoints[1]["crate_position"][2] - endpoints[0]["crate_position"][2],
        )
        for phase in dict.fromkeys(row["phase"] for row in observations):
            samples = [row for row in observations if row["phase"] == phase]
            first, last = samples[0], samples[-1]
            following = next((row for row in observations if row["wall_time"] > last["wall_time"]), last)
            angles, heights, crate_heights, knees, angular = [], [], [], [], []
            for row in samples:
                if row["state_seq"] not in lookup:
                    continue
                index = lookup[row["state_seq"]]
                data.qpos[:] = trace["qpos"][index]
                mujoco.mj_kinematics(model, data)
                angular.append(float(np.linalg.norm(trace["qvel"][index, base + 3 : base + 6])))
                crate_heights.append((row["wall_time"], float(data.xpos[crate, 2])))
                if row["wall_time"] >= last["wall_time"] - 0.3:
                    angles.append(
                        float(Rotation.from_matrix(data.xmat[torso].reshape(3, 3)).as_euler("xyz", degrees=True)[1])
                    )
                    heights.append(float(data.xpos[pelvis, 2]))
                    knees.append(np.degrees(np.asarray(row["joints"])[[3, 9]]))
            result["phases"][phase] = dict(
                start=first["wall_time"] - started,
                seconds=following["wall_time"] - first["wall_time"],
                samples=len(samples),
                visual_fraction=float(np.mean([row["vision"] is not None for row in samples])),
                torso_pitch_median=float(np.median(angles)) if angles else None,
                pelvis_height_median=float(np.median(heights)) if heights else None,
                knees_median=np.median(knees, axis=0).tolist() if knees else None,
                base_angular_p95=float(np.percentile(angular, 95)) if angular else None,
                final_second_crate_height_range=float(
                    np.ptp([z for stamp, z in crate_heights if stamp >= last["wall_time"] - 1])
                )
                if crate_heights
                else None,
            )
        ready = [row for row in observations if row["navigation"] is not None and row["navigation"]["ready"]]
        if ready:
            seq = ready[-1]["state_seq"]
            mask = (trace["seq"] >= seq) & (trace["seq"] < seq + 65)
            angular = np.linalg.norm(trace["qvel"][mask, base + 3 : base + 6], axis=1)
            result["takeover_65_state_samples"] = dict(
                start_seq=seq,
                count=int(sum(mask)),
                angular_peak=float(np.max(angular)),
                angular_p95=float(np.percentile(angular, 95)),
            )
        result["observation_sha256"] = hashlib.sha256(
            (run / "observations" / label / "observations.jsonl").read_bytes()
        ).hexdigest()
        report.append(result)
    return dict(
        run=str(run),
        attempts=len(attempts),
        successes=sum(row["ok"] for row in attempts),
        scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        results=report,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps([summarize(run) for run in args.runs], indent=2) + "\n")


if __name__ == "__main__":
    main()
