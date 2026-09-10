"""CPU unloaded reach calibration; geometry is a fixed robot-relative test target.

This measures the mode-2 posture preference, not grasp success. The bottle is
moved away for the probe so object contact cannot mask the reachable envelope.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

import run_bottle as B
from probe_visual_expert import atomic_json, pose_angles
from visual_motion import GraspMotion
from visual_planner import GraspIntent, RobotObservation


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--target-height', type=float, default=.683)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit('Calibration ID already exists; inspect saved results')
    out.mkdir(parents=True)
    sources = [Path(__file__), Path(B.__file__), Path(B.R.__file__), *Path(__file__).parent.glob('visual_*.py')]
    (out / 'source').mkdir()
    for p in sources:
        shutil.copy2(p, out / 'source' / p.name)
    atomic_json(out / 'config.json', dict(front_distances=[.25, .30, .35, .40],
        shoulder_left=.15, target_height=args.target_height, no_object_contact=True,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    torch.set_num_threads(1)
    policy = B.Policy(device='cpu'); sim = B.Sim(policy.meta)
    B.R.GRIP_KP, B.R.GRIP_KD = 1., .025
    results = []
    target = np.array([-.45, 0, args.target_height])
    for distance in (.25, .30, .35, .40):
        for facing in ('edge', 'bottle'):
            xy = target[:2] + [-distance, .15]
            yaw = 0. if facing == 'edge' else float(np.arctan2(-.15, distance))
            sim.reset(xy, yaw, 0)
            sim.d.qpos[sim.crate_q:sim.crate_q + 3] = [3, 3, .1]
            mujoco.mj_forward(sim.m, sim.d)
            policy.reset(sim.state()); refs = B.Refs(policy, sim)
            motion = None; angles = []; errors = []; trace = []
            for k in range(300):
                robot = RobotObservation(sim.d.qpos[:3].copy(), sim.d.qpos[3:7].copy(), sim.pad('right'),
                                         sim.body('right_wrist_yaw_link')[1], np.zeros(3), .0245)
                if k < 75:
                    ref = dict(refs.pelvis(xy, yaw), **refs.hands_at_side(np.array([*xy, .76]), yaw))
                    targets = refs.futures(lambda _: ref, k * .02)
                else:
                    if motion is None:
                        wrist = targets['right_wrist_yaw_link']
                        motion = GraspMotion(robot, refs.hand_rel, previous_wrist=(wrist[0][0], wrist[1][0]))
                    quat = (Rotation.from_euler('z', yaw) * Rotation.from_euler('y', -13, degrees=True)).as_quat(scalar_first=True)
                    intent = GraspIntent('entry', xy, yaw, target, quat, .0245)
                    targets, _ = motion.targets(intent, robot)
                sim.step(policy(sim.state(), targets))
                angles.append(pose_angles(sim)); errors.append(np.linalg.norm(sim.pad('right') - target))
                trace.append(sim.d.qpos.copy())
            tail = np.asarray(angles)[-50:]
            result = dict(front_distance=distance, facing=facing,
                          steady_posture_mean_deg=tail.mean(axis=0).tolist(),
                          wrist_error_cm=float(np.mean(errors[-50:]) * 100),
                          warnings={mujoco.mjtWarning(j).name: int(w.number) for j,w in enumerate(sim.d.warning) if w.number})
            results.append(result)
            np.savez_compressed(out / f'd{distance:.2f}_{facing}.npz', qpos=np.array(trace), angles=np.array(angles), wrist_error=errors)
            atomic_json(out / 'results.json', results)
            print(result, flush=True)


if __name__ == '__main__':
    main()
