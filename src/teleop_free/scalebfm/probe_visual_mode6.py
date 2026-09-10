"""Focused CPU comparison of valid FK targets in modes 2 and 6.

Unloaded static reaching only: no expert success claim, data collection or VLA.
Mode 6 receives shoulder/elbow/wrist references from an arm IK; it has no torso
mask. Record whether those references actually improve the executed posture.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

import run_bottle as B
from probe_visual_expert import atomic_json, pose_angles
from visual_reference import PoseTransition
from visual_scene import make_sim


def solve_arm(sim, target, pitch):
    data = mujoco.MjData(sim.m)
    data.qpos[:] = sim.d.qpos
    for name, angle in [('waist_yaw_joint', 0), ('waist_roll_joint', 0), ('waist_pitch_joint', pitch)]:
        data.qpos[sim.m.jnt_qposadr[sim.m.joint(name).id]] = np.radians(angle)
    joints = [sim.m.joint(n).id for n in ('right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
              'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
              'right_wrist_pitch_joint', 'right_wrist_yaw_joint')]
    qa = sim.m.jnt_qposadr[joints]
    bounds = sim.m.jnt_range[joints]
    margin = .05 * (bounds[:, 1] - bounds[:, 0])
    low, high = bounds[:, 0] + margin, bounds[:, 1] - margin
    initial = np.clip(data.qpos[qa], low, high)
    bid = sim.bid['right_wrist_yaw_link']
    def residual(q):
        data.qpos[qa] = q
        mujoco.mj_forward(sim.m, data)
        r = Rotation.from_quat(data.xquat[bid], scalar_first=True)
        pad = data.xpos[bid] + r.apply([.1515, 0, 0])
        return np.r_[pad - target, .1 * r.as_rotvec(), .001 * (q - initial)]
    solutions = []
    rng = np.random.default_rng(61)
    for start in [initial, *[rng.uniform(low, high) for _ in range(3)]]:
        result = least_squares(residual, start, bounds=(low, high), max_nfev=200)
        solutions.append(result)
    result = min(solutions, key=lambda r: np.linalg.norm(r.fun[:6]))
    error = residual(result.x)
    names = ['pelvis'] + [f'{side}_{link}_link' for side in ('left', 'right')
                         for link in ('shoulder_roll', 'elbow', 'wrist_yaw')]
    poses = {name: (data.xpos[sim.m.body(name).id].copy(), data.xquat[sim.m.body(name).id].copy()) for name in names}
    return poses, dict(position_error_cm=float(np.linalg.norm(error[:3]) * 100),
                       rotation_error_deg=float(np.degrees(np.linalg.norm(error[3:6]) / .1)),
                       arm_joints=result.x.tolist())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit('Probe ID already exists; inspect results instead of repeating')
    out.mkdir(parents=True)
    files = [Path(__file__), Path(B.__file__), Path(B.R.__file__), *Path(__file__).parent.glob('visual_*.py')]
    (out / 'source').mkdir()
    for p in files:
        shutil.copy2(p, out / 'source' / p.name)
    atomic_json(out / 'config.json', dict(mode_comparison=[2, 6],
        cases=[['high_near', .25, .823, 0], ['low_near', .30, .683, 25], ['low_deep', .45, .683, 30]],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    torch.set_num_threads(1)
    policy = B.Policy(device='cpu'); sim = make_sim(policy)
    B.R.GRIP_KP, B.R.GRIP_KD = 1., .025
    results = []
    for name, distance, height, pitch in [('high_near', .25, .823, 0), ('low_near', .30, .683, 25), ('low_deep', .45, .683, 30)]:
        for mode in (2, 6):
            sim.reset([-2, 0], 0, 0); sim.d.qpos[2] = .76
            mujoco.mj_forward(sim.m, sim.d)
            target = np.array([-2 + distance, -.15, height])
            poses, ik = solve_arm(sim, target, pitch)
            initial = {n: (sim.d.xpos[sim.m.body(n).id].copy(), sim.d.xquat[sim.m.body(n).id].copy()) for n in poses}
            bridges = {n: PoseTransition(initial[n], poses[n], 2.5, pad_offset=np.zeros(3)) for n in poses}
            policy.mode = torch.tensor([mode], dtype=torch.long)
            policy.reset(sim.state())
            trace, angles, errors = [], [], []
            for k in range(325):
                elapsed = max(0., k * .02 - 1.5)
                targets = {}
                for n in poses:
                    if mode == 2 and 'shoulder' in n or mode == 2 and 'elbow' in n:
                        continue
                    pairs = [bridges[n].sample(*poses[n], elapsed + j * .02 if k >= 75 else 0.) for j in range(6)]
                    targets[n] = np.stack([v[0] for v in pairs]), np.stack([v[1] for v in pairs])
                sim.step(policy(sim.state(), targets))
                trace.append(sim.d.qpos.copy()); angles.append(pose_angles(sim))
                errors.append(np.linalg.norm(sim.pad('right') - target))
            result = dict(case=name, mode=mode, ik=ik,
                          steady_posture_mean_deg=np.mean(angles[-50:], axis=0).tolist(),
                          steady_wrist_error_cm=float(np.mean(errors[-50:]) * 100),
                          warnings={mujoco.mjtWarning(j).name: int(w.number) for j,w in enumerate(sim.d.warning) if w.number})
            results.append(result)
            np.savez_compressed(out / f'{name}_mode{mode}.npz', qpos=trace, angles=angles, wrist_error=errors)
            atomic_json(out / 'results.json', results)
            print(result, flush=True)


if __name__ == '__main__':
    main()
