"""CPU-only, restartable paired experiments for the visual expert Goal.

Does not render, record a dataset, open ports or initialize CUDA. Each case stores
the exact config/source hashes and applied commands; completed cases are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from locomani.backends.scalebfm.reference import MotionLimits, PoseReference, PoseTransition
from locomani.tasks.bottle_visual import scene_legacy as B
from locomani.tasks.bottle_visual.pipeline.provenance import line_sources


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def pose_angles(sim):
    values = []
    for name in ("pelvis", "torso_link"):
        r = B.mat(sim.body(name)[1])
        values.extend(np.degrees([np.arctan2(r[2, 1], r[2, 2]), np.arcsin(np.clip(-r[2, 0], -1, 1))]))
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        values.append(float(np.degrees(sim.d.qpos[sim.m.jnt_qposadr[sim.m.joint(name).id]])))
    return values


class AppliedPolicy:
    def __init__(self, policy, sim, limits=None, transition=False):
        self.policy, self.sim, self.limits = policy, sim, limits
        self.reference = None
        self.rows = []
        self.phase = "settle"
        self.use_transition = transition
        self.previous_command = None
        self.previous_phase = None
        self.transition = None
        self.transition_tick = 0

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def reset(self, state):
        self.policy.reset(state)
        self.reference = None
        self.rows = []
        self.previous_command = None
        self.previous_phase = None
        self.transition = None

    def __call__(self, state, targets):
        wrist = "right_wrist_yaw_link"
        if self.limits is not None:
            if self.reference is None:
                self.reference = PoseReference(*self.sim.body(wrist), self.limits)
            targets = dict(targets)
            targets[wrist] = self.reference.futures(*targets[wrist], B.DT)
        elif self.use_transition:
            raw_p, raw_q = targets[wrist]
            if self.phase != self.previous_phase:
                previous = self.previous_command or self.sim.body(wrist)
                self.transition = PoseTransition(previous, (raw_p[0], raw_q[0]), 2.5 if self.phase == "reach" else 0.5)
                self.transition_tick = len(self.rows)
            elapsed = (len(self.rows) - self.transition_tick) * B.DT
            pairs = [self.transition.sample(p, q, elapsed + i * B.DT) for i, (p, q) in enumerate(zip(raw_p, raw_q))]
            targets = dict(targets)
            targets[wrist] = np.stack([v[0] for v in pairs]), np.stack([v[1] for v in pairs])
            self.previous_command = targets[wrist][0][0].copy(), targets[wrist][1][0].copy()
            self.previous_phase = self.phase
        self.rows.append(
            dict(
                t=len(self.rows) * B.DT,
                phase=self.phase,
                angles=pose_angles(self.sim),
                wrist_position=targets[wrist][0][0].tolist(),
                wrist_quaternion=targets[wrist][1][0].tolist(),
                pelvis_position=state["root_pos"].tolist(),
            )
        )
        return self.policy(state, targets)

    def tick(self, sim, state, command, t, phase, cmd_next=None):
        # run_round recorder hook supplies phase only. Applied targets are captured
        # at the policy boundary AFTER the reference adapter, not from raw ref(t).
        self.phase = phase


def summarize(rows):
    angles = np.array([r["angles"] for r in rows])
    pos = np.array([r["wrist_position"] for r in rows])
    rot = Rotation.from_quat(np.array([r["wrist_quaternion"] for r in rows]), scalar_first=True)
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dr = np.degrees((rot[1:] * rot[:-1].inv()).magnitude())
    phases = {}
    for phase in dict.fromkeys(r["phase"] for r in rows):
        mask = np.array([r["phase"] == phase for r in rows])
        phases[phase] = np.round(abs(angles[mask]).max(0), 4).tolist()
    return dict(
        columns=["pelvis_roll", "pelvis_pitch", "torso_roll", "torso_pitch", "waist_yaw", "waist_roll", "waist_pitch"],
        phase_max_abs_deg=phases,
        torso_roll_max_deg=float(abs(angles[:, 2]).max()),
        torso_roll_above15_s=float((abs(angles[:, 2]) > 15).sum() * B.DT),
        waist_yaw_max_deg=float(abs(angles[:, 4]).max()),
        wrist_step_max_cm=float(dp.max(initial=0) * 100),
        wrist_step_max_deg=float(dr.max(initial=0)),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", choices=["baseline", "pose_servo", "pose_transition"], default="pose_transition")
    parser.add_argument("--seeds", default="101,102,103,104,105,106,107,108,109,110,111,112")
    parser.add_argument("--edges", default="x-,y-,y+")
    parser.add_argument("--linear-speed", type=float, default=0.6)
    parser.add_argument("--angular-speed", type=float, default=2.5)
    parser.add_argument("--gain", type=float, default=6.0)
    parser.add_argument("--grip-kp", type=float, default=40.0)
    parser.add_argument("--grip-kd", type=float, default=1.0)
    parser.add_argument("--stand-d", default="0.45,0.55")
    parser.add_argument("--carry-back", type=float, default=0.15)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = line_sources()
    config = vars(args).copy()
    config["source_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit("Configuration/source differs from saved run; use a new run ID")
    atomic_json(config_path, config)
    source_dir = out / "source"
    source_dir.mkdir(exist_ok=True)
    for source in sources:
        shutil.copy2(source, source_dir / source.name)
    limits = (
        None
        if args.variant != "pose_servo"
        else MotionLimits(linear_speed=args.linear_speed, angular_speed=args.angular_speed, gain=args.gain)
    )
    B.R.GRIP_KP, B.R.GRIP_KD = args.grip_kp, args.grip_kd
    B.STAND_D = tuple(float(x) for x in args.stand_d.split(","))
    B.CARRY_BACK = args.carry_back
    policy = B.Policy(device="cpu")
    sim = B.Sim(policy.meta, video=False)
    for g in range(sim.m.ngeom):
        name = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, sim.m.geom_bodyid[g]) or ""
        if "dex1_finger" in name and sim.m.geom_contype[g] != 0:
            sim.m.geom_priority[g] = 1
            sim.m.geom_condim[g] = 4
            sim.m.geom_solref[g] = (0.002, 1.0)
            sim.m.geom_solimp[g] = (0.995, 0.9999, 0.001, 0.5, 2.0)
    seeds = [int(s) for s in args.seeds.split(",")]
    edges = args.edges.split(",")
    cases = []
    for i, seed in enumerate(seeds):
        edge = edges[i % len(edges)]
        case = out / f"seed{seed}_{edge}"
        case.mkdir(exist_ok=True)
        (case / "trace").mkdir(exist_ok=True)
        result_path = case / "result.json"
        if result_path.exists():
            cases.append(json.loads(result_path.read_text()))
            print(f"SKIP completed seed={seed} edge={edge}", flush=True)
            continue
        os.environ["BFM_EDGE"] = edge
        wrapped = AppliedPolicy(policy, sim, limits, args.variant == "pose_transition")
        t0 = time.time()
        result = B.run_round(
            sim, wrapped, 0, np.random.default_rng(seed), str(case), False, None, None, recorder=wrapped
        )
        warnings = {mujoco.mjtWarning(k).name: int(w.number) for k, w in enumerate(sim.d.warning) if w.number}
        case_result = dict(
            seed=seed,
            edge=edge,
            manifest=result,
            metrics=summarize(wrapped.rows),
            physics_warnings=warnings,
            wall_s=round(time.time() - t0, 3),
        )
        atomic_json(case / "commands.json", wrapped.rows)
        atomic_json(result_path, case_result)
        cases.append(case_result)
        print(json.dumps(case_result), flush=True)
    report = dict(
        config=config,
        count=len(cases),
        successes=sum(r["manifest"]["ok"] and not r["physics_warnings"] for r in cases),
        cases=cases,
    )
    atomic_json(out / "summary.json", report)
    print(f"DONE {report['successes']}/{report['count']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
