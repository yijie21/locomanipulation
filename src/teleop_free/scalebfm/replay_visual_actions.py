"""Replay native visual-expert actions on CPU with the deployment decoder.

Scene truth is used only to reset and score the simulation, never to decode an
action. Compare every simulated qpos and commanded wrist against the recording.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco
import numpy as np
import torch

import run_bottle as B
from probe_visual_expert import atomic_json
from visual_action import decode_action
from visual_executor import ActionExecutor
from visual_scene import STRATA, make_sim


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    root, out = Path(args.run), Path(args.out)
    if out.exists():
        raise SystemExit("Replay result already exists; inspect it instead of repeating")
    cfg = json.loads((root / "config.json").read_text())
    if not cfg.get("native_codec"):
        raise SystemExit("This run did not execute the shared native decoder")
    modules = [Path(B.__file__), Path(B.R.__file__), Path(__file__).with_name("visual_action.py")]
    if cfg.get('action_hz'):
        modules.append(Path(__file__).with_name('visual_executor.py'))
    for module in modules:
        expected = cfg["source_sha256"].get(str(module.resolve()))
        if expected != hashlib.sha256(module.read_bytes()).hexdigest():
            raise SystemExit(f"Source changed since recording: {module}")
    torch.set_num_threads(1)
    B.R.GRIP_KP, B.R.GRIP_KD = cfg["grip_kp"], cfg["grip_kd"]
    policy = B.Policy(device=cfg.get('policy_device', 'cpu'))
    sim = B.Sim(policy.meta, video=False)
    for g in range(sim.m.ngeom):
        name = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, sim.m.geom_bodyid[g]) or ""
        if "dex1_finger" in name and sim.m.geom_contype[g]:
            sim.m.geom_priority[g] = 1
            sim.m.geom_condim[g] = 4
            sim.m.geom_solref[g] = (0.002, 1)
            sim.m.geom_solimp[g] = (0.995, 0.9999, 0.001, 0.5, 2)
    results = []
    for case in sorted(root.glob("seed*")):
        if args.limit is not None and len(results) >= args.limit:
            break
        if not (case / "actions.npz").exists():
            continue
        result = json.loads((case / "result.json").read_text())
        if result.get('recovery_kind', 'none') != 'none':
            raise ValueError('This exact-action replayer supports normal resets only; recovery events require the intervention-aware evaluator')
        if result.get('scene_directory'):
            sim = make_sim(policy, result['scene_directory'])
        if result.get('stratum'):
            B.EDGE_INBOARD = STRATA[result['stratum']][1]
        os.environ["BFM_EDGE"] = result["edge"]
        B.STAND_D = (cfg["stand_distance"], cfg["stand_distance"])
        bottle, _, spawn, yaw, _ = B.sample_scene(sim, np.random.default_rng(result["seed"]))
        sim.reset(spawn, yaw, 0.0)
        sim.d.qpos[sim.crate_q:sim.crate_q + 3] = bottle
        mujoco.mj_forward(sim.m, sim.d)
        sim.grip_clip = 0.6
        policy.reset(sim.state())
        refs = B.Refs(policy, sim)
        executor = ActionExecutor(refs.hand_rel, action_hz=cfg['action_hz']) if cfg.get('action_hz') else None
        actions = np.load(case / "actions.npz")["actions"]
        recorded = np.load(case / "trace.npz")["qpos"]
        commands = json.loads((case / "commands.json").read_text())
        state_error = command_error = 0.0
        for i, row in enumerate(actions):
            if executor is None:
                targets, grip = decode_action(row, sim.d.qpos[:3], sim.d.qpos[3:7], refs.hand_rel)
            else:
                targets, grip = executor.step(sim.d.qpos[:3], sim.d.qpos[3:7], row if executor.needs_action else None)
            command_error = max(command_error, float(np.linalg.norm(
                targets["right_wrist_yaw_link"][0][0] - commands[i]["wrist_position"])))
            sim.grip_target[2:] = grip
            sim.step(policy(sim.state(), targets))
            state_error = max(state_error, float(np.max(abs(sim.d.qpos - recorded[i]))))
        results.append(dict(case=case.name, expert_ok=result["ok"],
                            qpos_max_abs_error=state_error, wrist_max_error_m=command_error,
                            exact=state_error <= 1e-6 and command_error <= 1e-6))
        print(json.dumps(results[-1]), flush=True)
    atomic_json(out, dict(run=str(root), cases=results,
                         exact_count=sum(r["exact"] for r in results), count=len(results)))


if __name__ == "__main__":
    main()
