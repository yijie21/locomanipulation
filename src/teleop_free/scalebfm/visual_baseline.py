"""Frozen legacy task generator, adapted to the common student runtime.

Only the baseline task script is loaded from the snapshot. Physics, stable jaw
PD, recording and29-D execution are shared with the qualified expert.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import mujoco
import numpy as np

import run_bottle as B
from probe_visual_expert import AppliedPolicy, atomic_json, summarize
from visual_action import encode_action
from visual_executor import ActionExecutor
from visual_sampling import LEGACY_PROFILE, reset_sampler


def load_frozen(snapshot):
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot / 'manifest.json').read_text())['files']
    root = Path(__file__).resolve().parents[3]
    for relative in ['src/teleop_free/scalebfm/run_spike.py', 'src/teleop_free/scalebfm/bfm_policy.py']:
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != manifest[relative]:
            raise ValueError(f'Common low-level runtime changed since baseline: {relative}')
    relative = 'src/teleop_free/scalebfm/run_bottle.py'
    path = snapshot / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[relative]:
        raise ValueError('Frozen baseline task source changed')
    spec = importlib.util.spec_from_file_location('visual_frozen_legacy', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.SETTLE_S = 1.68
    return module


class BaselineAdapter(AppliedPolicy):
    def __init__(self, policy, sim, recording, sensor):
        super().__init__(policy, sim)
        self.recording, self.sensor = recording, sensor
        self.observations, self.actions = [], []

    def reset(self, state):
        super().reset(state)
        self.executor = ActionExecutor(B.Refs(self.policy, self.sim).hand_rel)
        self.sensor.reset()

    def __call__(self, state, targets):
        k = self.executor.tick
        if k % 20 == 0:
            self.sensor.capture(k * B.DT)
            self.observations.append(dict(self.sensor.last_evidence))
        record = self.recording is not None and k >= 80 and k % 2 == 0
        if record:
            self.recording.capture(k * B.DT, self.executor.row)
        row = encode_action(targets, self.sim.d.qpos[:3], self.sim.d.qpos[3:7], self.sim.grip_target[2])
        targets, grip = self.executor.step(self.sim.d.qpos[:3], self.sim.d.qpos[3:7], row if self.executor.needs_action else None)
        self.sim.grip_target[:2] = B.GRIP_OPEN  # Common runtime commands right jaw only.
        self.sim.grip_target[2:] = grip
        self.actions.append(self.executor.row.copy())
        if record:
            self.recording.append_action(self.executor.row)
        return super().__call__(state, targets)


def run_baseline(legacy, sim, policy, sensor, seed, edge, stratum, directory, out, recording, scene_profile=LEGACY_PROFILE):
    from visual_scene import STRATA
    os.environ['BFM_EDGE'] = edge
    legacy.EDGE_INBOARD = STRATA[stratum][1]
    wrapped = BaselineAdapter(policy, sim, recording, sensor)
    (out / 'trace').mkdir(exist_ok=True)
    with reset_sampler(legacy, edge, stratum, scene_profile):
        result = legacy.run_round(sim, wrapped, 0, np.random.default_rng(seed), str(out), False, None, None, recorder=wrapped)
    trace = np.load(out / 'trace/wall_s0_00_state_trace.npz')['qpos']
    np.savez_compressed(out / 'trace.npz', qpos=trace, dt=B.DT)
    np.savez_compressed(out / 'actions.npz', actions=np.asarray(wrapped.actions), dt=B.DT)
    warnings = {mujoco.mjtWarning(j).name: int(w.number) for j,w in enumerate(sim.d.warning) if w.number}
    original_ok = result['ok']
    result.update(seed=seed, edge=edge, stratum=stratum, scene_profile=scene_profile, scene_directory=str(directory.resolve()),
                  metrics=summarize(wrapped.rows), physics_warnings=warnings, baseline_original_ok=original_ok,
                  ok=bool(original_ok and not warnings and np.isfinite(trace).all() and trace[:, 2].min() >= .45),
                  recovery_kind='none')
    atomic_json(out / 'commands.json', wrapped.rows)
    atomic_json(out / 'observations.json', wrapped.observations)
    atomic_json(out / 'simulation_result.json', result)
    return result
