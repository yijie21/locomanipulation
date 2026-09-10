"""CPU development runner for the observation-constrained ScaleBFM expert.

MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES='' .venv_sim/bin/python run_visual_bottle.py --out ...
No dataset/training launch; restartable per-case traces and applied-target audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import mujoco
import numpy as np
import torch

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.action import decode_action, encode_action
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.scalebfm.executor import ActionExecutor
from locomani.tasks.bottle_visual import geometry
from locomani.tasks.bottle_visual.config import BottleVisualTaskConfig
from locomani.tasks.bottle_visual.harness import AppliedPolicy, atomic_json, summarize
from locomani.tasks.bottle_visual.motion import GraspMotion
from locomani.tasks.bottle_visual.observation import SceneEstimate, SceneMeasurement, SceneTracker
from locomani.tasks.bottle_visual.pipeline.provenance import line_sources, snapshot_sources
from locomani.tasks.bottle_visual.pipeline.recording import RawEpisodeRecorder
from locomani.tasks.bottle_visual.planner import GraspConfig, RobotObservation, VisualGraspPlanner
from locomani.tasks.bottle_visual.recovery import KINDS, RecoveryIntervention
from locomani.tasks.bottle_visual.sampling import (
    DEFAULT_PROFILE,
    LEGACY_PROFILE,
    PROFILES,
    profile_edges,
    sample_reset,
    validate_edges,
)
from locomani.tasks.bottle_visual.scene import STRATA, make_sim, scene_directory
from locomani.tasks.bottle_visual.sensor import BottleProxySensor
from locomani.tasks.bottle_visual.success import PROTOCOL, LiftSuccess
from locomani.tasks.bottle_visual.visibility import head_visible, initial_visibility


def proprio(sim, velocity):
    return RobotObservation(
        sim.d.qpos[:3].copy(),
        sim.d.qpos[3:7].copy(),
        sim.pad("right"),
        sim.body("right_wrist_yaw_link")[1],
        velocity.copy(),
        float(sim.grip_target[2]),
    )


def run_case(sim, policy, sensor, seed, edge, out, task, recording=None, recovery_kind="none", stratum="low_near"):
    """One expert case on `sim` under `task` (BottleVisualTaskConfig); the body below still reads the
    task's fields through the local names it was written against."""
    config, max_s, native_codec = task.grasp, task.max_s, task.native_codec
    action_hz, integral_gain, reference_style, scene_profile = (
        task.action_hz or None,
        task.integral_gain,
        task.reference_style,
        task.scene_profile,
    )
    bottle, stand, spawn, yaw, meta = sample_reset(
        geometry,
        sim,
        np.random.default_rng(seed),
        edge,
        stratum,
        scene_profile,
        stand_d=(config.stand_distance, config.stand_distance),
        inboard=STRATA[stratum][1],
    )
    intervention = RecoveryIntervention(recovery_kind)
    if recovery_kind == "stop_short":
        inward = np.zeros(3)
        inward[0 if edge[0] == "x" else 1] = 1 if edge[1] == "-" else -1
        edge_point = bottle - meta["inboard"] * inward
        edge_point[2] = sim.table_top
        setup_scene = SceneEstimate(tuple(bottle), sim.table_top, tuple(edge_point), tuple(inward), 0.0, 1.0, True)
        # Reset placement only; the rollout planner is created fresh below and
        # receives only the delayed, image-gated sensor observations.
        setup_robot = RobotObservation(
            np.array([*spawn, 0.76]),
            np.array([1.0, 0, 0, 0]),
            np.zeros(3),
            np.array([1.0, 0, 0, 0]),
            np.zeros(3),
            0.0245,
        )
        setup_intent = VisualGraspPlanner(config).decide(setup_scene, setup_robot)
        spawn = setup_intent.stand_xy - 0.10 * inward[:2]
        yaw = setup_intent.yaw
        intervention.events.append(dict(kind="stop_short", t=0.0, spawn=spawn.tolist(), offset_m=0.10))
    sim.reset(spawn, yaw, 0.0)
    sim.d.qpos[sim.crate_q : sim.crate_q + 3] = bottle
    mujoco.mj_forward(sim.m, sim.d)
    sim.grip_clip = 0.6
    sensor.reset()
    wrapped = AppliedPolicy(policy, sim)
    wrapped.reset(sim.state())
    refs = R.Refs(policy, sim)
    executor = ActionExecutor(refs.hand_rel, action_hz=action_hz) if action_hz else None
    planner, tracker = VisualGraspPlanner(config), SceneTracker()
    motion = None
    queue = []
    intent = None
    velocity = np.zeros(3)
    previous_pad = sim.pad("right")
    trace, observations, events = [], [], []
    actions = []
    forecast_error = 0.0
    phase = "settle"
    success = LiftSuccess()
    first_move = None
    why = "timeout"
    ok = False
    initial_failure = None
    first_record_evidence = None
    warnings = {}
    knock_reference = bottle[:2].copy()
    t0 = time.time()
    for k in range(int(max_s / R.DT)):
        t = k * R.DT
        changed = intervention.update(sim, t, phase)
        if changed == "object_shift":
            knock_reference = sim.crate()[0][:2].copy()
        if changed == "wrist_offset":
            previous_pad = sim.pad("right")
            velocity[:] = 0
        if intervention.error:
            why = intervention.error
            break
        pad = sim.pad("right")
        velocity = 0.8 * velocity + 0.2 * (pad - previous_pad) / R.DT
        previous_pad = pad
        if tracker.capture_due(t):
            if t < intervention.blackout_until:
                measurement = SceneMeasurement(t, None, None, None, None)
                sensor.last_evidence = dict(t=t, pixels=[0, 0], fractions=[0.0, 0.0], boxes=[None, None], blackout=True)
            else:
                measurement = sensor.capture(t)
            tracker.ingest(measurement)
            queue.append((t + tracker.latency, proprio(sim, velocity)))
            observations.append(dict(**sensor.last_evidence, bottle=measurement.bottle_position))
        if k in (0, 80) and not head_visible(sensor.last_evidence):
            initial_failure = initial_failure or f"head camera cannot see bottle at t={t:.2f}"
        record_now = (
            recording is not None
            and k >= 80
            and k % 2 == 0
            and (intervention.recording_ready() or initial_failure is not None)
        )
        if record_now and first_record_evidence is None:
            # Recovery recordings can start between planner captures. Audit the
            # actual first student frame without delivering an extra policy cue.
            if t < intervention.blackout_until:
                first_record_evidence = dict(t=t, pixels=[0, 0], fractions=[0.0, 0.0], blackout=True)
            else:
                sensor.capture(t)
                first_record_evidence = dict(sensor.last_evidence)
            if not head_visible(first_record_evidence):
                initial_failure = initial_failure or f"head camera cannot see bottle at first recorded frame t={t:.2f}"
        estimate = tracker.estimate(t)
        while queue and queue[0][0] <= t + 1e-9:
            _, captured_robot = queue.pop(0)
            if t >= R.SETTLE_S and initial_failure is None:
                intent = planner.decide(estimate, captured_robot)
        if t < R.SETTLE_S or intent is None or initial_failure is not None:
            ref = dict(refs.pelvis(spawn, yaw), **refs.hands_at_side(np.array([*spawn, R.PELVIS_Z]), yaw))
            targets = refs.futures(lambda _: ref, t)
        else:
            robot = proprio(sim, velocity)
            if motion is None:
                last = wrapped.rows[-1]
                previous = np.array(last["wrist_position"]), np.array(last["wrist_quaternion"])
                motion = GraspMotion(
                    robot,
                    refs.hand_rel,
                    previous_wrist=previous,
                    integral_gain=integral_gain,
                    reference_style=reference_style,
                )
            targets, grip = motion.targets(intent, robot)
            sim.grip_target[2:] = grip
            if intent.phase != phase:
                phase = intent.phase
                events.append(dict(t=t, phase=phase, pad=pad.tolist(), bottle=estimate.bottle_position))
        if record_now:
            recording.capture(t, executor.row, blackout=t < intervention.blackout_until)
        if native_codec:
            row = encode_action(targets, sim.d.qpos[:3], sim.d.qpos[3:7], sim.grip_target[2])
            if executor is None:
                decoded, grip = decode_action(row, sim.d.qpos[:3], sim.d.qpos[3:7], refs.hand_rel)
            else:
                decoded, grip = executor.step(sim.d.qpos[:3], sim.d.qpos[3:7], row if executor.needs_action else None)
                row = executor.row
            forecast_error = max(
                forecast_error,
                float(
                    np.max(
                        np.linalg.norm(decoded["right_wrist_yaw_link"][0] - targets["right_wrist_yaw_link"][0], axis=1)
                    )
                ),
            )
            targets = decoded
            sim.grip_target[2:] = grip
            actions.append(row)
            if record_now:
                recording.append_action(row)
            if motion is not None:
                motion.previous_wrist = (
                    decoded["right_wrist_yaw_link"][0][0].copy(),
                    decoded["right_wrist_yaw_link"][1][0].copy(),
                )
        wrapped.phase = phase
        sim.step(wrapped(sim.state(), targets))
        trace.append(sim.d.qpos.copy())
        if initial_failure is not None and k >= 80:
            why = initial_failure
            break
        bp, bq = sim.crate()
        if first_move is None and np.linalg.norm(bp[:2] - knock_reference) > 0.005:
            first_move = [round(t + R.DT, 3), phase]
        warnings = {mujoco.mjtWarning(j).name: int(w.number) for j, w in enumerate(sim.d.warning) if w.number}
        if warnings or sim.fallen() or not np.isfinite(sim.d.qpos).all():
            why = "physics warning/fall"
            break
        rise = float(bp[2] - bottle[2])
        lifted = success.update(t + R.DT, rise, True)
        if lifted:
            ok = True
            why = ""
            break
    np.savez_compressed(out / "trace.npz", qpos=np.asarray(trace), dt=R.DT)
    if native_codec:
        np.savez_compressed(out / "actions.npz", actions=np.asarray(actions), dt=R.DT)
    atomic_json(out / "commands.json", wrapped.rows)
    atomic_json(out / "observations.json", observations)
    result = dict(
        seed=seed,
        edge=edge,
        scene_profile=scene_profile,
        ok=ok,
        success_protocol=PROTOCOL,
        why=why,
        phase=phase,
        len_s=len(trace) * R.DT,
        wall_s=round(time.time() - t0, 3),
        events=events,
        scene=meta,
        rise_cm=float((sim.crate()[0][2] - bottle[2]) * 100),
        metrics=summarize(wrapped.rows),
        physics_warnings=warnings,
        first_move=first_move,
        forecast_linearization_max_cm=forecast_error * 100,
    )
    result.update(
        initial_visibility=initial_visibility(observations, first_record_evidence),
        first_record_evidence=first_record_evidence,
    )
    result.update(
        recovery_kind=recovery_kind, recovery_triggered=intervention.triggered, recovery_events=intervention.events
    )
    atomic_json(out / "simulation_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="101,102,103")
    parser.add_argument(
        "--config", help="BottleVisualTaskConfig JSON (configs/tasks/bottle_visual.json); default = the ruled expert"
    )
    parser.add_argument("--backend", help="ScaleBfmBackendConfig JSON (configs/backends/scalebfm.json)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="task config override, e.g. grasp.stand_distance=0.4",
    )
    parser.add_argument(
        "--set-backend", action="append", default=[], metavar="KEY=VALUE", help="backend config override"
    )
    parser.add_argument("--edges", help="Defaults to the selected profile edges; yplus_center_v1 only permits y+")
    parser.add_argument("--record-observations", action="store_true")
    parser.add_argument(
        "--case-offset", type=int, default=0, help="Original scene-plan index when resuming a disjoint seed suffix"
    )
    args = parser.parse_args()
    task = (BottleVisualTaskConfig.load(args.config) if args.config else BottleVisualTaskConfig()).with_overrides(
        args.set
    )
    backend = (ScaleBfmBackendConfig.load(args.backend) if args.backend else ScaleBfmBackendConfig()).with_overrides(
        args.set_backend
    )
    edges = (args.edges or ",".join(profile_edges(task.scene_profile))).split(",")
    try:
        validate_edges(task.scene_profile, edges)
    except ValueError as exc:
        parser.error(str(exc))
    if task.action_hz and not task.native_codec:
        parser.error("action_hz requires native_codec")
    if task.action_hz not in (0, 25, 50):
        parser.error("action_hz must be 0, 25 or 50")
    if args.record_observations and (not task.native_codec or task.action_hz != 25):
        parser.error("Student recording requires the native 25 Hz executor")
    if not set(task.recoveries) <= set(KINDS):
        parser.error("Unknown recovery intervention")
    if not set(task.strata) <= STRATA.keys():
        parser.error("Unknown scene stratum")
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = line_sources()
    cfg = dict(
        task=task.to_dict(),
        backend=backend.to_dict(),
        seeds=args.seeds,
        edges=",".join(edges),
        record_observations=args.record_observations,
        case_offset=args.case_offset,
        render_backend=os.environ["MUJOCO_GL"],
        render_device=os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )
    saved = out / "config.json"
    if saved.exists() and json.loads(saved.read_text()) != cfg:
        raise SystemExit("Saved config/source differs; use a new experiment ID")
    atomic_json(saved, cfg)
    run_meta = out / "run.json"
    status = (
        json.loads(run_meta.read_text())
        if run_meta.exists()
        else dict(started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv, pid=os.getpid())
    )
    status["status"] = "running"
    atomic_json(run_meta, status)
    snapshot_sources(out, files)
    policy = R.Policy(device=task.policy_device, backend=backend)
    environments = {}
    results = []
    seeds = [int(x) for x in args.seeds.split(",")]
    strata, recoveries = task.strata, task.recoveries
    try:
        for local_i, seed in enumerate(seeds):
            i = local_i + args.case_offset
            stratum = strata[i % len(strata)]
            recovery_kind = recoveries[(i // len(strata)) % len(recoveries)]
            edge = edges[(i // len(strata) + i % len(strata)) % len(edges)]
            case = out / f"seed{seed}_{edge}"
            case.mkdir(exist_ok=True)
            if (case / "result.json").exists():
                if args.record_observations and not (case / "student.npz").exists():
                    raise RuntimeError(
                        f"Completed physics but incomplete recording: {case}; recover artifacts before resuming"
                    )
                result = json.loads((case / "result.json").read_text())
            else:
                directory = scene_directory(out / "scene_assets", stratum, backend)
                if stratum not in environments:
                    sim = make_sim(policy, directory, backend)
                    environments[stratum] = sim, BottleProxySensor(sim)
                sim, sensor = environments[stratum]
                recording = RawEpisodeRecorder(sim, case) if args.record_observations else None
                try:
                    result = run_case(
                        sim,
                        policy,
                        sensor,
                        seed,
                        edge,
                        case,
                        task,
                        recording=recording,
                        recovery_kind=recovery_kind,
                        stratum=stratum,
                    )
                finally:
                    if recording is not None:
                        recording.close()
                result.update(stratum=stratum, scene_directory=str(directory.resolve()))
                atomic_json(case / "result.json", result)
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        for _, sensor in environments.values():
            sensor.close()
    atomic_json(
        out / "summary.json",
        dict(config=cfg, count=len(results), successes=sum(r["ok"] for r in results), cases=results),
    )
    status.update(status="complete", ended_utc=datetime.now(timezone.utc).isoformat(), count=len(results))
    atomic_json(run_meta, status)


if __name__ == "__main__":
    main()
