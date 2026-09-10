"""Paired visual VLA evaluation and full delayed-chunk replay contract.

The predictor receives only RobotCameraObserver outputs. Scene truth below is
restricted to reset and scoring. Every action uses the collection executor.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco
import numpy as np
import torch

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.action import encode_action
from locomani.backends.scalebfm.camera import RobotCameraObserver
from locomani.backends.scalebfm.chunks import ChunkScheduler
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.scalebfm.executor import ActionExecutor
from locomani.paths import REPO
from locomani.tasks.bottle_visual import geometry
from locomani.tasks.bottle_visual.harness import AppliedPolicy, atomic_json, summarize
from locomani.tasks.bottle_visual.pipeline.provenance import line_sources, verify_runtime_assets
from locomani.tasks.bottle_visual.recovery import RecoveryIntervention
from locomani.tasks.bottle_visual.sampling import LEGACY_PROFILE, PROFILES, profile_edges, sample_reset
from locomani.tasks.bottle_visual.scene import STRATA, make_sim, scene_directory
from locomani.tasks.bottle_visual.sensor import BottleProxySensor

ROOT = Path(REPO)


def load_vla(path):
    frozen_path = Path(path).parent / "frozen_training.json"
    frozen = json.loads(frozen_path.read_text())
    for relative, expected in frozen["gr00t_source_sha256"].items():
        source = ROOT / "src" / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"GR00T inference source differs from training: {source}")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "configs"))
    import gr00t_bfm_visual_2cam  # Registers the frozen modality contract.

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.eval.open_loop_eval import parse_action_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    tag = EmbodimentTag.UNITREE_G1_SONIC
    config = MODALITY_CONFIGS[tag.value]
    policy = Gr00tPolicy(model_path=path, embodiment_tag=tag, device="cuda")
    policy.model.action_head.num_inference_timesteps = 4

    def predict(obs):
        chunk, _ = policy.get_action(parse_observation_gr00t(obs, config))
        return np.asarray(parse_action_gr00t(chunk)["action.visual_command"], dtype=np.float32)

    predict.modality_config = config
    return predict


def validation_diagnostic(predict, dataset, out):
    """Three fixed observation times per held-out episode; never selects weights."""
    from copy import deepcopy

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag

    cfg = deepcopy(predict.modality_config)
    # The loader retains this dictionary; keep scoring labels while the
    # observation-only config below excludes actions from predictor inputs.
    loader = LeRobotEpisodeLoader(dataset, deepcopy(cfg))
    cfg.pop("action")
    reports = []
    for episode in range(20):
        trajectory = loader[episode]
        for position in (0.0, 0.5, 0.9):
            index = int(position * max(0, len(trajectory) - 40))
            dp = extract_step_data(trajectory, index, cfg, EmbodimentTag.UNITREE_G1_SONIC, allow_padding=True)
            obs = {f"state.{k}": v for k, v in dp.states.items()}
            obs.update({f"video.{k}": np.asarray(v) for k, v in dp.images.items()})
            obs.update({k: dp.text for k in cfg["language"].modality_keys})
            torch.manual_seed(700000 + episode * 1000 + index)
            prediction = predict(obs)
            truth = np.stack(trajectory["action.visual_command"].iloc[index : index + 40])
            error = abs(prediction[: len(truth)] - truth)
            reports.append(
                dict(
                    episode=episode,
                    frame=index,
                    finite=bool(np.isfinite(prediction).all()),
                    per_dimension_mae=error.mean(0).tolist(),
                )
            )
    atomic_json(
        out,
        dict(
            samples=len(reports),
            per_dimension_mae=np.mean([r["per_dimension_mae"] for r in reports], axis=0).tolist(),
            cases=reports,
        ),
    )


def rollout(
    sim,
    policy,
    seed,
    edge,
    predict,
    out,
    *,
    replay_case=None,
    max_s=24.0,
    recovery_kind="none",
    scene_profile=LEGACY_PROFILE,
    stratum="low_near",
    stand_distance=0.375,  # the ruled expert stand (BottleVisualTaskConfig().grasp.stand_distance)
):
    bottle, _, spawn, yaw, scene = sample_reset(
        geometry,
        sim,
        np.random.default_rng(seed),
        edge,
        stratum,
        scene_profile,
        stand_d=(stand_distance, stand_distance),
        inboard=STRATA[stratum][1],
    )
    intervention = RecoveryIntervention(recovery_kind)
    if recovery_kind == "stop_short":
        # Assessment reset only, identical geometric reset rule to recovery
        # collection. Neither this estimate nor the reset label reaches VLA.
        from locomani.tasks.bottle_visual.observation import SceneEstimate
        from locomani.tasks.bottle_visual.planner import GraspConfig, RobotObservation, VisualGraspPlanner

        inward = np.zeros(3)
        inward[0 if edge[0] == "x" else 1] = 1 if edge[1] == "-" else -1
        edge_point = bottle - scene["inboard"] * inward
        edge_point[2] = sim.table_top
        estimate = SceneEstimate(tuple(bottle), sim.table_top, tuple(edge_point), tuple(inward), 0.0, 1.0, True)
        robot = RobotObservation(
            np.array([*spawn, 0.76]),
            np.array([1.0, 0, 0, 0]),
            np.zeros(3),
            np.array([1.0, 0, 0, 0]),
            np.zeros(3),
            R.GRIP_OPEN,
        )
        reset_config = GraspConfig(stand_distance=0.375, edge_clearance=0.22, high_table_front=0.20, facing="edge")
        intent = VisualGraspPlanner(reset_config).decide(estimate, robot)
        spawn, yaw = intent.stand_xy - 0.1 * inward[:2], intent.yaw
    sim.reset(spawn, yaw, 0.0)
    sim.d.qpos[sim.crate_q : sim.crate_q + 3] = bottle
    mujoco.mj_forward(sim.m, sim.d)
    sim.grip_clip = 0.6
    wrapped = AppliedPolicy(policy, sim)
    wrapped.reset(sim.state())
    refs = R.Refs(policy, sim)
    executor, scheduler = ActionExecutor(refs.hand_rel), ChunkScheduler()
    observer = RobotCameraObserver(sim)
    audit_sensor = BottleProxySensor(sim)
    full = expected = expected_commands = None
    if replay_case:
        full = np.load(replay_case / "actions.npz")["actions"]
        expected = np.load(replay_case / "trace.npz")["qpos"]
        expected_commands = json.loads((replay_case / "commands.json").read_text())
    ticks = len(full) if full is not None else int(max_s / R.DT)
    trace, actions, requests, object_trace, evidence = [], [], [], [], []
    previous_pad = sim.pad("right")
    velocity = np.zeros(3)
    stable_s = 0.0
    closed_once = False
    close_started = None
    first_move = None
    ok = False
    why = "timeout"
    state_error = command_error = 0.0
    warnings = {}
    knock_reference = bottle[:2].copy()
    initial_local_pad = None
    arm_started = False
    started = time.time()
    try:
        for k in range(ticks):
            t = k * R.DT
            if k == 84:
                initial_local_pad = R.rot_z(R.yaw_of(sim.d.qpos[3:7])).T @ (sim.pad("right") - sim.d.qpos[:3])
            if initial_local_pad is not None:
                local = R.rot_z(R.yaw_of(sim.d.qpos[3:7])).T @ (sim.pad("right") - sim.d.qpos[:3])
                arm_started = (
                    arm_started or local[0] - initial_local_pad[0] >= 0.03 or local[2] - initial_local_pad[2] >= 0.01
                )
            changed = intervention.update(sim, t, "reach" if arm_started and not closed_once else "walk")
            if changed == "object_shift":
                knock_reference = sim.crate()[0][:2].copy()
            if changed == "wrist_offset":
                previous_pad = sim.pad("right")
                velocity[:] = 0.0
            pad = sim.pad("right")
            velocity = 0.8 * velocity + 0.2 * (pad - previous_pad) / R.DT
            previous_pad = pad
            # First prediction is captured at1.60s, available at1.68s. Pending
            # replans never interrupt the live chunk. Rows skip modeled delay.
            if k >= 80 and (k - 80) % 20 == 0:
                obs, _, _ = observer.capture(t, executor.row, blackout=t < intervention.blackout_until)
                audit_sensor.capture(t)
                evidence.append(dict(audit_sensor.last_evidence, blackout=t < intervention.blackout_until))
                start = time.time()
                chunk = full[np.minimum(k + np.arange(40) * 2, len(full) - 1)] if full is not None else predict(obs)
                try:
                    scheduler.submit(k, chunk)
                except ValueError as error:
                    np.savez_compressed(out / "invalid_prediction.npz", chunk=chunk)
                    why = f"invalid predicted action: {error}"
                    break
                requests.append(dict(t=t, inference_ms=1000 * (time.time() - start)))
            row = None
            if executor.needs_action:
                row = scheduler.row(k)
                if row is None:
                    ref = dict(refs.pelvis(spawn, yaw), **refs.hands_at_side(np.array([*spawn, R.PELVIS_Z]), yaw))
                    row = encode_action(
                        refs.futures(lambda _: ref, t), sim.d.qpos[:3], sim.d.qpos[3:7], sim.grip_target[2]
                    )
            try:
                targets, grip = executor.step(sim.d.qpos[:3], sim.d.qpos[3:7], row)
            except ValueError as error:
                why = f"invalid decoded action: {error}"
                break
            sim.grip_target[2:] = grip
            if close_started is None and grip < R.GRIP_OPEN - 0.002:
                close_started = t
            closed_once = closed_once or grip < R.GRIP_OPEN - 0.002
            wrapped.phase = "settle" if k < 84 else "vla"
            sim.step(wrapped(sim.state(), targets))
            trace.append(sim.d.qpos.copy())
            actions.append(executor.row.copy())
            bp, bq = sim.crate()
            rise = float(bp[2] - bottle[2])
            tilt = float(np.degrees(np.arccos(np.clip(R.mat(bq)[2, 2], -1, 1))))
            if first_move is None and np.linalg.norm(bp[:2] - knock_reference) > 0.005:
                first_move = dict(t=t + R.DT, before_close=not closed_once)
            warnings = {mujoco.mjtWarning(j).name: int(w.number) for j, w in enumerate(sim.d.warning) if w.number}
            invalid = bool(warnings) or sim.fallen() or not np.isfinite(sim.d.qpos).all()
            knocked = first_move is not None and first_move["before_close"]
            good = (
                closed_once
                and rise >= 0.08
                and tilt <= 30
                and np.linalg.norm(velocity) < 0.06
                and not knocked
                and not invalid
                and intervention.triggered
                and not intervention.error
            )
            stable_s = stable_s + R.DT if good else 0.0
            object_trace.append(dict(t=t, rise_m=rise, tilt_deg=tilt, pad_distance_m=float(np.linalg.norm(pad - bp))))
            if full is not None:
                state_error = max(state_error, float(np.max(abs(sim.d.qpos - expected[k]))))
                command_error = max(
                    command_error,
                    float(
                        np.linalg.norm(targets["right_wrist_yaw_link"][0][0] - expected_commands[k]["wrist_position"])
                    ),
                )
            if stable_s >= 1.0:
                ok = True
                why = ""
            if full is None and (ok or invalid or knocked):
                if invalid:
                    why = "physics warning/fall"
                elif knocked:
                    why = "bottle knocked before close"
                break
    finally:
        observer.close()
        audit_sensor.close()
    np.savez_compressed(out / "trace.npz", qpos=np.asarray(trace), dt=R.DT)
    np.savez_compressed(out / "actions.npz", actions=np.asarray(actions), dt=R.DT)
    atomic_json(out / "commands.json", wrapped.rows)
    atomic_json(out / "object_audit.json", object_trace)
    atomic_json(out / "requests.json", requests)
    atomic_json(out / "observations.json", evidence)
    atomic_json(out / "recovery_events.json", intervention.events)
    close_pitch = [
        abs(r["angles"][3])
        for r in wrapped.rows
        if close_started is not None and close_started <= r["t"] < close_started + 0.8
    ]
    recent = np.asarray(trace[-51:])
    base_speed = float(np.linalg.norm(recent[-1, :2] - recent[0, :2]) / max(R.DT, (len(recent) - 1) * R.DT))
    latencies = [r["inference_ms"] for r in requests]
    result = dict(
        seed=seed,
        edge=edge,
        scene=scene,
        ok=ok,
        why=why,
        len_s=len(trace) * R.DT,
        wall_s=time.time() - started,
        first_move=first_move,
        closed_once=closed_once,
        physics_warnings=warnings,
        metrics=summarize(wrapped.rows),
        min_pad_distance_m=min(r["pad_distance_m"] for r in object_trace),
        recovery_kind=recovery_kind,
        recovery_triggered=intervention.triggered,
        recovery_error=intervention.error,
        close_pitch_median_deg=float(np.median(close_pitch)) if close_pitch else None,
        final_base_speed_m_s=base_speed,
        stationary_without_close=base_speed < 0.02 and not closed_once,
        inference_median_ms=float(np.median(latencies)) if latencies else None,
        inference_p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
        visibility_fraction=float(
            np.mean(
                [
                    not r["blackout"] and any(p >= 32 and f >= 0.2 for p, f in zip(r["pixels"], r["fractions"]))
                    for r in evidence
                ]
            )
        )
        if evidence
        else 0.0,
    )
    if full is not None:
        result.update(
            qpos_max_abs_error=state_error,
            wrist_max_error_m=command_error,
            exact=state_error <= 1e-6 and command_error <= 1e-6,
            replay_horizon="original recording duration; no extra recovery time",
        )
    atomic_json(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model")
    mode.add_argument("--replay-run")
    parser.add_argument("--replay-perturbation", choices=["none", "wrist_offset"], default="none")
    parser.add_argument("--replay-limit", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="21001:21061", help="Half-open start:end")
    parser.add_argument("--policy-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--validation-dataset")
    parser.add_argument(
        "--recovery-episodes",
        type=int,
        choices=[0, 12],
        default=12,
        help="Use0 for an ordinary-reset100-episode evaluation; old registered protocol uses12",
    )
    parser.add_argument(
        "--scene-profile",
        choices=PROFILES,
        default=LEGACY_PROFILE,
        help="Old registered evaluation by default; use yplus_center_v1 for new collections",
    )
    args = parser.parse_args()
    if args.replay_limit and not args.replay_run:
        parser.error("--replay-limit applies only to contract probes")
    torch.set_num_threads(1)
    backend = ScaleBfmBackendConfig()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | dict(
        torch=torch.__version__,
        mujoco=mujoco.__version__,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in line_sources()},
    )
    config["runtime_assets"] = verify_runtime_assets()
    saved = out / "config.json"
    if saved.exists() and json.loads(saved.read_text()) != config:
        raise ValueError("Config changed; use a new evaluation ID")
    atomic_json(saved, config)
    policy = R.Policy(device=args.policy_device, backend=backend)
    predict = load_vla(args.model) if args.model else None
    if args.validation_dataset and not (out / "validation.json").exists():
        validation_diagnostic(predict, args.validation_dataset, out / "validation.json")
    if args.replay_run:
        cases = []
        for case in sorted(Path(args.replay_run).glob("seed*")):
            if (case / "result.json").exists():
                r = json.loads((case / "result.json").read_text())
                if r.get("recovery_kind", "none") != "none":
                    raise ValueError(
                        "Replay source must have an ordinary reset; use --replay-perturbation for a controlled intervention"
                    )
                cases.append(
                    (
                        r["seed"],
                        r["edge"],
                        r["stratum"],
                        r["scene_directory"],
                        case,
                        args.replay_perturbation,
                        r.get("scene_profile", LEGACY_PROFILE),
                    )
                )
        if args.replay_limit:
            cases = cases[: args.replay_limit]
    else:
        start, end = map(int, args.seeds.split(":"))
        strata = ["high_near", "low_near", "low_deep"]
        count = end - start
        if count not in (60, 100):
            raise ValueError("Formal selection/confirmation must contain60/100 cases")
        recovery_kinds = ["stop_short", "wrist_offset", "blackout", "object_shift"]
        edges = profile_edges(args.scene_profile)
        cases = [
            (
                seed,
                edges[(i // 3 + i % 3) % len(edges)],
                strata[i % 3],
                None,
                None,
                "none"
                if i < count - args.recovery_episodes
                else recovery_kinds[(i - count + args.recovery_episodes) // 3],
                args.scene_profile,
            )
            for i, seed in enumerate(range(start, end))
        ]
    results = []
    environments = {}
    for seed, edge, stratum, directory, replay_case, recovery_kind, scene_profile in cases:
        dest = out / f"seed{seed}_{edge}"
        dest.mkdir(exist_ok=True)
        if (dest / "result.json").exists():
            result = json.loads((dest / "result.json").read_text())
        else:
            directory = directory or scene_directory(out / "scene_assets", stratum, backend)
            if stratum not in environments:
                environments[stratum] = make_sim(policy, directory, backend)
            # Same policy-noise stream for each paired scene, independent of arm
            # and training seed; model stochasticity is not episode-order driven.
            torch.manual_seed(900000 + seed)
            result = rollout(
                environments[stratum],
                policy,
                seed,
                edge,
                predict,
                dest,
                replay_case=replay_case,
                recovery_kind=recovery_kind,
                scene_profile=scene_profile,
                stratum=stratum,
            )
            result.update(stratum=stratum, scene_directory=str(directory), scene_profile=scene_profile)
            atomic_json(dest / "result.json", result)
        results.append(result)
        print(json.dumps(result), flush=True)
    atomic_json(
        out / "summary.json",
        dict(
            count=len(results),
            successes=sum(r["ok"] for r in results),
            exact_count=sum(r.get("exact", False) for r in results),
            cases=results,
        ),
    )


if __name__ == "__main__":
    main()
