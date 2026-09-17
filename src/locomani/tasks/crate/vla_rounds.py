"""Closed-loop VLA rounds for the crate: the bimanual counterpart of vla_round.py.

Runs inside a token-only stack (NO_ENCODER=1) that eval_crate_closed_loop.py
started, one process per batch, exactly like the visual expert. Each round:

  1. placement -- the visual expert's sampling (reset_ready's jitter, yaw and
     table-edge distance), through the episode control client;
  2. token prelude -- initial_pose.prepare_token_pose with the crate token file
     (first-frame token and arm posture of the training data), so the policy is
     handed over from the posture its data starts in;
  3. blocking chunk loop -- observation (three cameras, 60-d state) -> policy
     chunk (token [40,64], left/right gripper [40]) -> `execution_horizon`
     frames published with both hands;
  4. verdict -- the crate expert's own criterion (crate/evaluation.evaluate):
     rise, tilt and BOTH hands touching, sustained for two seconds; the
     bottle line's walk / reach / close failure strings otherwise.

No privileged geometry enters the action: the policy sees only the cameras and
proprioception; `p.obs()` is used for placement, the verdict and the trace.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from locomani import paths
from locomani.backends.sonic import initial_pose
from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode
from locomani.backends.sonic.primitives import Primitives
from locomani.backends.sonic.sonic_token import GRIPPER_MAX, expand_hands_bimanual, hand_from_gripper
from locomani.backends.sonic.vla_config import NEAR_M
from locomani.backends.sonic.vla_obs import ObsBuilder, PolicyLink
from locomani.backends.sonic.vla_round import latency_index
from locomani.schema import SONIC_CRATE
from locomani.sim.privileged_pub import EpisodeControlClient
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.evaluation import evaluate
from locomani.tasks.crate.visual_approach import initial_robot_xy

CLOSE_THRESHOLD = 0.005  # a commanded jaw under this is a close (vla_round.py's rule)
LIFT_HOLD_S = 2.0


def classify(min_dist: float, commanded_close: bool, moved_m: float) -> str:
    """The bottle line's failure strings, so the two lines aggregate alike."""
    if min_dist > NEAR_M:
        return f"[vla] walk: never closer than {min_dist:.2f} m"
    if not commanded_close:
        return f"[vla] reach: at the table (min {min_dist:.2f} m) but never commanded a close"
    return f"[vla] close: commanded a grasp but no lift (object moved {moved_m * 100:.0f} cm laterally)"


def place(p, reset, meta, cfg, rng):
    """reset_ready's sampling without its planner-arm ready pose (the token prelude owns the arms)."""
    jitter = rng.uniform(-cfg.position_jitter, cfg.position_jitter, size=2)
    yaw = np.pi / 2 + np.radians(rng.uniform(-cfg.yaw_jitter_deg, cfg.yaw_jitter_deg))
    start_distance = float(rng.uniform(cfg.start_edge_min, cfg.start_edge_max))
    start_xy = initial_robot_xy(meta, start_distance)
    p.cmd.reset_pose()
    p.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)
    reset.pin_arms()
    reset.reset(
        objects={"blue_crate": {"dxy": jitter.tolist(), "yaw": float(yaw)}},
        robot={"dxy": (start_xy - meta["robot_start"]["xy"]).tolist(), "yaw": 0.0},
    )
    time.sleep(0.4)
    p.cmd.start_control()
    p.hold_yaw = 0.0
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE, facing=(1, 0, 0))
    return dict(
        object_dxy=jitter.tolist(),
        object_yaw=float(yaw),
        start_edge_distance=start_distance,
        start_xy=start_xy.tolist(),
    )


def prelude(p, reset, meta, cfg, rng):
    """Placement, reset and the token posture: everything a round does before the policy is asked.

    Returns (placement, why); `why` is None when the robot stands in the token posture.
    """
    placement = place(p, reset, meta, cfg.visual, rng)
    initial = p.obs()
    if float(initial["pelvis"]["pos"][2]) < 0.5 or initial["fallen"]:
        return placement, "[stand] robot did not stabilise after reset"
    expected_arms = initial_pose.initial_arms()
    if expected_arms is None:
        return placement, "[stand] crate token file declares no arms"
    if not initial_pose.prepare_token_pose(p, expected_arms):
        return placement, "[stand] token-only ready posture not delivered"
    return placement, None


def warm_up(p, reset, meta, cfg, rng) -> None:
    """One full prelude before the first recorded round, so no recorded round is the deploy's first.

    Measured 2026-09-17 on the crate scene (outputs/twitch_body/*): the deploy's
    FIRST round in a process visits its reference motion's frame-0 posture --
    crouched, knee 0.67, pelvis 0.755 m, elbows -0.29 -- and then steps to the
    token posture (knee 0.17, pelvis 0.788, elbows +0.83) at the mode switch:
    0.5 rad of knee, 3.3 cm of height, 7.5 deg of pitch and 9 rad/s at the elbow
    inside 0.3 s. That is the lurch at the start of every evaluation video.
    From the second round on, the identical sequence (reset, start_control,
    IDLE hold, enter_streamed) is transient-free: pitch within 0.1 deg, arms
    0.00 rad/s. Once the deploy has streamed a token it no longer returns to
    frame 0, whichever direction the mode switches.

    What does not work, each measured: pinning the arms on the sim covers the
    arms only (knee still 0.31->0.45->0.17, pitch to -9 deg); launching with
    the encoder moves the start point to standing, but the switch's safety
    reset (zmq_manager.hpp TriggerSafetyReset on every mode change, no bypass)
    still plays frame 0 for a few ticks (pitch -13.6..+4.7 deg, arms 26 rad/s);
    switching during the landing stacks the two transients (pitch +10.7 deg,
    knee 1.02); a bare start_control + token before any reset does not take.

    So the batch runs one unrecorded prelude first. It costs ~10 s per batch.
    The placement draws from its own generator, so the recorded rounds keep
    the seeds the manifest promises.
    """
    _, why = prelude(p, reset, meta, cfg, rng)
    print(f"[vla] warm-up round: {'ok' if why is None else why}", flush=True)


def run_round(p, reset, obs_src, link, meta, cfg, rng, *, task, budget_s, execution_horizon, dataset_fps, trace_path):
    placement, why = prelude(p, reset, meta, cfg, rng)
    if why is not None:
        return dict(ok=False, why=why, placement=placement)
    if not obs_src.wait_ready(10.0):
        return dict(ok=False, why="[vla] camera/state streams never became ready", placement=placement)

    dt = 1.0 / dataset_fps
    init_tok = initial_pose.initial_token()
    l_open, r_open = expand_hands_bimanual(GRIPPER_MAX, GRIPPER_MAX)
    initial = p.obs()
    obj0 = np.asarray(p.object_pos("blue_crate", initial))
    obs = obs_src.build()
    if obs is None:
        return dict(ok=False, why="[vla] observation stream dropped", placement=placement)
    tok0, left0, right0 = link.get_chunk_bimanual(obs)
    l0, r0 = expand_hands_bimanual(left0[0], right0[0])
    n_bl = max(1, int(1.0 * dataset_fps))
    for i in range(n_bl):  # the 1 s blend from the standing token into chunk 0 (vla_round.py)
        a = (i + 1) / n_bl
        p.cmd.token_frame((1.0 - a) * init_tok + a * np.asarray(tok0[0]), right_hand=r0, left_hand=l0, frame_index=0)
        time.sleep(dt)

    t_start = time.time()
    frame_index, commanded_close, lifted_at, min_dist = 0, False, None, float("inf")
    ok, why, trace, final = False, "[vla] timeout", [], initial
    try:
        while time.time() - t_start < budget_s:
            t_obs = time.time()
            obs = obs_src.build()
            if obs is None:
                why = "[vla] observation stream dropped"
                break
            tok, left, right = link.get_chunk_bimanual(obs)
            skip = latency_index(time.time() - t_obs, dataset_fps, len(tok))
            n = max(1, min(execution_horizon, len(tok) - skip))
            for j in range(skip, skip + n):
                lhj, rhj = hand_from_gripper(left[j]), hand_from_gripper(right[j])
                if left[j] < CLOSE_THRESHOLD or right[j] < CLOSE_THRESHOLD:
                    commanded_close = True
                p.cmd.token_frame(tok[j], right_hand=rhj, left_hand=lhj, frame_index=frame_index)
                frame_index += 1
                time.sleep(dt)
            st = p.obs()
            final = st
            obj = np.asarray(p.object_pos("blue_crate", st))
            pel = np.asarray(p.pelvis_xy(st))
            rise = float(obj[2] - obj0[2])
            dist = float(np.linalg.norm(obj[:2] - pel))
            min_dist = min(min_dist, dist)
            verdict = evaluate(
                initial, st, "completed", lift_height=cfg.visual.lift_height, max_tilt_deg=cfg.visual.max_tilt_deg
            )
            trace.append(
                dict(
                    t=time.time() - t_start,
                    rise=rise,
                    dist=dist,
                    tilt_deg=verdict["tilt_deg"],
                    touching=verdict["touching"],
                    fallen=bool(st["fallen"]),
                    left_grip=float(left[skip : skip + n].min()),
                    right_grip=float(right[skip : skip + n].min()),
                    pelvis_xy=[float(v) for v in pel],
                    pelvis_yaw=float(p.pelvis_yaw(st)),
                    obj_xy=[float(v) for v in obj[:2]],
                )
            )
            if st["fallen"]:
                why = "[vla] fell"
                break
            if verdict["ok"]:
                if lifted_at is None:
                    lifted_at = time.time()
                elif time.time() - lifted_at > LIFT_HOLD_S:
                    ok, why = True, "ok"
                    break
            else:
                lifted_at = None
    finally:
        p.cmd.exit_streamed()
        p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    if not ok and why == "[vla] timeout":
        moved = float(np.linalg.norm(np.asarray(p.object_pos("blue_crate", final))[:2] - obj0[:2]))
        why = classify(min_dist, commanded_close, moved)
    verdict = evaluate(
        initial,
        final,
        "completed" if ok else "failed",
        lift_height=cfg.visual.lift_height,
        max_tilt_deg=cfg.visual.max_tilt_deg,
    )
    if trace_path is not None:
        trace_path.write_text(json.dumps(dict(ok=ok, why=why, trace=trace)))
    return dict(
        ok=ok,
        why=why,
        placement=placement,
        rise=verdict["rise"],
        tilt_deg=verdict["tilt_deg"],
        touching=verdict["touching"],
        min_dist=min_dist,
        commanded_close=commanded_close,
        seconds=time.time() - t_start,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--policy-port", type=int, default=5700)
    ap.add_argument("--execution-horizon", type=int, default=16)
    ap.add_argument("--budget-s", type=float, default=90.0)
    ap.add_argument("--dataset-fps", type=float, default=25.0)
    ap.add_argument("--initial-token-file", required=True)
    ap.add_argument("--trace-dir", type=Path)
    args = ap.parse_args(argv)
    if args.rounds < 1:
        ap.error("--rounds must be positive")
    initial_pose.INIT_TOKEN_FILE = str(args.initial_token_file)
    config = CrateTaskConfig.load(args.config) if args.config else CrateTaskConfig()
    meta = json.loads(Path(paths.SCENE_META).read_text())
    link = PolicyLink(port=args.policy_port)
    if not link.ping():
        raise SystemExit(f"no policy server on :{args.policy_port}")
    modalities = link.identity()["modalities"]
    for kind in ("video", "state", "action"):
        if list(modalities[kind]) != list(getattr(SONIC_CRATE, f"{kind}_keys")):
            raise SystemExit(
                f"policy server {kind} schema differs from the crate observation contract: {modalities[kind]}"
            )
    p, reset = Primitives(), EpisodeControlClient()
    obs_src = ObsBuilder(config.prompt, schema=SONIC_CRATE)
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        warm_up(p, reset, meta, config, np.random.default_rng([args.seed, 0xB1A5]))
        for index in range(args.rounds):
            seed = args.seed + index
            started = time.time()
            reset.start_video(f"crate_vla_{index:03d}")  # the sim's overview mp4 under SIM_VIDEO_DIR
            result = run_round(
                p,
                reset,
                obs_src,
                link,
                meta,
                config,
                np.random.default_rng(seed),
                task=config.prompt,
                budget_s=args.budget_s,
                execution_horizon=args.execution_horizon,
                dataset_fps=args.dataset_fps,
                trace_path=None if args.trace_dir is None else args.trace_dir / f"round_{index:03d}.json",
            )
            reset.end_video()
            row = dict(
                round=index,
                seed=seed,
                task=config.prompt,
                target="blue_crate",
                video=f"crate_vla_{index:03d}",
                wall_seconds=time.time() - started,
                **result,
            )
            with args.manifest.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"[crate-vla] round {index}: {'OK' if row['ok'] else 'FAIL'} {row['why']} ({row['wall_seconds']:.0f} s)",
                flush=True,
            )
    finally:
        obs_src.close()


if __name__ == "__main__":
    main()
