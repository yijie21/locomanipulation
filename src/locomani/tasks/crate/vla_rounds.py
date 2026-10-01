"""Closed-loop VLA rounds for the crate: the bimanual counterpart of vla_round.py.

Runs inside a token-only stack (NO_ENCODER=1) that eval_crate_closed_loop.py
started, one call of `run` per batch in the evaluator's process, like the visual expert. Each round:

  1. placement -- the visual expert's own sampling (`expert.start_placement`:
     its start pose, object jitter and yaw, along the scene's walk-in heading),
     through the episode control client;
  2. token prelude -- initial_pose.prepare_token_pose with the crate token file
     (first-frame token and arm posture of the training data), so the policy is
     handed over from the posture its data starts in;
  3. blocking chunk loop -- observation (three cameras, 60-d state) -> policy
     chunk (token [40,64], left/right gripper [40]) -> `execution_horizon`
     frames published with both hands;
  4. verdict -- the crate expert's own criterion (crate/evaluation.evaluate):
     rise, tilt and BOTH hands touching, sustained for two seconds; the
     bottle line's walk / reach / close failure strings otherwise. With
     `visual.retreat_distance` set (the warehouse), a lifted round keeps
     running and records whether the policy also walks that far back with the
     load held -- the rest of the collection's success -- as `retreat`.

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
from locomani.backends.sonic.sonic_token import expand_hands_bimanual, hand_from_gripper
from locomani.backends.sonic.vla_config import NEAR_M
from locomani.backends.sonic.vla_obs import ObsBuilder, PolicyLink
from locomani.backends.sonic.vla_round import latency_index
from locomani.schema import SONIC_CRATE
from locomani.sim.privileged_pub import EpisodeControlClient
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.evaluation import evaluate
from locomani.tasks.crate.expert import start_placement
from locomani.tasks.crate.grasp_region import target_key
from locomani.tasks.crate.retreat import CONTACT_GAP, SETTLE
from locomani.tasks.crate.visual_approach import approach_heading

CLOSE_THRESHOLD = 0.005  # a commanded jaw under this is a close (vla_round.py's rule)
LIFT_HOLD_S = 2.0


def posture(st) -> dict:
    """Pelvis height and torso pitch, the two channels this trace never had.

    The single-arm trace has carried both since the bottle line existed
    (`vla_client.py`); the crate trace recorded only `pelvis_xy`/`pelvis_yaw`/
    `fallen`, so the one question repeatedly asked of it -- did the robot lurch or
    fall at the start of a round -- could not be answered from its own artifacts.
    On 2026-09-18 it took a separate probe script and frame differencing to
    establish that the opening transient is a yaw step, not a body collapse.
    """
    pelvis = st.get("pelvis") or {}
    position, quaternion = pelvis.get("pos"), pelvis.get("quat")
    pitch = None
    if quaternion is not None:
        w, x, y, z = (float(v) for v in quaternion)
        # pitch about the body y axis, +ve = leaning forward (same form as vla_client)
        pitch = float(np.arcsin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))))
    return dict(
        pelvis_z=None if position is None else float(position[2]),
        torso_pitch=pitch,
    )


def classify(min_dist: float, commanded_close: bool, moved_m: float) -> str:
    """The bottle line's failure strings, so the two lines aggregate alike."""
    if min_dist > NEAR_M:
        return f"[vla] walk: never closer than {min_dist:.2f} m"
    if not commanded_close:
        return f"[vla] reach: at the table (min {min_dist:.2f} m) but never commanded a close"
    return f"[vla] close: commanded a grasp but no lift (object moved {moved_m * 100:.0f} cm laterally)"


def takeover_sample(p, st) -> dict:
    """Where the body is, in the four numbers a handover transient moves."""
    return dict(
        yaw_deg=float(np.degrees(p.pelvis_yaw(st))),
        pelvis_xy=[float(v) for v in np.asarray(p.pelvis_xy(st))],
        **posture(st),
    )


def place(p, reset, meta, cfg, rng):
    """The collection expert's placement without its planner-arm ready pose (the token prelude owns the arms).

    `expert.start_placement` itself, so the rounds start where the training episodes did: until 2026-09-27
    this was a copy of the synthetic-table sampling (robot yaw 0, object yaw pi/2, `blue_crate`), which in
    the logistics room faced the robot 90 deg off the walk-in and never moved the box (the sim skips an
    unknown object name)."""
    placement, record = start_placement(reset, meta, cfg, rng)
    heading = approach_heading(meta)
    p.cmd.reset_pose()
    p.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)
    reset.pin_arms()
    placement("vla_round")
    time.sleep(0.4)
    # Third sample, and the reason for it: `held` is taken after this function returns,
    # so spawn->held is the settle and held->streamed is the handover. Without the split
    # the 6 cm measured on 2026-09-18 could not be charged to either one, and the first
    # guess -- that it was the handover -- cost an A/B that made the clean rounds worse.
    spawn = takeover_sample(p, p.obs())
    p.cmd.start_control()
    p.hold_yaw = heading
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE, facing=(float(np.cos(heading)), float(np.sin(heading)), 0.0))
    return dict(record, takeover=dict(spawn=spawn))


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
    # Three samples across the handover, as the bottle gym's win pinned/held/video.
    # The recorded round starts after `enter_streamed`, so a transient at the switch
    # sits before the trace's first row and before the video's first frame -- which is
    # how a systematic yaw step of a few degrees went unattributed until 2026-09-18.
    placement["takeover"]["held"] = takeover_sample(p, initial)
    delivered = initial_pose.prepare_token_pose(p, expected_arms)
    placement["takeover"]["streamed"] = takeover_sample(p, p.obs())
    if not delivered:
        return placement, "[stand] token-only ready posture not delivered"
    return placement, None


def warm_up(p, reset, meta, cfg, rng) -> None:
    """One full prelude before the first recorded round, so no recorded round is the deploy's first.

    Measured 2026-09-17 on the crate scene (outputs/twitch_body/*): until the
    deploy's FIRST external token arrives it stands in a posture nobody chose --
    knee 0.67, pelvis 0.755 m, elbows -0.29. That is not the reference motion's
    frame 0 (near-zero joints at 0.790 m) and not the reset pose
    (DEFAULT_DOF_ANGLES, knee 0.30): the evaluation stack launches with
    NO_ENCODER=1, so `GatherTokenState` (deploy.cpp:1662) feeds the policy
    `token_state_data_`, still an all-zero buffer, and the crouch is simply what
    the policy does with a zero token. With the encoder on, the same IDLE hold
    stands at knee 0.16 / pelvis 0.787. The first token then fills that buffer
    for good (is_using_encoder_ never turns back on), which is why the step to
    the token posture (knee 0.17, pelvis 0.788, elbows +0.83) happens once, at
    the mode switch:
    0.5 rad of knee, 3.3 cm of height, 7.5 deg of pitch and 9 rad/s at the elbow
    inside 0.3 s. That is the lurch at the start of every evaluation video.
    From the second round on, the identical sequence (reset, start_control,
    IDLE hold, enter_streamed) is transient-free: pitch within 0.1 deg, arms
    0.00 rad/s -- the buffer still holds the last token, so the IDLE hold now
    settles at the token posture instead of the zero-token crouch.

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


class RetreatWatch:
    """After the lift verdict: does the policy walk `retreat_distance` back along the walk-in with the load held?

    The collection's success (`tasks/crate/retreat.retreat`, where the expert walks back on the planner) scored
    here on the policy's own motion. Checked once per chunk, the only times this loop reads the state, so a
    contact gap is counted from the first check without both hands to the next one that has them again.
    """

    def __init__(self, state, heading, visual):
        self.axis = np.array([np.cos(heading), np.sin(heading)])
        self.start_xy = np.asarray(state["pelvis"]["pos"][:2], dtype=float)
        self.t0 = float(state["t"])
        self.distance, self.timeout = visual.retreat_distance, visual.retreat_timeout
        self.floor = 0.6 * visual.lift_height  # rise above the start, as `retreat` and `evaluate` count held
        self.walked, self.lost_since, self.settle_from = 0.0, None, None
        self.min_rise, self.reason, self.seconds = float("inf"), "budget", 0.0

    def update(self, state, rise, touching) -> bool:
        """One check; True when the retreat is decided (its `reason` is then set)."""
        now = float(state["t"])
        self.seconds = now - self.t0
        self.min_rise = min(self.min_rise, rise)
        self.walked = float((self.start_xy - np.asarray(state["pelvis"]["pos"][:2])) @ self.axis)
        held = rise >= self.floor and {"left", "right"} <= set(touching)
        self.lost_since = None if held else (now if self.lost_since is None else self.lost_since)
        if state["fallen"]:
            self.reason = "fallen"
        elif self.lost_since is not None and now - self.lost_since >= CONTACT_GAP:
            self.reason = "dropped"
        elif self.settle_from is None and self.seconds > self.timeout:
            self.reason = "timeout"
        else:
            if self.settle_from is None and self.walked >= self.distance:
                self.settle_from = now
            if self.settle_from is None or now - self.settle_from < SETTLE:
                return False
            self.reason = "completed"
        return True

    def record(self) -> dict:
        return dict(
            reason=self.reason,
            walked=round(self.walked, 3),
            distance_target=self.distance,
            seconds=round(self.seconds, 2),
            min_rise=round(self.min_rise, 3),
        )


def run_round(p, reset, obs_src, link, meta, cfg, rng, *, task, budget_s, execution_horizon, dataset_fps, trace_path):
    placement, why = prelude(p, reset, meta, cfg, rng)
    if why is not None:
        return dict(ok=False, why=why, placement=placement)
    if not obs_src.wait_ready(10.0):
        return dict(ok=False, why="[vla] camera/state streams never became ready", placement=placement)

    dt = 1.0 / dataset_fps
    key = target_key(meta)
    low = cfg.visual.close_target  # the closed end the expert commanded, which the policy was trained on
    init_tok = initial_pose.initial_token()
    initial = p.obs()
    obj0 = np.asarray(p.object_pos(key, initial))
    obs = obs_src.build()
    if obs is None:
        return dict(ok=False, why="[vla] observation stream dropped", placement=placement)
    tok0, left0, right0 = link.get_chunk_bimanual(obs)
    l0, r0 = expand_hands_bimanual(left0[0], right0[0], low)
    n_bl = max(1, int(1.0 * dataset_fps))
    for i in range(n_bl):  # the 1 s blend from the standing token into chunk 0 (vla_round.py)
        a = (i + 1) / n_bl
        p.cmd.token_frame((1.0 - a) * init_tok + a * np.asarray(tok0[0]), right_hand=r0, left_hand=l0, frame_index=0)
        time.sleep(dt)

    t_start = time.time()
    frame_index, commanded_close, lifted_at, min_dist = 0, False, None, float("inf")
    ok, why, trace, final = False, "[vla] timeout", [], initial
    retreat = None  # RetreatWatch once the lift verdict has held, when the config asks for a retreat
    try:
        while time.time() - t_start < budget_s:
            t_obs = time.time()
            obs = obs_src.build()
            if obs is None:
                why = "[vla] observation stream dropped"
                break
            image = obs_src.image_meta  # the Isaac mirror's reply header (seq, state age), None on the MuJoCo stream
            tok, left, right = link.get_chunk_bimanual(obs)
            latency = time.time() - t_obs
            skip = latency_index(latency, dataset_fps, len(tok))
            n = max(1, min(execution_horizon, len(tok) - skip))
            for j in range(skip, skip + n):
                lhj, rhj = hand_from_gripper(left[j], low), hand_from_gripper(right[j], low)
                if left[j] < CLOSE_THRESHOLD or right[j] < CLOSE_THRESHOLD:
                    commanded_close = True
                p.cmd.token_frame(tok[j], right_hand=rhj, left_hand=lhj, frame_index=frame_index)
                frame_index += 1
                time.sleep(dt)
            st = p.obs()
            final = st
            obj = np.asarray(p.object_pos(key, st))
            pel = np.asarray(p.pelvis_xy(st))
            rise = float(obj[2] - obj0[2])
            dist = float(np.linalg.norm(obj[:2] - pel))
            min_dist = min(min_dist, dist)
            verdict = evaluate(
                initial,
                st,
                "completed",
                lift_height=cfg.visual.lift_height,
                max_tilt_deg=cfg.visual.max_tilt_deg,
                key=key,
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
                    latency_s=round(latency, 4),
                    skip=skip,
                    image=None
                    if image is None
                    else {k: image.get(k) for k in ("seq", "t", "state_age_s", "render_ms")},
                    **posture(st),
                )
            )
            if retreat is not None:
                if retreat.update(st, rise, verdict["touching"]):
                    break
                continue
            if st["fallen"]:
                why = "[vla] fell"
                break
            if verdict["ok"]:
                if lifted_at is None:
                    lifted_at = time.time()
                elif time.time() - lifted_at > LIFT_HOLD_S:
                    ok, why = True, "ok"
                    if cfg.visual.retreat_distance <= 0:
                        break
                    retreat = RetreatWatch(st, approach_heading(meta), cfg.visual)
            else:
                lifted_at = None
    finally:
        p.cmd.exit_streamed()
        p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    if not ok and why == "[vla] timeout":
        moved = float(np.linalg.norm(np.asarray(p.object_pos(key, final))[:2] - obj0[:2]))
        why = classify(min_dist, commanded_close, moved)
    verdict = evaluate(
        initial,
        final,
        "completed" if ok else "failed",
        lift_height=cfg.visual.lift_height,
        max_tilt_deg=cfg.visual.max_tilt_deg,
        key=key,
    )
    retreat_record = None if retreat is None else retreat.record()
    if trace_path is not None:
        trace_path.write_text(json.dumps(dict(ok=ok, why=why, retreat=retreat_record, trace=trace)))
    return dict(
        ok=ok,
        why=why,
        retreat=retreat_record,
        retreat_ok=None if retreat is None else retreat_record["reason"] == "completed",
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
    ap.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="leave streamed mode between rounds, so the settle runs on a drifting planner IDLE. "
        "The pre-2026-09-18 behaviour, kept only to reproduce a historical number",
    )
    ap.set_defaults(clean=True)
    ap.add_argument(
        "--round-offset",
        type=int,
        default=0,
        help="number of rounds already run in this evaluation. The batch loop calls `run` once per "
        "batch, each numbering its rounds from zero, so without this every batch overwrote the "
        "previous batch's traces: a 100-round evaluation kept 10 of them",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--policy-port", type=int, default=5700)
    ap.add_argument("--execution-horizon", type=int, default=16)
    ap.add_argument("--budget-s", type=float, default=90.0)
    ap.add_argument("--dataset-fps", type=float, default=25.0)
    ap.add_argument("--initial-token-file", required=True)
    ap.add_argument("--trace-dir", type=Path)
    ap.add_argument(
        "--image-source",
        choices=("mujoco", "isaac"),
        default="mujoco",
        help="the policy's camera images: the sim's MuJoCo stream, or the Isaac mirror (EvalConfig.image_source)",
    )
    args = ap.parse_args(argv)
    if args.rounds < 1:
        ap.error("--rounds must be positive")
    run(
        CrateTaskConfig.load(args.config) if args.config else CrateTaskConfig(),
        rounds=args.rounds,
        seed=args.seed,
        manifest=args.manifest,
        initial_token_file=args.initial_token_file,
        policy_port=args.policy_port,
        execution_horizon=args.execution_horizon,
        budget_s=args.budget_s,
        dataset_fps=args.dataset_fps,
        trace_dir=args.trace_dir,
        image_source=args.image_source,
        clean=args.clean,
        round_offset=args.round_offset,
    )


def run(
    config: CrateTaskConfig,
    *,
    rounds: int,
    seed: int,
    manifest: Path,
    initial_token_file,
    policy_port: int = 5700,
    execution_horizon: int = 16,
    budget_s: float = 90.0,
    dataset_fps: float = 25.0,
    trace_dir: Path | None = None,
    image_source: str = "mujoco",
    clean: bool = True,
    round_offset: int = 0,
) -> None:
    """One batch of rounds on the stack this process has adopted (`run_context.adopt_env`).

    The evaluator calls this once per batch in its own process (2026-10-01), so what is opened here is closed
    here: a clean clock left running would keep publishing into the next batch's restarted deploy.
    """
    initial_pose.INIT_TOKEN_FILE = str(initial_token_file)
    meta = json.loads(Path(paths.SCENE_META).read_text())
    link = PolicyLink(port=policy_port)
    if not link.ping():
        raise SystemExit(f"no policy server on :{policy_port}")
    modalities = link.identity()["modalities"]
    for kind in ("video", "state", "action"):
        if list(modalities[kind]) != list(getattr(SONIC_CRATE, f"{kind}_keys")):
            raise SystemExit(
                f"policy server {kind} schema differs from the crate observation contract: {modalities[kind]}"
            )
    clock = None
    if clean:
        # Keep the deploy in streamed mode for the whole batch, on a standing token.
        # Without this every round ends with `exit_streamed()` -- a planner command --
        # so the NEXT round's 3.4 s settle runs in planner mode, where IDLE drifts at
        # 8-24 cm/s. Measured 2026-09-18 over 30 rounds: the settle moved the base up
        # to 0.54 m, and the only clean rounds in the run were the three that followed
        # `warm_up`, which never leaves streamed mode. The single-arm line has had this
        # since the clean-deploy work; this line never did.
        from dataclasses import replace as dataclass_replace

        from locomani.backends.sonic.clean_clock import CleanClock, clean_commander_class
        from locomani.backends.sonic.vla_config import DEFAULT_VLA

        vla_cfg = dataclass_replace(
            DEFAULT_VLA,
            backend=dataclass_replace(DEFAULT_VLA.backend, dataset_fps=25.0, clean_deploy=True),
        )
        clock = CleanClock(config=vla_cfg)  # reads INIT_TOKEN_FILE, set above
        p = Primitives(commander=clean_commander_class(clock)())
        clock.attach(p.cmd)
        clock.prime()
        clock.resume()
    else:
        p = Primitives()
    reset = EpisodeControlClient()
    obs_src = ObsBuilder(config.prompt, schema=SONIC_CRATE, image_source=image_source)
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        warm_up(p, reset, meta, config, np.random.default_rng([seed, 0xB1A5]))
        for index in range(rounds):
            round_seed = seed + index
            started = time.time()
            # the sim's overview mp4 under SIM_VIDEO_DIR and its state trace; numbered across batches like the traces
            label = f"crate_vla_{index + round_offset:03d}"
            reset.start_video(label)
            result = run_round(
                p,
                reset,
                obs_src,
                link,
                meta,
                config,
                np.random.default_rng(round_seed),
                task=config.prompt,
                budget_s=budget_s,
                execution_horizon=execution_horizon,
                dataset_fps=dataset_fps,
                trace_path=None if trace_dir is None else trace_dir / f"round_{index + round_offset:03d}.json",
            )
            reset.end_video()
            row = dict(
                round=index,
                seed=round_seed,
                task=config.prompt,
                target=target_key(meta),
                video=label,
                wall_seconds=time.time() - started,
                **result,
            )
            with manifest.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"[crate-vla] round {index}: {'OK' if row['ok'] else 'FAIL'} {row['why']} ({row['wall_seconds']:.0f} s)",
                flush=True,
            )
    finally:
        obs_src.close()
        if clock is not None:
            clock.stop()
        reset.close()
        p.close()
        link.sock.close(linger=0)
        link.ctx.term()


if __name__ == "__main__":
    main()
