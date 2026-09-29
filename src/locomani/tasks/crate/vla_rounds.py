"""Closed-loop VLA rounds for the crate: the bimanual counterpart of vla_round.py.

Runs inside a token-only stack (NO_ENCODER=1) that eval_crate_closed_loop.py
started, one process per batch, exactly like the visual expert. Each round:

  1. placement -- the visual expert's own sampling (`expert.start_placement`:
     its start pose, object jitter and yaw, along the scene's walk-in heading),
     through the episode control client;
  2. token prelude -- initial_pose.prepare_token_pose with the crate token file
     (first-frame token and arm posture of the training data), so the policy is
     handed over from the posture its data starts in;
  3. blocking chunk loop -- observation (three cameras, 60-d state) -> policy
     chunk (token [40,64], left/right gripper [40]) -> `execution_horizon`
     frames published with both hands, from the frame the inference latency
     has reached. While the loop waits for a picture and a chunk, `gap`
     decides what the robot does: `continue` (default) keeps executing the
     running chunk (GapFeeder), `hold` sends nothing and the deploy clamps
     its last frame -- this line's loop until 2026-09-27;
  4. verdict -- the crate expert's own criterion (crate/evaluation.evaluate):
     rise, tilt and BOTH hands touching, sustained for two seconds; the
     bottle line's walk / reach / close failure strings otherwise. With
     `visual.retreat_distance` set (the warehouse), a lifted round keeps
     running and records whether the policy also walks that far back with the
     load held -- the rest of the collection's success -- as `retreat`.

No privileged geometry enters the action: the policy sees only the cameras and
proprioception; `p.obs()` is used for placement, the verdict and the trace.

`--client lockstep` runs the same rounds on the simulation clock instead
(`vla_lockstep.py`): the world stands still while the policy looks.
"""

from __future__ import annotations

import argparse
import json
import threading
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
from locomani.sim.mirror import MIRRORED_SOURCES
from locomani.sim.privileged_pub import EpisodeControlClient
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.evaluation import evaluate
from locomani.tasks.crate.expert import facing_edge, start_placement
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


class GapFeeder:
    """What the robot executes while the loop waits for a picture and a chunk, under `gap: continue`.

    Without it the deploy starves for the whole observation + inference (166 ms p50 with the Isaac mirror, 2026-09-27)
    and clamps its reference on the last frame it got, then the next chunk starts four frames later: the body chases a
    frozen reference and then a jump, where the training data's reference is continuous. The single-arm blocking
    client fixed this on 2026-08-31 (ledger bottle.md, "blocking 客户端三修复"); this is its feeder: a thread keeps
    publishing the running chunk's next frames at the dataset rate, holding its last, until the new chunk is in hand.
    The main thread publishes nothing meanwhile, so the commander socket has one writer at a time.
    """

    def __init__(self, p, dt: float, low: float):
        self.p, self.dt, self.low = p, dt, low
        self._stop, self._thread, self._next = None, None, 0
        self.sent = 0

    def start(self, chunk, j0: int, frame_index: int) -> None:
        """Feed `chunk` (token, left, right) from its frame `j0`, numbering from `frame_index`."""
        tok, left, right = chunk
        stop = threading.Event()
        self._next, self.sent = frame_index, 0

        def run():
            j = j0
            while not stop.is_set():
                jj = min(j, len(tok) - 1)  # past the chunk's end: hold its last frame
                lhj, rhj = hand_from_gripper(left[jj], self.low), hand_from_gripper(right[jj], self.low)
                self.p.cmd.token_frame(tok[jj], right_hand=rhj, left_hand=lhj, frame_index=self._next)
                self._next += 1
                self.sent += 1
                j += 1
                time.sleep(self.dt)

        self._stop, self._thread = stop, threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self, frame_index: int) -> int:
        """Stop feeding; the frame index the main thread continues from."""
        if self._thread is None:
            return frame_index
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None
        return self._next


class RoundScore:
    """One round's verdict, checked once per chunk: the wall-clock loop and the lockstep one (`vla_lockstep`) share it.

    `now` is whatever clock the loop runs on -- wall seconds here, simulation seconds under lockstep -- and only
    measures the lift hold; the retreat reads the state's own simulation time.
    """

    def __init__(self, p, meta, cfg, initial, now: float):
        self.p, self.meta, self.cfg = p, meta, cfg
        self.key = target_key(meta)
        self.initial, self.final, self.t_start = initial, initial, now
        self.obj0 = np.asarray(p.object_pos(self.key, initial))
        self.ok, self.why, self.trace = False, "[vla] timeout", []
        self.commanded_close, self.lifted_at, self.min_dist = False, None, float("inf")
        self.retreat = None  # RetreatWatch once the lift verdict has held, when the config asks for a retreat

    def check(self, st, now: float, **row) -> bool:
        """Score the state after one chunk; True when the round is decided. `row` joins the trace line."""
        p, cfg, key = self.p, self.cfg, self.key
        self.final = st
        obj = np.asarray(p.object_pos(key, st))
        pel = np.asarray(p.pelvis_xy(st))
        rise = float(obj[2] - self.obj0[2])
        dist = float(np.linalg.norm(obj[:2] - pel))
        self.min_dist = min(self.min_dist, dist)
        verdict = evaluate(
            self.initial,
            st,
            "completed",
            lift_height=cfg.visual.lift_height,
            max_tilt_deg=cfg.visual.max_tilt_deg,
            key=key,
        )
        self.trace.append(
            dict(
                t=now - self.t_start,
                rise=rise,
                dist=dist,
                tilt_deg=verdict["tilt_deg"],
                touching=verdict["touching"],
                fallen=bool(st["fallen"]),
                pelvis_xy=[float(v) for v in pel],
                pelvis_yaw=float(p.pelvis_yaw(st)),
                obj_xy=[float(v) for v in obj[:2]],
                **row,
                **posture(st),
            )
        )
        if self.retreat is not None:
            return self.retreat.update(st, rise, verdict["touching"])
        if st["fallen"]:
            self.why = "[vla] fell"
            return True
        if verdict["ok"]:
            if self.lifted_at is None:
                self.lifted_at = now
            elif now - self.lifted_at > LIFT_HOLD_S:
                self.ok, self.why = True, "ok"
                if cfg.visual.retreat_distance <= 0:
                    return True
                self.retreat = RetreatWatch(st, approach_heading(self.meta), cfg.visual)
        else:
            self.lifted_at = None
        return False

    def result(self, placement, seconds: float, trace_path, **extra) -> dict:
        """The manifest row's verdict fields; writes the per-chunk trace to `trace_path` when given."""
        p, cfg, key = self.p, self.cfg, self.key
        if not self.ok and self.why == "[vla] timeout":
            moved = float(np.linalg.norm(np.asarray(p.object_pos(key, self.final))[:2] - self.obj0[:2]))
            self.why = classify(self.min_dist, self.commanded_close, moved)
        verdict = evaluate(
            self.initial,
            self.final,
            "completed" if self.ok else "failed",
            lift_height=cfg.visual.lift_height,
            max_tilt_deg=cfg.visual.max_tilt_deg,
            key=key,
        )
        retreat_record = None if self.retreat is None else self.retreat.record()
        if trace_path is not None:
            trace_path.write_text(
                json.dumps(dict(ok=self.ok, why=self.why, retreat=retreat_record, **extra, trace=self.trace))
            )
        return dict(
            ok=self.ok,
            why=self.why,
            retreat=retreat_record,
            retreat_ok=None if self.retreat is None else retreat_record["reason"] == "completed",
            placement=placement,
            rise=verdict["rise"],
            tilt_deg=verdict["tilt_deg"],
            touching=verdict["touching"],
            min_dist=self.min_dist,
            commanded_close=self.commanded_close,
            seconds=seconds,
        )


def run_round(
    p,
    reset,
    obs_src,
    link,
    meta,
    cfg,
    rng,
    *,
    task,
    budget_s,
    execution_horizon,
    dataset_fps,
    trace_path,
    gap="continue",
):
    meta = facing_edge(meta, cfg.visual, rng)  # the round's walk-in, as the collection expert draws it
    placement, why = prelude(p, reset, meta, cfg, rng)
    if why is not None:
        return dict(ok=False, why=why, placement=placement)
    if not obs_src.wait_ready(10.0):
        return dict(ok=False, why="[vla] camera/state streams never became ready", placement=placement)

    dt = 1.0 / dataset_fps
    low = cfg.visual.close_target  # the closed end the expert commanded, which the policy was trained on
    init_tok = initial_pose.initial_token()
    initial = p.obs()
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
    score = RoundScore(p, meta, cfg, initial, t_start)
    frame_index = 0
    feeder = GapFeeder(p, dt, low)
    running = ((tok0, left0, right0), 0)  # the chunk being executed and its next frame: the blend ended on tok0[0]
    try:
        while time.time() - t_start < budget_s:
            if gap == "continue":
                feeder.start(running[0], running[1], frame_index)
            t_obs = time.time()
            obs = obs_src.build()
            if obs is None:
                frame_index = feeder.stop(frame_index)
                score.why = "[vla] observation stream dropped"
                break
            image = obs_src.image_meta  # the Isaac mirror's reply header (seq, state age), None on the MuJoCo stream
            tok, left, right = link.get_chunk_bimanual(obs)
            frame_index = feeder.stop(frame_index)
            latency = time.time() - t_obs
            skip = latency_index(latency, dataset_fps, len(tok))
            n = max(1, min(execution_horizon, len(tok) - skip))
            for j in range(skip, skip + n):
                lhj, rhj = hand_from_gripper(left[j], low), hand_from_gripper(right[j], low)
                if left[j] < CLOSE_THRESHOLD or right[j] < CLOSE_THRESHOLD:
                    score.commanded_close = True
                p.cmd.token_frame(tok[j], right_hand=rhj, left_hand=lhj, frame_index=frame_index)
                frame_index += 1
                time.sleep(dt)
            running = ((tok, left, right), skip + n)
            done = score.check(
                p.obs(),
                time.time(),
                left_grip=float(left[skip : skip + n].min()),
                right_grip=float(right[skip : skip + n].min()),
                latency_s=round(latency, 4),
                skip=skip,
                gap=gap,
                gap_frames=feeder.sent if gap == "continue" else 0,  # old-chunk frames executed while waiting
                image=None if image is None else {k: image.get(k) for k in ("seq", "t", "state_age_s", "render_ms")},
            )
            if done:
                break
    finally:
        feeder.stop(frame_index)  # an exception mid-gap must not leave a thread publishing after exit_streamed
        p.cmd.exit_streamed()
        p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    return score.result(placement, time.time() - t_start, trace_path)


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
        help="number of rounds already run in this evaluation. The batch loop runs one process per "
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
        choices=("mujoco", *MIRRORED_SOURCES),
        default="mujoco",
        help="the policy's camera images: the sim's MuJoCo stream, or the Isaac / Unreal mirror (EvalConfig.image_source)",
    )
    ap.add_argument(
        "--client",
        choices=("blocking", "lockstep"),
        default="blocking",
        help="blocking: this loop, on the wall clock; lockstep: vla_lockstep, on the simulation clock (a lockstep stack)",
    )
    ap.add_argument(
        "--gap",
        choices=("continue", "hold"),
        default="continue",
        help="what the robot executes while the policy looks and thinks: the running chunk's next frames, or "
        "nothing (the deploy clamps its last frame; this loop before 2026-09-27). EvalConfig.gap",
    )
    ap.add_argument("--latency-frames", type=int, help="lockstep: fixed inference latency, frames (EvalConfig)")
    ap.add_argument("--latency-dist", help="lockstep: latency distribution JSON (vla_lockstep latency-dist)")
    ap.add_argument("--camera-frames", type=int, default=0, help="lockstep: camera delay frames (EvalConfig)")
    ap.add_argument(
        "--no-policy-seed",
        action="store_true",
        help="lockstep: leave the server's diffusion noise unseeded (backend.lockstep.no_seed)",
    )
    args = ap.parse_args(argv)
    if args.rounds < 1:
        ap.error("--rounds must be positive")
    if args.client == "blocking" and (args.latency_frames is not None or args.latency_dist or args.camera_frames):
        ap.error("--latency-frames/--latency-dist/--camera-frames simulate latency on the lockstep clock only")
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
    if args.client == "lockstep":
        from locomani.tasks.crate import vla_lockstep

        vla_lockstep.run_batch(args, config, meta, link)
        return
    clock = None
    if args.clean:
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
    obs_src = ObsBuilder(config.prompt, schema=SONIC_CRATE, image_source=args.image_source)
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        warm_up(p, reset, meta, config, np.random.default_rng([args.seed, 0xB1A5]))
        for index in range(args.rounds):
            seed = args.seed + index
            started = time.time()
            # the sim's overview mp4 under SIM_VIDEO_DIR and its state trace; numbered across batches like the traces
            label = f"crate_vla_{index + args.round_offset:03d}"
            reset.start_video(label)
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
                trace_path=None
                if args.trace_dir is None
                else args.trace_dir / f"round_{index + args.round_offset:03d}.json",
                gap=args.gap,
            )
            reset.end_video()
            row = dict(
                round=index,
                seed=seed,
                task=config.prompt,
                target=target_key(meta),
                video=label,
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
