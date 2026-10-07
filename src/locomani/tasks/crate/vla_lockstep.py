"""Closed-loop VLA rounds for the crate on the simulation clock: the world stands still while the policy looks.

`vla_rounds.py --client lockstep` runs these instead of the wall-clock loop, inside a stack started with
`backend.lockstep.enabled` (sim LOCKSTEP=1, deploy --sim-lockstep, docs/archive/lockstep.md). Every tick of the
sim and the deploy happens because this process asked for it, so nothing moves between two requests. One cycle
(docs/handoff/lockstep-isaac-eval-plan-20260927.md §5; a frame is one dataset row: at 25 fps one token held two
50 Hz ticks, at 50 fps one tick):

    execute   chunk[D .. D+H-1]                  the world advances H frames; the reply names the boundary
    observe   world frozen                       deploy proprioception + images of the boundary state: the step
                                                 reply's MuJoCo cameras, or the Isaac mirror asked for that seq
    infer     world frozen                       however long the policy server takes in wall time
    gap       D frames, the old chunk continued  D = L inference frames + C camera frames (`latency_frames` /
              (gap: continue) or its last frame  `latency_dist`, `camera_frames`); `hold` is the wall-clock loop
              held (gap: hold)                   before 2026-09-27, whose deploy clamped its last frame
    adopt     the new chunk from index D         back to "execute"

So the renderer's time never enters the world (Isaac takes ~95 ms a picture, a camera nearly none), while the
inference latency still does, as frames. The prelude -- placement, the 3 s stand, the token posture -- counts ticks
instead of sleeping, and the policy's diffusion noise is seeded per inference, so a round with MuJoCo images
replays bit for bit when the cameras render without multisampling; under MuJoCo's default 4x MSAA the wrist images
differ by a grey level and the rounds part from the first chunk (docs/bimanual_design.md, time model; seen again
2026-10-07 at 25 and 50 fps alike). The verdict is `vla_rounds.RoundScore`, on simulation time.

A 50 fps run follows NVIDIA's client (`gear_sonic/scripts/run_vla_inference.py`; DECISIONS 2026-10-07): the first
chunk takes over from the standing token at its latency-compensated index with no blend (their `i` blend to the
initial token is our prelude's 3 s on it), and a chunk is always played `execution_horizon` frames, holding its last
frame past its end, as their index clamps at the horizon. A 25 fps run keeps the 2026-09 cycle (the 1 s blend, the
adopted span cut at the chunk's end), so its baselines replay. With `images.camera_delay` the pictures are as late
as a real camera's: each observation draws them from the earlier state `pipeline.camera_delay.camera_seq` names,
by the per-round draw the dataset's re-render uses per episode (`Camera`), while the proprioception stays the
boundary's.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from locomani.backends.sonic import initial_pose
from locomani.backends.sonic.agent_server import HAND_OPEN
from locomani.backends.sonic.sonic_token import GRIPPER_MAX, expand_hands, hand_from_gripper
from locomani.pipeline import camera_delay
from locomani.pipeline.world_lift import SRC_FPS
from locomani.sim.mirror import MIRRORED_SOURCES
from locomani.tasks.crate.expert import facing_edge, start_placement
from locomani.tasks.crate.visual_approach import approach_heading
from locomani.tasks.crate.vla_rounds import CLOSE_THRESHOLD, RoundScore, takeover_sample

RATE = 25  # dataset frames per second of every run before `dataset.fps` (2026-10-06): one token per frame
TICKS = 2  # 50 Hz control ticks per frame at RATE
OFFICIAL_RATE = 50  # a run at this rate follows NVIDIA's client (module docstring)
ENGAGE_TICKS = 250  # before the first reset: the INIT ramp is 150 ticks, and a policy not yet engaged falls over
SETTLE_TICKS = 20  # vla_rounds.place's 0.4 s after the reset request
STAND_TICKS = 150  # vla_rounds.place's 3 s IDLE hold, which the clean commander spends on the standing token
TOKEN_POSE_TICKS = 150  # initial_pose.prepare_token_pose's 3 s on the token
AFTER_TICKS = 50  # vla_rounds.run_round's closing 1 s hold
ARM_TOLERANCE = 0.30  # prepare_token_pose's ready-arm check, rad


def ticks_per_frame(rate: int) -> int:
    """The 50 Hz control ticks one dataset frame lasts at `rate` frames per second."""
    if rate <= 0 or SRC_FPS % rate:
        raise SystemExit(f"a lockstep round at {rate} fps: the rate must divide the {SRC_FPS} Hz control clock")
    return SRC_FPS // rate


class Session:
    """The client side of a lockstep stack: the step link, the clock that primes each deploy session, the robot.

    Publishers PUSH into the sim's relay (TF_SONIC_PUB_CONNECT), because the sim owns the deploy's only PUB under
    lockstep; that has to be set before any commander is built. The clock's background pump stays paused: every
    tick is asked for here, so a round is a function of its seed.
    """

    def __init__(self, ticks: int = TICKS):
        from locomani.backends.sonic import agent_server
        from locomani.backends.sonic.lockstep_client import LockstepClock, LockstepLink, lockstep_commander_class
        from locomani.backends.sonic.primitives import Primitives

        os.environ.setdefault(
            "TF_SONIC_PUB_CONNECT",
            os.environ.get("TF_LOCKSTEP_XSUB_PORT", str(int(os.environ.get("TF_SONIC_PORT", 5556)) + 28)),
        )
        self.link = LockstepLink()
        if not self.link.ping():
            raise SystemExit("no lockstep sim answers: start the eval stack with the experiment's eval.client lockstep")
        self.clock = LockstepClock(self.link, agent_server.AgentCommander())
        self.p = Primitives(commander=lockstep_commander_class(self.clock)())
        self.init_tok = self.clock.init_tok
        self.open_hands = expand_hands(GRIPPER_MAX)  # (left, right), what LockstepClock stands on
        self.ticks = ticks  # control ticks per dataset frame (`ticks_per_frame`)

    def step(self, frames, want_obs: bool = False) -> dict:
        """`frames` [(token, left7, right7)], each held `ticks` ticks; the reply of the boundary after the last."""
        rep = self.clock._tick(frames, self.ticks, want_obs=want_obs)
        if not self.clock.primed:
            raise RuntimeError(f"the deploy session changed mid-round ({rep.get('session')}): a reset or a restart")
        return rep

    def stand(self, ticks: int, chunk: int = 10) -> dict:
        """`ticks` control ticks on the standing token, re-priming whenever a reset or a deploy restart opened a
        new session. A reset lands in the sim's next tick and reopens the session one tick later, inside the 60
        ticks of kinematic warm-up (`backend.lockstep.warmup_ticks`), so ticking `chunk` at a time is safe."""
        clock = self.clock
        clock.pause()
        if not clock.ensure_primed():
            raise RuntimeError("the deploy never became ready (lockstep)")
        left, right = self.open_hands
        rep = {}
        while ticks > 0:
            if not clock.primed:
                clock._prime()
            k = min(ticks, chunk)
            rep = clock._tick([(self.init_tok, left, right)] * k, 1)
            ticks -= k
        return rep

    def state_at(self, seq: int, timeout: float = 2.0) -> dict:
        """The privileged state numbered `seq`: the frozen boundary's, for the verdict and the trace."""
        end = time.time() + timeout
        st = self.p.state.latest(0)
        while st is None or int(st.get("seq", -1)) < seq:
            if time.time() > end:
                raise RuntimeError(f"privileged state {seq} never arrived (newest {None if st is None else st['seq']})")
            st = self.p.state.latest(20) or st
        if int(st["seq"]) != seq:
            raise RuntimeError(f"asked for privileged state {seq}, the newest is {st['seq']}: the world moved")
        return st


def prelude(sess: Session, reset, meta, cfg, rng):
    """`vla_rounds.prelude` counted in ticks: placement, the stand, the token posture. Returns (placement, why)."""
    p = sess.p
    placement_fn, record = start_placement(reset, meta, cfg.visual, rng)
    heading = approach_heading(meta)
    p.cmd.reset_pose()
    p.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)
    reset.pin_arms()
    placement_fn("vla_round")  # PUSH: the sim applies it in its next tick
    sess.stand(SETTLE_TICKS)
    placement = dict(record, takeover=dict(spawn=takeover_sample(p, p.obs())))
    p.hold_yaw = heading
    sess.stand(STAND_TICKS)
    initial = p.obs()
    if float(initial["pelvis"]["pos"][2]) < 0.5 or initial["fallen"]:
        return placement, "[stand] robot did not stabilise after reset"
    expected = initial_pose.initial_arms()
    if expected is None:
        return placement, "[stand] crate token file declares no arms"
    placement["takeover"]["held"] = takeover_sample(p, initial)
    sess.stand(TOKEN_POSE_TICKS)  # the token posture is the standing token: prepare_token_pose streams exactly it
    state = p.obs()
    placement["takeover"]["streamed"] = takeover_sample(p, state)
    arms = np.asarray(state["joints"], dtype=float)[15:29]
    if state["fallen"] or arms.shape != (14,) or np.max(np.abs(arms - np.asarray(expected))) >= ARM_TOLERANCE:
        return placement, "[stand] token-only ready posture not delivered"
    return placement, None


class Latency:
    """Frames of simulated inference latency, drawn per chunk: fixed (`latency_frames`) or from a distribution
    measured on wall-clock rounds (`latency_dist`, see `latency_dist` below). Seeded per round."""

    def __init__(self, dist: dict, camera_frames: int, rng):
        self.dist, self.camera, self.rng = dist, int(camera_frames), rng

    def draw(self, key: str) -> tuple[int, int]:
        """(L, D): the inference frames drawn and the frames the world moves before the chunk is adopted."""
        inference = int(self.rng.choice(self.dist[key]))
        return inference, inference + self.camera


class Camera:
    """A round's camera under `images.camera_delay`: the earlier state each observation's pictures show. One draw
    of latency and capture phase per round, keyed by the round's seed, by the function the dataset's re-render
    draws per episode (`camera_delay.episode_draw`)."""

    def __init__(self, delay: camera_delay.CameraDelay, seed: int):
        self.delay = delay
        self.draw = camera_delay.episode_draw(delay, f"lockstep/{seed}")

    def seq(self, boundary: int) -> int:
        return camera_delay.camera_seq(self.delay, self.draw, boundary)

    def record(self) -> dict:
        latency, phase = self.draw
        return dict(self.delay.spec(), latency_draw_ms=round(latency * 1e3, 1), phase_ms=round(phase * 1e3, 1))


def observe(sess: Session, obs_src, rep: dict, image_source: str, last_index, camera: Camera | None = None):
    """The policy's observation at the boundary `rep` closed; (obs, trace fields, the deploy state's index). With
    a `camera` the pictures show the earlier state it names (the sim keeps it: `LockstepLink.observe`)."""
    proprio = obs_src.settle_proprio(last_index)
    if camera is not None:
        boundary = rep["priv_seq"]
        seq = camera.seq(boundary)
        got = sess.link.observe(seq)
        meta = None
        if image_source in MIRRORED_SOURCES:
            rendered = obs_src.mirror.render(state={k: got[k] for k in ("seq", "t", "wall", "qpos")})
            if rendered is None:
                raise RuntimeError(f"the mirror did not render state {seq}: {obs_src.mirror.last_error}")
            images, meta = rendered
            obs_src.image_meta = meta
        else:
            images = got.get("images")
            if images is None:
                raise RuntimeError(
                    "the lockstep sim returned no images: its eval stack renders offline (images.source?)"
                )
        obs = obs_src.build(images=images)
        if obs is None:
            raise RuntimeError("observation incomplete at the boundary")
        image = dict(seq=seq, age_ms=round((boundary - seq) * 1e3 / SRC_FPS, 1))
        if meta is not None:
            image["render_ms"] = meta.get("render_ms")
        return obs, dict(boundary_seq=boundary, proprio_index=proprio.get("index"), image=image), proprio.get("index")
    if image_source in MIRRORED_SOURCES:
        obs = obs_src.build(seq=rep["priv_seq"])
    else:
        if rep.get("images") is None:
            raise RuntimeError("the lockstep sim returned no images: its eval stack renders offline (images.source?)")
        obs = obs_src.build(images=rep["images"])
    if obs is None:
        raise RuntimeError("observation incomplete at the boundary")
    image = obs_src.image_meta if image_source in MIRRORED_SOURCES else None
    fields = dict(boundary_seq=rep.get("priv_seq"), proprio_index=proprio.get("index"))
    if image is not None:
        fields["image"] = {k: image.get(k) for k in ("seq", "t", "state_age_s", "render_ms")}
        if image.get("seq") != rep.get("priv_seq"):
            raise RuntimeError(f"the mirror rendered state {image.get('seq')}, the boundary is {rep.get('priv_seq')}")
    return obs, fields, proprio.get("index")


def frame_indices(start: int, count: int, length: int) -> np.ndarray:
    """The chunk indices `count` frames from `start` play: past the chunk's end its last frame is held."""
    return np.minimum(np.arange(start, start + count), length - 1)


def chunk_frames(chunk, start: int, count: int, low: float) -> list:
    """`count` wire frames of `chunk` (token, left, right) from index `start`, holding its last frame past the end."""
    tok, left, right = chunk
    return [
        (tok[j], hand_from_gripper(left[j], low), hand_from_gripper(right[j], low))
        for j in frame_indices(start, count, len(tok))
    ]


def run_round(
    sess: Session,
    reset,
    obs_src,
    link,
    meta,
    cfg,
    rng,
    *,
    budget_s,
    execution_horizon,
    trace_path,
    latency: Latency,
    gap: str,
    image_source: str,
    seed_policy: bool,
    obs_dump: Path | None = None,
    rate: int = RATE,
    camera: Camera | None = None,
):
    meta = facing_edge(meta, cfg.visual, rng)  # the round's walk-in, as the collection expert draws it
    placement, why = prelude(sess, reset, meta, cfg, rng)
    if why is not None:
        sess.stand(AFTER_TICKS)
        return dict(ok=False, why=why, placement=placement)
    p = sess.p
    low = cfg.visual.close_target  # the closed end the expert commanded, which the policy was trained on
    left_open, right_open = sess.open_hands
    stand_frame = (sess.init_tok, left_open, right_open)
    official = rate == OFFICIAL_RATE
    budget_f = int(budget_s * rate)
    record = dict(
        clock="lockstep",
        gap=gap,
        rate=rate,
        latency_source=latency.dist.get("source"),
        camera_frames=latency.camera,
        camera_delay=None if camera is None else camera.record(),
    )

    calls = []
    timing = {}

    def infer(obs):
        if seed_policy:  # the server seeds its diffusion noise with this: a MuJoCo-image round replays exactly
            obs["policy.seed"] = int(rng.integers(0, 2**31 - 1))
        t0 = time.perf_counter()
        tok, left, right = link.get_chunk_bimanual(obs)
        # the policy server's wall time, never the world's: what `latency-dist` builds the next distribution from
        # (DECISIONS 2026-10-07: drawn, not measured live, so a round replays)
        timing["infer_s"] = round(time.perf_counter() - t0, 4)
        if obs_dump is not None:  # backend.obs_dump: what the policy saw and answered, one file per inference
            obs_dump.mkdir(parents=True, exist_ok=True)
            arrays = {k: np.asarray(v) for k, v in obs.items() if not isinstance(v, str)}
            np.savez_compressed(
                obs_dump / f"{trace_path.stem if trace_path else 'round'}_{len(calls):04d}.npz",
                tok=tok,
                left=left,
                right=right,
                **arrays,
            )
        calls.append(1)
        return tok, left, right

    score = None
    try:
        rep = sess.step([stand_frame], want_obs=True)  # a boundary to look from
        initial = sess.state_at(rep["priv_seq"])
        obs, _, index = observe(sess, obs_src, rep, image_source, None, camera)
        tok0, left0, right0 = infer(obs)
        _, d0 = latency.draw("first")
        score = RoundScore(p, meta, cfg, initial, 0.0)
        if official:
            # NVIDIA's client: the deploy holds the standing token until the first chunk arrives, which then plays
            # from its latency-compensated index -- no blend into it
            first = frame_indices(d0, execution_horizon, len(tok0))
            if np.min(np.minimum(left0[first], right0[first])) < CLOSE_THRESHOLD:
                score.commanded_close = True
            adopted = chunk_frames((tok0, left0, right0), d0, execution_horizon, low)
            rep = sess.step([stand_frame] * d0 + adopted, want_obs=True)
            f = d0 + execution_horizon
            running, last = ((tok0, left0, right0), d0 + execution_horizon), adopted[-1]
        else:
            # the deploy stands on the standing token while the first chunk is computed, then the 1 s blend into
            # it: the wall-clock prelude (vla_rounds.run_round), in frames
            l0, r0 = hand_from_gripper(left0[0], low), hand_from_gripper(right0[0], low)
            blend = [((1 - a) * sess.init_tok + a * np.asarray(tok0[0]), l0, r0) for a in np.arange(1, rate + 1) / rate]
            rep = sess.step([stand_frame] * d0 + blend, want_obs=True)
            f = 0
            running, last = ((tok0, left0, right0), 0), blend[-1]
        while f < budget_f:
            obs, fields, index = observe(sess, obs_src, rep, image_source, index, camera)
            tok, left, right = infer(obs)
            inference, d = latency.draw("steady")
            waiting = chunk_frames(*running, d, low) if gap == "continue" else [last] * d
            # official: a chunk plays `execution_horizon` frames whatever its length; 25 fps: cut at its end
            n = execution_horizon if official else max(1, min(execution_horizon, len(tok) - d))
            played = frame_indices(d, n, len(tok))
            adopted = chunk_frames((tok, left, right), d, n, low)
            if np.min(np.minimum(left[played], right[played])) < CLOSE_THRESHOLD:
                score.commanded_close = True
            rep = sess.step(waiting + adopted, want_obs=True)
            f += d + n
            running, last = ((tok, left, right), d + n), adopted[-1]
            done = score.check(
                sess.state_at(rep["priv_seq"]),
                f / rate,
                frame=f,
                left_grip=float(np.min(left[played])),
                right_grip=float(np.min(right[played])),
                inference_frames=inference,
                skip=d,
                gap_frames=d,
                infer_s=timing["infer_s"],
                **fields,
            )
            if done or rep.get("fallen"):
                break
    except RuntimeError as exc:
        why = f"[vla] lockstep: {exc}"
    finally:
        sess.stand(AFTER_TICKS)
    if score is None:
        return dict(ok=False, why=why or "[vla] lockstep: no boundary", placement=placement)
    if why is not None:
        score.why = why
    return score.result(placement, f / rate, trace_path, **record)


def run_batch(args, config, meta, link) -> None:
    """`vla_rounds.main`'s batch, on the simulation clock."""
    from locomani.backends.sonic.config import DEFAULT_BACKEND  # the stack's backend block, from its environment
    from locomani.backends.sonic.lockstep_client import load_latency_dist
    from locomani.backends.sonic.vla_obs import ObsBuilder
    from locomani.schema import SONIC_CRATE
    from locomani.sim.privileged_pub import EpisodeControlClient
    from locomani.tasks.crate.grasp_region import target_key

    rate = int(args.dataset_fps)
    dist = load_latency_dist(args.latency_dist, args.latency_frames)
    if args.latency_dist and int(dist.get("fps", RATE)) != rate:
        raise SystemExit(
            f"{args.latency_dist} counts frames at {dist.get('fps', RATE)} fps (a file without `fps` is 25) and this "
            f"run's dataset is {rate}: rebuild it with `vla_lockstep latency-dist --fps {rate}`"
        )
    delay = camera_delay.parse(json.loads(args.camera_delay)) if args.camera_delay else None
    if delay is not None and args.camera_frames:
        raise SystemExit("camera_frames and images.camera_delay both delay the pictures; the run sets camera_delay")
    obs_dump = Path(DEFAULT_BACKEND.obs_dump) if DEFAULT_BACKEND.obs_dump else None
    sess = Session(ticks_per_frame(rate))
    sess.link.begin()  # a fresh session: after a deploy restart the old one would only time out
    reset = EpisodeControlClient()
    obs_src = ObsBuilder(config.prompt, schema=SONIC_CRATE, image_source=args.image_source)
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        sess.stand(ENGAGE_TICKS)  # lockstep pitfall 4: engage the policy before the first reset
        _, why = prelude(sess, reset, meta, config, np.random.default_rng([args.seed, 0xB1A5]))
        print(f"[vla] warm-up round: {'ok' if why is None else why}", flush=True)
        for index in range(args.rounds):
            seed = args.seed + index
            started = time.time()
            label = f"crate_vla_{index + args.round_offset:03d}"
            reset.start_video(label)
            rng = np.random.default_rng(seed)
            result = run_round(
                sess,
                reset,
                obs_src,
                link,
                meta,
                config,
                rng,
                budget_s=args.budget_s,
                execution_horizon=args.execution_horizon,
                trace_path=None
                if args.trace_dir is None
                else args.trace_dir / f"round_{index + args.round_offset:03d}.json",
                latency=Latency(dist, args.camera_frames, np.random.default_rng([seed, 0x1A7])),
                gap=args.gap,
                image_source=args.image_source,
                seed_policy=not args.no_policy_seed,
                obs_dump=obs_dump,
                rate=rate,
                camera=None if delay is None else Camera(delay, seed),
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
        sess.stand(5)  # the last end_video (and its state trace) lands in a tick
    finally:
        obs_src.close()


def latency_dist(trace_dir: Path, *, inference_only: bool, fps: int = RATE) -> dict:
    """A `latency_dist` measured on evaluation rounds (`<eval>/traces/round_*.json`), in frames at `fps`.

    Wall-clock (default): the frames each chunk actually skipped (`skip`), observation, rendering and inference
    together -- what a lockstep run needs to reproduce the wall-clock loop (plan W0). `inference_only`: the same
    latency without the Isaac render (`image.render_ms`), in frames by NVIDIA's rounding -- the target model's L. A
    lockstep round's row records the policy server's own time (`infer_s`, the world frozen meanwhile), which is
    already inference only (DECISIONS 2026-10-07: the next run draws from it). The file records `fps`; a lockstep
    run refuses one counted at another rate.
    """
    frames = []
    for path in sorted(Path(trace_dir).glob("round_*.json")):
        for row in json.loads(path.read_text()).get("trace", []):
            if "infer_s" in row:
                if not inference_only:
                    raise SystemExit(f"{path} is a lockstep round: its skips are drawn, measure with --inference-only")
                frames.append(int(np.round(row["infer_s"] * fps)))
            elif inference_only:
                render = ((row.get("image") or {}).get("render_ms") or 0.0) / 1e3
                frames.append(int(np.round(max(0.0, row["latency_s"] - render) * fps)))
            else:
                frames.append(int(row["skip"]))
    if not frames:
        raise SystemExit(f"no chunk rows in {trace_dir}/round_*.json")
    kind = "inference only (render removed)" if inference_only else "wall-clock skip"
    return {"first": frames, "steady": frames, "fps": int(fps), "source": f"{trace_dir} ({kind}, {len(frames)} chunks)"}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Lockstep helpers for the crate evaluation.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("latency-dist", help="write a latency_dist JSON from wall-clock round traces")
    d.add_argument("traces", type=Path, help="<eval>/traces of an evaluation (wall clock, or lockstep's infer_s)")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--inference-only", action="store_true", help="drop the Isaac render time: the target model")
    d.add_argument("--fps", type=int, default=RATE, help="the dataset rate of the runs that will draw from it")
    args = ap.parse_args(argv)
    dist = latency_dist(args.traces, inference_only=args.inference_only, fps=args.fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dist, indent=1))
    counts = {k: dist["steady"].count(k) for k in sorted(set(dist["steady"]))}
    print(f"{args.out}: {dist['source']}; frames {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
