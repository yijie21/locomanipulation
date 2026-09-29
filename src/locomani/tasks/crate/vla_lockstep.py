"""Closed-loop VLA rounds for the crate on the simulation clock: the world stands still while the policy looks.

`vla_rounds.py --client lockstep` runs these instead of the wall-clock loop, inside a stack started with
`backend.lockstep.enabled` (sim LOCKSTEP=1, deploy --sim-lockstep, docs/archive/lockstep.md). Every tick of the
sim and the deploy happens because this process asked for it, so nothing moves between two requests. One cycle
(docs/handoff/lockstep-isaac-eval-plan-20260927.md §5; a frame is one 25 fps token, two 50 Hz ticks):

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
replays bit for bit. The verdict is `vla_rounds.RoundScore`, on simulation time.
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
from locomani.sim.mirror import MIRRORED_SOURCES
from locomani.tasks.crate.expert import facing_edge, start_placement
from locomani.tasks.crate.visual_approach import approach_heading
from locomani.tasks.crate.vla_rounds import CLOSE_THRESHOLD, RoundScore, takeover_sample

RATE = 25  # dataset frames per second: one token per frame
TICKS = 2  # 50 Hz control ticks per frame
ENGAGE_TICKS = 250  # before the first reset: the INIT ramp is 150 ticks, and a policy not yet engaged falls over
SETTLE_TICKS = 20  # vla_rounds.place's 0.4 s after the reset request
STAND_TICKS = 150  # vla_rounds.place's 3 s IDLE hold, which the clean commander spends on the standing token
TOKEN_POSE_TICKS = 150  # initial_pose.prepare_token_pose's 3 s on the token
AFTER_TICKS = 50  # vla_rounds.run_round's closing 1 s hold
ARM_TOLERANCE = 0.30  # prepare_token_pose's ready-arm check, rad


class Session:
    """The client side of a lockstep stack: the step link, the clock that primes each deploy session, the robot.

    Publishers PUSH into the sim's relay (TF_SONIC_PUB_CONNECT), because the sim owns the deploy's only PUB under
    lockstep; that has to be set before any commander is built. The clock's background pump stays paused: every
    tick is asked for here, so a round is a function of its seed.
    """

    def __init__(self):
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

    def step(self, frames, want_obs: bool = False) -> dict:
        """`frames` [(token, left7, right7)], each held TICKS ticks; the reply of the boundary after the last."""
        rep = self.clock._tick(frames, TICKS, want_obs=want_obs)
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


def observe(sess: Session, obs_src, rep: dict, image_source: str, last_index):
    """The policy's observation at the boundary `rep` closed; (obs, trace fields, the deploy state's index)."""
    proprio = obs_src.settle_proprio(last_index)
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


def chunk_frames(chunk, start: int, count: int, low: float) -> list:
    """`count` wire frames of `chunk` (token, left, right) from index `start`, holding its last frame past the end."""
    tok, left, right = chunk
    out = []
    for j in range(start, start + count):
        jj = min(j, len(tok) - 1)
        out.append((tok[jj], hand_from_gripper(left[jj], low), hand_from_gripper(right[jj], low)))
    return out


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
    budget_f = int(budget_s * RATE)
    record = dict(clock="lockstep", gap=gap, latency_source=latency.dist.get("source"), camera_frames=latency.camera)

    calls = []

    def infer(obs):
        if seed_policy:  # the server seeds its diffusion noise with this: a MuJoCo-image round replays exactly
            obs["policy.seed"] = int(rng.integers(0, 2**31 - 1))
        tok, left, right = link.get_chunk_bimanual(obs)
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
        obs, _, index = observe(sess, obs_src, rep, image_source, None)
        tok0, left0, right0 = infer(obs)
        _, d0 = latency.draw("first")
        # the deploy stands on the standing token while the first chunk is computed, then the 1 s blend into it:
        # the wall-clock prelude (vla_rounds.run_round), in frames
        l0, r0 = hand_from_gripper(left0[0], low), hand_from_gripper(right0[0], low)
        blend = [((1 - a) * sess.init_tok + a * np.asarray(tok0[0]), l0, r0) for a in np.arange(1, RATE + 1) / RATE]
        rep = sess.step([stand_frame] * d0 + blend, want_obs=True)
        f = 0
        score = RoundScore(p, meta, cfg, initial, 0.0)
        running, last = ((tok0, left0, right0), 0), blend[-1]
        while f < budget_f:
            obs, fields, index = observe(sess, obs_src, rep, image_source, index)
            tok, left, right = infer(obs)
            inference, d = latency.draw("steady")
            waiting = chunk_frames(*running, d, low) if gap == "continue" else [last] * d
            n = max(1, min(execution_horizon, len(tok) - d))
            adopted = chunk_frames((tok, left, right), d, n, low)
            if any(min(left[j], right[j]) < CLOSE_THRESHOLD for j in range(d, d + n)):
                score.commanded_close = True
            rep = sess.step(waiting + adopted, want_obs=True)
            f += d + n
            running, last = ((tok, left, right), d + n), adopted[-1]
            done = score.check(
                sess.state_at(rep["priv_seq"]),
                f / RATE,
                frame=f,
                left_grip=float(np.min(left[d : d + n])),
                right_grip=float(np.min(right[d : d + n])),
                inference_frames=inference,
                skip=d,
                gap_frames=d,
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
    return score.result(placement, f / RATE, trace_path, **record)


def run_batch(args, config, meta, link) -> None:
    """`vla_rounds.main`'s batch, on the simulation clock."""
    from locomani.backends.sonic.config import DEFAULT_BACKEND  # the stack's backend block, from its environment
    from locomani.backends.sonic.lockstep_client import load_latency_dist
    from locomani.backends.sonic.vla_obs import ObsBuilder
    from locomani.schema import SONIC_CRATE
    from locomani.sim.privileged_pub import EpisodeControlClient
    from locomani.tasks.crate.grasp_region import target_key

    dist = load_latency_dist(args.latency_dist, args.latency_frames)
    obs_dump = Path(DEFAULT_BACKEND.obs_dump) if DEFAULT_BACKEND.obs_dump else None
    sess = Session()
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


def latency_dist(trace_dir: Path, *, inference_only: bool, fps: float = RATE) -> dict:
    """A `latency_dist` measured on wall-clock rounds (`<eval>/traces/round_*.json`).

    Wall-clock (default): the frames each chunk actually skipped (`skip`), observation, rendering and inference
    together -- what a lockstep run needs to reproduce the wall-clock loop (plan W0). `inference_only`: the same
    latency without the Isaac render (`image.render_ms`), in frames by NVIDIA's rounding -- the target model's L.
    """
    frames = []
    for path in sorted(Path(trace_dir).glob("round_*.json")):
        for row in json.loads(path.read_text()).get("trace", []):
            if inference_only:
                render = ((row.get("image") or {}).get("render_ms") or 0.0) / 1e3
                frames.append(int(np.round(max(0.0, row["latency_s"] - render) * fps)))
            else:
                frames.append(int(row["skip"]))
    if not frames:
        raise SystemExit(f"no chunk rows in {trace_dir}/round_*.json")
    kind = "inference only (render removed)" if inference_only else "wall-clock skip"
    return {"first": frames, "steady": frames, "source": f"{trace_dir} ({kind}, {len(frames)} chunks)"}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Lockstep helpers for the crate evaluation.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("latency-dist", help="write a latency_dist JSON from wall-clock round traces")
    d.add_argument("traces", type=Path, help="<eval>/traces of a wall-clock evaluation")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--inference-only", action="store_true", help="drop the Isaac render time: the target model")
    args = ap.parse_args(argv)
    dist = latency_dist(args.traces, inference_only=args.inference_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dist, indent=1))
    counts = {k: dist["steady"].count(k) for k in sorted(set(dist["steady"]))}
    print(f"{args.out}: {dist['source']}; frames {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
