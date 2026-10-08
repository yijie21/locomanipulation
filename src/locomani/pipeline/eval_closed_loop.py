"""Closed-loop evaluation: the finetuned VLA drives the real stack, N rounds.

    .venv_sim $ python eval_closed_loop.py [--config path/to/eval_config.json] \
                    [--set policy_port=5590] [--out outputs/eval/bottle200/closed_loop.json]

Every semantic knob lives in `EvalConfig` (locomani/pipeline/eval_config.py);
see `configs/eval/closed_loop_100.yaml` (`conf.load_group("eval", "closed_loop_100")`)
for the canonical 100-round protocol.
`--set key=value` (dotted for `backend.*`) overrides one field at a time.

PREREQUISITES (this script checks, it does not arrange):
  * the stack is up with NO_ENCODER=1 -- the token-only deployment mode.
    (This used to claim an encoder-loaded deploy makes the token stream
    "never hold"; that was wrong -- see the corrected note in start_stack.sh
    and docs/official_parity_audit.md C5. NO_ENCODER is kept because the gym
    restarts the deploy every round and skipping a second TensorRT engine
    keeps that restart shorter, not because the stream needs it.)
  * serve_policy.py is running on --policy-port with the checkpoint to test.

WHAT IT RUNS

grasp_gym's OWN round loop with vla_round installed as grasp_gym.VLA_POLICY,
and the same spawn arguments the collection used. Nothing about the rounds is
re-implemented here. The printed 200/210 reference is historical and unpaired;
it does not measure the current expert or a supplied fixed case bank. Current
comparisons need their own saved expert results and matching case metadata.

WHAT COMES OUT

HOW MANY ROUNDS

`rounds` defaults to 20 (2026-08-28). A 20-round PERCENTAGE is not a result:
its Wilson interval is 30-40 points wide. 20 rounds exist to support PAIRED
comparison against a same-seed run, where they resolve large effects only
(6-0 or 8-1 discordant reaches p<0.05 -- roughly a doubling). Numbers that
go into the ladder, or into any document, still need 100 (eval=closed_loop_100,
configs/eval/closed_loop_100.yaml).

Success rate with a Wilson 95% interval, and the per-stage failure breakdown
(walk / reach / close / fell / timeout) from vla_round's classifier. The
breakdown is the actionable part: failures concentrated where the script's
own failures live (close) mean faithfully inherited data, failures in the
walk mean the policy did not learn the task -- and trigger Q5's history
ablation, not more data.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# reach src/ for `python src/locomani/pipeline/eval_closed_loop.py`. THREE levels up, not two:
# src/locomani/pipeline/<this> -> pipeline -> locomani -> src. Two levels reached src/locomani, so
# `import locomani` failed with ModuleNotFoundError under every launcher that does not already have
# src on the path -- which is both of them (run_learn_pilot's eval stage passes an env without
# PYTHONPATH; run_eval_arm.sh cds into this directory). Broken since the 4a package move (29bdf84,
# 2026-09-09) and invisible until 2026-09-10, because verifying the eval path needs a trained model
# and every model was lost in the 09-09 incident -- it is the refactor follow-up STATE listed as
# "not yet verified: the eval path".
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from locomani.backends.sonic import initial_pose  # noqa: E402
from locomani.pipeline import stack  # noqa: E402
from locomani.pipeline.eval_config import EvalConfig  # noqa: E402
from locomani.pipeline.provenance import write_config_snapshot  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def gym_options(
    cfg,
    manifest,
    *,
    rounds: int,
    seed: int,
    cases: str | None = None,
    reach_config: str | None = None,
    vla: bool = True,
    video_include_init: bool = False,
):
    """Which gym options this evaluation sets, and what the rest is left at.

    The same spawn distribution the collection ran, WITHOUT `record` (no dataset is written) and
    WITHOUT `until_ok` (failures are the measurement, not something to retry past). `rounds` and
    `seed` are arguments rather than read off `cfg` because a resumed run offsets both.

    Keys are `GymOptions` field names and values are what argparse would have produced: until
    2026-09-22 this was an argv list assembled for a process that is never started, because the
    gym runs in THIS one (docs/handoff/flat-functional-scripts-20260922.md, decision 11). It is
    resolved through `cli_args.resolve_options`, the same merge `grasp_gym.main` uses, so an
    option set here and an option set on a command line cannot diverge.
    """
    from locomani.tasks.bottle import cli_args
    from locomani.tasks.bottle.reach_config import ReachConfig

    explicit = dict(
        rounds=rounds,
        stand_retries=cfg.stand_retries,
        targets=cfg.target,
        seed=seed,
        manifest=str(manifest),
        spawn_edge=[float(v) for v in cfg.spawn_edge],
        face_object=True,
        spawn_yaw_err=20.0,
        spawn_lateral=0.20,
        edge_jitter=0.20,
    )
    if cfg.video_include_init:
        explicit["video_include_init"] = True
    if cases:
        explicit["cases"] = cases
    if reach_config:
        explicit["reach_config"] = reach_config
    return cli_args.resolve_options(
        explicit,
        reach=ReachConfig.load(reach_config) if reach_config else None,
        vla=vla,
        video_include_init=video_include_init,
    )


def round_outcomes(rows):
    """One row per round: the attempt the round ended on.

    The manifest holds one line per ATTEMPT, and a stand retry re-runs the same
    case -- same case_id, same spawn, same object pose, same appearance seed --
    because the robot never stabilised after the reset teleport, before the task
    began. Those lines are one sample, not two, so counting them as rounds both
    inflates the denominator with setup failures and, when the retry is the one
    that works, reports a round that succeeded as a 50% success rate.

    Rounds are kept in first-seen order. A manifest without `round` (anything
    older than this field) degrades to one round per line, which is what the
    reader assumed all along.
    """
    ended: dict = {}
    for i, r in enumerate(rows):
        ended[r.get("round", i)] = r
    return list(ended.values())


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "run",
        nargs="?",
        type=Path,
        default=None,
        help="the run directory train wrote. Given this, every option below is read from the run's "
        "own config.yaml and summary.json, the stack is attached and taken down here, and the "
        "result is recorded in summary.json. The explicit options remain for diagnostics that "
        "evaluate something other than a run (scripts/experiments/)",
    )
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--keep-stack", action="store_true", help="leave the stack running when the rounds end")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    ap.add_argument("--config", type=Path, help="EvalConfig JSON; defaults if omitted")
    ap.add_argument("--set", dest="set_", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--cases", help="exact case bank generated by grasp_gym --sample-only")
    ap.add_argument("--reach-config", help="collection configuration used to generate cases")
    ap.add_argument(
        "--initial-token-file",
        type=Path,
        help="explicit fixed initialization token JSON; ready collection profiles prepare it before the visibility gate",
    )
    return ap.parse_args(argv)


def evaluate_run(args):
    """Evaluate the run `args.run` names: attach the stack, run the rounds here, record the result.

    This is what `run_eval_single_arm.py` was until 2026-09-22. It is not a separate process any
    more -- the rounds run in THIS one, so there is no argv to assemble for them and no second
    place for the environment to differ (docs/handoff/flat-functional-scripts-20260922.md). The
    admission, the stack teardown and the summary are `pipeline/closed_loop.py`, shared with the
    bimanual evaluator.
    """
    from locomani.pipeline import closed_loop, run_context, run_vla_server
    from locomani.pipeline.stages import Stage

    run_dir = args.run if args.run.is_absolute() else REPO / args.run
    if args.dry_run:
        return closed_loop.show(run_dir)
    run = closed_loop.attach(run_dir, args.run, args.gpu)
    # The instance's port block and the scene, before anything opens a client. The subprocess this
    # used to be was handed exactly this as its whole environment; in-process it is assigned.
    run_context.adopt_stack_env(run.stack)
    run_vla_server.wait_for_server(run.evaluation.policy_port)

    # Fill in what the driver used to pass on a command line, then run the evaluation below.
    args.config = run.ev / "eval_config.json"
    # The stack is pinned to the dataset's rate. The lockstep clients count frames at 25 (tasks/crate/vla_lockstep
    # RATE, latency_frames, the 1 s blends), so a dataset stored at another rate has no closed-loop evaluation
    # here yet; it is exercised through the official-client path (src/real_robot, sim2sim) -- see
    # docs/handoff/real-robot-sim2real-plan-20261006.md.
    fps = run_context.dataset_fps(run.cfg)
    if fps != 25:
        raise SystemExit(
            f"dataset.fps is {fps}: the closed-loop evaluation counts frames at 25 fps and would replay this "
            "checkpoint at the wrong speed; evaluate it through the real-robot client (scripts/run_real_robot_inference.sh, sim2sim)"
        )
    args.set_ = list(args.set_) + [f"backend.dataset_fps={fps}"]
    args.reach_config = args.reach_config or str(run_dir / "reach_config.json")
    args.initial_token_file = args.initial_token_file or (REPO / run.cfg["eval"]["initial_token"])
    args.out = str(run.ev / "closed_loop.json")
    args.trace_dir = str(run.ev / "traces")

    stage = Stage(run_dir)
    stage.record_config("eval", run.evaluation.to_dict(), env=dict(run.stack.env))
    try:
        with stage.timed("eval"):
            _evaluate(args)
    finally:
        closed_loop.finish(run, keep_stack=args.keep_stack)
    closed_loop.record(run)
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.run is not None:
        return evaluate_run(args)
    return _evaluate(args)


def instance_deploy_cmdline(instance: int) -> str | None:
    """The command line of `instance`'s own deploy, or None when that instance has no live deploy.

    Until 2026-09-28 this was `pgrep -af g1_deploy_onnx_ref` over the whole machine, so another
    session's collection deploy (WITH its encoder, on instance 1) refused a clean evaluation on
    instance 0 -- the bottle smoke of the documentation rewrite, the first time two stacks overlapped.
    """
    from locomani.stack import Stack

    try:
        pid = Stack.attach(instance).pids.get("deploy")
    except FileNotFoundError:
        return None
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, TypeError):
        return None


def _evaluate(args):
    cfg = EvalConfig.load(args.config) if args.config else EvalConfig()
    if args.set_:
        cfg = cfg.with_overrides(args.set_)

    if cfg.rtc and cfg.client not in ("async", "lockstep"):
        raise SystemExit("rtc=true is implemented for the async/lockstep clients only")
    if cfg.transition_commit and (
        not cfg.rtc or cfg.client != "async" or cfg.ensemble_n != 1 or not cfg.transition_decoder
    ):
        raise SystemExit("transition_commit=true requires rtc=true, client=async, ensemble_n=1, transition_decoder set")
    if cfg.client == "lockstep" and cfg.image_source != "mujoco":
        # The single-arm lockstep round takes its images from the step reply (MuJoCo, at the boundary); only the
        # bimanual one asks the mirror (Isaac or Unreal) for the boundary's state (tasks/crate/vla_lockstep.py).
        raise SystemExit(
            f"client=lockstep with image_source={cfg.image_source} is implemented by the bimanual evaluator only"
        )
    if cfg.gap != "continue" or cfg.camera_frames:
        raise SystemExit(
            f"gap={cfg.gap!r} camera_frames={cfg.camera_frames} are bimanual-evaluator knobs; the single-arm "
            "clients always continue the old chunk through the inference gap"
        )

    if args.initial_token_file:
        initial_pose.INIT_TOKEN_FILE = str(args.initial_token_file.resolve())
        initial_pose.initial_token()  # Validate before starting a round.

    # --- preflight, loudly -------------------------------------------------
    from locomani.backends.sonic import vla_round

    link = vla_round.PolicyLink(port=cfg.policy_port, config=cfg)
    if not link.ping():
        sys.exit(f"no policy server on :{cfg.policy_port} -- start tools/serve_policy.py (Isaac-GR00T venv) first")

    dep = instance_deploy_cmdline(int(os.environ.get("INSTANCE", "0")))
    if not dep:
        sys.exit("deploy is not running -- bring the stack up: bash scripts/start_stack.sh eval <run-dir>")
    if "--encoder-file" in dep:
        sys.exit(
            "deploy is running WITH its encoder. Token streaming will "
            "not hold (initial_encoder_mode_ 0 keeps re-enabling it). "
            "Restart the stack with NO_ENCODER=1."
        )

    out = Path(args.out or REPO / "outputs/eval/bottle200/closed_loop.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = out.with_suffix(".manifest.jsonl")
    trace_dir = args.trace_dir or str(out.parent / "closed_loop_traces")

    # The gym restarts the deploy every round (fresh_deploy ->
    # scripts/restart_deploy.sh, a subprocess that inherits this environment).
    # Without this the round-2 restart brings the encoder back and the token
    # stream stops holding -- the preflight above would only have caught it on
    # the NEXT invocation. stack.export_env is the one place that turns a
    # SonicBackendConfig into environment for that inherited-env subprocess
    # boundary (decision 8).
    backend_cfg = dataclasses.replace(
        cfg.backend,
        no_encoder=True,
        clean_deploy=(cfg.client == "async"),
        lockstep=dataclasses.replace(cfg.backend.lockstep, enabled=(cfg.client == "lockstep")),
    )
    stack.export_env(backend_cfg)
    os.environ["LOCOMANI_DEPLOY_RECEIPTS"] = str(out.parent / "deploy_receipts.jsonl")
    from locomani.schema import SONIC_BOTTLE

    model_identity = link.identity()
    for kind in ("video", "state", "action"):
        expected = list(getattr(SONIC_BOTTLE, kind + "_keys"))
        if model_identity["modalities"][kind] != expected:
            raise ValueError(f"policy server {kind} schema differs from the bottle observation contract")
    write_config_snapshot(
        out.parent,
        backend_cfg=backend_cfg,
        eval_cfg=cfg,
        started=time.time(),
        inputs={"served_checkpoint": model_identity},
        runtime_env=dict(os.environ),
    )

    # --- install the policy and run the gym's own loop ---------------------
    from locomani.tasks.bottle import grasp_gym

    # Collection stopped restarting the deploy per round on 2026-09-10 (ledger
    # bottle §11); the eval path was never measured without the restart and
    # the clean-clock shim below pauses/primes around it, so eval KEEPS it by
    # default. Now a config field, because the two paths establish the deploy's
    # heading anchor differently and that is the suspect in §18's 2/100.
    from locomani.tasks.bottle.runtime import GymRuntime

    restart, robot_factory, clock = None, None, None

    if cfg.client == "lockstep":
        # Needs a stack started with LOCKSTEP=1 (lockstep_sim.py serving the
        # step RPC, deploy --sim-lockstep). Ping it rather than trusting env.
        ls = vla_round.LockstepLink()
        if not ls.ping():
            raise SystemExit(f"no lockstep sim on :{vla_round.LOCKSTEP_PORT} -- start the stack with LOCKSTEP=1")
        latency = vla_round.load_latency_dist(cfg.latency_dist, cfg.latency_frames)
        # The clock owns a commander of its own (a second PUB into the
        # deploy's SUB) for the per-session prime + enter_streamed; the shim
        # must be installed BEFORE grasp_gym builds its Primitives.
        # publishers connect to the sim's XSUB (lockstep_sim.py) -- set before
        # ANY AgentCommander is built (this one and grasp_gym's Primitives)
        os.environ["TF_SONIC_PUB_CONNECT"] = os.environ.get(
            "TF_LOCKSTEP_XSUB_PORT", str(int(os.environ.get("TF_SONIC_PORT", 5556)) + 28)
        )
        from locomani.backends.sonic import agent_server

        clock = vla_round.LockstepClock(ls, agent_server.AgentCommander())
        commander_cls = vla_round.lockstep_commander_class(clock)

        def robot_factory(*a, **kw):
            return grasp_gym.Primitives(*a, commander=commander_cls(), **kw)

        clock.resume()
        print(
            f"[closed-loop] LOCKSTEP client: simulation clock, K ~ "
            f"{latency['source']} (first {sorted(set(latency['first']))[:6]}.. "
            f"steady {sorted(set(latency['steady']))[:8]}..). Not comparable "
            f"with wall-clock numbers until calibrated (docs/single_arm_design.md)."
        )
        policy = vla_round.make_vla_policy_lockstep(
            link, ls, clock, cfg.task, latency, seed=cfg.seed, trace_dir=trace_dir, config=cfg
        )
    elif cfg.client == "async":
        clock = vla_round.CleanClock(config=cfg)
        commander_cls = vla_round.clean_commander_class(clock)
        _orig_fresh = grasp_gym.fresh_deploy

        def fresh_deploy(p):
            clock.pause()
            ok = _orig_fresh(p)
            if ok:
                clock.prime()
            clock.resume()
            return ok

        restart = fresh_deploy
        _robot = grasp_gym.Primitives

        def robot_factory(*a, **kw):
            robot = _robot(*a, commander=commander_cls(), **kw)
            if not clock.primed:
                clock.prime()
                clock.resume()
            return robot

        print(
            "[closed-loop] CLEAN DEPLOY: auto-start, no planner command in the "
            "deploy's lifetime, standing token between rounds."
        )
        policy = vla_round.make_vla_policy_async(
            link, cfg.task, inference_interval=cfg.replan_interval, trace_dir=trace_dir, config=cfg
        )
    else:
        policy = vla_round.make_vla_policy(
            link, cfg.task, execution_horizon=cfg.execution_horizon, trace_dir=trace_dir, config=cfg
        )

    if args.cases and cfg.resume:
        raise SystemExit("fixed case evaluation does not support RNG-offset resume; run a separate case subset")
    done = 0
    if cfg.resume and manifest.exists():
        done = sum(1 for l in manifest.read_text().splitlines() if l.strip())
        # Refuse to top up a manifest that came from a different seed. The
        # rounds would still be same-distribution and the arithmetic would
        # still work, which is exactly why this has to be an error rather than
        # a warning: nothing downstream would ever show that the sample is a
        # blend of two seed lineages.
        if done and out.exists():
            try:
                prev = json.loads(out.read_text()).get("seed")
            except Exception:
                prev = None
            if prev is not None and prev != cfg.seed and not cfg.force_resume:
                sys.exit(
                    f"--resume: the {done} rounds already in\n  {manifest}\n"
                    f"were run with seed {prev}, not {cfg.seed}. Re-run with "
                    f"seed {prev} (--set seed={prev}), point --out somewhere else, or set "
                    f"force_resume=true if mixing them is genuinely what you want."
                )
    todo = max(0, cfg.rounds - done)
    if cfg.resume and done:
        # Offset so the top-up cannot redraw the rounds already measured. The
        # sample is still i.i.d. from the same distribution, which is all a
        # rate estimate needs.
        print(f"[closed-loop] resuming: {done} rounds already in the manifest, {todo} to go (seed {cfg.seed}+{done})")
        if todo == 0:
            print("[closed-loop] nothing to do; aggregating what is there")
    else:
        manifest.unlink(missing_ok=True)

    print(
        f"[closed-loop] {cfg.rounds} rounds, horizon {cfg.execution_horizon}, "
        f"seed {cfg.seed}\n[closed-loop] manifest -> {manifest}"
    )
    runtime = GymRuntime(
        task=dataclasses.replace(GymRuntime().task, restart_deploy=cfg.restart_deploy),
        policy=policy,
        restart=restart,
        robot_factory=robot_factory,
    )
    options = gym_options(
        cfg,
        manifest,
        rounds=todo if (cfg.resume and done) else cfg.rounds,
        seed=cfg.seed + done if (cfg.resume and done) else cfg.seed,
        cases=args.cases,
        reach_config=args.reach_config,
        vla=runtime.policy is not None,
        video_include_init=runtime.task.video_include_init,
    )
    try:
        if todo > 0:
            grasp_gym.run(options, runtime=runtime)
    finally:
        if clock is not None:
            clock.stop()

    # --- aggregate ----------------------------------------------------------
    attempts = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    rows = round_outcomes(attempts)
    from locomani.pipeline.closed_loop import score

    scored = score(rows)  # one definition with the bimanual evaluator
    n, k, stages = scored["rounds"], scored["successes"], scored["failure_stages"]
    lo, hi = scored["wilson95"]

    results = dict(
        rounds=n,
        successes=k,
        success_rate=scored["success_rate"],
        wilson95=[lo, hi],
        # Kept so a run that needed many re-teleports to get going is still
        # visible: it says the stack was struggling, not that the policy was.
        stand_retries_used=len(attempts) - n,
        scripted_reference=dict(
            successes=200,
            attempts=210,
            rate=200 / 210,
            scope="historical, unpaired; not this case bank or current expert",
        ),
        initial_pose=dict(
            choice="file"
            if initial_pose.INIT_TOKEN_FILE
            else ("official" if initial_pose.INIT_TOKEN_OFFICIAL else "ours"),
            file=initial_pose.INIT_TOKEN_FILE,
            file_sha256=(
                hashlib.sha256(Path(initial_pose.INIT_TOKEN_FILE).read_bytes()).hexdigest()
                if initial_pose.INIT_TOKEN_FILE
                else None
            ),
        ),
        execution_horizon=cfg.execution_horizon,
        client=cfg.client,
        failure_stages=stages,
        seed=cfg.seed,
        manifest=str(manifest),
        stall_abort_s=cfg.stall_abort_s,
        snap=cfg.snap,
        ensemble_n=cfg.ensemble_n,
        ensemble_m=cfg.ensemble_m,
        replan_interval=cfg.replan_interval,
        grip_gain=cfg.grip_gain,
        rtc=cfg.rtc,
        transition_commit=cfg.transition_commit,
        budget_s=cfg.budget_s,
        full_budget=(cfg.stall_abort_s <= 0 and cfg.budget_s >= 90.0),
        resumed_from=done if cfg.resume else 0,
        # Parallel-measured numbers are not ladder numbers (docs/evaluation).
        # seq_ab exports EvalConfig backend.eval_parallel=true for its arms;
        # anything else is a single-instance run and records False by default.
        parallel=cfg.backend.eval_parallel,
    )
    # A generic no-close timeout may follow a bottle knocked over much
    # earlier. Always expose the existing finer trace diagnosis as well.
    from locomani.tasks.bottle.failure_taxonomy import summarize_traces

    results["trace_taxonomy"] = summarize_traces(trace_dir, max_rounds=n)
    out.write_text(json.dumps(results, indent=2))

    print(f"\n=== closed loop: {k}/{n} = {100 * k / max(n, 1):.0f}% (Wilson 95%: {100 * lo:.0f}-{100 * hi:.0f}%)")
    print("    historical expert reference: 200/210 = 95.2% (unpaired; not the current expert/case bank)")
    for s, c in sorted(stages.items(), key=lambda kv: -kv[1]):
        print(f"    failed at {s:8s} {c}")
    for leaf, count in results["trace_taxonomy"]["counts"].items():
        if leaf != "OK":
            print(f"    trace diagnosis: {leaf}: {count}")
    if results["trace_taxonomy"]["trace_count"] != n:
        print(
            f"    trace coverage: {results['trace_taxonomy']['trace_count']}/{n} (pre-VLA failures may have no trace)"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
