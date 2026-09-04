"""Closed-loop evaluation: the finetuned VLA drives the real stack, N rounds.

    .venv_sim $ python eval_closed_loop.py [--fresh 20] \
                    [--policy-port 5590] [--execution-horizon 16] \
                    [--out outputs/eval/bottle200/closed_loop.json]

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
re-implemented here, deliberately: the number this result is read against is
the scripted expert's 200/210 = 95.2% on this distribution, measured by the
collection itself, and that comparison survives only if the VLA is dropped
into the SAME sampler. (The original plan's "20 matched holdout initial
conditions" is impossible for this dataset -- the collection predated the
manifest recording initial conditions -- and same-distribution fresh draws
against 95.2% is the stronger comparison anyway.)

WHAT COMES OUT

HOW MANY ROUNDS

--fresh defaults to 20 (2026-08-28). A 20-round PERCENTAGE is not a result:
its Wilson interval is 30-40 points wide. 20 rounds exist to support PAIRED
comparison against a same-seed run, where they resolve large effects only
(6-0 or 8-1 discordant reaches p<0.05 -- roughly a doubling). Numbers that
go into the ladder, or into any document, still need --fresh 100.

Success rate with a Wilson 95% interval, and the per-stage failure breakdown
(walk / reach / close / fell / timeout) from vla_round's classifier. The
breakdown is the actionable part: failures concentrated where the script's
own failures live (close) mean faithfully inherited data, failures in the
walk mean the policy did not learn the task -- and trigger Q5's history
ablation, not more data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

REPO = Path(__file__).resolve().parents[3]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", type=int, default=20,
                    help="rounds to run, sampled fresh from the collection "
                         "distribution. 20 is the standing default (set "
                         "2026-08-28) because a 100-round run costs 4.5 h. "
                         "KNOW WHAT 20 BUYS: a rate estimate at 20 rounds "
                         "carries a Wilson 95%% interval ~30-40 points wide "
                         "(4/20 = 20%% means 8-42%%), so a bare 20-round "
                         "PERCENTAGE is not a result. What 20 rounds CAN do "
                         "is PAIRED comparison against a run with the same "
                         "seed: 6-0 or 8-1 in discordant pairs reaches "
                         "p<0.05, i.e. it resolves large effects only "
                         "(roughly a doubling). Anything smaller, or "
                         "anything going into the ladder, still needs 100.")
    ap.add_argument("--execution-horizon", type=int, default=16,
                    help="blocking client only: chunk steps per replan")
    ap.add_argument("--clean-deploy", action="store_true",
                    help="wall-clock clients: deploy engaged by SONIC_AUTO_START, "
                         "never a planner command (stack started with CLEAN_DEPLOY=1); "
                         "docs/lockstep.md '校准诊断'")
    ap.add_argument("--client", choices=("async", "lockstep", "blocking"), default="async",
                    help="async = NVIDIA's own streaming pattern (worker-thread "
                         "inference, uninterrupted publishing, latency-"
                         "compensated chunk indexing); blocking = execute-N-"
                         "then-replan, the round-1/round-2 baseline client")
    ap.add_argument("--policy-port", type=int, default=5590)
    ap.add_argument("--task", default="pick up the water bottle")
    ap.add_argument("--target", default="water_bottle")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--no-snap", action="store_true",
                    help="send the predicted token RAW instead of snapping it "
                         "to the FSQ grid -- what NVIDIA's own client does. "
                         "Round 3 open-loop measured the snap costing ~16%% "
                         "nmse_token; this runs the closed-loop half of that "
                         "ablation. Default stays snapped: the 5.9/13.7/28.0 "
                         "ladder was measured with the snap on.")
    # 1 = off. N=3 M=0.3 is a measured configuration, not an untried one:
    # docs/evaluation.md's three-way paired table has it at 35/100 against
    # 48/101 for the same model without it (+14/-27, p=0.060). The parameter
    # table in that same document calls it a deployment switch that is not
    # promoted -- so it stays off by default and is asked for explicitly.
    ap.add_argument("--ensemble", type=int, default=1, metavar="N",
                    help="blend the N most recent overlapping action chunks "
                         "for the current instant (ACT-style temporal "
                         "ensembling). 1 = off, the behaviour the "
                         "5.9/13.7/28.0/47.5 ladder was measured with. Chunks "
                         "are 40 frames and arrive every 0.4 s, so ~3 overlap; "
                         "N=3 is the natural setting. Aimed at this round's "
                         "dominant failure -- 44 of 52 reach failures are "
                         "'arm up, palm never converges', which is what "
                         "accumulated error looks like.")
    ap.add_argument("--ensemble-m", type=float, default=0.3, metavar="M",
                    help="ensemble weight decay per chunk of age (0 = newest). "
                         "M>0 favours the freshest observation, M=0 is a "
                         "uniform mean, M<0 favours older predictions the way "
                         "ACT's paper does. Ignored unless --ensemble > 1.")
    ap.add_argument("--grip-gain", type=float, default=1.0, metavar="K",
                    help="amplify the predicted gripper's deviation from "
                         "fully-open: g' = MAX - K*(MAX - g), clipped. 1.0 = "
                         "off, what every number on record was measured with. "
                         "Aimed at the one discriminator that reproduces "
                         "across every batch: successes drive the gripper to "
                         "the clip floor, `reach` failures park it at "
                         "+0.0115..+0.0195 and 0%% ever cross 0.010, inside a "
                         "band the training data occupies 3.3%% of the time. "
                         "A hard threshold does not work -- the model does not "
                         "separate near from at (measured: +0.0189 with the "
                         "palm still 0.195 m out, +0.0178 at 0.136 m) -- so "
                         "this scales instead, preserving the model's own "
                         "ordering. K=3 puts the effective trigger at 0.0178, "
                         "the p75 of the failure band.")
    ap.add_argument("--latency-frames", type=int, default=None, metavar="K",
                    help="lockstep client: FIXED simulated inference latency "
                         "in 25 fps frames (for latency sweeps). Mutually "
                         "exclusive with --latency-dist.")
    ap.add_argument("--latency-dist", default=None, metavar="JSON",
                    help="lockstep client: empirical latency distribution "
                         "from tools/latency_dist.py (drawn per inference, "
                         "seeded by round). The calibration default.")
    ap.add_argument("--rtc", action="store_true",
                    help="async client only: real-time chunking (test-time "
                         "inpainting). Each inference freezes the first "
                         "rtc.frozen frames of the new chunk to the previous "
                         "chunk's overlapping rows (frozen = rolling p90 of "
                         "measured obs->chunk latency) and exponentially "
                         "ramps the rest, removing the chunk-boundary "
                         "discontinuity. Needs a serve_policy.py new enough "
                         "to answer rtc.* keys (2026-09-01).")
    ap.add_argument("--replan-interval", type=float, default=0.4, metavar="SEC",
                    help="async client only: seconds between chunk requests. "
                         "0.4 (~12 frames) is the ladder's setting; 1.33 "
                         "executes a whole 40-frame chunk per plan, which "
                         "open-loop says is still at the noise floor.")
    ap.add_argument("--budget", type=float, default=90.0, metavar="SEC",
                    help="per-round VLA time budget. 90 is the ladder's "
                         "setting and the only one comparable with "
                         "5.9/13.7/28.0/47.5. Shorter budgets are SCREENING "
                         "settings for paired A/Bs: the bias is shared by "
                         "both arms and always downward (round 4 measured "
                         "7/48 successes landing after 60 s). Confirm any "
                         "winner at 90 s before it becomes a headline.")
    ap.add_argument("--stall-abort", type=float, default=0.0, metavar="SEC",
                    help="END a round early once the pelvis has moved less "
                         "than 3 cm for SEC seconds with no close commanded. "
                         "OFF by default and it must stay off for anything "
                         "published: measured on round 3's 100 traces, a 30 s "
                         "window fires on 70 rounds, saves ~33%% of wall clock, "
                         "and scores 1 of 28 successes as a failure -- the "
                         "error is always in that direction. 5.9/13.7/28.0 "
                         "were all measured on the full 90 s budget. Use it "
                         "for diagnostic sweeps, never for a headline number.")
    ap.add_argument("--force-resume", action="store_true",
                    help="allow --resume across a change of --seed. Off by "
                         "default: the blended sample is still valid but "
                         "nothing downstream records that it is a blend.")
    ap.add_argument("--resume", action="store_true",
                    help="keep the existing manifest and top it up to --fresh "
                         "rounds instead of starting over. The rounds are NOT "
                         "reproduced bit for bit -- grasp_gym consumes one RNG "
                         "stream, so skipping rounds shifts every later draw. "
                         "They do not need to be: a success rate is a Monte "
                         "Carlo estimate, so what has to match is the "
                         "DISTRIBUTION, and the top-up draws from the same one "
                         "with a seed offset by the rounds already done.")
    ap.add_argument("--spawn-edge", type=float, nargs="+", default=[0.7],
                    metavar="M",
                    help="spawn this far from the table; one value fixes it, "
                         "two give a range. Default 0.7 -- the whole closed-"
                         "loop ladder (5.9/13.7/28.0/47.5%) was measured "
                         "there, so changing it makes the number incomparable "
                         "to the ladder. Match it to what the checkpoint was "
                         "TRAINED on, and say which in the report.")
    ap.add_argument("--seed", type=int, default=1000,
                    help="round-sampling seed; deliberately disjoint from the "
                         "collection's seeds")
    args = ap.parse_args()

    # --- preflight, loudly -------------------------------------------------
    import vla_round

    link = vla_round.PolicyLink(port=args.policy_port)
    if not link.ping():
        sys.exit(f"no policy server on :{args.policy_port} -- start "
                 f"tools/serve_policy.py (Isaac-GR00T venv) first")

    dep = subprocess.run(["pgrep", "-af", "g1_deploy" + "_onnx_ref"],
                         capture_output=True, text=True).stdout
    if not dep.strip():
        sys.exit("deploy is not running -- bring the stack up with "
                 "NO_ENCODER=1 bash src/teleop_free/start_stack.sh")
    a18 = os.environ.get("TF_ACTION18", "0") == "1"
    if "--encoder-file" in dep and not a18:
        sys.exit("deploy is running WITH its encoder. Token streaming will "
                 "not hold (initial_encoder_mode_ 0 keeps re-enabling it). "
                 "Restart the stack with NO_ENCODER=1.")
    if a18 and "--encoder-file" not in dep:
        # The mirror image: the 18-D action drives the PLANNER channel, whose
        # upper body is only read on the encoder path (2026-09-03: a stack
        # started with NO_ENCODER silently drops the arm and crouch reference
        # and every round stalls in the up phase).
        sys.exit("TF_ACTION18=1 needs the deploy's encoder. Restart the stack "
                 "WITHOUT NO_ENCODER=1.")
    if a18 and args.clean_deploy:
        sys.exit("TF_ACTION18=1 is incompatible with --clean-deploy: the shim "
                 "drops planner commands, which is the whole 18-D channel.")

    out = Path(args.out or REPO / "outputs/eval/bottle200/closed_loop.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = out.with_suffix(".manifest.jsonl")
    trace_dir = args.trace_dir or str(out.parent / "closed_loop_traces")

    # The gym restarts the deploy every round (fresh_deploy ->
    # scripts/restart_deploy.sh, a subprocess that inherits this environment).
    # Without this the round-2 restart brings the encoder back and the token
    # stream stops holding -- the preflight above would only have caught it on
    # the NEXT invocation.
    os.environ["NO_ENCODER"] = "1"
    if args.client == "lockstep":
        # restart_deploy.sh (per-round fresh_deploy) reads this to relaunch
        # the --sim-lockstep binary; a wall-clock deploy on a lockstep sim
        # would just never tick.
        os.environ["LOCKSTEP"] = "1"

    # --- install the policy and run the gym's own loop ---------------------
    import grasp_gym

    if args.client == "lockstep":
        # Needs a stack started with LOCKSTEP=1 (lockstep_sim.py serving the
        # step RPC, deploy --sim-lockstep). Ping it rather than trusting env.
        ls = vla_round.LockstepLink()
        if not ls.ping():
            raise SystemExit(f"no lockstep sim on :{vla_round.LOCKSTEP_PORT} -- "
                             "start the stack with LOCKSTEP=1")
        latency = vla_round.load_latency_dist(args.latency_dist, args.latency_frames)
        # The clock owns a commander of its own (a second PUB into the
        # deploy's SUB) for the per-session prime + enter_streamed; the shim
        # must be installed BEFORE grasp_gym builds its Primitives.
        # publishers connect to the sim's XSUB (lockstep_sim.py) -- set before
        # ANY AgentCommander is built (this one and grasp_gym's Primitives)
        os.environ["TF_SONIC_PUB_CONNECT"] = os.environ.get(
            "TF_LOCKSTEP_XSUB_PORT",
            str(int(os.environ.get("TF_SONIC_PORT", 5556)) + 28))
        import agent_server
        clock = vla_round.LockstepClock(ls, agent_server.AgentCommander())
        vla_round.install_lockstep_shim(clock)
        clock.resume()
        print(f"[closed-loop] LOCKSTEP client: simulation clock, K ~ "
              f"{latency['source']} (first {sorted(set(latency['first']))[:6]}.. "
              f"steady {sorted(set(latency['steady']))[:8]}..). Not comparable "
              f"with wall-clock numbers until calibrated (docs/evaluation.md).")
        grasp_gym.VLA_POLICY = vla_round.make_vla_policy_lockstep(
            link, ls, clock, args.task, latency, seed=args.seed,
            trace_dir=trace_dir)
    elif args.client == "async":
        if args.clean_deploy:
            os.environ["CLEAN_DEPLOY"] = "1"      # restart_deploy.sh per round
            clock = vla_round.CleanClock()
            vla_round.install_clean_shim(clock)
            _orig_fresh = grasp_gym.fresh_deploy

            def fresh_deploy(p):
                clock.pause()
                ok = _orig_fresh(p)
                if ok:
                    clock.prime()
                clock.resume()
                return ok
            grasp_gym.fresh_deploy = fresh_deploy
            _orig_prim_init = grasp_gym.Primitives.__init__

            def prim_init(self, *a, **kw):
                _orig_prim_init(self, *a, **kw)
                if not clock.primed:
                    clock.prime()
                    clock.resume()
            grasp_gym.Primitives.__init__ = prim_init
            print("[closed-loop] CLEAN DEPLOY: auto-start, no planner command in the "
                  "deploy's lifetime, standing token between rounds.")
        grasp_gym.VLA_POLICY = vla_round.make_vla_policy_async(
            link, args.task, inference_interval=args.replan_interval,
            trace_dir=trace_dir)
    else:
        grasp_gym.VLA_POLICY = vla_round.make_vla_policy(
            link, args.task, execution_horizon=args.execution_horizon,
            trace_dir=trace_dir)

    # The same spawn distribution the collection ran (collect_pilot.sh),
    # WITHOUT --record (no dataset is being written) and WITHOUT --until-ok
    # (failures are the measurement, not something to retry past).
    argv = ["grasp_gym",
            "--rounds", str(args.fresh),
            "--targets", args.target,
            "--seed", str(args.seed),
            "--manifest", str(manifest),
            *(["--spawn-edge"] + [str(v) for v in args.spawn_edge]),
            "--face-object",
            "--spawn-yaw-err", "20", "--spawn-lateral", "0.20",
            "--edge-jitter", "0.20"]
    done = 0
    if args.resume and manifest.exists():
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
            if prev is not None and prev != args.seed and not args.force_resume:
                sys.exit(
                    f"--resume: the {done} rounds already in\n  {manifest}\n"
                    f"were run with --seed {prev}, not {args.seed}. Re-run with "
                    f"--seed {prev}, point --out somewhere else, or pass "
                    f"--force-resume if mixing them is genuinely what you want.")
    todo = max(0, args.fresh - done)
    if args.resume and done:
        # Offset so the top-up cannot redraw the rounds already measured. The
        # sample is still i.i.d. from the same distribution, which is all a
        # rate estimate needs.
        argv[argv.index("--seed") + 1] = str(args.seed + done)
        argv[argv.index("--rounds") + 1] = str(todo)
        print(f"[closed-loop] resuming: {done} rounds already in the manifest, "
              f"{todo} to go (seed {args.seed}+{done})")
        if todo == 0:
            print("[closed-loop] nothing to do; aggregating what is there")
    else:
        manifest.unlink(missing_ok=True)

    # NOT inside the ensemble branch: caught 2026-08-30 with an arm already
    # running, where --grip-gain 3.0 was silently a no-op because --ensemble
    # defaults to 1. Every vla_round toggle gets its own guard.
    vla_round.GRIP_GAIN = args.grip_gain
    if args.grip_gain != 1.0:
        print(f"[closed-loop] GRIPPER GAIN {args.grip_gain} -- effective close "
              f"trigger at raw g < "
              f"{0.0245 - (0.0245 - 0.005) / args.grip_gain:.4f} "
              f"(failure band measured p25 0.0159 / p50 0.0168 / p75 0.0178)")
    if args.rtc:
        if args.client not in ("async", "lockstep"):
            raise SystemExit("--rtc is implemented for the async/lockstep clients only")
        vla_round.RTC_ON = True
        print("[closed-loop] RTC on: frozen-prefix inpainting at every "
              "replan; frozen = p90 latency, overlap = whole remainder, "
              "ramp server-side (serve_policy --rtc-ramp-rate).")
    if args.ensemble > 1:
        vla_round.ENSEMBLE_N = args.ensemble
        vla_round.ENSEMBLE_M = args.ensemble_m
        print(f"[closed-loop] TEMPORAL ENSEMBLING on: N={args.ensemble} "
              f"M={args.ensemble_m} -- not comparable with the ladder, which "
              f"ran without it.")
    if args.budget != 90.0:
        vla_round.TIMEOUT_S = args.budget
        print(f"[closed-loop] BUDGET {args.budget:.0f}s (ladder ran 90s) -- "
              f"screening setting, biased downward; not a headline number.")
    if args.no_snap:
        vla_round.SNAP = False
        print("[closed-loop] FSQ SNAP OFF -- raw floats on the wire (NVIDIA's "
              "own behaviour). Not comparable with the snapped ladder.")
    if args.replan_interval != 0.4:
        print(f"[closed-loop] replan interval {args.replan_interval}s (ladder "
              f"ran 0.4s) -- diagnostic setting.")
    if args.stall_abort > 0:
        vla_round.STALL_ABORT_S = args.stall_abort
        print(f"[closed-loop] STALL-ABORT {args.stall_abort:.0f}s IS ON -- this "
              f"run is NOT comparable with the 5.9/13.7/28.0 ladder (measured "
              f"on the full {vla_round.TIMEOUT_S:.0f}s budget). Diagnostic use "
              f"only.")

    print(f"[closed-loop] {args.fresh} rounds, horizon {args.execution_horizon}, "
          f"seed {args.seed}\n[closed-loop] manifest -> {manifest}")
    if todo > 0:
        old_argv = sys.argv
        sys.argv = argv
        try:
            grasp_gym.main()
        finally:
            sys.argv = old_argv

    # --- aggregate ----------------------------------------------------------
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    n = len(rows)
    k = sum(1 for r in rows if r["ok"])
    lo, hi = wilson(k, n)
    stages = {}
    for r in rows:
        if r["ok"]:
            continue
        why = r.get("why", "")
        stage = why.split(":")[0].replace("[vla] ", "") if "[vla]" in why else "pre-vla"
        stages[stage] = stages.get(stage, 0) + 1

    results = dict(
        rounds=n, successes=k, success_rate=k / n if n else 0.0,
        wilson95=[lo, hi],
        scripted_reference=dict(successes=200, attempts=210, rate=200 / 210),
        execution_horizon=args.execution_horizon, client=args.client,
        failure_stages=stages, seed=args.seed, manifest=str(manifest),
        stall_abort_s=args.stall_abort, snap=not args.no_snap,
        ensemble_n=args.ensemble, ensemble_m=args.ensemble_m,
        replan_interval=args.replan_interval, grip_gain=args.grip_gain,
        budget_s=args.budget,
        full_budget=(args.stall_abort <= 0 and args.budget >= 90.0),
        resumed_from=done if args.resume else 0,
        # Parallel-measured numbers are not ladder numbers (docs/evaluation).
        # seq_ab exports EVAL_PARALLEL=1 for its arms; anything else is a
        # single-instance run and records False by default.
        parallel=os.environ.get("EVAL_PARALLEL", "0") == "1")
    out.write_text(json.dumps(results, indent=2))

    print(f"\n=== closed loop: {k}/{n} = {100 * k / max(n, 1):.0f}% "
          f"(Wilson 95%: {100 * lo:.0f}-{100 * hi:.0f}%)")
    print(f"    scripted expert on this distribution: 200/210 = 95.2%")
    for s, c in sorted(stages.items(), key=lambda kv: -kv[1]):
        print(f"    failed at {s:8s} {c}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
