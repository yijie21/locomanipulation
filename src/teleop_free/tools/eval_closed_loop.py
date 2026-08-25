"""Closed-loop evaluation: the finetuned VLA drives the real stack, N rounds.

    .venv_sim $ python eval_closed_loop.py --fresh 100 \
                    [--policy-port 5590] [--execution-horizon 16] \
                    [--out outputs/eval/bottle200/closed_loop.json]

PREREQUISITES (this script checks, it does not arrange):
  * the stack is up with NO_ENCODER=1 -- the token-only deployment mode. With
    an encoder loaded the deploy keeps flipping its token source back
    (g1_deploy_onnx_ref.cpp:3096) and an external stream never holds.
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
    ap.add_argument("--fresh", type=int, default=100,
                    help="rounds to run, sampled fresh from the collection "
                         "distribution")
    ap.add_argument("--execution-horizon", type=int, default=16,
                    help="blocking client only: chunk steps per replan")
    ap.add_argument("--client", choices=("async", "blocking"), default="async",
                    help="async = NVIDIA's own streaming pattern (worker-thread "
                         "inference, uninterrupted publishing, latency-"
                         "compensated chunk indexing); blocking = execute-N-"
                         "then-replan, the round-1/round-2 baseline client")
    ap.add_argument("--policy-port", type=int, default=5590)
    ap.add_argument("--task", default="pick up the water bottle")
    ap.add_argument("--target", default="water_bottle")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-dir", default=None)
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
    if "--encoder-file" in dep:
        sys.exit("deploy is running WITH its encoder. Token streaming will "
                 "not hold (initial_encoder_mode_ 0 keeps re-enabling it). "
                 "Restart the stack with NO_ENCODER=1.")

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

    # --- install the policy and run the gym's own loop ---------------------
    import grasp_gym

    if args.client == "async":
        grasp_gym.VLA_POLICY = vla_round.make_vla_policy_async(
            link, args.task, trace_dir=trace_dir)
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
            "--spawn-edge", "0.7", "--face-object",
            "--spawn-yaw-err", "20", "--spawn-lateral", "0.20",
            "--edge-jitter", "0.20"]
    print(f"[closed-loop] {args.fresh} rounds, horizon {args.execution_horizon}, "
          f"seed {args.seed}\n[closed-loop] manifest -> {manifest}")
    manifest.unlink(missing_ok=True)
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
        failure_stages=stages, seed=args.seed, manifest=str(manifest))
    out.write_text(json.dumps(results, indent=2))

    print(f"\n=== closed loop: {k}/{n} = {100 * k / max(n, 1):.0f}% "
          f"(Wilson 95%: {100 * lo:.0f}-{100 * hi:.0f}%)")
    print(f"    scripted expert on this distribution: 200/210 = 95.2%")
    for s, c in sorted(stages.items(), key=lambda kv: -kv[1]):
        print(f"    failed at {s:8s} {c}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
