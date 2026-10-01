"""Closed-loop evaluation of a crate (bimanual) VLA checkpoint.

The crate counterpart of run_learn_pilot.py's eval stage: start a token-only
stack on the chosen instance/GPU, serve the checkpoint with the bimanual
modality config on its own port, run `locomani.tasks.crate.vla_rounds` in this
process on that stack's environment in batches (the stack restarts per batch, as
every crate pilot does), stop everything, and aggregate the manifest into
closed_loop.json with the same fields the single-arm evaluation reports.

    .venv_sim/bin/python -m locomani.pipeline.eval_crate_closed_loop \\
        --checkpoint outputs/crate_vla200_record_20260914/model/checkpoint-10000 \\
        --scene outputs/scene/crate_table30_depth04 --out outputs/crate_eval_10 \\
        --instance 1 --gpu 1 --serve-gpu 1 --rounds 10 --seed 6000

A checkpoint trained on another machine records that machine's Cosmos-Reason2-2B
directory; add `--backbone-path <local Cosmos-Reason2-2B dir>` to serve it here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from locomani import paths, profiles
from locomani.backends.sonic.config import SonicBackendConfig
from locomani.pipeline import run_vla_server, stack
from locomani.pipeline.closed_loop import score
from locomani.pipeline.provenance import file_digests, write_config_snapshot
from locomani.pipeline.stages import Stage
from locomani.tasks.crate.config import CrateTaskConfig

ROOT = Path(paths.REPO)
SRC = ROOT / "src/locomani"
TRAIN = ROOT / "third_party/Isaac-GR00T/.venv/bin/python"
MODALITY = ROOT / "configs/gr00t_sonic_3cam_bimanual.py"
DEFAULT_TOKEN = ROOT / "configs/deployment/crate_initial_token.json"
DEFAULT_TASK_GROUP = "crate_visual_low_reach"  # configs/task/crate_visual_low_reach.yaml


def aggregate(manifest: Path) -> dict:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    result = dict(
        score(rows),  # one definition with the single-arm evaluator (pipeline/closed_loop.py)
        why=dict(Counter(r.get("why", "")[:70] for r in rows if not r.get("ok"))),
    )
    # Lifted rounds of a config with visual.retreat_distance also walk back (vla_rounds.RetreatWatch): the
    # collection's own success is those that did so with the load held.
    watched = [r for r in rows if r.get("retreat") is not None]
    if watched:
        result["retreated"] = sum(1 for r in watched if r.get("retreat_ok"))
        result["retreat_reasons"] = dict(Counter(r["retreat"]["reason"] for r in watched))
    return result


# The EvalConfig fields this evaluator actually reads. Everything else in the block
# is not implemented here: `client` is refused outright because a config that says
# "async" while the loop runs blocking is exactly the kind of quiet lie the 2026-09-18
# cold-start run caught (eval_config.json said async, closed_loop.json said blocking),
# and the rest is written to `ignored_eval_fields.json` so "declared" and "delivered"
# can be compared without reading this file.
HONOURED = ("rounds", "seed", "policy_port", "execution_horizon", "budget_s", "initial_token", "image_source")
PINNED = dict(client="blocking")


def apply_eval_config(ap, args) -> None:
    """Take the run's whole eval block, then say out loud what is not used."""
    from locomani.pipeline.eval_config import EvalConfig

    evaluation = EvalConfig.load(args.eval_config)
    for field, value in PINNED.items():
        if getattr(evaluation, field) != value:
            ap.error(
                f"{args.eval_config}: {field}={getattr(evaluation, field)!r}, but the crate evaluator only "
                f"implements {field}={value!r}. Set it in the experiment file rather than letting the run "
                f"record a value it never used."
            )
    args.rounds = evaluation.rounds
    args.seed = evaluation.seed
    args.port = evaluation.policy_port
    args.execution_horizon = evaluation.execution_horizon
    args.budget_s = evaluation.budget_s
    args.initial_token_file = ROOT / evaluation.initial_token
    args.image_source = evaluation.image_source
    args.ignored_eval_fields = {
        name: getattr(evaluation, name)
        for name in evaluation.to_dict()
        if name not in HONOURED and name not in PINNED and name != "schema_version"
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "run",
        nargs="?",
        type=Path,
        default=None,
        help="the run directory train wrote. Given this, the checkpoint, scene, task config and "
        "eval block are read from the run itself and the result is recorded in its summary.json; "
        "the explicit options below remain for evaluating something that is not a run",
    )
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    ap.add_argument("--checkpoint", type=Path, required=False)
    ap.add_argument(
        "--scene", type=Path, help="crate scene dir (copied into --out); read from the run when one is given"
    )
    ap.add_argument("--out", type=Path, help="output directory; read from the run when one is given")
    ap.add_argument(
        "--config", type=Path, default=None, help=f"CrateTaskConfig JSON; defaults to task={DEFAULT_TASK_GROUP}"
    )
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch-rounds", type=int, default=10, help="stack restart cadence, as the crate pilots")
    ap.add_argument(
        "--attach",
        action="store_true",
        help="use a stack and a policy server that are already running (scripts/start_stack.sh eval and "
        "scripts/run_vla_server.sh), instead of starting and stopping both here. The deploy is "
        "restarted between batches rather than the whole stack, which is what the per-batch "
        "restart was for: the exporter is not running in an evaluation.",
    )
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--instance", type=int, default=1)
    ap.add_argument("--gpu", default="1", help="CUDA/EGL device for the stack")
    ap.add_argument("--serve-gpu", default="1", help="CUDA_VISIBLE_DEVICES for serve_policy")
    ap.add_argument(
        "--backbone-path",
        type=Path,
        default=None,
        help="local Cosmos-Reason2-2B directory for serve_policy, needed when the checkpoint was trained "
        "on another machine and records a backbone path that does not exist here",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=5700,
        help="policy server port. Keep it outside the instance blocks (5555-5592 for instances 0-1, "
        "locomani.sim.ports): 5591 collided with instance 1's privileged-state port and the bottle eval's "
        "5590 is instance 1's keyboard port",
    )
    ap.add_argument("--execution-horizon", type=int, default=16)
    ap.add_argument("--budget-s", type=float, default=90.0)
    ap.add_argument("--initial-token-file", type=Path, default=DEFAULT_TOKEN)
    ap.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help="the run's whole EvalConfig JSON. Fields this evaluator implements are taken from it; "
        "every other field is refused or recorded, never silently dropped (see HONOURED)",
    )
    ap.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="hand the deploy back to the planner between rounds, as this line did before "
        "2026-09-18. Only for reproducing a historical number: measured over 30 rounds it lets "
        "the settle drift the base a median 0.09 m and up to 0.83 m before the policy is asked",
    )
    ap.set_defaults(clean=True)
    ap.set_defaults(image_source="mujoco")  # an --eval-config (a run's eval block) may name the Isaac mirror
    ap.add_argument("--allow-instance-0", action="store_true")
    ap.add_argument(
        "--keep-stack", action="store_true", help="with a run directory: leave the stack running when the rounds end"
    )
    args = ap.parse_args(argv)
    if args.eval_config:
        apply_eval_config(ap, args)
    if args.run is not None:
        # A run directory supplies the instance, the checkpoint and the scene; validating the
        # defaults of options it is about to overwrite would refuse a perfectly good run.
        return args
    if args.instance == 0 and not args.allow_instance_0:
        ap.error("refusing INSTANCE=0: that is the collection/eval stack (pass --allow-instance-0 to override)")
    if args.rounds < 1 or args.batch_rounds < 1:
        ap.error("--rounds and --batch-rounds must be positive")
    if args.checkpoint is not None and not (args.checkpoint / "config.json").exists():
        ap.error(f"no checkpoint at {args.checkpoint}")
    if args.backbone_path is not None and not (args.backbone_path / "config.json").exists():
        ap.error(f"no backbone model at {args.backbone_path}")
    return args


def serve_command(args) -> list[str]:
    cmd = [
        str(TRAIN),
        str(SRC / "pipeline/serve_policy.py"),
        "--model-path",
        str(args.checkpoint),
        "--port",
        str(args.port),
        "--modality-config",
        str(MODALITY),
    ]
    if args.backbone_path is not None:
        cmd += ["--backbone-path", str(args.backbone_path)]
    return cmd


def serve_env(args) -> dict:
    # PYTHONUNBUFFERED: the "listening" line serve() waits for is a plain print in
    # PolicyServer.run(), and stdout into serve.log is block-buffered. Without this the
    # line stays in the child's buffer, the 600 s wait expires and a healthy server is
    # reported as "did not come up" (2026-09-16). run_learn_pilot.py sets the same.
    return dict(stack.os.environ, CUDA_VISIBLE_DEVICES=str(args.serve_gpu), PYTHONUNBUFFERED="1")


def serve(args, out: Path, stage: Stage) -> subprocess.Popen:
    log = (out / "serve.log").open("a")
    server = subprocess.Popen(
        serve_command(args),
        cwd=ROOT,
        env=serve_env(args),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    end = time.monotonic() + 600
    while "listening" not in (out / "serve.log").read_text():
        if server.poll() is not None or time.monotonic() > end:
            raise RuntimeError(f"policy server did not come up; see {out / 'serve.log'}")
        time.sleep(3)
    return server


def evaluate_run(args):
    """Evaluate the run `args.run` names, filling the options in from the run itself.

    This is what `run_eval_bimanual.py` was until 2026-09-22. The rounds below attach the stack
    (`--attach`) and restart its deploy between batches. The admission (this run's stack, the
    deploy patches, the headroom left beside what the stack already holds), the teardown and the
    summary are `pipeline/closed_loop.py`, shared with the single-arm evaluator (TODO 1.18).
    """
    from locomani.pipeline import closed_loop
    from locomani.tasks.crate import prepare

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    if args.dry_run:
        return closed_loop.show(run_dir)
    run = closed_loop.attach(run_dir, args.run, args.gpu)

    args.attach = True  # the stack and the policy server are already up, in the other terminals
    args.checkpoint = run.checkpoint
    args.scene = prepare.scene_name(run.cfg, run_dir)
    args.out = run.ev
    args.config = run_dir / "task_config.json"
    args.eval_config = run.ev / "eval_config.json"
    args.batch_rounds = run.cfg["collect"]["batch_rounds"]  # deploy restart cadence, as the crate pilots
    args.instance = run.instance
    args.serve_gpu = args.gpu
    apply_eval_config(argparse.ArgumentParser(), args)
    try:
        _evaluate(args)
    finally:
        closed_loop.finish(run, keep_stack=args.keep_stack)
    closed_loop.record(run)
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.run is not None:
        return evaluate_run(args)
    missing = [n for n in ("checkpoint", "scene", "out") if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"give a run directory, or every one of {', '.join('--' + n for n in missing)}")
    return _evaluate(args)


def _evaluate(args):
    out = args.out.resolve()
    manifest = out / "closed_loop.manifest.jsonl"
    if manifest.exists():
        raise SystemExit(f"{manifest} exists; pick another --out")
    out.mkdir(parents=True, exist_ok=True)
    scene = out / args.scene.name
    if not scene.exists():
        shutil.copytree(args.scene, scene)
    task_cfg = CrateTaskConfig.load(args.config) if args.config else profiles.load("task", DEFAULT_TASK_GROUP)
    if not args.config:  # the record of what was evaluated when it came from the YAML group
        task_cfg.dump(out / "task_config.json")
    # clean_deploy gives the child SONIC_AUTO_START (stack/manager.py:199), so the deploy
    # engages itself and no planner command is needed to start it. Paired with --clean below:
    # ledger/bottle.md:9 calls a planner settle before enter_streamed the `degraded` deploy,
    # which executes walking tokens at ~1/3. This line had been degraded in every round.
    backend = SonicBackendConfig(no_encoder=True, dataset_fps=25.0, clean_deploy=args.clean)
    stage = Stage(out)
    started = time.time()
    write_config_snapshot(
        out,
        task_cfg=task_cfg,
        backend_cfg=backend,
        started=started,
        inputs=dict(
            scene=file_digests(scene),
            checkpoint=str(args.checkpoint),
            initial_token=hashlib.sha256(Path(args.initial_token_file).read_bytes()).hexdigest(),
        ),
    )
    (out / "inputs.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    server = None
    live = None
    try:
        if args.attach:
            from locomani.stack import Stack

            try:
                live = Stack.attach(args.instance)
            except FileNotFoundError:
                raise SystemExit(
                    f"no stack on instance {args.instance}; start one in another terminal:\n"
                    f"  bash scripts/start_stack.sh eval <run> --gpu {args.gpu}"
                )
            if live.spec.auto_start != args.clean:
                # Clean rounds drop every planner command, the start included, so only
                # SONIC_AUTO_START engages the deploy; unclean rounds start it themselves and an
                # auto-started deploy fights them. `start_stack.sh eval` decides this from
                # `lines.Line.eval_backend`; a stack from anywhere else may disagree (2026-09-23).
                raise SystemExit(
                    f"the stack on instance {args.instance} runs auto_start={live.spec.auto_start}, but these "
                    f"rounds are {'clean' if args.clean else '--no-clean'}; restart the stack to match"
                )
            run_vla_server.wait_for_server(args.port)
        else:
            server = serve(args, out, stage)
        batches = -(-args.rounds // args.batch_rounds)
        for b in range(batches):
            tag = f"{b:02d}"
            rounds = min(args.batch_rounds, args.rounds - b * args.batch_rounds)
            videos = out / f"videos_{tag}"
            videos.mkdir(exist_ok=True)
            if args.attach:
                # The video directory is fixed when the sim starts, so an attached run writes
                # every batch into the one the stack was given (eval/videos), not videos_<batch>.
                if b and not live.restart_deploy():
                    raise RuntimeError(f"deploy restart before batch {tag} failed; see {live.root / 'deploy.log'}")
                env = dict(live.env)
            else:
                env = stack.start_stack(
                    backend,
                    task_cfg,
                    instance=args.instance,
                    gpu=str(args.gpu),
                    video_dir=str(videos),
                    extra_env={"LOCOMANI_SIM_DIR": str(scene), "NO_ENCODER": "1"},
                    runner=stage.run,
                    name=f"start_stack_{tag}",
                    receipt_path=out / "deploy_receipts.jsonl",
                )
            # In this process since 2026-10-01 (TODO 1.16). It used to be a `.venv_sim` child of this
            # `.venv_sim` process, handed `env` as its whole environment; adopting `env` is the same hand-off
            # (the crate collection does it the same way), and `vla_rounds.run` closes its clients and clock.
            # Adopt BEFORE importing vla_rounds: its imports build the IK from the scene `locomani.paths`
            # names, and adopt_sim_dir refuses to move a scene a module already holds (the gate's first try).
            from locomani.pipeline import run_context

            run_context.adopt_env(env)
            from locomani.tasks.crate import vla_rounds

            batch = dict(
                rounds=rounds,
                seed=args.seed + b * args.batch_rounds,
                manifest=manifest,
                initial_token_file=args.initial_token_file,
                policy_port=args.port,
                execution_horizon=args.execution_horizon,
                budget_s=args.budget_s,
                trace_dir=out / "traces",
                image_source=args.image_source,
                clean=args.clean,
                round_offset=b * args.batch_rounds,
            )
            try:
                stage.record_config(f"rounds_{tag}", dict(batch, config=task_cfg.to_dict()), env=env)
                with stage.timed(f"rounds_{tag}"):
                    vla_rounds.run(task_cfg, **batch)
            finally:
                if not args.attach:
                    stack.stop_instance(args.instance, runner=stage.run)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(15)
            except subprocess.TimeoutExpired:
                server.kill()
    summary = dict(
        aggregate(manifest),
        checkpoint=str(args.checkpoint),
        modality_config=str(MODALITY),
        seed=args.seed,
        execution_horizon=args.execution_horizon,
        budget_s=args.budget_s,
        client="blocking",
        manifest=str(manifest),
        backend=dataclasses.asdict(backend),
    )
    ignored = getattr(args, "ignored_eval_fields", None)
    if ignored:
        (out / "ignored_eval_fields.json").write_text(json.dumps(ignored, indent=1, default=str))
        print(f"[crate-eval] {len(ignored)} eval fields not implemented here: {', '.join(sorted(ignored))}", flush=True)
    (out / "closed_loop.json").write_text(json.dumps(summary, indent=1))
    print(
        f"[crate-eval] {summary['successes']}/{summary['rounds']} ok  failure stages {summary['failure_stages']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
