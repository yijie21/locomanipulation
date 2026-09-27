"""Closed-loop evaluation of a crate (bimanual) VLA checkpoint.

The crate counterpart of run_learn_pilot.py's eval stage: start a token-only
stack on the chosen instance/GPU, serve the checkpoint with the bimanual
modality config on its own port, run `locomani.tasks.crate.vla_rounds` inside
that stack's environment in batches (the stack restarts per batch, as every
crate pilot does), stop everything, and aggregate the manifest into
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
import math
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from locomani import paths, profiles
from locomani.backends.sonic.config import SonicBackendConfig
from locomani.pipeline import run_vla_server, stack
from locomani.pipeline.provenance import file_digests, write_config_snapshot
from locomani.pipeline.stages import Stage
from locomani.tasks.crate.config import CrateTaskConfig

ROOT = Path(paths.REPO)
SRC = ROOT / "src/locomani"
SIM = ROOT / "third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"
TRAIN = ROOT / "third_party/Isaac-GR00T/.venv/bin/python"
MODALITY = ROOT / "configs/gr00t_sonic_3cam_bimanual.py"
DEFAULT_TOKEN = ROOT / "configs/deployment/crate_initial_token.json"
DEFAULT_TASK_GROUP = "crate_visual_low_reach"  # configs/task/crate_visual_low_reach.yaml


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def failure_stage(why: str) -> str:
    return why.split(":")[0].replace("[vla] ", "") if "[vla]" in why else "pre-vla"


def aggregate(manifest: Path) -> dict:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    ok = sum(1 for r in rows if r.get("ok"))
    stages = Counter(failure_stage(r.get("why", "")) for r in rows if not r.get("ok"))
    result = dict(
        rounds=len(rows),
        successes=ok,
        success_rate=ok / len(rows) if rows else 0.0,
        wilson95=wilson(ok, len(rows)),
        failure_stages=dict(stages),
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
    (`--attach`) and restart its deploy between batches; this function checks the stack is this
    run's, admits the evaluation against what the stack already holds, and takes the stack down
    at the end -- as the single-arm evaluator does. Until 2026-09-23 it did none of the three: the
    headroom check charged the stack's own cores a second time, and an unattended run left the
    stack up after its rounds.
    """
    from locomani.pipeline import run_context
    from locomani.tasks.crate import prepare

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_context.load_run_config(run_dir)
    from locomani.pipeline.eval_config import EvalConfig

    evaluation = EvalConfig.from_dict(cfg["eval"])
    ev = run_dir / "eval"
    collect = cfg["collect"]

    if args.dry_run:
        run_context.checkpoint_of(run_dir)
        run_context.show_config("eval", evaluation.to_dict())
        print(f"\n  attaches to instance {collect['instance']}, policy port {evaluation.policy_port}")
        return 0

    checkpoint = run_context.checkpoint_of(run_dir)
    from locomani.stack import Stack

    try:
        stack = Stack.attach(int(collect["instance"]))
    except FileNotFoundError:
        raise SystemExit(
            f"no stack on instance {collect['instance']}; start one in another terminal:\n"
            f"  bash scripts/start_stack.sh eval {args.run} --gpu {args.gpu}"
        )
    run_context.require_stack_of(stack, run_dir, "eval")
    run_context.require_headroom(  # an evaluation runs no exporter; the live stack is already paid for
        "eval", gpu=args.gpu, record=False, already=run_context.group_threads(stack.pids.values())
    )
    ev.mkdir(parents=True, exist_ok=True)
    evaluation.dump(ev / "eval_config.json")

    args.attach = True  # the stack and the policy server are already up, in the other terminals
    args.checkpoint = checkpoint
    args.scene = prepare.scene_name(cfg, run_dir)
    args.out = ev
    args.config = run_dir / "task_config.json"
    args.eval_config = ev / "eval_config.json"
    args.batch_rounds = collect["batch_rounds"]  # deploy restart cadence, as the crate pilots
    args.instance = int(collect["instance"])
    args.serve_gpu = args.gpu
    apply_eval_config(argparse.ArgumentParser(), args)
    try:
        _evaluate(args)
    finally:
        if not args.keep_stack:
            try:  # re-read the state file: the rounds restarted the deploy, so its pid is new
                Stack.attach(args.instance).down()
            except FileNotFoundError:
                pass  # already down

    result = json.loads((ev / "closed_loop.json").read_text())
    run_context.summarise(
        run_dir,
        "eval",
        checkpoint=str(checkpoint),
        **{k: result[k] for k in ("rounds", "successes", "success_rate", "retreated") if k in result},
    )
    print(f"eval: {result.get('successes')}/{result.get('rounds')} -> {ev / 'closed_loop.json'}")
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
    # vla_rounds.py runs in a separate process (a fresh stack's venv) and reads --config as a JSON
    # path (CrateTaskConfig.load); when no explicit path was given, dump the resolved YAML-group
    # config so that subprocess boundary still gets a real file.
    config_path = args.config if args.config else out / "task_config.json"
    if not args.config:
        task_cfg.dump(config_path)
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
            try:
                stage.check(
                    f"rounds_{tag}",
                    [
                        SIM,
                        "-m",
                        "locomani.tasks.crate.vla_rounds",
                        "--rounds",
                        rounds,
                        "--seed",
                        args.seed + b * args.batch_rounds,
                        "--manifest",
                        manifest,
                        "--config",
                        config_path,
                        "--policy-port",
                        args.port,
                        "--execution-horizon",
                        args.execution_horizon,
                        "--budget-s",
                        args.budget_s,
                        "--initial-token-file",
                        args.initial_token_file,
                        "--trace-dir",
                        out / "traces",
                        "--image-source",
                        args.image_source,
                        *([] if args.clean else ["--no-clean"]),
                        "--round-offset",
                        b * args.batch_rounds,
                    ],
                    env=env,
                )
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
