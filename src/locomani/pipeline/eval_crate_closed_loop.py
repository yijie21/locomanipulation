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

from locomani import conf, paths
from locomani.backends.sonic.config import SonicBackendConfig
from locomani.pipeline import stack
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
    return dict(
        rounds=len(rows),
        successes=ok,
        success_rate=ok / len(rows) if rows else 0.0,
        wilson95=wilson(ok, len(rows)),
        failure_stages=dict(stages),
        why=dict(Counter(r.get("why", "")[:70] for r in rows if not r.get("ok"))),
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--scene", type=Path, required=True, help="crate scene dir (copied into --out, like run_crate_pilot)"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--config", type=Path, default=None, help=f"CrateTaskConfig JSON; defaults to task={DEFAULT_TASK_GROUP}"
    )
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch-rounds", type=int, default=10, help="stack restart cadence, as the crate pilots")
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
    ap.add_argument("--allow-instance-0", action="store_true")
    args = ap.parse_args(argv)
    if args.instance == 0 and not args.allow_instance_0:
        ap.error("refusing INSTANCE=0: that is the collection/eval stack (pass --allow-instance-0 to override)")
    if args.rounds < 1 or args.batch_rounds < 1:
        ap.error("--rounds and --batch-rounds must be positive")
    if not (args.checkpoint / "config.json").exists():
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


def main(argv=None):
    args = parse_args(argv)
    out = args.out.resolve()
    manifest = out / "closed_loop.manifest.jsonl"
    if manifest.exists():
        raise SystemExit(f"{manifest} exists; pick another --out")
    out.mkdir(parents=True, exist_ok=True)
    scene = out / args.scene.name
    if not scene.exists():
        shutil.copytree(args.scene, scene)
    task_cfg = CrateTaskConfig.load(args.config) if args.config else conf.load_group("task", DEFAULT_TASK_GROUP)
    # vla_rounds.py runs in a separate process (a fresh stack's venv) and reads --config as a JSON
    # path (CrateTaskConfig.load); when no explicit path was given, dump the resolved YAML-group
    # config so that subprocess boundary still gets a real file.
    config_path = args.config if args.config else out / "task_config.json"
    if not args.config:
        task_cfg.dump(config_path)
    backend = SonicBackendConfig(no_encoder=True, dataset_fps=25.0)
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
    try:
        server = serve(args, out, stage)
        batches = -(-args.rounds // args.batch_rounds)
        for b in range(batches):
            tag = f"{b:02d}"
            rounds = min(args.batch_rounds, args.rounds - b * args.batch_rounds)
            videos = out / f"videos_{tag}"
            videos.mkdir(exist_ok=True)
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
                    ],
                    env=env,
                )
            finally:
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
    (out / "closed_loop.json").write_text(json.dumps(summary, indent=1))
    print(
        f"[crate-eval] {summary['successes']}/{summary['rounds']} ok  failure stages {summary['failure_stages']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
