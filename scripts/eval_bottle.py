#!/usr/bin/env python3
"""Run the single-arm policy in closed loop: an evaluation stack, a policy server, and the rounds.

    python scripts/eval_bottle.py outputs/runs/bottle_simple_200 --gpu 1
    python scripts/eval_bottle.py outputs/runs/bottle_simple_200 --gpu 1 --dry-run

The stack comes from the run's `eval.backend` block, which is not the collection block: the deploy
must run WITHOUT its encoder here or the token stream will not hold, and it must run WITH it during
collection or the planner's arm references are dropped. The whole `eval` block is handed to the
evaluator as one file -- copying fields by hand is how a hundred rounds once spawned at 0.7 m while
the config said 1.1-1.3 m (2026-09-17).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_common  # noqa: E402

from locomani import paths  # noqa: E402
from locomani.pipeline.eval_config import EvalConfig  # noqa: E402
from locomani.pipeline.stages import Stage  # noqa: E402
from locomani.stack import Stack, spec_from_config  # noqa: E402

SIM_PY = Path(paths.VENV_SIM_PY)
TRAIN_PY = Path(paths.THIRD_PARTY) / "Isaac-GR00T" / ".venv" / "bin" / "python"
EVALUATOR = ROOT / "src/locomani/pipeline/eval_closed_loop.py"
SERVER = ROOT / "src/locomani/pipeline/serve_policy.py"


def checkpoint_of(run_dir: Path) -> Path:
    entry = (
        json.loads((run_dir / "summary.json").read_text()).get("train") if (run_dir / "summary.json").exists() else None
    )
    if not entry or not entry.get("checkpoint"):
        raise SystemExit(f"{run_dir}/summary.json has no checkpoint; run the train stage first")
    return Path(entry["checkpoint"])


def server_argv(cfg: dict, checkpoint: Path, port: int) -> list:
    return [
        TRAIN_PY, SERVER,
        "--model-path", checkpoint,
        "--port", str(port),
        "--modality-config", ROOT / cfg["embodiment"]["gr00t_modality_config"],
        "--embodiment-tag", cfg["embodiment"]["gr00t_tag"].upper(),
    ]  # fmt: skip


def rounds_argv(cfg: dict, run_dir: Path, ev: Path) -> list:
    return [
        SIM_PY, EVALUATOR,
        "--config", ev / "eval_config.json",   # the whole block, not a hand-picked few fields
        "--set", "backend.dataset_fps=25",     # the stack is pinned to the dataset's rate
        "--reach-config", run_dir / "reach_config.json",
        "--initial-token-file", ROOT / cfg["eval"]["initial_token"],
        "--out", ev / "closed_loop.json",
        "--trace-dir", ev / "traces",
    ]  # fmt: skip


def serve(argv: list, gpu: str, log: Path) -> subprocess.Popen:
    """Start the policy server and wait until it says it is listening."""
    log.parent.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [str(a) for a in argv],
        cwd=ROOT,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            PYTHONPATH=str(ROOT / "src"),
            HF_HOME=os.environ.get("HF_HOME", os.path.join(paths.WORKSPACE, "huggingface_cache")),
        ),
        stdout=log.open("a"),
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 600
    while "listening" not in log.read_text():
        if server.poll() is not None or time.monotonic() > deadline:
            server.terminate()
            raise SystemExit(f"the policy server did not come up ({log})")
        time.sleep(3)
    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory train wrote")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_common.load_run_config(run_dir)
    checkpoint = checkpoint_of(run_dir)
    ev = run_dir / "eval"
    evaluation = EvalConfig.from_dict(cfg["eval"])  # validated here, so a bad field fails before a stack starts
    spec = spec_from_config(
        cfg, run_dir=run_dir, gpu=args.gpu, backend=cfg["eval"]["backend"], video_dir=ev / "videos", record=False
    )

    if args.dry_run:
        run_common.show("eval_stack_up", ["locomani.stack", "up", str(spec.instance)], spec.settings())
        run_common.show(
            "serve_policy", server_argv(cfg, checkpoint, evaluation.policy_port), {"CUDA_VISIBLE_DEVICES": args.gpu}
        )
        run_common.show("eval", rounds_argv(cfg, run_dir, ev))
        return 0

    if (ev / "closed_loop.json").exists():
        raise SystemExit(f"{ev / 'closed_loop.json'} exists; move the finished evaluation aside before rerunning")
    run_common.require_headroom("eval", gpu=args.gpu)
    run_common.require_deploy_patches(cfg)
    ev.mkdir(parents=True, exist_ok=True)
    evaluation.dump(ev / "eval_config.json")
    stage = Stage(run_dir)
    stack = Stack(spec, base_env=spec.settings(), record_command=stage.record, run_dir=run_dir)
    env = stack.up()
    server = None
    try:
        server = serve(server_argv(cfg, checkpoint, evaluation.policy_port), args.gpu, ev / "serve.log")
        stage.check("eval", rounds_argv(cfg, run_dir, ev), env=env)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        stack.down()

    result = json.loads((ev / "closed_loop.json").read_text())
    run_common.summarise(
        run_dir,
        "eval",
        checkpoint=str(checkpoint),
        **{k: result[k] for k in ("rounds", "successes", "success_rate") if k in result},
    )
    print(f"eval: {result.get('successes')}/{result.get('rounds')} -> {ev / 'closed_loop.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
