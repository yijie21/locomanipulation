#!/usr/bin/env python3
"""Serve a run's checkpoint for closed-loop evaluation, in the foreground, until interrupted.

    python -m locomani.pipeline.run_vla_server outputs/runs/bottle_simple_200 --gpu 1

`scripts/run_vla_server.sh` is one line over this. It is terminal 2 of three: terminal 1 holds the
stack (`scripts/start_stack.sh eval`), terminal 3 drives the rounds. Ctrl-C here frees the card.

Which checkpoint, which modality config and which embodiment tag are read from the run, not passed
in: they are recorded in the run's own `summary.json` and `config.yaml`, and a second hand-typed
copy of them is a way for an evaluation to be served by something other than what it says it is.

The server's output is both shown here and appended to `<run>/eval/serve.log`, because the line
the round driver waits for ("listening") and the load failures worth reading afterwards are the
same stream.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context

ROOT = Path(paths.REPO)
TRAIN_PY = Path(paths.THIRD_PARTY) / "Isaac-GR00T" / ".venv" / "bin" / "python"
SERVER = ROOT / "src/locomani/pipeline/serve_policy.py"


def wait_for_server(port: int, seconds: float = 900.0) -> None:
    """Loading a checkpoint takes minutes, so wait -- but say what is missing instead of hanging mute."""
    end = time.monotonic() + seconds
    announced = False
    while time.monotonic() < end:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", int(port))) == 0:
                return
        if not announced:
            print(f"waiting for the policy server on port {port} ...", flush=True)
            announced = True
        time.sleep(3)
    raise SystemExit(
        f"nothing is listening on the policy port {port}; start the server in another terminal:\n"
        f"  bash scripts/run_vla_server.sh <run> --gpu <card>"
    )


def _interrupt(signum, frame) -> None:
    """Funnel SIGTERM into the same path Ctrl-C takes, so there is one place that stops the child."""
    raise KeyboardInterrupt


def server_argv(cfg: dict, checkpoint: Path, port: int) -> list:
    return [
        TRAIN_PY, SERVER,
        "--model-path", checkpoint,
        "--port", str(port),
        "--modality-config", ROOT / cfg["embodiment"]["gr00t_modality_config"],
        "--embodiment-tag", cfg["embodiment"]["gr00t_tag"].upper(),
    ]  # fmt: skip


def server_env(gpu: str) -> dict:
    # PYTHONUNBUFFERED: the "listening" line the round driver waits for is a plain print in
    # PolicyServer.run(), and a block-buffered stdout keeps it in the child for minutes.
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=str(ROOT / "src"),
        PYTHONUNBUFFERED="1",
        HF_HOME=os.environ.get("HF_HOME", os.path.join(paths.WORKSPACE, "huggingface_cache")),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory train wrote")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_context.load_run_config(run_dir)
    port = run_context.evaluation_of(cfg).policy_port
    checkpoint = run_context.checkpoint_of(run_dir)
    command = server_argv(cfg, checkpoint, port)

    if args.dry_run:
        run_context.show("serve_policy", command, {"CUDA_VISIBLE_DEVICES": args.gpu})
        return 0

    log = run_dir / "eval" / "serve.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"serving {checkpoint} on port {port} (log: {log})", flush=True)
    with log.open("a") as sink:
        # Own session, so the child is never killed behind this process's back and stopping it is
        # always this function's job -- which it then always does, in the finally below.
        server = subprocess.Popen(
            [str(a) for a in command],
            cwd=ROOT,
            env=server_env(args.gpu),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        # SIGTERM as well as Ctrl-C: `run_overnight.sh` stops this with kill, and handling only
        # KeyboardInterrupt left serve_policy.py holding ~7 GB of a shared card after the chain
        # finished (2026-09-21). The manual three-terminal path never showed it, because Ctrl-C
        # reaches the whole foreground group.
        signal.signal(signal.SIGTERM, _interrupt)
        try:
            for line in server.stdout:  # shown here and kept in the run, one stream
                sys.stdout.write(line)
                sink.write(line)
                sink.flush()
        except KeyboardInterrupt:
            pass
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        return server.returncode


if __name__ == "__main__":
    raise SystemExit(main())
