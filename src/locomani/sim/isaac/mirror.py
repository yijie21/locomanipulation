"""The Isaac mirror: the eval sim's newest MuJoCo state rendered in Isaac, the dataset cameras out, when the
policy asks (TODO 4.7.9). The protocol, the request handling and the policy client are `locomani.sim.mirror`;
this is the Isaac server over `view.IsaacView`. Measured on the warehouse, three cameras, one request per 0.4 s
on GPU1: p50 89 / p90 108 ms (docs/ledger/crate.md, 2026-09-24); 2x supersampling since 2026-09-29, p50 218 ms.

    bash scripts/start_stack.sh eval <run> --gpu 0          # terminal 1
    bash scripts/run_isaac_mirror.sh <run> --gpu 1          # this; after the stack, before the rounds

Every render is appended to `<run>/eval/isaac_mirror.jsonl`. Kit's imports are inside `_run`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
import zmq

from locomani.sim.mirror import STREAMS, serve, state_problem


def _stop(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    from locomani.pipeline import run_context
    from locomani.stack import Stack

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory whose eval stack is up")
    ap.add_argument("--gpu", required=True, help="the card Isaac renders on; not the stack's")
    ap.add_argument("--dry-run", action="store_true", help="say what would be served, start nothing")
    args = ap.parse_args(argv)

    run_dir = (args.run if args.run.is_absolute() else Path.cwd() / args.run).resolve()
    cfg = run_context.load_run_config(run_dir)
    source = cfg["images"]["source"]
    if source != "isaac":
        raise SystemExit(
            f"{run_dir.name}: images.source is {source!r}; the policy would never ask this mirror. "
            "Set images.source: isaac in the experiment file"
        )
    isaac = run_context.isaac_scene(cfg)
    instance = int(cfg["collect"]["instance"])
    try:
        stack = Stack.attach(instance)
    except FileNotFoundError:
        raise SystemExit(
            f"no stack on instance {instance}; the mirror renders the running sim's state, so start it first:\n"
            f"  bash scripts/start_stack.sh eval {args.run} --gpu <card>"
        )
    run_context.require_stack_of(stack, run_dir, "eval")
    scene_xml = stack.spec.scene_path() / "scene_locomani.xml"
    ports = stack.spec.ports()
    plan = dict(
        scene_xml=str(scene_xml),
        isaac_scene=isaac["dir"],
        rtx_settings=isaac["rtx_settings"],
        supersample=isaac["supersample"],
        streams=list(STREAMS),
        privileged_port=ports["TF_PRIVILEGED_PORT"],
        mirror_port=ports["TF_ISAAC_MIRROR_PORT"],
        gpu=str(args.gpu),
        stack_gpu=str(stack.spec.gpu),
    )
    if args.dry_run:
        run_context.show_config("isaac_mirror", plan)
        return 0
    if str(args.gpu) == str(stack.spec.gpu):
        print(f"WARNING: the mirror and the stack share GPU {args.gpu}; the render then competes with the sim")
    return _run(run_dir, plan)


def _run(run_dir: Path, plan: dict) -> int:
    """Kit's lifetime. On failure os._exit(1) BEFORE app.close(): close() ends the process with status 0."""
    os.environ.update(CUDA_VISIBLE_DEVICES=plan["gpu"], OMNI_KIT_ACCEPT_EULA="YES")
    from locomani.sim.isaac import stage as S
    from locomani.sim.isaac.view import IsaacView
    from locomani.sim.privileged_pub import PrivilegedSubscriber

    log_path = run_dir / "eval" / "isaac_mirror.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = S.StageCfg(
        scene_dir=plan["isaac_scene"],
        robot_mjcf=plan["scene_xml"],
        absolute_camera=True,
        supersample=plan["supersample"],
    )
    app = S.start_app(cfg)
    try:
        view = IsaacView(cfg, plan["scene_xml"], plan["streams"], plan["rtx_settings"])
        nq = view.replay.model.nq
        sub = PrivilegedSubscriber(port=plan["privileged_port"])
        first = sub.wait(30.0)
        problem = state_problem(first, nq)  # the check every request gets, once before any request
        if problem:
            raise SystemExit(problem)
        view.warm_up(np.asarray(first["qpos"]))
        sock = zmq.Context.instance().socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://*:{plan['mirror_port']}")
        signal.signal(signal.SIGTERM, _stop)
        print(f"isaac mirror listening on :{plan['mirror_port']} (state from :{plan['privileged_port']})", flush=True)
        with log_path.open("a") as sink:

            def record(header):
                sink.write(json.dumps(header) + "\n")
                sink.flush()

            try:
                serve(sock, view, lambda: sub.latest(0), nq, record, lambda: False)
            except KeyboardInterrupt:
                pass
    except SystemExit as exc:
        print(f"isaac mirror: {exc}", flush=True)
        os._exit(1)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
