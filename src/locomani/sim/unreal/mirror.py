"""The Unreal mirror: the eval sim's newest MuJoCo state rendered in Unreal (URLab), the dataset cameras out, when
the policy asks. The protocol and the policy client are `locomani.sim.mirror`, shared with the Isaac mirror, on
the same port; this is the server over `view.UnrealView`, for a run whose `images.source` is unreal.

    bash scripts/start_unreal.sh --gpu 1                    # the editor, once (its GPU is chosen here)
    bash scripts/start_stack.sh eval <run> --gpu 0          # terminal 1
    bash scripts/run_unreal_mirror.sh <run>                 # this; after the stack, before the rounds

The editor must be running with the scene's level built (`build_level`); the mirror opens the level, sets the
scene's capture count and starts PIE itself. Measured (warehouse, three cameras, 8 captures a request, no Nanite):
about 125-140 ms a request (docs/handoff/unreal-urlab-20260928.md). Every render is appended to
`<run>/eval/unreal_mirror.jsonl`. On exit it stops PIE, leaving the editor for the next run.
"""

from __future__ import annotations

import argparse
import json
import signal
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
    ap.add_argument("--dry-run", action="store_true", help="say what would be served, start nothing")
    ap.add_argument(
        "--light-scale",
        type=float,
        default=1.0,
        help="multiply every light and emissive panel of the level (view.set_lighting); a lighting-shift "
        "evaluation, TODO 21.3.1. Recorded on every render in eval/unreal_mirror.jsonl",
    )
    ap.add_argument("--light-temperature", type=float, default=None, help="every light at this kelvin")
    args = ap.parse_args(argv)

    run_dir = (args.run if args.run.is_absolute() else Path.cwd() / args.run).resolve()
    cfg = run_context.load_run_config(run_dir)
    source = cfg["images"]["source"]
    if source != "unreal":
        raise SystemExit(
            f"{run_dir.name}: images.source is {source!r}; the policy would never ask this mirror. "
            "Set images.source: unreal in the experiment file"
        )
    unreal = run_context.unreal_scene(cfg)
    instance = int(cfg["collect"]["instance"])
    try:
        stack = Stack.attach(instance)
    except FileNotFoundError:
        raise SystemExit(
            f"no stack on instance {instance}; the mirror renders the running sim's state, so start it first:\n"
            f"  bash scripts/start_stack.sh eval {args.run} --gpu <card>"
        )
    run_context.require_stack_of(stack, run_dir, "eval")
    ports = stack.spec.ports()
    plan = dict(
        scene_xml=str(stack.spec.scene_path() / "scene_locomani.xml"),
        level=unreal["level_path"],
        repeats=unreal["repeats"],
        streams=list(STREAMS),
        privileged_port=ports["TF_PRIVILEGED_PORT"],
        mirror_port=ports["TF_ISAAC_MIRROR_PORT"],
        stack_gpu=str(stack.spec.gpu),
        lighting={"scale": args.light_scale, "temperature": args.light_temperature},
    )
    if args.dry_run:
        run_context.show_config("unreal_mirror", plan)
        return 0
    return _run(run_dir, plan)


def _run(run_dir: Path, plan: dict) -> int:
    from locomani.sim.privileged_pub import PrivilegedSubscriber
    from locomani.sim.unreal.view import UnrealView

    log_path = run_dir / "eval" / "unreal_mirror.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    view = UnrealView(plan["scene_xml"], plan["level"], plan["repeats"], plan["streams"])
    lighting = plan["lighting"]
    try:
        if lighting != {"scale": 1.0, "temperature": None}:
            view.set_lighting(lighting["scale"], lighting["temperature"])
        sub = PrivilegedSubscriber(port=plan["privileged_port"])
        first = sub.wait(30.0)
        problem = state_problem(first, view.nq)  # the check every request gets, once before any request
        if problem:
            raise SystemExit(problem)
        view.warm_up(np.asarray(first["qpos"]))
        sock = zmq.Context.instance().socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://*:{plan['mirror_port']}")
        signal.signal(signal.SIGTERM, _stop)
        print(f"unreal mirror listening on :{plan['mirror_port']} (state from :{plan['privileged_port']})", flush=True)
        with log_path.open("a") as sink:

            def record(header):
                sink.write(json.dumps({**header, "renderer": "unreal", "lighting": lighting}) + "\n")
                sink.flush()

            try:
                serve(sock, view, lambda: sub.latest(0), view.nq, record, lambda: False)
            except KeyboardInterrupt:
                pass
    finally:
        view.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
