"""Dispatch collection, conversion, training and evaluation to task owners.

    python -m locomani.cli <collect|convert|train|eval|gate> --task NAME [flags]

Use --config for the selected entry's configuration (EvalConfig for evaluation).
Unknown flags pass to that entry. Supported task/verb pairs are in _DISPATCH;
ScaleBFM has its own in-process pipeline under tasks/bottle_visual/pipeline/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable

from locomani import paths
from locomani.tasks.registry import TASKS, get_task

Dispatch = Callable[[str | None, str | None, list[str], list[str]], None]


def _run_learn_pilot_stage(stage: str) -> Dispatch:
    def dispatch(cfg, out, set_, extra):
        from locomani.pipeline import run_learn_pilot

        argv = list(extra)
        if cfg is not None:
            argv += ["--task", cfg]  # run_learn_pilot's --task IS the BottleTaskConfig path
        if out is not None:
            argv += ["--out", out]
        for kv in set_:
            argv += ["--set", kv]
        argv += ["--set", f"through={stage}"]  # authoritative: the verb picked the stage
        run_learn_pilot.main(argv)

    return dispatch


def _eval_closed_loop(cfg, out, set_, extra):
    from locomani.pipeline import eval_closed_loop

    argv = list(extra)
    if cfg is not None:
        argv += ["--config", cfg]  # eval_closed_loop's own --config IS its EvalConfig path
    if out is not None:
        argv += ["--out", out]
    for kv in set_:
        argv += ["--set", kv]
    eval_closed_loop.main(argv)


def _run_crate_pilot(expert: str) -> Dispatch:
    """crate collect: the pilot driver, which owns the stack the way
    run_learn_pilot.py does for bottle. Until 2026-09-11 this dispatched
    straight into `bimanual_carry.main()` and refused --config/--set with
    "bimanual_carry.py has no --config/--set yet" -- wrong since refactor
    step 3 gave both experts those flags (bimanual_carry.py:360)."""

    def dispatch(cfg, out, set_, extra):
        from locomani.pipeline import run_crate_pilot

        argv = ["--expert", expert, *extra]
        if cfg is not None:
            argv += ["--config", cfg]
        if out is not None:
            argv += ["--out", out]
        for kv in set_:
            argv += ["--set", kv]
        run_crate_pilot.main(argv)

    return dispatch


def _gate(cfg, out, set_, extra, *, task_name):
    if cfg is not None or set_:
        raise SystemExit("locomani cli gate: scripts/gate.py takes mode/manifests/baseline positionally; use EXTRA")
    gate_py = Path(paths.REPO) / "scripts" / "gate.py"
    # gate.py's own positionals are `mode arm manifests baseline`; `arm` is
    # always this dispatch's --task, spliced in after the mode extra supplies.
    argv = [sys.executable, str(gate_py), *extra[:1], task_name, *extra[1:]]
    raise SystemExit(subprocess.call(argv))


_DISPATCH: dict[tuple[str, str], Dispatch] = {
    ("collect", "bottle"): _run_learn_pilot_stage("collect"),
    ("convert", "bottle"): _run_learn_pilot_stage("convert"),
    ("train", "bottle"): _run_learn_pilot_stage("train"),
    ("eval", "bottle"): _eval_closed_loop,
    # pick_place IS the bottle task with another config (tasks/pick_place/__init__.py
    # wraps grasp_gym's own entries), so run_learn_pilot drives it unchanged --
    # scripts/experiments/pick_place_pilot_20260910.sh has been calling it that way
    # since 2026-09-10 while this table still had no row for the task at all.
    ("collect", "pick_place"): _run_learn_pilot_stage("collect"),
    ("convert", "pick_place"): _run_learn_pilot_stage("convert"),
    ("train", "pick_place"): _run_learn_pilot_stage("train"),
    ("collect", "crate"): _run_crate_pilot("wall"),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="locomani", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("verb", choices=("collect", "convert", "train", "eval", "gate"))
    ap.add_argument(
        "--task", required=True, choices=tuple(TASKS), help="registry task name (docs/reference/standards.md)"
    )
    ap.add_argument("--config", default=None, help="selected entry config JSON (EvalConfig for evaluation)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", dest="set_", action="append", default=[], metavar="KEY=VALUE")
    args, extra = ap.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    get_task(args.task)  # validates --task against the registry before dispatch
    if args.verb == "gate":
        _gate(args.config, args.out, args.set_, extra, task_name=args.task)
        return
    dispatch = _DISPATCH.get((args.verb, args.task))
    if dispatch is None:
        raise SystemExit(f"locomani cli: no entry point wired yet for verb={args.verb!r} task={args.task!r}")
    dispatch(args.config, args.out, args.set_, extra)


if __name__ == "__main__":
    main()
