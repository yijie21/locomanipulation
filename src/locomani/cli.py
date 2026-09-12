"""Thin CLI dispatch onto the existing pipeline entry points (decision 3, 12).

    python -m locomani.cli <collect|convert|train|eval|gate> --task NAME
        [--config PATH] [--out PATH] [--set KEY=VALUE ...] [-- EXTRA ...]

No new logic: every verb sets `sys.argv` to the target entry point's OWN
flags and calls its existing `main()` (run_learn_pilot.py / eval_closed_loop.py)
or shells out to it (scripts/gate.py, which is not part of the `locomani`
package). `--task NAME` selects the row in the dispatch table below through
`locomani.tasks.registry`; it is not forwarded literally, because each target
already has its own, differently-named notion of "task"/config path (e.g.
run_learn_pilot.py's own `--task` flag IS a BottleTaskConfig path -- exactly
what this wrapper calls `--config`, matching `Task.config_cls`). Anything
this wrapper does not recognise (run_learn_pilot's required `--scene`, gate's
`mode`/manifests/baseline positionals, ...) passes through verbatim via
`parse_known_args`.

Not every (verb, task) is wired: only what already exists today.
  * bottle: collect/convert/train (run_learn_pilot.py's `through=` stage
    gate), eval (eval_closed_loop.py), gate.
  * pick_place: collect/convert/train, the same run_learn_pilot.py stages as
    bottle (the task is the bottle expert with another config). eval is not
    wired: eval_closed_loop.py's --config is an EvalConfig, not this task's
    config_cls, so the wrapper's --config would mean two different things.
  * crate: collect (run_crate_pilot.py -- the wall expert; pass
    `-- --expert carry` for the carry one), gate. convert/train/eval have no
    entry point: the crate expert does not drive the data exporter yet, so
    there is no dataset to convert or train on (docs/crate_collection.md §11).
  * bottle_visual (ScaleBFM): nothing here. That line has no SONIC stack at
    all -- its policy and its MuJoCo both run in-process -- and its own
    drivers live under tasks/bottle_visual/pipeline/.
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
    ap.add_argument("--config", default=None, help="Task.config_cls JSON, forwarded to the target's own config flag")
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
