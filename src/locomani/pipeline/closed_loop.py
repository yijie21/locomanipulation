"""What the two closed-loop evaluators share: the run-directory admission and record, and the score.

`eval_closed_loop` (single arm: grasp_gym's own round loop with the policy installed) and
`eval_crate_closed_loop` (bimanual: `vla_rounds` in batches, the deploy restarted between them) drive
different round loops, so they stay two files. Everything around the loop is here, once (TODO 1.18).
Until 2026-10-01 each evaluator carried its own copy, and the copies had drifted: the bimanual one
neither checked the deploy patches nor refused a finished evaluation before attaching, its Wilson
interval said (0, 0) for no rounds, and the single-arm one took the stack down through the deploy
pid it attached to, which its own per-round deploy restarts had already replaced.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from locomani.pipeline.batches import wilson


def failure_stage(why: str) -> str:
    """The stage a failed round's `why` names (`[vla] close: ...` -> `close`); before the policy took over, `pre-vla`."""
    return why.split(":")[0].replace("[vla] ", "") if "[vla]" in why else "pre-vla"


def score(rows: list[dict]) -> dict:
    """Success rate, its Wilson 95% interval and the failed rounds by stage, over one row per ROUND.

    A row per attempt would count a stand retry as a second sample; the single-arm evaluator folds those
    first (`eval_closed_loop.round_outcomes`), the bimanual rounds never retry.
    """
    ok = sum(1 for r in rows if r.get("ok"))
    return dict(
        rounds=len(rows),
        successes=ok,
        success_rate=ok / len(rows) if rows else 0.0,
        wilson95=list(wilson(ok, len(rows))),
        failure_stages=dict(Counter(failure_stage(r.get("why", "")) for r in rows if not r.get("ok"))),
    )


@dataclass
class RunEval:
    """A run-directory evaluation that passed admission: the run's config and eval block, and the stack it holds."""

    run_dir: Path
    cfg: dict
    evaluation: object  # EvalConfig
    ev: Path
    checkpoint: Path
    instance: int
    stack: object  # locomani.stack.Stack


def show(run_dir: Path) -> int:
    """`--dry-run`: the eval block the run would be evaluated with, and the instance it would attach to."""
    from locomani.pipeline import run_context

    cfg = run_context.load_run_config(run_dir)
    evaluation = run_context.evaluation_of(cfg)
    run_context.checkpoint_of(run_dir)
    run_context.show_config("eval", evaluation.to_dict())
    print(f"\n  attaches to instance {cfg['collect']['instance']}, policy port {evaluation.policy_port}")
    return 0


def attach(run_dir: Path, run_arg, gpu: str) -> RunEval:
    """Every check a run-directory evaluation makes before its first round, in one order for both lines.

    Attach FIRST, then ask only for what the running stack does not already hold (an evaluation runs no
    exporter, hence record=False): before 2026-09-23 the headroom check charged the stack's own cores twice.
    """
    from locomani.pipeline import run_context
    from locomani.stack import Stack

    cfg = run_context.load_run_config(run_dir)
    evaluation = run_context.evaluation_of(cfg)  # validated here, so a bad field fails first
    ev = run_dir / "eval"
    instance = int(cfg["collect"]["instance"])
    if (ev / "closed_loop.json").exists():
        raise SystemExit(f"{ev / 'closed_loop.json'} exists; move the finished evaluation aside before rerunning")
    checkpoint = run_context.checkpoint_of(run_dir)
    run_context.require_deploy_patches(cfg)
    try:
        stack = Stack.attach(instance)
    except FileNotFoundError:
        raise SystemExit(
            f"no stack on instance {instance}; start one in another terminal:\n"
            f"  bash scripts/start_stack.sh eval {run_arg} --gpu {gpu}"
        )
    run_context.require_stack_of(stack, run_dir, "eval")
    run_context.require_headroom("eval", gpu=gpu, record=False, already=run_context.group_threads(stack.pids.values()))
    ev.mkdir(parents=True, exist_ok=True)
    evaluation.dump(ev / "eval_config.json")
    return RunEval(run_dir, cfg, evaluation, ev, checkpoint, instance, stack)


def finish(run: RunEval, *, keep_stack: bool) -> None:
    """Take the stack down unless asked not to, through its state file: both lines' rounds restart the deploy,
    so the deploy pid attached at the start is no longer the running one."""
    if keep_stack:
        return
    from locomani.stack import Stack

    try:
        Stack.attach(run.instance).down()
    except FileNotFoundError:
        pass  # already down


def record(run: RunEval) -> dict:
    """Copy the evaluation's headline into the run's summary.json and say where the result is."""
    from locomani.pipeline import run_context

    result = json.loads((run.ev / "closed_loop.json").read_text())
    run_context.summarise(
        run.run_dir,
        "eval",
        checkpoint=str(run.checkpoint),
        **{k: result[k] for k in ("rounds", "successes", "success_rate", "retreated") if k in result},
    )
    print(f"eval: {result.get('successes')}/{result.get('rounds')} -> {run.ev / 'closed_loop.json'}")
    return result
