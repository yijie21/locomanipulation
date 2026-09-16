"""Stages as jobs (decision 4 of docs/handoff/infra-plan-20260916.md).

Each stage is a function `run(run_cfg, cfg, out) -> dict` that reads its upstream stage's manifest,
does its work through child processes, and writes `<out>/<stage>/manifest.json` (config hash,
upstream manifest hashes, outputs, metrics, resources measured at start). Stages know each other
only through those manifests; `locomani.cli <stage>` is the entry point and the launcher's DAG
reads the same files.
"""

from locomani.stages import collect, convert, eval_stage, stats, train

RUNNERS = {
    "collect": collect.run,
    "convert": convert.run,
    "train": train.run,
    "eval": eval_stage.run,
    "stats": stats.run,
}

__all__ = ["RUNNERS"]
