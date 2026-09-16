"""The one command line: `python -m locomani.cli <stage> [hydra overrides]`.

    python -m locomani.cli collect +experiment=bottle_simple_lowcom
    python -m locomani.cli eval    +experiment=bottle_simple_lowcom gpu=1 eval.rounds=20
    python -m locomani.cli collect task=crate_table30 embodiment=g1_bimanual scene=crate_table30_depth04
    python -m locomani.cli show    +experiment=pick_place_default      # print the composed run, run nothing

The stage composes the run (`locomani.conf`), writes `run_config.yaml|json` into the run directory
and runs that one stage (`locomani.stages`), which reads its upstream stage's manifest and writes
its own `<out>/<stage>/manifest.json`. Stages only meet through those manifests (decision 4 of
docs/handoff/infra-plan-20260916.md); the crate task's collect stage runs the crate experts through
`locomani.tasks.crate.prepare`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from locomani import conf


def dispatch(run: conf.RunConfig, cfg, out: Path, stage: str) -> dict:
    """Run one stage in this process; the stage writes `<out>/<stage>/manifest.json`."""
    from locomani.stages import RUNNERS

    return RUNNERS[stage](run, cfg, out)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    verb, overrides = argv[0], argv[1:]
    if verb not in (*conf.STAGES, "show"):
        raise SystemExit(f"unknown stage {verb!r}; choices {conf.STAGES + ('show',)}")
    if verb != "show":
        overrides = [*overrides, f"stage={verb}"]
    run, cfg = conf.load_run(overrides)
    if verb == "show":
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    out = conf.resolve_repo_path(run.out)
    conf.snapshot(cfg, out)
    manifest = dispatch(run, cfg, out, verb)
    print(f"{verb}: ok={manifest['ok']} metrics={manifest['metrics']} -> {out / verb / 'manifest.json'}")


if __name__ == "__main__":
    main()
