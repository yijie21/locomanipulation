"""The one command line: `python -m locomani.cli <stage> [hydra overrides]`.

    python -m locomani.cli collect +experiment=bottle_simple_lowcom
    python -m locomani.cli eval    +experiment=bottle_simple_lowcom gpu=1 eval.rounds=20
    python -m locomani.cli collect task=crate_table30 embodiment=g1_bimanual scene=crate_table30_depth04
    python -m locomani.cli show    +experiment=pick_place_default      # print the composed run, run nothing

The stage composes the run (`locomani.conf`), writes `run_config.yaml` into the run directory, and
dispatches to the task plugin's pipeline. Until step 3 of docs/handoff/infra-plan-20260916.md turns
the stages into their own processes, the dispatch is a bridge onto the existing drivers
(`run_learn_pilot` for the single-arm tasks, `run_crate_pilot` for the crate): it dumps the composed
task/pilot configs into the run directory and calls them with those files. Knobs those drivers still
hard-code (the collection backend, the crate expert's instance) are listed in that plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from locomani import conf


def _bottle_stage(run: conf.RunConfig, out: Path, stage: str) -> None:
    from locomani.pipeline import run_learn_pilot

    task_path, pilot_path = out / "task_config.json", out / "pilot_config.json"
    if not task_path.exists():
        run.task.dump(task_path)
    conf.pilot_from(run, through=stage).dump(pilot_path)
    argv = ["--config", str(pilot_path), "--task", str(task_path), "--out", str(out)]
    argv += ["--scene", str(conf.resolve_repo_path(run.scene.dir))]
    if run.scene.cases:
        argv += ["--cases", str(conf.resolve_repo_path(run.scene.cases))]
    run_learn_pilot.main(argv)


def _crate_stage(run: conf.RunConfig, out: Path, stage: str) -> None:
    if stage != "collect":
        raise SystemExit(
            f"crate {stage}: not bridged yet (docs/handoff/infra-plan-20260916.md step 3); use the crate drivers"
        )
    from locomani.pipeline import run_crate_pilot

    task_path = out / "task_config.json"
    if not task_path.exists():
        run.task.dump(task_path)
    rounds = run.collect.batch_rounds * run.collect.max_batches
    argv = ["--expert", "visual", "--out", str(out), "--config", str(task_path)]
    argv += ["--scene", str(conf.resolve_repo_path(run.scene.dir))]
    argv += ["--rounds", str(rounds), "--batch-rounds", str(run.collect.batch_rounds), "--seed", str(run.collect.seed0)]
    argv += ["--instance", str(run.collect.instance), "--gpu", str(run.gpu)]
    if run.collect.record:
        argv.append("--record")
    if (out / "inputs.json").exists():
        argv.append("--resume")
    run_crate_pilot.main(argv)


DISPATCH = {"bottle": _bottle_stage, "pick_place": _bottle_stage, "crate": _crate_stage}


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
    plugin = run.task.plugin
    if plugin not in DISPATCH:
        raise SystemExit(f"task plugin {plugin!r} has no CLI dispatch yet ({sorted(DISPATCH)})")
    DISPATCH[plugin](run, out, verb)


if __name__ == "__main__":
    main()
