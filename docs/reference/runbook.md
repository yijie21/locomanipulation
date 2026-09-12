# Runbook — implemented entry points

These commands identify the current drivers. Check `--help` and resolved config
before launching; they do not promise a historical success rate. Runtime setup is
in [setup](../setup.md); shared-resource rules are in [AGENTS](../../AGENTS.md).

## Inspect bottle options without a simulator

```bash
PYTHONPATH=src python3 -m locomani.tasks.bottle.cli_args \
  --task-config configs/tasks/bottle_visible.json
```

The report covers resolved gym options and their sources, not every child-process
setting. `reach.strategy` selects a controller. Version-1 reach JSON is migrated
explicitly; newly written reach snapshots use version 2.

## Bottle collection, conversion and training

```bash
make cli ARGS="collect --task bottle --config configs/tasks/bottle_visible.json --out outputs/experiments/<name> --scene outputs/scene/sim"
```

This dispatches to `pipeline/run_learn_pilot.py`. Its direct `--config` selects
PilotConfig; direct `--task` selects BottleTaskConfig. The wrapper currently renames
that task argument to `--config`. Use a fresh output directory. The stage verbs
`convert` and `train` use the same driver; read its `--help` before resuming a run.
After conversion run `src/locomani/pipeline/check_dataset.py` on the dataset.

## Other entries

| Operation | Driver / guide |
|---|---|
| Closed-loop SONIC evaluation | `src/locomani/pipeline/eval_closed_loop.py --help`; [evaluation](../evaluation.md) |
| Crate collection | `src/locomani/pipeline/run_crate_pilot.py --help`; [crate collection](../crate_collection.md) |
| ScaleBFM collection/evaluation | `src/locomani/tasks/bottle_visual/pipeline/collect.py` / `eval.py`; [ScaleBFM collection](../scalebfm_collection.md) |
| Pick and place | Bottle driver with `configs/tasks/pick_place.json`; [task guide](../pick_place_collection.md) |
| Behavior comparison | `python3 scripts/gate.py --help`; match baseline and candidate protocols |
| Stop an owned SONIC stack | `bash scripts/stack/stop_instance.sh <instance>` |

The previous runbook is preserved [verbatim](../archive/runbook_20260912.md).
Its old variables, running claims and empirical recommendations require verification.
