# Runbook — implemented entry points

These commands identify the current drivers. Check `--help` and resolved config
before launching; they do not promise a historical success rate. Runtime setup is
in [setup](../setup.md); shared-resource rules are in [AGENTS](../../AGENTS.md).

## Inspect bottle options without a simulator

```bash
PYTHONPATH=src python3 -m locomani.tasks.bottle.cli_args \
  --task-config configs/task/bottle_visible.yaml
```

The report covers resolved gym options and their sources, not every child-process
setting. `reach.strategy` selects a controller. Version-1 reach JSON is migrated
explicitly; newly written reach snapshots use version 2.

## Bottle collection, conversion and training

```bash
export PYTHONPATH=src
"$VENV_SIM_PY" -m locomani.cli collect task=bottle_visible scene.dir=outputs/scene/sim gpu=0 name=<name>
"$VENV_SIM_PY" -m locomani.cli convert name=<name>
"$VENV_SIM_PY" -m locomani.cli train name=<name>
"$VENV_SIM_PY" -m locomani.cli eval name=<name>
```

This dispatches to the unified CLI. Each stage reads the previous stage's manifest and writes its own.
Use `locomani.cli show name=<name>` to inspect the resolved configuration. Use a fresh run name each time.
After conversion run `src/locomani/pipeline/check_dataset.py` on the dataset in `outputs/runs/<name>/`.

## Manual bottle or pick-and-place check

Use this when inspecting motion without recording an episode, or when using a
custom `LOCOMANI_TMP` that the automatic learn runner does not discover yet.
Prepare the scene using the relevant [bottle](../collection.md) or
[pick-and-place](../pick_place_collection.md) guide. In Bash at the repository root:

```bash
export INSTANCE=0
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export SIM_VIS=none SCENE_RANDOM=0
TASK=configs/task/bottle_visible.yaml
PROMPT="pick up the water bottle"
export LOCOMANI_SIM_DIR="$LOCOMANI_ROOT/outputs/scene/sim"
SIM_VIDEO_DIR="$LOCOMANI_ROOT/outputs/bottle_manual/videos" \
  python -m locomani.stack.cli up --instance "$INSTANCE" --gpu 0 --prompt "$PROMPT"
"$VENV_SIM_PY" -u -m locomani.tasks.bottle.grasp_gym \
  --task-config "$TASK" --rounds 1 --seed 30000 --video-include-init \
  --manifest outputs/bottle_manual/manifest.jsonl
python -m locomani.stack.cli down --instance "$INSTANCE"
```

For pick-and-place, set `TASK=configs/task/pick_place.yaml`,
`PROMPT="pick up the red can and put it in the basket"` and
`LOCOMANI_SIM_DIR="$LOCOMANI_ROOT/outputs/scene/pick_place"`; use a separate output directory.
GPU/instance selection requires ownership as usual. Stop the expert before cleanup.

Without `--record` the gym does not record episodes; instance 0 still starts an
exporter and can create an empty session. Add `--record` to the gym command to keep
accepted episodes. Pair this stack's session under `$LOCOMANI_TMP/p03/outputs` with
its manifest; always use fresh output paths. For batching use the task guide.

## Read a run's evidence

`config_snapshot.json` records the pre-launch configuration and input identities.
`commands.jsonl` / `timings.jsonl` record actual stage settings and exit status.
`deploy_receipts.jsonl` appends the launcher's final cwd, executable and checkpoint
hashes on start/restart, including alternate deployment trees. Closed-loop evaluation
also records the serving process's checkpoint hashes and checks its model field order.
An older policy server must be restarted to supply the identity endpoint.

## Other entries

| Operation | Driver / guide |
|---|---|
| Closed-loop SONIC evaluation | `src/locomani/pipeline/eval_closed_loop.py --help`; [evaluation](../evaluation.md) |
| Crate collection | `python -m locomani.cli show task=crate_visual embodiment=g1_bimanual scene=crate_table30`; [crate collection](../crate_collection.md) |
| ScaleBFM collection/evaluation | `src/locomani/tasks/bottle_visual/pipeline/collect.py` / `eval.py`; [ScaleBFM collection](../scalebfm_collection.md) |
| Pick and place | Bottle driver with `configs/task/pick_place.yaml`; [task guide](../pick_place_collection.md) |
| Behavior comparison | `python3 scripts/gate.py --help`; match baseline and candidate protocols |
| Stop an owned SONIC stack | `python -m locomani.stack.cli down --instance <instance>` |

The previous runbook is preserved [verbatim](../archive/runbook_20260912.md).
Its old variables, running claims and empirical recommendations require verification.
