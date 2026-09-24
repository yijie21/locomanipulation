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
. scripts/paths.sh
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"   # original author's machine only; a fresh setup builds both patches (setup step 3) and skips this

bash scripts/run_overnight.sh configs/experiments/bottle_simple_200.yaml --gpu 1   # the whole path
# collection is two terminals: the recording stack, then the agent that drives it
bash scripts/start_stack.sh collect         configs/experiments/bottle_simple_gate8.yaml --gpu 1
bash scripts/single_arm/run_agent_server.sh configs/experiments/bottle_simple_gate8.yaml --gpu 1
bash scripts/convert_data.sh outputs/runs/<name>
bash scripts/train.sh        outputs/runs/<name> --gpu 1
# closed-loop evaluation is three: the stack, the policy server, the rounds
bash scripts/start_stack.sh eval             outputs/runs/<name> --gpu 1
bash scripts/run_vla_server.sh               outputs/runs/<name> --gpu 1
bash scripts/single_arm/run_vla_inference.sh outputs/runs/<name> --gpu 1
bash scripts/stop_stack.sh --instance 0   # only when a driver was killed before it could
```

Every script takes `--dry-run`, which prints the commands and environment it would use and writes
nothing. The first stage opens the run from an experiment file; the later ones take the run
directory and read the copy of that file inside it. A stage whose artifact already exists is
skipped by `run_overnight.sh` (collection only once `summary.json` records enough kept episodes); delete
the artifact to redo it. A driver refuses a stack that another run, or the other mode, started. After conversion, `check_dataset.py` has
already run with `--run-dir`; `artifact_contract.py check <run_dir> <name>` says whether a stage
left behind everything it owes.

## Manual bottle or pick-and-place check

Use this when inspecting motion without recording an episode, or when using a
custom `LOCOMANI_TMP` that the collection script does not discover yet.
Prepare the scene using the relevant [bottle](../single_arm.md) or
[pick-and-place](../single_arm.md) guide. In Bash at the repository root:

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
| Closed-loop SONIC evaluation | `src/locomani/pipeline/eval_closed_loop.py --help`; [evaluation](../single_arm.md) |
| Crate / warehouse collection | `bash scripts/bimanual/run_agent_server.sh configs/experiments/crate_table30_record.yaml --dry-run` (warehouse: `box_room_grasp_baseline_20260924.yaml`); [crate](../bimanual.md), [warehouse](../warehouse.md) |
| ScaleBFM collection/evaluation | `src/locomani/tasks/bottle_visual/pipeline/collect.py` / `eval.py`; [ScaleBFM collection](../scalebfm.md) |
| Pick and place | `tasks/bottle/collect.py` with `configs/experiments/pick_place.yaml`; [task guide](../single_arm.md) |
| Behavior comparison | `python3 scripts/gate.py --help`; match baseline and candidate protocols |
| Stop an owned SONIC stack | `python -m locomani.stack.cli down --instance <instance>` |

The previous runbook is preserved [verbatim](../archive/runbook_20260912.md).
Its old variables, running claims and empirical recommendations require verification.
