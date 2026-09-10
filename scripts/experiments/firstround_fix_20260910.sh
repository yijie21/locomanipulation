#!/bin/bash
# Verification of fix/first-round-anchor (384381e): 3 batches x 4 rounds from the FIX WORKTREE's src.
# Round 1 of each batch is the test (was 19/19 failing at the spawn on main).
set -u
cd /workspace/code/locomanipulation/.claude/worktrees/fix-review
E=outputs/experiments/firstround_fix_20260910
echo "$(date -u +%FT%TZ) launching" >> $E/runner.log
LOCOMANI_ROOT=$PWD third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u src/locomani/pipeline/run_learn_pilot.py \
  --config configs/pilot/learn_200.json --task configs/tasks/bottle_visible.json \
  --out "$E" --scene outputs/scene/sim \
  --set through=collect --set successes=12 --set batch_rounds=4 --set max_batches=3 --set seed0=32000 \
  --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0 >> "$E/pilot.log" 2>&1
echo "$(date -u +%FT%TZ) pilot exit=$?" >> $E/runner.log
bash scripts/stack/stop_instance.sh 0 >> $E/runner.log 2>&1
echo "$(date -u +%FT%TZ) stack stopped" >> $E/runner.log
