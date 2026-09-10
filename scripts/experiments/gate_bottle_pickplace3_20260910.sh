#!/bin/bash
# Bottle behaviour gate candidate after the pick_place changes (same recipe as outputs/experiments/gate_bottle_main0:
# bottle_visible.json, seed0 29000, 10 rounds until 8 ok, GYM_SHUFFLE_BAND=0), on GPU1 / instance 0.
#   bash scripts/experiments/gate_bottle_pickplace3_20260910.sh   (run from a scratch copy)
set -u
cd /workspace/code/locomanipulation
E=outputs/experiments/gate_bottle_pickplace3_20260910; mkdir -p "$E"
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u src/locomani/pipeline/run_learn_pilot.py \
  --config configs/pilot/learn_200.json --task configs/tasks/bottle_visible.json \
  --out "$E" --scene outputs/scene/sim \
  --set through=collect --set successes=8 --set batch_rounds=10 --set max_batches=1 --set seed0=29000 --set holdout=1 \
  --set gym_env=GYM_SHUFFLE_BAND=0 --set stack_env=CUDA_VISIBLE_DEVICES=1,MUJOCO_EGL_DEVICE_ID=1 >> "$E/pilot.log" 2>&1
echo "pilot exit=$?" >> "$E/runner.log"
bash scripts/stack/stop_instance.sh 0 >> "$E/runner.log" 2>&1
python3 scripts/gate.py compare bottle "$E/collect/manifest_*.jsonl" tests/gate/baseline_bottle.json --json "$E/gate.json" > "$E/gate.txt" 2>&1
echo "gate exit=$? done" >> "$E/runner.log"
