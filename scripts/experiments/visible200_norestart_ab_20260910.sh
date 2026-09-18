#!/bin/bash
# Paired 20-round A/B of ONE variable: the deploy restart in the eval.
# visible200 was collected with restart_deploy=false (hold + reanchor after each reset), but the
# eval pins the per-round restart, so the two establish the deploy's heading anchor by different
# mechanisms -- 65 measured them 8.5 deg apart (round_phases.py's comment). That is the suspect for
# ledger bottle §18's 2/100, where the robot arrives near the table with the object at the extreme
# left of the ego view and never closes.
# Same seed 6000, so these 20 rounds are the SAME cases as rounds 0-19 of the 100-round run
# (which scored 1/20 on that prefix). Only restart_deploy differs.
set -u
cd /workspace/code/locomanipulation
E=outputs/experiments/visible200_norestart_ab_20260910
SRC=outputs/experiments/visible200_20260910
LOG=$E/runner.log
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
for i in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | awk '{print ($1 > 500) ? 1 : 0}')
    live=$(ps -eo comm,args --no-headers | awk '($1 ~ /^python/ || $1 ~ /^g1_deploy/) && /run_sim_locomani\.py|g1_deploy_onnx_ref|run_data_exporter/ {n++} END {print n+0}')
    head=$(bash scripts/cpu_headroom.sh 5)
    echo "$(date -u +%T) wait $i: gpu0 busy=$busy procs=$live | $head" >> "$LOG"
    if [ "$busy" = "0" ] && [ "$live" = "0" ] && echo "$head" | grep -q "OK for a stack"; then break; fi
    sleep 60
done
echo "$(date -u +%FT%TZ) start" >> "$LOG"
# ABSOLUTE paths: the sim process runs with cwd third_party/GR00T-WholeBodyControl/, so a relative
# LOCOMANI_SIM_DIR / SIM_VIDEO_DIR resolves over there (documented in docs/single_arm_design.md; I hit it anyway).
LOCOMANI_SIM_DIR=$PWD/$SRC/scene NO_ENCODER=1 SIM_VIDEO_DIR=$PWD/$E/videos \
  bash scripts/stack/start_stack.sh "pick up the water bottle" >> "$E/stack.log" 2>&1
rc=$?; echo "$(date -u +%FT%TZ) stack rc=$rc" >> "$LOG"
if [ "$rc" != "0" ]; then echo "stack failed, aborting" >> "$LOG"; bash scripts/stack/stop_instance.sh 0 >> "$LOG" 2>&1; exit 1; fi
third_party/Isaac-GR00T/.venv/bin/python src/locomani/pipeline/serve_policy.py \
  --model-path "$SRC/model/checkpoint-10000" --port 5590 >> "$E/serve.log" 2>&1 &
SP=$!
for i in $(seq 1 200); do grep -q listening "$E/serve.log" 2>/dev/null && break; sleep 3; done
echo "$(date -u +%FT%TZ) server up" >> "$LOG"
$PY -u src/locomani/pipeline/eval_closed_loop.py \
  --config configs/eval/closed_loop_100.json \
  --set rounds=20 --set restart_deploy=false --set backend.dataset_fps=25 \
  --out "$E" --trace-dir "$E/trace" >> "$E/eval.log" 2>&1
rc=$?; echo "$(date -u +%FT%TZ) eval rc=$rc" >> "$LOG"
kill $SP 2>/dev/null
bash scripts/stack/stop_instance.sh 0 >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) stopped" >> "$LOG"
