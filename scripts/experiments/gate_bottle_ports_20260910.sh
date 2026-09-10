#!/usr/bin/env bash
# Behaviour gate candidate for the 2026-09-10 port/guard changes.
# Reproduces the baseline's Bp arm, batch 0 (tests/gate/baseline_bottle.json,
# source outputs/experiments/restart_ab_20260910/manifest2_Bp*.jsonl): same
# reach config, same case bank, same seed, same flags, RESTART_DEPLOY=False.
# Only the GPU differs -- the baseline ran on GPU0, which is another session's
# training today, so this runs on GPU1. Instance 0 is required for the exporter.
set -uo pipefail
cd /workspace/code/locomanipulation
. scripts/paths.sh
E="$PWD/outputs/experiments/restart_ab_20260910"
O="$PWD/outputs/experiments/gate_bottle_ports_20260910"
mkdir -p "$O"
export INSTANCE=0 CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 SIM_VIDEO_DIR="$O/sim_video"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

alive() { ps -eo comm,args --no-headers | awk -v p="$1" '$1 ~ /^python/ && $0 ~ p {print}'; }
for pat in "run_sim_""locomani[.]py" "run_data_""exporter[.]py"; do
    hit=$(alive "$pat"); [ -n "$hit" ] && { echo "REFUSING: '$pat' already running"; echo "$hit" | cut -c1-120; exit 1; }
done
read -r free < <(bash scripts/cpu_headroom.sh 2 | grep -oE "free [-0-9.]+" | cut -d' ' -f2)
echo "cpu free ${free} cores"

bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1; sleep 2
echo "start stack $(date -u +%H:%M:%S)"
bash scripts/stack/start_stack.sh "pick up the water bottle" > "$O/stack.log" 2>&1
grep -q "stack up" "$O/stack.log" || { echo "STACK FAILED"; tail -5 "$O/stack.log"; bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1; exit 1; }
echo "collect $(date -u +%H:%M:%S)"
PYTHONPATH=src timeout 1800 "$VENV_SIM_PY" -u -c "
import sys
from locomani.tasks.bottle import grasp_gym
grasp_gym.RESTART_DEPLOY = False
sys.argv = ['grasp_gym.py'] + sys.argv[1:]
grasp_gym.main()
" --reach-config "$E/reach_config.json" --targets water_bottle --rounds 10 --stand-retries 0 --record \
  --video-include-init --spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 20.0 --spawn-lateral 0.2 \
  --edge-jitter 0.2 --seed 29000 --cases "$E/cases_0.json" --manifest "$O/manifest_Bp0.jsonl" \
  > "$O/gym.log" 2>&1
echo "gym rc=$? $(date -u +%H:%M:%S)  $(grep -E '^\[gym\] [0-9]+/' "$O/gym.log" | tail -1)"
bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1
echo GATERUNDONE
