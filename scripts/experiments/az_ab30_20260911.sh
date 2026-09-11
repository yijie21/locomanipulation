#!/usr/bin/env bash
# Ledger 29.1: does the 29 deg planned stand cost table clearance, or did the
# sampled distribution just cover harder placements?
#
# The 30 cases are REBUILT FROM the gate-baseline batch's own manifest (init:
# obj_xy / stand_xy / psi / yaw / spawn_off / spawn_lat; `rel` inverted out of
# `stand = can_xy - R(psi)*rel`, verified to reproduce 29.00 deg on all 30), so
# this runs exactly the placements where the clearance failures appeared. Two
# banks differ ONLY in `rel` and the `stand` recomputed from it -- 29.0 vs the
# 37.4 the depth-only backshift produces -- a 4.75 cm move of the planned stand.
#
# Note the power: only 2 of those 30 rounds failed on clearance, so this cannot
# settle the clearance question by itself. What it IS powered for is the overall
# paired comparison on the SAMPLED distribution, which ledger 27 only measured
# on a different fixed bank.
set -uo pipefail
cd /workspace/code/locomanipulation
O="$PWD/outputs/experiments/az_ab30_20260911"
ROUNDS=30
export INSTANCE=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export SCENE_RANDOM=0 SIM_VIS=bbox,readout
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
export SONIC_POLICY="${SONIC_POLICY:-policy/sonic_v1_1}"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
PROMPT="pick up the water bottle"
clean() { bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1; sleep 3; }
trap 'clean; echo ABORTED; exit 130' INT TERM
alive_here() {
    ps -eo comm,args --no-headers | awk -v p="$1" -v i="$INSTANCE" '
        $1 ~ /^python/ && $0 ~ p {
            n = "0"; for (k = 1; k < NF; k++) if ($k == "--instance") n = $(k+1)
            if (n == i) print
        }'
}
for pat in "run_sim_""locomani[.]py" "run_data_""exporter[.]py"; do
    hit=$(alive_here "$pat")
    [ -n "$hit" ] && { echo "REFUSING: '$pat' already on instance $INSTANCE"; exit 1; }
done
# 37.4 first: if the machine degrades over the session, the arm that runs second
# is the one that looks worse, and 29 is the arm this experiment is defending.
for arm in az374 az29; do
    out="$O/$arm"; mkdir -p "$out/videos"
    [ -f "$out/done" ] && { echo "== $arm already done"; continue; }
    case "$arm" in
        az29)  BANK="$O/cases30_az29.json" ;;
        az374) BANK="$O/cases30_az374.json" ;;
    esac
    echo "===== arm $arm  $(date -u +%H:%M:%S)"
    clean
    SIM_VIDEO_DIR="$out/videos" bash scripts/stack/start_stack.sh "$PROMPT" > "$out/stack.log" 2>&1
    grep -q "stack up" "$out/stack.log" || { echo "stack failed"; tail -3 "$out/stack.log"; clean; exit 1; }
    grep -m1 "policy:" "$out/stack.log"
    timeout 5400 "$PY" -u src/locomani/tasks/bottle/grasp_gym.py \
        --reach-config configs/collection/ready_grasp_visible.json \
        --task-config configs/tasks/bottle_visible.json \
        --targets water_bottle \
        --spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 20.0 \
        --spawn-lateral 0.2 --edge-jitter 0.2 \
        --stand-retries 0 --record --video-include-init \
        --cases "$BANK" --rounds "$ROUNDS" \
        --manifest "$out/manifest.jsonl" > "$out/gym.log" 2>&1
    echo "  gym exit $? $(date -u +%H:%M:%S)"
    grep -E "^\[gym\] [0-9]+/" "$out/gym.log" | tail -1
    clean; touch "$out/done"
done
echo "===== done $(date -u +%FT%TZ)"
