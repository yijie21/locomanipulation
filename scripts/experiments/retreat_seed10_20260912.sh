#!/usr/bin/env bash
# Ledger 32: measure the squat's DELIVERED retreat directly, and collect the
# terminal-IK seeds.
#
# Two things this run produces that no earlier run could:
#   1. `coupled_transport[].squat_delivered_pose` -- the retreat/lateral/pitch
#      the squat actually delivers, against the 0.8 ratio `project_squat`
#      predicts. Reconstructed offline over 111 rounds it is 1.15x on release
#      and 1.43x on v1.1 (AUC 0.945); this is the direct measurement, on the
#      SAME 10 placements as az_ab30/az374 rounds 1-10, so it also validates
#      that reconstruction.
#   2. `coupled_prediction_ik_errors[].seed_q` + the two pelvis frames -- the
#      live seed the `use_waist=False` solve carries, without which the
#      rejections cannot be re-solved offline (ledger 31.5 item 1).
#
# Run it as `bash scripts/experiments/retreat_seed10_20260912.sh <out-dir-name>`.
# The three batches of 2026-09-12 (retreat_seed10_20260912_run1 / ...10b / ...10c) are this
# same script; they differ only in how much of the instrumentation existed yet -- run1 had the
# delivered-pose record, run2 added the waist shadow solve, run3 the byte-identical repeat.
#
# Bank, not sampling: the retreat is a per-round geometric quantity and fixed
# placements keep placement variance out of it (and pair it with az374).
set -uo pipefail
cd /workspace/code/locomanipulation
O="$PWD/outputs/experiments/${1:-retreat_seed10c_20260912}"
ROUNDS=10
BANK="$O/cases10_az374.json"
export INSTANCE=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export SCENE_RANDOM=0 SIM_VIS=bbox,readout
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
export SONIC_POLICY="${SONIC_POLICY:-policy/sonic_v1_1}"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
PROMPT="pick up the water bottle"
clean() { bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1; sleep 3; }
trap 'clean; echo ABORTED; exit 130' INT TERM
# Scoped to THIS instance: an unscoped match once refused to start because
# another session was running its own sim on instance 1 (ledger 31).
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
mkdir -p "$O/videos"
[ -f "$O/done" ] && { echo "already done"; exit 0; }
echo "===== retreat+seed run  $(date -u +%H:%M:%S)"
clean
SIM_VIDEO_DIR="$O/videos" bash scripts/stack/start_stack.sh "$PROMPT" > "$O/stack.log" 2>&1
grep -q "stack up" "$O/stack.log" || { echo "stack failed"; tail -3 "$O/stack.log"; clean; exit 1; }
grep -m1 "policy:" "$O/stack.log"
timeout 3600 "$PY" -u src/locomani/tasks/bottle/grasp_gym.py \
    --reach-config configs/collection/ready_grasp_visible.json \
    --task-config configs/tasks/bottle_visible.json \
    --targets water_bottle \
    --spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 20.0 \
    --spawn-lateral 0.2 --edge-jitter 0.2 \
    --stand-retries 0 --record --video-include-init \
    --cases "$BANK" --rounds "$ROUNDS" \
    --manifest "$O/manifest.jsonl" > "$O/gym.log" 2>&1
echo "  gym exit $? $(date -u +%H:%M:%S)"
grep -E "^\[gym\] [0-9]+/" "$O/gym.log" | tail -1
clean; touch "$O/done"
echo "===== done $(date -u +%FT%TZ)"
