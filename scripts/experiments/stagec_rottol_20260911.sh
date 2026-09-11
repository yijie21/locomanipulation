#!/usr/bin/env bash
# One variable on top of Stage B: TRANSPORT_ROTATION_TOL 0.20 -> 0.35 rad, the
# coupled-squat transport IK gate that was tighter than the pelvis pitch the
# squat delivers (ledger bottle.md 23). Same ten cases as Stage B, so this is
# paired against release 4/10, v1.1-coarse-0.30 0/10 and v1.1-coarse-0.40 3/10.
# Acceptance: `coupled safe-hand IK did not track` gone, and NO new failure kind
# that implies the looser gate let a bad wrist solution through (a topple, or a
# `coupled safe-hand path blocked` from the swept-motion screen).
set -uo pipefail
cd /workspace/code/locomanipulation
AB="$PWD/outputs/experiments/v11_bottle_ab20_20260910"
O="$PWD/outputs/experiments/stagec_rottol_20260911"
mkdir -p "$O/videos"
ROUNDS=10

export INSTANCE=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export SCENE_RANDOM=0 SIM_VIS=bbox,readout
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
# The point of the run: the default checkpoint. Stated anyway so the log records it.
export SONIC_POLICY="${SONIC_POLICY:-policy/sonic_v1_1}"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
PROMPT="pick up the water bottle"

clean() { bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1; sleep 3; }
trap 'clean; echo "ABORTED"; exit 130' INT TERM

alive() { ps -eo comm,args --no-headers | awk -v p="$1" '$1 ~ /^python/ && $0 ~ p {print}'; }
for pat in "run_sim_""locomani[.]py" "run_data_""exporter[.]py"; do
    hit=$(alive "$pat")
    [ -n "$hit" ] && { echo "REFUSING: '$pat' already running"; echo "$hit" | cut -c1-140; exit 1; }
done

[ -f "$O/done" ] && { echo "already done"; exit 0; }
clean
SIM_VIDEO_DIR="$O/videos" bash scripts/stack/start_stack.sh "$PROMPT" > "$O/stack.log" 2>&1
rc=$?
[ "$rc" = 3 ] && { echo "instance-0 lock held"; tail -3 "$O/stack.log"; exit 3; }
grep -q "stack up" "$O/stack.log" || { echo "stack failed, see $O/stack.log"; clean; exit 1; }
grep -m1 "policy:" "$O/stack.log"
echo "stack up $(date -u +%H:%M:%S), $ROUNDS fixed cases"
timeout 3600 "$PY" -u src/locomani/tasks/bottle/grasp_gym.py \
    --reach-config "$AB/reach_visible.json" \
    --task-config configs/tasks/bottle_visible.json \
    --targets water_bottle \
    --spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 20.0 \
    --spawn-lateral 0.2 --edge-jitter 0.2 \
    --stand-retries 0 --record --video-include-init \
    --cases "$AB/cases20.json" --rounds "$ROUNDS" \
    --manifest "$O/manifest.jsonl" > "$O/gym.log" 2>&1
echo "gym exit $? $(date -u +%H:%M:%S)"
grep -E "^\[gym\] [0-9]+/" "$O/gym.log" | tail -2
clean
touch "$O/done"
echo "===== done $(date -u +%FT%TZ)"
