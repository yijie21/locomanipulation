#!/usr/bin/env bash
# Stage B of the v1.1 re-calibration (ledger bottle.md 21.4): 10 rounds on v1.1
# with APPR_COARSE re-picked 0.30 -> 0.40, over the FIRST TEN cases of the
# 2026-09-10 A/B bank, so the result is paired against two existing arms on the
# same cases: release 4/10 and v1.1-with-0.30 0/10.
#
# One variable: only COARSE moved (gait_speeds.py says why MID/SHUFFLE cannot).
# The acceptance is telemetry, not a success rate -- no `stalled` approach exit,
# the coarse phase down from ~13 s to ~3 s, and the brake/park distribution.
set -uo pipefail
cd /workspace/code/locomanipulation
AB="$PWD/outputs/experiments/v11_bottle_ab20_20260910"
O="$PWD/outputs/experiments/stageb_coarse_pilot_20260911"
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
