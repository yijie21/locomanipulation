#!/usr/bin/env bash
# THIRD attempt, and the CONFIRMATION run: the full 20-case bank at plan 29 deg,
# so the result is paired against release's own 20 rounds (9/20) on the same
# cases rather than against a 10-round subset.
#
# SECOND attempt. The first (stagee_az29_20260911) was VOID: with --cases the
# planned stand comes from the BANK, so round_loop never calls sample_stand and
# `stand_azimuth_deg` was never consulted -- a config that silently did nothing,
# the exact failure shape in docs/reference/silent_failures.md. This run uses a
# bank derived from the same ten cases with ONLY `rel` rotated to 29 deg and
# `stand` recomputed from it (can_xy / psi / spawn_off / spawn_lat / dpsi /
# appearance byte-identical), so the planned stand is the only difference.
#
# Plan the stand at 29 deg instead of the 37.5 the depth-only backshift produces
# (ledger bottle.md 25): the FK template's comfortable centre is at 30.3 deg,
# release never executed the 37.5 because its lateral channel is dead at its
# COARSE speed, and the achieved azimuth is what predicts the grasp -- on
# release's own 20 rounds [0,30) is 5/7 ok against [38,50) 1/6.
#
# Same ten cases as every other arm today. ACCEPTANCE, in order: the ACHIEVED
# azimuth must come down (v1.1 tracks the plan to +2.5..+5.4 deg, so expect
# ~32-35); `stand infeasible` and `local alignment outside comfortable IK`
# must drop from 4/10 and 2/10; and no new failure kind. The success RATE is
# not the acceptance -- ten rounds cannot measure it (ledger 23.3).
set -uo pipefail
cd /workspace/code/locomanipulation
AB="$PWD/outputs/experiments/v11_bottle_ab20_20260910"
O="$PWD/outputs/experiments/stagef_az29_20260911"
mkdir -p "$O/videos"
ROUNDS=20

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

# Scoped to OUR instance. The first version of this guard matched any
# run_sim_locomani.py and refused because another session was running one on
# instance 1 -- the same shape as ports.require_exporter matching only the
# default entry point (STATE, 2026-09-10). The real cross-session conflict is
# the port block, and start_stack.sh takes a flock for that; this guard only
# has to catch a sim left behind on THIS instance. A sim with no --instance flag
# is instance 0.
alive_here() {
    ps -eo comm,args --no-headers | awk -v p="$1" -v i="$INSTANCE" '
        $1 ~ /^python/ && $0 ~ p {
            n = "0"; for (k = 1; k < NF; k++) if ($k == "--instance") n = $(k+1)
            if (n == i) print
        }'
}
for pat in "run_sim_""locomani[.]py" "run_data_""exporter[.]py"; do
    hit=$(alive_here "$pat")
    [ -n "$hit" ] && { echo "REFUSING: '$pat' already running on instance $INSTANCE"; echo "$hit" | cut -c1-140; exit 1; }
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
    --reach-config "$PWD/outputs/experiments/stagee_az29_20260911/reach_visible_az29.json" \
    --task-config configs/tasks/bottle_visible.json \
    --targets water_bottle \
    --spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 20.0 \
    --spawn-lateral 0.2 --edge-jitter 0.2 \
    --stand-retries 0 --record --video-include-init \
    --cases "$PWD/outputs/experiments/stagef_az29_20260911/cases20_az29.json" --rounds "$ROUNDS" \
    --manifest "$O/manifest.jsonl" > "$O/gym.log" 2>&1
echo "gym exit $? $(date -u +%H:%M:%S)"
grep -E "^\[gym\] [0-9]+/" "$O/gym.log" | tail -2
clean
touch "$O/done"
echo "===== done $(date -u +%FT%TZ)"
