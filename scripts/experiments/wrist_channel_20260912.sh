#!/usr/bin/env bash
# Ledger 28 reopened: measure the VR 3-point wrist channel on BOTH checkpoints.
#
# v1.1 is the checkpoint trained with wrist-pose augmentation (model card), and
# the observation configs carry the same vr_3point blocks, so the only way the
# two can differ is in what the policy does with an identical input. release is
# the control because it has a known-true prior (22%, the inverse of
# TRACK_ROT_BOOST = 4.5) -- without a leg whose answer is known in advance, a
# broken instrument reads as a measurement, which is exactly what happened on
# 2026-09-11 (both arms read ratio 0.01).
set -uo pipefail
cd /workspace/code/locomanipulation
O="$PWD/outputs/diagnostics/wrist_channel_20260912"
mkdir -p "$O"
export INSTANCE=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export SCENE_RANDOM=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
clean() { bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1; sleep 3; }
trap 'clean; echo ABORTED; exit 130' INT TERM
for arm in sonic_v1_1 release; do
    [ -f "$O/$arm.done" ] && { echo "== $arm already done"; continue; }
    echo "===== $arm  $(date -u +%H:%M:%S)"
    clean
    export SONIC_POLICY="policy/$arm"
    bash scripts/stack/start_stack.sh "wrist channel diag" > "$O/stack_$arm.log" 2>&1
    grep -q "stack up" "$O/stack_$arm.log" || { echo "stack failed"; tail -3 "$O/stack_$arm.log"; clean; exit 1; }
    # The provenance trap of ledger 21.5: trust the stack log, not the diag's
    # own self-report, for which checkpoint actually ran.
    grep -m1 "policy:" "$O/stack_$arm.log"
    timeout 600 "$PY" -u -m locomani.backends.sonic.diag_wrist \
        --side right --json "$O/wrist_$arm.json" > "$O/diag_$arm.log" 2>&1
    echo "  diag exit $?"
    tail -14 "$O/diag_$arm.log"
    clean; touch "$O/$arm.done"
done
echo "===== done $(date -u +%FT%TZ)"
