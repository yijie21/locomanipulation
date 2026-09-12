#!/bin/bash
# One experiment (2026-09-12): is the VR 3-point wrist channel live once the
# deploy is really in encoder mode "teleop"? Starts the stack on instance 0,
# runs locomani.backends.sonic.diag_vr3pt, stops the stack. Usage:
#   bash scripts/experiments/vr3pt_probe_20260912.sh policy/sonic_v1_1 [legs]
set -u
POLICY="${1:-policy/sonic_v1_1}"
LEGS="${2:-0,1,2,3,4,5}"
TAG=$(basename "$POLICY")
cd /workspace/code/locomanipulation
export INSTANCE=0
. scripts/paths.sh
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export SIM_VIDEO_DIR="$PWD/outputs/videos/vr3pt_20260912/$TAG"
mkdir -p "$SIM_VIDEO_DIR" outputs/diag/vr3pt_20260912
export SONIC_POLICY="$POLICY"
echo "[probe] $(date -u +%FT%TZ) policy=$POLICY legs=$LEGS"
bash scripts/stack/start_stack.sh "vr3pt probe $TAG" || { echo "[probe] start_stack failed rc=$?"; exit 3; }
PYTHONPATH=src "$VENV_SIM_PY" -m locomani.backends.sonic.diag_vr3pt --legs "$LEGS" \
    --json "outputs/diag/vr3pt_20260912/vr3pt_${TAG}.json"
rc=$?
echo "[probe] diag rc=$rc"
cp "$(locomani_session_root 0)/deploy.log" "outputs/diag/vr3pt_20260912/deploy_${TAG}.log" 2>/dev/null
bash scripts/stack/stop_instance.sh 0
echo "[probe] $(date -u +%FT%TZ) done"
exit $rc
