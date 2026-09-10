#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="$PWD/outputs/diagnostics/sonic_lean_phase2_20260910"
exec 9>"$OUT/run.lock"
flock -n 9 || exit 1
[[ ! -e "$OUT/encoder.bin" ]] || { echo 'Existing encoder capture: inspect before another run'; exit 1; }
export INSTANCE=1 CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export TF_CAMERA_PORT=5565 TF_SONIC_PORT=5566 TF_OUT_PORT=5567 TF_KEYBOARD_PORT=5590
export TF_PRIVILEGED_PORT=5591 TF_EPISODE_PORT=5592 SONIC_DDS_DOMAIN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/src"
export SONIC_DEPLOY_BIN="$OUT/deploy/target/release/g1_deploy_onnx_ref"
export SONIC_ENCODER_AUDIT_FILE="$OUT/encoder.bin"
cleanup() {
 rc=$?
 trap - EXIT
 bash scripts/stack/stop_instance.sh 1
 echo "$rc" > "$OUT/run.exit"
 exit "$rc"
}
trap cleanup EXIT
bash scripts/stack/start_stack.sh 'SONIC coordinated reference diagnosis' > "$OUT/stack.log" 2>&1
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u scripts/experiments/sonic_lean_phase2_20260910.py --repeats 3 > "$OUT/cases.log" 2>&1
# The binary writer uses buffered I/O; report tolerates a partial final record.
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/sonic_encoder_capture_report_20260910.py > "$OUT/report.log" 2>&1
