#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${SONIC_NATIVE_OUT:-$PWD/outputs/diagnostics/sonic_native_phase3_20260910/retry_reset}"
export SONIC_NATIVE_OUT="$OUT"
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || exit 1
export INSTANCE=1 SONIC_DDS_DOMAIN=1 CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export TF_PRIVILEGED_PORT=5591 TF_EPISODE_PORT=5592 TF_SONIC_PORT=5566 TF_OUT_PORT=5567
export TF_CAMERA_PORT=5565 TF_KEYBOARD_PORT=5590 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
# Video is opt-in: the recorder has to exist before the stack starts, so
# SIM_VIDEO_DIR cannot be set later from the replay process.
if [[ -n "${SONIC_NATIVE_VIDEO:-}" ]]; then export SIM_VIDEO_DIR="$OUT/videos"; mkdir -p "$SIM_VIDEO_DIR"; fi
source scripts/env_deploy.sh
cleanup() {
 rc=$?
 trap - EXIT
 if [[ "${started:-0}" == 1 ]]; then bash scripts/stack/stop_instance.sh 1; fi
 echo "$rc" > "$OUT/run.exit"
 exit "$rc"
}
trap cleanup EXIT
# Wait for three consecutive headroom windows before creating a stack.
python - <<'PYGATE' > "$OUT/resources.jsonl"
import json, time
from pathlib import Path
p=Path('/sys/fs/cgroup/cpu.stat')
q, period=map(int, Path('/sys/fs/cgroup/cpu.max').read_text().split())
def read(): return dict((k,int(v)) for k,v in (x.split() for x in p.read_text().splitlines()))
streak=0
for _ in range(120):
    a=read(); start=time.monotonic(); time.sleep(5); b=read()
    free=q/period-(b['usage_usec']-a['usage_usec'])/1e6/(time.monotonic()-start)
    good=free>=6 and b['throttled_usec']==a['throttled_usec']
    streak=streak+1 if good else 0
    print(json.dumps(dict(wall=time.time(),free=free,streak=streak)),flush=True)
    if streak>=3: break
else: raise SystemExit('CPU gate timeout; no stack started')
PYGATE
started=1
bash scripts/stack/start_stack.sh 'SONIC native motion diagnosis' > "$OUT/stack.log" 2>&1
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u scripts/experiments/sonic_native_replay_20260910.py > "$OUT/cases.log" 2>&1
