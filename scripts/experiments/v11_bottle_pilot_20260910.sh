#!/usr/bin/env bash
# Paired 10-round bottle pilot: policy/sonic_v1_1 against policy/release on the
# SAME fixed case bank, so the only variable is the checkpoint. Instance 1 /
# GPU1 only; GPU0 belongs to another session's training.
set -uo pipefail
cd /workspace/code/locomanipulation
O="$PWD/outputs/experiments/v11_bottle_pilot_20260910"
CASES="$O/cases10.json"
[ -f "$CASES" ] || { echo "missing $CASES"; exit 1; }

export INSTANCE=1 CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export SCENE_RANDOM=0 SIM_VIS=bbox,readout
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
PROMPT="pick up the water bottle"

# Only this instance's leftovers; never a bare pkill (it would take other
# sessions' sims down with it).
clean() {
    bash scripts/stack/stop_instance.sh 1 >/dev/null 2>&1
    for p in $(pgrep -f "root-output-""dir.*p03_i1" || true); do kill -9 "$p" 2>/dev/null; done
    sleep 3
}

gate() {
    "$PY" - <<'PYGATE'
import sys, time
from pathlib import Path
q, period = map(int, Path('/sys/fs/cgroup/cpu.max').read_text().split())
p = Path('/sys/fs/cgroup/cpu.stat')
read = lambda: {k: int(v) for k, v in (x.split() for x in p.read_text().splitlines())}
streak = 0
for _ in range(120):
    a = read(); t0 = time.monotonic(); time.sleep(5); b = read()
    free = q / period - (b['usage_usec'] - a['usage_usec']) / 1e6 / (time.monotonic() - t0)
    good = free >= 6 and b['throttled_usec'] == a['throttled_usec']
    streak = streak + 1 if good else 0
    print(f"free {free:.2f} streak {streak}", flush=True)
    if streak >= 3:
        sys.exit(0)
sys.exit(1)
PYGATE
}

trap 'clean; echo "ABORTED"; exit 130' INT TERM

for arm in sonic_v1_1 release; do
    out="$O/$arm"
    if [ -f "$out/done" ]; then echo "== $arm already done, skipping"; continue; fi
    mkdir -p "$out/videos"
    echo "===== arm $arm  $(date -u +%H:%M:%S)"
    gate > "$out/resources.log" || { echo "CPU gate timeout; no stack started"; exit 1; }
    clean
    SIM_VIDEO_DIR="$out/videos" SONIC_POLICY="policy/$arm" \
        bash scripts/stack/start_stack.sh "$PROMPT" > "$out/stack.log" 2>&1
    if ! grep -q "stack up" "$out/stack.log"; then
        echo "  stack failed to come up, see $out/stack.log"; clean; exit 1
    fi
    echo "  stack up $(date -u +%H:%M:%S), running 10 fixed cases"
    timeout 3600 "$PY" -u src/locomani/tasks/bottle/grasp_gym.py \
        --cases "$CASES" --rounds 10 --record --stand-retries 0 \
        --manifest "$out/manifest.jsonl" > "$out/gym.log" 2>&1
    echo "  gym exit $? $(date -u +%H:%M:%S)"
    grep -E "^\[gym\] [0-9]+/|^EPISODE " "$out/gym.log" | tail -12
    clean
    touch "$out/done"
done

echo "===== totals"
for arm in sonic_v1_1 release; do
    printf '%-12s ' "$arm"
    grep -h -oE "^EPISODE (OK|FAILED \[[a-z_]+\])" "$O/$arm/gym.log" 2>/dev/null | sort | uniq -c | tr '\n' ' '
    echo "  videos: $(ls "$O/$arm/videos"/*.mp4 2>/dev/null | wc -l)"
done
echo ALLDONE
