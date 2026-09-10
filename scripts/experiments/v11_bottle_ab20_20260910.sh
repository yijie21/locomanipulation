#!/usr/bin/env bash
# Clean 20-round A/B: policy/release vs policy/sonic_v1_1 over ONE fixed case
# bank, instance 0 with all GPU work on GPU1. Release runs first this time so a
# systematic order effect would show up against the 2026-09-10 pilot, whose
# first ten cases are byte-identical to this bank's first ten.
set -uo pipefail
cd /workspace/code/locomanipulation
O="$PWD/outputs/experiments/v11_bottle_ab20_20260910"
CASES="$O/cases20.json"
TASKCFG="configs/tasks/bottle_visible.json"
ROUNDS=20
[ -f "$CASES" ] || { echo "missing $CASES"; exit 1; }

export INSTANCE=0 CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export SCENE_RANDOM=0 SIM_VIS=bbox,readout
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
PROMPT="pick up the water bottle"

# start_stack.sh now takes a per-instance flock (session 65, 2026-09-10), so the
# old wait-for-ports loop here is redundant: a second stack refuses loudly
# instead of racing. Teardown is just the scoped stop.
clean() {
    bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1
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

# pgrep -f matches any shell whose command line contains the name; require python.
alive() { ps -eo comm,args --no-headers | awk -v p="$1" '$1 ~ /^python/ && $0 ~ p {print}'; }
for pat in "run_sim_""locomani[.]py" "run_data_""exporter[.]py"; do
    hit=$(alive "$pat")
    if [ -n "$hit" ]; then
        echo "REFUSING: a python process matching '$pat' is already running"
        echo "$hit" | cut -c1-140; exit 1
    fi
done

trap 'clean; echo "ABORTED"; exit 130' INT TERM

for arm in release sonic_v1_1; do
    out="$O/$arm"
    if [ -f "$out/done" ]; then echo "== $arm already done, skipping"; continue; fi
    mkdir -p "$out/videos"
    echo "===== arm $arm  $(date -u +%H:%M:%S)"
    gate > "$out/resources.log" || { echo "CPU gate timeout; no stack started"; exit 1; }
    clean
    SIM_VIDEO_DIR="$out/videos" SONIC_POLICY="policy/$arm" \
        bash scripts/stack/start_stack.sh "$PROMPT" > "$out/stack.log" 2>&1
    rc=$?
    if [ "$rc" = 3 ]; then
        echo "  another session holds the instance-0 lock; backing off"; tail -3 "$out/stack.log"; exit 3
    fi
    if ! grep -q "stack up" "$out/stack.log"; then
        echo "  stack failed to come up, see $out/stack.log"; clean; exit 1
    fi
    echo "  stack up $(date -u +%H:%M:%S), running $ROUNDS fixed cases"
    timeout 5400 "$PY" -u src/locomani/tasks/bottle/grasp_gym.py \
        --reach-config "$O/reach_visible.json" \
        --task-config "$TASKCFG" \
        --targets water_bottle \
        --spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 20.0 \
        --spawn-lateral 0.2 --edge-jitter 0.2 \
        --stand-retries 0 --record --video-include-init \
        --cases "$CASES" --rounds "$ROUNDS" \
        --manifest "$out/manifest.jsonl" > "$out/gym.log" 2>&1
    echo "  gym exit $? $(date -u +%H:%M:%S)"
    grep -E "^\[gym\] [0-9]+/" "$out/gym.log" | tail -2
    clean
    touch "$out/done"
done

echo "===== totals"
for arm in release sonic_v1_1; do
    printf '%-12s ' "$arm"
    grep -h -oE "^EPISODE (OK|FAILED \[[a-z_]+\])" "$O/$arm/gym.log" 2>/dev/null | sort | uniq -c | tr '\n' ' '
    echo "  videos: $(ls "$O/$arm/videos"/*.mp4 2>/dev/null | wc -l)"
done
echo ALLDONE
