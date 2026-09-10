#!/usr/bin/env bash
# Paired 10-round bottle pilot: policy/sonic_v1_1 against policy/release on the
# SAME fixed case bank, so the only variable is the checkpoint. Instance 1 /
# GPU1 only; GPU0 belongs to another session's training.
set -uo pipefail
cd /workspace/code/locomanipulation
O="$PWD/outputs/experiments/v11_bottle_pilot_20260910"
CASES="$O/cases10_visible.json"
# The collection profile, not hand-rolled flags: bottle_visible.json is what the
# visible200 line actually ran (ready_grasp_recovery / visible_expert / the
# 1.1-1.3 m spawn band / stand_retries 0 / restart_deploy false). Without it the
# reach config falls back to the dataclass defaults, ready_grasp_recovery is
# False, and every round dies at the grasp gate with "is retired".
TASKCFG="configs/tasks/bottle_visible.json"
[ -f "$CASES" ] || { echo "missing $CASES"; exit 1; }

# Instance 0, GPU1. The exporter only runs on instance 0 ("instance 1 records no
# episodes"), and the gym resolves its ports from the instance-0 defaults, so a
# recording run has to BE instance 0. The GPU is a separate choice: GPU0 is
# another session's training.
export INSTANCE=0 CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
export SCENE_RANDOM=0 SIM_VIS=bbox,readout
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
PROMPT="pick up the water bottle"

# Only this instance's leftovers; never a bare pkill (it would take other
# sessions' sims down with it).
clean() {
    bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1
    :
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

# Never start on top of another session's stack: stop_instance.sh 0 kills the
# exporter by pattern, which is not instance-scoped.
# pgrep -f matches any SHELL whose command line merely contains the name, so
# require the process to actually be a python interpreter.
alive() { ps -eo comm,args --no-headers | awk -v p="$1" '$1 ~ /^python/ && $0 ~ p {print}'; }
for pat in "run_sim_""locomani[.]py" "run_data_""exporter[.]py"; do
    hit=$(alive "$pat")
    if [ -n "$hit" ]; then
        echo "REFUSING: a python process matching '$pat' is already running"
        echo "$hit" | cut -c1-140; exit 1
    fi
done

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
        --task-config "$TASKCFG" --cases "$CASES" --rounds 10 --record \
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
