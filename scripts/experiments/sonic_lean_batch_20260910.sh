#!/usr/bin/env bash
# Phase-1 physical reproduction. Requires the already-owned instance 1 stack.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=outputs/diagnostics/sonic_lean_20260910/live
mkdir -p "$OUT"
# An exclusive lease prevents two copies commanding this diagnostic instance.
exec 9>"$OUT/batch.lock"
flock -n 9 || exit 1
export TF_CAMERA_PORT=5565 TF_SONIC_PORT=5566 TF_OUT_PORT=5567 TF_KEYBOARD_PORT=5590
export TF_PRIVILEGED_PORT=5591 TF_EPISODE_PORT=5592 SONIC_DDS_DOMAIN=1 CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
cleanup() {
    rc=$?
    trap - EXIT
    bash scripts/stack/stop_instance.sh 1
    printf '%s\n' "$rc" > "$OUT/batch.exit"
    exit "$rc"
}
trap cleanup EXIT
for rep in 1 2 3 4 5; do
    # Alternate order to expose carry-over effects across resets.
    arms='neutral reach'
    if (( rep % 2 == 0 )); then arms='reach neutral'; fi
    for arm in $arms; do
        file="$OUT/${arm}_${rep}.json"
        if [[ -f "$file" ]]; then
            # Resume only a complete, successful case with both channel traces.
            "$PY" - "$file" <<'PYVERIFY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = json.loads(p.read_text())
assert len(rows) == 12 and all(not r['fallen'] for r in rows), 'incomplete/failed case: inspect manually'
for channel in ['planner', 'streamed']:
    r = [x for x in rows if x['channel'] == channel]
    assert [x['cmd'] for x in r] == [0, .1, .2, .3, .4, .52]
    trace = json.loads(Path(str(p) + '.' + channel + '.trace.json').read_text())
    assert trace and all(not x['fallen'] for x in trace)
log = p.with_suffix('.log').read_text()
assert '[lean] wrote ' in log and 'Traceback' not in log, 'case did not finish cleanly'
PYVERIFY
            echo "SKIP verified complete $arm repeat $rep"
            continue
        fi
        "$PY" - <<'PY' >> "$OUT/resources.jsonl"
import json, time
from pathlib import Path
p=Path('/sys/fs/cgroup/cpu.stat')
def read(): return {k:int(v) for k,v in (l.split() for l in p.read_text().splitlines())}
quota, period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
streak=0
for attempt in range(120):
    a=read(); t=time.monotonic(); time.sleep(5); b=read(); dt=time.monotonic()-t
    free=int(quota)/int(period)-(b['usage_usec']-a['usage_usec'])/1e6/dt
    pids=int(Path('/sys/fs/cgroup/pids.current').read_text())
    throttle=b['throttled_usec']-a['throttled_usec']
    good=free >= 6 and pids < 2800 and throttle == 0
    streak=streak+1 if good else 0
    print(json.dumps(dict(wall=time.time(), cpu_free=free, pids=pids, throttled_usec=throttle, consecutive_passes=streak)),flush=True)
    if streak >= 3:
        break
else:
    raise RuntimeError('resource gate did not stabilize in 10 minutes')
PY
        echo "RUN $arm repeat $rep $(date -u +%FT%TZ)"
        timeout 180 "$PY" -u scripts/experiments/sonic_lean_live_20260910.py \
            --arm "$arm" --out "$file" > "$OUT/${arm}_${rep}.log" 2>&1
    done
done
