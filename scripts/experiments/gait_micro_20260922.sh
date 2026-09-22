#!/usr/bin/env bash
# Can SONIC v1.1 make small in-place adjustments? Three probes on the shuffle
# band (0.30-0.35, just above v1.1's speed step, ledger bottle.md 21):
#   --march : SLOW_WALK with a zero movement vector -- does it step in place?
#   --crab  : movement vector at 0/30/45/90 deg to the facing -- is there a
#             lateral/diagonal micro-step below the 0.40 command where the
#             lateral channel is known to open (ledger 25.1)?
# Measurement only; no control constant changes. Modelled on
# gait_speed_map_20260911.sh (single arm, no arm loop).
set -uo pipefail
cd /workspace/code/locomanipulation
O="${GAIT_OUT:-$PWD/outputs/diagnostics/gait_micro_v11_20260922}"
mkdir -p "$O"
SPEEDS="${GAIT_SPEEDS:-0.30,0.32,0.35}"
CRAB="${GAIT_CRAB:-0,30,45,90}"
MARCH="${GAIT_MARCH:-0.30,0.35,0.45}"
REPEATS="${GAIT_REPEATS:-3}"
ARM="${GAIT_ARM:-sonic_v1_1}"

export INSTANCE=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export SCENE_RANDOM=0 SIM_VIS=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
export SONIC_POLICY="policy/$ARM"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python

clean() { "$PY" -m locomani.stack.cli down --instance 0 >/dev/null 2>&1; sleep 3; }
trap 'clean; echo "ABORTED"; exit 130' INT TERM

alive() { ps -eo comm,args --no-headers | awk -v p="$1" '$1 ~ /^python/ && $0 ~ p {print}'; }
for pat in "run_sim_""locomani[.]py" "run_data_""exporter[.]py"; do
    hit=$(alive "$pat")
    [ -n "$hit" ] && { echo "REFUSING: '$pat' already running"; echo "$hit" | cut -c1-140; exit 1; }
done
[ -f "$O/STOP" ] && rm -f "$O/STOP"

echo "===== $ARM  $(date -u +%H:%M:%S)"
# no exporter: a recording stack's exporter holds ~2500 pids of a 3840 container cap
"$PY" -m locomani.stack.cli up --instance 0 --gpu 0 --no-record --policy "$SONIC_POLICY" > "$O/stack.log" 2>&1
rc=$?
[ "$rc" = 3 ] && { echo "instance-0 lock held; backing off"; tail -3 "$O/stack.log"; exit 3; }
[ "$rc" = 0 ] || { echo "stack failed rc=$rc, see $O/stack.log"; clean; exit 1; }
tail -1 "$O/stack.log"
timeout 3000 "$PY" -u -m locomani.backends.sonic.diag_shuffle \
    --speeds "$SPEEDS" --crab "$CRAB" --march "$MARCH" --repeats "$REPEATS" \
    --json "$O/legs.json" > "$O/sweep.log" 2>&1
echo "sweep exit $? $(date -u +%H:%M:%S)"
clean
touch "$O/done"
echo "===== done $(date -u +%FT%TZ)"
