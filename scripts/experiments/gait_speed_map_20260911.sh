#!/usr/bin/env bash
# Stage A of the v1.1 re-calibration (docs/ledger/bottle.md 20.4): measure the
# speed-command -> delivered-gait map for BOTH checkpoints in one session.
#
# The commanded `speed` is not m/s and its mapping belongs to the CHECKPOINT.
# Every walking constant in the repo was picked off release's map
# (primitives_tuning.py:306-317), so this is the measurement the rest of the
# re-calibration is arithmetic on. Measurement only: nothing here changes a
# control constant, and no object/grasp is involved.
set -uo pipefail
cd /workspace/code/locomanipulation
O="${GAIT_OUT:-$PWD/outputs/diagnostics/gait_speed_v11_20260911}"
mkdir -p "$O"
# Overridable so the knee can be filled in without re-running the whole grid
# (the first pass had one point between 0.26 and 0.40, and that is exactly
# where v1.1's curve steps).
SPEEDS="${GAIT_SPEEDS:-0.185,0.20,0.215,0.26,0.30,0.40,0.45}"
REPEATS="${GAIT_REPEATS:-3}"
ARMS="${GAIT_ARMS:-sonic_v1_1 release}"
# 0 = forward. The tabletop approach steers at the planned stand while the
# facing stays on the approach heading, so its movement vector is at an ANGLE
# to the facing and forward delivery does not imply lateral (ledger 24.5).
CRAB="${GAIT_CRAB:-0}"

export INSTANCE=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export SCENE_RANDOM=0 SIM_VIS=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python

clean() { bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1; sleep 3; }
trap 'clean; echo "ABORTED"; exit 130' INT TERM

# Refuse to start beside someone else's sim (the flock covers the port block,
# this covers a sim launched outside start_stack).
alive() { ps -eo comm,args --no-headers | awk -v p="$1" '$1 ~ /^python/ && $0 ~ p {print}'; }
for pat in "run_sim_""locomani[.]py" "run_data_""exporter[.]py"; do
    hit=$(alive "$pat")
    [ -n "$hit" ] && { echo "REFUSING: '$pat' already running"; echo "$hit" | cut -c1-140; exit 1; }
done

for arm in $ARMS; do
    out="$O/$arm"; mkdir -p "$out"
    if [ -f "$out/done" ]; then echo "== $arm already done"; continue; fi
    echo "===== arm $arm  $(date -u +%H:%M:%S)"
    clean
    # export, not a per-command prefix: the sweep process reads
    # locomani.paths.SONIC_POLICY for its own provenance line, and the first run
    # (2026-09-11) labelled the release arm's legs.json "sonic_v1_1" because the
    # prefix reached start_stack.sh only. The MEASUREMENT was unaffected (the
    # sweep just talks ZMQ to whatever deploy is up, and stack.log/deploy.log
    # both show policy/release), but the label in the artifact was wrong.
    export SONIC_POLICY="policy/$arm"
    bash scripts/stack/start_stack.sh "walk forward" > "$out/stack.log" 2>&1
    rc=$?
    if [ "$rc" = 3 ]; then echo "  instance-0 lock held; backing off"; tail -3 "$out/stack.log"; exit 3; fi
    grep -q "stack up" "$out/stack.log" || { echo "  stack failed, see $out/stack.log"; clean; exit 1; }
    grep -m1 "policy:" "$out/stack.log"
    timeout 2400 "$PY" -u -m locomani.backends.sonic.diag_shuffle \
        --speeds "$SPEEDS" --repeats "$REPEATS" --crab "$CRAB" --json "$out/legs.json" \
        > "$out/sweep.log" 2>&1
    echo "  sweep exit $? $(date -u +%H:%M:%S)"
    tail -32 "$out/sweep.log"
    clean
    touch "$out/done"
done
echo "===== done $(date -u +%FT%TZ)"
