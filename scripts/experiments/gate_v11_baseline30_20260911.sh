#!/usr/bin/env bash
# Stage E: record the bottle gate's baseline on v1.1, because the one in
# tests/gate/baseline_bottle.json came from a release-era run
# (outputs/experiments/restart_ab_20260910) and every metric in it -- stand
# medians, durations, the failure-reason set -- is release's.
#
# THIRTY rounds, matching the release-era baseline's size. A 10-round baseline
# cannot support the gate's +-0.15 yield tolerance: at p=0.3 the binomial sd over
# 10 rounds is 0.145, so a candidate run would trip it by chance alone.
#
# Deliberately a COLLECTION-style batch: sampled cases (no fixed bank) and the
# REPO profile, which now carries stand_azimuth_deg 29 (ledger 26/27), so the
# baseline records the line as production actually runs it. Recording before
# that landed would have enshrined the 0/20 configuration.
set -uo pipefail
cd /workspace/code/locomanipulation
AB="$PWD/outputs/experiments/v11_bottle_ab20_20260910"
O="$PWD/outputs/experiments/gate_v11_baseline30_20260911"
mkdir -p "$O/videos"
ROUNDS=30

export INSTANCE=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export SCENE_RANDOM=0 SIM_VIS=bbox,readout
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
# The point of the run: the default checkpoint. Stated anyway so the log records it.
export SONIC_POLICY="${SONIC_POLICY:-policy/sonic_v1_1}"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
PROMPT="pick up the water bottle"

clean() { bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1; sleep 3; }
trap 'clean; echo "ABORTED"; exit 130' INT TERM

# Scoped to OUR instance. The first version of this guard matched any
# run_sim_locomani.py and refused because another session was running one on
# instance 1 -- the same shape as ports.require_exporter matching only the
# default entry point (STATE, 2026-09-10). The real cross-session conflict is
# the port block, and start_stack.sh takes a flock for that; this guard only
# has to catch a sim left behind on THIS instance. A sim with no --instance flag
# is instance 0.
alive_here() {
    ps -eo comm,args --no-headers | awk -v p="$1" -v i="$INSTANCE" '
        $1 ~ /^python/ && $0 ~ p {
            n = "0"; for (k = 1; k < NF; k++) if ($k == "--instance") n = $(k+1)
            if (n == i) print
        }'
}
for pat in "run_sim_""locomani[.]py" "run_data_""exporter[.]py"; do
    hit=$(alive_here "$pat")
    [ -n "$hit" ] && { echo "REFUSING: '$pat' already running on instance $INSTANCE"; echo "$hit" | cut -c1-140; exit 1; }
done

[ -f "$O/done" ] && { echo "already done"; exit 0; }
clean
SIM_VIDEO_DIR="$O/videos" bash scripts/stack/start_stack.sh "$PROMPT" > "$O/stack.log" 2>&1
rc=$?
[ "$rc" = 3 ] && { echo "instance-0 lock held"; tail -3 "$O/stack.log"; exit 3; }
grep -q "stack up" "$O/stack.log" || { echo "stack failed, see $O/stack.log"; clean; exit 1; }
grep -m1 "policy:" "$O/stack.log"
echo "stack up $(date -u +%H:%M:%S), $ROUNDS fixed cases"
timeout 3600 "$PY" -u src/locomani/tasks/bottle/grasp_gym.py \
    --reach-config configs/collection/ready_grasp_visible.json \
    --task-config configs/tasks/bottle_visible.json \
    --targets water_bottle \
    --spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 20.0 \
    --spawn-lateral 0.2 --edge-jitter 0.2 \
    --stand-retries 0 --record --video-include-init \
    --rounds "$ROUNDS" \
    --manifest "$O/manifest.jsonl" > "$O/gym.log" 2>&1
echo "gym exit $? $(date -u +%H:%M:%S)"
grep -E "^\[gym\] [0-9]+/" "$O/gym.log" | tail -2
clean
touch "$O/done"
echo "===== done $(date -u +%FT%TZ)"
