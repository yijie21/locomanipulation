#!/bin/bash
# Diagnosis step 1: re-run the first few visible200 eval rounds with OBS_DUMP on,
# so the observations the policy actually saw can be replayed offline
# (src/locomani/pipeline/replay_obs.py --dump).
#
#   bash scripts/experiments/visible200_obsdump_20260911.sh [ROUNDS]
#
# The protocol is copied from run_learn_pilot.py's eval stage -- the one that
# produced the 2/100 -- NOT from run_eval_arm.sh, which is a different
# calibration (memory two-eval-calibrations-in-the-ledger). Same config file,
# same --set overrides, same reach config, same initial token, same seed 6000,
# so rounds 0..N-1 draw the same cases the 100-round eval drew. Only two things
# differ: the round count, and backend.obs_dump.
#
# The scene is deliberately NOT rebuilt: that stage inherits outputs/scene/sim,
# which is byte-identical to the eval's frozen copy under the experiment dir
# (checked 2026-09-11). Rebuilding it here would change the table height and
# make the dumped observations a different distribution than the 2/100 run's.
set -uo pipefail
# Absolute, not $0-relative: this script is executed from a scratch COPY (a bash
# script edited while running is read by byte offset and corrupts -- memory
# never-edit-a-running-bash-script), and a copy's $0 does not locate the repo.
REPO=/workspace/code/locomanipulation
cd "$REPO" || exit 1

ROUNDS=${1:-5}
NAME=visible200_obsdump_20260911
OUT=outputs/experiments/$NAME
SRC=outputs/experiments/visible200_20260910
INSTANCE=0
PORT=5590
GPU=0

SIM=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
TRAIN=third_party/Isaac-GR00T/.venv/bin/python
CKPT=$SRC/model/checkpoint-10000

[ -d "$OUT" ] && { echo "refusing to overwrite $OUT -- move it or pick a new NAME"; exit 1; }
mkdir -p "$OUT/eval" "$OUT/eval/videos" "$OUT/obs"

# Admission: this takes instance 0's whole port block. start_stack.sh holds a
# per-instance flock and exits 3 if another runner has it, but check the cheap
# things first so a busy machine fails here rather than half-way in.
for p in 5555 5556 5580 5581 5582 5590; do
  ss -ltn 2>/dev/null | grep -q ":$p " && { echo "ABORT: port $p already listening"; exit 1; }
done
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
[ "$free_mib" -lt 20000 ] && { echo "ABORT: GPU$GPU has only ${free_mib} MiB free"; exit 1; }
echo "ADMITTED rounds=$ROUNDS gpu$GPU free=${free_mib}MiB pids=$(cat /sys/fs/cgroup/pids.current)"

env CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU NO_ENCODER=1 \
    SIM_VIDEO_DIR="$PWD/$OUT/eval/videos" \
    bash scripts/stack/start_stack.sh "pick up the water bottle" \
    > "$OUT/start_stack.log" 2>&1
rc=$?
grep -qi "stack up" "$OUT/start_stack.log" \
  || { echo "STACK_FAILED rc=$rc"; tail -12 "$OUT/start_stack.log"; exit 1; }
echo "STACK_UP"

CUDA_VISIBLE_DEVICES=$GPU nohup "$TRAIN" src/locomani/pipeline/serve_policy.py \
  --model-path "$CKPT" --port "$PORT" > "$OUT/eval/serve.log" 2>&1 &
for _ in $(seq 1 120); do
  grep -q "listening" "$OUT/eval/serve.log" 2>/dev/null && break; sleep 3
done
grep -q "listening" "$OUT/eval/serve.log" \
  || { echo "SERVE_FAILED"; tail -5 "$OUT/eval/serve.log"; bash scripts/stack/stop_instance.sh $INSTANCE $PORT >/dev/null 2>&1; exit 1; }
echo "SERVE_UP"

env NO_ENCODER=1 SIM_VIDEO_DIR="$PWD/$OUT/eval/videos" \
  "$SIM" -u src/locomani/pipeline/eval_closed_loop.py \
    --config configs/eval/closed_loop_100.json \
    --set rounds=$ROUNDS \
    --set seed=6000 \
    --set stand_retries=0 \
    --set backend.dataset_fps=25 \
    --set "backend.obs_dump=$PWD/$OUT/obs" \
    --reach-config "$PWD/$SRC/reach_config.json" \
    --initial-token-file "$PWD/configs/deployment/ready_grasp_initial_token.json" \
    --out "$PWD/$OUT/eval/closed_loop.json" \
    --trace-dir "$PWD/$OUT/eval/traces" \
    > "$OUT/eval.log" 2>&1
echo "EVAL_RC=$?"

bash scripts/stack/stop_instance.sh $INSTANCE $PORT > /dev/null 2>&1
ok=$(grep -c '"ok": true' "$OUT/eval/closed_loop.manifest.jsonl" 2>/dev/null || echo 0)
echo "DONE ok=$ok traces=$(ls "$OUT/eval"/traces/round_*.json 2>/dev/null | wc -l) dumps=$(ls "$OUT"/obs/obs_*.npz 2>/dev/null | wc -l) dumpsize=$(du -sh "$OUT/obs" 2>/dev/null | cut -f1)"
