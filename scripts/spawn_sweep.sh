#!/bin/bash
# Randomised spawn-distance sweep: TWO instances, SAME config, DIFFERENT seeds.
#
#   nohup bash scripts/spawn_sweep.sh outputs/eval/spawn_sweep > outputs/logs/spawn_sweep.log 2>&1 &
#
# WHY NOT seq_ab.sh. seq_ab pairs two ARMS on one seed to compare two configs.
# Here there is only one config; the independent variable is --spawn-edge, and
# it is already randomised inside the run. So the two instances are just two
# halves of one sample, run on different seeds and pooled -- 200 rounds in one
# night instead of 100.
#
# WHY A SWEEP AND NOT TWO FIXED ARMS. Measured 2026-08-30: the SAME checkpoint,
# same seed, same budget scored 9/20 with --spawn-edge 1.2 and 1/12 with
# --spawn-edge 1.1 1.3. Absolute rates are not comparable across runs, so a
# two-point comparison of two separate runs cannot be trusted. Inside ONE run
# the spawn is drawn per round, which makes spawn -> success causal and needs
# no cross-run assumption.
#
# WHAT IT IS TESTING. Pooled over four earlier runs, success binned by the
# distance the robot actually PARKS at peaks hard in [0.40, 0.45) -- 93% at the
# 1.1-1.3 m spawn, 40% at the 0.7 m spawn -- which is exactly the expert's own
# gripper-close distance (p25 0.383 / median 0.422 / p75 0.452). But park
# distance is an OUTCOME, not a treatment. Spawn distance IS a treatment. A
# two-point fit says park = 0.383 + 0.194 x (spawn - 0.7), so the sweet spot
# should sit near --spawn-edge 0.9-1.0. This sweep either finds that peak or
# kills the whole story.
#
# READING THE RESULT: analyse with scripts/spawn_sweep_report.py, which reports
# the pre-VLA (view-gate) rejections SEPARATELY. They rise with spawn distance
# for a reason that has nothing to do with the task -- the head camera is
# pitched 52 deg down and a tabletop object leaves the top of the frame -- so
# leaving them in the denominator would depress the far end of the curve for
# the wrong reason.
set -uo pipefail
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

OUT_DIR="${1:?usage: spawn_sweep.sh <out_dir>}"
ROUNDS="${ROUNDS:-100}"
BUDGET="${BUDGET:-90}"
EDGE="${EDGE:-0.7 1.4}"
SEED0="${SEED0:-7000}"
SEED1="${SEED1:-8000}"
CKPT="${CKPT:-$LOCOMANI_OUTPUTS/train/walk200/checkpoint-10000}"
DATASET_FPS="${DATASET_FPS:-25}"
TASK="${TASK:-pick up the water bottle}"
COMMON="${COMMON:---client async}"
GR00T_PY="${GR00T_PY:-$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python}"
PY="$VENV_SIM_PY"
LOGS="$LOCOMANI_TMP/spawn_sweep"
mkdir -p "$OUT_DIR" "$LOGS"
[ -d "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }

P_SRV=$(printf '%s_%s.py' serve policy)

kill_instance() {
    local i=$1 port pid
    for port in $((5555+10*$1)) $((5581+10*$1)) $((5557+10*$1)); do
        pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -1)
        [ -n "${pid:-}" ] && kill -9 "$pid" 2>/dev/null
    done
    for pid in $(pgrep -f "g1_deploy" | head -20); do
        if { tr '\0' ' ' < "/proc/$pid/environ" | grep -q "SONIC_DDS_DOMAIN=$i"; } 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    for pid in $(pgrep -f run_data_exporter); do kill -9 "$pid" 2>/dev/null; done
    return 0
}

PIDS=""
cleanup() {
    for p in $PIDS; do kill "$p" 2>/dev/null; done
    sleep 2
    for p in $PIDS; do kill -9 "$p" 2>/dev/null; done
    for p in $(pgrep -f "eval_closed_loop.*$OUT_DIR" || true); do kill -9 "$p" 2>/dev/null; done
    for p in $(pgrep -f "$P_SRV" || true); do kill -9 "$p" 2>/dev/null; done
    kill_instance 0; kill_instance 1
}
trap cleanup EXIT INT TERM

for pid in $(pgrep -f "$P_SRV" || true); do kill -9 "$pid" 2>/dev/null; done
sleep 1
for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i nohup "$GR00T_PY" "$LOCOMANI_SRC/pipeline/serve_policy.py" \
        --model-path "$CKPT" --port $((6590+i)) > "$LOGS/serve_$i.log" 2>&1 &
    PIDS="$PIDS $!"
done
echo "[sweep] loading policy servers..."
for i in 0 1; do
    up=0
    for _ in $(seq 1 90); do
        grep -q "\[serve_policy\]" "$LOGS/serve_$i.log" 2>/dev/null && { up=1; break; }; sleep 5
    done
    [ "$up" = 1 ] || { echo "server $i never came up"; tail -5 "$LOGS/serve_$i.log"; exit 1; }
done
echo "[sweep] servers up"

for i in 0 1; do
    kill_instance $i; sleep 2
    env INSTANCE=$i NO_ENCODER=1 SIM_VIDEO_DIR="$LOCOMANI_OUTPUTS/videos/sweep_i$i" \
        nohup bash scripts/stack/start_stack.sh "$TASK" > "$LOGS/stack_$i.log" 2>&1 &
    ok=0
    for _ in $(seq 1 90); do
        grep -q "stack up" "$LOGS/stack_$i.log" 2>/dev/null && { ok=1; break; }; sleep 5
    done
    [ "$ok" = 1 ] || { echo "stack $i never came up"; tail -20 "$LOGS/stack_$i.log"; exit 1; }
done
echo "[sweep] stacks up"

for i in 0 1; do
    seed=$([ "$i" = 0 ] && echo "$SEED0" || echo "$SEED1")
    # shellcheck disable=SC2086
    env INSTANCE=$i SONIC_DDS_DOMAIN=$i NO_ENCODER=1 EVAL_PARALLEL=1 \
        TF_CAMERA_PORT=$((5555+10*i)) TF_SONIC_PORT=$((5556+10*i)) \
        TF_OUT_PORT=$((5557+10*i)) TF_KEYBOARD_PORT=$((5580+10*i)) \
        TF_PRIVILEGED_PORT=$((5581+10*i)) TF_EPISODE_PORT=$((5582+10*i)) \
        TF_DATASET_FPS="$DATASET_FPS" \
        nohup "$PY" -u src/locomani/pipeline/eval_closed_loop.py \
        --fresh "$ROUNDS" --seed "$seed" --budget "$BUDGET" \
        --spawn-edge $EDGE --task "$TASK" --policy-port $((6590+i)) $COMMON \
        --out "$OUT_DIR/half$i.json" --trace-dir "$OUT_DIR/half${i}_traces" \
        > "$LOGS/eval_$i.log" 2>&1 &
    PIDS="$PIDS $!"
done
cat > "$OUT_DIR/run_meta.json" <<META
{"sweep": true, "rounds_per_half": $ROUNDS, "budget_s": $BUDGET,
 "spawn_edge": "$EDGE", "seeds": [$SEED0, $SEED1], "ckpt": "$CKPT",
 "dataset_fps": $DATASET_FPS, "common": "$COMMON", "task": "$TASK"}
META
echo "[sweep] both halves launched: seeds $SEED0 / $SEED1, --spawn-edge $EDGE, $ROUNDS rounds each"
wait
echo "[sweep] done"
"$PY" scripts/spawn_sweep_report.py "$OUT_DIR" || true
