#!/bin/bash
# Closed-loop a checkpoint: 20 rounds for a fast read, then resume to 100.
#
#   RUN=ramp200 nohup bash scripts/run_closed_loop.sh > outputs/logs/cl_ramp200.log 2>&1 &
#
# CKPT defaults to the LAST checkpoint in the run's directory -- the user's
# standing choice (2026-08-29): no open-loop selector, the longest-trained
# weights are the ones evaluated. That costs the selector's protection against
# late-checkpoint overfitting (fast200's open-loop picked 9000 over 10000), and
# it buys back the ~2 h the four-horizon sweep was spending to provide it.
#
# The three settings that fail SILENTLY, written here instead of remembered:
#   NO_ENCODER=1     the token-only deploy mode eval_closed_loop CHECKS for.
#   TF_DATASET_FPS   must equal the fps of the data the checkpoint was trained
#                    on. Wrong value replays every motion at the wrong speed
#                    and raises nothing.
#   --seed 6000      what round 3 (fast200, 28.0%) and round 4 used, so the
#                    fresh spawns are the same draws and the comparison is
#                    PAIRED. A different seed makes it incomparable.
#
# Both passes write the SAME --out: --resume derives its manifest from it
# (eval_closed_loop.py:181), so a different name would start over at round 1.
set -uo pipefail

# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

RUN="${RUN:-ramp200}"
CKPT_DIR="$LOCOMANI_OUTPUTS/train/$RUN"
CKPT="${CKPT:-$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)}"
EVAL_DIR="${EVAL_DIR:-$LOCOMANI_OUTPUTS/eval/$RUN}"
EVAL_SEED="${EVAL_SEED:-6000}"
DATASET_FPS="${DATASET_FPS:-25}"
ROUNDS="${ROUNDS:-20 100}"
REPORT="${REPORT:-$LOCOMANI_OUTPUTS/logs/${RUN}_report.md}"
GR00T_PY="$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python"

[ -n "$CKPT" ] && [ -d "$CKPT" ] || { echo "no checkpoint in $CKPT_DIR"; exit 1; }
mkdir -p "$EVAL_DIR"
say() { echo; echo "=========== $* ($(date '+%F %T')) ==========="; }
note() { echo "$*" >> "$REPORT"; }

kill_stack() {
    local P_SIM P_DEP P_EXP P_SRV pat pid
    P_SIM=$(printf '%s_%s' run_sim locomani); P_DEP=$(printf '%s_%s' g1_deploy onnx_ref)
    P_EXP=$(printf '%s_%s' run_data exporter); P_SRV=$(printf '%s_%s' serve policy)
    for pat in "$P_SIM" "$P_DEP" "$P_EXP" "$P_SRV"; do
        for pid in $(pgrep -f "$pat" || true); do kill -9 "$pid" 2>/dev/null || true; done
    done
    sleep 3
}

serve() {  # policy server on :5590, GPU 1
    local log="$LOCOMANI_TMP/serve_$RUN.log"
    for pid in $(pgrep -f "$(printf '%s_%s' serve policy)" || true); do
        kill -9 "$pid" 2>/dev/null || true; done
    sleep 2
    CUDA_VISIBLE_DEVICES=1 nohup "$GR00T_PY" "$LOCOMANI_SRC/tools/serve_policy.py" \
        --model-path "$1" --port 5590 > "$log" 2>&1 &
    for _ in $(seq 1 60); do
        grep -q "listening" "$log" 2>/dev/null && return 0; sleep 5
    done
    echo "policy server never came up"; tail -5 "$log"; return 1
}

say "closed-loop $RUN: $(basename "$CKPT"), fps $DATASET_FPS, seed $EVAL_SEED"
kill_stack
sleep 2
NO_ENCODER=1 SIM_VIDEO_DIR="$LOCOMANI_OUTPUTS/sim_video" \
    bash "$LOCOMANI_SRC/start_stack.sh" "pick up the water bottle" \
    > "$LOCOMANI_OUTPUTS/logs/stack_cl_$RUN.log" 2>&1
grep -qi "stack up" "$LOCOMANI_OUTPUTS/logs/stack_cl_$RUN.log" \
    || { echo "stack did not come up"; tail -5 "$LOCOMANI_OUTPUTS/logs/stack_cl_$RUN.log"; exit 1; }
serve "$CKPT" || exit 1

CL_OUT="$EVAL_DIR/closed_loop.json"
for N in $ROUNDS; do
    RESUME=""; [ -f "${CL_OUT%.json}.manifest.jsonl" ] && RESUME="--resume"
    say "--fresh $N  (live: $EVAL_DIR/eval_fresh${N}.log)"
    # Redirect to a file rather than piping to tail: a pipe to `tail` shows
    # NOTHING until the command exits, which for a 2.5 h eval means no way to
    # see progress. The per-round manifest is written live either way, but the
    # log is what says WHY a round failed.
    # shellcheck disable=SC2086
    ( cd "$LOCOMANI_SRC" && TF_DATASET_FPS="$DATASET_FPS" "$VENV_SIM_PY" -u \
        tools/eval_closed_loop.py --fresh "$N" --seed "$EVAL_SEED" \
        --client async --execution-horizon 16 $RESUME \
        --out "$CL_OUT" --trace-dir "$EVAL_DIR/traces" ) \
        > "$EVAL_DIR/eval_fresh${N}.log" 2>&1
    tail -25 "$EVAL_DIR/eval_fresh${N}.log"
    [ -f "$CL_OUT" ] || { note "- $N 轮没有产出 JSON"; continue; }
    cp "$CL_OUT" "$EVAL_DIR/closed_loop_at_${N}.json"
    "$VENV_SIM_PY" - "$CL_OUT" "$REPORT" "$(basename "$CKPT")" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
w = d.get("wilson95") or [float("nan")] * 2
with open(sys.argv[2], "a") as out:
    out.write(f"- **{d['successes']}/{d['rounds']} = {d['success_rate']*100:.1f}%**"
              f"  Wilson95 [{w[0]*100:.1f}, {w[1]*100:.1f}]"
              f"  ({sys.argv[3]}) — round 3 (fast200) 是 28.0%\n")
    fs = d.get("failure_stages") or {}
    if fs:
        out.write("  - 失败分布: " + ", ".join(f"{k} {v}" for k, v in
                  sorted(fs.items(), key=lambda x: -x[1])) + "\n")
PY
done
kill_stack
say "done -- $CL_OUT"
