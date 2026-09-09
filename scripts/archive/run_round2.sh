#!/bin/bash
#
# ARCHIVED -- HISTORICAL RECORD, NOT MAINTAINED, KNOWN NOT TO RUN.
#
# This is round 2 (bottle250 = 200 + 50 DAgger, 14/102 = 13.7%). It is kept because it is the
# only place the exact configuration behind that number is written down, and
# docs/evaluation.md's ladder is only reproducible if it survives. It is NOT
# kept as something to run: it calls tools whose interfaces have since moved
# (eval_closed_loop.py no longer has --model-path, --val-dataset,
# --matched-manifest-dir or --wandb-run, so any stage that reaches the
# closed-loop call dies there), and it carries its own copy of kill_stack that
# predates stop_stack.sh becoming the single implementation.
#
# For a new experiment use scripts/run_overnight.sh (the full chain) or
# scripts/run_closed_loop.sh (evaluation only).
#
# Round 2: retrain with the DAgger recovery episodes, evaluate both ways.
#
#   nohup bash scripts/run_round2.sh > outputs/logs/round2.log 2>&1 &
#
# The chain is the round-1 pipeline with the lessons baked in:
#   convert(250, mixed-era)  ->  stats  ->  audit  ->  train(bottle250)
#   ->  open-loop h16 sweep + full horizons on best
#   ->  stack flip to NO_ENCODER + policy server + closed-loop 100
#
# Every stage's mechanism was exercised in round 1; what this script adds is
# only the sequencing, so a crash leaves resumable state (each stage checks
# for its own outputs before redoing work).

set -euo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT"

RUN=bottle250
TRAIN_DS="$LOCOMANI_ROOT/datasets/g1_locomani_${RUN}_train"
VAL_DS="$LOCOMANI_ROOT/datasets/g1_locomani_${RUN}_val"
EVAL_DIR="$LOCOMANI_OUTPUTS/eval/$RUN"
CKPT_DIR="$LOCOMANI_OUTPUTS/train/$RUN"
GR00T_PY="$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python"
MODALITY="$LOCOMANI_ROOT/configs/gr00t_sonic_2cam.py"
export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-$LOCOMANI_ROOT/third_party/GR00T-WholeBodyControl/gear_sonic_deploy}"
mkdir -p "$EVAL_DIR"
say() { echo; echo "=========== $* ($(date '+%F %T')) ==========="; }

# --- 1. convert: 200 collection + 50 recovery episodes, one dataset ----------
# --strip-pose-columns because the pose column set is era-dependent (the
# TF_POSE_OBJECTS restart bug); everything else must still match exactly.
say "1/6 convert"
if [ ! -f "$VAL_DS/meta/info.json" ]; then
    SESS="$(tr '\n' ' ' < "$LOCOMANI_OUTPUTS/logs/bottle200/sessions.txt") $(tr '\n' ' ' < "$LOCOMANI_OUTPUTS/logs/dagger50/sessions.txt")"
    MAN="$(ls -1 "$LOCOMANI_OUTPUTS"/logs/bottle200/manifest_*.jsonl | sort | tr '\n' ' ') $(ls -1 "$LOCOMANI_OUTPUTS"/logs/dagger50/manifest_*.jsonl | sort | tr '\n' ' ')"
    # shellcheck disable=SC2086
    "$VENV_DATA_PY" "$LOCOMANI_SRC/pipeline/convert_to_hf_format.py" \
        --sessions $SESS --manifest $MAN \
        --strip-pose-columns --holdout 20 --holdout-seed 0 \
        --out "$TRAIN_DS" --holdout-out "$VAL_DS"
else
    echo "  datasets exist, skipping"
fi
echo "train $(ls "$TRAIN_DS/data/chunk-000" | wc -l) episodes, val $(ls "$VAL_DS/data/chunk-000" | wc -l)"

# --- 2. stats ----------------------------------------------------------------
say "2/6 statistics"
for ds in "$TRAIN_DS" "$VAL_DS"; do
    [ -f "$ds/meta/stats.json" ] && { echo "  have $ds"; continue; }
    "$GR00T_PY" -m gr00t.data.stats --dataset-path "$ds" \
        --embodiment-tag UNITREE_G1_SONIC --modality-config-path "$MODALITY" | tail -1
done

# --- 3. audit ----------------------------------------------------------------
# --allow-missing-object-pose: the pose columns were STRIPPED at conversion, so
# grasp causality is vouched for by the gym's own per-episode predicate.
say "3/6 audit"
for ds in "$TRAIN_DS" "$VAL_DS"; do
    "$VENV_DATA_PY" "$LOCOMANI_SRC/pipeline/check_dataset.py" "$ds" \
        --allow-missing-object-pose
done

# --- 4. train ----------------------------------------------------------------
say "4/6 train"
if [ ! -d "$CKPT_DIR/checkpoint-10000" ]; then
    RUN="$RUN" DATA="$TRAIN_DS" bash "$LOCOMANI_ROOT/scripts/train_bottle.sh"
else
    echo "  checkpoint-10000 exists, skipping"
fi

# --- 5. open-loop: h16 sweep -> best -> full horizons ------------------------
say "5/6 open-loop"
for c in $(ls -d "$CKPT_DIR"/checkpoint-* | sort -t- -k2 -n); do
    n=$(basename "$c")
    [ -f "$EVAL_DIR/open_loop_$n.json" ] && continue
    echo "--- $n"
    "$GR00T_PY" "$LOCOMANI_SRC/pipeline/eval_open_loop.py" \
        --model-path "$c" --dataset "$VAL_DS" --horizons 16 \
        --out "$EVAL_DIR/open_loop_$n.json" --wandb-run "$RUN" \
        || echo "  (failed on $n)"
done
BEST=$("$GR00T_PY" - "$EVAL_DIR" <<'PYEOF'
import json, re, sys
from pathlib import Path
rows = []
for f in sorted(Path(sys.argv[1]).glob("open_loop_checkpoint-*.json")):
    step = int(re.search(r"checkpoint-(\d+)", f.name).group(1))
    try:
        rows.append((step, json.loads(f.read_text())["horizons"]["16"]["model"]["nmse_token"]))
    except Exception:
        pass
rows.sort()
for s, m in rows:
    print(f"  step {s:6d}  nmse_token@h16 {m:.4f}", file=sys.stderr)
best = min(rows, key=lambda r: (round(r[1], 4), r[0]))
print(f"checkpoint-{best[0]}")
PYEOF
)
echo "BEST: $BEST"; echo "$CKPT_DIR/$BEST" > "$EVAL_DIR/best_checkpoint.txt"
[ -f "$EVAL_DIR/open_loop_best_full.json" ] || \
"$GR00T_PY" "$LOCOMANI_SRC/pipeline/eval_open_loop.py" \
    --model-path "$CKPT_DIR/$BEST" --dataset "$VAL_DS" --horizons 1 8 16 40 \
    --out "$EVAL_DIR/open_loop_best_full.json" --wandb-run "$RUN" \
    || echo "  (full-horizon pass failed)"

# --- 6. closed-loop ----------------------------------------------------------
say "6/6 closed-loop"
# 6a. stack -> token-only. Kill by PID, then reap the orphaned image-publish
# worker a kill -9'd sim leaves holding :5555 (it reparents to init).
for pat in "run_sim_""locomani" "g1_deploy_""onnx_ref" "run_data_""exporter"; do
    for pid in $(pgrep -f "$pat"); do kill -9 "$pid" 2>/dev/null || true; done
done
sleep 3
for pid in $(pgrep -f "venv_sim/bin/python -c from multiprocessing" || true); do
    [ "$(ps -o ppid= -p "$pid" | tr -d ' ')" = "1" ] && kill -9 "$pid" 2>/dev/null || true
done
sleep 1
NO_ENCODER=1 SCENE_RANDOM=0 SIM_VIDEO_DIR="$LOCOMANI_OUTPUTS/sim_video" \
    bash "$LOCOMANI_STACK/start_stack.sh" "pick up the water bottle"
# 6b. policy server on the second GPU
for pid in $(pgrep -f "serve_""policy" || true); do kill "$pid" 2>/dev/null || true; done
CUDA_VISIBLE_DEVICES=1 nohup "$GR00T_PY" "$LOCOMANI_SRC/pipeline/serve_policy.py" \
    --model-path "$CKPT_DIR/$BEST" --port 5590 \
    > "$LOCOMANI_TMP/serve_policy_$RUN.log" 2>&1 &
for _ in $(seq 1 60); do
    grep -q "listening" "$LOCOMANI_TMP/serve_policy_$RUN.log" 2>/dev/null && break
    sleep 5
done
grep -q "listening" "$LOCOMANI_TMP/serve_policy_$RUN.log" || { echo "policy server never came up"; exit 1; }
# 6c. the run itself (venv_sim; exports NO_ENCODER for the per-round restarts)
cd "$LOCOMANI_SRC/pipeline"
"$LOCOMANI_ROOT/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python" \
    eval_closed_loop.py --fresh 100 --seed 3000 \
    --out "$EVAL_DIR/closed_loop.json" \
    --trace-dir "$EVAL_DIR/closed_loop_traces"

say "round 2 complete"
cat "$EVAL_DIR/closed_loop.json"
