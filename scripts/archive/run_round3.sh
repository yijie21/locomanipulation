#!/bin/bash
#
# ARCHIVED -- HISTORICAL RECORD, NOT MAINTAINED, KNOWN NOT TO RUN.
#
# This is round 3 (fast200, zero DAgger, 28/100 = 28.0%). It is kept because it is the
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
# Round 3, unattended end to end: wait out the fast-reach collection, convert
# (single-era schema -- the pose columns are REAL again, so the grasp-causality
# audit is back in force), train, open-loop sweep, closed-loop 100.
#
#   nohup bash scripts/run_round3.sh > outputs/logs/round3.log 2>&1 &
#
# Self-healing covers every failure this pipeline has actually had:
# disk preflight (round 2 died of ENOSPC mid-eval), the orphaned :5555 image
# publisher after a stack kill, the encoder-loaded deploy that eats token
# streams, wandb credential hangs (train_bottle preflights), and per-stage
# resumability (rerun the script and finished stages are skipped).

set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT"

RUN=fast200
TRAIN_DS="$LOCOMANI_ROOT/datasets/g1_locomani_${RUN}_train"
VAL_DS="$LOCOMANI_ROOT/datasets/g1_locomani_${RUN}_val"
EVAL_DIR="$LOCOMANI_OUTPUTS/eval/$RUN"
CKPT_DIR="$LOCOMANI_OUTPUTS/train/$RUN"
GR00T_PY="$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python"
MODALITY="$LOCOMANI_ROOT/configs/gr00t_sonic_2cam.py"
COLLECT_LOG="$LOCOMANI_TMP/collect_fast200.log"
export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-$LOCOMANI_ROOT/third_party/GR00T-WholeBodyControl/gear_sonic_deploy}"
mkdir -p "$EVAL_DIR"
say() { echo; echo "=========== $* ($(date '+%F %T')) ==========="; }

# Patterns assembled at run time so they can never match this script's own
# command line (a literal pattern here kills the invoking shell: exit 144).
kill_stack() {
    local P_SIM P_DEP P_EXP P_SRV pat pid
    P_SIM=$(printf '%s_%s' run_sim locomani); P_DEP=$(printf '%s_%s' g1_deploy onnx_ref)
    P_EXP=$(printf '%s_%s' run_data exporter); P_SRV=$(printf '%s_%s' serve policy)
    for pat in "$P_SIM" "$P_DEP" "$P_EXP" "$P_SRV"; do
        for pid in $(pgrep -f "$pat" || true); do kill -9 "$pid" 2>/dev/null || true; done
    done
}

# --- 0. disk preflight --------------------------------------------------------
# Training writes ~65 GB of checkpoints; round 2's collection, eval AND a
# second training all died of ENOSPC at once. Demand 95 GB up front, and make
# room from the known reclaimables (superseded frozen-dit checkpoints).
say "0/6 disk"
free_gb() { df --output=avail -BG /workspace | tail -1 | tr -dc 0-9; }
if [ "$(free_gb)" -lt 95 ]; then
    rm -rf "$LOCOMANI_OUTPUTS/train/bottle250_frozen_dit/checkpoint-5000" \
           "$LOCOMANI_OUTPUTS/train/bottle250_frozen_dit/checkpoint-6000"
    echo "  reclaimed superseded frozen-dit checkpoints"
fi
echo "  free: $(free_gb) GB"
[ "$(free_gb)" -ge 90 ] || { echo "still under 90 GB -- refusing to start a 65 GB training"; exit 1; }

# --- 1. wait for the collection ----------------------------------------------
say "1/6 collection"
PAT=$(printf '%s_%s' collect pilot)
if [ ! -f "$VAL_DS/meta/info.json" ]; then
    while ! grep -q "convert with" "$COLLECT_LOG" 2>/dev/null; do
        pgrep -f "$PAT" >/dev/null || { echo "collection process gone without finishing:"; tail -10 "$COLLECT_LOG"; exit 1; }
        sleep 300
    done
fi
echo "  kept: $(cat "$LOCOMANI_OUTPUTS"/logs/fastreach200/manifest_*.jsonl 2>/dev/null | grep -c '"ok": true')/200"

# --- 2. convert ---------------------------------------------------------------
# No --strip-pose-columns: every session is post-restart-fix, the privileged
# pose column is real, and the audit's grasp-causality pass runs again.
say "2/6 convert"
if [ ! -f "$VAL_DS/meta/info.json" ]; then
    SESS=$(tr '\n' ' ' < "$LOCOMANI_OUTPUTS/logs/fastreach200/sessions.txt")
    MAN=$(ls -1 "$LOCOMANI_OUTPUTS"/logs/fastreach200/manifest_*.jsonl | sort | tr '\n' ' ')
    # shellcheck disable=SC2086
    "$VENV_DATA_PY" "$LOCOMANI_SRC/tools/convert_to_hf_format.py" \
        --sessions $SESS --manifest $MAN \
        --holdout 20 --holdout-seed 0 \
        --out "$TRAIN_DS" --holdout-out "$VAL_DS"
else
    echo "  datasets exist, skipping"
fi

# --- 3. stats + audit ---------------------------------------------------------
say "3/6 stats + audit"
for ds in "$TRAIN_DS" "$VAL_DS"; do
    [ -f "$ds/meta/stats.json" ] || \
    "$GR00T_PY" -m gr00t.data.stats --dataset-path "$ds" \
        --embodiment-tag UNITREE_G1_SONIC --modality-config-path "$MODALITY" | tail -1
    "$VENV_DATA_PY" "$LOCOMANI_SRC/tools/check_dataset.py" "$ds"
done

# --- 4. train -----------------------------------------------------------------
say "4/6 train"
# Tear the collection stack down FIRST. Everything that needed it (collection,
# conversion) is done, and stage 6 brings a fresh one up anyway -- but while it
# is alive the sim + TensorRT engines hold ~6.5 GB on GPU 0, and training
# defaults to GPUS=2, so rank 0 lands on that card and dies:
#   torch.OutOfMemoryError: Tried to allocate 3.02 GiB.
#   GPU 0 has ... 66.38 MiB is free
# (round 3, 2026-08-26 05:43 -- cost a full training run.)
kill_stack
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'

if [ ! -d "$CKPT_DIR/checkpoint-10000" ]; then
    if ! RUN="$RUN" DATA="$TRAIN_DS" bash "$LOCOMANI_ROOT/scripts/train_bottle.sh"; then
        echo "TRAINING FAILED -- stopping."
        echo "Everything downstream evaluates a model that does not exist:"
        echo "the open-loop sweep would find no checkpoints and the closed-loop"
        echo "run would report 0/100 that means nothing. Fix and re-run; stages"
        echo "0-3 are already done and will be skipped."
        exit 1
    fi
fi
if [ -z "$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null)" ]; then
    echo "no checkpoints in $CKPT_DIR -- refusing to evaluate nothing."; exit 1
fi

# --- 5. open-loop -------------------------------------------------------------
say "5/6 open-loop"
for c in $(ls -d "$CKPT_DIR"/checkpoint-* | sort -t- -k2 -n); do
    n=$(basename "$c")
    [ -f "$EVAL_DIR/open_loop_$n.json" ] && continue
    echo "--- $n"
    "$GR00T_PY" "$LOCOMANI_SRC/tools/eval_open_loop.py" \
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
    print(f"  step {s:6d}  nmse@h16 {m:.4f}", file=sys.stderr)
print(f"checkpoint-{min(rows, key=lambda r: (round(r[1], 4), r[0]))[0]}")
PYEOF
)
echo "BEST: $BEST"; echo "$CKPT_DIR/$BEST" > "$EVAL_DIR/best_checkpoint.txt"
[ -f "$EVAL_DIR/open_loop_best_full.json" ] || \
"$GR00T_PY" "$LOCOMANI_SRC/tools/eval_open_loop.py" \
    --model-path "$CKPT_DIR/$BEST" --dataset "$VAL_DS" --horizons 1 8 16 40 \
    --out "$EVAL_DIR/open_loop_best_full.json" --wandb-run "$RUN" \
    || echo "  (full-horizon pass failed)"

# --- 6. closed-loop -----------------------------------------------------------
say "6/6 closed-loop"
kill_stack
sleep 3
for pid in $(pgrep -f "venv_sim/bin/python -c from multiprocessing" || true); do
    [ "$(ps -o ppid= -p "$pid" | tr -d ' ')" = "1" ] && kill -9 "$pid" 2>/dev/null || true
done
sleep 1
NO_ENCODER=1 SCENE_RANDOM=0 SIM_VIDEO_DIR="$LOCOMANI_OUTPUTS/sim_video" \
    bash "$LOCOMANI_SRC/start_stack.sh" "pick up the water bottle"
CUDA_VISIBLE_DEVICES=1 nohup "$GR00T_PY" "$LOCOMANI_SRC/tools/serve_policy.py" \
    --model-path "$CKPT_DIR/$BEST" --port 5590 \
    > "$LOCOMANI_TMP/serve_policy_$RUN.log" 2>&1 &
for _ in $(seq 1 60); do
    grep -q "listening" "$LOCOMANI_TMP/serve_policy_$RUN.log" 2>/dev/null && break
    sleep 5
done
grep -q "listening" "$LOCOMANI_TMP/serve_policy_$RUN.log" || { echo "server never up"; exit 1; }
cd "$LOCOMANI_SRC"
"$LOCOMANI_ROOT/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python" \
    tools/eval_closed_loop.py --fresh 100 --seed 6000 --client async \
    --out "$EVAL_DIR/closed_loop.json" \
    --trace-dir "$EVAL_DIR/closed_loop_traces"

say "round 3 complete"
cat "$EVAL_DIR/closed_loop.json"
