#!/bin/bash
# End-to-end: wait for the conversion, audit, train, then both evaluations.
#
#   nohup bash scripts/run_pipeline.sh > outputs/logs/pipeline.log 2>&1 &
#
# Stages are numbered and each one fails the whole run rather than carrying a
# broken artefact forward. The expensive stage (train) is in the middle, so
# everything cheap that can invalidate it -- the schema audit especially --
# happens before, not after.

set -euo pipefail

# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT"

TRAIN_DS="${TRAIN_DS:-$LOCOMANI_ROOT/datasets/g1_locomani_bottle200_train}"
VAL_DS="${VAL_DS:-$LOCOMANI_ROOT/datasets/g1_locomani_bottle200_val}"
RUN="${RUN:-bottle200}"
OUT_ROOT="${OUT_ROOT:-$LOCOMANI_OUTPUTS/train}"
CKPT_DIR="$OUT_ROOT/$RUN"
EVAL_DIR="${EVAL_DIR:-$LOCOMANI_OUTPUTS/eval/$RUN}"
CONVERT_LOG="${CONVERT_LOG:-$LOCOMANI_TMP/convert200.log}"
GR00T_PY="$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python"
DATA_PY="$VENV_DATA_PY"
MODALITY="$LOCOMANI_ROOT/configs/gr00t_sonic_2cam.py"
# Chosen in advance from the open-loop horizon sweep; overridden below by
# whatever that sweep actually says.
CLOSED_HORIZON="${CLOSED_HORIZON:-16}"

mkdir -p "$EVAL_DIR"
say() { echo; echo "=========== $* ($(date '+%F %T')) ==========="; }

# --- 1. wait for the conversion ----------------------------------------------
# Two "done:" lines, one per sink (train and holdout). Waiting on the line
# rather than on the process means a crashed converter is caught here instead
# of being mistaken for a finished one.
say "1/6 waiting for conversion"
if [ ! -f "$TRAIN_DS/meta/info.json" ] || [ ! -f "$VAL_DS/meta/info.json" ]; then
    # `grep -c` prints 0 AND exits 1 when it matches nothing, so `|| echo 0`
    # emits a SECOND zero and `[` then chokes on "0\n0" -- which under `set -e`
    # aborts the wait and walks straight into a half-written dataset. `|| true`
    # keeps the printed 0 and drops only the exit status.
    while [ "$(grep -c '^done:' "$CONVERT_LOG" 2>/dev/null || true)" -lt 2 ]; do
        if grep -qE 'Traceback|\[manifest\]' "$CONVERT_LOG" 2>/dev/null; then
            echo "conversion failed:"; tail -20 "$CONVERT_LOG"; exit 1
        fi
        sleep 30
    done
fi
grep '^done:' "$CONVERT_LOG" || true
echo "train $(ls "$TRAIN_DS/data/chunk-000" | wc -l) episodes, val $(ls "$VAL_DS/data/chunk-000" | wc -l)"

# --- 2. normalization statistics ---------------------------------------------
# Per dataset. The TRAIN set's stats are the ones that matter: training freezes
# them into the checkpoint's processor (gr00t_policy.py:124), so evaluation
# normalizes with train statistics no matter what the val directory contains.
# The val set still needs a stats.json of its own only because the loader
# asserts one exists.
say "2/6 statistics"
for ds in "$TRAIN_DS" "$VAL_DS"; do
    if [ -f "$ds/meta/stats.json" ]; then echo "  have $ds/meta/stats.json"; continue; fi
    "$GR00T_PY" -m gr00t.data.stats --dataset-path "$ds" \
        --embodiment-tag UNITREE_G1_SONIC --modality-config-path "$MODALITY" \
        | tail -2
done

# --- 3. audit -----------------------------------------------------------------
# Before the GPU hours, not after. check_dataset exits non-zero on any failure.
say "3/6 dataset audit"
# --allow-missing-object-pose is passed DELIBERATELY, not defensively. The
# first bottle collection lost the privileged object-pose channel for 9 of its
# 10 sessions (restart_exporter.sh did not carry TF_POSE_OBJECTS, since fixed),
# and the columns were stripped so every parquet matches info.json. Grasp
# causality is therefore unverified by this audit on that dataset; the
# surviving evidence is the gym's own per-episode predicate. Drop this flag as
# soon as a re-collected set has the channel back.
for ds in "$TRAIN_DS" "$VAL_DS"; do
    "$DATA_PY" "$LOCOMANI_SRC/tools/check_dataset.py" "$ds" \
        ${ALLOW_NO_POSE:+--allow-missing-object-pose}
done

# --- 4. train -----------------------------------------------------------------
say "4/6 training"
RUN="$RUN" OUT_ROOT="$OUT_ROOT" DATA="$TRAIN_DS" bash "$LOCOMANI_ROOT/scripts/train_bottle.sh"

# --- 5. open-loop evaluation, every checkpoint --------------------------------
# Every one, not just the last: GR00T asserts eval_strategy == "no", so there is
# no in-training validation and the stopping point can only be chosen after the
# fact. Scoring only the final checkpoint would throw that choice away.
say "5/6 open-loop evaluation"
CKPTS=$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n || true)
[ -n "$CKPTS" ] || { echo "no checkpoints in $CKPT_DIR"; exit 1; }
for c in $CKPTS; do
    n=$(basename "$c")
    echo "--- $n"
    "$GR00T_PY" "$LOCOMANI_SRC/tools/eval_open_loop.py" \
        --model-path "$c" --dataset "$VAL_DS" \
        --modality-config "$MODALITY" \
        --out "$EVAL_DIR/open_loop_$n.json" \
        --wandb-run "$RUN" || echo "  (failed on $n, continuing)"
done

# Pick by held-out token nMSE at the horizon closed-loop will actually run.
# Ties and near-ties go to the EARLIER checkpoint: with 1.62B trainable
# parameters on 180 episodes the later one is the more overfit of the two.
BEST=$("$GR00T_PY" - "$EVAL_DIR" "$CLOSED_HORIZON" <<'PY'
import json, re, sys
from pathlib import Path
d, h = Path(sys.argv[1]), sys.argv[2]
best, rows = None, []
for f in sorted(d.glob("open_loop_checkpoint-*.json")):
    step = int(re.search(r"checkpoint-(\d+)", f.name).group(1))
    try:
        m = json.loads(f.read_text())["horizons"][h]["model"]["nmse_token"]
    except Exception:
        continue
    rows.append((step, m, f))
rows.sort()
for step, m, f in rows:
    print(f"  step {step:6d}  nmse_token@h{h} {m:.4f}", file=sys.stderr)
if rows:
    best = min(rows, key=lambda r: (round(r[1], 4), r[0]))
    print(str(best[2]).replace("open_loop_", "").replace(".json", "")
          .replace(str(d) + "/", ""))
PY
) || true
BEST_CKPT="$CKPT_DIR/${BEST:-$(basename "$(echo "$CKPTS" | tail -1)")}"
echo "best checkpoint: $BEST_CKPT"
echo "$BEST_CKPT" > "$EVAL_DIR/best_checkpoint.txt"

# --- 6. closed-loop evaluation ------------------------------------------------
# The one that actually answers the question. Open-loop token accuracy is only
# loosely coupled to task success here: replaying the GROUND TRUTH token open
# loop still drifts 0.77 m (diag_token_replay.py), so a good nMSE is necessary
# and nowhere near sufficient.
say "6/6 closed-loop evaluation"
CL="$LOCOMANI_SRC/tools/eval_closed_loop.py"
if [ ! -f "$CL" ]; then
    echo "$CL does not exist yet -- stopping here."
    echo "Checkpoints and open-loop results are complete; run stage 6 with:"
    echo "  $GR00T_PY $CL --model-path $BEST_CKPT --matched --fresh 100"
    exit 0
fi
"$GR00T_PY" "$CL" \
    --model-path "$BEST_CKPT" \
    --execution-horizon "$CLOSED_HORIZON" \
    --matched-manifest-dir "$LOCOMANI_OUTPUTS/logs/bottle200" \
    --val-dataset "$VAL_DS" \
    --fresh "${FRESH:-100}" \
    --out "$EVAL_DIR/closed_loop.json" \
    --wandb-run "$RUN"

say "pipeline complete"
echo "checkpoints  $CKPT_DIR"
echo "eval         $EVAL_DIR"
