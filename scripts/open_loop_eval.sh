#!/bin/bash
# Run NVIDIA's official open-loop eval (gr00t/eval/open_loop_eval.py) over
# EVERY episode of a dataset for one checkpoint: saves every trajectory's
# plot, the per-trajectory MSE/MAE, and the averaged MSE/MAE.
#
#   bash scripts/open_loop_eval.sh \
#       --model-path PATH --dataset-path PATH [--out-dir PATH] \
#       [--horizon N] [--embodiment-tag TAG] [--backbone-path PATH]
#
# --backbone-path: the local Cosmos-Reason2-2B directory. A checkpoint records
# the backbone path its training machine used (config.json model_name); one
# trained on the cluster points at /group-volume/..., which does not exist
# here, and open_loop_eval.py refuses to load it until told where the local
# copy is. Also settable as BACKBONE_PATH=... in the environment. Not needed
# for checkpoints trained on this machine.
#
# Named flags, not positional args: --model-path and --dataset-path are two
# unrelated absolute paths of the same shape (a directory), and swapping two
# positional args of the same shape is a mistake that is invisible until the
# error message from three layers down. A --flag typo instead fails
# immediately and says which one is wrong (see the swap check below).
#
# WHAT IT REUSES FROM scripts/crosscheck_open_loop.sh
#
# Same two derivations, for the same reasons:
#   - TRAJ_IDS comes from meta/info.json's total_episodes, not a fixed count,
#     so "every trajectory" tracks whatever dataset is pointed at instead of
#     going stale.
#   - STEPS comes from the longest episode in meta/episodes.jsonl. The
#     official script's --steps defaults to 200 and silently truncates
#     (open_loop_eval.py:149); on an episode where the grasp happens well past
#     frame 200, a fixed default would score only the walk and never see it.
#
# WHERE THE PLOTS GO
#
# --save-plot-dir $OUT_DIR/plots, so every trajectory's plot lands in the
# output directory as it is produced. (Not --save-plot-path: that is one
# literal file, and main() passes the same one for every traj_id, so each
# trajectory would overwrite the last.) Earlier this script let the Python
# default write to /tmp/open_loop_eval and moved the files afterward; a run
# that died between the eval and the move left 76 plots in /tmp, where the
# next run's cleanup deleted them (2026-09-16). Nothing is staged in /tmp now.
set -u
set -o pipefail   # so the `python | tee` pipeline below still reports python's exit code
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

usage() {
    cat >&2 <<EOF
usage: $(basename "$0") --model-path PATH --dataset-path PATH [--out-dir PATH]
                         [--horizon N] [--embodiment-tag TAG] [--backbone-path PATH]
EOF
}

MODEL_PATH=""
DATASET_PATH=""
OUT_DIR=""
EMBODIMENT_TAG="${EMBODIMENT_TAG:-UNITREE_G1_SONIC}"
HORIZON="${HORIZON:-16}"
BACKBONE_PATH="${BACKBONE_PATH:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --model-path)     MODEL_PATH="${2:?--model-path needs a value}"; shift 2 ;;
        --dataset-path)   DATASET_PATH="${2:?--dataset-path needs a value}"; shift 2 ;;
        --out-dir)        OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --horizon)        HORIZON="${2:?--horizon needs a value}"; shift 2 ;;
        --embodiment-tag) EMBODIMENT_TAG="${2:?--embodiment-tag needs a value}"; shift 2 ;;
        --backbone-path)  BACKBONE_PATH="${2:?--backbone-path needs a value}"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done
[ -z "$MODEL_PATH" ]   && { echo "missing --model-path"   >&2; usage; exit 1; }
[ -z "$DATASET_PATH" ] && { echo "missing --dataset-path" >&2; usage; exit 1; }
OUT_DIR="${OUT_DIR:-$LOCOMANI_OUTPUTS/eval/$(basename "$MODEL_PATH")}"

# Catch the classic swap (--model-path and --dataset-path pointing at each
# other) before spending minutes loading the model: a LeRobot dataset has
# meta/info.json, a checkpoint directory does not.
[ -f "$DATASET_PATH/meta/info.json" ] || {
    echo "error: --dataset-path '$DATASET_PATH' has no meta/info.json" >&2
    echo "  (looks like it's not a LeRobot dataset -- swapped with --model-path?)" >&2
    exit 1
}
[ -f "$MODEL_PATH/meta/info.json" ] && {
    echo "error: --model-path '$MODEL_PATH' has a meta/info.json (looks like a dataset)" >&2
    echo "  (--model-path and --dataset-path look swapped)" >&2
    exit 1
}

BACKBONE_ARGS=()
if [ -n "$BACKBONE_PATH" ]; then
    [ -f "$BACKBONE_PATH/config.json" ] || {
        echo "error: --backbone-path '$BACKBONE_PATH' has no config.json (not a Cosmos-Reason2-2B directory)" >&2
        exit 1
    }
    BACKBONE_ARGS=(--backbone-path "$BACKBONE_PATH")
fi

GR00T_PY="${GR00T_PY:-$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python}"
LOG="$OUT_DIR/run.log"
PLOTS="$OUT_DIR/plots"

mkdir -p "$PLOTS"

# Every episode, derived not hardcoded -- see header.
n_val=$("$GR00T_PY" -c "
import json
print(json.load(open('$DATASET_PATH/meta/info.json'))['total_episodes'])" 2>/dev/null) || n_val=""
if [ -z "$n_val" ]; then echo "cannot read $DATASET_PATH/meta/info.json"; exit 1; fi
TRAJ_IDS=$(seq 0 $((n_val - 1)) | tr '\n' ' ')

# Longest episode, so --steps never silently truncates. See header.
STEPS=$("$GR00T_PY" -c "
import json
print(max(json.loads(l)['length'] for l in open('$DATASET_PATH/meta/episodes.jsonl')))" 2>/dev/null) || STEPS=""
[ -z "$STEPS" ] && { echo "cannot read episode lengths from $DATASET_PATH"; exit 1; }

echo "open-loop eval: $MODEL_PATH"
echo "  dataset    $DATASET_PATH  ($n_val episodes)"
echo "  horizon    $HORIZON   steps $STEPS"
[ -n "$BACKBONE_PATH" ] && echo "  backbone   $BACKBONE_PATH"
echo "  out        $OUT_DIR"
echo

# shellcheck disable=SC2086
"$GR00T_PY" "$LOCOMANI_ROOT/src/gr00t/eval/open_loop_eval.py" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --model-path "$MODEL_PATH" \
    --traj-ids $TRAJ_IDS \
    --steps "$STEPS" \
    --execution-horizon "$HORIZON" \
    --save-plot-dir "$PLOTS" \
    "${BACKBONE_ARGS[@]}" \
    2>&1 | tee "$LOG"
status=$?
if [ "$status" != 0 ]; then
    echo "open_loop_eval.py exited $status -- see $LOG"; exit "$status"
fi

n_plots=0
for plot in "$PLOTS"/traj_*.jpeg; do
    [ -f "$plot" ] && n_plots=$((n_plots + 1))
done
echo
echo "saved $n_plots plots to $PLOTS"

"$GR00T_PY" - "$LOG" "$OUT_DIR/metrics.json" <<'PYEOF'
import json, re, sys
from pathlib import Path

log, out = sys.argv[1], sys.argv[2]
text = Path(log).read_text()

rows = []
for m in re.finditer(
    r"MSE for trajectory (\d+): ([0-9.eE+-]+), MAE: ([0-9.eE+-]+)", text
):
    rows.append({"traj_id": int(m.group(1)), "mse": float(m.group(2)), "mae": float(m.group(3))})
rows.sort(key=lambda r: r["traj_id"])

avg_mse = re.search(r"Average MSE across all trajs: ([0-9.eE+-]+)", text)
avg_mae = re.search(r"Average MAE across all trajs: ([0-9.eE+-]+)", text)
if not (rows and avg_mse and avg_mae):
    print(f"could not parse per-trajectory / average metrics out of {log}", file=sys.stderr)
    sys.exit(1)

result = {
    "n_traj": len(rows),
    "per_trajectory": rows,
    "average": {"mse": float(avg_mse.group(1)), "mae": float(avg_mae.group(1))},
}
Path(out).write_text(json.dumps(result, indent=2))

print(f"{'traj_id':>8} {'mse':>14} {'mae':>14}")
for r in rows:
    print(f"{r['traj_id']:>8} {r['mse']:>14.6f} {r['mae']:>14.6f}")
print()
print(f"{'average':>8} {result['average']['mse']:>14.6f} {result['average']['mae']:>14.6f}")
print()
print(f"written {out}")
PYEOF
