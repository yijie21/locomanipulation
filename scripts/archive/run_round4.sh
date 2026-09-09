#!/bin/bash
#
# ARCHIVED -- HISTORICAL RECORD, NOT MAINTAINED, KNOWN NOT TO RUN.
#
# This is round 4 (fast250 = 200 + 50 targeted DAgger, 48/101 = 47.5%). It is kept because it is the
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
# Round 4, end to end, unattended:
#
#   0. wait for the tier-1 A/B sweep (no-snap, h40) to finish
#   1. failure anatomy over its traces (palm_z / palm_obj decomposition)
#   2. pick the deployment config from the A/B (combo tiebreak if both won),
#      then CONFIRM it: 100 rounds, FULL budget, round-3 model. This point is
#      what separates "config gain" from "data gain" on the ladder.
#   3. flip the stack to collection mode, targeted DAgger 50 episodes
#      (near-table spawns; the pose sidecar records world poses)
#   4. convert round-3 + DAgger together -> fast250 datasets, stats, audit
#   5. train fast250 (hardware-picked optimizer)
#   6. open-loop sweep -> best checkpoint
#   7. closed-loop 100 rounds, full budget, winner config
#   8. summary
#
# Per-stage resumable like run_round3.sh: each stage skips if its artifact
# already exists. Fails loudly; never evaluates a model that does not exist.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

RUN=fast250
T1="$LOCOMANI_OUTPUTS/eval/tier1"
R4="$LOCOMANI_OUTPUTS/eval/$RUN"
CKPT_DIR="$LOCOMANI_OUTPUTS/train/$RUN"
TRAIN_DS="$LOCOMANI_ROOT/datasets/g1_locomani_${RUN}_train"
VAL_DS="$LOCOMANI_ROOT/datasets/g1_locomani_${RUN}_val"
MODALITY="$LOCOMANI_ROOT/configs/gr00t_sonic_2cam.py"
GR00T_PY="$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python"
DAGGER_OUT="$LOCOMANI_OUTPUTS/logs/dagger50_r4"
mkdir -p "$R4"

say() { echo; echo "=========== $1 ($(date '+%F %T')) ==========="; }

kill_stack() {
    local P_SIM P_DEP P_EXP P_SRV pat pid
    P_SIM=$(printf '%s_%s' run_sim locomani); P_DEP=$(printf '%s_%s' g1_deploy onnx_ref)
    P_EXP=$(printf '%s_%s' run_data exporter); P_SRV=$(printf '%s_%s' serve policy)
    for pat in "$P_SIM" "$P_DEP" "$P_EXP" "$P_SRV"; do
        for pid in $(pgrep -f "$pat" || true); do kill -9 "$pid" 2>/dev/null || true; done
    done
    sleep 3
    for pid in $(pgrep -f "venv_sim/bin/python -c from multiprocessing" || true); do
        [ "$(ps -o ppid= -p "$pid" | tr -d ' ')" = "1" ] && kill -9 "$pid" 2>/dev/null || true
    done
}

serve() {  # serve <model_dir> ; leaves server on :5590 GPU1
    local log="$LOCOMANI_TMP/serve_$RUN.log"
    for pid in $(pgrep -f "$(printf '%s_%s' serve policy)" || true); do
        kill -9 "$pid" 2>/dev/null || true; done
    sleep 2
    CUDA_VISIBLE_DEVICES=1 nohup "$GR00T_PY" "$LOCOMANI_SRC/pipeline/serve_policy.py" \
        --model-path "$1" --port 5590 > "$log" 2>&1 &
    for _ in $(seq 1 60); do
        grep -q "listening" "$log" 2>/dev/null && return 0; sleep 5
    done
    echo "policy server never came up"; tail -5 "$log"; return 1
}

# --- 0. wait for tier 1 -------------------------------------------------------
say "0/8 wait for tier-1 A/B"
while ! grep -q "===== done" "$LOCOMANI_TMP/tier1.log" 2>/dev/null; do
    if ! pgrep -f "$(printf 'tier%s.sh' 1)" > /dev/null; then
        grep -q "===== done" "$LOCOMANI_TMP/tier1.log" || {
            echo "tier-1 runner died before finishing"; exit 1; }
    fi
    sleep 120
done
echo "  tier-1 finished"

# --- 1. anatomy ---------------------------------------------------------------
say "1/8 failure anatomy"
"$VENV_SIM_PY" "$LOCOMANI_SRC/tasks/bottle/failure_anatomy.py" \
    "$T1/nosnap_traces" "$T1/h40_traces" --out "$T1/anatomy.json"

# --- 2. config decision + full-budget confirm ---------------------------------
say "2/8 config decision"
FLAGS_FILE="$R4/deploy_flags.txt"
if [ ! -f "$FLAGS_FILE" ]; then
    python3 - "$T1" "$FLAGS_FILE" <<'PYEOF'
import json, sys
t1, out = sys.argv[1], sys.argv[2]
BASE = 13                     # round-3 baseline, first 50 rounds, same seed
ns  = json.load(open(f"{t1}/nosnap.json"))["successes"]
h40 = json.load(open(f"{t1}/h40.json"))["successes"]
win_ns, win_h = ns >= BASE + 3, h40 >= BASE + 3
print(f"  no-snap {ns}/50  h40 {h40}/50  baseline {BASE}/50 (threshold +3)")
if win_ns and win_h:
    flags = "COMBO"           # both won -> paired combo run decides
elif win_ns:
    flags = "--no-snap"
elif win_h:
    flags = "--replan-interval 1.33"
else:
    flags = ""
open(out, "w").write(flags + "\n")
print(f"  -> {flags or '(default config)'}")
PYEOF
fi
FLAGS=$(head -1 "$FLAGS_FILE")

if [ "$FLAGS" = "COMBO" ]; then
    say "2b/8 combo tiebreak (both variants won their A/B)"
    if [ ! -f "$T1/combo.json" ]; then
        serve "$LOCOMANI_OUTPUTS/train/fast200/checkpoint-9000" || exit 1
        (cd "$LOCOMANI_SRC/pipeline" && "$VENV_SIM_PY" eval_closed_loop.py \
            --fresh 50 --seed 6000 --client async --stall-abort 30 \
            --no-snap --replan-interval 1.33 \
            --out "$T1/combo.json" --trace-dir "$T1/combo_traces")
    fi
    python3 - "$T1" "$FLAGS_FILE" <<'PYEOF'
import json, sys
t1, out = sys.argv[1], sys.argv[2]
cand = {
    "--no-snap": json.load(open(f"{t1}/nosnap.json"))["successes"],
    "--replan-interval 1.33": json.load(open(f"{t1}/h40.json"))["successes"],
    "--no-snap --replan-interval 1.33": json.load(open(f"{t1}/combo.json"))["successes"],
}
best = max(cand, key=cand.get)
print("  combo tiebreak:", {k: f"{v}/50" for k, v in cand.items()}, "->", best)
open(out, "w").write(best + "\n")
PYEOF
    FLAGS=$(head -1 "$FLAGS_FILE")
fi
echo "  deployment flags: '${FLAGS}'"

if [ -n "$FLAGS" ] && [ ! -f "$R4/confirm_config.json" ]; then
    say "2c/8 full-budget confirm of '$FLAGS' on the ROUND-3 model"
    serve "$LOCOMANI_OUTPUTS/train/fast200/checkpoint-9000" || exit 1
    # shellcheck disable=SC2086
    (cd "$LOCOMANI_SRC/pipeline" && "$VENV_SIM_PY" eval_closed_loop.py \
        --fresh 100 --seed 6000 --client async $FLAGS \
        --out "$R4/confirm_config.json" \
        --trace-dir "$R4/confirm_traces")
    grep -o '"successes": [0-9]*' "$R4/confirm_config.json" | head -1
fi

# --- 3. DAgger collection -----------------------------------------------------
say "3/8 DAgger collection (50 episodes, near-table)"
kept_dagger() {
    cat "$DAGGER_OUT"/manifest_*.jsonl 2>/dev/null | python3 -c '
import sys, json
print(sum(1 for l in sys.stdin
          if l.strip() and json.loads(l).get("ok") and json.loads(l).get("produced")))'
}
if [ "$(kept_dagger)" -lt 50 ]; then
    kill_stack
    sleep 2
    # Collection stack: encoder ON (token labels), same appearance settings as
    # the round-3 collection (randomization at its default). NO_ENCODER may
    # linger in the environment from an eval stage -- strip it explicitly.
    env -u NO_ENCODER SIM_VIDEO_DIR="$LOCOMANI_OUTPUTS/sim_video" \
        bash "$LOCOMANI_STACK/start_stack.sh" "pick up the water bottle"
    DAGGER_OUT="$DAGGER_OUT" bash "$LOCOMANI_ROOT/scripts/collect_dagger.sh" 50
    echo "  kept: $(kept_dagger)/50"
    [ "$(kept_dagger)" -ge 45 ] || { echo "DAgger collection fell short -- stopping."; exit 1; }
else
    echo "  already collected: $(kept_dagger)/50"
fi

# --- 4. convert ---------------------------------------------------------------
say "4/8 convert -> $RUN"
if [ ! -f "$VAL_DS/meta/info.json" ]; then
    SESS="$(tr '\n' ' ' < "$LOCOMANI_OUTPUTS/logs/fastreach200/sessions.txt") $(tr '\n' ' ' < "$DAGGER_OUT/sessions.txt")"
    MAN="$(ls -1 "$LOCOMANI_OUTPUTS"/logs/fastreach200/manifest_*.jsonl | sort | tr '\n' ' ') $(ls -1 "$DAGGER_OUT"/manifest_*.jsonl | sort | tr '\n' ' ')"
    # shellcheck disable=SC2086
    "$VENV_DATA_PY" "$LOCOMANI_SRC/pipeline/convert_to_hf_format.py" \
        --sessions $SESS --manifest $MAN \
        --holdout 20 --holdout-seed 0 \
        --out "$TRAIN_DS" --holdout-out "$VAL_DS"
else
    echo "  datasets exist, skipping"
fi

# --- 5. stats + audit + train -------------------------------------------------
say "5/8 stats + audit + train"
for ds in "$TRAIN_DS" "$VAL_DS"; do
    [ -f "$ds/meta/stats.json" ] || \
    "$GR00T_PY" -m gr00t.data.stats --dataset-path "$ds" \
        --embodiment-tag UNITREE_G1_SONIC --modality-config-path "$MODALITY" | tail -1
    "$VENV_DATA_PY" "$LOCOMANI_SRC/pipeline/check_dataset.py" "$ds"
done

kill_stack
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
if [ ! -d "$CKPT_DIR/checkpoint-10000" ]; then
    if ! RUN="$RUN" DATA="$TRAIN_DS" bash "$LOCOMANI_ROOT/scripts/train_bottle.sh"; then
        echo "TRAINING FAILED -- stopping."; exit 1
    fi
fi
[ -n "$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null)" ] || {
    echo "no checkpoints -- refusing to evaluate nothing."; exit 1; }

# --- 6. open-loop -------------------------------------------------------------
say "6/8 open-loop"
for c in $(ls -d "$CKPT_DIR"/checkpoint-* | sort -t- -k2 -n); do
    n=$(basename "$c")
    [ -f "$R4/open_loop_$n.json" ] && continue
    echo "--- $n"
    "$GR00T_PY" "$LOCOMANI_SRC/pipeline/eval_open_loop.py" \
        --model-path "$c" --dataset "$VAL_DS" --horizons 16 \
        --out "$R4/open_loop_$n.json" --wandb-run "$RUN" \
        || echo "  (failed on $n)"
done
BEST=$("$GR00T_PY" - "$R4" <<'PYEOF'
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
echo "BEST: $BEST"; echo "$CKPT_DIR/$BEST" > "$R4/best_checkpoint.txt"
[ -f "$R4/open_loop_best_full.json" ] || \
"$GR00T_PY" "$LOCOMANI_SRC/pipeline/eval_open_loop.py" \
    --model-path "$CKPT_DIR/$BEST" --dataset "$VAL_DS" --horizons 1 8 16 40 \
    --out "$R4/open_loop_best_full.json" --wandb-run "$RUN" \
    || echo "  (full-horizon pass failed)"

# --- 7. closed-loop -----------------------------------------------------------
say "7/8 closed-loop (full budget, config: '${FLAGS:-default}')"
if [ ! -f "$R4/closed_loop.json" ]; then
    kill_stack
    sleep 2
    NO_ENCODER=1 SCENE_RANDOM=0 SIM_VIDEO_DIR="$LOCOMANI_OUTPUTS/sim_video" \
        bash "$LOCOMANI_STACK/start_stack.sh" "pick up the water bottle"
    serve "$CKPT_DIR/$BEST" || exit 1
    # shellcheck disable=SC2086
    (cd "$LOCOMANI_SRC/pipeline" && "$VENV_SIM_PY" eval_closed_loop.py \
        --fresh 100 --seed 6000 --client async $FLAGS \
        --out "$R4/closed_loop.json" \
        --trace-dir "$R4/closed_loop_traces")
fi

# --- 8. summary ---------------------------------------------------------------
say "8/8 summary"
python3 - "$T1" "$R4" <<'PYEOF'
import json, sys
t1, r4 = sys.argv[1], sys.argv[2]
def g(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
out = {
    "tier1": {"nosnap": g(f"{t1}/nosnap.json"), "h40": g(f"{t1}/h40.json"),
              "combo": g(f"{t1}/combo.json"), "anatomy": g(f"{t1}/anatomy.json")},
    "confirm_config": g(f"{r4}/confirm_config.json"),
    "round4": {"closed_loop": g(f"{r4}/closed_loop.json"),
               "open_loop_best": g(f"{r4}/open_loop_best_full.json")},
    "ladder": {"round1": "6/101", "round2": "14/102", "round3": "28/100"},
}
json.dump(out, open(f"{r4}/SUMMARY.json", "w"), indent=2)
cl = out["round4"]["closed_loop"] or {}
cf = out["confirm_config"] or {}
print(f"  confirm (round-3 model + config): {cf.get('successes')}/100")
print(f"  round 4 (new data + config):      {cl.get('successes')}/{cl.get('rounds')}")
print(f"  -> {r4}/SUMMARY.json")
PYEOF

say "round 4 complete"
