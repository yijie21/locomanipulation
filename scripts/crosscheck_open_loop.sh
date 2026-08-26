#!/bin/bash
# Cross-check our open-loop numbers against NVIDIA's own open-loop eval.
#
#   bash scripts/crosscheck_open_loop.sh [RUN]        # RUN defaults to fast200
#
# WHY THIS EXISTS
#
# src/teleop_free/tools/eval_open_loop.py reports nMSE split by group, a
# snapped hit rate, joint-command degrees and a persistence baseline, because
# a pooled MSE over this action space is close to unreadable: 64 of the 65 dims
# are an FSQ-quantised latent, so the one dim that decides whether the bottle
# is grasped carries 1/65 of the number, and Euclidean distance in a learned
# latent has no physical unit.
#
# All of which is reasoning, not evidence. The metrics are ours, the code is
# ours, and nothing independent has ever agreed with them. NVIDIA's
# gr00t/eval/open_loop_eval.py computes a plain unnormalised action MSE/MAE
# over the same trajectories through the same policy server. It is a weaker
# metric -- and that is the point: it shares no code with ours downstream of
# the server, so if the two disagree about WHICH CHECKPOINT IS BEST, one of
# them has a bug.
#
# Ranking agreement is what is being tested, not the absolute values; the two
# numbers are not comparable in magnitude and are not meant to be.
#
# WHAT IT DOES NOT TEST
#
# Everything up to and including the server is shared: serve_policy.py is a
# 71-line wrapper around the official Gr00tPolicy + PolicyServer and even
# imports parse_action_gr00t from the official eval. So a bug in model
# loading, normalisation or the embodiment projection would fool both. What
# this catches is the part that is genuinely ours: trajectory iteration, the
# execution-horizon indexing, and the metric arithmetic.
#
# WHEN IT RUNS
#
# After run_round3.sh has fully exited -- not alongside it. Stage 5 wants a GPU
# and stage 6's closed-loop measurement is timing-sensitive (the async client
# re-plans every 0.4 s, so anything that slows inference changes the number it
# reports). A cross-check is not worth corrupting the headline result for.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

RUN="${1:-fast200}"
EVAL_DIR="${EVAL_DIR:-$LOCOMANI_OUTPUTS/eval/$RUN}"
CKPT_DIR="${CKPT_DIR:-$LOCOMANI_OUTPUTS/train/$RUN}"
VAL_DS="${VAL_DS:-$LOCOMANI_ROOT/datasets/g1_locomani_${RUN}_val}"
GR00T_PY="${GR00T_PY:-$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python}"
# 5590 is the closed-loop server's port; stay off it even though this runs
# after that has finished, so a stray overlap cannot silently talk to the
# wrong model.
PORT="${PORT:-5591}"
GPU="${GPU:-1}"
HORIZON="${HORIZON:-16}"
OUT="$EVAL_DIR/crosscheck_official.json"
LOG_DIR="${LOG_DIR:-$LOCOMANI_TMP/crosscheck}"
mkdir -p "$EVAL_DIR" "$LOG_DIR"

# Assembled at run time: a literal pattern on the command line matches this
# script's own cmdline and kills the invoking shell (exit 144, twice learned).
P_SRV=$(printf '%s_%s.py' serve policy)

wait_for_driver() {
    local drv; drv=$(printf 'run_%s.sh' round3)
    if pgrep -f "$drv" > /dev/null; then
        echo "waiting for run_round3.sh to finish (checked every 5 min)..."
        while pgrep -f "$drv" > /dev/null; do sleep 300; done
        echo "  driver exited, starting"
        sleep 30      # let its last teardown settle
    fi
}

kill_server() {
    for pid in $(pgrep -f "$P_SRV" || true); do kill -9 "$pid" 2>/dev/null || true; done
    sleep 2
}

# Every val episode, so the trajectory set matches what our eval uses by
# default (n_traj 20). Derived, not hardcoded: a re-holdout would silently
# change it otherwise.
n_val=$("$GR00T_PY" -c "
import json,sys
print(json.load(open('$VAL_DS/meta/info.json'))['total_episodes'])" 2>/dev/null) || n_val=""
if [ -z "$n_val" ]; then echo "cannot read $VAL_DS/meta/info.json"; exit 1; fi
TRAJ_IDS=$(seq 0 $((n_val - 1)) | tr '\n' ' ')

# --steps, and why it has to be set. The official client defaults to 200 and
# caps at the trajectory length (open_loop_eval.py:149), so on this dataset it
# scores only the first 200 of a 546-832 frame episode -- the walk, and none of
# the grasp, which happens around frame 327. The first cross-check run
# disagreed with ours about the best checkpoint for exactly that reason: the
# two were scoring different parts of the episode. Derived from the longest
# episode so it can never silently truncate again.
STEPS=$("$GR00T_PY" -c "
import json
print(max(json.loads(l)['length'] for l in open('$VAL_DS/meta/episodes.jsonl')))" 2>/dev/null) || STEPS=""
[ -z "$STEPS" ] && { echo "cannot read episode lengths from $VAL_DS"; exit 1; }

wait_for_driver

mapfile -t CKPTS < <(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "no checkpoints in $CKPT_DIR -- nothing to cross-check"; exit 1
fi

echo "cross-check: NVIDIA open_loop_eval vs ours"
echo "  checkpoints  ${#CKPTS[@]}"
echo "  val set      $VAL_DS ($n_val episodes)"
echo "  horizon      $HORIZON   steps $STEPS   port $PORT   gpu $GPU"
echo

: > "$LOG_DIR/results.tsv"
for c in "${CKPTS[@]}"; do
    n=$(basename "$c")
    slog="$LOG_DIR/server_$n.log"; clog="$LOG_DIR/client_$n.log"

    kill_server
    CUDA_VISIBLE_DEVICES="$GPU" nohup "$GR00T_PY" \
        "$LOCOMANI_SRC/tools/serve_policy.py" \
        --model-path "$c" --port "$PORT" > "$slog" 2>&1 &
    # serve_policy prints its banner only after Gr00tPolicy is constructed, so
    # the banner IS the ready signal -- no sleep-and-hope.
    up=0
    for _ in $(seq 1 90); do
        grep -q "\[serve_policy\]" "$slog" 2>/dev/null && { up=1; break; }
        sleep 5
    done
    if [ "$up" != "1" ]; then
        echo "  $n: server never came up"; tail -5 "$slog" | sed 's/^/    /'
        kill_server; continue
    fi

    # shellcheck disable=SC2086
    "$GR00T_PY" "$LOCOMANI_ROOT/src/gr00t/eval/open_loop_eval.py" \
        --dataset-path "$VAL_DS" \
        --embodiment-tag UNITREE_G1_SONIC \
        --host 127.0.0.1 --port "$PORT" \
        --traj-ids $TRAJ_IDS \
        --steps "$STEPS" \
        --execution-horizon "$HORIZON" > "$clog" 2>&1
    kill_server

    mse=$(grep -oE "Average MSE across all trajs: [0-9.eE+-]+" "$clog" | tail -1 | awk '{print $NF}')
    mae=$(grep -oE "Average MAE across all trajs: [0-9.eE+-]+" "$clog" | tail -1 | awk '{print $NF}')
    if [ -z "$mse" ]; then
        echo "  $n: client produced no MSE"; tail -5 "$clog" | sed 's/^/    /'
        continue
    fi
    printf '%s\t%s\t%s\n' "$n" "$mse" "$mae" >> "$LOG_DIR/results.tsv"
    printf '  %-20s official MSE %-12s MAE %s\n' "$n" "$mse" "$mae"
done

echo
"$GR00T_PY" - "$LOG_DIR/results.tsv" "$EVAL_DIR" "$OUT" "$HORIZON" <<'PYEOF'
import json, re, sys
from pathlib import Path

tsv, eval_dir, out, horizon = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]

official = {}
for line in Path(tsv).read_text().splitlines():
    if not line.strip():
        continue
    name, mse, mae = line.split("\t")
    official[name] = {"mse": float(mse), "mae": float(mae)}

ours = {}
for f in sorted(eval_dir.glob("open_loop_checkpoint-*.json")):
    name = re.search(r"(checkpoint-\d+)", f.name).group(1)
    try:
        h = json.loads(f.read_text())["horizons"][horizon]["model"]
        ours[name] = {"nmse_token": h["nmse_token"], "nmse_gripper": h.get("nmse_gripper")}
    except Exception:
        pass

shared = [k for k in official if k in ours]
step = lambda n: int(n.split("-")[1])
shared.sort(key=step)

rows = []
for n in shared:
    rows.append({"checkpoint": n, "step": step(n),
                 "official_mse": official[n]["mse"], "official_mae": official[n]["mae"],
                 "ours_nmse_token": ours[n]["nmse_token"],
                 "ours_nmse_gripper": ours[n]["nmse_gripper"]})

verdict = {}
if len(shared) >= 2:
    best_off = min(shared, key=lambda n: official[n]["mse"])
    best_ours = min(shared, key=lambda n: ours[n]["nmse_token"])
    # Spearman without scipy: rank correlation over the shared checkpoints.
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0] * len(vals)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    a = ranks([official[n]["mse"] for n in shared])
    b = ranks([ours[n]["nmse_token"] for n in shared])
    k = len(shared)
    d2 = sum((x - y) ** 2 for x, y in zip(a, b))
    rho = 1 - 6 * d2 / (k * (k * k - 1))
    verdict = {"best_official": best_off, "best_ours": best_ours,
               "same_best": best_off == best_ours, "spearman": rho, "n": k}

Path(out).write_text(json.dumps(
    {"horizon": horizon, "rows": rows, "verdict": verdict}, indent=2))

print(f"{'checkpoint':<20} {'official MSE':>14} {'ours nMSE(token)':>18} {'ours nMSE(grip)':>17}")
for r in rows:
    g = "n/a" if r["ours_nmse_gripper"] is None else f"{r['ours_nmse_gripper']:.4f}"
    print(f"{r['checkpoint']:<20} {r['official_mse']:>14.6f} "
          f"{r['ours_nmse_token']:>18.4f} {g:>17}")

if verdict:
    print()
    print(f"best by NVIDIA's MSE : {verdict['best_official']}")
    print(f"best by our nMSE     : {verdict['best_ours']}")
    print(f"Spearman rho         : {verdict['spearman']:+.3f}  (over {verdict['n']} checkpoints)")
    print()
    if verdict["same_best"]:
        print("AGREE on the best checkpoint. The two metrics rank it the same way,")
        print("so our trajectory iteration, horizon indexing and metric arithmetic")
        print("are doing what the official implementation does. They are still the")
        print("same numbers about the same server, so this does not clear anything")
        print("upstream of it.")
    else:
        print("DISAGREE on the best checkpoint. One of the two is wrong, and it is")
        print("worth resolving before trusting either for a closed-loop pick.")
        print("Where to look first, in order:")
        print("  1. execution-horizon indexing -- which predicted step is compared")
        print("     against which ground-truth frame (an off-by-one survives a")
        print("     pooled MSE and shows up as a ranking flip)")
        print("  2. trajectory iteration -- same episodes, same frame ranges?")
        print("  3. the metrics themselves: a low rho with a shared best is normal")
        print("     (they measure different things); a HIGH rho with different bests")
        print("     is not, and points at a tie-break rather than a bug.")
print()
print(f"written {out}")
PYEOF
