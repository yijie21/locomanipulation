#!/bin/bash
# Paired closed-loop A/B on TWO parallel stack instances with sequential
# early stopping -- the A+C+B harness (docs/evaluation.md).
#
#   bash scripts/seq_ab.sh <out_dir> <flags_a> <flags_b>
#
#   FLAGS are the eval_closed_loop.py arm-specific arguments, e.g.
#     bash scripts/seq_ab.sh outputs/eval/seq_ens "" "--ensemble 3 --ensemble-m 0.3"
#
#   Env: ROUNDS=100 BUDGET=60 SEED=6000 CHECK_EVERY=20 CKPT=... COMMON=...
#
# WHAT IT DOES
#   * arm A on INSTANCE=0 (sim GPU0's EGL default, policy server GPU 0 :6590)
#   * arm B on INSTANCE=1 (policy server GPU 1 :6591)
#     -- 6590/6591 because INSTANCE=1's keyboard port IS 5590, serve_policy's
#     historical default (scripts/paths.sh warns about exactly this).
#   * both arms: same --seed, same --fresh, same --budget -> round i of A and
#     round i of B draw identical initial conditions (one RNG stream), so the
#     comparison is paired and seq_mcnemar.py verifies it via `appearance`.
#   * every CHECK_EVERY paired rounds, an exact McNemar look: DIFFERENT
#     (p<0.01 interim / p<0.05 final), FUTILE (cannot reach significance by
#     ROUNDS), or CONTINUE. DIFFERENT/FUTILE kills both arms early.
#
# WHAT IT IS FOR -- SCREENING. BUDGET defaults to 60 s (not the ladder's 90):
# the bias is shared by both arms and downward (7/48 round-4 successes landed
# after 60 s). A winner here still gets one full-budget 90 s confirmation
# before any conclusion; a FUTILE here is a real "not worth 6 hours".
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

OUT_DIR="${1:?usage: seq_ab.sh <out_dir> <flags_a> <flags_b>}"
FLAGS_A="${2-}"
FLAGS_B="${3-}"
ROUNDS="${ROUNDS:-100}"
BUDGET="${BUDGET:-60}"
SEED="${SEED:-6000}"
CHECK_EVERY="${CHECK_EVERY:-20}"
CKPT="${CKPT:-$LOCOMANI_OUTPUTS/train/fast250/checkpoint-10000}"
COMMON="${COMMON:---client async --no-snap}"
GR00T_PY="${GR00T_PY:-$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python}"
PY="$VENV_SIM_PY"
LOGS="$LOCOMANI_TMP/seq_ab"
mkdir -p "$OUT_DIR" "$LOGS"

[ -d "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }

# Assembled at run time: a literal pattern on the command line matches this
# script's own cmdline and kills the invoking shell (exit 144, learned twice).
P_SRV=$(printf '%s_%s.py' serve policy)

ports_for() {  # instance -> the three LISTENING ports worth sweeping
    echo "$((5555 + 10 * $1)) $((5581 + 10 * $1)) $((5557 + 10 * $1))"
}

kill_instance() {
    local i=$1 port pid
    for port in $(ports_for "$i"); do
        pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -1)
        [ -n "${pid:-}" ] && kill -9 "$pid" 2>/dev/null
    done
    # deploys hold no listening port; match by their per-instance DDS domain
    for pid in $(pgrep -f "g1_deploy" | head -20); do
        if { tr '\0' ' ' < "/proc/$pid/environ" | grep -q "SONIC_DDS_DOMAIN=$i"; } 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    # the exporter subscribes (no listening port) so the sweep above misses
    # it; instance 0's stack starts one and 35 of them once exhausted the
    # thread quota (par_batch.sh learned this first). Eval records nothing,
    # so killing every exporter is always safe here.
    for pid in $(pgrep -f run_data_exporter); do kill -9 "$pid" 2>/dev/null; done
    return 0
}

EVAL_PIDS=""
SRV_PIDS=""
cleanup() {
    for p in $EVAL_PIDS $SRV_PIDS; do kill "$p" 2>/dev/null; done
    sleep 2
    for p in $EVAL_PIDS $SRV_PIDS; do kill -9 "$p" 2>/dev/null; done
    # belt and braces: the stored PIDs can miss a re-exec'd child (an aborted
    # run left both arms alive once); match by OUR out dir so a concurrent
    # unrelated eval is never touched
    for p in $(pgrep -f "eval_closed_loop.*$OUT_DIR" || true); do
        kill -9 "$p" 2>/dev/null
    done
    kill_instance 0; kill_instance 1
}
trap cleanup EXIT INT TERM

# --- policy servers, one per GPU ------------------------------------------
for i in 0 1; do
    port=$((6590 + i))
    for pid in $(pgrep -f "$P_SRV.*$port" || true); do kill -9 "$pid" 2>/dev/null; done
done
sleep 1
for i in 0 1; do
    port=$((6590 + i)); slog="$LOGS/serve_$i.log"
    CUDA_VISIBLE_DEVICES=$i nohup "$GR00T_PY" "$LOCOMANI_SRC/tools/serve_policy.py" \
        --model-path "$CKPT" --port "$port" > "$slog" 2>&1 &
    SRV_PIDS="$SRV_PIDS $!"
done
echo "[seq_ab] loading policy servers :6590 (gpu0) :6591 (gpu1)..."
for i in 0 1; do
    up=0
    for _ in $(seq 1 90); do
        grep -q "\[serve_policy\]" "$LOGS/serve_$i.log" 2>/dev/null && { up=1; break; }
        sleep 5
    done
    [ "$up" = "1" ] || { echo "server $i never came up"; tail -5 "$LOGS/serve_$i.log"; exit 1; }
done
echo "[seq_ab] servers up"

# --- stacks, one per instance ---------------------------------------------
boot_stack() {
    local i=$1
    kill_instance "$i"; sleep 2
    env INSTANCE=$i NO_ENCODER=1 SIM_VIDEO_DIR="$PWD/outputs/videos/seq_ab_i$i" \
        nohup bash src/teleop_free/start_stack.sh > "$LOGS/stack_$i.log" 2>&1 &
    for _ in $(seq 1 90); do
        grep -q "stack up" "$LOGS/stack_$i.log" 2>/dev/null && return 0
        sleep 5
    done
    echo "stack $i never came up"; tail -20 "$LOGS/stack_$i.log"
    return 1
}
echo "[seq_ab] booting stacks (instance 0 and 1)..."
boot_stack 0 || exit 1
boot_stack 1 || exit 1
echo "[seq_ab] stacks up"

# --- the two arms ----------------------------------------------------------
launch_arm() {   # $1 instance  $2 name  $3 arm flags
    local i=$1 name=$2 flags=$3
    # shellcheck disable=SC2086
    env INSTANCE=$i SONIC_DDS_DOMAIN=$i NO_ENCODER=1 \
        TF_CAMERA_PORT=$((5555 + 10 * i)) TF_SONIC_PORT=$((5556 + 10 * i)) \
        TF_OUT_PORT=$((5557 + 10 * i)) TF_KEYBOARD_PORT=$((5580 + 10 * i)) \
        TF_PRIVILEGED_PORT=$((5581 + 10 * i)) TF_EPISODE_PORT=$((5582 + 10 * i)) \
        nohup "$PY" src/teleop_free/tools/eval_closed_loop.py \
        --fresh "$ROUNDS" --seed "$SEED" --budget "$BUDGET" \
        --policy-port $((6590 + i)) $COMMON $flags \
        --out "$OUT_DIR/$name.json" --trace-dir "$OUT_DIR/${name}_traces" \
        > "$LOGS/eval_$name.log" 2>&1 &
    echo $!
}
PID_A=$(launch_arm 0 a "$FLAGS_A")
PID_B=$(launch_arm 1 b "$FLAGS_B")
EVAL_PIDS="$PID_A $PID_B"
MAN_A="$OUT_DIR/a.manifest.jsonl"
MAN_B="$OUT_DIR/b.manifest.jsonl"

cat > "$OUT_DIR/run_meta.json" <<EOF
{"parallel": true, "rounds": $ROUNDS, "budget_s": $BUDGET, "seed": $SEED,
 "check_every": $CHECK_EVERY, "ckpt": "$CKPT",
 "common": "$COMMON", "flags_a": "$FLAGS_A", "flags_b": "$FLAGS_B"}
EOF
echo "[seq_ab] arms launched: A(inst0) pid $PID_A [$FLAGS_A]  B(inst1) pid $PID_B [$FLAGS_B]"
echo "[seq_ab] budget ${BUDGET}s, $ROUNDS rounds max, look every $CHECK_EVERY paired rounds"

# --- sequential looks ------------------------------------------------------
next_look=$CHECK_EVERY
VERDICT=""
while :; do
    sleep 60
    na=$( [ -f "$MAN_A" ] && wc -l < "$MAN_A" || echo 0 )
    nb=$( [ -f "$MAN_B" ] && wc -l < "$MAN_B" || echo 0 )
    n=$(( na < nb ? na : nb ))
    a_alive=1; b_alive=1
    kill -0 "$PID_A" 2>/dev/null || a_alive=0
    kill -0 "$PID_B" 2>/dev/null || b_alive=0

    if [ "$n" -ge "$next_look" ] || { [ "$a_alive" = 0 ] && [ "$b_alive" = 0 ] && [ "$n" -gt 0 ]; }; then
        echo "[seq_ab] look at n=$n ($(date +%T))"
        look=$("$PY" src/teleop_free/tools/seq_mcnemar.py "$MAN_A" "$MAN_B" \
               --n-max "$ROUNDS" --json "$OUT_DIR/seq_verdict.json") || {
            echo "$look"; echo "[seq_ab] PAIRING BROKEN -- aborting"; exit 1; }
        echo "  $look" | head -1
        VERDICT=$(echo "$look" | awk '/^VERDICT/{print $2}')
        case "$VERDICT" in
            DIFFERENT|FUTILE|NO_DIFFERENCE)
                echo "[seq_ab] stopping: $VERDICT"
                break;;
        esac
        while [ "$next_look" -le "$n" ]; do next_look=$((next_look + CHECK_EVERY)); done
    fi

    if [ "$a_alive" = 0 ] && [ "$b_alive" = 0 ]; then
        echo "[seq_ab] both arms exited (n=$n)"
        break
    fi
    # one arm died early with the other still short of ROUNDS -> broken run
    if [ "$a_alive" = 0 ] && [ "$na" -lt "$ROUNDS" ]; then
        echo "[seq_ab] arm A died at $na/$ROUNDS -- see $LOGS/eval_a.log"; exit 1
    fi
    if [ "$b_alive" = 0 ] && [ "$nb" -lt "$ROUNDS" ]; then
        echo "[seq_ab] arm B died at $nb/$ROUNDS -- see $LOGS/eval_b.log"; exit 1
    fi
done

cleanup
trap - EXIT

# Archive the sim logs with the results: frame_rate_guard.py needs the log
# that was live DURING the run, and stack restarts overwrite it -- round 4's
# obs-rate can never be audited for exactly this reason. A few hundred KB.
cp "$(locomani_session_root 0)/sim.log" "$OUT_DIR/sim_i0.log" 2>/dev/null
cp "$(locomani_session_root 1)/sim.log" "$OUT_DIR/sim_i1.log" 2>/dev/null

echo
echo "===== seq_ab result ====="
cat "$OUT_DIR/seq_verdict.json" 2>/dev/null
echo
case "$VERDICT" in
    DIFFERENT)
        echo "Arms DIFFER at screening budget (${BUDGET}s). Before concluding,"
        echo "confirm the winner at the full 90 s budget against the ladder.";;
    FUTILE|NO_DIFFERENCE)
        echo "No resolvable difference by $ROUNDS rounds at ${BUDGET}s budget.";;
esac
