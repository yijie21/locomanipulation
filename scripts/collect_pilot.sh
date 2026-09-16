#!/bin/bash
# Pilot collection: ONE target, ONE fixed appearance, ONE scene geometry.
#
#   bash scripts/collect_pilot.sh [how-many-successes] [target]
#
# This is the deliberately narrow batch that tests the PIPELINE (LeRobot
# conversion -> GR00T finetune -> 66-dim actions back into the deploy), not
# the policy's generalisation. Exactly two things vary per episode:
#
#   * where the robot spawns (1-3 m back, +-0.35 m lateral, +-90 deg yaw)
#   * where the object sits on its edge (4-8 cm inboard, +-JITTER along it)
#
# Everything else is pinned, and pinning it is the point: with a single
# target the instruction is constant, and with SCENE_RANDOM=0 the training
# and deployment pixels are the same, so a model that fails here fails for
# a reason we can name.
#
# It does NOT bring the stack up. Bring it up yourself, once, with the
# appearance pinned:
#
#   export SCENE_RANDOM=0 SIM_VIS=bbox,readout
#   SIM_VIDEO_DIR=outputs/sim_video \
#       python -m locomani.stack.cli up --prompt "pick up the red can"
#
# CHUNK is not a tuning knob -- it is a hard ceiling. The exporter leaks
# about 93 threads per episode and this container's cgroup caps the instance
# at pids.max = 3840, so a session wedges for good at roughly its 33rd
# episode (measured: 3061 exporter threads, pids.current 3707/3840, gym and
# exporter both frozen in a futex while sim and deploy kept running). Each
# chunk therefore ends by restarting ONLY the exporter, which frees the
# threads and rolls to a fresh session. MAX_ATTEMPTS (24) is what actually
# keeps a chunk clear of that ceiling -- CHUNK only sets the success target,
# and a chunk that misses it simply hands the shortfall to the next one.
#
# One session per chunk, one manifest per chunk, same order -- which is
# exactly the pairing convert_to_hf_format.py --sessions/--manifest wants.
# sessions.txt records that order so the conversion command can be built
# from it. Re-running after a crash counts what is already kept and tops up.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

TARGET_OK="${1:-200}"
OBJ="${2:-can}"
CHUNK="${CHUNK:-20}"          # successes per SESSION -- see the note above
JITTER="${JITTER:-0.20}"      # along-edge scatter, metres either way
SPARE="${SPARE:-20}"          # attempts a chunk may spend above its target
MAX_CHUNKS="${MAX_CHUNKS:-12}"
# Hard ceiling on ATTEMPTS per chunk, and the reason it is 24 and not 30.
# The exporter leaks ~93 threads per episode against pids.max = 3840, so a
# session wedges around its 33rd (see the header). A cap of 30 sat right on
# that cliff and duly fell off it: 2026-08-25, chunk 01 froze on round 30
# with the gym in a futex while sim and deploy kept running -- eight hours
# of silence before anyone noticed. 24 attempts is ~2900/3840, a quarter of
# the budget in hand. Fewer successes per chunk, so raise MAX_CHUNKS to match.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-24}"
# How long the gym may log nothing before it is presumed wedged. A round logs
# continuously; the longest legitimate gap is the per-round deploy restart
# (20-40 s), so this is ~9x the real worst case.
STALL="${STALL:-360}"
PROMPT="${PROMPT:-pick up the red can}"
# Where the robot wakes up. Empty keeps grasp_gym's own defaults (spawn 1-3 m
# behind the STAND, jitter around the approach line), which is what the first
# pilot used. SPAWN_ARGS lets a batch pin the near, object-facing start
# without this script growing a flag per knob.
SPAWN_ARGS="${SPAWN_ARGS:-}"
# The collection profile. Without it the reach calibration falls back to
# ReachConfig(), whose ready_grasp_recovery=False was retired in refactor step 2
# -- so from that commit until 2026-09-10 this script could not produce a single
# episode: every round walked to the object and then failed with "is retired".
# grasp_gym refuses at startup on that now, and --task-config carries the whole
# profile (reach + spawn + target), so this is one line rather than nine flags.
TASK_CONFIG="${TASK_CONFIG:-configs/tasks/bottle_visible.json}"
OUT="${OUT:-$LOCOMANI_OUTPUTS/logs/pilot}"
mkdir -p "$OUT"

# Successes already on disk. Counted from the manifests, not from the logs:
# the manifest is the same record the converter joins on, so this number and
# the dataset's episode count cannot drift apart.
kept() {
    cat "$OUT"/manifest_*.jsonl 2>/dev/null | python3 -c '
import sys, json
n = 0
for line in sys.stdin:
    line = line.strip()
    if line and json.loads(line).get("ok") and json.loads(line).get("produced"):
        n += 1
print(n)'
}

echo "pilot: $TARGET_OK successful episodes of \"$OBJ\", edge jitter +-$JITTER m"
echo "  out       $OUT"
echo "  spawn     ${SPAWN_ARGS:-<grasp_gym defaults>}"
i=0
while :; do
    have=$(kept)
    if [ "$have" -ge "$TARGET_OK" ]; then break; fi
    i=$((i + 1))
    if [ "$i" -gt "$MAX_CHUNKS" ]; then
        echo "stopping: $MAX_CHUNKS chunks and still $have/$TARGET_OK"; break
    fi
    need=$((TARGET_OK - have))
    [ "$need" -gt "$CHUNK" ] && need=$CHUNK
    # Never reuse a tag: a resumed run must not append into a previous
    # chunk's manifest, or that manifest stops matching its session.
    tag=$(printf %02d "$i")
    while [ -f "$OUT/manifest_$tag.jsonl" ]; do
        i=$((i + 1)); tag=$(printf %02d "$i")
    done
    # The cap counts ATTEMPTS, and every attempt that reaches P2 leaves an
    # episode in the session -- successes and discards alike -- so the cap,
    # not the success target, is what has to stay under the thread ceiling.
    cap=$((need + SPARE))
    [ "$cap" -gt "$MAX_ATTEMPTS" ] && cap=$MAX_ATTEMPTS
    sess=$(ls -dt "$(locomani_session_root)"/outputs/*/ 2>/dev/null | head -1)
    echo "===== chunk $tag: $have/$TARGET_OK kept, asking for $need more ====="
    echo "$sess" >> "$OUT/sessions.txt"
    # Run supervised, not blocking. A wedged gym never exits (it sits in a
    # futex), and this loop would wait on it for as long as the machine is
    # up. Killing it is safe and self-healing: the chunk below tolerates any
    # exit status, rolls the exporter to a fresh session, and takes a new tag,
    # and every episode already in the manifest stays kept.
    "$VENV_SIM_PY" -u src/locomani/tasks/bottle/grasp_gym.py \
        --task-config "$TASK_CONFIG" \
        --rounds "$cap" --until-ok "$need" \
        --targets "$OBJ" --edge-jitter "$JITTER" --record \
        $SPAWN_ARGS \
        --manifest "$OUT/manifest_$tag.jsonl" \
        > "$OUT/gym_$tag.log" 2>&1 &
    gym_pid=$!
    while kill -0 "$gym_pid" 2>/dev/null; do
        sleep 30
        silent=$(( $(date +%s) - $(stat -c %Y "$OUT/gym_$tag.log" 2>/dev/null || date +%s) ))
        if [ "$silent" -gt "$STALL" ]; then
            echo "  STALL: gym silent ${silent}s after round $(grep -c '^\[gym\] round' "$OUT/gym_$tag.log"), killing it"
            kill -9 "$gym_pid" 2>/dev/null
            break
        fi
    done
    wait "$gym_pid" 2>/dev/null || true
    grep -E "^\[gym\] [0-9]+/" "$OUT/gym_$tag.log" || echo "  (no result line)"
    # Roll to a fresh session before the next chunk fills this one's quota.
    if PYTHONPATH=src "$VENV_SIM_PY" -m locomani.stack.cli restart-exporter --instance "${INSTANCE:-0}"; then
        echo "  exporter restarted (new session under $(locomani_session_root "${INSTANCE:-0}")/outputs)"
    else
        echo "  WARNING: exporter restart failed"; break
    fi
done

echo
echo "===== totals ====="
grep -h -oE "^EPISODE (OK|FAILED \[[a-z]+\])" "$OUT"/gym_*.log | sort | uniq -c
echo "kept: $(kept)/$TARGET_OK"
echo
echo "convert with (sessions and manifests pair up in this order):"
echo "  --sessions $(tr '\n' ' ' < "$OUT/sessions.txt")"
echo "  --manifest $(ls "$OUT"/manifest_*.jsonl | sort | tr '\n' ' ')"
