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
#       bash src/teleop_free/start_stack.sh "pick up the red can"
#
# CHUNK is not a tuning knob -- it is a hard ceiling. The exporter leaks
# about 93 threads per episode and this container's cgroup caps the instance
# at pids.max = 3840, so a session wedges for good at roughly its 33rd
# episode (measured: 3061 exporter threads, pids.current 3707/3840, gym and
# exporter both frozen in a futex while sim and deploy kept running). Each
# chunk therefore ends by restarting ONLY the exporter, which frees the
# threads and rolls to a fresh session. Keep CHUNK well under 30.
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
PROMPT="${PROMPT:-pick up the red can}"
OUT="$LOCOMANI_OUTPUTS/logs/pilot"
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
    [ "$cap" -gt 30 ] && cap=30
    sess=$(ls -dt "$(locomani_session_root)"/outputs/*/ 2>/dev/null | head -1)
    echo "===== chunk $tag: $have/$TARGET_OK kept, asking for $need more ====="
    echo "$sess" >> "$OUT/sessions.txt"
    "$VENV_SIM_PY" -u src/teleop_free/grasp_gym.py \
        --rounds "$cap" --until-ok "$need" \
        --targets "$OBJ" --edge-jitter "$JITTER" --record \
        --manifest "$OUT/manifest_$tag.jsonl" \
        > "$OUT/gym_$tag.log" 2>&1
    grep -E "^\[gym\] [0-9]+/" "$OUT/gym_$tag.log" || echo "  (no result line)"
    # Roll to a fresh session before the next chunk fills this one's quota.
    if new=$(TASK_PROMPT="$PROMPT" bash "$(dirname "$0")/restart_exporter.sh"); then
        echo "  new session $new"
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
