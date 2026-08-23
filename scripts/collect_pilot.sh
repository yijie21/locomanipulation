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
# It does NOT bring the stack up, and that is deliberate: every chunk has to
# land in the SAME exporter session, and the session is created when the
# exporter starts. Bring it up yourself, once, with the appearance pinned:
#
#   export SCENE_RANDOM=0 SIM_VIS=bbox,readout
#   SIM_VIDEO_DIR=outputs/sim_video \
#       bash src/teleop_free/start_stack.sh "pick up the red can"
#
# Chunks exist for resumability, not for any per-chunk randomisation: run it
# again after a crash and it counts what is already on disk and tops up.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

TARGET_OK="${1:-200}"
OBJ="${2:-can}"
CHUNK="${CHUNK:-50}"          # successes per gym process
JITTER="${JITTER:-0.20}"      # along-edge scatter, metres either way
SPARE="${SPARE:-20}"          # attempts a chunk may spend above its target
MAX_CHUNKS="${MAX_CHUNKS:-12}"
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
    tag=$(printf %02d "$i")
    echo "===== chunk $tag: $have/$TARGET_OK kept, asking for $need more ====="
    "$VENV_SIM_PY" -u src/teleop_free/grasp_gym.py \
        --rounds $((need + SPARE)) --until-ok "$need" \
        --targets "$OBJ" --edge-jitter "$JITTER" --record \
        --manifest "$OUT/manifest_$tag.jsonl" \
        > "$OUT/gym_$tag.log" 2>&1
    grep -E "^\[gym\] [0-9]+/" "$OUT/gym_$tag.log" || echo "  (no result line)"
done

echo
echo "===== totals ====="
grep -h -oE "^EPISODE (OK|FAILED \[[a-z]+\])" "$OUT"/gym_*.log | sort | uniq -c
echo "kept: $(kept)/$TARGET_OK"
cat "$OUT"/manifest_*.jsonl > "$OUT/manifest_all.jsonl"
echo "manifest (in order): $OUT/manifest_all.jsonl"
echo "session:  $(ls -dt "$(locomani_session_root)"/outputs/*/ 2>/dev/null | head -1)"
