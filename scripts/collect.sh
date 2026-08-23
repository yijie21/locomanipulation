#!/bin/bash
# Collect a dataset across several scene dimensions.
#
#   bash scripts/collect.sh <batches> <rounds-per-batch> [first-seed]
#
# Table height and object sizes are baked into the compiled scene, so they can
# only change when it is rebuilt -- and the sim reads the XML once at boot.
# That makes them a PER-BATCH axis: rebuild, restart the stack, run N rounds,
# repeat. The rebuild is ~1 s; the stack restart is ~40 s, so at 20 rounds a
# batch the dimension axis costs about 2 s an episode.
#
# Everything else varies per episode inside the batch and needs none of this:
# target identity, spawn scatter, and appearance (scene_random.py).
#
# Each batch fills its own exporter session and writes its own manifest, in
# step, which is what convert_to_hf_format.py --manifest expects.
# NOTE: this rebuilds outputs/scene/sim in place, so the working scene is left
# at whatever the last batch used. `bash scripts/install_scene.sh` puts it back
# to the shipped dimensions.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

BATCHES="${1:-5}"
ROUNDS="${2:-20}"
SEED0="${3:-1}"
OUT="$LOCOMANI_OUTPUTS/logs/collect"
mkdir -p "$OUT"

echo "collect: $BATCHES batches x $ROUNDS rounds, scene seeds from $SEED0"
for b in $(seq 0 $((BATCHES - 1))); do
    seed=$((SEED0 + b))
    echo "===== batch $((b + 1))/$BATCHES  scene seed $seed ====="

    bash src/teleop_free/stop_stack.sh > /dev/null 2>&1
    SCENE_SEED="$seed" "$VENV_SIM_PY" src/teleop_free/build_sim_scene.py \
        outputs/scene outputs/scene/sim | grep -E '^\[scene\]' || {
        echo "  scene rebuild FAILED"; continue; }

    nohup env SIM_VIDEO_DIR="$LOCOMANI_OUTPUTS/sim_video" \
        bash src/teleop_free/start_stack.sh "pick up the object" \
        > "$OUT/stack_$seed.log" 2>&1 &
    # Wait for the stack rather than sleeping: the deploy's first boot after a
    # scene change still loads cached TensorRT engines, but "still booting" and
    # "failed" look identical to a fixed sleep.
    for _ in $(seq 1 120); do
        grep -qiE "stack up|FAILED|Traceback" "$OUT/stack_$seed.log" && break
        sleep 5
    done
    if ! grep -qi "stack up" "$OUT/stack_$seed.log"; then
        echo "  stack did not come up, skipping batch"; tail -5 "$OUT/stack_$seed.log"
        continue
    fi

    "$VENV_SIM_PY" -u src/teleop_free/grasp_gym.py \
        --rounds "$ROUNDS" --record \
        --manifest "$OUT/manifest_$seed.jsonl" \
        > "$OUT/gym_$seed.log" 2>&1
    grep -E "^\[gym\] [0-9]+/" "$OUT/gym_$seed.log" || echo "  (no result line)"
done

bash src/teleop_free/stop_stack.sh > /dev/null 2>&1
echo
echo "===== totals ====="
grep -h -oE "^EPISODE (OK|FAILED \[[a-z]+\])" "$OUT"/gym_*.log | sort | uniq -c
echo "manifests: $OUT/manifest_*.jsonl"
echo "sessions:  $(locomani_session_root)/outputs/"
