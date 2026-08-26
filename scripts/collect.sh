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
# The spawn envelope is NOT baked in here. It used to be missing entirely --
# this script called grasp_gym with only --rounds/--record/--manifest, so it
# could not reproduce the distribution every experiment in this repo was run
# on, while README pointed people here for exactly that. SPAWN_ARGS, TARGETS
# and JITTER now pass through, with grasp_gym's own (wide) defaults unchanged
# so old invocations behave as before. To reproduce the experiments:
#
#   SCENE_PRESET=bottle_only TARGETS=water_bottle JITTER=0.20 \
#   SPAWN_ARGS="--spawn-edge 0.7 --face-object --spawn-yaw-err 20 --spawn-lateral 0.20" \
#       bash scripts/collect.sh 10 20 1
#
# SCENE_PRESET matters here and not only at install time: this script rebuilds
# the scene once per batch, and the preset is read at that moment.
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
# Passed straight to grasp_gym. Defaults match grasp_gym's own, so leaving them
# unset is exactly the old behaviour -- and exactly the wide distribution.
SPAWN_ARGS="${SPAWN_ARGS:-}"
TARGETS="${TARGETS:-pool}"
JITTER="${JITTER:-0.10}"
OUT="$LOCOMANI_OUTPUTS/logs/collect"
mkdir -p "$OUT"

echo "collect: $BATCHES batches x $ROUNDS rounds, scene seeds from $SEED0"
echo "  preset  ${SCENE_PRESET:-<none> (default multi-object table)}"
echo "  targets $TARGETS   edge-jitter $JITTER"
echo "  spawn   ${SPAWN_ARGS:-<grasp_gym defaults: 1-3 m out, off the approach line, +-90 deg>}"
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
        --targets "$TARGETS" --edge-jitter "$JITTER" \
        $SPAWN_ARGS \
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
