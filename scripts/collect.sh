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
# DART collection noise (recovery demonstrations; see grasp_gym --dart-frac):
#
#   DART_ARGS="--dart-frac 0.7" ... bash scripts/collect.sh 12 20 1
#
# 0.7 of rounds first-stop at ~0.45-0.70 m from the object (see --dart-far:
# the draw is the brake trigger; deceleration carry lands the realized stop
# 0.06-0.16 m short of it) and the repair loop
# walks the rest in -- calibrated to the measured policy park tail (19% of
# closed-loop rounds park >= 0.50 m, where success is 5%). Budget ~20% more
# rounds than the target episode count: the far stop costs some successes.
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
DART_ARGS="${DART_ARGS:-}"
TARGETS="${TARGETS:-pool}"
JITTER="${JITTER:-0.10}"
# Overridable so a run can keep its manifests in their own directory: the
# converter pairs sessions with manifests POSITIONALLY, so a shared
# directory holding an older run's manifests silently shifts that join.
OUT="${OUT:-$LOCOMANI_OUTPUTS/logs/collect}"
mkdir -p "$OUT"

# Visual backend, NOT physics. `mujoco` (the default) is the status quo to the
# byte: nothing below the loop runs. `isaac` appends a per-episode offline
# re-render stage after the batch, reading the pose sidecar and this batch's
# scene snapshot -- the physics loop, the stack and the ZMQ timing are
# untouched either way. Unknown values are refused rather than silently
# falling back, because a typo that quietly collected 400 episodes without
# pixels is exactly the failure this flag has to not have.
RENDER_BACKEND="${RENDER_BACKEND:-mujoco}"
case "$RENDER_BACKEND" in
    mujoco|isaac) ;;
    *) echo "collect: RENDER_BACKEND must be mujoco or isaac, got '$RENDER_BACKEND'" >&2
       exit 2 ;;
esac

echo "collect: $BATCHES batches x $ROUNDS rounds, scene seeds from $SEED0"
echo "  preset  ${SCENE_PRESET:-<none> (default multi-object table)}"
echo "  targets $TARGETS   edge-jitter $JITTER"
echo "  spawn   ${SPAWN_ARGS:-<grasp_gym defaults: 1-3 m out, off the approach line, +-90 deg>}"
echo "  dart    ${DART_ARGS:-<off>}"
echo "  render  $RENDER_BACKEND"
for b in $(seq 0 $((BATCHES - 1))); do
    seed=$((SEED0 + b))
    echo "===== batch $((b + 1))/$BATCHES  scene seed $seed ====="

    # COLLECT_NO_STOP=1: another instance's stack is running (an eval);
    # stop only THIS instance instead of every stack on the machine.
    if [ "${COLLECT_NO_STOP:-0}" = "1" ]; then
        bash src/teleop_free/stop_instance.sh "${INSTANCE:-0}" > /dev/null 2>&1
    else
        bash src/teleop_free/stop_stack.sh > /dev/null 2>&1
    fi
    SCENE_SEED="$seed" "$VENV_SIM_PY" src/teleop_free/build_sim_scene.py \
        outputs/scene outputs/scene/sim | grep -E '^\[scene\]' || {
        echo "  scene rebuild FAILED"; continue; }

    # Snapshot the scene this batch was collected in. build_sim_scene.py
    # rewrites outputs/scene/sim IN PLACE, so without this the XML an episode
    # was recorded against is gone the moment the next batch starts, and an
    # offline re-render (Isaac / 3DGS) can only regenerate it from the seed --
    # which stops being exact the first time build_sim_scene.py or scene_lib.py
    # changes. Table height varies 0.62-0.65 m across seeds, so a wrong scene
    # is a table at the wrong height under a correct trajectory, not a subtle
    # error. One cp and the two hashes that make it reproducible.
    SNAP="$OUT/scene_$seed"; mkdir -p "$SNAP"
    cp outputs/scene/sim/scene_locomani.xml outputs/scene/sim/scene_meta.json \
       "$SNAP/" 2>/dev/null || echo "  scene snapshot failed (non-fatal)"
    printf '{"scene_seed":%s,"preset":"%s","built":"%s","git":{"build_sim_scene.py":"%s","scene_lib.py":"%s"}}\n' \
        "$seed" "${SCENE_PRESET:-}" "$(date -Is)" \
        "$(git log -1 --format=%h -- src/teleop_free/build_sim_scene.py 2>/dev/null)" \
        "$(git log -1 --format=%h -- src/teleop_free/scene_lib.py 2>/dev/null)" \
        > "$SNAP/provenance.json"

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
        $SPAWN_ARGS $DART_ARGS \
        --manifest "$OUT/manifest_$seed.jsonl" \
        > "$OUT/gym_$seed.log" 2>&1
    grep -E "^\[gym\] [0-9]+/" "$OUT/gym_$seed.log" || echo "  (no result line)"

    # Isaac re-render stage. Ticket 06 wires the flag only: this prints the
    # plan and starts nothing, so the mujoco path stays provably unchanged
    # while the stage itself is built in ticket 07.
    if [ "$RENDER_BACKEND" = "isaac" ]; then
        echo "  [isaac] would re-render batch $seed:"
        echo "  [isaac]   scene    $OUT/scene_$seed/scene_locomani.xml"
        echo "  [isaac]   sidecar  $(locomani_session_root)/pose_sidecar/"
        echo "  [isaac]   cam traj $RENDER_PY src/teleop_free/isaac/compute_cam_traj.py"
        echo "  [isaac]   render   $RENDER_PY src/teleop_free/isaac/isaac_render.py"
        echo "  [isaac] not implemented yet (ticket 07); nothing started"
    fi
done

if [ "${COLLECT_NO_STOP:-0}" = "1" ]; then
    bash src/teleop_free/stop_instance.sh "${INSTANCE:-0}" > /dev/null 2>&1
else
    bash src/teleop_free/stop_stack.sh > /dev/null 2>&1
fi
echo
echo "===== totals ====="
grep -h -oE "^EPISODE (OK|FAILED \[[a-z]+\])" "$OUT"/gym_*.log | sort | uniq -c
echo "manifests: $OUT/manifest_*.jsonl"
echo "sessions:  $(locomani_session_root)/outputs/"
