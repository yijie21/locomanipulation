#!/bin/bash
# Targeted recovery collection: the scripted expert, spawned where the VLA
# stalls.
#
#   bash scripts/collect_dagger.sh [N]          # default 50 episodes
#
# WHY THIS EXISTS -- the measured failure mode it is aimed at
#
# The first closed-loop evaluation (checkpoint-10000, 100 fresh rounds) showed
# one collapsed failure mode: the VLA walks to the grasp stand cleanly and
# then freezes -- at the table, arm hanging, gripper open -- because the
# training episodes never contain that state. The scripted expert raises its
# arm DURING the walk, so "arrived with the arm still down" appears in zero
# of 200 episodes, and the policy, asked what to do there, answers with the
# nearest thing it knows: stand. Which is the same state. A fixed point.
#
# THE DAGGER IDEA, AND WHY IT IS CHEAP HERE
#
# The textbook fix is DAgger: collect the EXPERT's actions on the STUDENT's
# states, so the training distribution grows to cover where the student
# actually goes. The usual price is a live handover, which this stack cannot
# do in one round -- the expert's token labels come from the encoder, the
# student needs the encoder absent (they cannot share a deploy).
#
# But the student's stall state is, to within noise, A NATURAL SETTLED STAND
# at the table: arms down is just what standing looks like. So spawning the
# expert close to the table and letting it settle REPRODUCES the student's
# state without any handover -- and its recovery from there (raise at the
# table, reach, close, lift) is exactly the missing supervision. The near
# spawn also compresses the walk phase, which rebalances a dataset that is
# currently 55% walking by frames.
#
# Spawn range 0.35-0.55 m from the table edge (the eval's stall poses sit at
# 0.17-0.44 m pelvis-to-object, which is roughly 0.35-0.6 m pelvis-to-edge
# once the object's inset is added). The floor is watched empirically: the
# stand-up needs forward clearance (the rising knees ride ~0.25 m ahead), so
# run PROBE=1 first -- 10 rounds, keep nothing -- and read the stand failure
# rate before spending the full budget. If [stand] failures dominate, raise
# DAGGER_EDGE_LO.
#
# PREREQUISITE: the stack must be running WITH its encoder (a normal
# collection stack, NOT the NO_ENCODER eval stack) -- the encoder is what
# writes the action.motion_token labels.
#
# Afterwards, convert old + new together (the pose columns differ by era --
# the TF_POSE_OBJECTS restart bug -- so strip them):
#   ... convert_to_hf_format.py --sessions <old...> <new...> \
#       --manifest <old...> <new...> --strip-pose-columns \
#       --holdout 20 --out datasets/g1_locomani_bottle250_train ...

set -eu
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT"

N="${1:-50}"
LO="${DAGGER_EDGE_LO:-0.35}"
HI="${DAGGER_EDGE_HI:-0.55}"

# Refuse to label with zeros: no encoder means no motion_token labels.
if pgrep -f "g1_deploy""_onnx_ref" >/dev/null; then
    if ! ps -o args= -p "$(pgrep -f 'g1_deploy''_onnx_ref' | head -1)" \
            | grep -q "encoder-file"; then
        echo "the running deploy has NO ENCODER -- its token labels would be"
        echo "zeros. Restart the collection stack first:"
        echo "  SCENE_RANDOM=0 bash src/teleop_free/start_stack.sh \"pick up the water bottle\""
        exit 1
    fi
else
    echo "no deploy running -- bring the collection stack up first"; exit 1
fi

SPAWN_ARGS="--spawn-edge $LO $HI --face-object --spawn-yaw-err 10 --spawn-lateral 0.10"

if [ "${PROBE:-0}" = "1" ]; then
    echo "probe: 10 rounds at edge $LO-$HI m, recording nothing --"
    echo "       read the [stand] failure rate before the real run"
    "$VENV_SIM_PY" "$LOCOMANI_SRC/grasp_gym.py" \
        --rounds 10 --targets water_bottle \
        --manifest "$LOCOMANI_OUTPUTS/logs/dagger_probe.jsonl" \
        $SPAWN_ARGS --edge-jitter 0.20
    echo; echo "stand failures:"
    grep -c '"why": "\[stand\]' "$LOCOMANI_OUTPUTS/logs/dagger_probe.jsonl" || true
    exit 0
fi

OUT="$LOCOMANI_OUTPUTS/logs/dagger$N" \
SPAWN_ARGS="$SPAWN_ARGS" \
OBJ=water_bottle PROMPT="pick up the water bottle" JITTER=0.20 \
CHUNK=20 SPARE=10 \
exec bash "$LOCOMANI_ROOT/scripts/collect_pilot.sh" "$N" water_bottle
