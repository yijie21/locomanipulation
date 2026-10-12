#!/bin/bash
# TODO 21.3.4 / 23.2: every generated object over N lanes (instance = first instance + lane, all on --gpu), run_grasp.py's
# defaults (cached grasp, IK planner, one round) unless flags say otherwise.
#   bash run_batch.sh [lanes 4] [first instance 1] [gpu 0] [tag] [run_grasp.py flags]   touch <ROOT>/STOP to stop
#   bash run_batch.sh 4 1 0 _mine                                   the default configuration
#   bash run_batch.sh 4 1 0 _gn --grasp-source graspnet             graspnet-baseline grasps (servers.sh start graspnet)
#   ROUNDS=2 bash run_batch.sh 4 1 0 _cmp --grasp-source graspnet --planners ik,curobo --rounds 2   the 21.3.4 comparison
#   ROUNDS=3 bash run_batch.sh 4 1 0 _cache --rounds 3 --yaws 3     the cache acceptance (TODO 23.2.5)
# A lane takes the next object not yet done (outputs/runs/gnb_obj_XXX<tag>/skills/manifest.jsonl has ROUNDS rounds); one
# object's crash does not stop the lane. Logs: outputs/graspnet_bottle_20261008/logs/batch<tag>_lane<k>.log,
# obj_XXX<tag>.log.
set -u
cd "$(dirname "$0")/../../.." || exit 1
LANES=${1:-4}; FIRST=${2:-1}; GPU=${3:-0}; TAG=${4:-}
EXTRA=("${@:5}")
ROOT=outputs/graspnet_bottle_20261008
L=$ROOT/logs
OBJECTS=$(python3 -c "import json;print(' '.join(str(o['id']) for o in json.load(open('$ROOT/objects.json'))['objects']))")
ROUNDS=${ROUNDS:-1}   # an object is done when its manifest has this many rounds (match --rounds in the extra flags)
done_object() { local m="outputs/runs/gnb_obj_$(printf %03d "$1")$TAG/skills/manifest.jsonl"; [ -f "$m" ] && [ "$(wc -l < "$m")" -ge "$ROUNDS" ]; }
lane() {
    local k=$1 inst=$((FIRST + $1))
    for o in $OBJECTS; do
        [ -f "$ROOT/STOP" ] && { echo "lane $k: STOP"; return; }
        done_object "$o" && continue
        mkdir "$ROOT/claims$TAG/obj_$o" 2>/dev/null || continue   # another lane has it
        echo "lane $k: object $o on instance $inst $(date +%T)"
        timeout 900 bash scripts/experiments/graspnet_bottle_20261008/run_one.sh "$o" "$inst" "$GPU" "$TAG" "${EXTRA[@]}" \
            > "$L/obj_$(printf %03d "$o")$TAG.log" 2>&1
        echo "lane $k: object $o exit $? $(date +%T)"
    done
}
mkdir -p "$ROOT/claims$TAG"
for k in $(seq 0 $((LANES - 1))); do lane "$k" > "$L/batch${TAG}_lane$k.log" 2>&1 & sleep 5; done
wait
echo "batch finished $(date +%T)"
