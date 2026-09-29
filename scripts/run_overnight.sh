#!/bin/bash
# The whole line, unattended: collect -> convert -> train -> evaluate.
#
#   bash scripts/run_overnight.sh configs/experiments/bottle_simple_200.yaml --gpu 1
#   bash scripts/run_overnight.sh configs/experiments/crate_table30_record.yaml --gpu 1 --line bimanual
#   bash scripts/run_overnight.sh configs/experiments/box_room_grasp_200_cap120_20260925.yaml --gpu 0 --render-gpu 1 --line bimanual
#
# --render-gpu (old spelling --isaac-gpu): an experiment with `images.source: isaac` or `unreal` renders its dataset
# cameras in that renderer and is evaluated through its mirror, both on this card (the stack, training and policy
# stay on --gpu). Unreal renders in the long-lived editor: started here on this card if none is running
# (scripts/start_unreal.sh), left running at the end.
# --collect-gpu / --lanes: the bimanual line collects in parallel lanes (src/locomani/pipeline/lanes.py), as many
# as the machine fits, spread over every card by default; --lanes 1 collects in one. The single-arm line uses --gpu.
#
# THIS IS THE ONLY SCRIPT IN scripts/ THAT CALLS OTHER SCRIPTS. Everything else is flat: one bash,
# one exec, one python entry. Read the stage scripts to understand the pipeline -- read this one
# only to run it overnight. It exists because 20-100 round A/B batches and gates are how most of
# the numbers in docs/ledger were produced, and a human cannot sit through them.
#
# Each block below is one stage and can be run on its own, in the terminals the stage documents.
# A stage whose artifact is already in the run directory is skipped; to redo one, delete what it
# produced (the dataset directory, model/, eval/closed_loop.json). Collection counts as done only
# when summary.json's collect entry kept as many episodes as the experiment asks for: a collection
# that stopped short writes that entry too, and continuing past it converted the short dataset.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

EXPERIMENT=""; GPU="0"; RENDER_GPU=""; LINE="single_arm"; FROM="collect"; TO="eval"; COLLECT_GPU=""; LANES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)  GPU="${2:?--gpu needs a value}"; shift 2 ;;
        --render-gpu|--isaac-gpu) RENDER_GPU="${2:?$1 needs a value}"; shift 2 ;;
        --collect-gpu) COLLECT_GPU="${2:?--collect-gpu needs N, N,M or all}"; shift 2 ;;
        --lanes) LANES="${2:?--lanes needs a number or auto}"; shift 2 ;;
        --line) LINE="${2:?--line needs single_arm or bimanual}"; shift 2 ;;
        --from) FROM="${2:?--from needs collect|convert|train|eval}"; shift 2 ;;
        --to)   TO="${2:?--to needs collect|convert|train|eval}"; shift 2 ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) EXPERIMENT="$1"; shift ;;
    esac
done
[ -n "$EXPERIMENT" ] || { echo "usage: $(basename "$0") <experiment.yaml> [--gpu N] [--render-gpu N] [--collect-gpu N,M|all] [--lanes N|auto] [--line single_arm|bimanual] [--from STAGE] [--to STAGE]" >&2; exit 1; }
case "$LINE" in single_arm|bimanual) : ;; *) echo "--line must be single_arm or bimanual" >&2; exit 1 ;; esac

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RUN="$LOCOMANI_OUTPUTS/runs/$("$VENV_SIM_PY" -c "
import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$EXPERIMENT")"
echo "run directory  $RUN"
IMAGE_SOURCE=$("$VENV_SIM_PY" -c "
import sys; from locomani.pipeline import run_context
print(run_context.open_run(sys.argv[1], write=False)[0]['images']['source'])" "$EXPERIMENT") || exit 1
if [ "$IMAGE_SOURCE" != mujoco ] && [ -z "$RENDER_GPU" ]; then
    echo "images.source is $IMAGE_SOURCE: give the card it renders on, --render-gpu N" >&2; exit 1
fi
# a stage runs when it is inside [--from, --to]; the stages are ordered collect < convert < train < eval
rank() { case "$1" in collect) echo 0 ;; convert) echo 1 ;; train) echo 2 ;; eval) echo 3 ;; *) echo 9 ;; esac ; }
wanted() { [ "$(rank "$1")" -ge "$(rank "$FROM")" ] && [ "$(rank "$1")" -le "$(rank "$TO")" ] ; }

# --- collect -------------------------------------------------------------------------------
# Two terminals in the documented path: the stack in one, the agent server in the other. Here the
# stack is started in the background and the agent server runs in the foreground; the agent server
# takes the stack down when it finishes, and the trap covers the case where it never gets there.
collected() {
    "$VENV_SIM_PY" -c "
import json, pathlib, sys, yaml
summary = pathlib.Path(sys.argv[1]) / 'summary.json'
entry = json.loads(summary.read_text()).get('collect') if summary.exists() else None
wanted = yaml.safe_load(open(sys.argv[2]))['collect']['successes']
sys.exit(0 if entry and entry.get('kept', 0) >= wanted else 1)" "$RUN" "$EXPERIMENT"
}
if wanted collect && ! collected; then
    echo "=== collect ==="
    # The bimanual line collects in lanes spread over every card unless --collect-gpu says otherwise; the
    # single-arm line runs one stack, on --gpu.
    [ -n "$COLLECT_GPU" ] || { [ "$LINE" = bimanual ] && COLLECT_GPU=all || COLLECT_GPU="$GPU"; }
    bash "$LOCOMANI_ROOT/scripts/start_stack.sh" collect "$EXPERIMENT" --gpu "$COLLECT_GPU" ${LANES:+--lanes "$LANES"} || exit 1
    # every lane's instance (collect/lanes.json), else the one stack's
    INSTANCES=$("$VENV_SIM_PY" -c "
import json, pathlib, sys, yaml
lanes = pathlib.Path(sys.argv[1]) / 'collect' / 'lanes.json'
if lanes.exists():
    print(' '.join(str(lane['instance']) for lane in json.loads(lanes.read_text())['lanes']))
else:
    print(yaml.safe_load(open(sys.argv[2]))['collect']['instance'])" "$RUN" "$EXPERIMENT")
    trap 'for i in $INSTANCES; do bash "$LOCOMANI_ROOT/scripts/stop_stack.sh" --instance "$i" 2>/dev/null; done' EXIT
    bash "$LOCOMANI_ROOT/scripts/$LINE/run_agent_server.sh" "$EXPERIMENT" --gpu "$GPU" || exit 1
    trap - EXIT
fi

# --- convert -------------------------------------------------------------------------------
if wanted convert && [ ! -d "$RUN/train_dataset" ]; then
    echo "=== convert ==="
    # backend.render offline recorded black placeholder frames; this fills them (a live run is a no-op).
    # A policy evaluated on Isaac / Unreal images trains on that renderer's pictures of the same states.
    if [ "$IMAGE_SOURCE" = isaac ]; then
        bash "$LOCOMANI_ROOT/scripts/render_dataset_cameras_isaac.sh" "$RUN" --gpu "$RENDER_GPU" || exit 1
    elif [ "$IMAGE_SOURCE" = unreal ]; then
        bash "$LOCOMANI_ROOT/scripts/start_unreal.sh" --gpu "$RENDER_GPU" || exit 1
        bash "$LOCOMANI_ROOT/scripts/render_dataset_cameras_unreal.sh" "$RUN" || exit 1
    else
        bash "$LOCOMANI_ROOT/scripts/render_dataset_cameras.sh" "$RUN" --gpu "$GPU" || exit 1
    fi
    bash "$LOCOMANI_ROOT/scripts/convert_data.sh" "$RUN" || exit 1
fi

# --- train ---------------------------------------------------------------------------------
if wanted train && [ ! -d "$RUN/model" ]; then
    echo "=== train ==="
    bash "$LOCOMANI_ROOT/scripts/train.sh" "$RUN" --gpu "$GPU" || exit 1
fi

# --- evaluate ------------------------------------------------------------------------------
# Three terminals in the documented path: stack, policy server, rounds (a fourth, the Isaac or Unreal mirror,
# for images.source isaac / unreal). Here all but the rounds run in the background; the rounds driver waits
# for the policy port before the first round and takes the stack down at the end. The mirror has no
# port to wait on from there, so this waits for its "listening" line before the rounds start.
if wanted eval && [ ! -f "$RUN/eval/closed_loop.json" ]; then
    echo "=== evaluate ==="
    bash "$LOCOMANI_ROOT/scripts/start_stack.sh" eval "$RUN" --gpu "$GPU" || exit 1
    INSTANCE=$("$VENV_SIM_PY" -c "
import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['collect']['instance'])" "$RUN/config.yaml")
    bash "$LOCOMANI_ROOT/scripts/run_vla_server.sh" "$RUN" --gpu "$GPU" &
    SERVER=$!
    MIRROR=""
    # the rounds driver takes the stack down when it ends; this covers it never getting there
    trap 'kill "$SERVER" $MIRROR 2>/dev/null; bash "$LOCOMANI_ROOT/scripts/stop_stack.sh" --instance "$INSTANCE" 2>/dev/null' EXIT
    if [ "$IMAGE_SOURCE" != mujoco ]; then
        mkdir -p "$RUN/eval"
        MLOG="$RUN/eval/${IMAGE_SOURCE}_mirror.log"
        if [ "$IMAGE_SOURCE" = isaac ]; then
            bash "$LOCOMANI_ROOT/scripts/run_isaac_mirror.sh" "$RUN" --gpu "$RENDER_GPU" > "$MLOG" 2>&1 &
        else
            bash "$LOCOMANI_ROOT/scripts/start_unreal.sh" --gpu "$RENDER_GPU" || exit 1
            bash "$LOCOMANI_ROOT/scripts/run_unreal_mirror.sh" "$RUN" > "$MLOG" 2>&1 &
        fi
        MIRROR=$!
        for _ in $(seq 600); do
            grep -q "$IMAGE_SOURCE mirror listening" "$MLOG" && break
            kill -0 "$MIRROR" 2>/dev/null || { echo "the $IMAGE_SOURCE mirror exited; see $MLOG" >&2; exit 1; }
            sleep 1
        done
        grep -q "$IMAGE_SOURCE mirror listening" "$MLOG" || { echo "the $IMAGE_SOURCE mirror never listened" >&2; exit 1; }
    fi
    bash "$LOCOMANI_ROOT/scripts/$LINE/run_vla_inference.sh" "$RUN" --gpu "$GPU"
    status=$?
    kill "$SERVER" $MIRROR 2>/dev/null
    trap - EXIT
    [ "$status" = 0 ] || exit "$status"
fi

echo "done -> $RUN"
