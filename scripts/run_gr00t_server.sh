#!/bin/bash
# GR00T policy server for the real robot (src/real_robot/run_gr00t_server.py), in the Isaac-GR00T venv.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_gr00t_server.sh --model-path outputs/runs/<run>/checkpoint-<step>
#   bash scripts/run_gr00t_server.sh --model-path <checkpoint> --backbone-path /some/other/Cosmos-Reason2-2B
#   bash scripts/run_gr00t_server.sh --help
#
# Cosmos-Reason2-2B, the VLM backbone: a checkpoint records the backbone path training used, and one trained on
# the cluster records a directory that does not exist here. Unless --backbone-path is given, this script adds
#   --backbone-path "$COSMOS_BACKBONE_PATH"
# which paths.sh defaults to ~/Checkpoints/Cosmos-Reason2-2B; set COSMOS_BACKBONE_PATH in
# $LOCOMANI_WORKSPACE/locomani.local.sh (or export it) when the weights live elsewhere.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1
export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$LOCOMANI_WORKSPACE/huggingface_cache}"

COSMOS_BACKBONE_PATH="${COSMOS_BACKBONE_PATH:-$HOME/Checkpoints/Cosmos-Reason2-2B}"
backbone_args=()
case " $* " in
    *" --backbone-path "*|*" --backbone-path="*|*" --help "*|*" -h "*) ;;  # given explicitly, or just asking
    *)
        if [ -f "$COSMOS_BACKBONE_PATH/config.json" ]; then
            backbone_args=(--backbone-path "$COSMOS_BACKBONE_PATH")
            echo "[run_gr00t_server] --backbone-path $COSMOS_BACKBONE_PATH (COSMOS_BACKBONE_PATH; pass --backbone-path to override)"
        else
            echo "[run_gr00t_server] COSMOS_BACKBONE_PATH=$COSMOS_BACKBONE_PATH has no config.json; the checkpoint's own backbone record is used" >&2
        fi
        ;;
esac

exec "$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python" -m real_robot.run_gr00t_server "$@" "${backbone_args[@]}"
