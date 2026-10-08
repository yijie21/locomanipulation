#!/bin/bash
# The GR00T policy server for the real G1, in the Isaac-GR00T venv with this repository's vendored src/gr00t.
# Terminal 2 of three: 1 the robot's C++ deploy and camera server, 3 scripts/run_real_robot_inference.sh.
# Foreground: Ctrl-C frees the card. Same flags as NVIDIA's gr00t/eval/run_gr00t_server.py.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_gr00t_server.sh --model-path outputs/runs/<run>/checkpoint-XXXX
#   bash scripts/run_gr00t_server.sh --help
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1
export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$LOCOMANI_WORKSPACE/huggingface_cache}"
exec "$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python" -m real_robot.run_gr00t_server "$@"
