#!/bin/bash
# The GR00T + SONIC inference client for the real G1, in SONIC's own venv (gear_sonic: camera client, robot model,
# protocol-v4 packing). Terminal 3 of three: 1 the robot's C++ deploy and camera server, 2 scripts/run_vla_server.sh.
#
#   bash scripts/run_real_robot_inference.sh --host <policy server> --init-token-file configs/deployment/warehouse_initial_token.json
#   bash scripts/run_real_robot_inference.sh --help
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1
export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m real_robot.inference "$@"
