#!/bin/bash
# The bimanual CARRY pilot (bimanual_carry.py). Same three-line shim as
# run_wall_pilot.sh: scripts/run_crate_pilot.sh is the runner,
# src/locomani/pipeline/run_crate_pilot.py the driver.
#
#   bash scripts/run_bimanual_pilot.sh [rounds] [seed] [extra args to run_crate_pilot.py]
exec env CRATE_EXPERT=carry bash "$(dirname "$0")/run_crate_pilot.sh" "$@"
