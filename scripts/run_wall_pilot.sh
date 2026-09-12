#!/bin/bash
# The WALL-TOP crate grasp pilot. Kept as its own name because docs and habit
# point here; the runner is scripts/run_crate_pilot.sh and the driver is
# src/locomani/pipeline/run_crate_pilot.py (which owns the stack, the run
# directory and the stop -- see that file for why this one is three lines).
#
#   bash scripts/run_wall_pilot.sh [rounds] [seed] [extra args to run_crate_pilot.py]
#   OUT=<dir> INSTANCE=2 bash scripts/run_wall_pilot.sh 10 0 --config configs/tasks/crate_wall.json
exec env CRATE_EXPERT=wall bash "$(dirname "$0")/run_crate_pilot.sh" "$@"
