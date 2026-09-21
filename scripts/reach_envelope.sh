#!/bin/bash
# Measure the bimanual reach envelope: bracket the x boundary of every (y, h, d, yaw) line, round by round.
#
#   bash scripts/reach_envelope.sh outputs/reach_envelope/<name>            # rounds until every line is bracketed
#   bash scripts/reach_envelope.sh outputs/reach_envelope/<name> --from 3   # resume at round 3
#
# <run_dir>/search.json (written with defaults on first use) fixes the grid, the whole-body envelope,
# the floor scene and the shard list; edit it before the first round, never between rounds. Each
# round plans one bank from the previous results, runs every shard in parallel (one SONIC stack per
# shard: its own GPU and instance), and stops when the planner has nothing left. Touch
# <run_dir>/STOP to halt between rounds. Afterwards:
#   python -m locomani.tasks.crate.reach_envelope.boundary_search table <run_dir>
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run_dir> [--from N]" >&2; exit 1; }
run_dir=$1; shift
n=0
[ "${1:-}" = "--from" ] && n=$2

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
search=locomani.tasks.crate.reach_envelope.boundary_search
while :; do
  [ -f "$run_dir/STOP" ] && { echo "STOP before round $n"; break; }
  rd=$run_dir/round_$n
  [ -f "$rd/bank.json" ] || "$VENV_SIM_PY" -m $search plan "$run_dir" "$n"
  [ -f "$rd/bank.json" ] || { echo "round $n: nothing planned"; break; }
  shards=$(ls -d "$rd"/shard_* | wc -l)
  echo "=== round $n: $(python3 -c "import json;print(len(json.load(open('$rd/bank.json'))['samples']))") trials over $shards shards $(date -u +%H:%M:%S)"
  pids=()
  for ((s = 0; s < shards; s++)); do
    "$VENV_SIM_PY" scripts/probe_ghost_reach.py "$rd/shard_$s/request.json" --output "$rd/shard_$s/result.json" \
      --shard-index "$s" --shard-count "$shards" > "$rd/shard_$s/launch.log" 2>&1 &
    pids+=($!)
    sleep 20 # stagger stack bring-up
  done
  for p in "${pids[@]}"; do wait "$p" || echo "shard exited $?"; done
  echo "=== round $n done $(date -u +%H:%M:%S)"
  n=$((n + 1))
done
"$VENV_SIM_PY" -m $search status "$run_dir" | tail -1
