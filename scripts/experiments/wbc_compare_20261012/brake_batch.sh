#!/usr/bin/env bash
# SONIC vs ScaleBFM: walk to a finish line on the flat walk scene and stop (TODO 24.7).
# ScaleBFM first (GPU 1, deterministic sim, minutes), then SONIC's stack alone on instance 0 / GPU 0 (wall-clock:
# nothing runs beside it), then the marker renders and the merged table + side-by-side videos.
#   bash scripts/experiments/wbc_compare_20261012/brake_batch.sh [out dir]    (default outputs/wbc_compare_20261012/brake)
# Touch <run dir>/STOP to end a run after its current trial.
set -uo pipefail
cd /workspace/code/locomanipulation
export TMPDIR=/workspace/tmp
O="${1:-outputs/wbc_compare_20261012/brake}"
mkdir -p "$O"
D=scripts/experiments/wbc_compare_20261012
BFM_PY=third_party/Isaac-GR00T/.venv/bin/python
SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
SPEEDS="${SPEEDS:-0.55,0.35,0.25}"
REPEATS="${REPEATS:-10}"
MARKER="${MARKER:-1.5}"

stamp() { date -u +%H:%M:%S; }

for prog in marker plan; do
  echo "===== $(stamp) scalebfm $prog"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=2 "$BFM_PY" "$D/bfm_brake.py" --program "$prog" \
    --speeds "$SPEEDS" --repeats "$REPEATS" --marker "$MARKER" --out "$O/bfm_$prog" > "$O/bfm_$prog.log" 2>&1
  echo "bfm $prog exit $?"
  MUJOCO_GL=egl "$SIM_PY" scripts/experiments/render_brake_marker_20260923.py "$O/bfm_$prog" > "$O/bfm_$prog/render.log" 2>&1
  echo "render exit $?"
done

# SONIC: the 2026-09-23 diagnostic as is (brake_marker_20260923.sh brings the stack up on instance 0, runs
# diag_brake, brings it down, renders). guard = extra 0.20 / margin 0.03, the setting validated 50/50 on 09-23.
run_sonic() {
  local name=$1; shift
  echo "===== $(stamp) sonic $name"
  BRAKE_OUT="$PWD/$O/sonic_$name" bash scripts/experiments/brake_marker_20260923.sh "$@" > "$O/sonic_$name.log" 2>&1
  echo "sonic $name exit $?  $(tail -2 "$O/sonic_$name.log" | tr '\n' ' ')"
}
run_sonic marker --speeds "$SPEEDS" --repeats "$REPEATS" --marker "$MARKER"
run_sonic guard --guard --extra 0.20 --margin 0.03 --speeds "$SPEEDS" --repeats "$REPEATS" --marker "$MARKER"
run_sonic guard_final020 --guard --extra 0.20 --margin 0.03 --speeds 0.55 --final-speed 0.20 --final-distance 0.3 \
  --repeats "$REPEATS" --marker "$MARKER"

echo "===== $(stamp) summary"
"$SIM_PY" "$D/brake_summary.py" --out "$O" > "$O/summary.log" 2>&1
echo "summary exit $?"
echo "===== $(stamp) done"
