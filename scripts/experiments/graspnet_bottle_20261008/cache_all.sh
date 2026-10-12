#!/bin/bash
# TODO 23.2: build the offline grasp cache (grasp_cache.py) for every generated object scene, N processes at a time.
#   bash scripts/experiments/graspnet_bottle_20261008/cache_all.sh [lanes 12] [grasp_cache.py flags]
# Log: outputs/graspnet_bottle_20261008/logs/cache_obj_XXX.log; summary per scene in grasp_cache_summary.json.
set -u
cd "$(dirname "$0")/../../.." || exit 1
LANES=${1:-12}
EXTRA=("${@:2}")
ROOT=outputs/graspnet_bottle_20261008
mkdir -p "$ROOT/logs"
export PYTHONPATH="$PWD/src" TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
python3 -c "import json;print('\n'.join(str(o['id']) for o in json.load(open('$ROOT/objects.json'))['objects']))" |
    xargs -P "$LANES" -I{} sh -c "$PY -u scripts/experiments/graspnet_bottle_20261008/grasp_cache.py {} ${EXTRA[*]} > $ROOT/logs/cache_obj_\$(printf %03d {}).log 2>&1; tail -1 $ROOT/logs/cache_obj_\$(printf %03d {}).log"
echo "cache_all finished $(date +%T)"
