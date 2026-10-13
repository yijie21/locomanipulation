#!/bin/bash
# W0 admission batch (TODO 25.1): stand x10, walk 3 speeds x10, bend {33, 18 cm} x modes {2, 1} x10, ScaleBFM on
# the mars warehouse scenes. Minutes on one GPU, no video. Usage: bash w0_batch.sh [outdir]
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=${1:-outputs/controller_compare_20261013/w0}
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
PY=third_party/Isaac-GR00T/.venv/bin/python
W0=scripts/experiments/controller_compare_20261013/w0_admission.py
$PY $W0 --test stand --out "$OUT/stand"
$PY $W0 --test walk --out "$OUT/walk"
for box in 33 18; do
  for mode in 2 1; do
    $PY $W0 --test bend --box $box --mode $mode --out "$OUT/bend_${box}_m${mode}"
  done
done
echo "=== W0 DONE"
