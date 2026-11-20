#!/usr/bin/env bash
# The install check: the pipeline on the examples (simulated head-camera looks), compared with what the benchmark got.
#   bash run_examples.sh [OUT]        (default OUT: out/examples)
# 1. the looks' own depth (--depth raw): must reproduce the benchmark's masks, points and table heights
# 2. their stereo pairs (--depth stereo): the depth worked out here, as the benchmark did
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/setup/env.sh"
out=${1:-$here/out/examples}
bash "$here/run.sh" "$out/raw" "$here"/examples/looks/*.npz -- --depth raw
bash "$here/run.sh" "$out/stereo" "$here"/examples/looks/*.npz -- --depth stereo
"$GRASP_PY" "$here/tools/check_examples.py" "$here/examples" "$out/raw" "$out/stereo"
