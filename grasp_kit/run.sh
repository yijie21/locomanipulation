#!/usr/bin/env bash
# The whole pipeline on one or more looks (g1grasp/look.py):
#   bash run.sh OUT LOOK.npz [LOOK.npz ...] [-- perceive.py options, e.g. --object "red cup" --depth raw]
# -> OUT/<look>.grasp.json (the grasp), OUT/<look>.check.png (to check it by eye); README.md says how to read them.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/setup/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}  # the weights are local (setup/download_weights.sh)
[ $# -ge 2 ] || { sed -n 2,4p "$0"; exit 1; }
out=$1
shift
looks=()
opts=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; opts=("$@"); break; fi
  looks+=("$1")
  shift
done
"$PERCEPTION_PY" "$here/perceive.py" "${looks[@]}" --out "$out" ${opts[@]+"${opts[@]}"}
targets=()
for f in "${looks[@]}"; do targets+=("$out/$(basename "$f" .npz).target.npz"); done
"$GRASP_PY" "$here/plan_grasp.py" "${targets[@]}"
