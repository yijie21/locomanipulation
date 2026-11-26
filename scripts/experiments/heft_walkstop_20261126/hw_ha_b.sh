#!/usr/bin/env bash
# heft_walkstop_20261126 revision 2 on 4090b (no /workspace): hw_ha.sh from code_b ($O/hw_mk_code_b.sh: the round's copy
# with its own outputs/ holding hadata's frozen scenes spelled on vePFS, as hadata's dev/b4090b/repo_b), checked against
# asrun/CODE_MD5_B first; LOCOMANI_WORKSPACE as hadata's hd_job_b.sh (the 4090's /workspace target). Lives at $O/, outside
# code_b, so code_b stays as recorded.
#   hw_ha_b.sh <gpu> --controller heft|sonic --out DIR [--arms ... --cells ... --reps a:b --traces | --probe]
set -euo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/heft_walkstop_20261126
B=$O/code_b
(cd $B && md5sum -c --quiet $O/asrun/CODE_MD5_B) || { echo "code_b changed since CODE_MD5_B"; exit 2; }
export LC_CODE_ROOT=$B LOCOMANI_WORKSPACE=/wuji-vefps-D/wuji-il/yijie/ws
exec bash $B/scripts/experiments/heft_walkstop_20261126/hw_ha.sh "$@"
