#!/usr/bin/env bash
# TODO 25.9.1 phase 0 exit checks (plan §9): the runner reproduces 25.7 round by round, its rounds do not depend on the
# process / shard / order, and a fork with an empty delta is the unforked round. ScaleBFM on GPU 1; no SONIC stack up.
# 25.7's reference runs are cold: the first round of each of its processes ran the network's first, unfused call
# (bfm_round.warm_up), so those rounds (L1 s7000_00, L2 s47000_00) are expected to differ; every other round must not.
#   bash scripts/experiments/scalebfm_warehouse_rsi_20261014/phase0_checks.sh [log]
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1
H=scripts/experiments/scalebfm_warehouse_rsi_20261014
O=outputs/scalebfm_warehouse_rsi_20261014
C=$O/phase0
L=${1:-$C/checks.log}
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
POOL=$O/pool_diag/states
L1=sonic_s7000_00,sonic_s7000_01,sonic_s7000_02,sonic_s7000_03,sonic_s7000_04,sonic_s7000_05,sonic_s7000_06,sonic_s7000_07,sonic_s7000_08,sonic_s7000_09
mkdir -p $C
if pgrep -f 'g1_deploy_onnx_ref|run_agent_server' >/dev/null; then echo "a SONIC stack is up" >> $L; exit 1; fi
HASH=$($PY -c "import sys; sys.path.insert(0, '$H'); from bfm_round import load_spec, spec_hash; print(spec_hash(load_spec('$H/incumbent.yaml')))")
INC=$O/diag/incumbent_$HASH
shards() {  # out, n, extra args...
  local out=$1 n=$2; shift 2
  for i in $(seq 0 $((n - 1))); do $PY $H/bfm_round.py $POOL --spec $H/incumbent.yaml --out $out --shard $i/$n "$@" >> $L 2>&1 & done
  wait
}
echo "=== A incumbent on the diagnostic pool (4 shards) $(date -u)" >> $L
shards $INC 4
echo "=== B vs 25.7 (L1, gain 0.5) and L2 (40 rounds) $(date -u)" >> $L
$PY $H/determinism.py $O/reference/m2_g05_l1 $INC --out $C/b_l1_g05.json >> $L 2>&1; echo "rc $?" >> $L
$PY $H/determinism.py outputs/controller_compare_20261013/l2/warehouse/bfm_nominal $INC --out $C/b_l2.json >> $L 2>&1; echo "rc $?" >> $L
echo "=== C gain 1.0 on L1 vs 25.7 $(date -u)" >> $L
printf 'mode: 2\nstance: near\nset:\n  visual.tracking_gain: 1.0\n' > $C/g10.yaml
for i in 0 1 2 3; do $PY $H/bfm_round.py $POOL --spec $C/g10.yaml --only $L1 --out $C/c_g10 --shard $i/4 >> $L 2>&1 & done; wait
$PY $H/determinism.py $O/reference/m2_g10_l1 $C/c_g10 --out $C/c_g10.json >> $L 2>&1; echo "rc $?" >> $L
echo "=== D other histories (L1): one process, and 3 shards (each round follows different rounds) vs A $(date -u)" >> $L
$PY $H/bfm_round.py $POOL --spec $H/incumbent.yaml --only $L1 --out $C/d_one >> $L 2>&1 &
for i in 0 1 2; do $PY $H/bfm_round.py $POOL --spec $H/incumbent.yaml --only $L1 --out $C/d_three --shard $i/3 >> $L 2>&1 & done; wait
$PY $H/determinism.py $INC $C/d_one --out $C/d_one.json >> $L 2>&1; echo "rc one $?" >> $L
$PY $H/determinism.py $INC $C/d_three --out $C/d_three.json >> $L 2>&1; echo "rc three $?" >> $L
echo "=== E empty forks at walk / ready / reach (L1) vs A $(date -u)" >> $L
for ph in walk ready reach; do $PY $H/bfm_round.py $POOL --spec $H/incumbent.yaml --fork "$ph:{}" --only $L1 --out $C/e_$ph >> $L 2>&1 & done; wait
for ph in walk ready reach; do $PY $H/determinism.py $INC $C/e_$ph --out $C/e_$ph.json >> $L 2>&1; echo "rc $ph $?" >> $L; done
echo "=== F a real fork at reach (gain 1.0) vs A: same prefix, different after $(date -u)" >> $L
$PY $H/bfm_round.py $POOL --spec $H/incumbent.yaml --fork "reach:{set: {visual.tracking_gain: 1.0}}" --only $L1 --out $C/f_reach_g10 >> $L 2>&1
$PY - >> $L 2>&1 <<PYEOF
import json, numpy as np, glob, os
bad = 0
for f in sorted(glob.glob("$C/f_reach_g10/*.json")):
    a, b = json.load(open(f)), json.load(open("$INC/" + os.path.basename(f)))
    za, zb = np.load(f[:-5] + ".npz"), np.load("$INC/" + os.path.basename(f)[:-5] + ".npz")
    k = int(np.searchsorted(za["t"], a["events"]["fork"]))
    same_prefix = np.array_equal(za["qpos"][:k], zb["qpos"][:k])
    n = min(len(za["qpos"]), len(zb["qpos"]))
    differs_after = not np.array_equal(za["qpos"][k:n], zb["qpos"][k:n]) or len(za["qpos"]) != len(zb["qpos"])
    bad += not (same_prefix and differs_after)
    print(os.path.basename(f), "fork t", a["events"]["fork"], "prefix same", same_prefix, "after differs", differs_after, a["expert_reason"], b["expert_reason"])
print("F bad", bad)
PYEOF
echo "=== G handoff records + ScaleBFM envelope $(date -u)" >> $L
$PY $H/handoff.py $POOL $INC --out $INC/records.jsonl >> $L 2>&1
$PY $H/envelope.py --name scalebfm_m22200_mode2_incumbent $INC --out $O/envelope/scalebfm_m22200_mode2_incumbent.json >> $L 2>&1
$PY $H/envelope.py --name sonic_v1_1 $POOL --out $O/envelope/sonic_v1_1.json >> $L 2>&1
$PY $H/envelope.py --name sonic_v1_1_ok --ok-only $POOL --out $O/envelope/sonic_v1_1_ok.json >> $L 2>&1
echo "=== PHASE0 CHECKS DONE $(date -u)" >> $L
