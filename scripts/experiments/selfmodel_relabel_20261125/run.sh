#!/usr/bin/env bash
# notebook: selfmodel_relabel_20261125
# selfmodel_relabel_20261125: data (mid_data.py), split check, models A, B, C (train.py E3_ensemble, the recipe as is:
# 30k iterations, batch 1024, 5 members, seed 0, REACHWM_GOAL_REF=1) on GPUs 1-3, then eval_entry.py. Exploratory, not
# registered: model Am (C plus the mid starts at which the reference still runs), same recipe, sharing C's GPU.
#   run.sh            the formal run, code from $O/asrun (checked against the repo's reachwm/selfwm files by md5)
#   SMOKE=1 run.sh    300 resets a dataset, 200 iterations, into $O/smoke, code from this script's directory
set -uo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_relabel_20261125
SRC=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_selfmodel_20261119
cd /wuji-vefps-D/wuji-il/yijie/ws/code/loco
export TMPDIR=/workspace/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 REACHWM_GOAL_REF=1
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
R=scripts/experiments/reachwm_20261030
G=(${GPUS:-1 2 3})
if [ "${SMOKE:-0}" = 1 ]; then
  X=$O/smoke; CODE=$(cd "$(dirname "$0")" && pwd); MAXR="--max-resets 300"; IT="--iters 200"
  BS=$X/data/B_graspr_odo.npz,$X/data/B_graspr_wrist_odo.npz
else
  X=$O; CODE=$O/asrun; MAXR=""; IT=""
  BS=$SRC/data_graspr_odo/all.npz,$SRC/data_graspr_wrist_odo/all.npz
fi
mkdir -p $X/data
ST=$X/STATUS
phase() { echo "$1" > "$ST"; echo "[$(date -u +%FT%TZ)] $1"; }
die() { phase "FAIL $1"; exit 1; }
spec() { for k in ${1//,/ }; do printf "$X/data/$k.npz,"; done | sed 's/,$//'; }

if [ "${SMOKE:-0}" != 1 ]; then
  (cd $CODE && md5sum -c --quiet MD5) || die "asrun md5"
  for f in reachwm_20261030/{data,train,models}.py selfwm_20261029/{env,wm,fk}.py; do
    cmp -s scripts/experiments/$f $CODE/$f || die "repo $f differs from asrun"
  done
  echo "code: asrun md5 ok, repo reachwm/selfwm files identical to asrun; repo HEAD $(git log -1 --format=%h)"
fi
phase "data"
$PY $CODE/mid_data.py --out $X/data $MAXR > $X/mid_data.log 2>&1 || die "mid_data"
cat $X/mid_data.log
AS=$(spec C_graspr_odo,C_graspr_wrist_odo,mid_graspr_odo,mid_graspr_wrist_odo)
CS=$(spec C_graspr_odo,C_graspr_wrist_odo)
AMS=$(spec C_graspr_odo,C_graspr_wrist_odo,midmove_graspr_odo,midmove_graspr_wrist_odo)
phase "split check"
$PY $CODE/split_check.py --A "$AS" --B "$BS" --C "$CS" --Am "$AMS" > $X/split_check.log 2>&1
rc=$?
cat $X/split_check.log
[ $rc = 0 ] || die "split check"
phase "train"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
CUDA_VISIBLE_DEVICES=${G[0]} $PY $R/train.py E3_ensemble $IT --data "$AS" --out-root $X/models_A > $X/train_A.log 2>&1 &
pa=$!
CUDA_VISIBLE_DEVICES=${G[1]} $PY $R/train.py E3_ensemble $IT --data "$BS" --out-root $X/models_B > $X/train_B.log 2>&1 &
pb=$!
CUDA_VISIBLE_DEVICES=${G[2]} $PY $R/train.py E3_ensemble $IT --data "$CS" --out-root $X/models_C > $X/train_C.log 2>&1 &
pc=$!
CUDA_VISIBLE_DEVICES=${G[2]} $PY $R/train.py E3_ensemble $IT --data "$AMS" --out-root $X/models_Am > $X/train_Am.log 2>&1 &
pm=$!
echo "train pids A $pa B $pb C $pc Am $pm" | tee $X/pids.txt
wait $pa || die "train A"
wait $pb || die "train B"
wait $pc || die "train C"
wait $pm || die "train Am"
for m in A B C Am; do tail -n 3 $X/train_$m.log; done
phase "eval"
CUDA_VISIBLE_DEVICES=${G[0]} $PY $CODE/eval_entry.py --data "$BS" --A $X/models_A --B $X/models_B --C $X/models_C \
  --Am $X/models_Am --json $X/eval.json > $X/eval.log 2>&1 || die "eval"
cat $X/eval.log
phase "done"
