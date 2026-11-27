#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# selfmodel_hadata_20261126 revision 2c: evaluation 3 only (evaluations 1 and 2 do not read the real episodes and are not
# rerun), from the frozen eval_code2 (eval_code with hd_real.py's merge fix):
#   hd_eval3.sh <collection root> <controller>
# hd_eval.sh's environment, checks and eval 3 lines unchanged, except the code dir (eval_code2) and its own phase names:
#   code     eval_code2's MD5, asrun_train's MD5.check; the training's STATUS "done"; $ROOT/real/DONE
#   eval 3   hd_real.py on $ROOT/real/<ctrl>_* -> $EV/real/<dir>/ (merged), eval3.{txt,json}
#   done
# Phases in $EV/STATUS and $EV/run.log (prefixed "eval_code2"); any failure stops the run, STATUS "FAIL ...".
set -uo pipefail
ROOT=$1 C=$2
Y=/wuji-vefps-D/wuji-il/yijie; L=$Y/loco_outputs; WS=$Y/ws
OH=$L/selfmodel_hadata_20261126
EC=$OH/eval_code2
E=$EC/selfmodel_hacollect_20261126
AT=$OH/asrun_train
R=$L/selfmodel_simpledata_20261126/asrun/repo_b
SDR=$L/selfmodel_simpledata_20261126/asrun/repo/scripts/experiments/selfmodel_simpledata_20261126/sd_range.py
HSS=$OH/asrun_real/selfmodel_hacollect_20261126/hr_smoke_stats.py
TOUCH=$AT/selfmodel_hacollect_20261126/hd_touch.py
O=$ROOT/train/$C
EV=$O/eval
PY="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1 LOCO_REPO=$R SX_REPO=$R
export OMP_NUM_THREADS=${HE_THREADS:-4} MKL_NUM_THREADS=${HE_THREADS:-4} REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1
unset REACHWM_X0_REF REACHWM_SUBSET REACHWM_TRAIN_HELD REACHWM_ROOT
export CUDA_VISIBLE_DEVICES=${HE_GPU:-}
NI=${HE_NICE:-10}
mkdir -p $EV/logs
phase() { echo "$1" > $EV/STATUS; echo "$(date -u +%FT%TZ) $1" | tee -a $EV/run.log; }
die() { phase "FAIL $1"; exit 1; }
run() {  # run <name> <script> args...: <name>.raw, <name>.txt (without the warnings)
  local n=$1; shift
  nice -n $NI $PY "$@" > $EV/logs/$n.raw 2>&1 || { tail -n 5 $EV/logs/$n.raw >> $EV/run.log; return 1; }
  grep -v Warn $EV/logs/$n.raw | grep -v "^skip " > $EV/$n.txt
}
cd $R || die "no $R"
(cd $EC && md5sum -c --quiet MD5) || die "eval_code2 md5"
(cd $AT && md5sum -c --quiet MD5.check) || die "asrun_train md5"
phase "code: eval_code2 $EC ($(cat $EC/FROZEN_UTC 2>/dev/null)); eval 3 only; $C on $(hostname), CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'; train $O; out $EV"
[ "$(cat $O/STATUS 2>/dev/null)" = done ] || die "training not done: $(cat $O/STATUS 2>/dev/null)"
[ -e $EV/real ] && die "$EV/real exists (move the earlier eval 3 aside first)"
phase "eval 3 (eval_code2)"
[ -f $ROOT/real/DONE ] || die "real episodes not done ($ROOT/real/DONE)"
REAL=$(ls -d $ROOT/real/${C}_*/ | sed 's:/$::' | paste -sd, -)
[ -n "$REAL" ] || die "no $ROOT/real/${C}_* directories"
echo "real directories: $REAL" >> $EV/run.log
run eval3 $E/hd_real.py --root $O --real $REAL --out $EV/real --sd-range $SDR --light $HSS --touch $TOUCH --json $EV/eval3.json \
  || die "eval 3"
grep "^REAL MERGE" $EV/eval3.txt | tee -a $EV/run.log
phase "done"
