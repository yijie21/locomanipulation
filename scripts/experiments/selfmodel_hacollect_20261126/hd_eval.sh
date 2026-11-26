#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# selfmodel_hadata_20261126 revision 2b, section 3: one controller's three evaluations, from the frozen eval_code (read-only
# on the data and the models; everything written under $O/eval):
#   hd_eval.sh <collection root> <controller>
# (<collection root>: selfmodel_hadata_20261126 for scalebfm / heft, selfmodel_hadata_sonic_20261126 for sonic.)
# Reader and models: simpledata's frozen asrun/repo_b (data.py 174071cb, train.py 8acdaea1, models.py 0ac27f6b) through its
# py4090b.py, REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1, X0_REF off; CPU unless HE_GPU is set. Phases ($EV/STATUS,
# $EV/run.log; any failure stops the run, STATUS "FAIL ..."):
#   code     eval_code's MD5 (its scripts and every file they import or call by path), asrun_train's MD5.check
#   wait     $O/STATUS "done" (hd_train.sh's last phase)
#   eval 1   hd_eval.py --split val, then test_iid -> eval1_<split>.{txt,json}
#   eval 2   the xfer HA record (selfmodel_xfer_20261125/<ctrl>_ha) touch-filtered by hd_touch.py -> $EV/xfer/data/all.npz
#            (+ its run.json); $EV/lml/data/all.npz -> $O/data/d3.npz (the HA d3, touch-filtered); the frozen sx_eval.py
#            (81f61381) with --keep settled and ended for S_f1_s0..2, T_f1_s0..2 and F (fdvel <ctrl>/models_s0) ->
#            eval2_<model>_<keep>.{txt,json}; hd_xfer_pair.py (1 - model / F, paired, checked against those jsons)
#            -> eval2_pair.{txt,json}
#   eval 3   $ROOT/real/DONE, then hd_real.py on $ROOT/real/<ctrl>_* (seeds 200-239) -> $EV/real/<dir>/ (merged), eval3.{txt,json}
#   done
# Test only: HE_TEST=1 (no STATUS / DONE wait, no md5 checks) with HE_CODE (code dir), HE_TRAIN (the train dir), HE_OUT.
set -uo pipefail
ROOT=$1 C=$2
Y=/wuji-vefps-D/wuji-il/yijie; L=$Y/loco_outputs; WS=$Y/ws
OH=$L/selfmodel_hadata_20261126
EC=${HE_CODE:-$OH/eval_code}
E=$EC/selfmodel_hacollect_20261126
AT=$OH/asrun_train
R=$L/selfmodel_simpledata_20261126/asrun/repo_b
SX=$L/selfmodel_xfer_20261125/asrun_ha/selfmodel_xfer_20261125/sx_eval.py
SDR=$L/selfmodel_simpledata_20261126/asrun/repo/scripts/experiments/selfmodel_simpledata_20261126/sd_range.py
HSS=$OH/asrun_real/selfmodel_hacollect_20261126/hr_smoke_stats.py
TOUCH=$AT/selfmodel_hacollect_20261126/hd_touch.py
FD=$L/selfmodel_fdvel_20261126/$C/models_s0
XREC=$L/selfmodel_xfer_20261125/${C}_ha
O=${HE_TRAIN:-$ROOT/train/$C}
EV=${HE_OUT:-$O/eval}
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
if [ -z "${HE_TEST:-}" ]; then
  (cd $EC && md5sum -c --quiet MD5) || die "eval_code md5"
  (cd $AT && md5sum -c --quiet MD5.check) || die "asrun_train md5"
fi
phase "code: eval_code $EC ($(cat $EC/FROZEN_UTC 2>/dev/null)); $C on $(hostname), CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'; train $O; out $EV"
if [ -z "${HE_TEST:-}" ]; then
  [ "$(cat $O/STATUS 2>/dev/null)" = done ] || die "training not done: $(cat $O/STATUS 2>/dev/null)"
fi
md5sum $O/models_*/E3_ensemble_f*/model.pt $FD/E3_ensemble_f1/model.pt >> $EV/run.log
phase "eval 1"
for s in val test_iid; do
  run eval1_$s $E/hd_eval.py --root $O --split $s --json $EV/eval1_$s.json || die "eval 1 $s"
done
phase "eval 2: the xfer record, touch filter"
mkdir -p $EV/lml/data $EV/xfer/data
ln -sfn $O/data/d3.npz $EV/lml/data/all.npz
nice -n $NI $PY $TOUCH $XREC $EV/xfer/data/all.npz > $EV/logs/xfer_touch.log 2>&1 || die "xfer touch filter"
grep -q "TOUCH FILTER PASS" $EV/logs/xfer_touch.log || die "xfer touch filter: no PASS"
grep -v Warn $EV/logs/xfer_touch.log | grep -v "^skip " >> $EV/run.log
cp $XREC/run.json $EV/xfer/run.json
phase "eval 2: sx_eval"
for m in S_f1_s0 S_f1_s1 S_f1_s2 T_f1_s0 T_f1_s1 T_f1_s2 F; do
  if [ $m = F ]; then md=$FD; else md=$O/models_${m%%_*}_${m##*_}; fi
  for k in settled ended; do
    run eval2_${m}_$k $SX --lml $EV/lml --xfer $EV/xfer --models $md --keep $k --json $EV/eval2_${m}_$k.json || die "eval 2 $m $k"
  done
done
phase "eval 2: pair"
run eval2_pair $E/hd_xfer_pair.py --sx $SX --lml $EV/lml --xfer $EV/xfer --root $O --F $FD --check $EV --json $EV/eval2_pair.json \
  || die "eval 2 pair"
grep "equal sx_eval" $EV/eval2_pair.txt | tee -a $EV/run.log
grep -q "equal sx_eval's json: False" $EV/eval2_pair.txt && die "eval 2 pair differs from sx_eval"
phase "eval 3"
if [ -z "${HE_TEST:-}" ]; then
  [ -f $ROOT/real/DONE ] || die "real episodes not done ($ROOT/real/DONE)"
fi
REAL=$(ls -d $ROOT/real/${C}_*/ | sed 's:/$::' | paste -sd, -)
[ -n "$REAL" ] || die "no $ROOT/real/${C}_* directories"
echo "real directories: $REAL" >> $EV/run.log
run eval3 $E/hd_real.py --root $O --real $REAL --out $EV/real --sd-range $SDR --light $HSS --touch $TOUCH --json $EV/eval3.json \
  || die "eval 3"
grep "^REAL MERGE" $EV/eval3.txt | tee -a $EV/run.log
phase "done"
