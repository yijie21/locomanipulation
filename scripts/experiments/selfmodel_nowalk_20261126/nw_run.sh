#!/usr/bin/env bash
# notebook: selfmodel_nowalk_20261126
# selfmodel_nowalk_20261126: the formal trainings and evaluations on the H20 machine (ssh -p 27840), from the frozen copy
# $P/asrun (this experiment's scripts) and simpledata's asrun/repo_b (data.py, train.py, models.py) with train3b's frozen
# t3b_train.py, launched as on 4090b (py4090b.py, LOCOMANI_WORKSPACE, SONIC_POLICY).
#   nw_run.sh                      in tmux nowalk; $OUT/run.log, $OUT/STATUS ("FAIL ..." stops the run)
# Phases:
#   code    md5 of $P/asrun/MD5.check, splits.npz (its md5 in MD5.check), t3b_train.py c7ecf581, the shared code
#   train   JOBS (arm seed gpu): N 0, C 0, C2 0, N 1, C 1 on cards 0 / 1, at most 2 of ours per card, a launch only
#           when the card's used memory + 12 GB <= 80 GB (nvidia-smi; the user's H20 cap); waits otherwise.
#           Each: nw_train.py --arm A -- E3_ensemble --seed S --frac 1 --data <D's spec> --out-root $OUT/models_<A>_s<S>,
#           REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1, nice 10, 4 CPU threads.
#   eval    on CPU: nw_eval.py arms D (simpledata's model, trained on 4090b), N, C, C2 (seed 0) -> $OUT/eval_s0;
#           arms D, N, C (seed 1) -> $OUT/eval_s1; nw_range.py arm N seed 0 -> $OUT/range_N.{txt,json}
# Test only: NW_TEST=1 with NW_OUT, NW_ITERS, NW_JOBS ("A:S:G ..."; G "cpu" = no card), NW_CODE (scripts dir), no md5.
# The evaluations' reproduction check (arm D, simpledata's model) runs in tests too.
set -uo pipefail
P=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126
R=$O/asrun/repo_b
SDC=$O/asrun/repo/scripts/experiments/selfmodel_simpledata_20261126
T3B=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125/asrun/selfmodel_train3b_20261125
WS=/wuji-vefps-D/wuji-il/yijie/ws
CODE=${NW_CODE:-$P/asrun/code}
OUT=${NW_OUT:-$P/run}
SPL=$P/splits/splits.npz
DS=$O/data3/data/all.npz,$O/data3b/data/all.npz,$O/train/data/mid_data3.npz,$O/train/data/mid_data3b.npz
JOBS=(${NW_JOBS:-N:0:0 C:0:1 C2:0:0 N:1:1 C:1:0})
IT=${NW_ITERS:+--iters $NW_ITERS}
export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=$R SX_REPO=$R
unset REACHWM_X0_REF
PY="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
mkdir -p $OUT/logs
cd $R || exit 1
phase() { echo "$1" > $OUT/STATUS; echo "$(date -u +%FT%TZ) $1" | tee -a $OUT/run.log; }
die() { phase "FAIL $1"; exit 1; }
ours_on() { local n=0 p; for p in "${!GPU_OF[@]}"; do [ "${GPU_OF[$p]}" = "$1" ] && kill -0 $p 2>/dev/null && n=$((n + 1)); done; echo $n; }
if [ -z "${NW_TEST:-}" ]; then
  (cd $P/asrun && md5sum -c --quiet MD5.check) || die "asrun md5"
  (cd $O/asrun && md5sum -c --quiet MD5.check && md5sum -c --quiet MD5_r2.check) || die "simpledata asrun md5"
fi
[ "$(md5sum < $T3B/t3b_train.py | cut -c1-8)" = c7ecf581 ] || die "t3b_train md5"
phase "code ok (test ${NW_TEST:-0}); code $CODE; out $OUT; jobs ${JOBS[*]}"
grep -h "SPLIT CHECK" $P/splits/split.log | tee -a $OUT/run.log | grep -q "SPLIT CHECK PASS" || die "split check"
phase "train"
declare -A GPU_OF NAME_OF
for j in "${JOBS[@]}"; do
  IFS=: read -r arm seed g <<< "$j"
  log=$OUT/logs/train_${arm}_s$seed.log
  if [ "$g" = cpu ]; then cvd=""; else
    cvd=$g
    while true; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      [ "$(ours_on $g)" -lt 2 ] && [ $((used + 12000)) -le 80000 ] && break
      echo "$(date -u +%FT%TZ) GPU $g: used $used MiB, ours $(ours_on $g); waiting to launch $arm s$seed" >> $OUT/run.log; sleep 60
    done
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ' | sed "s/^/$(date -u +%FT%TZ) nvidia-smi before $arm s$seed: /" >> $OUT/run.log; echo >> $OUT/run.log
  CUDA_VISIBLE_DEVICES=$cvd nice -n 10 $PY $CODE/nw_train.py --arm $arm --splits $SPL --t3b $T3B -- \
    E3_ensemble --seed $seed --frac 1 --data $DS $IT --out-root $OUT/models_${arm}_s$seed > $log 2>&1 &
  p=$!; GPU_OF[$p]=$g; NAME_OF[$p]="$arm s$seed"
  echo "$(date -u +%FT%TZ) train $arm seed $seed on GPU $g: pid $p" >> $OUT/run.log
  until grep -q " segments, splits" $log || ! kill -0 $p 2>/dev/null; do sleep 10; done
  echo "$(date -u +%FT%TZ)   $(grep -m1 "^nw_train arm" $log | cut -c1-200)" >> $OUT/run.log
done
for p in "${!GPU_OF[@]}"; do wait $p || die "training ${NAME_OF[$p]} (pid $p)"; done
grep -h "best val" $OUT/logs/train_*.log | tee -a $OUT/run.log
phase "eval s0"
A0="D=$O/train/models_D_s0,N=$OUT/models_N_s0,C=$OUT/models_C_s0"
[ -d $OUT/models_C2_s0 ] && A0="$A0,C2=$OUT/models_C2_s0"
CUDA_VISIBLE_DEVICES= nice -n 10 $PY $CODE/nw_eval.py --sd $O --arms $A0 --splits $SPL --out $OUT/eval_s0 > $OUT/eval_s0.raw 2>&1 || die "eval s0"
grep -v Warn $OUT/eval_s0.raw > $OUT/eval_s0.txt
grep -A4 "PRIMARY" $OUT/eval_s0.txt | tee -a $OUT/run.log
if [ -d $OUT/models_N_s1 ] && [ -d $OUT/models_C_s1 ]; then
  phase "eval s1"
  CUDA_VISIBLE_DEVICES= nice -n 10 $PY $CODE/nw_eval.py --sd $O --arms D=$O/train/models_D_s0,N=$OUT/models_N_s1,C=$OUT/models_C_s1 \
    --splits $SPL --out $OUT/eval_s1 > $OUT/eval_s1.raw 2>&1 || die "eval s1"
  grep -v Warn $OUT/eval_s1.raw > $OUT/eval_s1.txt
  grep -A4 "PRIMARY" $OUT/eval_s1.txt | tee -a $OUT/run.log
fi
phase "range N"
CUDA_VISIBLE_DEVICES= nice -n 10 $PY $CODE/nw_range.py --arm N --splits $SPL --sd-code $SDC -- --train $DS --data3 $O/data3/data/all.npz \
  --models $OUT/models_N_s0 --real main=$O/real/main/data/all.npz,task=$O/real/task/data/all.npz --json $OUT/range_N.json \
  > $OUT/range_N.raw 2>&1 || die "range N"
grep -v Warn $OUT/range_N.raw > $OUT/range_N.txt
phase "done"
