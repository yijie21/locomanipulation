#!/usr/bin/env bash
# notebook: selfmodel_nowalk_20261126
# selfmodel_nowalk_20261126 phase 2 (LocoManiLab; ScaleBFM, SONIC, HEFT): trainings and evaluations on the H20 machine
# (ssh -p 27840), from the frozen copy $P/asrun2/code and simpledata's asrun/repo_b (data.py, train.py, models.py; FDVEL
# off: train3b's D) with train3b's frozen t3b_train.py, launched as on 4090b (py4090b.py).
#   nw2_run.sh                     in tmux nowalk2; $OUT/run.log, $OUT/STATUS ("FAIL ..." stops the run)
# Phases:
#   code    md5 of $P/asrun2/MD5.check (scripts and the three splits.npz), simpledata's asrun md5, t3b_train.py c7ecf581
#   train   for each controller N and C, seeds 0, 1, 2, on cards 0 / 1 in turn, at most 2 of ours per card, a launch only
#           when the card's used memory + 12 GB <= 80 GB; nw_train.py --arm A -- E3_ensemble --seed S --frac 1 --data
#           <train3b D's spec> --out-root $OUT/<be>/models_<A>_s<S>; REACHWM_GOAL_REF / APPROACH / QFIX = 1, FDVEL and
#           X0_REF off; nice 10, 4 CPU threads
#   eval    on CPU, per controller: nw2_eval.py -> $OUT/<be>/eval.{txt,json}
# Test only: NW_TEST=1 with NW_OUT, NW_ITERS, NW_BES, NW_SEEDS, NW_CODE; no md5 (the D reproduction check still runs).
set -uo pipefail
P=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
O=$L/selfmodel_simpledata_20261126
R=$O/asrun/repo_b
T3B=$L/selfmodel_train3b_20261125/asrun/selfmodel_train3b_20261125
WS=/wuji-vefps-D/wuji-il/yijie/ws
CODE=${NW_CODE:-$P/asrun2/code}
OUT=${NW_OUT:-$P/lml/run}
BES=(${NW_BES:-scalebfm sonic heft})
SEEDS=(${NW_SEEDS:-0 1 2})
IT=${NW_ITERS:+--iters $NW_ITERS}
export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 LOCO_REPO=$R SX_REPO=$R
unset REACHWM_X0_REF REACHWM_FDVEL
PY="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
mkdir -p $OUT/logs
cd $R || exit 1
phase() { echo "$1" > $OUT/STATUS; echo "$(date -u +%FT%TZ) $1" | tee -a $OUT/run.log; }
die() { phase "FAIL $1"; exit 1; }
declare -A GPU_OF NAME_OF
ours_on() { local n=0 p; for p in "${!GPU_OF[@]}"; do [ "${GPU_OF[$p]}" = "$1" ] && kill -0 $p 2>/dev/null && n=$((n + 1)); done; echo $n; }
if [ -z "${NW_TEST:-}" ]; then
  (cd $P/asrun2 && md5sum -c --quiet MD5.check) || die "asrun2 md5"
  (cd $O/asrun && md5sum -c --quiet MD5.check && md5sum -c --quiet MD5_r2.check) || die "simpledata asrun md5"
fi
[ "$(md5sum < $T3B/t3b_train.py | cut -c1-8)" = c7ecf581 ] || die "t3b_train md5"
phase "code ok (test ${NW_TEST:-0}); code $CODE; out $OUT; controllers ${BES[*]}; seeds ${SEEDS[*]}"
for be in "${BES[@]}"; do
  grep -h "SPLIT CHECK" $P/lml/$be/splits/split.log | sed "s/^/$be: /" | tee -a $OUT/run.log | grep -q "SPLIT CHECK PASS" || die "$be split check"
done
phase "train"
NG=0
for be in "${BES[@]}"; do
  DS=$L/selfmodel_data3_20261125/$be/data/all.npz,$L/selfmodel_data3b_20261125/$be/data/all.npz
  DS=$DS,$L/selfmodel_train3b_20261125/$be/data/mid_data3.npz,$L/selfmodel_train3b_20261125/$be/data/mid_data3b.npz
  for seed in "${SEEDS[@]}"; do
    for arm in N C; do
      g=$((NG % 2)); NG=$((NG + 1))
      while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
        [ "$(ours_on $g)" -lt 2 ] && [ $((used + 12000)) -le 80000 ] && break
        o=$((1 - g)); used_o=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $o)
        if [ "$(ours_on $o)" -lt 2 ] && [ $((used_o + 12000)) -le 80000 ]; then g=$o; break; fi
        sleep 60
      done
      log=$OUT/logs/train_${be}_${arm}_s$seed.log
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ' | sed "s/^/$(date -u +%FT%TZ) nvidia-smi before $be $arm s$seed: /" >> $OUT/run.log; echo >> $OUT/run.log
      mkdir -p $OUT/$be
      CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY $CODE/nw_train.py --arm $arm --splits $P/lml/$be/splits/splits.npz --t3b $T3B -- \
        E3_ensemble --seed $seed --frac 1 --data $DS $IT --out-root $OUT/$be/models_${arm}_s$seed > $log 2>&1 &
      p=$!; GPU_OF[$p]=$g; NAME_OF[$p]="$be $arm s$seed"
      echo "$(date -u +%FT%TZ) train $be $arm seed $seed on GPU $g: pid $p" >> $OUT/run.log
      until grep -q " segments, splits" $log || ! kill -0 $p 2>/dev/null; do sleep 10; done
      kill -0 $p 2>/dev/null || { wait $p || die "training $be $arm s$seed (pid $p) exited before training"; }
      echo "$(date -u +%FT%TZ)   $(grep -m1 "^nw_train arm" $log | cut -c1-160)" >> $OUT/run.log
    done
  done
done
for p in "${!GPU_OF[@]}"; do wait $p || die "training ${NAME_OF[$p]} (pid $p)"; done
grep -H "best val" $OUT/logs/train_*.log | sed "s|$OUT/logs/||" | tee -a $OUT/run.log
for be in "${BES[@]}"; do
  phase "eval $be"
  CUDA_VISIBLE_DEVICES= nice -n 10 $PY $CODE/nw2_eval.py --backend $be --root $OUT/$be --splits $P/lml/$be/splits/splits.npz \
    --out $OUT/$be/eval > $OUT/$be/eval.raw 2>&1 || die "eval $be"
  grep -v Warn $OUT/$be/eval.raw > $OUT/$be/eval.txt
  grep -A6 "PRIMARY" $OUT/$be/eval.txt | tee -a $OUT/run.log
done
phase "done"
