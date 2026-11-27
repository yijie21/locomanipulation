#!/usr/bin/env bash
# notebook: selfmodel_walkmix_20261126
# selfmodel_walkmix_20261126 part 1 (SIMPLE, ScaleBFM): the middle arms Ma, Mb, their shared control Cm, and Dh (D's data
# on this machine), seeds 0 and 1, on the H20 machine (ssh -p 27840), from the frozen copy $Q/asrun/code and
# simpledata's asrun/repo_b (data.py FDVEL version, train.py, models.py) with train3b's frozen t3b_train.py; nowalk's
# phase-1 recipe (REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1, X0_REF off), launched as on 4090b (py4090b.py).
# The machine (H20b, ssh -p 16093, or H20) is whichever runs it; its cards, trainings per card and CPU cap are arguments:
#   wm_run.sh "<cards>" <per card> <cores cap>     e.g. wm_run.sh "0 1 2 3 4 5 6 7" 1 150, in tmux walkmix
# $OUT/run.log, $OUT/STATUS ("FAIL ..." stops the run).
# Phases:
#   code    md5 of $Q/asrun/MD5.check (scripts and splits.npz), simpledata's asrun md5, t3b_train.py c7ecf581, the split check
#   train   8 trainings on the given cards: at most <per card> of ours per card; a launch only when the card's used memory
#           + 12 GB <= 80 GB and the cores used by every process in the container (10 s of /proc/<pid>/stat CPU time) + 5
#           <= <cores cap>; wm_train.py --arm A -- E3_ensemble --seed S --frac 1 --data <D's spec> --out-root $OUT/models_<A>_s<S>;
#           nice 10, 4 CPU threads
#   eval    on CPU: wm_eval.py, these 8 models with nowalk's N0, N1, C0, C1 and simpledata's D0 -> $OUT/eval.{txt,json}
# Test only: WM_TEST=1 with WM_OUT, WM_ITERS, WM_JOBS ("arm:seed ..."), WM_CODE; no md5 (the eval's checks still run).
set -uo pipefail
CARDS=($1); PER=$2; CAP=$3
Q=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_walkmix_20261126
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
O=$L/selfmodel_simpledata_20261126
P=$L/selfmodel_nowalk_20261126
R=$O/asrun/repo_b
T3B=$L/selfmodel_train3b_20261125/asrun/selfmodel_train3b_20261125
WS=/wuji-vefps-D/wuji-il/yijie/ws
CODE=${WM_CODE:-$Q/asrun/code}
OUT=${WM_OUT:-$Q/run}
JOBS=(${WM_JOBS:-Ma:0 Mb:0 Cm:0 Dh:0 Ma:1 Mb:1 Cm:1 Dh:1})
IT=${WM_ITERS:+--iters $WM_ITERS}
SPL=$Q/splits/splits.npz
DS=$O/data3/data/all.npz,$O/data3b/data/all.npz,$O/train/data/mid_data3.npz,$O/train/data/mid_data3b.npz
export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=$R SX_REPO=$R
unset REACHWM_X0_REF
PY="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
mkdir -p $OUT/logs
cd $R || exit 1
phase() { echo "$1" > $OUT/STATUS; echo "$(date -u +%FT%TZ) $1" | tee -a $OUT/run.log; }
die() { phase "FAIL $1"; exit 1; }
declare -A GPU_OF NAME_OF
ours_on() { local n=0 p; for p in "${!GPU_OF[@]}"; do [ "${GPU_OF[$p]}" = "$1" ] && kill -0 $p 2>/dev/null && n=$((n + 1)); done; echo $n; }
cpu_t() { for f in /proc/[0-9]*/stat; do sed 's/.*) //' "$f" 2>/dev/null; done | awk '{s += $12 + $13} END {print s}'; }
cores() { local a b; a=$(cpu_t); sleep 10; b=$(cpu_t); echo "$a $b $(getconf CLK_TCK)" | awk '{printf "%d\n", ($2 - $1) / (10 * $3) + 0.999}'; }
if [ -z "${WM_TEST:-}" ]; then
  (cd $Q/asrun && md5sum -c --quiet MD5.check) || die "asrun md5"
  (cd $O/asrun && md5sum -c --quiet MD5.check && md5sum -c --quiet MD5_r2.check) || die "simpledata asrun md5"
fi
[ "$(md5sum < $T3B/t3b_train.py | cut -c1-8)" = c7ecf581 ] || die "t3b_train md5"
phase "code ok (test ${WM_TEST:-0}); code $CODE; out $OUT; jobs ${JOBS[*]}; cards ${CARDS[*]}, $PER per card, cores cap $CAP; host $(hostname)"
grep -h "SPLIT CHECK" $Q/splits/split.log | tee -a $OUT/run.log | grep -q "SPLIT CHECK PASS" || die "split check"
phase "train"
for job in "${JOBS[@]}"; do
  arm=${job%%:*}; seed=${job##*:}
  g=""
  while [ -z "$g" ]; do
    c=$(cores)
    if [ $((c + 5)) -le $CAP ]; then
      for k in "${CARDS[@]}"; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $k)
        if [ "$(ours_on $k)" -lt $PER ] && [ $((used + 12000)) -le 80000 ]; then g=$k; break; fi
      done
    fi
    [ -z "$g" ] && sleep 50
  done
  log=$OUT/logs/train_${arm}_s$seed.log
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ' | sed "s/^/$(date -u +%FT%TZ) cores $c; nvidia-smi before $arm s$seed: /" >> $OUT/run.log; echo >> $OUT/run.log
  CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY $CODE/wm_train.py --arm $arm --splits $SPL --t3b $T3B -- \
    E3_ensemble --seed $seed --frac 1 --data $DS $IT --out-root $OUT/models_${arm}_s$seed > $log 2>&1 &
  p=$!; GPU_OF[$p]=$g; NAME_OF[$p]="$arm s$seed"
  echo "$(date -u +%FT%TZ) train $arm seed $seed on GPU $g: pid $p" >> $OUT/run.log
  until grep -q " segments, splits" $log || ! kill -0 $p 2>/dev/null; do sleep 10; done
  kill -0 $p 2>/dev/null || { wait $p || die "training $arm s$seed (pid $p) exited before training"; }
  echo "$(date -u +%FT%TZ)   $(grep -m1 "^nw_train arm" $log | cut -c1-160)" >> $OUT/run.log
done
for p in "${!GPU_OF[@]}"; do wait $p || die "training ${NAME_OF[$p]} (pid $p)"; done
grep -H "best val" $OUT/logs/train_*.log | sed "s|$OUT/logs/||" | tee -a $OUT/run.log
phase "eval"
MS=""
for job in "${JOBS[@]}"; do arm=${job%%:*}; seed=${job##*:}; MS="$MS,$arm$seed=$OUT/models_${arm}_s$seed"; done
MS="${MS#,},N0=$P/run_b/models_N_s0:z,N1=$P/run_b/models_N_s1:z,C0=$P/run_b/models_C_s0,C1=$P/run_b/models_C_s1,D0=$O/train/models_D_s0"
CUDA_VISIBLE_DEVICES= nice -n 10 $PY $CODE/wm_eval.py --sd $O --models "$MS" \
  --pairs Ma/Cm,Mb/Cm,Ma/N,Mb/N,Cm/N,Ma/C,Mb/C,Ma/Dh,Mb/Dh,Cm/Dh,Dh/D,N/C --nowalk-eval $P/run_b --out $OUT/eval \
  > $OUT/eval.raw 2>&1 || die "eval"
grep -v Warn $OUT/eval.raw > $OUT/eval.txt
grep -A12 "PRIMARY" $OUT/eval.txt | grep -E "PRIMARY|all:|Ma/Cm|Mb/Cm" | tee -a $OUT/run.log
phase "done"
