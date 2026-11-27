#!/usr/bin/env bash
# notebook: selfmodel_nowalk_20261126
# selfmodel_nowalk_20261126 revision 1b: takes over nw_run_b.sh's run on 4090b (run_b/) once HA's b0a left card 0
# (coordinator, 09:24Z): two of ours at a time on card 0 instead of one, the same check per launch (used + 9 GB <= 20 GB).
# nw_run_b.sh (pid ORIG) is stopped with SIGSTOP while it waits to launch C s0; its N s0 training (pid NPID, its child)
# runs on and is counted as ours; this script launches C s0, C2 s0, N s1, C s1 (nw_run_b.sh's order), waits for all,
# checks N s0's log and model, then kills the stopped nw_run_b.sh (SIGKILL; N s0 has ended by then, so nothing of ours is
# left in that tmux session) and runs nw_run_b.sh's evaluations unchanged. Same frozen scripts, splits.npz and OUT.
#   nw_run_b2.sh ORIG NPID         in tmux nowalk_b2; $OUT/run.log (appended), $OUT/STATUS ("FAIL ..." stops the run)
# Phases:
#   code    md5 of $P/asrun/MD5.check, splits.npz (its md5 in MD5.check), t3b_train.py c7ecf581, the shared code
#   train   JOBS (arm seed gpu): C 0, C2 0, N 1, C 1 (N 0 adopted), all on card 0, at most two of ours at a time, a
#           launch only when the card's used memory + 9 GB <= 20 GB (nvidia-smi; the user's 4090b cap); waits otherwise.
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
OUT=${NW_OUT:-$P/run_b}
SPL=$P/splits/splits.npz
DS=$O/data3/data/all.npz,$O/data3b/data/all.npz,$O/train/data/mid_data3.npz,$O/train/data/mid_data3b.npz
JOBS=(${NW_JOBS:-C:0:0 C2:0:0 N:1:0 C:1:0})
ORIG=$1 NPID=$2
IT=${NW_ITERS:+--iters $NW_ITERS}
export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=$R SX_REPO=$R
unset REACHWM_X0_REF
PY="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
mkdir -p $OUT/logs
cd $R || exit 1
phase() { echo "$1" > $OUT/STATUS; echo "$(date -u +%FT%TZ) $1" | tee -a $OUT/run.log; }
die() { phase "FAIL $1"; exit 1; }
# alive: a zombie counts as ended (NPID's parent, nw_run_b.sh, is stopped and cannot reap it)
alive() { [ -r /proc/$1/stat ] && [ "$(awk '{print $3}' /proc/$1/stat)" != Z ]; }
ours_on() { local n=0 p; for p in "${!GPU_OF[@]}"; do [ "${GPU_OF[$p]}" = "$1" ] && alive $p && n=$((n + 1)); done; echo $n; }
if [ -z "${NW_TEST:-}" ]; then
  (cd $P/asrun && md5sum -c --quiet MD5.check && md5sum -c --quiet MD5_r1.check && md5sum -c --quiet MD5_r2.check) || die "asrun md5"
  (cd $O/asrun && md5sum -c --quiet MD5.check && md5sum -c --quiet MD5_r2.check) || die "simpledata asrun md5"
fi
[ "$(md5sum < $T3B/t3b_train.py | cut -c1-8)" = c7ecf581 ] || die "t3b_train md5"
phase "code ok (test ${NW_TEST:-0}); code $CODE; out $OUT; jobs ${JOBS[*]}"
grep -h "SPLIT CHECK" $P/splits/split.log | tee -a $OUT/run.log | grep -q "SPLIT CHECK PASS" || die "split check"
[ "$(awk '{print $3}' /proc/$ORIG/stat)" = T ] && tr '\0' ' ' < /proc/$ORIG/cmdline | grep -q "nw_run_b.sh" || die "ORIG $ORIG: not a stopped nw_run_b.sh"
tr '\0' ' ' < /proc/$NPID/cmdline | grep -q "nw_train.py --arm N .*--out-root $OUT/models_N_s0" && [ "$(awk '{print $4}' /proc/$NPID/stat)" = "$ORIG" ] \
  || die "NPID $NPID: not nw_run_b.sh's N s0 training"
phase "train"
declare -A GPU_OF NAME_OF
GPU_OF[$NPID]=0; NAME_OF[$NPID]="N s0 (adopted from nw_run_b.sh)"
echo "$(date -u +%FT%TZ) takeover: nw_run_b.sh pid $ORIG stopped; N s0 pid $NPID adopted; two of ours at a time on card 0" >> $OUT/run.log
for j in "${JOBS[@]}"; do
  IFS=: read -r arm seed g <<< "$j"
  log=$OUT/logs/train_${arm}_s$seed.log
  if [ "$g" = cpu ]; then cvd=""; else
    cvd=$g
    while true; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      [ "$(ours_on $g)" -lt 2 ] && [ $((used + 9000)) -le 20000 ] && break
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
for p in "${!GPU_OF[@]}"; do
  if [ "$p" = "$NPID" ]; then
    while alive $NPID; do sleep 30; done
    grep -q "best val median" $OUT/logs/train_N_s0.log && [ -f $OUT/models_N_s0/E3_ensemble_f1/model.pt ] || die "training N s0 (adopted pid $NPID)"
    kill -KILL $ORIG && echo "$(date -u +%FT%TZ) N s0 ended; stopped nw_run_b.sh (pid $ORIG) killed" >> $OUT/run.log
  else
    wait $p || die "training ${NAME_OF[$p]} (pid $p)"
  fi
done
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
