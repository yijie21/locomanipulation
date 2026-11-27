#!/usr/bin/env bash
# notebook: simple_relook_20261126 -- the ScaleBFM run continued by hand (coordinator, 11-26 ~10:10Z) from the step
# run.sh was waiting on: the relook group's SF and kin_rr (sfull_run.sh), then J0 (plan and JR), then the evaluation.
# The commands are run.sh's own (asrun/repo, RL_scalebfm, the same environment); only the GPU memory gate differs.
# run.sh asked 2.9 GB free on cards 1 and 3 each for the sfull step, which blocked it from 09:26:54Z on; measured on this
# run's machine (development, 11-26): a cuRobo planner ~1.0 GB (card 0 of 4090b, 8493 -> 9527 MiB over one plan2.py), a
# ScaleBFM sim ~0.82-0.95 GB (nvidia-smi's compute apps during the relook A, four processes). So a step's need per card:
#   sfull  the planner card (s_check, shifts2, two planners at a time, pick, jref_data) 2.3 GB; the other card one sim at
#          a time (ARM_PAR=2: SF on the first card, kin_rr on the second) 1.0 GB
#   J0 plan  one card 2.3 GB (two planners)
#   J0 JR  four sims, 1.0 GB each, over cards 1 and 3 as they have room
# Before every step nvidia-smi is read again: <used> + <need> <= 20480 MiB on each card the step uses, card 3 preferred
# for the heavier part, else wait (2 min). Card 0 is never used. Progress into $X/scalebfm/run.log as run.sh's.
set -uo pipefail
X=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_relook_20261126
WS=/wuji-vefps-D/wuji-il/yijie/ws
OLD=$WS/code/loco/outputs/simple_eval_20261112
PC=$WS/code/loco/outputs/simple_dev_20261124/precision_const
C=scalebfm RL=$(cat $X/asrun/RL_scalebfm) O=$X/scalebfm R=$X/asrun/repo
export BENCH_REPO=$R SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE BENCH_SET=sensor=rgbd BENCH_PLAN_PAR=2 J_ARM=JR
unset SIMPLE_EVAL_OUT SIMPLE_LEVELS BENCH_TASKS BENCH_EPISODES BENCH_EXTRA ONLY CONST_LOO S_CHOICE MODEL DATA REACH_SONIC_NETS
E=$R/scripts/experiments/simple_eval_20261112 H=$R/scripts/experiments/simple_relook_20261126
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
D=$O/relook/bench
log() { echo "[$(date -u +%FT%TZ)] $*" >> $O/run.log; }
md5check() { (cd $X/asrun && md5sum -c --quiet MD5.check > /dev/null 2>&1) && echo ok || echo CHANGED; }
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 || echo 99999; }
fits() { [ $(($(used $1) + $2)) -le 20480 ]; }
smi() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; }
pick2() {  # pick2 <need heavy> <need light> -> "<heavy card> <light card>", card 3 preferred for the heavy part; waits
  while :; do
    if fits 3 $1 && fits 1 $2; then echo "3 1"; return; fi
    if fits 1 $1 && fits 3 $2; then echo "1 3"; return; fi
    log "manual wait: need $1 MiB on one of cards 3/1 and $2 MiB on the other ($(smi))"; sleep 120
  done
}
pick1() {  # pick1 <need> -> a card, 3 preferred; waits
  while :; do
    fits 3 $1 && { echo 3; return; }
    fits 1 $1 && { echo 1; return; }
    log "manual wait: need $1 MiB on card 3 or 1 ($(smi))"; sleep 120
  done
}
sims() {  # sims <n> -> n card numbers (1 GB each), as cards 3 and 1 have room, card 3 first; waits
  local n=$1 c3 c1 out
  while :; do
    c3=$(( (20480 - $(used 3)) / 1000 )); c1=$(( (20480 - $(used 1)) / 1000 ))
    [ $c3 -lt 0 ] && c3=0; [ $c1 -lt 0 ] && c1=0
    if [ $((c3 + c1)) -ge $n ]; then
      out=""; local k3=0 k1=0 i
      for i in $(seq 1 $n); do  # alternate while both have room, so the tasks spread
        if [ $k3 -lt $c3 ] && { [ $k3 -le $k1 ] || [ $k1 -ge $c1 ]; }; then out="$out 3"; k3=$((k3 + 1)); else out="$out 1"; k1=$((k1 + 1)); fi
      done
      echo $out; return
    fi
    log "manual wait: $n sims of 1000 MiB, room on card 3 $c3, card 1 $c1 ($(smi))"; sleep 120
  done
}
step() {  # step <name> <command...>
  local n=$1; shift
  log "relook/j0 $n: start (manual; $(smi))"
  "$@" || { log "$n: FAIL"; echo "FAIL $n" >> $O/FAILED; return 1; }
  log "$n: done"
}

log "manual resume (coordinator): run.sh stopped while waiting for 2.9 GB on cards 1 and 3; asrun $(md5check); gates as measured"
echo running > $O/STATUS
G=$(pick2 2300 1000)
step "relook sfull" env SF_ARMS="SF kin_rr" SF_BENCH_EXTRA="$RL" ARM_PAR=2 PLAN_PAR=2 bash $E/sfull_run.sh \
  $O/relook/sfull $C $D $PC/const_$C.json $PC/ctrack_$C.json $PC/ctrack_rr_$C.json "$G"
J=$O/j0/bench
mkdir -p $J
ln -sfn $OLD/bench_rgbd_20261124/scalebfm/films_A $J/films_A; ln -sfn $OLD/bench_rgbd_20261124/scalebfm/rows_ep $J/rows_ep
g=$(pick1 2300)
step "j0 plan" env BENCH_GPU=$g BENCH_GPUS=$g bash $E/bench_via.sh $J $C none plan 0 29 && {
  GS=$(sims 4)
  step "j0 JR" env BENCH_GPU=${GS%% *} BENCH_GPUS="$GS" bash $E/bench_via.sh $J $C none JR 0 29; }
(cd $R && $PY -I $H/relook_eval.py scalebfm $O/relook/bench $O/relook/sfull $OLD/bench_rgbd_20261124/scalebfm $OLD/sfull_20261125/scalebfm J) > $O/eval_relook.txt 2>&1
(cd $R && $PY -I $H/j0_eval.py $O/j0/bench $OLD/bench_rgbd_20261124/scalebfm J) > $O/eval_j0.txt 2>&1
log "asrun after the run: $(md5check)"
if [ -s $O/FAILED ]; then echo "FAIL $(head -1 $O/FAILED)" > $O/STATUS; else echo done > $O/STATUS; fi
log "end: $(cat $O/STATUS)"
