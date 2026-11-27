#!/usr/bin/env bash
# notebook: simple_relook_20261126 -- the formal run of one controller, from the frozen copy asrun/ (MD5.check):
#   run.sh scalebfm  the 4090, asrun/repo; sims, rendering and cuRobo on cards 1 and 3. Groups one after another:
#                    relook (A, plan, JR, then SF and kin_rr, all with RL_scalebfm), then J0 (plan and JR from the main
#                    metric's A films, bench_rgbd_20261124/scalebfm, no relook). The no-relook side is the main metric's
#                    own records (the switch-off identity: development, and again from asrun/repo).
#   run.sh sonic     4090b, asrun/repo_b (py4090b_rl.py as its python); SONIC's networks on the CPU (allcpu), rendering
#                    and cuRobo on a card per group (none and look_only card 0, relook and J0 card 1). Groups side by
#                    side, each its own pipeline: none (A, plan, JR, SF and kin_rr without the second look: the new
#                    baseline, as the GPU records do not equal the CPU's), relook (the same with RL_sonic), look_only
#                    (A with RL_sonic --relook-steps 0), J0 (plan and JR from none's A films, once look_only's A and
#                    none's A are done). At most 12 sims at once (three groups of four task processes).
# Every step waits until each card it uses has <used> + <need> <= 20 GB. Outputs $X/<ctrl>/<group>/{bench,sfull};
# progress $X/<ctrl>/run.log and STATUS; evaluation $X/<ctrl>/eval_*.txt.
set -uo pipefail
C=$1
X=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_relook_20261126
WS=/wuji-vefps-D/wuji-il/yijie/ws
OLD=$WS/code/loco/outputs/simple_eval_20261112
PC=$WS/code/loco/outputs/simple_dev_20261124/precision_const
RL=$(cat $X/asrun/RL_$C)  # the relook options, frozen: SONIC's add the arms' tuck before its first step
O=$X/$C
case $C in
  scalebfm) R=$X/asrun/repo ;;
  sonic) R=$X/asrun/repo_b; export REACH_SONIC_NETS=allcpu LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1 ;;
  *) echo "scalebfm or sonic"; exit 2 ;;
esac
export BENCH_REPO=$R SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE BENCH_SET=sensor=rgbd BENCH_PLAN_PAR=2 J_ARM=JR
unset SIMPLE_EVAL_OUT SIMPLE_LEVELS BENCH_TASKS BENCH_EPISODES BENCH_EXTRA ONLY CONST_LOO S_CHOICE MODEL DATA
E=$R/scripts/experiments/simple_eval_20261112 H=$R/scripts/experiments/simple_relook_20261126
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
[ -e $O ] && { echo "$O exists: a run was made there"; exit 2; }
mkdir -p $O
log() { echo "[$(date -u +%FT%TZ)] $*" >> $O/run.log; }
md5check() { (cd $X/asrun && md5sum -c --quiet MD5.check > /dev/null 2>&1) && echo ok || echo CHANGED; }
gpu_wait() {  # gpu_wait <need MiB> <cards...>
  local need=$1 c u ok; shift
  while :; do
    ok=1
    for c in "$@"; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c) || u=99999
      [ $((u + need)) -le 20480 ] || ok=0
    done
    [ $ok = 1 ] && return
    log "wait: cards $* need $need MiB each"; sleep 120
  done
}
step() {  # step <group> <name> <need MiB> <cards> <command...>
  local g=$1 n=$2 need=$3 cards=$4; shift 4
  gpu_wait $need $cards
  log "$g $n: start (cards $cards; $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '))"
  "$@" || { log "$g $n: FAIL"; echo "FAIL $g $n" >> $O/FAILED; return 1; }
  log "$g $n: done"
}
sims=1300; [ $C = sonic ] && sims=600  # four task processes over the cards (ScaleBFM ~0.6 GB each; SONIC renders only)
cards() { export BENCH_GPUS="$*" BENCH_GPU=$1; }  # the cards of the group in this subshell: sims in turn, cuRobo on the first
bench_group() {  # bench_group <group> <relook options, maybe empty>: A, plan, JR, then SF and kin_rr
  local g=$1 rl=$2 D=$O/$1/bench
  mkdir -p $O/$g
  step $g A $sims "$BENCH_GPUS" env BENCH_EXTRA="$rl" bash $E/bench_via.sh $D $C none A 0 29 || return 1
  touch $O/$g/A.done
  step $g plan 2300 $BENCH_GPU bash $E/bench_via.sh $D $C none plan 0 29 || return 1
  step $g JR $sims "$BENCH_GPUS" env BENCH_EXTRA="$rl" bash $E/bench_via.sh $D $C none JR 0 29 || return 1
  step $g sfull 2900 "$BENCH_GPUS" env SF_ARMS="SF kin_rr" SF_BENCH_EXTRA="$rl" ARM_PAR=2 PLAN_PAR=2 bash $E/sfull_run.sh \
    $O/$g/sfull $C $D $PC/const_$C.json $PC/ctrack_$C.json $PC/ctrack_rr_$C.json "$BENCH_GPUS" || return 1
}
j0_group() {  # j0_group <bench whose A films to replan>
  local D=$O/j0/bench
  mkdir -p $D
  ln -s $1/films_A $D/films_A; ln -s $1/rows_ep $D/rows_ep
  step j0 plan 2300 $BENCH_GPU bash $E/bench_via.sh $D $C none plan 0 29 || return 1
  step j0 JR $sims "$BENCH_GPUS" bash $E/bench_via.sh $D $C none JR 0 29
}
look_only() {
  mkdir -p $O/look_only
  step look_only A $sims "$BENCH_GPUS" env BENCH_EXTRA="$RL --relook-steps 0" bash $E/bench_via.sh $O/look_only/bench $C none A 0 29
}
ev() { (cd $R && $PY -I "$@") 2>&1; }

echo running > $O/STATUS
log "start $C from $R; asrun $(md5check); RL '$RL'; nets ${REACH_SONIC_NETS:-}"
if [ $C = scalebfm ]; then
  ( cards 1 3; bench_group relook "$RL" )
  ( cards 1 3; j0_group $OLD/bench_rgbd_20261124/scalebfm )
  ev $H/relook_eval.py scalebfm $O/relook/bench $O/relook/sfull $OLD/bench_rgbd_20261124/scalebfm $OLD/sfull_20261125/scalebfm J > $O/eval_relook.txt
  ev $H/j0_eval.py $O/j0/bench $OLD/bench_rgbd_20261124/scalebfm J > $O/eval_j0.txt
else
  ( cards 0; bench_group none "" ) & p1=$!
  ( cards 1; bench_group relook "$RL" ) & p2=$!
  ( cards 0; look_only )
  while [ ! -e $O/none/A.done ] && kill -0 $p1 2> /dev/null; do sleep 60; done
  [ -e $O/none/A.done ] && ( cards 1; j0_group $O/none/bench )
  wait $p1 $p2
  ev $H/relook_eval.py sonic $O/relook/bench $O/relook/sfull $O/none/bench $O/none/sfull JR > $O/eval_relook.txt
  ev $H/relook_eval.py sonic $O/look_only/bench $O/none/sfull_absent $O/none/bench $O/none/sfull JR > $O/eval_look_only.txt
  ev $H/j0_eval.py $O/j0/bench $O/none/bench JR > $O/eval_j0.txt
  ev $H/relook_eval.py sonic $O/none/bench $O/none/sfull $OLD/bench_rgbd_20261124/sonic $OLD/sfull_20261125/sonic J > $O/eval_cpu_vs_gpu.txt
fi
log "asrun after the run: $(md5check)"
if [ -s $O/FAILED ]; then echo "FAIL $(head -1 $O/FAILED)" > $O/STATUS; else echo done > $O/STATUS; fi
log "end: $(cat $O/STATUS)"
