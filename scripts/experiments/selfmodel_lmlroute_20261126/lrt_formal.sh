#!/usr/bin/env bash
# notebook: selfmodel_lmlroute_20261126
# Formal run of selfmodel_lmlroute_20261126 on 4090 (tmux lrt_formal): lrt_formal.sh <backend>...
# (revision 1: lrt_formal.sh scalebfm sonic heft)
# Revision 1 (+ the addendum before the start): A R C CR of every controller, the 7 tasks x episodes 30..119, the round's
# looks (simulator mask), everything from the private copy C0 (lrt_run's code check, LC_CODE_ROOT).
# - inputs, code and looks checked against asrun/;
# - planners (lrt_planner.sh, one per card 1-3 when the card stays <= 20 GB; never card 0) into $O/routes;
# - workers (lrt_run.py work: mkdir claims on vePFS, one episode = its four groups in one process; every record states
#   the machine), so that H20 workers can join the same queue later: a worker is launched for the first controller in
#   the order given whose unclaimed episodes outnumber its running workers, while every process of this container
#   together (all our experiments; other tenants are not visible here) uses < CPU_CAP cores (lrt_cpu.py: utime + stime
#   over 5 s, not loadavg) and at most MAXP of ours run; each worker goes through the tasks from its own starting task;
# - claims of this machine left without done.json by a dead worker are released (lrt_run.py release) and rerun whole;
# - per controller once its 630 episodes are done: merge, lrt_check, lrt_eval, the copy's md5 again.
# Progress: $O/formal.log; worker pids: $O/pids.txt.
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlroute_20261126
C0=$O/code
A=$O/asrun
CODE=$C0/scripts/experiments/selfmodel_lmlroute_20261126
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
VPY=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
CPU_CAP=${CPU_CAP:-78}; MAXP=${MAXP:-70}
METHODS=alone,reachable,route,route_reachable
TASKS=(IThorBathroomLiftCandleFromFloor IThorKitchenApproachTableAndLiftApple IThorKitchenApproachTableAndLiftBowl IThorKitchenApproachTableAndLiftSaltShaker IThorKitchenTakeStatueFromCabinet IThorLivingRoomApproachTableAndLiftDumbbell IThorLivingRoomApproachTableAndLiftWateringCan)
NEP=$((90 * ${#TASKS[@]}))
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $O/formal.log; }
md5ok() { (cd $C0 && md5sum -c --quiet $A/CODE_MD5); }
inok() { md5sum -c --quiet $A/MD5 && md5sum -c --quiet $A/LOOKS_MD5; }
inok || { log "aborted: inputs differ from asrun/MD5 or LOOKS_MD5"; exit 1; }
md5ok || { log "aborted: code differs from asrun/CODE_MD5"; exit 1; }
echo "$@" > $O/FORMAL_BACKENDS
log "formal start on $(hostname): backends $*, groups $METHODS, episodes 30..119, claim queue $O/claims, CPU cap $CPU_CAP, max $MAXP"
mkdir -p $O/routes $O/claims
for g in 1 2 3; do
  (LRT_CODE=$C0 setsid nohup bash $CODE/lrt_planner.sh $g $O/routes $O/routes/planner$g.log > /dev/null 2>&1 < /dev/null &)
done
until ls $O/routes/planner_*.json > /dev/null 2>&1; do sleep 20; done  # at least one planner up before any episode
log "planner up: $(ls $O/routes/planner_*.json | xargs -n1 basename | tr '\n' ' ')"
cd $G || exit 1
export PYTHONPATH=$G:$C0/src SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export LC_CODE_ROOT=$C0 LC_OUT=$O LRT_ROUTES=$O/routes LRT_LOOKS=$O/hg/looks
for BE in "$@"; do mkdir -p $O/$BE/logs; $PY $CODE/lrt_run.py release --backend $BE --methods $METHODS >> $O/$BE/logs/release.log 2>&1; done
nclaim() { ls -d $O/claims/$1/*/ep* 2>/dev/null | wc -l; }
ndone() { ls $O/claims/$1/*/ep*/done.json 2>/dev/null | wc -l; }
finish() {  # merge, check, eval of a controller whose episodes are all done
  local BE=$1
  log "$BE runs done: $(ls $O/$BE/episodes/*/*/ep*.json 2>/dev/null | wc -l) episode records, hosts $(cat $O/claims/$BE/*/ep*/done.json | grep -o '"host": "[^"]*"' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  local nst=$(cat $O/$BE/logs/work_*.log | grep -c "^lc code check start: env.py $C0/")
  local nf=$(cat $O/$BE/logs/work_*.log | grep -c "^lc code check .*FAIL")
  log "$BE code check: $nst driver starts with env.py from the copy; failures $nf"
  (cd $G && OMP_NUM_THREADS=4 nice -n 19 $PY $CODE/lrt_run.py merge --backend $BE > $O/$BE/merge.log 2>&1)
  (cd $C0 && nice -n 19 $VPY $CODE/lrt_check.py --backend $BE --out $O --ar $O > $O/$BE/check.txt 2>&1)
  log "$BE merged and checked: $(tail -n 1 $O/$BE/check.txt)"
  (cd $C0 && LC_OUT=$O LC_AR=$O nice -n 19 $VPY $CODE/lrt_eval.py --backend $BE --json $O/$BE/eval.json 2>&1 | grep -v Warn > $O/$BE/eval.txt)
  log "$BE evaluated"
  if md5ok; then log "$BE md5 after: same as asrun"; else log "$BE md5 after: CHANGED"; fi
}
declare -A RUNNING NRUN FIN  # pid -> backend; backend -> running workers; backend -> finished
for BE in "$@"; do NRUN[$BE]=0; FIN[$BE]=0; done
k=0
while :; do
  for p in "${!RUNNING[@]}"; do
    kill -0 $p 2>/dev/null && continue
    BE=${RUNNING[$p]}; unset "RUNNING[$p]"; NRUN[$BE]=$((NRUN[$BE] - 1))
  done
  pick=; left=0
  for BE in "$@"; do
    [ ${FIN[$BE]} = 1 ] && continue
    nd=$(ndone $BE); nc=$(nclaim $BE)
    if [ $nd -ge $NEP ] && [ ${NRUN[$BE]} -eq 0 ]; then FIN[$BE]=1; { finish $BE & }; continue; fi
    left=1
    if [ ${NRUN[$BE]} -eq 0 ] && [ $nc -ge $NEP ] && [ $nd -lt $NEP ]; then  # all claimed, none of ours running: dead claims?
      $PY $CODE/lrt_run.py release --backend $BE --methods $METHODS >> $O/$BE/logs/release.log 2>&1
      msg=$(tail -n 1 $O/$BE/logs/release.log)
      case "$msg" in *": 0 claims released"*) ;; *) log "$BE: $msg";; esac
    fi
    [ -z "$pick" ] && [ $((NEP - nc)) -gt ${NRUN[$BE]} ] && pick=$BE
  done
  [ $left = 0 ] && break
  if [ -n "$pick" ] && [ ${#RUNNING[@]} -lt $MAXP ]; then
    read -r cpu _ <<< "$(python3 $CODE/lrt_cpu.py 5)"
    if python3 -c "import sys; sys.exit(0 if $cpu + 1 < $CPU_CAP else 1)"; then
      r=$((k % ${#TASKS[@]})); TL=$(printf '%s\n' "${TASKS[@]:$r}" "${TASKS[@]:0:$r}" | paste -sd,)
      nohup nice -n 19 $PY $CODE/lrt_run.py work --backend $pick --tasks $TL --eps=30:120 --methods $METHODS > $O/$pick/logs/work_$k.log 2>&1 &
      RUNNING[$!]=$pick; NRUN[$pick]=$((NRUN[$pick] + 1))
      echo "$! $pick $k $(hostname)" >> $O/pids.txt
      [ $((k % 5)) -eq 0 ] && log "launched worker $k ($pick, from ${TASKS[$r]}); container CPU $cpu cores before it; ours running ${#RUNNING[@]}; $pick claimed $(nclaim $pick) done $(ndone $pick)"
      k=$((k + 1))
      continue
    fi
  fi
  sleep 20
done
wait  # the last finish
for p in $(awk '{print $1}' $O/routes/planner_pids.txt); do kill $p 2>/dev/null; done; log "planners stopped (by pid)"
log "formal done"
