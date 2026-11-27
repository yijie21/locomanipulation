#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# hd_formal2.sh <4090|4090b> (revision 1 addendum, 2026-11-26): the frozen asrun's hd_formal.sh (206f1d33) for two
# machines drawing from one queue on vePFS, run from outside the asrun, one instance per machine. Changes, nothing else:
#  (1) claims: a shard is taken by mkdir $L/claims/<tag> (atomic on vePFS from either machine; no flock across
#      machines), the owner (machine, slot, time) written inside; .run gets the machine as a 4th field; each machine
#      adopts only its own .run files (no 4th field = 4090, the earlier schedulers').
#  (2) slots: 4090 1a 1b, 2a 2b 3a 3b once $VR exists, 0a while $L/card0_ok exists (hd_card0_guard.sh); 4090b b0a b1a,
#      b1b while $L/gate_b1b exists, b0b while $L/gate_b0b exists. The card is the slot's digit.
#  (3) the shard launcher: 4090 the asrun's hd_job.sh; 4090b dev/b4090b/hd_job_b.sh (same arguments). 4090b never takes
#      a w00 shard (shard 0 writes the collection's run.json, which stays the 4090's).
#  (4) the merge and checks (hd_finish.sh) run on the 4090 only; the 4090 instance ends when every shard has a .done,
#      the 4090b one when nothing is left to claim and its slots are empty.
#  (5) the lock is local (/tmp), against a second instance on the same machine.
# A shard with a nonzero exit keeps its .done and claim (not retried, as before: the poll shows it); one handed back
# (hd_card0_guard.sh) loses .done, .run and claim and is claimed again by either machine.
# The formal HA collections' scheduler on the 4090 (one instance; nohup): selfmodel_hadata_20261126 (ScaleBFM, HEFT) and
# selfmodel_hadata_sonic_20261126 (SONIC), each data3-style 3000 resets in 24 shards and data3b-style 1500 in 12 shards
# (125 resets a shard), in this order: ScaleBFM d3, d3b, HEFT d3, d3b, SONIC d3, d3b. Slots: two processes on card 1;
# cards 2 and 3 (two each) only once $VR/DONE exists (simple_sonic_vrref_20261126's end marker). Card 0 never.
# A slot with $L/hold_<slot> starts nothing new. A shard is a hd_job.sh process; its exit code goes to
# $L/jobs/<tag>.done; a nonzero one is not retried (the poll shows it). A shard interrupted with the scheduler (.run
# without .done, its process gone) is run again on restart (hc_ha.py skips the chunks already written). When every
# shard of a collection has exited 0, hd_finish.sh merges and checks it (CPU).
#   nohup bash hd_formal.sh > $L/formal.log 2>&1 &
set -uo pipefail
OH=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126
OS=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_sonic_20261126
VR=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_sonic_vrref_20261126/DONE
L=$OH/logs
MACH=${1:?4090 or 4090b}
CL=$L/claims
case $MACH in 4090|4090b) ;; *) echo "machine 4090 or 4090b"; exit 1;; esac
mkdir -p $L/jobs $OS/logs $CL
exec 9> /tmp/hd_formal2_$MACH.lock
flock -n 9 || { echo "another scheduler holds /tmp/hd_formal2_$MACH.lock"; exit 1; }
echo "$(date -u +%FT%TZ) scheduler start pid $$ on $MACH ($(hostname))"
# job lines: tag asrun out ctrl mode shard shards resets seed0 id0
JOBS=()
add() {  # exp-root ctrl mode shards resets seed0 id0
  local R=$1 C=$2 M=$3 P=$4 N=$5 S0=$6 I0=$7
  for k in $(seq 0 $((P - 1))); do JOBS+=("${C}_${M}_w$(printf %02d $k) $R/asrun $R $C $M $k $P $N $S0 $I0"); done
}
for C in scalebfm heft; do
  add $OH $C d3 24 3000 89100000 91000000
  add $OH $C d3b 12 1500 89600000 92000000
done
add $OS sonic d3 24 3000 89300000 93000000
add $OS sonic d3b 12 1500 89800000 94000000
printf '%s\n' "${JOBS[@]}" > $L/jobs.txt
declare -A PID TAG
for r in $L/jobs/*.run; do  # adopt shards still running from an earlier scheduler (its subshell writes the .done)
  [ -e "$r" ] || continue
  t=$(basename $r .run); [ -e $L/jobs/$t.done ] && continue
  read s p _ m < $r
  [ "${m:-4090}" = $MACH ] || continue
  if kill -0 $p 2>/dev/null; then PID[$s]=$p; TAG[$s]=$t; echo "$(date -u +%FT%TZ) slot $s: adopted $t (pid $p)"
  else mv $r $r.dead; rm -rf $CL/$t; echo "$(date -u +%FT%TZ) $t: interrupted on $MACH (pid $p gone), back in the queue"; fi
done
slots() {
  if [ $MACH = 4090 ]; then echo 1a 1b; [ -e $VR ] && echo 2a 2b 3a 3b; [ -e $L/card0_ok ] && echo 0a
  else echo b0a b1a; [ -e $L/gate_b1b ] && echo b1b; [ -e $L/gate_b0b ] && echo b0b; fi
}
gpu_of() { [ $MACH = 4090 ] && echo ${1:0:1} || echo ${1:1:1}; }
launcher() { [ $MACH = 4090 ] && echo $1/selfmodel_hacollect_20261126/hd_job.sh || echo $OH/../selfmodel_hacollect_20261126/dev/b4090b/hd_job_b.sh; }
next_job() {  # $1 slot: the first shard without .done and claim, claimed for this machine and slot
  for j in "${JOBS[@]}"; do
    set -- $j
    [ -e $L/jobs/$1.done ] && continue
    [ -d $CL/$1 ] && continue
    [ $MACH = 4090b ] && [[ $1 == *_w00 ]] && continue
    mkdir $CL/$1 2>/dev/null || continue
    echo "$MACH $SLOT $(date -u +%FT%TZ)" > $CL/$1/owner
    echo "$j"; return
  done
}
finish_ready() {  # collections whose shards all exited 0 and that are not finished yet
  for R in $OH $OS; do
    for C in scalebfm heft sonic; do
      for M in d3 d3b; do
        local tags=$(printf '%s\n' "${JOBS[@]}" | awk -v r="$R/asrun" -v c=$C -v m=$M '$2 == r && $4 == c && $5 == m {print $1}')
        [ -z "$tags" ] && continue
        [ -e $R/$C/$M/FINISHED ] || [ -e $R/$C/$M/FINISHING ] && continue
        local ok=1
        for t in $tags; do [ "$(cut -d' ' -f1 $L/jobs/$t.done 2>/dev/null)" = 0 ] || ok=0; done
        if [ $ok = 1 ]; then
          set -- $(printf '%s\n' "${JOBS[@]}" | awk -v r="$R/asrun" -v c=$C -v m=$M '$2 == r && $4 == c && $5 == m {print; exit}')
          touch $R/$C/$M/FINISHING
          echo "$(date -u +%FT%TZ) finish $C $M"
          (bash $2/selfmodel_hacollect_20261126/hd_finish.sh $2 $3 $C $M $8 ${10} > $R/$C/$M/finish.log 2>&1) &
        fi
      done
    done
  done
}
while :; do
  for s in "${!PID[@]}"; do
    kill -0 ${PID[$s]} 2>/dev/null && continue
    echo "$(date -u +%FT%TZ) slot $s: ${TAG[$s]} exited ($(cat $L/jobs/${TAG[$s]}.done 2>/dev/null || echo 'no .done'))"
    unset PID[$s] TAG[$s]
  done
  for s in $(slots); do
    [ -n "${PID[$s]:-}" ] && continue
    [ -e $L/hold_$s ] && continue
    SLOT=$s
    j=$(next_job)
    [ -z "$j" ] && break
    set -- $j
    g=$(gpu_of $s)
    ( bash $(launcher $2) $2 $3 $g $4 $5 $6 $7 $8 $9 ${10} > $L/$1.log 2>&1
      echo "$? $(date -u +%FT%TZ)" > $L/jobs/$1.done ) &
    PID[$s]=$!; TAG[$s]=$1
    echo "$s $! $(date -u +%FT%TZ) $MACH" > $L/jobs/$1.run
    echo "$(date -u +%FT%TZ) slot $s: start $1 (pid $!, card $g, $MACH)"
    sleep 20  # Kit start-ups one at a time
  done
  [ $MACH = 4090 ] && finish_ready
  if [ $MACH = 4090 ]; then
    left=$(for j in "${JOBS[@]}"; do set -- $j; [ -e $L/jobs/$1.done ] || echo $1; done | wc -l)
  else
    left=$(for j in "${JOBS[@]}"; do set -- $j; [ -e $L/jobs/$1.done ] || [ -d $CL/$1 ] || [[ $1 == *_w00 ]] || echo $1; done | wc -l)
  fi
  if [ "$left" = 0 ] && [ ${#PID[@]} = 0 ]; then
    [ $MACH = 4090 ] && { finish_ready; echo "$(date -u +%FT%TZ) all shards exited"; date -u +%FT%TZ > $L/all_exited; }
    [ $MACH = 4090b ] && echo "$(date -u +%FT%TZ) 4090b: nothing left to claim"
    break
  fi
  sleep 60
done
wait
echo "$(date -u +%FT%TZ) scheduler end"
