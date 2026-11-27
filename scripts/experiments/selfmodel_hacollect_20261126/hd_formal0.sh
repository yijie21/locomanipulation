#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# hd_formal0.sh (revision 2a, 2026-11-26): the frozen asrun's hd_formal.sh (206f1d33) with two changes, run from outside
# the asrun (the shards still run the frozen asrun's hd_job.sh): (1) one more slot, 0a (card 0, one process), while
# $L/card0_ok exists (hd_card0_guard.sh removes it and sets hold_0a when someone else needs card 0); (2) its own lock
# file (the first scheduler's lock stays held by the shard subshells it forked). The queue, order, adoption, holds and
# finishing are the original's. It replaces the first scheduler (stopped by pid; its running shards are adopted from
# their .run files).
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
mkdir -p $L/jobs $OS/logs
exec 9> $L/formal0.lock
flock -n 9 || { echo "another scheduler holds $L/formal0.lock"; exit 1; }
echo "$(date -u +%FT%TZ) scheduler start pid $$"
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
  read s p _ < $r
  if kill -0 $p 2>/dev/null; then PID[$s]=$p; TAG[$s]=$t; echo "$(date -u +%FT%TZ) slot $s: adopted $t (pid $p)"; fi
done
slots() { echo 1a 1b; [ -e $VR ] && echo 2a 2b 3a 3b; [ -e $L/card0_ok ] && echo 0a; }
next_job() {
  for j in "${JOBS[@]}"; do
    set -- $j
    [ -e $L/jobs/$1.done ] && continue
    local busy=0
    for s in "${!TAG[@]}"; do [ "${TAG[$s]}" = "$1" ] && busy=1; done
    [ $busy = 1 ] && continue
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
    j=$(next_job)
    [ -z "$j" ] && break
    set -- $j
    g=${s:0:1}
    ( bash $2/selfmodel_hacollect_20261126/hd_job.sh $2 $3 $g $4 $5 $6 $7 $8 $9 ${10} > $L/$1.log 2>&1
      echo "$? $(date -u +%FT%TZ)" > $L/jobs/$1.done ) &
    PID[$s]=$!; TAG[$s]=$1
    echo "$s $! $(date -u +%FT%TZ)" > $L/jobs/$1.run
    echo "$(date -u +%FT%TZ) slot $s: start $1 (pid $!)"
    sleep 20  # Kit start-ups one at a time
  done
  finish_ready
  left=$(for j in "${JOBS[@]}"; do set -- $j; [ -e $L/jobs/$1.done ] || echo $1; done | wc -l)
  if [ "$left" = 0 ] && [ ${#PID[@]} = 0 ]; then
    finish_ready
    echo "$(date -u +%FT%TZ) all shards exited"; date -u +%FT%TZ > $L/all_exited; break
  fi
  sleep 60
done
wait
echo "$(date -u +%FT%TZ) scheduler end"
