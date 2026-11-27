#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# Card 0's guard for the collection's slot 0a (revision 1 addendum, 2026-11-26; hd_formal0.sh, then hd_formal2.sh). Card 0 is shared: our SAM 3
# service and someone else's process were on it at the start (BASE MiB in all). Every minute:
#   memory  card 0's total use > BASE + our shard's use (its nvidia-smi entry, the one new since the start) + 1024 MiB;
#   compute our shard paused (SIGSTOP, 1 s, then 4 utilization samples 0.5 s apart, SIGCONT): every sample > 30 %.
#           With no shard of ours running: utilization > 30 % or total > BASE + 1024 MiB closes the slot the same way.
# Either one: hold_0a set and card0_ok removed (the slot opens no new shard), our shard's python stopped by pid (TERM,
# KILL after 30 s; nothing else on card 0 is touched), and once its .done is written, .done and .run are renamed
# (*.card0) so the scheduler runs that shard again on another slot (hc_ha.py skips the chunks already written).
# The guard then exits (the slot stays closed; reopening is by hand). Log: $L/card0_guard.log.
#   nohup bash hd_card0_guard.sh <BASE MiB> [<our running shard's nvidia-smi pid>] > /dev/null 2>&1 < /dev/null &
# (the second argument when the guard is restarted while our shard runs: that entry is ours, not someone else's)
set -uo pipefail
BASE=$1 OURS0=${2:-}
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/logs
U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
log() { echo "$(date -u +%FT%TZ) $*" >> $L/card0_guard.log; }
apps() { nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v u=$U '$1 == u {print $2, $3}'; }
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0; }
util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0; }
OLD=" $(apps | awk -v o="$OURS0" '$1 != o {print $1}' | tr '\n' ' ') "
log "start pid $$: BASE $BASE MiB, card 0 entries at the start (not ours):$OLD${OURS0:+; ours $OURS0}"
shard() {  # the slot's running shard: tag and its python's pid (hd_job.sh -> hc_ha.sh -> nice -> python, all exec)
  local r t s p c
  for r in $(grep -l "^0a " $L/jobs/*.run 2>/dev/null); do
    t=$(basename $r .run); [ -e $L/jobs/$t.done ] && continue
    read s p _ < $r
    c=$(pgrep -P $p -f "[h]c_ha.py" | head -1)
    [ -n "$c" ] && { echo "$t $c"; return; }
  done
}
close() {  # $1 why, $2 tag, $3 python pid (may be empty)
  touch $L/hold_0a; rm -f $L/card0_ok
  log "TRIGGER: $1; slot 0a closed${2:+, stopping $2 python pid $3}"
  [ -z "${3:-}" ] && exit 0
  kill -TERM $3 2>/dev/null
  for _ in $(seq 1 30); do kill -0 $3 2>/dev/null || break; sleep 1; done
  kill -0 $3 2>/dev/null && { kill -KILL $3; log "pid $3 needed SIGKILL"; }
  for _ in $(seq 1 300); do [ -e $L/jobs/$2.done ] && break; sleep 1; done
  log "$2 .done: $(cat $L/jobs/$2.done 2>/dev/null || echo none)"
  mv $L/jobs/$2.done $L/jobs/$2.done.card0 2>/dev/null; mv $L/jobs/$2.run $L/jobs/$2.run.card0 2>/dev/null
  rm -rf $L/claims/$2  # hd_formal2.sh's claim (either machine may take it again)
  log "$2 handed back to the queue"
  exit 0
}
OURS=$OURS0
while :; do
  read t c < <(shard; echo)
  tot=$(used)
  if [ -n "${c:-}" ]; then
    new=$(apps | awk -v o="$OLD" 'index(o, " " $1 " ") == 0 {print $1, $2}')
    [ -z "$OURS" ] && OURS=$(echo "$new" | awk 'NF {print $1; exit}')
    mine=$(echo "$new" | awk -v p="$OURS" '$1 == p {print $2}'); mine=${mine:-0}
    lim=$((BASE + mine + 1024))
    [ -n "$OURS" ] && [ "$tot" -gt "$lim" ] && close "memory $tot MiB > $BASE + ours $mine + 1024" $t $c
    kill -STOP $c 2>/dev/null; sleep 1
    s1=$(util); sleep 0.5; s2=$(util); sleep 0.5; s3=$(util); sleep 0.5; s4=$(util)
    kill -CONT $c 2>/dev/null
    [ "$s1" -gt 30 ] && [ "$s2" -gt 30 ] && [ "$s3" -gt 30 ] && [ "$s4" -gt 30 ] && close "utilization $s1/$s2/$s3/$s4 % with ours paused" $t $c
    log "ok: $t pid $c (nvidia entry ${OURS:-not yet}, $mine MiB), total $tot MiB (limit $lim), paused utilization $s1/$s2/$s3/$s4 %"
  else
    OURS=""
    OLD=" $(apps | awk '{print $1}' | tr '\n' ' ') "
    [ "$tot" -gt $((BASE + 1024)) ] && close "memory $tot MiB > $BASE + 1024 with no shard of ours"
    u=$(util); [ "$u" -gt 30 ] && close "utilization $u % with no shard of ours"
    log "ok: no shard of ours; total $tot MiB, utilization $u %"
  fi
  sleep 60
done
