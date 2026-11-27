#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# Frees one collection slot for someone else's run (coordinator's call): hold_<slot> set (the slot opens no new shard),
# the slot's running shard stopped by its python's pid (TERM, KILL after 30 s; nothing else touched), its .done/.run
# renamed *.<suffix> and its claim removed so either machine runs it again (hc_ha.py skips the chunks already written),
# then the card's memory before and after. Reopening is by hand (rm hold_<slot>).
#   bash hd_release_slot.sh <slot e.g. b0b> <suffix e.g. yield1>
set -uo pipefail
S=$1 X=$2
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/logs
LOG=$L/formal_4090b.log; [[ $S == b* ]] || LOG=$L/formal.log
case $S in b0?|0?) G=0;; b1?|1?) G=1;; 2?) G=2;; 3?) G=3;; *) echo "slot $S?"; exit 2;; esac
mem() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i $G; }
echo "card $G before: $(mem)"
touch $L/hold_$S
echo "$(date -u +%FT%TZ) hold_$S set (coordinator: slot lent out); stopping its shard" >> $LOG
T="" C=""
for r in $(grep -l "^$S " $L/jobs/*.run 2>/dev/null); do
  t=$(basename $r .run); [ -e $L/jobs/$t.done ] && continue
  read s p _ < $r
  c=$(pgrep -P $p -f "[h]c_ha.py" | head -1)
  [ -n "$c" ] && { T=$t C=$c; break; }
done
if [ -z "$C" ]; then echo "no running shard in $S (hold set)"; echo "$(date -u +%FT%TZ) $S: no running shard" >> $LOG; exit 0; fi
echo "stopping $T python pid $C ($(ps -o etime= -p $C | tr -d ' ') running)"
kill -TERM $C
for _ in $(seq 1 30); do kill -0 $C 2>/dev/null || break; sleep 1; done
kill -0 $C 2>/dev/null && { kill -KILL $C; echo "SIGKILL needed"; }
for _ in $(seq 1 90); do [ -e $L/jobs/$T.done ] && break; sleep 1; done
echo "$T .done: $(cat $L/jobs/$T.done 2>/dev/null || echo none)"
mv $L/jobs/$T.done $L/jobs/$T.done.$X 2>/dev/null; mv $L/jobs/$T.run $L/jobs/$T.run.$X 2>/dev/null; rm -rf $L/claims/$T
echo "$(date -u +%FT%TZ) $T stopped (python pid $C; .done/.run renamed *.$X, claim removed: back in the queue)" >> $LOG
sleep 5
echo "card $G after: $(mem)"
