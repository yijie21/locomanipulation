#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# Opens 4090b's slot b0b (gate_b0b) once 4090b's CPU allows a fourth HA process (coordinator 2026-11-26): read-only until
# one of two clocks reaches 10 minutes without a break (a sample every 30 s; one failing sample restarts it):
#   loadavg  /proc/loadavg's 1-minute value < 38 (the coordinator's rule; note: the node's, shared with the 4090's pod);
#   own      4090b's own CPU use (its processes' CPU time over the sample, /proc/[pid]/stat) + 2.5 cores < OWN_MAX
#            (arg 2, default 30: the coordinator's 07:58Z decision, room for SONIC's second-look CPU runs, up to 12).
# The loadavg clock acts unless $L/b0b_gate_own exists (the coordinator's choice). Gives up (no gate) when the 4090b
# scheduler has nothing left to claim or is gone. Never touches any process. Log: $L/b0b_gate.log.
#   nohup bash hd_b0b_gate.sh <4090b scheduler pid> [OWN_MAX] > /dev/null 2>&1 < /dev/null &
set -uo pipefail
SCH=$1 OWN_MAX=${2:-30}
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/logs
NEED=600
HZ=$(getconf CLK_TCK)
log() { echo "$(date -u +%FT%TZ) $*" >> $L/b0b_gate.log; }
tot() { awk '{s += $14 + $15 + $16 + $17} END {print s}' /proc/[0-9]*/stat 2>/dev/null; }
log "start pid $$: loadavg < 38 acts unless $L/b0b_gate_own exists; own use + 2.5 < $OWN_MAX tracked too"
t_l=$(date +%s) t_o=$(date +%s) last=0 a=$(tot) ta=$(date +%s.%N)
while :; do
  sleep 30
  [ -e $L/gate_b0b ] && { log "gate_b0b already there; exit"; exit 0; }
  kill -0 $SCH 2>/dev/null || { log "4090b scheduler $SCH gone; no gate; exit"; exit 0; }
  grep -q "nothing left to claim" $L/formal_4090b.log && { log "4090b: nothing left to claim; no gate; exit"; exit 0; }
  b=$(tot); tb=$(date +%s.%N); now=$(date +%s)
  own=$(python3 -c "print('%.1f' % (($b - $a) / $HZ / ($tb - $ta)))"); a=$b; ta=$tb
  la=$(cut -d' ' -f1 /proc/loadavg)
  python3 -c "import sys; sys.exit(0 if $la < 38 else 1)" || { [ $((now - t_l)) -ge 60 ] && log "loadavg clock restarts after $((now - t_l)) s: loadavg $la"; t_l=$now; }
  python3 -c "import sys; sys.exit(0 if $own + 2.5 < $OWN_MAX else 1)" || { [ $((now - t_o)) -ge 60 ] && log "own clock restarts after $((now - t_o)) s: own $own cores"; t_o=$now; }
  [ $((now - last)) -ge 300 ] && { log "quiet for: loadavg $((now - t_l)) s, own $((now - t_o)) s (loadavg $la, own $own cores)"; last=$now; }
  mode=loadavg q=$((now - t_l)); [ -e $L/b0b_gate_own ] && { mode=own; q=$((now - t_o)); }
  if [ $q -ge $NEED ]; then
    touch $L/gate_b0b
    log "OPENED ($mode, $q s): gate_b0b written (loadavg $la, own $own cores)"
    echo "$(date -u +%FT%TZ) gate_b0b set by hd_b0b_gate.sh ($mode rule, $q s)" >> $L/formal_4090b.log
    exit 0
  fi
done
