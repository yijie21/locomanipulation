#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# Reopens the collection's slot 0a on card 0 (coordinator 2026-11-26, after the guard closed it at 07:14:49Z): read-only
# until card 0 has met all three, without a break, for 30 minutes (a sample every 20 s; one failing sample restarts the
# clock):
#   (1) utilization < 5 %;
#   (2) no compute entry on card 0 other than the allowed ones;
#   (3) the memory of every entry other than our SAM 3 <= 1536 MiB.
# Allowed entries: SAM 3's (arg 1, nvidia-smi's host pid; the container cannot map host pids, so it is the 4678 MiB entry
# present since before the collection) -- strict; or SAM 3's and the entries in arg 2 (present at the guard's first
# BASE 6207 MiB: the 1458 MiB entry, most likely the bigym cuRobo plan server, not ours) -- tolerant. Both clocks run;
# the strict one acts unless $L/card0_reopen_tolerant exists (the coordinator's choice). Keepalive is not on card 0.
# Then: the guard started with BASE = card 0's total now (no shard of ours yet: it adopts the first new entry), hold_0a
# removed, card0_ok written; "REOPENED" in the log; exits. Never touches any process. Log: $L/card0_reopen.log.
#   nohup bash hd_card0_reopen.sh <SAM 3 entry> "<tolerated entries>" > /dev/null 2>&1 < /dev/null &
set -uo pipefail
SAM=$1 TOL=" ${2:-} "
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/logs
G=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/sched/hd_card0_guard2.sh
NEED=1800
U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
log() { echo "$(date -u +%FT%TZ) $*" >> $L/card0_reopen.log; }
log "start pid $$: SAM 3 entry $SAM, tolerated:$TOL; strict acts unless $L/card0_reopen_tolerant exists"
t_s=$(date +%s) t_t=$(date +%s) last=0
while :; do
  now=$(date +%s)
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v u=$U '$1 == u {print $2, $3}')
  oth=$(echo "$apps" | awk -v s=$SAM 'NF && $1 != s {print $1}' | tr '\n' ' ')
  othm=$(echo "$apps" | awk -v s=$SAM 'NF && $1 != s {m += $2} END {print m + 0}')
  foreign_s=$(for p in $oth; do echo $p; done | wc -l)
  foreign_t=$(for p in $oth; do [[ "$TOL" == *" $p "* ]] || echo $p; done | wc -l)
  ok_c=1; [ "$u" -lt 5 ] && [ "$othm" -le 1536 ] || ok_c=0
  why="util $u %, entries other than SAM 3: [${oth% }] $othm MiB"
  if [ $ok_c = 1 ] && [ $foreign_s = 0 ]; then :; else [ $((now - t_s)) -ge 60 ] && log "strict clock restarts after $((now - t_s)) s: $why"; t_s=$now; fi
  if [ $ok_c = 1 ] && [ $foreign_t = 0 ]; then :; else [ $((now - t_t)) -ge 60 ] && log "tolerant clock restarts after $((now - t_t)) s: $why"; t_t=$now; fi
  [ $((now - last)) -ge 300 ] && { log "quiet for: strict $((now - t_s)) s, tolerant $((now - t_t)) s ($why)"; last=$now; }
  mode=strict q=$((now - t_s)); [ -e $L/card0_reopen_tolerant ] && { mode=tolerant; q=$((now - t_t)); }
  if [ $q -ge $NEED ]; then
    if pgrep -f "[h]d_card0_guard" > /dev/null; then log "a guard is already running; not reopening"; exit 1; fi
    BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    nohup bash $G $BASE > /dev/null 2>&1 < /dev/null &
    gp=$!
    sleep 2
    rm -f $L/hold_0a; touch $L/card0_ok
    log "REOPENED ($mode, quiet $q s): guard pid $gp with BASE $BASE MiB; hold_0a removed, card0_ok written ($why)"
    exit 0
  fi
  sleep 20
done
