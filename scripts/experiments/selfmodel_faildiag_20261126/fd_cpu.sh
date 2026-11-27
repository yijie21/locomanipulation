#!/usr/bin/env bash
# fd_cpu.sh: cores used by our faildiag processes (fd_run / fd_heft / fd_gt / our GraspGen-X), from /proc CPU time over 10 s
pids() { ps -eo pid,args | grep -E '[f]d_run.py|[f]d_heft.py|[f]d_gt.py|[h]g_ggx_server.py --port 18873' | grep -v 'bash -c' | awk '{print $1}'; }
tick() { for p in $(pids); do awk '{print $1, $14+$15}' /proc/$p/stat 2>/dev/null; done; }
a=$(tick); sleep 10; b=$(tick)
join <(echo "$a" | sort) <(echo "$b" | sort) | awk -v hz=$(getconf CLK_TCK) '{d+=$3-$2; n++} END {printf "%d processes, %.1f cores (CPU time over 10 s)\n", n, d/hz/10}'
