#!/bin/bash
# Free CPU cores under this container's cgroup quota, measured over a window.
#
#   bash scripts/cpu_headroom.sh [seconds=5]      # one reading
#   watch -n 5 bash scripts/cpu_headroom.sh 3     # live
#
# `nproc` (32) and the load average describe the HOST; the container is capped by
# /sys/fs/cgroup/cpu.max (15.36 cores on this instance). When the quota is
# saturated the MuJoCo sim drops below real time and every wall-clock expert
# fails as control errors (2026-09-10: sim RTF 0.18, all arms 0/10). A recording
# stack needs ~5 cores; do not start one with fewer than 6 free or any throttling.
set -u
W="${1:-5}"
read -r quota period < /sys/fs/cgroup/cpu.max
u0=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat); t0=$(awk '/^throttled_usec/{print $2}' /sys/fs/cgroup/cpu.stat)
sleep "$W"
u1=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat); t1=$(awk '/^throttled_usec/{print $2}' /sys/fs/cgroup/cpu.stat)
awk -v q="$quota" -v p="$period" -v du="$((u1 - u0))" -v dt="$((t1 - t0))" -v w="$W" 'BEGIN {
    cap = (q == "max") ? -1 : q / p; used = du / (w * 1e6);
    if (cap < 0) printf "quota: none  used %.2f cores  throttled %d ms\n", used, dt / 1000;
    else printf "quota %.2f cores  used %.2f  free %.2f  throttled %d ms  %s\n", cap, used, cap - used, dt / 1000,
        (cap - used >= 6 && dt == 0) ? "OK for a stack" : "NOT enough for a stack";
}'
