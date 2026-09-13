#!/bin/bash
# Stop a running scripts/render_wall_pilot.sh job: the driver loop and whichever
# mujoco_render / isaac_render worker it currently has in flight.
#
#   scripts/stack/stop_render.sh [pattern]
#
# pattern: optional substring of the job's <out_dir> (render_wall_pilot.sh's
# 2nd argument) to pick one job when more than one is running on this shared
# machine. Omit it only when exactly one render_wall_pilot.sh is running --
# with more than one and no pattern, this refuses rather than guess which one
# you own (the same "stop only the owned instance" rule as stop_instance.sh;
# render jobs carry no lock of their own to check against).
#
# Why not a plain `pkill -f render_wall_pilot.sh`: each isaac_render call is
# wrapped in `timeout`, which puts it in its OWN process group precisely so it
# can kill that whole subtree on its own deadline -- killing only the driver's
# group leaves that worker orphaned and still holding the GPU (observed
# 2026-09-13: the driver had already moved on to round 50's mujoco_render
# while round 48's isaac_render was still running under a different pgid).
# This stops every process group involved: SIGTERM first so Isaac/RTX gets a
# chance to release its GPU context, SIGKILL only for stragglers.
#
# Identification reads each candidate's REAL argv from /proc/<pid>/cmdline
# (NUL-separated), not the flattened, shell-requoted `ps args` column: a
# process is a driver/worker only if one of its OWN argv elements is EXACTLY
# (by basename) `render_wall_pilot.sh`, or the module name, not merely a
# substring somewhere in the line. A plain `pgrep -f` / `ps args ~ pattern`
# version of this script matched (and stopped) the unrelated wrapper shell
# running this script's own test harness, whose command line happened to
# quote "render_wall_pilot.sh" inside a `bash -c '...'` blob -- the exact
# pkill-self-match-in-shell-chains failure this codebase's other stop
# scripts already comment on, just hit here via `comm`+substring instead of
# `pgrep -f` directly. Found the hard way 2026-09-13; fixed by requiring an
# exact argv-element match instead of a text search.
set -uo pipefail

pattern="${1:-}"

argv() {  # $1 = pid; its real argv, one element per line (empty if gone)
    # 2>/dev/null must come BEFORE the `<` redirection: bash sets up a
    # command's redirections left to right and abandons the rest on the
    # first failure, so with `< file 2>/dev/null` a missing file's error
    # never reaches the /dev/null redirect that was meant to hide it.
    tr '\0' '\n' 2>/dev/null < "/proc/$1/cmdline"
}

# Pure-bash matching below, deliberately not `... | xargs basename | grep -q`:
# under `set -o pipefail`, `grep -q` exits the instant it sees a match, and
# xargs (which forks a fresh `basename` per argv line) can still be mid-write
# on a later line at that moment -- that write gets SIGPIPE, xargs reports
# exit 125, and pipefail turns a REAL match into a reported failure. Found
# the hard way 2026-09-13: a genuine `bash .../render_wall_pilot.sh` driver
# process failed `is_driver` intermittently, only inside the scan loop where
# timing made the race likely, never in a one-off manual check. `${a##*/}`
# (bash's own basename) and a `case` pattern for the substring test avoid the
# external pipeline entirely, so there is no early-exit reader to race.
is_driver() {  # $1 = pid; a real `bash .../render_wall_pilot.sh ...` invocation
    local all first=1 a
    all=$(argv "$1") || return 1
    while IFS= read -r a; do
        if [ "$first" = 1 ]; then
            [ "${a##*/}" = "bash" ] || return 1
            first=0
            continue
        fi
        [ "${a##*/}" = "render_wall_pilot.sh" ] && return 0
    done <<<"$all"
    return 1
}

is_worker() {  # $1 = pid; a real `python* -m locomani.sim.isaac.(mujoco_render|isaac_render) ...`
    local all first=1 a
    all=$(argv "$1") || return 1
    while IFS= read -r a; do
        if [ "$first" = 1 ]; then
            case "${a##*/}" in python*) ;; *) return 1 ;; esac
            first=0
            continue
        fi
        case "$a" in
            locomani.sim.isaac.mujoco_render | locomani.sim.isaac.isaac_render) return 0 ;;
        esac
    done <<<"$all"
    return 1
}

all_pids() { for f in /proc/[0-9]*/cmdline; do p=${f#/proc/}; echo "${p%/cmdline}"; done; }

matches_pattern() {  # $1 = pid; the optional --out-dir pattern, checked against real argv only
    [ -z "$pattern" ] && return 0
    local all
    all=$(argv "$1") || return 1
    case "$all" in *"$pattern"*) return 0 ;; esac
    return 1
}

drivers() { for p in $(all_pids); do is_driver "$p" && matches_pattern "$p" && echo "$p"; done; }
workers() { for p in $(all_pids); do is_worker "$p" && matches_pattern "$p" && echo "$p"; done; }

mapfile -t driver_pids < <(drivers)
mapfile -t worker_pids < <(workers)
if [ "${#driver_pids[@]}" -eq 0 ] && [ "${#worker_pids[@]}" -eq 0 ]; then
    echo "[stop_render] nothing matching render_wall_pilot.sh${pattern:+ ('$pattern')} is running"
    exit 0
fi
if [ "${#driver_pids[@]}" -gt 1 ] && [ -z "$pattern" ]; then
    echo "REFUSED: ${#driver_pids[@]} render_wall_pilot.sh jobs are running; pass part of one job's"
    echo "  --out directory as an argument to pick one, so this does not stop someone else's job:"
    for p in "${driver_pids[@]}"; do ps -o pid,etime,args --no-headers -p "$p" | sed 's/^/    /'; done
    exit 3
fi

pgids_of() {  # unique process-group ids owning the given PIDs (one per line)
    [ "$#" -eq 0 ] && return 0
    ps -o pgid= -p "$(IFS=,; echo "$*")" 2>/dev/null | tr -d ' '
}

mapfile -t groups < <(pgids_of "${driver_pids[@]}" "${worker_pids[@]}" | sort -u)
echo "[stop_render] stopping process group(s): ${groups[*]}"
# `ps -g` is unreliable across ps implementations (BSD "real group ID" vs
# "session ID" ambiguity); filter a full listing by the known-good pgid column
# instead of trusting it.
ps -eo pid,pgid,etime,args --no-headers | awk -v gs="${groups[*]}" '
    BEGIN { n = split(gs, g, " "); for (i = 1; i <= n; i++) want[g[i]] = 1 }
    want[$2] { print "    " $0 }'

for g in "${groups[@]}"; do kill -TERM -- "-$g" 2>/dev/null || true; done
sleep 3
for g in "${groups[@]}"; do
    kill -0 -- "-$g" 2>/dev/null && kill -KILL -- "-$g" 2>/dev/null
done
sleep 1

mapfile -t still < <(drivers; workers)
if [ "${#still[@]}" -gt 0 ]; then
    echo "[stop_render] WARNING: still running after SIGKILL: ${still[*]}"
    exit 1
fi
echo "[stop_render] stopped"
