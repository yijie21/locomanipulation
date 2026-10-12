#!/bin/bash
# TODO 21.3.4: start / stop the two helper servers of the graspnet comparison (pid files, no pkill -f).
#   bash scripts/experiments/graspnet_bottle_20261008/servers.sh start|stop|restart [graspnet|curobo] [--gpu 1]
set -u
cd "$(dirname "$0")/../../.." || exit 1
L=outputs/graspnet_bottle_20261008/logs
mkdir -p "$L"
ACTION=${1:-start}
WHICH=${2:-all}
GPU=${GPU:-1}
start_one() {
    case $1 in
        graspnet) cmd=(third_party/graspnet-baseline/.venv/bin/python -u scripts/experiments/graspnet_bottle_20261008/graspnet_server.py --port 5790 --gpu "$GPU") ;;
        curobo) cmd=(env CUDA_VISIBLE_DEVICES="$GPU" third_party/SIMPLE/.venv/bin/python -u scripts/experiments/graspnet_bottle_20261008/curobo_server.py outputs/graspnet_bottle_20261008/curobo/g1_right_curobo.json --port 5791) ;;
    esac
    PYTHONUNBUFFERED=1 nohup setsid "${cmd[@]}" > "$L/$1_server.log" 2>&1 < /dev/null &
    echo $! > "$L/$1_server.pid"
    for _ in $(seq 1 60); do grep -q "listening" "$L/$1_server.log" && break; sleep 1; done
    tail -1 "$L/$1_server.log"
}
stop_one() {
    [ -f "$L/$1_server.pid" ] && kill "$(cat "$L/$1_server.pid")" 2>/dev/null
    rm -f "$L/$1_server.pid"
}
for s in graspnet curobo; do
    [ "$WHICH" = all ] || [ "$WHICH" = "$s" ] || continue
    case $ACTION in
        start) start_one $s ;;
        stop) stop_one $s ;;
        restart) stop_one $s; sleep 2; start_one $s ;;
    esac
done
