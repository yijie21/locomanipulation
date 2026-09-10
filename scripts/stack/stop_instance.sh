#!/bin/bash
# Stop ONE instance's stack + its policy server. Matches on the process's own comm/args and
# environment, never on a substring of an arbitrary command line: pgrep -f also matches any
# shell whose command line merely quotes the pattern (memory pkill-self-match-in-shell-chains),
# and the old unscoped `run_data_exporter` pattern reached every instance's exporter.
# Killing the stack also releases start_stack.sh's per-instance lock (held via an inherited fd).
k=${1:-0}; port=${2:-}
SP=$((5556 + 10*k)); CP=$((5555 + 10*k))
pyprocs() {  # $1 = args regex; only processes whose comm is python*
    ps -eo pid,comm,args --no-headers | awk -v pat="$1" '$2 ~ /^python/ && $0 ~ pat {print $1}'
}
for p in $(pyprocs "run_sim_locomani[.]py --instance $k( |$)"); do pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done   # children first: THIS instance's image publisher
for p in $(pgrep -f "$(printf 'g1_deploy_onnx_ref lo .*zmq-port %s ' "$SP")" || true); do kill -9 "$p" 2>/dev/null; done
for p in $(pyprocs "run_data_exporter[.]py|run_instance_exporter[.]py"); do   # only the exporter that serves THIS instance's camera port
    grep -qz "TF_CAMERA_PORT=$CP" "/proc/$p/environ" 2>/dev/null && kill -9 "$p" 2>/dev/null
done
[ -n "$port" ] && for p in $(pyprocs "serve_policy[.]py .*--port $port( |$)"); do kill -9 "$p" 2>/dev/null; done
sleep 2
