#!/bin/bash
# Stop ONE instance's stack + its policy server (patterns built at runtime: pkill self-match trap).
k=${1:-0}; port=${2:-}
SP=$((5556 + 10*k))
for p in $(pgrep -f "$(printf 'run_sim_locomani.py --instance %s' "$k")" || true); do pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done   # children first: THIS instance's image publisher
for p in $(pgrep -f "$(printf 'g1_deploy_onnx_ref lo .*zmq-port %s ' "$SP")" || true); do kill -9 "$p" 2>/dev/null; done
[ "$k" = 0 ] && for p in $(pgrep -f "$(printf '%s_%s' run_data exporter)" || true); do kill -9 "$p" 2>/dev/null; done
[ -n "$port" ] && for p in $(pgrep -f "$(printf '%s_%s.py .*--port %s' serve policy "$port")" || true); do kill -9 "$p" 2>/dev/null; done
sleep 2
