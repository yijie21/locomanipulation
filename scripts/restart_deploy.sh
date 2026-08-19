#!/bin/bash
# Restart the C++ deploy for one instance. The deploy keeps locomotion and
# heading state across SIM resets — round 1 of every fresh deploy is clean,
# rounds 2+ wander and fall — so the gym restarts it between rounds to make
# every round a "round 1". The TensorRT engines are cached after the first
# stack boot, so a restart takes ~20-40 s, not the first-run 10 min.
#
# Usage: INSTANCE=<i> bash scripts/restart_deploy.sh
# Prints "deploy restarted" on success (the gym greps for it).
set -u

INSTANCE="${INSTANCE:-0}"
export SONIC_DDS_DOMAIN=$INSTANCE
TF_SONIC_PORT=$((5556 + 10 * INSTANCE))
TF_OUT_PORT=$((5557 + 10 * INSTANCE))
if [ "$INSTANCE" = "0" ]; then LOGS=/workspace/tmp/p03; else LOGS=/workspace/tmp/p03_i$INSTANCE; fi

# shellcheck disable=SC1091
source /workspace/code/locomanipulation/scripts/env_deploy.sh

# Instance-selective kill: the port is part of the deploy's own cmdline and
# never part of this script's, so the pattern cannot self-match.
pkill -f "g1_deploy_onnx_ref lo .*zmq-port $TF_SONIC_PORT " 2>/dev/null
sleep 1
pkill -9 -f "g1_deploy_onnx_ref lo .*zmq-port $TF_SONIC_PORT " 2>/dev/null

SONIC_POLICY="${SONIC_POLICY:-policy/release}"
cd "$SONIC_DEPLOY_ROOT" || exit 1
echo "----- deploy restart $(date +%T) -----" >> "$LOGS/deploy.log"
mark=$(wc -l < "$LOGS/deploy.log")
nohup ./target/release/g1_deploy_onnx_ref lo "$SONIC_POLICY/model_decoder.onnx" reference/example/ \
    --obs-config "$SONIC_POLICY/observation_config.yaml" \
    --encoder-file "$SONIC_POLICY/model_encoder.onnx" \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager --output-type all --zmq-host localhost \
    --zmq-port "$TF_SONIC_PORT" --zmq-out-port "$TF_OUT_PORT" --disable-crc-check \
    >> "$LOGS/deploy.log" 2>&1 &

end=$((SECONDS + 300))
while [ $SECONDS -lt $end ]; do
    if tail -n +$((mark + 1)) "$LOGS/deploy.log" | grep -qi "Init Done"; then
        echo "deploy restarted"
        exit 0
    fi
    sleep 2
done
echo "deploy restart FAILED"
grep -viE "^INFO|^WARNING" "$LOGS/deploy.log" | tail -10
exit 1
