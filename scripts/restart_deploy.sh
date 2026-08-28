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
# shellcheck disable=SC1091
source "$(dirname "$0")/env_deploy.sh"        # also pulls in paths.sh
LOGS=$(locomani_session_root "$INSTANCE")

# Instance-selective kill: the port is part of the deploy's own cmdline and
# never part of this script's, so the pattern cannot self-match.
pkill -f "g1_deploy_onnx_ref lo .*zmq-port $TF_SONIC_PORT " 2>/dev/null
sleep 1
pkill -9 -f "g1_deploy_onnx_ref lo .*zmq-port $TF_SONIC_PORT " 2>/dev/null

SONIC_POLICY="${SONIC_POLICY:-policy/release}"
# NO_ENCODER=1 relaunches WITHOUT the encoder, mirroring start_stack.sh. This
# must be honoured HERE and not only at stack boot: grasp_gym restarts the
# deploy through this script every round, so a closed-loop VLA run that booted
# the stack token-only got its deploy silently swapped for an encoder-loaded
# one at the first round-2/stand-retry, which changes the deployment config
# mid-run. (The older claim here -- that an encoder-loaded deploy makes the
# token stream "never hold" -- was wrong; see the corrected note in
# start_stack.sh and docs/official_parity_audit.md C5. Honouring NO_ENCODER
# per round still matters: the run must not silently change form halfway.)
ENC_ARG=(--encoder-file "$SONIC_POLICY/model_encoder.onnx")
if [ "${NO_ENCODER:-0}" = "1" ]; then
    ENC_ARG=()
    echo "encoder: OFF (token-only / VLA deployment mode)"
fi
cd "$SONIC_DEPLOY_ROOT" || exit 1
echo "----- deploy restart $(date +%T) -----" >> "$LOGS/deploy.log"
mark=$(wc -l < "$LOGS/deploy.log")
nohup ./target/release/g1_deploy_onnx_ref lo "$SONIC_POLICY/model_decoder.onnx" reference/example/ \
    --obs-config "$SONIC_POLICY/observation_config.yaml" \
    "${ENC_ARG[@]}" \
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
