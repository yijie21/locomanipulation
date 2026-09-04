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
# LOCKSTEP=1: SIMPLE's simulation-clock patch (--sim-lockstep), built from a
# copy of their tree plus our SONIC_DDS_DOMAIN patch (/workspace/deps/
# gear_sonic_deploy_lockstep). Same protocol v4, same policy/release files
# (md5-identical decoder); the binary only ticks when the sim hands it an
# envelope-stamped LowState. See lockstep_sim.py.
# SONIC_DEPLOY_BIN: run a different build of the deploy WITHOUT lockstep
# (e.g. the patched SIMPLE build in wall-clock mode, so a lockstep-vs-
# wall-clock calibration compares the same binary).
DEPLOY_BIN="${SONIC_DEPLOY_BIN:-./target/release/g1_deploy_onnx_ref}"
if [ -n "${SONIC_DEPLOY_BIN:-}" ]; then
    # SONIC_DEPLOY_LIBS=simple: SIMPLE's TRT 10.9 libs (its own tree only);
    # our builds under /workspace/deps carry their engine caches for the
    # toolchain they were built with, and the wrong libs fail with
    # "Serialization assertion stdVersionRead == kSERIALIZATION_VERSION".
    [ "${SONIC_DEPLOY_LIBS:-}" = "simple" ] && \
    export LD_LIBRARY_PATH="$HOME/tools/zmq-install/lib:$HOME/tools/onnxruntime-linux-x64-gpu-1.22.0/lib:$HOME/tools/tensorrt-root/lib:${LD_LIBRARY_PATH:-}"
    echo "  deploy binary: $DEPLOY_BIN"
fi
LOCKSTEP_ARG=()
if [ "${CLEAN_DEPLOY:-0}" = "1" ] && [ "${LOCKSTEP:-0}" != "1" ]; then
    # Wall-clock "clean deploy" (docs/lockstep.md 校准诊断): the lockstep-port
    # build with NVIDIA's default_angles, engaged by SONIC_AUTO_START, never
    # sent a planner command. One planner hold before enter_streamed leaves
    # the deploy executing walking tokens at ~1/3 amplitude.
    CD_ROOT="${CLEAN_DEPLOY_ROOT:-/workspace/deps/gear_sonic_deploy_ours_lockstep_nvdef}"
    DEPLOY_BIN="$CD_ROOT/target/release/g1_deploy_onnx_ref"
    LOCKSTEP_ARG=(--default-motion neutral_kick_R_001__A543)
    export SONIC_AUTO_START=1
    echo "  CLEAN_DEPLOY: $DEPLOY_BIN (auto-start, no planner)"
fi
if [ "${LOCKSTEP:-0}" = "1" ]; then
    # Default: OUR deploy source + SIMPLE's lockstep commits (branch
    # lockstep-port in third_party/GR00T-WholeBodyControl), built with our
    # TensorRT/onnxruntime under /workspace/deps -- so the lockstep binary is
    # behaviourally the production deploy. LOCKSTEP_DEPLOY_ROOT=/workspace/
    # deps/gear_sonic_deploy_lockstep (+ LOCKSTEP_DEPLOY_LIBS=simple) selects
    # the copy of SIMPLE's own tree instead.
    LS_ROOT="${LOCKSTEP_DEPLOY_ROOT:-/workspace/deps/gear_sonic_deploy_ours_lockstep}"
    DEPLOY_BIN="$LS_ROOT/target/release/g1_deploy_onnx_ref"
    # --default-motion: under lockstep there is no planner thread, so the
    # reference-motion features in the policy observation come from
    # current_motion_ frame 0 -- which is the FIRST clip in reference/example
    # (a dance) unless a default is named. SIMPLE's choice; its frame 0 is a
    # neutral stance, the lockstep counterpart of the planner's idle pose.
    LOCKSTEP_ARG=(--sim-lockstep --default-motion neutral_kick_R_001__A543)
    # No planner thread under lockstep, and the zmq 'start' command only
    # takes effect inside the planner-mode handler -- so the policy is
    # engaged the way SIMPLE does it: auto-start on the first
    # WAIT_FOR_CONTROL tick after the 3 s init ramp (one-shot; session
    # resets keep CONTROL state).
    export SONIC_AUTO_START=1
    if [ "${LOCKSTEP_DEPLOY_LIBS:-}" = "simple" ]; then
        export LD_LIBRARY_PATH="$HOME/tools/zmq-install/lib:$HOME/tools/onnxruntime-linux-x64-gpu-1.22.0/lib:$HOME/tools/tensorrt-root/lib:${LD_LIBRARY_PATH:-}"
    fi
    echo "  LOCKSTEP: $DEPLOY_BIN --sim-lockstep"
fi
# A foreign build runs from ITS tree: the TensorRT engine caches next to the
# policy (policy/release/*.trt) are serialized by the TensorRT the binary was
# linked against (10.9 there, 10.13 here) and do not cross over; planner and
# reference clips resolve relative to cwd as well. The policy files that
# matter (observation_config.yaml, model_decoder.onnx) are md5-identical.
case "$DEPLOY_BIN" in
    /*) cd "$(dirname "$DEPLOY_BIN")/../.." || exit 1
        echo "  deploy cwd: $PWD" ;;
esac
nohup "$DEPLOY_BIN" lo "$SONIC_POLICY/model_decoder.onnx" reference/example/ \
    --obs-config "$SONIC_POLICY/observation_config.yaml" \
    "${ENC_ARG[@]}" \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager --output-type all --zmq-host localhost \
    --zmq-port "$TF_SONIC_PORT" --zmq-out-port "$TF_OUT_PORT" --disable-crc-check "${LOCKSTEP_ARG[@]}" ${DEPLOY_EXTRA_ARGS:-} \
    >> "$LOGS/deploy.log" 2>&1 &

end=$((SECONDS + 300))
while [ $SECONDS -lt $end ]; do
    if tail -n +$((mark + 1)) "$LOGS/deploy.log" | grep -qiE "Init Done|simulation-clock control thread active"; then
        if [ "${CLEAN_DEPLOY:-0}" = "1" ] && [ "${LOCKSTEP:-0}" != "1" ]; then
            # auto-start engages ~3 s after Init Done; the round must not
            # stream before the policy is in CONTROL
            tail -n +$((mark + 1)) "$LOGS/deploy.log" | grep -q "transitioning to CONTROL state" || { sleep 1; continue; }
        fi
        echo "deploy restarted"
        exit 0
    fi
    sleep 2
done
echo "deploy restart FAILED"
grep -viE "^INFO|^WARNING" "$LOGS/deploy.log" | tail -10
exit 1
