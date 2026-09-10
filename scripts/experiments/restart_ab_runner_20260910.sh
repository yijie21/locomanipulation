#!/bin/bash
# Round 2 of the paired A/B, on a quiet machine: A = restart_deploy=true (default),
# Bp = restart_deploy=false + spawn-phase reanchor. Same case banks as round 1
# (cases_0..2.json), interleaved A0 Bp0 A1 Bp1 A2 Bp2. Refuses to start an arm when
# the container's CPU quota has < 6 cores free or was throttled in the last 5 s.
set -u
cd /workspace/code/locomanipulation
. scripts/paths.sh
E=/workspace/code/locomanipulation/outputs/experiments/restart_ab_20260910
export INSTANCE=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 SIM_VIDEO_DIR=$E/sim_video
log() { echo "$(date +%T) $*" | tee -a "$E/runner2.log"; }
cpu_free() {  # prints free cores (quota - 5 s usage) and throttled ms
    local q u0 t0 u1 t1
    q=$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max); u0=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat); t0=$(awk '/^throttled_usec/{print $2}' /sys/fs/cgroup/cpu.stat)
    sleep 5
    u1=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat); t1=$(awk '/^throttled_usec/{print $2}' /sys/fs/cgroup/cpu.stat)
    echo "$(( q/100000 - (u1-u0)/5000000 )) $(( (t1-t0)/1000 ))"
}
COMMON=(--reach-config "$E/reach_config.json" --targets water_bottle --rounds 10 --stand-retries 0 --record
        --video-include-init --spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 20.0 --spawn-lateral 0.2 --edge-jitter 0.2)
run_arm() {  # $1=arm(A|Bp) $2=batch
    local arm=$1 b=$2 restart=True; [ "$arm" = Bp ] && restart=False
    read -r free thr < <(cpu_free)
    log "batch $b arm $arm: cpu free $free cores, throttled ${thr} ms"
    if [ "$free" -lt 6 ] || [ "$thr" -gt 0 ]; then log "batch $b arm $arm: REFUSED (cpu)"; return 1; fi
    bash scripts/stack/stop_instance.sh 0 > /dev/null 2>&1
    log "batch $b arm $arm: start stack"
    if ! bash scripts/stack/start_stack.sh "pick up the water bottle" > "$E/stack2_${arm}${b}.log" 2>&1; then
        log "batch $b arm $arm: STACK FAILED"; return 1; fi
    log "batch $b arm $arm: collect (restart_deploy=$restart)"
    local t0=$(date +%s)
    PYTHONPATH=src "$VENV_SIM_PY" -u -c "
import sys
from locomani.tasks.bottle import grasp_gym
grasp_gym.RESTART_DEPLOY = $restart
sys.argv = ['grasp_gym.py'] + sys.argv[1:]
grasp_gym.main()
" "${COMMON[@]}" --seed $((29000 + b)) --cases "$E/cases_$b.json" --manifest "$E/manifest2_${arm}${b}.jsonl" > "$E/gym2_${arm}${b}.log" 2>&1
    log "batch $b arm $arm: rc=$? in $(( $(date +%s) - t0 ))s  $(grep -E '^\[gym\] [0-9]+/' "$E/gym2_${arm}${b}.log" | tail -1)"
    bash scripts/stack/stop_instance.sh 0 > /dev/null 2>&1
}
( while true; do echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) $(cpu_free)" >> "$E/cpu2.log"; sleep 25; done ) &
SAMPLER=$!
for b in 0 1 2; do run_arm A $b; run_arm Bp $b; done
kill $SAMPLER 2>/dev/null
log "AB2_DONE"
