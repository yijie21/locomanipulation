#!/usr/bin/env bash
# render_all.sh <npz dir> <out dir> [look]: every simple_bench.py --film state of <npz dir> drawn in SIMPLE's Isaac scene
# (isaac_film_vast.py), one process per task: env.close() ends the Kit app, so a second gym.make in the same process
# never runs. Two lanes at a time on one GPU (GPU, default 1). Logs: <out dir>/logs/render_<task>.log, one line per
# finished task in <out dir>/logs/done.txt.  MAIN: main's checkout, the owner of third_party/SIMPLE.
set -uo pipefail
E=$(cd "$(dirname "$0")" && pwd)
W=$(cd "$E/../../.." && pwd)
M=${MAIN:-/workspace/code/locomanipulation}
O=$W/outputs/simple_eval_20261112
NPZ=$(cd "$1" && pwd); mkdir -p "$2/logs"; OUT=$(cd "$2" && pwd)
LOOK=${3:--0.12,0,0.10}
GPU=${GPU:-1}
OLD=${SIMPLE_OLD_TASK:-$O/film_20261119/scripts/oldtask/g1_wholebody_xmove_bend_pick_teleop.py}
if [ ! -f "$OLD" ]; then  # SIMPLE dropped XMoveBendPickTeleop in eef474f; its episodes are from the version before
    mkdir -p "$(dirname "$OLD")"
    git -C $M/third_party/SIMPLE show eef474f^:src/simple/tasks/g1_wholebody_xmove_bend_pick_teleop.py > "$OLD"
fi
cd $M/third_party/SIMPLE || exit 1

one() {
    local task=$1
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TMPDIR=/workspace/tmp \
        timeout 7200 .venv/bin/python "$E/isaac_film_vast.py" "$O/x" "$OUT" "$NPZ"/"${task}"_L0_*.npz \
        --look="$LOOK" --old-task "$OLD" > "$OUT/logs/render_${task}.log" 2>&1
    echo "rc=$? $task $(date +%T)" >> "$OUT/logs/done.txt"
}

(one XMoveBendPickTeleop; one TabletopGraspMP) &
(one BendPickMP; one XMovePickTeleop) &
wait
echo "all done $(date +%T)" >> "$OUT/logs/done.txt"
