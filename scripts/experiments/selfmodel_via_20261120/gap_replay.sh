#!/usr/bin/env bash
# notebook: selfmodel_via_20261120 -- fourth step, part two (the plan entry's "第四步第二部分"): ScaleBFM replays of a gap
# route set on the 4090, one reset per (slot, pass); settings as onpol_dev_r2_pose / data_via_pose (the compensation's model
# and its defaults), the scene with the two mocap obstacles.
# gap_replay.sh <out dir> <gap routes dir> <seed> <ID0> <procs> <gpus> <obst 0|1> [<model dir> <pose|const>]
#   obst 1: the path's real obstacles stand in the simulation during the reach (REACH_GAP_OBST); 0: the open floor.
set -u
OUT=$1 RT=$2 SEED=$3 ID0=$4 PROCS=$5 GPUS=$6 OBST=$7 MD=${8:-} MODE=${9:-}
R=/wuji-vefps-D/wuji-il/yijie/ws/code/loco; cd $R
SC=${GAP_SCENE:-$R/outputs/scene/simple_g1_gap}; PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python; V=scripts/experiments/selfmodel_via_20261120
export BFM_ROOT=odo OMP_NUM_THREADS=1 MUJOCO_GL=egl SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE
N=$(python3 -c "import json; print(4 * len(json.load(open('$RT/slots.json'))['names']))")
mkdir -p $OUT/asrun && cp scripts/experiments/reachwm_20261030/gen_reach.py $V/via_comp.py $V/via_gen.py "$0" $RT/slots.json $OUT/asrun/
COMP=""; [ -n "$MD" ] && COMP="REACH_VIA_COMP=$MD REACH_VIA_COMP_MODE=$MODE"
GAP=""; [ "$OBST" = 1 ] && GAP="REACH_GAP_OBST=1"
{ echo "gap replay of $RT ($N resets = 4 passes x slots, seed $SEED, ID0 $ID0), mode ${MODE:-direct}${MD:+, model $MD}, obstacles $OBST; ScaleBFM; scene $SC"
  echo "commit $(git rev-parse HEAD), uncommitted: $(git status --short | tr '\n' ' ')"
  echo "started $(date -u +%FT%TZ) on the 4090, $PROCS processes on GPUs $GPUS, nice 10"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; } > $OUT/run_info.txt
env REACH_HAND_OFF=0.0415,0,0 REACH_SCENE_DIR=$SC REACH_GRASP=1 REACH_GRASP_ROOM=1 REACH_GRASP_SPEED=0.08 REACH_GRASP_CONT=0.5 \
    REACH_GRASP_IKREF=1 REACH_WRIST=1 REACH_VIA_REPLAY=$RT/via:$RT/prep REACH_VIA_SLOTS=$RT/slots.json REACH_ID0=$ID0 \
    REACH_OUT=$OUT $COMP $GAP nice -n 10 $PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs $PROCS --resets $N \
    --seed $SEED --gpus $GPUS > $OUT/run.log 2>&1
REACH_OUT=$OUT $PY scripts/experiments/reachwm_20261030/gen_reach.py merge > $OUT/merge.txt 2>&1
$PY -I $V/arm_spheres.py $OUT $SC > $OUT/arm.txt 2>&1
OMP_NUM_THREADS=2 $PY $V/via_model.py build $OUT > $OUT/build.txt 2>&1
echo "finished $(date -u +%FT%TZ) $(tail -1 $OUT/merge.txt)" >> $OUT/run_info.txt
