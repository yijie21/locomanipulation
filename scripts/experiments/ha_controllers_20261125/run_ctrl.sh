#!/usr/bin/env bash
# notebook: skip ha controller integration smoke
# One HumanoidArena smoke process with the controller chosen by CTRL (the switch):
#   CTRL=sonic            HA's own probe_sonic.py, unchanged (native SONIC path), via launch_sonic.py (GPU pinning only)
#   CTRL=scalebfm|heft    probe_ctrl.py (g1_backend's ScaleBFM / HEFT, LocoManiLab's control path)
#   GPU=2 CTRL=heft OUT=$R/probe/ctrl_heft PHASES=actuators,kin,stand,lag,reach,walk,seed bash run_ctrl.sh
# GPU is the physical index; only that card is made visible (CUDA_VISIBLE_DEVICES) and the process sees it as cuda:0.
# Same task, env-config YAML, seeds and SONIC arguments as run_probe.sh, so the env is built identically.
set -euo pipefail
R=/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena
HERE=$(cd "$(dirname "$0")" && pwd)
source $R/env.sh
cd $ISAACLAB_ROOT
export PYTHONPATH=$ISAACLAB_ROOT PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID ENABLE_DEPTH=${ENABLE_DEPTH:-1}
export SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
GPU=${GPU:?set GPU to the physical card}
export CUDA_VISIBLE_DEVICES=$GPU HA_PHYS_GPU=$GPU
CTRL=${CTRL:?set CTRL to sonic, scalebfm or heft}
TASK_YAML=${TASK_YAML:-tasks/common_test_config/base_test/pp_box_sonic_test.yaml}
OUT=${OUT:-$R/probe/ctrl_$CTRL}
TASK=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['task_name'])" $TASK_YAML)
EPISODE_SEED=$(python -c "import hashlib,sys; p=f'{sys.argv[1]}|0|0'.encode(); print(int.from_bytes(hashlib.sha256(p).digest()[-4:],'big')&0x7FFFFFFF)" "$TASK")
mkdir -p $OUT
echo "ctrl=$CTRL task=$TASK episode_seed=$EPISODE_SEED gpu=$GPU out=$OUT"
COMMON=(--task "$TASK" --env_config_yaml "$ISAACLAB_ROOT/$TASK_YAML"
  --seed 0 --repeat_idx 0 --episode_seed "$EPISODE_SEED"
  --sonic_encoder_path "$SONIC_POLICY_ROOT/model_encoder.onnx"
  --sonic_decoder_path "$SONIC_POLICY_ROOT/model_decoder.onnx"
  --lerobot_server_url http://127.0.0.1:9
  --headless --device cuda:0 --verbose_startup --probe_out "$OUT")
case "$CTRL" in
  sonic) exec nice -n 10 python $HERE/launch_sonic.py "${COMMON[@]}" --phases "${PHASES:-stand,seed}" "$@" ;;
  scalebfm|heft) exec nice -n 10 python $HERE/probe_ctrl.py "${COMMON[@]}" --controller "$CTRL" \
      --phases "${PHASES:-actuators,kin,stand,lag,reach,walk,seed}" "$@" ;;
  *) echo "CTRL must be sonic, scalebfm or heft" >&2; exit 2 ;;
esac
