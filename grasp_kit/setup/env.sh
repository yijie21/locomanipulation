# Paths of the kit's environments, sources and weights. Sourced by run.sh and the setup scripts; set any of these
# before sourcing to put things elsewhere (e.g. WEIGHTS on a bigger disk).
KIT=${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ENVS=${ENVS:-$KIT/envs}
SRC=${SRC:-$KIT/third_party}
WEIGHTS=${WEIGHTS:-$KIT/weights}
export HF_HOME=${HF_HOME:-$WEIGHTS/huggingface}
export GRASPGENX_CHECKPOINT_DIR=${GRASPGENX_CHECKPOINT_DIR:-$WEIGHTS/graspgenx_checkpoints}
export GRASPGENX_GRIPPER_CFG_DIR=${GRASPGENX_GRIPPER_CFG_DIR:-$WEIGHTS/gripper_descriptions}
PERCEPTION_PY=${PERCEPTION_PY:-$ENVS/perception/bin/python}
GRASP_PY=${GRASP_PY:-$ENVS/grasp/bin/python}
# PyTorch's CUDA build: cu128 for RTX 50xx and drivers >= 570; cu126 for older drivers (>= 560)
TORCH_CUDA=${TORCH_CUDA:-cu128}
