#!/bin/bash
# Finetune GR00T on the dataset a run converted.
#
#   bash scripts/train.sh outputs/runs/bottle_simple_200 --gpu 1
#   bash scripts/train.sh outputs/runs/bottle_simple_200 --gpu 1 --dry-run
#
# No stack, one card (or two, if the run's train block asks for two). Serves both lines: what
# differs between single-arm and bimanual is the dataset, which the run already names.
#
# The recipe -- the optimizer chosen from the card, the single-card batch, the squatter preflight,
# the vendored-gr00t check -- lives in the python entry, not here. Until 2026-09-21 it lived in
# scripts/train_bottle.sh and was driven through the environment by scripts/train.py: python
# calling bash calling python. The assembled torchrun command is byte-identical across that move
# for the single-card, two-card and 40 GB-card branches.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <run-dir> [--gpu N] [--dry-run] [extra launch_finetune args]" >&2; exit 1; }

export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_SIM_PY" -m locomani.pipeline.run_train "$@"
