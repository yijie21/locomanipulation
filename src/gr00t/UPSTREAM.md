# Provenance

Vendored copy of the `gr00t` python package from NVIDIA's Isaac-GR00T.

    upstream repo    https://github.com/NVIDIA/Isaac-GR00T
    commit           376ba890cff8c9de64d71d982772a9c36185fdd7  (2026-08-10)
    copied from      third_party/Isaac-GR00T/gr00t  (clean, no local diffs at copy time)

## Why it lives here

`third_party/` is gitignored -- each tree there is an upstream clone, and a
change made inside one is invisible to this repo. That was fine while GR00T was
only being *run*; it stops being fine the moment training code is *modified*.
This copy is the one that is imported, edited, and committed.

The venv stays at `third_party/Isaac-GR00T/.venv` (an environment is not
code). Its editable-install `.pth` is pointed at `src/` by
`scripts/link_gr00t_src.sh` -- run that once after creating the venv, and
verify with:

    third_party/Isaac-GR00T/.venv/bin/python -c "import gr00t; print(gr00t.__file__)"
    # must print .../src/gr00t/__init__.py

`scripts/train_bottle.sh` performs the same check before every run and refuses
to train against the third_party copy, because the failure is otherwise
silent: same package name, same version, stale code.

## Diffing against upstream

    git -C third_party/Isaac-GR00T fetch
    diff -ru third_party/Isaac-GR00T/gr00t src/gr00t --exclude=__pycache__ --exclude=UPSTREAM.md

To pull an upstream fix: update the clone, diff, apply what is wanted, note it
below.

## Local changes since the copy

- 2026-08-25  configs/finetune_config.py, experiment/launch_finetune.py:
  expose `optim` as a FinetuneConfig field (launch_finetune hardcoded
  "adamw_torch"). Motivation: "adamw_bnb_8bit" shrinks Adam state 13 GB ->
  3.2 GB, letting the full 1.62B-trainable finetune fit ONE 32 GB GPU
  (SINGLE_GPU=1 in scripts/train_bottle.sh).
