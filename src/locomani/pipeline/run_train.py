#!/usr/bin/env python3
"""Finetune GR00T on the dataset this run converted: the whole recipe, then torchrun.

    python -m locomani.pipeline.run_train outputs/runs/bottle_simple_200 --gpu 1
    python -m locomani.pipeline.run_train outputs/runs/bottle_simple_200 --gpu 1 --dry-run

`scripts/train.sh` is one line over this. Ported from `scripts/train_bottle.sh` on 2026-09-21,
which the old `scripts/train.py` used to drive through the environment -- python calling bash
calling python, with the run's `train` block translated into variables on the way. The recipe is
the same one; every choice below still carries the reason it was made.

Slot 11 (unitree_g1_sonic) is EMPTY in nvidia/GR00T-N1.7-3B -- only 24/25/26/27 carry trained
projector weights -- so `--tune-projector` is not optional: without it the state/action embedding
stays at its 0.02*randn init. The vision and language towers stay frozen: 200 episodes of one
scene with a fixed appearance seed is exactly the data that would replace a general visual encoder
with a memory of this orange table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context

ROOT = Path(paths.REPO)
GR00T = Path(paths.THIRD_PARTY) / "Isaac-GR00T"
PY = GR00T / ".venv" / "bin" / "python"
TORCHRUN = GR00T / ".venv" / "bin" / "torchrun"
BIG_CARD_MIB = 40000  # 40 GB+: A100/H100 class, where fp32 Adam has nothing to gain from 8-bit


def require_vendored_gr00t() -> None:
    """The venv lives with the upstream clone; the PACKAGE must be the vendored copy in src/.

    Not optional paranoia: the failure mode is a stale third_party copy imported under the same
    name and version, which trains to completion on the wrong code.
    """
    where = subprocess.run(
        [str(PY), "-c", "import gr00t; print(gr00t.__file__)"], capture_output=True, text=True
    ).stdout.strip()
    if not where.startswith(str(ROOT / "src" / "gr00t") + "/"):
        raise SystemExit(f"gr00t resolves to {where}, not src/gr00t --\nrun: bash scripts/link_gr00t_src.sh")


def gpu_probe(visible: str | None) -> tuple[int, int, int]:
    """(cards, smallest total MiB, smallest free MiB) for the cards THIS run will use.

    Scoped with -i because nvidia-smi does not honour CUDA_VISIBLE_DEVICES: without it a run
    pinned to a free GPU 0 is refused because somebody else's job sits on GPU 1, which is exactly
    what blocked the cover400 pair.
    """
    argv = ["nvidia-smi"]
    if visible:
        argv += ["-i", str(visible)]
    argv += ["--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"]
    try:
        rows = subprocess.run(argv, capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        pairs = [[int(x) for x in row.replace(",", " ").split()] for row in rows if row.strip()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0, 0, 0
    if not pairs:
        return 0, 0, 0
    return len(pairs), min(p[0] for p in pairs), min(p[1] for p in pairs)


def choose_optimizer(vram_min: int, *, pinned: str | None = None, auto: bool = True) -> tuple[str, str]:
    """fp32 Adam keeps m+v for 1.62B trainable params = 13 GB, which does not fit a 32 GB card.

    Quantising the states (adamw_bnb_8bit, 13 -> 3.2 GB) is a MEMORY choice, not an optimization
    one: effective batch, lr and step count are untouched, so the run stays the same experiment.
    On 40 GB+ cards there is nothing to buy, so stay on fused fp32 Adam, which is faster.
    """
    if pinned:
        return pinned, "pinned by OPTIM="
    if not auto:
        return "adamw_torch", "OPTIM_AUTO=0"
    if vram_min <= 0:
        return "adamw_torch", "could not read nvidia-smi, assuming headroom"
    if vram_min < BIG_CARD_MIB:
        return "adamw_bnb_8bit", f"auto: smallest card {vram_min} MiB < 40 GB"
    return "adamw_torch", f"auto: smallest card {vram_min} MiB >= 40 GB"


def squatter(vram_min: int, free_min: int, squat_max: int) -> str | None:
    """Preflight the FREE memory, not just the total.

    This is the failure that cost round 3 a training run: a healthy-looking box, both cards
    present, and 6.5 GB of somebody else's process on GPU 0. Better to refuse here than to OOM
    twenty minutes into model loading.
    """
    used = vram_min - free_min
    if vram_min > 0 and used > squat_max:
        return (
            f"REFUSING TO START: a GPU already has {used} MiB in use\n"
            f"  (smallest card: {free_min} of {vram_min} MiB free; limit is {squat_max})\n"
            f"  Training wants essentially the whole card. Stop whatever is holding it\n"
            f"  (the collection stack? bash scripts/stop_stack.sh --instance N) and re-run, or\n"
            f"  raise train.squat_max in the experiment if you know the residue is harmless."
        )
    return None


def batch_plan(gpus: int, ngpu: int, global_batch: int, grad_accum: int) -> tuple[int, int, int, str | None]:
    """One card changes the batch, whatever said so: GPUS=1, a pinned device, or a one-GPU box.

    Passing gpus=1 alone used to keep the two-card global batch of 64 and OOM at step 1 after ~18 s
    of model loading (cover400_tok, 2026-09-05). Per-device 8 is measured, not derived: at 32 the
    dual-camera VLM forward OOMed at a 32.1 GB peak, at 16 it ran but peaked at 31.66/32.6 GB --
    950 MiB of headroom, which a 10k-step run eats in fragmentation and checkpoint writes.
    Accumulation rises to match, so the optimizer still steps on the same effective 64.
    """
    if gpus == 1 or ngpu == 1:
        return 1, 8, 8, "single card"
    return gpus, global_batch, grad_accum, None


def dataset_of(run_dir: Path) -> Path:
    summary = run_dir / "summary.json"
    entry = json.loads(summary.read_text()).get("convert") if summary.exists() else None
    if not entry or not entry.get("train_dataset"):
        raise SystemExit(f"{run_dir}/summary.json has no converted dataset; run the convert stage first")
    return Path(entry["train_dataset"])


def checkpoint_of(run_dir: Path, train: dict) -> Path:
    """Where the recipe leaves the finished weights: a step directory when the last step saves."""
    steps, every = train["steps"], min(train["save_steps"], train["steps"])
    return run_dir / "model" / f"checkpoint-{steps}" if steps % every == 0 else run_dir / "model"


def finetune_argv(cfg: dict, run_dir: Path, dataset: Path, gpus: int, batch: int, accum: int, optim: str) -> list:
    train = cfg["train"]
    return [
        TORCHRUN, f"--nproc_per_node={gpus}", f"--master_port={os.environ.get('PORT', '29500')}",
        "-m", "gr00t.experiment.launch_finetune",
        "--base-model-path", train.get("base_model") or "nvidia/GR00T-N1.7-3B",
        "--dataset-path", dataset,
        "--embodiment-tag", cfg["embodiment"]["gr00t_tag"].upper(),
        "--modality-config-path", ROOT / cfg["embodiment"]["gr00t_modality_config"],
        "--num-gpus", str(gpus),
        # experiment.py rebuilds output_dir as <output-dir>/<experiment-name> whenever
        # --experiment-name is given, so the PARENT goes here and the leaf is the name: the two
        # then agree instead of burying the checkpoints one level deeper than the caller asked.
        "--output-dir", run_dir,
        "--max-steps", str(train["steps"]),
        "--global-batch-size", str(batch),
        "--gradient-accumulation-steps", str(accum),
        "--optim", optim,
        "--save-steps", str(min(train["save_steps"], train["steps"])),
        "--save-total-limit", "1",
        # GR00T asserts eval_strategy == "no", so there is no in-training validation loss and no
        # checkpoint selection to make afterwards; keeping optimizer state as well would be 340 GB
        # for ten checkpoints. The cost is that these cannot resume training.
        "--save-only-model",
        "--dataloader-num-workers", "2",
        "--no-use-wandb",
        "--experiment-name", "model",
        "--tune-projector",
        "--tune-diffusion-model",
        "--no-tune-visual",
        "--no-tune-llm",
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory convert wrote")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    ap.add_argument("extra", nargs="*", help="extra launch_finetune arguments, passed through")
    args = ap.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_context.load_run_config(run_dir)
    train = cfg["train"]
    dataset = dataset_of(run_dir)
    checkpoint = checkpoint_of(run_dir, train)

    ngpu, vram_min, free_min = gpu_probe(args.gpu)
    gpus, batch, accum, plan_why = batch_plan(int(train["gpus"]), ngpu, int(train["global_batch"]), 1)
    optim, optim_why = choose_optimizer(
        vram_min, pinned=os.environ.get("OPTIM"), auto=os.environ.get("OPTIM_AUTO", "1") == "1"
    )
    if plan_why:  # one card: the 8-bit optimizer is a property of the card, not of the flag
        optim, optim_why = "adamw_bnb_8bit", plan_why

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    env.setdefault("HF_HOME", os.path.join(paths.WORKSPACE, "huggingface_cache"))
    if optim == "adamw_bnb_8bit":  # fragmentation insurance for a long run close to the ceiling
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    command = finetune_argv(cfg, run_dir, dataset, gpus, batch, accum, optim) + list(args.extra)

    print(f"dataset      {dataset}")
    print(f"output       {checkpoint}")
    print(f"gpus         {gpus} of {ngpu} visible ({vram_min} MiB/card, {free_min} MiB free)")
    print(f"batch        {batch} x {accum} accum = {batch * accum} effective")
    print(f"optim        {optim}  [{optim_why}]")
    print(f"steps        {train['steps']}, checkpoint every {min(train['save_steps'], train['steps'])} (keep 1)")
    # consumed in this process rather than handed to torchrun, so a dry run has to say it out loud
    print(f"squat max    {train['squat_max']} MiB of foreign VRAM tolerated on the card")
    print()

    if args.dry_run:
        run_context.show(
            "train", command, {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF") if k in env}
        )
        return 0

    require_vendored_gr00t()
    if not dataset.is_dir():
        raise SystemExit(f"no dataset at {dataset}")
    if not (dataset / "meta" / "stats.json").is_file():
        raise SystemExit(f"no meta/stats.json in {dataset} -- run the convert stage first")
    if (refusal := squatter(vram_min, free_min, int(train["squat_max"]))) is not None:
        raise SystemExit(refusal)
    if (run_dir / "model").exists():
        raise SystemExit(
            f"{run_dir / 'model'} exists; a half-written checkpoint is worse than none -- inspect it first"
        )
    run_context.require_headroom("train", gpu=args.gpu)

    from locomani.pipeline.stages import Stage

    Stage(run_dir).check("train", command, env=env)
    if not checkpoint.exists():
        raise SystemExit(f"the recipe finished but {checkpoint} is not there")
    run_context.summarise(run_dir, "train", checkpoint=str(checkpoint), steps=train["steps"], dataset=str(dataset))
    print(f"train: {train['steps']} steps -> {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
