"""train: the GR00T finetune (scripts/train_bottle.sh) on the converted dataset, one checkpoint."""

from __future__ import annotations

import time
from pathlib import Path

from omegaconf import DictConfig

from locomani.pipeline.stages import Stage
from locomani.stages import common


def checkpoint_path(out: Path, train) -> Path:
    ckpt = out / "model"
    if train.steps % min(train.save_steps, train.steps) == 0:
        ckpt = ckpt / f"checkpoint-{train.steps}"
    return ckpt


def run(run_cfg, cfg: DictConfig, out: Path) -> dict:
    started, resources = time.time(), common.resources_now()
    up = common.require_upstream(out, "train")
    train_ds = Path(up["outputs"]["train_dataset"])
    if (out / "model").exists():
        raise RuntimeError("partial model exists; inspect it and use a new output directory")
    t = run_cfg.train
    env = dict(
        DATA=str(train_ds),
        RUN="model",
        OUT_ROOT=str(out),
        MAX_STEPS=str(t.steps),
        SAVE_STEPS=str(min(t.save_steps, t.steps)),
        SAVE_LIMIT="1",
        WANDB="0",
        WORKERS="2",
        GPUS=str(t.gpus),
        SINGLE_GPU="1" if t.gpus == 1 else "0",
        GLOBAL_BATCH=str(t.global_batch),
        CUDA_VISIBLE_DEVICES=str(run_cfg.gpu),
        SQUAT_MAX=str(t.squat_max),
        MODALITY=str(common.ROOT / run_cfg.embodiment.gr00t_modality_config),
    )
    if t.base_model:
        env["BASE_MODEL"] = str(t.base_model)
    import os

    Stage(out).check("train", ["bash", common.ROOT / "scripts/train_bottle.sh"], env=dict(os.environ, **env))
    ckpt = checkpoint_path(out, t)
    if not ckpt.exists():
        raise RuntimeError(f"no checkpoint at {ckpt} after training")
    return common.write_manifest(
        out,
        "train",
        cfg,
        ok=True,
        outputs=dict(checkpoint=ckpt, train_dataset=train_ds),
        metrics=dict(steps=t.steps, gpus=t.gpus, global_batch=t.global_batch),
        started=started,
        resources=resources,
    )
