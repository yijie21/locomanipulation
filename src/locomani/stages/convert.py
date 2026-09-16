"""convert: exporter sessions -> LeRobot train/val datasets, GR00T stats, dataset audit."""

from __future__ import annotations

import json
import time
from pathlib import Path

from omegaconf import DictConfig

from locomani.pipeline.stages import Stage
from locomani.stages import common


def run(run_cfg, cfg: DictConfig, out: Path) -> dict:
    started, resources = time.time(), common.resources_now()
    up = common.require_upstream(out, "convert")
    sessions = [s for s in json.loads(up["outputs"]["sessions"]) if s]
    manifests = json.loads(up["outputs"]["manifests"])
    train_ds, val_ds = out / "train_dataset", out / "val_dataset"
    for d in (train_ds, val_ds):
        if d.exists():
            raise RuntimeError(f"partial conversion exists at {d}; inspect it and use a new output directory")
    stages = Stage(out)
    stages.check(
        "convert",
        [
            common.DATA_PY,
            common.SRC / "pipeline/convert_to_hf_format.py",
            "--sessions",
            *sessions,
            "--manifest",
            *manifests,
            "--trim-verification-tail",
            "--out",
            train_ds,
            "--holdout",
            str(run_cfg.train.holdout),
            "--holdout-out",
            val_ds,
            "--jobs",
            "4",
        ],
    )
    modality = common.ROOT / run_cfg.embodiment.gr00t_modality_config
    splits = [("train", train_ds)] + (
        [("val", val_ds)] if (val_ds / "meta" / "info.json").exists() else []
    )  # holdout=0 -> no val
    for label, ds in splits:
        stages.check(
            f"stats_{label}",
            [
                common.TRAIN_PY,
                "-m",
                "gr00t.data.stats",
                "--dataset-path",
                ds,
                "--embodiment-tag",
                run_cfg.embodiment.gr00t_tag.upper(),
                "--modality-config-path",
                modality,
            ],
        )
        stages.check(f"audit_{label}", [common.DATA_PY, common.SRC / "pipeline/check_dataset.py", ds])
    info = json.loads((train_ds / "meta" / "info.json").read_text())
    metrics = dict(
        train_episodes=info.get("total_episodes"), train_frames=info.get("total_frames"), fps=info.get("fps")
    )
    return common.write_manifest(
        out,
        "convert",
        cfg,
        ok=True,
        outputs=dict(train_dataset=train_ds, val_dataset=val_ds if len(splits) > 1 else "", modality_config=modality),
        metrics=metrics,
        started=started,
        resources=resources,
    )
