"""stats: one summary of a run from its stage manifests (what the registry and `locomani.cli compare` read)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from omegaconf import DictConfig

from locomani.stages import common


def run(run_cfg, cfg: DictConfig, out: Path) -> dict:
    started, resources = time.time(), common.resources_now()
    summary = {}
    for stage in ("collect", "convert", "train", "eval"):
        m = common.read_manifest(out, stage)
        if m is not None:
            summary[stage] = dict(ok=m["ok"], metrics=m["metrics"], seconds=m["seconds"], config_hash=m["config_hash"])
    if not summary:
        raise RuntimeError(f"no stage manifests under {out}")
    (common.stage_dir(out, "stats") / "summary.json").write_text(json.dumps(summary, indent=1))
    return common.write_manifest(
        out,
        "stats",
        cfg,
        ok=True,
        outputs=dict(summary=Path(out) / "stats" / "summary.json"),
        metrics={s: v["metrics"] for s, v in summary.items()},
        started=started,
        resources=resources,
    )
