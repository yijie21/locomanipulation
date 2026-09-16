"""Manifests, config hashes and resource snapshots shared by every stage."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from locomani import paths

SIM_PY = Path(paths.VENV_SIM_PY)
DATA_PY = Path(paths.SONIC) / ".venv_data_collection" / "bin" / "python"
TRAIN_PY = Path(paths.THIRD_PARTY) / "Isaac-GR00T" / ".venv" / "bin" / "python"
SRC = Path(paths.SRC)
ROOT = Path(paths.REPO)

# Which config groups a stage's behaviour depends on: its config hash covers exactly these.
STAGE_GROUPS = {
    "collect": ("task", "embodiment", "scene", "backend", "collect"),
    "convert": ("embodiment", "train"),
    "train": ("embodiment", "train"),
    "eval": ("task", "embodiment", "scene", "backend", "eval"),
    "stats": (),
}
UPSTREAM = {"collect": None, "convert": "collect", "train": "convert", "eval": "train", "stats": "eval"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def config_hash(cfg: DictConfig, stage: str) -> str:
    tree = OmegaConf.to_container(cfg, resolve=True)
    part = {g: tree.get(g) for g in STAGE_GROUPS[stage]}
    return sha256_text(json.dumps(part, sort_keys=True))


def stage_dir(out: Path, stage: str) -> Path:
    d = Path(out) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(out: Path, stage: str) -> Path:
    return Path(out) / stage / "manifest.json"


def read_manifest(out: Path, stage: str) -> dict | None:
    p = manifest_path(out, stage)
    return json.loads(p.read_text()) if p.exists() else None


def manifest_hash(out: Path, stage: str) -> str | None:
    p = manifest_path(out, stage)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def require_upstream(out: Path, stage: str) -> dict:
    """The upstream stage's manifest, which must exist and be ok."""
    up = UPSTREAM[stage]
    m = read_manifest(out, up)
    if m is None or not m.get("ok"):
        raise RuntimeError(f"{stage} needs a completed {up} stage in {out} (run `locomani.cli {up}` first)")
    return m


def resources_now() -> dict:
    """CPU cores in use (1 s cgroup delta), pids in the container, foreign GPU memory per card."""
    out = dict(cores_in_use=None, pids=None, gpu_used_mib={}, load1=None)
    try:
        cpu = Path("/sys/fs/cgroup/cpu.stat")
        a = int(cpu.read_text().split()[1])
        time.sleep(1.0)
        b = int(cpu.read_text().split()[1])
        out["cores_in_use"] = round((b - a) / 1e6, 2)
        out["pids"] = int(Path("/sys/fs/cgroup/pids.current").read_text())
        out["load1"] = os.getloadavg()[0]
    except (OSError, ValueError, IndexError):
        pass
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in q.stdout.splitlines():
            idx, used = [x.strip() for x in line.split(",")]
            out["gpu_used_mib"][idx] = int(used)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return out


def write_manifest(
    out: Path,
    stage: str,
    cfg: DictConfig,
    *,
    ok: bool,
    outputs: dict,
    metrics: dict,
    started: float,
    resources: dict,
    error: str | None = None,
) -> dict:
    up = UPSTREAM[stage]
    m = dict(
        stage=stage,
        ok=bool(ok),
        error=error,
        run=str(out),
        name=str(cfg.get("name")),
        config_hash=config_hash(cfg, stage),
        upstream={up: manifest_hash(out, up)} if up else {},
        outputs={k: str(v) for k, v in outputs.items()},
        metrics=metrics,
        resources_at_start=resources,
        started=started,
        finished=time.time(),
        seconds=round(time.time() - started, 1),
    )
    p = manifest_path(out, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=1))
    return m


def rows(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if Path(path).exists()
        else []
    )
