"""What every stage script does before it does its own work: read the experiment, claim the run
directory, refuse to start without headroom, and -- with --dry-run -- say what it would execute.

    cfg, run_dir = open_run(args.experiment, args.name)
    require_headroom("collect", gpu=args.gpu)
    require_deploy_patches(cfg)

The experiment file is copied verbatim to `<run>/config.yaml`. That copy is the run's provenance
and is what a later stage reads, so a stage never depends on a file that may have changed since.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"
PID_MARGIN = 200  # a stack needs room to fork after the check passes
NEEDS = {  # measured, not theoretical: an exporter holds ~1800 pids and the container's ceiling is 3840
    "collect": dict(cores=8, pids=1800, gpu_free_mib=6000),
    "convert": dict(cores=4, pids=200, gpu_free_mib=0),
    "train": dict(cores=4, pids=300, gpu_free_mib=28000),
    "eval": dict(cores=8, pids=1800, gpu_free_mib=12000),
}


def open_run(experiment: Path | str, name: str | None = None) -> tuple[dict, Path]:
    """Read the experiment file and copy it verbatim into the run directory it names."""
    experiment = Path(experiment)
    cfg = yaml.safe_load(experiment.read_text())
    cfg["name"] = name or cfg["name"]
    run_dir = RUNS / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    copy = run_dir / "config.yaml"
    if copy.exists() and copy.read_text() != experiment.read_text():
        raise SystemExit(f"{copy} differs from {experiment}; use a new name rather than changing a run in flight")
    shutil.copyfile(experiment, copy)
    return cfg, run_dir


def load_run_config(run_dir: Path) -> dict:
    """The copy this run started from. A later stage reads this, never the file in configs/."""
    path = Path(run_dir) / "config.yaml"
    if not path.exists():
        raise SystemExit(f"{path} is missing; this directory was not opened by a stage script")
    return yaml.safe_load(path.read_text())


def measure(gpu: str | None = None) -> dict:
    """Cores in use (1 s cgroup delta), pids in the container, free MiB on the card."""
    live: dict = {}
    try:
        cpu = Path("/sys/fs/cgroup/cpu.stat")
        first = int(cpu.read_text().split()[1])
        time.sleep(1.0)
        live["cores"] = round((int(cpu.read_text().split()[1]) - first) / 1e6, 2)
        live["pids"] = int(Path("/sys/fs/cgroup/pids.current").read_text())
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        live["quota"] = float(os.cpu_count() or 16) if quota == "max" else float(quota) / float(period)
        limit = Path("/sys/fs/cgroup/pids.max").read_text().strip()
        live["pids_max"] = 3840 if limit == "max" else int(limit)
    except (OSError, ValueError, IndexError):
        pass
    if gpu is not None:
        try:
            q = subprocess.run(
                ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            total, used = (int(x) for x in q.stdout.strip().splitlines()[0].split(","))
            live["gpu_free_mib"] = total - used
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
    return live


def shortfall(stage: str, live: dict) -> str | None:
    """Why this stage must not start now, or None. Pure, so the thresholds are testable."""
    need = NEEDS[stage]
    if (cores := live.get("cores")) is not None and cores + need["cores"] > live.get("quota", 16.0):
        return f"cpu: {cores} cores in use + {need['cores']} needed > quota {live.get('quota')}"
    if (pids := live.get("pids")) is not None and pids + need["pids"] > live.get("pids_max", 3840) - PID_MARGIN:
        return f"pids: {pids} + {need['pids']} needed > {live.get('pids_max', 3840) - PID_MARGIN}"
    free = live.get("gpu_free_mib")
    if need["gpu_free_mib"] and free is not None and free < need["gpu_free_mib"]:
        return f"gpu: {free} MiB free < {need['gpu_free_mib']} needed"
    return None


def require_headroom(stage: str, *, gpu: str | None = None) -> dict:
    """Refuse rather than wait: a saturated quota makes the sim miss its wall clock and every
    timed expert phase fail as a control error, which reads like a robot problem for hours."""
    live = measure(gpu)
    if (why := shortfall(stage, live)) is not None:
        raise SystemExit(f"{stage}: not enough headroom -- {why}")
    return live


def require_deploy_patches(cfg: dict) -> None:
    """The streamed-pose experts need a patched deploy; check before anything is started."""
    sys.path.insert(0, str(ROOT / "src"))
    from locomani.pipeline.provenance import require_pose_ownership

    strategy = (cfg.get("task", {}).get("reach") or {}).get("strategy")
    require_pose_ownership(dict(os.environ), hand_handoff=strategy in ("simple", "whole_body"))


def show(name: str, argv: list, env: dict | None = None) -> None:
    """The one dry-run format: what would run, and the settings that are this command's own."""
    print(f"\n[{name}]\n  " + " ".join(str(a) for a in argv))
    for key, value in sorted((env or {}).items()):
        print(f"    {key}={value}")


def summarise(run_dir: Path, stage: str, **metrics) -> Path:
    """One line per stage in `<run>/summary.json`: what the next stage and the reader both read."""
    path = Path(run_dir) / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary[stage] = dict(metrics, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    path.write_text(json.dumps(summary, indent=1, sort_keys=True))
    return path
