"""`<out>/config_snapshot.json` -- the provenance every pipeline entry writes
(decision 8): the four config dicts (task/backend/eval/pilot -- `None` for
whichever this entry point does not construct), the git sha, the start time,
and a sha256 digest of every src/configs/scripts source file -- the same
coverage run_learn_pilot.py's retired `provenance.json` had. Replaces that
file; nothing writes `provenance.json` any more.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from locomani import paths


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_digests(root: Path | None = None) -> dict[str, str]:
    root = Path(root or paths.REPO)
    files = sorted(
        p
        for b in ("src/locomani", "configs", "scripts")
        for p in (root / b).rglob("*")
        if p.suffix in (".py", ".json", ".sh")
    )
    return {str(p.relative_to(root)): _digest(p) for p in files}


def file_digests(directory) -> dict[str, str]:
    """{file name: sha256} of every regular file directly under `directory` (a frozen scene dir)."""
    d = Path(directory)
    return {p.name: _digest(p) for p in sorted(d.iterdir()) if p.is_file()} if d.is_dir() else {}


def digest(path) -> str:
    return _digest(Path(path))


def git_sha(root: Path | None = None) -> str:
    root = Path(root or paths.REPO)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()


def write_config_snapshot(
    out_dir,
    *,
    task_cfg=None,
    backend_cfg=None,
    eval_cfg=None,
    pilot_cfg=None,
    started: float | None = None,
    inputs: dict | None = None,
) -> Path:
    """Write `<out_dir>/config_snapshot.json`; returns the path written.

    `inputs`: identity of external inputs the source digests cannot cover -- the frozen scene
    files, the initial token, a reach calibration -- as {name: sha256 or {file: sha256}}
    (2026-09-10 review: the retired provenance.json hashed these; the snapshot had dropped them).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = out_dir / "config_snapshot.json"
    snap.write_text(
        json.dumps(
            dict(
                task=task_cfg.to_dict() if task_cfg is not None else None,
                backend=backend_cfg.to_dict() if backend_cfg is not None else None,
                eval=eval_cfg.to_dict() if eval_cfg is not None else None,
                pilot=pilot_cfg.to_dict() if pilot_cfg is not None else None,
                git_sha=git_sha(),
                started=started if started is not None else time.time(),
                sources=source_digests(),
                inputs=inputs or {},
            ),
            indent=1,
        )
        + "\n"
    )
    return snap
