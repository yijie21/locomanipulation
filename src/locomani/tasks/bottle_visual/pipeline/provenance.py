"""Provenance of a ScaleBFM-line run: the source files it hashes, and the registered assets.

Every driver in this line writes `source_sha256` (one entry per source file) into its
`config.json`, and refuses to resume into a directory whose hashes differ. `line_sources()`
is the one list they all hash -- the whole ScaleBFM backend and this task package -- so a
change anywhere in the line invalidates a resume, which is the behaviour the drivers relied
on when they each kept a hand-written list of sibling files.
"""

import hashlib
import json
from pathlib import Path

from locomani.paths import REPO

_BACKEND = Path(__file__).resolve().parents[4] / "backends" / "scalebfm"
_TASK = Path(__file__).resolve().parents[1]


def line_sources() -> list[Path]:
    """Every .py file of the ScaleBFM backend and the bottle_visual task (incl. pipeline/)."""
    files = [*_BACKEND.glob("*.py"), *_TASK.glob("*.py"), *(_TASK / "pipeline").glob("*.py")]
    return sorted(p for p in files if p.name != "__init__.py")


def source_sha256(files=None) -> dict[str, str]:
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (files or line_sources())}


def verify_runtime_assets():
    manifest = Path(REPO) / "outputs/experiments/scalebfm_visual_goal_20260908/runtime_assets.json"
    assets = json.loads(manifest.read_text())["files"]
    for name, expected in assets.items():
        path = Path(name)
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"Registered runtime asset changed: {path}")
    return assets
