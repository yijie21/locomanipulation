"""Provenance of a ScaleBFM-line run: the source files it hashes, and the registered assets.

Every driver in this line writes `source_sha256` (one entry per source file) into its
`config.json`, and refuses to resume into a directory whose hashes differ. `line_sources()`
is the one list they all hash -- the whole ScaleBFM backend and this task package -- so a
change anywhere in the line invalidates a resume, which is the behaviour the drivers relied
on when they each kept a hand-written list of sibling files.
"""

import hashlib
import json
import shutil
from pathlib import Path

from locomani.paths import REPO

_PKG = Path(__file__).resolve().parents[3]  # src/locomani
_BACKEND = _PKG / "backends" / "scalebfm"
_TASK = _PKG / "tasks" / "bottle_visual"


def line_sources() -> list[Path]:
    """Every .py file of the ScaleBFM backend and the bottle_visual task (incl. pipeline/)."""
    files = [*_BACKEND.glob("*.py"), *_TASK.glob("*.py"), *(_TASK / "pipeline").glob("*.py")]
    files = sorted(p for p in files if p.name != "__init__.py")
    names = {p.name for p in files}
    # 2026-09-10 review: a wrong parents[] index silently dropped the whole backend from the
    # fingerprint; fail loudly if either package is missing rather than hash a partial line.
    if not {"runtime.py", "network.py", "round.py", "planner.py"} <= names:
        raise RuntimeError(f"line_sources() is incomplete: {sorted(names)}")
    return files


def source_relpath(p: Path) -> Path:
    """`backends/scalebfm/config.py`, not `config.py`: snapshot copies keep the package layout
    (two config.py files would otherwise overwrite each other in <out>/source/)."""
    return Path(p).resolve().relative_to(_PKG)


def snapshot_sources(out_dir, files=None) -> None:
    for p in files or line_sources():
        dest = Path(out_dir) / "source" / source_relpath(p)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)


def source_sha256(files=None) -> dict[str, str]:
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (files or line_sources())}


def verify_runtime_assets(manifest=None):
    manifest = (
        Path(manifest)
        if manifest
        else Path(REPO) / "outputs/experiments/scalebfm_visual_goal_20260908/runtime_assets.json"
    )
    assets = json.loads(manifest.read_text())["files"]
    for name, expected in assets.items():
        path = Path(name)
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"Registered runtime asset changed: {path}")
    return assets
