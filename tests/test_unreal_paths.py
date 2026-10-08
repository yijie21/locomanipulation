"""Where the Unreal level records live: beside the install ($UNREAL_ROOT/levels), not in a checkout.

The levels are in the machine's one host project, so every checkout must find their source.json. Under
$LOCOMANI_ROOT a level built from a worktree was unknown to the main checkout (TODO 12.9.11, 2026-09-29).
paths.sh and paths.py each resolve the default; they must agree.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("UNREAL_ROOT", "UNREAL_ASSETS", "LOCOMANI_WORKSPACE")}
    env["LOCOMANI_WORKSPACE"] = (
        "/nonexistent-no-local-settings"  # paths.sh sources $LOCOMANI_WORKSPACE/locomani.local.sh
    )
    return {**env, **extra}


def test_the_level_records_default_beside_the_unreal_install_in_both_path_files():
    env = _env(UNREAL_ROOT="/opt/ue")
    py = subprocess.run(
        [sys.executable, "-c", "from locomani import paths; print(paths.UNREAL_ASSETS)"],
        env={**env, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    sh = subprocess.run(
        ["bash", "-c", f'. "{ROOT}/scripts/paths.sh" && echo "$UNREAL_ASSETS"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert py == sh == "/opt/ue/levels"
