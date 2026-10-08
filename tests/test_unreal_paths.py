"""Where the Unreal level records live: beside the install ($UNREAL_ROOT/levels), not in a checkout.

The levels are in the machine's one host project, so every checkout must find their source.json. Under
$LOCOMANI_ROOT a level built from a worktree was unknown to the main checkout (TODO 12.9.11, 2026-09-29).
paths.sh and paths.py each resolve the default; they must agree.
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml

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


def test_a_scene_level_copies_its_own_room_level_and_the_warehouse_by_default():
    """The apartment's level is a copy of its own room (scenes.yaml `unreal.room_level`), imported into its own asset
    directory so building it never replaces the warehouse room's meshes."""
    from locomani.sim.unreal import build_level

    assert build_level.room_level(None) == (build_level.ROOM_LEVEL, {})
    path, options = build_level.room_level("AdtApartment")
    assert path == "/Game/Levels/AdtApartment"
    assert options == {"WAREHOUSE_LEVEL": path, "WAREHOUSE_ASSETS": "/Game/Rooms/AdtApartment"}
    entry = yaml.safe_load((ROOT / "configs/scenes.yaml").read_text())["adt_apartment_mars"]["unreal"]
    assert entry["room_level"] == "AdtApartment" and "--room-level AdtApartment" in entry["room_build"]
