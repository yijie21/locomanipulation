"""`scripts/build_scene.sh <name>` runs the recipe configs/scenes.yaml records for that scene.

The three collection lines' runbooks build their scenes with it, so every scene an experiment
file names by `scene.name` must carry a `build` the script can find, and a name it does not
know must stop it before anything runs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_scene.sh"


def dry(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args, "--dry-run"], cwd=ROOT, capture_output=True, text=True)


@pytest.mark.parametrize("name", ["bottle", "crate", "warehouse_textured"])
def test_each_line_scene_prints_its_registry_recipe(name):
    entry = yaml.safe_load((ROOT / "configs/scenes.yaml").read_text())[name]
    got = dry(name)
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == f"+ {entry['build']}"
    assert entry["dir"].split("/")[-1] in entry["build"]  # the recipe makes the directory the name points at


def test_the_isaac_stage_is_its_own_part():
    got = dry("warehouse_textured", "--isaac")
    assert got.returncode == 0 and "build_isaac_warehouse_scene.py" in got.stdout


@pytest.mark.parametrize("args", [("nosuch",), ("bottle", "--isaac")])
def test_a_scene_without_that_recipe_is_refused(args):
    got = dry(*args)
    assert got.returncode != 0 and "configs/scenes.yaml" in got.stderr and "+ " not in got.stdout


def test_paths_sh_puts_src_on_pythonpath_once_and_reads_the_local_file(tmp_path):
    (tmp_path / "locomani.local.sh").write_text('export SONIC_DEPLOY_ROOT="${SONIC_DEPLOY_ROOT:-/elsewhere}"\n')
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "LOCOMANI_WORKSPACE": str(tmp_path)}
    out = subprocess.run(
        ["bash", "-c", 'set -eu; . scripts/paths.sh; . scripts/paths.sh; echo "$PYTHONPATH|$SONIC_DEPLOY_ROOT"'],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == f"{ROOT / 'src'}|/elsewhere"


def test_the_isaac_stage_python_is_the_first_one_that_imports_pxr(tmp_path):
    """USD_PY is tried first but must import pxr; then python3, then the python bundled with the `blender` on PATH."""
    bin_dir, blender_dir = tmp_path / "bin", tmp_path / "blender-9.9-linux-x64"
    bundled = blender_dir / "9.9" / "python" / "bin" / "python3.99"
    bundled.parent.mkdir(parents=True)
    bin_dir.mkdir()
    (bin_dir / "python3").write_text("#!/bin/sh\nexit 1\n")  # a system python without pxr
    (blender_dir / "blender").write_text("#!/bin/sh\nexit 0\n")
    bundled.write_text("#!/bin/sh\nexit 0\n")  # imports pxr
    (bin_dir / "blender").symlink_to(blender_dir / "blender")
    for f in (bin_dir / "python3", blender_dir / "blender", bundled):
        f.chmod(0o755)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path), "LOCOMANI_WORKSPACE": str(tmp_path)}

    def run(**extra: str) -> subprocess.CompletedProcess:
        probe = "set -eu; . scripts/paths.sh; locomani_usd_python"
        return subprocess.run(["bash", "-c", probe], cwd=ROOT, env={**env, **extra}, capture_output=True, text=True)

    assert run().stdout.strip() == str(bundled)
    assert run(USD_PY=str(blender_dir / "blender")).stdout.strip() == str(blender_dir / "blender")  # exits 0: "has pxr"
    stale = run(USD_PY="python3")  # left over from a shell that sourced the older paths.sh default
    assert stale.stdout.strip() == str(bundled) and "USD_PY=python3 has no pxr" in stale.stderr
    bundled.write_text("#!/bin/sh\nexit 1\n")
    got = run()
    assert got.returncode != 0 and got.stdout == ""
