"""Every absolute path this repo uses, resolved once, in one place.

Two rules, and the split between them is the whole point:

* **Inside the repo** — scenes, reach maps, logs, models — nothing is
  configurable, because nothing needs to be. The repo root is *derived* from
  this file's own location, so a fresh clone at any path just works, with no
  environment variable to set and no sed pass to run. An explicit absolute
  ``LOCOMANI_SIM_DIR`` isolates the sim, IK and stance diagnostics when
  concurrent experiments need separate composed scenes. (``scripts/install_scene.sh``
  has done this since the beginning; the rest of the tree had drifted into
  hard-coded ``/workspace/code/locomanipulation`` strings.)
* **Outside the repo** — TensorRT/ONNX Runtime under ``deps``, the conda
  install, the deploy's session output — genuinely varies per machine, so it
  comes from ``$LOCOMANI_WORKSPACE`` (default ``/workspace``). A collaborator
  whose home is ``/home/workspace`` sets that one variable.

A boolean "am I on the other machine" flag was the obvious alternative and is
worse: it only ever describes two machines, and it still breaks if the clone
does not sit at exactly ``<base>/code/locomanipulation``. Deriving the repo
root has neither failure mode.

The shell half of this lives in ``scripts/paths.sh`` and exports the same
names.
"""

from __future__ import annotations

import os

# src/locomani/paths.py -> src/locomani -> src -> <repo>
REPO = os.environ.get("LOCOMANI_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Everything that is NOT in the repo. Caches, deps, conda, deploy sessions.
WORKSPACE = os.environ.get("LOCOMANI_WORKSPACE", "/workspace")

SRC = os.path.join(REPO, "src", "locomani")
# The locomani package's own directory, computed straight from this file's location
# rather than joined from REPO/SRC. Same value as SRC by construction (paths.py sits
# at the package root) -- kept as a distinct name for code that wants "this package's
# root" without going through the REPO override (refactor step 4a package-layout move).
PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(REPO, "scripts")
THIRD_PARTY = os.path.join(REPO, "third_party")
SONIC = os.path.join(THIRD_PARTY, "GR00T-WholeBodyControl")
VENV_SIM_PY = os.path.join(SONIC, ".venv_sim", "bin", "python")

# --- the SONIC deploy and the checkpoint it runs -----------------------------
# SONIC_DEPLOY_ROOT is the one thing here that may live outside the repo (a
# lockstep or NVIDIA-default build under $DEPS_DIR), so it keeps an override,
# exactly like scripts/paths.sh.
SONIC_DEPLOY = os.environ.get("SONIC_DEPLOY_ROOT") or os.path.join(SONIC, "gear_sonic_deploy")
# Which checkpoint. DEFAULT v1.1 since 2026-09-11 (user ruling) -- the same
# default as scripts/paths.sh, and that is the point: the encoder token is a
# checkpoint-private latent, so an offline tool reading a different policy
# directory than the running deploy produces numbers that mean nothing.
SONIC_POLICY = os.environ.get("SONIC_POLICY", "policy/sonic_v1_1")
SONIC_POLICY_DIR = os.path.join(SONIC_DEPLOY, SONIC_POLICY)


def sonic_policy_file(name: str) -> str:
    """One file of the ACTIVE SONIC checkpoint, e.g. ``model_encoder.onnx``."""
    return os.path.join(SONIC_POLICY_DIR, name)


OUTPUTS = os.path.join(REPO, "outputs")
SCENE_DIR = os.path.join(OUTPUTS, "scene")  # MJCF + caches
SIM_DIR = os.path.abspath(os.environ.get("LOCOMANI_SIM_DIR", os.path.join(SCENE_DIR, "sim")))
SCENE_META = os.path.join(SIM_DIR, "scene_meta.json")
SCENE_XML = os.path.join(SIM_DIR, "scene_locomani.xml")
DIAG_DIR = os.path.join(OUTPUTS, "diag")  # diag_*.py results
LOGS_DIR = os.path.join(OUTPUTS, "logs")  # episode jsonl
MODELS_DIR = os.path.join(OUTPUTS, "models")  # exported grasp models
VIDEOS_DIR = os.path.join(OUTPUTS, "videos")  # diagnostic renders
RENDERS_ISAAC = os.path.join(OUTPUTS, "renders_isaac")

# --- Isaac rendering ---------------------------------------------------------
# Mirrors scripts/paths.sh. SIMPLE is a pinned checkout under third_party/
# (docs/setup.md step 7), not a vendored dependency: the render path never imports the
# `simple` package (it converts our own scene_locomani.xml through Isaac's MJCF
# importer). See patches/simple.md.
SIMPLE_ROOT = os.environ.get("SIMPLE_ROOT", os.path.join(REPO, "third_party", "SIMPLE"))
SIMPLE_PIN = os.environ.get("SIMPLE_PIN", "6d10628794d9c7de4596b4f2afb2c054a637c2bc")
# The python carrying isaacsim. Deliberately separate from SIMPLE_ROOT: SIMPLE
# pins isaacsim==4.5.0 while 3DGS/NuRec needs 5.0+, so the render runtime has to
# be able to move without waiting for SIMPLE.
RENDER_PY = os.environ.get("RENDER_PY", os.path.join(SIMPLE_ROOT, ".venv", "bin", "python"))
# Converted USD assets. Ticket 03 moves the build under assets/isaac/; until
# then the first-light assets still live where they were generated.
ISAAC_ASSETS = os.environ.get("ISAAC_ASSETS", os.path.join(REPO, "assets", "isaac"))
# The robot-only MJCF the render articulation is imported from. NOT the scene
# XML: the scene changes per batch, the robot does not, so they are built and
# cached separately (scripts/build_isaac_assets.sh).
ROBOT_MJCF = os.path.join(SONIC, "gear_sonic", "data", "robot_model", "model_data", "g1", "g1_29dof_with_dex1.xml")

RESTART_DEPLOY = os.path.join(SCRIPTS, "restart_deploy.sh")


def session_root(instance: int = 0) -> str:
    """Where start_stack.sh puts logs and the exporter puts raw sessions.

    Instance 0 is the plain path; instance i gets a suffix, matching
    start_stack.sh so multi-instance runs do not collide.
    """
    tail = "p03" if int(instance) == 0 else f"p03_i{int(instance)}"
    return os.path.join(WORKSPACE, "tmp", tail)


def describe() -> str:
    """One-line-per-path dump — what `python paths.py` prints."""
    keys = ("REPO", "WORKSPACE", "SONIC", "SONIC_POLICY_DIR", "SCENE_META", "OUTPUTS", "LOGS_DIR")
    body = "\n".join(f"  {k:<17} {globals()[k]}" for k in keys)
    return f"{body}\n  {'session_root()':<17} {session_root()}\n  {'derived from':<17} {os.path.abspath(__file__)}"


if __name__ == "__main__":
    print(describe())
