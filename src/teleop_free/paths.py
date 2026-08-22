"""Every absolute path this repo uses, resolved once, in one place.

Two rules, and the split between them is the whole point:

* **Inside the repo** — scenes, reach maps, logs, models — nothing is
  configurable, because nothing needs to be. The repo root is *derived* from
  this file's own location, so a fresh clone at any path just works, with no
  environment variable to set and no sed pass to run. (``scripts/install_scene.sh``
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

# src/teleop_free/paths.py -> src/teleop_free -> src -> <repo>
REPO = os.environ.get("LOCOMANI_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Everything that is NOT in the repo. Caches, deps, conda, deploy sessions.
WORKSPACE = os.environ.get("LOCOMANI_WORKSPACE", "/workspace")

SRC = os.path.join(REPO, "src", "teleop_free")
SCRIPTS = os.path.join(REPO, "scripts")
THIRD_PARTY = os.path.join(REPO, "third_party")
SONIC = os.path.join(THIRD_PARTY, "GR00T-WholeBodyControl")
VENV_SIM_PY = os.path.join(SONIC, ".venv_sim", "bin", "python")

OUTPUTS = os.path.join(REPO, "outputs")
SCENE_DIR = os.path.join(OUTPUTS, "scene")          # MJCF + caches
SIM_DIR = os.path.join(SCENE_DIR, "sim")            # the composed scene
SCENE_META = os.path.join(SIM_DIR, "scene_meta.json")
SCENE_XML = os.path.join(SIM_DIR, "scene_locomani.xml")
DIAG_DIR = os.path.join(OUTPUTS, "diag")            # diag_*.py results
LOGS_DIR = os.path.join(OUTPUTS, "logs")            # episode jsonl
MODELS_DIR = os.path.join(OUTPUTS, "models")        # exported grasp models
VIDEOS_DIR = os.path.join(OUTPUTS, "videos")        # diagnostic renders
RENDERS_ISAAC = os.path.join(OUTPUTS, "renders_isaac")

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
    keys = ("REPO", "WORKSPACE", "SONIC", "SCENE_META", "OUTPUTS", "LOGS_DIR")
    body = "\n".join(f"  {k:<12} {globals()[k]}" for k in keys)
    return (f"{body}\n  {'session_root()':<12} {session_root()}\n"
            f"  {'derived from':<12} {os.path.abspath(__file__)}")


if __name__ == "__main__":
    print(describe())
