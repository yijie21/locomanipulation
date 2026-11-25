"""TODO 33.5: the preparatory-action selector from the command line (locomani.tasks.scene.prep), without a simulation --
for an object at a start the chooser cannot serve, which goals (turn it, pull it toward an edge, move a neighbour) would
give a standing stance from which the hand is delivered, best first, and which of them an action in the skills package
implements.

    PYTHONPATH=src .venv_sim/bin/python scripts/rsi/prep.py <scene dir> <object> [--start START.json] [--skills DIR] [--json]

`--start` is a pool's start file (placements and robot_xy); `--skills` a skills package directory whose modules register
actions (`prep.register`), so the listing says which plans are implemented. A plan is a prediction for choosing what
to try; only a closed-loop round is evidence ([evidence])."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from locomani.tasks.scene import graspability as G  # noqa: E402
from locomani.tasks.scene import prep as P  # noqa: E402


def load_skills(d: Path) -> None:
    """Import a skills package directory under a private name so its `prep.register` calls run."""
    spec = importlib.util.spec_from_file_location("rsi_prep_skills", d / "__init__.py",
                                                  submodule_search_locations=[str(d)])  # fmt: skip
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rsi_prep_skills"] = mod
    spec.loader.exec_module(mod)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("object")
    ap.add_argument("--start", type=Path, default=None)
    ap.add_argument("--skills", type=Path, default=None, help="a skills package whose modules register prep actions")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.skills is not None:
        load_skills(a.skills)
    scene = G.StaticScene(a.scene)
    start_xy = None
    if a.start is not None:
        st = json.loads(a.start.read_text())["start"]
        st = ast.literal_eval(st) if isinstance(st, str) else st
        scene.place(st.get("placements") or {})
        start_xy = st.get("robot_xy")
    plans = P.propose(scene, a.object, start_xy)
    if a.json:
        print(json.dumps(dict(object=a.object, implemented=P.implemented(), wanted=P.wanted(plans),
                              plans=[x.as_dict() for x in plans[:12]]), ensure_ascii=False))  # fmt: skip
        return 0
    print(
        f"{a.object}: {len(plans)} plans; actions implemented: {P.implemented() or 'none'}; wanted: {P.wanted(plans) or 'none'}"
    )
    if not plans:
        print("nothing to prepare (the chooser already has a stance), or no goal helps")
    else:
        print(P.explain(plans[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
