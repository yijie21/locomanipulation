"""selfmodel_via_20261120, fourth step, part two: the route sets of the three self-checks before anything is scored.

    python3 -I gap_check_routes.py <gap routes dir> <out dir> [n paths, default 24]

Takes the first n planned paths of <gap routes dir> (slots.json order) and writes three route sets, each with its own
slots.json (walk key: the base scene, as gap_slots.py):
  far   the obstacles moved 10 m ahead: with REACH_GAP_OBST=1 in the gap scene, a run must equal the same paths' run in the
        original scene bit for bit (placing, the contact callback and the two extra bodies change nothing while untouched)
  real  the obstacles where the plan had them (as the test runs them)
  hit   the bar lowered onto the goal (its centre at the goal's height): the hand has to touch it -- the contact record must
        show it, with a time after the command
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
SL = json.load(open(D / "slots.json"))
picked = [(n, k) for n, k in zip(SL["names"], SL["key"]) if n is not None][:N]
for tag in ("far", "real", "hit"):
    for sub in ("via", "prep"):
        (OUT / tag / sub).mkdir(parents=True, exist_ok=True)
    for name, _ in picked:
        shutil.copy(D / "via" / f"via_{name}.npz", OUT / tag / "via" / f"via_{name}.npz")
        pr = dict(np.load(D / "prep" / f"prep_{name}.npz"))
        B = pr["boxes"].copy()
        if tag == "far":
            B[:, 0] += 10.0
        elif tag == "hit":
            B[1, 2] = float(pr["goal_jaw"][2])
        pr["boxes"] = B
        np.savez(OUT / tag / "prep" / f"prep_{name}.npz", **pr)
    json.dump(dict(names=[n for n, _ in picked], walk=[int(k[0]) for _, k in picked], key=[k for _, k in picked],
                   levels=SL["levels"], margins=SL["margins"]), open(OUT / tag / "slots.json", "w"))  # fmt: skip
print(f"{len(picked)} paths -> {OUT}/far, real, hit")
