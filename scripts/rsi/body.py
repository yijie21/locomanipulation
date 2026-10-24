"""TODO 33.1.4: the body model's reach predictor from the command line (locomani.tasks.scene.body) -- for a role or a
person asking whether the controller delivers a reach command, without running a round.

    PYTHONPATH=src .venv_sim/bin/python scripts/rsi/body.py reach --x 0.45 --y-out 0.05 --z 0.9 --att level --hand right
    PYTHONPATH=src .venv_sim/bin/python scripts/rsi/body.py world --point X Y Z --pelvis X Y Z QW QX QY QZ --hand right
    PYTHONPATH=src .venv_sim/bin/python scripts/rsi/body.py best --name pick.STANCE_MIN_D_M [--task T]   (TODO 33.1.7)

Frame: x forward and y outward of the hand's own side from the pelvis (heading frame), z above the floor, metres.
The answer is the odds of landing 3 cm or more off and the error to expect, from the envelope grid's table
(`--table` names another file). `best`: the level of a parameter the design probes' curves say works best, or
null -- what a skill gets from body.best_level. A prediction is for choosing what to try; only a closed-loop round
is evidence."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from locomani.tasks.scene.body import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
