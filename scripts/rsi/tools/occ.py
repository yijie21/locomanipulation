"""TODO 32.6: the navigation map around a point -- an ASCII window of the occupancy the walk plans on (`#` occupied for
furniture and walls, `+` within the robot's body radius of them -- no place for the robot's centre --, `.` free), marked
points, and the clearance (distance to the nearest occupied cell) at each.
What 18 of 44 Reviewers rebuilt by hand (`occ.py`, `occ_look.py`, `occ_ascii.py`).

    PYTHONPATH=src python3 scripts/rsi/tools/occ.py <x> <y> [--r 1.2] [--mark name=x,y ...] [--scene <scene dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "src")]
SCENE = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars"


def main(argv=None) -> int:
    from locomani.primitives.navigator import build
    from locomani.tasks.scene.world import BODY_RADIUS_M

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("--r", type=float, default=1.2, help="half width of the window, m")
    ap.add_argument("--mark", action="append", default=[], help="name=x,y (the centre is marked C)")
    ap.add_argument("--scene", type=Path, default=SCENE)
    a = ap.parse_args(argv)
    meta = json.loads((a.scene / "scene_meta.json").read_text())
    occ, _ = build(meta, body_radius=BODY_RADIUS_M)
    marks = {"C": (a.x, a.y)}
    for m in a.mark:
        k, v = m.split("=")
        marks[k[:1].upper()] = tuple(float(u) for u in v.split(","))
    lo = occ.cell_of((a.x - a.r, a.y - a.r))
    hi = occ.cell_of((a.x + a.r, a.y + a.r))
    cells = {tuple(occ.cell_of(p)): k for k, p in marks.items()}
    print(
        f"cell {occ.res} m; # obstacle, + within the body radius {BODY_RADIUS_M} m, . free; rows top = +y, columns left = -x"
    )
    for gy in range(hi[1], lo[1] - 1, -1):
        row = ""
        for gx in range(lo[0], hi[0] + 1):
            inside = 0 <= gx < occ.occupied.shape[0] and 0 <= gy < occ.occupied.shape[1]
            sym = "#" if not inside or occ.occupied[gx, gy] else ("." if occ.free[gx, gy] else "+")
            row += cells.get((gx, gy)) or sym
        print(row)
    for k, p in marks.items():
        c = occ.cell_of(p)
        inside = 0 <= c[0] < occ.clearance.shape[0] and 0 <= c[1] < occ.clearance.shape[1]
        clr = float(occ.clearance[c[0], c[1]]) if inside else float("nan")
        state = "free for the robot's centre" if occ.is_free(p) else "not free (obstacle or within the body radius)"
        print(f"{k} ({p[0]:.3f}, {p[1]:.3f}): {state}, clearance to the nearest obstacle {clr:.2f} m")
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    raise SystemExit(main())
