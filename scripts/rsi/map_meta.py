"""Vision RSI D2 step 2d (docs/handoff/vision-rsi-design-20261023.md section 17.7): write the perceived scene description
(locomani.tasks.scene.mapped) from a map built by map_build.py, with the parts asked for, for
`perceive.RenderConfig.mapped`. Prints what it replaced and what kept the truth.

    PYTHONPATH=src /workspace/envs/vision_map/bin/python scripts/rsi/map_meta.py \\
        outputs/vision_rsi_20261023/tour/map_kiss outputs/vision_rsi_20261023/tour/meta_grid.json --parts grid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from locomani.tasks.scene.mapped import PARTS, perceived_meta  # noqa: E402

BODY_RADIUS_M = 0.35  # scene.world's


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", type=Path, help="a map_build.py output directory")
    ap.add_argument("out", type=Path)
    ap.add_argument("--parts", nargs="+", choices=PARTS, default=["grid"])
    ap.add_argument("--record", type=Path, default=REPO / "outputs/vision_rsi_20261023/tour/record",
                    help="the tour's record directory (its scene)")  # fmt: skip
    a = ap.parse_args(argv)
    tour = json.loads((a.record / "tour.json").read_text())
    meta = json.loads((REPO / tour["scene"] / "scene_meta.json").read_text())
    grid = dict(np.load(a.map / "grid.npz"))
    surfaces = json.loads((a.map / "surfaces.json").read_text())
    match = json.loads((a.map / "surface_match.json").read_text())
    out = perceived_meta(meta, grid, surfaces, match, {}, parts=tuple(a.parts), body_radius=BODY_RADIUS_M)
    out["perceived"]["map"] = str(a.map)
    a.out.write_text(json.dumps(out))
    p = out["perceived"]
    print(json.dumps(dict(parts=p["parts"], obstacles=len(out["room"]["obstacles"]),
                          surfaces_replaced=len(match), surfaces_kept=len(p["surfaces_kept"]))))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
