"""Query the measured reach envelope: how much x margin does a box pose have?

    python -m locomani.tasks.crate.reach_envelope.query <run_dir> [<run_dir>...] -- x y h d yaw_deg

Merges the boundary tables of several search runs (later runs override earlier ones on the same
(y, h, d, yaw) line, so list the crouch runs last), interpolates the 2 cm position boundary x over
(y, h, d, yaw), and reports margin = boundary - x. The envelope is a floor-only, planted-feet,
position-only measurement (see docs/reach_envelope.md §7); treat margin as necessary, not sufficient.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

X_FLOOR = 0.175  # a line that never passed down to 0.20 m is scored as a boundary just below it
NOISE_M = 0.05  # one search step: repeat trials flipped the verdict within this band (13/100)


class ReachEnvelope:
    def __init__(self, run_dirs):
        cells = {}
        for rd in run_dirs:
            rd = Path(rd)
            for r in json.load(open(rd / "boundary_table.json")):
                if r["trials"] == 0:
                    continue
                xb = X_FLOOR if r["x_pass"] is None else r["x_pass"] + 0.025
                cells[(round(r["y"], 3), round(r["h"], 3), round(r["d"], 3), round(r["yaw"], 1))] = xb
        self.axes = tuple(sorted({k[i] for k in cells}) for i in range(4))
        grid = np.full([len(a) for a in self.axes], np.nan)
        for k, v in cells.items():
            grid[tuple(a.index(k[i]) for i, a in enumerate(self.axes))] = v
        self.grid, self.coverage = grid, float(np.isfinite(grid).mean())
        self._f = RegularGridInterpolator(self.axes, grid, bounds_error=False, fill_value=np.nan)

    def boundary_x(self, y, h, d, yaw):
        """Interpolated 2 cm-position boundary along x, NaN outside the measured (y, h, d, yaw) box."""
        return float(self._f([[y, h, d, yaw]])[0])

    def query(self, x, y, h, d, yaw):
        xb = self.boundary_x(y, h, d, yaw)
        inside = all(a[0] - 1e-9 <= v <= a[-1] + 1e-9 for a, v in zip(self.axes, (y, h, d, yaw)))
        margin = None if np.isnan(xb) else xb - x
        return {
            "inside_measured_box": inside,
            "boundary_x_m": None if np.isnan(xb) else round(xb, 3),
            "margin_m": None if margin is None else round(margin, 3),
            "verdict": (
                "unknown"
                if margin is None
                else "reach"
                if margin >= NOISE_M
                else "marginal"
                if margin >= 0
                else "no_reach"
            ),
            "crouch_needed": h < 0.50,
            "axes": {n: (a[0], a[-1]) for n, a in zip(("y", "h", "d", "yaw"), self.axes)},
        }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        print(__doc__)
        return 2
    i = argv.index("--")
    env = ReachEnvelope(argv[:i])
    x, y, h, d, yaw = map(float, argv[i + 1 :])
    out = env.query(x, y, h, d, yaw)
    out["grid_coverage"] = round(env.coverage, 3)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
