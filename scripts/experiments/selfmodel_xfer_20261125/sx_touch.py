"""selfmodel_xfer_20261125 (revision 2), HumanoidArena: the scene-contact sensitivity cut, read-only.

First the touched fractions: a segment touched the scene when its peak contact force (sx_ha.add_touch: the task's
contact_forces sensor, every robot body except the ankle links, from the command to the end label) is above --thresh N.
Then sx_eval.py unchanged, with the touched xfer segments left out of its kept set (LocoManiLab's side untouched).
The main result stays sx_eval.py's on every segment; this is the supplement revision 2 asks for.
--zmin Z counts only contacts whose touching body was at least Z m high (0.2: the table and box, not the floor).
  sx_touch.py --lml <data3>/<ctrl> --xfer <out> [--keep settled|ended] [--thresh 1] [--zmin 0] [--json <out>]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sx_eval as E  # noqa: E402


def frac(m: np.ndarray, sel: np.ndarray) -> str:
    n = int(sel.sum())
    k = int((m & sel).sum())
    return f"{k}/{n} ({100 * k / max(n, 1):.1f} %)"


def main() -> None:
    argv = sys.argv[1:]
    thresh, zmin = 1.0, 0.0
    for flag in ("--thresh", "--zmin"):
        if flag in argv:
            i = argv.index(flag)
            v = float(argv[i + 1])
            del argv[i : i + 2]
            thresh, zmin = (v, zmin) if flag == "--thresh" else (thresh, v)
    xfer = Path(argv[argv.index("--xfer") + 1])
    D = E.DT.load(str(xfer / "data" / "all.npz"))
    tn, tz = np.asarray(D["touch_n"], float), np.asarray(D["touch_z"], float)
    fell, st, ch = np.asarray(D["fell"]) != 0, np.asarray(D["settled_end"]) > 0, np.asarray(D["chain"])
    hit = (tn > thresh) & (tz >= zmin)
    ok = ~fell
    print(f"scene contact (peak force from the command to the end label, ankle links excluded) > {thresh:g} N"
          f"{f', touching body at least {zmin:g} m high' if zmin > 0 else ''}, one-hand segments:")
    print(f"  all {frac(hit, np.ones_like(hit))}; not fallen / lost {frac(hit, ok)}; "
          f"kept by the main metric (not fallen, settled) {frac(hit, ok & st)}")
    print(f"  kept, first reach {frac(hit, ok & st & (ch == 0))}; chained {frac(hit, ok & st & (ch > 0))}; "
          f"fallen / lost {frac(hit, fell)}")
    if hit.any():
        print(f"  touched segments: peak force median {np.median(tn[hit]):.1f} N (p90 {np.percentile(tn[hit], 90):.1f}); "
              f"height of the touching body median {np.median(tz[hit]):.3f} m (p10 {np.percentile(tz[hit], 10):.3f}, "
              f"p90 {np.percentile(tz[hit], 90):.3f})")
    orig = E.kept

    def kept(Dd: dict, idx: np.ndarray | None = None, ended: bool = False) -> np.ndarray:
        k = orig(Dd, idx, ended)
        if "touch_n" in Dd:  # the xfer record only (LocoManiLab's data3 has no contact field)
            k = k[~((np.asarray(Dd["touch_n"], float)[k] > thresh) & (np.asarray(Dd["touch_z"], float)[k] >= zmin))]
        return k

    E.kept = kept
    sys.argv = [sys.argv[0], *argv]
    print(f"sensitivity: sx_eval.py with those touched xfer segments left out")
    E.main()


if __name__ == "__main__":
    main()
