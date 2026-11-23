"""selfmodel_via_20261120: how the hand's attitude answered a wrist turn that a compensated command asked for.

    $PY via_turn.py <direct dir> <compensated dir> [table point, default 24 = the settled end; 12 = mid-path] [model dir]

Pairs as via_comp_eval.py (same reset index, the same wanted sweep within 1 mm). The hand is the 15 spheres of the palm
and the fingers; a rotation between two placements of it is the least-squares one (via_comp.kabsch), as a rotation vector
in the heading frame. Per pair at the table point:
  error      wanted -> direct run: the attitude error to undo
  commanded  wanted -> the compensated command's plan: the turn the command asked for
  actual     direct run -> compensated run: what the hand did
  predicted  the model's prediction for the direct command -> for the compensated one: the predictions the run saved
             (comp_off0, comp_off), or <model dir>'s for the same two commands when it is given
Printed: medians of the four sizes, the cosines actual . commanded and predicted . commanded, the error left; then the
same by size of the commanded turn. A model that knows the controller has predicted / commanded = actual / commanded and
both cosines near 1.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import via_comp as VC  # noqa: E402
import via_gen as VG  # noqa: E402
import via_point as VP  # noqa: E402


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    DD, DC = Path(a[0]), Path(a[1])
    q = int(a[2]) if len(a) > 2 else 24
    Td, Tc, zc = VP.load([DD]), VP.load([DC]), dict(np.load(DC / "all.npz"))
    kd, kc = Td["reset"] % 100000, Tc["reset"] % 100000
    hand = np.array([str(x).startswith("hand_") for x in np.load(DD / "arm.npz")["link"]])
    off_d = off_c = None
    if len(a) > 3:  # another model's predictions for the two commands
        import torch

        M = torch.load(Path(a[3]) / "via_point.pt", map_location="cpu", weights_only=False)
        off_d = VP.offsets(M, Td, np.arange(len(kd)))["direction"]
        off_c = VP.offsets(M, Tc, np.arange(len(kc)))["direction"]
    rot = lambda A, B: VG.logm(VC.kabsch(A, B))  # noqa: E731
    full = lambda r: np.concatenate([r, r[-1:]])  # noqa: E731
    cos = lambda u, v: float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9))  # noqa: E731
    deg = lambda v: float(np.degrees(np.linalg.norm(v)))  # noqa: E731
    rows = []
    for k in np.intersect1d(kd, kc):
        jd, jc = int(np.where(kd == k)[0][0]), int(np.where(kc == k)[0][0])
        s = Tc["seg"][jc]
        ref0 = zc["comp_ref0"][s].astype(float)
        if np.abs(ref0 - Td["ref"][jd]).max() > 1e-3:
            continue
        W, P = full(ref0)[q, hand], full(zc["comp_ref"][s].astype(float))[q, hand]
        pd = W + (zc["comp_off0"][s] if off_d is None else off_d[jd])[q, hand]
        pc = P + (zc["comp_off"][s] if off_c is None else off_c[jc])[q, hand]
        Ad = (full(Td["ref"][jd].astype(float)) + Td["off"][jd])[q, hand]
        Ac = (full(Tc["ref"][jc].astype(float)) + Tc["off"][jc])[q, hand]
        e, c, r, p = rot(W, Ad), rot(W, P), rot(Ad, Ac), rot(pd, pc)
        rows.append([deg(e), deg(c), deg(r), deg(p), cos(r, c), cos(p, c), deg(rot(W, Ac))])
    t = np.array(rows)
    print(f"{DC.parent.name}/{DC.name}, table point {q}: {len(t)} pairs")
    line = lambda m: (  # noqa: E731
        f"error {np.median(t[m, 0]):5.1f}  commanded {np.median(t[m, 1]):5.1f}  actual {np.median(t[m, 2]):5.1f}  predicted "
        f"{np.median(t[m, 3]):5.1f} deg | actual / commanded {np.median(t[m, 2] / np.maximum(t[m, 1], 1e-6)):.2f}, predicted / "
        f"commanded {np.median(t[m, 3] / np.maximum(t[m, 1], 1e-6)):.2f} | cos actual {np.median(t[m, 4]):+.2f}, predicted "
        f"{np.median(t[m, 5]):+.2f} | error left {np.median(t[m, 6]):5.1f} deg"
    )
    print("  all                   " + line(np.ones(len(t), bool)))
    for lo, hi in ((5, 15), (15, 25), (25, 90)):
        m = (t[:, 1] > lo) & (t[:, 1] <= hi)
        if m.sum() >= 3:
            print(f"  commanded {lo:2d}-{hi:2d} ({int(m.sum()):3d}) " + line(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
