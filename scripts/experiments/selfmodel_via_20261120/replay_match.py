"""selfmodel_via_20261120: the same replays run on two machines -- do they agree?

    $PY replay_match.py <model dir> <replay dir A> [<replay dir B>]

A: the swept-region error of a replay set by the route's kind and by pass, as the model dirs' by_kind.py prints it (same
functions: via_point.offsets and errors), plus one line for the detour routes (every kind but FreshUnder). Needs only the
model file and the replay, not the model's training data.
B: the same routes, seed and passes run on another machine. Each reset of A is set against the same reset of B (reset id
minus its run's ID0, which is a multiple of 100000): how far apart the palm started, and the difference of the error.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import via_point as VP  # noqa: E402

MD, RS = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
M = torch.load(MD / "via_point.pt", map_location="cpu", weights_only=False)
WAYS = ("reference", "constant", "direction")


def table(R: Path):
    T = VP.load([R])
    rows = np.arange(len(T["seg"]))
    off = VP.offsets(M, T, rows)
    e = {p: VP.errors([R], T, off[p], rows)[0] for p in WAYS}
    names = json.load(open(R / "routes.json"))["names"]
    kind = np.array([names[r].split("_")[0] for r in T["route"]])
    return T, e, kind


def line(tag: str, T: dict, e: dict, m: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    b, d, c = e["reference"][m], e["direction"][m], e["constant"][m]
    ids = T["route"][m]
    g = [np.where(ids == u)[0] for u in np.unique(ids)]
    bs = []
    for _ in range(2000):
        j = np.concatenate([g[k] for k in rng.integers(len(g), size=len(g))])
        bs.append(1 - np.median(d[j]) / np.median(b[j]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(
        f"{tag:34s} routes {len(g):2d} replays {int(m.sum()):3d} | reference {100 * np.median(b):5.2f} "
        f"constant {100 * np.median(c):5.2f} direction {100 * np.median(d):5.2f} cm | drop "
        f"{1 - np.median(d) / np.median(b):.2f} ({lo:.2f} to {hi:.2f}) | direction better on {np.mean(d < b):.2f} | "
        f"palm path {np.median(T['length'][m]):.2f} m"
    )


tabs = []
for R in RS:
    T, e, kind = table(R)
    tabs.append((T, e))
    print(f"== {R}")
    line("all", T, e, np.ones(len(kind), bool))
    for k in dict.fromkeys(kind.tolist()):
        line(k, T, e, kind == k)
    line("first pass (no walk)", T, e, T["walked"] == 0)
    line("later passes (walked)", T, e, T["walked"] > 0)
    line("detour (box + edge)", T, e, kind != "FreshUnder")

if len(tabs) == 2:
    (TA, eA), (TB, eB) = tabs
    ka, kb = TA["reset"] % 100000, TB["reset"] % 100000
    common = np.intersect1d(ka, kb)
    ia = np.array([np.where(ka == k)[0][0] for k in common])
    ib = np.array([np.where(kb == k)[0][0] for k in common])
    assert (TA["route"][ia] == TB["route"][ib]).all(), "the two runs put different routes on the same reset"
    palm = TA["key"][-1]
    start = 100 * np.linalg.norm(TA["ref"][ia, 0, palm] - TB["ref"][ib, 0, palm], axis=1)
    print(f"== A against B, reset by reset: {len(common)} resets in both (A {len(ka)}, B {len(kb)})")
    for tag, m in (("all", np.ones(len(ia), bool)), ("first pass (no walk)", TA["walked"][ia] == 0),
                   ("later passes (walked)", TA["walked"][ia] > 0)):  # fmt: skip
        s = start[m]
        print(f"  {tag:22s} palm's start apart: median {np.median(s):.3f} cm, 95 % {np.percentile(s, 95):.3f}, "
              f"max {s.max():.3f}; identical (< 0.001 cm) on {np.mean(s < 1e-3):.2f}")  # fmt: skip
        for p in ("reference", "direction"):
            d = 100 * (eA[p][ia] - eB[p][ib])[m]
            print(f"    {p:9s} error A - B: median {np.median(d):+.3f} cm, |A - B| median {np.median(np.abs(d)):.3f}, "
                  f"95 % {np.percentile(np.abs(d), 95):.3f}, max {np.abs(d).max():.3f}")  # fmt: skip
