"""selfmodel_via_20261120: one compensated run against another on the same routes, pair by pair.

    python3 -I via_vs.py <baseline compensated dir> <compensated dir> ...

Every dir has comp_pairs.npz (via_comp_eval.py against the same direct run, so the pairs line up). Only the detour routes
are compared (every kind but FreshUnder). Printed per run and per measure (wrist + palm + fingers, palm, all 24 spheres,
the palm's settled end; errors against the wanted sweep): both medians, the drop of the median and of the 90 % level with
95 % intervals from resampling routes, and the share of pairs on which the run beats the baseline. Then the share of
reaches whose wrist + palm + fingers still leave the wanted sweep by more than 9 cm, in both runs, and the difference of
the two shares with its 95 % interval (routes resampled).
"""

import sys

import numpy as np

P = dict(np.load(sys.argv[1] + "/comp_pairs.npz"))
rng = np.random.default_rng(0)
for d in sys.argv[2:]:
    Q = dict(np.load(d + "/comp_pairs.npz"))
    _, i, j = np.intersect1d(P["reset"], Q["reset"], return_indices=True)  # a pair dropped in one run only is left out
    P1, Q = {k: v[i] for k, v in P.items()}, {k: v[j] for k, v in Q.items()}
    keep = P1["kind"] != "FreshUnder"
    P1, Q = {k: v[keep] for k, v in P1.items()}, {k: v[keep] for k, v in Q.items()}
    route = P1["route"]
    g = [np.where(route == r)[0] for r in np.unique(route)]

    def drop(a, b, f=np.median):
        bs = []
        for _ in range(2000):
            j = np.concatenate([g[k] for k in rng.integers(len(g), size=len(g))])
            bs.append(1 - f(b[j]) / f(a[j]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return f"{1 - f(b) / f(a):.2f} ({lo:.2f} to {hi:.2f})"

    p90 = lambda v: np.percentile(v, 90)
    print(d.rstrip("/").split("/")[-1], f"| {len(route)} pairs, {len(g)} routes")
    for k, name in (
        ("through", "wrist + palm + fingers"),
        ("palm", "palm"),
        ("arm", "arm (24 spheres)"),
        ("end", "palm's settled end"),
    ):
        a, b = P1[k + "_comp"], Q[k + "_comp"]
        print(
            f"  {name:22s} baseline {100 * np.median(a):5.2f} cm, this run {100 * np.median(b):5.2f} cm; drop of the median {drop(a, b)}; of the 90 % level {100 * p90(a):5.2f} -> {100 * p90(b):5.2f}: {drop(a, b, p90)}; better on {np.mean(b < a):.2f} of the pairs"
        )
    a, b = P1["through_comp"] > 0.09, Q["through_comp"] > 0.09
    bs = []
    for _ in range(2000):
        j = np.concatenate([g[k] for k in rng.integers(len(g), size=len(g))])
        bs.append(b[j].mean() - a[j].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(
        f"  wrist + palm + fingers still over 9 cm: baseline {a.sum()}/{len(a)} ({a.mean():.3f}), this run {b.sum()}/{len(b)} ({b.mean():.3f}); "
        f"difference {b.mean() - a.mean():+.3f} ({lo:+.3f} to {hi:+.3f}); routes with any: {len(set(route[a]))} and {len(set(route[b]))}"
    )
