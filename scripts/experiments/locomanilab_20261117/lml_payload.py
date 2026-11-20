"""Step 5 (2026-11-19): a payload the self model never saw, and the self model + residual servo.

Step 4 degraded only feedback's measurement. Here the self model is made wrong instead: a point mass in the acting hand
(lml_tasks `@p<kg>`; the controller is not told, the self model was trained empty-handed). Methods per payload:

  alone, selfmodel                  as step 4
  fb_k<ki>_<sensor>                 step 4's servo (from the command, reference minus measured hand)
  smfb_k<ki>_<sensor>               the self model's command, then the servo from the label time on goal minus hand
  sensors: ideal, n2 (2 cm white noise per update), o6 (blind within 6 cm of the goal), b2 (2 cm hand-only offset);
  ideal also with the cap at 30 cm (`ideal_c30`): smoke 2026-11-19, 2 kg sank the hand ~20 cm beyond the empty hand

Placement, sets and episodes as step 4 (lml_sense): first waypoint, +2 s hold, error 1.0 s after the label time;
cabinet / oven 50-99, FreeSpace 0-99. The better of the two gains per (method, sensor, payload) is taken, by median.

    python lml_payload.py conds               # the comma list for lml_tasks.py run --methods
    python lml_payload.py stats --out DIR     # the pre-registered analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import lml_sense as S
import numpy as np

PAYLOADS = (0.0, 0.5, 1.0, 2.0)
SENSORS = ("ideal", "ideal_c30", "n2", "o6", "b2")
GAINS = (1, 2)
BASE = ("alone", "selfmodel") + tuple(f"{k}_k{g}_{s}" for k in ("fb", "smfb") for g in GAINS for s in SENSORS)
CONDS = tuple(f"{b}@p{p:g}" for p in PAYLOADS for b in BASE)
SETS = S.SETS
PRIMARY = S.PRIMARY
ALPHA = 0.01


def load(out: Path) -> tuple[dict, dict]:
    R, skip = {}, {}
    for s in SETS:
        for c in CONDS:
            for p in glob.glob(str(out / "episodes" / s / c / "ep*.json")):
                r = json.loads(Path(p).read_text())
                R[(s, c, r["ep"])] = r
            for p in glob.glob(str(out / "episodes" / s / c / "ep*.skip")):
                skip.setdefault(s, set()).add(int(Path(p).stem[2:]))
    return R, skip


def verdict(a: np.ndarray, b: np.ndarray, na: str, nb: str) -> str:
    """a vs b (lower is better), sign test."""
    k, n, d, p = S.paired(a, b)
    who = na if (p < ALPHA and d > 0) else (nb if (p < ALPHA and d < 0) else "tie")
    return f"{na} better {k:3d}/{n:3d}, diff ({nb} - {na}) {d:+5.2f}, p {p:.1e} -> {who}"


def stats(out: Path) -> None:
    R, skip = load(out)
    print(f"{len(R)} episodes; skipped {dict((k, sorted(v)) for k, v in skip.items())}")
    for s, (lo, hi) in SETS.items():
        eps = [e for e in range(lo, hi) if all((s, c, e) in R for c in CONDS)]
        missing = [(c, e) for c in CONDS for e in range(lo, hi) if (s, c, e) not in R and e not in skip.get(s, set())]
        print(f"\n== {s}: {len(eps)} common episodes; missing (not skipped) {len(missing)}")
        E = {c: np.array([[S.err(R[(s, c, e)], i) for e in eps] for i in range(5)]) for c in CONDS}
        best = {}
        for p in PAYLOADS:
            tag = f"@p{p:g}"
            sm = R[(s, f"selfmodel{tag}", eps[0])]
            pe = [R[(s, f"selfmodel{tag}", e)]["waypoints"][0].get("pred_err_cm") for e in eps]
            pe = [v for v in pe if v is not None]
            print(f"  -- payload {p:g} kg (self model's own prediction miss at the label time, median {np.median(pe):.2f} cm; "
                  f"payload side {sm.get('payload_side')})")  # fmt: skip
            for b in BASE:
                c = b + tag
                rs = [R[(s, c, e)] for e in eps]
                m = lambda i: np.median(E[c][i])  # noqa: E731
                ex = ""
                if "_k" in b:
                    st = [r["waypoints"][0].get("servo_stat", {}) for r in rs]
                    ex = f" | at cap {sum(x.get('at_cap', 0) > 0 for x in st):3d}"
                print(f"    {c:22s} +1 s {m(PRIMARY):6.2f} | label time {m(0):6.2f} +2 s {m(4):6.2f} | <=1 cm {np.mean(E[c][PRIMARY] <= 1):.2f} "
                      f"<=2 cm {np.mean(E[c][PRIMARY] <= 2):.2f} | fell {sum(r['fell'] is not None for r in rs)} lost {sum(r['lost'] is not None for r in rs)}{ex}")  # fmt: skip
            for kind in ("fb", "smfb"):
                for sen in SENSORS:
                    g = min(GAINS, key=lambda k: np.median(E[f"{kind}_k{k}_{sen}{tag}"][PRIMARY]))
                    best[(kind, sen, p)] = E[f"{kind}_k{g}_{sen}{tag}"][PRIMARY]
                    best[(kind, sen, p, "gain")] = g
        sm = {p: E[f"selfmodel@p{p:g}"][PRIMARY] for p in PAYLOADS}
        print("  -- comparisons at +1 s (better gain per method, sensor and payload; sign test, p < 0.01)")
        for p in PAYLOADS:
            g = lambda k, x: best[(k, x, p, "gain")]  # noqa: E731
            print(f"   payload {p:g} kg")
            print(
                f"     selfmodel vs fb[ideal k{g('fb', 'ideal')}]      : {verdict(sm[p], best[('fb', 'ideal', p)], 'selfmodel', 'fb')}"
            )
            print(
                f"     selfmodel vs fb[ideal_c30 k{g('fb', 'ideal_c30')}]  : {verdict(sm[p], best[('fb', 'ideal_c30', p)], 'selfmodel', 'fb')}"
            )
            print(
                f"     smfb[ideal_c30 k{g('smfb', 'ideal_c30')}] vs fb[ideal_c30 k{g('fb', 'ideal_c30')}] : {verdict(best[('smfb', 'ideal_c30', p)], best[('fb', 'ideal_c30', p)], 'smfb', 'fb')}"
            )
            print(
                f"     smfb[ideal k{g('smfb', 'ideal')}] vs fb[ideal k{g('fb', 'ideal')}] : {verdict(best[('smfb', 'ideal', p)], best[('fb', 'ideal', p)], 'smfb', 'fb')}"
            )
            print(
                f"     smfb[ideal k{g('smfb', 'ideal')}] vs selfmodel     : {verdict(best[('smfb', 'ideal', p)], sm[p], 'smfb', 'selfmodel')}"
            )
            for sen in SENSORS[2:]:
                print(
                    f"     [{sen}] smfb k{g('smfb', sen)} vs fb k{g('fb', sen)}     : {verdict(best[('smfb', sen, p)], best[('fb', sen, p)], 'smfb', 'fb')}"
                )
                print(
                    f"     [{sen}] smfb k{g('smfb', sen)} vs selfmodel   : {verdict(best[('smfb', sen, p)], sm[p], 'smfb', 'selfmodel')}"
                )
                print(
                    f"     [{sen}] selfmodel vs fb k{g('fb', sen)}  : {verdict(sm[p], best[('fb', sen, p)], 'selfmodel', 'fb')}"
                )
        print("  -- growth with payload (median at +1 s): " + " | ".join(
            f"{n} " + " / ".join(f"{np.median(v[p]):.2f}" for p in PAYLOADS)
            for n, v in (("alone", {p: E[f'alone@p{p:g}'][PRIMARY] for p in PAYLOADS}), ("selfmodel", sm),
                         ("fb ideal", {p: best[('fb', 'ideal', p)] for p in PAYLOADS}),
                         ("smfb ideal", {p: best[('smfb', 'ideal', p)] for p in PAYLOADS}),
                         ("fb ideal_c30", {p: best[('fb', 'ideal_c30', p)] for p in PAYLOADS}),
                         ("smfb ideal_c30", {p: best[('smfb', 'ideal_c30', p)] for p in PAYLOADS}))))  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["conds", "stats"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.phase == "conds":
        print(",".join(CONDS))
    else:
        stats(a.out)
