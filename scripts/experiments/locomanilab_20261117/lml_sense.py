"""Step 4 (2026-11-18): does the self model beat feedback once feedback's hand measurement is not ideal?

Step 3's review: with the servo's cap lifted, an ideally measured feedback servo placed the hand as well as the self
model; what is left to the self model is that it needs no measurement during the reach. Here the servo (lml_tasks
fb_k<ki>_<sensor>: cap FB_CAP, reference read where the policy tracks it, running to the end) gets a degraded hand
measurement, four families at three levels, two gains each (the better gain per level is taken, which favours feedback):

  noise      n1 n2 n4        white noise per 0.1 s update, sigma cm per axis
  latency    l100 l200 l400  ms, timestamp aligned
  occlusion  o3 o6 o10       no measurement while the hand is within r cm of the goal
  bias       b1 b2 b4        constant offset, cm, random direction per episode

All methods get the same exact target at the command; only the measurement feedback needs during the reach is degraded.
Placement = the hand point's distance to the goal ERR_OFFS[PRIMARY] frames (1.0 s) after the self model's label time
(2.0 s after the reference arrives), the first waypoint only (the pre-touch point, in free space), held 2 s longer.

Sets: IThorKitchenCloseCabinet and IThorKitchenCloseOven (episodes 50-99, as step 3) and FreeSpace (0-99).

    python lml_sense.py conds                 # the comma list for lml_tasks.py run --methods
    python lml_sense.py stats --out DIR       # the pre-registered analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

SENSORS = {"noise": ("n1", "n2", "n4"), "latency": ("l100", "l200", "l400"), "occlusion": ("o3", "o6", "o10"),
           "bias": ("b1", "b2", "b4")}  # fmt: skip
GAINS = (1, 2)
LEVELS = ("ideal",) + tuple(s for v in SENSORS.values() for s in v)
CONDS = ("alone", "selfmodel", "selfmodel_s1") + tuple(f"fb_k{k}_{s}" for k in GAINS for s in LEVELS)
SETS = {"IThorKitchenCloseCabinet": (50, 100), "IThorKitchenCloseOven": (50, 100), "FreeSpace": (0, 100)}
PRIMARY = 2  # index into lml_tasks.ERR_OFFS (0, 25, 50, 75, 100): +50 frames = 1.0 s after the label time
ALPHA = 0.01


def sign_p(k: int, n: int) -> float:
    """Two-sided exact sign test: k successes of n."""
    if n == 0:
        return 1.0
    t = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2**n)


def err(r: dict, idx: int) -> float:
    """The first waypoint's placement error (cm) at ERR_OFFS[idx]; inf if the episode ended before it (fell / lost)."""
    w = r["waypoints"][0] if r["waypoints"] else None
    if w is None or w.get("err_t_cm") is None or w["err_t_cm"][idx] is None:
        return math.inf
    return float(w["err_t_cm"][idx])


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


def paired(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float, float]:
    """a vs b per episode (lower is better): a better count, n non-tied, median of b - a (finite pairs), sign-test p."""
    with np.errstate(invalid="ignore"):
        d = b - a
    nz = (d != 0) & ~np.isnan(d)  # both inf (both ended early): a tie
    k = int((d[nz] > 0).sum())
    fin = np.isfinite(d)
    return k, int(nz.sum()), float(np.median(d[fin])) if fin.any() else math.nan, sign_p(k, int(nz.sum()))


def stats(out: Path) -> None:
    R, skip = load(out)
    print(f"{len(R)} episodes; skipped {dict((k, sorted(v)) for k, v in skip.items())}")
    for s, (lo, hi) in SETS.items():
        eps = [e for e in range(lo, hi) if all((s, c, e) in R for c in CONDS)]
        missing = [(c, e) for c in CONDS for e in range(lo, hi) if (s, c, e) not in R and e not in skip.get(s, set())]
        print(f"\n== {s}: {len(eps)} common episodes; missing (not skipped) {len(missing)}")
        E = {c: np.array([[err(R[(s, c, e)], i) for e in eps] for i in range(5)]) for c in CONDS}
        for c in CONDS:
            rs = [R[(s, c, e)] for e in eps]
            ex = ""
            if c.startswith("fb_"):
                st = [r["waypoints"][0].get("servo_stat", {}) for r in rs]
                ex = (f" | at cap {sum(x.get('at_cap', 0) > 0 for x in st):3d} eps, blind updates "
                      f"{np.mean([x.get('blind', 0) / max(x.get('updates', 0) + x.get('blind', 0), 1) for x in st]):.2f}")  # fmt: skip
            m = lambda i: np.median(E[c][i])  # noqa: E731
            print(f"  {c:14s} +1 s {m(PRIMARY):6.2f} | label time {m(0):6.2f} +2 s {m(4):6.2f} | <=1 cm {np.mean(E[c][PRIMARY] <= 1):.2f} "
                  f"<=2 cm {np.mean(E[c][PRIMARY] <= 2):.2f} | fell {sum(r['fell'] is not None for r in rs)} lost {sum(r['lost'] is not None for r in rs)}{ex}")  # fmt: skip
        sm = E["selfmodel"][PRIMARY]
        print(f"  -- self model (s0) vs feedback at +1 s; better gain per level = lower median (oracle, favours feedback);"
              f" diff = feedback - self model (cm, > 0: self model better)")
        for lv in LEVELS:
            best = min(GAINS, key=lambda k: np.median(E[f"fb_k{k}_{lv}"][PRIMARY]))
            parts = []
            for k in GAINS:
                fb = E[f"fb_k{k}_{lv}"][PRIMARY]
                nb, n, dmed, p = paired(sm, fb)
                parts.append(f"k{k}{'*' if k == best else ' '} {np.median(fb):5.2f}: sm better {nb:3d}/{n:3d} diff {dmed:+5.2f} p {p:.1e}")
            fb = E[f"fb_k{best}_{lv}"][PRIMARY]
            nb, n, dmed, p = paired(sm, fb)
            verdict = "SELF MODEL" if (p < ALPHA and dmed > 0) else ("FEEDBACK" if (p < ALPHA and dmed < 0) else "tie")
            print(f"    {lv:6s} " + " | ".join(parts) + f" || best-gain: {verdict}")
        print("  -- crossover (smallest level where the self model beats the better gain, p < 0.01, diff > 0):")
        for fam, lvs in SENSORS.items():
            won = []
            for lv in lvs:
                best = min(GAINS, key=lambda k: np.median(E[f"fb_k{k}_{lv}"][PRIMARY]))
                nb, n, dmed, p = paired(sm, E[f"fb_k{best}_{lv}"][PRIMARY])
                won.append(p < ALPHA and dmed > 0)
            first = next((lv for lv, w in zip(lvs, won) if w), None)
            print(f"    {fam:9s} {first or 'none'} ({' '.join(f'{lv}:{int(w)}' for lv, w in zip(lvs, won))})")
        nb, n, dmed, p = paired(sm, E["selfmodel_s1"][PRIMARY])
        print(f"  -- s0 vs s1 at +1 s: s0 better {nb}/{n}, diff {dmed:+.2f}, p {p:.1e}; alone {np.median(E['alone'][PRIMARY]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["conds", "stats"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.phase == "conds":
        print(",".join(CONDS))
    else:
        stats(a.out)
