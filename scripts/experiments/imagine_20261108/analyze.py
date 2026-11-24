"""How much of the real diagnostic outcomes an imagined batch predicts, against predictors that cost nothing (TODO 45).

    python3 scripts/experiments/imagine_20261108/analyze.py <batch dir>

Baselines, on the drafts' rounds: "always ok"; "start" = the share of the other drafts that won this start, so it knows
which starts are hard without looking at the draft; "draft" = the same draft's real wins on its other starts.
Within an iteration: does the draft imagination ranks first really win most, against picking at random.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load(p) -> dict:
    return json.loads(Path(p).read_text())


def main(out: str) -> int:
    jobs = load(Path(out) / "manifest.json")
    R, I, meta = {}, {}, {}
    for j in jobs:
        for l in j["labels"]:
            f = Path(j["out"]) / f"{l}.json"
            if f.exists():
                R[j["id"], l], I[j["id"], l] = load(Path(j["real"]) / f"{l}.json"), load(f)
        m = re.search(r"/([^/]+)/outputs/.*/loop/(iter_\d+)/(candidate[^/]*)/skills", j["pkg"])
        meta[j["id"]] = dict(
            group=(m.group(1), m.group(2)) if m else None,
            name=m.group(3) if m else "base",
            inc=j["incumbent"],
            pkg=j["pkg"],
        )
    drafts = sorted({k[0] for k in R if not meta[k[0]]["inc"]})
    labels = sorted({k[1] for k in R})
    keys = [(d, l) for d in drafts for l in labels if (d, l) in R]
    y = np.array([R[k]["ok"] for k in keys], float)
    imag = np.array([I[k]["ok"] for k in keys], float)
    A = np.full((len(drafts), len(labels)), np.nan)
    for (d, l), v in zip(keys, y):
        A[drafts.index(d), labels.index(l)] = v
    start, draft = np.zeros(len(keys)), np.zeros(len(keys))
    for n, (d, l) in enumerate(keys):
        i, j = drafts.index(d), labels.index(l)
        start[n] = np.nanmean(np.delete(A[:, j], i))
        draft[n] = np.nanmean(np.delete(A[i], j))
    print(f"draft rounds {len(keys)}, real win rate {y.mean():.3f}")
    print(
        f"agreement: imagined {np.mean(imag == y):.3f}; always ok {max(y.mean(), 1 - y.mean()):.3f}; start {np.mean((start >= 0.5) == y):.3f}; draft {np.mean((draft >= 0.5) == y):.3f}; start+draft {np.mean(((start + draft) / 2 >= 0.5) == y):.3f}"
    )

    def auc(score) -> float:
        pos, neg = score[y == 1], score[y == 0]
        return float(np.mean([(pos > v).mean() + 0.5 * (pos == v).mean() for v in neg]))

    print(
        f"AUC: imagined {auc(imag):.3f}; start {auc(start):.3f}; draft {auc(draft):.3f}; start+draft {auc(start + draft):.3f}; start+draft+imagined {auc(start + draft + imag):.3f}"
    )
    groups = defaultdict(list)
    for d in drafts:
        if meta[d]["group"]:
            groups[meta[d]["group"]].append(d)
    hit = rnd = n = 0
    for _g, ds in groups.items():
        if len(ds) < 2:
            continue
        real = {d: sum(R[d, l]["ok"] for l in labels if (d, l) in R) for d in ds}
        im = {d: sum(I[d, l]["ok"] for l in labels if (d, l) in R) for d in ds}
        top_i = [d for d in ds if im[d] == max(im.values())]
        best = max(real.values())
        hit += np.mean([real[d] == best for d in top_i])
        rnd += np.mean([real[d] == best for d in ds])
        n += 1
    print(
        f"within an iteration ({n} groups): imagined top is a real best {hit / max(n, 1):.2f}; random draft {rnd / max(n, 1):.2f}"
    )
    print(f"AUC with the start prior: start+imagined {auc(start + imag):.3f}")

    def miss(rec) -> float:
        """The body table's worst odds of a 3 cm miss over the round's reaches (recorded by Motor.reach_note)."""
        return max(
            [float(s["pred_p_miss"]) for s in rec.get("steps") or [] if s.get("pred_p_miss") is not None] or [0.0]
        )

    pm_i = np.array([miss(I[k]) for k in keys])
    pm_r = np.array([miss(R[k]) for k in keys])
    print(
        f"body table: rounds with a prediction, imagined {np.mean(pm_i > 0):.2f}, real {np.mean(pm_r > 0):.2f}; worst odds alone AUC imagined {auc(-pm_i):.3f}, real-state {auc(-pm_r):.3f}"
    )
    print(
        f"imagined x (1 - worst odds): AUC {auc(imag * (1 - pm_i)):.3f}; with start prior {auc(start + imag * (1 - pm_i)):.3f}"
    )
    for tau in (0.3, 0.5, 0.7):
        pred = (imag == 1) & (pm_i < tau)
        print(
            f"  tau {tau}: agreement {np.mean(pred == y):.3f}, says fail {np.mean(~pred):.3f}, of those really fail {y[~pred].size and 1 - y[~pred].mean():.3f}"
        )
    print("real class -> imagined ok / imagined fail (draft rounds):")
    c = Counter((R[k].get("failure_class"), bool(I[k]["ok"])) for k in keys)
    for cls in sorted({x[0] for x in c}, key=str):
        print(f"  {cls!s:12s} {c[cls, True]:4d} / {c[cls, False]:4d}")
    gaps = sorted(drafts, key=lambda d: sum(I[d, l]["ok"] - R[d, l]["ok"] for l in labels if (d, l) in R))
    print("largest gaps (draft, real, imagined, real-ok imagined-fail steps):")
    for d in gaps[:4] + gaps[-3:]:
        ls = [l for l in labels if (d, l) in R]
        cc = Counter()
        for l in ls:
            if R[d, l]["ok"] and not I[d, l]["ok"]:
                bad = next((s for s in I[d, l].get("steps", []) if not s.get("ok", True)), {})
                cc[(I[d, l].get("failure_class"), bad.get("step"), str(bad.get("why"))[:40])] += 1
        print(
            f"  {d} {meta[d]['group']} {meta[d]['name']} real {sum(R[d, l]['ok'] for l in ls)} imag {sum(I[d, l]['ok'] for l in ls)} {cc.most_common(3)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
