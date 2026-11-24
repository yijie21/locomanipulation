"""Read arrival_run.py's rounds beside the real ones and beside the score without arrival errors (TODO 45.8).

    python3 arrival_report.py <out of arrival_run.py> [--pen]

A start's new score is its share of imagined successes over the K draws, or that share weighted by (1 - worst body-table
odds of a miss) as the old score was. Metrics, the same as shadow_report.py: the real win rate of the top-scored draft
against the mean draft of the iteration, whether the top draft is a real best, whether the lowest is a real worst. Beside
them, per round: AUC of the score against the real result, and what a veto at share < 0.5 catches by real failure class.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BOOT = 4000
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "outputs" / "scene_rsi_20261016" / "imagine_20261108"


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def miss(r: dict) -> float:
    return max([float(s["pred_p_miss"]) for s in r.get("steps") or [] if s.get("pred_p_miss") is not None] or [0.0])


def old_score(job: dict, label: str) -> float | None:
    """The score without arrival errors, on the same start: v7 for the recorded drafts, shadow.json for the shadow ones."""
    f = BASE / "v7_kin_gf" / job["id"] / f"{label}.json"
    if f.exists():
        r = load(f)
        return float(r["ok"]) * (1 - miss(r))
    if "-iter_" not in job["id"]:
        return None
    sb, it = job["id"].split("-iter_")
    it, num = it.split("-", 1)
    s = load(
        Path("/workspace/code/rsi-scene")
        / sb
        / "outputs/scene_rsi_20261016_hidden/grasp_floor/shadow"
        / f"iter_{it}"
        / "shadow.json"
    )
    c = next(c for c in s["candidates"] if str(c["number"]) == num)
    return float(c["rounds"][label]["score"])


def rows_of(out: Path) -> tuple[list[dict], dict]:
    m = load(out / "manifest.json")
    rows = []
    for j in m["jobs"]:
        for label in j["labels"]:
            real = load(Path(j["real"]) / f"{label}.json")
            sims = [
                load(k / f"{label}.json") for k in sorted((out / j["id"]).glob("k*")) if (k / f"{label}.json").exists()
            ]
            if not sims:
                continue
            rows.append(dict(group=j["group"], id=j["id"], inc=j["incumbent"], label=label, y=bool(real["ok"]), cls=real.get("failure_class"),
                             share=float(np.mean([bool(s["ok"]) for s in sims])),
                             share_body=float(np.mean([float(s["ok"]) * (1 - miss(s)) for s in sims])), old=old_score(j, label), n=len(sims)))  # fmt: skip
    return rows, m


def auc(a: np.ndarray, y: np.ndarray) -> float:
    p, n = a[y], a[~y]
    if not len(p) or not len(n):
        return float("nan")
    return float(np.mean([(p > v).mean() + 0.5 * (p == v).mean() for v in n]))


def picks(rows: list[dict], key: str) -> dict:
    G = defaultdict(lambda: defaultdict(list))
    for x in rows:
        if x["group"] and x[key] is not None:
            G[x["group"]][x["id"]].append(x)
    res = []
    for g, ds in sorted(G.items()):
        if len(ds) < 2:
            continue
        real = {d: float(np.mean([x["y"] for x in r])) for d, r in ds.items()}
        sc = {d: sum(x[key] for x in r) for d, r in ds.items()}
        top = [d for d in sc if abs(sc[d] - max(sc.values())) < 1e-9]
        low = [d for d in sc if abs(sc[d] - min(sc.values())) < 1e-9]
        res.append(dict(group=g, pick=float(np.mean([real[d] for d in top])), mean=float(np.mean(list(real.values()))), best=max(real.values()),
                        hit=float(np.mean([real[d] == max(real.values()) for d in top])), low=float(np.mean([real[d] == min(real.values()) for d in low])),
                        chance_hit=float(np.mean([v == max(real.values()) for v in real.values()])), chance_low=float(np.mean([v == min(real.values()) for v in real.values()])),
                        real=[round(real[d], 2) for d in sorted(real)], score=[round(sc[d], 1) for d in sorted(real)]))  # fmt: skip
    return dict(groups=res)


def line(rows: list[dict], key: str, name: str) -> str:
    G = picks(rows, key)["groups"]
    if not G:
        return f"{name}: no groups"
    d = np.array([g["pick"] - g["mean"] for g in G])
    rng = np.random.default_rng(0)
    b = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(BOOT)]
    cand = [x for x in rows if x["group"] and x[key] is not None]
    return (f"{name:24s} chosen {np.mean([g['pick'] for g in G]):.3f}, mean draft {np.mean([g['mean'] for g in G]):.3f}, best {np.mean([g['best'] for g in G]):.3f}; "
            f"diff {d.mean():+.3f} [{np.percentile(b, 2.5):+.3f}..{np.percentile(b, 97.5):+.3f}]; "
            f"top is best {sum(g['hit'] for g in G):.1f}/{len(G)} (chance {sum(g['chance_hit'] for g in G):.1f}); "
            f"lowest is worst {sum(g['low'] for g in G):.1f}/{len(G)} (chance {sum(g['chance_low'] for g in G):.1f}); "
            f"round AUC {auc(np.array([x[key] for x in cand], float), np.array([x['y'] for x in cand])):.3f}")  # fmt: skip


def main(out: str, *flags: str) -> int:
    root = Path(out)
    rows, m = rows_of(root)
    print(
        f"{root.name}: noise {m['noise']}, fit {m['fit']}; rounds {len(rows)}; draws per round {sorted(Counter(x['n'] for x in rows))}"
    )
    for key, name in (
        ("old", "no arrival errors"),
        ("share", "share of successes"),
        ("share_body", "share x (1 - body miss)"),
    ):
        print(line(rows, key, name))
    cand = [x for x in rows if x["group"]]
    fail = [x for x in cand if x["share"] < 0.5]
    keep = [x for x in cand if x["share"] >= 0.5]
    print(
        f"veto at share < 0.5: {len(fail)} rounds, real win {np.mean([x['y'] for x in fail]) if fail else float('nan'):.2f}; kept {len(keep)} rounds, real win {np.mean([x['y'] for x in keep]) if keep else float('nan'):.2f}"
    )
    for c, n in Counter(x["cls"] for x in cand).most_common():
        v = [x for x in cand if x["cls"] == c]
        print(
            f"  real {str(c):12s} n {n:4d}: vetoed {sum(x['share'] < 0.5 for x in v):4d}; mean share {np.mean([x['share'] for x in v]):.2f}"
        )
    if "--groups" in flags:
        for key in ("share", "share_body"):
            for g in picks(rows, key)["groups"]:
                print(
                    f"  {key:10s} {g['group']:30s} score {g['score']} real {g['real']} pick {g['pick']:.2f} best {g['best']:.2f}"
                )
    if "--pen" in flags:
        by = defaultdict(list)
        for j in m["jobs"]:
            for label in j["labels"]:
                real = load(Path(j["real"]) / f"{label}.json")
                for k in sorted((root / j["id"]).glob("k*")):
                    f = k / f"{label}.kin.json"
                    if f.exists():
                        pen = [g["pen_cm"] for g in load(f)["grip_pen"]]
                        if pen:
                            by[real.get("failure_class")].append(max(pen))
        for c, v in sorted(by.items(), key=lambda t: -len(t[1])):
            a = np.array(v)
            print(
                f"  pen cm, real {str(c):12s} n {len(a):5d}: p50 {np.median(a):.2f} p90 {np.percentile(a, 90):.2f}; >0.5 {np.mean(a > 0.5):.2f}, >1 {np.mean(a > 1):.2f}, >2 {np.mean(a > 2):.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
