"""Time-ordered check of the contact table learned from real rounds (TODO 45.3).

contact_paired.py trained on every other start, later iterations included. Here a group of drafts (one sandbox, one
iteration) may only use groups that had completely finished before its first real round was written. Everything else is
frozen from contact_paired.py and was fixed before looking at the result: K = 10, 1 / (d + 0.01) weights, other starts
only, top and side grasps apart, rounds decided at the grip train it, score = geometry x (1 - body odds) x contact.

Variants:
    everything        contact_paired.py's training set; must reproduce its numbers, or this script is wrong
    earlier + start   earlier groups plus the incumbent's rounds, which a loop has before its first draft (the primary)
    earlier           earlier groups only

    PYTHONPATH=scripts/experiments/imagine_20261108 python3 scripts/experiments/imagine_20261108/contact_time.py <imagine root> v7
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from contact_eval import load, query
from contact_paired import GRIP, ROOT, K, miss

BOOT = 4000


def rounds(root: Path, ver: str, t: str, fam_name: str) -> list[dict]:
    fam = load(ROOT / "scripts" / "rsi" / "families" / f"{fam_name}.json")
    scene = load(ROOT / fam["scene"] / "scene_meta.json")["objects"]
    tasks = {x["id"]: x for x in map(json.loads, (ROOT / fam["tasks_file"]).read_text().splitlines())}
    rows = []
    for j in load(root / f"{ver}_kin_{t}" / "manifest.json"):
        m = re.search(r"/([^/]+)/outputs/.*/loop/(iter_\d+)/", j["pkg"])
        group = ("incumbent", j["id"]) if j["incumbent"] else (m.group(1), m.group(2))
        for l in j["labels"]:
            real = Path(j["real"]) / f"{l}.json"
            r, i, kin = load(real), load(Path(j["out"]) / f"{l}.json"), load(Path(j["out"]) / f"{l}.kin.json")
            goal = tasks[load(Path(j["pool"]) / f"{l}.json")["task"]]["goal"][0]
            key = goal[goal.index("(") + 1 : goal.index(")")].split(",")[0].strip()
            q = query(i, kin, scene[key]["body"])
            rows.append(dict(id=j["id"], group=group, inc=bool(j["incumbent"]), label=l, key=key, t=os.path.getmtime(real), y=float(r["ok"]), imag=float(i["ok"]), pm=miss(i),
                             rel=None if q is None else q[0], down=None if q is None else q[2], cls=r.get("failure_class")))  # fmt: skip
    return rows


def contact(x: dict, train: dict, base: float) -> float:
    pool = [] if x["rel"] is None else [s for s in train.get((x["key"], x["down"]), []) if s["label"] != x["label"]]
    if not pool:
        return base
    d = np.linalg.norm(np.array([s["rel"] for s in pool]) - x["rel"], axis=1)
    o = np.argsort(d)[:K]
    w = 1.0 / (d[o] + 0.01)
    return float((w * np.array([pool[i]["y"] for i in o])).sum() / w.sum())


def table(rows: list[dict]) -> tuple[dict, float, int]:
    train = defaultdict(list)
    for x in rows:
        if x["rel"] is not None and x["cls"] in GRIP:
            train[(x["key"], x["down"])].append(x)
    n = sum(len(v) for v in train.values())
    return train, (float(np.mean([x["y"] for v in train.values() for x in v])) if n else 1.0), n


def score(rows: list[dict], variant: str) -> tuple[list[dict], dict]:
    """Copies of the draft rounds with their contact odds p; sizes of each group's training set."""
    span = defaultdict(list)
    for x in rows:
        span[x["group"]].append(x["t"])
    out, sizes = [], {}
    test = [x for x in rows if not x["inc"]]
    if variant == "everything":
        train, base, n = table(test)
        return [dict(x, p=contact(x, train, base)) for x in test], {g: n for g in {x["group"] for x in test}}
    for g in sorted({x["group"] for x in test}, key=lambda g: min(span[g])):
        start = min(span[g])
        seen = [
            x
            for x in rows
            if x["group"] != g and max(span[x["group"]]) < start and (variant == "earlier + start" or not x["inc"])
        ]
        train, base, sizes[g] = table(seen)
        out += [dict(x, p=contact(x, train, base)) for x in test if x["group"] == g]
    return out, sizes


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(pos > v).mean() + 0.5 * (pos == v).mean() for v in neg]))


def picks(rows: list[dict]) -> dict:
    """For each iteration with two drafts or more: is the top draft a real best, and how many of its rounds really won."""
    G = defaultdict(lambda: defaultdict(list))
    for x in rows:
        G[x["group"]][x["id"]].append(x)
    res = {}
    for g, ds in G.items():
        if len(ds) < 2:
            continue
        real = {d: float(np.mean([x["y"] for x in r])) for d, r in ds.items()}
        best = max(real.values())
        row = dict(
            random=float(np.mean([real[d] == best for d in ds])),
            random_win=float(np.mean(list(real.values()))),
            best_win=best,
        )
        for name, f in (
            ("base", lambda x: x["imag"] * (1 - x["pm"])),
            ("contact", lambda x: x["imag"] * (1 - x["pm"]) * x["p"]),
        ):
            sc = {d: sum(f(x) for x in r) for d, r in ds.items()}
            top = [d for d in sc if abs(sc[d] - max(sc.values())) < 1e-9]
            row[name], row[name + "_win"] = (
                float(np.mean([real[d] == best for d in top])),
                float(np.mean([real[d] for d in top])),
            )
        res[g] = row
    return res


def report(rows: list[dict], name: str, sizes: dict, rng: np.random.Generator, detail: bool) -> None:
    y = np.array([x["y"] for x in rows])
    s0 = np.array([x["imag"] * (1 - x["pm"]) for x in rows])
    p = np.array([x["p"] for x in rows])
    grip = np.array([x["cls"] in GRIP for x in rows])
    names = sorted({x["group"] for x in rows})
    groups = np.array([names.index(x["group"]) for x in rows])
    idx = {g: np.flatnonzero(groups == g) for g in set(groups.tolist())}
    keys = list(idx)
    deltas = []
    for _ in range(BOOT):
        b = np.concatenate([idx[keys[i]] for i in rng.integers(len(keys), size=len(keys))])
        if 0 < y[b].sum() < len(b):
            deltas.append(auc(y[b], s0[b] * p[b]) - auc(y[b], s0[b]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"-- {name}")
    print(
        f"  AUC geometry x body {auc(y, s0):.3f}; x contact {auc(y, s0 * p):.3f} (change {auc(y, s0 * p) - auc(y, s0):+.3f}, 95% over groups {lo:+.3f} .. {hi:+.3f}); contact alone {auc(y, p):.3f}; decided at the grip, contact alone {auc(y[grip], p[grip]):.3f}"
    )
    P = picks(rows)
    n = len(P)
    m = lambda k: float(np.mean([v[k] for v in P.values()]))  # noqa: E731
    d = np.array([v["contact"] - v["base"] for v in P.values()])
    w = np.array([v["contact_win"] - v["base_win"] for v in P.values()])
    Bd = [d[rng.integers(n, size=n)].mean() for _ in range(BOOT)]
    Bw = [w[rng.integers(n, size=n)].mean() for _ in range(BOOT)]
    print(
        f"  best draft ({n} groups): geometry x body {m('base'):.2f}; x contact {m('contact'):.2f}; random {m('random'):.2f}. contact better in {(d > 1e-9).sum()}, worse in {(d < -1e-9).sum()}, same in {(abs(d) <= 1e-9).sum()}; change {d.mean():+.2f}, 95% over groups {np.percentile(Bd, 2.5):+.2f} .. {np.percentile(Bd, 97.5):+.2f}"
    )
    print(
        f"  real win rate of the chosen draft: geometry x body {m('base_win'):.3f}; x contact {m('contact_win'):.3f}; random {m('random_win'):.3f}; true best {m('best_win'):.3f}; contact change {w.mean():+.3f}, 95% over groups {np.percentile(Bw, 2.5):+.3f} .. {np.percentile(Bw, 97.5):+.3f}"
    )
    if detail:
        print(
            "  group, rounds it could train on, hit base -> contact, win rate of the pick base -> contact, random, best (in time order):"
        )
        t0 = defaultdict(lambda: float("inf"))
        for x in rows:
            t0[x["group"]] = min(t0[x["group"]], x["t"])
        ordered = [g for g in sorted(sizes, key=lambda g: t0[g]) if g in P]
        for g in ordered:
            v = P[g]
            print(
                f"    {g[0]:26s} {g[1]:8s} train {sizes[g]:4d}  hit {v['base']:.2f} -> {v['contact']:.2f}  win {v['base_win']:.3f} -> {v['contact_win']:.3f}  random {v['random_win']:.3f}  best {v['best_win']:.3f}"
            )
        half = len(ordered) // 2
        for part, gs in (("first half", ordered[:half]), ("second half", ordered[half:])):
            if gs:
                print(
                    f"  {part}: hit base {np.mean([P[g]['base'] for g in gs]):.2f}, contact {np.mean([P[g]['contact'] for g in gs]):.2f}, random {np.mean([P[g]['random'] for g in gs]):.2f}; win base {np.mean([P[g]['base_win'] for g in gs]):.3f}, contact {np.mean([P[g]['contact_win'] for g in gs]):.3f}, random {np.mean([P[g]['random_win'] for g in gs]):.3f}"
                )


def main(root: str, ver: str) -> int:
    rng = np.random.default_rng(0)
    for t, fam_name in (("gf", "grasp_floor"), ("ga", "grasp_atomic")):
        rows = rounds(Path(root), ver, t, fam_name)
        print(
            f"== {fam_name}: {sum(not x['inc'] for x in rows)} draft rounds, {sum(x['inc'] for x in rows)} incumbent rounds"
        )
        for variant in ("everything", "earlier + start", "earlier"):
            scored, sizes = score(rows, variant)
            report(scored, variant, sizes, rng, variant == "earlier + start")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
