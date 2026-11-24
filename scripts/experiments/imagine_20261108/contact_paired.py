"""Contact learned from the real rounds themselves (TODO 45.3): the hand-to-object relation an imagined round reached,
labelled by whether the paired real round ended holding the object. Only rounds the real run decided at the grip train
it; a round is scored from other starts of the same object, nearest neighbours, top and side grasps apart.

    python3 scripts/experiments/imagine_20261108/contact_paired.py <imagine root> v7
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from contact_eval import load, query

ROOT = Path(__file__).resolve().parents[3]
GRIP = {"ok", "slip", "too_wide", "knocked", "miss_near"}
K = 10


def miss(r):
    return max([float(s["pred_p_miss"]) for s in r.get("steps") or [] if s.get("pred_p_miss") is not None] or [0.0])


def main(root: str, ver: str) -> int:
    root = Path(root)
    for t, fam_name in (("gf", "grasp_floor"), ("ga", "grasp_atomic")):
        fam = load(ROOT / "scripts" / "rsi" / "families" / f"{fam_name}.json")
        scene = load(ROOT / fam["scene"] / "scene_meta.json")["objects"]
        tasks = {x["id"]: x for x in map(json.loads, (ROOT / fam["tasks_file"]).read_text().splitlines())}
        rows = []
        for j in load(root / f"{ver}_kin_{t}" / "manifest.json"):
            if j["incumbent"]:
                continue
            m = re.search(r"/([^/]+)/outputs/.*/loop/(iter_\d+)/", j["pkg"])
            for l in j["labels"]:
                r, i, kin = (
                    load(Path(j["real"]) / f"{l}.json"),
                    load(Path(j["out"]) / f"{l}.json"),
                    load(Path(j["out"]) / f"{l}.kin.json"),
                )
                goal = tasks[load(Path(j["pool"]) / f"{l}.json")["task"]]["goal"][0]
                key = goal[goal.index("(") + 1 : goal.index(")")].split(",")[0].strip()
                q = query(i, kin, scene[key]["body"])
                rows.append(dict(id=j["id"], group=(m.group(1), m.group(2)) if m else None, label=l, key=key, y=float(r["ok"]), imag=float(i["ok"]), pm=miss(i),
                                 rel=None if q is None else q[0], down=None if q is None else q[2], cls=r.get("failure_class")))  # fmt: skip
        train = defaultdict(list)
        for x in rows:
            if x["rel"] is not None and x["cls"] in GRIP:
                train[(x["key"], x["down"])].append(x)
        base = float(np.mean([x["y"] for v in train.values() for x in v]))
        for x in rows:
            pool = [] if x["rel"] is None else [s for s in train[(x["key"], x["down"])] if s["label"] != x["label"]]
            if not pool:
                x["p"] = base
                continue
            d = np.linalg.norm(np.array([s["rel"] for s in pool]) - x["rel"], axis=1)
            o = np.argsort(d)[:K]
            w = 1.0 / (d[o] + 0.01)
            x["p"] = float((w * np.array([pool[i]["y"] for i in o])).sum() / w.sum())
        y = np.array([x["y"] for x in rows])
        s0 = np.array([x["imag"] * (1 - x["pm"]) for x in rows])
        p = np.array([x["p"] for x in rows])

        def auc(score, mask=None):
            yy, ss = (y, score) if mask is None else (y[mask], score[mask])
            pos, neg = ss[yy == 1], ss[yy == 0]
            return float(np.mean([(pos > v).mean() + 0.5 * (pos == v).mean() for v in neg]))

        grip = np.array([x["cls"] in GRIP for x in rows])
        print(
            f"== {fam_name}: {len(rows)} rounds, {int(grip.sum())} decided at the grip, {sum(len(v) for v in train.values())} usable"
        )
        print(f"  decided at the grip: contact alone AUC {auc(p, grip):.3f} (win rate {y[grip].mean():.3f})")
        print(f"  all rounds: geometry x body {auc(s0):.3f}; x contact {auc(s0 * p):.3f}; contact alone {auc(p):.3f}")
        G = defaultdict(lambda: defaultdict(list))
        for x, a, b in zip(rows, s0, s0 * p):
            if x["group"]:
                G[x["group"]][x["id"]].append((x["y"], a, b))
        hit, rnd, n = np.zeros(2), 0.0, 0
        for ds in G.values():
            if len(ds) < 2:
                continue
            n += 1
            real = {d: sum(v[0] for v in r) for d, r in ds.items()}
            best = max(real.values())
            rnd += np.mean([real[d] == best for d in ds])
            for q in range(2):
                sc = {d: sum(v[q + 1] for v in r) for d, r in ds.items()}
                top = [d for d in sc if abs(sc[d] - max(sc.values())) < 1e-9]
                hit[q] += np.mean([real[d] == best for d in top])
        print(
            f"  best draft in an iteration ({n} groups): geometry x body {hit[0] / max(n, 1):.2f}; x contact {hit[1] / max(n, 1):.2f}; random {rnd / max(n, 1):.2f}"
        )
        for cls in ("slip", "too_wide", "knocked", "ok"):
            m_ = np.array([x["cls"] == cls for x in rows])
            if m_.any():
                print(f"  {cls}: {int(m_.sum())} rounds, mean contact odds {p[m_].mean():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
