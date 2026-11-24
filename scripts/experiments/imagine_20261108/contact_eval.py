"""What the contact replays add to the imagined rounds (TODO 45.3).

    python3 scripts/experiments/imagine_20261108/contact_eval.py <imagine root> v7 contact2

"lookup" answers from the replays of this very grasp at this very start, which no new draft would have: an upper bound.
"other starts" answers from the same object's grasps at other starts only (nearest neighbours in the target's position
relative to the object, top and side grasps apart), which is what a table built from history could do.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MOVES = {"walk_to", "turn", "sidestep"}
K = 10


def load(p):
    return json.loads(Path(p).read_text())


def down(reach: dict) -> bool:
    a = reach["args"]
    return len(a) > 2 and a[2] is not None and float(a[2][2][0]) < -0.7


def query(rec: dict, kin: dict, body: str):
    """(rel, side, down) of the round's last grip that replays, or None."""
    for g in reversed(kin.get("grips", [])):
        calls, n = rec["calls"], g["n_calls"]
        last = max([i for i in range(n) if calls[i]["name"] in MOVES], default=-1)
        reaches = [c for c in calls[last + 1 : n] if c["name"] in ("reach", "posture")]
        if not reaches or reaches[-1]["name"] != "reach" or body not in g["free"]:
            continue
        target = reaches[-1]["args"][1]
        if isinstance(target[0], list):
            continue
        pos = g["free"][body][:3]
        return np.array([float(t - p) for t, p in zip(target, pos)]), g["side"], down(reaches[-1])
    return None


def main(root: str, ver: str, prefix: str = "contact") -> int:
    root = Path(root)
    S = []
    errs = Counter()
    for t in ("gf", "ga"):
        jobs = {j["id"]: j for j in load(root / f"{prefix}_{t}" / "jobs.json")}
        for p in sorted((root / f"{prefix}_{t}" / "rows").glob("*.json")):
            r = load(p)
            if r.get("error"):
                errs[r["error"][:70]] += 1
                continue
            j = jobs[r["id"]]
            last = [c for c in j["reaches"] if c["name"] == "reach"][-1]
            S.append(dict(r, fam=t, rel=np.array(j["rel"], float), down=down(last), body=j["body"]))
    print(f"samples {len(S)}, errors {sum(errs.values())} {errs.most_common(2)}")
    print(
        f"held {np.mean([s['held'] for s in S]):.3f}, knocked {np.mean([s['knocked'] for s in S]):.3f}, fell {np.mean([s['fell'] for s in S]):.3f}, wall median {np.median([s['wall_s'] for s in S]):.1f} s"
    )
    groups = defaultdict(list)
    for s in S:
        groups[(s["fam"], s["label"], tuple(np.round(s["rel"] * 200) / 200), s["side"])].append(s)
    centre, flips, n = 0, 0, 0
    for g in groups.values():
        c = [s for s in g if s["offset"] == [0.0, 0.0]]
        o = [s for s in g if s["offset"] != [0.0, 0.0]]
        if c and o:
            n += len(o)
            flips += sum(s["held"] != c[0]["held"] for s in o)
            centre += c[0]["held"]
    print(
        f"distinct grasps {len(groups)}; held at the centre {centre}; a 1.5 cm shift of the object flips {flips} of {n} outcomes ({flips / max(n, 1):.2f})"
    )
    by = defaultdict(list)
    for s in S:
        by[s["key"]].append(s["held"])
    print(
        "held by object:", sorted(((round(float(np.mean(v)), 2), k, len(v)) for k, v in by.items()), key=lambda x: x[0])
    )

    for t in ("gf", "ga"):  # does a replay at the centre say what the real round said?
        manifest = {j["id"]: j for j in load(root / f"{ver}_kin_{t}" / "manifest.json")}
        c = Counter()
        GRIP = {"ok", "slip", "too_wide", "knocked", "miss_near"}
        for s in S:
            if s["fam"] != t or s["offset"] != [0.0, 0.0] or s["candidate"] not in manifest:
                continue
            r = load(Path(manifest[s["candidate"]]["real"]) / f"{s['label']}.json")
            if r.get("failure_class") in GRIP:
                c[(bool(r["ok"]), bool(s["held"]))] += 1
        n = sum(c.values())
        print(
            f"{t}: replay vs real round, rounds decided at the grip: {n}; real ok & held {c[True, True]}, real ok & not held {c[True, False]}, real fail & held {c[False, True]}, both fail {c[False, False]}; agree {(c[True, True] + c[False, False]) / max(n, 1):.2f}"
        )

    def lookup(fam, label, rel, side):
        g = groups.get((fam, label, tuple(np.round(rel * 200) / 200), side))
        return None if not g else float(np.mean([s["held"] for s in g]))

    idx = defaultdict(list)
    for s in S:
        idx[(s["key"], s["down"])].append(s)

    def other_starts(key, dn, label, rel):
        pool = [s for s in idx[(key, dn)] if s["label"] != label]
        if not pool:
            return None
        X = np.array([s["rel"] - np.r_[s["offset"], 0.0] for s in pool])
        y = np.array([s["held"] for s in pool], float)
        d = np.linalg.norm(X - rel, axis=1)
        o = np.argsort(d)[:K]
        w = 1.0 / (d[o] + 0.01)
        return float((w * y[o]).sum() / w.sum())

    def miss(r):
        return max([float(s["pred_p_miss"]) for s in r.get("steps") or [] if s.get("pred_p_miss") is not None] or [0.0])

    for t, fam_name in (("gf", "grasp_floor"), ("ga", "grasp_atomic")):
        out = root / f"{ver}_kin_{t}"
        jobs = load(out / "manifest.json")
        sys.path[:0] = [str(Path(__file__).resolve().parents[3] / "scripts" / "rsi")]
        scene = None
        rows, cov = [], Counter()
        for j in jobs:
            if j["incumbent"]:
                continue
            m = re.search(r"/([^/]+)/outputs/.*/loop/(iter_\d+)/", j["pkg"])
            for l in j["labels"]:
                r, i, kin = (
                    load(Path(j["real"]) / f"{l}.json"),
                    load(Path(j["out"]) / f"{l}.json"),
                    load(Path(j["out"]) / f"{l}.kin.json"),
                )
                pool = load(Path(j["pool"]) / f"{l}.json")
                if scene is None:
                    fam = load(
                        Path(__file__).resolve().parents[3] / "scripts" / "rsi" / "families" / f"{fam_name}.json"
                    )
                    scene = load(Path(__file__).resolve().parents[3] / fam["scene"] / "scene_meta.json")["objects"]
                    tasks = {
                        x["id"]: x
                        for x in map(
                            json.loads,
                            (Path(__file__).resolve().parents[3] / fam["tasks_file"]).read_text().splitlines(),
                        )
                        if True
                    }
                goal = tasks[pool["task"]]["goal"][0]
                key = goal[goal.index("(") + 1 : goal.index(")")].split(",")[0].strip()
                body = scene[key]["body"]
                q = query(i, kin, body)
                p_l = p_o = None
                if q is not None:
                    rel, side, dn = q
                    p_l, p_o = lookup(t, l, rel, side), other_starts(key, dn, l, rel)
                cov["no grip to query" if q is None else "lookup hit" if p_l is not None else "lookup miss"] += 1
                rows.append(dict(group=(m.group(1), m.group(2)) if m else None, id=j["id"], label=l, y=float(r["ok"]), imag=float(i["ok"]), pm=miss(i),
                                 p_l=p_l, p_o=p_o, cls=r.get("failure_class")))  # fmt: skip
        y = np.array([x["y"] for x in rows])
        imag = np.array([x["imag"] for x in rows])
        pm = np.array([x["pm"] for x in rows])
        base = float(np.mean([s["held"] for s in S if s["fam"] == t]))
        pl = np.array([base if x["p_l"] is None else x["p_l"] for x in rows])
        po = np.array([base if x["p_o"] is None else x["p_o"] for x in rows])

        def auc(score):
            pos, neg = score[y == 1], score[y == 0]
            return float(np.mean([(pos > v).mean() + 0.5 * (pos == v).mean() for v in neg]))

        ids = sorted({x["id"] for x in rows})
        labels = sorted({x["label"] for x in rows})
        A = np.full((len(ids), len(labels)), np.nan)
        for x in rows:
            A[ids.index(x["id"]), labels.index(x["label"])] = x["y"]
        start = np.array([np.nanmean(np.delete(A[:, labels.index(x["label"])], ids.index(x["id"]))) for x in rows])

        print(f"== {fam_name}: {len(rows)} draft rounds, real win rate {y.mean():.3f}; coverage {dict(cov)}")
        s0 = imag * (1 - pm)
        for name, s in (
            ("geometry", imag),
            ("geometry x body", s0),
            ("+ contact, other starts", s0 * po),
            ("+ contact, lookup (upper bound)", s0 * pl),
        ):
            print(
                f"  {name:34s} AUC {auc(s):.3f}; with start prior {auc(start + s):.3f}; agreement at 0.5 {np.mean((s >= 0.5) == y):.3f}"
            )
        print(f"  start prior alone AUC {auc(start):.3f}; always ok {max(y.mean(), 1 - y.mean()):.3f}")
        G = defaultdict(lambda: defaultdict(list))
        for x, a, b, c in zip(rows, s0, s0 * po, s0 * pl):
            if x["group"]:
                G[x["group"]][x["id"]].append((x["y"], a, b, c))
        hit, rnd, n = np.zeros(3), 0.0, 0
        for g, ds in G.items():
            if len(ds) < 2:
                continue
            n += 1
            real = {d: sum(v[0] for v in r) for d, r in ds.items()}
            best = max(real.values())
            rnd += np.mean([real[d] == best for d in ds])
            for q in range(3):
                sc = {d: sum(v[q + 1] for v in r) for d, r in ds.items()}
                top = [d for d in sc if abs(sc[d] - max(sc.values())) < 1e-9]
                hit[q] += np.mean([real[d] == best for d in top])
        print(
            f"  best draft in an iteration ({n} groups): geometry x body {hit[0] / max(n, 1):.2f}, + other starts {hit[1] / max(n, 1):.2f}, + lookup {hit[2] / max(n, 1):.2f}, random {rnd / max(n, 1):.2f}"
        )
        c = defaultdict(lambda: [0, 0, 0])
        for x in rows:
            bad = None if x["p_o"] is None else x["p_o"] < 0.5
            bl = None if x["p_l"] is None else x["p_l"] < 0.5
            c[x["cls"]][0] += 1
            c[x["cls"]][1] += bool(bad)
            c[x["cls"]][2] += bool(bl)
        print(
            "  real class: rounds / contact says fail (other starts) / (lookup):",
            {k: tuple(v) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))},
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
