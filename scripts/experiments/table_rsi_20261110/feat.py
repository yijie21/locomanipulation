"""TODO 47.3: cells keyed by features, not by object names, and a test on objects the search never saw.

Features come from the scene file alone (the object's box, its support's height), so a new object has a cell before
anything runs. The 54 single-hand objects of the apartment outside grasp_atomic's ten are split by prototype into
27 more training objects and 27 held-out ones. One shared sweep (each number alone, a common grid, every training
start) gives per-start outcomes for every setting; from them, for each way of cutting the objects into cells, the
setting each cell would take and its leave-one-object-out score. The cut with the best left-out score is the table.

    feat.py draw             # pool_feat_train (10 x 16 old starts + new objects x 6), pool_feat_test (held-out x 8)
    feat.py sweep            # base + every (number, value) on the training starts; base on the held-out starts
    feat.py fit              # leave-one-object-out over the cuts -> feat_table.json, feat_fit.txt
    feat.py test             # the table on the held-out objects, once
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import search as S  # noqa: E402
import table as T  # noqa: E402

FAMILY = "pick_any"
META = T.REPO / "outputs/scene_discovery/adt_apartment_mars/scene_meta.json"
D = T.OUT / "feat"
SEEN = ("Orange_A", "NIT_CoffeeCan_Anon", "DishSoap_1", "Mug_2", "TvRemote_1", "WoodenSpoon", "CandlePattern02_1",
        "VaseSmall_2", "Plant4_1", "DisplayWolf")  # fmt: skip
KNOBS = {  # a common grid for every object: key -> (lo, hi)
    "pick.STANCE_MIN_D_M": (0.30, 0.54),
    "pick.STANCE_CLEAR_M": (0.12, 0.30),
    "pick.VIA_UP_M": (0.05, 0.20),
    "pick.PRE_BACK_M": (0.06, 0.20),
    "pick.TOP_MIN_D_M": (0.30, 0.60),
}
CUTS = {  # name -> the features a cell is keyed by
    "one cell": (),
    "shape": ("shape",),
    "support": ("support",),
    "shape x support": ("shape", "support"),
    "shape x width": ("shape", "width"),
    "shape x support x width": ("shape", "support", "width"),
}
MARGIN_SHARE = (
    0.06  # a cell takes a setting when its smoothed wins beat the cell's baseline by this share of its starts
)
JAW_M = 0.092


def meta() -> dict:
    return json.loads(META.read_text())


def features(key: str, m: dict) -> dict:
    o = m["objects"][key]
    x, y, z = (2 * v for v in o["box"]["half_extent"])
    top = m["surfaces"][o["surface"]]["top"]
    narrow = min(x, y)
    return dict(
        shape="flat" if z <= 0.04 else "tall" if z >= 0.15 else "mid",
        support="low" if top < 0.6 else "high" if top > 1.0 else "table",
        width="fits" if narrow <= JAW_M - 0.012 else "tight" if narrow <= JAW_M else "wide",
    )


def split(m: dict) -> tuple[list, list]:
    groups = collections.defaultdict(list)
    for k, o in m["objects"].items():
        if o.get("free") and o["semantic"].get("graspable") == "single" and k not in SEEN:
            groups[o["semantic"]["adt"].get("prototype_name") or re.sub(r"_\d+$", "", k)].append(k)
    names = sorted(groups)
    random.Random(47).shuffle(names)
    train, test = [], []
    for g in names:  # whole prototypes go to one side, the smaller side first
        (train if len(train) <= len(test) else test).extend(sorted(groups[g]))
    return sorted(train), sorted(test)


def draw() -> None:
    sys.path[:0] = [str(T.REPO / "scripts/rsi"), str(T.REPO / "src")]
    import family as F

    m = meta()
    train, test = split(m)
    fam = F.load(FAMILY)
    D.mkdir(exist_ok=True)
    (D / "split.json").write_text(json.dumps(dict(train=train, test=test), indent=1))
    for name, objs, seed, n in (
        ("pool_feat_new", train, 860000, 6 * len(train)),
        ("pool_feat_test", test, 870000, 8 * len(test)),
    ):
        if not (D / name / "_pool.json").exists():
            r = F.draw(fam, "transfer", D / name, seed=seed, n=n, tasks=[f"pick_{k}" for k in objs])
            print(name, {k: (v if not isinstance(v, list) else len(v)) for k, v in r.items()})
    tr = D / "pool_feat_train"
    tr.mkdir(exist_ok=True)
    for src in (T.TRAIN, D / "pool_feat_new"):
        for f in src.glob("pick_*.json"):
            shutil.copy(f, tr / f.name)
    print("train objects", len(train), "held-out objects", len(test), "training starts", len(list(tr.glob("pick_*.json"))),
          "held-out starts", len(list((D / "pool_feat_test").glob("pick_*.json"))))  # fmt: skip
    cells = collections.Counter(tuple(features(k, m).values()) for k in [*SEEN, *train])
    print("training cells (shape, support, width):", dict(cells))
    print("held-out cells:", dict(collections.Counter(tuple(features(k, m).values()) for k in test)))


def variants() -> list[tuple[str, dict]]:
    dflt, out = T.defaults(), [("base", {})]
    for k, (lo, hi) in KNOBS.items():
        out += [(f"{k.split('.')[1]}={v:g}", {k: v}) for v in S.grid(k, lo, hi, dflt)]
    return out


def sweep(procs: int, gpus: str) -> None:
    import fast

    tr = D / "pool_feat_train"
    job = D / "sweep.json"
    job.write_text(json.dumps(dict(family=FAMILY, pool=str(tr), memo=True, starts=sorted(p.stem for p in tr.glob("pick_*.json")),
                                   variants=[dict(name=n, skills=str(T.SKILLS), set={**T.BASE, **st}, out=str(D / "train" / n))
                                             for n, st in variants()])))  # fmt: skip
    print("sweep rc", fast.run(job, procs, gpus))
    te = D / "pool_feat_test"
    job = D / "test_base.json"
    job.write_text(json.dumps(dict(family=FAMILY, pool=str(te), memo=False, starts=sorted(p.stem for p in te.glob("pick_*.json")),
                                   variants=[dict(name="base", skills=str(T.SKILLS), set=dict(T.BASE), out=str(D / "test/base"))])))  # fmt: skip
    print("test base rc", fast.run(job, procs, gpus))


def load() -> tuple[dict, dict]:
    """wins[object][variant] = (won, n); only starts every variant has."""
    per = collections.defaultdict(dict)
    for n, _ in variants():
        for lab, r in T.rounds(D / "train" / n).items():
            per[lab][n] = r["ok"]
    names = [n for n, _ in variants()]
    wins = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for lab, d in per.items():
        if all(n in d for n in names):
            obj = T.task_of(lab)[5:]
            for n in names:
                wins[obj][n][0] += d[n]
                wins[obj][n][1] += 1
    return wins, dict(variants())


def choose(objs: list, wins: dict, sets: dict) -> tuple[str, float, float]:
    """The setting a cell made of `objs` takes: (variant name, smoothed wins, baseline wins); 'base' when no number's
    plateau beats the baseline by MARGIN_SHARE of the cell's starts."""
    n = sum(wins[o]["base"][1] for o in objs)
    base = sum(wins[o]["base"][0] for o in objs)
    dflt, best = T.defaults(), ("base", float(base), float(base))
    for k in KNOBS:
        pts = {round(dflt[k], 5): base}
        for name, st in sets.items():
            if list(st) == [k]:
                pts[round(st[k], 5)] = sum(wins[o][name][0] for o in objs)
        v, s = S.plateau(pts)
        if abs(v - dflt[k]) > 1e-9 and s >= base + MARGIN_SHARE * n and s > best[1]:
            best = (next(nm for nm, st in sets.items() if list(st) == [k] and abs(st[k] - v) < 1e-9), s, float(base))
    return best


def fit() -> None:
    m = meta()
    wins, sets = load()
    objs = sorted(wins)
    feats = {o: features(o, m) for o in objs}
    n_all, base_all = sum(wins[o]["base"][1] for o in objs), sum(wins[o]["base"][0] for o in objs)
    lines = [f"training: {len(objs)} objects, {n_all} starts, incumbent {base_all}"]
    best_single = sum(max(w[0] for w in wins[o].values()) for o in objs)
    lines.append(f"best single setting per object (optimistic, in-sample): {best_single}")
    scores = {}
    for cut, keys in CUTS.items():
        cell = lambda o, keys=keys: tuple(feats[o][k] for k in keys)  # noqa: E731
        members = collections.defaultdict(list)
        for o in objs:
            members[cell(o)].append(o)
        fit_wins = sum(sum(wins[o][choose(g, wins, sets)[0]][0] for o in g) for g in members.values())
        loo = 0
        for o in objs:  # the cell's setting chosen without this object, scored on it
            others = [x for x in members[cell(o)] if x != o]
            loo += wins[o][choose(others, wins, sets)[0] if others else "base"][0]
        scores[cut] = loo
        lines.append(f"cut '{cut}': {len(members)} cells, in-sample {fit_wins}, left-one-object-out {loo}")
    cut = max(scores, key=lambda c: (scores[c], -len(CUTS[c])))
    keys = CUTS[cut]
    members = collections.defaultdict(list)
    for o in objs:
        members[tuple(feats[o][k] for k in keys)].append(o)
    table = {}
    lines.append(f"chosen cut: '{cut}' (left-out {scores[cut]} vs incumbent {base_all})")
    for c, g in sorted(members.items()):
        name, s, b = choose(g, wins, sets)
        n = sum(wins[o]["base"][1] for o in g)
        table["|".join(c)] = dict(set=sets[name], objects=g, n=n, base=b, wins=sum(wins[o][name][0] for o in g))
        lines.append(
            f"  cell {c or ('all',)}: {len(g)} objects, {n} starts, incumbent {b:.0f} -> {table['|'.join(c)]['wins']} with {sets[name] or 'nothing'}"
        )
    (D / "feat_table.json").write_text(json.dumps(dict(cut=cut, keys=list(keys), cells=table), indent=1))
    (D / "feat_fit.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def test(procs: int, gpus: str) -> None:
    import fast

    m, tab = meta(), json.loads((D / "feat_table.json").read_text())
    te = D / "pool_feat_test"
    labels = sorted(p.stem for p in te.glob("pick_*.json"))
    by_set = collections.defaultdict(list)
    for lab in labels:
        f = features(T.task_of(lab)[5:], m)
        cell = tab["cells"].get("|".join(f[k] for k in tab["keys"]))
        if cell and cell["set"]:
            by_set[json.dumps(cell["set"], sort_keys=True)].append(lab)
    vs = [dict(name=f"cell{i}", skills=str(T.SKILLS), set={**T.BASE, **json.loads(s)}, starts=ls, out=str(D / "test/table"))
          for i, (s, ls) in enumerate(sorted(by_set.items()))]  # fmt: skip
    job = D / "test_table.json"
    job.write_text(json.dumps(dict(family=FAMILY, pool=str(te), memo=False, starts=labels, variants=vs)))
    if vs:
        fast.run(job, procs, gpus)
    base, new = T.rounds(D / "test/base"), T.rounds(D / "test/table")
    per = collections.defaultdict(lambda: [0, 0, 0, 0])
    for lab, o in base.items():
        r = new.get(lab, o)
        f = features(T.task_of(lab)[5:], m)
        p = per["|".join(f[k] for k in tab["keys"]) or "all"]
        p[0] += 1
        p[1] += o["ok"]
        p[2] += r["ok"]
        p[3] += lab in new
    won = sum(new[x]["ok"] and not base[x]["ok"] for x in new if x in base)
    lost = sum(base[x]["ok"] and not new[x]["ok"] for x in new if x in base)
    text = [f"held-out objects ({len(set(T.task_of(x) for x in base))} objects, {len(base)} starts, none used by the search): "
            f"incumbent {sum(p[1] for p in per.values())} -> feature table {sum(p[2] for p in per.values())} (gained {won}, lost {lost})"]  # fmt: skip
    text += [
        f"  cell {c}: {p[1]} -> {p[2]} of {p[0]} ({p[3]} starts ran a changed setting)" for c, p in sorted(per.items())
    ]
    (D / "feat_test.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("draw", "sweep", "fit", "test"))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    if a.cmd in ("draw", "fit"):
        globals()[a.cmd]()
    else:
        globals()[a.cmd](a.procs, a.gpus)
