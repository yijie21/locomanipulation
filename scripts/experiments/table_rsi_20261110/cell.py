"""TODO 47.6: a structural step for a whole cell. Where no number moves a cell (feat.py), a model writes a new way of
grasping for it, and the same search that sets numbers also judges the code.

The writer sees only some of the cell's training objects (`dev`) and may run its package on their starts a few times.
Its package is then handed to the search: the numbers the writer named (knobs.json) and the stance numbers are swept
on every training start of the cell, values come from plateaus, and the package is taken only when its
leave-one-prototype-out wins beat what the cell has now by feat.MARGIN_SHARE of the cell's starts. A cell of the table is
then (skills package, set); other cells keep running the incumbent package, so nothing else can change.

    cell.py prep <cell>      # sandbox: skills copy, cards of the dev objects, brief.md, eval.py
    cell.py search <cell>    # judge the sandbox's package -> search.txt, final.json
    cell.py test <cell>      # the judged package on the held-out objects of the cell, once -> test.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import feat as F  # noqa: E402
import search as S  # noqa: E402
import table as T  # noqa: E402

CELLS = {
    "tall_low": dict(
        match=lambda f: f["shape"] == "tall" and f["support"] == "low",
        dev=("BookFiction10_00", "BookLearning1_00", "VaseBig_2"),
        what="tall objects (15 cm or taller) standing on low shelves (top below 0.6 m): upright books standing in a "
        "row on a shelf board, and tall vases",
        facts="A book is 2-5 cm thick, 14-22 cm deep and 21-29 cm tall, so the jaw (9.2 cm) fits only across its "
        "thickness. The shelves are boards of a display unit or bookcase and carry several objects each, so expect "
        "neighbours beside the object and possibly a board above it (not checked for you: the round records and the "
        "skill's own world queries tell).",
    ),
    "flat": dict(
        match=lambda f: f["shape"] == "flat" and f["support"] != "low",
        dev=("WoodenSpoon", "KitchenKnife", "Donut_B", "CarKey_2"),
        what="flat objects (4 cm or lower) lying on tables, counters and high shelves: cutlery, keys, a wallet, a "
        "carrot, a donut",
        facts="An earlier edit for the wooden spoon alone (wrist tilted 45 degrees, jaw across the neck) raised it "
        "from 3 to 7 of 16; half of what was left were objects held and then lost in the lift.",
    ),
}
STANCE = ("pick.STANCE_MIN_D_M", "pick.STANCE_CLEAR_M", "pick.TOP_MIN_D_M")
MAX_KNOBS = 6
STAGES = 3
EVALS = 6
TRAIN = F.D / "pool_feat_train"
HELD = F.D / "pool_feat_test"
BOOKS = dict(  # TODO 47.7: the packed row of upright books; tuned on one shelf, tested on another piece of furniture
    dev=(
        "BookFiction10_00",
        "BookFiction10_01",
        "BookFiction1_00",
        "BookFiction1_01",
        "BookFiction8_00",
        "BookFiction8_01",
    ),
    train=(T.OUT / "books/pool_dev", None),  # every start of that pool (the bedroom display shelves' row)
    held=(
        HELD,
        ("BookComic1_00", "BookComic1_01", "BookCooking1_00", "BookCooking1_01", "BookFiction9_00", "BookFiction9_01"),
    ),
    ref=T.OUT / "books/ref",  # the tall_low package with its searched stance on the dev starts
    base={"pick.ALT_STANCE_D_M": 0.55},
)
CELLS.update(books_sonnet=BOOKS, books_opus=BOOKS)


def pool_of(cell: str, which: str) -> Path:
    return CELLS[cell][which][0] if which in CELLS[cell] else (TRAIN if which == "train" else HELD)


def base_of(cell: str) -> dict:
    return {**T.BASE, **CELLS[cell].get("base", {})}


def objects(cell: str, pool: Path) -> dict:
    """{object: [labels]} of the pool's starts whose object falls in the cell."""
    m, out = F.meta(), collections.defaultdict(list)
    which = "train" if pool == TRAIN else "held"
    for p in sorted(pool_of(cell, which).glob("pick_*.json")):
        o = T.task_of(p.stem)[5:]
        names = CELLS[cell].get(which, (None, None))[1]
        if (names is None or o in names) if which in CELLS[cell] else CELLS[cell]["match"](F.features(o, m)):
            out[o].append(p.stem)
    return dict(out)


def now(cell: str) -> dict:
    """{label: won} under what the number table gives the cell's objects today."""
    if "ref" in CELLS[cell]:
        return {x: bool(r["ok"]) for x, r in T.rounds(CELLS[cell]["ref"]).items()}
    m, tab = F.meta(), json.loads((F.D / "feat_table.json").read_text())
    names = {json.dumps(st, sort_keys=True): n for n, st in F.variants()}
    out = {}
    for o, labels in objects(cell, TRAIN).items():
        c = tab["cells"].get("|".join(F.features(o, m)[k] for k in tab["keys"]), {})
        rs = T.rounds(F.D / "train" / names[json.dumps(c.get("set") or {}, sort_keys=True)])
        out.update({x: bool(rs[x]["ok"]) for x in labels})
    return out


def prep(cell: str) -> None:
    c, d = CELLS[cell], T.OUT / f"cell_{cell}"
    d.mkdir(exist_ok=True)
    if not (d / "skills").exists():
        shutil.copytree(T.SKILLS, d / "skills", ignore=shutil.ignore_patterns("__pycache__"))
    m, objs, base, have = F.meta(), objects(cell, TRAIN), T.rounds(F.D / "train/base"), now(cell)
    (d / "cards").mkdir(exist_ok=True)
    rows, dev_labels = [], []
    for o in c["dev"]:
        rs = [base[x] for x in objs[o]]
        dev_labels += objs[o]
        s = T.summary(rs)
        ob = m["objects"][o]
        size = " x ".join(f"{200 * v:.1f}" for v in ob["box"]["half_extent"])
        rows.append(f"- `{o}`: box {size} cm, on `{ob['surface']}` (top {m['surfaces'][ob['surface']]['top']:.2f} m); "
                    f"the skill wins {s['wins']} of {s['n']}, losses {s['classes']}")  # fmt: skip
        (d / "cards" / f"pick_{o}.md").write_text("\n".join(
            [f"# pick_{o}", f"The skill as it is on {s['n']} training starts: {s['wins']} won. Losses: {s['classes']}.",
             "", "Rounds (what each step measured):", *[T.one_line(r) for r in sorted(rs, key=lambda r: r["ok"])]]) + "\n")  # fmt: skip
    n_all = sum(len(v) for v in objs.values())
    (
        d / "eval.py"
    ).write_text(f'''"""Run the package in skills/ on the {len(dev_labels)} training starts of this sandbox. Usage: eval.py <tag>
Rounds go to rounds/<tag>/ (a tag is never reused, at most {EVALS} tags); prints wins per object and each round's verdict."""
import collections, json, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent
tag = sys.argv[1]
out = H / "rounds" / tag
assert not out.exists(), "use a new tag"
assert len(list((H / "rounds").glob("*"))) < {EVALS} if (H / "rounds").exists() else True, "all {EVALS} runs are used"
job = H / f"job_{{tag}}.json"
job.write_text(json.dumps(dict(family="{F.FAMILY}", pool="{TRAIN}", memo=False, starts={json.dumps(dev_labels)},
    variants=[dict(name=tag, skills=str(H / "skills"), set={json.dumps(T.BASE)}, out=str(out))])))
py = "{T.REPO}/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"
subprocess.run([py, "{HERE}/fast.py", str(job), "--procs", "4", "--gpus", "0,1"], check=False)
rs = [json.loads(f.read_text()) for f in sorted(out.glob("*.json"))]
per = collections.Counter(), collections.Counter()
for r in rs:
    per[0][r["task"]] += 1
    per[1][r["task"]] += r["ok"]
print("rounds", len(rs), "of {len(dev_labels)}; won", sum(r["ok"] for r in rs), {{t: f"{{per[1][t]}}/{{per[0][t]}}" for t in per[0]}})
for r in rs:
    print(r["label"], "WON" if r["ok"] else "LOST " + str(r.get("failure_class")), "|", r["why"][:90], "|", r.get("error", "")[:160])
if len(rs) < {len(dev_labels)}:
    print("some rounds did not run: see", job.with_suffix(".log"))
''')
    (d / "brief.md").write_text(f"""# A new way to grasp one kind of object

A humanoid robot (Unitree G1, a parallel jaw on each hand that opens 9.2 cm, whole-body controller ScaleBFM) picks objects off furniture in a simulated apartment. One grasp skill written in Python does every object: it chooses a stance beside the furniture, walks there, moves the hand to a pre-grasp point, enters level from the side or comes down from above, closes the jaw, lifts and holds. A round is won when the object is held between both fingers at the end.

A search program has tuned the skill's numbers for each kind of object. For most kinds that was enough. For one kind it was not: **{c["what"]}**. Over the {len(objs)} training objects of this kind ({n_all} starts) the skill wins {sum(bool(base[x]["ok"]) for v in objs.values() for x in v)}, and {sum(have.values())} with the best numbers the search found. So the way the skill grasps this kind has to change, not its numbers.

{c["facts"]}

**The objects you can see** ({len(c["dev"])} of the {len(objs)}; the rest are kept back to check that your way works on the kind, not on these objects):
{chr(10).join(rows)}

`cards/pick_<object>.md` has, for every round of these objects, what each step measured (how far the walk stopped from the stance, how far the jaw centre ended from its target and how far the wrist's attitude was off, what the jaw closed on, how much the object moved before the grip and what moved it). Loss classes: `side_reach` / `pinch_reach` the hand ended far from its target on a side / top grasp; `too_wide` the jaw came down on the object instead of around it; `slip` held, then lost; `knocked` the object was pushed or tipped before the jaw closed; `miss_near` the jaw closed beside it; `walk_off` the walk did not reach the stance.

Known facts about this robot: the hand lands where it is sent only when the target is roughly 0.35-0.55 m ahead of the shoulder and between about 0.55 and 1.2 m above the floor (lower targets need a crouch or a bend, which `m.posture` gives); a wrist pointing straight down at table height is hardly ever delivered by the controller, a wrist tilted about 45 degrees is; a hand that starts its next move from where the last command ended (not from where it was measured) does not recoil.

**Your job.** Change the skills package in `skills/` (a private copy; edit only files under `{d}/`) so that this kind of object is picked up more often. Any approach is allowed: another grasp axis or wrist attitude, another stance or body posture, a different approach path, pulling or tilting the object out before grasping it, a re-grasp. Your package will run only on this kind of object, so you need not keep other kinds working. Read `skills/pick.py` first; `{T.OUT}/params.md` documents its numbers. The motor interface is the `m.reach(side, pos, rot, via=..., speed=...)`, `m.grip(side)`, `m.walk_to(...)`, `m.posture(...)` calls the code already uses.

**Name your numbers.** Every number your new way depends on (distances, angles, speeds, a switch between two variants) must be a module-level constant in `skills/pick.py` whose name starts with `ALT_`, with a comment saying what it does. Write `knobs.json` in this directory: `{{"pick.ALT_NAME": [lowest sensible value, highest sensible value], ...}}`, at most {MAX_KNOBS} entries, the ones you believe matter most (a switch is `[0, 1]`). After you finish, a search program will sweep these numbers on all {n_all} starts of the kind and set them; it, not you, decides the final values, and it accepts your package only if it wins clearly more on objects left out of the tuning. So spend your runs on making the way of grasping right, not on fine-tuning values.

**How to test.** `python3 eval.py <new tag>` in this directory runs your package on the {len(dev_labels)} starts of the objects above (about 10 minutes; another job may share the machine) and prints wins and each round's verdict; round records with every step's measurements are in `rounds/<tag>/`. You may run it at most {EVALS} times. Do not run anything else that simulates, and do not read or use any pool or result outside `{d}/` and `{T.OUT}/params.md`.

**Report** (final message, short): what you changed and why, the wins of each eval run by tag, and the tag of your best version. Leave `skills/` in its best state and `knobs.json` written.
""")
    print(
        cell,
        "objects",
        {o: len(v) for o, v in objs.items()},
        "dev starts",
        len(dev_labels),
        "now",
        sum(have.values()),
        "of",
        n_all,
    )


def knobs(cell: str) -> dict:
    """{key: (default, lo, hi)}: the writer's ALT_ numbers that exist in its package, then the stance numbers."""
    d = T.OUT / f"cell_{cell}"
    src = (d / "skills/pick.py").read_text()
    dflt, out = T.defaults(), {}
    raw = json.loads((d / "knobs.json").read_text()) if (d / "knobs.json").exists() else {}
    for k, r in list(raw.items())[:MAX_KNOBS]:
        name = k.split(".")[-1]
        mm = re.search(rf"^{re.escape(name)}\s*(?::[^=]+)?=\s*(-?[\d.]+(?:e-?\d+)?)", src, re.M)
        if k.startswith("pick.ALT_") and mm and isinstance(r, list) and len(r) == 2 and float(r[0]) < float(r[1]):
            out[k] = (float(mm.group(1)), float(r[0]), float(r[1]))
        else:
            print("knob dropped:", k, r)
    for k in STANCE:
        out[k] = (dflt[k], *F.KNOBS[k])
    return out


def proto(o: str) -> str:
    return re.sub(r"_\d+$", "", o)


def values(k: str, spec: tuple) -> list[float]:
    return S.grid(k, spec[1], spec[2], {k: spec[0]})


def pick(objs: list, ks: dict, wins: dict, stage: int, cur_name: str, margin_n: int):
    """The (key, value, variant name) the starts of `objs` would take at this stage, or None."""
    have = sum(wins[cur_name][o] for o in objs)
    best = None
    for k, spec in ks.items():
        pts = {spec[0]: have}
        for v in values(k, spec):
            n = f"s{stage}/{k.split('.')[1]}/{v:g}"
            if n in wins:
                pts[v] = sum(wins[n][o] for o in objs)
        if len(pts) < 2:
            continue
        x, s = S.plateau(pts)
        if abs(x - spec[0]) > 1e-9 and s >= have + F.MARGIN_SHARE * margin_n and (best is None or s > best[3]):
            best = (k, x, f"s{stage}/{k.split('.')[1]}/{x:g}", s)
    return best


def search(cell: str, procs: int, gpus: str) -> None:
    import fast

    d = T.OUT / f"cell_{cell}"
    objs, have = objects(cell, TRAIN), now(cell)
    labels = sorted(x for v in objs.values() for x in v)
    n_of = {o: len(v) for o, v in objs.items()}
    ks = knobs(cell)
    wins, lines = (
        {},
        [f"cell {cell}: {len(objs)} objects, {len(labels)} starts; the number table has {sum(have.values())}"],
    )

    def run(names_sets: list) -> None:
        job = d / f"search_{names_sets[0][0].split('/')[0]}.json"
        job.write_text(json.dumps(dict(family=F.FAMILY, pool=str(pool_of(cell, "train")), memo=True, starts=labels, variants=[
            dict(name=n, skills=str(d / "skills"), set={**base_of(cell), **st}, out=str(d / "search" / n)) for n, st in names_sets])))  # fmt: skip
        fast.run(job, procs, gpus)
        for n, _ in names_sets:
            rs = T.rounds(d / "search" / n)
            wins[n] = {o: sum(bool(rs[x]["ok"]) for x in v if x in rs) for o, v in objs.items()}

    run([("s0", {})])
    lines.append(f"package as written: {sum(wins['s0'].values())}  {wins['s0']}")
    cur, cur_name, path = {}, "s0", []
    for stage in range(1, STAGES + 1):
        todo = {k: v for k, v in ks.items() if k not in cur}
        run([(f"s{stage}/{k.split('.')[1]}/{v:g}", {**cur, k: v}) for k, spec in todo.items() for v in values(k, spec)])
        for k, spec in todo.items():
            pts = {spec[0]: sum(wins[cur_name].values())}
            pts.update({v: sum(wins[f"s{stage}/{k.split('.')[1]}/{v:g}"].values()) for v in values(k, spec)})
            lines.append(f"  stage {stage} {k}: " + " ".join(f"{a:g}:{pts[a]}" for a in sorted(pts)))
        b = pick(list(objs), todo, wins, stage, cur_name, len(labels))
        path.append((stage, cur_name, dict(todo)))
        if b is None:
            lines.append(
                f"stage {stage}: no plateau beats {sum(wins[cur_name].values())} by {F.MARGIN_SHARE:.0%} of the starts; stop"
            )
            break
        cur[b[0]] = b[1]
        ks[b[0]] = (b[1], *ks[b[0]][1:])
        cur_name = b[2]
        lines.append(f"stage {stage}: take {b[0]} = {b[1]:g} -> {sum(wins[cur_name].values())} (smoothed {b[3]:.1f})")
    # leave one prototype out: follow the same stages choosing without it; it leaves the path where its choice differs
    loo, per = 0, {}
    for o in objs:
        others = [x for x in objs if proto(x) != proto(o)]  # twins of one prototype share a shelf and a fate
        name = "s0"
        for stage, at, todo in path:
            if at != name:
                break
            b = pick(others, todo, wins, stage, name, sum(n_of[x] for x in others))
            if b is None:
                break
            name = b[2]
        per[o] = (wins[name][o], n_of[o], sum(have[x] for x in objs[o]))
        loo += wins[name][o]
    ref = sum(have.values())
    ok = loo >= ref + F.MARGIN_SHARE * len(labels)
    dev = CELLS[cell]["dev"]
    lines.append(f"final set {cur}: {sum(wins[cur_name].values())} of {len(labels)} in-sample")
    lines.append(f"left-one-prototype-out {loo} vs the number table {ref} (needs {ref + F.MARGIN_SHARE * len(labels):.1f}): "
                 f"{'ACCEPT' if ok else 'REJECT'}")  # fmt: skip
    for o, (w, n, h) in sorted(per.items()):
        lines.append(f"  {o}{'' if o in dev else ' (kept back from the writer)'}: {h} -> {w} of {n}")
    kept = [o for o in objs if o not in dev]
    lines.append(f"objects kept back from the writer: {sum(per[o][2] for o in kept)} -> {sum(per[o][0] for o in kept)} "
                 f"of {sum(per[o][1] for o in kept)}")  # fmt: skip
    (d / "search.txt").write_text("\n".join(lines) + "\n")
    (d / "final.json").write_text(json.dumps(dict(accept=bool(ok), set=cur, loo=loo, ref=ref, n=len(labels))))
    print("\n".join(x for x in lines if not x.startswith("  stage")))


def test(cell: str, procs: int, gpus: str) -> None:
    import fast

    d = T.OUT / f"cell_{cell}"
    fin = json.loads((d / "final.json").read_text())
    objs = objects(cell, HELD)
    labels = sorted(x for v in objs.values() for x in v)
    job = d / "test.json"
    job.write_text(json.dumps(dict(family=F.FAMILY, pool=str(pool_of(cell, "held")), memo=False, starts=labels, variants=[
        dict(name="cell", skills=str(d / "skills"), set={**base_of(cell), **fin["set"]}, out=str(d / "test"))])))  # fmt: skip
    fast.run(job, procs, gpus)
    base, tab, new = T.rounds(F.D / "test/base"), T.rounds(F.D / "test/table"), T.rounds(d / "test")
    lines = []
    tot = [0, 0, 0, 0]
    for o, v in sorted(objs.items()):
        row = [len(v), sum(bool(base[x]["ok"]) for x in v), sum(bool(tab.get(x, base[x])["ok"]) for x in v),
               sum(bool(new[x]["ok"]) for x in v if x in new)]  # fmt: skip
        tot = [a + b for a, b in zip(tot, row)]
        cls = collections.Counter(new[x].get("failure_class") for x in v if x in new and not new[x]["ok"])
        lines.append(f"  {o}: incumbent {row[1]}, number table {row[2]}, new grasp {row[3]} of {row[0]}  {dict(cls)}")
    head = (f"cell {cell}, held-out objects ({len(objs)} objects, {tot[0]} starts; search verdict "
            f"{'ACCEPT' if fin['accept'] else 'REJECT'}): incumbent {tot[1]}, number table {tot[2]}, new grasp {tot[3]}")  # fmt: skip
    (d / "test.txt").write_text("\n".join([head, *lines]) + "\n")
    print("\n".join([head, *lines]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "search", "test"))
    ap.add_argument("cell", choices=sorted(CELLS))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.cell)
    else:
        globals()[a.cmd](a.cell, a.procs, a.gpus)
