"""TODO 46: the skill as a table. One cell per object; a cell holds parameter overrides of the incumbent's skills.

An edit changes one cell, so it can only change that object's rounds (the simulation is deterministic): no gate is
needed to show nothing else regressed. A proposer (a language model, small or large) reads a cell's card and names a
few moves, each a handful of parameter values; a plain search evaluates them on the cell's training starts and keeps
the best; the table is tested once on starts no step here has seen.

    table.py cards                      # train/base rounds -> cards/<task>.md
    table.py eval <source> [--round 2]  # proposals_<source>.json -> rounds under train/<source>/
    table.py select <source>            # -> table_<source>.json
    table.py test <source>              # the table on the 120 held-out starts, against the recorded incumbent
    table.py oracle                     # per training start, did any variant of any source win
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path("/workspace/code/locomanipulation")
HERE = Path(__file__).resolve().parent
OUT = REPO / "outputs/scene_rsi_20261016/table_rsi_20261110"
AB = REPO / "outputs/scene_rsi_20261016/body/comp_ab_20261026"
SKILLS = AB / "skills"
BASE = {"pick.VIA_BACK_M": 0.2}  # the incumbent's own override
TRAIN = OUT / "pool_train"
TEST = {
    "diag": AB / "pool_diag_d",
    "accept": Path(
        "/workspace/code/rsi-scene/comp-ab-20261026/outputs/scene_rsi_20261016_hidden/grasp_atomic/pool_accept"
    ),
    "sealed": AB / "pool_sealed",
}
RECORDED = REPO / "outputs/scene_rsi_20261016/walkmap_20261027/rounds"
MAX_MOVES = 10  # proposals read per cell and round
MAX_KEYS = 4  # parameters one move may set
MIN_GAIN = 2  # training wins a move needs over the base before the cell takes it
ARM = ("pregrasp", "entry", "descend", "hand_entry", "hand_descent", "insert")
sys.path.insert(0, str(HERE))


def task_of(label: str) -> str:
    return label.rsplit("_s", 1)[0]


def rounds(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        if "label" in r:
            out[r["label"]] = r
    return out


def progress(r: dict) -> float:
    """TODO 46.4: how far a round got, 0..1: stood at the stance, hand on target, object between the fingers, lifted,
    held to the end; a little off for the jaw error and for other objects moved."""
    steps = r.get("steps", [])
    walk = next((s for s in reversed(steps) if "stop_err_cm" in s), None)
    gi = next((i for i, s in enumerate(steps) if s["step"] == "grip"), None)
    last = next((s for s in reversed(steps[: gi if gi is not None else len(steps)]) if "jaw_err_cm" in s), None)
    key = r["task"].split("_", 1)[1]
    grips = [s for s in steps if s["step"] == "grip"]
    held = any(key in " ".join(s.get("held", [])) for s in grips)
    stage = 0.0
    if walk is not None and walk["stop_err_cm"] < 5.0 and not r.get("fell"):
        stage = 0.2
        if last is not None and last["jaw_err_cm"] < (5.0 if last["step"] == "descend" else 3.0):
            stage = 0.4
        if held:
            stage = 0.6
            if r.get("plan_ok"):
                stage = 0.8
    if r["ok"]:
        stage = 1.0
    fine = 0.0 if last is None else min(1.0, last["jaw_err_cm"] / 10.0) * 0.05
    return round(stage - fine - 0.02 * min(3, len(r.get("others_moved", {}))), 3)


def summary(rs: list[dict]) -> dict:
    return dict(n=len(rs), wins=sum(r["ok"] for r in rs), progress=round(sum(progress(r) for r in rs) / max(1, len(rs)), 3),
                classes=dict(collections.Counter(r.get("failure_class") or "?" for r in rs if not r["ok"])))  # fmt: skip


def one_line(r: dict) -> str:
    parts = []
    for s in r["steps"]:
        if s["step"] == "stance":
            parts.append(f"stance {s.get('kind')} shoulder-to-object {s.get('shoulder_to_obj')} m")
        elif "stop_err_cm" in s:
            parts.append(f"{s['step']} stop {s['stop_err_cm']} cm")
        elif "jaw_err_cm" in s and s["step"] in (*ARM, "lift"):
            parts.append(f"{s['step']} jaw {s['jaw_err_cm']} cm off, wrist {s.get('attitude_err_deg')} deg off")
        elif s["step"] == "grip":
            parts.append(f"grip held {s.get('held')} opening {s.get('opening_cm')} cm")
    ev = r.get("events", {})
    facts = {k: ev[k] for k in ("obj_moved_before_grip_cm", "obj_first_move_by", "jaw_obj_at_grip_cm", "min_jaw_obj_cm",
                                "obj_tilt_before_grip_deg") if ev.get(k) is not None}  # fmt: skip
    return (
        f"- {r['label']}: {'WON' if r['ok'] else 'LOST ' + str(r.get('failure_class'))}. "
        + "; ".join(parts)
        + f". {facts}"
    )


def cards() -> None:
    base = rounds(OUT / "train/base")
    (OUT / "cards").mkdir(exist_ok=True)
    by = collections.defaultdict(list)
    for r in base.values():
        by[r["task"]].append(r)
    for t, rs in sorted(by.items()):
        s = summary(rs)
        text = [f"# {t}", f"Baseline on {s['n']} training starts: {s['wins']} won. Losses by class: {s['classes']}.",
                "", "Rounds (what each step measured):", *[one_line(r) for r in sorted(rs, key=lambda r: r["ok"])]]  # fmt: skip
        (OUT / "cards" / f"{t}.md").write_text("\n".join(text) + "\n")
        print(t, s)
    print("all", summary(list(base.values())))


def parameters() -> set:
    """The keys the skills package's own `apply_set` accepts (param_keys.json, read from the loaded package)."""
    return set(json.loads((OUT / "param_keys.json").read_text()))


def moves(source: str, rnd: int) -> dict:
    """{task: [(name, set)]} from the proposer's file, keeping only moves the interface allows."""
    f = OUT / (f"proposals_{source}.json" if rnd == 1 else f"proposals_{source}_r{rnd}.json")
    raw, known, out, dropped = json.loads(f.read_text()), parameters(), {}, 0
    for t, ms in raw.items():
        keep = []
        for i, m in enumerate(ms[: MAX_MOVES if rnd < 3 else 99]):
            st = m.get("set", {})
            ok = 0 < len(st) <= MAX_KEYS and all(k in known and isinstance(v, (int, float)) and not isinstance(v, bool)
                                                 for k, v in st.items())  # fmt: skip
            if ok:
                keep.append((f"r{rnd}m{i:02d}", st))
            else:
                dropped += 1
        out[t] = keep
    print(source, "round", rnd, "moves", sum(len(v) for v in out.values()), "dropped", dropped)
    return out


def evaluate(source: str, rnd: int, procs: int, gpus: str) -> None:
    import fast

    labels = sorted(p.stem for p in TRAIN.glob("pick_*.json"))
    variants = []
    for t, ms in moves(source, rnd).items():
        mine = [x for x in labels if task_of(x) == t]
        for name, st in ms:
            variants.append(dict(name=name, skills=str(SKILLS), set={**BASE, **st}, starts=mine,
                                 out=str(OUT / "train" / source / t / name)))  # fmt: skip
    job = OUT / f"eval_{source}_r{rnd}.json"
    job.write_text(
        json.dumps(dict(family="grasp_atomic", pool=str(TRAIN), memo=True, starts=labels, variants=variants))
    )
    print("variants", len(variants), "rc", fast.run(job, procs, gpus))


def tried(source: str) -> dict:
    """{task: {name: (set, [rounds])}} for every evaluated move of `source`."""
    out = collections.defaultdict(dict)
    sets = {}
    for f in sorted(OUT.glob(f"eval_{source}_r*.json")):
        for v in json.loads(f.read_text())["variants"]:
            sets[v["out"]] = {k: x for k, x in v["set"].items() if BASE.get(k) != x}
    for d in sorted((OUT / "train" / source).glob("pick_*/*")):
        rs = list(rounds(d).values())
        if rs:
            out[d.parent.name][d.name] = (sets.get(str(d), {}), rs)
    return out


def select(source: str) -> None:
    base = collections.defaultdict(list)
    for r in rounds(OUT / "train/base").values():
        base[r["task"]].append(r)
    table, lines = {}, []
    for t in sorted(base):
        b = summary(base[t])
        best, best_s = None, b
        for name, (st, rs) in sorted(tried(source).get(t, {}).items()):
            s = summary(rs)
            if s["n"] == b["n"] and (s["wins"], s["progress"]) > (best_s["wins"], best_s["progress"]):
                best, best_s = (name, st), s
        take = best is not None and best_s["wins"] >= b["wins"] + MIN_GAIN
        table[t] = dict(set=best[1] if take else {}, move=best[0] if take else None, train_base=b["wins"],
                        train=best_s["wins"] if take else b["wins"], n=b["n"])  # fmt: skip
        lines.append(f"{t}: base {b['wins']}/{b['n']} progress {b['progress']}; best {best and best[0]} "
                     f"{best_s['wins']} progress {best_s['progress']} {best and best[1]} -> {'TAKEN' if take else 'kept base'}")  # fmt: skip
    (OUT / f"table_{source}.json").write_text(json.dumps(table, indent=1))
    lines.append(f"train total: base {sum(v['train_base'] for v in table.values())} -> {sum(v['train'] for v in table.values())} "
                 f"of {sum(v['n'] for v in table.values())}; cells changed {sum(bool(v['set']) for v in table.values())}")  # fmt: skip
    (OUT / f"select_{source}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def test(source: str, procs: int, gpus: str, tag: str = "") -> None:
    import fast

    table = json.loads((OUT / f"table_{source}.json").read_text())
    changed = {t: v["set"] for t, v in table.items() if v["set"]}
    won = lost = n = base_wins = new_wins = 0
    detail = []
    for pool, d in TEST.items():
        labels = sorted(p.stem for p in d.glob("pick_*.json"))
        rec = rounds(RECORDED / pool)
        todo = [x for x in labels if task_of(x) in changed and x in rec]
        variants = [dict(name=t, skills=str(SKILLS), set={**BASE, **st}, starts=[x for x in todo if task_of(x) == t],
                         out=str(OUT / "test" / (source + tag) / pool)) for t, st in changed.items()]  # fmt: skip
        job = OUT / f"test_{source}{tag}_{pool}.json"
        job.write_text(json.dumps(dict(family="grasp_atomic", pool=str(d), memo=False, starts=todo, variants=variants)))
        if todo:
            fast.run(job, procs, gpus)
        new = rounds(OUT / "test" / (source + tag) / pool)
        for x in labels:
            if x not in rec:
                continue
            o = rec[x]
            r = new.get(x, o) if task_of(x) in changed else o
            n += 1
            base_wins += o["ok"]
            new_wins += r["ok"]
            won += r["ok"] and not o["ok"]
            lost += o["ok"] and not r["ok"]
            if r["ok"] != o["ok"]:
                detail.append(
                    f"{pool}/{x}: {'won' if r['ok'] else 'lost ' + str(r.get('failure_class'))} (was {o.get('failure_class') or 'won'})"
                )
    by = collections.Counter()
    text = [f"{source}: held-out {n} starts, incumbent {base_wins} -> table {new_wins} (gained {won}, lost {lost}); "
            f"cells changed: {sorted(changed)}", *detail]  # fmt: skip
    del by
    (OUT / f"test_{source}{tag}.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


def oracle() -> None:
    base = rounds(OUT / "train/base")
    best = {k: r["ok"] for k, r in base.items()}
    for src in sorted(p.name for p in (OUT / "train").iterdir() if p.is_dir() and p.name != "base"):
        for t, ms in tried(src).items():
            for _, (_, rs) in ms.items():
                for r in rs:
                    best[r["label"]] = best.get(r["label"], False) or r["ok"]
    by = collections.defaultdict(lambda: [0, 0, 0])
    for k, ok in best.items():
        b = by[task_of(k)]
        b[0] += 1
        b[1] += base[k]["ok"]
        b[2] += ok
    lines = [f"{t}: base {b[1]}/{b[0]}, any variant {b[2]}/{b[0]}" for t, b in sorted(by.items())]
    lines.append(
        f"total: base {sum(b[1] for b in by.values())}, any variant {sum(b[2] for b in by.values())} of {len(best)}"
    )
    (OUT / "oracle.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)  # a sweep's values: the default plus this share of the taken step


def defaults() -> dict:
    out = {}
    for line in (OUT / "params.md").read_text().split("\n"):
        m = re.match(r"^- `([\w.]+)` = (\S+)", line)
        if m and re.match(r"^-?[\d.]+$", m.group(2)):
            out[m.group(1)] = float(m.group(2))
    return {**out, **BASE}


def refine(source: str) -> None:
    """TODO 46.2: the values are the search's, not the proposer's. For each cell that took a move, sweep each of the
    move's numbers from the default to four times the step the move took (the others held at the move's values); written
    as round 3's proposals."""
    table, dflt, out = json.loads((OUT / f"table_{source}_r2.json").read_text()), defaults(), {}
    for t, v in table.items():
        ms = []
        for k, x in v["set"].items():
            d = dflt.get(k)
            if d is None or abs(x - d) < 1e-9 or (float(x).is_integer() and float(d).is_integer() and abs(x - d) <= 1):
                continue  # a switch, or nothing to sweep
            for g in GRID:
                val = round(d + g * (x - d), 5)
                if abs(val - x) > 1e-9:
                    ms.append(dict(why=f"sweep {k} x{g}", set={**v["set"], k: val}))
        out[t] = ms[: 4 * len(GRID)]
    (OUT / f"proposals_{source}_r3.json").write_text(json.dumps(out, indent=1))
    print(source, {t: len(m) for t, m in out.items() if m})


def plateau(source: str) -> None:
    """After round 3's sweeps: per cell and number, the value in the middle of the widest run of best smoothed wins;
    the cell's set with those values is round 4's one proposal."""
    table, dflt, tr, out = json.loads((OUT / f"table_{source}_r2.json").read_text()), defaults(), tried(source), {}
    for t, v in table.items():
        if not v["set"]:
            continue
        final = dict(v["set"])
        for k, x in v["set"].items():
            pts = {}
            for _, (st, rs) in tr.get(t, {}).items():
                full = {**v["set"], **{kk: vv for kk, vv in st.items()}}
                if (
                    set(st) <= set(v["set"]) | {k}
                    and all(abs(full.get(kk, dflt.get(kk, 0)) - vv) < 1e-9 for kk, vv in v["set"].items() if kk != k)
                    and len(rs) == v["n"]
                ):
                    pts[round(st.get(k, dflt.get(k, 0.0)), 5)] = sum(r["ok"] for r in rs)
            xs = sorted(pts)
            if len(xs) < 4:
                continue
            w = [pts[a] for a in xs]
            sm = [(w[max(i - 1, 0)] + 2 * w[i] + w[min(i + 1, len(w) - 1)]) / 4 for i in range(len(w))]
            top = max(sm)
            good = [i for i, y in enumerate(sm) if y >= top - 0.26]
            runs, cur = [], [good[0]]
            for i in good[1:]:
                if i == cur[-1] + 1:
                    cur.append(i)
                else:
                    runs.append(cur)
                    cur = [i]
            runs.append(cur)
            best = max(runs, key=len)
            final[k] = xs[best[len(best) // 2]]
            print(f"{source} {t} {k}: " + " ".join(f"{a}:{pts[a]}" for a in xs) + f" -> {final[k]} (was {x})")
        out[t] = [dict(why="plateau centres", set=final)] if final != v["set"] else []
    (OUT / f"proposals_{source}_r4.json").write_text(json.dumps(out, indent=1))


def select_refined(source: str) -> None:
    """The refined table: a cell takes round 4's plateau set when it wins within one round of the taken move on the
    training starts (the plateau's middle is preferred to its edge at equal evidence)."""
    table, tr = json.loads((OUT / f"table_{source}_r2.json").read_text()), tried(source)
    for t, v in table.items():
        for name, (st, rs) in tr.get(t, {}).items():
            if name.startswith("r4") and len(rs) == v["n"] and sum(r["ok"] for r in rs) >= v["train"] - 1 and v["set"]:
                print(f"{source} {t}: {v['set']} ({v['train']}) -> {st} ({sum(r['ok'] for r in rs)})")
                v.update(set=st, move=name, train=sum(r["ok"] for r in rs))
    (OUT / f"table_{source}.json").write_text(json.dumps(table, indent=1))


def jitter(source: str, procs: int, gpus: str, tag: str = "") -> None:
    """TODO 46.4: each held-out start run three times, as drawn and with the robot's start shifted 1 cm twice
    (pool_jitter): a start's score is the share of the three it wins, for the incumbent and for the table."""
    import fast

    pool = OUT / "pool_jitter"
    labels = sorted(p.stem for p in pool.glob("pick_*.json"))
    table = json.loads((OUT / f"table_{source}.json").read_text())
    changed = {t: v["set"] for t, v in table.items() if v["set"]}
    variants = [dict(name="base", skills=str(SKILLS), set=BASE, out=str(OUT / "jitter/base"))]
    variants += [dict(name=t, skills=str(SKILLS), set={**BASE, **st}, starts=[x for x in labels if task_of(x) == t],
                      out=str(OUT / "jitter" / (source + tag))) for t, st in changed.items()]  # fmt: skip
    job = OUT / f"jitter_{source}{tag}.json"
    job.write_text(json.dumps(dict(family="grasp_atomic", pool=str(pool), memo=True, starts=labels, variants=variants)))
    fast.run(job, procs, gpus)
    jb, jt = rounds(OUT / "jitter/base"), rounds(OUT / "jitter" / (source + tag))
    rows = []
    for name in TEST:
        rec, new = rounds(RECORDED / name), rounds(OUT / "test" / (source + tag) / name)
        for x, o in rec.items():
            if not (TEST[name] / f"{x}.json").exists():
                continue
            b = [o["ok"]] + [jb[f"{x}j{j}"]["ok"] for j in (1, 2) if f"{x}j{j}" in jb]
            if task_of(x) in changed:
                t = [new[x]["ok"]] + [jt[f"{x}j{j}"]["ok"] for j in (1, 2) if f"{x}j{j}" in jt]
            else:
                t = b
            rows.append((x, sum(b), len(b), sum(t), len(t)))
    n = len(rows)
    cnt = lambda i, k: sum(r[i] == k for r in rows)  # noqa: E731
    text = [
        f"{source}{tag}: {n} held-out starts x 3 (as drawn, two 1 cm shifts of the robot's start)",
        f"incumbent: rounds won {sum(r[1] for r in rows)}/{sum(r[2] for r in rows)}; starts won 3/3 {cnt(1, 3)}, 2/3 {cnt(1, 2)}, 1/3 {cnt(1, 1)}, 0/3 {cnt(1, 0)}",
        f"table:     rounds won {sum(r[3] for r in rows)}/{sum(r[4] for r in rows)}; starts won 3/3 {cnt(3, 3)}, 2/3 {cnt(3, 2)}, 1/3 {cnt(3, 1)}, 0/3 {cnt(3, 0)}",
        f"starts whose share rose {sum(r[3] > r[1] for r in rows)}, fell {sum(r[3] < r[1] for r in rows)}",
    ]
    (OUT / f"jitter_{source}{tag}.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd", choices=("cards", "eval", "select", "test", "oracle", "jitter", "refine", "plateau", "select_refined")
    )
    ap.add_argument("source", nargs="?")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    if a.cmd == "cards":
        cards()
    elif a.cmd == "eval":
        evaluate(a.source, a.round, a.procs, a.gpus)
    elif a.cmd == "select":
        select(a.source)
    elif a.cmd == "test":
        test(a.source, a.procs, a.gpus, a.tag)
    elif a.cmd in ("refine", "plateau", "select_refined"):
        globals()[a.cmd](a.source)
    elif a.cmd == "jitter":
        jitter(a.source, a.procs, a.gpus, a.tag)
    else:
        oracle()
