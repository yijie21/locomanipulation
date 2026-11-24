"""TODO 47: search first, compress after. A small model only names, per object, the numbers worth searching and a
range; everything else is a program.

    search.py sweeps <knobs.json>   # stage 1 proposals: every named number swept alone over its range
    search.py anchor                # after stage 1: per object the best plateau; stage 2 re-sweeps the rest around it
    search.py compress              # after stage 2: table_search.json and the residual (what no setting solves)
    search.py test                  # the table on pool_test2 (never used before), against the incumbent there

The rounds run through table.py's evaluate ("search" is the source name). A value is taken from the middle of the
widest run of best smoothed wins, and only when that smoothed count beats the baseline by MARGIN: the best of many
tries on 16 starts is biased upward, a plateau is not.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import table as T  # noqa: E402

SRC = "search"
N_VALUES = 7
MAX_KNOBS = 6
MARGIN = 1.5  # smoothed training wins a plateau needs over what the cell has already
RESIDUAL = 13  # a cell still under this many training wins of 16 is reported as beyond the numbers' reach


def knobs(path: Path) -> dict:
    raw, keys, dflt, out = json.loads(path.read_text()), T.parameters(), T.defaults(), {}
    for t, ks in raw.items():
        keep = []
        for k in ks[:MAX_KNOBS]:
            key, lo, hi = k.get("key"), k.get("lo"), k.get("hi")
            if (
                key in keys
                and key in dflt
                and isinstance(lo, (int, float))
                and isinstance(hi, (int, float))
                and lo < hi
            ):
                keep.append((key, float(lo), float(hi)))
        out[t] = keep
    return out


def grid(key: str, lo: float, hi: float, dflt: dict) -> list[float]:
    d = dflt[key]
    if float(d).is_integer() and float(lo).is_integer() and float(hi).is_integer() and hi - lo <= 3:
        return [float(v) for v in range(int(lo), int(hi) + 1) if v != d]  # a switch or a small count
    vals = {round(lo + i * (hi - lo) / (N_VALUES - 1), 5) for i in range(N_VALUES)}
    return sorted(v for v in vals if abs(v - d) > 1e-9)


def sweeps(path: Path) -> None:
    dflt, out = T.defaults(), {}
    (T.OUT / "knobs_used.json").write_text(json.dumps(knobs(path), indent=1))
    for t, ks in knobs(path).items():
        out[t] = [dict(why=f"sweep {k}", set={k: v}) for k, lo, hi in ks for v in grid(k, lo, hi, dflt)]
    (T.OUT / f"proposals_{SRC}_r3.json").write_text(json.dumps(out, indent=1))  # rounds >= 3 have no move cap
    print({t: len(m) for t, m in out.items()}, "total", sum(len(m) for m in out.values()))


def curve(tried: dict, fixed: dict, key: str, dflt: dict, n: int, base_wins: int) -> dict:
    """{value: wins} of the variants that are `fixed` plus `key` alone; the default's value is `fixed`'s own result."""
    pts = {}
    for _, (st, rs) in tried.items():
        if len(rs) != n or set(st) != set(fixed) | {key} or any(abs(st[k] - v) > 1e-9 for k, v in fixed.items()):
            continue
        pts[round(st[key], 5)] = sum(r["ok"] for r in rs)
    pts.setdefault(round(dflt[key], 5), base_wins)
    return pts


def plateau(pts: dict) -> tuple[float, float]:
    """(value, smoothed wins) at the middle of the widest run of best smoothed wins."""
    xs = sorted(pts)
    w = [pts[x] for x in xs]
    if len(xs) < 3:
        i = max(range(len(xs)), key=lambda j: w[j])
        return xs[i], min(w) if len(xs) > 1 else w[0]  # a switch: no neighbours, be conservative
    sm = [(w[max(i - 1, 0)] + 2 * w[i] + w[min(i + 1, len(w) - 1)]) / 4 for i in range(len(w))]
    good = [i for i, y in enumerate(sm) if y >= max(sm) - 0.26]
    runs, cur = [], [good[0]]
    for i in good[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    best = max(runs, key=len)
    i = best[len(best) // 2]
    return xs[i], sm[i]


def base_wins() -> dict:
    out = collections.Counter()
    for r in T.rounds(T.OUT / "train/base").values():
        out[r["task"]] += r["ok"]
    return out


def best_knob(tried: dict, fixed: dict, have: float, ks: list, dflt: dict, lines: list, t: str):
    best = None
    for key, _, _ in ks:
        if key in fixed:
            continue
        pts = curve(tried, fixed, key, dflt, 16, round(have))
        if len(pts) < 2:
            continue
        v, s = plateau(pts)
        lines.append(
            f"  {t} {key} given {fixed}: "
            + " ".join(f"{x:g}:{pts[x]}" for x in sorted(pts))
            + f" -> {v:g} smoothed {s:.2f}"
        )
        if s >= have + MARGIN and (best is None or s > best[2]):
            best = (key, v, s)
    return best


def anchor() -> None:
    dflt, used, tr, base = T.defaults(), json.loads((T.OUT / "knobs_used.json").read_text()), T.tried(SRC), base_wins()
    out, anchors, lines = {}, {}, []
    for t, ks in used.items():
        b = best_knob(tr.get(t, {}), {}, base[t], ks, dflt, lines, t)
        if b is None:
            out[t] = []
            continue
        anchors[t] = {b[0]: b[1]}
        lines.append(f"{t}: anchor {b[0]} = {b[1]:g} (smoothed {b[2]:.2f}, baseline {base[t]})")
        out[t] = [dict(why=f"sweep {k} around the anchor", set={**anchors[t], k: v})
                  for k, lo, hi in ks if k != b[0] for v in [*grid(k, lo, hi, dflt)]]  # fmt: skip
        out[t].append(dict(why="the anchor alone", set=anchors[t]))
    (T.OUT / "anchors.json").write_text(json.dumps(anchors, indent=1))
    (T.OUT / f"proposals_{SRC}_r4.json").write_text(json.dumps(out, indent=1))
    (T.OUT / "anchor.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(x for x in lines if not x.startswith("  ")))
    print("stage 2 variants", sum(len(m) for m in out.values()))


def compress() -> None:
    dflt, used, tr, base = T.defaults(), json.loads((T.OUT / "knobs_used.json").read_text()), T.tried(SRC), base_wins()
    anchors, table, lines = json.loads((T.OUT / "anchors.json").read_text()), {}, []
    for t in sorted(base):
        st, wins = dict(anchors.get(t, {})), base[t]
        if st:
            exact = [sum(r["ok"] for r in rs) for _, (s, rs) in tr.get(t, {}).items() if s == st and len(rs) == 16]
            wins = exact[0] if exact else wins
            b = best_knob(tr.get(t, {}), st, wins, used.get(t, []), dflt, lines, t)
            if b is not None:
                st[b[0]] = b[1]
                exact = [sum(r["ok"] for r in rs) for _, (s, rs) in tr.get(t, {}).items() if s == st and len(rs) == 16]
                wins = exact[0] if exact else round(b[2])
        won_any = set()
        for _, (_, rs) in tr.get(t, {}).items():
            won_any |= {r["label"] for r in rs if r["ok"]}
        table[t] = dict(set=st, move="search" if st else None, train_base=base[t], train=wins, n=16, any=len(won_any))
        lines.append(f"{t}: baseline {base[t]} -> {wins} with {st or 'nothing taken'}; any setting won {len(won_any)}")
    resid = [t for t, v in table.items() if v["train"] < RESIDUAL]
    lines.append(
        f"train total {sum(v['train_base'] for v in table.values())} -> {sum(v['train'] for v in table.values())} of 160"
    )
    lines.append(f"beyond the numbers' reach (under {RESIDUAL} of 16 after the search): {resid}")
    (T.OUT / f"table_{SRC}.json").write_text(json.dumps(table, indent=1))
    (T.OUT / "compress.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(x for x in lines if not x.startswith("  ")))


def proven() -> list:
    """Numbers whose plateau beat some cell's baseline by MARGIN, each with the widest range any cell gave it: what
    worked somewhere is worth trying everywhere."""
    dflt, used, tr, base = T.defaults(), json.loads((T.OUT / "knobs_used.json").read_text()), T.tried(SRC), base_wins()
    out = {}
    for t, ks in used.items():
        for key, lo, hi in ks:
            pts = curve(tr.get(t, {}), {}, key, dflt, 16, base[t])
            if len(pts) >= 2 and plateau(pts)[1] >= base[t] + MARGIN:
                o = out.get(key, (lo, hi))
                out[key] = (min(o[0], lo), max(o[1], hi))
    return [(k, lo, hi) for k, (lo, hi) in sorted(out.items())]


def spread() -> None:
    """Stage 3 proposals: every proven number swept in every cell that has not had it, around the cell's own set."""
    dflt, used, tr = T.defaults(), json.loads((T.OUT / "knobs_used.json").read_text()), T.tried(SRC)
    table, ks, out = json.loads((T.OUT / f"table_{SRC}.json").read_text()), proven(), {}
    for t, v in table.items():
        ms = []
        for key, lo, hi in ks:
            if key in v["set"] or any(key == k for k, _, _ in used.get(t, [])):
                continue  # already in the cell's set, or already swept in this cell (alone, or around its anchor)
            ms += [dict(why=f"spread {key}", set={**v["set"], key: x}) for x in grid(key, lo, hi, dflt)]
        out[t] = ms
    del tr
    (T.OUT / "proven.json").write_text(json.dumps(ks))
    (T.OUT / f"proposals_{SRC}_r5.json").write_text(json.dumps(out, indent=1))
    print("proven", ks)
    print({t: len(m) for t, m in out.items()}, "total", sum(len(m) for m in out.values()))


def compress2() -> None:
    """After stage 3: each cell may add proven numbers, one at a time, best first, while a plateau beats what it has."""
    dflt, tr, ks = T.defaults(), T.tried(SRC), [tuple(k) for k in json.loads((T.OUT / "proven.json").read_text())]
    table, lines = json.loads((T.OUT / f"table_{SRC}.json").read_text()), []
    for t, v in sorted(table.items()):
        b = best_knob(tr.get(t, {}), v["set"], v["train"], ks, dflt, lines, t)
        if b is not None:
            st = {**v["set"], b[0]: b[1]}
            exact = [sum(r["ok"] for r in rs) for _, (s, rs) in tr.get(t, {}).items() if s == st and len(rs) == 16]
            lines.append(f"{t}: {v['set']} ({v['train']}) -> {st} ({exact[0] if exact else round(b[2])})")
            v.update(set=st, move="search", train=exact[0] if exact else round(b[2]))
    lines.append(
        f"train total {sum(v['train_base'] for v in table.values())} -> {sum(v['train'] for v in table.values())} of 160"
    )
    lines.append(
        f"beyond the numbers' reach (under {RESIDUAL} of 16): {[t for t, v in table.items() if v['train'] < RESIDUAL]}"
    )
    (T.OUT / f"table_{SRC}2.json").write_text(json.dumps(table, indent=1))
    (T.OUT / "compress2.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(x for x in lines if not x.startswith("  ")))


def test(procs: int, gpus: str, source: str = SRC) -> None:
    import fast

    pool = T.OUT / "pool_test2"
    labels = sorted(p.stem for p in pool.glob("pick_*.json"))
    table = json.loads((T.OUT / f"table_{source}.json").read_text())
    changed = {t: v["set"] for t, v in table.items() if v["set"]}
    variants = [dict(name=t, skills=str(T.SKILLS), set={**T.BASE, **st}, starts=[x for x in labels if T.task_of(x) == t],
                     out=str(T.OUT / "test2" / source)) for t, st in changed.items()]  # fmt: skip
    job = T.OUT / f"test2_{source}.json"
    job.write_text(
        json.dumps(dict(family="grasp_atomic", pool=str(pool), memo=False, starts=labels, variants=variants))
    )
    fast.run(job, procs, gpus)
    base, new = T.rounds(T.OUT / "test2/base"), T.rounds(T.OUT / "test2" / source)
    per = collections.defaultdict(lambda: [0, 0, 0])
    for x, o in base.items():
        r = new.get(x, o) if T.task_of(x) in changed else o
        p = per[T.task_of(x)]
        p[0] += 1
        p[1] += o["ok"]
        p[2] += r["ok"]
    won = sum(new[x]["ok"] and not base[x]["ok"] for x in new if x in base)
    lost = sum(base[x]["ok"] and not new[x]["ok"] for x in new if x in base)
    text = [f"{source} on pool_test2 ({len(base)} fresh starts): incumbent {sum(p[1] for p in per.values())} -> table "
            f"{sum(p[2] for p in per.values())} (gained {won}, lost {lost})"]  # fmt: skip
    text += [f"  {t}: {p[1]} -> {p[2]} of {p[0]}  {changed.get(t, '')}" for t, p in sorted(per.items())]
    (T.OUT / f"test2_{source}.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sweeps", "anchor", "compress", "test", "spread", "compress2"))
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    if a.cmd == "sweeps":
        sweeps(Path(a.arg))
    elif a.cmd == "anchor":
        anchor()
    elif a.cmd in ("compress", "spread", "compress2"):
        globals()[a.cmd]()
    else:
        test(a.procs, a.gpus, a.arg or SRC)
