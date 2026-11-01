"""TODO 39.4 (plan docs/handoff/past-scene-graph-plan-20261030.md section 4.9): read a round's PAST records.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/past.py show <diag dir>/<label> [--step N] [--json]
    PYTHONPATH=src $PY scripts/rsi/past.py compare <failed round> <successful round>

`show` prints every step: roles, G_before -> G_after per edge with its expectation, the events in between, the
bodies that joined the scope, and where the NPZ ticks are; `compare` aligns two rounds' steps by kind (longest common
subsequence) and prints the edges whose values differ. Reading only; the records are written by scene_round.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src")]

from locomani.tasks.scene import past as PAST  # noqa: E402

SKIP_IN_NUMS = ("s", "r", "o", "v", "src")


def side_file(p: Path) -> Path:
    """<dir>/<label>, <label>.json or <label>.past.jsonl -> the side file."""
    p = Path(p)
    if p.name.endswith(".past.jsonl"):
        return p
    if p.suffix == ".json":
        return p.with_name(p.stem + ".past.jsonl")
    return p.with_name(p.name + ".past.jsonl")


def records(p: Path) -> list[dict]:
    return [r for r in PAST.read(side_file(p)) if "record_id" in r]


def _b(v) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, bool):
        return str(v).lower()
    return json.dumps(v)


def _edge(s, r, o) -> str:
    return f"{s} —{r}→ {o}" if o is not None else f"{s} —{r}"


def _key(e: dict) -> tuple:
    return (e["s"], e["r"], e["o"])


def record_lines(r: dict) -> list[str]:
    a = r.get("action") or {}
    kind = r["kind"] + ("?" if r.get("kind_src") == "primitive" else "")  # '?': from the trace, no call log
    head = (f"{r['step_index']}.{r['attempt']} {r['step']} [{kind}] {a.get('t0', 0):.1f}-{a.get('t1', 0):.1f} s "
            f"{'OK' if a.get('ok') else 'FAIL'}")  # fmt: skip
    if a.get("why"):
        head += f"  {a['why']}"
    out = [head]
    roles = {k: v for k, v in (r.get("roles") or {}).items() if v}
    if roles:
        out.append("  roles " + ", ".join(f"{k}={v}" for k, v in roles.items()))
    if a.get("check"):
        out.append("  check " + json.dumps(a["check"]))
    exp = {tuple(x["edge"]): x for x in r.get("expected") or []}
    before = {_key(e): e for e in r["G_before"]["edges"]}
    after = {_key(e): e for e in r["G_after"]["edges"]}
    bad = {tuple(d["edge"]) for d in r.get("discrepancies") or []}
    for k in sorted(set(before) | set(after), key=lambda k: (k[1], str(k[0]), str(k[2]))):
        b, f = before.get(k), after.get(k)
        e = f or b
        line = f"  {_edge(*k)}  {_b(b['v']) if b else '-'}→{_b(f['v']) if f else '-'}"  # '-': no such edge then
        nums = ", ".join(f"{kk} {vv}" for kk, vv in e.items() if kk not in SKIP_IN_NUMS)
        if nums:
            line += f"  ({nums})"
        if k in exp:
            line += f"  (expected {_b(exp[k]['to'])} {'✗' if k in bad else '✓'})"
        out.append(line)
    for d in r.get("discrepancies") or []:
        if tuple(d["edge"]) not in before and tuple(d["edge"]) not in after:
            out.append(f"  {_edge(*d['edge'])}  expected {_b(d['to'])}, observed {_b(d.get('observed'))} ✗")
    if r.get("events"):
        out.append("  events:")
        for e in r["events"]:
            extra = "".join(f"  {k} {v}" for k, v in e.items() if k not in ("t", "edge", "from", "to", "src"))
            out.append(f"    {e['t']:.1f}  {_edge(*e['edge'])}  {_b(e['from'])}→{_b(e['to'])}{extra}")
    added = (r.get("scope") or {}).get("added") or []
    if added:
        out.append("  joined: " + ", ".join(f"{x['id']} at {x['t']:.1f} s ({x['why']})" for x in added))
    ev = r.get("evidence") or {}
    out.append(f"  evidence {ev.get('npz')} ticks {ev.get('k0')}-{ev.get('k1')}")
    return out


def show(path: Path, step: int | None = None, as_json: bool = False) -> str:
    lines = PAST.read(side_file(path))
    recs = [r for r in lines if "record_id" in r]
    if step is not None:
        recs = [r for r in recs if r["step_index"] == step]
    if as_json:
        return "\n".join(json.dumps(r, indent=1) for r in recs)
    if not recs:
        notes = [x.get("note") or f"error: {x['error']}" for x in lines if x.get("note") or x.get("error")]
        return "; ".join(notes) if notes else "no records"
    ok = recs[0].get("outcome", {}).get("round_ok")
    out = [f"{recs[0]['label']}  task {recs[0]['task']}  round {'OK' if ok else 'FAIL'}"]
    for r in recs:
        out += [""] + record_lines(r)
    return "\n".join(out)


def _lcs(a: list[str], b: list[str]) -> list[tuple[int | None, int | None]]:
    """Alignment of two sequences: (i, j) matched, (i, None) only in a, (None, j) only in b."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    i = j = 0
    out: list[tuple[int | None, int | None]] = []
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j))
            i, j = i + 1, j + 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            out.append((i, None))
            i += 1
        else:
            out.append((None, j))
            j += 1
    out += [(k, None) for k in range(i, n)] + [(None, k) for k in range(j, m)]
    return out


def compare(a: Path, b: Path) -> str:
    ra, rb = records(a), records(b)
    out = [f"A {ra[0]['label'] if ra else a}  B {rb[0]['label'] if rb else b}  (steps aligned by kind)"]
    for i, j in _lcs([r["kind"] for r in ra], [r["kind"] for r in rb]):
        ka = ra[i]["kind"] if i is not None else "-"
        kb = rb[j]["kind"] if j is not None else "-"
        left = f"{ra[i]['step']} {'OK' if ra[i]['action'].get('ok') else 'FAIL'}" if i is not None else "-"
        right = f"{rb[j]['step']} {'OK' if rb[j]['action'].get('ok') else 'FAIL'}" if j is not None else "-"
        out.append(f"\n{ka} | {kb}   A {left}   B {right}")
        if i is None or j is None:
            continue
        ea = {_key(e): e for e in ra[i]["G_after"]["edges"]}
        eb = {_key(e): e for e in rb[j]["G_after"]["edges"]}
        for k in sorted(set(ea) | set(eb), key=lambda k: (k[1], str(k[0]), str(k[2]))):
            va, vb = (ea.get(k) or {}).get("v"), (eb.get(k) or {}).get("v")
            if va != vb or (k in ea) != (k in eb):
                sa, sb = (_b(va) if k in ea else "-"), (_b(vb) if k in eb else "-")  # '-': no such edge in that round
                out.append(f"  {_edge(*k)}  A {sa}  B {sb}  <-- differs")
        fa = [(e["edge"][1], e["to"]) for e in ra[i].get("events") or []]
        fb = [(e["edge"][1], e["to"]) for e in rb[j].get("events") or []]
        if fa != fb:
            out.append(f"  events A {fa}  B {fb}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="one round's records, step by step")
    s.add_argument("round", type=Path, help="<diag dir>/<label>, its .json or its .past.jsonl")
    s.add_argument("--step", type=int, help="only this step index")
    s.add_argument("--json", action="store_true", help="the records as JSON")
    c = sub.add_parser("compare", help="two rounds' steps aligned by kind, edges that differ")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    a = ap.parse_args(argv)
    print(show(a.round, a.step, a.json) if a.cmd == "show" else compare(a.a, a.b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
