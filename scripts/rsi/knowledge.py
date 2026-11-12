"""The two-layer knowledge library (docs/handoff/shared-knowledge-plan-20261111.md): entries of the form condition ->
failure signature -> repair, built from a finished loop's ledger and PAST scene-graph records, scoped `shared` or
`local` by their per-object evidence, and retrieved by evaluating each entry's condition on the incumbent's
diagnostic rounds. Roles read the hits (HITS.md); skills may not (rules.md 39).

    python3 scripts/rsi/knowledge.py build <sandbox> <family> [--iters 0-3] [--out scripts/rsi/knowledge] [--stub]
    python3 scripts/rsi/knowledge.py evidence <sandbox> <family> [--iters 0-3]        # the tables only, no model call
    python3 scripts/rsi/knowledge.py eval <sandbox root> <rounds dir> --out <dir>      # knowledge_hits.json + HITS.md
    python3 scripts/rsi/knowledge.py index [--out scripts/rsi/knowledge]
    python3 scripts/rsi/knowledge.py report <sandbox> [<sandbox> ...]                 # the replay A/B table
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE)]

import records as RC  # noqa: E402

VISIBLE = "outputs/scene_rsi_20261016"
OUT = HERE / "knowledge"
MODEL = os.environ.get("RSI_MODEL", "claude-opus-5-5")
MIN_FLIPS = 2  # triggered flipped rounds an entry needs
SEPARATION_P = 0.10  # Fisher's p for "the condition separates flipped from unchanged rounds" (tables of >= records.MIN_REFUTE rows)
PAD = 0.10  # a narrowing range is widened by this fraction on each side
START_COLS = ("start.obj_height_m", "start.obj_min_width_m", "start.support_top_m", "start.container",
              "start.obj_mass_kg", "start.over_support_m", "start.side_fits_n")  # fmt: skip


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def jload(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def _iter_of(ref: str) -> tuple[int, str]:
    """A Manager `same_as` reference "NN:id" -> (NN, id)."""
    n, _, hid = ref.partition(":")
    return int(n), hid


def mechanisms(loop: Path, iters: set[int] | None = None) -> list[dict]:
    """One dict per mechanism (module docstring): hypotheses linked by the Manager's `repeats`, their probe and draft
    flips, the gate's verdict on their drafts. Mechanisms with no flip and no gate verdict are left out."""
    rows = [
        r for r in jl(loop / "ledger.jsonl") if isinstance(r.get("iter"), int) and (iters is None or r["iter"] in iters)
    ]
    by_iter: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_iter[r["iter"]].append(r)
    parent: dict[tuple[int, str], tuple[int, str]] = {}

    def find(k):
        while parent.setdefault(k, k) != k:
            k = parent[k]
        return k

    hyps: dict[tuple[int, str], dict] = {}
    for n in sorted(by_iter):
        for h in (jload(loop / f"iter_{n:02d}" / "hypotheses.json", {}) or {}).get("hypotheses") or []:
            if isinstance(h, dict) and h.get("id"):
                hyps[(n, h["id"])] = h
        for r in by_iter[n]:
            if r.get("step") == "manager":
                for rep in r.get("repeats") or []:
                    for ref in rep.get("same_as") or []:
                        try:
                            parent[find((n, rep["now"]))] = find(_iter_of(ref))
                        except (ValueError, KeyError):
                            continue
    groups: dict[tuple[int, str], list[tuple[int, str]]] = defaultdict(list)
    for k in sorted(hyps):
        groups[find(k)].append(k)
    out = []
    for root, members in groups.items():
        members.sort()
        ids = set(members)
        flips: dict[str, str] = {}
        drafts, gate = [], None
        incumbent = {}
        for n in sorted({m[0] for m in members}):
            inc = (jload(loop / f"iter_{n:02d}" / "reviewer_context.json", {}) or {}).get("incumbent") or {}
            if inc.get("hash"):
                incumbent[n] = inc["hash"]
            for r in by_iter[n]:
                if r.get("step") == "probe":
                    for p in r.get("results") or []:
                        if (n, p.get("id")) in ids:
                            flips.update({x: "fixed" for x in p.get("fixed") or []})
                            flips.update({x: "broken" for x in p.get("broken") or []})
            cands = [c for c in jload(loop / f"iter_{n:02d}" / "candidates.json", []) or []
                     if isinstance(c, dict) and (n, c.get("hypothesis")) in ids]  # fmt: skip
            for c in cands:
                flips.update({x: "fixed" for x in c.get("fixed") or []})
                flips.update({x: "broken" for x in c.get("broken") or []})
                drafts.append(
                    dict(iter=n, number=c.get("number"), hash=c.get("hash"), mechanism=c.get("mechanism") or "")
                )
                for g in by_iter[n]:
                    if g.get("step") == "gate" and g.get("hash") == c.get("hash") and not g.get("void"):
                        gate = dict(b=g.get("b"), c=g.get("c"), accepted=bool(g.get("accepted")))
        if not flips and gate is None:
            continue
        first = members[0]
        out.append(dict(slug=first[1], ids=members, claims=[hyps[m].get("claim", "") for m in members],
                        insteads=[hyps[m].get("instead", "") for m in members], incumbent=incumbent, flips=flips,
                        drafts=drafts, gate=gate))  # fmt: skip
    return out


def version_of_dir(source: str) -> str | None:
    """The skills version a diag round was run under, from its directory: diag/incumbent_<hash> or candidate_<hash>."""
    name = Path(source).name
    return name.split("_", 1)[1] if name.startswith(("incumbent_", "candidate_")) and "_" in name else None


def diag_rows(sandbox: Path, family: str) -> list[dict]:
    """Every visible diagnostic round of the family in the sandbox, as record-library rows with the scene columns
    and `version` (the incumbent or candidate hash of the directory the round sits in; a round's `skills` field is
    the package hash, a different value). Retired rounds (`_retired_*.json`, pool_diag_retired) count too: they
    were in the pool when the flips were measured. records.scan takes the start files from the directories given."""
    fam = sandbox / VISIBLE / family
    dirs = [(fam / d, family) for d in ("diag", "pool_diag", "pool_diag_retired")]
    rows = RC.scan(dirs, RC.machine(), RC.scene_meta(sandbox, family), retired=True)
    for r in rows:
        r["version"] = version_of_dir(r["source"])
    return rows


def evidence_table(mech: dict, rows: list[dict]) -> list[dict]:
    """The rounds the mechanism's condition must separate: of the tasks it flipped, every round under the incumbent
    version the flip was measured against (the situation before the change), with `flip`."""
    versions = set(mech["incumbent"].values())
    tasks = {r["task"] for r in rows if r["label"] in mech["flips"]}
    out = []
    for r in rows:
        if r["task"] in tasks and r.get("version") in versions:
            out.append(dict(r, flip=mech["flips"].get(r["label"], "unchanged")))
    return out


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def rank_columns(table: list[dict]) -> list[dict]:
    """Per column, how differently the flipped and the unchanged rounds are distributed, as a score in [0, 1]:
    numeric columns d / (d + IQR) with d the difference of the group medians and IQR the pooled interquartile range
    (with each group's [p10, p90]); categorical ones the share of the flipped rounds' majority category minus its
    share among the unchanged rounds, floored at 0 (with counts). A column with no unchanged rounds scores 0.0:
    there is nothing to separate from. Best first, ties by column name."""
    flipped = [r for r in table if r["flip"] != "unchanged"]
    rest = [r for r in table if r["flip"] == "unchanged"]
    cols = {k for r in table for k in r if "." in k}
    out = []
    for k in sorted(cols):
        fv = [r[k] for r in flipped if k in r]
        rv = [r[k] for r in rest if k in r]
        if not fv:
            continue
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in fv + rv):
            pooled = sorted(fv + rv)
            iqr = _pct(pooled, 0.75) - _pct(pooled, 0.25)
            d = abs(_pct(fv, 0.5) - _pct(rv, 0.5)) if rv else 0.0
            score = d / (d + iqr) if d else 0.0
            out.append(dict(col=k, kind="num", score=round(score, 3), flipped=[_pct(fv, 0.1), _pct(fv, 0.9)],
                            unchanged=[_pct(rv, 0.1), _pct(rv, 0.9)] if rv else None))  # fmt: skip
        else:
            fc, rc = Counter(map(str, fv)), Counter(map(str, rv))
            top, n = fc.most_common(1)[0]
            score = max(0.0, n / len(fv) - rc.get(top, 0) / len(rv)) if rv else 0.0
            out.append(dict(col=k, kind="cat", score=round(score, 3), flipped=dict(fc), unchanged=dict(rc)))
    return sorted(out, key=lambda c: (-c["score"], c["col"]))


def render_evidence(mech: dict, table: list[dict], ranked: list[dict], top: int = 12) -> str:
    lines = [f"# mechanism {mech['slug']}", "", "hypotheses: " + "; ".join(f"{n}:{h}" for n, h in mech["ids"]), ""]
    for c, i in zip(mech["claims"], mech["insteads"], strict=True):
        lines += [f"- claim: {c}", f"  instead: {i}"]
    if mech["gate"]:
        lines.append(f"- gate: b {mech['gate']['b']} c {mech['gate']['c']} accepted {mech['gate']['accepted']}")
    lines += ["", "## columns that separate flipped from unchanged rounds (best first)", ""]
    for c in ranked[:top]:
        lines.append(
            f"- {c['col']} ({c['kind']}, score {c['score']}): flipped {c['flipped']} | unchanged {c['unchanged']}"
        )
    lines += ["", "## rounds", "", "| label | task | ok | flip | " + " | ".join(c["col"] for c in ranked[:top]) + " |",
              "|---|---|---|---|" + "---|" * min(top, len(ranked))]  # fmt: skip
    for r in sorted(table, key=lambda r: (r["flip"] == "unchanged", r["label"])):
        vals = [str(r.get(c["col"], "")) for c in ranked[:top]]
        lines.append(f"| {r['label']} | {r['task']} | {r['ok']} | {r['flip']} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _clause(c: dict, row: dict) -> bool:
    f, op, v = c.get("feature"), c.get("op"), c.get("value")
    if f not in row or op not in RC.OPS:
        return False
    try:
        return bool(RC.OPS[op](row[f], v))
    except TypeError:
        return False


def holds(condition: dict, row: dict) -> bool:
    """The condition on one record-library row: every `all` clause, one of `any` (when given), every `narrowed_by`
    clause. A missing column or a type mismatch is false, never unknown-as-true."""
    if not all(_clause(c, row) for c in condition.get("all") or []):
        return False
    anys = condition.get("any") or []
    if anys and not any(_clause(c, row) for c in anys):
        return False
    return all(_clause(c, row) for c in condition.get("narrowed_by") or [])


def trigger(condition: dict, rows: list[dict]) -> list[dict]:
    return [r for r in rows if holds(condition, r)]


def object_of(task: str) -> str:
    """The task's object key: the id without its verb prefix (pick_Wolf_1 -> Wolf_1)."""
    return task.split("_", 1)[1] if "_" in task else task


def by_object(triggered: list[dict], flips: dict[str, str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in triggered:
        o = out.setdefault(object_of(r["task"]), dict(triggered=0, fixed=0, broken=0))
        o["triggered"] += 1
        f = flips.get(r["label"])
        if f in ("fixed", "broken"):
            o[f] += 1
    return out


def scope_of(bo: dict[str, dict]) -> tuple[str, str]:
    """Plan section 4.1: shared, local or drop, with the reason."""
    flipped = sum(o["fixed"] + o["broken"] for o in bo.values())
    if flipped < MIN_FLIPS:
        return "drop", f"{flipped} triggered flipped round(s), fewer than {MIN_FLIPS}"
    net = {k: o["fixed"] - o["broken"] for k, o in bo.items()}
    negative = sorted(k for k, v in net.items() if v < 0)
    positive = sorted(k for k, v in net.items() if v > 0)
    if negative:
        return "local", f"net negative on {', '.join(negative)}; positive on {', '.join(positive) or 'none'}"
    if len(bo) >= 2 and positive:
        return "shared", f"net non-negative on {len(bo)} objects, positive on {', '.join(positive)}"
    if len(bo) == 1 and positive:
        return "local", f"positive on one object ({positive[0]})"
    return "drop", "no object is net positive"


def separation(condition: dict, table: list[dict]) -> dict:
    """Does the condition pick out the flipped rounds? Fisher's exact p on triggered x flipped."""
    hit = [holds(condition, r) for r in table]
    fl = [r["flip"] != "unchanged" for r in table]
    a = sum(h and f for h, f in zip(hit, fl, strict=True))
    b = sum(h and not f for h, f in zip(hit, fl, strict=True))
    c = sum(f and not h for h, f in zip(hit, fl, strict=True))
    d = len(table) - a - b - c
    return dict(n_in=a + b, flipped_in=a, n_out=c + d, flipped_out=c, p=round(RC.fisher(a, b, c, d), 4))


def narrow(condition: dict, table: list[dict], flips: dict[str, str]) -> dict:
    """Plan section 4.2: when an object is net negative, add start.* ranges taken from the net-positive objects'
    triggered rounds (padded by PAD) one column at a time until every negative object leaves the trigger set."""
    cond = dict(condition)
    cond.pop("narrowed_by", None)
    bo = by_object(trigger(cond, table), flips)
    negative = {k for k, o in bo.items() if o["fixed"] - o["broken"] < 0}
    positive = {k for k, o in bo.items() if o["fixed"] - o["broken"] > 0}
    if not negative or not positive:
        return condition
    pos_rows = [r for r in trigger(cond, table) if object_of(r["task"]) in positive]
    added = []
    for col in START_COLS:
        vals = [r[col] for r in pos_rows if isinstance(r.get(col), (int, float))]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        pad = PAD * (hi - lo) if hi > lo else PAD * abs(hi) or PAD
        clauses = [dict(feature=col, op=">=", value=round(lo - pad, 4), why="positives' range"),
                   dict(feature=col, op="<=", value=round(hi + pad, 4), why="positives' range")]  # fmt: skip
        trial = dict(cond, narrowed_by=added + clauses)
        still = {k for k, o in by_object(trigger(trial, table), flips).items() if o["fixed"] - o["broken"] < 0}
        if len(still) < len(negative):
            added += clauses
            negative = still
        if not negative:
            return dict(cond, narrowed_by=added)
    return condition


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evidence")
    ev.add_argument("sandbox", type=Path)
    ev.add_argument("family")
    ev.add_argument("--iters", help="N-M: iterations whose evidence is used")
    ev.add_argument("--out", type=Path, help="write one .md per mechanism here instead of printing")
    a = ap.parse_args(argv)
    if a.cmd == "evidence":
        iters = None
        if a.iters:
            lo, _, hi = a.iters.partition("-")
            iters = set(range(int(lo), int(hi or lo) + 1))
        loop = a.sandbox / VISIBLE / a.family / "loop"
        rows = diag_rows(a.sandbox, a.family)
        for m in mechanisms(loop, iters):
            t = evidence_table(m, rows)
            md = render_evidence(m, t, rank_columns(t))
            if a.out:
                a.out.mkdir(parents=True, exist_ok=True)
                (a.out / f"{m['slug']}.md").write_text(md)
            else:
                print(md)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
