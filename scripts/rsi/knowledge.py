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
