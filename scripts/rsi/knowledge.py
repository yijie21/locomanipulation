"""The two-layer knowledge library (docs/handoff/shared-knowledge-plan-20261111.md): entries of the form condition ->
failure signature -> repair, built from a loop's ledger and PAST scene-graph records, each with a `generality` from
its per-object evidence (generality_of): `multi_object` held and helped on two or more objects and hurt none,
`single_object` only one object in evidence, `contested` some object lost rounds (user 2026-11-23; before that
`scope` shared / local, read back by modern_entry). Skills may not read it (rules.md 39).

One shared file (user 2026-11-18): every entry is a line of `<library>/knowledge.jsonl`, tagged by the kind of task its
family learned it on (`tag`: `single` one-hand pick, `bimanual` two-hand pick, `mixed` otherwise; tag_of) with its
`source` (family, sandbox, iterations). A build replaces the entries of its own family and keeps every other one.
Before each Reviewer the loop writes `<library>/KNOWLEDGE.md` (write_hits): the WHOLE library, one section per tag,
each entry marked with the incumbent's diagnostic rounds its condition holds on (a mark, never a filter), and the code
syncs of an alternating run (`sync.jsonl`, scripts/rsi/alternate.py). Roles read KNOWLEDGE.md whole.

Every entry has a standing (user 2026-11-18): each Reviewer hypothesis that cites it in `knowledge_refs` is one use,
recorded in `usage.jsonl` with how the hypothesis ended (usage_rows: helped when the Prober verified it or its code
passed the gate, no effect when the Prober refuted it, else unclear). `tested`: it helped at least once and at least
as often as not; `doubtful`: no effect twice or more and more often than it helped; else `candidate`; `resolved`, for
the reader only: its condition holds just where the reader's incumbent already succeeds. KNOWLEDGE.md shows tested and
candidate entries in full, doubtful and resolved ones a line each. No model call: a build and `usage` count.

    python3 scripts/rsi/knowledge.py build <sandbox> <family> [--iters 0-3] [--out scripts/rsi/knowledge] [--tag single] [--stub]
    python3 scripts/rsi/knowledge.py evidence <sandbox> <family> [--iters 0-3]        # the tables only, no model call
    python3 scripts/rsi/knowledge.py eval <sandbox root> <rounds dir> --out <dir>      # knowledge_hits.json + KNOWLEDGE.md
    python3 scripts/rsi/knowledge.py usage <sandbox> <family> [--out scripts/rsi/knowledge]   # citations only, no build
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
LIBRARY = "knowledge.jsonl"  # the entries, one per line (user 2026-11-18: one shared file, tagged)
SYNCS = "sync.jsonl"  # an alternating run's code syncs (scripts/rsi/alternate.py), shown in KNOWLEDGE.md
ROLE_FILE = "KNOWLEDGE.md"  # what the Reviewer and the Engineer read, rewritten before each Reviewer (gitignored)
USAGE = "usage.jsonl"  # every citation of an entry and how the citing hypothesis ended (user 2026-11-18, usage_rows)
STATUS_ORDER = ("tested", "candidate", "doubtful", "resolved")  # KNOWLEDGE.md: full text for the first two
TAGS = {"single": "one-hand pick", "bimanual": "two-hand pick", "mixed": "mixed tasks"}
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


def flip_key(version: str | None, label: str) -> str:
    """A mechanism's `flips` key: the incumbent version the flip was measured against and the round's label
    ("<hash>:<label>"); a row is looked up by (row["version"], row["label"])."""
    return f"{version}:{label}"


def _note_flips(flips: dict[str, str], version: str | None, res: dict) -> None:
    """A probe result's or a draft's fixed / broken labels, under the incumbent version they were measured against
    (none without a known version); a later note of the same key wins."""
    if version:
        flips.update({flip_key(version, x): "fixed" for x in res.get("fixed") or []})
        flips.update({flip_key(version, x): "broken" for x in res.get("broken") or []})


def gate_bc(g: dict) -> tuple:
    """A gate ledger row's wins and losses: loop.py writes them under `result`; older rows and tests at the top."""
    r = g.get("result") or {}
    return g.get("b", r.get("b")), g.get("c", r.get("c"))


def draft_verdicts(rows: list[dict], drafts: list[dict]) -> dict[int, dict]:
    """{draft number: {b, c, accepted, via}} for one iteration's gate calls (its ledger rows), sync verdicts aside.
    A gated hash that is a draft's is that draft's (`draft`: alone or as the second shot); a composed candidate is the
    adopted draft's (`adopted`: the Composer took it, with overrides, so another hash) or, for a merge, the drafts the
    Composer lists in `merged` (`merged`; 2026-11-18). A merge row without that list (an older ledger) credits no
    draft here; usage_rows then credits every draft of the iteration."""
    by_hash = {c.get("hash"): c.get("number") for c in drafts if c.get("hash")}
    comp = next((r for r in reversed(rows) if r.get("step") == "compose" and not r.get("void")), {})
    out: dict[int, dict] = {}
    for g in rows:
        if g.get("step") != "gate" or g.get("void") or g.get("sync") is not None or not g.get("hash"):
            continue
        b, c = gate_bc(g)
        v = dict(b=b, c=c, accepted=bool(g.get("accepted")))
        if g["hash"] in by_hash:
            out[by_hash[g["hash"]]] = dict(v, via="draft")
        elif comp.get("decision") == "adopt" and comp.get("adopted") is not None:
            out[comp["adopted"]] = dict(v, via="adopted")
        elif comp.get("decision") == "merge":
            for k in comp.get("merged") or []:
                out[k] = dict(v, via="merged")
    return out


def mechanisms(loop: Path, iters: set[int] | None = None) -> list[dict]:
    """One dict per mechanism (module docstring): hypotheses linked by the Manager's `repeats`, their probe and draft
    flips, the gate's verdict on their drafts. Mechanisms with no flip and no gate verdict are left out. `flips` is
    keyed by flip_key(incumbent hash of the iteration, label): a label flipped under one incumbent says nothing about
    its round under another. An iteration without a known incumbent hash contributes no flips (no evidence table
    could hold them)."""
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
                            _note_flips(flips, inc.get("hash"), p)
            all_drafts = [c for c in jload(loop / f"iter_{n:02d}" / "candidates.json", []) or [] if isinstance(c, dict)]
            verdicts = draft_verdicts(by_iter[n], all_drafts)
            cands = [c for c in all_drafts if (n, c.get("hypothesis")) in ids]
            for c in cands:
                _note_flips(flips, inc.get("hash"), c)
                drafts.append(
                    dict(iter=n, number=c.get("number"), hash=c.get("hash"), mechanism=c.get("mechanism") or "")
                )
                if c.get("number") in verdicts:
                    gate = verdicts[c["number"]]
        if not flips and gate is None:
            continue
        first = members[0]
        # `key` names the entry: a hypothesis id is unique within its iteration only (the Manager links `NN:id`)
        out.append(dict(slug=first[1], key=f"i{first[0]:02d}_{first[1]}", ids=members, claims=[hyps[m].get("claim", "") for m in members],
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
    version the flip was measured against (the situation before the change), with `flip` looked up by the row's
    (version, label)."""
    versions = set(mech["incumbent"].values())
    tasks = {r["task"] for r in rows if flip_key(r.get("version"), r["label"]) in mech["flips"]}
    out = []
    for r in rows:
        if r["task"] in tasks and r.get("version") in versions:
            out.append(dict(r, flip=mech["flips"].get(flip_key(r["version"], r["label"]), "unchanged")))
    return out


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def rank_columns(table: list[dict], allowed: set[str] | frozenset[str] | None = None) -> list[dict]:
    """Per column (every dotted column, or those in `allowed`), how differently the flipped and the unchanged rounds are distributed, as a score in [0, 1]:
    numeric columns d / (d + IQR) with d the difference of the group medians and IQR the pooled interquartile range
    (with each group's [p10, p90]); categorical ones the share of the flipped rounds' majority category minus its
    share among the unchanged rounds, floored at 0 (with counts). A column with no unchanged rounds scores 0.0:
    there is nothing to separate from. Best first, ties by column name."""
    flipped = [r for r in table if r["flip"] != "unchanged"]
    rest = [r for r in table if r["flip"] == "unchanged"]
    cols = {k for r in table for k in r if (k in allowed if allowed is not None else "." in k)}
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
        g = mech["gate"]
        lines.append(
            f"- gate: b {g['b']} c {g['c']} accepted {g['accepted']}" + (f" ({g['via']})" if g.get("via") else "")
        )
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
    """Per object: the triggered rounds and how many of them the mechanism fixed or broke (flips keyed by flip_key)."""
    out: dict[str, dict] = {}
    for r in triggered:
        o = out.setdefault(object_of(r["task"]), dict(triggered=0, fixed=0, broken=0))
        o["triggered"] += 1
        f = flips.get(flip_key(r.get("version"), r["label"]))
        if f in ("fixed", "broken"):
            o[f] += 1
    return out


GENERALITIES = ("multi_object", "single_object", "contested")  # generality_of; "drop" keeps no entry


def generality_of(bo: dict[str, dict]) -> tuple[str, str]:
    """Plan section 4.1: how far the per-object evidence reaches, with the reason. `multi_object`: two or more
    objects, none net negative, at least one net positive; `single_object`: one object in evidence, net positive;
    `contested`: some object is net negative (after narrowing failed to exclude it); `drop`: too few flipped rounds
    or no object net positive. Says nothing about other task families: that is the entry's `tag`."""
    flipped = sum(o["fixed"] + o["broken"] for o in bo.values())
    if flipped < MIN_FLIPS:
        return "drop", f"{flipped} triggered flipped round(s), fewer than {MIN_FLIPS}"
    net = {k: o["fixed"] - o["broken"] for k, o in bo.items()}
    negative = sorted(k for k, v in net.items() if v < 0)
    positive = sorted(k for k, v in net.items() if v > 0)
    if not positive:
        return "drop", "no object is net positive"
    if negative:
        return "contested", f"net negative on {', '.join(negative)}; positive on {', '.join(positive)}"
    if len(bo) >= 2 and positive:
        return "multi_object", f"net non-negative on {len(bo)} objects, positive on {', '.join(positive)}"
    if len(bo) == 1 and positive:
        return "single_object", f"positive on one object ({positive[0]})"
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


CLAUSE = {"type": "object", "required": ["feature", "op", "value"],
          "properties": {"feature": {"type": "string"}, "op": {"enum": sorted(RC.OPS)},
                         "value": {"type": ["number", "string"]}}}  # fmt: skip
SCHEMA = {
    "type": "object",
    "properties": {
        "condition": {"type": "object", "required": ["all"],
                      "properties": {"all": {"type": "array", "items": CLAUSE}, "any": {"type": "array", "items": CLAUSE}}},
        "signature": {"type": "object", "required": ["reads_as"],
                      "properties": {"first_bad_kind": {"type": "string"}, "first_bad_edge": {"type": "string"},
                                     "failure_class": {"type": "array", "items": {"type": "string"}},
                                     "reads_as": {"type": "string"}}},
        "repair": {"type": "object", "required": ["works", "does_not", "where"],
                   "properties": {"works": {"type": "string"}, "does_not": {"type": "array", "items": {"type": "string"}},
                                  "where": {"type": "string"},
                                  "parameters_touched": {"type": "array", "items": {"type": "string"}}}},
        "decline": {"type": "string"},
    },
}  # fmt: skip


def names_from_meta(meta: dict) -> set[str]:
    """Names an entry may not contain: the loop's rule for a scene (records.object_names_from)."""
    return RC.object_names_from((meta or {}).get("objects") or {})


def _reflect_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rsi_reflect", HERE / "reflect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LEAK = re.compile(r"_hidden|rsi_kit|scene_rsi_20261016/kit/")  # the real leak targets (cf. loop.leak_patterns)


def consolidate(
    mech: dict,
    table: list[dict],
    ranked: list[dict],
    ctx: Path,
    scene_note: str,
    stub: dict | None = None,
    problems: list[str] | None = None,
) -> tuple[dict, dict]:
    """One `claude -p` call per mechanism in an isolated context directory (reflect.py's way); `stub` skips it."""
    ctx.mkdir(parents=True, exist_ok=True)
    (ctx / "evidence.md").write_text(render_evidence(mech, table, ranked))
    (ctx / "mechanism.md").write_text("\n\n".join(f"## iteration {d['iter']} draft {d['number']} ({d['hash']})\n\n{d['mechanism']}"
                                                  for d in mech["drafts"]) or "no draft was built\n")  # fmt: skip
    (ctx / "columns.txt").write_text("".join(f"{c['col']} {c['kind']}\n" for c in ranked))
    if problems:
        (ctx / "problems.txt").write_text("\n".join(problems) + "\n")
    if stub is not None:
        return stub, dict(cost_usd=0.0, seconds=0, session="stub", outside=[])
    R = _reflect_module()
    prompt = (
        (HERE / "prompts" / "consolidator.md")
        .read_text()
        .replace("{CONTEXT_DIR}", str(ctx))
        .replace("{SCENE_NOTE}", scene_note)
    )
    retry = ("A previous attempt was rejected by the driver's checks for the reasons in problems.txt; "
             "write an entry that passes them.\n\n" if problems else "")  # fmt: skip
    prompt = prompt.replace("{PROBLEMS}\n\n", retry).replace("{PROBLEMS}", retry)
    session = str(uuid.uuid4())
    cmd = ["claude", "-p", "--model", MODEL, "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
           "--settings", json.dumps({"disableAllHooks": True}), "--strict-mcp-config", "--effort", "high",
           "--permission-mode", "bypassPermissions", "--disallowedTools", "Edit,Write,NotebookEdit,Agent,WebFetch,WebSearch",
           "--session-id", session]  # fmt: skip
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_SSE")}
    env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY="1", CLAUDE_CODE_DISABLE_CLAUDE_MDS="1", TMPDIR=str(ctx.parent))
    t0 = time.time()
    try:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=ctx, env=env, timeout=1800)
        err = f"rc {r.returncode}: {(r.stderr or '')[-200:]}"
        out = json.loads(r.stdout) if r.returncode == 0 else None
    except subprocess.TimeoutExpired:
        out, err = None, "timeout after 1800 s"
    except ValueError:  # JSONDecodeError
        out = None
    if out is None:
        return {"decline": f"call failed: {err}"}, dict(
            cost_usd=None, seconds=round(time.time() - t0), session=session, outside=[]
        )
    so = out.get("structured_output") or {}
    named = set()
    results = f"/{session}/tool-results/"
    for p in Path.home().glob(f".claude/projects/*/{session}.jsonl"):
        named |= {
            m
            for m in R.PATH.findall(p.read_text(errors="replace"))
            if not m.startswith((str(ctx.parent), str(Path.home() / ".claude/shell-snapshots")))
            and not (".claude/projects/" in m and results in m)
        }
    leaked = next((m for m in sorted(named) if LEAK.search(m)), None)
    if leaked:
        so = {"decline": f"void: the session reached {leaked}"}
    return so, dict(
        cost_usd=out.get("total_cost_usd"), seconds=round(time.time() - t0), session=session, outside=sorted(named)[:50]
    )


def _texts(so: dict) -> list[str]:
    sig, rep = so.get("signature") or {}, so.get("repair") or {}
    return [str(sig.get("reads_as", "")), str(rep.get("works", "")), str(rep.get("where", "")),
            *map(str, rep.get("does_not") or []), *map(str, rep.get("parameters_touched") or [])]  # fmt: skip


def validate(so: dict, table: list[dict], flips: dict[str, str], names: set[str]) -> tuple[dict | None, list[str]]:
    """Plan section 3.4: syntax against the table's columns, only the columns of section 3.2
    (records.CONDITION_COLUMNS), no object names, the condition separates (p <= SEPARATION_P) and triggers at least
    MIN_FLIPS flipped rounds; then generality (4.1) after narrowing (4.2)."""
    problems: list[str] = []
    if so.get("decline"):
        return None, [f"declined: {so['decline']}"]
    cond = so.get("condition") or {}
    cols = {k for r in table for k in r}
    for c in (cond.get("all") or []) + (cond.get("any") or []):
        if c.get("feature") not in RC.CONDITION_COLUMNS:
            problems.append(f"column {c.get('feature')!r} is not a condition column (plan section 3.2)")
        elif c.get("feature") not in cols:
            problems.append(f"unknown column {c.get('feature')!r}")
        if c.get("op") not in RC.OPS:
            problems.append(f"unknown op {c.get('op')!r}")
    if not cond.get("all"):
        problems.append("condition.all is empty")
    texts = _texts(so) + [
        c["value"] for c in (cond.get("all") or []) + (cond.get("any") or []) if isinstance(c.get("value"), str)
    ]
    for t in texts:
        for n in sorted(names):
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", t):
                problems.append(f"names an object: {n}")
    if problems:
        return None, sorted(set(problems))
    cond = narrow(cond, table, flips)
    sep = separation(cond, table)
    if sep["flipped_in"] < MIN_FLIPS:
        problems.append(f"triggers {sep['flipped_in']} flipped round(s), fewer than {MIN_FLIPS}")
    if len(table) >= RC.MIN_REFUTE and sep["p"] > SEPARATION_P:  # a smaller table cannot reach the threshold
        problems.append(f"does not separate flipped from unchanged rounds (p {sep['p']})")
    if problems:
        return None, problems
    bo = by_object(trigger(cond, table), flips)
    generality, why = generality_of(bo)
    if generality == "drop":
        return None, [f"dropped: {why}"]
    entry = dict(condition=cond, signature=so.get("signature") or {}, repair=so.get("repair") or {},
                 evidence=dict(by_object=bo, separation=sep,
                               diag_rounds=sorted(r["label"] for r in trigger(cond, table) if r["flip"] != "unchanged"),
                               counterexamples=sorted(r["label"] for r in table if r["flip"] == "broken")),
                 generality=generality, generality_why=why)  # fmt: skip
    return entry, []


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60] or "entry"


def _stub_output(table: list[dict]) -> dict:
    """A test double: a condition over the flipped rounds' first separating categorical column, else an always-true
    numeric clause."""
    ranked = rank_columns(table, RC.CONDITION_COLUMNS)
    cat = next((c for c in ranked if c["kind"] == "cat"), None)
    if cat:
        top = max(cat["flipped"], key=cat["flipped"].get)
        cond = {"all": [dict(feature=cat["col"], op="==", value=top)]}
    else:
        num = ranked[0]
        cond = {"all": [dict(feature=num["col"], op=">=", value=num["flipped"][0])]}
    return dict(condition=cond, signature=dict(reads_as="stub"), repair=dict(works="stub", does_not=[], where="stub"))


def build(
    sandbox: Path,
    family: str,
    out: Path,
    iters: set[int] | None = None,
    stub: bool = False,
    ctx_root: Path | None = None,
    tag: str | None = None,
) -> dict:
    """Plan section 3: mechanisms -> evidence tables -> the Consolidator -> validation, generality -> entries tagged `tag`
    (default tag_of) replacing the family's earlier ones in `out`/knowledge.jsonl, the build log build/<family>.md and
    INDEX.md. `ctx_root` holds the Consolidator's context directories."""
    loop = sandbox / VISIBLE / family / "loop"
    meta = RC.scene_meta(sandbox, family) or {}
    names = names_from_meta(meta)
    fam = jload(sandbox / "scripts" / "rsi" / "families" / f"{family}.json", {}) or {}
    scene_note = fam.get("scene_note") or "in a simulated scene"
    rows = diag_rows(sandbox, family)
    if ctx_root is None:
        ctx_root = (
            Path(os.environ.get("LOCOMANI_WORKSPACE", str(Path.home() / "workspace")))
            / "tmp"
            / "rsi_knowledge"
            / family
        )
    tag = tag or tag_of(sandbox, family)
    new: list[dict] = []
    log = [f"# build {family} ({tag}) from {sandbox} ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    kept, cost = 0, 0.0
    # a mechanism whose evidence is as at the last build keeps that build's outcome without a model call: an
    # alternating run (scripts/rsi/alternate.py) rebuilds a family after each of its iterations
    cache_path = out / "build" / f"{family}.cache.json"
    cache, fresh_cache = jload(cache_path, {}) or {}, {}
    for m in mechanisms(loop, iters):
        table = evidence_table(m, rows)
        if not table:
            log.append(f"- {m['key']}: no diagnostic rows under its incumbent version -- skipped")
            continue
        ranked = rank_columns(table, RC.CONDITION_COLUMNS)  # the Consolidator sees only section 3.2's columns
        key = evidence_key(m, table, ranked, scene_note, tag, stub)
        if key in cache:
            fresh_cache[key] = hit = cache[key]
            if hit.get("entry"):
                hit["entry"] = modern_entry(hit["entry"])  # a cache written before the rename
                new.append(hit["entry"])
                kept += 1
                log.append(f"- {m['key']}: {hit['entry']['generality']} (as at the last build: same evidence)")
            else:
                log.append(f"- {m['key']}: dropped as at the last build (same evidence) -- {hit.get('why', '')}")
            continue
        entry, problems, outside = None, [], []
        for attempt in range(2):
            so, acct = consolidate(
                m,
                table,
                ranked,
                ctx_root / slugify(m["key"]) / str(attempt),
                scene_note,
                stub=_stub_output(table) if stub else None,
                problems=problems or None,
            )
            cost += float(acct.get("cost_usd") or 0.0)
            outside = acct.get("outside") or []
            entry, problems = validate(so, table, m["flips"], names)
            if entry is not None or problems and problems[0].startswith("declined"):
                break
        note = f"; {len(outside)} paths outside the context dir" if outside else ""
        if entry is None:
            log.append(f"- {m['key']}: dropped -- " + "; ".join(problems) + note)
            if not (problems and problems[0].startswith(("declined: call failed", "declined: void"))):
                fresh_cache[key] = dict(entry=None, why="; ".join(problems))  # a failed or voided call: next build
            continue
        entry.update(id=f"{family}/{slugify(m['key'])}", version=1, status="candidate", tag=tag,
                     source=dict(family=family, sandbox=sandbox.name, iters=sorted({n for n, _ in m["ids"]}),
                                 mechanism_ids=[h for _, h in m["ids"]], gate=m["gate"],
                                 built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                     audit=dict(names_checked=True, hidden_checked=True))  # fmt: skip
        new.append(entry)
        fresh_cache[key] = dict(entry=entry)
        kept += 1
        log.append(
            f"- {m['key']}: {entry['generality']} ({entry['generality_why']}); separation p {entry['evidence']['separation']['p']}"
        )
    log.append(f"\n{kept} entries, model cost {cost:.2f} USD")
    # a rebuild replaces the family's entries (a dropped mechanism keeps none) and leaves every other family's
    save_entries(out, [e for e in entries(out) if source_family(e) != family] + new)
    # how this family's hypotheses that cited entries ended (user 2026-11-18): the entries' standing, no model call
    cited = update_usage(out, str(sandbox), family, usage_rows(loop, family, str(sandbox)))
    log.append(f"{cited} citation(s) of library entries in this family's ledger recorded in {USAGE}")
    (out / "build").mkdir(parents=True, exist_ok=True)
    (out / "build" / f"{family}.md").write_text("\n".join(log) + "\n")
    _write_atomic(cache_path, json.dumps(fresh_cache, ensure_ascii=False) + "\n")
    index(out)
    return dict(entries=kept, cost_usd=round(cost, 2), tag=tag, citations=cited)


def evidence_key(m: dict, table: list[dict], ranked: list[dict], scene_note: str, tag: str, stub: bool) -> str:
    """What a mechanism's entry is built from -- its hypotheses, flips, gate verdict, evidence table, the scene note,
    the tag --, hashed: the same key, the same entry."""
    import hashlib

    blob = json.dumps(dict(mech={k: m.get(k) for k in ("key", "slug", "ids", "claims", "insteads", "flips", "gate")},
                           table=render_evidence(m, table, ranked), scene_note=scene_note, tag=tag, stub=stub),
                      sort_keys=True, default=str)  # fmt: skip
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def source_family(e: dict) -> str:
    """The family an entry was built from (`source.family`; an older entry without it: its id's prefix)."""
    return (e.get("source") or {}).get("family") or str(e.get("id", "")).split("/", 1)[0]


def modern_entry(e: dict) -> dict:
    """An entry as the current code writes it: one written before 2026-11-23 has `scope` shared / local instead of
    `generality`; shared was multi_object, local was single_object ("positive on one object") or contested."""
    if "generality" in e or "scope" not in e:
        return e
    e = dict(e)
    scope, why = e.pop("scope"), e.pop("scope_why", "")
    if scope == "shared":
        e["generality"] = "multi_object"
    elif scope == "local":
        e["generality"] = "single_object" if str(why).startswith("positive on one object") else "contested"
    else:
        e["generality"] = scope
    e["generality_why"] = why
    return e


def entries(out: Path = OUT) -> list[dict]:
    """Every entry of the library at `out` (its knowledge.jsonl), in file order, as the current code writes them
    (modern_entry); none when it has no file."""
    return [modern_entry(e) for e in jl(Path(out) / LIBRARY)]


def save_entries(out: Path, ents: list[dict]) -> None:
    """knowledge.jsonl := `ents`, sorted by tag then id, written atomically."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ents = sorted(ents, key=lambda e: (e.get("tag") or "", e.get("id") or ""))
    _write_atomic(out / LIBRARY, "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ents))


def usage_rows(loop: Path, family: str, run: str) -> list[dict]:
    """Every citation of a library entry in one family's loop (user 2026-11-18): one row per (hypothesis, entry id in
    its `knowledge_refs`), with how the hypothesis ended. `status`: the Prober's (verified, refuted, mixed, hypothesis);
    `picked`: the Manager chose it to build; `accepted`: `draft` when a draft of it passed the gate (alone, adopted by
    the Composer, or as the second shot), `merged` when a merged candidate that passed kept its draft (the Composer's
    `merged` list; an older ledger without it credits every draft of the iteration; draft_verdicts);
    `outcome`: helped (verified or accepted), failed (refuted, not accepted) or unclear. `run`: the sandbox, so two runs
    of one family do not overwrite each other."""
    rows = jl(loop / "ledger.jsonl")
    out = []
    for it in sorted(p for p in loop.glob("iter_*") if p.is_dir() and p.name[5:].isdigit()):
        n = int(it.name[5:])
        hyps = [h for h in (jload(it / "hypotheses.json", {}) or {}).get("hypotheses") or [] if isinstance(h, dict)]
        cited = [h for h in hyps if h.get("id") and any(isinstance(x, str) for x in h.get("knowledge_refs") or [])]
        if not cited:
            continue
        mine = [r for r in rows if r.get("iter") == n]
        probe = {p.get("id"): p for r in mine if r.get("step") == "probe" for p in r.get("results") or []}
        picked = set()
        for r in mine:
            if r.get("step") == "manager" and not r.get("void"):
                picked |= {x for x in [r.get("hypothesis"), *(r.get("hypotheses") or [])] if x}
        drafts = [c for c in jload(it / "candidates.json", []) or [] if isinstance(c, dict)]
        hyp_of = {c.get("number"): c.get("hypothesis") for c in drafts}
        credit: dict[str, str] = {}
        for k, v in draft_verdicts(mine, drafts).items():
            if v["accepted"] and hyp_of.get(k):
                credit[hyp_of[k]] = "merged" if v["via"] == "merged" else "draft"
        comp = next((r for r in reversed(mine) if r.get("step") == "compose" and not r.get("void")), {})
        if comp.get("decision") == "merge" and not comp.get("merged") and any(
            g.get("step") == "gate" and g.get("accepted") and not g.get("void") and g.get("sync") is None
            and g.get("hash") not in {c.get("hash") for c in drafts}
            for g in mine
        ):  # an older ledger's merge, which drafts went in unknown: every draft of the iteration  # fmt: skip
            for c in drafts:
                credit.setdefault(c.get("hypothesis"), "merged")
        for h in cited:
            status = (probe.get(h["id"]) or {}).get("status") or h.get("status") or "hypothesis"
            acc = credit.get(h["id"])
            outcome = "helped" if status == "verified" or acc else "failed" if status == "refuted" else "unclear"
            for ref in dict.fromkeys(x for x in h["knowledge_refs"] if isinstance(x, str)):
                out.append(dict(entry=ref, run=run, family=family, iter=n, hypothesis=h["id"], status=status,
                                picked=h["id"] in picked, accepted=acc, outcome=outcome))  # fmt: skip
    return out


def update_usage(lib: Path, run: str, family: str, rows: list[dict]) -> int:
    """usage.jsonl := its rows of other runs or families, plus `rows` (this run's family, recounted from its whole
    ledger: nothing is counted twice). Returns the rows written for this run and family."""
    keep = [r for r in jl(Path(lib) / USAGE) if (r.get("run"), r.get("family")) != (run, family)]
    Path(lib).mkdir(parents=True, exist_ok=True)
    allr = sorted(keep + rows, key=lambda r: (r["entry"], r["family"], r["run"], r["iter"], r["hypothesis"]))
    _write_atomic(Path(lib) / USAGE, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in allr))
    return len(rows)


def entry_stats(usage: list[dict]) -> dict[str, dict]:
    """{entry id: {uses, helped, failed, unclear, accepted}} over every citation in usage.jsonl."""
    out: dict[str, dict] = {}
    for r in usage:
        s = out.setdefault(r["entry"], dict(uses=0, helped=0, failed=0, unclear=0, accepted=0))
        s["uses"] += 1
        s[r.get("outcome", "unclear")] = s.get(r.get("outcome", "unclear"), 0) + 1
        s["accepted"] += bool(r.get("accepted"))
    return out


def status_of(st: dict | None) -> str:
    """An entry's standing from its citations (user 2026-11-18): `tested` -- helped at least once and at least as often
    as it failed; `doubtful` -- failed at least twice and more often than it helped; else `candidate`. `resolved` is
    the reader's (render_library): it depends on whose incumbent reads it."""
    if not st:
        return "candidate"
    if st["helped"] >= 1 and st["helped"] >= st["failed"]:
        return "tested"
    if st["failed"] >= 2 and st["failed"] > st["helped"]:
        return "doubtful"
    return "candidate"


def tag_of(sandbox: Path, family: str) -> str:
    """The kind of task the family's entries come from: `single` when every task's object is a one-hand object of the
    scene file (`semantic.graspable`), `bimanual` when every one is a two-hand object, else `mixed`. A family file's
    `knowledge_tag` names it outright."""
    fam = jload(Path(sandbox) / "scripts" / "rsi" / "families" / f"{family}.json", {}) or {}
    if fam.get("knowledge_tag"):
        return str(fam["knowledge_tag"])
    meta = RC.scene_meta(Path(sandbox), family) or {}
    tasks_file = Path(sandbox) / str(fam.get("tasks_file") or "")
    rows = {r.get("id"): r for r in jl(tasks_file)} if fam.get("tasks_file") else {}
    kinds = set()
    for t in fam.get("tasks") or []:
        goal = (rows.get(t) or {}).get("goal") or []
        m = re.match(r"\s*\w+\(\s*([^,\s)]+)", goal[0]) if goal else None
        obj = (meta.get("objects") or {}).get(m.group(1)) if m else None
        kinds.add(((obj or {}).get("semantic") or {}).get("graspable"))
    return kinds.pop() if kinds in ({"single"}, {"bimanual"}) else "mixed"


def index(out: Path = OUT) -> None:
    """INDEX.md, the people's table of the library (the roles read KNOWLEDGE.md)."""
    rows = ["# Knowledge library", "",
            f"One entry per mechanism, in {LIBRARY} (scripts/rsi/knowledge.py build). The roles read {ROLE_FILE}, the "
            "whole library marked with the incumbent's rounds, not this index.", "",
            "| id | tag | generality | status | cited (helped / no effect / unclear) | condition | objects | reads as |",
            "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    stats = entry_stats(jl(Path(out) / USAGE))
    for e in entries(out):
        cond = " and ".join(f"{c['feature']} {c['op']} {c['value']}" for c in e["condition"].get("all", []))
        st = stats.get(e["id"])
        cites = f"{st['uses']} ({st['helped']} / {st['failed']} / {st['unclear']})" if st else "0"
        rows.append(
            f"| {e['id']} | {e.get('tag', '')} | {e['generality']} | {status_of(st)} | {cites} | {cond} | {len(e['evidence']['by_object'])} | {e['signature'].get('reads_as', '')} |"
        )
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "INDEX.md").write_text("\n".join(rows) + "\n")


def hits(ents: list[dict], rows: list[dict]) -> list[dict]:
    """Plan section 5.2: each entry whose condition holds on at least one round, with the rounds and how many failed;
    entries that hit only successes go last."""
    out = []
    for e in ents:
        trig = trigger(e["condition"], rows)
        if not trig:
            continue
        rounds = sorted((r["label"], bool(r["ok"])) for r in trig)
        out.append(dict(id=e["id"], generality=e["generality"], version=e.get("version", 1), status=e.get("status", ""),
                        rounds=rounds, failed=sum(not ok for _, ok in rounds)))  # fmt: skip
    return sorted(out, key=lambda h: (h["failed"] == 0, -h["failed"], h["id"]))


def _cond_text(c: dict) -> str:
    parts = [f"{x['feature']} {x['op']} {x['value']}" for x in c.get("all") or []]
    if c.get("any"):
        parts.append("(" + " or ".join(f"{x['feature']} {x['op']} {x['value']}" for x in c["any"]) + ")")
    parts += [f"{x['feature']} {x['op']} {x['value']}" for x in c.get("narrowed_by") or []]
    return " and ".join(parts)


def _uses_text(st: dict | None) -> str:
    if not st:
        return "Not cited yet"
    acc = f" ({st['accepted']} passed the gate)" if st["accepted"] else ""
    return (f"Cited {st['uses']} time(s): helped {st['helped']}{acc}, no effect {st['failed']}, "
            f"unclear {st['unclear']}")  # fmt: skip


def _entry_lines(e: dict, h: dict | None, status: str = "candidate", st: dict | None = None) -> list[str]:
    """One entry of KNOWLEDGE.md, in full. Object names from the evidence never reach it (no `by_object`, no
    `generality_why`): only the reading family's own round labels."""
    src = e.get("source") or {}
    head = f"### {e['id']}  ({e['generality']}, v{e.get('version', 1)}, {status}; from {source_family(e)}"
    head += f" iterations {src['iters']})" if src.get("iters") else ")"
    if h is None:
        mark = "This incumbent: the condition holds on none of its diagnostic rounds"
    else:
        failed = [l for l, ok in h["rounds"] if not ok]
        ok = [l for l, good in h["rounds"] if good]
        mark = f"This incumbent: the condition holds on {len(failed)} failed round(s)" + (
            f": {', '.join(failed)}" if failed else ""
        )
        if ok:
            mark += f"; and on {len(ok)} round(s) it succeeds on: {', '.join(ok)}"
    lines = [head, mark, f"Condition: {_cond_text(e['condition'])}", f"Reads as: {e['signature'].get('reads_as', '')}",
             f"Worked: {e['repair'].get('works', '')} (where: {e['repair'].get('where', '')})"]  # fmt: skip
    if e["repair"].get("does_not"):
        lines.append("Did not work: " + "; ".join(e["repair"]["does_not"]))
    lines.append(_uses_text(st))
    return lines + [""]


def _short_line(e: dict, status: str, st: dict | None, h: dict | None) -> str:
    """A doubtful or resolved entry of KNOWLEDGE.md: one line."""
    why = (f"{_uses_text(st).lower()}; this way did not work" if status == "doubtful"
           else f"its condition holds on {len((h or {}).get('rounds') or [])} round(s), all of which this incumbent already succeeds on")  # fmt: skip
    return f"- {e['id']} ({status}): {e['signature'].get('reads_as', '')} -- {why}"


def reader_status(e: dict, h: dict | None, st: dict | None) -> str:
    """The entry's standing for this reader: `resolved` when its condition holds on at least one of this incumbent's
    diagnostic rounds and every one of them succeeds; else status_of its citations."""
    if h and h.get("rounds") and h.get("failed", 0) == 0:
        return "resolved"
    return status_of(st)


def _sync_line(s: dict) -> str:
    """One code sync of an alternating run (scripts/rsi/alternate.py `sync.jsonl`)."""
    head = f"- iteration {s.get('iter')}: {s.get('from')} -> {s.get('to')}: "
    if s.get("skipped"):  # a skipped sync may still have met a conflict: the roles should see where they differ
        kept = f"; kept {s.get('to')}'s side of " + "; ".join(s["conflicts"]) if s.get("conflicts") else ""
        return head + f"nothing merged ({s['skipped']}){kept}"
    files = ", ".join(s.get("merged", []) + s.get("copied", [])) or "none"
    if s.get("deleted"):
        files += "; deleted " + ", ".join(s["deleted"])
    text = head + f"merged {files}"
    if s.get("conflicts"):
        text += f"; kept {s.get('to')}'s side of " + "; ".join(s["conflicts"])
    r = s.get("result") or {}
    if "b" in r:
        text += f"; on {s.get('to')}'s acceptance pool {r['b']} wins, {r['c']} losses"
    return (
        text
        + f" -> {'accepted' if s.get('accepted') else 'rejected'}"
        + (f" ({s['reason']})" if s.get("reason") and not s.get("accepted") else "")
    )


def render_library(
    ents: list[dict], hs: list[dict], syncs: list[dict] | None = None, usage: list[dict] | None = None
) -> str:
    """KNOWLEDGE.md (user 2026-11-18): every entry, one section per tag, each marked with the incumbent's diagnostic
    rounds its condition holds on (hits) and with how its citations ended (usage); within a section tested, then
    candidate entries in full, then doubtful and resolved ones a line each; then the code syncs between the families
    of an alternating run."""
    by_hit = {h["id"]: h for h in hs}
    stats = entry_stats(usage or [])
    lines = ["# Knowledge library: every entry, marked with this incumbent's diagnostic rounds", "",
             "Entries (condition -> failure signature -> repair) the loops of this family and of others have learned, "
             "tagged by the kind of task: one-hand pick (single) and two-hand pick (bimanual). Read them all. What "
             "worked on other objects or another task is a hypothesis here; this family's rounds decide. Under each "
             "entry: the rounds of this incumbent its condition holds on, and how the hypotheses that cited it ended. "
             "Tested entries helped where they were cited; doubtful ones were cited and did not help; resolved ones "
             "hold only where this incumbent already succeeds. Cite an entry you use by its id in `knowledge_refs`.",
             ""]  # fmt: skip
    tags = sorted(
        {e.get("tag") or "untagged" for e in ents},
        key=lambda t: (t not in TAGS, list(TAGS).index(t) if t in TAGS else 0, t),
    )
    for t in tags:
        sec = [e for e in ents if (e.get("tag") or "untagged") == t]
        st_of = {e["id"]: reader_status(e, by_hit.get(e["id"]), stats.get(e["id"])) for e in sec}
        sec.sort(
            key=lambda e: (STATUS_ORDER.index(st_of[e["id"]]), -(by_hit.get(e["id"]) or {}).get("failed", 0), e["id"])
        )
        lines += [f"## {TAGS.get(t, t).capitalize()} (tag {t}): {len(sec)} entr{'y' if len(sec) == 1 else 'ies'}", ""]
        short = []
        for e in sec:
            status, h, st = st_of[e["id"]], by_hit.get(e["id"]), stats.get(e["id"])
            if status in ("doubtful", "resolved"):
                short.append(_short_line(e, status, st, h))
            else:
                lines += _entry_lines(e, h, status, st)
        if short:
            lines += ["Doubtful or resolved here (one line each):", *short, ""]
    if syncs:
        lines += ["## Code syncs between the families", "",
                  "After each family's iteration its incumbent's skills package was merged into the other family's "
                  "and kept there only if it lost no more starts than it won on that family's acceptance pool.", ""]  # fmt: skip
        lines += [_sync_line(x) for x in syncs] + [""]
    return "\n".join(lines)


def write_hits(root: Path, rounds_dir: Path, out_dir: Path, family: str, pool: Path | None = None) -> int:
    """Evaluate the library under `root` on the rounds in `rounds_dir` (start files from `pool`, by default the
    pool_diag beside the rounds' diag directory); write <out_dir>/knowledge_hits.json and the whole library, marked,
    as <root>/scripts/rsi/knowledge/KNOWLEDGE.md. Returns the entries hitting a failed round. No entry and no sync:
    nothing written."""
    lib = root / "scripts" / "rsi" / "knowledge"
    ents = entries(lib) if lib.is_dir() else []
    syncs = jl(lib / SYNCS) if lib.is_dir() else []
    if not ents and not syncs:
        return 0
    hs: list[dict] = []
    if ents:
        pool = pool or rounds_dir.parents[1] / "pool_diag"
        rows = RC.scan([(rounds_dir, family), (pool, family)], RC.machine(), RC.scene_meta(root, family))
        hs = hits(ents, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(out_dir / "knowledge_hits.json", json.dumps(hs, indent=1) + "\n")
    _write_atomic(lib / ROLE_FILE, render_library(ents, hs, syncs, jl(lib / USAGE)))
    return sum(h["failed"] > 0 for h in hs)


def _write_atomic(p: Path, text: str) -> None:
    """Write to <p>.tmp beside it, then rename: a reader never sees half a file."""
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def report(sandboxes: list[Path]) -> list[dict]:
    """Plan section 7.2: per replayed sandbox, the last iteration's Reviewer and Prober numbers."""
    out = []
    for sb in sandboxes:
        fams = (
            [d for d in sorted((sb / VISIBLE).iterdir()) if d.is_dir() and d.name not in ("body", "records")]
            if (sb / VISIBLE).exists()
            else []
        )
        if not fams:
            continue
        loop = fams[0] / "loop"
        es = jl(loop / "ledger.jsonl")
        rev = next((e for e in reversed(es) if e.get("step") == "review"), {})
        probe = next((e for e in reversed(es) if e.get("step") == "probe"), {})
        n = probe.get("iter", rev.get("iter"))
        if n is None:  # no review or probe row yet: nothing to compare
            continue
        res = probe.get("results") or []
        hyps = (jload(loop / f"iter_{n:02d}" / "hypotheses.json", {}) or {}).get("hypotheses") or []
        st = Counter(r.get("status") for r in res)
        out.append(dict(sandbox=sb.name, with_knowledge=(loop / f"iter_{n:02d}" / "knowledge_hits.json").exists(),
                        n_hyp=len(res), verified=st.get("verified", 0), refuted=st.get("refuted", 0),
                        mixed=st.get("mixed", 0), hypothesis=st.get("hypothesis", 0),
                        net_fixed=sum(len(r.get("fixed") or []) - len(r.get("broken") or []) for r in res),
                        pred_hit=sum(r.get("predicted_fixed_hit") or 0 for r in res),
                        pred_n=sum(r.get("predicted_fixed_n") or 0 for r in res),
                        cites=sum(bool(h.get("knowledge_refs")) for h in hyps if isinstance(h, dict)),
                        review_cost_usd=rev.get("cost_usd"), review_s=rev.get("seconds")))  # fmt: skip
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evidence")
    ev.add_argument("sandbox", type=Path)
    ev.add_argument("family")
    ev.add_argument("--iters", help="N-M: iterations whose evidence is used")
    ev.add_argument("--out", type=Path, help="write one .md per mechanism here instead of printing")
    b = sub.add_parser("build")
    b.add_argument("sandbox", type=Path)
    b.add_argument("family")
    b.add_argument("--iters")
    b.add_argument("--out", type=Path, default=OUT)
    b.add_argument("--ctx-root", type=Path, help="where the Consolidator's context directories go")
    b.add_argument("--stub", action="store_true", help="no model call: a test double writes the conditions")
    b.add_argument(
        "--tag", help="the entries' tag (default: tag_of -- single, bimanual or mixed by the tasks' objects)"
    )
    us = sub.add_parser("usage", help="record how this family's hypotheses that cited entries ended; no build")
    us.add_argument("sandbox", type=Path)
    us.add_argument("family")
    us.add_argument("--out", type=Path, default=OUT)
    ix = sub.add_parser("index")
    ix.add_argument("--out", type=Path, default=OUT)
    ev2 = sub.add_parser("eval")
    ev2.add_argument("root", type=Path, help="the sandbox (or repository) holding scripts/rsi/knowledge")
    ev2.add_argument("rounds", type=Path, help="the incumbent's diagnostic rounds directory")
    ev2.add_argument("family")
    ev2.add_argument("--out", type=Path, required=True)
    rp = sub.add_parser("report")
    rp.add_argument("sandboxes", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd in ("evidence", "build", "usage"):  # a wrong path read as "no mechanisms" and built an empty library
        ledger = a.sandbox / VISIBLE / a.family / "loop" / "ledger.jsonl"
        if not ledger.exists():
            ap.error(f"no ledger at {ledger}: the sandbox path or the family is wrong")
    if a.cmd == "report":
        rows = report([p.resolve() for p in a.sandboxes])
        keys = ["sandbox", "with_knowledge", "n_hyp", "verified", "refuted", "mixed", "hypothesis", "net_fixed", "pred_hit", "pred_n", "cites", "review_cost_usd", "review_s"]  # fmt: skip
        print("| " + " | ".join(keys) + " |")
        print("|" + "---|" * len(keys))
        for r in rows:
            print("| " + " | ".join(str(r.get(k)) for k in keys) + " |")
        return 0
    if a.cmd == "evidence":
        iters = None
        if a.iters:
            lo, _, hi = a.iters.partition("-")
            iters = set(range(int(lo), int(hi or lo) + 1))
        loop = a.sandbox / VISIBLE / a.family / "loop"
        rows = diag_rows(a.sandbox, a.family)
        for m in mechanisms(loop, iters):
            t = evidence_table(m, rows)
            md = render_evidence(m, t, rank_columns(t, RC.CONDITION_COLUMNS))
            if a.out:
                a.out.mkdir(parents=True, exist_ok=True)
                (a.out / f"{m['key']}.md").write_text(md)
            else:
                print(md)
        return 0
    if a.cmd == "build":
        iters = None
        if a.iters:
            lo, _, hi = a.iters.partition("-")
            iters = set(range(int(lo), int(hi or lo) + 1))
        print(json.dumps(build(a.sandbox.resolve(), a.family, a.out, iters, a.stub, a.ctx_root, a.tag)))
        return 0
    if a.cmd == "usage":
        sb = a.sandbox.resolve()
        n = update_usage(a.out, str(sb), a.family, usage_rows(sb / VISIBLE / a.family / "loop", a.family, str(sb)))
        index(a.out)
        print(json.dumps(dict(family=a.family, citations=n, out=str(a.out / USAGE))))
        return 0
    if a.cmd == "index":
        index(a.out)
        return 0
    if a.cmd == "eval":
        print(write_hits(a.root.resolve(), a.rounds.resolve(), a.out, a.family))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
