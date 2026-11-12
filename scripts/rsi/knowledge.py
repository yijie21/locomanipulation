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
    if not positive:
        return "drop", "no object is net positive"
    if negative:
        return "local", f"net negative on {', '.join(negative)}; positive on {', '.join(positive)}"
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
    """Plan section 3.4: syntax against the table's columns, no object names, the condition separates (p <=
    SEPARATION_P) and triggers at least MIN_FLIPS flipped rounds; then scope (4.1) after narrowing (4.2)."""
    problems: list[str] = []
    if so.get("decline"):
        return None, [f"declined: {so['decline']}"]
    cond = so.get("condition") or {}
    cols = {k for r in table for k in r if "." in k}
    for c in (cond.get("all") or []) + (cond.get("any") or []):
        if c.get("feature") not in cols:
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
    scope, why = scope_of(bo)
    if scope == "drop":
        return None, [f"dropped: {why}"]
    entry = dict(condition=cond, signature=so.get("signature") or {}, repair=so.get("repair") or {},
                 evidence=dict(by_object=bo, separation=sep,
                               diag_rounds=sorted(r["label"] for r in trigger(cond, table) if r["flip"] != "unchanged"),
                               counterexamples=sorted(r["label"] for r in table if r["flip"] == "broken")),
                 scope=scope, scope_why=why)  # fmt: skip
    return entry, []


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60] or "entry"


def _stub_output(table: list[dict]) -> dict:
    """A test double: a condition over the flipped rounds' first separating categorical column, else an always-true
    numeric clause."""
    ranked = rank_columns(table)
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
) -> dict:
    """Plan section 3: mechanisms -> evidence tables -> the Consolidator -> validation, scope -> entries,
    BUILD.md and INDEX.md under `out`. `ctx_root` holds the Consolidator's context directories."""
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
    dest = out / family
    dest.mkdir(parents=True, exist_ok=True)
    log = [f"# build {family} from {sandbox} ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    kept, cost = 0, 0.0
    for m in mechanisms(loop, iters):
        table = evidence_table(m, rows)
        if not table:
            log.append(f"- {m['slug']}: no diagnostic rows under its incumbent version -- skipped")
            continue
        ranked = rank_columns(table)
        entry, problems, outside = None, [], []
        for attempt in range(2):
            so, acct = consolidate(
                m,
                table,
                ranked,
                ctx_root / slugify(m["slug"]) / str(attempt),
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
            log.append(f"- {m['slug']}: dropped -- " + "; ".join(problems) + note)
            continue
        entry.update(id=f"{family}/{slugify(m['slug'])}", version=1, status="candidate",
                     source=dict(family=family, sandbox=sandbox.name, iters=sorted({n for n, _ in m["ids"]}),
                                 mechanism_ids=[h for _, h in m["ids"]], gate=m["gate"],
                                 built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                     audit=dict(names_checked=True, hidden_checked=True))  # fmt: skip
        (dest / f"{slugify(m['slug'])}.json").write_text(json.dumps(entry, indent=1, ensure_ascii=False) + "\n")
        kept += 1
        log.append(
            f"- {m['slug']}: {entry['scope']} ({entry['scope_why']}); separation p {entry['evidence']['separation']['p']}"
        )
    log.append(f"\n{kept} entries, model cost {cost:.2f} USD")
    (dest / "BUILD.md").write_text("\n".join(log) + "\n")
    index(out)
    return dict(entries=kept, cost_usd=round(cost, 2))


def entries(out: Path = OUT) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out.glob("*/*.json"))]


def index(out: Path = OUT) -> None:
    rows = ["# Knowledge library", "", "One entry per mechanism (scripts/rsi/knowledge.py build). Roles read HITS.md, not this index.", "",
            "| id | scope | status | condition | objects | reads as |", "|---|---|---|---|---|---|"]  # fmt: skip
    for e in entries(out):
        cond = " and ".join(f"{c['feature']} {c['op']} {c['value']}" for c in e["condition"].get("all", []))
        rows.append(
            f"| {e['id']} | {e['scope']} | {e['status']} | {cond} | {len(e['evidence']['by_object'])} | {e['signature'].get('reads_as', '')} |"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "INDEX.md").write_text("\n".join(rows) + "\n")


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
    ix = sub.add_parser("index")
    ix.add_argument("--out", type=Path, default=OUT)
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
    if a.cmd == "build":
        iters = None
        if a.iters:
            lo, _, hi = a.iters.partition("-")
            iters = set(range(int(lo), int(hi or lo) + 1))
        print(json.dumps(build(a.sandbox.resolve(), a.family, a.out, iters, a.stub, a.ctx_root)))
        return 0
    if a.cmd == "index":
        index(a.out)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
