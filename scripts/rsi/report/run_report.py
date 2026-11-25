"""One HTML page per finished RSI run (user 2026-11-25), in English, from the run's own records:

* Table 1, per family: every iteration's incumbent on the acceptance pool -- per object, wins and rate; the last row
  the whole pool.
* Table 2, per family: the same on the sealed pool (sealed.py; `--all` gives every iteration's incumbent, without it
  only the seed and the final version have rounds).
* Table 3: every iteration's steps -- wall time and model tokens (input, cache write, cache read, output) and cost;
  a subtotal per iteration, the run's total in the last row.
* Table 4: Table 3's subtotals alone -- per iteration each family's time, total tokens and cost side by side, then
  all families together; the run total in the last row.

    python3 scripts/rsi/report/run_report.py <run dir | sandbox> [--family F ...] [--out FILE]

A run directory is an alternate.py run (mode 3: alternate.json, sandboxes a/ and b/); a sandbox is one loop.py run's
checkout (modes 1 and 2). The page goes to <dir>/rsi_report.html unless --out says otherwise. Nothing is simulated
and no model is called. Sources: each family's loop ledger and its roles' *_raw.json (modelUsage; the ledger's own
`usage` when a raw file is missing), the hidden gate tree (its ledger, the seed's and every passed gate's acceptance
rounds), the hidden sealed rounds, alternate.json (the halves' knowledge build and sync), and the knowledge builds'
session transcripts under ~/.claude/projects (their tokens; the build events carry only the cost).

An iteration's incumbent is the last version accepted by then: its own gate, or in mode 3 the sync from the other
family in the same iteration. An iteration that accepted nothing keeps the previous incumbent (column marked "kept").
A candidate rejected at the gate's stage 1 ran only STAGE1 starts, so it has no per-object numbers of its own; the
notes under Table 1 give its stage-1 tally.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import re
from datetime import datetime
from pathlib import Path

OUT_REL = Path("outputs") / "scene_rsi_20261016"
HIDDEN_REL = Path("outputs") / "scene_rsi_20261016_hidden"
PROJECTS = Path.home() / ".claude" / "projects"
TOKENS = ("input", "cache_write", "cache_read", "output")
# ledger usage keys and *_raw.json modelUsage keys, in TOKENS order
LEDGER_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
RAW_KEYS = ("inputTokens", "cacheCreationInputTokens", "cacheReadInputTokens", "outputTokens")
ROLE_STEP = {"review": "Reviewer", "manager": "Manager", "engineer": "Engineers", "compose": "Composer"}
PROBE_STEPS = {"probe", "design", "park"}
GATE_STEPS = {"gate", "second_shot"}


# -- records ------------------------------------------------------------------------------------------------------------


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def zero() -> dict:
    return dict.fromkeys(TOKENS, 0)


def add(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in TOKENS}


def pool_sum(per: dict) -> tuple[int, int]:
    """(wins, rounds) of a tally."""
    return sum(a for a, _ in per.values()), sum(n for _, n in per.values())


def tally(rounds: Path) -> dict:
    """{task: [wins, rounds]} of a rounds directory (batch.outcomes' files: every *.json not starting with "_")."""
    per = collections.defaultdict(lambda: [0, 0])
    for p in sorted(rounds.glob("*.json")) if rounds.is_dir() else []:
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text())
        if "task" in r:
            per[r["task"]][0] += bool(r.get("ok"))
            per[r["task"]][1] += 1
    return dict(per)


def raw_usage(loop: Path) -> dict:
    """{session id: (tokens, cost)} of every role's *_raw.json under the loop's iteration directories."""
    out = {}
    for p in loop.glob("iter_*/**/*_raw.json"):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        mu = d.get("modelUsage") or {}
        if d.get("session_id") and mu:
            tok = {k: sum(int(m.get(rk) or 0) for m in mu.values()) for k, rk in zip(TOKENS, RAW_KEYS)}
            out[d["session_id"]] = (tok, float(d.get("total_cost_usd") or 0.0))
    return out


def transcript_usage(run: Path) -> list[tuple[float, dict]]:
    """[(first timestamp, tokens)] of the knowledge builds' sessions (their cwd is under <run>/ctx), one count per
    API message id."""
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(run.resolve()))
    out = []
    for f in PROJECTS.glob(f"{slug}-ctx-*/*.jsonl"):
        seen, tok, first = set(), zero(), None
        for line in f.read_text(errors="replace").splitlines():
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if first is None and m.get("timestamp"):
                first = ts(m["timestamp"])
            msg = m.get("message") or {}
            if m.get("type") == "assistant" and msg.get("usage") and msg.get("id") not in seen:
                seen.add(msg.get("id"))
                tok = add(tok, {k: int(msg["usage"].get(lk) or 0) for k, lk in zip(TOKENS, LEDGER_KEYS)})
        if first is not None:
            out.append((first, tok))
    return out


# -- a family's run -----------------------------------------------------------------------------------------------------


def task_objects(sandbox: Path, fam: dict) -> dict:
    """{task: "object name, ..."}: the free objects its goal names, by the scene's catalog name."""
    meta_p = sandbox / fam["scene"] / "scene_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"objects": {}}
    rows = {r["id"]: r for r in jl(sandbox / fam["tasks_file"])} if fam.get("tasks_file") else {}
    out = {}
    for t in fam.get("tasks", []):
        names = []
        for g in rows.get(t, {}).get("goal", []):
            for a in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", g.split("(", 1)[-1]):
                o = meta["objects"].get(a) or {}
                if o.get("free") and a not in [n for n, _ in names]:
                    names.append((a, ((o.get("gso") or {}).get("name") or a).replace("_", " ")))
        out[t] = ", ".join(n for _, n in names) or t
    return out


def iterations(loop: Path, alt: dict | None, family: str) -> list[dict]:
    """The loop's iterations in ledger order: label (mode 3: alternate.py's iteration, else loop iteration + 1), the
    loop iteration, its window start, its ledger rows and, in mode 3, the half's knowledge build and sync events."""
    rows = jl(loop / "ledger.jsonl")
    meta = json.loads((loop / "meta.json").read_text()) if (loop / "meta.json").exists() else {}
    halves = []
    if alt:
        ev = [e for e in alt.get("events", []) if e.get("family") == family]
        for e in ev:
            if e["step"] == "loop" and not e.get("stop"):
                same = [x for x in ev if x["iter"] == e["iter"]]
                halves.append(dict(iter=e["iter"], loop=e, build=next((x for x in same if x["step"] == "build"), None),
                                   sync=next((x for x in same if x["step"] == "sync"), None)))  # fmt: skip
    ks = sorted({r["iter"] for r in rows if isinstance(r.get("iter"), int)})
    out, prev_end = [], ts(meta["started"]) if meta.get("started") else None
    for n, k in enumerate(ks):
        idx = [i for i, r in enumerate(rows) if r.get("iter") == k]
        start = prev_end
        for r in rows[: idx[0]][::-1]:  # mode 3: the half starts when the loop resumes
            if r.get("step") == "resume":
                start = ts(r["t"])
                break
            if isinstance(r.get("iter"), int):
                break
        mine = [rows[i] for i in idx]
        half = halves[n] if n < len(halves) else None
        out.append(dict(label=half["iter"] if half else k + 1, loop_iter=k, start=start, rows=mine, half=half,
                        end=max(ts(r["t"]) for r in mine)))  # fmt: skip
        prev_end = out[-1]["end"]
    return out


def steps(it: dict, raw: dict, builds: list) -> list[dict]:
    """[{step, sec, calls, tok, cost, note}] of one iteration; sec None: the step did not run."""
    rows = it["rows"]

    def at(names):
        return [r for r in rows if r.get("step") in names]

    def usage(rs):
        tok, cost, calls = zero(), 0.0, 0
        for r in rs:
            if r.get("session") in raw:
                t, c = raw[r["session"]]
            elif r.get("usage"):
                t, c = {k: int(r["usage"].get(lk) or 0) for k, lk in zip(TOKENS, LEDGER_KEYS)}, r.get("cost_usd") or 0.0
            else:
                continue
            tok, cost, calls = add(tok, t), cost + float(c or 0.0), calls + 1
        return tok, cost, calls

    out, cur = [], it["start"]

    def put(name, end, rs=(), note=""):
        nonlocal cur
        sec = None if end is None or cur is None else max(end - cur, 0.0)
        tok, cost, calls = usage(rs)
        out.append(dict(step=name, sec=sec, calls=calls, tok=tok, cost=cost, note=note))
        if end is not None:
            cur = end

    rev = at({"review"})
    rev_start = min(ts(r["t"]) - (r.get("seconds") or 0) for r in rev) if rev else None
    put("Diagnosis", rev_start)
    put("Reviewer", max(ts(r["t"]) for r in rev) if rev else None, rev,
        "void" if any(r.get("void") for r in rev) else "")  # fmt: skip
    pr = at(PROBE_STEPS)
    n_h, n_d = sum(len(r.get("results") or []) for r in at({"probe"})), len(at({"design"}))
    put("Probe", max(ts(r["t"]) for r in pr) if pr else None,
        note=f"{n_h} hypotheses probed" + (f", {n_d} parameter design(s)" if n_d else "") if pr else "")  # fmt: skip
    mg = at({"manager"})
    put("Manager", max(ts(r["t"]) for r in mg) if mg else None, mg)
    en = at({"engineer"})
    void_en = sum(1 for r in en if r.get("void"))
    put(f"Engineers (×{len(en)}, parallel)" if en else "Engineers", max(ts(r["t"]) for r in en) if en else None, en,
        f"{void_en} draft(s) void" if void_en else "")  # fmt: skip
    co = at({"compose"})
    void = at({"void_iter"})
    put("Composer", max(ts(r["t"]) for r in co) if co else None, co,
        "; ".join(f"iteration void: {v.get('role')} {v.get('leaks')}" for v in void))  # fmt: skip
    ga = at(GATE_STEPS)
    verdicts = [r.get("reason", "") for r in ga if r.get("step") == "gate"]
    put("Gate", max(ts(r["t"]) for r in ga) if ga else None,
        note="; ".join(v if i == 0 else f"second shot: {v}" for i, v in enumerate(verdicts)))  # fmt: skip
    other = [r for r in rows if r.get("usage") and r.get("step") not in ROLE_STEP]
    if other:
        tok, cost, calls = usage(other)
        out.append(dict(step="Other model calls", sec=None, calls=calls, tok=tok, cost=cost,
                        note=", ".join(sorted({r["step"] for r in other}))))  # fmt: skip
    h = it["half"]
    if h and h["build"]:
        t0, t1 = ts(h["loop"]["t"]), ts(h["build"]["t"])
        tok, calls = zero(), 0
        for first, t in builds:
            if t0 - 5 <= first <= t1 + 5:
                tok, calls = add(tok, t), calls + 1
        out.append(dict(step="Knowledge build", sec=t1 - t0, calls=calls, tok=tok,
                        cost=float(h["build"].get("cost_usd") or 0.0),
                        note=f"{h['build'].get('entries', 0)} entries"))  # fmt: skip
        if h["sync"]:
            s = h["sync"]
            note = "accepted" if s.get("accepted") else s.get("skipped") or s.get("reason") or "rejected"
            out.append(dict(step="Sync gate", sec=ts(s["t"]) - t1, calls=0, tok=zero(), cost=0.0,
                            note=f"to the other family: {note}"))  # fmt: skip
    return out


def versions(hidden: Path, loop_rows: list[dict], its: list[dict], syncs: list[dict], family: str) -> list[dict]:
    """Every accepted version from the gate ledger, with the iteration label it belongs to: its own gate's iteration,
    a mode 3 sync's iteration (sync.jsonl: from, to, iter, the new incumbent's hash), else the iteration that had
    started by the gate's start."""
    gate_iter = {
        r.get("hash"): r["iter"] for r in loop_rows if r.get("step") == "gate" and isinstance(r.get("iter"), int)
    }
    label_of = {it["loop_iter"]: it["label"] for it in its}
    sync_label = {s["hash"]: s["iter"] for s in syncs if s.get("to") == family and s.get("accepted") and s.get("hash")}
    out = []
    for v in jl(hidden / "gate" / "ledger.jsonl"):
        if not v.get("accepted") or v.get("dry_run"):
            continue
        h = v["hash"]
        if h in gate_iter and gate_iter[h] in label_of:
            label, how = label_of[gate_iter[h]], "accepted"
        elif h in sync_label:
            label, how = sync_label[h], "synced"
        else:  # the iteration that had started by the gate's start
            t = ts(v["started"]) if v.get("started") else None
            before = [it for it in its if t is not None and it["start"] is not None and it["start"] <= t]
            label, how = (before[-1]["label"] if before else (its[0]["label"] if its else 1)), "accepted"
        if v.get("reason", "").startswith("accepted (non-inferior)"):
            how = "synced"
        name = f"{v['call']:02d}_{h}"
        out.append(dict(label=label, hash=h, name=name, how=how, result=v.get("result") or {},
                        accept=hidden / "gate" / name / "candidate"))  # fmt: skip
    return out


def notes(it: dict, syncs: list[dict], family: str) -> list[str]:
    """What happened at this iteration's gate (and, in mode 3, to the sync into this family)."""
    out = []
    for r in it["rows"]:
        if r.get("step") == "gate":
            s1 = r.get("stage1") or {}
            res = r.get("result") or {}
            if r.get("accepted"):
                out.append(f"candidate {r.get('hash')} accepted: {res.get('incumbent_ok')} -> {res.get('candidate_ok')}"
                           f" / {res.get('n')}")  # fmt: skip
            elif s1 and not res:
                out.append(f"candidate {r.get('hash')} rejected at stage 1 ({s1.get('candidate_ok')}/{s1.get('n')}"
                           f" against the incumbent's {s1.get('incumbent_ok')}/{s1.get('n')}): {r.get('reason')}")  # fmt: skip
            else:
                out.append(f"candidate {r.get('hash')} rejected: {r.get('reason')}")
        elif r.get("step") == "void_iter":
            out.append(
                f"iteration void: the {r.get('role')} named off-limits paths ({', '.join(r.get('leaks') or [])})"
            )
    for s in syncs:
        if s.get("to") == family and s.get("iter") == it["label"]:
            res = s.get("result") or {}
            what = (f"accepted as non-inferior, {res.get('incumbent_ok')} -> {res.get('candidate_ok')} / {res.get('n')}"
                    if s.get("accepted") else s.get("skipped") or s.get("reason") or "rejected")  # fmt: skip
            out.append(f"sync from {s.get('from')}: {what}")
    return out


def rate_text(col: dict) -> str:
    """ "acceptance w/n (r %), sealed w/n (r %)" of one column."""
    (a, n), (s, m) = pool_sum(col["acc"]), pool_sum(col["seal"])
    acc = f"acceptance {a}/{n} ({100 * a / n:.1f} %)" if n else "acceptance: no rounds"
    return acc + (f", sealed {s}/{m} ({100 * s / m:.1f} %)" if m else ", sealed not run")


def family_report(sandbox: Path, family: str, alt: dict | None, syncs: list[dict], builds: list) -> dict:
    loop = sandbox / OUT_REL / family / "loop"
    hidden = sandbox / HIDDEN_REL / family
    fam_p = sandbox / "scripts" / "rsi" / "families" / f"{family}.json"
    fam = json.loads(fam_p.read_text()) if fam_p.exists() else {"tasks": []}
    rows = jl(loop / "ledger.jsonl")
    its = iterations(loop, alt, family)
    vs = versions(hidden, rows, its, syncs, family)
    cols = [dict(label="Seed", name="", how="start", accept=hidden / "gate" / "incumbent_seed_rounds",
                 sealed=hidden / "sealed" / "seed", notes=[])]  # fmt: skip
    for it in its:
        mine = [v for v in vs if v["label"] <= it["label"]]
        v = mine[-1] if mine else None
        new = [x for x in vs if x["label"] == it["label"]]
        cols.append(dict(label=f"Iteration {it['label']}", name=v["name"] if v else "seed",
                         how=(new[-1]["how"] if new else "kept"),
                         accept=v["accept"] if v else cols[0]["accept"],
                         sealed=hidden / "sealed" / (v["name"] if v else "seed"), notes=notes(it, syncs, family)))  # fmt: skip
    for c in cols:
        c["acc"], c["seal"] = tally(c["accept"]), tally(c["sealed"])
        c["rate"] = rate_text(c)
    stop = [r.get("reason") for r in rows if r.get("step") == "stop"]
    meta = json.loads((loop / "meta.json").read_text()) if (loop / "meta.json").exists() else {}
    raw = raw_usage(loop)
    return dict(family=family, sandbox=sandbox, objects=task_objects(sandbox, fam), tasks=sorted(fam.get("tasks", [])),
                cols=cols, its=[dict(it, steps=steps(it, raw, builds), rate=c["rate"]) for it, c in zip(its, cols[1:])],
                stop=stop, meta=meta,
                sealed_run=(hidden / "sealed").is_dir())  # fmt: skip


def collect(target: Path, families: list[str] | None) -> dict:
    alt_p = target / "alternate.json"
    if alt_p.exists():
        alt = json.loads(alt_p.read_text())
        pairs = list(zip(("a", "b"), alt["families"]))
        fams = [(target / side, f) for side, f in pairs if not families or f in families]
        builds, syncs = transcript_usage(target), jl(target / "knowledge" / "sync.jsonl")
        mode = "alternating (mode 3)"
    else:
        alt, builds, syncs, mode = None, [], [], "single family (modes 1 and 2)"
        found = sorted(p.parent.parent.name for p in (target / OUT_REL).glob("*/loop/ledger.jsonl"))
        fams = [(target, f) for f in (families or found)]
    if not fams:
        raise SystemExit(f"{target}: no family's loop ledger under {OUT_REL}/*/loop and no alternate.json")
    reps = [family_report(sb, f, alt, syncs, builds) for sb, f in fams]
    return dict(target=target, mode=mode, alt=alt, families=reps)


# -- HTML ---------------------------------------------------------------------------------------------------------------

CSS = """
:root{--bg:#f7f8f6;--card:#ffffff;--fg:#1d2321;--muted:#5d6763;--line:#dfe4e1;--head:#eef2ef;--accent:#1f7a5a;
--warn:#a1561b;--rate:31,122,90}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121615;--card:#1a201e;--fg:#e4ebe7;
--muted:#9aa7a1;--line:#2c3532;--head:#222a27;--accent:#5cc79c;--warn:#e0a060;--rate:92,199,156;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121615;--card:#1a201e;--fg:#e4ebe7;--muted:#9aa7a1;--line:#2c3532;--head:#222a27;
--accent:#5cc79c;--warn:#e0a060;--rate:92,199,156;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0}
main{max-width:1760px;margin:0 auto;padding-block:24px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));
padding-right:max(16px,env(safe-area-inset-right,0px));display:flex;flex-direction:column;gap:28px}
h1{font-size:22px;margin:0;text-wrap:balance}h2{font-size:18px;margin:0 0 4px;text-wrap:balance}
h3{font-size:15px;margin:0 0 6px}
p{margin:0;max-width:75ch}.muted{color:var(--muted)}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;display:flex;
flex-direction:column;gap:12px;min-width:0}
dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:0}dt{color:var(--muted)}dd{margin:0;
min-width:0;overflow-wrap:anywhere}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;white-space:nowrap}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:right}
th{background:var(--head);font-weight:600}
td.l,th.l{text-align:left}td.name{white-space:normal;min-width:180px;max-width:280px}
tr.total td{font-weight:700;background:var(--head)}tr.sub td{font-weight:600;background:var(--head)}
td.rate{background:rgba(var(--rate),calc(var(--r)*0.35))}
th.kept,td.kept{color:var(--muted)}
.tag{display:inline-block;font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.tag.accepted,.tag.synced{color:var(--accent)}
ul{margin:0;padding-left:20px}li{max-width:110ch}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.warn{color:var(--warn)}
"""


def e(x) -> str:
    return html.escape(str(x))


def fmt_tok(n: int) -> str:
    return f"{n / 1e6:.2f} M" if n >= 1e6 else (f"{n / 1e3:.1f} k" if n >= 1e3 else str(n))


def fmt_min(sec) -> str:
    return "&mdash;" if sec is None else f"{sec / 60:.1f}"


def rate_cells(w: int, n: int, kept: bool) -> str:
    """Wins and rate cells; the rate tinted by its value, a kept column greyed."""
    attr = ' class="kept"' if kept else ""
    if not n:
        return f"<td{attr}>&mdash;</td><td{attr}>&mdash;</td>"
    r = w / n
    rate = "rate kept" if kept else "rate"
    return f'<td{attr}>{w}/{n}</td><td class="{rate}" style="--r:{r:.3f}">{100 * r:.1f} %</td>'


def pool_table(rep: dict, key: str) -> str:
    cols = [c for c in rep["cols"] if key == "acc" or c[key]]
    tasks = rep["tasks"] or sorted({t for c in cols for t in c[key]})
    h1 = "".join(f'<th colspan="2" class="{"kept" if c["how"] == "kept" else ""}">{e(c["label"])}<br>'
                 f'<span class="tag {e(c["how"])}">{e(c["how"])}</span>'
                 + (f' <code>{e(c["name"])}</code>' if c["name"] else "") + "</th>" for c in cols)  # fmt: skip
    h2 = "<th>Wins</th><th>Rate</th>" * len(cols)
    body = []
    for t in tasks:
        cells = "".join(rate_cells(*(c[key].get(t) or (0, 0)), c["how"] == "kept") for c in cols)
        body.append(f'<tr><td class="l"><code>{e(t)}</code></td><td class="l name">{e(rep["objects"].get(t, ""))}</td>'
                    f"{cells}</tr>")  # fmt: skip
    tot = "".join(rate_cells(*pool_sum(c[key]), False) for c in cols)
    body.append(f'<tr class="total"><td class="l">Total</td><td class="l name"></td>{tot}</tr>')
    return (f'<div class="scroll"><table><thead><tr><th class="l" rowspan="2">Task</th>'
            f'<th class="l" rowspan="2">Object</th>{h1}</tr><tr>{h2}</tr></thead><tbody>{"".join(body)}</tbody>'
            f"</table></div>")  # fmt: skip


def steps_table(data: dict) -> str:
    multi = len(data["families"]) > 1
    rows, grand, grand_sec, grand_cost, grand_calls = [], zero(), 0.0, 0.0, 0
    order = sorted(((it["label"], i, rep, it) for i, rep in enumerate(data["families"]) for it in rep["its"]),
                   key=lambda x: (x[0], x[1]))  # fmt: skip
    for label, _, rep, it in order:
        st = it["steps"]
        span = len(st) + 1
        fam = f'<td class="l" rowspan="{span}"><code>{e(rep["family"])}</code></td>' if multi else ""
        sub, sub_sec, sub_cost, sub_calls = zero(), 0.0, 0.0, 0
        for j, s in enumerate(st):
            lead = f'<td class="l" rowspan="{span}">{label}</td>{fam}' if j == 0 else ""
            t = s["tok"]
            toks = ("".join(f"<td>{fmt_tok(t[k])}</td>" for k in TOKENS) + f"<td>{fmt_tok(sum(t.values()))}</td>"
                    f"<td>{s['cost']:.2f}</td>") if s["calls"] else '<td colspan="6" class="muted l">no model call</td>'  # fmt: skip
            rows.append(f'<tr>{lead}<td class="l">{e(s["step"])}</td><td>{fmt_min(s["sec"])}</td><td>{s["calls"]}</td>'
                        f'{toks}<td class="l name">{e(s["note"])}</td></tr>')  # fmt: skip
            sub, sub_sec = add(sub, t), sub_sec + (s["sec"] or 0.0)
            sub_cost, sub_calls = sub_cost + s["cost"], sub_calls + s["calls"]
        rows.append(f'<tr class="sub"><td class="l">Iteration {label} total</td><td>{sub_sec / 60:.1f}</td>'
                    f"<td>{sub_calls}</td>" + "".join(f"<td>{fmt_tok(sub[k])}</td>" for k in TOKENS)
                    + f'<td>{fmt_tok(sum(sub.values()))}</td><td>{sub_cost:.2f}</td><td class="l name">'
                    f"incumbent: {e(it['rate'])}</td></tr>")  # fmt: skip
        grand, grand_sec = add(grand, sub), grand_sec + sub_sec
        grand_cost, grand_calls = grand_cost + sub_cost, grand_calls + sub_calls
    lead = '<td class="l" colspan="3">Run total</td>' if multi else '<td class="l" colspan="2">Run total</td>'
    final = "; ".join(
        (f"{r['family']}: " if multi else "") + f"final {r['cols'][-1]['rate']}" for r in data["families"]
    )
    rows.append(f'<tr class="total">{lead}<td>{grand_sec / 60:.1f}<br><span class="muted">{grand_sec / 3600:.2f} h'
                f"</span></td><td>{grand_calls}</td>" + "".join(f"<td>{fmt_tok(grand[k])}</td>" for k in TOKENS)
                + f'<td>{fmt_tok(sum(grand.values()))}</td><td>{grand_cost:.2f}</td><td class="l name">{e(final)}</td>'
                "</tr>")  # fmt: skip
    head = ('<tr><th class="l">Iteration</th>' + ('<th class="l">Family</th>' if multi else "")
            + '<th class="l">Step</th><th>Time (min)</th><th>Model calls</th><th>Input</th><th>Cache write</th>'
            '<th>Cache read</th><th>Output</th><th>Total tokens</th><th>Cost (USD)</th><th class="l">Note</th></tr>')  # fmt: skip
    return f'<div class="scroll"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def it_total(it: dict) -> tuple[float, int, float]:
    """(seconds, tokens, cost) of one iteration: Table 3's subtotal row of it."""
    st = it["steps"]
    return (sum(s["sec"] or 0.0 for s in st), sum(sum(s["tok"].values()) for s in st), sum(s["cost"] for s in st))


def family_table(data: dict) -> str:
    """Table 4 (user 2026-11-25): per iteration each family's time, total tokens and cost side by side, then all
    families together; the run total in the last row (the same sums as Table 3's)."""
    reps = data["families"]
    multi = len(reps) > 1
    by = [{it["label"]: it_total(it) for it in r["its"]} for r in reps]
    labels = sorted({k for b in by for k in b})
    groups = [*(f"<code>{e(r['family'])}</code>" for r in reps), *(["All families"] if multi else [])]

    def cells(v, total=False) -> str:
        if v is None:
            return '<td class="muted">&mdash;</td>' * 3
        sec, tok, cost = v
        hours = f'<br><span class="muted">{sec / 3600:.2f} h</span>' if total else ""
        return f"<td>{sec / 60:.1f}{hours}</td><td>{fmt_tok(tok)}</td><td>{cost:.2f}</td>"

    def both(vs):
        got = [v for v in vs if v is not None]
        return tuple(sum(x) for x in zip(*got)) if got else None

    rows = []
    for k in labels:
        vs = [b.get(k) for b in by]
        rows.append(f'<tr><td class="l">{k}</td>' + "".join(cells(v) for v in vs) + (cells(both(vs)) if multi else "")
                    + "</tr>")  # fmt: skip
    tot = [both(list(b.values())) or (0.0, 0, 0.0) for b in by]
    rows.append('<tr class="total"><td class="l">Run total</td>' + "".join(cells(v, True) for v in tot)
                + (cells(both(tot), True) if multi else "") + "</tr>")  # fmt: skip
    head = ('<tr><th class="l" rowspan="2">Iteration</th>' + "".join(f'<th colspan="3">{g}</th>' for g in groups)
            + "</tr><tr>" + "<th>Time (min)</th><th>Total tokens</th><th>Cost (USD)</th>" * len(groups) + "</tr>")  # fmt: skip
    return f'<div class="scroll"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def render(data: dict) -> str:
    target, alt = data["target"], data["alt"]
    m0 = data["families"][0]["meta"]
    info = [("Directory", f"<code>{e(target)}</code>"), ("Mode", e(data["mode"])),
            ("Families", ", ".join(f"<code>{e(r['family'])}</code>" for r in data["families"])),
            ("Model", e(m0.get("model", "unknown"))), ("Started", e(m0.get("started", "unknown")))]  # fmt: skip
    if alt:
        info += [("Code ref", f"<code>{e(alt.get('ref', ''))}</code>"), ("Seed package", e(alt.get("seed") or "main library")),
                 ("End", e(next((x.get("why") for x in alt.get("events", [])[::-1] if x.get("step") == "end"), "not ended")))]  # fmt: skip
    for r in data["families"]:
        if r["stop"]:
            info.append((f"Stop, {e(r['family'])}", e("; ".join(r["stop"]))))
    finals = []
    for r in data["families"]:
        c = r["cols"][-1]
        info.append((f"Final, {e(r['family'])}", f"<code>{e(c['name'] or 'seed')}</code>: {e(c['rate'])}"))
    parts = [
        f"<h1>RSI run report: {e(target.name)}</h1><dl>{''.join(f'<dt>{k}</dt><dd>{v}</dd>' for k, v in info)}</dl>"
    ]
    for n, r in enumerate(data["families"], 1):
        note_items = "".join(f"<li><b>{e(c['label'])}</b>: {e('; '.join(c['notes']) or 'no gate call')}</li>"
                             for c in r["cols"][1:])  # fmt: skip
        if r["stop"]:
            note_items += (
                f"<li><b>Stopped</b> after iteration {r['its'][-1]['label'] if r['its'] else 0}: "
                f"{e('; '.join(r['stop']))}</li>"
            )
        sec = [f"<h2>{e(r['family'])}</h2>",
               f"<h3>Table 1.{n}: acceptance test per object</h3>",
               '<p class="muted">Each iteration\'s incumbent on the acceptance pool. A column marked kept is the '
               "previous iteration's incumbent: nothing was accepted in that iteration.</p>",
               pool_table(r, "acc"), f"<ul>{note_items}</ul>",
               f"<h3>Table 2.{n}: sealed test per object</h3>"]  # fmt: skip
        if any(c["seal"] for c in r["cols"]):
            missing = [c["label"] for c in r["cols"] if not c["seal"]]
            sec.append('<p class="muted">Each iteration\'s incumbent on the sealed pool, drawn after the loop ended.'
                       + (f' No sealed rounds for: {e(", ".join(missing))} (run <code>sealed.py --all</code>).'
                          if missing else "") + "</p>")  # fmt: skip
            sec.append(pool_table(r, "seal"))
        else:
            sec.append(f'<p class="warn">No sealed rounds yet. In <code>{e(r["sandbox"])}</code> run '
                       f'<code>$PY scripts/rsi/sealed.py {e(r["family"])} --all</code>, then this report again.</p>')  # fmt: skip
        parts.append(f"<section>{''.join(sec)}</section>")
    parts.append('<section><h2>Table 3: time and tokens per iteration and step</h2><p class="muted">Wall-clock '
                 "minutes. Diagnosis is the incumbent on the diagnostic pool, its redraw and the knowledge lookup. "
                 "The Engineers run in parallel, so their row is the wall time of the slowest. Tokens are the model "
                 "calls' usage; output includes thinking. Diagnosis, Probe and the gates are simulation only. Each "
                 "iteration's total row gives its incumbent's success rate, the last row the run's totals and the "
                 "final success rate.</p>" + steps_table(data) + "</section>")  # fmt: skip
    both = (" The halves of mode 3 run one after the other, so All families is the run's wall time."
            if len(data["families"]) > 1 and alt else "")  # fmt: skip
    parts.append('<section><h2>Table 4: time, tokens and cost per iteration and family</h2><p class="muted">Each '
                 "family's iteration total from Table 3: wall-clock minutes, total tokens and cost." + both
                 + " The last row is the run total.</p>" + family_table(data) + "</section>")  # fmt: skip
    return (f"<title>RSI report {e(target.name)}</title><meta name=\"viewport\" content=\"width=device-width,"
            f'initial-scale=1"><style>{CSS}</style><main>{"".join(parts)}</main>')  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="an alternate.py run directory or a loop.py sandbox")
    ap.add_argument("--family", action="append", help="only this family (repeatable)")
    ap.add_argument("--out", type=Path, help="default: <target>/rsi_report.html")
    a = ap.parse_args(argv)
    data = collect(a.target.resolve(), a.family)
    out = a.out or a.target / "rsi_report.html"
    out.write_text('<!doctype html><meta charset="utf-8">' + render(data))
    for r in data["families"]:
        c = r["cols"][-1]
        print(f"{r['family']}: {len(r['its'])} iterations, final {c['name'] or 'seed'}, {c['rate']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
