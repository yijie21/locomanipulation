"""What the two families of one mode 3 run shared (user 2026-11-25): one English HTML page in run_report.py's style,
from the run's own records. Next to rsi_report.html, which has the per-object, time and token tables.

* Table 1: both families' incumbents per iteration, acceptance and sealed pools, and how each version came in.
* Table 2: the two channels -- the knowledge library and the code sync -- and where each one is stored.
* Table 3: the library's entries with their standing, citations (also by the other family) and whether the change
  that carried each one passed its own family's gate.
* Table 4: every citation of one family's entry by the other family's Reviewer, and how the citing hypothesis ended.
* One case per citing hypothesis: the entry's mechanism and repair, what the hypothesis proposed, the sentence that
  names the entry, the Prober's verdict, the Manager's picks, the drafts and the Composer, and that iteration's gate.
* Table 5: every code sync -- the files merged and copied, the verdict, and the first error of a failed test log.

    python3 scripts/rsi/report/knowledge_report.py <run dir> [--out FILE]

The run directory is an alternate.py run (alternate.json, sandboxes a/ and b/, knowledge/). The page goes to
<run>/rsi_knowledge_sharing.html unless --out says otherwise. Nothing is simulated and no model is called. Sources:
knowledge/knowledge.jsonl, usage.jsonl and sync.jsonl, logs/NN_<side>_<family>_sync_tests.log, and per family the
loop's ledger and iter_NN/{hypotheses,candidates}.json; Table 1 is run_report.collect's. A citation's outcome is
knowledge.py's (usage_rows): helped when the Prober verified the hypothesis or its code passed the gate, failed when
the Prober refuted it, else unclear -- so a verified hypothesis that was never built counts as helped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import knowledge as KN  # noqa: E402
import run_report as RR  # noqa: E402

e = RR.e
OUT_NAME = "rsi_knowledge_sharing.html"
SIDES = ("a", "b")  # alternate.py: the first family's sandbox, the second's
TEXT_FIELDS = ("claim", "evidence", "instead", "needs_code")  # a hypothesis's fields searched for the cited entry
READS_AS_N, CLAIM_N = 160, 320  # characters of an entry's reads_as and a hypothesis's claim shown (whole sentences)
CSS = (
    RR.CSS
    + """
th{white-space:normal}td.text{white-space:normal;min-width:380px;max-width:600px}
.case{border-top:1px solid var(--line);padding-top:12px;display:flex;flex-direction:column;gap:8px}
blockquote{margin:0;padding:4px 12px;border-left:3px solid var(--accent);color:var(--muted);max-width:110ch}
"""
)


def c(x) -> str:
    return f"<code>{e(x)}</code>"


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", str(text or "").strip()) if s]


def first(text: str) -> str:
    s = sentences(text)
    return s[0] if s else ""


def lead(text: str, n: int) -> str:
    """The text's first sentences, up to the first that brings it to `n` characters."""
    out = ""
    for s in sentences(text):
        out = f"{out} {s}".strip()
        if len(out) >= n:
            break
    return out


def table(head: list[str], rows: list[list[str]], left=frozenset(), text=frozenset(), narrow=frozenset()) -> str:
    """Cells are HTML already; `left` columns align left, `text` ones wrap, `narrow` ones wrap in run_report's
    name width."""

    def cls(j):
        names = " ".join([x for x, on in (("l", j in left), ("text", j in text), ("name", j in narrow)) if on])
        return f' class="{names}"' if names else ""

    th = "".join(f"<th{cls(j)}>{h}</th>" for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f"<td{cls(j)}>{x}</td>" for j, x in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def section(title: str, *parts: str) -> str:
    return f"<section><h2>{title}</h2>{''.join(parts)}</section>"


def muted(text: str) -> str:
    return f'<p class="muted">{text}</p>'


def ul(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def dl(pairs: list[tuple[str, str]]) -> str:
    return "<dl>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in pairs) + "</dl>"


def fam_of(entry_id: str) -> str:
    return entry_id.split("/", 1)[0]


def name_of(entry_id: str) -> str:
    return entry_id.split("/", 1)[-1]


# -- records ------------------------------------------------------------------------------------------------------------


def own_gate(entry: dict) -> str:
    """Whether the change that carried the entry passed its own family's gate (the entry's source.gate)."""
    g = (entry.get("source") or {}).get("gate")
    if not g:
        return "never reached a gate"
    via = f" ({g['via']})" if g.get("via") else ""
    return f"accepted{via}" if g.get("accepted") else f"not accepted{via}"


def loop_dir(run: Path, side: str, family: str) -> Path:
    return run / side / RR.OUT_REL / family / "loop"


def jload(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def case(run: Path, side: str, family: str, it: int, hyp: str, refs: list[dict], ledger: list[dict]) -> dict:
    """One citing hypothesis: its record, the Manager's picks, its drafts, the Composer and the iteration's gate."""
    d = loop_dir(run, side, family) / f"iter_{it:02d}"
    hyps = (jload(d / "hypotheses.json", {}) or {}).get("hypotheses") or []
    h = next((x for x in hyps if isinstance(x, dict) and x.get("id") == hyp), {})
    rows = [r for r in ledger if r.get("iter") == it]
    picks = []
    for r in rows:
        if r.get("step") == "manager" and not r.get("void"):
            picks += [x for x in [r.get("hypothesis"), *(r.get("hypotheses") or [])] if x]
    drafts = [x.get("number") for x in jload(d / "candidates.json", []) or [] if isinstance(x, dict)
              and x.get("hypothesis") == hyp]  # fmt: skip
    comp = next((r for r in reversed(rows) if r.get("step") == "compose" and not r.get("void")), None)
    quotes = []
    for ref in refs:
        for f in TEXT_FIELDS:
            quotes += [s for s in sentences(h.get(f)) if ref["entry"] in s or name_of(ref["entry"]) in s]
    return dict(family=family, iter=it, hyp=hyp, refs=refs, h=h, picks=list(dict.fromkeys(picks)), drafts=drafts,
                compose=comp, gate=RR.notes(dict(rows=rows, label=it + 1), [], family),
                quotes=list(dict.fromkeys(quotes)))  # fmt: skip


def first_error(log: Path) -> str:
    """The first pytest error line (`E   ...`) of a sync's test log, else its last line."""
    if not log.exists():
        return ""
    lines = [x for x in log.read_text(errors="replace").splitlines() if x.strip()]
    err = next((x for x in lines if x.startswith("E ")), None)
    return (err[1:].strip() if err else (lines[-1] if lines else ""))[:240]


def collect(run: Path) -> dict:
    alt_p = run / "alternate.json"
    if not alt_p.exists():
        raise SystemExit(f"{run}: no alternate.json -- knowledge sharing needs a mode 3 run (alternate.py)")
    alt = json.loads(alt_p.read_text())
    lib = run / "knowledge"
    side = dict(zip(alt["families"], SIDES))
    usage = RR.jl(lib / KN.USAGE)
    stats = KN.entry_stats(usage)
    ents = KN.entries(lib)
    cross = [u for u in usage if fam_of(u["entry"]) != u["family"]]
    by_other = {}
    for u in cross:
        by_other[u["entry"]] = by_other.get(u["entry"], 0) + 1
    groups: dict[tuple, list[dict]] = {}
    for u in sorted(cross, key=lambda u: (u["iter"], alt["families"].index(u["family"]), u["hypothesis"])):
        groups.setdefault((u["family"], u["iter"], u["hypothesis"]), []).append(u)
    ledgers = {f: RR.jl(loop_dir(run, side[f], f) / "ledger.jsonl") for f in alt["families"]}
    cases = [case(run, side[f], f, it, hyp, refs, ledgers[f]) for (f, it, hyp), refs in groups.items()]
    syncs = RR.jl(lib / KN.SYNCS)
    for s in syncs:
        s["error"] = first_error(
            run / "logs" / f"{s['iter']:02d}_{side.get(s['from'], '?')}_{s['from']}_sync_tests.log"
        )
    return dict(run=run, alt=alt, side=side, report=RR.collect(run, None), entries=ents, stats=stats, usage=usage,
                cross=cross, by_other=by_other, cases=cases, syncs=syncs)  # fmt: skip


# -- HTML ---------------------------------------------------------------------------------------------------------------


def rate(per: dict) -> str:
    w, n = RR.pool_sum(per)
    return f"{w}/{n} ({100 * w / n:.1f} %)" if n else "&mdash;"


def results_table(data: dict) -> str:
    reps = data["report"]["families"]
    stopped = data["alt"].get("stopped") or {}
    head, rows = ["Iteration"], []
    for r in reps:
        head += [f"{c(r['family'])}<br>acceptance", "sealed", "version"]
    for k in range(max(len(r["cols"]) for r in reps)):
        row = [e(next(r["cols"][k]["label"] for r in reps if k < len(r["cols"])))]
        for r in reps:
            if k >= len(r["cols"]):
                row += [
                    "&mdash;",
                    "&mdash;",
                    f'<span class="muted">stopped: {e(stopped.get(r["family"], "no half"))}</span>',
                ]
                continue
            col = r["cols"][k]
            tag = f'<span class="tag {e(col["how"])}">{e(col["how"])}</span>'
            row += [rate(col["acc"]), rate(col["seal"]), f"{tag} {c(col['name'])}" if col["name"] else tag]
        rows.append(row)
    left = {0} | {3 * j + 3 for j in range(len(reps))}
    return table(head, rows, left=left)


def channels_table() -> str:
    return table(
        ["Channel", "What is shared", "Where it is stored", "How it reaches the other family"],
        [
            ["Knowledge library",
             "One entry per failure mechanism: its condition, what it looks like (reads as), the repair, the evidence "
             "per object and the gate result of the change that carried it. A family's entries are rebuilt after each "
             "of its halves (knowledge.py build) and tagged by the kind of task.",
             f"{c('knowledge/knowledge.jsonl')} (the entries), {c('knowledge/INDEX.md')} (an overview), "
             f"{c('knowledge/usage.jsonl')} (every citation and how it ended)",
             f"Before each half the library is copied into the sandbox ({c('scripts/rsi/knowledge/')}). Before the "
             f"Reviewer the loop writes {c('KNOWLEDGE.md')} there: the whole library, each entry marked with the "
             f"incumbent's rounds it holds on ({c('loop/iter_NN/knowledge_hits.json')}). A hypothesis that builds on "
             f"an entry names it in {c('knowledge_refs')} ({c('loop/iter_NN/hypotheses.json')})."],
            ["Code sync",
             "The skills package: one family's incumbent merged three-way into the other's.",
             f"{c('sync/NN_<from>_to_<to>/cand/skills/')} (each merge), {c('sync/base/skills/')} (the merge base), "
             f"{c('knowledge/sync.jsonl')} (the verdicts), {c('logs/NN_<side>_<from>_sync_tests.log')}",
             "After each half. The merged package's own tests must pass; then the receiving family's gate judges it "
             "as non-inferior (no more starts lost than won)."],
        ],
        left={0, 1, 2, 3}, text={1, 2, 3},
    )  # fmt: skip


def library_table(data: dict) -> str:
    rows = []
    for en in data["entries"]:
        st = data["stats"].get(en["id"])
        cited = f"{st['uses']} ({st['helped']} / {st['failed']} / {st['unclear']})" if st else "0"
        rows.append([c(name_of(en["id"])), c(fam_of(en["id"])), e(KN.status_of(st)), cited,
                     str(data["by_other"].get(en["id"], 0)), str(len((en.get("evidence") or {}).get("by_object") or {})),
                     e(own_gate(en)), e(lead((en.get("signature") or {}).get("reads_as"), READS_AS_N))])  # fmt: skip
    return table(["Entry", "From", "Standing", "Cited (helped / no effect / unclear)", "By the other family",
                  "Objects", "Its own family's gate", "Reads as (its first sentences)"], rows,
                 left={0, 1, 2, 6, 7}, text={7})  # fmt: skip


def citations_table(data: dict) -> str:
    rows = [[str(u["iter"] + 1), c(u["family"]), c(u["hypothesis"]), c(name_of(u["entry"])), c(fam_of(u["entry"])),
             "yes" if u.get("picked") else "no", e(u.get("accepted") or "no"), e(u.get("status")), e(u.get("outcome"))]
            for u in sorted(data["cross"], key=lambda u: (u["iter"], u["family"], u["hypothesis"]))]  # fmt: skip
    return table(["Iteration", "Citing family", "Its hypothesis", "Entry cited", "Entry from", "Built", "Shipped",
                  "Prober", "Outcome"], rows, left={1, 2, 3, 4, 6, 7, 8})  # fmt: skip


def composer_text(cs: dict) -> str:
    comp, drafts = cs["compose"], cs["drafts"]
    if not drafts:
        return "no draft of this hypothesis"
    mine = ", ".join(str(n) for n in drafts)
    if not comp:
        return f"draft {mine}; no Composer call"
    dec, merged, adopted = comp.get("decision"), comp.get("merged") or [], comp.get("adopted")
    if dec == "merge":
        how = (f"merged drafts {', '.join(map(str, merged))}" + (" (this one included)" if set(drafts) & set(merged)
               else " (not this one)")) if merged else "merged"  # fmt: skip
    elif dec == "adopt":
        how = f"adopted draft {adopted}" + (" (this one)" if adopted in drafts else "")
    else:
        how = str(dec or "no decision")
    mech = first(comp.get("mechanism"))
    return f"draft {mine}; the Composer {e(how)}" + (f": <i>{e(mech)}</i>" if mech else "")


def case_html(data: dict, cs: dict) -> str:
    ents = {en["id"]: en for en in data["entries"]}
    h = cs["h"]
    cited = []
    for ref in cs["refs"]:
        en = ents.get(ref["entry"], {})
        cited.append(f"{c(ref['entry'])}, its own family's gate: {e(own_gate(en))}.<br>Reads as: "
                     f"{e(lead((en.get('signature') or {}).get('reads_as'), READS_AS_N))}<br>Repair: "
                     f"{e((en.get('repair') or {}).get('works') or '')}")  # fmt: skip
    pairs = [("Cites", "<br><br>".join(cited)), ("Claim", e(lead(h.get("claim"), CLAIM_N) or "(no hypotheses.json)"))]
    if h.get("instead"):
        pairs.append(("Proposes", e(h["instead"])))
    pairs += [("Prober", e(cs["refs"][0].get("status"))),
              ("Manager", ("built it" if cs["refs"][0].get("picked") else "did not build it") + "; picked "
               + (", ".join(c(x) for x in cs["picks"]) or "nothing")),
              ("Drafts, Composer", composer_text(cs)),
              ("Gate", e("; ".join(cs["gate"]) or "no gate call")),
              ("Recorded", f"shipped: {e(cs['refs'][0].get('accepted') or 'no')}; outcome: "
               f"{e(cs['refs'][0].get('outcome'))}")]  # fmt: skip
    quote = "".join(f"<blockquote>{e(q)}</blockquote>" for q in cs["quotes"])
    where = (muted("Where the hypothesis names the entry:") + quote if quote
             else muted("The hypothesis names the entry only in its knowledge_refs."))  # fmt: skip
    return (f'<div class="case"><h3>Iteration {cs["iter"] + 1}: {c(cs["family"])} hypothesis {c(cs["hyp"])}</h3>'
            f"{dl(pairs)}{where}</div>")  # fmt: skip


def sync_table(data: dict) -> str:
    rows = []
    for s in data["syncs"]:
        res = s.get("result") or {}
        if s.get("accepted"):
            same = res.get("candidate_ok") == res.get("incumbent_ok") and not res.get("b") and not res.get("c")
            out = (f'<span class="tag accepted">{e(s.get("reason") or "accepted")}</span> {res.get("incumbent_ok")} '
                   f"&rarr; {res.get('candidate_ok')} / {res.get('n')} (won {res.get('b')}, lost {res.get('c')})"
                   + (" &mdash; no start changed" if same else ""))  # fmt: skip
        else:
            why = s.get("skipped") or s.get("reason") or "rejected"
            out = f'<span class="warn">{e(why)}</span>' if "fails" in why else e(why)
        rows.append([str(s["iter"]), f"{c(s['from'])} &rarr;<br>{c(s['to'])}",
                     ", ".join(c(x) for x in s.get("merged") or []) or "&mdash;",
                     ", ".join(c(x) for x in s.get("copied") or []) or "&mdash;", out,
                     c(s["error"]) if s.get("error") and not s.get("accepted") else ""])  # fmt: skip
    return table(["Iteration", "From &rarr; to", "Merged", "Copied", "Verdict", "First test error"], rows,
                 left={1, 2, 3, 4, 5}, text={5}, narrow={2, 3, 4})  # fmt: skip


def summary(data: dict) -> list[str]:
    fams = data["alt"]["families"]
    out = []
    n_by = {f: sum(1 for en in data["entries"] if fam_of(en["id"]) == f) for f in fams}
    out.append(f"<b>Library:</b> {len(data['entries'])} entries ("
               + ", ".join(f"{n_by[f]} from {c(f)}" for f in fams) + f"); {len(data['usage'])} citations, "
               f"{len(data['cross'])} of them by the other family.")  # fmt: skip
    for src in fams:
        for dst in fams:
            rows = [u for u in data["cross"] if fam_of(u["entry"]) == src and u["family"] == dst]
            if src == dst or not rows:
                continue
            hyps = {(u["iter"], u["hypothesis"]) for u in rows}
            built = {(u["iter"], u["hypothesis"]) for u in rows if u.get("picked")}
            shipped = {(u["iter"], u["hypothesis"]) for u in rows if u.get("accepted")}
            helped = sum(u.get("outcome") == "helped" for u in rows)
            out.append(f"<b>{c(src)} &rarr; {c(dst)}:</b> {len(rows)} citation(s) by {len(hyps)} hypothesis(es); "
                       f"{len(built)} built, {len(shipped)} shipped in a candidate that passed the gate; outcome helped "
                       f"{helped}, other {len(rows) - helped}.")  # fmt: skip
    if not data["cross"]:
        out.append("<b>No citation crossed between the families.</b>")
    syncs = data["syncs"]
    acc = [s for s in syncs if s.get("accepted")]
    fails = sum("fails its tests" in (s.get("skipped") or "") for s in syncs)
    nothing = sum("changes nothing" in (s.get("skipped") or "") for s in syncs)
    out.append(f"<b>Code sync:</b> {len(syncs)} attempts; {len(acc)} accepted, {fails} skipped because the merged "
               f"package failed its tests, {nothing} with nothing to merge, {len(syncs) - len(acc) - fails - nothing} "
               "rejected otherwise.")  # fmt: skip
    for r in data["report"]["families"]:
        col = r["cols"][-1]
        out.append(f"<b>Final, {c(r['family'])}:</b> {c(col['name'] or 'seed')}, {e(col['rate'])}.")
    return out


def render(data: dict) -> str:
    run, alt = data["run"], data["alt"]
    info = [("Run directory", c(run)), ("Mode", "alternating (mode 3), one half of each family per iteration"),
            ("Families", ", ".join(f"{c(f)} (sandbox {c(s + '/')})" for f, s in data["side"].items())),
            ("Code ref", c(alt.get("ref", ""))), ("Seed package", e(alt.get("seed") or "main library")),
            ("End", e(next((x.get("why") for x in alt.get("events", [])[::-1] if x.get("step") == "end"), "not ended"))),
            ("Companion page", f"{c('rsi_report.html')} in the run directory (scripts/rsi/report/run_report.py)")]  # fmt: skip
    cases = "".join(case_html(data, cs) for cs in data["cases"]) or muted("No citation crossed between the families.")
    parts = [
        f"<h1>RSI knowledge sharing: {e(run.name)}</h1>{dl(info)}",
        section("Summary", ul(summary(data))),
        section("Table 1: both families per iteration",
                muted("Each iteration's incumbent on the acceptance pool and on the sealed pool (sealed.py --all; "
                      "&mdash; when a version has no sealed rounds). synced: the version came from the other family "
                      "by the code sync; kept: nothing was accepted in that iteration."), results_table(data)),
        section("Table 2: what is shared and where",
                muted(f"Paths are relative to the run directory. {c('alternate.py publish')} copies the run's "
                      f"library into the repository's {c('scripts/rsi/knowledge/')}."), channels_table()),
        section(f"Table 3: the library at the end of the run ({len(data['entries'])} entries)",
                muted("Standing from every citation (knowledge.py: tested -- helped at least once and at least as "
                      "often as not; doubtful; else candidate). By the other family: citations by the family that "
                      "did not write the entry. Objects: the objects in the entry's evidence. Its own family's gate: "
                      "whether the change that carried the entry was accepted there."), library_table(data)),
        section("Table 4: citations across the families",
                muted("Built: the Manager picked the hypothesis for an Engineer. Shipped: a draft of it was in a "
                      "candidate that passed the gate (draft alone, or merged by the Composer). Prober: verified, refuted, "
                      "mixed, or hypothesis (not probed). Outcome: helped when "
                      "the Prober verified it or it shipped, failed when refuted, else unclear."),
                citations_table(data)),
        section("The cases: each citing hypothesis",
                muted("Iteration is alternate.py's. The gate line is that iteration's gate in the citing family."),
                cases),
        section("Table 5: every code sync",
                muted("The files the three-way merge combined and the files only the source had (copied); the "
                      "verdict from knowledge/sync.jsonl; for a failed test run, the first error line of "
                      "logs/NN_&lt;side&gt;_&lt;from&gt;_sync_tests.log."), sync_table(data)),
        section("Sources", muted(
            f"All under {c(run)}: {c('alternate.json')}; {c('knowledge/')} (knowledge.jsonl, usage.jsonl, sync.jsonl); "
            f"{c('logs/*_sync_tests.log')}; per sandbox {c(str(RR.OUT_REL) + '/<family>/loop/')} (ledger.jsonl, "
            f"iter_NN/hypotheses.json, iter_NN/candidates.json) and {c(str(RR.HIDDEN_REL) + '/<family>/')} (gate and "
            "sealed rounds). Made by scripts/rsi/report/knowledge_report.py; no model call.")),
    ]  # fmt: skip
    return (f"<title>RSI knowledge sharing {e(run.name)}</title><meta name=\"viewport\" content=\"width=device-width,"
            f'initial-scale=1"><style>{CSS}</style><main>{"".join(parts)}</main>')  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="an alternate.py run directory")
    ap.add_argument("--out", type=Path, help=f"default: <run>/{OUT_NAME}")
    a = ap.parse_args(argv)
    data = collect(a.run.resolve())
    out = a.out or a.run / OUT_NAME
    out.write_text('<!doctype html><meta charset="utf-8">' + render(data))
    print(f"{len(data['entries'])} entries, {len(data['usage'])} citations ({len(data['cross'])} across), "
          f"{len(data['syncs'])} syncs")  # fmt: skip
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
