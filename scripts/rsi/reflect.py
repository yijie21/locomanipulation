"""TODO 32.10 (Harness2's harnessing-level recursion): after a family's loop has stopped, a Reflector reads its whole
record and writes a library entry for the loops of later families -- per change a verdict (helped / hurt / no observed
effect / confounded) and at most one transferable lesson (cue, lesson, evidence, confidence), next to the code the run
ended with. The Reviewer's and the Engineer's prompts point at the library when the sandbox holds its INDEX.md.

    python3 scripts/rsi/reflect.py <finished sandbox> <family> [--out scripts/rsi/experience]
    python3 scripts/rsi/reflect.py --index [--out scripts/rsi/experience]       # rebuild INDEX.md only

The record is gathered into /workspace/tmp/rsi_reflect/<family>/context, outside the repository so the session
picks up no project memory (the first run read the operator's MEMORY.md through the repository's project) -- the ledger, every iteration's
hypotheses, probes, Manager decisions and candidate patches, each gated candidate's diagnostic flips against its
incumbent, the final code against the seed, the sealed summary); the Reflector (claude -p, Opus, effort xhigh) runs
there and may read nothing else -- paths it names outside are listed in the entry's `outside`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL = "claude-opus-5-5"
VISIBLE, HIDDEN = "outputs/scene_rsi_20261016", "outputs/scene_rsi_20261016_hidden"
SCHEMA = {
    "type": "object",
    "required": ["edits"],
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["iteration", "component", "change", "observed", "verdict"],
                "properties": {
                    "iteration": {"type": "integer"},
                    "component": {"type": "string"},
                    "change": {"type": "string"},
                    "observed": {"type": "string"},
                    "verdict": {"enum": ["helped", "hurt", "no observed effect", "confounded"]},
                },
            },
        },
        "experience": {
            "type": "object",
            "required": ["cue", "lesson", "evidence", "confidence"],
            "properties": {k: {"type": "string"} for k in ("cue", "lesson", "evidence", "confidence")},
        },
        "no_entry": {"type": "boolean"},
        "why": {"type": "string"},
    },
}
BUILD = Path("/workspace/tmp/rsi_reflect")
PATH = re.compile(r"(?:/workspace|/home|/tmp)/[^\s\"'`;|&)]+")


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def outcomes(d: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if not p.name.startswith("_")}


def diff(a: Path, b: Path) -> str:
    """`a` -> `b` with the two roots written as a/ and b/: an absolute root names the source sandbox's hidden tree
    (the gate's candidate), and a role that copies such a path from the library voids its iteration (2026-10-21)."""
    out = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc", str(a), str(b)], capture_output=True,
                         text=True).stdout  # fmt: skip
    return relabel(out, str(a), str(b))


def relabel(patch: str, a: str, b: str) -> str:
    return patch.replace(a + "/", "a/").replace(b + "/", "b/")


def gather(src: Path, family: str, ctx: Path) -> None:
    loop, hid = src / VISIBLE / family, src / HIDDEN / family
    state = loop / "loop"
    if ctx.exists():
        shutil.rmtree(ctx)
    ctx.mkdir(parents=True)
    fam = src / "scripts/rsi/families" / f"{family}.json"
    shutil.copy2(fam if fam.exists() else REPO / "scripts/rsi/families" / f"{family}.json", ctx / "family.json")
    rows = jl(state / "ledger.jsonl")
    (ctx / "ledger.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k not in ("usage", "session")},
                                                         ensure_ascii=False) + "\n" for r in rows))  # fmt: skip
    for d in sorted(state.glob("iter_*")):
        out = ctx / d.name
        out.mkdir()
        for f in ("hypotheses.json", "probe.json", "manager.json", "candidates.json", "composer.json"):
            if (d / f).exists():
                shutil.copy2(d / f, out / f)
        for f in d.glob("code*.patch"):
            shutil.copy2(f, out / f.name)
    # every gated candidate's diagnostic rounds against the incumbent it was built on
    (ctx / "flips").mkdir()
    seed = next((p.name for p in (loop / "incumbents").iterdir() if p.name not in {r.get("hash") for r in rows}), None)
    inc = seed
    for r in rows:
        if r.get("step") != "gate" or not r.get("hash") or r.get("void"):
            continue
        a, b = loop / "diag" / f"incumbent_{inc}", loop / "diag" / f"candidate_{r['hash']}"
        if not b.exists():
            b = loop / "diag" / f"incumbent_{r['hash']}"
        A, Bo = outcomes(a), outcomes(b)
        both = sorted(set(A) & set(Bo))
        (ctx / "flips" / f"iter_{r['iter']:02d}_{r['hash']}.json").write_text(json.dumps(dict(
            iteration=r["iter"], candidate=r["hash"], incumbent=inc, accepted=r.get("accepted"), n=len(both),
            candidate_ok=sum(bool(Bo[x]["ok"]) for x in both), incumbent_ok=sum(bool(A[x]["ok"]) for x in both),
            fixed=[x for x in both if Bo[x]["ok"] and not A[x]["ok"]],
            broken=[x for x in both if A[x]["ok"] and not Bo[x]["ok"]],
            note="the candidate's diagnostic rounds stop when the gate rejects it: n may be short"), indent=1))  # fmt: skip
        if r.get("accepted"):
            inc = r["hash"]
    final = loop / "incumbents" / inc
    if seed and final.exists():
        (ctx / "final.patch").write_text(diff(loop / "incumbents" / seed / "skills", final / "skills"))
    elif seed:
        g = hid / "gate"
        last = sorted(g.glob(f"*_{inc}"))
        if last:
            (ctx / "final.patch").write_text(diff(loop / "incumbents" / seed / "skills", last[-1] / "cand" / "skills"))
    if (hid / "sealed" / "summary.json").exists():
        s = json.loads((hid / "sealed" / "summary.json").read_text())
        (ctx / "sealed.json").write_text(json.dumps({k: {kk: v.get(kk) for kk in ("ok", "n", "per_task")}
                                                     for k, v in s.get("versions", {}).items()}, indent=1))  # fmt: skip


def reflect(ctx: Path, family: str) -> tuple[dict, dict]:
    prompt = (
        (HERE / "prompts" / "reflector.md").read_text().replace("{FAMILY}", family).replace("{CONTEXT_DIR}", str(ctx))
    )
    session = str(uuid.uuid4())
    cmd = ["claude", "-p", "--model", MODEL, "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
           "--settings", json.dumps({"disableAllHooks": True}), "--strict-mcp-config", "--effort", "xhigh",
           "--permission-mode", "bypassPermissions", "--disallowedTools", "Edit,Write,NotebookEdit,Agent,WebFetch,WebSearch",
           "--session-id", session]  # fmt: skip
    t0 = time.time()
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_SSE")}
    env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY="1", TMPDIR=str(ctx.parent))
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=ctx, env=env)
    out = json.loads(r.stdout)
    so = out.get("structured_output")
    if so is None:
        raise SystemExit(f"no structured output: {str(out.get('result'))[-300:]}")
    named = set()
    for p in Path.home().glob(f".claude/projects/*/{session}.jsonl"):
        named |= {m for m in PATH.findall(p.read_text(errors="replace"))
                  if not m.startswith((str(ctx.parent), str(Path.home() / ".claude/shell-snapshots")))}  # fmt: skip
    acct = dict(cost_usd=out.get("total_cost_usd"), seconds=round(time.time() - t0), session=session,
                outside=sorted(named)[:50])  # fmt: skip
    return so, acct


def render(family: str, so: dict, acct: dict) -> str:
    lines = [f"# {family}", ""]
    e = so.get("experience")
    if e and not so.get("no_entry"):
        lines += [f"**Cue:** {e['cue']}", "", f"**Lesson:** {e['lesson']}", "", f"**Evidence:** {e['evidence']}", "",
                  f"**Confidence:** {e['confidence']}", ""]  # fmt: skip
    else:
        lines += [f"No entry: {so.get('why', '')}", ""]
    lines += [
        "## What each change did",
        "",
        "| iter | component | change | observed | verdict |",
        "|---|---|---|---|---|",
    ]
    for r in so.get("edits", []):
        cell = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
        lines.append(
            f"| {r['iteration']} | {cell(r['component'])} | {cell(r['change'])} | {cell(r['observed'])} | {r['verdict']} |"
        )
    lines += ["", "The code the run ended with, against its seed: `final.patch`.", ""]
    return "\n".join(lines)


def index(out: Path) -> None:
    rows = ["# Experience library", "",
            "One entry per finished family (scripts/rsi/reflect.py, TODO 32.10): what each change did, and at most one "
            "lesson that may transfer. An entry's numbers come from its own family's objects and surfaces: treat them "
            "as hypotheses here and check them on this family's rounds.", "",
            "| family | cue | confidence |", "|---|---|---|"]  # fmt: skip
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        so = json.loads((d / "entry.json").read_text()) if (d / "entry.json").exists() else {}
        e = so.get("experience") if not so.get("no_entry") else None
        cue = (e or {}).get("cue", "(no entry)").replace("|", "/").replace("\n", " ")
        rows.append(f"| [{d.name}]({d.name}/EXPERIENCE.md) | {cue} | {(e or {}).get('confidence', '').split('.')[0]} |")
    (out / "INDEX.md").write_text("\n".join(rows) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sandbox", type=Path, nargs="?")
    ap.add_argument("family", nargs="?")
    ap.add_argument("--out", type=Path, default=HERE / "experience")
    ap.add_argument("--index", action="store_true", help="only rebuild INDEX.md")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.index:
        ctx = BUILD / a.family / "context"
        gather(a.sandbox.resolve(), a.family, ctx)
        so, acct = reflect(ctx, a.family)
        d = a.out / a.family
        d.mkdir(parents=True, exist_ok=True)
        # the paths the session named outside its context stay beside the context: the library is read by later
        # roles, and a path into another sandbox's hidden tree copied from it voids their iteration (2026-10-21)
        (ctx.parent / "outside.json").write_text(json.dumps(acct["outside"], indent=1))
        entry = dict(so, **{k: v for k, v in acct.items() if k != "outside"})
        (d / "entry.json").write_text(json.dumps(entry, indent=1, ensure_ascii=False))
        (d / "EXPERIENCE.md").write_text(render(a.family, so, acct))
        if (ctx / "final.patch").exists():
            shutil.copy2(ctx / "final.patch", d / "final.patch")
        print(f"{a.family}: {len(so.get('edits', []))} rows, {'no entry' if so.get('no_entry') else 'an entry'}, "
              f"{acct['cost_usd']} USD, {acct['seconds']} s, outside {acct['outside'][:5]}")  # fmt: skip
    index(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
