"""TODO 27.10.6 (plan docs/handoff/scene-rsi-plan-20261016.md §12.2): replay a finished iteration's Reviewer with and
without the diagnostic digest, then probe every replay's hypotheses the same way, to see whether the digest keeps
the diagnosis while costing less.

Runs inside a replay sandbox (sandbox.py, built from the commit that has digest.py) that holds one finished loop's
iteration:

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/scene_rsi_20261016/review_replay.py setup <finished sandbox> <family> <iter>
    $PY scripts/experiments/scene_rsi_20261016/review_replay.py review <family> --arm digest --tag d1
    $PY scripts/experiments/scene_rsi_20261016/review_replay.py review <family> --arm plain --tag p1
    $PY scripts/experiments/scene_rsi_20261016/review_replay.py probe <family> orig p1 d1 ...
    $PY scripts/experiments/scene_rsi_20261016/review_replay.py report <family>

`setup` copies the iteration's Reviewer context (paths moved into this sandbox), the incumbent's skills snapshot, its
recorded diagnostic rounds (`replay/recorded_diag`, what the Reviewer reads) and the diagnostic pool, and the
original hypotheses and their cost as tag `orig`. `review` puts the incumbent's skills in place and runs the Reviewer
with the loop's own call (`loop.claude`): `plain` is reviewer.md without the digest item, `digest` is reviewer.md with
the digest built from the recorded rounds, `library` adds item 5, an experience library (`--library`), after the
digest. Every review records the absolute paths it named outside this sandbox (`outside`): a replay sandbox must hold
only what the replayed iteration had (build it from that run's own sandbox commit, plus this tooling). `probe` re-runs the incumbent on the diagnostic pool in this scene once
(the baseline) and each distinct parameter set the tags' hypotheses ask for, as the loop's Prober does,
then scores every tag's hypotheses against the baseline. `report` writes `replay/report.json`: per tag the
cost, minutes, turns, tool calls, mean context per turn, hypotheses, and the probes' best and summed gains and
prediction hit rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import batch as B  # noqa: E402
import digest as DG  # noqa: E402
import loop as L  # noqa: E402

DIGEST_ITEM = re.compile(r"^4\. \{DIGEST\}.*\n", re.M)
CONTEXT_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")  # what a turn re-reads
LIBRARY_ITEM = (
    "5. {LIBRARY} -- the experience library: what earlier families established about this robot, its skills and the "
    "scene, one entry per trigger (when it applies; the rule and its numbers; the evidence; its limits). Check a failure "
    "against it before re-deriving its mechanism. It was true when written: where the code or a round disagrees, trust "
    "them and say which entry is wrong.\n"
)
PATH = re.compile(r"(?:/workspace|/home)/[^\s\"'`;|&)]+")


def replay_dir() -> Path:
    return L.LOOP / "replay"


def setup(src: Path, family: str, n: int) -> None:
    L.configure(family)
    src_fam = src / "outputs" / "scene_rsi_20261016" / family
    src_loop = src_fam / "loop"
    it = src_loop / f"iter_{n:02d}"
    ctx_text = (it / "reviewer_context.json").read_text().replace(str(src), str(REPO))
    ctx = json.loads(ctx_text)
    inc = ctx["incumbent"]
    out = replay_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "reviewer_context.json").write_text(ctx_text)
    shutil.copytree(src_fam / "incumbents" / inc["hash"], L.LOOP / "incumbents" / inc["hash"], dirs_exist_ok=True)
    shutil.copytree(src_fam / "diag" / f"incumbent_{inc['hash']}", out / "recorded_diag", dirs_exist_ok=True)
    pool = L.LOOP / "pool_diag"
    if pool.exists():
        shutil.rmtree(pool)
    shutil.copytree(src_fam / "pool_diag", pool)
    orig = out / "orig"
    orig.mkdir(exist_ok=True)
    shutil.copy(it / "hypotheses.json", orig / "hypotheses.json")
    acct = next(e for e in map(json.loads, (src_loop / "ledger.jsonl").read_text().splitlines())
                if e.get("step") == "review" and e.get("iter") == n)  # fmt: skip
    (orig / "acct.json").write_text(json.dumps({k: acct.get(k) for k in ("cost_usd", "seconds", "session", "usage")}))
    (out / "source.json").write_text(json.dumps(dict(sandbox=str(src), family=family, iter=n, incumbent=inc)))
    print(
        f"[replay] {family} iteration {n} from {src}: incumbent {inc['hash']}, {len(list(pool.glob('*.json')))} starts"
    )


def outside_reads(session: str | None, allowed: list[Path]) -> list[str]:
    """Absolute paths a session's tool calls named outside `allowed` (this sandbox, its scratch dir): another checkout's
    docs are knowledge the replayed iteration never had."""
    p = L.transcript(session)
    if p is None:
        return []
    roots, found = [str(a) for a in allowed], set()
    for line in p.open():
        e = json.loads(line)
        if e.get("type") != "assistant":
            continue
        for b in e["message"].get("content") or ():
            if b.get("type") == "tool_use":
                found |= {m for m in PATH.findall(json.dumps(b["input"])) if not m.startswith(tuple(roots))}
    return sorted(found)


def review(family: str, arm: str, tag: str, library: Path | None = None) -> None:
    L.configure(family)
    out = replay_dir()
    ctx = out / "reviewer_context.json"
    inc = json.loads(ctx.read_text())["incumbent"]
    L.put_skills(Path(inc["dir"]) / "skills")
    it_dir = out / tag
    it_dir.mkdir(exist_ok=True)
    base = out / "recorded_diag"
    template = (HERE / "prompts" / "reviewer.md").read_text()
    digest = it_dir / "diag_digest.md"
    if arm in ("digest", "library"):
        digest.write_text(DG.build(base, family))
    else:
        template = DIGEST_ITEM.sub("", template)
    if arm == "library":  # the digest's item, then the library's
        shutil.copy(library, it_dir / "library.md")
        template = DIGEST_ITEM.sub(lambda m: m.group(0) + LIBRARY_ITEM, template)
    common = dict(RULES=L.RULES, PLAN=L.PLAN, LOOP=L.LOOP, SIM_PY=L.SIM_PY, DIAG_INC=base, SKILLS=L.SKILLS,
                  FAMILY=family, FAMILY_FILE=HERE / "families" / f"{family}.json", POOL=L.LOOP / "pool_diag",
                  INC_DIR=inc["dir"], TMP=L.role_tmp(), DIGEST=digest, LIBRARY=it_dir / "library.md")  # fmt: skip
    prompt = L.fill(template, CONTEXT=ctx, **common)
    hyps, acct = L.claude("reviewer", prompt, L.HYPOTHESES_SCHEMA, it_dir, "Edit,Write,NotebookEdit,Agent")
    (it_dir / "hypotheses.json").write_text(json.dumps(hyps, indent=1))
    acct["leaks"] = L.transcript_leaks(acct["session"])
    acct["outside"] = outside_reads(acct["session"], [REPO, L.role_tmp(), Path("/workspace/tmp")])
    acct["arm"] = arm
    (it_dir / "acct.json").write_text(json.dumps(acct, indent=1, default=str))
    print(f"[replay] {tag} ({arm}): {len(hyps['hypotheses'])} hypotheses, {acct['cost_usd']:.2f} USD, "
          f"{acct['seconds'] / 60:.0f} min, leaks {acct['leaks']}, outside reads {acct['outside']}")  # fmt: skip


def delta_key(h: dict) -> str:
    return hashlib.sha1(json.dumps(h["delta"].get("set", {}), sort_keys=True).encode()).hexdigest()[:12]


def probe(family: str, tags: list[str], skip: list[str]) -> None:
    """The loop's probe (loop.probe) once per distinct parameter set over all tags (tags repeat each other's
    deltas), on the diagnostic starts minus those of the tasks in `skip` (tasks the scene no longer has)."""
    L.configure(family)
    out = replay_dir()
    inc = json.loads((out / "reviewer_context.json").read_text())["incumbent"]
    skills = Path(inc["dir"]) / "skills"
    L.put_skills(skills)
    pool = L.LOOP / "pool_diag"
    labels = tuple(x for x in B.pool_labels(pool) if not any(x.startswith(f"{t}_s") for t in skip))
    hyps = {tag: json.loads((out / tag / "hypotheses.json").read_text())["hypotheses"] for tag in tags}
    sets = {delta_key(h): h["delta"].get("set", {}) for hs in hyps.values() for h in hs}
    base, shared = out / "baseline", out / "probes"
    jobs = [B.Job(skills, base, inc["set"], labels)]
    jobs += [B.Job(skills, shared / k, {**inc["set"], **s}, labels) for k, s in sorted(sets.items())]
    (out / "probes.json").write_text(json.dumps(dict(skip=skip, labels=len(labels), sets=sets), indent=1))
    B.run_jobs(jobs, family, pool, L.PROCS, out / "prober.log", L.GPUS)
    recorded = {k: r["ok"] for k, r in B.outcomes(out / "recorded_diag").items() if k in labels}
    before = {k: r["ok"] for k, r in B.outcomes(base).items() if k in labels}
    differ = sorted(k for k in before if recorded.get(k) != before[k])
    print(f"[replay] baseline {sum(before.values())}/{len(before)}, recorded {sum(recorded.values())}/{len(recorded)}, "
          f"rounds that differ {differ}")  # fmt: skip
    for tag, hs in hyps.items():
        res = []
        for h in hs:
            after = {k: r["ok"] for k, r in B.outcomes(shared / delta_key(h)).items() if k in labels}
            if not after:
                res.append(dict(id=h["id"], error="no rounds ran (an unknown parameter? see prober.log)"))
                continue
            fixed = sorted(k for k in after if after[k] and not before.get(k))
            broken = sorted(k for k in after if before.get(k) and not after[k])
            pf, pb = set(h["predicted_fixed"]) & set(labels), set(h["predicted_broken"]) & set(labels)
            res.append(dict(id=h["id"], key=delta_key(h), n=len(after), ok_before=sum(before.values()),
                            ok_after=sum(after.values()), fixed=fixed, broken=broken,
                            predicted_fixed_hit=len(pf & set(fixed)), predicted_fixed_n=len(pf),
                            predicted_broken_hit=len(pb & set(broken)), predicted_broken_n=len(pb),
                            summary=L.summarize(shared / delta_key(h))))  # fmt: skip
        (out / tag / "probe.json").write_text(json.dumps(res, indent=1))
        print(f"[replay] {tag}: " + ", ".join(f"{r['id']} {r.get('ok_before')}->{r.get('ok_after')}" for r in res))


def turns(session: str | None) -> dict:
    """Model turns, tool calls and the context each turn re-read (k tokens), from the session's transcript; `orig`'s
    lives under the finished sandbox's project."""
    p = L.transcript(session) or next(iter((Path.home() / ".claude" / "projects").glob(f"*/{session}.jsonl")), None)
    if not session or p is None:
        return dict(turns=None, tool_calls=None, context_k_mean=None, context_k_max=None)
    seen, ctx, tools = set(), [], 0
    for line in p.open():
        e = json.loads(line)
        if e.get("type") != "assistant":
            continue
        m = e["message"]
        tools += sum(b.get("type") == "tool_use" for b in m.get("content") or ())
        if m.get("id") not in seen:
            seen.add(m.get("id"))
            u = m.get("usage") or {}
            ctx.append(sum(u.get(k, 0) for k in CONTEXT_KEYS))
    return dict(turns=len(ctx), tool_calls=tools, context_k_mean=round(sum(ctx) / len(ctx) / 1000) if ctx else None,
                context_k_max=round(max(ctx) / 1000) if ctx else None)  # fmt: skip


def report(family: str) -> None:
    L.configure(family)
    out = replay_dir()
    rows = {}
    for d in sorted(p for p in out.iterdir() if (p / "acct.json").exists()):
        acct = json.loads((d / "acct.json").read_text())
        hyps = json.loads((d / "hypotheses.json").read_text())
        pr = json.loads((d / "probe.json").read_text()) if (d / "probe.json").exists() else []
        gains = [r["ok_after"] - r["ok_before"] for r in pr if "ok_after" in r]
        hit = sum(r.get("predicted_fixed_hit", 0) for r in pr)
        pred = sum(r.get("predicted_fixed_n", 0) for r in pr)
        rows[d.name] = dict(
            arm=acct.get("arm", "plain (original)"), cost_usd=round(float(acct.get("cost_usd") or 0), 2),
            minutes=round(float(acct.get("seconds") or 0) / 60, 1), **turns(acct.get("session")),
            hypotheses=len(hyps["hypotheses"]), claims=len(hyps.get("infeasible") or []),
            probe_ok_before=pr[0]["ok_before"] if pr and "ok_before" in pr[0] else None,
            best_gain=max(gains) if gains else None, gains=gains, predicted_fixed_hit=f"{hit}/{pred}",
            leaks=acct.get("leaks"), outside=acct.get("outside"), claims_text=[h["claim"][:160] for h in hyps["hypotheses"]],
        )  # fmt: skip
    (out / "report.json").write_text(json.dumps(rows, indent=1))
    for k, r in rows.items():
        print(k, {x: r[x] for x in ("arm", "cost_usd", "minutes", "turns", "tool_calls", "context_k_mean", "hypotheses",
                                    "best_gain", "gains", "predicted_fixed_hit")})  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("src", type=Path)
    s.add_argument("family")
    s.add_argument("iter", type=int)
    r = sub.add_parser("review")
    r.add_argument("family")
    r.add_argument("--arm", choices=("plain", "digest", "library"), required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--library", type=Path, help="the experience library (arm library)")
    p = sub.add_parser("probe")
    p.add_argument("family")
    p.add_argument("tags", nargs="+")
    p.add_argument("--skip", default="", help="comma-separated tasks to leave out (gone from the scene)")
    q = sub.add_parser("report")
    q.add_argument("family")
    a = ap.parse_args(argv)
    if a.cmd == "setup":
        setup(a.src, a.family, a.iter)
    elif a.cmd == "review":
        if a.arm == "library" and a.library is None:
            ap.error("--arm library needs --library")
        review(a.family, a.arm, a.tag, a.library.resolve() if a.library else None)
    elif a.cmd == "probe":
        probe(a.family, a.tags, [t for t in a.skip.split(",") if t])
    else:
        report(a.family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
