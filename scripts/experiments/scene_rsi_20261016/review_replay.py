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
the digest built from the recorded rounds. `probe` re-runs the incumbent on the diagnostic pool in this scene once
(the baseline) and each tag's hypotheses as the loop's Prober does. `report` writes `replay/report.json`: per tag the
cost, minutes, tool calls, hypotheses, and the probes' best and summed gains and prediction hit rates.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import digest as DG  # noqa: E402
import loop as L  # noqa: E402

DIGEST_ITEM = re.compile(r"^4\. \{DIGEST\}.*\n", re.M)


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


def review(family: str, arm: str, tag: str) -> None:
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
    if arm == "digest":
        digest.write_text(DG.build(base, family))
    else:
        template = DIGEST_ITEM.sub("", template)
    common = dict(RULES=L.RULES, PLAN=L.PLAN, LOOP=L.LOOP, SIM_PY=L.SIM_PY, DIAG_INC=base, SKILLS=L.SKILLS,
                  FAMILY=family, FAMILY_FILE=HERE / "families" / f"{family}.json", POOL=L.LOOP / "pool_diag",
                  INC_DIR=inc["dir"], TMP=L.role_tmp(), DIGEST=digest)  # fmt: skip
    prompt = L.fill(template, CONTEXT=ctx, **common)
    hyps, acct = L.claude("reviewer", prompt, L.HYPOTHESES_SCHEMA, it_dir, "Edit,Write,NotebookEdit,Agent")
    (it_dir / "hypotheses.json").write_text(json.dumps(hyps, indent=1))
    acct["leaks"] = L.transcript_leaks(acct["session"])
    acct["arm"] = arm
    (it_dir / "acct.json").write_text(json.dumps(acct, indent=1, default=str))
    print(f"[replay] {tag} ({arm}): {len(hyps['hypotheses'])} hypotheses, {acct['cost_usd']:.2f} USD, "
          f"{acct['seconds'] / 60:.0f} min, leaks {acct['leaks']}")  # fmt: skip


def probe(family: str, tags: list[str]) -> None:
    L.configure(family)
    out = replay_dir()
    inc = json.loads((out / "reviewer_context.json").read_text())["incumbent"]
    L.put_skills(Path(inc["dir"]) / "skills")
    base_dir = out / "baseline"
    base_dir.mkdir(exist_ok=True)
    base = L.diag_incumbent(inc, base_dir)  # this scene's incumbent rounds on the same starts
    for tag in tags:
        hyps = json.loads((out / tag / "hypotheses.json").read_text())["hypotheses"]
        res = L.probe(hyps, inc, base, out / tag)
        (out / tag / "probe.json").write_text(json.dumps(res, indent=1))
        print(f"[replay] {tag}: " + ", ".join(f"{r['id']} {r.get('ok_before')}->{r.get('ok_after')}" for r in res))


def tool_calls(session: str | None) -> int | None:
    p = L.transcript(session)
    if p is None:
        return None
    return sum(line.count('"type":"tool_use"') + line.count('"type": "tool_use"') for line in p.open())


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
            minutes=round(float(acct.get("seconds") or 0) / 60, 1), tool_calls=tool_calls(acct.get("session")),
            hypotheses=len(hyps["hypotheses"]), claims=len(hyps.get("infeasible") or []),
            probe_ok_before=pr[0]["ok_before"] if pr and "ok_before" in pr[0] else None,
            best_gain=max(gains) if gains else None, gains=gains, predicted_fixed_hit=f"{hit}/{pred}",
            leaks=acct.get("leaks"), claims_text=[h["claim"][:160] for h in hyps["hypotheses"]],
        )  # fmt: skip
    (out / "report.json").write_text(json.dumps(rows, indent=1))
    for k, r in rows.items():
        print(k, {x: r[x] for x in ("arm", "cost_usd", "minutes", "tool_calls", "hypotheses", "best_gain", "gains")})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("src", type=Path)
    s.add_argument("family")
    s.add_argument("iter", type=int)
    r = sub.add_parser("review")
    r.add_argument("family")
    r.add_argument("--arm", choices=("plain", "digest"), required=True)
    r.add_argument("--tag", required=True)
    p = sub.add_parser("probe")
    p.add_argument("family")
    p.add_argument("tags", nargs="+")
    q = sub.add_parser("report")
    q.add_argument("family")
    a = ap.parse_args(argv)
    if a.cmd == "setup":
        setup(a.src, a.family, a.iter)
    elif a.cmd == "review":
        review(a.family, a.arm, a.tag)
    elif a.cmd == "probe":
        probe(a.family, a.tags)
    else:
        report(a.family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
