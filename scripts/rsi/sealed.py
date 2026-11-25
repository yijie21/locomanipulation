"""TODO 27 (plan docs/handoff/scene-rsi-plan-20261016.md D8): a family's sealed pool, drawn only after its inner loop
ends, run under the seed and under the gate's accepted incumbents (each one's skills snapshot and overrides, exactly
as the gate judged it). The final incumbent and the seed run first; with --all the intermediate incumbents follow, for
the growth curve. Rounds land in <hidden>/sealed/<name>/ (a rerun skips rounds already written); the summary --
per-version and per-task wins with their 95% Wilson intervals, falls and drops, the final against the seed paired by
start -- goes to
<hidden>/sealed/summary.json. D8: a family whose final incumbent wins >= 50% of the sealed pool enters bucket 1.
`confirm` (TODO 48.19.2, user 2026-11-23): the final incumbent is confirmed when, paired by start against the seed, it
wins more than it loses, the one-sided sign test p is at most the family's p_max (0.05 without one), and no task's
loss is significant at caps["alpha_task"] (verdict.py, when the family has that cap). Every run appends one line (final, confirmed, counts)
to <hidden>/sealed/history.jsonl, so a second evaluation of a later final is visible as such. The loop runs this file
by itself when a run ends (any stop, TODO 54) of a family with caps sealed_at_stop (loop.run_sealed).

Run it in the loop's sandbox checkout, detached (nohup setsid), like the gate:
    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/sealed.py pick_single [--all] [--procs 12 --gpus 0,1]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import backend as BK  # noqa: E402
import batch as B  # noqa: E402
import verdict as V  # noqa: E402
from gate import mcnemar_one_sided, overrides, wilson  # noqa: E402

BUCKET1_RATE = 0.50


def versions(gate: Path) -> list[tuple[str, Path]]:
    """[(name, candidate dir)]: the seed, then every accepted incumbent in gate order."""
    out = [("seed", gate / "incumbent_seed")]
    for line in (gate / "ledger.jsonl").read_text().splitlines():
        v = json.loads(line)
        if v.get("accepted") and not v.get("dry_run"):
            out.append((f"{v['call']:02d}_{v['hash']}", gate / f"{v['call']:02d}_{v['hash']}" / "cand"))
    return out


def tally(rounds: dict) -> dict:
    per = collections.defaultdict(lambda: [0, 0])
    for r in rounds.values():
        per[r["task"]][0] += bool(r["ok"])
        per[r["task"]][1] += 1
    ok = sum(r["ok"] for r in rounds.values())
    return dict(ok=ok, n=len(rounds), ci=wilson(ok, len(rounds)), fell=sum(r["fell"] for r in rounds.values()),
                dropped=sum(r["dropped"] for r in rounds.values()),
                per_task={k: f"{a}/{n}" for k, (a, n) in sorted(per.items())},
                per_task_ci={k: wilson(a, n) for k, (a, n) in sorted(per.items())})  # fmt: skip


def confirm(final: dict, seed: dict, p_max: float | None, alpha_task: float | None) -> dict:
    """The final incumbent against the seed, paired by start: confirmed, the reasons it is not, and per-task counts
    (verdict.decide, kind "sealed", TODO 54: the sign test at the family's p_max, 0.05 without one, and the per-task
    test at `alpha_task`, Holm across the tasks)."""
    v = V.judge(final, seed, "sealed", dict(p_max=p_max if p_max is not None else 0.05, alpha_task=alpha_task))
    return dict(confirmed=v["decision"], why=v["reasons"], n=v["n"], b=v["b"], c=v["c"],
                p_one_sided=round(v["p_one_sided"], 5), per_task=v["per_task"])  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("--all", action="store_true", help="also run the intermediate incumbents")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("RSI_PROCS", "12")))
    ap.add_argument("--gpus", default=os.environ.get("RSI_GPUS", "0,1"))
    a = ap.parse_args(argv)
    hidden = REPO / "outputs" / "scene_rsi_20261016_hidden" / a.family
    pool, out = hidden / "pool_sealed", hidden / "sealed"
    if not pool.exists():
        res = BK.draw(BK.load(a.family), "sealed", pool)
        print(f"sealed pool: {res['written']} starts, unavailable {res['unavailable']}")
    vs = versions(hidden / "gate")
    # TODO 36 (review 2026-10-22): nothing accepted -> vs[-1] is the seed; it ran twice as [seed, seed]
    order = list(dict.fromkeys([vs[-1], vs[0]] + (vs[1:-1] if a.all else [])))
    jobs = [B.Job(d / "skills", out / name, overrides(d)) for name, d in order]
    # where="gate": the sealed starts are hidden rows (the full practice files) and the final evaluation runs in the
    # gate's environment (bed_env_gate: cameras rendered every step, user 2026-11-23)
    missing = B.run_jobs(jobs, a.family, pool, a.procs, out / "run.log", a.gpus, where="gate")
    summary = dict(pool=str(pool), versions={name: tally(B.outcomes(out / name)) for name, _ in order},
                   missing={order[k][0]: v for k, v in missing.items() if v})  # fmt: skip
    scene = BK.of(a.family) is None
    if scene:  # TODO 33.1.2: the body model's predictions against the delivered reaches, beside the success rate
        summary["prediction"] = {name: _score(B.outcomes(out / name).values()) for name, _ in order}
    final, seed = B.outcomes(out / order[0][0]), B.outcomes(out / "seed")
    keys = sorted(set(final) & set(seed))
    b = sum(final[k]["ok"] and not seed[k]["ok"] for k in keys)
    c = sum(seed[k]["ok"] and not final[k]["ok"] for k in keys)
    rate = sum(final[k]["ok"] for k in keys) / max(len(keys), 1)
    k_ok = sum(final[k]["ok"] for k in keys)
    summary.update(final=order[0][0], final_vs_seed=dict(n=len(keys), b=b, c=c, p_one_sided=mcnemar_one_sided(b, c)),
                   final_rate=round(rate, 3), final_rate_ci=wilson(k_ok, len(keys)))  # fmt: skip
    if scene:
        summary["bucket1"] = rate >= BUCKET1_RATE
    caps = BK.caps(a.family)
    summary["confirm"] = confirm(final, seed, caps.get("p_max"), caps.get("alpha_task"))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    hist = out / "history.jsonl"
    before = len(hist.read_text().splitlines()) if hist.exists() else 0
    with hist.open("a") as fh:
        fh.write(json.dumps(dict(final=order[0][0], evaluation=before + 1, confirmed=summary["confirm"]["confirmed"],
                                 why=summary["confirm"]["why"], b=b, c=c, final_rate=summary["final_rate"])) + "\n")  # fmt: skip
    print(json.dumps(summary, indent=1))
    return 0


def _score(rounds):
    """body.score for the scene backend's rounds; None where the records have no reach predictions (other backends)."""
    try:
        from locomani.tasks.scene import body

        return body.score(rounds)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
