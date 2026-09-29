#!/usr/bin/env python3
"""Pair a fixed-plan run with an agent run that got the same failure injection (agentic-plan Q11).

    $VENV_SIM_PY scripts/injection_report.py --plan <run> --agent <run> [--out outputs/analysis/injection_<date>]

Both runs are `scripts/plan_episode.py --inject <same file>` on the same `--seed`, one with `--plan`, one with
`--agent`. Rounds are paired by seed; only rounds where both arms actually fired the injection count. The
report gives the paired OK counts, the discordant pairs with an exact two-sided sign test, and the agent's
recovery segments (its calls after the event) grouped by the first thing it did. Writes `report.md` and
`pairs.json` under `--out`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _run_dir(name: str) -> Path:
    p = Path(name)
    return p if p.is_dir() else REPO / "outputs" / "runs" / name


def _rows(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "skills" / "manifest.jsonl").read_text().splitlines() if line.strip()]


def _traces(run: Path) -> dict[int, list[dict]]:
    out = {}
    for path in sorted((run / "skills").glob("agent_trace_*.jsonl")):
        rnd = int(path.stem.rsplit("_", 1)[1])
        out[rnd] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def main(argv=None) -> int:
    from locomani.skills.injection import paired_summary, render_report

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="run name or directory of the fixed-plan arm")
    ap.add_argument("--agent", required=True, help="run name or directory of the agent arm")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    plan, agent = _run_dir(args.plan), _run_dir(args.agent)
    plan_rows, agent_rows = _rows(plan), _rows(agent)
    pairs, summary = paired_summary(plan_rows, agent_rows, _traces(agent))
    injection = next((r["injection"]["injection"] for r in agent_rows if r.get("injection")), None)
    out = args.out or REPO / "outputs" / "analysis" / f"injection_{time.strftime('%Y%m%d')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(
        render_report(pairs, summary, plan_run=plan.name, agent_run=agent.name, injection=injection)
    )
    (out / "pairs.json").write_text(json.dumps({"summary": summary, "pairs": pairs}, indent=1))
    print(
        f"[report] {out / 'report.md'}: plan {summary['plan_ok']}/{summary['pairs_both_injected']} "
        f"agent {summary['agent_ok']}/{summary['pairs_both_injected']} p={summary['sign_test_p']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
