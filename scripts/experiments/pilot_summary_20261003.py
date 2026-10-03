"""Round-1 envelope pilots (TODO 9.9): per pilot, success and where each failure happened, from the run manifests.

    python3 scripts/experiments/pilot_summary_20261003.py outputs/runs/sd_pilot_*_20261003b > summary.md

Reads `<run>/skills/manifest.jsonl` and the per-round traces; prints one row per pilot (ok / rounds, approach ok,
grasp ok) and the failure reasons grouped by their `[tag]` prefix and first clause.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def stage_ok(trace: Path, name: str) -> bool | None:
    if not trace.exists():
        return None
    for line in trace.read_text().splitlines():
        r = json.loads(line)
        if r.get("primitive") == name:
            return bool((r.get("outcome") or {}).get("ok"))
    return None


def main() -> None:
    print("| pilot | OK | approach OK | grasp OK | failures |\n|---|---|---|---|---|")
    for run in sorted(Path(p) for p in sys.argv[1:]):
        rows = [json.loads(line) for line in (run / "skills/manifest.jsonl").read_text().splitlines() if line.strip()]
        traces = [run / f"skills/agent_trace_{r['round']:02d}.jsonl" for r in rows]
        approach = [stage_ok(t, "approach") for t in traces]
        grasp = [stage_ok(t, "grasp_single") for t in traces]
        fails = collections.Counter(r["why"].split(":")[0][:70] for r in rows if not r["ok"])
        n = len(rows)
        print(
            f"| {run.name} | {sum(r['ok'] for r in rows)}/{n} | {sum(bool(a) for a in approach)}/{n} "
            f"| {sum(bool(g) for g in grasp)}/{sum(a is True for a in approach)} | "
            + "; ".join(f"{k} x{v}" for k, v in fails.most_common())
            + " |"
        )


if __name__ == "__main__":
    main()
