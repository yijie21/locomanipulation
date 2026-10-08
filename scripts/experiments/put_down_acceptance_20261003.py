"""put_down_single acceptance (TODO 19.12.2, docs/handoff/put-down-single-20261003.md §3): from the run traces.

    python3 scripts/experiments/put_down_acceptance_20261003.py outputs/runs/sd_put_down_p4_*

Counts over every round of the given runs, in order: rounds, grasps that held, and of the first 20 rounds with a
held grasp how many also got through `approach(surface)` and `put_down_single` (the gate: >= 10). Prints the
put-down failures by `[tag]` and first clause, and per successful put-down the squat, the landing, the tilt and the
offset from the spot.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

GATE_ROUNDS, GATE_OK = 20, 10


def calls(trace: Path) -> list[dict]:
    return [json.loads(line) for line in trace.read_text().splitlines() if line.strip()] if trace.exists() else []


def main() -> None:
    held, fails, ok_rows, rounds = [], collections.Counter(), [], 0
    for run in sorted(Path(p) for p in sys.argv[1:]):
        for line in (run / "skills/manifest.jsonl").read_text().splitlines():
            row = json.loads(line)
            rounds += 1
            by = {c.get("primitive"): c for c in calls(run / f"skills/agent_trace_{row['round']:02d}.jsonl")}
            grasp = (by.get("grasp_single") or {}).get("outcome") or {}
            if not grasp.get("ok"):
                continue
            tail = [c for c in calls(run / f"skills/agent_trace_{row['round']:02d}.jsonl")][2:]
            ok = bool(row["ok"]) and all((c.get("outcome") or {}).get("ok") for c in tail)
            held.append(ok)
            if ok:
                m = (by.get("put_down_single") or {}).get("outcome", {}).get("metrics", {})
                ok_rows.append((run.name, row["round"], m))
            else:
                bad = next((c for c in tail if "rejected" in c or not (c.get("outcome") or {}).get("ok")), None)
                why = row.get("why") or ""
                if bad is not None:
                    why = (bad.get("outcome") or {}).get("reason") or json.dumps(bad.get("rejected"))[:70]
                where = (bad or {}).get("primitive", "criterion")
                fails[f"{where}: {why.split(':')[0][:70]}"] += 1
    gate = held[:GATE_ROUNDS]
    print(f"rounds {rounds}; grasp held {len(held)}; approach(surface)+put_down ok {sum(held)}/{len(held)}")
    print(
        f"gate: first {len(gate)} held grasps -> {sum(gate)} ok (need {GATE_OK} of {GATE_ROUNDS})"
        + ("" if len(gate) == GATE_ROUNDS else "  [incomplete]")
    )
    for k, v in fails.most_common():
        print(f"  fail x{v}: {k}")
    for run, rnd, m in ok_rows:
        print(
            f"  ok {run} r{rnd}: squat {m.get('squat_m', 0):.3f} landed {m.get('landed')} "
            f"tilt {m.get('tilt_deg', float('nan')):.1f} deg offset {m.get('final_offset', float('nan')) * 100:.1f} cm"
        )


if __name__ == "__main__":
    main()
