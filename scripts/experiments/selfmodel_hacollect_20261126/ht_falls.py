"""selfmodel_hatask_20261126 revision 5 (descriptive, enters no decision): HA falls per task and arm, the phase each
fall happened in, and how the falls pair up by seed. Reads only the episode logs (fell_step, phase_at_end, end).

    python3 ht_falls.py <formal>/<ctrl>_ppbox,<formal>/<ctrl>_doubledesk

A fall is an episode whose log has fell_step set (HA's fall detector, confirmed 5 steps; the episode ends there, so
phase_at_end is the phase the fall happened in). Episodes that ended in error (end starting with "error") are left out,
as in ht_formal_stats.py.
"""

import collections
import json
import sys
from pathlib import Path


def logs(run: Path, arm: str) -> dict:
    out = {}
    for f in sorted((run / arm / "log").glob("*_r0.json")):
        lg = json.loads(f.read_text())
        if str(lg.get("end", "")).startswith("error"):
            continue
        out[int(f.stem.split("_s")[1].split("_")[0])] = lg
    return out


def main(runs: list[Path]) -> None:
    for run in runs:
        task = run.name.split("_", 1)[1]
        la, lr = logs(run, "A"), logs(run, "R")
        seeds = sorted(set(la) & set(lr))
        for arm, lg in (("A", la), ("R", lr)):
            fell = [s for s in seeds if lg[s].get("fell_step") is not None]
            ph = collections.Counter(lg[s].get("phase_at_end") for s in fell)
            groups = [sum(lo <= s < lo + 10 for s in fell) for lo in range(min(seeds) // 10 * 10, max(seeds) + 1, 10)] if seeds else []
            print(f"{task} {arm}: falls {len(fell)}/{len(seeds)}; phase at the fall {dict(ph.most_common())}; per 10 seeds {groups}")
        fa = {s for s in seeds if la[s].get("fell_step") is not None}
        fr = {s for s in seeds if lr[s].get("fell_step") is not None}
        print(f"{task} pairs {len(seeds)}: both fell {len(fa & fr)}, A only {len(fa - fr)}, R only {len(fr - fa)}, neither {len(seeds) - len(fa | fr)}")


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1].split(",") if p])
