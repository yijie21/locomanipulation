"""Pair the two checkpoint arms case by case and report where they differ.

Ten rounds cannot resolve a success-rate difference, so the point here is the
paired per-case outcome plus the continuous quantities (where the robot parked,
how long the round took), which ten rounds CAN say something about.
"""

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("V11_OUT", ROOT / "outputs/experiments/v11_bottle_pilot_20260910"))
ARMS = ["sonic_v1_1", "release"]


def load(arm):
    path = OUT / arm / "manifest.jsonl"
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            rows.setdefault(r["case_id"], r)  # first attempt per case
    return rows


def num(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    return dict(
        n=len(vals),
        mean=round(statistics.mean(vals), 4),
        sd=round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
    )


def main():
    arms = {a: load(a) for a in ARMS}
    cases = sorted({c for rows in arms.values() for c in rows}, key=lambda c: int(c.split(":")[1]))
    if not cases:
        print("no manifests yet")
        return 1

    print(f"{'case':10} " + " ".join(f"{a:<34}" for a in ARMS))
    agree = {"both_ok": 0, "both_fail": 0}
    only = {a: 0 for a in ARMS}
    for c in cases:
        cells = []
        for a in ARMS:
            r = arms[a].get(c)
            cells.append("-" if r is None else ("OK" if r["ok"] else f"FAIL {r['why']}")[:34].ljust(34))
        print(f"{c:10} " + " ".join(cells))
        got = {a: (arms[a].get(c) or {}).get("ok") for a in ARMS}
        if all(v is not None for v in got.values()):
            if all(got.values()):
                agree["both_ok"] += 1
            elif not any(got.values()):
                agree["both_fail"] += 1
            else:
                only[ARMS[0] if got[ARMS[0]] else ARMS[1]] += 1

    print()
    for a in ARMS:
        rows = list(arms[a].values())
        ok = sum(1 for r in rows if r["ok"])
        kept = sum(1 for r in rows if r.get("produced"))
        why = {}
        for r in rows:
            if not r["ok"]:
                why[r["why"]] = why.get(r["why"], 0) + 1
        print(f"{a:12} {ok}/{len(rows)} ok, {kept} episodes kept   failures: {why or '-'}")
        print(f"{'':12} park      {num([r.get('park') for r in rows])}")
        print(f"{'':12} park_first{num([r.get('park_first') for r in rows])}")
    print()
    print(
        f"paired over {sum(agree.values()) + sum(only.values())} shared cases: "
        f"both ok {agree['both_ok']}, both fail {agree['both_fail']}, " + ", ".join(f"only {a} {only[a]}" for a in ARMS)
    )
    print(
        "Ten paired rounds cannot resolve a success-rate difference; read the "
        "failure kinds and the park numbers, not the ratio."
    )
    (OUT / "report.json").write_text(
        json.dumps(dict(cases={c: {a: arms[a].get(c) for a in ARMS} for c in cases}, agree=agree, only=only), indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
