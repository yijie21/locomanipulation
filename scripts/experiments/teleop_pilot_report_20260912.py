"""Paired report for the teleop-channel pilots: per case, per arm, plus the
grasp-phase telemetry the channel makes available (delivered squat retreat and
pitch against NO prediction, and the servo residual at the capture).

    python3 scripts/experiments/teleop_pilot_report_20260912.py \
        outputs/experiments/teleop_pilot_20260912/sonic_v1_1_run5 \
        outputs/experiments/teleop_pilot_20260912/release_run5

Case 0 of `cases10_az374.json` fails `[stand] could not turn -19 deg` on EVERY
arm measured so far (v1.1 coupled twice, release coupled, teleop), so it is a
property of the bank; the report prints both the raw and the reachable-only
denominators rather than quietly dropping it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(run_dir: Path) -> list[dict]:
    path = run_dir / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def short(why: str) -> str:
    why = (why or "").strip()
    if not why:
        return "ok"
    return why[:58]


def telemetry(row: dict) -> dict:
    ex = row.get("execution") or {}
    metrics = ex.get("metrics") or {}
    ct = (metrics.get("coupled_transport") or [{}])[-1]
    pose = ct.get("squat_delivered_pose") or {}
    return dict(
        retreat_m=pose.get("retreat_m"),
        drop_m=pose.get("drop_m"),
        residual_m=pose.get("residual_m"),
        align_s=ct.get("alignment_s"),
        lifted_m=ct.get("lifted_m"),
    )


def main() -> None:
    runs = [Path(a) for a in sys.argv[1:]]
    if not runs:
        raise SystemExit(__doc__)
    tables = {r.name: {row.get("case_id"): row for row in load(r)} for r in runs}
    cases = sorted({c for t in tables.values() for c in t})
    names = list(tables)

    print(f"{'case':<10}" + "".join(f"{n:<24}" for n in names))
    for case in cases:
        cells = []
        for n in names:
            row = tables[n].get(case)
            cells.append("-" if row is None else ("OK" if row.get("ok") else short(row.get("why"))[:22]))
        print(f"{case:<10}" + "".join(f"{c:<24}" for c in cells))

    print()
    for n in names:
        rows = list(tables[n].values())
        ok = [r for r in rows if r.get("ok")]
        stand = [r for r in rows if not r.get("ok") and str(r.get("why", "")).startswith("[stand]")]
        reach = len(rows) - len(stand)
        produced = [r for r in ok if r.get("produced")]
        print(
            f"{n}: {len(ok)}/{len(rows)} raw, {len(ok)}/{reach} of the rounds that got past the spawn, {len(produced)} episodes recorded"
        )
        tel = [telemetry(r) for r in ok]
        ret = [t["retreat_m"] for t in tel if t["retreat_m"] is not None]
        res = [t["residual_m"] for t in tel if t["residual_m"] is not None]
        lift = [t["lifted_m"] for t in tel if t["lifted_m"] is not None]
        al = [t["align_s"] for t in tel if t["align_s"] is not None]
        if ret:
            print(
                f"   delivered squat retreat {min(ret) * 100:.1f}-{max(ret) * 100:.1f} cm (nothing predicts it on this channel)"
            )
        if res:
            print(f"   wrist residual at the end of the squat {min(res) * 100:.1f}-{max(res) * 100:.1f} cm")
        if al:
            print(f"   local align {min(al):.1f}-{max(al):.1f} s")
        if lift:
            print(f"   lifted {min(lift) * 100:.1f}-{max(lift) * 100:.1f} cm")
        fails: dict[str, int] = {}
        for r in rows:
            if not r.get("ok"):
                fails[short(r.get("why"))] = fails.get(short(r.get("why")), 0) + 1
        for why, k in sorted(fails.items(), key=lambda kv: -kv[1]):
            print(f"   {k}x {why}")

    if len(names) == 2:
        a, b = names
        disc = [(c, tables[a].get(c, {}).get("ok"), tables[b].get(c, {}).get("ok")) for c in cases]
        ab = sum(1 for _, x, y in disc if x and not y)
        ba = sum(1 for _, x, y in disc if y and not x)
        print(f"\npaired, same cases: {a} only {ab}, {b} only {ba} (discordant {ab}:{ba})")
        print(
            "10 rounds cannot resolve a small difference on this line -- memory ten-round-pilots-cannot-measure-this-line"
        )


if __name__ == "__main__":
    main()
