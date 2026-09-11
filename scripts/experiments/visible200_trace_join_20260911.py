#!/usr/bin/env python3
"""Re-derive ledger bottle.md section 18.1 on a CORRECT trace/manifest join.

The visible200 eval wrote its traces from the VLA client's own call counter, so
`traces/round_NNN.json` describes manifest round NNN+1 (manifest round 0 died in
`[stand]`, before the VLA stage, and the counter never saw it). Every per-round
claim in section 18.1 that joined the two by round number was reading one round's
gripper data against the next round's geometry.

This script joins them correctly -- by the `round` field when the trace carries
one (written since the 2026-09-11 fix), otherwise by finding the single integer
offset that makes every `why` string agree -- and recomputes the section's four
claims:

  1. how many rounds the model never predicted a close in
  2. whether the rounds that did predict one succeeded
  3. whether a predicted close ever failed to reach the executed stream
  4. whether stop distance and azimuth separate success from failure

Usage:
  python3 scripts/experiments/visible200_trace_join_20260911.py [EVAL_DIR] [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = REPO / "outputs/experiments/visible200_20260910/eval"
CLOSE_THRESHOLD = 0.005  # vla_round.py: a frame below this counts as a commanded close


def load(eval_dir: Path):
    manifest = {}
    with (eval_dir / "closed_loop.manifest.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            manifest[r["round"]] = r
    traces = {}
    for p in sorted(glob.glob(str(eval_dir / "traces/round_*.json"))):
        traces[int(os.path.basename(p)[6:9])] = json.load(open(p))
    return manifest, traces


def resolve_offset(manifest: dict, traces: dict) -> tuple[int, str]:
    """The integer k with traces[n] == manifest[n + k], proven on the `why` strings."""
    if traces and all("round" in t for t in traces.values()):
        offs = {t["round"] - n for n, t in traces.items()}
        if len(offs) == 1:
            return offs.pop(), "trace files carry their own `round` field"
    best = None
    for k in range(-5, 6):
        hits = sum(
            1
            for n, t in traces.items()
            if n + k in manifest and (manifest[n + k].get("why") or "") == (t.get("why") or "")
        )
        if best is None or hits > best[1]:
            best = (k, hits)
    k, hits = best
    if hits != len(traces):
        raise SystemExit(f"no offset aligns every trace: best k={k} matched {hits}/{len(traces)} `why` strings")
    return k, f"offset {k:+d} proven on {hits}/{len(traces)} exact `why` matches"


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def stop_geometry(trace: dict) -> tuple[float | None, float | None]:
    """(closest approach in m, azimuth to the object in deg) at the closest row.

    Azimuth is measured at the moment the robot is nearest the bottle, from its
    own heading -- the stop pose the policy actually chose, not the sampled stand
    the manifest recorded before the round ran.
    """
    rows = [r for r in (trace.get("trace") or []) if r.get("dist") is not None]
    if not rows:
        return None, None
    near = min(rows, key=lambda r: r["dist"])
    px, py = near["pelvis_xy"]
    ox, oy = near["obj_xy"]
    az = wrap(math.atan2(oy - py, ox - px) - near["pelvis_yaw"])
    return float(near["dist"]), math.degrees(az)


def summarise(values, label):
    v = sorted(x for x in values if x is not None)
    if not v:
        return f"  {label:<28} n=0"
    q = lambda f: v[min(len(v) - 1, int(f * len(v)))]  # noqa: E731
    return (
        f"  {label:<28} n={len(v):<3} p10={q(0.1):+7.3f} median={statistics.median(v):+7.3f} "
        f"p90={q(0.9):+7.3f} min={v[0]:+7.3f} max={v[-1]:+7.3f}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", nargs="?", default=str(DEFAULT_EVAL))
    ap.add_argument("--json", dest="out_json")
    args = ap.parse_args(argv)
    eval_dir = Path(args.eval_dir)

    manifest, traces = load(eval_dir)
    k, how = resolve_offset(manifest, traces)
    print(f"eval dir : {eval_dir}")
    print(f"manifest : {len(manifest)} rounds     traces: {len(traces)} files")
    print(f"JOIN     : traces[n] == manifest[n{k:+d}]   ({how})")
    if k:
        print(f"           ^ a naive join by round number is wrong by {k:+d} for EVERY round")
    print()

    rows = []
    for n, t in sorted(traces.items()):
        m = manifest.get(n + k)
        if m is None:
            continue
        chunk = t.get("latency", {}).get("chunk_grip_min") or []
        steps = t.get("trace") or []
        executed = [r["grip"] for r in steps if r.get("grip") is not None]
        dist, az = stop_geometry(t)
        rows.append(
            dict(
                trace_file=n,
                round=n + k,
                ok=bool(m["ok"]),
                why=m.get("why") or "",
                chunk_min=min(chunk) if chunk else None,
                predicted_close=bool(chunk) and min(chunk) < CLOSE_THRESHOLD,
                first_predicted_step=next((i for i, v in enumerate(chunk) if v < CLOSE_THRESHOLD), None),
                n_infer=len(chunk),
                executed_min=min(executed) if executed else None,
                executed_close=bool(executed) and min(executed) < CLOSE_THRESHOLD,
                stop_dist_m=dist,
                stop_azimuth_deg=az,
            )
        )

    pred = [r for r in rows if r["predicted_close"]]
    never = [r for r in rows if not r["predicted_close"]]
    ok = [r for r in rows if r["ok"]]

    print("CLAIM 1 -- how often does the model command a close?")
    print(
        f"  never predicted a close : {len(never):3} of {len(rows)}   (of which succeeded: {sum(r['ok'] for r in never)})"
    )
    print(
        f"  predicted a close       : {len(pred):3} of {len(rows)}   (of which succeeded: {sum(r['ok'] for r in pred)})"
    )
    print()

    print("CLAIM 2/3 -- the rounds that predicted a close, and whether it was executed")
    for r in sorted(pred, key=lambda r: r["chunk_min"]):
        gap = "" if r["executed_close"] else "   <-- predicted but NEVER executed"
        print(
            f"  round {r['round']:03d} (trace_file {r['trace_file']:03d}) ok={str(r['ok']):5} "
            f"chunk_min={r['chunk_min']:+.4f} executed_min={r['executed_min']:+.4f} "
            f"first@{r['first_predicted_step']}/{r['n_infer']}{gap}"
        )
        print(f"      why: {r['why'][:96]}")
    print()

    print("CLAIM 4 -- do stop distance and azimuth separate success from failure?")
    print(" by outcome:")
    print(summarise([r["stop_dist_m"] for r in ok], "stop distance m  (success)"))
    print(summarise([r["stop_dist_m"] for r in rows if not r["ok"]], "stop distance m  (failure)"))
    print(summarise([r["stop_azimuth_deg"] for r in ok], "stop azimuth deg (success)"))
    print(summarise([r["stop_azimuth_deg"] for r in rows if not r["ok"]], "stop azimuth deg (failure)"))
    print(" by whether the model decided to close:")
    print(summarise([r["stop_dist_m"] for r in pred], "stop distance m  (closed)"))
    print(summarise([r["stop_dist_m"] for r in never], "stop distance m  (never closed)"))
    print(summarise([r["stop_azimuth_deg"] for r in pred], "stop azimuth deg (closed)"))
    print(summarise([r["stop_azimuth_deg"] for r in never], "stop azimuth deg (never closed)"))
    print()

    print("GRIPPER CHANNEL RANGE (the model's own output, not the robot's)")
    print(summarise([r["chunk_min"] for r in never], "chunk min (never closed)"))
    print(summarise([r["chunk_min"] for r in pred], "chunk min (closed)"))
    print("  reference: GRIPPER_MIN -0.0200  GRIPPER_MAX +0.0245  close threshold < +0.0050")
    print()

    print("FAILURE STRINGS vs WHAT THE MODEL ACTUALLY DID")
    tally = {}
    for r in rows:
        key = re.sub(r"\(min [0-9.]+ m\)", "(min ...)", r["why"])[:62]
        t = tally.setdefault(key, [0, 0])
        t[0] += 1
        t[1] += r["predicted_close"]
    for key, (n, c) in sorted(tally.items(), key=lambda kv: -kv[1][0]):
        print(f"  {n:3} rounds ({c} predicted a close)  {key}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps({"eval_dir": str(eval_dir), "join_offset": k, "how": how, "rounds": rows}, indent=1) + "\n"
        )
        print(f"\nper-round records -> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
