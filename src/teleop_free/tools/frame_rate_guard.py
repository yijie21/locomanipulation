#!/usr/bin/env python3
"""Flag rounds whose observation stream starved.

The VLA's only input is the camera stream. When the box is loaded the sim
cannot keep up and the publish rate collapses -- measured 2026-08-27 during a
two-instance A/B, with 15 of 32 cores taken by processes belonging to other
projects: median 14.6 Hz on one instance and 13.2 Hz on the other, against a
30 Hz target, with troughs at 5-6 Hz. Nothing in the stack noticed. Closed-loop
numbers were being produced from a half-speed observation stream and reported
as if they were comparable to runs recorded at full rate.

This tool does not fix that. It makes it visible, per round, after the fact,
and it can shout while a run is still going.

**It marks; it never discards.** Dropping degraded rounds would bias the
result: starvation correlates with what else is running, which correlates with
nothing about the policy, but the *rounds it hits* are not random -- long
rounds overlap more windows than short ones, and long rounds are
disproportionately failures. Marking lets a paired analysis run a sensitivity
check ("drop pairs where either arm was degraded") without baking the choice
into the data.

Reads only: a sim log and a manifest. Writes only its own sidecar json. It
touches neither the manifest nor any file the evaluation harness owns.

    # after a run
    frame_rate_guard.py --sim-log /workspace/tmp/p03/sim.log \
        --manifest outputs/eval/fast250/closed_loop.manifest.jsonl \
        --out outputs/eval/fast250/obs_rate.json

    # how bad was it
    frame_rate_guard.py --summary outputs/eval/*/obs_rate.json

    # while a run is going
    frame_rate_guard.py --watch --sim-log /workspace/tmp/p03/sim.log \
        --sim-log /workspace/tmp/p03_i1/sim.log

Exit status is 0 whether or not rounds were flagged: this is an instrument,
and an instrument that fails the build is an instrument people stop running.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time

FREQ_RE = re.compile(r"Image publish frequency:\s+([0-9.]+)")
# The gym resets the scene through :5582 once per round. When these are
# present they are exact round boundaries; see _segment for what happens when
# they are not.
RESET_RE = re.compile(r"^\[episode\].*reset")

DEFAULT_THRESHOLD = 25.0     # Hz, median over a round. Target is 30.


def parse_log(path: str) -> tuple[list[float], list[int]]:
    """-> (publish-rate samples in order, indices into that list where a reset
    happened). The rate lines carry no timestamp, which is the whole reason
    _segment has to work as hard as it does."""
    samples: list[float] = []
    resets: list[int] = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = FREQ_RE.search(line)
            if m:
                samples.append(float(m.group(1)))
            elif RESET_RE.match(line):
                resets.append(len(samples))
    return samples, resets


def _segment(samples: list[float], resets: list[int],
             durations: list[float]) -> tuple[list[list[float]], str]:
    """Cut the sample stream into one bucket per round.

    Two ways, and the tool reports which one it used because they are not
    equally trustworthy:

    `marker` -- one reset per round, so the boundaries are exact.

    `proportional` -- no usable markers, so samples are handed out in
    proportion to each round's wall-clock `seconds`. This assumes the rate
    lines are emitted at a roughly constant wall cadence. They are not: the
    sim prints one every fixed number of frames, so a starved stretch emits
    them more slowly and is therefore *under*-sampled by this split. The bias
    runs toward optimism -- a round marked degraded under `proportional` was
    degraded; a round that passed may still have starved. Treat a
    `proportional` pass as weak evidence and a `proportional` flag as strong.
    """
    n = len(durations)
    if len(resets) >= n >= 1:
        bounds = resets[:n] + [len(samples)]
        buckets = [samples[bounds[i]:bounds[i + 1]] for i in range(n)]
        if all(buckets):
            return buckets, "marker"

    total = sum(durations) or float(n)
    buckets, cursor = [], 0.0
    for d in durations:
        nxt = cursor + (d / total) * len(samples)
        buckets.append(samples[int(round(cursor)):int(round(nxt))])
        cursor = nxt
    return buckets, "proportional"


def analyse(sim_log: str, manifest: str, threshold: float) -> dict:
    rows = [json.loads(l) for l in open(manifest) if l.strip()]
    durations = [float(r.get("seconds") or 0.0) for r in rows]
    samples, resets = parse_log(sim_log)
    buckets, method = _segment(samples, resets, durations)

    rounds = []
    for r, b in zip(rows, buckets):
        if b:
            med, lo, n = statistics.median(b), min(b), len(b)
            degraded = med < threshold
        else:
            # No samples landed in this round. Not evidence of health.
            med = lo = None
            n = 0
            degraded = False
        rounds.append({
            "round": r.get("round"), "attempt": r.get("attempt"),
            "ok": r.get("ok"), "seconds": r.get("seconds"),
            "obs_hz_med": None if med is None else round(med, 2),
            "obs_hz_min": None if lo is None else round(lo, 2),
            "n_samples": n, "degraded": degraded,
        })

    covered = [x for x in rounds if x["n_samples"]]
    flagged = [x for x in covered if x["degraded"]]
    meds = [x["obs_hz_med"] for x in covered]
    return {
        "sim_log": os.path.abspath(sim_log),
        "manifest": os.path.abspath(manifest),
        "threshold_hz": threshold,
        "segmentation": method,
        "n_rounds": len(rounds),
        "n_covered": len(covered),
        "n_degraded": len(flagged),
        "degraded_frac": (len(flagged) / len(covered)) if covered else None,
        "obs_hz_med_overall": round(statistics.median(meds), 2) if meds else None,
        "rounds": rounds,
    }


def cmd_summary(paths: list[str]) -> None:
    print(f"{'run':34s} {'seg':13s} {'rounds':>6s} {'cov':>5s} "
          f"{'degr':>5s} {'frac':>6s} {'med Hz':>7s}")
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:                                # noqa: BLE001
            print(f"{p}: unreadable ({e})")
            continue
        name = os.path.basename(os.path.dirname(os.path.abspath(p))) or p
        frac = d.get("degraded_frac")
        print(f"{name[:34]:34s} {d.get('segmentation',''):13s} "
              f"{d.get('n_rounds',0):6d} {d.get('n_covered',0):5d} "
              f"{d.get('n_degraded',0):5d} "
              f"{'-' if frac is None else format(frac, '6.1%')} "
              f"{d.get('obs_hz_med_overall') or '-':>7}")


def cmd_watch(logs: list[str], threshold: float, window: int,
              poll: float) -> None:
    """Tail the logs; shout when `window` consecutive samples sit below the
    threshold. Prints one line per state change, not per sample, so it can be
    left running inside another loop without drowning it."""
    fhs = {}
    for p in logs:
        fh = open(p, errors="ignore")
        fh.seek(0, os.SEEK_END)
        fhs[p] = fh
    recent: dict[str, list[float]] = {p: [] for p in logs}
    starving: dict[str, bool] = {p: False for p in logs}
    print(f"[guard] watching {len(logs)} log(s), threshold {threshold} Hz, "
          f"window {window}", flush=True)
    while True:
        for p, fh in fhs.items():
            for line in fh:
                m = FREQ_RE.search(line)
                if not m:
                    continue
                buf = recent[p]
                buf.append(float(m.group(1)))
                del buf[:-window]
                if len(buf) < window:
                    continue
                med = statistics.median(buf)
                bad = med < threshold
                if bad != starving[p]:
                    starving[p] = bad
                    tag = "STARVED" if bad else "recovered"
                    print(f"[guard] {tag}: {os.path.basename(os.path.dirname(p))}"
                          f"/{os.path.basename(p)} median {med:.1f} Hz "
                          f"over last {window}", flush=True)
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Flag rounds whose observation stream starved.")
    ap.add_argument("--sim-log", action="append", default=[],
                    help="sim.log to read; repeat for multiple instances")
    ap.add_argument("--manifest", help="the run's manifest jsonl")
    ap.add_argument("--out", help="where to write the sidecar json")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"median Hz below which a round is degraded "
                         f"(default {DEFAULT_THRESHOLD}; target rate is 30)")
    ap.add_argument("--summary", nargs="+", metavar="JSON",
                    help="print the degraded fraction for these sidecars")
    ap.add_argument("--watch", action="store_true",
                    help="tail --sim-log(s) and shout on starvation")
    ap.add_argument("--window", type=int, default=10,
                    help="--watch: consecutive samples in the median")
    ap.add_argument("--poll", type=float, default=2.0)
    args = ap.parse_args()

    if args.summary:
        cmd_summary(args.summary)
        return
    if args.watch:
        if not args.sim_log:
            ap.error("--watch needs at least one --sim-log")
        cmd_watch(args.sim_log, args.threshold, args.window, args.poll)
        return
    if not args.sim_log or not args.manifest:
        ap.error("give --sim-log and --manifest, or --summary, or --watch")

    report = analyse(args.sim_log[0], args.manifest, args.threshold)
    if len(args.sim_log) > 1:
        report["other_instances"] = [
            analyse(p, args.manifest, args.threshold) for p in args.sim_log[1:]]
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.out}")
    frac = report["degraded_frac"]
    print(f"segmentation={report['segmentation']}  "
          f"covered={report['n_covered']}/{report['n_rounds']}  "
          f"degraded={report['n_degraded']}"
          f"{'' if frac is None else f' ({frac:.1%})'}  "
          f"median={report['obs_hz_med_overall']} Hz "
          f"(threshold {report['threshold_hz']})")
    if report["segmentation"] == "proportional":
        print("  note: no per-round markers in the log, so boundaries are "
              "estimated from round durations -- flags are trustworthy, "
              "passes are weak (see _segment).", file=sys.stderr)


if __name__ == "__main__":
    main()
