"""Empirical inference-latency distribution from wall-clock async traces.

    python latency_dist.py outputs/eval/<run>/traces [...] --out latency.json

Reads the `latency` record the async client writes per round
(vla_round.make_vla_policy_async: infer_s = obs->chunk seconds per
inference, cycle_s = spacing between consecutive observations) and turns it
into the two frame-count populations the lockstep client draws K from:

    first  : the FIRST inference of each round, in frames (ceil) -- it is
             slow (cold caches, 1-2 s) and the prelude holds the standing
             token through it;
    steady : every later obs->obs spacing, in frames (round) -- the number of
             frames the body keeps executing the old plan before the new
             chunk exists, which under the async client's "re-observe as soon
             as inference returns" policy IS the latency.

Frames are at TF_DATASET_FPS (25). The json keeps the raw populations, not a
histogram, so `rng.choice` reproduces the distribution exactly.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=float(os.environ.get("TF_DATASET_FPS", "25")))
    args = ap.parse_args()

    first, steady, adopt, rounds = [], [], [], 0
    for d in args.trace_dirs:
        for fn in sorted(glob.glob(os.path.join(d, "round_*.json"))):
            rec = json.load(open(fn)).get("latency")
            if not rec or not rec.get("infer_s"):
                continue
            rounds += 1
            first.append(int(math.ceil(rec["infer_s"][0] * args.fps)))
            steady += [max(1, int(round(c * args.fps))) for c in rec.get("cycle_s", [])]
            # adopt: frames from obs until the chunk EXISTS (inference time);
            # the ensemble starts using a chunk then, before the next obs
            adopt += [max(1, int(math.ceil(v * args.fps))) for v in rec["infer_s"][1:]]
    if not steady:
        raise SystemExit(
            "no latency records found -- traces must come from the async client after 2026-09-02 (they carry `latency`)"
        )

    def desc(x):
        xs = sorted(x)
        q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1)))]  # noqa: E731
        return f"n={len(xs)} p10={q(0.1)} p50={q(0.5)} p90={q(0.9)} max={xs[-1]}"

    print(f"rounds {rounds}")
    print(f"first  frames: {desc(first)}")
    print(f"steady frames: {desc(steady)}")
    print(f"adopt  frames: {desc(adopt) if adopt else 'n=0'}")
    json.dump(
        {
            "first": first,
            "steady": steady,
            "adopt": adopt or steady,
            "fps": args.fps,
            "rounds": rounds,
            "sources": args.trace_dirs,
        },
        open(args.out, "w"),
    )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
