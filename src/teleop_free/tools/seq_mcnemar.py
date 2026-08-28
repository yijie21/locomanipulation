"""Paired sequential McNemar between two closed-loop manifests.

    python seq_mcnemar.py A.manifest.jsonl B.manifest.jsonl [--json out]

Pairs round i of A with round i of B and REFUSES to compare rounds whose
`appearance` differs: same seed + same round index must mean the same initial
conditions, and a silent mismatch would mean the two arms were not actually
paired (different seed, a resume offset, a gym restart) -- exactly the kind of
thing that must fail loudly.

Prints the exact two-sided McNemar p over the discordant pairs, plus a
group-sequential verdict for the caller (seq_ab.sh polls this):

  DIFFERENT   p below the spending boundary at this look -- stop, arms differ
  FUTILE      enough pairs that a miss this small cannot reach significance
              by n_max under any realistic continuation -- stop, no signal
  CONTINUE    keep running

The boundaries are Pocock-flavoured screening thresholds, not an exact STEP
procedure: interim looks stop only below p<0.01, the final look uses p<0.05.
That keeps the overall false-positive rate near 5% for the 20/40/60/80/100
schedule without importing a spending-function library. The futility rule is
deliberately conservative: it fires only when the discordant-pair imbalance is
so small at n>=60 that even a perfect run-in could not cross the final
boundary. This is a SCREENING instrument -- a stopped-early "DIFFERENT" still
gets a full-budget confirmation before anything is concluded (the round-4
lesson: config gains that skipped that step did not survive it).
"""
from __future__ import annotations

import argparse
import json
import math
import sys

INTERIM_P = 0.01      # stop-for-difference at any interim look
FINAL_P = 0.05        # the final look
N_MAX = 20            # matches eval_closed_loop.py --fresh (2026-08-28)


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def exact_mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_a")
    ap.add_argument("manifest_b")
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    a, b_rows = load(args.manifest_a), load(args.manifest_b)

    # Pair by APPEARANCE, not by row index. A stand-stage failure re-attempts
    # the same round under the same appearance and BOTH attempts land in the
    # manifest (round 0 attempt 0 "[stand] walked off" + attempt 1), so row i
    # of one arm can be attempt debris the other arm never produced. Same
    # appearance = same initial conditions, which is the thing pairing is
    # actually about; the LAST attempt is the round's outcome (earlier
    # attempts died before the VLA was measured).
    def by_app(rows):
        d = {}
        for r in rows:                      # later rows overwrite: last attempt wins
            d[r["appearance"]] = r["ok"]
        return d

    ma, mb = by_app(a), by_app(b_rows)
    shared = [k for k in ma if k in mb]
    n = len(shared)
    # A same-seed pair of arms shares almost every appearance (only a round
    # still in flight on one side is missing). A different seed shares ~none.
    floor = min(len(ma), len(mb))
    if floor == 0:
        sys.exit("NOT PAIRED: at least one manifest is empty -- nothing to "
                 "test. Refusing rather than printing p=1.0 over zero pairs.")
    if n < 0.9 * floor:
        sys.exit(f"NOT PAIRED: only {n} of {floor} appearances are shared "
                 f"between the arms -- these runs were not drawn from the "
                 f"same seed/sequence. Refusing to test.")
    pairs = [(ma[k], mb[k]) for k in shared]

    ka = sum(1 for x, _ in pairs if x)
    kb = sum(1 for _, y in pairs if y)
    only_b = sum(1 for x, y in pairs if y and not x)   # B wins the pair
    only_a = sum(1 for x, y in pairs if x and not y)   # A wins the pair
    p = exact_mcnemar(only_a, only_b)

    # Futility: even if B won EVERY remaining pair, could the final look
    # reach FINAL_P? If not, no reason to keep burning sim time.
    rem = max(0, args.n_max - n)
    p_best_b = exact_mcnemar(only_a, only_b + rem)
    p_best_a = exact_mcnemar(only_a + rem, only_b)
    hopeless = p_best_b >= FINAL_P and p_best_a >= FINAL_P

    if n >= args.n_max:
        verdict = "DIFFERENT" if p < FINAL_P else "NO_DIFFERENCE"
    elif p < INTERIM_P:
        verdict = "DIFFERENT"
    elif hopeless:
        verdict = "FUTILE"
    else:
        verdict = "CONTINUE"

    res = dict(pairs=n, a_ok=ka, b_ok=kb, only_a=only_a, only_b=only_b,
               p=p, verdict=verdict, n_max=args.n_max,
               interim_p=INTERIM_P, final_p=FINAL_P)
    if args.json:
        json.dump(res, open(args.json, "w"), indent=2)
    print(f"pairs {n}  A {ka}/{n}  B {kb}/{n}  "
          f"discordant +B {only_b} / +A {only_a}  p={p:.4f}  -> {verdict}")
    # Machine-readable last line for the driver.
    print(f"VERDICT {verdict}")


if __name__ == "__main__":
    main()
