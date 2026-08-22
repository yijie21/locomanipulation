"""Run N episodes back to back and report where they fail.

The point is not to collect data — `--no-record` is the default — but to turn
"episodes are flaky" into a per-stage failure rate. Single-episode debugging has
been misleading: the same script has ended in a converged pre-grasp, a walk
timeout, and a failure to stand, so which stage to work on next is a question
about a distribution, not about any one run.

One AgentCommander for the whole batch. It binds :5556 and :5580, and rebinding
those between episodes races with the socket's TIME_WAIT — reusing it is both
faster and less fragile. The privileged subscriber is shared for the same
reason.

    python src/teleop_free/run_batch.py --n 20 [--perturb] [--record]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collect_episode import SCENE_META, EpisodeFailed, run_episode  # noqa: E402
from primitives import Primitives  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402
import paths

# Perturbation ranges for P2. Modest on purpose: the aim is a policy that
# generalises over where things are, not one that has seen the object fall off
# the table.
OBJECT_DXY = 0.06          # metres, uniform in a square
OBJECT_DYAW = np.pi        # any heading — an orange has no meaningful one
ROBOT_DXY = 0.25
ROBOT_DYAW = np.deg2rad(20.0)

STAGES = ["stand", "approach_A", "grasp", "transport", "approach_B",
          "place", "retreat"]


def sample_perturbation(rng: np.random.Generator, meta: dict) -> dict:
    obj = meta["target_object"]

    return {
        "label": "p2",
        "objects": {
            obj: {"dxy": rng.uniform(-OBJECT_DXY, OBJECT_DXY, 2).tolist(),
                  "yaw": float(rng.uniform(-OBJECT_DYAW, OBJECT_DYAW))},
        },
        "robot": {"dxy": rng.uniform(-ROBOT_DXY, ROBOT_DXY, 2).tolist(),
                  "yaw": float(rng.uniform(-ROBOT_DYAW, ROBOT_DYAW))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="ep", help="prefix for per-episode video files")
    ap.add_argument("--out", default=f"{paths.REPO}/outputs/logs")
    args = ap.parse_args()

    meta = json.load(open(SCENE_META))
    rng = np.random.default_rng(args.seed)
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()
    os.makedirs(args.out, exist_ok=True)

    results = []
    failed_at = collections.Counter()
    t_batch = time.time()

    for i in range(args.n):
        perturb = sample_perturbation(rng, meta) if args.perturb else None
        # The sim names each episode's video from the reset label.
        perturb = perturb or {}
        perturb["label"] = perturb.get("label", "ep")
        perturb["video"] = f"{args.tag}_{i:03d}"
        print(f"\n=== episode {i + 1}/{args.n} ===")
        t0 = time.time()
        try:
            r = run_episode(p, meta, side=args.side, record=args.record,
                            reset=reset, perturb=perturb)
            print(f"  OK in {time.time() - t0:.0f}s")
        except EpisodeFailed as e:
            r = {"ok": False, "stage": e.stage, "why": e.why}
            print(f"  FAILED {e}")
        except Exception:
            # One bad episode must not end the batch — the whole point is the
            # distribution over many.
            r = {"ok": False, "stage": "exception", "why": traceback.format_exc(-1)}
            print(f"  EXCEPTION\n{r['why']}")

        r.update(index=i, duration_s=time.time() - t0,
                 perturb=perturb, aborted=p.aborted)
        p.aborted = None
        results.append(r)
        failed_at[r.get("stage") if not r.get("ok") else "ok"] += 1

        if args.record:
            p.cmd.key("c" if r.get("ok") else "x")
            time.sleep(2.0)
        p.cmd.pause_control()
        time.sleep(1.0)

    # --- report ---------------------------------------------------------
    n_ok = failed_at.get("ok", 0)
    print(f"\n{'=' * 56}")
    print(f"{args.n} episodes in {(time.time() - t_batch) / 60:.1f} min — "
          f"{n_ok} succeeded ({100 * n_ok / max(args.n, 1):.0f}%)")
    print(f"\n{'stage':>12}  {'reached':>8} {'failed here':>12}")
    # "Reached" counts episodes that got at least as far as each stage, so the
    # table reads as a funnel rather than as isolated counts.
    order = ["ok"] + STAGES
    reached = args.n
    for stage in STAGES:
        n_fail = failed_at.get(stage, 0)
        print(f"{stage:>12}  {reached:>8} {n_fail:>12}")
        reached -= n_fail
    for other in set(failed_at) - set(order):
        print(f"{other:>12}  {'-':>8} {failed_at[other]:>12}")

    reasons = collections.Counter(
        r["why"].split("(")[0].strip() for r in results if not r.get("ok"))
    print("\nfailure reasons:")
    for why, n in reasons.most_common(8):
        print(f"  {n:>3}x  {why[:90]}")

    path = os.path.join(args.out, f"batch_{int(t_batch)}.json")
    with open(path, "w") as f:
        json.dump({"n": args.n, "perturb": args.perturb, "seed": args.seed,
                   "results": results}, f, indent=2)
    print(f"\nwrote {path}")

    reset.end_video()
    reset.close()
    p.close()


if __name__ == "__main__":
    main()
