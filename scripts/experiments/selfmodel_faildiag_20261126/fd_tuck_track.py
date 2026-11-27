"""selfmodel_faildiag_20261126 revision 6: the tucked walks' hand-point tracking in the formal run (fd_smoke6.track: from
the tuck's start + 90 rows to the first reach command after it, every 5th row, FK of the measured joints against FK of
the pushed reference, the larger of the two hands), per tucked group over question 1's 139 episode runs.

    python fd_tuck_track.py --root <q1tuck>      (locomanilab_py311, from code_tuck/)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fd_smoke6 as S
import fd_tuckpose as P

FD = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126")
TN = {"dumbbell": "IThorLivingRoomApproachTableAndLiftDumbbell", "wcan": "IThorLivingRoomApproachTableAndLiftWateringCan"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    eps = [x for x in json.load(open(FD / "q1hit/an/q1_moved_hits.json")) if x.get("moved") and x["backend"] == "sonic" and "hit_bodies_1s" in x]
    so2mj = [S.SO.index(n) for n in P.MJ_NAMES]
    B = P.Body()
    for g in ("tuck1", "tuck1b", "tuck2b", "noturn_tuck1", "noturn_tuck1b"):
        p50, p90, none = [], [], 0
        for x in eps:
            r, rows, C = S.load(Path(a.root) / g / "sonic", x["task"], x["method"], x["ep"])
            t = S.track(B, so2mj, r, rows, C)
            if not t or not t.get("n"):
                none += 1
                continue
            p50.append(t["p50"])
            p90.append(t["p90"])
        q = lambda v: np.percentile(v, [10, 50, 90]).round(1).tolist()  # noqa: E731
        print(f"{g:14s} episodes with a tucked window {len(p50)} (none {none}): per-episode median hand point error cm p10/50/90 {q(p50)}; "
              f"per-episode p90 cm p10/50/90 {q(p90)}; episodes with median > 3 cm {sum(v > 3 for v in p50)}")  # fmt: skip


if __name__ == "__main__":
    main()
