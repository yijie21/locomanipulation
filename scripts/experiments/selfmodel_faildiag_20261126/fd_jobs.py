"""selfmodel_faildiag_20261126: job lists for fd_pool.sh (one line per job: fd_run.py arguments).

    python fd_jobs.py q1 > jobs_q1.txt      # rerun: SONIC alone / selfmodel / reachable, ScaleBFM alone / reachable,
                                            # dumbbell + watering can, every episode past the split (lmlcmp3's alone records)
    python fd_jobs.py q3 > jobs_q3.txt      # rerun HEFT alone with the full qpos: candle, apple, dumbbell, watering can
    python fd_jobs.py q2 > jobs_q2.txt      # servo / servopos / gtgrasp / gtservo (measurement tools)

Episodes are cut into chunks of CHUNK so the pool stays balanced; fd_run.py skips an episode whose record exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LC3 = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126")
DB, WC = "IThorLivingRoomApproachTableAndLiftDumbbell", "IThorLivingRoomApproachTableAndLiftWateringCan"
CANDLE, APPLE = "IThorBathroomLiftCandleFromFloor", "IThorKitchenApproachTableAndLiftApple"
CHUNK = 4


def post_split(backend: str, task: str) -> list[int]:
    """Episodes 30..119 whose lmlcmp3 alone record got past the split (a plan after the look, no error)."""
    out = []
    for ep in range(30, 120):
        f = LC3 / backend / "episodes" / task / "alone" / f"ep{ep:+05d}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if r.get("error") or not r.get("plan") or r["plan"].get("fail") is not None:
            continue
        out.append(ep)
    return out


def lines(group: str, backend: str, task: str, method: str, eps: list[int], extra: str = "") -> list[str]:
    return [f"run --group {group} --backend {backend} --task {task} --method {method} --ep-list {','.join(map(str, eps[i:i + CHUNK]))}{extra}"
            for i in range(0, len(eps), CHUNK)]  # fmt: skip


def main() -> None:
    which = sys.argv[1]
    out: list[str] = []
    if which == "q1":
        for task in (DB, WC):
            for be, methods in (("sonic", ("alone", "selfmodel", "reachable")), ("scalebfm", ("alone", "reachable"))):
                eps = post_split(be, task)
                for m in methods:
                    out += lines("rerun", be, task, m, eps)
    elif which == "q3":
        for task in (CANDLE, APPLE, DB, WC):
            out += lines("rerun", "heft", task, "alone", post_split("heft", task), " --qpos")
    elif which == "q2":
        for be in ("scalebfm", "sonic"):
            sets = [(DB, post_split(be, DB)), (WC, post_split(be, WC)), (CANDLE, post_split(be, CANDLE)[:20]), (APPLE, post_split(be, APPLE)[:20])]
            groups = ("servo", "gtgrasp", "gtservo") + (("servopos",) if be == "sonic" else ())
            for g in groups:
                for task, eps in sets:
                    if g == "servopos" and task not in (DB, WC):
                        continue
                    out += lines(g, be, task, "reachable" if g == "gtgrasp" else "alone", eps)
    print("\n".join(out))


if __name__ == "__main__":
    main()
