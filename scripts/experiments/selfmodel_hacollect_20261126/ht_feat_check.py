"""selfmodel_hatask_20261126: R's model input at the command (ht_task's features, logged as feat_x0 / feat_u for the
executed candidate) against the episode's own record of that reach (data/<name>.npz read by data.load with the run's
flags: what evaluation reads), read-only. lc_check's comparison for lc_run, here for ht_task.

    cd <repo_b>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=<the run's> LOCO_REPO=<repo_b> \
        $PY ht_feat_check.py <ht_task out>[,<out>...]

Per R command through the model that executed a candidate: its segment(s) in the record (same chain; a two-hand command
has one per hand, by side) -- x0 and u byte-equal, else the largest |difference| and the columns. A segment the record
lacks (the episode ended inside the reach) is counted, not failed. Known source of a difference: data3's `walked` is
the episode's (any walk at all), so a reach made before any walk in an episode that walks later (carrying) records
walked 1 while the command saw 0 -- listed separately.
Prints the counts and "FEAT CHECK PASS" / "... FAIL".
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "reachwm_20261030"))
import data as DT  # noqa: E402

WALKED_COL = 204  # x0: S[k0], S[k0-1], S[k0-2], jaw0, rot0, walked (205 is chain, lc3_run's CHAIN_COL)


def main() -> int:
    ok, n_cmp, n_same, n_missing, n_walked, bad = True, 0, 0, 0, 0, []
    for run in sys.argv[1].split(","):
        for lp in sorted(Path(run).glob("R/log/*.json")):
            lg = json.loads(lp.read_text())
            npz = Path(run) / "R" / "data" / f"{lp.stem}.npz"
            D = DT.load(str(npz)) if npz.exists() else None
            for cm in lg.get("commands", []):
                r = cm.get("r")
                if r is None or "feat_x0" not in r:
                    continue
                x0, u = np.asarray(r["feat_x0"], np.float32), np.asarray(r["feat_u"], np.float32)
                if cm.get("chain") is None:  # the episode ended inside this reach's settle (the log has no chain)
                    n_missing += 1
                    continue
                rows = [] if D is None else list(np.flatnonzero(np.asarray(D["chain"]) == cm["chain"]))
                if not rows:
                    n_missing += 1
                    continue
                if len(rows) != len(x0):
                    bad.append(dict(ep=lp.stem, label=cm["label"], why=f"{len(rows)} segments for {len(x0)} feature rows"))
                    ok = False
                    continue
                order = {int(D["side"][i]): i for i in rows}
                for j in range(len(x0)):
                    i = order[int(np.sign(u[j][0]))] if len(rows) > 1 else rows[0]
                    a, b = np.asarray(D["x0"][i], np.float32), x0[j]
                    au, bu = np.asarray(D["u"][i], np.float32), u[j]
                    n_cmp += 1
                    if a.tobytes() == b.tobytes() and au.tobytes() == bu.tobytes():
                        n_same += 1
                        continue
                    cols = np.flatnonzero(a != b).tolist()
                    if cols == [WALKED_COL] and au.tobytes() == bu.tobytes():
                        n_walked += 1
                        continue
                    bad.append(dict(ep=lp.stem, label=cm["label"], x0_cols=cols[:10], x0_max=float(np.abs(a - b).max()),
                                    u_max=float(np.abs(au - bu).max())))  # fmt: skip
                    ok = False
    for b in bad:
        print("differs:", json.dumps(b))
    print(f"compared {n_cmp} segment inputs: {n_same} byte-equal, {n_walked} differ only in walked (column {WALKED_COL}), "
          f"{len(bad)} other; {n_missing} commands without a recorded segment")  # fmt: skip
    print("FEAT CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
