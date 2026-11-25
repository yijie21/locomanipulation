"""TODO 52 mechanism 2, speed-up: a sweep of one post-approach parameter (the move speed, art_speed) over 15 starts of one task,
run with the memo off and on in one bed each (one server: the saved states live there), wall time per run and in all.

    python sweep_speed.py <Task> --out DIR --memo on|off [--factors 0.6,0.8,1.0,1.25,1.5] [--n 15] [--slots 1]

Order: start-major (all values of a start in a row), so the first value of a start stores the states the others restore.
Rows (one per run) and the summary go to DIR/<Task>_<memo>/.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import common as X

from locomani.bench.simple import run as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--memo", choices=("on", "off"), required=True)
    ap.add_argument("--factors", default="0.6,0.8,1.0,1.25,1.5")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--only", default=None)
    ap.add_argument("--extra", default="", help="more --set values, comma separated")
    ap.add_argument("--slots", default="1")
    ap.add_argument("--warm", type=int, default=48)
    a = ap.parse_args()
    out = Path(a.out) / f"{a.task}_{a.memo}"
    ov = (
        "lazy_render=true",
        f"lazy_warm={a.warm}",
        f"memo={'true' if a.memo == 'on' else 'false'}",
        *[x for x in a.extra.split(",") if x],
    )
    rows = json.loads((out / "rows.json").read_text()) if (out / "rows.json").exists() else []  # resumable
    done = {(r["id"], r["factor"]) for r in rows}
    t_all = time.time()
    with X.bed_on_slot(a.task, out, a.slots, ov) as (bed, sess):
        base = float(bed.S.defaults(a.task, bed.S.K_CUT)["art_speed"])
        for eid in [a.only] if a.only else X.pool_ids(a.task)[: a.n]:
            for f in [float(x) for x in a.factors.split(",")]:
                if (eid, f) in done:
                    continue
                t = time.time()
                row = R.run(bed, eid, dict(X.SETTING, art_speed=round(base * f, 4)), None)
                rows.append(dict(id=eid, factor=f, success=row["success"], t_success=row["t_success"], steps=row["counted_steps"], wall=round(time.time() - t, 1),
                                 loss=row["loss"], end=row.get("end"), memo=row.get("memo"), phases=row.get("phases")))  # fmt: skip
                print(json.dumps(rows[-1]), flush=True)
                (out / "rows.json").write_text(json.dumps(rows, default=str))
    summ = dict(task=a.task, memo=a.memo, runs=len(rows), wall_total=round(time.time() - t_all, 1), wall_runs=round(sum(r["wall"] for r in rows), 1),
                success=sum(bool(r["success"]) for r in rows))  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summ))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
