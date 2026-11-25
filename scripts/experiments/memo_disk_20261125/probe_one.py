"""TODO 53: one start, memo with a disk store: sizes, what persists, timings.   python probe_one.py <Task> --out DIR --store DIR"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import common as X

from locomani.bench.simple import run as R

ap = argparse.ArgumentParser()
ap.add_argument("task")
ap.add_argument("--out", required=True)
ap.add_argument("--store", required=True)
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--slots", default="1")
a = ap.parse_args()
ov = ("lazy_render=true", "lazy_warm=48", f"memo_dir={a.store}", "memo_sig=probe", "memo=true")
with X.bed_on_slot(a.task, Path(a.out) / a.task, a.slots, ov) as (bed, sess):
    for eid in X.pool_ids(a.task)[: a.n]:
        for k in range(2):
            t = time.time()
            if k == 1:
                bed.memo_store.clear()
                bed.memo_eyes.clear()
            row = R.run(bed, eid, dict(X.SETTING), None)
            print(eid, k, round(time.time() - t, 1), row["success"], json.dumps(row["memo"]), flush=True)
