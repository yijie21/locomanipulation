"""Debug: run a start twice, the second time from the first's saved states with every restore checked (what differs from the state stored)."""

import json
import sys
from pathlib import Path

import common as X
import replay_check as RC

task = sys.argv[1]
RC.install_patches()
eid = X.pool_ids(task)[0]
with X.bed_on_slot(task, Path(sys.argv[2]) / task, "1", ("lazy_render=true", "lazy_warm=8", "memo_check=true")) as (
    bed,
    sess,
):
    rb, tb = RC.run_one(bed, eid, dict(X.SETTING), True, "record")
    rc, tc = RC.run_one(bed, eid, dict(X.SETTING), True, "replay")
    print(json.dumps(rc["memo"], default=str)[:6000])
