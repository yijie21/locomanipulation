"""TODO 53: what in the server's and the client's saved states the disk format refuses.   python probe_kinds.py <Task> --out DIR"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import common as X

from locomani.bench.simple import memo as MEMO
from locomani.bench.simple import run as R
from locomani.bench.simple import statecopy as SC

ap = argparse.ArgumentParser()
ap.add_argument("task")
ap.add_argument("--out", required=True)
ap.add_argument("--slots", default="1")
a = ap.parse_args()
SEEN: dict = {}
orig = MEMO.Memo._persist


def spy(self, key, e):
    self.link.send(("snap_unpersistable", e.sid))
    for p, why in self.link.recv():
        SEEN.setdefault(("server", p), why)
    for n, part in e.changes.items():
        for p, why in SC.unpersistable({n: part}):
            SEEN.setdefault(("client", p), why)
    return orig(self, key, e)


MEMO.Memo._persist = spy
with X.bed_on_slot(a.task, Path(a.out) / a.task, a.slots, ("lazy_render=true", "lazy_warm=48", "memo=true")) as (
    bed,
    sess,
):
    R.run(bed, X.pool_ids(a.task)[0], dict(X.SETTING), None)
for (side, p), why in sorted(SEEN.items()):
    print(side, p, why)
