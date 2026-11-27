"""selfmodel_freespace_20261126: HA collection's seconds per reset in time windows, from the HA shard logs (read-only;
for the coordinator's 15 % rule: E2a must not slow HA's resets by more than 15 %, else 16 processes).

    python ha_rate.py --win 2026-11-26T05:00:00Z,2026-11-26T07:00:00Z [--win ...]

Each shard log's "drive shard K: N/M resets, ..., T s" lines (T: wall seconds since the shard started) are dated by the
nearest earlier loguru timestamp in the same log (UTC). Per window: sum of T differences / sum of N differences over
consecutive lines of one shard that both fall in the window (a restart, where T goes down, starts afresh).
"""

from __future__ import annotations

import argparse
import glob
import re
from datetime import datetime, timezone

R = "/wuji-vefps-D/wuji-il/yijie/loco_outputs"
TS = re.compile(rb"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.\d+ \|")
DR = re.compile(rb"drive shard (\d+): (\d+)/(\d+) resets, .*?, (\d+) s \(")


def parse(path: str) -> list[tuple[float, int, float]]:
    out, ts = [], None
    with open(path, "rb") as fh:
        for line in fh:
            m = TS.match(line)
            if m:
                ts = datetime.strptime(m.group(1).decode(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
                continue
            m = DR.search(line)
            if m and ts is not None:
                out.append((ts, int(m.group(2)), float(m.group(4))))
    return out


def main(wins: list[tuple[float, float]]) -> None:
    logs = sorted(glob.glob(f"{R}/selfmodel_hadata_20261126/logs/*_d3_w*.log") + glob.glob(f"{R}/selfmodel_hadata_sonic_20261126/logs/*_d3_w*.log"))
    recs = {p: parse(p) for p in logs}
    for a, b in wins:
        dt = dn = 0.0
        shards = 0
        for p, rs in recs.items():
            used = False
            for (t0, n0, s0), (t1, n1, s1) in zip(rs, rs[1:]):
                if a <= t0 and t1 < b and n1 > n0 and s1 > s0:
                    dt, dn, used = dt + (s1 - s0), dn + (n1 - n0), True
            shards += used
        fa = datetime.fromtimestamp(a, timezone.utc).strftime("%H:%MZ")
        fb = datetime.fromtimestamp(b, timezone.utc).strftime("%H:%MZ")
        print(f"{fa}-{fb}: {int(dn)} resets over {shards} shards, {dt / dn if dn else float('nan'):.1f} s per reset per shard")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", action="append", required=True)
    a = ap.parse_args()
    w = []
    for s in a.win:
        x, y = s.split(",")
        f = lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()  # noqa: E731
        w.append((f(x), f(y)))
    main(w)
