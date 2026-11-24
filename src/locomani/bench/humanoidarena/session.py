"""One round of one task's ids on the HumanoidArena bed (TODO 49.5): the machine's one-simulator lock, the server, the rows.

HumanoidArena's physics runs on the CPU (PhysX, the benchmark's own setting) and Kit renders on one GPU; on 2026-11-23 an overload of simulators stalled the machine, so one HumanoidArena
simulator runs at a time: `Lock` takes an exclusive file lock (waits up to `lock_wait_s`) and holds it while the server lives.
"""

from __future__ import annotations

import fcntl
import json
import sys
import time
from pathlib import Path

from locomani.bench.humanoidarena import episodes as EP
from locomani.bench.humanoidarena import record
from locomani.bench.humanoidarena import run as RUN
from locomani.bench.humanoidarena.config import HumanoidArenaConfig

EXIT_OK, EXIT_ARGS, EXIT_ROWS, EXIT_STOPPED = 0, 2, 1, 3


class Lock:
    def __init__(self, path: str, wait_s: float):
        self.path, self.wait_s, self.fh = path, wait_s, None

    def __enter__(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w")
        t0 = time.time()
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.time() - t0 > self.wait_s:
                    raise TimeoutError(f"another HumanoidArena simulator holds {self.path}")
                time.sleep(2.0)

    def __exit__(self, *exc):
        if self.fh is not None:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()


class Session:
    def __init__(
        self, cfg: HumanoidArenaConfig, task: str, ids: list[str], out: Path, setting: dict, pkg_dir: str, extra: dict
    ):
        self.cfg, self.task, self.ids, self.out, self.setting = cfg, task, ids, Path(out), setting
        self.pkg = RUN.load_package(pkg_dir)
        self.bed = RUN.Bed(cfg, self.pkg, extra)

    def stop_all(self) -> None:
        self.bed.link.close()

    def run_round(self) -> int:
        """Run every id; one JSON row per id to `out`. -> 0 when every id has a row without an error, else 1."""
        rows = []
        with Lock(self.cfg.lock_path(), self.cfg.lock_wait_s):
            self.out.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out, "w") as fh:
                for eid in self.ids:
                    row = RUN.run(self.bed, eid, self.setting)
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(
                        f"{eid}: success={row['success']} bench={row['bench_success']} why={row['why'] or '-'} steps={row['steps']} {row['wall_s']}s",
                        file=sys.stderr,
                    )
            self.stop_all()
        print(json.dumps(record.summary(rows)), file=sys.stderr)
        return EXIT_OK if rows and all(r["error"] is None for r in rows) else EXIT_ROWS


__all__ = ["Session", "Lock", "EP"]
