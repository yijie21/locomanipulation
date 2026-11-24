"""python -m locomani.bench.locomanilab: the LocoManiLab benchmark bed (TODO 49.4).

    run <Task> --ids F --setting J [--stance J] --out F [--skills DIR] [--bench-set k=v ...] [--set k=v ...]
        one round of one task's ids (a file, one "<Task>/<kind>/<n>" per line): slot, look + GraspGen-X cache, batch, row check
        (session.py). Exit 0 only when every id has a row. Runs under any python that has locomani; the bed itself is started
        under the sim venv, and it starts the environment server (LocoManiLab's venv) itself.
    batch --ids F|SPEC --out F ...      the bed's episodes (under the sim venv; SAM 3 must be up)
    look  --ids F|SPEC --out F ...      the look pass
    cachedir <Task> ...                 prints the cache folder of this package and these switches

--skills DIR is the skill package (default: the seed package of this module). --bench-set k=v are the run's switches
(locomani.bench.options; resolved once, recorded in every row as `bench`); --set k=v are LocoManiLabConfig fields (recorded as
`lml_config`). The ids' kinds must be allowed by `kinds` (default Q, the loops' practice rows; `--set kinds=P` for development
episodes); the benchmark's own episodes (kind E) are always refused, before a slot or a server is taken.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from locomani.bench import options
from locomani.bench.locomanilab import config as cfgmod
from locomani.bench.locomanilab.session import EXIT_ARGS, EXIT_STOPPED, Session


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m locomani.bench.locomanilab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("cmd", choices=("run", "batch", "look", "cachedir"))
    ap.add_argument("task", nargs="?", help="run and cachedir: the task")
    ap.add_argument("--ids")
    ap.add_argument("--out")
    ap.add_argument("--setting", default="{}")
    ap.add_argument("--stance")
    ap.add_argument("--skills", help="the skill package folder (default: the seed package)")
    ap.add_argument(
        "--set", dest="set", action="append", default=[], metavar="KEY=VALUE", help="a LocoManiLabConfig field"
    )
    ap.add_argument(
        "--slot-timeout", type=float, default=None, help="run: give up waiting for a slot after this many seconds"
    )
    options.add_arguments(ap)
    return ap


def forwarded(a: argparse.Namespace) -> list[str]:
    """The arguments a session hands on to the bed's processes: the same configs, so each process resolves them alike."""
    out = []
    if a.bench:
        out += ["--bench", a.bench]
    for kv in a.bench_set:
        out += ["--bench-set", kv]
    for kv in a.set:
        out += ["--set", kv]
    if a.skills:
        out += ["--skills", a.skills]
    return out


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        # this bed's profile (configs/bench/locomanilab.yaml); planned values resolve, check_scope refuses what is not wired
        resolved = options.resolve(a.bench or "locomanilab", a.bench_set, allow_planned=True)
        if resolved.options.bench != "locomanilab":
            raise ValueError(f"this is the LocoManiLab bed; bench={resolved.options.bench}")
        cfg = cfgmod.LocoManiLabConfig().with_overrides(a.set)
        cfg.validate_scope()
        json.loads(a.setting)
        if a.stance:
            json.loads(a.stance)
        cfgmod.check_scope(resolved.options)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ARGS
    from locomani.bench.locomanilab import episodes as EP

    if a.cmd == "run":
        if not (a.task and a.ids and a.out):
            print("usage: run <Task> --ids F --setting J [--stance J] --out F", file=sys.stderr)
            return EXIT_ARGS
        ids = [s.strip() for s in open(a.ids) if s.strip()]
        wrong = [i for i in ids if i.split("/")[0] != a.task]
        if wrong:
            print(f"error: ids of another task than {a.task}: {wrong[:3]}", file=sys.stderr)
            return EXIT_ARGS
        try:
            for i in ids:
                EP.check_id(cfg, i)  # a benchmark-own or foreign-pool id is refused before any slot or server is taken
        except EP.EpisodeRefused as e:
            print(f"error: {e}", file=sys.stderr)
            return EXIT_ARGS
        session = Session(cfg, a.task, Path(a.ids), Path(a.out), a.setting, a.stance, forwarded(a), a.slot_timeout,
                          look=resolved.options.target == "perception")  # fmt: skip
        signal.signal(
            signal.SIGTERM, lambda *_: (session.stop_all(), sys.exit(EXIT_STOPPED))
        )  # (a handler that only returned would let the retry loops start a new child)
        signal.signal(signal.SIGINT, lambda *_: (session.stop_all(), sys.exit(EXIT_STOPPED)))
        return session.run_round()
    from locomani.bench.locomanilab import run as bed_run  # the heavy imports (mujoco, torch) are the bed's own

    pkg_dir = a.skills or cfg.seed_skills or str(cfgmod.SEED_DIR)
    bed = bed_run.Bed(
        cfg, bed_run.load_package(pkg_dir), dict(bench=resolved.record(), lml_config=cfg.to_dict()), resolved.options
    )
    bed.record.update(view="camera", depth_source=bed.depth_source)
    if a.cmd == "cachedir":
        print(bed.cache_dir())
        return 0
    setting = json.loads(a.setting)
    stance = json.loads(a.stance) if a.stance else None
    if a.cmd == "look":
        bed_run.each(bed, EP.ids_of(a.ids), a.out, lambda eid: bed_run.look(bed, eid))
    else:
        bed_run.each(bed, EP.ids_of(a.ids), a.out, lambda eid: bed_run.run(bed, eid, setting, stance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
