"""python -m locomani.bench.simple: the SIMPLE benchmark bed (TODO 49.2).

    run <Task> --ids F --setting J [--stance J] --out F [--skills DIR] [--bench-set k=v ...] [--set k=v ...]
        one round of one task's ids (a file, one "<Task>/<kind>/<n>" per line): slot, services, look + GraspGen-X cache,
        batch, row check (session.py). Exit 0 only when every id has a row. Runs under any python that has locomani;
        the bed itself is started under the sim venv.
    batch --ids F|SPEC --out F --slot K ...      the bed's episodes (under the sim venv; the slot's services must be up)
    look  --ids F|SPEC --out F --slot K ...      the look pass
    cachedir <Task> ...                          prints the cache folder of this package and these parameters
    check --ids SPEC --slot K                    the environment's layout against the layout asked for

--skills DIR is the skill package (default: the seed package of this module). --bench-set k=v are the run's switches
(locomani.bench.options; resolved once, recorded in every row as `bench`); --set k=v are SimpleConfig fields (recorded as
`simple_config`). Switch values outside the loops' set are refused (TODO 49.2 step 6 / 49.3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from locomani.bench import options
from locomani.bench.simple import config as cfgmod
from locomani.bench.simple.session import EXIT_ARGS, Session


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m locomani.bench.simple", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("cmd", choices=("run", "batch", "look", "cachedir", "check"))
    ap.add_argument("task", nargs="?", help="run and cachedir: the task")
    ap.add_argument("--ids")
    ap.add_argument("--out")
    ap.add_argument("--setting", default="{}")
    ap.add_argument("--stance")
    ap.add_argument("--skills", help="the skill package folder (default: the seed package)")
    ap.add_argument("--slot", type=int, default=1, help="batch / look / check: the slot whose sockets to use")
    ap.add_argument("--set", dest="set", action="append", default=[], metavar="KEY=VALUE", help="a SimpleConfig field")
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
        resolved = options.from_args(
            a, allow_planned=True
        )  # planned values are allowed to resolve; run.check_scope refuses what is not ported
        if resolved.options.bench != "simple":
            raise ValueError(f"this is the SIMPLE bed; bench={resolved.options.bench}")
        cfg = cfgmod.SimpleConfig().with_overrides(a.set)
        setting_name, _, extra = a.setting.partition("+")
        if setting_name in ("L2", "L3", "L4"):
            raise ValueError(
                "the named speed settings L2/L3/L4 (sweep_simple) are not ported; give the setting as json"
            )
        json.loads(a.setting)
        if a.stance:
            json.loads(a.stance)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ARGS
    try:
        cfgmod.check_scope(resolved.options)
    except cfgmod.ScopeError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ARGS
    if a.cmd == "run":
        if not (a.task and a.ids and a.out):
            print("usage: run <Task> --ids F --setting J [--stance J] --out F", file=sys.stderr)
            return EXIT_ARGS
        ids = [s.strip() for s in open(a.ids) if s.strip()]
        wrong = [i for i in ids if i.split("/")[0] != a.task]
        if wrong:
            print(f"error: ids of another task than {a.task}: {wrong[:3]}", file=sys.stderr)
            return EXIT_ARGS
        from locomani.bench.simple import episodes as EP

        try:
            for i in ids:
                EP.check_id(i, cfg)  # a released or foreign-pool id is refused before any slot or server is taken
        except EP.EpisodeRefused as e:
            print(f"error: {e}", file=sys.stderr)
            return EXIT_ARGS
        return Session(
            cfg,
            a.task,
            Path(a.ids),
            Path(a.out),
            a.setting,
            a.stance,
            forwarded(a),
            a.slot_timeout,
            stereo=resolved.options.target == "perception" and resolved.options.sensor == "stereo",
            look=resolved.options.target == "perception",
        ).run_round()
    from locomani.bench.simple import run as bed_run  # the heavy imports (mujoco, torch) are the bed's own

    pkg_dir = a.skills or cfg.seed_skills or str(cfgmod.SEED_DIR)
    bed = bed_run.Bed(
        cfg,
        bed_run.load_package(pkg_dir),
        a.slot,
        dict(
            bench=resolved.record(), simple_config={k: v for k, v in cfg.to_dict().items() if k != "memo_dir"}
        ),  # (the store's folder is the loop's scratch area: not in a row)
        resolved.options,
    )
    bed.record.update(view="truth" if resolved.options.target == "truth" else "camera", depth_source=bed.depth_source)
    setting = json.loads(a.setting)
    stance = json.loads(a.stance) if a.stance else None
    from locomani.bench.simple import episodes as EP

    if a.cmd == "cachedir":
        print(bed.cache_dir(a.task, setting))
    elif a.cmd == "check":
        for eid in EP.ids_of(a.ids):
            print(eid, bed_run.layout_check(bed, eid))
    elif a.cmd == "look":
        bed_run.each(bed, EP.ids_of(a.ids), a.out, lambda eid: bed_run.look(bed, eid, setting))
    else:
        bed_run.each(bed, EP.ids_of(a.ids), a.out, lambda eid: bed_run.run(bed, eid, setting, stance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
