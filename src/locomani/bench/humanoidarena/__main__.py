"""python -m locomani.bench.humanoidarena: the HumanoidArena benchmark bed (TODO 49.5b).

    run <task> --ids F|SPEC --setting J --out F [--skills DIR] [--bench-set k=v ...] [--set k=v ...]
        one round of one task's ids (a file, one "<task>/<pool>/<n>" per line, or a spec "task/P/0-4"): the machine's one-simulator
        lock, the environment server (the benchmark's own venv), the episodes, one JSON row each. Exit 0 only when every id has a row without
        an error. Runs under any python that has locomani (numpy); the simulator runs in the benchmark's own venv (Isaac Sim 5.0, Isaac Lab 2.2).
    check --ids F|SPEC                          refuse or accept the ids (no simulator)

--skills DIR is the skill package (default: the seed package of this module). --bench-set k=v are the run's switches
(locomani.bench.options; resolved once, recorded in every row as `bench`); --set k=v are HumanoidArenaConfig fields (`humanoidarena_config`),
e.g. --set pools=PA to open the acceptance pool too, --set overlay_dir=DIR to save the first rgbd frame with the truth drawn on it.
Ids' pools must be allowed by `pools` (default P, the practice pool); the benchmark's evaluation episodes (groups 0, 1, 2) are refused in every layer.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from locomani.bench import options
from locomani.bench.humanoidarena import config as cfgmod
from locomani.bench.humanoidarena import episodes as EP
from locomani.bench.humanoidarena.session import EXIT_ARGS, EXIT_STOPPED, Session


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m locomani.bench.humanoidarena", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("cmd", choices=("run", "check"))
    ap.add_argument("task", nargs="?", help="run: the task")
    ap.add_argument("--ids")
    ap.add_argument("--out")
    ap.add_argument("--setting", default="{}")
    ap.add_argument("--skills", help="the skill package folder (default: the seed package)")
    ap.add_argument(
        "--set", dest="set", action="append", default=[], metavar="KEY=VALUE", help="a HumanoidArenaConfig field"
    )
    options.add_arguments(ap)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        resolved = options.resolve(a.bench or "humanoidarena", a.bench_set, allow_planned=False)
        if resolved.options.bench != "humanoidarena":
            raise ValueError(f"this is the HumanoidArena bed; bench={resolved.options.bench}")
        cfg = cfgmod.HumanoidArenaConfig().with_overrides(a.set)
        cfg.validate_scope()
        setting = json.loads(a.setting)
        cfgmod.check_scope(resolved.options)
        if not a.ids:
            raise ValueError("--ids is required")
        ids = EP.ids_of(a.ids)
        for i in ids:
            EP.check_id(cfg, i)  # the evaluation block or a foreign pool is refused before a server is started
    except (ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ARGS
    if a.cmd == "check":
        print(f"{len(ids)} ids ok")
        return 0
    if not (a.task and a.out):
        print("usage: run <task> --ids F --setting J --out F", file=sys.stderr)
        return EXIT_ARGS
    wrong = [i for i in ids if i.split("/")[0] != a.task]
    if wrong:
        print(f"error: ids of another task than {a.task}: {wrong[:3]}", file=sys.stderr)
        return EXIT_ARGS
    extra = dict(bench=resolved.record(), humanoidarena_config=cfg.to_dict(), view="truth")
    session = Session(
        cfg, a.task, ids, Path(a.out), setting, a.skills or cfg.seed_skills or str(cfgmod.SEED_DIR), extra
    )
    signal.signal(signal.SIGTERM, lambda *_: (session.stop_all(), sys.exit(EXIT_STOPPED)))
    signal.signal(signal.SIGINT, lambda *_: (session.stop_all(), sys.exit(EXIT_STOPPED)))
    return session.run_round()


if __name__ == "__main__":
    raise SystemExit(main())
