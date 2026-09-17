#!/usr/bin/env python3
"""What a stage must leave behind, recorded from a run we accept and checked against a new one.

    python scripts/artifact_contract.py record outputs/runs/<name> bottle_eval --under eval --also eval.log
    python scripts/artifact_contract.py check  outputs/runs/<name> bottle_eval

A contract is a list of glob patterns relative to the run directory, each with the smallest number
of files that must match. It says nothing about content: it is the net for the failure that has no
other alarm -- an artifact that quietly stops being produced. The eval stage rendered no round
video for a day (2026-09-17) while every test passed and the stage reported ok, because nothing
asserted that `eval/videos` had anything in it.

`record` collapses batch numbering (`videos_07` -> `videos_*`, `collect_07.log` -> `collect_*.log`)
so a contract taken from a 30-batch run also fits a 3-batch one. Review what it writes before
committing it: a reference run can contain artifacts that are incidental, and a contract is only
worth having if every line in it is something a later run genuinely owes you.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CONTRACTS = Path(__file__).resolve().parents[1] / "tests" / "contracts"
IN_DIR = re.compile(r"\d{2,}")  # videos_07, checkpoint-10000, chunk-000, visual_crate_s7000_00
IN_FILE = re.compile(r"(?<=[_\-])\d{2,}(?=[_\-.]|$)")  # manifest_07.jsonl, but not modality_gr00t_sonic.json
UNIQUE = re.compile(r"\d{4,}")  # a timestamp or an episode index: no rerun repeats it


def pattern_for(rel: Path) -> str:
    """One pattern per file family: batch numbers become `*`, run-unique names become `*.ext`.

    A name is only genericised when it carries something a rerun cannot repeat -- a batch index or
    a timestamp. `inputs.json` and `manifest.json` stay themselves, so a contract that lists both
    still fails when one of them disappears.
    """
    parts, last = [], len(rel.parts) - 1
    for i, part in enumerate(rel.parts):
        if i < last:
            parts.append(IN_DIR.sub("*", part))  # a batch, chunk, step or seed index
        elif UNIQUE.search(part):
            parts.append("*" + part[part.index(".") :] if "." in part else "*")
        else:
            parts.append(IN_FILE.sub("*", part))
    return "/".join(parts)


def record(run: Path, under: list[str], also: list[str]) -> list[dict]:
    counts: dict[str, int] = {}
    for prefix in under:
        root = run / prefix if prefix != "." else run
        for path in sorted(root.rglob("*")):
            if path.is_file():
                counts[pattern_for(path.relative_to(run))] = counts.get(pattern_for(path.relative_to(run)), 0) + 1
    for extra in also:
        matches = len(list(run.glob(extra)))
        counts[extra] = max(matches, 1)
    return [dict(pattern=p, min=1) for p in sorted(counts)]


def check(run: Path, contract: list[dict]) -> list[str]:
    """The patterns this run does not satisfy, as human sentences."""
    missing = []
    for entry in contract:
        found = len([p for p in run.glob(entry["pattern"]) if p.is_file()])
        if found < entry["min"]:
            missing.append(f"{entry['pattern']}: {found} file(s), contract wants {entry['min']}")
    return missing


def load(name: str) -> list[dict]:
    return json.loads((CONTRACTS / f"{name}.json").read_text())["required"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("record", "check"))
    ap.add_argument("run", type=Path)
    ap.add_argument("name")
    ap.add_argument("--under", action="append", default=[], help="record everything under this subdirectory")
    ap.add_argument("--also", action="append", default=[], help="an extra pattern, relative to the run directory")
    ap.add_argument("--note", default="", help="what the contract is for, stored next to it")
    args = ap.parse_args(argv)
    if args.verb == "record":
        entries = record(args.run, args.under or ["."], args.also)
        CONTRACTS.mkdir(parents=True, exist_ok=True)
        path = CONTRACTS / f"{args.name}.json"
        path.write_text(json.dumps(dict(recorded_from=str(args.run), note=args.note, required=entries), indent=1))
        print(f"{path}: {len(entries)} patterns from {args.run}")
        return 0
    missing = check(args.run, load(args.name))
    print("\n".join(missing) if missing else f"{args.name}: all patterns satisfied by {args.run}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
