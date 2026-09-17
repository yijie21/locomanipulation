#!/usr/bin/env python3
"""Turn the exporter's sessions into a LeRobot dataset, compute its statistics, audit it.

    python scripts/convert.py outputs/runs/bottle_simple_200
    python scripts/convert.py outputs/runs/bottle_simple_200 --dry-run

Serves both lines: the pairing of exporter sessions with collection manifests comes from the run's
own summary, so this script does not need to know which expert produced them. The pairing is
positional and the converter trusts it -- session k holds the episodes manifest k recorded.

The audit runs with `--run-dir`, which judges the lift in world z from the run's state traces. The
exporter's object-pose column is in the pelvis frame, so without it the part of a lift the pelvis
performs cancels out and good episodes are failed (15 of 190 on 2026-09-17).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_common  # noqa: E402

from locomani import paths  # noqa: E402
from locomani.pipeline.stages import Stage  # noqa: E402

DATA_PY = Path(paths.SONIC) / ".venv_data_collection" / "bin" / "python"
TRAIN_PY = Path(paths.THIRD_PARTY) / "Isaac-GR00T" / ".venv" / "bin" / "python"
# The children run in other virtual environments and import locomani by name (check_dataset pulls
# world_lift in for --run-dir), so the path is handed over explicitly rather than inherited from
# whatever started this script.
CHILD_ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
CONVERTER = ROOT / "src/locomani/pipeline/convert_to_hf_format.py"
AUDIT = ROOT / "src/locomani/pipeline/check_dataset.py"


def collected(run_dir: Path) -> dict:
    """What collection left behind: the session/manifest pairing, in order."""
    summary = run_dir / "summary.json"
    if not summary.exists():
        raise SystemExit(f"{summary} is missing; run the collection stage in this directory first")
    entry = json.loads(summary.read_text()).get("collect")
    if not entry or not entry.get("sessions"):
        raise SystemExit(f"{summary} has no finished collection to convert")
    if len(entry["sessions"]) != len(entry["manifests"]):
        raise SystemExit("sessions and manifests must pair one to one, in order")
    return entry


def convert_argv(entry: dict, cfg: dict, train_ds: Path, val_ds: Path) -> list:
    return [
        DATA_PY, CONVERTER,
        "--sessions", *entry["sessions"],
        "--manifest", *entry["manifests"],
        "--trim-verification-tail",
        "--out", train_ds,
        "--holdout", str(cfg["train"]["holdout"]),
        "--holdout-out", val_ds,
        "--jobs", "4",
    ]  # fmt: skip


def stats_argv(dataset: Path, cfg: dict) -> list:
    return [
        TRAIN_PY, "-m", "gr00t.data.stats",
        "--dataset-path", dataset,
        "--embodiment-tag", cfg["embodiment"]["gr00t_tag"].upper(),
        "--modality-config-path", ROOT / cfg["embodiment"]["gr00t_modality_config"],
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory collection wrote")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_common.load_run_config(run_dir)
    entry = collected(run_dir)
    train_ds, val_ds = run_dir / "train_dataset", run_dir / "val_dataset"
    stage = Stage(run_dir)

    if not args.dry_run:
        run_common.require_headroom("convert")
        for existing in (train_ds, val_ds):
            if existing.exists():
                raise SystemExit(f"{existing} exists; inspect it and remove it, or convert into a new run")

    commands = [("convert", convert_argv(entry, cfg, train_ds, val_ds))]
    for label, dataset in (("train", train_ds), ("val", val_ds)):
        if label == "val" and not cfg["train"]["holdout"]:
            continue  # holdout 0 keeps every episode for training, and there is no split to audit
        commands.append((f"stats_{label}", stats_argv(dataset, cfg)))
        commands.append((f"audit_{label}", [DATA_PY, AUDIT, dataset, "--run-dir", run_dir]))

    if args.dry_run:
        for name, command in commands:
            run_common.show(name, command, {"PYTHONPATH": CHILD_ENV["PYTHONPATH"]})
        return 0
    for name, command in commands:
        stage.check(name, command, env=CHILD_ENV)

    info = json.loads((train_ds / "meta" / "info.json").read_text())
    run_common.summarise(
        run_dir,
        "convert",
        train_dataset=str(train_ds),
        val_dataset=str(val_ds) if val_ds.exists() else None,
        episodes=info.get("total_episodes"),
        frames=info.get("total_frames"),
        fps=info.get("fps"),
    )
    print(f"convert: {info.get('total_episodes')} episodes, {info.get('total_frames')} frames -> {train_ds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
