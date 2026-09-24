#!/usr/bin/env python3
"""Generate deterministic joint 5D source targets for actual reach trials."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from locomani.tasks.crate.joint_reach_sampling import generate_joint_bank, load_observations  # noqa: E402


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    observations = None
    if config.get("method") == "boundary_refinement":
        observations_value = config.get("observations")
        if not isinstance(observations_value, str) or not observations_value:
            raise ValueError("boundary_refinement config requires a non-empty observations path")
        observations_path = Path(observations_value)
        if not observations_path.is_absolute():
            observations_path = (config_path.parent / observations_path).resolve()
        observations = load_observations(observations_path)

    bank = generate_joint_bank(config, observations=observations)
    _atomic_json(args.output.resolve(), bank)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sample_count": len(bank["samples"]),
                "generation": bank["generation"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
