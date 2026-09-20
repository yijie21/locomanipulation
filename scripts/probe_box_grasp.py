#!/usr/bin/env python3
"""Launch one isolated SONIC stack and run the generic box-rim probe child."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from locomani import paths  # noqa: E402
from locomani.backends.sonic.config import SonicBackendConfig  # noqa: E402
from locomani.stack import Stack, StackSpec  # noqa: E402
from locomani.tasks.crate.box_rim_execution_cli import load_execution_request  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    request = load_execution_request(args.request.resolve())
    runtime = request["runtime"]
    backend = SonicBackendConfig.from_dict(runtime["backend"])
    command = [
        sys.executable,
        "-m",
        "locomani.tasks.crate.box_rim_execution_cli",
        str(args.request.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "instance": runtime["instance"],
                    "gpu": str(runtime["gpu"]),
                    "scene_dir": str(runtime["scene_dir"]),
                    "output": str(args.output.resolve()),
                },
                indent=2,
            )
        )
        return 0
    spec = StackSpec(
        instance=runtime["instance"],
        gpu=str(runtime["gpu"]),
        scene_dir=str(runtime["scene_dir"]),
        prompt="grasp the box rim with both hands",
        record=False,
        policy=os.environ.get("SONIC_POLICY", paths.SONIC_POLICY),
        deploy_root=os.environ.get("SONIC_DEPLOY_ROOT", paths.SONIC_DEPLOY),
        deploy_bin=os.environ.get("SONIC_DEPLOY_BIN") or None,
        encoder=not backend.no_encoder,
        lockstep=bool(backend.lockstep.enabled),
        auto_start=bool(backend.clean_deploy or backend.lockstep.enabled),
        wrist_cams=bool(backend.wrist_cams),
        video_dir=str(args.output.resolve().parent / "videos"),
        sim_vis=backend.sim_vis,
        scene_random=False,
        dds_domain=backend.sonic_dds_domain,
        env={"LOCOMANI_BACKEND_JSON": json.dumps(backend.to_dict(), sort_keys=True)},
    )
    stack = Stack(spec, run_dir=args.output.resolve().parent)
    try:
        env = stack.up()
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        return int(completed.returncode)
    finally:
        stack.down()


if __name__ == "__main__":
    raise SystemExit(main())
