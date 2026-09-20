#!/usr/bin/env python3
"""Launch one isolated SONIC stack for the bounded dynamic ghost reach probe."""

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
from locomani.stack import Stack, StackSpec  # noqa: E402
from locomani.tasks.crate.ghost_reach_execution_cli import (  # noqa: E402
    load_execution_request,
    request_description,
)


def stack_environment(backend, output: Path) -> dict[str, str]:
    output = output.resolve()
    return {
        "LOCOMANI_BACKEND_JSON": json.dumps(backend.to_dict(), sort_keys=True),
        "SIM_STATE_TRACE_DIR": str(output.parent / "videos" / "trace"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case-index", type=int, choices=range(1, 15))
    args = parser.parse_args(argv)
    request = load_execution_request(args.request.resolve())
    runtime = request["runtime"]
    backend = runtime["backend"]
    command = [
        sys.executable,
        "-m",
        "locomani.tasks.crate.ghost_reach_execution_cli",
        str(request["path"]),
        "--output",
        str(args.output.resolve()),
    ]
    if args.case_index is not None:
        command.extend(["--case-index", str(args.case_index)])
    stack_env = stack_environment(backend, args.output)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "stack_env": stack_env,
                    **request_description(request, args.output, case_index=args.case_index),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    spec = StackSpec(
        instance=runtime["instance"],
        gpu=runtime["gpu"],
        scene_dir=str(runtime["scene_dir"]),
        prompt="reach both open hands to fixed virtual targets without walking",
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
        env=stack_env,
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
