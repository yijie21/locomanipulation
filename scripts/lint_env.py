#!/usr/bin/env python3
"""Knobs live in config dataclasses, not in the environment (standards.md, decision 8).

Only process-level variables may be read from os.environ in main code: devices, ports,
paths, and the few external services. Everything else is a knob and must move into a
config object (refactor step 3).

    python scripts/lint_env.py            # report (exit 0)
    python scripts/lint_env.py --strict   # exit 1 on any non-whitelisted read
"""

import glob
import re
import sys
from collections import defaultdict

WHITELIST = {
    "CUDA_VISIBLE_DEVICES",
    "MUJOCO_EGL_DEVICE_ID",
    "INSTANCE",
    "HF_HOME",
    "PYTHONUNBUFFERED",
    "WANDB",
    "LOCOMANI_ROOT",
    "LOCOMANI_WORKSPACE",
    "LOCOMANI_SIM_DIR",
    "SONIC_ROOT",
    "SONIC_DEPLOY_ROOT",
    "SONIC_DEPLOY_BIN",
    "SIM_VIDEO_DIR",
    "SIM_STATE_TRACE_DIR",
    "TF_CAMERA_PORT",
    "TF_SONIC_PORT",
    "TF_OUT_PORT",
    "TF_KEYBOARD_PORT",
    "TF_PRIVILEGED_PORT",
    "TF_EPISODE_PORT",
    "TF_LOCKSTEP_PORT",
    "TF_LOCKSTEP_XSUB_PORT",
    "TF_SONIC_PUB_CONNECT",
    "PORT",
    "HOME",
    "PATH",
    "TMPDIR",
}
EXCLUDE = ("third_party/", "src/gr00t/", "src/teleop_free/scalebfm/", "outputs/", ".claude/", "legacy/")
PAT = re.compile(r"os\.environ(?:\.get)?\s*[\(\[]\s*['\"]([A-Za-z_0-9]+)['\"]")


def main():
    strict = "--strict" in sys.argv
    hits = defaultdict(list)
    for f in sorted(set(glob.glob("src/**/*.py", recursive=True) + glob.glob("scripts/*.py"))):
        if any(x in f for x in EXCLUDE):
            continue
        for i, line in enumerate(open(f), 1):
            for var in PAT.findall(line):
                if var not in WHITELIST:
                    hits[var].append(f"{f}:{i}")
    for var, where in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        more = f" (+{len(where) - 1})" if len(where) > 1 else ""
        print(f"{var:32s} {len(where):3d}  {where[0]}{more}")
    total = sum(len(v) for v in hits.values())
    fail = strict and bool(hits)
    print(f"[lint_env] {len(hits)} non-whitelisted variables, {total} reads" + (" -- FAIL" if fail else ""))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
