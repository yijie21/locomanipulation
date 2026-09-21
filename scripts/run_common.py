"""Transitional shim: the implementation moved to `locomani.pipeline.run_context` (2026-09-21).

The old python stage scripts in this directory and the new module entries behind the flat bash
scripts must not drift apart while both paths exist, so there is exactly one implementation and
this file re-exports it. Deleted together with the old scripts once the gate passes
(docs/handoff/refactor-flat-bash-20260921.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locomani.pipeline.run_context import (  # noqa: E402,F401
    NEEDS,
    PID_MARGIN,
    ROOT,
    RUNS,
    load_run_config,
    measure,
    open_run,
    require_deploy_patches,
    require_headroom,
    shortfall,
    show,
    summarise,
)
